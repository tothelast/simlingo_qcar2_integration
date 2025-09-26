"""
SimLingo model wrapper (Hydra-based, real inference, no fallback).

Loads the SimLingo architecture from the local repo via Hydra configs and
restores weights from the checkpoint. Returns raw predictions (speed
waypoints and optional route waypoints). No dummy policy is provided.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from threading import Lock

import numpy as np
import torch

logger = logging.getLogger(__name__)

class SimLingoModel:
    """Instantiate and run the real SimLingo model from the local repository.

    This expects the SimLingo repo to be cloned locally (read‑only import), e.g. at
    ../simlingo relative to this package, or provide SIMLINGO_REPO env/path.
    """

    def __init__(self, model_root: str | Path = "./models/simlingo", simlingo_repo: Optional[str | Path] = None) -> None:
        self.model_root = Path(model_root)
        self.checkpoint_dir: Optional[Path] = None
        self.checkpoint_file: Optional[Path] = None
        self.model: Optional[torch.nn.Module] = None
        self.available: bool = False
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Allow CPU fallback if GPU arch unsupported at runtime
        try:
            if self.device.type == "cuda":
                _ = torch.cuda.get_device_name(0)
        except Exception:
            self.device = torch.device("cpu")

        # Live instruction (thread-safe)
        self._instr_lock = Lock()
        self._instruction = "follow the road. Predict the waypoints."

        # Resolve SimLingo repo path
        if simlingo_repo is None:
            # Try <workspace root>/simlingo from this file
            here = Path(__file__).resolve()
            candidate = here.parents[3] / "simlingo"
            self.simlingo_repo = candidate
        else:
            self.simlingo_repo = Path(simlingo_repo)

    def _locate_checkpoint(self) -> None:
        ckpt_root = self.model_root / "checkpoints"
        if not ckpt_root.exists():

            return
        epoch_dirs = sorted(ckpt_root.glob("epoch=*"))
        if epoch_dirs:
            for d in epoch_dirs:
                for fname in ("pytorch_model.bin", "pytorch_model.pt"):
                    candidate = d / fname
                    if candidate.exists():
                        self.checkpoint_dir = d
                        self.checkpoint_file = candidate
                        return
        for pattern in ("pytorch_model.bin", "pytorch_model.pt"):
            candidates = list(ckpt_root.rglob(pattern))
            if candidates:
                self.checkpoint_file = candidates[0]
                self.checkpoint_dir = candidates[0].parent
                return

    def _ensure_repo_on_path(self) -> None:
        repo = str(self.simlingo_repo)
        if repo not in sys.path:
            sys.path.insert(0, repo)

    def set_instruction(self, text: str) -> None:
        """Set the current language instruction (thread-safe)."""
        if text is None:
            return
        t = text.strip()
        if not t:
            return
        with self._instr_lock:
            self._instruction = t


    def get_instruction(self) -> str:
        with self._instr_lock:
            return self._instruction

    def load(self, try_load_weights: bool = False) -> bool:
        """Load SimLingo components directly (no Hydra) and restore weights from checkpoint."""
        try:
            if not self.model_root.exists():
                raise ModelLoadError(f"Model root not found: {self.model_root}")
            if not self.simlingo_repo.exists():
                raise ModelLoadError(f"SimLingo repo not found at: {self.simlingo_repo}")

            self._locate_checkpoint()
            if not self.checkpoint_file or not self.checkpoint_file.exists():
                raise ModelLoadError("SimLingo checkpoint not found under models/simlingo/")
    

            # Load model via original Hydra config to match training exactly
            self._ensure_repo_on_path()
            from omegaconf import OmegaConf
            import hydra
            from transformers import AutoProcessor
            from simlingo_training.utils.internvl2_utils import get_num_image_tokens_per_patch

            hydra_cfg_path = self.model_root / ".hydra" / "config.yaml"
            if not hydra_cfg_path.exists():
                raise ModelLoadError(f"Hydra config not found: {hydra_cfg_path}")
            cfg = OmegaConf.load(str(hydra_cfg_path))

            # Processor/tokenizer from language/vision variant used in training
            lm_variant = cfg.model.language_model.variant
            self._vision_variant = cfg.model.vision_model.variant
            processor = AutoProcessor.from_pretrained(lm_variant, trust_remote_code=True)

            # Instantiate model exactly like training
            cache_dir = str(self.model_root.parent / "pretrained")
            model = hydra.utils.instantiate(
                cfg.model,
                cfg_data_module=cfg.data_module,
                processor=processor,
                cache_dir=cache_dir,
                _recursive_=False,
            )

            # Load checkpoint strictly and verify keys
            state = torch.load(str(self.checkpoint_file), map_location="cpu")
            sd = state.get("state_dict", state) if isinstance(state, dict) else state
            load_res = model.load_state_dict(sd, strict=False)
            missing = list(load_res.missing_keys)
            unexpected = list(load_res.unexpected_keys)
            if missing or unexpected:
                logger.error(f"Checkpoint key mismatch. Missing: {missing[:8]}{'...' if len(missing)>8 else ''}; Unexpected: {unexpected[:8]}{'...' if len(unexpected)>8 else ''}")
                raise ModelLoadError("Model architecture does not match checkpoint (strict load failed)")
            # Enforce strict match
            model.load_state_dict(sd, strict=True)

            # Keep handy attributes used at inference
            self.model = model.to(self.device)
            self.model.eval()
            self.available = True
            self.processor = processor
            self._num_img_tokens_per_patch = get_num_image_tokens_per_patch(self._vision_variant)
            # training flag
            try:
                self._use_global_img = bool(cfg.data_module.use_global_img)
            except Exception:
                self._use_global_img = False
            logger.info(f"SimLingo model initialized on {self.device} with Hydra config. Strict checkpoint load succeeded.")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize SimLingo model: {e}")
            self.available = False
            return False

    @torch.inference_mode()
    def inference(self, processed_image: np.ndarray, camera_info: Optional[Dict[str, Any]] = None, vehicle_info: Optional[Dict[str, Any]] = None, instruction: Optional[str] = None) -> Dict[str, Any]:
        """Run the SimLingo model with full training-parity preprocessing.

        Args:
            processed_image: RGB float32 image in [0,1] range (H,W,3).
            camera_info: Optional dict with keys {width, height, fov_deg, intrinsics (3x3), extrinsics (4x4)}.
            vehicle_info: Optional dict with keys {speed_mps: float}.
            instruction: Optional override for current instruction string.
        Returns:
            { "pred_speed_wps": np.ndarray[N,2], "pred_route": Optional[np.ndarray[M,2]] }
        """
        if self.model is None or not self.available:
            raise ModelInferenceError("Model not loaded/available.")
        if processed_image is None or processed_image.ndim != 3 or processed_image.shape[2] != 3:
            raise ModelInferenceError("Invalid input image.")

        try:
            # Lazy imports from repo for preprocessing and types
            from PIL import Image
            from transformers import AutoProcessor
            from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess, get_custom_chat_template
            from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
            from hydra.utils import to_absolute_path

            img = (np.clip(processed_image, 0.0, 1.0) * 255.0).astype(np.uint8)
            H, W = img.shape[:2]

            # Build processor/tokenizer once from model config if accessible
            processor = getattr(self, "processor", None)
            if processor is None:
                # Fallback to variant from model
                variant = getattr(getattr(self.model, "language_model", object()), "variant", None) or \
                          getattr(getattr(self.model, "vision_model", object()), "variant", None) or "OpenGVLab/InternVL2-1B"
                processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)

            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

            # Image preprocessing: dynamic multi-patch to match training
            pil = Image.fromarray(img).convert("RGB")
            transform = build_transform(input_size=448)
            tiles = dynamic_preprocess(pil, image_size=448, use_thumbnail=self._use_global_img, max_num=6)
            pv = torch.stack([transform(t) for t in tiles])  # [NP,3,448,448]
            pixel_values = pv.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)  # [B=1,T=1,NP,C,H,W]

            # Build prompt via InternVL chat template with proper image tokens
            text = (instruction if instruction is not None else self.get_instruction()).strip()
            prompt_text = f"Command: {text}."
            # Determine total number of image tokens based on number of patches (NP)
            npatches = int(pixel_values.shape[2])
            num_image_tokens_total = int(self._num_img_tokens_per_patch) * max(1, npatches)

            variant = getattr(self.model.language_model, 'variant', getattr(self, '_vision_variant', 'OpenGVLab/InternVL2-1B'))
            conversations = [[
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]}
            ]]
            _, question_dict = get_custom_chat_template(
                conversations=conversations,
                tokenizer=tokenizer,
                encoder_variant=variant,
                num_image_tokens_total=num_image_tokens_total,
                cache_root_dir='pretrained'
            )
            ids = question_dict['phrase_ids']
            valid = question_dict['phrase_valid']
            mask = question_dict['phrase_mask']
            ll = LanguageLabel(
                phrase_ids=ids.to(self.device),
                phrase_valid=valid.to(self.device),
                phrase_mask=mask.to(self.device),
                placeholder_values=[],
                language_string=[prompt_text],
                loss_masking=question_dict['loss_masking'].to(self.device),
            )

            # Camera intrinsics/extrinsics and vehicle context
            import math
            if camera_info and isinstance(camera_info, dict) and camera_info.get('intrinsics') is not None:
                K_np = np.array(camera_info['intrinsics'], dtype=np.float32)

            else:
                fov = float(camera_info.get('fov_deg', 90.0)) if isinstance(camera_info, dict) else 90.0
                fx = W / (2.0 * math.tan(fov * math.pi / 360.0))
                fy = fx
                cx = W / 2.0
                cy = H / 2.0
                K_np = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

            K = torch.tensor(K_np, dtype=torch.float32, device=self.device)

            if camera_info and isinstance(camera_info, dict) and camera_info.get('extrinsics') is not None:
                E_np = np.array(camera_info['extrinsics'], dtype=np.float32)

            else:
                E_np = np.eye(4, dtype=np.float32)

            E = torch.tensor(E_np, dtype=torch.float32, device=self.device)

            spd = float(vehicle_info.get('speed_mps', 0.0)) if isinstance(vehicle_info, dict) else 0.0
            speed = torch.tensor([[spd]], dtype=torch.float32, device=self.device)
            target_point = torch.tensor([[0.0, 0.0]], dtype=torch.float32, device=self.device)

            din = DrivingInput(
                camera_images=pixel_values,
                image_sizes=torch.tensor([[H, W]], dtype=torch.int32, device=self.device),
                camera_intrinsics=K,
                camera_extrinsics=E,
                vehicle_speed=speed,
                target_point=target_point,
                prompt=ll,
                prompt_inference=ll,
            )

            pred_speed_wps, pred_route, _ = self.model(din)

            out: Dict[str, Any] = {}
            if pred_speed_wps is not None and isinstance(pred_speed_wps, torch.Tensor):
                arr = pred_speed_wps.detach().float().cpu().numpy()
                if arr.ndim >= 3:
                    out["pred_speed_wps"] = arr[0]
            if pred_route is not None and isinstance(pred_route, torch.Tensor):
                arr = pred_route.detach().float().cpu().numpy()
                if arr.ndim >= 3:
                    out["pred_route"] = arr[0]

            if "pred_speed_wps" not in out and "pred_route" not in out:
                raise ModelInferenceError("No predictions returned by model.")
            return out

        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"Inference failed: {e}")

