"""
SimLingo model wrapper (Hydra-based, real inference, no fallback).

Loads the SimLingo architecture from the local repo via Hydra configs and
restores weights from the checkpoint. Returns raw predictions (speed
waypoints and optional route waypoints). No dummy policy is provided.
"""
from __future__ import annotations


from pathlib import Path

import logging
import math
from typing import Dict, Any, Optional

import numpy as np
import torch

from omegaconf import OmegaConf
import hydra
from transformers import AutoProcessor
from simlingo_training.utils.internvl2_utils import (
    get_num_image_tokens_per_patch,
    build_transform,
    dynamic_preprocess,
    get_custom_chat_template,
)
from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
from PIL import Image


logger = logging.getLogger(__name__)


class SimLingoModel:
    """Instantiate and run the real SimLingo model from the local repository.

    Requires simlingo_training to be importable (install the submodule, e.g.
    `pip install -e external/simlingo`). No path fallbacks are used.
    """

    def __init__(self, model_root: str | Path = "./models/simlingo") -> None:
        self.model_root = Path(model_root)
        self.checkpoint_dir: Optional[Path] = None
        self.checkpoint_file: Optional[Path] = None
        self.model: Optional[torch.nn.Module] = None
        self.available: bool = False
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._instruction = "follow the road. Predict the waypoints."


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


    def set_instruction(self, text: str) -> None:
        """Set the current language instruction."""
        if text and text.strip():
            self._instruction = text.strip()

    def get_instruction(self) -> str:
        return self._instruction

    def load(self) -> bool:
        """Load SimLingo components directly (no Hydra) and restore weights from checkpoint."""
        self._locate_checkpoint()

        hydra_cfg_path = self.model_root / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(str(hydra_cfg_path))


        # Processor/tokenizer from vision variant to match official training/inference
        self._vision_variant = cfg.model.vision_model.variant
        processor = AutoProcessor.from_pretrained(cfg.model.vision_model.variant, trust_remote_code=True)

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
        model.load_state_dict(sd, strict=True)

        # Keep handy attributes used at inference
        self.model = model.to(self.device)
        self.model.eval()
        self.available = True
        self.processor = processor
        self._num_img_tokens_per_patch = get_num_image_tokens_per_patch(self._vision_variant)
        self._use_global_img = bool(cfg.data_module.use_global_img)

        logger.info(f"SimLingo model initialized on {self.device} with Hydra config. Strict checkpoint load succeeded.")
        return True

    @torch.inference_mode()
    def inference(self, processed_image: np.ndarray, camera_info: Optional[Dict[str, Any]] = None, vehicle_info: Optional[Dict[str, Any]] = None, instruction: Optional[str] = None) -> Dict[str, Any]:
        """Run the SimLingo model with full training-parity preprocessing.

        Args:
            processed_image: RGB image (H,W,3), either uint8 [0..255] or float32 in [0,1].
            camera_info: Optional dict with keys {width, height, fov_deg, intrinsics (3x3), extrinsics (4x4)}.
            vehicle_info: Optional dict with keys {speed_mps: float}.
            instruction: Optional override for current instruction string.
        Returns:
            { "pred_speed_wps": np.ndarray[N,2], "pred_route": Optional[np.ndarray[M,2]] }
        """

        # Support both uint8 RGB (preferred) and float32 [0,1] RGB for backward compatibility
        arr = processed_image
        if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
            img = np.ascontiguousarray(arr)
        else:
            # assume float in [0,1]; clip and scale to uint8
            img = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
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
        tiles = dynamic_preprocess(pil, image_size=448, use_thumbnail=self._use_global_img, max_num=2)
        pv = torch.stack([transform(t) for t in tiles])  # [NP,3,448,448]
        model_dtype = next(self.model.parameters()).dtype
        pixel_values = pv.unsqueeze(0).unsqueeze(0).to(self.device, dtype=model_dtype)  # [B=1,T=1,NP,C,H,W]

        # Build commentary (Chain-of-Thought) prompt via InternVL chat template
        # Paper: first generate commentary (language) then actions conditioned on it
        user_instr = (instruction if instruction is not None else self.get_instruction()).strip()

        commentary_question = (
            "Instruction: " + user_instr + "\n"
            "Question: What should the ego vehicle do next and why? "
            "Respond briefly (1-2 sentences)."
        )
        # Determine total number of image tokens based on number of patches (NP)
        npatches = int(pixel_values.shape[2])
        num_image_tokens_total = int(self._num_img_tokens_per_patch) * max(1, npatches)


        variant = getattr(self.model.language_model, 'variant', getattr(self, '_vision_variant', 'OpenGVLab/InternVL2-1B'))
        conversations = [[
            {"role": "user", "content": [{"type": "text", "text": commentary_question}]},
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
            language_string=[commentary_question],
            loss_masking=question_dict['loss_masking'].to(self.device),
        )

        # Camera intrinsics/extrinsics and vehicle context
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

        # Run the upstream model (CoT commentary + actions)
        outputs = self.model(din)

        # Unpack: upstream returns (speed_wps, route, language)
        pred_speed_wps: Optional[torch.Tensor] = None
        pred_route: Optional[torch.Tensor] = None
        language = None
        if isinstance(outputs, (list, tuple)):
            if len(outputs) > 0:
                pred_speed_wps = outputs[0]
            if len(outputs) > 1:
                pred_route = outputs[1]
            if len(outputs) > 2:
                language = outputs[2]
        else:
            pred_speed_wps = outputs

        out: Dict[str, Any] = {}
        if isinstance(pred_speed_wps, torch.Tensor):
            arr = pred_speed_wps.detach().float().cpu().numpy()
            if arr.ndim >= 3:
                out["pred_speed_wps"] = arr[0]
        if isinstance(pred_route, torch.Tensor):
            arr = pred_route.detach().float().cpu().numpy()
            if arr.ndim >= 3:
                out["pred_route"] = arr[0]

        # Language is already decoded to strings by upstream DrivingModel
        if isinstance(language, (list, tuple)) and len(language) > 0 and isinstance(language[0], str):
            out["language_text"] = language[0].strip()

        return out


