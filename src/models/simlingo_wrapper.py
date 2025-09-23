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

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    pass


class ModelInferenceError(RuntimeError):
    pass


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
            logger.warning(f"No checkpoints directory found at: {ckpt_root}")
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
            logger.info(f"Found SimLingo checkpoint: {self.checkpoint_file}")

            # Import repo modules and assemble model manually
            self._ensure_repo_on_path()
            from transformers import AutoProcessor
            from simlingo_training.models.language_model.llm import LLM
            from simlingo_training.models.encoder.internvl2_model import LingoInternVLModel
            from simlingo_training.models.adaptors.adaptors import (
                DrivingAdaptor, LanguageAdaptor, WaypointInputAdaptor, AdaptorList,
            )
            import torch.nn as nn

            variant = "OpenGVLab/InternVL2-1B"
            processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)

            # Language model (LoRA-enabled to match checkpoint keys)
            language_model = LLM(
                variant=variant,
                lora=True,
                lora_alpha=64,
                lora_r=32,
                lora_dropout=0.1,
            )

            # Vision encoder (feature extractor)
            image_encoder = LingoInternVLModel(variant)
            image_encoder.processor = processor
            image_encoder.use_global_img = False

            # Adaptors and waypoint encoder
            driving = DrivingAdaptor(
                language_model.hidden_size,
                speed_wps_mode='2d',
                predict_route_as_wps=True,
            )
            language = LanguageAdaptor(language_model)
            adaptors = AdaptorList(language=language, driving=driving)
            wp_encoder = WaypointInputAdaptor(
                token_size=language_model.hidden_size,
                hidden_size=256,
                hidden_size2=512,
            )

            class MinimalDriving(nn.Module):
                def __init__(self, language_model, processor, image_encoder, adaptors, wp_encoder):
                    super().__init__()
                    self.language_model = language_model
                    self.processor = processor
                    self.image_encoder = image_encoder
                    self.adaptors = adaptors
                    self.wp_encoder = wp_encoder
                    # for state_dict compatibility
                    self.wp_encoder = wp_encoder

                def forward(self, example):
                    try:
                        driving_input = example.driving_input
                    except AttributeError:
                        driving_input = example

                    adaptor_dict = self.adaptors(example, inference=True)
                    adaptor_dict = self.image_encoder.replace_placeholder_tokens(
                        adaptor_dict=adaptor_dict,
                        pixel_values=driving_input.camera_images,
                        placeholder_values=driving_input.prompt_inference.placeholder_values,
                        wp_encoder=self.wp_encoder,
                    )

                    input_embeds_all = adaptor_dict["language_inputs"]
                    attention_masks = adaptor_dict["language_inputs_mask"]

                    speed_wps, route, language_out = None, None, []
                    for b_idx, (input_embed, attention_mask) in enumerate(zip(input_embeds_all, attention_masks)):
                        input_embed = input_embed.unsqueeze(0)
                        attention_mask = attention_mask.unsqueeze(0)
                        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
                        if self.language_model.variant == 'OpenGVLab/InternVL2-4B':
                            eos = tok.added_tokens_encoder.get('<|end|>', tok.eos_token_id)
                        elif self.language_model.variant == 'OpenGVLab/InternVL2-2B':
                            eos = tok.added_tokens_encoder.get('<|im_end|>', tok.eos_token_id)
                        else:
                            eos = tok.eos_token_id

                        sampled_tokens, input_embeds = self.language_model.greedy_sample(
                            input_embed,
                            eos_token_id=eos,
                            max_new_tokens=100,
                            input_embed_matrix=self.adaptors.language.embed_tokens.weight,
                            logit_matrix=self.adaptors.language.lm_head.weight,
                            attention_mask=attention_mask,
                        )

                        inputs_driving = self.adaptors.driving(example)
                        input_embed_concat = torch.cat((input_embeds, inputs_driving["inputs"][b_idx].unsqueeze(0)), dim=1)
                        features, logits = self.language_model.forward(input_embed_concat)

                        len_driving = inputs_driving["inputs"].size(1)
                        driving_features = features[:, -len_driving:]
                        driving_logits = logits[:, -len_driving:]
                        predictions = self.adaptors.driving.get_predictions(driving_features, driving_logits)

                        # Collect first (batch 0)
                        if predictions.get('speed_wps') is not None:
                            speed_wps = predictions['speed_wps'] if speed_wps is None else torch.cat((speed_wps, predictions['speed_wps']), dim=0)
                        if predictions.get('route') is not None:
                            route = predictions['route'] if route is None else torch.cat((route, predictions['route']), dim=0)
                        language_out.append((tok.batch_decode(sampled_tokens, skip_special_tokens=True)[0]))

                    return speed_wps, route, language_out

            composite = MinimalDriving(language_model, processor, image_encoder, adaptors, wp_encoder)

            # Load state dict from checkpoint (strict=False for safety)
            state = torch.load(str(self.checkpoint_file), map_location="cpu")
            sd = state.get('state_dict', state) if isinstance(state, dict) else state
            composite.load_state_dict(sd, strict=False)

            self.model = composite.to(self.device)
            self.model.eval()
            self.available = True
            logger.info(f"SimLingo model initialized on {self.device}.")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize SimLingo model: {e}")
            self.available = False
            return False

    @torch.inference_mode()
    def inference(self, processed_image: np.ndarray) -> Dict[str, Any]:
        """Run the SimLingo model and return predicted waypoints.

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
            # Fallback to InternVL2 default via model.hparams
            processor = getattr(self.model, "processor", None)
            if processor is None:
                # Try rebuilding from vision model variant stored in lightning module
                variant = getattr(getattr(self.model, "language_model", object()), "variant", None) or \
                          getattr(getattr(self.model, "vision_model", object()), "variant", None)
                if variant is None:
                    raise ModelInferenceError("Cannot resolve vision/language variant for processor.")
                processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)

            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

            # Image preprocessing: single centered patch to ensure NP>=1
            transform = build_transform(input_size=448)
            pil = Image.fromarray(img)
            images = [pil]  # single image patch
            pixel_values = torch.stack([transform(im) for im in images])  # [1,3,448,448]
            pixel_values = pixel_values.unsqueeze(0).unsqueeze(0)  # [1,T=1,N=1,3,448,448]
            # Keep float32 to match linear layer biases in auxiliary heads
            pixel_values = pixel_values.to(self.device, dtype=torch.float32)

            # Minimal prompt: follow the road; build prompt via InternVL chat template with proper image tokens
            prompt_text = "Current speed: 0.0 m/s. Command: follow the road. Predict the waypoints."
            # Determine total number of image tokens based on number of patches (NP)
            npatches = int(pixel_values.shape[2])  # NP from [B,T,NP,C,H,W]
            num_image_token = getattr(getattr(self.model, 'image_encoder').model, 'num_image_token', None)
            if num_image_token is None:
                num_image_token = getattr(getattr(self.model, 'image_encoder').model.config, 'num_image_token', 256)
            num_image_tokens_total = int(num_image_token) * max(1, npatches)
            variant = getattr(self.model.language_model, 'variant', 'OpenGVLab/InternVL2-1B')
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

            # Dummy intrinsics/extrinsics (not used by inference heads)
            K = torch.eye(3, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)[:, 0]
            E = torch.eye(4, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)[:, 0]
            speed = torch.tensor([[0.0]], dtype=torch.float32, device=self.device)
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

