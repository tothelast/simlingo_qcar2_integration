# Changelog

## 2025-09-22 to 2025-09-23
- Switched environment to CUDA 12.8 stack (torch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128)
- Installed InternVL/PEFT dependencies: einops, timm, peft; ensured hydra-core, omegaconf, pytorch-lightning
- Implemented real SimLingo inference in `src/models/simlingo_wrapper.py` (no fallback policy)
- Corrected InternVL2 image token handling and chat template usage (IMG_CONTEXT block)
- Enforced stable preprocessing (float32, 448-sized patch)
- Added graceful failure: on `ModelInferenceError`, stop vehicle and exit loop
- Implemented spawn system with default yaw=90°, auto-search safe candidates, and CLI overrides
- Updated CLI (`tests/run_basic_integration.py`) with `--spawn-x/y/z/yaw` and `--no-autosearch`
- Created `.gitignore` (exclude venv, caches, logs, large model artifacts)
- Updated README and docs with architecture, setup, usage, troubleshooting, and changelog

