# Setup Guide (GPU: CUDA 12.8)

## 1) Create Python environment
```bash
python -m venv simlingo_env
source simlingo_env/bin/activate
pip install --upgrade pip
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
# Extras needed by InternVL2 + PEFT + Lightning
pip install einops timm peft hydra-core==1.3.2 omegaconf==2.3.0 'pytorch-lightning>=2.4'
```

## 3) Verify GPU (RTX 5070 / CUDA 12.8 stack)
```bash
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY
```
Expected: torch 2.8.0+cu128, cuda available True, device name shows your RTX GPU.

## 4) Model assets
- SimLingo repo clone: `../simlingo/` (relative to this project root)
- Checkpoint file: `models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- HuggingFace: `OpenGVLab/InternVL2-1B` downloads on first use (large)

## 5) QVL SDK
- The QVL Python library is expected under `../../0_libraries/python/qvl` relative to project files
- Ensure QLabs is running with an open layout before starting the integration

## 6) Quick smoke run
```bash
source simlingo_env/bin/activate
python tests/run_basic_integration.py --duration 10 --hz 5 --spawn-yaw 90
```

## Notes
- FlashAttention2 is optional (not installed by default)
- If you keep a local HF cache outside this project, add a symlink or configure TRANSFORMERS_CACHE to avoid repeated downloads

