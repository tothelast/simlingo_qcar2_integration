# SimLingo ↔ QCar2 (QLabs) Integration

Real SimLingo Vision–Language–Action (VLA) inference driving a QCar2 in QLabs. No fallback policies. GPU‑accelerated on RTX 5070 (CUDA 12.8).

## Highlights
- True neural inference: InternVL2‑1B backbone with LoRA (PEFT), loading epoch=013.ckpt
- End‑to‑end pipeline: QLabs → QCar2 camera → preprocessing → SimLingo VLA → waypoint→control → vehicle commands
- Graceful failure: stops vehicle and exits loop if inference fails (no hand‑crafted fallback)
- Spawn system with auto‑search for a safe starting pose and CLI overrides (x,y,z,yaw)

## Project Structure
```
simlingo_qcar2_integration/
├── src/
│   ├── adapters/
│   │   ├── data_adapter.py          # QCar2 camera → model input
│   │   └── control_adapter.py       # SimLingo outputs → QCar2 controls
│   ├── models/
│   │   └── simlingo_wrapper.py      # Real model load + inference
│   └── integration/
│       └── main_bridge.py           # QLabs orchestration + control loop
├── models/
│   └── simlingo/
│       └── checkpoints/epoch=013.ckpt/pytorch_model.pt
├── tests/
│   └── run_basic_integration.py     # CLI to run an end‑to‑end session
├── docs/                            # Architecture, setup, usage, troubleshooting
└── requirements.txt
```

## Requirements (GPU stack)
- Ubuntu 24.04, Python 3.12 (project uses venv: `simlingo_env/`)
- NVIDIA RTX 5070 (or CUDA 12.8 capable GPU)
- PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, torchaudio 2.8.0+cu128
- Core deps: transformers, opencv‑python, pillow, numpy
- SimLingo/InternVL extras: hydra‑core 1.3.2, omegaconf 2.3.0, pytorch‑lightning ≥ 2.4, einops, timm, peft

## Setup
1) Create and activate venv
```bash
python -m venv simlingo_env
source simlingo_env/bin/activate
```
2) Install requirements
```bash
pip install -r requirements.txt
# Ensure extras are present (used by InternVL2 & PEFT):
pip install einops timm peft hydra-core==1.3.2 omegaconf==2.3.0 'pytorch-lightning>=2.4'
```
3) Verify GPU is detected
```bash
python - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device 0:', torch.cuda.get_device_name(0))
PY
```
4) Check model assets
- SimLingo repo: `../simlingo/` (cloned)
- Checkpoint: `models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- InternVL2‑1B will be auto‑downloaded from HuggingFace on first run (large)

## Usage
Run with QLabs open and a scene loaded.
```bash
source simlingo_env/bin/activate
python tests/run_basic_integration.py --duration 20 --hz 10 \
  --spawn-x 0 --spawn-y 0 --spawn-z 0.12 --spawn-yaw 90
```
CLI options:
- `--duration` (s), `--hz` (loop rate)
- `--try-load-weights` (optional, loads additional large weights if available)
- Spawn overrides: `--spawn-x`, `--spawn-y`, `--spawn-z`, `--spawn-yaw`
- `--no-autosearch` to disable safe‑spawn auto‑search

## Control Loop Architecture
QLabs → QCar2 camera → image preprocessing → SimLingo VLA inference → waypoint→control → QCar2 commands
- Data: CSI/RGB camera → resize/normalize to model input
- Model: InternVL2‑1B + PEFT (LoRA) → loads epoch=013.ckpt → predicts waypoints/route
- Control: waypoints → forward speed & steering via proportional mapping → `set_velocity_and_request_state`
- Safety: on `ModelInferenceError`, stop vehicle and exit (no fallback policy)

## Spawn System
- Default spawn: `(x=0.0, y=0.0, z=0.1, yaw=90°)`
- Auto‑search: tries several nearby safe candidates if bumpers report collision
- Override exact pose with CLI for your layout

## Recent Changes
- Switched to CUDA 12.8 stack (torch/vision/audio +cu128)
- Real SimLingo model path wired; removed all fallback/handcrafted driving
- Correct InternVL2 chat template and image token handling
- Robust preprocessing to avoid dtype/shape mismatches
- Graceful failure: stop on inference errors
- Spawn auto‑search and CLI overrides for track‑aligned placement

## More Documentation
See docs/ for detailed architecture, setup, usage, troubleshooting, and change log.
