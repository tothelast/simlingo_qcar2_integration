# SimLingo ↔ QCar2 (QLabs) Integration

Real SimLingo Vision–Language–Action (VLA) inference driving a QCar2 in QLabs. GPU‑accelerated on RTX 5070 (CUDA 12.8).

## Highlights
- True neural inference: InternVL2‑1B backbone with LoRA (PEFT), loading epoch=013.ckpt
- End‑to‑end pipeline: QLabs → QCar2 camera → preprocessing → SimLingo VLA → waypoint→control → vehicle commands

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
│   |    └── main_bridge.py          # QLabs orchestration + control loop
|   └── main.py                      # Entry point
├── models/
    └── simlingo/
        └── checkpoints/epoch=013.ckpt/pytorch_model.pt
```

## Requirements (GPU stack)
- Ubuntu 24.04, Python 3.12 (project uses venv: `simlingo_env/`)
- NVIDIA GPU with CUDA 12.8 support (tested on RTX 5070, works with other CUDA-capable GPUs)
- Core ML/AI: PyTorch 2.7+, transformers, pytorch-lightning, peft
- Computer Vision: opencv-python, pillow, numpy
- Configuration: hydra-core, omegaconf
- Utilities: einops, timm, huggingface-hub, safetensors

## Setup

### 1) Create and activate virtual environment
```bash
python3 -m venv simlingo_env
source simlingo_env/bin/activate
```

### 2) Install PyTorch with CUDA support
```bash
# Install PyTorch with CUDA 12.8 support first
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3) Install remaining requirements
```bash
# Install all other dependencies
pip install -r requirements.txt
```

### 4) Verify GPU detection
```bash
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### 5) Check model assets
The project loads models from these locations:

**SimLingo Checkpoint** (required):
- Path: `models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- Config: `models/simlingo/.hydra/config.yaml`
- Status: ✅ Present in repository

**InternVL2-1B Pretrained Model** (auto-downloaded):
- Path: `pretrained/InternVL2-1B/`
- Source: Auto-downloaded from HuggingFace (`OpenGVLab/InternVL2-1B`)
- Status: ✅ Present in repository
- Size: ~2GB (includes model.safetensors, tokenizer, config files)

**SimLingo Training Code** (required):
- Path: `simlingo_training/` directory
- Status: ✅ Present in repository
- Contains: Model architectures, utilities, data loaders

## Usage
Run with QLabs open and a scene loaded.
```bash
source simlingo_env/bin/activate
python src/main.py --duration 30 --hz 10 --try-load-weights
```
CLI options:
- `--duration` (s): How long to run the simulation (default: 30)
- `--hz` (float): Control loop frequency (default: 10.0)
- `--try-load-weights`: Attempt to load the large checkpoint weights into CPU memory (optional)


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

