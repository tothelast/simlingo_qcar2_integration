# SimLingo + QCar2 Integration — Updated Summary

## Overview
This repository runs the real SimLingo VLA stack in QLabs on a QCar2:
- InternVL2-1B backbone with PEFT (LoRA)
- Loads epoch=013.ckpt state dict into the assembled architecture
- End-to-end: QLabs -> QCar2 camera -> preprocessing -> SimLingo inference -> waypoint-to-control -> vehicle commands
- Graceful failure: on model errors, the vehicle is stopped and the loop exits (no fallback driving)

## What We Implemented
- Real SimLingo inference in `src/models/simlingo_wrapper.py` (no dummy policy)
- Robust prompt and image token handling for InternVL2 chat template
- PyTorch CUDA 12.8 stack (torch/vision/audio +cu128) for RTX 5070 acceleration
- Safe spawn system with auto-search + CLI overrides in `tests/run_basic_integration.py`
- Control adapter that converts predicted waypoints to QCar2 commands

## Architecture (Data Flow)
1) Camera acquisition (QVL): CSI/RGB camera frame captured from QCar2
2) Preprocessing (data_adapter): resize/normalize to model input; 448-sized patch with correct dtype
3) Prompt & tokens: InternVL2 chat template with image token block (IMG_CONTEXT)
4) Model inference (simlingo_wrapper):
   - Vision encoder (InternVL2) produces image features
   - Language model (LLM with LoRA) integrates image+text context
   - Adaptors produce driving outputs (waypoints/route)
5) Output mapping (control_adapter): waypoints -> forward speed & steering
6) Actuation (QVL): `set_velocity_and_request_state()` applied each cycle

## Environment & Dependencies
- Python venv: `simlingo_env/`
- PyTorch: 2.8.0+cu128; torchvision 0.23.0+cu128; torchaudio 2.8.0+cu128
- Transformers, OpenCV, Pillow, NumPy
- hydra-core 1.3.2, omegaconf 2.3.0, pytorch-lightning >= 2.4, einops, timm, peft
- Optional: FlashAttention2 (not required; logs show it is not installed)

## Model Assets
- SimLingo training repo path: `../simlingo`
- Checkpoint: `models/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt`
- HuggingFace model: `OpenGVLab/InternVL2-1B` auto-downloaded on first run

## Running
With QLabs open and a scene loaded:
```bash
source simlingo_env/bin/activate
python tests/run_basic_integration.py --duration 20 --hz 10 \
  --spawn-x 0 --spawn-y 0 --spawn-z 0.12 --spawn-yaw 90
```
Flags:
- Spawn: `--spawn-x --spawn-y --spawn-z --spawn-yaw` (degrees)
- `--no-autosearch` to disable safe-spawn candidate search
- `--try-load-weights` for heavier optional weight loads

## Spawn System
- Default: (0.0, 0.0, 0.1) @ yaw 90°; uses `set_transform_and_request_state_degrees`
- Auto-search: tries several nearby poses if bumper hit flags indicate a collision at spawn

## Safety & Failure Behavior
- Any `ModelInferenceError` triggers a safe stop (0 forward, 0 steering) and exits the control loop
- No handcrafted fallback policy is used at any time

## Recent Changes
- CUDA 12.8 stack (cu128) and dependency fixes (einops, timm, peft)
- InternVL2 tokenization & chat template corrected (IMG_CONTEXT handling)
- Hard failure on init/inference errors (no fallback); explicit stopping on exceptions
- Spawn defaults switched to yaw=90° with safe auto-search candidates
- CLI spawn overrides added to the test runner

## Logs (examples)
- Model ready: `SimLingo model ready (real VLA inference; no fallback)`
- Spawn pose: `Repositioned QCar2 to [...] yaw=90.0° (front_hit=False, rear_hit=False)`
- GPU use: torch 2.8.0+cu128; device is RTX 5070
