# Usage Guide

With QLabs open and a QCar2 scene loaded:

## Basic run
```bash
source simlingo_env/bin/activate
python tests/run_basic_integration.py --duration 20 --hz 10
```

## Spawn pose overrides
If the initial pose collides or points off-track, supply your own pose:
```bash
python tests/run_basic_integration.py --duration 20 --hz 10 \
  --spawn-x 0 --spawn-y 0 --spawn-z 0.12 --spawn-yaw 90
```
Flags:
- `--spawn-x`, `--spawn-y`, `--spawn-z` (meters, world frame)
- `--spawn-yaw` (degrees, roll/pitch fixed at 0)
- `--no-autosearch` to disable safe spawn candidate search

## Other flags
- `--try-load-weights` enables optional heavy weight loads (not required for the default run)

## Expected logs
- Model readiness: `SimLingo model ready (real VLA inference; no fallback)`
- Spawn confirmation with bumper flags: `Repositioned QCar2 to [...] yaw=90.0° (front_hit=False, rear_hit=False)`
- Control loop start: `Starting control loop at 10.0 Hz for 20.0 s`

## How to stop
- Ctrl+C in the terminal, or wait for `--duration` to elapse

## No fallback behavior
- If model inference fails, the vehicle is stopped and the loop exits automatically. There is no handcrafted driving policy in this integration.

