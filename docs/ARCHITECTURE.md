# Architecture and Integration Details

This document explains how the QLabs ↔ QCar2 ↔ SimLingo VLA system is wired in this repository.

## Components Overview
- Integration bridge: `src/integration/main_bridge.py`
- SimLingo wrapper (real model): `src/models/simlingo_wrapper.py`
- Data adapter (camera → model input): `src/adapters/data_adapter.py`
- Control adapter (waypoints → QCar2 controls): `src/adapters/control_adapter.py`

## End-to-end Flow
1) QLabs connection and QCar2 spawn
   - Uses QVL (Quanser) Python API from `0_libraries/python/qvl`
   - Spawns the QCar2 and attempts a safe transform (position + yaw) with auto-search fallback
2) Camera acquisition
   - Front CSI camera by default (820x410), RGB camera also supported (640x480)
   - Frame is retrieved via QVL and decoded to an RGB ndarray
3) Preprocessing
   - Resize to model input, convert to float32, normalization
   - InternVL2-specific transform pipeline builds a single 448 patch per frame for stability
4) Prompt & Tokens
   - Uses SimLingo’s InternVL2 utilities to form a chat-style prompt
   - Expands image tokens to the expected block of IMG_CONTEXT tokens
5) Model inference (real SimLingo)
   - InternVL2-1B (vision-language) with LoRA (PEFT) is instantiated
   - Checkpoint epoch=013.ckpt is loaded (state_dict into assembled components)
   - Adaptors produce driving outputs (predicted waypoints / route)
6) Waypoint → control
   - The control adapter applies simple proportional logic to compute forward speed and steering from the nearest predicted waypoint(s)
7) Command sending
   - Commands are sent to QCar2 via `set_velocity_and_request_state()` every cycle
8) Failure handling
   - Any model failure raises ModelInferenceError; the vehicle is stopped and the loop exits

## Spawn System
- Default pose: `(x=0.0, y=0.0, z=0.1, yaw=90°)`
- After spawn, the code attempts the configured pose; if bumpers hit (collision), a small list of nearby candidate poses is tried
- CLI overrides are supported for precise scene alignment

## Control Loop
- Frequency: configurable (`--hz`), default 10 Hz
- Duration: configurable (`--duration`)
- Anti-stall: if front bumper hit is detected, a short “unstick” maneuver is performed (reverse + small steer) before stopping; this is integration-side safety, not model policy

## Code Pointers (short excerpts)
- Spawn (degrees-based yaw):
  - `QLabsQCar2.set_transform_and_request_state_degrees([... rotation=[0,0,yaw_deg] ...])`
- Model call site:
  - `out = self.model.inference(model_input)`
- Control mapping:
  - `fwd, turn = self.control_adapter.process_simlingo_output(out)`

## Notes on Domain Gap
- SimLingo was trained on CARLA; QLabs scenery, textures, and camera intrinsics differ
- Expect conservative movement, hesitation, or non-lane-following until domain adaptation is addressed
- Improving behavior typically requires fine-tuning with QLabs data or stronger priors in the prompt

