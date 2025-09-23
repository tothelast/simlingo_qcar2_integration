# Troubleshooting

## The car spawns off-track or collides immediately
- Use spawn CLI overrides (x, y, z, yaw) to place the car on-track
- Auto-search is enabled by default; it will try a few nearby candidates if the initial pose collides
- Provide a known-good pose from your scene and make it the default

## The car moves slowly or appears to wander
- SimLingo was trained on CARLA; QLabs differs in visuals and camera geometry (domain gap)
- The default prompt is generic ("follow the road; predict the waypoints")
- Waypoint-to-control mapping is conservative and prioritizes safety
- Improvement options:
  - Refine the prompt with scene-specific guidance
  - Increase maximum forward speed in the control adapter (with caution)
  - Fine-tune the model with QLabs data for better lane following

## Repeated "Front bumper hit" messages
- The control loop includes a basic unstick maneuver (reverse briefly, small steer) if the front bumper hit flag is raised
- This is integration-side safety (not model policy). Adjust the spawn yaw/pose or provide a wider start area

## GPU not detected / wrong CUDA
- Ensure torch 2.8.0+cu128 and torchvision 0.23.0+cu128 are installed in `simlingo_env`
- Confirm `torch.cuda.is_available()` returns True and the GPU name is shown
- If needed, reinstall torch/vision/torchaudio matching CUDA 12.8 wheels

## Missing packages (einops/timm/peft)
- Install with:
```bash
pip install einops timm peft
```

## HuggingFace model downloads are slow or blocked
- Preload `OpenGVLab/InternVL2-1B` via another machine and copy the cache
- Set the `TRANSFORMERS_CACHE` environment variable to a shared cache directory
- Verify credentials if the repo requires a token

## Where are errors handled?
- Model init/inference errors raise `ModelInferenceError`
- The main loop catches it, sends a safe stop command, and exits (no fallback)

