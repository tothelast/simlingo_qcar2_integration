#!/usr/bin/env python3
"""
Basic end-to-end validation of the SimLingo-Qcar2 integration in QLabs.

This script performs:
- Model loading verification
- Camera acquisition + preprocessing
- Model inference (fallback policy)
- Control command generation and sending

Run QLabs, open a layout, then execute:
  source simlingo_env/bin/activate
  python tests/run_basic_integration.py --duration 20
"""
from __future__ import annotations

import sys
from pathlib import Path
import argparse
import logging

# Make src importable
HERE = Path(__file__).resolve()
SRC_PATH = HERE.parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from integration.main_bridge import run_cli
from models.simlingo_wrapper import SimLingoModel


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run basic SimLingo-Qcar2 integration test")
    parser.add_argument("--duration", type=float, default=20.0, help="Run duration (s)")
    parser.add_argument("--hz", type=float, default=10.0, help="Control loop frequency")
    parser.add_argument("--try-load-weights", action="store_true", help="Attempt to load large weights (optional)")
    # Spawn overrides
    parser.add_argument("--spawn-x", type=float, default=None, help="Spawn X (meters)")
    parser.add_argument("--spawn-y", type=float, default=None, help="Spawn Y (meters)")
    parser.add_argument("--spawn-z", type=float, default=None, help="Spawn Z (meters)")
    parser.add_argument("--spawn-yaw", type=float, default=None, help="Spawn yaw (degrees)")
    parser.add_argument("--no-autosearch", action="store_true", help="Disable auto-search for safe spawn")
    args = parser.parse_args()

    # Step 1: Model presence check
    model = SimLingoModel(model_root=HERE.parents[1] / "models" / "simlingo")
    if not model.load(try_load_weights=args.try_load_weights):
        logging.error("Model loading verification failed.")
        return 2
    logging.info("Model loading verification: OK")

    # Step 2-4: End-to-end via main bridge
    rc = run_cli(
        hz=args.hz,
        duration=args.duration,
        try_load_weights=args.try_load_weights,
        spawn_x=args.spawn_x,
        spawn_y=args.spawn_y,
        spawn_z=args.spawn_z,
        spawn_yaw=args.spawn_yaw,
        spawn_autosearch=(not args.no_autosearch),
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

