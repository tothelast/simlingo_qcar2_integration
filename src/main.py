#!/usr/bin/env python3
"""
Entry point for SimLingo-Qcar2 minimal integration demo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the repository root is on sys.path so `src` is importable
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.integration.main_bridge import run_cli


def main():
    p = argparse.ArgumentParser(description="Run SimLingo-Qcar2 integration demo")
    p.add_argument("--hz", type=float, default=10.0, help="Control loop frequency")
    p.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    p.add_argument(
        "--try-load-weights",
        action="store_true",
        help="Attempt to load the large checkpoint weights into CPU memory (optional)",
    )
    args = p.parse_args()
    return run_cli(hz=args.hz, duration=args.duration, try_load_weights=args.try_load_weights)


if __name__ == "__main__":
    raise SystemExit(main())

