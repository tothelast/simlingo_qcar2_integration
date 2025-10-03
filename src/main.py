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
    p.add_argument("--hz", type=float, default=5.0, help="Control loop frequency (default: 5 Hz)")
    p.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    p.add_argument("--show_agents_comments", action="store_true", help="Show model's reasoning/commentary")
    p.add_argument("--show_current_instruction", action="store_true", help="Show current instruction")
    args = p.parse_args()
    return run_cli(hz=args.hz, duration=args.duration, show_agents_comments=args.show_agents_comments, show_current_instruction=args.show_current_instruction)


if __name__ == "__main__":
    raise SystemExit(main())

