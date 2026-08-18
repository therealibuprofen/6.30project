#!/usr/bin/env python3
"""Run the CPU-only read-only ROI artifact and provenance audit v1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.roi_artifact_audit_v1 import RUN_NAME, run_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_audit(args.project_root, args.output_dir, device=args.device)
    print(json.dumps({"run_name": RUN_NAME, "status": "PASS", **summary}, indent=2))


if __name__ == "__main__":
    main()
