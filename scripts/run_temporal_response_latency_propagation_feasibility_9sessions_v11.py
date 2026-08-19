#!/usr/bin/env python3
"""Run discrete-frame temporal response-latency feasibility analysis v11."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrasound_decoding.temporal_latency_v11 import (
    CONSECUTIVE_FRAMES,
    FIXED_ORIENTATIONS,
    FRAMES_PER_CYCLE,
    IMAGE_SHAPE,
    MIN_COMMON_PATCHES,
    N_SPLITS,
    ONSET_CRITERIA,
    PATCH_SIZE,
    PEAK_WINDOW_SECONDS,
    RANDOM_SEED,
    RUN_NAME,
    SESSIONS,
    STABLE_DETECTION_RATE,
    STABLE_ONSET_IQR,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    run_analysis,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/temporal_response_latency_propagation_feasibility_9sessions_v11.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / RUN_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--v9-output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/spatial_glm_contrast_reproducibility_9sessions_v9",
    )
    parser.add_argument("--smoke-sessions", nargs=2, default=("626", "708"))
    parser.add_argument("--smoke-cycles", type=int, default=3)
    parser.add_argument("--smoke-patches", type=int, default=12)
    parser.add_argument("--smoke-splits", type=int, default=10)
    return parser.parse_args()


def validate_config(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "run_name": RUN_NAME,
        "sessions": list(SESSIONS),
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "image_shape": list(IMAGE_SHAPE),
        "frames_per_cycle": FRAMES_PER_CYCLE,
        "orientation": FIXED_ORIENTATIONS,
        "patch_size": list(PATCH_SIZE),
        "primary_onset_abs_z": ONSET_CRITERIA["PRIMARY_Z2"],
        "sensitivity_onset_abs_z": ONSET_CRITERIA["SENSITIVITY_Z1_5"],
        "consecutive_frames": CONSECUTIVE_FRAMES,
        "peak_window_seconds": list(PEAK_WINDOW_SECONDS),
        "stable_detection_rate": STABLE_DETECTION_RATE,
        "stable_onset_iqr_frames": STABLE_ONSET_IQR,
        "n_splits": N_SPLITS,
        "minimum_common_patches": MIN_COMMON_PATCHES,
        "random_seed": RANDOM_SEED,
        "external_threshold_0_4_used": False,
        "subframe_fitting": False,
        "temporal_interpolation": False,
        "hrf_fitting": False,
        "registration": False,
        "roi_selection": False,
        "decoder_training": False,
        "formal_device": "cpu",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AssertionError(f"config mismatch for {key}: {value.get(key)!r} != {expected_value!r}")
    return value


def log(path: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    validate_config(args.config)
    if args.device != "cpu":
        raise RuntimeError("v11 formal analysis is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        if args.smoke_cycles not in (2, 3):
            raise ValueError("smoke cycles must be 2 or 3")
        if not 1 <= args.smoke_patches <= 32 or not 1 <= args.smoke_splits <= 50:
            raise ValueError("smoke patch/split limits are outside the fixed small range")
        smoke_dir = args.output_dir / "smoke"
        result = run_analysis(
            data_root=args.data_root, output_dir=smoke_dir, config_path=args.config,
            v9_root=args.v9_output_dir, sessions=tuple(map(str, args.smoke_sessions)),
            max_cycles=args.smoke_cycles, patch_limit=args.smoke_patches,
            n_splits=args.smoke_splits, require_nine_sessions=False,
        )
        (smoke_dir / "SMOKE_NOT_SCIENTIFIC.txt").write_text(
            "SMOKE PASS: limited sessions/cycles/patches/splits; schema only; NOT A SCIENTIFIC RESULT.\n",
            encoding="utf-8",
        )
        print("SMOKE PASS: " + json.dumps(result, sort_keys=True), flush=True)
        return
    run_log = args.output_dir / "run_log_server.txt"
    log(run_log, "FORMAL v11 START: CPU; nine sessions; complete-cycle discrete-frame latency")
    result = run_analysis(
        data_root=args.data_root, output_dir=args.output_dir, config_path=args.config,
        v9_root=args.v9_output_dir, sessions=SESSIONS, n_splits=N_SPLITS,
        require_nine_sessions=True,
    )
    log(run_log, "FORMAL v11 PASS: " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
