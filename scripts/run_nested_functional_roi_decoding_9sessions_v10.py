#!/usr/bin/env python3
"""Run nested training-fold functional ROI decoding v10."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultrasound_decoding.multiframe.dataset import BlockSequenceData, load_block_sequence_session
from ultrasound_decoding.nested_functional_roi_v10 import (
    FIXED_ORIENTATIONS, MAX_FOLDS, MODELS, ROI_RULES, RUN_NAME, SESSIONS,
    config_freeze_text, run_formal, run_session,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/nested_functional_roi_decoding_9sessions_v10.json"
DEFAULT_DATA = PROJECT_ROOT / "processed_data/block_sequences_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / RUN_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-output-dir", type=Path, default=PROJECT_ROOT / "outputs/block_clean4_binary_all_models_9sessions_v1")
    parser.add_argument("--v9-output-dir", type=Path, default=PROJECT_ROOT / "outputs/spatial_glm_contrast_reproducibility_9sessions_v9")
    parser.add_argument("--smoke-sessions", nargs=2, default=("626", "708"))
    parser.add_argument("--smoke-cycles", type=int, default=3)
    return parser.parse_args()


def validate_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "run_name": RUN_NAME, "sessions": list(SESSIONS), "task": "binary",
        "input_shape": [4, 128, 501], "orientation": FIXED_ORIENTATIONS,
        "max_folds": MAX_FOLDS, "eps": 1e-8, "roi_rules": ROI_RULES,
        "pca_variance": 0.95, "models": list(MODELS),
        "user_drawn_roi_used": False, "full_session_roi_used": False,
        "v9_map_used_for_roi": False, "cross_session_transfer": False,
        "registration": False, "morphology": False, "formal_device": "cpu",
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise AssertionError(f"config mismatch for {key}: {value.get(key)!r} != {expected_value!r}")
    return value


def subset_cycles(data: BlockSequenceData, n_cycles: int) -> BlockSequenceData:
    cycles = np.unique(data.groups)[:int(n_cycles)]
    keep = np.isin(data.groups, cycles)
    metadata = data.metadata.loc[keep].reset_index(drop=True)
    return BlockSequenceData(
        session=data.session, task=data.task, X=data.X[keep], y=data.y[keep], groups=data.groups[keep],
        metadata=metadata, clean4_relative_time_s=data.clean4_relative_time_s[keep],
        clean4_original_frame_indices=data.clean4_original_frame_indices[keep],
        source_h5_path=data.source_h5_path, source_metadata_path=data.source_metadata_path,
    )


def run_smoke(args: argparse.Namespace) -> None:
    if args.smoke_cycles not in (2, 3):
        raise ValueError("smoke cycles must be 2 or 3")
    rows = []
    for session in map(str, args.smoke_sessions):
        data = subset_cycles(load_block_sequence_session(args.project_root, session, "binary", data_dir=args.data_dir), args.smoke_cycles)
        result = run_session(
            data, models=("whole_brain_clean4_flat4_pca_lda", "roi_mean4_top10_rlda"),
            roi_rules={"functional_roi_top10": 0.10}, max_folds=args.smoke_cycles,
            baseline_predictions=None,
        )
        rows.extend(result.summary_rows)
        if any(row["test_cycles_used_for_roi"] for row in result.roi_audit_rows):
            raise AssertionError("smoke leakage audit failed")
    smoke_dir = args.output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(smoke_dir / "smoke_summary.csv", index=False)
    (smoke_dir / "SMOKE_NOT_SCIENTIFIC.txt").write_text(
        "SMOKE PASS: two sessions, 2-3 cycles, top10, whole-brain plus ROI mean4 only. NOT A FORMAL RESULT.\n",
        encoding="utf-8",
    )
    print("SMOKE PASS: two sessions; nested train-cycle ROI selection, whole-brain and roi_mean4 top10 completed")


def log(output_dir: Path, message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with (output_dir / "run_log_server.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    validate_config(args.config)
    if args.device != "cpu":
        raise RuntimeError("v10 is CPU-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        run_smoke(args)
        return
    log(args.output_dir, "FORMAL v10 START: CPU, nine sessions, nested training-fold functional ROI")
    result = run_formal(
        project_root=args.project_root, data_dir=args.data_dir, output_dir=args.output_dir,
        config_path=args.config, baseline_root=args.baseline_output_dir, v9_root=args.v9_output_dir,
    )
    log(args.output_dir, "FORMAL v10 PASS: " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
