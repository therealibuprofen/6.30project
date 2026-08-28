#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
for search_path in (PROJECT_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.baselines import run_multiscale_temporal1d as framework
from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.canonical_single_frame import (
    EXPECTED_EPOCH,
    EXPECTED_IMAGE_SHAPE,
    EXPECTED_PARAMETERS,
    HISTORICAL_BASE_MODEL,
    HISTORICAL_METHOD,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    NORMALIZATION_TRANSFORM,
    TEMPORAL_MIDPOINT_S,
    apply_saved_normalization,
    file_sha256,
    load_validated_checkpoint,
    predict_single_frame_probabilities,
    reconstruct_late_fusion_probabilities,
    select_canonical_frames,
    select_canonical_positions,
)
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
)


OUTPUT_VERSION = "fcnn_canonical_single_frame_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
EXPECTED_FOLDS = 82
EXPECTED_TASKS = 246
EXPECTED_BLOCKS = 456
EXPECTED_TEST_FORWARDS = EXPECTED_BLOCKS * len(SEEDS)
EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS = 11_640
EXPECTED_CANONICAL_TOTAL_FORWARDS = (
    EXPECTED_TEST_FORWARDS + EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
)
EXPECTED_LATE_FUSION_FRAME_FORWARDS = 4 * EXPECTED_TEST_FORWARDS
EXPECTED_TOTAL_FRAME_FORWARDS = (
    EXPECTED_CANONICAL_TOTAL_FORWARDS + EXPECTED_LATE_FUSION_FRAME_FORWARDS
)
LATE_FUSION_PROBABILITY_ATOL = 2e-6
LATE_FUSION_PROBABILITY_RTOL = 1e-6
MEANPOOL_OUTPUT_VERSION = "fcnn_mean_std_temporal_statistics_v1"
MEANPOOL_MODEL_NAME = "fcnn_bottleneck_temporal_statistics"
MEANPOOL_MODEL_IMPLEMENTATION_VERSION = (
    "fcnn_mean_std_temporal_statistics_v1.0.0"
)
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
TIE_TOLERANCE = 1e-12
REQUIRED_FINAL_OUTPUTS = (
    "config.json",
    "task_plan.csv",
    "checkpoint_manifest.csv",
    "normalization_audit.csv",
    "canonical_frame_manifest.csv",
    "predictions.csv",
    "late_fusion_reconstructed_predictions.csv",
    "late_fusion_reconstruction_audit.csv",
    "late_fusion_reconstruction_audit.json",
    "fold_summary.csv",
    "session_seed_summary.csv",
    "session_summary.csv",
    "pairwise_comparison.csv",
    "statistical_audit.json",
    "provenance_audit.json",
    "fcnn_canonical_single_frame_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only reconstruction of canonical-midpoint single-frame FCNN "
            "predictions from frozen historical fcnn_late_fusion checkpoints."
        )
    )
    parser.add_argument(
        "--stage", choices=("plan", "sanity", "full", "status"), required=True
    )
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / OUTPUT_VERSION,
    )
    parser.add_argument(
        "--historical-base-run-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/runs/multiframe/block_clean4_binary_v1",
        help="Historical source for sessions 626 and 628.",
    )
    parser.add_argument(
        "--historical-fcnn-run-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/runs/multiframe/block_clean4_binary_fcnn_v1",
        help="Historical source for sessions 708/709/710/807/813/817/822.",
    )
    parser.add_argument(
        "--historical-aggregate-dir",
        type=Path,
        default=PROJECT_DIR
        / "results/runs/multiframe/block_clean4_binary_all_models_9sessions_v1/aggregate",
    )
    parser.add_argument(
        "--meanpool-run-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/fcnn_mean_std_temporal_statistics_v1",
    )
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--synthetic-sanity", action="store_true")
    parser.add_argument(
        "--review-approved",
        action="store_true",
        help="Required for the formal 246-task CPU reconstruction.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def source_paths(project_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        project_root
        / "src/ultrasound_decoding/multiframe/canonical_single_frame.py",
        project_root / "src/ultrasound_decoding/multiframe/models.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/deep.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "configs/fcnn_canonical_single_frame_v1.json",
        project_root / "docs/fcnn_canonical_single_frame_v1.md",
        project_root
        / "scripts/baselines/run_fcnn_mean_std_temporal_statistics.py",
        project_root / "scripts/baselines/run_multiscale_temporal1d.py",
    ]


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "model": MODEL_NAME,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "scientific_name": (
            "Canonical-midpoint single-frame FCNN with frame-wise clean4 training"
        ),
        "paper_wording": (
            "Berthon-style single-frame FCNN adapted to the same clean4 "
            "frame-wise training pool; not a full Berthon training reproduction"
        ),
        "task": "binary_presence",
        "sessions": list(EXPECTED_SESSIONS),
        "seeds": list(SEEDS),
        "expected_folds": EXPECTED_FOLDS,
        "expected_tasks": EXPECTED_TASKS,
        "expected_test_block_forwards": EXPECTED_TEST_FORWARDS,
        "expected_train_diagnostic_block_forwards": (
            EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
        ),
        "expected_canonical_block_forwards": EXPECTED_CANONICAL_TOTAL_FORWARDS,
        "expected_late_fusion_verification_frame_forwards": (
            EXPECTED_LATE_FUSION_FRAME_FORWARDS
        ),
        "expected_total_model_frame_forwards": EXPECTED_TOTAL_FRAME_FORWARDS,
        "device": "cpu_only",
        "training_performed": False,
        "weights_updated": False,
        "checkpoint_method": HISTORICAL_METHOD,
        "checkpoint_base_model": HISTORICAL_BASE_MODEL,
        "checkpoint_epoch": EXPECTED_EPOCH,
        "checkpoint_parameters": EXPECTED_PARAMETERS,
        "checkpoint_training": (
            "historical clean4 frames treated as independent frame samples"
        ),
        "canonical_rule": {
            "formula": "argmin_k(abs(clean4_relative_time_s[k] - 15.0))",
            "midpoint_s": TEMPORAL_MIDPOINT_S,
            "tie_break": "earlier_timestamp",
            "uses_block_label": False,
            "uses_prediction": False,
            "uses_balanced_accuracy": False,
        },
        "normalization": {
            "transform": NORMALIZATION_TRANSFORM,
            "statistics": "checkpoint-saved outer-training-fold mean/std",
            "fit_or_recomputed_for_inference": False,
            "test_used_for_fit": False,
        },
        "canonical_inference": (
            "one canonical frame -> historical FCNN -> one probability vector; "
            "no logit/probability averaging, vote, temporal mean/std, or fusion"
        ),
        "late_fusion_verification": (
            "same checkpoint and saved normalization -> independently forward all "
            "four clean4 frames -> softmax per frame -> arithmetic mean of four "
            "probability vectors; must match the historical aggregate"
        ),
        "evaluation_unit": "one held-out block equals one prediction",
        "oof_aggregation": (
            "concatenate all outer-held-out block predictions within session/seed, "
            "then compute Balanced Accuracy"
        ),
        "train_accuracy_diagnostic": (
            "canonical block-level accuracy on each outer training fold; not the "
            "historical frame-wise training objective accuracy"
        ),
        "comparators": {
            "late_fusion": (
                "same-checkpoint reconstructed fcnn_late_fusion OOF predictions; "
                "historical aggregate retained as a validated external reference"
            ),
            "meanpool": (
                "current fcnn_mean_std_temporal_statistics_v1 mean_only OOF block predictions"
            ),
        },
        "primary_comparison": "meanpool_vs_canonical_single_frame",
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "automatic_next_stage": False,
    }


def data_dir(args: argparse.Namespace) -> Path:
    return (
        args.data_dir
        if args.data_dir is not None
        else default_block_data_dir(args.project_root)
    )


def load_formal_task_plan(args: argparse.Namespace) -> pd.DataFrame:
    path = args.meanpool_run_dir / "task_plan.csv"
    if not path.is_file():
        raise FileNotFoundError(f"current formal task plan is missing: {path}")
    raw = pd.read_csv(path, dtype={"session": str})
    plan = raw[raw["variant"].eq("mean_only")].copy()
    plan["session"] = plan["session"].astype(str)
    plan["seed"] = pd.to_numeric(plan["seed"]).astype(int)
    plan["fold"] = pd.to_numeric(plan["fold"]).astype(int)
    plan = plan.sort_values(["session", "seed", "fold"]).reset_index(drop=True)
    required = {
        "session",
        "seed",
        "fold",
        "n_train_samples",
        "n_test_samples",
        "train_cycles",
        "test_cycles",
    }
    if not required.issubset(plan.columns):
        raise AssertionError("current formal task plan lacks required columns")
    if len(plan) != EXPECTED_TASKS:
        raise AssertionError(f"expected {EXPECTED_TASKS} tasks, got {len(plan)}")
    if plan[["session", "seed", "fold"]].duplicated().any():
        raise AssertionError("formal task plan contains duplicate task keys")
    if set(plan["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("formal task plan session coverage differs")
    if set(plan["seed"]) != set(SEEDS):
        raise AssertionError("formal task plan seed coverage differs")
    fold_keys = plan[["session", "fold"]].drop_duplicates()
    if len(fold_keys) != EXPECTED_FOLDS:
        raise AssertionError(f"expected {EXPECTED_FOLDS} folds, got {len(fold_keys)}")
    if int(plan["n_test_samples"].sum()) != EXPECTED_TEST_FORWARDS:
        raise AssertionError("formal test block forward count differs")
    if (
        int(plan["n_train_samples"].sum())
        != EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
    ):
        raise AssertionError("formal train diagnostic forward count differs")
    plan["task_key"] = plan.apply(
        lambda row: f"{row.session}:{int(row.seed)}:{int(row.fold)}", axis=1
    )
    return plan[
        [
            "session",
            "seed",
            "fold",
            "n_train_samples",
            "n_test_samples",
            "train_cycles",
            "test_cycles",
            "task_key",
        ]
    ]


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        bytes(value).decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def build_canonical_frame_manifest(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        h5_path = data_dir(args) / f"session_{session}_blocks.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        with h5py.File(h5_path, "r") as handle:
            times = handle["/clean4/relative_time_s"][:]
            indices = handle["/clean4/original_frame_indices"][:]
            cycles = handle["/metadata/cycle"][:].astype(np.int64)
            blocks = _decode_strings(handle["/metadata/block_name"][:])
            sample_ids = _decode_strings(handle["/metadata/block_id"][:])
        selection = select_canonical_positions(times)
        for row_i in range(len(times)):
            position = int(selection.positions[row_i])
            rows.append(
                {
                    "session": session,
                    "cycle": int(cycles[row_i]),
                    "block": blocks[row_i],
                    "sample_id": sample_ids[row_i],
                    "clean4_original_frame_indices": framework.canonical_json(
                        [int(value) for value in indices[row_i].tolist()]
                    ),
                    "clean4_relative_time_s": framework.canonical_json(
                        [float(value) for value in times[row_i].tolist()]
                    ),
                    "canonical_position": position,
                    "canonical_original_frame_index": int(indices[row_i, position]),
                    "canonical_relative_time_s": float(
                        selection.relative_times_s[row_i]
                    ),
                    "distance_to_midpoint_s": float(
                        selection.distances_to_midpoint_s[row_i]
                    ),
                    "tie": bool(selection.ties[row_i]),
                }
            )
    manifest = pd.DataFrame(rows).sort_values(
        ["session", "cycle", "block"]
    ).reset_index(drop=True)
    if len(manifest) != EXPECTED_BLOCKS:
        raise AssertionError(
            f"expected {EXPECTED_BLOCKS} canonical blocks, got {len(manifest)}"
        )
    if manifest["sample_id"].duplicated().any():
        raise AssertionError("canonical manifest contains duplicate block IDs")
    if manifest["canonical_original_frame_index"].isna().any():
        raise AssertionError("a block lacks a canonical frame")
    return manifest


def write_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = load_formal_task_plan(args)
    canonical = build_canonical_frame_manifest(args)
    config = experiment_config(args)
    framework.atomic_json(args.output_dir / "config.json", config)
    framework.atomic_csv(args.output_dir / "task_plan.csv", plan)
    framework.atomic_csv(
        args.output_dir / "canonical_frame_manifest.csv", canonical
    )
    framework.atomic_json(
        args.output_dir / "PLAN_COMPLETE.json",
        {
            "status": "complete",
            "training_started": False,
            "formal_inference_started": False,
            "expected_tasks": EXPECTED_TASKS,
            "expected_folds": EXPECTED_FOLDS,
            "expected_test_block_forwards": EXPECTED_TEST_FORWARDS,
            "expected_train_diagnostic_block_forwards": (
                EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
            ),
            "expected_canonical_block_forwards": EXPECTED_CANONICAL_TOTAL_FORWARDS,
            "expected_late_fusion_verification_frame_forwards": (
                EXPECTED_LATE_FUSION_FRAME_FORWARDS
            ),
            "expected_total_model_frame_forwards": EXPECTED_TOTAL_FRAME_FORWARDS,
            "task_plan_sha256": file_sha256(args.output_dir / "task_plan.csv"),
            "canonical_frame_manifest_sha256": file_sha256(
                args.output_dir / "canonical_frame_manifest.csv"
            ),
            "created_utc": utc_now(),
        },
    )
    print(
        f"PLAN COMPLETE tasks={len(plan)} folds={EXPECTED_FOLDS} "
        f"canonical_blocks={len(canonical)} canonical_block_forwards="
        f"{EXPECTED_CANONICAL_TOTAL_FORWARDS} late_fusion_verification_frames="
        f"{EXPECTED_LATE_FUSION_FRAME_FORWARDS} training_started=False",
        flush=True,
    )
    return plan, canonical


def run_plan(args: argparse.Namespace) -> None:
    """Build the plan and validate every historical asset without inference."""

    plan, _canonical = write_plan(args)
    checkpoint_manifest = validate_all_checkpoints(args, plan)
    historical_late_predictions = validate_late_fusion_reference(args, plan)
    meanpool_predictions, meanpool_provenance = validate_meanpool_reference(
        args, plan
    )
    framework.atomic_csv(
        args.output_dir / "checkpoint_manifest.csv", checkpoint_manifest
    )
    plan_complete_path = args.output_dir / "PLAN_COMPLETE.json"
    complete = json.loads(plan_complete_path.read_text(encoding="utf-8"))
    complete.update(
        {
            "checkpoint_validation_complete": True,
            "checkpoint_coverage": f"{len(checkpoint_manifest)}/{EXPECTED_TASKS}",
            "checkpoint_manifest_sha256": file_sha256(
                args.output_dir / "checkpoint_manifest.csv"
            ),
            "historical_late_fusion_oof_rows": int(
                len(historical_late_predictions)
            ),
            "current_meanpool_oof_rows": int(len(meanpool_predictions)),
            "comparator_provenance_validation_complete": True,
            "meanpool_run_fingerprint": meanpool_provenance["run_fingerprint"],
        }
    )
    framework.atomic_json(plan_complete_path, complete)
    print(
        f"CHECKPOINT PLAN VALIDATED {len(checkpoint_manifest)}/{EXPECTED_TASKS}; "
        "formal_inference_started=False",
        flush=True,
    )


def source_run_dir(args: argparse.Namespace, session: str) -> Path:
    return (
        args.historical_base_run_dir
        if session in {"626", "628"}
        else args.historical_fcnn_run_dir
    )


def checkpoint_path_for_task(
    args: argparse.Namespace, session: str, seed: int, fold: int
) -> Path:
    return (
        source_run_dir(args, session)
        / f"session_{session}/checkpoints/{HISTORICAL_METHOD}"
        / f"seed_{seed}/fold_{fold}/checkpoint.pt"
    )


def validate_source_config(path: Path, expected_session: str) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    deep = config.get("deep_config", {})
    checks = {
        "session": str(config.get("session")) == expected_session,
        "task": config.get("task") == "binary",
        "input_shape": config.get("input_shape") == [4, 128, 501],
        "data_version": config.get("data_version") == "block_sequences_v1_clean4",
        "cv_group": config.get("cv_group") == "cycle",
        "seeds": config.get("seeds") == [0, 1, 2],
        "normalization": config.get("normalization_protocol")
        == "arcsinh_then_train_fold_all_frames_pixel_zscore",
        "optimizer": deep.get("optimizer") == "adamw",
        "lr": float(deep.get("lr", np.nan)) == 1e-3,
        "weight_decay": float(deep.get("weight_decay", np.nan)) == 1e-3,
        "batch_size": int(deep.get("batch_size", -1)) == 16,
        "max_epochs": int(deep.get("max_epochs", -1)) == EXPECTED_EPOCH,
        "loss": deep.get("loss") == "cross_entropy",
    }
    failures = sorted(key for key, value in checks.items() if not value)
    if failures:
        raise AssertionError(
            f"historical source config mismatch for {expected_session}: {failures}"
        )
    return config


def source_checkpoint_rows(
    args: argparse.Namespace, session: str
) -> dict[tuple[int, int], dict[str, str]]:
    run_dir = source_run_dir(args, session) / f"session_{session}"
    validate_source_config(run_dir / "config.json", session)
    manifest_path = run_dir / "checkpoint_manifest.csv"
    rows = pd.read_csv(manifest_path, dtype={"session": str})
    rows = rows[
        rows["method"].eq(HISTORICAL_METHOD) & rows["status"].eq("available")
    ].copy()
    result = {
        (int(row.seed), int(row.fold)): row._asdict()
        for row in rows.itertuples(index=False)
    }
    return result


def validate_all_checkpoints(
    args: argparse.Namespace, plan: pd.DataFrame
) -> pd.DataFrame:
    source_manifests = {
        session: source_checkpoint_rows(args, session)
        for session in EXPECTED_SESSIONS
    }
    rows: list[dict[str, Any]] = []
    for expected in plan.itertuples(index=False):
        session = str(expected.session)
        seed = int(expected.seed)
        fold = int(expected.fold)
        source_row = source_manifests[session].get((seed, fold))
        if source_row is None:
            raise AssertionError(f"checkpoint source manifest lacks {(session, seed, fold)}")
        path = checkpoint_path_for_task(args, session, seed, fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        _model, payload, audit = load_validated_checkpoint(
            path,
            expected_sha256=str(source_row["checkpoint_sha256"]),
            expected_session=session,
            expected_seed=seed,
            expected_fold=fold,
            expected_train_cycles=str(expected.train_cycles),
            expected_test_cycles=str(expected.test_cycles),
        )
        rows.append(
            {
                "session": session,
                "seed": seed,
                "fold": fold,
                "task_key": str(expected.task_key),
                "checkpoint_path": str(path.resolve()),
                "checkpoint_sha256": audit["checkpoint_sha256"],
                "source_manifest_sha256": str(source_row["checkpoint_sha256"]),
                "source_code_version": str(payload["code_version"]),
                "method": str(payload["method"]),
                "base_model": str(payload["model_config"]["base_model"]),
                "model_parameters": int(payload["model_parameters"]),
                "final_epoch": int(payload["final_epoch"]),
                "train_cycles": str(payload["train_cycles"]),
                "test_cycles": str(payload["test_cycles"]),
                "normalization_transform": str(
                    payload["normalization_transform"]
                ),
                "normalization_shape": framework.canonical_json(
                    list(payload["normalization_mean"].shape)
                ),
                "valid": True,
            }
        )
        del _model, payload
    manifest = pd.DataFrame(rows).sort_values(
        ["session", "seed", "fold"]
    ).reset_index(drop=True)
    if len(manifest) != EXPECTED_TASKS or not manifest["valid"].all():
        raise AssertionError("checkpoint validation did not reach 246/246")
    if manifest[["session", "seed", "fold"]].duplicated().any():
        raise AssertionError("checkpoint validation contains duplicate task keys")
    return manifest


def validate_late_fusion_reference(
    args: argparse.Namespace, plan: pd.DataFrame
) -> pd.DataFrame:
    predictions_path = (
        args.historical_aggregate_dir / "multiframe_all_models_predictions.csv"
    )
    folds_path = (
        args.historical_aggregate_dir / "multiframe_all_models_fold_summary.csv"
    )
    predictions = pd.read_csv(predictions_path, dtype={"session": str})
    predictions = predictions[predictions["method"].eq(HISTORICAL_METHOD)].copy()
    folds = pd.read_csv(folds_path, dtype={"session": str})
    folds = folds[folds["method"].eq(HISTORICAL_METHOD)].copy()
    keys = {
        (str(row.session), int(row.seed), int(row.fold))
        for row in folds.itertuples(index=False)
    }
    expected_keys = {
        (str(row.session), int(row.seed), int(row.fold))
        for row in plan.itertuples(index=False)
    }
    if keys != expected_keys:
        raise AssertionError("historical late-fusion task keys differ")
    expected_membership = {
        (str(row.session), int(row.seed), int(row.fold)): (
            str(row.train_cycles),
            str(row.test_cycles),
        )
        for row in plan.itertuples(index=False)
    }
    for row in folds.itertuples(index=False):
        key = (str(row.session), int(row.seed), int(row.fold))
        if (str(row.train_cycles), str(row.test_cycles)) != expected_membership[key]:
            raise AssertionError("historical late-fusion fold membership differs")
    if predictions[["session", "seed", "block_id"]].duplicated().any():
        raise AssertionError("historical late-fusion OOF blocks are duplicated")
    if len(predictions) != EXPECTED_TEST_FORWARDS:
        raise AssertionError("historical late-fusion OOF coverage differs")
    return predictions


def build_late_fusion_reconstruction_audit(
    reconstructed: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    expected_blocks: int = EXPECTED_TEST_FORWARDS,
    expected_tasks: int = EXPECTED_TASKS,
    probability_atol: float = LATE_FUSION_PROBABILITY_ATOL,
    probability_rtol: float = LATE_FUSION_PROBABILITY_RTOL,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare same-checkpoint late fusion against the formal aggregate."""

    required = {
        "session",
        "seed",
        "fold",
        "sample_i",
        "block_id",
        "cycle",
        "block_name",
        "truth",
        "pred",
        "prob_no_stimulus",
        "prob_stimulus",
    }
    for name, frame in (("reconstructed", reconstructed), ("historical", historical)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise AssertionError(f"{name} late-fusion predictions lack {missing}")
    reconstructed = reconstructed[list(required)].copy()
    historical = historical[list(required)].copy()
    for frame in (reconstructed, historical):
        frame["session"] = frame["session"].astype(str)
        for column in ("seed", "fold", "sample_i", "cycle", "truth", "pred"):
            frame[column] = pd.to_numeric(frame[column]).astype(int)
        frame["block_id"] = frame["block_id"].astype(str)
        frame["block_name"] = frame["block_name"].astype(str)
    key_columns = ["session", "seed", "fold", "block_id"]
    if reconstructed[key_columns].duplicated().any():
        raise AssertionError("reconstructed late-fusion block identity is duplicated")
    if historical[key_columns].duplicated().any():
        raise AssertionError("historical late-fusion block identity is duplicated")
    task_keys = sorted(
        set(
            map(
                tuple,
                pd.concat(
                    [
                        reconstructed[["session", "seed", "fold"]],
                        historical[["session", "seed", "fold"]],
                    ],
                    ignore_index=True,
                )
                .drop_duplicates()
                .itertuples(index=False, name=None),
            )
        )
    )
    rows: list[dict[str, Any]] = []
    maximum_probability_difference = 0.0
    total_probability_mismatches = 0
    total_prediction_mismatches = 0
    total_truth_mismatches = 0
    total_identity_mismatches = 0
    for session, seed, fold in task_keys:
        reconstructed_task = reconstructed[
            reconstructed["session"].eq(str(session))
            & reconstructed["seed"].eq(int(seed))
            & reconstructed["fold"].eq(int(fold))
        ]
        historical_task = historical[
            historical["session"].eq(str(session))
            & historical["seed"].eq(int(seed))
            & historical["fold"].eq(int(fold))
        ]
        merged = reconstructed_task.merge(
            historical_task,
            on=key_columns,
            how="outer",
            suffixes=("_reconstructed", "_historical"),
            indicator=True,
            validate="one_to_one",
        )
        missing_identity = int((~merged["_merge"].eq("both")).sum())
        matched = merged[merged["_merge"].eq("both")].copy()
        metadata_mismatches = int(
            (
                (matched["sample_i_reconstructed"] != matched["sample_i_historical"])
                | (matched["cycle_reconstructed"] != matched["cycle_historical"])
                | (
                    matched["block_name_reconstructed"]
                    != matched["block_name_historical"]
                )
            ).sum()
        )
        truth_mismatches = int(
            (matched["truth_reconstructed"] != matched["truth_historical"]).sum()
        )
        prediction_mismatches = int(
            (matched["pred_reconstructed"] != matched["pred_historical"]).sum()
        )
        reconstructed_probability = matched[
            ["prob_no_stimulus_reconstructed", "prob_stimulus_reconstructed"]
        ].to_numpy(float)
        historical_probability = matched[
            ["prob_no_stimulus_historical", "prob_stimulus_historical"]
        ].to_numpy(float)
        absolute_difference = np.abs(
            reconstructed_probability - historical_probability
        )
        max_difference = (
            float(absolute_difference.max()) if absolute_difference.size else 0.0
        )
        probability_mismatches = int(
            (~np.isclose(
                reconstructed_probability,
                historical_probability,
                rtol=float(probability_rtol),
                atol=float(probability_atol),
            )).sum()
        )
        passed = bool(
            len(reconstructed_task) == len(historical_task)
            and missing_identity == 0
            and metadata_mismatches == 0
            and truth_mismatches == 0
            and prediction_mismatches == 0
            and probability_mismatches == 0
        )
        rows.append(
            {
                "session": str(session),
                "seed": int(seed),
                "fold": int(fold),
                "reconstructed_blocks": int(len(reconstructed_task)),
                "historical_blocks": int(len(historical_task)),
                "matched_blocks": int(len(matched)),
                "identity_mismatches": missing_identity + metadata_mismatches,
                "truth_mismatches": truth_mismatches,
                "probability_value_mismatches": probability_mismatches,
                "prediction_mismatches": prediction_mismatches,
                "maximum_probability_absolute_difference": max_difference,
                "probability_atol": float(probability_atol),
                "probability_rtol": float(probability_rtol),
                "status": "PASS" if passed else "FAIL",
            }
        )
        maximum_probability_difference = max(
            maximum_probability_difference, max_difference
        )
        total_probability_mismatches += probability_mismatches
        total_prediction_mismatches += prediction_mismatches
        total_truth_mismatches += truth_mismatches
        total_identity_mismatches += missing_identity + metadata_mismatches
    audit = pd.DataFrame(rows).sort_values(["session", "seed", "fold"])
    session_seed_ba_differences: list[float] = []
    if (
        len(reconstructed) == len(historical)
        and total_identity_mismatches == 0
        and total_truth_mismatches == 0
    ):
        aligned = reconstructed.merge(
            historical,
            on=key_columns,
            suffixes=("_reconstructed", "_historical"),
            validate="one_to_one",
        )
        for _key, group in aligned.groupby(["session", "seed"], sort=True):
            reconstructed_ba = balanced_accuracy(
                group["truth_reconstructed"].to_numpy(int),
                group["pred_reconstructed"].to_numpy(int),
            )
            historical_ba = balanced_accuracy(
                group["truth_historical"].to_numpy(int),
                group["pred_historical"].to_numpy(int),
            )
            session_seed_ba_differences.append(
                abs(reconstructed_ba - historical_ba)
            )
    maximum_ba_difference = (
        max(session_seed_ba_differences)
        if session_seed_ba_differences
        else float("inf")
    )
    passed = bool(
        len(reconstructed) == expected_blocks
        and len(historical) == expected_blocks
        and len(audit) == expected_tasks
        and audit["status"].eq("PASS").all()
        and total_identity_mismatches == 0
        and total_truth_mismatches == 0
        and total_probability_mismatches == 0
        and total_prediction_mismatches == 0
        and maximum_ba_difference <= 1e-12
    )
    summary = {
        "status": "PASS" if passed else "FAIL",
        "reconstructed_blocks": int(len(reconstructed)),
        "historical_blocks": int(len(historical)),
        "expected_blocks": int(expected_blocks),
        "audited_tasks": int(len(audit)),
        "expected_tasks": int(expected_tasks),
        "maximum_probability_absolute_difference": float(
            maximum_probability_difference
        ),
        "probability_atol": float(probability_atol),
        "probability_rtol": float(probability_rtol),
        "probability_value_mismatches": int(total_probability_mismatches),
        "prediction_mismatches": int(total_prediction_mismatches),
        "truth_mismatches": int(total_truth_mismatches),
        "identity_mismatches": int(total_identity_mismatches),
        "session_seed_groups_compared": int(len(session_seed_ba_differences)),
        "maximum_session_seed_oof_ba_difference": float(maximum_ba_difference),
        "historical_aggregate_used_for_final_comparison": False,
        "reconstructed_predictions_used_for_final_comparison": True,
    }
    return audit.reset_index(drop=True), summary


def require_late_fusion_reconstruction_pass(summary: dict[str, Any]) -> None:
    if summary.get("status") != "PASS":
        raise AssertionError(
            "same-checkpoint late-fusion reconstruction differs from the "
            "historical formal aggregate; formal summary is blocked"
        )


def validate_meanpool_reference(
    args: argparse.Namespace, plan: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    complete_path = args.meanpool_run_dir / "RUN_COMPLETE.json"
    config_path = args.meanpool_run_dir / "config.json"
    task_plan_path = args.meanpool_run_dir / "task_plan.csv"
    predictions_path = args.meanpool_run_dir / "predictions.csv"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_meanpool_protocol_payloads(complete, config)
    full_plan = pd.read_csv(task_plan_path, dtype={"session": str})
    validate_meanpool_task_plan(full_plan, plan)
    predictions = pd.read_csv(predictions_path, dtype={"session": str})
    predictions = predictions[predictions["variant"].eq("mean_only")].copy()
    keys = {
        (str(row.session), int(row.seed), int(row.fold))
        for row in predictions[["session", "seed", "fold"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected_keys = {
        (str(row.session), int(row.seed), int(row.fold))
        for row in plan.itertuples(index=False)
    }
    if keys != expected_keys:
        raise AssertionError("current MeanPool task keys differ")
    if predictions[["session", "seed", "cycle", "block_name"]].duplicated().any():
        raise AssertionError("current MeanPool OOF blocks are duplicated")
    if len(predictions) != EXPECTED_TEST_FORWARDS:
        raise AssertionError("current MeanPool OOF coverage differs")
    expected_by_task = {
        (str(row.session), int(row.seed), int(row.fold)): row
        for row in plan.itertuples(index=False)
    }
    for key, group in predictions.groupby(["session", "seed", "fold"], sort=True):
        normalized_key = (str(key[0]), int(key[1]), int(key[2]))
        expected = expected_by_task[normalized_key]
        expected_test_cycles = {
            int(value) for value in str(expected.test_cycles).split(",")
        }
        if len(group) != int(expected.n_test_samples):
            raise AssertionError("current MeanPool per-task sample count differs")
        if set(group["cycle"].astype(int)) != expected_test_cycles:
            raise AssertionError("current MeanPool test-cycle membership differs")
        if not set(group["y_true"].astype(int)).issubset({0, 1}):
            raise AssertionError("current MeanPool labels are not binary presence")
    return predictions, {
        "run_complete_sha256": file_sha256(complete_path),
        "config_sha256": file_sha256(config_path),
        "task_plan_sha256": file_sha256(task_plan_path),
        "predictions_sha256": file_sha256(predictions_path),
        "run_fingerprint": str(complete["run_fingerprint"]),
        "git_commit": str(config.get("git_commit", "")),
        "model_implementation_version": str(
            complete["model_implementation_version"]
        ),
        "output_version": MEANPOOL_OUTPUT_VERSION,
        "model": MEANPOOL_MODEL_NAME,
        "variant": "mean_only",
        "mean_only_task_coverage": EXPECTED_TASKS,
        "mean_only_oof_block_coverage": EXPECTED_TEST_FORWARDS,
        "protocol_validation": "passed",
    }


def validate_meanpool_protocol_payloads(
    complete: dict[str, Any], config: dict[str, Any]
) -> None:
    """Lock the formal MeanPool scientific protocol, not only completion counts."""

    experiment = config.get("experiment_config", {})
    mean_architecture = experiment.get("architectures", {}).get("mean_only", {})
    training = experiment.get("training", {})
    checks = {
        "run_status": complete.get("status") == "complete",
        "completed_tasks": int(complete.get("completed_tasks", -1)) == 492,
        "expected_tasks": int(complete.get("expected_tasks", -1)) == 492,
        "total_tasks": int(complete.get("total_tasks", -1)) == 492,
        "number_of_sessions": int(complete.get("number_of_sessions", -1)) == 9,
        "number_of_variants": int(complete.get("number_of_variants", -1)) == 2,
        "number_of_seeds": int(complete.get("number_of_seeds", -1)) == 3,
        "number_of_folds": int(complete.get("number_of_folds", -1))
        == EXPECTED_FOLDS,
        "run_model_version": complete.get("model_implementation_version")
        == MEANPOOL_MODEL_IMPLEMENTATION_VERSION,
        "config_model_version": config.get("model_implementation_version")
        == MEANPOOL_MODEL_IMPLEMENTATION_VERSION,
        "output_version": experiment.get("output_version")
        == MEANPOOL_OUTPUT_VERSION,
        "task": experiment.get("task") == "binary_presence",
        "class_mapping": experiment.get("class_mapping")
        == {"0": "no_stimulus", "1": "stimulus"},
        "input_protocol": experiment.get("input_protocol") == "clean4",
        "raw_input_shape": experiment.get("raw_input_shape") == [4, 128, 501],
        "sessions": list(experiment.get("sessions", []))
        == list(EXPECTED_SESSIONS),
        "seeds": list(experiment.get("seeds", [])) == list(SEEDS),
        "variants": list(experiment.get("variants", []))
        == ["mean_only", "mean_std"],
        "mean_only_model": mean_architecture.get("method")
        == MEANPOOL_MODEL_NAME,
        "mean_only_variant": mean_architecture.get("variant") == "mean_only",
        "mean_only_model_version": mean_architecture.get(
            "model_implementation_version"
        )
        == MEANPOOL_MODEL_IMPLEMENTATION_VERSION,
        "mean_only_temporal_length": int(
            mean_architecture.get("temporal_length", -1)
        )
        == 4,
        "mean_only_temporal_reduction": mean_architecture.get(
            "temporal_reduction"
        )
        == "mean",
        "mean_only_parameters": int(
            mean_architecture.get("trainable_parameters", -1)
        )
        == EXPECTED_PARAMETERS,
        "fixed_epoch_40": int(training.get("max_epochs", -1)) == EXPECTED_EPOCH
        and experiment.get("epoch_selection")
        == "fixed 40 epochs; no validation or early stopping",
        "cycle_grouped_folds": experiment.get("cv")
        == "exact formal clean4 cycle-grouped folds, max_folds=10",
        "train_fold_normalization": experiment.get("normalization")
        == "pixel z-score fit on outer-training blocks and all four real frames only",
        "normalization_preprocessing": experiment.get("preprocessing")
        == (
            "clean4 -> per-frame arcsinh -> outer-train-fold all-frame "
            "pixel z-score -> unchanged shared FCNN frame encoder -> "
            "bottleneck temporal statistics"
        ),
        "no_test_normalization": not bool(
            experiment.get("test_used_for_normalization", True)
        ),
        "no_test_feature_scaling": not bool(
            experiment.get("test_used_for_feature_scaling", True)
        ),
        "no_test_model_selection": not bool(
            experiment.get("test_used_for_model_selection", True)
        ),
        "no_test_early_stopping": not bool(
            experiment.get("test_used_for_early_stopping", True)
        ),
    }
    failures = sorted(name for name, valid in checks.items() if not valid)
    if failures:
        raise AssertionError(
            "current MeanPool scientific protocol mismatch: " + ", ".join(failures)
        )


def validate_meanpool_task_plan(
    full_plan: pd.DataFrame, expected_mean_only_plan: pd.DataFrame
) -> None:
    required = {
        "session",
        "variant",
        "seed",
        "fold",
        "n_train_samples",
        "n_test_samples",
        "train_cycles",
        "test_cycles",
    }
    if not required.issubset(full_plan.columns):
        raise AssertionError("current MeanPool task plan lacks required columns")
    if len(full_plan) != 492 or set(full_plan["variant"].astype(str)) != {
        "mean_only",
        "mean_std",
    }:
        raise AssertionError("current MeanPool task plan is not paired 492 tasks")
    counts = full_plan.groupby("variant").size().to_dict()
    if counts != {"mean_only": EXPECTED_TASKS, "mean_std": EXPECTED_TASKS}:
        raise AssertionError("current MeanPool candidate task coverage differs")
    observed = full_plan[full_plan["variant"].eq("mean_only")].copy()
    columns = [
        "session",
        "seed",
        "fold",
        "n_train_samples",
        "n_test_samples",
        "train_cycles",
        "test_cycles",
    ]
    observed = observed[columns].sort_values(["session", "seed", "fold"])
    expected = expected_mean_only_plan[columns].sort_values(
        ["session", "seed", "fold"]
    )
    observed = observed.reset_index(drop=True).astype(
        {"session": str, "seed": int, "fold": int}
    )
    expected = expected.reset_index(drop=True).astype(
        {"session": str, "seed": int, "fold": int}
    )
    if not observed.equals(expected):
        raise AssertionError("current MeanPool task/fold membership differs")


def validate_meanpool_sample_identity(
    canonical_predictions: pd.DataFrame,
    meanpool_predictions: pd.DataFrame,
    *,
    expected_blocks: int = EXPECTED_TEST_FORWARDS,
    expected_session_seed_groups: int = len(EXPECTED_SESSIONS) * len(SEEDS),
) -> dict[str, Any]:
    """Require exact OOF sample metadata and truth alignment before comparison."""

    identity = ["session", "seed", "fold", "source_index", "cycle", "block_name"]
    required_canonical = set(identity + ["y_true"])
    required_meanpool = set(identity + ["y_true"])
    if not required_canonical.issubset(canonical_predictions.columns):
        raise AssertionError("canonical predictions lack MeanPool identity columns")
    if not required_meanpool.issubset(meanpool_predictions.columns):
        raise AssertionError("MeanPool predictions lack identity columns")
    canonical = canonical_predictions[identity + ["y_true"]].copy()
    meanpool = meanpool_predictions[identity + ["y_true"]].copy()
    for frame in (canonical, meanpool):
        frame["session"] = frame["session"].astype(str)
        for column in ("seed", "fold", "source_index", "cycle", "y_true"):
            frame[column] = pd.to_numeric(frame[column]).astype(int)
    if canonical[identity].duplicated().any() or meanpool[identity].duplicated().any():
        raise AssertionError("canonical/MeanPool OOF identity is duplicated")
    merged = canonical.merge(
        meanpool,
        on=identity,
        how="outer",
        suffixes=("_canonical", "_meanpool"),
        indicator=True,
        validate="one_to_one",
    )
    if len(canonical) != expected_blocks or len(meanpool) != expected_blocks:
        raise AssertionError("canonical/MeanPool OOF coverage differs")
    if not merged["_merge"].eq("both").all():
        raise AssertionError("canonical and MeanPool OOF sample identities differ")
    truth_mismatches = int(
        (merged["y_true_canonical"] != merged["y_true_meanpool"]).sum()
    )
    if truth_mismatches:
        raise AssertionError("canonical and MeanPool OOF truth labels differ")
    group_counts = merged.groupby(["session", "seed"]).size()
    if len(group_counts) != expected_session_seed_groups:
        raise AssertionError("canonical/MeanPool session-seed coverage differs")
    return {
        "status": "PASS",
        "matched_oof_blocks": int(len(merged)),
        "truth_mismatches": truth_mismatches,
        "session_seed_groups": int(len(group_counts)),
        "identity_columns": identity,
    }


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        classification_metrics(
            np.asarray(y_true, dtype=np.int64),
            np.asarray(y_pred, dtype=np.int64),
        )["balanced_accuracy"]
    )


def infer_task(
    args: argparse.Namespace,
    expected: Any,
    data: Any,
    source_row: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    session = str(expected.session)
    seed = int(expected.seed)
    fold = int(expected.fold)
    train_cycles = [int(value) for value in str(expected.train_cycles).split(",")]
    test_cycles = [int(value) for value in str(expected.test_cycles).split(",")]
    train_mask = np.isin(data.groups, train_cycles)
    test_mask = np.isin(data.groups, test_cycles)
    if train_mask.any() and test_mask.any() and np.any(train_mask & test_mask):
        raise AssertionError("train/test blocks overlap")
    if int(train_mask.sum()) != int(expected.n_train_samples):
        raise AssertionError("training block count differs from task plan")
    if int(test_mask.sum()) != int(expected.n_test_samples):
        raise AssertionError("test block count differs from task plan")
    path = checkpoint_path_for_task(args, session, seed, fold)
    model, payload, checkpoint_audit = load_validated_checkpoint(
        path,
        expected_sha256=str(source_row["source_manifest_sha256"]),
        expected_session=session,
        expected_seed=seed,
        expected_fold=fold,
        expected_train_cycles=str(expected.train_cycles),
        expected_test_cycles=str(expected.test_cycles),
    )
    train_frames, train_selection = select_canonical_frames(
        data.X[train_mask], data.clean4_relative_time_s[train_mask]
    )
    test_frames, test_selection = select_canonical_frames(
        data.X[test_mask], data.clean4_relative_time_s[test_mask]
    )
    mean = payload["normalization_mean"]
    std = payload["normalization_std"]
    transform = str(payload["normalization_transform"])
    train_normalized = apply_saved_normalization(
        train_frames, mean, std, transform=transform
    )
    test_normalized = apply_saved_normalization(
        test_frames, mean, std, transform=transform
    )
    test_blocks = np.asarray(data.X[test_mask], dtype=np.float32)
    test_blocks_normalized = apply_saved_normalization(
        test_blocks.reshape(-1, *EXPECTED_IMAGE_SHAPE),
        mean,
        std,
        transform=transform,
    ).reshape(test_blocks.shape)
    train_prob = predict_single_frame_probabilities(
        model, train_normalized, batch_size=args.inference_batch_size
    )
    test_prob = predict_single_frame_probabilities(
        model, test_normalized, batch_size=args.inference_batch_size
    )
    late_fusion_prob = reconstruct_late_fusion_probabilities(
        model,
        test_blocks_normalized,
        batch_size=args.inference_batch_size,
    )
    train_pred = train_prob.argmax(axis=1).astype(np.int64)
    test_pred = test_prob.argmax(axis=1).astype(np.int64)
    late_fusion_pred = late_fusion_prob.argmax(axis=1).astype(np.int64)
    train_truth = data.y[train_mask].astype(np.int64)
    test_truth = data.y[test_mask].astype(np.int64)
    test_meta = data.metadata.loc[test_mask].reset_index(drop=True)
    test_source_indices = np.flatnonzero(test_mask)
    predictions = pd.DataFrame(
        {
            "session": session,
            "seed": seed,
            "fold": fold,
            "sample_index": np.arange(len(test_truth), dtype=np.int64),
            "source_index": test_source_indices,
            "sample_id": test_meta["block_id"].astype(str),
            "cycle": data.groups[test_mask].astype(np.int64),
            "block_name": test_meta["block_name"].astype(str),
            "canonical_position": test_selection.positions,
            "canonical_original_frame_index": data.clean4_original_frame_indices[
                test_mask
            ][np.arange(len(test_truth)), test_selection.positions],
            "canonical_relative_time_s": test_selection.relative_times_s,
            "distance_to_midpoint_s": test_selection.distances_to_midpoint_s,
            "tie": test_selection.ties,
            "y_true": test_truth,
            "y_pred": test_pred,
            "probability_0": test_prob[:, 0],
            "probability_1": test_prob[:, 1],
        }
    )
    if len(predictions) != int(test_mask.sum()):
        raise AssertionError("one block did not yield exactly one test prediction")
    if predictions["sample_id"].duplicated().any():
        raise AssertionError("a test block yielded multiple predictions")
    late_fusion_predictions = pd.DataFrame(
        {
            "session": session,
            "seed": seed,
            "fold": fold,
            # Historical aggregate `sample_i` is the session-level block/source
            # index, not a fold-local row number.
            "sample_i": test_source_indices,
            "source_index": test_source_indices,
            "block_id": test_meta["block_id"].astype(str),
            "cycle": data.groups[test_mask].astype(np.int64),
            "block_name": test_meta["block_name"].astype(str),
            "truth": test_truth,
            "pred": late_fusion_pred,
            "prob_no_stimulus": late_fusion_prob[:, 0],
            "prob_stimulus": late_fusion_prob[:, 1],
        }
    )
    if len(late_fusion_predictions) != int(test_mask.sum()):
        raise AssertionError("late-fusion reconstruction coverage differs")
    if late_fusion_predictions["block_id"].duplicated().any():
        raise AssertionError("late-fusion reconstruction duplicated a block")
    fold_metrics = classification_metrics(test_truth, test_pred)
    fold_result = {
        "session": session,
        "seed": seed,
        "fold": fold,
        "task_key": str(expected.task_key),
        "train_cycles": str(expected.train_cycles),
        "test_cycles": str(expected.test_cycles),
        "n_train_blocks": int(train_mask.sum()),
        "n_test_blocks": int(test_mask.sum()),
        "canonical_train_accuracy_diagnostic": float(
            np.mean(train_pred == train_truth)
        ),
        "test_accuracy": float(fold_metrics["accuracy"]),
        "test_balanced_accuracy": float(fold_metrics["balanced_accuracy"]),
        "test_macro_f1": float(fold_metrics["macro_f1"]),
        "train_canonical_forward_count": int(len(train_frames)),
        "test_canonical_forward_count": int(len(test_frames)),
        "late_fusion_verification_frame_forward_count": int(
            4 * len(test_frames)
        ),
        "model_eval": not model.training,
        "torch_no_grad_path": True,
        "device": "cpu",
    }
    normalization_audit = {
        "session": session,
        "seed": seed,
        "fold": fold,
        "checkpoint_sha256": checkpoint_audit["checkpoint_sha256"],
        "normalization_transform": transform,
        "normalization_shape": framework.canonical_json(list(mean.shape)),
        "saved_checkpoint_stats_used": True,
        "normalization_refit": False,
        "test_used_for_fit": False,
        "mean_mean": float(np.asarray(mean).mean()),
        "mean_std": float(np.asarray(mean).std()),
        "std_mean": float(np.asarray(std).mean()),
        "std_std": float(np.asarray(std).std()),
        "n_train_canonical_frames_transformed": int(len(train_frames)),
        "n_test_canonical_frames_transformed": int(len(test_frames)),
        "n_test_late_fusion_verification_frames_transformed": int(
            4 * len(test_frames)
        ),
        "train_tie_count": int(train_selection.ties.sum()),
        "test_tie_count": int(test_selection.ties.sum()),
    }
    return predictions, late_fusion_predictions, fold_result, normalization_audit


def reference_seed_ba(
    predictions: pd.DataFrame,
    *,
    truth_column: str,
    prediction_column: str,
    method_name: str,
) -> pd.DataFrame:
    rows = []
    for (session, seed), group in predictions.groupby(
        ["session", "seed"], sort=True
    ):
        rows.append(
            {
                "session": str(session),
                "seed": int(seed),
                f"{method_name}_BA": balanced_accuracy(
                    group[truth_column].to_numpy(int),
                    group[prediction_column].to_numpy(int),
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_SESSIONS) * len(SEEDS):
        raise AssertionError(f"{method_name} session/seed coverage differs")
    return result


def build_summaries(
    fold_summary: pd.DataFrame,
    predictions: pd.DataFrame,
    late_predictions: pd.DataFrame,
    meanpool_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    single = reference_seed_ba(
        predictions,
        truth_column="y_true",
        prediction_column="y_pred",
        method_name="single_frame",
    )
    late = reference_seed_ba(
        late_predictions,
        truth_column="truth",
        prediction_column="pred",
        method_name="late_fusion",
    )
    meanpool = reference_seed_ba(
        meanpool_predictions,
        truth_column="y_true",
        prediction_column="y_pred",
        method_name="meanpool",
    )
    train = (
        fold_summary.groupby(["session", "seed"], as_index=False)[
            "canonical_train_accuracy_diagnostic"
        ]
        .mean()
        .rename(
            columns={
                "canonical_train_accuracy_diagnostic": (
                    "mean_fold_canonical_train_accuracy_diagnostic"
                )
            }
        )
    )
    seed_summary = single.merge(late, on=["session", "seed"], validate="one_to_one")
    seed_summary = seed_summary.merge(
        meanpool, on=["session", "seed"], validate="one_to_one"
    ).merge(train, on=["session", "seed"], validate="one_to_one")
    seed_summary["canonical_train_test_gap_diagnostic"] = (
        seed_summary["mean_fold_canonical_train_accuracy_diagnostic"]
        - seed_summary["single_frame_BA"]
    )
    session_summary = (
        seed_summary.groupby("session", as_index=False)
        .agg(
            single_frame_BA=("single_frame_BA", "mean"),
            late_fusion_BA=("late_fusion_BA", "mean"),
            meanpool_BA=("meanpool_BA", "mean"),
            canonical_train_accuracy_diagnostic=(
                "mean_fold_canonical_train_accuracy_diagnostic",
                "mean",
            ),
            canonical_train_test_gap_diagnostic=(
                "canonical_train_test_gap_diagnostic",
                "mean",
            ),
        )
        .sort_values("session")
    )
    session_summary["late_fusion_minus_single"] = (
        session_summary["late_fusion_BA"] - session_summary["single_frame_BA"]
    )
    session_summary["meanpool_minus_single"] = (
        session_summary["meanpool_BA"] - session_summary["single_frame_BA"]
    )
    session_summary["meanpool_minus_late_fusion"] = (
        session_summary["meanpool_BA"] - session_summary["late_fusion_BA"]
    )
    comparisons = {
        "late_fusion_vs_single_frame": "late_fusion_minus_single",
        "meanpool_vs_single_frame": "meanpool_minus_single",
        "meanpool_vs_late_fusion": "meanpool_minus_late_fusion",
    }
    paired_rows = []
    audits: dict[str, Any] = {}
    for comparison, column in comparisons.items():
        values = session_summary[column].to_numpy(float)
        for row in session_summary.itertuples(index=False):
            paired_rows.append(
                {
                    "comparison": comparison,
                    "session": str(row.session),
                    "delta_BA": float(getattr(row, column)),
                }
            )
        audits[comparison] = pairwise_statistical_audit(
            session_summary["session"].astype(str).tolist(), values
        )
    statistical = {
        "unit": "nine paired session-level seed-mean OOF Balanced Accuracies",
        "test": "exact two-sided paired sign-flip over all 2^9 sign patterns",
        "ordinary_t_test_used": False,
        "comparisons": audits,
        "method_group_summaries": method_group_summaries(session_summary),
    }
    return seed_summary, session_summary, pd.DataFrame(paired_rows), statistical


def exact_two_sided_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def pairwise_statistical_audit(
    sessions: list[str], values: np.ndarray
) -> dict[str, Any]:
    delta = pd.Series(np.asarray(values, dtype=float), index=sessions)
    largest_improvement_session = str(delta.idxmax())
    largest_degradation_session = str(delta.idxmin())
    return {
        "n_sessions": int(len(delta)),
        "mean_delta_BA": float(delta.mean()),
        "median_delta_BA": float(delta.median()),
        "improved": int((delta > TIE_TOLERANCE).sum()),
        "tied": int((delta.abs() <= TIE_TOLERANCE).sum()),
        "worsened": int((delta < -TIE_TOLERANCE).sum()),
        "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(
            delta.to_numpy(float)
        ),
        "largest_improvement_session": largest_improvement_session,
        "largest_improvement": float(delta.loc[largest_improvement_session]),
        "largest_degradation_session": largest_degradation_session,
        "largest_degradation": float(delta.loc[largest_degradation_session]),
        "leave_largest_improvement_out_delta": float(
            delta.drop(index=largest_improvement_session).mean()
        ),
        "session_deltas": {key: float(value) for key, value in delta.items()},
    }


def method_group_summaries(session_summary: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method, column in (
        ("single_frame", "single_frame_BA"),
        ("late_fusion", "late_fusion_BA"),
        ("meanpool", "meanpool_BA"),
    ):
        strong = session_summary[session_summary["session"].isin(STRONG_SESSIONS)]
        weak = session_summary[session_summary["session"].isin(WEAK_SESSIONS)]
        result[method] = {
            "overall_9_mean_BA": float(session_summary[column].mean()),
            "strong_3_mean_BA": float(strong[column].mean()),
            "weak_6_mean_BA": float(weak[column].mean()),
        }
    return result


def build_report(
    session_summary: pd.DataFrame, statistical: dict[str, Any]
) -> str:
    lines = [
        "# Canonical-midpoint single-frame FCNN v1",
        "",
        "This is a CPU-only reconstruction from frozen historical "
        "`fcnn_late_fusion` checkpoints. No model was trained, fine-tuned, or "
        "updated. It is a Berthon-style architectural adaptation using the same "
        "frame-wise clean4 training pool, not a full reproduction of the original "
        "paper training protocol.",
        "",
        "Each held-out block contributes exactly one frame selected by "
        "`argmin_k |t_k - 15 s|`; an exact tie selects the earlier timestamp. "
        "Checkpoint-saved outer-training-fold normalization is applied, followed "
        "by one single-frame FCNN forward. No fusion or averaging is used.",
        "",
        "## Session results",
        "",
        "| session | single-frame BA | late-fusion BA | meanpool BA | LF-single | MP-single | MP-LF |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in session_summary.itertuples(index=False):
        lines.append(
            f"| {row.session} | {row.single_frame_BA:.4f} | "
            f"{row.late_fusion_BA:.4f} | {row.meanpool_BA:.4f} | "
            f"{row.late_fusion_minus_single:+.4f} | "
            f"{row.meanpool_minus_single:+.4f} | "
            f"{row.meanpool_minus_late_fusion:+.4f} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, audit in statistical["comparisons"].items():
        lines.append(
            f"- `{name}`: mean Δ={audit['mean_delta_BA']:+.4f}, median "
            f"Δ={audit['median_delta_BA']:+.4f}, improved/tied/worsened="
            f"{audit['improved']}/{audit['tied']}/{audit['worsened']}, exact "
            f"p={audit['exact_two_sided_sign_flip_p']:.6f}, leave-largest-out "
            f"Δ={audit['leave_largest_improvement_out_delta']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Single-frame versus late fusion is the clean test-time multi-frame "
            "inference comparison because it uses the same checkpoint. The late-"
            "fusion values in this report are reconstructed from that checkpoint "
            "and passed the formal aggregate probability/identity audit. Late fusion "
            "versus MeanPool compares decoder strategies. Single-frame versus "
            "MeanPool is the primary overall baseline comparison, but their "
            "training units differ (frame-wise CE versus block-wise post-mean CE).",
        ]
    )
    return "\n".join(lines) + "\n"


def runtime_provenance(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "cuda_available_ignored": bool(torch.cuda.is_available()),
        "device_used": "cpu",
        "git_commit": git_output(args.project_root, "rev-parse", "HEAD"),
        "git_parent_commit": git_output(args.project_root, "rev-parse", "HEAD^"),
        "source_sha256": {
            str(path.relative_to(args.project_root)): file_sha256(path)
            for path in source_paths(args.project_root)
        },
    }


def run_sanity(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise RuntimeError("sanity is CPU-only")
    if args.synthetic_sanity:
        from ultrasound_decoding.deep import FCNN

        session = "710"
        data = load_block_sequence_session(
            args.project_root, session, TASK_NAME, data_dir=data_dir(args)
        )
        sample_indices = np.arange(2, dtype=np.int64)
        torch.manual_seed(0)
        model = FCNN(input_shape=EXPECTED_IMAGE_SHAPE, n_classes=2).cpu().eval()
        mean = np.zeros((1, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
        std = np.ones((1, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
        checkpoint_used = False
        task_key = "synthetic:no-training"
    else:
        plan = load_formal_task_plan(args)
        expected = plan.iloc[0]
        session = str(expected.session)
        data = load_block_sequence_session(
            args.project_root, session, TASK_NAME, data_dir=data_dir(args)
        )
        test_cycles = [
            int(value) for value in str(expected.test_cycles).split(",")
        ]
        sample_indices = np.flatnonzero(np.isin(data.groups, test_cycles))[:2]
        if len(sample_indices) != 2:
            raise AssertionError("formal sanity requires two held-out blocks")
        sources = source_checkpoint_rows(args, str(expected.session))
        source = sources[(int(expected.seed), int(expected.fold))]
        model, payload, _audit = load_validated_checkpoint(
            checkpoint_path_for_task(
                args, str(expected.session), int(expected.seed), int(expected.fold)
            ),
            expected_sha256=str(source["checkpoint_sha256"]),
            expected_session=str(expected.session),
            expected_seed=int(expected.seed),
            expected_fold=int(expected.fold),
            expected_train_cycles=str(expected.train_cycles),
            expected_test_cycles=str(expected.test_cycles),
        )
        mean = payload["normalization_mean"]
        std = payload["normalization_std"]
        checkpoint_used = True
        task_key = str(expected.task_key)
    frames, selection = select_canonical_frames(
        data.X[sample_indices], data.clean4_relative_time_s[sample_indices]
    )
    normalized = apply_saved_normalization(
        frames, mean, std, transform=NORMALIZATION_TRANSFORM
    )
    probabilities = predict_single_frame_probabilities(
        model, normalized, batch_size=2
    )
    late_fusion_sanity: dict[str, Any] = {
        "status": "not_run_for_synthetic_sanity",
        "frame_forwards": 0,
    }
    if checkpoint_used:
        blocks = np.asarray(data.X[sample_indices], dtype=np.float32)
        normalized_blocks = apply_saved_normalization(
            blocks.reshape(-1, *EXPECTED_IMAGE_SHAPE),
            mean,
            std,
            transform=NORMALIZATION_TRANSFORM,
        ).reshape(blocks.shape)
        reconstructed_probabilities = reconstruct_late_fusion_probabilities(
            model, normalized_blocks, batch_size=2
        )
        meta = data.metadata.iloc[sample_indices].reset_index(drop=True)
        reconstructed = pd.DataFrame(
            {
                "session": session,
                "seed": int(expected.seed),
                "fold": int(expected.fold),
                "sample_i": sample_indices,
                "block_id": meta["block_id"].astype(str),
                "cycle": data.groups[sample_indices].astype(np.int64),
                "block_name": meta["block_name"].astype(str),
                "truth": data.y[sample_indices].astype(np.int64),
                "pred": reconstructed_probabilities.argmax(axis=1),
                "prob_no_stimulus": reconstructed_probabilities[:, 0],
                "prob_stimulus": reconstructed_probabilities[:, 1],
            }
        )
        historical = validate_late_fusion_reference(args, plan)
        historical = historical[
            historical["session"].astype(str).eq(session)
            & historical["seed"].astype(int).eq(int(expected.seed))
            & historical["fold"].astype(int).eq(int(expected.fold))
            & historical["block_id"].astype(str).isin(
                reconstructed["block_id"].astype(str)
            )
        ].copy()
        _audit_rows, late_fusion_sanity = build_late_fusion_reconstruction_audit(
            reconstructed,
            historical,
            expected_blocks=2,
            expected_tasks=1,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    framework.atomic_json(
        args.output_dir / "SANITY_COMPLETE.json",
        {
            "status": "complete",
            "device": "cpu",
            "training_started": False,
            "formal_full_inference_started": False,
            "synthetic_sanity": bool(args.synthetic_sanity),
            "historical_checkpoint_used": checkpoint_used,
            "task_key": task_key,
            "n_blocks": 2,
            "canonical_frame_forwards": 2,
            "late_fusion_verification_frame_forwards": int(
                late_fusion_sanity["frame_forwards"]
                if "frame_forwards" in late_fusion_sanity
                else 8
            ),
            "canonical_positions": selection.positions.tolist(),
            "probabilities_finite": bool(np.isfinite(probabilities).all()),
            "one_prediction_per_block": len(probabilities) == 2,
            "late_fusion_reconstruction": late_fusion_sanity,
            "completed_utc": utc_now(),
        },
    )
    if checkpoint_used:
        require_late_fusion_reconstruction_pass(late_fusion_sanity)
    print(
        "SANITY COMPLETE device=cpu training_started=False "
        f"synthetic={args.synthetic_sanity}",
        flush=True,
    )


def run_full(args: argparse.Namespace) -> None:
    if not args.review_approved:
        raise RuntimeError("full reconstruction is locked until --review-approved")
    if args.device != "cpu":
        raise RuntimeError("formal reconstruction is CPU-only")
    if args.synthetic_sanity:
        raise RuntimeError("synthetic sanity cannot enter the formal full stage")
    plan, canonical_manifest = write_plan(args)
    # Strictly validate all 246 assets before the first formal model forward.
    checkpoint_manifest = validate_all_checkpoints(args, plan)
    historical_late_predictions = validate_late_fusion_reference(args, plan)
    meanpool_predictions, meanpool_provenance = validate_meanpool_reference(
        args, plan
    )
    framework.atomic_csv(
        args.output_dir / "checkpoint_manifest.csv", checkpoint_manifest
    )
    checkpoint_lookup = {
        (str(row.session), int(row.seed), int(row.fold)): row._asdict()
        for row in checkpoint_manifest.itertuples(index=False)
    }
    prediction_frames: list[pd.DataFrame] = []
    late_fusion_reconstruction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(
            args.project_root, session, TASK_NAME, data_dir=data_dir(args)
        )
        session_plan = plan[plan["session"].eq(session)]
        for expected in session_plan.itertuples(index=False):
            source_row = checkpoint_lookup[
                (session, int(expected.seed), int(expected.fold))
            ]
            (
                task_predictions,
                task_late_fusion_predictions,
                fold_result,
                normalization,
            ) = infer_task(args, expected, data, source_row)
            prediction_frames.append(task_predictions)
            late_fusion_reconstruction_frames.append(
                task_late_fusion_predictions
            )
            fold_rows.append(fold_result)
            normalization_rows.append(normalization)
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["session", "seed", "fold", "sample_index"]
    )
    reconstructed_late_predictions = pd.concat(
        late_fusion_reconstruction_frames, ignore_index=True
    ).sort_values(["session", "seed", "fold", "sample_i"])
    fold_summary = pd.DataFrame(fold_rows).sort_values(
        ["session", "seed", "fold"]
    )
    normalization_audit = pd.DataFrame(normalization_rows).sort_values(
        ["session", "seed", "fold"]
    )
    if len(fold_summary) != EXPECTED_TASKS:
        raise AssertionError("formal fold result coverage is not 246/246")
    if len(predictions) != EXPECTED_TEST_FORWARDS:
        raise AssertionError("formal canonical OOF prediction coverage differs")
    if (
        int(fold_summary["train_canonical_forward_count"].sum())
        != EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
    ):
        raise AssertionError("formal train diagnostic forward coverage differs")
    if predictions[["session", "seed", "sample_id"]].duplicated().any():
        raise AssertionError("a held-out block has more than one OOF prediction")
    if len(reconstructed_late_predictions) != EXPECTED_TEST_FORWARDS:
        raise AssertionError("reconstructed late-fusion OOF coverage differs")
    if (
        int(
            fold_summary[
                "late_fusion_verification_frame_forward_count"
            ].sum()
        )
        != EXPECTED_LATE_FUSION_FRAME_FORWARDS
    ):
        raise AssertionError("late-fusion verification frame coverage differs")
    late_fusion_audit, late_fusion_audit_summary = (
        build_late_fusion_reconstruction_audit(
            reconstructed_late_predictions, historical_late_predictions
        )
    )
    framework.atomic_csv(
        args.output_dir / "late_fusion_reconstructed_predictions.csv",
        reconstructed_late_predictions,
    )
    framework.atomic_csv(
        args.output_dir / "late_fusion_reconstruction_audit.csv",
        late_fusion_audit,
    )
    framework.atomic_json(
        args.output_dir / "late_fusion_reconstruction_audit.json",
        late_fusion_audit_summary,
    )
    # Persist mismatch evidence, then block every formal summary on any drift.
    require_late_fusion_reconstruction_pass(late_fusion_audit_summary)
    meanpool_identity_audit = validate_meanpool_sample_identity(
        predictions, meanpool_predictions
    )
    seed_summary, session_summary, paired, statistical = build_summaries(
        fold_summary,
        predictions,
        reconstructed_late_predictions,
        meanpool_predictions,
    )
    provenance = {
        "status": "validated",
        "training_performed": False,
        "weights_updated": False,
        "device": "cpu",
        "checkpoint_coverage": f"{len(checkpoint_manifest)}/{EXPECTED_TASKS}",
        "checkpoint_manifest_sha256": file_sha256(
            args.output_dir / "checkpoint_manifest.csv"
        ),
        "historical_late_fusion": {
            "prediction_path": str(
                (
                    args.historical_aggregate_dir
                    / "multiframe_all_models_predictions.csv"
                ).resolve()
            ),
            "prediction_sha256": file_sha256(
                args.historical_aggregate_dir
                / "multiframe_all_models_predictions.csv"
            ),
            "task_coverage": EXPECTED_TASKS,
            "same_checkpoint_as_single_frame": True,
            "role": "externally validated provenance reference only",
            "reconstruction_audit_status": late_fusion_audit_summary["status"],
            "maximum_probability_absolute_difference": (
                late_fusion_audit_summary[
                    "maximum_probability_absolute_difference"
                ]
            ),
        },
        "reconstructed_late_fusion": {
            "prediction_sha256": file_sha256(
                args.output_dir / "late_fusion_reconstructed_predictions.csv"
            ),
            "audit_csv_sha256": file_sha256(
                args.output_dir / "late_fusion_reconstruction_audit.csv"
            ),
            "audit_json_sha256": file_sha256(
                args.output_dir / "late_fusion_reconstruction_audit.json"
            ),
            "used_for_final_comparison": True,
        },
        "current_meanpool": {
            **meanpool_provenance,
            "canonical_sample_identity_audit": meanpool_identity_audit,
        },
        "canonical_frame_manifest_sha256": file_sha256(
            args.output_dir / "canonical_frame_manifest.csv"
        ),
        "runtime": runtime_provenance(args),
    }
    framework.atomic_csv(args.output_dir / "predictions.csv", predictions)
    framework.atomic_csv(args.output_dir / "fold_summary.csv", fold_summary)
    framework.atomic_csv(
        args.output_dir / "normalization_audit.csv", normalization_audit
    )
    framework.atomic_csv(
        args.output_dir / "session_seed_summary.csv", seed_summary
    )
    framework.atomic_csv(args.output_dir / "session_summary.csv", session_summary)
    framework.atomic_csv(args.output_dir / "pairwise_comparison.csv", paired)
    framework.atomic_json(args.output_dir / "statistical_audit.json", statistical)
    framework.atomic_json(args.output_dir / "provenance_audit.json", provenance)
    framework.atomic_text(
        args.output_dir / "fcnn_canonical_single_frame_report.md",
        build_report(session_summary, statistical),
    )
    missing = [
        name for name in REQUIRED_FINAL_OUTPUTS if not (args.output_dir / name).is_file()
    ]
    if missing:
        raise AssertionError(f"formal reconstruction outputs are missing: {missing}")
    framework.atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "training_performed": False,
            "device": "cpu",
            "completed_tasks": len(fold_summary),
            "expected_tasks": EXPECTED_TASKS,
            "number_of_folds": EXPECTED_FOLDS,
            "number_of_sessions": len(EXPECTED_SESSIONS),
            "number_of_seeds": len(SEEDS),
            "test_block_predictions": len(predictions),
            "expected_test_block_predictions": EXPECTED_TEST_FORWARDS,
            "train_diagnostic_block_predictions": int(
                fold_summary["train_canonical_forward_count"].sum()
            ),
            "expected_train_diagnostic_block_predictions": (
                EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
            ),
            "canonical_block_forwards": int(
                len(predictions)
                + fold_summary["train_canonical_forward_count"].sum()
            ),
            "expected_canonical_block_forwards": (
                EXPECTED_CANONICAL_TOTAL_FORWARDS
            ),
            "late_fusion_reconstructed_block_predictions": int(
                len(reconstructed_late_predictions)
            ),
            "expected_late_fusion_reconstructed_block_predictions": (
                EXPECTED_TEST_FORWARDS
            ),
            "late_fusion_verification_frame_forwards": int(
                fold_summary[
                    "late_fusion_verification_frame_forward_count"
                ].sum()
            ),
            "expected_late_fusion_verification_frame_forwards": (
                EXPECTED_LATE_FUSION_FRAME_FORWARDS
            ),
            "total_model_frame_forwards": EXPECTED_TOTAL_FRAME_FORWARDS,
            "expected_total_model_frame_forwards": EXPECTED_TOTAL_FRAME_FORWARDS,
            "late_fusion_reconstruction_status": late_fusion_audit_summary[
                "status"
            ],
            "meanpool_sample_identity_status": meanpool_identity_audit["status"],
            "required_outputs": list(REQUIRED_FINAL_OUTPUTS),
            "completed_utc": utc_now(),
        },
    )
    print(
        f"RUN COMPLETE tasks={len(fold_summary)}/{EXPECTED_TASKS} "
        f"predictions={len(predictions)} device=cpu training_performed=False",
        flush=True,
    )


def run_status(args: argparse.Namespace) -> None:
    path = args.output_dir / "RUN_COMPLETE.json"
    if not path.is_file():
        print("STATUS incomplete; RUN_COMPLETE.json is absent")
        return
    complete = json.loads(path.read_text(encoding="utf-8"))
    valid = validate_run_complete_payload(complete) and all(
        (args.output_dir / name).is_file() for name in REQUIRED_FINAL_OUTPUTS
    )
    print(
        f"STATUS {'complete' if valid else 'invalid'} "
        f"tasks={complete.get('completed_tasks')}/{EXPECTED_TASKS}",
        flush=True,
    )
    if not valid:
        raise AssertionError("RUN_COMPLETE validation failed")


def validate_run_complete_payload(payload: dict[str, Any]) -> bool:
    """RUN_COMPLETE is valid only for the full 246-task CPU reconstruction."""

    return bool(
        payload.get("status") == "complete"
        and int(payload.get("completed_tasks", -1)) == EXPECTED_TASKS
        and int(payload.get("expected_tasks", -1)) == EXPECTED_TASKS
        and int(payload.get("number_of_folds", -1)) == EXPECTED_FOLDS
        and int(payload.get("number_of_sessions", -1)) == len(EXPECTED_SESSIONS)
        and int(payload.get("number_of_seeds", -1)) == len(SEEDS)
        and int(payload.get("test_block_predictions", -1))
        == EXPECTED_TEST_FORWARDS
        and int(payload.get("expected_test_block_predictions", -1))
        == EXPECTED_TEST_FORWARDS
        and int(payload.get("train_diagnostic_block_predictions", -1))
        == EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
        and int(payload.get("expected_train_diagnostic_block_predictions", -1))
        == EXPECTED_TRAIN_DIAGNOSTIC_FORWARDS
        and int(payload.get("canonical_block_forwards", -1))
        == EXPECTED_CANONICAL_TOTAL_FORWARDS
        and int(payload.get("expected_canonical_block_forwards", -1))
        == EXPECTED_CANONICAL_TOTAL_FORWARDS
        and int(
            payload.get("late_fusion_reconstructed_block_predictions", -1)
        )
        == EXPECTED_TEST_FORWARDS
        and int(
            payload.get(
                "expected_late_fusion_reconstructed_block_predictions", -1
            )
        )
        == EXPECTED_TEST_FORWARDS
        and int(payload.get("late_fusion_verification_frame_forwards", -1))
        == EXPECTED_LATE_FUSION_FRAME_FORWARDS
        and int(
            payload.get("expected_late_fusion_verification_frame_forwards", -1)
        )
        == EXPECTED_LATE_FUSION_FRAME_FORWARDS
        and int(payload.get("total_model_frame_forwards", -1))
        == EXPECTED_TOTAL_FRAME_FORWARDS
        and int(payload.get("expected_total_model_frame_forwards", -1))
        == EXPECTED_TOTAL_FRAME_FORWARDS
        and payload.get("late_fusion_reconstruction_status") == "PASS"
        and payload.get("meanpool_sample_identity_status") == "PASS"
        and not bool(payload.get("training_performed", True))
        and payload.get("device") == "cpu"
    )


def main() -> None:
    args = parse_args()
    if args.device != "cpu":
        raise RuntimeError("this experiment is CPU-only")
    if args.inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    if args.stage == "plan":
        run_plan(args)
    elif args.stage == "sanity":
        run_sanity(args)
    elif args.stage == "full":
        run_full(args)
    else:
        run_status(args)


if __name__ == "__main__":
    main()
