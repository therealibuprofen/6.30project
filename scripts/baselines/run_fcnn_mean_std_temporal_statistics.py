#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import shlex
import socket
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
for search_path in (PROJECT_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.baselines import run_mamba_visual_binary as fold_audit
from scripts.baselines import run_multiscale_temporal1d as framework
from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
    split_manifest,
)
from ultrasound_decoding.multiframe.fcnn_temporal_statistics import (
    BOTTLENECK_DIM,
    INPUT_VARIANTS,
    MEAN_ONLY_VARIANT,
    MEAN_STD_VARIANT,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    STD_CORRECTION,
    architecture_config,
    parameter_audit,
    train_fold,
)
from ultrasound_decoding.multiframe.training import DeepTrainingConfig


OUTPUT_VERSION = "fcnn_mean_std_temporal_statistics_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
MAX_FOLDS = 10
FORMAL_EPOCHS = 40
FORMAL_BATCH_SIZE = 16
FORMAL_FOLD_SOURCE_VERSION = "multiscale_temporal1d_v1.0.0"
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
TIE_TOLERANCE = 1e-12
HISTORICAL_FCNN_MEANPOOL_BA = 0.6016
REQUIRED_TASK_FILES = (
    "result.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "training_history.csv",
    "normalization_audit.json",
    "temporal_statistics_audit.json",
    "model_config.json",
)
REQUIRED_FINAL_OUTPUTS = (
    "task_level_results.csv",
    "seed_summary.csv",
    "session_summary.csv",
    "overall_summary.csv",
    "paired_sign_flip.csv",
    "overfitting_audit.csv",
    "predictions.csv",
    "confusion_matrices.csv",
    "training_history.csv",
    "temporal_statistics_audit.csv",
    "parameter_audit.csv",
    "decision_rule_audit.json",
    "fcnn_mean_std_temporal_statistics_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired formal clean4 FCNN bottleneck temporal mean-only versus "
            "mean+population-std experiment."
        )
    )
    parser.add_argument(
        "--stage", choices=("plan", "sanity", "full", "status"), required=True
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sessions", nargs="+", default=list(EXPECTED_SESSIONS))
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs" / OUTPUT_VERSION
    )
    parser.add_argument(
        "--formal-fold-run-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "multiscale_temporal1d_v1",
    )
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--sanity-epochs", type=int, default=1)
    parser.add_argument(
        "--review-approved",
        action="store_true",
        help="Required before the formal server run can start.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def frozen_training_config(
    batch_size: int = FORMAL_BATCH_SIZE, epochs: int = FORMAL_EPOCHS
) -> DeepTrainingConfig:
    return DeepTrainingConfig(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=int(batch_size),
        max_epochs=int(epochs),
        dropout=0.25,
        loss="cross_entropy",
    )


def source_paths(project_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        project_root
        / "src/ultrasound_decoding/multiframe/fcnn_temporal_statistics.py",
        project_root / "src/ultrasound_decoding/multiframe/models.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/cv.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "configs/fcnn_mean_std_temporal_statistics_v1.json",
        project_root / "docs/fcnn_mean_std_temporal_statistics_v1.md",
        project_root / "scripts/baselines/run_multiscale_temporal1d.py",
        project_root / "scripts/baselines/run_mamba_visual_binary.py",
    ]


def frozen_experiment_config(batch_size: int) -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "question": (
            "does population temporal std in the unchanged formal FCNN "
            "three-dimensional bottleneck improve held-out-cycle generalization"
        ),
        "sessions": list(EXPECTED_SESSIONS),
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "task": "binary_presence",
        "class_mapping": TASK_CLASS_NAMES[TASK_NAME],
        "input_protocol": "clean4",
        "raw_input_shape": list(EXPECTED_BLOCK_SHAPE),
        "model_input_shape": ["B", 4, 1, 128, 501],
        "bottleneck_shape": ["B", 4, BOTTLENECK_DIM],
        "variants": list(INPUT_VARIANTS),
        "temporal_statistics": {
            MEAN_ONLY_VARIANT: "mean_t(z) -> Linear(3,2)",
            MEAN_STD_VARIANT: (
                "concat(mean_t(z),std_t(z,correction=0)) -> Linear(6,2)"
            ),
            "std_correction": STD_CORRECTION,
            "statistics_space": "normalized-frame shared-FCNN bottleneck",
            "secondary_feature_scaling": False,
        },
        "preprocessing": (
            "clean4 -> per-frame arcsinh -> outer-train-fold all-frame "
            "pixel z-score -> unchanged shared FCNN frame encoder -> "
            "bottleneck temporal statistics"
        ),
        "normalization": (
            "pixel z-score fit on outer-training blocks and all four real frames only"
        ),
        "architectures": {
            variant: architecture_config(variant) for variant in INPUT_VARIANTS
        },
        "parameter_audit": parameter_audit(),
        "cv": "exact formal clean4 cycle-grouped folds, max_folds=10",
        "seeds": list(SEEDS),
        "training": frozen_training_config(batch_size).__dict__,
        "epoch_selection": "fixed 40 epochs; no validation or early stopping",
        "final_train_accuracy": (
            "mean across folds of fixed epoch-40 real-training-sample accuracy"
        ),
        "historical_control_reused": False,
        "paired_controls": (
            "same samples, folds, seeds, encoder initialization procedure, "
            "normalization, optimizer, learning rate, batch size and 40 epochs"
        ),
        "test_used_for_normalization": False,
        "test_used_for_feature_scaling": False,
        "test_used_for_model_selection": False,
        "test_used_for_early_stopping": False,
        "decision_rule": {
            "overall_mean_delta_BA_at_least": 0.010,
            "weak_mean_delta_BA_at_least": 0.020,
            "weak_improved_at_least": 4,
            "strong_mean_delta_BA_strictly_greater_than": -0.010,
            "single_session_dominance_diagnostic": (
                "largest improvement and leave-largest-improvement-out delta"
            ),
        },
        "automatic_next_stage": False,
    }


def formal_fold_source_identity(
    project_root: Path, run_dir: Path
) -> dict[str, Any]:
    completion_path = run_dir / "RUN_COMPLETE.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"formal fold source is incomplete: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or int(completion.get("completed_tasks", -1)) != 492
        or int(completion.get("total_tasks", -1)) != 492
        or completion.get("model_implementation_version")
        != FORMAL_FOLD_SOURCE_VERSION
    ):
        raise AssertionError("formal fold source must be completed 492/492")
    manifest_hashes = {}
    for session in EXPECTED_SESSIONS:
        path = run_dir / "audit" / f"session_{session}" / "split_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(f"formal fold source lacks {path}")
        manifest_hashes[session] = framework.file_sha256(path)
    try:
        display_path = str(run_dir.relative_to(project_root))
    except ValueError:
        display_path = str(run_dir)
    return {
        "run_dir": display_path,
        "completion_sha256": framework.file_sha256(completion_path),
        "run_fingerprint": str(completion["run_fingerprint"]),
        "model_implementation_version": FORMAL_FOLD_SOURCE_VERSION,
        "completed_tasks": 492,
        "session_manifest_sha256": manifest_hashes,
    }


def run_identity(args: argparse.Namespace) -> dict[str, Any]:
    runtime = framework.runtime_environment_signature()
    runtime.update(
        {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_names": [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ],
        }
    )
    return {
        "experiment_config": frozen_experiment_config(args.batch_size),
        "runtime_environment_signature": runtime,
        "git_commit": framework.git_text(args.project_root, "rev-parse", "HEAD"),
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "formal_fold_source": formal_fold_source_identity(
            args.project_root, args.formal_fold_run_dir
        ),
        "project_source_sha256": {
            str(path.relative_to(args.project_root)): framework.file_sha256(path)
            for path in source_paths(args.project_root)
        },
    }


def write_run_metadata(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if framework.fingerprint(observed) != framework.fingerprint(identity):
            if (args.output_dir / "task_plan.csv").exists() or (
                args.output_dir / "tasks"
            ).exists():
                raise RuntimeError(
                    "existing tasks have another code/config/environment fingerprint"
                )
    framework.atomic_json(config_path, identity)
    framework.atomic_json(
        args.output_dir / "environment.json",
        {
            "runtime_environment_signature": identity[
                "runtime_environment_signature"
            ],
            "created_utc": utc_now(),
        },
    )
    command = shlex.join(sys.argv) + "\n"
    framework.atomic_text(args.output_dir / "command.txt", command)
    framework.atomic_text(args.output_dir / f"{args.stage}_command.txt", command)
    framework.atomic_json(
        args.output_dir / "git_state.json",
        {
            "commit": framework.git_text(args.project_root, "rev-parse", "HEAD"),
            "branch": framework.git_text(
                args.project_root, "branch", "--show-current"
            ),
            "changed_files": framework.git_text(
                args.project_root, "status", "--short"
            ).splitlines(),
            "diff_stat": framework.git_text(
                args.project_root, "diff", "--stat"
            ),
        },
    )


def audit_session(args: argparse.Namespace, session: str):
    manifest_path = (
        args.formal_fold_run_dir
        / "audit"
        / f"session_{session}"
        / "split_manifest.csv"
    )
    data = load_block_sequence_session(
        args.project_root,
        session,
        TASK_NAME,
        data_dir=args.data_dir or default_block_data_dir(args.project_root),
    )
    if tuple(data.X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(
            f"session {session}: expected {EXPECTED_BLOCK_SHAPE}, got {data.X.shape}"
        )
    splits = grouped_cv_splits(data.groups, max_folds=MAX_FOLDS)
    current = fold_audit.canonical_manifest(
        split_manifest(
            session,
            TASK_NAME,
            data.y,
            data.groups,
            splits=splits,
            max_folds=MAX_FOLDS,
        )
    )
    formal = fold_audit.canonical_manifest(pd.read_csv(manifest_path))
    if not current.equals(formal):
        raise AssertionError(f"session {session}: formal fold manifest drift")
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        overlap = np.intersect1d(data.groups[train_idx], data.groups[test_idx])
        if overlap.size:
            raise AssertionError(
                f"session {session} fold {fold}: cycle leakage {overlap.tolist()}"
            )
    audit_dir = args.output_dir / "audit" / f"session_{session}"
    framework.atomic_csv(audit_dir / "split_manifest.csv", current)
    framework.atomic_json(
        audit_dir / "dataset.json",
        {
            "session": session,
            "source_h5": str(data.source_h5_path),
            "source_metadata": str(data.source_metadata_path),
            "shape": list(data.X.shape),
            "n_cycles": int(data.n_cycles),
            "n_samples": int(data.n_blocks),
            "formal_clean4_fold_match": True,
            "train_test_cycle_overlap_all_folds": False,
            "formal_manifest_source": str(manifest_path),
        },
    )
    return data, splits


def task_key(session: str, variant: str, seed: int, fold: int) -> str:
    return f"{session}:{variant}:{seed}:{fold}"


def task_dir(
    output_dir: Path, session: str, variant: str, seed: int, fold: int
) -> Path:
    return (
        output_dir
        / "tasks"
        / f"session_{session}"
        / variant
        / f"seed_{seed}"
        / f"fold_{fold:02d}"
    )


def task_fingerprint(run_fp: str, row: dict[str, Any]) -> str:
    return framework.fingerprint(
        {
            "run_fingerprint": run_fp,
            "session": str(row["session"]),
            "variant": str(row["variant"]),
            "seed": int(row["seed"]),
            "fold": int(row["fold"]),
            "n_test_samples": int(row["n_test_samples"]),
            "train_cycles": str(row["train_cycles"]),
            "test_cycles": str(row["test_cycles"]),
            "config_fingerprint": str(row["config_fingerprint"]),
            "runtime_environment_fingerprint": str(
                row["runtime_environment_fingerprint"]
            ),
        }
    )


def validate_task_plan(plan: pd.DataFrame) -> dict[str, int]:
    required = {
        "session",
        "variant",
        "seed",
        "fold",
        "n_test_samples",
        "train_cycles",
        "test_cycles",
        "task_key",
        "task_fingerprint",
    }
    if not required.issubset(plan.columns):
        raise AssertionError(f"task plan missing {sorted(required-set(plan.columns))}")
    if plan["task_key"].duplicated().any():
        raise AssertionError("duplicate formal task key")
    if set(plan["session"].astype(str)) != set(EXPECTED_SESSIONS):
        raise AssertionError("formal session coverage drift")
    if set(plan["variant"].astype(str)) != set(INPUT_VARIANTS):
        raise AssertionError("formal variant coverage drift")
    if set(pd.to_numeric(plan["seed"]).astype(int)) != set(SEEDS):
        raise AssertionError("formal seed coverage drift")
    paired = plan.groupby(["session", "seed", "fold"])["variant"].nunique()
    if not paired.eq(2).all():
        raise AssertionError("mean-only and mean+std are not fully paired")
    fold_rows = plan[
        ["session", "fold", "train_cycles", "test_cycles"]
    ].drop_duplicates()
    if fold_rows.duplicated(["session", "fold"]).any():
        raise AssertionError("paired variants do not share fold membership")
    counts = {
        "number_of_sessions": 9,
        "number_of_variants": 2,
        "number_of_seeds": 3,
        "number_of_folds": len(fold_rows),
        "expected_total_tasks": len(plan),
    }
    expected = {
        "number_of_sessions": 9,
        "number_of_variants": 2,
        "number_of_seeds": 3,
        "number_of_folds": 82,
        "expected_total_tasks": 492,
    }
    if counts != expected:
        raise AssertionError(f"unexpected formal task counts: {counts}")
    return counts


def print_task_counts(counts: dict[str, int]) -> None:
    print(f"sessions: {counts['number_of_sessions']}", flush=True)
    print(f"variants: {counts['number_of_variants']}", flush=True)
    print(f"seeds: {counts['number_of_seeds']}", flush=True)
    print(f"total folds: {counts['number_of_folds']}", flush=True)
    print(f"expected tasks: {counts['expected_total_tasks']}", flush=True)


def build_task_plan(
    args: argparse.Namespace, identity: dict[str, Any]
) -> pd.DataFrame:
    run_fp = framework.fingerprint(identity)
    config_fp = framework.fingerprint(identity["experiment_config"])
    runtime_fp = framework.fingerprint(identity["runtime_environment_signature"])
    rows, audits = [], []
    for session in EXPECTED_SESSIONS:
        data, splits = audit_session(args, session)
        audits.append(
            {
                "session": session,
                "n_cycles": int(data.n_cycles),
                "n_samples": int(data.n_blocks),
                "n_folds": len(splits),
                "input_shape": framework.canonical_json(list(data.X.shape[1:])),
                "formal_clean4_fold_match": True,
                "cycle_overlap": False,
            }
        )
        for variant in INPUT_VARIANTS:
            for seed in SEEDS:
                for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                    row = {
                        "session": session,
                        "variant": variant,
                        "seed": seed,
                        "fold": fold,
                        "n_train_samples": len(train_idx),
                        "n_test_samples": len(test_idx),
                        "train_cycles": cycle_text(data.groups[train_idx]),
                        "test_cycles": cycle_text(data.groups[test_idx]),
                        "task_key": task_key(session, variant, seed, fold),
                        "config_fingerprint": config_fp,
                        "runtime_environment_fingerprint": runtime_fp,
                    }
                    row["task_fingerprint"] = task_fingerprint(run_fp, row)
                    rows.append(row)
        del data
    plan = pd.DataFrame(rows)
    counts = validate_task_plan(plan)
    framework.atomic_csv(args.output_dir / "task_plan.csv", plan)
    framework.atomic_csv(
        args.output_dir / "dataset_and_fold_audit.csv", pd.DataFrame(audits)
    )
    framework.atomic_json(
        args.output_dir / "task_plan_metadata.json",
        {
            "run_fingerprint": run_fp,
            **counts,
            "task_definition": "session x variant x seed x fold",
            "historical_control_reused": False,
            "paired_fold_indices": True,
            "created_utc": utc_now(),
        },
    )
    print_task_counts(counts)
    return plan


def load_or_build_task_plan(
    args: argparse.Namespace, identity: dict[str, Any]
) -> pd.DataFrame:
    plan_path = args.output_dir / "task_plan.csv"
    metadata_path = args.output_dir / "task_plan_metadata.json"
    run_fp = framework.fingerprint(identity)
    if not plan_path.exists() and not metadata_path.exists():
        return build_task_plan(args, identity)
    if not plan_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("task plan is incomplete; use a fresh output directory")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_fingerprint") != run_fp:
        raise RuntimeError("existing task plan has a different run fingerprint")
    plan = pd.read_csv(plan_path, dtype={"session": str})
    for row in plan.to_dict(orient="records"):
        if row.get("task_fingerprint") != task_fingerprint(run_fp, row):
            raise AssertionError("task plan contains an invalid task fingerprint")
    counts = validate_task_plan(plan)
    for key, value in counts.items():
        if int(metadata.get(key, -1)) != value:
            raise AssertionError(f"task-plan metadata mismatch for {key}")
    print_task_counts(counts)
    return plan


def _task_hashes(path: Path) -> dict[str, str]:
    return {name: framework.file_sha256(path / name) for name in REQUIRED_TASK_FILES}


def validate_result_model_provenance(
    result: dict[str, Any], expected_variant: str
) -> tuple[bool, str]:
    """Validate experiment-level model lineage and the paired variant label."""

    if expected_variant not in INPUT_VARIANTS:
        return False, f"unknown expected variant {expected_variant!r}"
    if result.get("model") != MODEL_NAME:
        return False, "result model identifier mismatch"
    if result.get("variant") != expected_variant:
        return False, "result variant identifier mismatch"
    return True, "validated"


def validate_completed_task(
    path: Path,
    expected: dict[str, Any],
    run_fp: str,
    *,
    raise_on_error: bool = False,
) -> tuple[bool, str]:
    def fail(message: str) -> tuple[bool, str]:
        if raise_on_error:
            raise AssertionError(f"invalid completed task {path}: {message}")
        return False, message

    missing = [
        name for name in ("COMPLETE.json", *REQUIRED_TASK_FILES)
        if not (path / name).is_file()
    ]
    if missing:
        return fail(f"missing files {missing}")
    try:
        complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        history = pd.read_csv(path / "training_history.csv", dtype={"session": str})
        normalization = json.loads(
            (path / "normalization_audit.json").read_text(encoding="utf-8")
        )
        statistics = json.loads(
            (path / "temporal_statistics_audit.json").read_text(encoding="utf-8")
        )
        model_config = json.loads(
            (path / "model_config.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")
    session = str(expected["session"])
    variant = str(expected["variant"])
    seed = int(expected["seed"])
    fold = int(expected["fold"])
    identity = (session, variant, seed, fold)
    task_fp = task_fingerprint(run_fp, expected)
    if complete.get("status") != "complete":
        return fail("completion status is not complete")
    if complete.get("task_key") != task_key(*identity):
        return fail("task key mismatch")
    if complete.get("artifact_sha256") != _task_hashes(path):
        return fail("artifact hash mismatch")
    for name, payload in (("complete", complete), ("result", result)):
        if payload.get("run_fingerprint") != run_fp:
            return fail(f"{name} run fingerprint mismatch")
        if payload.get("task_fingerprint") != task_fp:
            return fail(f"{name} task fingerprint mismatch")
        if payload.get("config_fingerprint") != str(expected["config_fingerprint"]):
            return fail(f"{name} config fingerprint mismatch")
        if payload.get("runtime_environment_fingerprint") != str(
            expected["runtime_environment_fingerprint"]
        ):
            return fail(f"{name} runtime fingerprint mismatch")
    observed = (
        str(result.get("session")),
        str(result.get("variant")),
        int(result.get("seed", -1)),
        int(result.get("fold", -1)),
    )
    if observed != identity:
        return fail("result identity mismatch")
    model_provenance_valid, model_provenance_reason = (
        validate_result_model_provenance(result, variant)
    )
    if not model_provenance_valid:
        return fail(model_provenance_reason)
    expected_parameters = parameter_audit()[
        f"{variant}_trainable_parameters"
    ]
    if int(result.get("trainable_parameters", -1)) != expected_parameters:
        return fail("parameter count mismatch")
    if framework.fingerprint(model_config) != framework.fingerprint(
        architecture_config(variant)
    ):
        return fail("architecture config mismatch")
    if int(result.get("actual_batch_size", -1)) != FORMAL_BATCH_SIZE:
        return fail("batch size mismatch")
    if int(result.get("final_epoch", -1)) != FORMAL_EPOCHS:
        return fail("result is not fixed epoch 40")
    if len(history) != FORMAL_EPOCHS or not np.array_equal(
        history["epoch"].to_numpy(int), np.arange(1, FORMAL_EPOCHS + 1)
    ):
        return fail("training history is not exactly epochs 1..40")
    if not np.isfinite(
        history[["train_loss", "train_accuracy"]].to_numpy(float)
    ).all():
        return fail("training history contains non-finite values")
    if not np.isclose(
        float(result.get("final_train_accuracy", np.nan)),
        float(history.iloc[-1]["train_accuracy"]),
        atol=1e-12,
    ):
        return fail("final train accuracy is not epoch-40 accuracy")
    if (
        normalization.get("phase") != "outer_train_fold_only"
        or bool(normalization.get("target_used_for_stats", True))
        or bool(normalization.get("test_used_for_normalization_fit", True))
        or bool(normalization.get("test_used_for_feature_scaling", True))
        or bool(normalization.get("secondary_bottleneck_scaling", True))
    ):
        return fail("normalization leakage or protocol drift")
    if (
        str(statistics.get("session")),
        str(statistics.get("variant")),
        int(statistics.get("seed", -1)),
        int(statistics.get("fold", -1)),
    ) != identity:
        return fail("statistics audit identity mismatch")
    if bool(statistics.get("test_used_for_fitted_statistics", True)) or bool(
        statistics.get("test_used_for_model_selection", True)
    ):
        return fail("test information entered fitted statistics or selection")
    train_stats = statistics.get("train", {})
    if (
        int(train_stats.get("std_correction", -1)) != STD_CORRECTION
        or not bool(train_stats.get("std_nonnegative", False))
        or int(train_stats.get("mean_channel_nan_count", -1)) != 0
        or int(train_stats.get("std_channel_nan_count", -1)) != 0
        or int(train_stats.get("mean_channel_inf_count", -1)) != 0
        or int(train_stats.get("std_channel_inf_count", -1)) != 0
    ):
        return fail("invalid temporal-statistics audit")
    expected_n = int(expected["n_test_samples"])
    if len(predictions) != expected_n:
        return fail("prediction count mismatch")
    probabilities = predictions[["probability_0", "probability_1"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        return fail("invalid probabilities")
    metrics = classification_metrics(
        predictions["y_true"].to_numpy(int),
        predictions["y_pred"].to_numpy(int),
    )
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        if not np.isclose(
            float(result.get(metric, np.nan)), metrics[metric], atol=1e-12
        ):
            return fail(f"stored {metric} differs from predictions")
    return True, "validated"


def update_status(
    args: argparse.Namespace, plan: pd.DataFrame, run_fp: str
) -> pd.DataFrame:
    rows = []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir,
            str(expected["session"]),
            str(expected["variant"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        valid, reason = validate_completed_task(path, expected, run_fp)
        rows.append(
            {
                "session": str(expected["session"]),
                "variant": str(expected["variant"]),
                "seed": int(expected["seed"]),
                "fold": int(expected["fold"]),
                "status": "complete" if valid else "pending",
                "validation": reason,
                "task_dir": str(path),
            }
        )
    status = pd.DataFrame(rows)
    framework.atomic_csv(args.output_dir / "run_status.csv", status)
    completed = int(status["status"].eq("complete").sum())
    print(
        f"STATUS completed={completed} pending={len(status)-completed} total={len(status)}",
        flush=True,
    )
    return status


def _flatten_statistics_audit(audit: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: audit[key]
        for key in (
            "session",
            "variant",
            "seed",
            "fold",
            "train_cycles",
            "test_cycles",
            "test_used_for_fitted_statistics",
            "test_used_for_model_selection",
        )
    }
    for split_name in ("train", "test_diagnostic_only"):
        for key, value in audit[split_name].items():
            if isinstance(value, (list, dict)):
                value = framework.canonical_json(value)
            row[f"{split_name}_{key}"] = value
    return row


def run_sanity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if args.workers != 0:
        raise ValueError("sanity requires --workers 0")
    if args.sanity_epochs not in (1, 2):
        raise ValueError("sanity is restricted to 1 or 2 epochs")
    data, splits = audit_session(args, "710")
    train_idx, test_idx = splits[0]
    config = frozen_training_config(
        batch_size=args.batch_size, epochs=args.sanity_epochs
    )
    audits = []
    for variant in INPUT_VARIANTS:
        result = train_fold(
            data.X[train_idx],
            data.y[train_idx],
            data.X[test_idx],
            np.asarray([0, 1]),
            variant=variant,
            session="710",
            fold=1,
            seed=0,
            train_cycles=cycle_text(data.groups[train_idx]),
            test_cycles=cycle_text(data.groups[test_idx]),
            training_config=config,
            device=args.device,
            workers=0,
        )
        audit = result.statistics_audit
        train_stats = audit["train"]
        if variant == MEAN_STD_VARIANT and bool(train_stats["std_all_zero"]):
            raise AssertionError("sanity std channel is unexpectedly all zero")
        print(
            f"{variant} mean min/max/mean/std="
            f"{train_stats['mean_channel_min']:.6f}/"
            f"{train_stats['mean_channel_max']:.6f}/"
            f"{train_stats['mean_channel_mean']:.6f}/"
            f"{train_stats['mean_channel_std']:.6f}",
            flush=True,
        )
        print(
            f"{variant} std min/max/mean/std="
            f"{train_stats['std_channel_min']:.6f}/"
            f"{train_stats['std_channel_max']:.6f}/"
            f"{train_stats['std_channel_mean']:.6f}/"
            f"{train_stats['std_channel_std']:.6f}",
            flush=True,
        )
        audits.append(_flatten_statistics_audit(audit))
    parameter_row = parameter_audit()
    framework.atomic_csv(
        args.output_dir / "audit" / "sanity_temporal_statistics.csv",
        pd.DataFrame(audits),
    )
    framework.atomic_json(
        args.output_dir / "audit" / "parameter_audit.json", parameter_row
    )
    framework.atomic_json(
        args.output_dir / "SANITY_COMPLETE.json",
        {
            "status": "complete",
            "formal_training_started": False,
            "session": "710",
            "fold": 1,
            "seed": 0,
            "epochs": args.sanity_epochs,
            "variants": list(INPUT_VARIANTS),
            "std_correction": STD_CORRECTION,
            "run_fingerprint": framework.fingerprint(identity),
            "completed_utc": utc_now(),
        },
    )
    print("SANITY COMPLETE; formal training was not started", flush=True)


def write_fold_task(
    args: argparse.Namespace,
    identity: dict[str, Any],
    expected: dict[str, Any],
    data,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    session = str(expected["session"])
    variant = str(expected["variant"])
    seed = int(expected["seed"])
    fold = int(expected["fold"])
    run_fp = framework.fingerprint(identity)
    task_fp = task_fingerprint(run_fp, expected)
    observed_train_cycles = cycle_text(data.groups[train_idx])
    observed_test_cycles = cycle_text(data.groups[test_idx])
    if (
        observed_train_cycles != str(expected["train_cycles"])
        or observed_test_cycles != str(expected["test_cycles"])
    ):
        raise AssertionError("runtime cycle membership differs from task plan")
    if np.intersect1d(data.groups[train_idx], data.groups[test_idx]).size:
        raise AssertionError("outer train/test cycle leakage")
    result = train_fold(
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        np.asarray([0, 1]),
        variant=variant,
        session=session,
        fold=fold,
        seed=seed,
        train_cycles=observed_train_cycles,
        test_cycles=observed_test_cycles,
        training_config=frozen_training_config(args.batch_size, FORMAL_EPOCHS),
        device=args.device,
        workers=args.workers,
    )
    metrics = classification_metrics(data.y[test_idx], result.predictions)
    path = task_dir(args.output_dir, session, variant, seed, fold)
    path.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for sample_index, source_index in enumerate(test_idx):
        prediction_rows.append(
            {
                "session": session,
                "variant": variant,
                "seed": seed,
                "fold": fold,
                "sample_index": int(sample_index),
                "source_index": int(source_index),
                "cycle": int(data.groups[source_index]),
                "block_name": str(data.metadata.iloc[int(source_index)]["block_name"]),
                "y_true": int(data.y[source_index]),
                "y_pred": int(result.predictions[sample_index]),
                "probability_0": float(result.probabilities[sample_index, 0]),
                "probability_1": float(result.probabilities[sample_index, 1]),
            }
        )
    cm = confusion_matrix(
        data.y[test_idx], result.predictions, np.asarray([0, 1])
    )
    confusion_rows = [
        {
            "session": session,
            "variant": variant,
            "seed": seed,
            "fold": fold,
            "true_label": truth,
            "predicted_label": predicted,
            "count": int(cm[truth, predicted]),
        }
        for truth in range(2)
        for predicted in range(2)
    ]
    history = pd.DataFrame(result.history)
    history.insert(0, "fold", fold)
    history.insert(0, "seed", seed)
    history.insert(0, "variant", variant)
    history.insert(0, "session", session)
    best_i = int(history["train_accuracy"].to_numpy(float).argmax())
    final_train_accuracy = float(history.iloc[-1]["train_accuracy"])
    result_payload = {
        "session": session,
        "variant": variant,
        "seed": seed,
        "fold": fold,
        "model": MODEL_NAME,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "train_cycles": observed_train_cycles,
        "test_cycles": observed_test_cycles,
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "final_train_accuracy": final_train_accuracy,
        "train_accuracy": final_train_accuracy,
        "final_epoch": FORMAL_EPOCHS,
        "trained_epochs": result.final_trained_epochs,
        "best_epoch": int(history.iloc[best_i]["epoch"]),
        "best_epoch_definition": (
            "descriptive max train accuracy; never used for train-test gap"
        ),
        "best_train_accuracy": float(history.iloc[best_i]["train_accuracy"]),
        "final_training_loss": float(result.final_training_loss),
        "mean_training_loss": float(history["train_loss"].mean()),
        "trainable_parameters": int(result.model_parameters),
        "actual_batch_size": min(args.batch_size, len(train_idx)),
        "device": result.device,
        "std_correction": STD_CORRECTION,
        "historical_control_reused": False,
        "early_stopping_used": False,
        "selected_epoch": FORMAL_EPOCHS,
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": str(expected["config_fingerprint"]),
        "runtime_environment_fingerprint": str(
            expected["runtime_environment_fingerprint"]
        ),
    }
    framework.atomic_json(path / "result.json", result_payload)
    framework.atomic_csv(path / "predictions.csv", pd.DataFrame(prediction_rows))
    framework.atomic_csv(
        path / "confusion_matrix.csv", pd.DataFrame(confusion_rows)
    )
    framework.atomic_csv(path / "training_history.csv", history)
    framework.atomic_json(
        path / "normalization_audit.json", result.normalization_audit
    )
    framework.atomic_json(
        path / "temporal_statistics_audit.json", result.statistics_audit
    )
    framework.atomic_json(path / "model_config.json", result.model_config)
    framework.atomic_json(
        path / "COMPLETE.json",
        {
            "status": "complete",
            "task_key": task_key(session, variant, seed, fold),
            "run_fingerprint": run_fp,
            "task_fingerprint": task_fp,
            "config_fingerprint": str(expected["config_fingerprint"]),
            "runtime_environment_fingerprint": str(
                expected["runtime_environment_fingerprint"]
            ),
            "artifact_sha256": _task_hashes(path),
            "completed_utc": utc_now(),
        },
    )
    validate_completed_task(path, expected, run_fp, raise_on_error=True)


def read_all_tasks(
    args: argparse.Namespace, plan: pd.DataFrame, run_fp: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results, predictions, confusions, histories, audits = [], [], [], [], []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir,
            str(expected["session"]),
            str(expected["variant"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        validate_completed_task(path, expected, run_fp, raise_on_error=True)
        results.append(json.loads((path / "result.json").read_text(encoding="utf-8")))
        predictions.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        confusions.append(pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str}))
        histories.append(pd.read_csv(path / "training_history.csv", dtype={"session": str}))
        audits.append(
            _flatten_statistics_audit(
                json.loads(
                    (path / "temporal_statistics_audit.json").read_text(
                        encoding="utf-8"
                    )
                )
            )
        )
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True),
        pd.concat(confusions, ignore_index=True),
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(audits),
    )


def exact_two_sided_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    statistics = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def build_seed_summary(
    per_fold: pd.DataFrame, predictions: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    final = history[pd.to_numeric(history["epoch"]).eq(FORMAL_EPOCHS)].copy()
    expected_folds = per_fold.groupby(["session", "variant", "seed"])[
        "fold"
    ].nunique()
    final_folds = final.groupby(["session", "variant", "seed"])["fold"].nunique()
    if not expected_folds.equals(final_folds):
        raise AssertionError("epoch-40 fold coverage differs from task coverage")
    rows = []
    for (session, variant, seed), group in per_fold.groupby(
        ["session", "variant", "seed"], sort=True
    ):
        pred = predictions[
            predictions["session"].eq(session)
            & predictions["variant"].eq(variant)
            & pd.to_numeric(predictions["seed"]).eq(seed)
        ]
        oof_ba = classification_metrics(
            pred["y_true"].to_numpy(int), pred["y_pred"].to_numpy(int)
        )["balanced_accuracy"]
        final_group = final[
            final["session"].eq(session)
            & final["variant"].eq(variant)
            & pd.to_numeric(final["seed"]).eq(seed)
        ]
        final_train = float(final_group["train_accuracy"].mean())
        rows.append(
            {
                "session": session,
                "variant": variant,
                "seed": int(seed),
                "mean_oof_BA": float(oof_ba),
                "final_train_accuracy": final_train,
                "train_accuracy": final_train,
                "train_test_gap": final_train - float(oof_ba),
                "final_epoch": FORMAL_EPOCHS,
                "number_of_folds": int(group["fold"].nunique()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 9 * 2 * 3:
        raise AssertionError("seed summary coverage is incomplete")
    return result


def build_session_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        seed_summary.groupby(["session", "variant"], as_index=False)
        .agg(
            BA=("mean_oof_BA", "mean"),
            train_acc=("final_train_accuracy", "mean"),
            gap=("train_test_gap", "mean"),
        )
        .sort_values(["session", "variant"])
    )
    ba = grouped.pivot(index="session", columns="variant", values="BA")
    train = grouped.pivot(index="session", columns="variant", values="train_acc")
    gap = grouped.pivot(index="session", columns="variant", values="gap")
    rows = []
    for session in EXPECTED_SESSIONS:
        rows.append(
            {
                "session": session,
                "mean_only_BA": float(ba.loc[session, MEAN_ONLY_VARIANT]),
                "mean_std_BA": float(ba.loc[session, MEAN_STD_VARIANT]),
                "delta_BA": float(
                    ba.loc[session, MEAN_STD_VARIANT]
                    - ba.loc[session, MEAN_ONLY_VARIANT]
                ),
                "mean_only_train_acc": float(
                    train.loc[session, MEAN_ONLY_VARIANT]
                ),
                "mean_std_train_acc": float(train.loc[session, MEAN_STD_VARIANT]),
                "mean_only_gap": float(gap.loc[session, MEAN_ONLY_VARIANT]),
                "mean_std_gap": float(gap.loc[session, MEAN_STD_VARIANT]),
                "delta_gap": float(
                    gap.loc[session, MEAN_STD_VARIANT]
                    - gap.loc[session, MEAN_ONLY_VARIANT]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_overall_and_decision(
    session_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    deltas = session_summary.set_index("session")["delta_BA"]
    strong = session_summary[session_summary["session"].isin(STRONG_SESSIONS)]
    weak = session_summary[session_summary["session"].isin(WEAK_SESSIONS)]
    mean_only = float(session_summary["mean_only_BA"].mean())
    mean_std = float(session_summary["mean_std_BA"].mean())
    strong_only = float(strong["mean_only_BA"].mean())
    strong_std = float(strong["mean_std_BA"].mean())
    weak_only = float(weak["mean_only_BA"].mean())
    weak_std = float(weak["mean_std_BA"].mean())
    weak_deltas = weak.set_index("session")["delta_BA"]
    max_session = str(deltas.idxmax())
    leave_max_out = float(deltas.drop(index=max_session).mean())
    overall = pd.DataFrame(
        [
            {
                "historical_fcnn_meanpool_reference_BA": HISTORICAL_FCNN_MEANPOOL_BA,
                "mean_only_mean_BA": mean_only,
                "mean_std_mean_BA": mean_std,
                "overall_delta_BA": float(deltas.mean()),
                "median_delta_BA": float(deltas.median()),
                "mean_only_strong_mean_BA": strong_only,
                "mean_std_strong_mean_BA": strong_std,
                "strong_delta_BA": strong_std - strong_only,
                "mean_only_weak_mean_BA": weak_only,
                "mean_std_weak_mean_BA": weak_std,
                "weak_delta_BA": weak_std - weak_only,
                "weak_improved": int((weak_deltas > TIE_TOLERANCE).sum()),
                "weak_tied": int((weak_deltas.abs() <= TIE_TOLERANCE).sum()),
                "weak_worsened": int((weak_deltas < -TIE_TOLERANCE).sum()),
                "all_improved": int((deltas > TIE_TOLERANCE).sum()),
                "all_tied": int((deltas.abs() <= TIE_TOLERANCE).sum()),
                "all_worsened": int((deltas < -TIE_TOLERANCE).sum()),
                "largest_improvement_session": max_session,
                "largest_single_session_improvement": float(deltas.loc[max_session]),
                "leave_largest_improvement_out_overall_delta_BA": leave_max_out,
            }
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "comparison": "mean_std_vs_mean_only",
                "n_sessions": len(deltas),
                "mean_delta_BA": float(deltas.mean()),
                "median_delta_BA": float(deltas.median()),
                "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(
                    deltas.to_numpy(float)
                ),
                "session_deltas_json": framework.canonical_json(deltas.to_dict()),
            }
        ]
    )
    checks = {
        "overall_mean_delta_BA_ge_0.010": float(deltas.mean()) >= 0.010,
        "weak_6_mean_delta_BA_ge_0.020": weak_std - weak_only >= 0.020,
        "at_least_4_of_6_weak_sessions_improved": int(
            (weak_deltas > TIE_TOLERANCE).sum()
        )
        >= 4,
        "strong_3_mean_delta_BA_gt_minus_0.010": strong_std - strong_only
        > -0.010,
    }
    not_single_session_driven = leave_max_out > 0.0
    decision = {
        "criteria_predefined": True,
        "checks": checks,
        "single_session_dominance_diagnostic": {
            "largest_improvement_session": max_session,
            "largest_single_session_improvement": float(deltas.loc[max_session]),
            "leave_largest_improvement_out_overall_delta_BA": leave_max_out,
            "trend_remains_positive_without_largest_improvement": (
                not_single_session_driven
            ),
        },
        "decision": (
            "supports_continue_temporal_statistics_route"
            if all(checks.values()) and not_single_session_driven
            else "does_not_support_temporal_statistics_route"
        ),
        "automatic_next_stage_started": False,
    }
    return overall, paired, decision


def build_report(
    session_summary: pd.DataFrame,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    summary = overall.iloc[0]
    test = paired.iloc[0]
    params = parameter_audit()
    lines = [
        "# FCNN Mean+Std Temporal Statistics v1",
        "",
        "## Frozen protocol",
        "",
        "Both variants are rerun and paired by clean4 samples, cycle folds, seeds, "
        "shared FCNN encoder initialization procedure, outer-train normalization, "
        "optimizer, batch size, and fixed 40 epochs. Historical outputs are not "
        "used as the paired control.",
        "",
        "Actual preprocessing is `clean4 -> per-frame arcsinh -> outer-train-fold "
        "all-frame pixel z-score -> unchanged shared FCNN encoder -> [B,4,3] "
        "bottleneck`. Mean-only applies `mean_t -> Linear(3,2)`, exactly the formal "
        "FCNNMeanPool implementation. Mean+Std applies "
        "`concat(mean_t, std_t(correction=0)) -> Linear(6,2)`.",
        "",
        "No secondary bottleneck scaling is fitted. Test data do not enter "
        "normalization, feature scaling, early stopping, or model selection.",
        "",
        "## Parameter audit",
        "",
        f"- Mean-only: {params['mean_only_trainable_parameters']}",
        f"- Mean+Std: {params['mean_std_trainable_parameters']}",
        f"- Delta: {params['parameter_delta']} "
        f"({params['parameter_delta_percentage']:.6f}%)",
        "- The delta is only the classifier input expansion from 3 to 6.",
        "",
        "## Session results",
        "",
        "| session | mean-only train | mean-only BA | mean-only gap | mean+std train | mean+std BA | mean+std gap | delta BA | delta gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in session_summary.itertuples(index=False):
        lines.append(
            f"| {row.session} | {row.mean_only_train_acc:.4f} | "
            f"{row.mean_only_BA:.4f} | {row.mean_only_gap:.4f} | "
            f"{row.mean_std_train_acc:.4f} | {row.mean_std_BA:.4f} | "
            f"{row.mean_std_gap:.4f} | {row.delta_BA:+.4f} | "
            f"{row.delta_gap:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Historical FCNN mean-pool reference: approximately {HISTORICAL_FCNN_MEANPOOL_BA:.4f}",
            f"- New mean-only mean BA: {summary.mean_only_mean_BA:.4f}",
            f"- New mean+std mean BA: {summary.mean_std_mean_BA:.4f}",
            f"- Overall delta BA: {summary.overall_delta_BA:+.4f}",
            f"- Median delta BA: {summary.median_delta_BA:+.4f}",
            f"- Strong-3 mean-only/mean+std/delta: "
            f"{summary.mean_only_strong_mean_BA:.4f} / "
            f"{summary.mean_std_strong_mean_BA:.4f} / "
            f"{summary.strong_delta_BA:+.4f}",
            f"- Weak-6 mean-only/mean+std/delta: "
            f"{summary.mean_only_weak_mean_BA:.4f} / "
            f"{summary.mean_std_weak_mean_BA:.4f} / "
            f"{summary.weak_delta_BA:+.4f}",
            f"- Weak improved/tied/worsened: {int(summary.weak_improved)}/"
            f"{int(summary.weak_tied)}/{int(summary.weak_worsened)}",
            f"- All improved/tied/worsened: {int(summary.all_improved)}/"
            f"{int(summary.all_tied)}/{int(summary.all_worsened)}",
            f"- Exact paired two-sided sign-flip p: "
            f"{test.exact_two_sided_sign_flip_p:.6f}",
            f"- Largest improvement: session {summary.largest_improvement_session}, "
            f"delta={summary.largest_single_session_improvement:+.4f}",
            f"- Leave-largest-improvement-out delta: "
            f"{summary.leave_largest_improvement_out_overall_delta_BA:+.4f}",
            "",
            "## Predefined decision",
            "",
            f"`{decision['decision']}`",
            "",
            "No additional temporal statistic or downstream experiment is started automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_outputs(
    args: argparse.Namespace, plan: pd.DataFrame, identity: dict[str, Any]
) -> None:
    run_fp = framework.fingerprint(identity)
    per_fold, predictions, confusions, history, statistics = read_all_tasks(
        args, plan, run_fp
    )
    per_fold = per_fold.sort_values(["session", "seed", "fold", "variant"])
    predictions = predictions.sort_values(
        ["session", "seed", "fold", "variant", "sample_index"]
    )
    confusions = confusions.sort_values(
        ["session", "seed", "fold", "variant", "true_label", "predicted_label"]
    )
    history = history.sort_values(["session", "seed", "fold", "variant", "epoch"])
    statistics = statistics.sort_values(["session", "seed", "fold", "variant"])
    seed_summary = build_seed_summary(per_fold, predictions, history)
    session_summary = build_session_summary(seed_summary)
    overfitting = seed_summary[
        [
            "session",
            "variant",
            "seed",
            "final_train_accuracy",
            "mean_oof_BA",
            "train_test_gap",
            "final_epoch",
        ]
    ].rename(columns={"mean_oof_BA": "OOF_test_BA"})
    overall, paired, decision = build_overall_and_decision(session_summary)
    framework.atomic_csv(args.output_dir / "task_level_results.csv", per_fold)
    framework.atomic_csv(args.output_dir / "predictions.csv", predictions)
    framework.atomic_csv(args.output_dir / "confusion_matrices.csv", confusions)
    framework.atomic_csv(args.output_dir / "training_history.csv", history)
    framework.atomic_csv(
        args.output_dir / "temporal_statistics_audit.csv", statistics
    )
    framework.atomic_csv(args.output_dir / "seed_summary.csv", seed_summary)
    framework.atomic_csv(args.output_dir / "session_summary.csv", session_summary)
    framework.atomic_csv(args.output_dir / "overfitting_audit.csv", overfitting)
    framework.atomic_csv(args.output_dir / "overall_summary.csv", overall)
    framework.atomic_csv(args.output_dir / "paired_sign_flip.csv", paired)
    framework.atomic_csv(
        args.output_dir / "parameter_audit.csv", pd.DataFrame([parameter_audit()])
    )
    framework.atomic_json(args.output_dir / "decision_rule_audit.json", decision)
    framework.atomic_text(
        args.output_dir / "fcnn_mean_std_temporal_statistics_report.md",
        build_report(session_summary, overall, paired, decision),
    )


def run_cuda_preflight(args: argparse.Namespace) -> None:
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise AssertionError("formal protocol requires batch size 16")
    device = torch.device(args.device if args.device != "auto" else "cuda")
    rows = []
    for variant in INPUT_VARIANTS:
        from ultrasound_decoding.multiframe.fcnn_temporal_statistics import build_model

        model = build_model(variant).to(device)
        inputs = torch.zeros((FORMAL_BATCH_SIZE, 4, 1, 128, 501), device=device)
        targets = torch.arange(FORMAL_BATCH_SIZE, device=device) % 2
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-3
        )
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        optimizer.step()
        if (
            tuple(logits.shape) != (FORMAL_BATCH_SIZE, 2)
            or not bool(torch.isfinite(loss).item())
            or not gradients_finite
        ):
            raise AssertionError(f"{variant}: CUDA preflight failed")
        rows.append(
            {
                "variant": variant,
                "loss_finite": True,
                "gradients_finite": True,
                "optimizer_step_success": True,
                "trainable_parameters": parameter_audit()[
                    f"{variant}_trainable_parameters"
                ],
                "batch_size": FORMAL_BATCH_SIZE,
            }
        )
        del model, inputs, targets, optimizer, logits, loss
        torch.cuda.empty_cache()
    framework.atomic_json(
        args.output_dir / "audit" / "cuda_memory_preflight.json",
        {"formal_training_started": False, "device": str(device), "variants": rows},
    )


def run_full(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if not args.review_approved:
        raise RuntimeError("full run is locked until --review-approved is supplied")
    if not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    if args.device != "auto" and not str(args.device).startswith("cuda"):
        raise RuntimeError("formal run requires a CUDA device")
    expected_env = Path("/data2/yuq1ngr/conda_envs/fus")
    if expected_env not in Path(sys.executable).resolve().parents:
        raise RuntimeError(f"formal run must use {expected_env}; got {sys.executable}")
    invalid = sorted(set(map(str, args.sessions)) - set(EXPECTED_SESSIONS))
    if invalid:
        raise ValueError(f"unknown sessions: {invalid}")
    run_cuda_preflight(args)
    plan = load_or_build_task_plan(args, identity)
    run_fp = framework.fingerprint(identity)
    status = update_status(args, plan, run_fp)
    completed = int(status["status"].eq("complete").sum())
    total = len(plan)
    for session in map(str, args.sessions):
        data, splits = audit_session(args, session)
        session_plan = plan[plan["session"].eq(session)]
        for expected in session_plan.to_dict(orient="records"):
            variant = str(expected["variant"])
            fold = int(expected["fold"])
            seed = int(expected["seed"])
            path = task_dir(args.output_dir, session, variant, seed, fold)
            valid, _reason = validate_completed_task(path, expected, run_fp)
            if valid:
                print(
                    f"SKIP [{completed}/{total}] session={session} "
                    f"variant={variant} fold={fold} seed={seed}",
                    flush=True,
                )
                continue
            print(
                f"RUN  [{completed}/{total}] session={session} "
                f"variant={variant} fold={fold} seed={seed}",
                flush=True,
            )
            train_idx, test_idx = splits[fold - 1]
            write_fold_task(
                args, identity, expected, data, train_idx, test_idx
            )
            completed += 1
            print(
                f"DONE [{completed}/{total}] session={session} "
                f"variant={variant} fold={fold} seed={seed}",
                flush=True,
            )
        del data
    status = update_status(args, plan, run_fp)
    if not status["status"].eq("complete").all():
        print("PARTIAL RUN SAVED; rerun the identical GNU screen command", flush=True)
        return
    aggregate_outputs(args, plan, identity)
    missing = [
        name for name in REQUIRED_FINAL_OUTPUTS
        if not (args.output_dir / name).is_file()
    ]
    if missing:
        raise AssertionError(f"finalization missing outputs: {missing}")
    framework.atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "compute_environment": "server",
            "run_fingerprint": run_fp,
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "expected_tasks": len(plan),
            "total_tasks": len(plan),
            "completed_tasks": len(plan),
            "number_of_sessions": 9,
            "number_of_variants": 2,
            "number_of_seeds": 3,
            "number_of_folds": 82,
            "historical_control_reused": False,
            "required_outputs": list(REQUIRED_FINAL_OUTPUTS),
            "automatic_next_stage_started": False,
            "completed_utc": utc_now(),
        },
    )
    print("FULL RUN COMPLETE; STOP for manual analysis", flush=True)


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.formal_fold_run_dir = args.formal_fold_run_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise ValueError("formal and sanity protocol require --batch-size 16")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    identity = run_identity(args)
    write_run_metadata(args, identity)
    if args.stage == "plan":
        load_or_build_task_plan(args, identity)
    elif args.stage == "sanity":
        run_sanity(args, identity)
    elif args.stage == "full":
        run_full(args, identity)
    else:
        plan_path = args.output_dir / "task_plan.csv"
        metadata_path = args.output_dir / "task_plan_metadata.json"
        if not plan_path.is_file() or not metadata_path.is_file():
            print(f"NOT STARTED: no complete formal task plan at {plan_path}")
            return
        plan = load_or_build_task_plan(args, identity)
        update_status(args, plan, framework.fingerprint(identity))


if __name__ == "__main__":
    main()
