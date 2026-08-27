#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from ultrasound_decoding.multiframe.models import (
    CNN2DTemporal1D,
    count_trainable_parameters,
)
from ultrasound_decoding.multiframe.cycle_consistent_temporal1d import (
    CONSISTENCY_IMPLEMENTATION_VERSION,
    CYCLE_CONSISTENT_VARIANT,
    INPUT_VARIANTS,
    LAMBDA_CONSISTENCY,
    MODEL_IMPLEMENTATION_VERSION,
    RAW_VARIANT,
    cycle_consistency_loss,
    formal_architecture_config,
    total_training_loss,
    train_fold,
)
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_sequence_tensor,
    resolve_device,
)


OUTPUT_VERSION = "cycle_consistent_temporal1d_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
MAX_FOLDS = 10
FORMAL_EPOCHS = 40
FORMAL_BATCH_SIZE = 16
EXPECTED_PARAMETER_COUNT = 115890
FORMAL_FOLD_SOURCE_VERSION = "multiscale_temporal1d_v1.0.0"
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
TIE_TOLERANCE = 1e-12
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
    "pair_coverage_audit.csv",
    "representation_audit.csv",
    "decision_rule_audit.json",
    "cycle_consistent_temporal1d_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired formal clean4 raw vs cycle-consistent Temporal1D "
            "within-session experiment."
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
        "--benchmark-root",
        type=Path,
        default=(
            PROJECT_DIR
            / "results"
            / "runs"
            / "multiframe"
            / "block_clean4_binary_v1"
        ),
    )
    parser.add_argument(
        "--formal-fold-run-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "multiscale_temporal1d_v1",
        help=(
            "Exact completed formal run whose audit/session_*/split_manifest.csv "
            "files lock the cycle folds. No candidate-path fallback is used."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
    parser.add_argument("--sanity-epochs", type=int, default=1)
    parser.add_argument(
        "--review-approved",
        action="store_true",
        help="Required for full stage after code-review approval.",
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
        / "src/ultrasound_decoding/multiframe/cycle_consistent_temporal1d.py",
        project_root / "src/ultrasound_decoding/multiframe/models.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/cv.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "scripts/baselines/run_multiscale_temporal1d.py",
        project_root / "scripts/baselines/run_local_global_residual_mamba.py",
        project_root / "scripts/baselines/run_mamba_visual_binary.py",
    ]


def frozen_experiment_config(batch_size: int) -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "question": (
            "does same-block cross-cycle representation consistency improve "
            "generalization"
        ),
        "sessions": list(EXPECTED_SESSIONS),
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "task": TASK_NAME,
        "class_mapping": TASK_CLASS_NAMES[TASK_NAME],
        "stimulus_blocks": ["grating", "dot"],
        "nonstimulus_blocks": ["stop_after_grating", "static"],
        "input_unit": "one formal clean4 block",
        "input_shape": list(EXPECTED_BLOCK_SHAPE),
        "input_variants": list(INPUT_VARIANTS),
        "cycle_consistency": {
            "implementation_version": CONSISTENCY_IMPLEMENTATION_VERSION,
            "embedding": "64D temporal_conv output immediately before binary classifier",
            "positive_pairs": "same block identity and different training cycle; i<j",
            "loss": "mean(1-cosine_similarity(L2-normalized embeddings))",
            "negatives_used": False,
            "lambda_consistency": LAMBDA_CONSISTENCY,
            "lambda_searched": False,
            "pair_pool": "current mini-batch outer-training samples only",
            "test_metadata_or_embeddings_used_for_training": False,
        },
        "preprocessing": {
            RAW_VARIANT: "clean4 -> arcsinh -> train_fold_pixel_zscore",
            CYCLE_CONSISTENT_VARIANT: "clean4 -> arcsinh -> train_fold_pixel_zscore",
        },
        "cv": "exact formal clean4 cycle-grouped folds, max_folds=10",
        "normalization": "pixel z-score fit on outer-train blocks/all four frames only",
        "oof_primary_metric": "balanced_accuracy",
        "seeds": list(SEEDS),
        "architecture": formal_architecture_config(),
        "training": frozen_training_config(batch_size).__dict__,
        "epoch_selection": "fixed 40 epochs; no validation or early stopping",
        "patience": None,
        "test_used_for_normalization": False,
        "test_used_for_early_stopping": False,
        "test_used_for_model_selection": False,
        "test_block_identity_used": False,
        "test_embedding_used_for_training_or_selection": False,
        "historical_raw_reused": False,
        "paired_controls": (
            "same fold indices, seed, initialization procedure, model, optimizer, "
            "learning rate, batch size, epochs, and classifier"
        ),
        "decision_rule": {
            "overall_mean_delta_BA_at_least": 0.010,
            "weak_mean_delta_BA_at_least": 0.020,
            "weak_improved_at_least": 4,
            "strong_mean_delta_BA_strictly_greater_than": -0.010,
            "single_session_dominance_diagnostic": (
                "report largest improvement and leave-one-largest-out overall delta"
            ),
        },
        "automatic_next_stage": False,
    }


def formal_fold_source_identity(project_root: Path, run_dir: Path) -> dict[str, Any]:
    completion_path = run_dir / "RUN_COMPLETE.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"explicit formal fold source lacks RUN_COMPLETE.json: {completion_path}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or int(completion.get("completed_tasks", -1)) != 492
        or int(completion.get("total_tasks", -1)) != 492
        or completion.get("model_implementation_version")
        != FORMAL_FOLD_SOURCE_VERSION
    ):
        raise AssertionError(
            "formal fold source must be the completed multiscale Temporal1D v1 "
            f"run with 492/492 tasks: {completion_path}"
        )
    manifest_hashes = {}
    for session in EXPECTED_SESSIONS:
        path = run_dir / "audit" / f"session_{session}" / "split_manifest.csv"
        if not path.is_file():
            raise FileNotFoundError(f"formal fold source lacks manifest: {path}")
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
        "total_tasks": 492,
        "session_manifest_sha256": manifest_hashes,
    }


def run_identity(
    project_root: Path, batch_size: int, formal_fold_run_dir: Path
) -> dict[str, Any]:
    runtime = framework.runtime_environment_signature()
    runtime.update(
        {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    )
    paths = source_paths(project_root)
    return {
        "experiment_config": frozen_experiment_config(batch_size),
        "runtime_environment_signature": runtime,
        "git_commit": framework.git_text(project_root, "rev-parse", "HEAD"),
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "formal_fold_source": formal_fold_source_identity(
            project_root, formal_fold_run_dir
        ),
        "project_source_sha256": {
            str(path.relative_to(project_root)): framework.file_sha256(path)
            for path in paths
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
                    "existing tasks have another code/config/environment fingerprint; "
                    "use a new output directory"
                )
    framework.atomic_json(config_path, identity)
    framework.atomic_json(
        args.output_dir / "environment.json",
        {
            "compute_environment": (
                "server" if str(args.device).startswith("cuda") else "local"
            ),
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
            "diff_stat": framework.git_text(args.project_root, "diff", "--stat"),
        },
    )


def audit_session(args: argparse.Namespace, session: str):
    """Rebuild clean4 folds and compare them to one explicitly named formal run."""

    completion_path = args.formal_fold_run_dir / "RUN_COMPLETE.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"explicit formal fold source lacks RUN_COMPLETE.json: {completion_path}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if (
        completion.get("status") != "complete"
        or int(completion.get("completed_tasks", -1)) != 492
        or int(completion.get("total_tasks", -1)) != 492
        or completion.get("model_implementation_version")
        != FORMAL_FOLD_SOURCE_VERSION
    ):
        raise AssertionError(f"formal fold source is not complete: {completion_path}")
    manifest_path = (
        args.formal_fold_run_dir / "audit" / f"session_{session}" / "split_manifest.csv"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"explicit formal fold source lacks session manifest: {manifest_path}"
        )
    data_dir = args.data_dir or default_block_data_dir(args.project_root)
    data = load_block_sequence_session(
        args.project_root, session, TASK_NAME, data_dir=data_dir
    )
    if tuple(data.X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(
            f"session {session}: expected {EXPECTED_BLOCK_SHAPE}, got {data.X.shape}"
        )
    splits = grouped_cv_splits(data.groups, max_folds=MAX_FOLDS)
    current = split_manifest(
        session,
        TASK_NAME,
        data.y,
        data.groups,
        splits=splits,
        max_folds=MAX_FOLDS,
    )
    canonical_current = fold_audit.canonical_manifest(current)
    canonical_formal = fold_audit.canonical_manifest(pd.read_csv(manifest_path))
    if not canonical_current.equals(canonical_formal):
        raise AssertionError(
            f"session {session}: fold content differs from {manifest_path}"
        )
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        overlap = np.intersect1d(data.groups[train_idx], data.groups[test_idx])
        if overlap.size:
            raise AssertionError(
                f"session {session} fold {fold}: cycle leakage {overlap.tolist()}"
            )
    audit_dir = args.output_dir / "audit" / f"session_{session}"
    framework.atomic_csv(audit_dir / "split_manifest.csv", canonical_current)
    framework.atomic_json(
        audit_dir / "dataset.json",
        {
            "session": session,
            "source_h5": str(data.source_h5_path),
            "source_metadata": str(data.source_metadata_path),
            "shape": list(data.X.shape),
            "n_cycles": data.n_cycles,
            "n_samples": data.n_blocks,
            "formal_clean4_fold_match": True,
            "formal_manifest_source": str(manifest_path),
            "formal_run_complete": str(completion_path),
            "formal_run_fingerprint": str(completion.get("run_fingerprint")),
            "manifest_comparison": "canonical exact table equality",
            "train_test_cycle_overlap_all_folds": False,
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
        raise AssertionError(f"task plan missing columns {sorted(required-set(plan.columns))}")
    if plan["task_key"].duplicated().any():
        raise AssertionError("task plan contains duplicate task keys")
    if set(plan["session"].astype(str)) != set(EXPECTED_SESSIONS):
        raise AssertionError("task plan session coverage differs from frozen 9 sessions")
    if set(plan["variant"].astype(str)) != set(INPUT_VARIANTS):
        raise AssertionError("task plan variant coverage differs from raw + cycle_consistent")
    if set(pd.to_numeric(plan["seed"]).astype(int)) != set(SEEDS):
        raise AssertionError("task plan seed coverage differs from 0,1,2")
    pair_counts = plan.groupby(["session", "seed", "fold"])["variant"].nunique()
    if not pair_counts.eq(len(INPUT_VARIANTS)).all():
        raise AssertionError("raw and cycle_consistent tasks are not perfectly paired")
    fold_rows = plan[["session", "fold", "train_cycles", "test_cycles"]].drop_duplicates()
    if fold_rows.duplicated(["session", "fold"]).any():
        raise AssertionError("paired variants do not share identical cycle folds")
    counts = {
        "number_of_sessions": len(EXPECTED_SESSIONS),
        "number_of_variants": len(INPUT_VARIANTS),
        "number_of_seeds": len(SEEDS),
        "number_of_folds": len(fold_rows),
        "expected_total_tasks": len(plan),
    }
    if counts != {
        "number_of_sessions": 9,
        "number_of_variants": 2,
        "number_of_seeds": 3,
        "number_of_folds": 82,
        "expected_total_tasks": 492,
    }:
        raise AssertionError(f"unexpected formal task counts: {counts}")
    return counts


def print_task_counts(counts: dict[str, int]) -> None:
    for key in (
        "number_of_sessions",
        "number_of_variants",
        "number_of_seeds",
        "number_of_folds",
        "expected_total_tasks",
    ):
        print(f"{key.replace('_', ' ')}: {counts[key]}", flush=True)


def build_task_plan(args: argparse.Namespace, identity: dict[str, Any]) -> pd.DataFrame:
    run_fp = framework.fingerprint(identity)
    config_fp = framework.fingerprint(identity["experiment_config"])
    runtime_fp = framework.fingerprint(identity["runtime_environment_signature"])
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        data, splits = audit_session(args, session)
        audits.append(
            {
                "session": session,
                "n_cycles": data.n_cycles,
                "n_samples": data.n_blocks,
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
            "task_definition": "session x input_variant x seed x fold",
            "historical_raw_reused": False,
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
            raise AssertionError(f"task plan metadata mismatch for {key}")
    print_task_counts(counts)
    return plan


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

    required = (
        "COMPLETE.json",
        "result.json",
        "predictions.csv",
        "confusion_matrix.csv",
        "training_history.csv",
        "normalization_audit.json",
        "pair_coverage_audit.json",
        "representation_audit.json",
        "model_config.json",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return fail(f"missing files {missing}")
    try:
        complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        confusion = pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str})
        history = pd.read_csv(path / "training_history.csv", dtype={"session": str})
        normalization = json.loads(
            (path / "normalization_audit.json").read_text(encoding="utf-8")
        )
        pair_audit = json.loads(
            (path / "pair_coverage_audit.json").read_text(encoding="utf-8")
        )
        representation_audit = json.loads(
            (path / "representation_audit.json").read_text(encoding="utf-8")
        )
        model_config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")

    session = str(expected["session"])
    variant = str(expected["variant"])
    seed = int(expected["seed"])
    fold = int(expected["fold"])
    expected_task_fp = task_fingerprint(run_fp, expected)
    if complete.get("task_key") != task_key(session, variant, seed, fold):
        return fail("task key mismatch")
    for name, payload in (("complete", complete), ("result", result)):
        if payload.get("run_fingerprint") != run_fp:
            return fail(f"{name} run fingerprint mismatch")
        if payload.get("task_fingerprint") != expected_task_fp:
            return fail(f"{name} task fingerprint mismatch")
        if payload.get("config_fingerprint") != str(expected["config_fingerprint"]):
            return fail(f"{name} config fingerprint mismatch")
        if payload.get("runtime_environment_fingerprint") != str(
            expected["runtime_environment_fingerprint"]
        ):
            return fail(f"{name} runtime fingerprint mismatch")
    identity = (session, variant, seed, fold)
    observed = (
        str(result.get("session")),
        str(result.get("variant")),
        int(result.get("seed", -1)),
        int(result.get("fold", -1)),
    )
    if observed != identity:
        return fail("result identity mismatch")
    if result.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        return fail("model implementation version mismatch")
    if int(result.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNT:
        return fail("parameter count mismatch")
    if int(result.get("actual_batch_size", -1)) != FORMAL_BATCH_SIZE:
        return fail("batch size mismatch")
    if (
        str(result.get("train_cycles")) != str(expected["train_cycles"])
        or str(result.get("test_cycles")) != str(expected["test_cycles"])
    ):
        return fail("result cycle membership mismatch")
    expected_model_config = formal_architecture_config()
    expected_model_config["parameter_count"] = EXPECTED_PARAMETER_COUNT
    if framework.fingerprint(model_config) != framework.fingerprint(expected_model_config):
        return fail("architecture config mismatch")
    if bool(normalization.get("target_used_for_stats", True)):
        return fail("target used for normalization statistics")
    if normalization.get("phase") != "outer_train_fold_only":
        return fail("normalization is not outer-train-fold-only")
    if normalization.get("input_variant") != variant:
        return fail("normalization variant mismatch")
    if normalization.get("preprocessing_order") != (
        "clean4 -> arcsinh -> train_fold_pixel_zscore"
    ):
        return fail("input preprocessing differs from frozen raw Temporal1D")
    if bool(normalization.get("test_block_identity_used", True)):
        return fail("test block identity used by training/inference")
    expected_train_cycles = [
        int(value) for value in str(expected["train_cycles"]).split(",")
    ]
    expected_test_cycles = [
        int(value) for value in str(expected["test_cycles"]).split(",")
    ]
    if (
        str(pair_audit.get("session")),
        str(pair_audit.get("variant")),
        int(pair_audit.get("seed", -1)),
        int(pair_audit.get("fold", -1)),
    ) != identity:
        return fail("pair-audit identity mismatch")
    if pair_audit.get("train_cycle_ids") != expected_train_cycles:
        return fail("pair-pool training-cycle provenance mismatch")
    if pair_audit.get("test_cycle_ids") != expected_test_cycles:
        return fail("pair-audit test-cycle provenance mismatch")
    if (
        bool(pair_audit.get("train_test_cycle_overlap", True))
        or bool(pair_audit.get("test_samples_in_pair_pool", True))
        or bool(pair_audit.get("test_block_identity_loaded_for_training", True))
        or bool(pair_audit.get("negatives_used", True))
        or not bool(pair_audit.get("same_block_required", False))
        or not bool(pair_audit.get("different_cycle_required", False))
        or not bool(pair_audit.get("upper_triangle_unique_pairs", False))
    ):
        return fail("pair-pool leakage or positive-pair rule violation")
    expected_lambda = LAMBDA_CONSISTENCY if variant == CYCLE_CONSISTENT_VARIANT else 0.0
    if not np.isclose(
        float(pair_audit.get("lambda_consistency", np.nan)), expected_lambda
    ):
        return fail("consistency lambda mismatch")
    if not (
        (
            str(representation_audit.get("session")),
            str(representation_audit.get("variant")),
            int(representation_audit.get("seed", -1)),
            int(representation_audit.get("fold", -1)),
        )
        == identity
        and
        bool(representation_audit.get("diagnostic_only", False))
        and not bool(representation_audit.get("used_for_model_selection", True))
        and not bool(representation_audit.get("test_embeddings_computed", True))
        and int(representation_audit.get("embedding_dimension", -1)) == 64
    ):
        return fail("training-representation diagnostic scope mismatch")
    if (
        str(normalization.get("session")),
        str(normalization.get("method")),
        int(normalization.get("seed", -1)),
        int(normalization.get("fold", -1)),
    ) != identity:
        return fail("normalization identity mismatch")
    if (
        str(normalization.get("train_cycles")) != str(expected["train_cycles"])
        or str(normalization.get("test_cycles")) != str(expected["test_cycles"])
    ):
        return fail("normalization cycle membership mismatch")

    expected_n = int(expected["n_test_samples"])
    if len(predictions) != expected_n or int(result.get("n_test_samples", -1)) != expected_n:
        return fail("prediction count mismatch")
    if not (
        predictions["session"].eq(session).all()
        and predictions["variant"].eq(variant).all()
        and pd.to_numeric(predictions["seed"]).eq(seed).all()
        and pd.to_numeric(predictions["fold"]).eq(fold).all()
    ):
        return fail("prediction identity mismatch")
    probabilities = predictions[["probability_0", "probability_1"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        return fail("invalid probabilities")
    metrics = classification_metrics(
        predictions["y_true"].to_numpy(int), predictions["y_pred"].to_numpy(int)
    )
    for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
        if not np.isclose(float(result.get(metric, np.nan)), metrics[metric], atol=1e-12):
            return fail(f"stored {metric} differs from predictions")
    if len(confusion) != 4 or int(confusion["count"].sum()) != expected_n:
        return fail("confusion matrix invalid")
    if not (
        confusion["session"].eq(session).all()
        and confusion["variant"].eq(variant).all()
        and pd.to_numeric(confusion["seed"]).eq(seed).all()
        and pd.to_numeric(confusion["fold"]).eq(fold).all()
    ):
        return fail("confusion matrix identity mismatch")
    stored_cm = np.zeros((2, 2), dtype=int)
    for row in confusion.itertuples(index=False):
        stored_cm[int(row.true_label), int(row.predicted_label)] = int(row.count)
    expected_cm = confusion_matrix(
        predictions["y_true"].to_numpy(int),
        predictions["y_pred"].to_numpy(int),
        np.asarray([0, 1]),
    )
    if not np.array_equal(stored_cm, expected_cm):
        return fail("confusion matrix differs from predictions")
    history_required = {
        "session",
        "variant",
        "seed",
        "fold",
        "epoch",
        "train_loss",
        "classification_loss",
        "consistency_loss",
        "total_loss",
        "train_accuracy",
        "number_of_batches",
        "batches_with_valid_pairs",
        "valid_pair_fraction_of_batches",
        "total_valid_positive_pairs",
        "mean_valid_pairs_per_batch",
        "lambda_consistency",
    }
    if not history_required.issubset(history.columns):
        return fail("training history columns missing")
    if len(history) != FORMAL_EPOCHS or not np.array_equal(
        history["epoch"].to_numpy(int), np.arange(1, FORMAL_EPOCHS + 1)
    ):
        return fail("training history is not exactly 40 epochs")
    finite_columns = [
        "train_loss",
        "classification_loss",
        "consistency_loss",
        "total_loss",
        "train_accuracy",
        "valid_pair_fraction_of_batches",
        "mean_valid_pairs_per_batch",
    ]
    if not np.isfinite(history[finite_columns].to_numpy(float)).all():
        return fail("training history contains non-finite values")
    if not np.allclose(history["train_loss"], history["total_loss"], atol=1e-12):
        return fail("train_loss and total_loss differ")
    if not np.allclose(history["lambda_consistency"], expected_lambda, atol=1e-12):
        return fail("history consistency lambda mismatch")
    if variant == RAW_VARIANT:
        if not np.allclose(history["consistency_loss"], 0.0, atol=1e-12):
            return fail("raw history has nonzero consistency loss")
        if not np.allclose(
            history["total_loss"], history["classification_loss"], atol=1e-12
        ):
            return fail("raw total loss differs from classification loss")
    if not (
        history["session"].eq(session).all()
        and history["variant"].eq(variant).all()
        and pd.to_numeric(history["seed"]).eq(seed).all()
        and pd.to_numeric(history["fold"]).eq(fold).all()
    ):
        return fail("training history identity mismatch")
    if int(result.get("trained_epochs", -1)) != FORMAL_EPOCHS:
        return fail("trained epoch count mismatch")
    if int(pair_audit.get("number_of_batches", -1)) != int(
        history["number_of_batches"].sum()
    ):
        return fail("pair-audit batch count differs from history")
    if int(pair_audit.get("total_valid_positive_pairs", -1)) != int(
        history["total_valid_positive_pairs"].sum()
    ):
        return fail("pair-audit positive-pair count differs from history")
    best_i = int(history["train_accuracy"].to_numpy(float).argmax())
    if int(result.get("best_epoch", -1)) != int(history.iloc[best_i]["epoch"]):
        return fail("best_epoch differs from descriptive maximum train accuracy epoch")
    if not np.isclose(
        float(result.get("best_train_accuracy", np.nan)),
        float(history.iloc[best_i]["train_accuracy"]),
        atol=1e-12,
    ):
        return fail("stored best_train_accuracy differs from history")
    if not np.isclose(
        float(result.get("final_train_accuracy", np.nan)),
        float(history.iloc[-1]["train_accuracy"]),
        atol=1e-12,
    ):
        return fail("stored final_train_accuracy differs from history")
    if not np.isclose(
        float(result.get("train_accuracy", np.nan)),
        float(history.iloc[-1]["train_accuracy"]),
        atol=1e-12,
    ):
        return fail("stored train_accuracy is not fixed-epoch final accuracy")
    final_history = history.iloc[-1]
    for result_key, history_key in (
        ("classification_loss_final", "classification_loss"),
        ("consistency_loss_final", "consistency_loss"),
        ("total_loss_final", "total_loss"),
    ):
        if not np.isclose(
            float(result.get(result_key, np.nan)),
            float(final_history[history_key]),
            atol=1e-12,
        ):
            return fail(f"stored {result_key} differs from final history")
    for key in (
        "total_valid_positive_pairs",
        "mean_valid_pairs_per_batch",
        "valid_pair_fraction_of_batches",
    ):
        if not np.isclose(
            float(result.get(key, np.nan)), float(pair_audit.get(key, np.nan))
        ):
            return fail(f"stored {key} differs from pair audit")
    return True, "validated"


def update_status(args: argparse.Namespace, plan: pd.DataFrame, run_fp: str) -> pd.DataFrame:
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


def run_sanity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if args.workers != 0:
        raise ValueError("sanity requires --workers 0")
    if args.sanity_epochs not in (1, 2):
        raise ValueError("sanity is restricted to 1 or 2 epochs")
    data, splits = audit_session(args, "710")
    train_idx, test_idx = splits[0]
    selected_train_cycles = np.unique(data.groups[train_idx])[:2]
    selected_test_cycles = np.unique(data.groups[test_idx])[:1]
    tiny_train = train_idx[np.isin(data.groups[train_idx], selected_train_cycles)]
    tiny_test = test_idx[np.isin(data.groups[test_idx], selected_test_cycles)]
    device = resolve_device(args.device)
    audits = []
    for variant in INPUT_VARIANTS:
        model = CNN2DTemporal1D(n_classes=2, temporal_length=4).to(device)
        model.eval()
        inputs = blocks_to_sequence_tensor(data.X[tiny_train[:2]]).to(device)
        targets = torch.from_numpy(data.y[tiny_train[:2]].astype(np.int64)).to(device)
        logits = model(inputs)
        logits_with_embedding, embedding = model.forward_with_embedding(inputs)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        if (
            tuple(logits.shape) != (2, 2)
            or tuple(embedding.shape) != (2, 64)
            or not torch.equal(logits, logits_with_embedding)
            or not bool(torch.isfinite(loss).item())
        ):
            raise AssertionError("invalid Temporal1D forward/backward")
        trained = train_fold(
            data.X[tiny_train],
            data.y[tiny_train],
            data.X[tiny_test],
            np.asarray([0, 1]),
            block_names_train=data.metadata.iloc[tiny_train]["block_name"]
            .astype(str)
            .to_numpy(),
            cycle_ids_train=data.groups[tiny_train],
            cycle_ids_test=data.groups[tiny_test],
            input_variant=variant,
            session="710",
            fold=1,
            seed=0,
            train_cycles=cycle_text(data.groups[tiny_train]),
            test_cycles=cycle_text(data.groups[tiny_test]),
            training_config=frozen_training_config(
                min(FORMAL_BATCH_SIZE, len(tiny_train)), args.sanity_epochs
            ),
            device=str(device),
            workers=0,
        )
        if len(trained.history) != args.sanity_epochs:
            raise AssertionError("sanity epoch count mismatch")
        if trained.pair_audit["total_valid_positive_pairs"] <= 0:
            raise AssertionError("sanity batch did not exercise consistency pairs")
        audits.append(
            {
                "variant": variant,
                "input_shape": list(data.X[tiny_train].shape),
                "output_shape": list(trained.X_test_normalized.shape),
                "parameter_count": trained.model_parameters,
                "normalization_target_used_for_stats": False,
                "lambda_consistency": trained.pair_audit["lambda_consistency"],
                "pair_coverage": trained.pair_audit,
                "training_representation": trained.representation_audit,
                "debug_only_not_formal": True,
            }
        )
        del model, inputs, targets, logits, loss, trained
    framework.atomic_json(
        args.output_dir / "sanity" / "sanity_audit.json",
        {
            "session": "710",
            "fold": 1,
            "seed": 0,
            "variants": audits,
            "formal_clean4_fold_match": True,
            "cycle_overlap": False,
            "test_metadata_used_for_inference": False,
            "formal_results": False,
        },
    )
    framework.atomic_json(
        args.output_dir / "sanity" / "SANITY_COMPLETE.json",
        {"run_fingerprint": framework.fingerprint(identity), "checks_passed": True},
    )
    print("SANITY PASS (debug only; no formal result)", flush=True)


def write_fold_task(
    args: argparse.Namespace,
    identity: dict[str, Any],
    expected: dict[str, Any],
    data: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    run_fp = framework.fingerprint(identity)
    session = str(expected["session"])
    variant = str(expected["variant"])
    seed = int(expected["seed"])
    fold = int(expected["fold"])
    path = task_dir(args.output_dir, session, variant, seed, fold)
    trained = train_fold(
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        np.asarray([0, 1]),
        block_names_train=data.metadata.iloc[train_idx]["block_name"]
        .astype(str)
        .to_numpy(),
        cycle_ids_train=data.groups[train_idx],
        cycle_ids_test=data.groups[test_idx],
        input_variant=variant,
        session=session,
        fold=fold,
        seed=seed,
        train_cycles=cycle_text(data.groups[train_idx]),
        test_cycles=cycle_text(data.groups[test_idx]),
        training_config=DeepTrainingConfig(**identity["experiment_config"]["training"]),
        device=args.device,
        workers=args.workers,
    )
    metrics = classification_metrics(data.y[test_idx], trained.predictions)
    # Inference has already completed through the X-only model path above.
    # Held-out metadata below is attached to result rows for audit only.
    prediction_rows = []
    for local_i, sample_i in enumerate(test_idx):
        metadata = data.metadata.iloc[int(sample_i)]
        prediction_rows.append(
            {
                "session": session,
                "seed": seed,
                "fold": fold,
                "variant": variant,
                "sample_index": int(sample_i),
                "block_id": str(metadata["block_id"]),
                "cycle": int(data.groups[sample_i]),
                "block_name": str(metadata["block_name"]),
                "y_true": int(data.y[sample_i]),
                "y_pred": int(trained.predictions[local_i]),
                "probability_0": float(trained.probabilities[local_i, 0]),
                "probability_1": float(trained.probabilities[local_i, 1]),
            }
        )
    cm = confusion_matrix(data.y[test_idx], trained.predictions, np.asarray([0, 1]))
    confusion_rows = [
        {
            "session": session,
            "seed": seed,
            "fold": fold,
            "variant": variant,
            "true_label": truth,
            "predicted_label": prediction,
            "count": int(cm[truth, prediction]),
        }
        for truth in (0, 1)
        for prediction in (0, 1)
    ]
    history = pd.DataFrame(trained.history)
    for column, value in reversed(
        (("session", session), ("seed", seed), ("fold", fold), ("variant", variant))
    ):
        history.insert(0, column, value)
    best_i = int(history["train_accuracy"].to_numpy(float).argmax())
    best_epoch = int(history.iloc[best_i]["epoch"])
    best_train_accuracy = float(history.iloc[best_i]["train_accuracy"])
    final_train_accuracy = float(history.iloc[-1]["train_accuracy"])
    task_fp = task_fingerprint(run_fp, expected)
    shared = {
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": str(expected["config_fingerprint"]),
        "runtime_environment_fingerprint": str(
            expected["runtime_environment_fingerprint"]
        ),
    }
    result = {
        **shared,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "session": session,
        "seed": seed,
        "fold": fold,
        "variant": variant,
        "model": "cnn2d_temporal1d",
        "n_cycles": data.n_cycles,
        "n_samples": data.n_blocks,
        "n_train_samples": len(train_idx),
        "n_test_samples": len(test_idx),
        "train_cycles": cycle_text(data.groups[train_idx]),
        "test_cycles": cycle_text(data.groups[test_idx]),
        "train_accuracy": final_train_accuracy,
        "best_train_accuracy": best_train_accuracy,
        "final_train_accuracy": final_train_accuracy,
        "final_epoch": FORMAL_EPOCHS,
        "test_balanced_accuracy": float(metrics["balanced_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "best_epoch": best_epoch,
        "best_epoch_definition": "descriptive max train accuracy; not model selection",
        "selected_epoch": FORMAL_EPOCHS,
        "early_stopping_used": False,
        "parameter_count": int(trained.model_parameters),
        "actual_batch_size": FORMAL_BATCH_SIZE,
        "final_training_loss": float(trained.final_training_loss),
        "classification_loss_final": float(history.iloc[-1]["classification_loss"]),
        "consistency_loss_final": float(history.iloc[-1]["consistency_loss"]),
        "total_loss_final": float(history.iloc[-1]["total_loss"]),
        "lambda_consistency": float(trained.pair_audit["lambda_consistency"]),
        "total_valid_positive_pairs": int(
            trained.pair_audit["total_valid_positive_pairs"]
        ),
        "number_of_batches": int(trained.pair_audit["number_of_batches"]),
        "batches_with_valid_pairs": int(
            trained.pair_audit["batches_with_valid_pairs"]
        ),
        "mean_valid_pairs_per_batch": float(
            trained.pair_audit["mean_valid_pairs_per_batch"]
        ),
        "valid_pair_fraction_of_batches": float(
            trained.pair_audit["valid_pair_fraction_of_batches"]
        ),
        "training_same_block_cross_cycle_cosine": trained.representation_audit[
            "same_block_cross_cycle_mean_cosine_similarity"
        ],
        "trained_epochs": int(trained.final_trained_epochs),
        "device": trained.device,
    }
    framework.atomic_json(path / "result.json", result)
    framework.atomic_csv(path / "predictions.csv", pd.DataFrame(prediction_rows))
    framework.atomic_csv(path / "confusion_matrix.csv", pd.DataFrame(confusion_rows))
    framework.atomic_csv(path / "training_history.csv", history)
    framework.atomic_json(path / "normalization_audit.json", trained.normalization_audit)
    framework.atomic_json(path / "pair_coverage_audit.json", trained.pair_audit)
    framework.atomic_json(
        path / "representation_audit.json",
        {
            "session": session,
            "fold": fold,
            "seed": seed,
            "variant": variant,
            **trained.representation_audit,
        },
    )
    framework.atomic_json(path / "model_config.json", trained.model_config)
    framework.atomic_json(
        path / "COMPLETE.json",
        {
            **shared,
            "task_key": task_key(session, variant, seed, fold),
            "completed_utc": utc_now(),
        },
    )
    validate_completed_task(path, expected, run_fp, raise_on_error=True)


def read_all_tasks(args: argparse.Namespace, plan: pd.DataFrame, run_fp: str):
    results, predictions, confusions, histories = [], [], [], []
    pair_audits, representation_audits = [], []
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
        pair_audit = json.loads(
            (path / "pair_coverage_audit.json").read_text(encoding="utf-8")
        )
        pair_audits.append(
            {
                **pair_audit,
                "train_cycle_ids": framework.canonical_json(
                    pair_audit["train_cycle_ids"]
                ),
                "test_cycle_ids": framework.canonical_json(
                    pair_audit["test_cycle_ids"]
                ),
            }
        )
        representation_audits.append(
            json.loads((path / "representation_audit.json").read_text(encoding="utf-8"))
        )
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True),
        pd.concat(confusions, ignore_index=True),
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(pair_audits),
        pd.DataFrame(representation_audits),
    )


def exact_two_sided_sign_flip(values: np.ndarray) -> float:
    return framework.exact_two_sided_sign_flip(np.asarray(values, dtype=float))


def build_seed_summary(
    per_fold: pd.DataFrame, predictions: pd.DataFrame, history: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for (session, variant, seed), group in predictions.groupby(
        ["session", "variant", "seed"], sort=True
    ):
        fold_source = per_fold[
            per_fold["session"].eq(str(session))
            & per_fold["variant"].eq(str(variant))
            & per_fold["seed"].eq(int(seed))
        ]
        expected_n = int(fold_source.iloc[0]["n_samples"])
        if group["sample_index"].duplicated().any() or set(group["sample_index"]) != set(
            range(expected_n)
        ):
            raise AssertionError("OOF coverage is incomplete or duplicated")
        metrics = classification_metrics(
            group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int)
        )
        history_source = history[
            history["session"].eq(str(session))
            & history["variant"].eq(str(variant))
            & history["seed"].eq(int(seed))
        ]
        final_history = history_source[
            pd.to_numeric(history_source["epoch"]).eq(FORMAL_EPOCHS)
        ].copy()
        expected_folds = set(pd.to_numeric(fold_source["fold"]).astype(int))
        observed_folds = set(pd.to_numeric(final_history["fold"]).astype(int))
        if (
            len(final_history) != len(expected_folds)
            or final_history["fold"].duplicated().any()
            or observed_folds != expected_folds
        ):
            raise AssertionError("seed final-epoch train-accuracy coverage is incomplete")
        # Match the historical formal multiframe overfitting audit: use each
        # fold's fixed epoch-40 final train accuracy, then average across folds
        # for this seed. Descriptive best_epoch never enters the gap.
        train_accuracy = float(final_history["train_accuracy"].astype(float).mean())
        mean_oof_ba = float(metrics["balanced_accuracy"])
        number_of_batches = int(fold_source["number_of_batches"].sum())
        total_valid_pairs = int(fold_source["total_valid_positive_pairs"].sum())
        batches_with_pairs = int(fold_source["batches_with_valid_pairs"].sum())
        rows.append(
            {
                "session": str(session),
                "variant": str(variant),
                "seed": int(seed),
                "mean_oof_BA": mean_oof_ba,
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "train_accuracy": train_accuracy,
                "final_train_accuracy": train_accuracy,
                "train_test_gap": train_accuracy - mean_oof_ba,
                "mean_consistency_loss": float(
                    fold_source["consistency_loss_final"].astype(float).mean()
                ),
                "mean_valid_pairs_per_batch": float(
                    total_valid_pairs / max(number_of_batches, 1)
                ),
                "valid_pair_batch_fraction": float(
                    batches_with_pairs / max(number_of_batches, 1)
                ),
                "mean_training_same_block_cross_cycle_cosine": float(
                    fold_source["training_same_block_cross_cycle_cosine"]
                    .astype(float)
                    .mean()
                ),
                "n_cycles": int(fold_source.iloc[0]["n_cycles"]),
                "n_samples": expected_n,
                "n_folds": int(group["fold"].nunique()),
            }
        )
    result = pd.DataFrame(rows).sort_values(["session", "variant", "seed"])
    expected = len(EXPECTED_SESSIONS) * len(INPUT_VARIANTS) * len(SEEDS)
    if len(result) != expected:
        raise AssertionError("seed summary coverage is incomplete")
    return result


def build_session_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        seed_summary.groupby(["session", "variant"], as_index=False)
        .agg(
            BA=("mean_oof_BA", "mean"),
            train_acc=("final_train_accuracy", "mean"),
            gap=("train_test_gap", "mean"),
            consistency_loss=("mean_consistency_loss", "mean"),
            pairs_per_batch=("mean_valid_pairs_per_batch", "mean"),
            pair_batch_fraction=("valid_pair_batch_fraction", "mean"),
            training_cosine=("mean_training_same_block_cross_cycle_cosine", "mean"),
        )
        .sort_values(["session", "variant"])
    )
    ba = grouped.pivot(index="session", columns="variant", values="BA")
    train = grouped.pivot(index="session", columns="variant", values="train_acc")
    gap = grouped.pivot(index="session", columns="variant", values="gap")
    consistency = grouped.pivot(index="session", columns="variant", values="consistency_loss")
    pairs = grouped.pivot(index="session", columns="variant", values="pairs_per_batch")
    pair_fraction = grouped.pivot(index="session", columns="variant", values="pair_batch_fraction")
    training_cosine = grouped.pivot(index="session", columns="variant", values="training_cosine")
    rows = []
    for session in EXPECTED_SESSIONS:
        rows.append(
            {
                "session": session,
                "raw_BA": float(ba.loc[session, RAW_VARIANT]),
                "cycle_consistent_BA": float(ba.loc[session, CYCLE_CONSISTENT_VARIANT]),
                "delta_BA": float(
                    ba.loc[session, CYCLE_CONSISTENT_VARIANT]
                    - ba.loc[session, RAW_VARIANT]
                ),
                "raw_train_acc": float(train.loc[session, RAW_VARIANT]),
                "consistent_train_acc": float(train.loc[session, CYCLE_CONSISTENT_VARIANT]),
                "raw_gap": float(gap.loc[session, RAW_VARIANT]),
                "consistent_gap": float(gap.loc[session, CYCLE_CONSISTENT_VARIANT]),
                "delta_gap": float(
                    gap.loc[session, CYCLE_CONSISTENT_VARIANT]
                    - gap.loc[session, RAW_VARIANT]
                ),
                "mean_consistency_loss": float(
                    consistency.loc[session, CYCLE_CONSISTENT_VARIANT]
                ),
                "mean_valid_pairs_per_batch": float(
                    pairs.loc[session, CYCLE_CONSISTENT_VARIANT]
                ),
                "valid_pair_batch_fraction": float(
                    pair_fraction.loc[session, CYCLE_CONSISTENT_VARIANT]
                ),
                "raw_training_same_block_cross_cycle_cosine": float(
                    training_cosine.loc[session, RAW_VARIANT]
                ),
                "consistent_training_same_block_cross_cycle_cosine": float(
                    training_cosine.loc[session, CYCLE_CONSISTENT_VARIANT]
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != len(EXPECTED_SESSIONS):
        raise AssertionError("session summary coverage is incomplete")
    return result


def build_overfitting_audit(seed_summary: pd.DataFrame) -> pd.DataFrame:
    return seed_summary[
        [
            "session",
            "variant",
            "seed",
            "train_accuracy",
            "final_train_accuracy",
            "mean_oof_BA",
            "train_test_gap",
        ]
    ].rename(columns={"mean_oof_BA": "OOF_test_BA"})


def build_overall_and_decision(
    session_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    deltas = session_summary.set_index("session")["delta_BA"]
    strong = session_summary[session_summary["session"].isin(STRONG_SESSIONS)]
    weak = session_summary[session_summary["session"].isin(WEAK_SESSIONS)]
    raw_mean = float(session_summary["raw_BA"].mean())
    consistent_mean = float(session_summary["cycle_consistent_BA"].mean())
    overall_delta = float(deltas.mean())
    strong_raw = float(strong["raw_BA"].mean())
    strong_consistent = float(strong["cycle_consistent_BA"].mean())
    weak_raw = float(weak["raw_BA"].mean())
    weak_consistent = float(weak["cycle_consistent_BA"].mean())
    improved = int((deltas > TIE_TOLERANCE).sum())
    tied = int((deltas.abs() <= TIE_TOLERANCE).sum())
    worsened = int((deltas < -TIE_TOLERANCE).sum())
    weak_deltas = weak.set_index("session")["delta_BA"]
    weak_improved = int((weak_deltas > TIE_TOLERANCE).sum())
    weak_tied = int((weak_deltas.abs() <= TIE_TOLERANCE).sum())
    weak_worsened = int((weak_deltas < -TIE_TOLERANCE).sum())
    max_session = str(deltas.idxmax())
    max_improvement = float(deltas.loc[max_session])
    leave_max_out_delta = float(deltas.drop(index=max_session).mean())
    overall = pd.DataFrame(
        [
            {
                "raw_mean_BA": raw_mean,
                "cycle_consistent_mean_BA": consistent_mean,
                "overall_delta_BA": overall_delta,
                "median_delta_BA": float(deltas.median()),
                "raw_strong_mean_BA": strong_raw,
                "consistent_strong_mean_BA": strong_consistent,
                "strong_delta_BA": strong_consistent - strong_raw,
                "raw_weak_mean_BA": weak_raw,
                "consistent_weak_mean_BA": weak_consistent,
                "weak_delta_BA": weak_consistent - weak_raw,
                "weak_improved": weak_improved,
                "weak_tied": weak_tied,
                "weak_worsened": weak_worsened,
                "all_improved": improved,
                "all_tied": tied,
                "all_worsened": worsened,
                "max_improvement_session": max_session,
                "max_single_session_improvement": max_improvement,
                "leave_max_improvement_out_overall_delta_BA": leave_max_out_delta,
            }
        ]
    )
    paired = pd.DataFrame(
        [
            {
                "comparison": "cycle_consistent_vs_raw",
                "n_sessions": len(deltas),
                "mean_delta_BA": overall_delta,
                "median_delta_BA": float(deltas.median()),
                "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(
                    deltas.to_numpy(float)
                ),
                "session_deltas_json": framework.canonical_json(deltas.to_dict()),
            }
        ]
    )
    checks = {
        "overall_mean_delta_BA_ge_0.010": overall_delta >= 0.010,
        "weak_6_mean_delta_BA_ge_0.020": (weak_consistent - weak_raw) >= 0.020,
        "at_least_4_of_6_weak_sessions_improved": weak_improved >= 4,
        "strong_3_mean_delta_BA_gt_minus_0.010": (
            strong_consistent - strong_raw
        ) > -0.010,
    }
    # The final criterion is deliberately a transparent diagnostic, not a new
    # inferential rule.  Continue is supported only if the four numeric checks
    # pass and the leave-one-largest-out trend remains positive.
    not_single_session_driven = leave_max_out_delta > 0.0
    decision = {
        "criteria_predefined": True,
        "checks": checks,
        "single_session_dominance_diagnostic": {
            "largest_improvement_session": max_session,
            "largest_single_session_improvement": max_improvement,
            "leave_largest_improvement_out_overall_delta_BA": leave_max_out_delta,
            "trend_remains_positive_without_largest_improvement": not_single_session_driven,
        },
        "decision": (
            "supports_continue_cycle_consistency_route"
            if all(checks.values()) and not_single_session_driven
            else "does_not_support_cycle_consistency_route"
        ),
        "not_a_statistical_significance_claim": True,
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
    lines = [
        "# Cycle-Consistent Temporal1D v1",
        "",
        "## Frozen protocol",
        "",
        "Raw and cycle-consistent are both rerun. They share the exact clean4 samples, "
        "cycle folds, seeds, initialization procedure, CNN2DTemporal1D architecture, "
        "classifier, AdamW configuration, batch size, and fixed 40 epochs.",
        "",
        "Preprocessing order:",
        "",
        "- raw: `clean4 -> arcsinh -> outer-train-fold pixel z-score -> Temporal1D`",
        "- cycle_consistent: `clean4 -> arcsinh -> outer-train-fold pixel z-score -> Temporal1D`",
        "",
        "The 64D `temporal_conv` output immediately before the unchanged binary classifier "
        "is L2-normalized. Unique mini-batch pairs are valid only when block identity is "
        "the same, cycle identity differs, and i<j. Their mean `1-cosine` loss is added "
        "with fixed lambda 0.1. No negatives, projection head, test block identity, test "
        "embedding, input modification, validation selection, or early stopping is used.",
        "",
        "Historical raw baseline reused: **no**.",
        "",
        "## Session results",
        "",
        "| session | raw train | raw BA | raw gap | consistent train | consistent BA | "
        "consistent gap | delta BA | delta gap | pairs/batch | pair-batch fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in session_summary.itertuples(index=False):
        lines.append(
            f"| {row.session} | {row.raw_train_acc:.4f} | {row.raw_BA:.4f} | "
            f"{row.raw_gap:.4f} | {row.consistent_train_acc:.4f} | "
            f"{row.cycle_consistent_BA:.4f} | {row.consistent_gap:.4f} | "
            f"{row.delta_BA:+.4f} | {row.delta_gap:+.4f} | "
            f"{row.mean_valid_pairs_per_batch:.2f} | {row.valid_pair_batch_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Raw mean BA: {summary.raw_mean_BA:.4f}",
            f"- Cycle-consistent mean BA: {summary.cycle_consistent_mean_BA:.4f}",
            f"- Overall delta BA: {summary.overall_delta_BA:+.4f}",
            f"- Median session delta BA: {summary.median_delta_BA:+.4f}",
            f"- Strong-3 raw/consistent/delta: {summary.raw_strong_mean_BA:.4f} / "
            f"{summary.consistent_strong_mean_BA:.4f} / {summary.strong_delta_BA:+.4f}",
            f"- Weak-6 raw/consistent/delta: {summary.raw_weak_mean_BA:.4f} / "
            f"{summary.consistent_weak_mean_BA:.4f} / {summary.weak_delta_BA:+.4f}",
            f"- Weak improved/tied/worsened: {int(summary.weak_improved)}/"
            f"{int(summary.weak_tied)}/{int(summary.weak_worsened)}",
            f"- All improved/tied/worsened: {int(summary.all_improved)}/"
            f"{int(summary.all_tied)}/{int(summary.all_worsened)}",
            f"- Exact paired two-sided sign-flip p: {test.exact_two_sided_sign_flip_p:.6f}",
            f"- Mean consistency loss: {session_summary.mean_consistency_loss.mean():.6f}",
            f"- Mean valid pairs/batch: {session_summary.mean_valid_pairs_per_batch.mean():.3f}",
            f"- Mean valid-pair batch fraction: "
            f"{session_summary.valid_pair_batch_fraction.mean():.3f}",
            f"- Training same-block cross-cycle cosine, raw/consistent: "
            f"{session_summary.raw_training_same_block_cross_cycle_cosine.mean():.4f} / "
            f"{session_summary.consistent_training_same_block_cross_cycle_cosine.mean():.4f}",
            "",
            "## Single-session dominance diagnostic",
            "",
            f"- Largest improvement: session {summary.max_improvement_session}, "
            f"delta={summary.max_single_session_improvement:+.4f}",
            f"- Overall delta after removing it: "
            f"{summary.leave_max_improvement_out_overall_delta_BA:+.4f}",
            "",
            "## Predefined decision",
            "",
            f"`{decision['decision']}`",
            "",
            "This is a rule-based continuation decision, not a statistical-significance claim. "
            "No downstream experiment is started automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_outputs(
    args: argparse.Namespace, plan: pd.DataFrame, identity: dict[str, Any]
) -> None:
    run_fp = framework.fingerprint(identity)
    (
        per_fold,
        predictions,
        confusions,
        history,
        pair_audits,
        representation_audits,
    ) = read_all_tasks(args, plan, run_fp)
    per_fold = per_fold.sort_values(["session", "seed", "fold", "variant"])
    predictions = predictions.sort_values(
        ["session", "seed", "fold", "variant", "sample_index"]
    )
    confusions = confusions.sort_values(
        ["session", "seed", "fold", "variant", "true_label", "predicted_label"]
    )
    history = history.sort_values(["session", "seed", "fold", "variant", "epoch"])
    if len(pair_audits) != len(plan) or len(representation_audits) != len(plan):
        raise AssertionError("pair/representation task-audit coverage is incomplete")
    pair_audits = pair_audits.sort_values(["session", "seed", "fold", "variant"])
    representation_audits = representation_audits.sort_values(
        ["session", "seed", "fold", "variant"]
    )
    paired_coverage = pair_audits.groupby(["session", "seed", "fold"], sort=True)
    for column in (
        "number_of_batches",
        "batches_with_valid_pairs",
        "total_valid_positive_pairs",
    ):
        if not paired_coverage[column].nunique().eq(1).all():
            raise AssertionError(
                f"raw and cycle_consistent batch pairing differ for {column}"
            )
    seed_summary = build_seed_summary(per_fold, predictions, history)
    session_summary = build_session_summary(seed_summary)
    overfitting = build_overfitting_audit(seed_summary)
    overall, paired, decision = build_overall_and_decision(session_summary)
    framework.atomic_csv(args.output_dir / "task_level_results.csv", per_fold)
    framework.atomic_csv(args.output_dir / "predictions.csv", predictions)
    framework.atomic_csv(args.output_dir / "confusion_matrices.csv", confusions)
    framework.atomic_csv(args.output_dir / "training_history.csv", history)
    framework.atomic_csv(args.output_dir / "pair_coverage_audit.csv", pair_audits)
    framework.atomic_csv(
        args.output_dir / "representation_audit.csv", representation_audits
    )
    framework.atomic_csv(args.output_dir / "seed_summary.csv", seed_summary)
    framework.atomic_csv(args.output_dir / "session_summary.csv", session_summary)
    framework.atomic_csv(args.output_dir / "overfitting_audit.csv", overfitting)
    framework.atomic_csv(args.output_dir / "overall_summary.csv", overall)
    framework.atomic_csv(args.output_dir / "paired_sign_flip.csv", paired)
    framework.atomic_json(args.output_dir / "decision_rule_audit.json", decision)
    framework.atomic_text(
        args.output_dir / "cycle_consistent_temporal1d_report.md",
        build_report(session_summary, overall, paired, decision),
    )


def run_cuda_preflight(args: argparse.Namespace) -> None:
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise AssertionError("formal protocol requires batch size 16")
    device = torch.device(args.device if args.device != "auto" else "cuda")
    audits = []
    for variant in INPUT_VARIANTS:
        model = CNN2DTemporal1D(n_classes=2, temporal_length=4).to(device)
        inputs = torch.zeros((FORMAL_BATCH_SIZE, 4, 1, 128, 501), device=device)
        targets = torch.arange(FORMAL_BATCH_SIZE, device=device) % 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        if variant == CYCLE_CONSISTENT_VARIANT:
            logits, embedding = model.forward_with_embedding(inputs)
            block_ids = torch.arange(FORMAL_BATCH_SIZE, device=device) % 4
            cycle_ids = torch.arange(FORMAL_BATCH_SIZE, device=device) // 4
            consistency, pair_count, _mask = cycle_consistency_loss(
                embedding,
                block_ids,
                cycle_ids,
                allowed_train_cycle_ids=(0, 1, 2, 3),
            )
            loss = total_training_loss(
                nn.CrossEntropyLoss()(logits, targets),
                consistency,
                lambda_consistency=LAMBDA_CONSISTENCY,
            )
        else:
            logits = model(inputs)
            pair_count = 0
            loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        optimizer.step()
        parameter_count = count_trainable_parameters(model)
        if (
            tuple(logits.shape) != (FORMAL_BATCH_SIZE, 2)
            or not bool(torch.isfinite(loss).item())
            or not gradients_finite
            or parameter_count != EXPECTED_PARAMETER_COUNT
        ):
            raise AssertionError(f"{variant}: invalid CUDA batch-16 preflight")
        audits.append(
            {
                "variant": variant,
                "loss_finite": True,
                "gradients_finite": True,
                "optimizer_step_success": True,
                "parameter_count": parameter_count,
                "valid_positive_pairs": pair_count,
            }
        )
        del model, inputs, targets, optimizer, logits, loss
        torch.cuda.empty_cache()
    framework.atomic_json(
        args.output_dir / "audit" / "cuda_batch16_preflight.json",
        {"formal_training_started": False, "device": str(device), "variants": audits},
    )


def run_full(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if not args.review_approved:
        raise RuntimeError(
            "formal run is locked until code review is approved; pass "
            "--review-approved only after greenlight"
        )
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
            train_idx, test_idx = splits[int(expected["fold"]) - 1]
            path = task_dir(
                args.output_dir,
                session,
                variant,
                int(expected["seed"]),
                int(expected["fold"]),
            )
            valid, _ = validate_completed_task(path, expected, run_fp)
            if valid:
                print(
                    f"SKIP [{completed}/{total}] session={session} variant={variant} "
                    f"fold={expected['fold']} seed={expected['seed']}",
                    flush=True,
                )
                continue
            print(
                f"RUN  [{completed}/{total}] session={session} variant={variant} "
                f"fold={expected['fold']} seed={expected['seed']}",
                flush=True,
            )
            write_fold_task(
                args,
                identity,
                expected,
                data,
                train_idx,
                test_idx,
            )
            completed += 1
            print(
                f"DONE [{completed}/{total}] session={session} variant={variant} "
                f"fold={expected['fold']} seed={expected['seed']}",
                flush=True,
            )
        del data
    status = update_status(args, plan, run_fp)
    if not status["status"].eq("complete").all():
        print("PARTIAL RUN SAVED; rerun identical GNU screen command to resume", flush=True)
        return
    aggregate_outputs(args, plan, identity)
    missing = [name for name in REQUIRED_FINAL_OUTPUTS if not (args.output_dir / name).is_file()]
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
            "completed_tasks": len(plan),
            "number_of_sessions": len(EXPECTED_SESSIONS),
            "number_of_variants": len(INPUT_VARIANTS),
            "number_of_seeds": len(SEEDS),
            "number_of_folds": int(
                len(plan[["session", "fold"]].drop_duplicates())
            ),
            "historical_raw_reused": False,
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
    args.benchmark_root = args.benchmark_root.resolve()
    args.formal_fold_run_dir = args.formal_fold_run_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    if args.batch_size != FORMAL_BATCH_SIZE:
        raise ValueError("formal and sanity protocol require --batch-size 16")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    identity = run_identity(
        args.project_root, args.batch_size, args.formal_fold_run_dir
    )
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
