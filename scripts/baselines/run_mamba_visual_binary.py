#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import itertools
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
from ultrasound_decoding.multiframe.spatial_mamba import (
    MODEL_DISPLAY_NAME,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    TRANSFORMER_REFERENCE_PARAMETER_COUNT,
    SpatialMambaClassifier,
    SpatialMambaConfig,
    architecture_config,
    mamba_dependency_available,
    parameter_breakdown,
    require_mamba_dependency,
    train_spatial_mamba_fold,
)
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_sequence_tensor,
    normalize_blocks_train_fold_only_with_stats,
)


OUTPUT_VERSION = "mamba_visual_binary_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
MAX_FOLDS = 10
FORMAL_EPOCHS = 40
ALLOWED_BATCH_SIZES = (16,)
REQUIRED_FINAL_OUTPUTS = (
    "mamba_summary.csv",
    "mamba_per_seed.csv",
    "mamba_per_fold.csv",
    "mamba_predictions.csv",
    "mamba_confusion_matrices.csv",
    "mamba_training_history.csv",
    "mamba_vs_existing_baselines.csv",
    "paired_comparisons.csv",
    "overfitting_summary.csv",
    "mamba_report.md",
)
RUNTIME_DISTRIBUTIONS = {
    "mamba_ssm_version": "mamba-ssm",
    "causal_conv1d_version": "causal-conv1d",
    "transformers_version": "transformers",
    "numpy_version": "numpy",
    "scipy_version": "scipy",
    "pandas_version": "pandas",
    "scikit_learn_version": "scikit-learn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen clean4 controlled Spatial-Mamba within-session binary baseline."
    )
    parser.add_argument("--stage", choices=("sanity", "full", "status"), required=True)
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
        default=PROJECT_DIR / "results" / "runs" / "multiframe" / "block_clean4_binary_v1",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sanity-epochs", type=int, default=2)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git_text(project_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=project_root, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        ).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def distribution_version(distribution: str) -> str:
    try:
        return str(importlib_metadata.version(distribution))
    except importlib_metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def runtime_environment_signature() -> dict[str, str]:
    """Stable task identity fields; excludes host, time, GPU load, and paths."""
    signature = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or "NONE"),
    }
    signature.update(
        {
            field: distribution_version(distribution)
            for field, distribution in RUNTIME_DISTRIBUTIONS.items()
        }
    )
    return signature


def environment_payload(device: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    resolved = device
    if device == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    runtime_signature = runtime_environment_signature()
    return {
        "compute_environment": "server" if resolved.startswith("cuda") else "local_sanity_or_cpu",
        "created_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ] if cuda_available else [],
        "requested_device": device,
        "resolved_device": resolved,
        "mamba_dependency_available": mamba_dependency_available(),
        "mamba_dependency_required_for_training": True,
        **runtime_signature,
        "runtime_environment_signature": runtime_signature,
    }


def frozen_experiment_config(batch_size: int) -> dict[str, Any]:
    architecture = SpatialMambaConfig()
    training = DeepTrainingConfig(
        optimizer="adamw", lr=1e-3, weight_decay=1e-3,
        batch_size=int(batch_size), max_epochs=FORMAL_EPOCHS,
        dropout=0.25, loss="cross_entropy",
    )
    return {
        "output_version": OUTPUT_VERSION,
        "baseline_claim": "controlled Spatial-Mamba baseline/backbone candidate; not a proposed model",
        "sessions": list(EXPECTED_SESSIONS),
        "task": TASK_NAME,
        "class_mapping": TASK_CLASS_NAMES[TASK_NAME],
        "stimulus_blocks": ["grating", "dot"],
        "non_stimulus_blocks": ["stop_after_grating", "static"],
        "input_unit": "one clean4 block",
        "input_shape": list(EXPECTED_BLOCK_SHAPE),
        "complete_cycles_only": True,
        "cv": "formal clean4 cycle-grouped folds, max_folds=10",
        "normalization": "arcsinh_then_train_pixel_zscore; outer train fold only",
        "oof_primary_metric": "balanced_accuracy",
        "seeds": list(SEEDS),
        "models": [MODEL_NAME],
        "training": training.__dict__,
        "architecture": architecture_config(architecture),
        "epoch_selection": "fixed 40 epochs; no validation/test early stopping or model selection",
        "validation_protocol": "none, matching the frozen existing clean4 deep benchmark",
        "test_used_for_training_or_tuning": False,
        "oom_policy": "batch size fixed at 16; stop and report before formal run if memory is insufficient",
    }


def run_identity(project_root: Path, batch_size: int) -> dict[str, Any]:
    model_path = project_root / "src" / "ultrasound_decoding" / "multiframe" / "spatial_mamba.py"
    runner_path = Path(__file__).resolve()
    transitive_paths = [
        project_root / "src" / "ultrasound_decoding" / "multiframe" / "factorized_transformer.py",
        project_root / "src" / "ultrasound_decoding" / "multiframe" / "training.py",
        project_root / "src" / "ultrasound_decoding" / "multiframe" / "dataset.py",
        project_root / "src" / "ultrasound_decoding" / "multiframe" / "models.py",
        project_root / "src" / "ultrasound_decoding" / "cv.py",
        project_root / "src" / "ultrasound_decoding" / "evaluate.py",
    ]
    return {
        "experiment_config": frozen_experiment_config(batch_size),
        "runtime_environment_signature": runtime_environment_signature(),
        "git_commit": git_text(project_root, "rev-parse", "HEAD"),
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        # File hashes also protect dirty/uncommitted code whose commit string has
        # not changed. A code edit therefore invalidates old task markers.
        "model_source_sha256": file_sha256(model_path),
        "runner_source_sha256": file_sha256(runner_path),
        "transitive_project_source_sha256": {
            str(path.relative_to(project_root)): file_sha256(path) for path in transitive_paths
        },
    }


def write_run_metadata(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        observed = json.loads(config_path.read_text())
        if fingerprint(observed) != fingerprint(identity):
            has_formal_tasks = (args.output_dir / "task_plan.csv").exists() or (
                args.output_dir / "tasks"
            ).exists()
            if has_formal_tasks:
                raise RuntimeError(
                    "existing formal tasks have a different code/config fingerprint; "
                    "use a new output directory and never mix them"
                )
    atomic_json(config_path, identity)
    atomic_json(args.output_dir / "environment.json", environment_payload(args.device))
    command = shlex.join(sys.argv) + "\n"
    atomic_text(args.output_dir / "command.txt", command)
    atomic_text(args.output_dir / f"{args.stage}_command.txt", command)
    atomic_json(
        args.output_dir / "git_state.json",
        {
            "commit": git_text(args.project_root, "rev-parse", "HEAD"),
            "branch": git_text(args.project_root, "branch", "--show-current"),
            "changed_files": git_text(args.project_root, "status", "--short").splitlines(),
            "diff_stat": git_text(args.project_root, "diff", "--stat"),
        },
    )


def _canonical_cycle_list(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return ",".join(str(int(float(token.strip()))) for token in text.split(","))


def canonical_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session", "task", "fold", "train_cycles", "test_cycles",
        "n_train_blocks", "n_test_blocks", "train_class_counts", "test_class_counts",
    ]
    if not set(columns).issubset(frame.columns):
        raise AssertionError(f"fold manifest missing columns: {sorted(set(columns) - set(frame.columns))}")
    out = frame[columns].copy()
    out["session"] = out["session"].map(lambda value: str(int(float(value))))
    out["task"] = out["task"].astype(str)
    for column in ("fold", "n_train_blocks", "n_test_blocks"):
        out[column] = pd.to_numeric(out[column], errors="raise").astype(np.int64)
    for column in ("train_cycles", "test_cycles"):
        out[column] = out[column].map(_canonical_cycle_list)
    for column in ("train_class_counts", "test_class_counts"):
        out[column] = out[column].map(lambda value: canonical_json(json.loads(str(value))))
    return out.reset_index(drop=True)


def formal_manifest_candidates(args: argparse.Namespace, session: str) -> list[Path]:
    return [
        args.project_root / "outputs" / "block_clean4_binary_all_models_9sessions_v1"
        / f"session_{session}" / "split_manifest.csv",
        args.project_root / "results" / "runs" / "multiframe"
        / "block_clean4_binary_all_models_v1" / f"session_{session}" / "split_manifest.csv",
        args.benchmark_root / f"session_{session}" / "split_manifest.csv",
        args.project_root / "outputs" / "frame_count_ablation_v1" / "parts"
        / f"session_{session}" / "k_4" / "pca_lda_flat4" / "seed_0" / "split_manifest.csv",
    ]


def audit_session(args: argparse.Namespace, session: str) -> tuple[Any, list[tuple[np.ndarray, np.ndarray]]]:
    data_dir = args.data_dir or default_block_data_dir(args.project_root)
    data = load_block_sequence_session(args.project_root, session, TASK_NAME, data_dir=data_dir)
    if tuple(data.X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(f"session {session}: expected {EXPECTED_BLOCK_SHAPE}, got {data.X.shape}")
    splits = grouped_cv_splits(data.groups, max_folds=MAX_FOLDS)
    current = split_manifest(session, TASK_NAME, data.y, data.groups, splits=splits, max_folds=MAX_FOLDS)
    historical_path = next((path for path in formal_manifest_candidates(args, session) if path.exists()), None)
    if historical_path is None:
        raise FileNotFoundError(
            "verified formal clean4 split manifest required; checked: "
            + ", ".join(str(path) for path in formal_manifest_candidates(args, session))
        )
    historical = pd.read_csv(historical_path)
    canonical_current = canonical_manifest(current)
    canonical_historical = canonical_manifest(historical)
    if not canonical_current.equals(canonical_historical):
        raise AssertionError(
            f"session {session}: true content mismatch against formal clean4 manifest {historical_path}"
        )
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        overlap = np.intersect1d(data.groups[train_idx], data.groups[test_idx])
        if overlap.size:
            raise AssertionError(f"session {session} fold {fold}: cycle leakage {overlap.tolist()}")
    audit_dir = args.output_dir / "audit" / f"session_{session}"
    atomic_csv(audit_dir / "split_manifest.csv", canonical_current)
    atomic_json(
        audit_dir / "dataset.json",
        {
            "session": session,
            "source_h5": str(data.source_h5_path),
            "source_metadata": str(data.source_metadata_path),
            "shape": list(data.X.shape),
            "n_cycles": data.n_cycles,
            "n_samples": data.n_blocks,
            "class_counts": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(data.y, return_counts=True))
            },
            "formal_clean4_fold_match": True,
            "formal_manifest_source": str(historical_path),
            "manifest_comparison": "canonical numeric/text dtype normalization then exact table equality",
            "train_test_cycle_overlap_all_folds": False,
        },
    )
    return data, splits


def task_dir(output_dir: Path, session: str, seed: int, fold: int) -> Path:
    return output_dir / "tasks" / f"session_{session}" / MODEL_NAME / f"seed_{seed}" / f"fold_{fold:02d}"


def task_key(session: str, seed: int, fold: int) -> str:
    return f"{session}:{MODEL_NAME}:{seed}:{fold}"


def task_fingerprint(run_fingerprint: str, row: dict[str, Any]) -> str:
    return fingerprint(
        {
            "run_fingerprint": run_fingerprint,
            "session": str(row["session"]),
            "model": str(row["model"]),
            "seed": int(row["seed"]),
            "fold": int(row["fold"]),
            "n_test_samples": int(row["n_test_samples"]),
            "config_fingerprint": str(row["config_fingerprint"]),
            "runtime_environment_fingerprint": str(row["runtime_environment_fingerprint"]),
            "batch_size": int(row["batch_size"]),
        }
    )


def validate_completed_task(
    path: Path,
    expected: dict[str, Any],
    run_fingerprint: str,
    *,
    raise_on_error: bool = False,
) -> tuple[bool, str]:
    """Revalidate every task artifact; COMPLETE.json alone is never sufficient."""

    def fail(message: str) -> tuple[bool, str]:
        if raise_on_error:
            raise AssertionError(f"invalid completed task {path}: {message}")
        return False, message

    required_names = (
        "COMPLETE.json", "result.json", "predictions.csv", "confusion_matrix.csv",
        "training_history.csv", "normalization_audit.json", "model_config.json",
    )
    missing = [name for name in required_names if not (path / name).exists() or (path / name).stat().st_size == 0]
    if missing:
        return fail(f"missing/empty files {missing}")
    try:
        complete = json.loads((path / "COMPLETE.json").read_text())
        result = json.loads((path / "result.json").read_text())
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        confusion = pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str})
        history = pd.read_csv(path / "training_history.csv", dtype={"session": str})
        normalization = json.loads((path / "normalization_audit.json").read_text())
        model_config = json.loads((path / "model_config.json").read_text())
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")

    expected_task_fingerprint = task_fingerprint(run_fingerprint, expected)
    expected_key = task_key(str(expected["session"]), int(expected["seed"]), int(expected["fold"]))
    if complete.get("task_key") != expected_key:
        return fail("COMPLETE task_key mismatch")
    if complete.get("run_fingerprint") != run_fingerprint or result.get("run_fingerprint") != run_fingerprint:
        return fail("run fingerprint mismatch")
    if complete.get("task_fingerprint") != expected_task_fingerprint or result.get("task_fingerprint") != expected_task_fingerprint:
        return fail("task fingerprint mismatch")
    if (
        complete.get("config_fingerprint") != str(expected["config_fingerprint"])
        or result.get("config_fingerprint") != str(expected["config_fingerprint"])
    ):
        return fail("experiment config fingerprint mismatch")
    if (
        complete.get("runtime_environment_fingerprint")
        != str(expected["runtime_environment_fingerprint"])
        or result.get("runtime_environment_fingerprint")
        != str(expected["runtime_environment_fingerprint"])
    ):
        return fail("runtime environment fingerprint mismatch")
    expected_identity = (
        str(expected["session"]), MODEL_NAME, int(expected["seed"]), int(expected["fold"]),
    )
    result_identity = (
        str(result.get("session")), str(result.get("model")),
        int(result.get("seed", -1)), int(result.get("fold", -1)),
    )
    normalization_identity = (
        str(normalization.get("session")), str(normalization.get("method")),
        int(normalization.get("seed", -1)), int(normalization.get("fold", -1)),
    )
    if result_identity != expected_identity:
        return fail(f"result identity mismatch {result_identity} != {expected_identity}")
    if normalization_identity != expected_identity:
        return fail(
            f"normalization identity mismatch {normalization_identity} != {expected_identity}"
        )
    if result.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        return fail("result model implementation version mismatch")
    if model_config.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        return fail("model_config implementation version mismatch")
    if model_config.get("model") != MODEL_NAME:
        return fail("model_config model mismatch")
    expected_model_config = architecture_config(SpatialMambaConfig())
    observed_architecture = {
        key: value for key, value in model_config.items() if key != "parameter_breakdown"
    }
    if fingerprint(observed_architecture) != fingerprint(expected_model_config):
        return fail("model_config differs from the frozen architecture")
    breakdown = model_config.get("parameter_breakdown")
    required_parameter_fields = {
        "cnn_stem_parameters", "spatial_mamba_parameters",
        "temporal_transformer_parameters", "classifier_parameters",
        "total_parameter_count",
    }
    if not isinstance(breakdown, dict) or not required_parameter_fields.issubset(breakdown):
        return fail("model_config parameter breakdown missing")
    component_sum = sum(int(breakdown[name]) for name in required_parameter_fields if name != "total_parameter_count")
    if component_sum != int(breakdown["total_parameter_count"]):
        return fail("parameter breakdown does not sum to total")
    if int(result.get("parameter_count", -1)) != int(breakdown["total_parameter_count"]):
        return fail("result/model_config total parameter mismatch")
    for field in required_parameter_fields:
        if int(result.get(field, -1)) != int(breakdown[field]):
            return fail(f"result parameter field mismatch: {field}")
    if int(result.get("actual_batch_size", -1)) != int(expected["batch_size"]):
        return fail("actual batch size differs from frozen task config")
    if bool(normalization.get("target_used_for_stats", True)):
        return fail("normalization audit says target/test was used for statistics")
    if normalization.get("phase") != "outer_train_fold_only":
        return fail("normalization phase is not outer_train_fold_only")

    expected_n = int(expected["n_test_samples"])
    if int(result.get("n_test_samples", -1)) != expected_n or len(predictions) != expected_n:
        return fail("prediction count mismatch")
    required_prediction_columns = {
        "session", "model", "seed", "fold", "sample_index", "block_id", "cycle",
        "block_name", "y_true", "y_pred", "probability_0", "probability_1",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        return fail(f"prediction columns missing {sorted(required_prediction_columns - set(predictions.columns))}")
    if len(predictions) and not (
        predictions["session"].eq(str(expected["session"])).all()
        and predictions["model"].eq(MODEL_NAME).all()
        and pd.to_numeric(predictions["seed"], errors="coerce").eq(int(expected["seed"])).all()
        and pd.to_numeric(predictions["fold"], errors="coerce").eq(int(expected["fold"])).all()
    ):
        return fail("prediction identity columns mismatch")
    probability = predictions[["probability_0", "probability_1"]].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        return fail("prediction probabilities invalid")
    if not set(pd.to_numeric(predictions["y_true"], errors="raise").astype(int).unique()).issubset({0, 1}):
        return fail("invalid ground-truth labels")
    if not set(pd.to_numeric(predictions["y_pred"], errors="raise").astype(int).unique()).issubset({0, 1}):
        return fail("invalid predicted labels")

    required_metrics = ("balanced_accuracy", "accuracy", "macro_f1")
    if any(metric not in result or not np.isfinite(float(result[metric])) for metric in required_metrics):
        return fail("missing or non-finite required metric")
    recalculated = classification_metrics(
        predictions["y_true"].to_numpy(dtype=int), predictions["y_pred"].to_numpy(dtype=int)
    )
    if any(not np.isclose(float(result[name]), float(recalculated[name]), atol=1e-12) for name in required_metrics):
        return fail("stored metrics differ from prediction-derived metrics")

    required_confusion_columns = {"session", "model", "seed", "fold", "true_label", "predicted_label", "count"}
    if not required_confusion_columns.issubset(confusion.columns):
        return fail("confusion matrix columns missing")
    if len(confusion) != 4 or int(pd.to_numeric(confusion["count"], errors="raise").sum()) != expected_n:
        return fail("confusion matrix count mismatch")
    if not (
        confusion["session"].eq(str(expected["session"])).all()
        and confusion["model"].eq(MODEL_NAME).all()
        and pd.to_numeric(confusion["seed"], errors="coerce").eq(int(expected["seed"])).all()
        and pd.to_numeric(confusion["fold"], errors="coerce").eq(int(expected["fold"])).all()
    ):
        return fail("confusion identity columns mismatch")
    stored_cm = np.zeros((2, 2), dtype=int)
    for row in confusion.itertuples(index=False):
        stored_cm[int(row.true_label), int(row.predicted_label)] = int(row.count)
    expected_cm = confusion_matrix(
        predictions["y_true"].to_numpy(dtype=int), predictions["y_pred"].to_numpy(dtype=int),
        np.asarray([0, 1]),
    )
    if not np.array_equal(stored_cm, expected_cm):
        return fail("confusion matrix differs from predictions")

    required_history_columns = {"session", "model", "seed", "fold", "epoch", "train_loss", "train_accuracy"}
    if not required_history_columns.issubset(history.columns) or len(history) != FORMAL_EPOCHS:
        return fail("training history missing columns or epochs")
    if not (
        history["session"].eq(str(expected["session"])).all()
        and history["model"].eq(MODEL_NAME).all()
        and pd.to_numeric(history["seed"], errors="coerce").eq(int(expected["seed"])).all()
        and pd.to_numeric(history["fold"], errors="coerce").eq(int(expected["fold"])).all()
        and np.array_equal(pd.to_numeric(history["epoch"], errors="raise").to_numpy(dtype=int), np.arange(1, FORMAL_EPOCHS + 1))
    ):
        return fail("training history identity/epoch mismatch")
    if not np.isfinite(history[["train_loss", "train_accuracy"]].to_numpy(dtype=float)).all():
        return fail("non-finite training history")
    if int(result.get("trained_epochs", -1)) != FORMAL_EPOCHS:
        return fail("result does not contain exactly 40 trained epochs")
    return True, "validated"


def build_task_plan(args: argparse.Namespace, identity: dict[str, Any]) -> pd.DataFrame:
    run_fp = fingerprint(identity)
    config_fp = fingerprint(identity["experiment_config"])
    runtime_fp = fingerprint(identity["runtime_environment_signature"])
    batch_size = int(identity["experiment_config"]["training"]["batch_size"])
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        data, splits = audit_session(args, session)
        audits.append(
            {
                "session": session, "n_cycles": data.n_cycles, "n_samples": data.n_blocks,
                "n_folds": len(splits), "input_shape": canonical_json(list(data.X.shape[1:])),
                "formal_clean4_fold_match": True,
            }
        )
        for seed in SEEDS:
            for fold, (_, test_idx) in enumerate(splits, start=1):
                row = {
                    "session": session, "model": MODEL_NAME, "seed": seed, "fold": fold,
                    "n_test_samples": len(test_idx),
                    "task_key": task_key(session, seed, fold),
                    "config_fingerprint": config_fp,
                    "runtime_environment_fingerprint": runtime_fp,
                    "batch_size": batch_size,
                }
                row["task_fingerprint"] = task_fingerprint(run_fp, row)
                rows.append(row)
        del data
    plan = pd.DataFrame(rows)
    atomic_csv(args.output_dir / "task_plan.csv", plan)
    atomic_csv(args.output_dir / "dataset_and_fold_audit.csv", pd.DataFrame(audits))
    atomic_json(
        args.output_dir / "task_plan_metadata.json",
        {
            "run_fingerprint": run_fp,
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "runtime_environment_signature": identity["runtime_environment_signature"],
            "total_tasks": len(plan),
            "task_definition": "session x one_model x seed x fold",
            "created_utc": utc_now(),
        },
    )
    return plan


def load_or_build_task_plan(args: argparse.Namespace, identity: dict[str, Any]) -> pd.DataFrame:
    plan_path = args.output_dir / "task_plan.csv"
    metadata_path = args.output_dir / "task_plan_metadata.json"
    run_fp = fingerprint(identity)
    if plan_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("run_fingerprint") != run_fp:
            raise RuntimeError("existing task plan has a different code/config fingerprint")
        plan = pd.read_csv(plan_path, dtype={"session": str})
        for row in plan.to_dict(orient="records"):
            if row.get("task_fingerprint") != task_fingerprint(run_fp, row):
                raise AssertionError("task plan contains an invalid task fingerprint")
        return plan
    return build_task_plan(args, identity)


def update_status(args: argparse.Namespace, plan: pd.DataFrame, run_fp: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir, str(expected["session"]), int(expected["seed"]), int(expected["fold"])
        )
        valid, reason = validate_completed_task(path, expected, run_fp)
        rows.append(
            {
                "session": str(expected["session"]), "model": MODEL_NAME,
                "seed": int(expected["seed"]), "fold": int(expected["fold"]),
                "status": "complete" if valid else "pending", "validation": reason,
                "task_dir": str(path),
            }
        )
    status = pd.DataFrame(rows)
    atomic_csv(args.output_dir / "run_status.csv", status)
    completed = int(status["status"].eq("complete").sum())
    total = len(status)
    print(f"STATUS completed={completed} pending={total - completed} total={total}", flush=True)
    if completed < total:
        next_rows = status[status["status"].eq("pending")].head(5)
        for row in next_rows.itertuples(index=False):
            print(f"PENDING session={row.session} fold={row.fold} seed={row.seed} reason={row.validation}", flush=True)
    elif (args.output_dir / "RUN_COMPLETE.json").exists():
        print(f"FULLY_COMPLETE marker={args.output_dir / 'RUN_COMPLETE.json'}", flush=True)
    else:
        print("TASKS_COMPLETE_FINALIZATION_PENDING", flush=True)
    return status


def select_balanced_indices(y: np.ndarray, candidates: np.ndarray, per_class: int) -> np.ndarray:
    selected: list[int] = []
    candidate_array = np.asarray(candidates, dtype=np.int64)
    for label in sorted(np.unique(y[candidate_array]).tolist()):
        selected.extend(candidate_array[y[candidate_array] == label][:per_class].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def run_sanity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    require_mamba_dependency()
    if args.workers != 0:
        raise ValueError("local sanity requires --workers 0")
    if args.sanity_epochs not in (2, 3):
        raise ValueError("sanity is restricted to 2 or 3 epochs")
    data, splits = audit_session(args, "710")
    train_idx, test_idx = splits[0]
    tiny_train = select_balanced_indices(data.y, train_idx, per_class=4)
    tiny_test = select_balanced_indices(data.y, test_idx, per_class=2)
    if len(tiny_train) < 4 or len(tiny_test) < 2:
        raise AssertionError("session 710 fold 1 does not support the tiny balanced sanity subset")

    config = SpatialMambaConfig()
    model = SpatialMambaClassifier(config).to(args.device)
    x = blocks_to_sequence_tensor(data.X[tiny_train[:2]]).to(args.device)
    y = torch.from_numpy(data.y[tiny_train[:2]].astype(np.int64)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    logits, shapes = model.forward_with_shapes(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients_finite = all(
        item.grad is None or bool(torch.isfinite(item.grad).all().item())
        for item in model.parameters()
    )
    optimizer.step()
    parameter_changed = not torch.equal(before, parameter.detach())
    expected_shapes = {
        "input": (2, 4, 1, 128, 501),
        "cnn_output": (2, 4, 64, 8, 32),
        "spatial_tokens": (8, 256, 64),
        "spatial_mamba_output": (8, 256, 64),
        "temporal_transformer_input": (2, 4, 64),
        "temporal_transformer_output": (2, 4, 64),
        "pooled": (2, 64),
        "logits": (2, 2),
    }
    if shapes != expected_shapes:
        raise AssertionError(f"unexpected sanity shapes: {shapes}")
    if not bool(torch.isfinite(loss).item()) or not gradients_finite or not parameter_changed:
        raise AssertionError("forward/backward/update sanity failed")

    tiny_training = DeepTrainingConfig(
        optimizer="adamw", lr=1e-3, weight_decay=1e-3,
        batch_size=min(4, len(tiny_train)), max_epochs=args.sanity_epochs,
        dropout=0.25, loss="cross_entropy",
    )
    result = train_spatial_mamba_fold(
        data.X[tiny_train], data.y[tiny_train], data.X[tiny_test],
        np.asarray([0, 1], dtype=np.int64),
        session="710", fold=1, seed=0,
        train_cycles=cycle_text(data.groups[tiny_train]),
        test_cycles=cycle_text(data.groups[tiny_test]),
        training_config=tiny_training, architecture=config,
        device=args.device, workers=0,
    )
    if len(result.history) != args.sanity_epochs:
        raise AssertionError("tiny fit did not execute the requested sanity epochs")
    if not np.isfinite(result.probabilities).all() or not np.allclose(
        result.probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise AssertionError("sanity prediction probabilities are invalid")
    if bool(result.normalization_audit["target_used_for_stats"]):
        raise AssertionError("test data participated in normalization")
    sanity_dir = args.output_dir / "sanity"
    atomic_csv(
        sanity_dir / "sanity_results.csv",
        pd.DataFrame(
            [
                {
                    "session": "710", "model": MODEL_NAME, "fold": 1, "seed": 0,
                    "input_shape_without_channel": canonical_json(list(data.X.shape[1:])),
                    "cnn_output_shape": canonical_json(list(shapes["cnn_output"])),
                    "spatial_tokens_shape": canonical_json(list(shapes["spatial_tokens"])),
                    "temporal_input_shape": canonical_json(list(shapes["temporal_transformer_input"])),
                    "logits_shape": canonical_json(list(shapes["logits"])),
                    "spatial_tokens_per_frame": 256,
                    "formal_clean4_fold_match": True,
                    "cycle_overlap": False,
                    "loss_finite": bool(torch.isfinite(loss).item()),
                    "backward_success": gradients_finite,
                    "parameter_changed": parameter_changed,
                    "probabilities_finite": bool(np.isfinite(result.probabilities).all()),
                    "normalization_target_used_for_stats": False,
                    "tiny_epochs": args.sanity_epochs,
                    "tiny_train_samples": len(tiny_train), "tiny_test_samples": len(tiny_test),
                    "debug_only_not_formal": True,
                }
            ]
        ),
    )
    atomic_json(
        sanity_dir / "SANITY_COMPLETE.json",
        {
            "completed_utc": utc_now(), "run_fingerprint": fingerprint(identity),
            "session": "710", "fold": 1, "seed": 0,
            "formal_results": False, "checks_passed": True,
        },
    )
    print(f"SANITY PASS: {sanity_dir / 'SANITY_COMPLETE.json'}", flush=True)


def write_fold_task(
    args: argparse.Namespace,
    identity: dict[str, Any],
    expected: dict[str, Any],
    data: Any,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    run_fp = fingerprint(identity)
    session = str(expected["session"])
    seed = int(expected["seed"])
    fold = int(expected["fold"])
    path = task_dir(args.output_dir, session, seed, fold)
    train_cycles = cycle_text(data.groups[train_idx])
    test_cycles = cycle_text(data.groups[test_idx])
    result = train_spatial_mamba_fold(
        data.X[train_idx], data.y[train_idx], data.X[test_idx],
        np.asarray([0, 1], dtype=np.int64),
        session=session, fold=fold, seed=seed,
        train_cycles=train_cycles, test_cycles=test_cycles,
        training_config=DeepTrainingConfig(**identity["experiment_config"]["training"]),
        architecture=SpatialMambaConfig(), device=args.device, workers=args.workers,
    )
    metrics = classification_metrics(data.y[test_idx], result.predictions)
    predictions: list[dict[str, Any]] = []
    for local_i, sample_i in enumerate(test_idx):
        metadata = data.metadata.iloc[int(sample_i)]
        predictions.append(
            {
                "session": session, "model": MODEL_NAME, "seed": seed, "fold": fold,
                "sample_index": int(sample_i), "block_id": str(metadata["block_id"]),
                "cycle": int(data.groups[sample_i]), "block_name": str(metadata["block_name"]),
                "y_true": int(data.y[sample_i]), "y_pred": int(result.predictions[local_i]),
                "probability_0": float(result.probabilities[local_i, 0]),
                "probability_1": float(result.probabilities[local_i, 1]),
            }
        )
    cm = confusion_matrix(data.y[test_idx], result.predictions, np.asarray([0, 1]))
    confusions = [
        {
            "session": session, "model": MODEL_NAME, "seed": seed, "fold": fold,
            "true_label": true_label, "predicted_label": predicted_label,
            "count": int(cm[true_label, predicted_label]), "scope": "fold",
        }
        for true_label in (0, 1) for predicted_label in (0, 1)
    ]
    history = pd.DataFrame(result.history)
    history.insert(0, "fold", fold)
    history.insert(0, "seed", seed)
    history.insert(0, "model", MODEL_NAME)
    history.insert(0, "session", session)
    task_fp = task_fingerprint(run_fp, expected)
    breakdown = result.model_config["parameter_breakdown"]
    result_payload = {
        "run_fingerprint": run_fp, "task_fingerprint": task_fp,
        "config_fingerprint": str(expected["config_fingerprint"]),
        "runtime_environment_fingerprint": str(expected["runtime_environment_fingerprint"]),
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "runtime_environment_signature": identity["runtime_environment_signature"],
        "session": session, "model": MODEL_NAME, "seed": seed, "fold": fold,
        "n_cycles": data.n_cycles, "n_samples": data.n_blocks,
        "n_train_samples": len(train_idx), "n_test_samples": len(test_idx),
        "train_cycles": train_cycles, "test_cycles": test_cycles,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"]),
        "parameter_count": int(result.model_parameters),
        "cnn_stem_parameters": int(breakdown["cnn_stem_parameters"]),
        "spatial_mamba_parameters": int(breakdown["spatial_mamba_parameters"]),
        "temporal_transformer_parameters": int(breakdown["temporal_transformer_parameters"]),
        "classifier_parameters": int(breakdown["classifier_parameters"]),
        "total_parameter_count": int(breakdown["total_parameter_count"]),
        "transformer_reference_parameter_count": TRANSFORMER_REFERENCE_PARAMETER_COUNT,
        "parameter_delta_vs_transformer": int(
            breakdown["total_parameter_count"] - TRANSFORMER_REFERENCE_PARAMETER_COUNT
        ),
        "actual_batch_size": int(identity["experiment_config"]["training"]["batch_size"]),
        "final_training_loss": float(result.final_training_loss),
        "trained_epochs": int(result.final_trained_epochs), "device": result.device,
    }
    # COMPLETE is written last, after every artifact has reached disk.
    atomic_json(path / "result.json", result_payload)
    atomic_csv(path / "predictions.csv", pd.DataFrame(predictions))
    atomic_csv(path / "confusion_matrix.csv", pd.DataFrame(confusions))
    atomic_csv(path / "training_history.csv", history)
    atomic_json(path / "normalization_audit.json", result.normalization_audit)
    atomic_json(path / "model_config.json", result.model_config)
    atomic_json(
        path / "COMPLETE.json",
        {
            "task_key": task_key(session, seed, fold),
            "run_fingerprint": run_fp, "task_fingerprint": task_fp,
            "config_fingerprint": str(expected["config_fingerprint"]),
            "runtime_environment_fingerprint": str(expected["runtime_environment_fingerprint"]),
            "completed_utc": utc_now(),
            "validated_files": [
                "result.json", "predictions.csv", "confusion_matrix.csv",
                "training_history.csv", "normalization_audit.json", "model_config.json",
            ],
        },
    )
    validate_completed_task(path, expected, run_fp, raise_on_error=True)


def read_all_validated_tasks(
    args: argparse.Namespace, plan: pd.DataFrame, run_fp: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    confusions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir, str(expected["session"]), int(expected["seed"]), int(expected["fold"])
        )
        validate_completed_task(path, expected, run_fp, raise_on_error=True)
        results.append(json.loads((path / "result.json").read_text()))
        predictions.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        confusions.append(pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str}))
        histories.append(pd.read_csv(path / "training_history.csv", dtype={"session": str}))
    return (
        pd.DataFrame(results), pd.concat(predictions, ignore_index=True),
        pd.concat(confusions, ignore_index=True), pd.concat(histories, ignore_index=True),
    )


EXISTING_BASELINE_DISPLAY = {
    "pca_lda_flat4": "PCA+LDA",
    "cpca_lda_flat4": "cPCA+LDA",
    "fcnn_meanpool": "FCNN mean-pool",
    "cnn2d_meanpool": "CNN mean-pool",
    "cnn2d_lstm": "CNN-LSTM",
    "cnn2d_temporal1d": "Temporal 1D-CNN",
    "sbind_noatt": "SBIND-adapted-NoAtt",
    "sbind": "SBIND-adapted",
    "cnn_factorized_transformer": "CNN Factorized Transformer",
    MODEL_NAME: MODEL_DISPLAY_NAME,
}


def clean4_long_candidates(args: argparse.Namespace) -> list[Path]:
    return [
        args.project_root / "outputs" / "block_clean4_binary_all_models_9sessions_v1"
        / "aggregate" / "multiframe_all_models_master_long.csv",
        args.project_root / "results" / "runs" / "multiframe"
        / "block_clean4_binary_all_models_v1" / "aggregate" / "multiframe_all_models_master_long.csv",
        args.benchmark_root / "aggregate" / "multiframe_master_summary.csv",
    ]


def sbind_summary_candidates(args: argparse.Namespace) -> list[Path]:
    base = args.project_root / "outputs" / "sbind_visual_binary_v1"
    return [
        base / "sbind_summary.csv",
        base / "sbind_visual_binary_v1" / "sbind_summary.csv",
    ]


def transformer_summary_candidates(args: argparse.Namespace) -> list[Path]:
    base = args.project_root / "outputs" / "transformer_visual_binary_v1"
    return [
        base / "transformer_summary.csv",
        base / "transformer_visual_binary_v1" / "transformer_summary.csv",
    ]


def load_existing_baselines(args: argparse.Namespace) -> pd.DataFrame:
    selected_models = set(EXISTING_BASELINE_DISPLAY) - {
        MODEL_NAME, "sbind_noatt", "sbind", "cnn_factorized_transformer"
    }
    rows: list[dict[str, Any]] = []
    for priority, path in enumerate(clean4_long_candidates(args)):
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"session": str})
        if "task" in frame.columns:
            frame = frame[frame["task"].astype(str).eq(TASK_NAME)]
        required = {"session", "method", "balanced_accuracy", "accuracy"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[frame["method"].astype(str).isin(selected_models)]
        for (session, method), group in frame.groupby(["session", "method"], sort=True):
            rows.append(
                {
                    "session": str(session), "model": str(method),
                    "model_display": EXISTING_BASELINE_DISPLAY[str(method)],
                    "mean_BA": float(group["balanced_accuracy"].astype(float).mean()),
                    "std_BA": float(group["balanced_accuracy"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
                    "mean_accuracy": float(group["accuracy"].astype(float).mean()),
                    "n_seeds": int(group["seed"].nunique()) if "seed" in group.columns else len(group),
                    "source": str(path), "source_priority": priority,
                }
            )
    if rows:
        clean = (
            pd.DataFrame(rows).sort_values("source_priority")
            .drop_duplicates(["session", "model"], keep="first")
        )
    else:
        clean = pd.DataFrame()

    sbind_rows: list[dict[str, Any]] = []
    for priority, path in enumerate(sbind_summary_candidates(args)):
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"session": str})
        required = {"session", "model", "mean_BA", "std_BA", "mean_accuracy"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[frame["model"].isin(["sbind_noatt", "sbind"])]
        for row in frame.itertuples(index=False):
            sbind_rows.append(
                {
                    "session": str(row.session), "model": str(row.model),
                    "model_display": EXISTING_BASELINE_DISPLAY[str(row.model)],
                    "mean_BA": float(row.mean_BA), "std_BA": float(row.std_BA),
                    "mean_accuracy": float(row.mean_accuracy), "n_seeds": len(SEEDS),
                    "source": str(path), "source_priority": priority,
                }
            )
    sbind = (
        pd.DataFrame(sbind_rows).sort_values("source_priority")
        .drop_duplicates(["session", "model"], keep="first")
        if sbind_rows else pd.DataFrame()
    )
    transformer_rows: list[dict[str, Any]] = []
    for priority, path in enumerate(transformer_summary_candidates(args)):
        if not path.exists():
            continue
        frame = pd.read_csv(path, dtype={"session": str})
        required = {"session", "model", "mean_BA", "std_BA", "mean_accuracy"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[frame["model"].astype(str).eq("cnn_factorized_transformer")]
        for row in frame.itertuples(index=False):
            transformer_rows.append(
                {
                    "session": str(row.session), "model": "cnn_factorized_transformer",
                    "model_display": EXISTING_BASELINE_DISPLAY["cnn_factorized_transformer"],
                    "mean_BA": float(row.mean_BA), "std_BA": float(row.std_BA),
                    "mean_accuracy": float(row.mean_accuracy), "n_seeds": len(SEEDS),
                    "source": str(path), "source_priority": priority,
                }
            )
    transformer = (
        pd.DataFrame(transformer_rows).sort_values("source_priority")
        .drop_duplicates(["session", "model"], keep="first")
        if transformer_rows else pd.DataFrame()
    )
    combined = pd.concat([clean, sbind, transformer], ignore_index=True)
    expected_pairs = {
        (session, model)
        for session in EXPECTED_SESSIONS
        for model in EXISTING_BASELINE_DISPLAY
        if model != MODEL_NAME
    }
    observed_pairs = set(zip(combined.get("session", []), combined.get("model", [])))
    missing = sorted(expected_pairs - observed_pairs)
    if missing:
        raise FileNotFoundError(
            "existing formal baseline rows are incomplete; no old model will be retrained. "
            f"Missing session/model pairs: {missing}. Checked clean4={clean4_long_candidates(args)}; "
            f"SBIND={sbind_summary_candidates(args)}; "
            f"Transformer={transformer_summary_candidates(args)}"
        )
    return combined.drop(columns=["source_priority"]).sort_values(["session", "model"])


def exact_two_sided_sign_flip(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("sign-flip deltas must be a finite non-empty vector")
    observed = abs(float(values.mean()))
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted.append(abs(float(np.mean(values * np.asarray(signs)))))
    return float(np.mean(np.asarray(permuted) >= observed - 1e-15))


def paired_comparison_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    rows: list[dict[str, Any]] = []
    for baseline in ("cnn2d_meanpool", "cnn_factorized_transformer", "sbind"):
        required_sessions = [session for session in EXPECTED_SESSIONS if session in pivot.index]
        if required_sessions != list(EXPECTED_SESSIONS) or baseline not in pivot or MODEL_NAME not in pivot:
            raise AssertionError(f"paired comparison is incomplete for {baseline}")
        deltas = (
            pivot.loc[list(EXPECTED_SESSIONS), MODEL_NAME]
            - pivot.loc[list(EXPECTED_SESSIONS), baseline]
        ).to_numpy(dtype=float)
        tolerance = 1e-12
        rows.append(
            {
                "comparison": f"{MODEL_NAME}_vs_{baseline}", "baseline": baseline,
                "n_sessions": len(deltas), "mean_delta_BA": float(deltas.mean()),
                "median_delta_BA": float(np.median(deltas)),
                "improved_sessions": int((deltas > tolerance).sum()),
                "tied_sessions": int((np.abs(deltas) <= tolerance).sum()),
                "worsened_sessions": int((deltas < -tolerance).sum()),
                "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(deltas),
                "session_deltas_json": canonical_json(
                    {session: float(delta) for session, delta in zip(EXPECTED_SESSIONS, deltas)}
                ),
            }
        )
    return pd.DataFrame(rows)


def build_overfitting_summary(history: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (session, seed), group in history.groupby(["session", "seed"], sort=True):
        by_epoch = group.groupby("epoch", as_index=True)["train_accuracy"].mean().sort_index()
        if list(by_epoch.index.astype(int)) != list(range(1, FORMAL_EPOCHS + 1)):
            raise AssertionError(f"overfitting audit history is incomplete for {session} seed {seed}")
        best_epoch = int(by_epoch.idxmax())
        oof = per_seed[
            per_seed["session"].astype(str).eq(str(session))
            & per_seed["seed"].astype(int).eq(int(seed))
        ]
        if len(oof) != 1:
            raise AssertionError(f"missing unique OOF row for {session} seed {seed}")
        oof_ba = float(oof.iloc[0]["balanced_accuracy"])
        rows.append(
            {
                "session": str(session), "model": MODEL_NAME, "seed": int(seed),
                "final_train_accuracy": float(by_epoch.loc[FORMAL_EPOCHS]),
                "best_train_accuracy": float(by_epoch.loc[best_epoch]),
                "validation_accuracy": np.nan,
                "validation_used": False,
                "validation_note": "not applicable: frozen clean4 deep benchmark uses fixed epochs and no validation fold",
                "OOF_test_BA": oof_ba,
                "best_epoch": best_epoch,
                "selected_epoch": FORMAL_EPOCHS,
                "epoch_selection": "fixed_40_no_test_or_validation_selection",
                "n_folds": int(group["fold"].nunique()),
                "possible_severe_overfit": bool(float(by_epoch.loc[best_epoch]) >= 0.95 and oof_ba <= 0.60),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    comparison: pd.DataFrame, paired: pd.DataFrame, overfitting: pd.DataFrame
) -> str:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    mamba_mean = float(pivot[MODEL_NAME].mean())
    cnn_mean = float(pivot["cnn2d_meanpool"].mean())
    transformer_mean = float(pivot["cnn_factorized_transformer"].mean())
    sbind_mean = float(pivot["sbind"].mean())
    cnn_stats = paired[paired["baseline"].eq("cnn2d_meanpool")].iloc[0]
    transformer_stats = paired[
        paired["baseline"].eq("cnn_factorized_transformer")
    ].iloc[0]
    sbind_stats = paired[paired["baseline"].eq("sbind")].iloc[0]
    strong = ["708", "709", "710"]
    strong_values = pivot.loc[strong, MODEL_NAME]
    weak = [session for session in EXPECTED_SESSIONS if float(pivot.loc[session, "cnn2d_meanpool"]) <= 0.60]
    weak_deltas = (
        pivot.loc[weak, MODEL_NAME] - pivot.loc[weak, "cnn2d_meanpool"]
        if weak else pd.Series(dtype=float)
    )
    severe = int(overfitting.groupby("session")["possible_severe_overfit"].any().sum())
    parameter_count = int(comparison.loc[comparison["model"].eq(MODEL_NAME), "parameter_count"].dropna().iloc[0]) if "parameter_count" in comparison else -1
    parameter_delta = parameter_count - TRANSFORMER_REFERENCE_PARAMETER_COUNT
    return "\n".join(
        [
            "# Visual fUS Spatial-Mamba Baseline v1",
            "",
            "compute_environment = server",
            "",
            "本报告只解释冻结的 `spatial_mamba` controlled baseline；没有据此修改模型或超参数。",
            "",
            f"1. Spatial Mamba 平均 BA={mamba_mean:.4f}；CNN mean-pool={cnn_mean:.4f}，"
            f"平均差={float(cnn_stats['mean_delta_BA']):+.4f}，exact p={float(cnn_stats['exact_two_sided_sign_flip_p']):.4f}。",
            f"2. Factorized Transformer 平均 BA={transformer_mean:.4f}；Mamba 平均差="
            f"{float(transformer_stats['mean_delta_BA']):+.4f}，exact p="
            f"{float(transformer_stats['exact_two_sided_sign_flip_p']):.4f}。",
            f"3. SBIND-adapted 平均 BA={sbind_mean:.4f}；Mamba 平均差="
            f"{float(sbind_stats['mean_delta_BA']):+.4f}，exact p={float(sbind_stats['exact_two_sided_sign_flip_p']):.4f}。",
            f"4. 相对 CNN 改善/持平/下降={int(cnn_stats['improved_sessions'])}/"
            f"{int(cnn_stats['tied_sessions'])}/{int(cnn_stats['worsened_sessions'])}；相对 SBIND："
            f"{int(sbind_stats['improved_sessions'])}/{int(sbind_stats['tied_sessions'])}/"
            f"{int(sbind_stats['worsened_sessions'])}；相对 Transformer："
            f"{int(transformer_stats['improved_sessions'])}/{int(transformer_stats['tied_sessions'])}/"
            f"{int(transformer_stats['worsened_sessions'])}。",
            f"5. 强 session 708/709/710 的 Spatial Mamba BA 分别为 "
            + ", ".join(f"{session}:{float(strong_values.loc[session]):.3f}" for session in strong)
            + f"；达到或超过 0.90 的数量为 {int((strong_values >= 0.90).sum())}/3。",
            f"6. 按既有 CNN BA<=0.60 定义的弱 session 为 {weak}；其中相对 CNN 改善 "
            f"{int((weak_deltas > 1e-12).sum())}/{len(weak_deltas)}，平均差="
            f"{float(weak_deltas.mean()) if len(weak_deltas) else float('nan'):+.4f}。",
            f"7. 训练准确率>=0.95 且 OOF BA<=0.60 的 session 数为 {severe}/9。"
            "验证准确率不适用，因为冻结 benchmark 没有 validation/early stopping；正式选择始终为 epoch 40。",
            f"8. Spatial Mamba 参数量={parameter_count}；Factorized Transformer 参数量="
            f"{TRANSFORMER_REFERENCE_PARAMETER_COUNT}；差异={parameter_delta:+d}。未为参数匹配修改 d_model。",
            "",
            "是否更适合当前小样本视觉 fUS，应依据相对 Transformer 的跨 session 一致性、exact p 值和过拟合审计人工判断。",
            "",
            "停止于 controlled baseline；不自动开发 Mamba2/Mamba3、其他扫描或 proposed model。",
        ]
    ) + "\n"


def run_cuda_batch16_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the frozen formal batch fits before any formal task is created."""
    if args.batch_size != 16:
        raise AssertionError("Spatial-Mamba v1 preflight requires frozen batch size 16")
    device = torch.device(args.device if args.device != "auto" else "cuda")
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = SpatialMambaClassifier(SpatialMambaConfig()).to(device)
        inputs = torch.zeros((16, 4, 1, 128, 501), dtype=torch.float32, device=device)
        targets = torch.arange(16, device=device, dtype=torch.long) % 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        optimizer.step()
        if tuple(logits.shape) != (16, 2) or not bool(torch.isfinite(loss).item()):
            raise AssertionError("CUDA batch-16 preflight produced invalid output")
        breakdown = parameter_breakdown(model)
        audit = {
            "status": "pass", "formal_training_started": False,
            "device": str(device), "batch_size": 16,
            "input_shape": [16, 4, 1, 128, 501], "logits_shape": [16, 2],
            "loss_finite": True, "backward_success": True, "optimizer_step_success": True,
            **breakdown,
            "transformer_reference_parameter_count": TRANSFORMER_REFERENCE_PARAMETER_COUNT,
            "parameter_delta_vs_transformer": (
                breakdown["total_parameter_count"] - TRANSFORMER_REFERENCE_PARAMETER_COUNT
            ),
        }
        del loss, logits, optimizer, targets, inputs, model
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA OOM during the mandatory batch-size-16 preflight. "
                "No formal task was started. Stop and report; do not automatically change batch size."
            ) from exc
        raise
    finally:
        torch.cuda.empty_cache()
    atomic_json(args.output_dir / "audit" / "cuda_batch16_preflight.json", audit)
    print(
        "PREFLIGHT PASS batch_size=16 "
        f"parameters={audit['total_parameter_count']} device={device}", flush=True,
    )
    return audit


def aggregate_outputs(
    args: argparse.Namespace, plan: pd.DataFrame, identity: dict[str, Any]
) -> None:
    run_fp = fingerprint(identity)
    # This call is deliberately strict and unconditional. Final tables cannot
    # inherit a task merely because its COMPLETE marker exists.
    per_fold, predictions, confusions, history = read_all_validated_tasks(args, plan, run_fp)
    per_fold = per_fold.sort_values(["session", "model", "seed", "fold"]).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["session", "model", "seed", "fold", "sample_index"]
    ).reset_index(drop=True)
    confusions = confusions.sort_values(
        ["session", "model", "seed", "fold", "true_label", "predicted_label"]
    ).reset_index(drop=True)
    history = history.sort_values(
        ["session", "model", "seed", "fold", "epoch"]
    ).reset_index(drop=True)
    atomic_csv(args.output_dir / "mamba_per_fold.csv", per_fold)
    atomic_csv(args.output_dir / "mamba_predictions.csv", predictions)
    atomic_csv(args.output_dir / "mamba_confusion_matrices.csv", confusions)
    atomic_csv(args.output_dir / "mamba_training_history.csv", history)

    seed_rows: list[dict[str, Any]] = []
    for (session, model, seed), group in predictions.groupby(
        ["session", "model", "seed"], sort=True
    ):
        source = per_fold[
            per_fold["session"].astype(str).eq(str(session))
            & per_fold["model"].eq(model)
            & per_fold["seed"].astype(int).eq(int(seed))
        ]
        if group["sample_index"].duplicated().any():
            raise AssertionError(f"duplicate OOF samples for {session} seed {seed}")
        expected_samples = int(source["n_samples"].iloc[0])
        if len(group) != expected_samples or set(group["sample_index"].astype(int)) != set(range(expected_samples)):
            raise AssertionError(f"incomplete OOF coverage for {session} seed {seed}")
        metrics = classification_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        seed_rows.append(
            {
                "session": str(session), "model": str(model), "seed": int(seed),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"]),
                "n_cycles": int(source["n_cycles"].iloc[0]), "n_samples": len(group),
                "n_folds": int(group["fold"].nunique()),
                "parameter_count": int(source["parameter_count"].iloc[0]),
                "cnn_stem_parameters": int(source["cnn_stem_parameters"].iloc[0]),
                "spatial_mamba_parameters": int(source["spatial_mamba_parameters"].iloc[0]),
                "temporal_transformer_parameters": int(source["temporal_transformer_parameters"].iloc[0]),
                "classifier_parameters": int(source["classifier_parameters"].iloc[0]),
                "transformer_reference_parameter_count": TRANSFORMER_REFERENCE_PARAMETER_COUNT,
                "parameter_delta_vs_transformer": int(source["parameter_delta_vs_transformer"].iloc[0]),
                "actual_batch_size": int(source["actual_batch_size"].iloc[0]),
            }
        )
    per_seed = pd.DataFrame(seed_rows).sort_values(["session", "model", "seed"])
    expected_seed_rows = len(EXPECTED_SESSIONS) * len(SEEDS)
    if len(per_seed) != expected_seed_rows:
        raise AssertionError(f"expected {expected_seed_rows} OOF seed rows, got {len(per_seed)}")
    atomic_csv(args.output_dir / "mamba_per_seed.csv", per_seed)
    summary = (
        per_seed.groupby(["session", "model"], as_index=False)
        .agg(
            mean_BA=("balanced_accuracy", "mean"),
            std_BA=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            n_cycles=("n_cycles", "first"), n_samples=("n_samples", "first"),
            parameter_count=("parameter_count", "first"),
            cnn_stem_parameters=("cnn_stem_parameters", "first"),
            spatial_mamba_parameters=("spatial_mamba_parameters", "first"),
            temporal_transformer_parameters=("temporal_transformer_parameters", "first"),
            classifier_parameters=("classifier_parameters", "first"),
            transformer_reference_parameter_count=("transformer_reference_parameter_count", "first"),
            parameter_delta_vs_transformer=("parameter_delta_vs_transformer", "first"),
        )
        .sort_values(["session", "model"])
    )
    required_summary_columns = [
        "session", "model", "mean_BA", "std_BA", "mean_accuracy",
        "n_cycles", "n_samples", "parameter_count", "cnn_stem_parameters",
        "spatial_mamba_parameters", "temporal_transformer_parameters",
        "classifier_parameters", "transformer_reference_parameter_count",
        "parameter_delta_vs_transformer",
    ]
    summary = summary[required_summary_columns]
    atomic_csv(args.output_dir / "mamba_summary.csv", summary)

    existing = load_existing_baselines(args)
    mamba_rows = summary.assign(
        model_display=MODEL_DISPLAY_NAME,
        n_seeds=len(SEEDS),
        source=str(args.output_dir / "mamba_summary.csv"),
    )[
        [
            "session", "model", "model_display", "mean_BA", "std_BA",
            "mean_accuracy", "n_seeds", "source", "parameter_count",
        ]
    ]
    comparison = pd.concat([existing, mamba_rows], ignore_index=True).sort_values(
        ["session", "model"]
    )
    expected_comparison_rows = len(EXPECTED_SESSIONS) * len(EXISTING_BASELINE_DISPLAY)
    if len(comparison) != expected_comparison_rows:
        raise AssertionError(
            f"baseline comparison expected {expected_comparison_rows} rows, got {len(comparison)}"
        )
    atomic_csv(args.output_dir / "mamba_vs_existing_baselines.csv", comparison)
    paired = paired_comparison_rows(comparison)
    atomic_csv(args.output_dir / "paired_comparisons.csv", paired)
    overfitting = build_overfitting_summary(history, per_seed)
    atomic_csv(args.output_dir / "overfitting_summary.csv", overfitting)
    atomic_text(args.output_dir / "mamba_report.md", build_report(comparison, paired, overfitting))


def run_full(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    require_mamba_dependency()
    if not torch.cuda.is_available():
        raise RuntimeError("Spatial-Mamba formal run requires CUDA, but torch.cuda.is_available() is False")
    if args.device != "auto" and not args.device.startswith("cuda"):
        raise RuntimeError("Spatial-Mamba formal run must use --device cuda or cuda:N")
    invalid_sessions = sorted(set(str(value) for value in args.sessions) - set(EXPECTED_SESSIONS))
    if invalid_sessions:
        raise ValueError(f"unknown sessions: {invalid_sessions}")
    run_fp = fingerprint(identity)
    run_cuda_batch16_preflight(args)
    plan = load_or_build_task_plan(args, identity)
    selected_sessions = [str(value) for value in args.sessions]
    selected = plan[plan["session"].astype(str).isin(selected_sessions)]
    status = update_status(args, plan, run_fp)
    completed = int(status["status"].eq("complete").sum())
    total = len(plan)
    for session in selected_sessions:
        session_plan = selected[selected["session"].astype(str).eq(session)]
        if session_plan.empty:
            continue
        # Fold identity is re-audited immediately before any session trains.
        data, splits = audit_session(args, session)
        for expected in session_plan.to_dict(orient="records"):
            path = task_dir(args.output_dir, session, int(expected["seed"]), int(expected["fold"]))
            valid, _ = validate_completed_task(path, expected, run_fp)
            if valid:
                print(
                    f"SKIP [{completed}/{total}] session={session} model={MODEL_NAME} "
                    f"fold={expected['fold']} seed={expected['seed']} device={args.device}", flush=True,
                )
                continue
            train_idx, test_idx = splits[int(expected["fold"]) - 1]
            print(
                f"RUN  [{completed}/{total}] session={session} model={MODEL_NAME} "
                f"fold={expected['fold']} seed={expected['seed']} device={args.device}", flush=True,
            )
            write_fold_task(args, identity, expected, data, train_idx, test_idx)
            completed += 1
            print(
                f"DONE [{completed}/{total}] session={session} model={MODEL_NAME} "
                f"fold={expected['fold']} seed={expected['seed']} device={args.device}", flush=True,
            )
        del data
    status = update_status(args, plan, run_fp)
    all_complete = bool(len(status) and status["status"].eq("complete").all())
    if not all_complete:
        print("PARTIAL RUN SAVED; rerun the same command to resume", flush=True)
        return
    aggregate_outputs(args, plan, identity)
    missing = [name for name in REQUIRED_FINAL_OUTPUTS if not (args.output_dir / name).exists()]
    if missing:
        raise AssertionError(f"finalization missing outputs: {missing}")
    atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete", "compute_environment": "server",
            "completed_utc": utc_now(), "run_fingerprint": run_fp,
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "runtime_environment_signature": identity["runtime_environment_signature"],
            "completed_tasks": len(plan), "total_tasks": len(plan),
            "required_outputs": list(REQUIRED_FINAL_OUTPUTS),
            "strict_task_revalidation_before_aggregation": True,
        },
    )
    print(f"FULL RUN COMPLETE: {args.output_dir / 'RUN_COMPLETE.json'}", flush=True)


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    if args.batch_size not in ALLOWED_BATCH_SIZES:
        raise ValueError(f"--batch-size must be one of {ALLOWED_BATCH_SIZES}")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    identity = run_identity(args.project_root, args.batch_size)
    write_run_metadata(args, identity)
    if args.stage == "sanity":
        run_sanity(args, identity)
    elif args.stage == "full":
        run_full(args, identity)
    else:
        plan_path = args.output_dir / "task_plan.csv"
        metadata_path = args.output_dir / "task_plan_metadata.json"
        if not plan_path.exists() or not metadata_path.exists():
            print(f"NOT STARTED: no formal task plan at {plan_path}")
            return
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("run_fingerprint") != fingerprint(identity):
            raise RuntimeError("task plan belongs to a different code/config fingerprint")
        plan = pd.read_csv(plan_path, dtype={"session": str})
        update_status(args, plan, fingerprint(identity))


if __name__ == "__main__":
    main()
