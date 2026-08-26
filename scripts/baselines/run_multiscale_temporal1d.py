#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
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
for search_path in (PROJECT_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.baselines import run_local_global_residual_mamba as prior_runner
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    cycle_text,
)
from ultrasound_decoding.multiframe.models import (
    CNN2DTemporal1D,
    count_trainable_parameters,
)
from ultrasound_decoding.multiframe.multiscale_temporal1d import (
    ENCODER_MODE_BY_MODEL_NAME,
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_TEMPORAL_BASELINE_NAME,
    MODEL_DISPLAY_NAMES,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    MODEL_NAMES,
    SINGLE_SCALE_MODEL_NAME,
    MultiScaleTemporal1DConfig,
    architecture_config,
    build_model,
    formal_temporal1d_audit,
    parameter_breakdown,
    train_fold,
)
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_sequence_tensor,
    resolve_device,
)


OUTPUT_VERSION = "multiscale_temporal1d_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
FORMAL_EPOCHS = 40
MAX_FOLDS = 10
ALLOWED_BATCH_SIZES = (16,)
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
NOTABLE_DECLINE_TOLERANCE = 0.02
EXTERNAL_BASELINES = (
    FORMAL_TEMPORAL_BASELINE_NAME,
    "fcnn_meanpool",
    "cnn_factorized_transformer",
    "spatial_mamba",
    "local_global_residual_mamba",
)
GATED_MAMBA_FORMAL_RUN_RELATIVE = Path(
    "outputs/local_global_residual_mamba_v1_1"
)
GATED_MAMBA_FORMAL_MODEL = "local_global_residual_mamba"
DISPLAY_NAMES = {
    **MODEL_DISPLAY_NAMES,
    FORMAL_TEMPORAL_BASELINE_NAME: "Existing formal Temporal 1D-CNN",
    "fcnn_meanpool": "FCNN mean-pool",
    "cnn_factorized_transformer": "Factorized Transformer",
    "spatial_mamba": "Spatial Mamba",
    "local_global_residual_mamba": "Gated Local+Global Mamba",
}
REQUIRED_FINAL_OUTPUTS = (
    "multiscale_summary.csv",
    "multiscale_per_seed.csv",
    "multiscale_per_fold.csv",
    "multiscale_predictions.csv",
    "multiscale_confusion_matrices.csv",
    "multiscale_training_history.csv",
    "parameter_count_audit.csv",
    "model_comparison.csv",
    "mechanistic_comparison.csv",
    "external_comparisons.csv",
    "paired_comparisons.csv",
    "strong_session_comparison.csv",
    "overfitting_comparison.csv",
    "decision_rule_audit.json",
    "multiscale_temporal1d_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen clean4 same-backbone single-scale vs lightweight multi-scale "
            "Spatial CNN + unchanged formal Temporal 1D-CNN."
        )
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
        default=(
            PROJECT_DIR
            / "results"
            / "runs"
            / "multiframe"
            / "block_clean4_binary_v1"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sanity-epochs", type=int, default=1)
    parser.add_argument(
        "--review-approved",
        action="store_true",
        help="Required for full stage after external code-review greenlight.",
    )
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
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git_text(project_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def distribution_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def runtime_environment_signature() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scipy_version": distribution_version("scipy"),
        "scikit_learn_version": distribution_version("scikit-learn"),
        "h5py_version": distribution_version("h5py"),
    }


def frozen_training_config(batch_size: int, epochs: int = FORMAL_EPOCHS) -> DeepTrainingConfig:
    return DeepTrainingConfig(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=int(batch_size),
        max_epochs=int(epochs),
        dropout=0.25,
        loss="cross_entropy",
    )


def frozen_experiment_config(batch_size: int) -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "claim": "lightweight multi-scale spatial receptive-field candidate",
        "sessions": list(EXPECTED_SESSIONS),
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "task": TASK_NAME,
        "class_mapping": TASK_CLASS_NAMES[TASK_NAME],
        "stimulus_blocks": ["grating", "dot"],
        "non_stimulus_blocks": ["stop_after_grating", "static"],
        "input_unit": "one formal clean4 block",
        "input_shape": list(EXPECTED_BLOCK_SHAPE),
        "complete_cycles_only": True,
        "cv": "exact formal clean4 cycle-grouped folds, max_folds=10",
        "normalization": "arcsinh_then_train_pixel_zscore; outer train fold only",
        "oof_primary_metric": "balanced_accuracy",
        "seeds": list(SEEDS),
        "mechanistic_models": list(MODEL_NAMES),
        "external_baselines_read_only": list(EXTERNAL_BASELINES),
        "training": frozen_training_config(batch_size).__dict__,
        "architectures": {
            model_name: architecture_config(model_name) for model_name in MODEL_NAMES
        },
        "formal_temporal1d_audit": formal_temporal1d_audit(),
        "epoch_selection": "fixed 40; no early stopping or test/validation selection",
        "test_used_for_training_or_tuning": False,
        "success_rule": {
            "multiscale_mean_BA_above_same_backbone_single_scale": True,
            "non_decreasing_sessions_vs_single_scale_at_least": 6,
            "strong_at_least_2_of_3_delta_ge_minus": NOTABLE_DECLINE_TOLERANCE,
            "mean_delta_vs_formal_temporal1d_at_least": 0.0,
            "severe_overfit_not_worse_than_complex_mamba": True,
        },
        "automatic_next_stage": False,
    }


def run_identity(project_root: Path, batch_size: int) -> dict[str, Any]:
    project_paths = [
        project_root / "src/ultrasound_decoding/multiframe/multiscale_temporal1d.py",
        project_root / "src/ultrasound_decoding/multiframe/models.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/cv.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "scripts/baselines/run_local_global_residual_mamba.py",
        project_root / "scripts/baselines/run_mamba_visual_binary.py",
        Path(__file__).resolve(),
    ]
    return {
        "experiment_config": frozen_experiment_config(batch_size),
        "runtime_environment_signature": runtime_environment_signature(),
        "git_commit": git_text(project_root, "rev-parse", "HEAD"),
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "project_source_sha256": {
            str(path.relative_to(project_root)): file_sha256(path) for path in project_paths
        },
    }


def write_run_metadata(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        observed = json.loads(config_path.read_text(encoding="utf-8"))
        if fingerprint(observed) != fingerprint(identity):
            formal_exists = (args.output_dir / "task_plan.csv").exists() or (
                args.output_dir / "tasks"
            ).exists()
            if formal_exists:
                raise RuntimeError(
                    "existing formal tasks have another code/config/environment fingerprint; "
                    "use a new output directory"
                )
    atomic_json(config_path, identity)
    atomic_json(
        args.output_dir / "environment.json",
        {
            "compute_environment": (
                "server" if str(args.device).startswith("cuda") else "local_sanity"
            ),
            "required_formal_environment": "/data2/yuq1ngr/conda_envs/fus",
            "runtime_environment_signature": runtime_environment_signature(),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
            "created_utc": utc_now(),
        },
    )
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


def audit_session(args: argparse.Namespace, session: str):
    return prior_runner.audit_session(args, session)


def task_dir(output_dir: Path, session: str, model: str, seed: int, fold: int) -> Path:
    return (
        output_dir
        / "tasks"
        / f"session_{session}"
        / model
        / f"seed_{seed}"
        / f"fold_{fold:02d}"
    )


def task_key(session: str, model: str, seed: int, fold: int) -> str:
    return f"{session}:{model}:{seed}:{fold}"


def task_fingerprint(run_fp: str, row: dict[str, Any]) -> str:
    return fingerprint(
        {
            "run_fingerprint": run_fp,
            "session": str(row["session"]),
            "model": str(row["model"]),
            "seed": int(row["seed"]),
            "fold": int(row["fold"]),
            "n_test_samples": int(row["n_test_samples"]),
            "config_fingerprint": str(row["config_fingerprint"]),
            "runtime_environment_fingerprint": str(
                row["runtime_environment_fingerprint"]
            ),
            "batch_size": int(row["batch_size"]),
        }
    )


def build_task_plan(args: argparse.Namespace, identity: dict[str, Any]) -> pd.DataFrame:
    run_fp = fingerprint(identity)
    config_fp = fingerprint(identity["experiment_config"])
    runtime_fp = fingerprint(identity["runtime_environment_signature"])
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
                "formal_clean4_fold_match": True,
                "cycle_overlap": False,
            }
        )
        for model_name in MODEL_NAMES:
            for seed in SEEDS:
                for fold, (_, test_idx) in enumerate(splits, start=1):
                    row = {
                        "session": session,
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "n_test_samples": len(test_idx),
                        "task_key": task_key(session, model_name, seed, fold),
                        "config_fingerprint": config_fp,
                        "runtime_environment_fingerprint": runtime_fp,
                        "batch_size": int(identity["experiment_config"]["training"]["batch_size"]),
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
            "total_tasks": len(plan),
            "task_definition": "session x two controlled spatial encoders x seed x fold",
            "external_baselines_retrained": False,
            "created_utc": utc_now(),
        },
    )
    return plan


def load_or_build_task_plan(args: argparse.Namespace, identity: dict[str, Any]) -> pd.DataFrame:
    plan_path = args.output_dir / "task_plan.csv"
    metadata_path = args.output_dir / "task_plan_metadata.json"
    run_fp = fingerprint(identity)
    if plan_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != run_fp:
            raise RuntimeError("existing task plan has a different run fingerprint")
        plan = pd.read_csv(plan_path, dtype={"session": str})
        for row in plan.to_dict(orient="records"):
            if row.get("task_fingerprint") != task_fingerprint(run_fp, row):
                raise AssertionError("task plan contains an invalid task fingerprint")
        return plan
    return build_task_plan(args, identity)


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
        model_config = json.loads((path / "model_config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")
    model_name = str(expected["model"])
    expected_fp = task_fingerprint(run_fp, expected)
    expected_identity = (
        str(expected["session"]),
        model_name,
        int(expected["seed"]),
        int(expected["fold"]),
    )
    if model_name not in MODEL_NAMES:
        return fail("unexpected model")
    if complete.get("task_key") != task_key(*expected_identity):
        return fail("task key mismatch")
    for name, payload in (("complete", complete), ("result", result)):
        if payload.get("run_fingerprint") != run_fp:
            return fail(f"{name} run fingerprint mismatch")
        if payload.get("task_fingerprint") != expected_fp:
            return fail(f"{name} task fingerprint mismatch")
        if payload.get("config_fingerprint") != str(expected["config_fingerprint"]):
            return fail(f"{name} config fingerprint mismatch")
        if payload.get("runtime_environment_fingerprint") != str(
            expected["runtime_environment_fingerprint"]
        ):
            return fail(f"{name} runtime fingerprint mismatch")
    observed_identity = (
        str(result.get("session")),
        str(result.get("model")),
        int(result.get("seed", -1)),
        int(result.get("fold", -1)),
    )
    if observed_identity != expected_identity:
        return fail("result identity mismatch")
    if result.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        return fail("implementation version mismatch")
    expected_architecture = architecture_config(model_name)
    observed_architecture = {
        key: value for key, value in model_config.items() if key != "parameter_breakdown"
    }
    if fingerprint(expected_architecture) != fingerprint(observed_architecture):
        return fail("architecture config mismatch")
    breakdown = model_config.get("parameter_breakdown", {})
    if int(breakdown.get("total_parameter_count", -1)) != EXPECTED_PARAMETER_COUNTS[
        model_name
    ]:
        return fail("parameter count differs from frozen architecture")
    if int(result.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNTS[model_name]:
        return fail("result parameter count mismatch")
    if int(result.get("actual_batch_size", -1)) != int(expected["batch_size"]):
        return fail("batch size mismatch")
    if bool(normalization.get("target_used_for_stats", True)):
        return fail("test fold used for normalization")
    if normalization.get("phase") != "outer_train_fold_only":
        return fail("normalization is not outer_train_fold_only")
    normalization_identity = (
        str(normalization.get("session")),
        str(normalization.get("method")),
        int(normalization.get("seed", -1)),
        int(normalization.get("fold", -1)),
    )
    if normalization_identity != expected_identity:
        return fail("normalization identity mismatch")
    expected_n = int(expected["n_test_samples"])
    if len(predictions) != expected_n or int(result.get("n_test_samples", -1)) != expected_n:
        return fail("prediction count mismatch")
    if not (
        predictions["session"].eq(expected_identity[0]).all()
        and predictions["model"].eq(model_name).all()
        and pd.to_numeric(predictions["seed"]).eq(expected_identity[2]).all()
        and pd.to_numeric(predictions["fold"]).eq(expected_identity[3]).all()
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
        if not np.isclose(float(result.get(metric, np.nan)), float(metrics[metric]), atol=1e-12):
            return fail(f"stored {metric} differs from predictions")
    confusion_columns = {
        "session",
        "model",
        "seed",
        "fold",
        "true_label",
        "predicted_label",
        "count",
    }
    if (
        not confusion_columns.issubset(confusion.columns)
        or len(confusion) != 4
        or int(confusion["count"].sum()) != expected_n
    ):
        return fail("confusion matrix invalid")
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
    history_columns = {
        "session",
        "model",
        "seed",
        "fold",
        "epoch",
        "train_loss",
        "train_accuracy",
    }
    if not history_columns.issubset(history.columns):
        return fail("training history columns missing")
    if len(history) != FORMAL_EPOCHS or not np.array_equal(
        history["epoch"].to_numpy(int), np.arange(1, FORMAL_EPOCHS + 1)
    ):
        return fail("training history is not exactly 40 epochs")
    if not (
        history["session"].eq(expected_identity[0]).all()
        and history["model"].eq(model_name).all()
        and pd.to_numeric(history["seed"]).eq(expected_identity[2]).all()
        and pd.to_numeric(history["fold"]).eq(expected_identity[3]).all()
    ):
        return fail("training history identity mismatch")
    if not np.isfinite(history[["train_loss", "train_accuracy"]].to_numpy(float)).all():
        return fail("training history contains non-finite values")
    if int(result.get("trained_epochs", -1)) != FORMAL_EPOCHS:
        return fail("trained epoch count mismatch")
    return True, "validated"


def update_status(args: argparse.Namespace, plan: pd.DataFrame, run_fp: str) -> pd.DataFrame:
    rows = []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir,
            str(expected["session"]),
            str(expected["model"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        valid, reason = validate_completed_task(path, expected, run_fp)
        rows.append(
            {
                "session": str(expected["session"]),
                "model": str(expected["model"]),
                "seed": int(expected["seed"]),
                "fold": int(expected["fold"]),
                "status": "complete" if valid else "pending",
                "validation": reason,
                "task_dir": str(path),
            }
        )
    status = pd.DataFrame(rows)
    atomic_csv(args.output_dir / "run_status.csv", status)
    completed = int(status["status"].eq("complete").sum())
    print(
        f"STATUS completed={completed} pending={len(status)-completed} total={len(status)}",
        flush=True,
    )
    return status


def select_balanced_indices(y: np.ndarray, candidates: np.ndarray, per_class: int) -> np.ndarray:
    return prior_runner.select_balanced_indices(y, candidates, per_class)


def run_sanity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if args.workers != 0:
        raise ValueError("sanity requires --workers 0")
    if args.sanity_epochs not in (1, 2):
        raise ValueError("sanity is restricted to 1 or 2 epochs")
    data, splits = audit_session(args, "710")
    train_idx, test_idx = splits[0]
    tiny_train = select_balanced_indices(data.y, train_idx, per_class=2)
    tiny_test = select_balanced_indices(data.y, test_idx, per_class=1)
    sanity_device = resolve_device(args.device)
    x = blocks_to_sequence_tensor(data.X[tiny_train[:2]]).to(sanity_device)
    y = torch.from_numpy(data.y[tiny_train[:2]].astype(np.int64)).to(sanity_device)
    audits = []
    for model_name in MODEL_NAMES:
        model = build_model(model_name).to(sanity_device)
        logits, shapes = model.forward_with_shapes(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        if tuple(logits.shape) != (2, 2) or not bool(torch.isfinite(loss).item()):
            raise AssertionError(f"{model_name}: invalid sanity forward/backward")
        result = train_fold(
            data.X[tiny_train],
            data.y[tiny_train],
            data.X[tiny_test],
            np.asarray([0, 1], dtype=np.int64),
            session="710",
            fold=1,
            seed=0,
            train_cycles=cycle_text(data.groups[tiny_train]),
            test_cycles=cycle_text(data.groups[tiny_test]),
            training_config=frozen_training_config(
                min(4, len(tiny_train)), args.sanity_epochs
            ),
            model_name=model_name,
            device=str(sanity_device),
            workers=0,
        )
        if len(result.history) != args.sanity_epochs:
            raise AssertionError("sanity epoch count mismatch")
        audits.append(
            {
                "model": model_name,
                "shapes": {key: list(value) for key, value in shapes.items()},
                "parameters": parameter_breakdown(model),
                "normalization_target_used_for_stats": False,
                "debug_only_not_formal": True,
            }
        )
        del result, model, logits, loss
    atomic_json(
        args.output_dir / "sanity" / "sanity_audit.json",
        {
            "session": "710",
            "fold": 1,
            "seed": 0,
            "models": audits,
            "formal_clean4_fold_match": True,
            "cycle_overlap": False,
            "formal_results": False,
        },
    )
    atomic_json(
        args.output_dir / "sanity" / "SANITY_COMPLETE.json",
        {"run_fingerprint": fingerprint(identity), "checks_passed": True},
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
    run_fp = fingerprint(identity)
    session = str(expected["session"])
    model_name = str(expected["model"])
    seed = int(expected["seed"])
    fold = int(expected["fold"])
    path = task_dir(args.output_dir, session, model_name, seed, fold)
    trained = train_fold(
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        np.asarray([0, 1], dtype=np.int64),
        session=session,
        fold=fold,
        seed=seed,
        train_cycles=cycle_text(data.groups[train_idx]),
        test_cycles=cycle_text(data.groups[test_idx]),
        training_config=DeepTrainingConfig(**identity["experiment_config"]["training"]),
        model_name=model_name,
        device=args.device,
        workers=args.workers,
    )
    metrics = classification_metrics(data.y[test_idx], trained.predictions)
    prediction_rows = []
    for local_index, sample_index in enumerate(test_idx):
        metadata = data.metadata.iloc[int(sample_index)]
        prediction_rows.append(
            {
                "session": session,
                "model": model_name,
                "seed": seed,
                "fold": fold,
                "sample_index": int(sample_index),
                "block_id": str(metadata["block_id"]),
                "cycle": int(data.groups[sample_index]),
                "block_name": str(metadata["block_name"]),
                "y_true": int(data.y[sample_index]),
                "y_pred": int(trained.predictions[local_index]),
                "probability_0": float(trained.probabilities[local_index, 0]),
                "probability_1": float(trained.probabilities[local_index, 1]),
            }
        )
    cm = confusion_matrix(
        data.y[test_idx], trained.predictions, np.asarray([0, 1], dtype=np.int64)
    )
    confusion_rows = [
        {
            "session": session,
            "model": model_name,
            "seed": seed,
            "fold": fold,
            "true_label": truth,
            "predicted_label": prediction,
            "count": int(cm[truth, prediction]),
        }
        for truth in (0, 1)
        for prediction in (0, 1)
    ]
    history = pd.DataFrame(trained.history)
    for column, value in reversed(
        (("session", session), ("model", model_name), ("seed", seed), ("fold", fold))
    ):
        history.insert(0, column, value)
    task_fp = task_fingerprint(run_fp, expected)
    shared = {
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": str(expected["config_fingerprint"]),
        "runtime_environment_fingerprint": str(expected["runtime_environment_fingerprint"]),
    }
    result = {
        **shared,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "session": session,
        "model": model_name,
        "seed": seed,
        "fold": fold,
        "n_cycles": data.n_cycles,
        "n_samples": data.n_blocks,
        "n_train_samples": len(train_idx),
        "n_test_samples": len(test_idx),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "parameter_count": int(trained.model_parameters),
        "actual_batch_size": int(identity["experiment_config"]["training"]["batch_size"]),
        "final_training_loss": float(trained.final_training_loss),
        "trained_epochs": int(trained.final_trained_epochs),
        "device": trained.device,
    }
    atomic_json(path / "result.json", result)
    atomic_csv(path / "predictions.csv", pd.DataFrame(prediction_rows))
    atomic_csv(path / "confusion_matrix.csv", pd.DataFrame(confusion_rows))
    atomic_csv(path / "training_history.csv", history)
    atomic_json(path / "normalization_audit.json", trained.normalization_audit)
    atomic_json(path / "model_config.json", trained.model_config)
    atomic_json(
        path / "COMPLETE.json",
        {
            **shared,
            "task_key": task_key(session, model_name, seed, fold),
            "completed_utc": utc_now(),
        },
    )
    validate_completed_task(path, expected, run_fp, raise_on_error=True)


def read_all_tasks(args: argparse.Namespace, plan: pd.DataFrame, run_fp: str):
    results, predictions, confusions, histories = [], [], [], []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir,
            str(expected["session"]),
            str(expected["model"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        validate_completed_task(path, expected, run_fp, raise_on_error=True)
        results.append(json.loads((path / "result.json").read_text(encoding="utf-8")))
        predictions.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        confusions.append(pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str}))
        histories.append(pd.read_csv(path / "training_history.csv", dtype={"session": str}))
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True),
        pd.concat(confusions, ignore_index=True),
        pd.concat(histories, ignore_index=True),
    )


def _first_existing(paths: list[Path], description: str) -> Path:
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"missing formal {description}; checked={paths}")
    return path


def validate_gated_mamba_formal_run(
    args: argparse.Namespace,
) -> tuple[Path, dict[str, Any]]:
    """Resolve the completed v1.1 formal run without falling back to legacy v1."""
    run_dir = args.project_root / GATED_MAMBA_FORMAL_RUN_RELATIVE
    completion_path = run_dir / "RUN_COMPLETE.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            "formal Gated Mamba v1.1 requires "
            f"{completion_path}; legacy local_global_residual_mamba_v1 is not an "
            "automatic fallback"
        )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssertionError(
            f"invalid formal Gated Mamba completion manifest: {completion_path}"
        ) from exc
    if not isinstance(completion, dict) or completion.get("status") != "complete":
        raise AssertionError(
            "formal Gated Mamba v1.1 RUN_COMPLETE.json must have status='complete': "
            f"{completion_path}"
        )
    return run_dir, completion


def load_external_baselines(args: argparse.Namespace) -> pd.DataFrame:
    clean = prior_runner.load_clean4_formal_seed_records(args)
    rows = []
    for model_name in (FORMAL_TEMPORAL_BASELINE_NAME, "fcnn_meanpool"):
        selected = clean[clean["method"].eq(model_name)]
        for session, group in selected.groupby("session", sort=True):
            rows.append(
                {
                    "session": str(session),
                    "model": model_name,
                    "model_display": DISPLAY_NAMES[model_name],
                    "mean_BA": float(group["balanced_accuracy"].mean()),
                    "std_BA": float(group["balanced_accuracy"].std(ddof=1)),
                    "mean_accuracy": float(group["accuracy"].mean()),
                    "n_seeds": 3,
                    "parameter_count": (
                        EXPECTED_PARAMETER_COUNTS[FORMAL_TEMPORAL_BASELINE_NAME]
                        if model_name == FORMAL_TEMPORAL_BASELINE_NAME
                        else np.nan
                    ),
                    "source": canonical_json(group["source"].astype(str).tolist()),
                    "retrained_by_this_runner": False,
                }
            )
    gated_run_dir, gated_completion = validate_gated_mamba_formal_run(args)
    gated_completion_path = gated_run_dir / "RUN_COMPLETE.json"
    gated_summary_path = gated_run_dir / "proposed_summary.csv"
    if not gated_summary_path.is_file():
        raise FileNotFoundError(
            f"completed formal Gated Mamba v1.1 summary is missing: {gated_summary_path}"
        )
    summary_specs = (
        (
            "spatial_mamba",
            _first_existing(prior_runner.mamba_summary_candidates(args), "Spatial Mamba summary"),
            "mamba_summary.csv",
        ),
        (
            "cnn_factorized_transformer",
            _first_existing(
                prior_runner.transformer_summary_candidates(args),
                "Factorized Transformer summary",
            ),
            "transformer_summary.csv",
        ),
        (
            GATED_MAMBA_FORMAL_MODEL,
            gated_summary_path,
            "proposed_summary.csv",
        ),
    )
    for model_name, path, _ in summary_specs:
        frame = pd.read_csv(path, dtype={"session": str})
        required = {"session", "model", "mean_BA", "std_BA", "mean_accuracy"}
        if not required.issubset(frame.columns):
            raise AssertionError(f"{path} lacks required formal summary columns")
        selected = frame[frame["model"].astype(str).eq(model_name)]
        if model_name == GATED_MAMBA_FORMAL_MODEL:
            observed_sessions = selected["session"].astype(str).tolist()
            if (
                len(observed_sessions) != len(EXPECTED_SESSIONS)
                or set(observed_sessions) != set(EXPECTED_SESSIONS)
            ):
                raise AssertionError(
                    "formal Gated Mamba v1.1 summary must contain exactly one row for "
                    f"each expected session; observed={observed_sessions}, "
                    f"expected={list(EXPECTED_SESSIONS)}"
                )
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "session": str(row.session),
                    "model": model_name,
                    "model_display": DISPLAY_NAMES[model_name],
                    "mean_BA": float(row.mean_BA),
                    "std_BA": float(row.std_BA),
                    "mean_accuracy": float(row.mean_accuracy),
                    "n_seeds": 3,
                    "parameter_count": (
                        float(row.parameter_count)
                        if hasattr(row, "parameter_count")
                        else np.nan
                    ),
                    "source": str(path),
                    "source_run_complete": (
                        str(gated_completion_path)
                        if model_name == GATED_MAMBA_FORMAL_MODEL
                        else np.nan
                    ),
                    "source_run_status": (
                        str(gated_completion["status"])
                        if model_name == GATED_MAMBA_FORMAL_MODEL
                        else np.nan
                    ),
                    "retrained_by_this_runner": False,
                }
            )
    result = pd.DataFrame(rows)
    expected = {
        (session, model) for session in EXPECTED_SESSIONS for model in EXTERNAL_BASELINES
    }
    observed = set(zip(result["session"], result["model"]))
    if observed != expected:
        raise AssertionError(
            f"external baseline coverage mismatch: missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )
    return result.sort_values(["session", "model"]).reset_index(drop=True)


def exact_two_sided_sign_flip(values: np.ndarray) -> float:
    return prior_runner.exact_two_sided_sign_flip(np.asarray(values, dtype=float))


def pairwise_rows(
    comparison: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...],
    comparison_type: str,
) -> pd.DataFrame:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    rows = []
    for candidate, baseline in pairs:
        deltas = (
            pivot.loc[list(EXPECTED_SESSIONS), candidate]
            - pivot.loc[list(EXPECTED_SESSIONS), baseline]
        )
        strong = deltas.loc[list(STRONG_SESSIONS)]
        weak = deltas.loc[list(WEAK_SESSIONS)]
        tolerance = 1e-12
        rows.append(
            {
                "comparison_type": comparison_type,
                "comparison": f"{candidate}_vs_{baseline}",
                "candidate": candidate,
                "baseline": baseline,
                "mean_delta_BA": float(deltas.mean()),
                "median_delta_BA": float(deltas.median()),
                "improved_sessions": int((deltas > tolerance).sum()),
                "tied_sessions": int((deltas.abs() <= tolerance).sum()),
                "worsened_sessions": int((deltas < -tolerance).sum()),
                "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(deltas.to_numpy()),
                "strong_session_mean_delta_BA": float(strong.mean()),
                "weak_session_mean_delta_BA": float(weak.mean()),
                "session_deltas_json": canonical_json(deltas.to_dict()),
            }
        )
    return pd.DataFrame(rows)


def build_overfit(history: pd.DataFrame, per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (session, model, seed), group in history.groupby(
        ["session", "model", "seed"], sort=True
    ):
        by_epoch = group.groupby("epoch")["train_accuracy"].mean().sort_index()
        oof = per_seed[
            per_seed["session"].eq(str(session))
            & per_seed["model"].eq(str(model))
            & per_seed["seed"].eq(int(seed))
        ]
        if len(by_epoch) != FORMAL_EPOCHS or len(oof) != 1:
            raise AssertionError("overfit audit source is incomplete")
        best_train = float(by_epoch.max())
        oof_ba = float(oof.iloc[0]["balanced_accuracy"])
        rows.append(
            {
                "session": str(session),
                "model": str(model),
                "seed": int(seed),
                "final_train_accuracy": float(by_epoch.iloc[-1]),
                "best_train_accuracy": best_train,
                "OOF_test_BA": oof_ba,
                "generalization_gap": best_train - oof_ba,
                "possible_severe_overfit": bool(best_train >= 0.95 and oof_ba <= 0.60),
            }
        )
    return pd.DataFrame(rows)


def load_complex_overfit(args: argparse.Namespace) -> pd.DataFrame:
    gated_run_dir, gated_completion = validate_gated_mamba_formal_run(args)
    gated_completion_path = gated_run_dir / "RUN_COMPLETE.json"
    mamba_path = gated_run_dir / "overfitting_comparison.csv"
    if not mamba_path.is_file():
        raise FileNotFoundError(
            "completed formal Gated Mamba v1.1 overfitting comparison is missing: "
            f"{mamba_path}"
        )
    mamba = pd.read_csv(mamba_path, dtype={"session": str})
    mamba = mamba[
        mamba["model"].isin(["spatial_mamba", "local_global_residual_mamba"])
    ].copy()
    mamba["source"] = str(mamba_path)
    mamba["source_run_complete"] = str(gated_completion_path)
    mamba["source_run_status"] = str(gated_completion["status"])
    transformer_path = _first_existing(
        [
            args.project_root
            / "outputs/transformer_visual_binary_v1/overfitting_summary.csv",
            args.project_root
            / "outputs/transformer_visual_binary_v1/transformer_visual_binary_v1/overfitting_summary.csv",
        ],
        "Factorized Transformer overfitting summary",
    )
    transformer = pd.read_csv(transformer_path, dtype={"session": str})
    transformer = transformer[
        transformer["model"].astype(str).eq("cnn_factorized_transformer")
    ].copy()
    transformer["source"] = str(transformer_path)
    selected = pd.concat([mamba, transformer], ignore_index=True)
    if len(selected) != len(EXPECTED_SESSIONS) * len(SEEDS) * 3:
        raise AssertionError("complex-model overfit audit is incomplete")
    return selected


def decision_rule_audit(
    comparison: pd.DataFrame, overfitting: pd.DataFrame, paired: pd.DataFrame
) -> dict[str, Any]:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    single_delta = pivot[MODEL_NAME] - pivot[SINGLE_SCALE_MODEL_NAME]
    formal_delta = pivot[MODEL_NAME] - pivot[FORMAL_TEMPORAL_BASELINE_NAME]
    strong = single_delta.loc[list(STRONG_SESSIONS)]
    severe = overfitting.assign(
        possible_severe_overfit=overfitting["possible_severe_overfit"].map(
            lambda value: value
            if isinstance(value, (bool, np.bool_))
            else str(value).lower() == "true"
        )
    ).groupby(["model", "session"])["possible_severe_overfit"].any()
    counts = {
        model: int(severe.loc[model].sum())
        for model in (
            MODEL_NAME,
            "cnn_factorized_transformer",
            "spatial_mamba",
            "local_global_residual_mamba",
        )
    }
    checks = {
        "multiscale_mean_BA_above_same_backbone_single_scale": bool(
            single_delta.mean() > 0
        ),
        "at_least_6_of_9_non_decreasing_vs_single_scale": bool(
            int((single_delta >= -1e-12).sum()) >= 6
        ),
        "strong_at_least_2_of_3_not_notably_down_vs_single_scale": bool(
            int((strong >= -NOTABLE_DECLINE_TOLERANCE).sum()) >= 2
        ),
        "mean_BA_not_below_formal_temporal1d": bool(formal_delta.mean() >= -1e-12),
        "severe_overfit_not_worse_than_transformer_spatial_or_gated_mamba": bool(
            counts[MODEL_NAME]
            <= min(
                counts["cnn_factorized_transformer"],
                counts["spatial_mamba"],
                counts["local_global_residual_mamba"],
            )
        ),
    }
    supports = all(checks.values())
    return {
        "criteria_frozen_before_formal_results": True,
        "notable_decline_tolerance_BA": NOTABLE_DECLINE_TOLERANCE,
        "checks": checks,
        "severe_overfit_session_counts": counts,
        "decision": (
            "supports_continue_multiscale_route_after_manual_review"
            if supports
            else "does_not_support_more_cnn_complexity"
        ),
        "automatic_next_stage_started": False,
        "paired_table_fingerprint": fingerprint(paired.to_dict(orient="records")),
    }


def build_report(
    comparison: pd.DataFrame,
    mechanistic: pd.DataFrame,
    external: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    lines = [
        "# Lightweight Multi-scale Spatial CNN + Temporal 1D-CNN v1",
        "",
        "compute_environment = server",
        "",
        "本轮只改变第二空间编码阶段的感受野；clean4、fold、normalization、Temporal1D、classifier 和训练协议均冻结。",
        "",
        "## Mechanistic comparison",
        "",
    ]
    for row in mechanistic.itertuples(index=False):
        lines.append(
            f"- Multi-scale vs same-backbone single-scale: mean ΔBA={row.mean_delta_BA:+.4f}, "
            f"median={row.median_delta_BA:+.4f}, improved/tied/worsened="
            f"{row.improved_sessions}/{row.tied_sessions}/{row.worsened_sessions}, "
            f"exact p={row.exact_two_sided_sign_flip_p:.4f}, strong Δ="
            f"{row.strong_session_mean_delta_BA:+.4f}, weak Δ={row.weak_session_mean_delta_BA:+.4f}."
        )
    lines.extend(["", "## External comparisons", ""])
    for row in external.itertuples(index=False):
        lines.append(
            f"- Multi-scale vs {DISPLAY_NAMES[row.baseline]}: mean ΔBA="
            f"{row.mean_delta_BA:+.4f}, median={row.median_delta_BA:+.4f}, "
            f"improved/tied/worsened={row.improved_sessions}/{row.tied_sessions}/"
            f"{row.worsened_sessions}, exact p={row.exact_two_sided_sign_flip_p:.4f}, "
            f"strong Δ={row.strong_session_mean_delta_BA:+.4f}, "
            f"weak Δ={row.weak_session_mean_delta_BA:+.4f}."
        )
    lines.extend(["", "## Strong sessions", ""])
    for session in STRONG_SESSIONS:
        lines.append(
            f"- {session}: formal Temporal1D={pivot.loc[session, FORMAL_TEMPORAL_BASELINE_NAME]:.4f}, "
            f"single-scale={pivot.loc[session, SINGLE_SCALE_MODEL_NAME]:.4f}, "
            f"multi-scale={pivot.loc[session, MODEL_NAME]:.4f}."
        )
    lines.extend(
        [
            "",
            "## Pre-registered decision",
            "",
            f"Decision: `{decision['decision']}`.",
            "",
            "本 runner 到此停止；不会自动增加 dilation、kernel、分支、attention、ROI 或其他结构。",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_outputs(args: argparse.Namespace, plan: pd.DataFrame, identity: dict[str, Any]) -> None:
    run_fp = fingerprint(identity)
    per_fold, predictions, confusions, history = read_all_tasks(args, plan, run_fp)
    per_fold = per_fold.sort_values(["session", "model", "seed", "fold"])
    predictions = predictions.sort_values(["session", "model", "seed", "fold", "sample_index"])
    confusions = confusions.sort_values(
        ["session", "model", "seed", "fold", "true_label", "predicted_label"]
    )
    history = history.sort_values(["session", "model", "seed", "fold", "epoch"])
    atomic_csv(args.output_dir / "multiscale_per_fold.csv", per_fold)
    atomic_csv(args.output_dir / "multiscale_predictions.csv", predictions)
    atomic_csv(args.output_dir / "multiscale_confusion_matrices.csv", confusions)
    atomic_csv(args.output_dir / "multiscale_training_history.csv", history)
    seed_rows = []
    for (session, model, seed), group in predictions.groupby(
        ["session", "model", "seed"], sort=True
    ):
        source = per_fold[
            per_fold["session"].eq(str(session))
            & per_fold["model"].eq(str(model))
            & per_fold["seed"].eq(int(seed))
        ]
        expected_n = int(source.iloc[0]["n_samples"])
        if group["sample_index"].duplicated().any() or set(group["sample_index"]) != set(
            range(expected_n)
        ):
            raise AssertionError("OOF coverage is incomplete or duplicated")
        metrics = classification_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        seed_rows.append(
            {
                "session": str(session),
                "model": str(model),
                "seed": int(seed),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "n_cycles": int(source.iloc[0]["n_cycles"]),
                "n_samples": expected_n,
                "n_folds": int(group["fold"].nunique()),
                "parameter_count": int(source.iloc[0]["parameter_count"]),
            }
        )
    per_seed = pd.DataFrame(seed_rows).sort_values(["session", "model", "seed"])
    if len(per_seed) != len(EXPECTED_SESSIONS) * len(MODEL_NAMES) * len(SEEDS):
        raise AssertionError("OOF per-seed coverage is incomplete")
    atomic_csv(args.output_dir / "multiscale_per_seed.csv", per_seed)
    summary = (
        per_seed.groupby(["session", "model"], as_index=False)
        .agg(
            mean_BA=("balanced_accuracy", "mean"),
            std_BA=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            n_cycles=("n_cycles", "first"),
            n_samples=("n_samples", "first"),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values(["session", "model"])
    )
    atomic_csv(args.output_dir / "multiscale_summary.csv", summary)
    parameter_rows = [
        {
            "model": model_name,
            "parameter_count": EXPECTED_PARAMETER_COUNTS[model_name],
            "source": "trained_same_backbone_model",
        }
        for model_name in MODEL_NAMES
    ] + [
        {
            "model": FORMAL_TEMPORAL_BASELINE_NAME,
            "parameter_count": EXPECTED_PARAMETER_COUNTS[FORMAL_TEMPORAL_BASELINE_NAME],
            "source": "formal_source_code_audit",
        }
    ]
    atomic_csv(args.output_dir / "parameter_count_audit.csv", pd.DataFrame(parameter_rows))
    external_baselines = load_external_baselines(args)
    proposed = summary.assign(
        model_display=summary["model"].map(DISPLAY_NAMES),
        n_seeds=3,
        source=str(args.output_dir / "multiscale_summary.csv"),
        retrained_by_this_runner=True,
    )
    comparison = pd.concat([external_baselines, proposed], ignore_index=True).sort_values(
        ["session", "model"]
    )
    if len(comparison) != len(EXPECTED_SESSIONS) * (
        len(EXTERNAL_BASELINES) + len(MODEL_NAMES)
    ):
        raise AssertionError("comparison coverage is incomplete")
    atomic_csv(args.output_dir / "model_comparison.csv", comparison)
    mechanistic = pairwise_rows(
        comparison,
        ((MODEL_NAME, SINGLE_SCALE_MODEL_NAME),),
        "mechanistic_same_backbone",
    )
    external = pairwise_rows(
        comparison,
        tuple((MODEL_NAME, baseline) for baseline in EXTERNAL_BASELINES),
        "external_baseline",
    )
    paired = pd.concat([mechanistic, external], ignore_index=True)
    atomic_csv(args.output_dir / "mechanistic_comparison.csv", mechanistic)
    atomic_csv(args.output_dir / "external_comparisons.csv", external)
    atomic_csv(args.output_dir / "paired_comparisons.csv", paired)
    strong = comparison[
        comparison["session"].isin(STRONG_SESSIONS)
        & comparison["model"].isin(
            [FORMAL_TEMPORAL_BASELINE_NAME, SINGLE_SCALE_MODEL_NAME, MODEL_NAME]
        )
    ]
    if len(strong) != len(STRONG_SESSIONS) * 3:
        raise AssertionError("strong-session table is incomplete")
    atomic_csv(args.output_dir / "strong_session_comparison.csv", strong)
    proposed_overfit = build_overfit(history, per_seed)
    complex_overfit = load_complex_overfit(args)
    overfitting = pd.concat([proposed_overfit, complex_overfit], ignore_index=True)
    atomic_csv(args.output_dir / "overfitting_comparison.csv", overfitting)
    decision = decision_rule_audit(comparison, overfitting, paired)
    atomic_json(args.output_dir / "decision_rule_audit.json", decision)
    atomic_text(
        args.output_dir / "multiscale_temporal1d_report.md",
        build_report(comparison, mechanistic, external, decision),
    )


def run_cuda_preflight(args: argparse.Namespace) -> None:
    if args.batch_size != 16:
        raise AssertionError("formal protocol requires batch size 16")
    device = torch.device(args.device if args.device != "auto" else "cuda")
    audits = []
    for model_name in MODEL_NAMES:
        model = build_model(model_name).to(device)
        inputs = torch.zeros((16, 4, 1, 128, 501), device=device)
        targets = torch.arange(16, device=device) % 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in model.parameters()
        )
        optimizer.step()
        if (
            tuple(logits.shape) != (16, 2)
            or not bool(torch.isfinite(loss).item())
            or not gradients_finite
        ):
            raise AssertionError(f"{model_name}: invalid CUDA batch-16 preflight")
        breakdown = parameter_breakdown(model)
        if breakdown["total_parameter_count"] != EXPECTED_PARAMETER_COUNTS[model_name]:
            raise AssertionError("preflight parameter count mismatch")
        audits.append(
            {
                "model": model_name,
                "loss_finite": True,
                "gradients_finite": True,
                "optimizer_step_success": True,
                **breakdown,
            }
        )
        del model, inputs, targets, optimizer, logits, loss
        torch.cuda.empty_cache()
    atomic_json(
        args.output_dir / "audit" / "cuda_batch16_preflight.json",
        {"formal_training_started": False, "device": str(device), "models": audits},
    )


def run_full(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if not args.review_approved:
        raise RuntimeError(
            "formal run is locked until external code review is approved; use "
            "--review-approved only after greenlight"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("formal run requires CUDA")
    if args.device != "auto" and not str(args.device).startswith("cuda"):
        raise RuntimeError("formal run requires CUDA device")
    expected_env = Path("/data2/yuq1ngr/conda_envs/fus")
    if expected_env not in Path(sys.executable).resolve().parents:
        raise RuntimeError(f"formal run must use {expected_env}; got {sys.executable}")
    invalid_sessions = sorted(set(map(str, args.sessions)) - set(EXPECTED_SESSIONS))
    if invalid_sessions:
        raise ValueError(f"unknown sessions: {invalid_sessions}")
    run_cuda_preflight(args)
    plan = load_or_build_task_plan(args, identity)
    run_fp = fingerprint(identity)
    selected_sessions = list(map(str, args.sessions))
    status = update_status(args, plan, run_fp)
    completed = int(status["status"].eq("complete").sum())
    total = len(plan)
    for session in selected_sessions:
        session_plan = plan[plan["session"].eq(session)]
        data, splits = audit_session(args, session)
        for expected in session_plan.to_dict(orient="records"):
            model_name = str(expected["model"])
            path = task_dir(
                args.output_dir,
                session,
                model_name,
                int(expected["seed"]),
                int(expected["fold"]),
            )
            valid, _ = validate_completed_task(path, expected, run_fp)
            if valid:
                print(
                    f"SKIP [{completed}/{total}] session={session} model={model_name} "
                    f"fold={expected['fold']} seed={expected['seed']} device={args.device}",
                    flush=True,
                )
                continue
            train_idx, test_idx = splits[int(expected["fold"]) - 1]
            print(
                f"RUN  [{completed}/{total}] session={session} model={model_name} "
                f"fold={expected['fold']} seed={expected['seed']} device={args.device}",
                flush=True,
            )
            write_fold_task(args, identity, expected, data, train_idx, test_idx)
            completed += 1
            print(
                f"DONE [{completed}/{total}] session={session} model={model_name} "
                f"fold={expected['fold']} seed={expected['seed']} device={args.device}",
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
    atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "compute_environment": "server",
            "run_fingerprint": run_fp,
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "completed_tasks": len(plan),
            "total_tasks": len(plan),
            "mechanistic_models_trained": list(MODEL_NAMES),
            "external_baselines_reused_read_only": list(EXTERNAL_BASELINES),
            "automatic_next_stage_started": False,
        },
    )
    print("FULL RUN COMPLETE; STOP for manual analysis", flush=True)


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
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != fingerprint(identity):
            raise RuntimeError("task plan belongs to another code/config/environment")
        plan = pd.read_csv(plan_path, dtype={"session": str})
        update_status(args, plan, fingerprint(identity))


if __name__ == "__main__":
    main()
