#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.baselines import run_mamba_visual_binary as audit_utils
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    cycle_text,
)
from ultrasound_decoding.multiframe.local_global_residual_mamba import (
    EXPECTED_FORMAL_PARAMETER_COUNT,
    INITIAL_GATE_LOGIT,
    LOCAL_BASELINE_NAME,
    MODEL_DISPLAY_NAME,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    LocalGlobalResidualMambaClassifier,
    LocalGlobalResidualMambaConfig,
    architecture_config,
    parameter_breakdown,
    train_local_global_residual_mamba_fold,
)
from ultrasound_decoding.multiframe.spatial_mamba import require_mamba_dependency
from ultrasound_decoding.multiframe.training import DeepTrainingConfig, blocks_to_sequence_tensor


OUTPUT_VERSION = "local_global_residual_mamba_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
MAX_FOLDS = 10
FORMAL_EPOCHS = 40
ALLOWED_BATCH_SIZES = (16,)
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
NOTABLE_DECLINE_TOLERANCE = 0.02
STRONG_MAMBA_RECOVERY_THRESHOLD = 0.02
COMPARISON_BASELINES = (
    LOCAL_BASELINE_NAME,
    "spatial_mamba",
    "cnn_factorized_transformer",
    "fcnn_meanpool",
)
DISPLAY_NAMES = {
    LOCAL_BASELINE_NAME: "Temporal 1D-CNN",
    "spatial_mamba": "Spatial Mamba",
    "cnn_factorized_transformer": "CNN Factorized Transformer",
    "fcnn_meanpool": "FCNN mean-pool",
    MODEL_NAME: MODEL_DISPLAY_NAME,
}
REQUIRED_FINAL_OUTPUTS = (
    "proposed_summary.csv",
    "proposed_per_seed.csv",
    "proposed_per_fold.csv",
    "proposed_predictions.csv",
    "proposed_confusion_matrices.csv",
    "proposed_training_history.csv",
    "gate_summary.csv",
    "model_comparison.csv",
    "paired_comparisons.csv",
    "strong_session_comparison.csv",
    "overfitting_comparison.csv",
    "decision_rule_audit.json",
    "proposed_v1_report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen proposed v1 clean4 validation: local CNN + gated residual "
            "Spatial Mamba + Temporal 1D-CNN."
        )
    )
    parser.add_argument("--stage", choices=("sanity", "full", "status"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sessions", nargs="+", default=list(EXPECTED_SESSIONS))
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / OUTPUT_VERSION,
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
    parser.add_argument("--sanity-epochs", type=int, default=2)
    parser.add_argument(
        "--review-approved",
        action="store_true",
        help="Required for --stage full after external code-review greenlight.",
    )
    return parser.parse_args()


utc_now = lambda: datetime.now(timezone.utc).isoformat()
canonical_json = audit_utils.canonical_json
fingerprint = audit_utils.fingerprint
file_sha256 = audit_utils.file_sha256
atomic_json = audit_utils.atomic_json
atomic_text = audit_utils.atomic_text
atomic_csv = audit_utils.atomic_csv
git_text = audit_utils.git_text
canonical_manifest = audit_utils.canonical_manifest
exact_two_sided_sign_flip = audit_utils.exact_two_sided_sign_flip


def runtime_environment_signature() -> dict[str, str]:
    signature = dict(audit_utils.runtime_environment_signature())
    signature["h5py_version"] = audit_utils.distribution_version("h5py")
    return signature


def environment_payload(device: str) -> dict[str, Any]:
    payload = dict(audit_utils.environment_payload(device))
    signature = runtime_environment_signature()
    payload.update(signature)
    payload["runtime_environment_signature"] = signature
    return payload


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
        "claim": "proposed-method first-round validation candidate",
        "sessions": list(EXPECTED_SESSIONS),
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
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
        "models": ["local_temporal_baseline_reused", MODEL_NAME],
        "local_temporal_baseline_source": (
            "read existing formal cnn2d_temporal1d results; never retrain"
        ),
        "training": frozen_training_config(batch_size).__dict__,
        "architecture": architecture_config(LocalGlobalResidualMambaConfig()),
        "epoch_selection": "fixed 40 epochs; no validation/test model selection",
        "test_used_for_training_or_tuning": False,
        "oom_policy": "batch size fixed at 16; stop before formal tasks on OOM",
        "comparison_baselines": list(COMPARISON_BASELINES),
        "success_rule": {
            "mean_BA_above_temporal1d": True,
            "at_least_6_of_9_non_decreasing_vs_temporal1d": True,
            "strong_at_least_2_of_3_delta_ge_minus": NOTABLE_DECLINE_TOLERANCE,
            "strong_mean_recovery_vs_spatial_mamba_at_least": (
                STRONG_MAMBA_RECOVERY_THRESHOLD
            ),
            "strong_at_least_2_of_3_improve_vs_spatial_mamba": True,
            "overfit_mitigation": (
                "proposed severe-overfit session count is lower than pure Spatial Mamba"
            ),
            "failure_if": (
                "mean not above Temporal1D OR all three strong sessions decline by >0.02 "
                "OR severe-overfit session count is not lower than pure Spatial Mamba"
            ),
        },
        "automatic_next_stage": False,
    }


def run_identity(project_root: Path, batch_size: int) -> dict[str, Any]:
    model_path = (
        project_root
        / "src"
        / "ultrasound_decoding"
        / "multiframe"
        / "local_global_residual_mamba.py"
    )
    runner_path = Path(__file__).resolve()
    transitive_paths = [
        project_root / "scripts" / "baselines" / "run_mamba_visual_binary.py",
        project_root / "src" / "ultrasound_decoding" / "multiframe" / "spatial_mamba.py",
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
        "model_source_sha256": file_sha256(model_path),
        "runner_source_sha256": file_sha256(runner_path),
        "transitive_project_source_sha256": {
            str(path.relative_to(project_root)): file_sha256(path)
            for path in transitive_paths
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
                    "existing formal tasks use another code/config/environment fingerprint; "
                    "use a new output directory"
                )
    atomic_json(config_path, identity)
    environment = environment_payload(args.device)
    environment["required_formal_environment"] = "/data2/yuq1ngr/conda_envs/fus_mamba"
    atomic_json(args.output_dir / "environment.json", environment)
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


def audit_session(
    args: argparse.Namespace, session: str
) -> tuple[Any, list[tuple[np.ndarray, np.ndarray]]]:
    # Reuse the already-reviewed exact formal clean4 audit and manifest matching.
    return audit_utils.audit_session(args, session)


def task_dir(output_dir: Path, session: str, seed: int, fold: int) -> Path:
    return (
        output_dir
        / "tasks"
        / f"session_{session}"
        / MODEL_NAME
        / f"seed_{seed}"
        / f"fold_{fold:02d}"
    )


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
                "input_shape": canonical_json(list(data.X.shape[1:])),
                "formal_clean4_fold_match": True,
                "cycle_overlap": False,
            }
        )
        for seed in SEEDS:
            for fold, (_, test_idx) in enumerate(splits, start=1):
                row = {
                    "session": session,
                    "model": MODEL_NAME,
                    "seed": seed,
                    "fold": fold,
                    "n_test_samples": len(test_idx),
                    "task_key": task_key(session, seed, fold),
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
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "runtime_environment_signature": identity["runtime_environment_signature"],
            "total_tasks": len(plan),
            "task_definition": "session x proposed model x seed x fold",
            "existing_temporal_baseline_retrained": False,
            "created_utc": utc_now(),
        },
    )
    return plan


def load_or_build_task_plan(
    args: argparse.Namespace, identity: dict[str, Any]
) -> pd.DataFrame:
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
    run_fingerprint: str,
    *,
    raise_on_error: bool = False,
) -> tuple[bool, str]:
    """Strictly revalidate a fold/seed; COMPLETE.json is never sufficient."""

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
        "gate.json",
    )
    missing = [
        name
        for name in required
        if not (path / name).exists() or (path / name).stat().st_size == 0
    ]
    if missing:
        return fail(f"missing/empty files {missing}")
    try:
        complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        confusion = pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str})
        history = pd.read_csv(path / "training_history.csv", dtype={"session": str})
        normalization = json.loads(
            (path / "normalization_audit.json").read_text(encoding="utf-8")
        )
        model_config = json.loads(
            (path / "model_config.json").read_text(encoding="utf-8")
        )
        gate = json.loads((path / "gate.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")

    expected_task_fp = task_fingerprint(run_fingerprint, expected)
    expected_key = task_key(
        str(expected["session"]), int(expected["seed"]), int(expected["fold"])
    )
    if complete.get("task_key") != expected_key:
        return fail("COMPLETE task_key mismatch")
    for payload_name, payload in (("COMPLETE", complete), ("result", result)):
        if payload.get("run_fingerprint") != run_fingerprint:
            return fail(f"{payload_name} run fingerprint mismatch")
        if payload.get("task_fingerprint") != expected_task_fp:
            return fail(f"{payload_name} task fingerprint mismatch")
        if payload.get("config_fingerprint") != str(expected["config_fingerprint"]):
            return fail(f"{payload_name} config fingerprint mismatch")
        if payload.get("runtime_environment_fingerprint") != str(
            expected["runtime_environment_fingerprint"]
        ):
            return fail(f"{payload_name} runtime fingerprint mismatch")

    expected_identity = (
        str(expected["session"]),
        MODEL_NAME,
        int(expected["seed"]),
        int(expected["fold"]),
    )
    observed_identity = (
        str(result.get("session")),
        str(result.get("model")),
        int(result.get("seed", -1)),
        int(result.get("fold", -1)),
    )
    normalization_identity = (
        str(normalization.get("session")),
        str(normalization.get("method")),
        int(normalization.get("seed", -1)),
        int(normalization.get("fold", -1)),
    )
    gate_identity = (
        str(gate.get("session")),
        str(gate.get("model")),
        int(gate.get("seed", -1)),
        int(gate.get("fold", -1)),
    )
    if observed_identity != expected_identity:
        return fail("result identity mismatch")
    if normalization_identity != expected_identity:
        return fail("normalization identity mismatch")
    if gate_identity != expected_identity:
        return fail("gate identity mismatch")
    if result.get("model_implementation_version") != MODEL_IMPLEMENTATION_VERSION:
        return fail("result implementation version mismatch")

    expected_architecture = architecture_config(LocalGlobalResidualMambaConfig())
    observed_architecture = {
        key: value
        for key, value in model_config.items()
        if key != "parameter_breakdown"
    }
    if fingerprint(expected_architecture) != fingerprint(observed_architecture):
        return fail("model_config differs from frozen architecture")
    breakdown = model_config.get("parameter_breakdown")
    components = {
        "cnn_stem_parameters",
        "spatial_position_parameters",
        "spatial_mamba_parameters",
        "gate_parameters",
        "temporal_1d_parameters",
        "classifier_parameters",
    }
    if not isinstance(breakdown, dict) or not components.issubset(breakdown):
        return fail("parameter breakdown missing")
    component_sum = sum(int(breakdown[name]) for name in components)
    if component_sum != int(breakdown.get("total_parameter_count", -1)):
        return fail("parameter breakdown does not sum")
    if int(result.get("parameter_count", -1)) != component_sum:
        return fail("result parameter count mismatch")
    for field in (*sorted(components), "total_parameter_count"):
        if int(result.get(field, -1)) != int(breakdown[field]):
            return fail(f"result parameter field mismatch: {field}")
    if component_sum != EXPECTED_FORMAL_PARAMETER_COUNT:
        return fail("formal parameter count differs from frozen architecture")
    if int(breakdown.get("gate_parameters", -1)) != 1:
        return fail("v1 must contain exactly one gate parameter")
    if int(result.get("actual_batch_size", -1)) != int(expected["batch_size"]):
        return fail("actual batch size differs from frozen task config")

    if bool(normalization.get("target_used_for_stats", True)):
        return fail("test fold was used for normalization statistics")
    if normalization.get("phase") != "outer_train_fold_only":
        return fail("normalization is not outer_train_fold_only")
    expected_n = int(expected["n_test_samples"])
    if int(result.get("n_test_samples", -1)) != expected_n or len(predictions) != expected_n:
        return fail("prediction count mismatch")
    prediction_columns = {
        "session",
        "model",
        "seed",
        "fold",
        "sample_index",
        "block_id",
        "cycle",
        "block_name",
        "y_true",
        "y_pred",
        "probability_0",
        "probability_1",
    }
    if not prediction_columns.issubset(predictions.columns):
        return fail("prediction columns missing")
    if len(predictions) and not (
        predictions["session"].eq(str(expected["session"])).all()
        and predictions["model"].eq(MODEL_NAME).all()
        and pd.to_numeric(predictions["seed"]).eq(int(expected["seed"])).all()
        and pd.to_numeric(predictions["fold"]).eq(int(expected["fold"])).all()
    ):
        return fail("prediction identity mismatch")
    probabilities = predictions[["probability_0", "probability_1"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        return fail("prediction probabilities invalid")
    metrics = classification_metrics(
        predictions["y_true"].to_numpy(int), predictions["y_pred"].to_numpy(int)
    )
    for name in ("balanced_accuracy", "accuracy", "macro_f1"):
        if name not in result or not np.isclose(
            float(result[name]), float(metrics[name]), atol=1e-12
        ):
            return fail(f"stored {name} differs from predictions")

    confusion_columns = {
        "session",
        "model",
        "seed",
        "fold",
        "true_label",
        "predicted_label",
        "count",
    }
    if not confusion_columns.issubset(confusion.columns) or len(confusion) != 4:
        return fail("confusion matrix shape/columns invalid")
    if int(pd.to_numeric(confusion["count"], errors="raise").sum()) != expected_n:
        return fail("confusion matrix count mismatch")
    if not (
        confusion["session"].eq(str(expected["session"])).all()
        and confusion["model"].eq(MODEL_NAME).all()
        and pd.to_numeric(confusion["seed"]).eq(int(expected["seed"])).all()
        and pd.to_numeric(confusion["fold"]).eq(int(expected["fold"])).all()
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

    history_columns = {
        "session",
        "model",
        "seed",
        "fold",
        "epoch",
        "train_loss",
        "train_accuracy",
        "alpha",
    }
    if not history_columns.issubset(history.columns) or len(history) != FORMAL_EPOCHS:
        return fail("training history missing columns or 40 epochs")
    if not (
        history["session"].eq(str(expected["session"])).all()
        and history["model"].eq(MODEL_NAME).all()
        and pd.to_numeric(history["seed"]).eq(int(expected["seed"])).all()
        and pd.to_numeric(history["fold"]).eq(int(expected["fold"])).all()
    ):
        return fail("training history identity mismatch")
    expected_epochs = np.arange(1, FORMAL_EPOCHS + 1)
    if not np.array_equal(history["epoch"].to_numpy(int), expected_epochs):
        return fail("training epoch sequence invalid")
    if not np.isfinite(
        history[["train_loss", "train_accuracy", "alpha"]].to_numpy(float)
    ).all():
        return fail("training history contains non-finite values")
    if not history["alpha"].between(0.0, 1.0, inclusive="neither").all():
        return fail("alpha is outside (0,1)")
    if int(result.get("trained_epochs", -1)) != FORMAL_EPOCHS:
        return fail("result is not exactly 40 epochs")

    initial_expected = float(torch.sigmoid(torch.tensor(INITIAL_GATE_LOGIT)).item())
    for field in ("initial_alpha", "final_alpha", "mean_alpha_last5_epochs"):
        if field not in gate or not np.isfinite(float(gate[field])):
            return fail(f"gate audit missing {field}")
    if not np.isclose(float(gate["initial_alpha"]), initial_expected, atol=1e-12):
        return fail("initial alpha differs from sigmoid(-2)")
    if not np.isclose(float(gate["final_alpha"]), float(history.iloc[-1]["alpha"]), atol=1e-12):
        return fail("final alpha differs from final history row")
    if not np.isclose(
        float(gate["mean_alpha_last5_epochs"]),
        float(history.tail(5)["alpha"].mean()),
        atol=1e-12,
    ):
        return fail("mean last-5 alpha differs from history")
    return True, "validated"


def update_status(
    args: argparse.Namespace, plan: pd.DataFrame, run_fp: str
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir,
            str(expected["session"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        valid, reason = validate_completed_task(path, expected, run_fp)
        rows.append(
            {
                "session": str(expected["session"]),
                "model": MODEL_NAME,
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
    total = len(status)
    print(
        f"STATUS completed={completed} pending={total - completed} total={total}",
        flush=True,
    )
    for row in status[status["status"].eq("pending")].head(5).itertuples(index=False):
        print(
            f"PENDING session={row.session} fold={row.fold} seed={row.seed} "
            f"reason={row.validation}",
            flush=True,
        )
    return status


def select_balanced_indices(
    y: np.ndarray, candidates: np.ndarray, per_class: int
) -> np.ndarray:
    return audit_utils.select_balanced_indices(y, candidates, per_class)


def run_sanity(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    require_mamba_dependency()
    if args.workers != 0:
        raise ValueError("sanity requires --workers 0")
    if args.sanity_epochs not in (1, 2):
        raise ValueError("sanity is restricted to 1 or 2 epochs")
    data, splits = audit_session(args, "710")
    train_idx, test_idx = splits[0]
    tiny_train = select_balanced_indices(data.y, train_idx, per_class=2)
    tiny_test = select_balanced_indices(data.y, test_idx, per_class=1)

    model = LocalGlobalResidualMambaClassifier().to(args.device)
    x = blocks_to_sequence_tensor(data.X[tiny_train[:2]]).to(args.device)
    y = torch.from_numpy(data.y[tiny_train[:2]].astype(np.int64)).to(args.device)
    before_gate = model.gate_logit.detach().clone()
    logits, shapes = model.forward_with_shapes(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    optimizer.step()
    expected_shapes = {
        "input": (2, 4, 1, 128, 501),
        "local_map": (2, 4, 64, 8, 32),
        "global_map": (2, 4, 64, 8, 32),
        "fused_map": (2, 4, 64, 8, 32),
        "frame_features": (2, 4, 64),
        "temporal_input": (2, 64, 4),
        "temporal_features": (2, 64),
        "logits": (2, 2),
    }
    if shapes != expected_shapes:
        raise AssertionError(f"unexpected sanity shapes: {shapes}")
    if not bool(torch.isfinite(loss).item()) or not gradients_finite:
        raise AssertionError("sanity forward/backward produced non-finite values")
    if torch.equal(before_gate, model.gate_logit.detach()):
        raise AssertionError("global gate did not receive an optimizer update")

    result, gate = train_local_global_residual_mamba_fold(
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
            min(4, len(tiny_train)), epochs=args.sanity_epochs
        ),
        device=args.device,
        workers=0,
    )
    if len(result.history) != args.sanity_epochs:
        raise AssertionError("tiny sanity fit epoch count mismatch")
    if not np.isfinite(result.probabilities).all() or not np.allclose(
        result.probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise AssertionError("sanity probabilities invalid")
    if bool(result.normalization_audit["target_used_for_stats"]):
        raise AssertionError("sanity normalization used test data")
    sanity_dir = args.output_dir / "sanity"
    atomic_json(
        sanity_dir / "sanity_audit.json",
        {
            "session": "710",
            "fold": 1,
            "seed": 0,
            "shapes": {key: list(value) for key, value in shapes.items()},
            "initial_alpha": gate["initial_alpha"],
            "final_alpha": gate["final_alpha"],
            "formal_clean4_fold_match": True,
            "cycle_overlap": False,
            "loss_finite": True,
            "backward_success": True,
            "normalization_target_used_for_stats": False,
            "tiny_epochs": args.sanity_epochs,
            "debug_only_not_formal": True,
        },
    )
    atomic_json(
        sanity_dir / "SANITY_COMPLETE.json",
        {
            "completed_utc": utc_now(),
            "run_fingerprint": fingerprint(identity),
            "formal_results": False,
            "checks_passed": True,
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
    trained, gate_audit = train_local_global_residual_mamba_fold(
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        np.asarray([0, 1], dtype=np.int64),
        session=session,
        fold=fold,
        seed=seed,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
        training_config=DeepTrainingConfig(
            **identity["experiment_config"]["training"]
        ),
        architecture=LocalGlobalResidualMambaConfig(),
        device=args.device,
        workers=args.workers,
    )
    metrics = classification_metrics(data.y[test_idx], trained.predictions)
    prediction_rows: list[dict[str, Any]] = []
    for local_index, sample_index in enumerate(test_idx):
        metadata = data.metadata.iloc[int(sample_index)]
        prediction_rows.append(
            {
                "session": session,
                "model": MODEL_NAME,
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
            "model": MODEL_NAME,
            "seed": seed,
            "fold": fold,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "count": int(cm[true_label, predicted_label]),
            "scope": "fold",
        }
        for true_label in (0, 1)
        for predicted_label in (0, 1)
    ]
    history = pd.DataFrame(trained.history)
    history.insert(0, "fold", fold)
    history.insert(0, "seed", seed)
    history.insert(0, "model", MODEL_NAME)
    history.insert(0, "session", session)
    task_fp = task_fingerprint(run_fp, expected)
    breakdown = trained.model_config["parameter_breakdown"]
    result_payload = {
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": str(expected["config_fingerprint"]),
        "runtime_environment_fingerprint": str(
            expected["runtime_environment_fingerprint"]
        ),
        "runtime_environment_signature": identity["runtime_environment_signature"],
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "session": session,
        "model": MODEL_NAME,
        "seed": seed,
        "fold": fold,
        "n_cycles": data.n_cycles,
        "n_samples": data.n_blocks,
        "n_train_samples": len(train_idx),
        "n_test_samples": len(test_idx),
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "parameter_count": int(trained.model_parameters),
        **{key: int(value) for key, value in breakdown.items()},
        **{key: float(value) for key, value in gate_audit.items()},
        "actual_batch_size": int(identity["experiment_config"]["training"]["batch_size"]),
        "final_training_loss": float(trained.final_training_loss),
        "trained_epochs": int(trained.final_trained_epochs),
        "device": trained.device,
    }
    gate_payload = {
        "session": session,
        "model": MODEL_NAME,
        "seed": seed,
        "fold": fold,
        **{key: float(value) for key, value in gate_audit.items()},
        "gate_scope": "one global trainable scalar",
        "test_fold_used_to_set_gate": False,
    }
    atomic_json(path / "result.json", result_payload)
    atomic_csv(path / "predictions.csv", pd.DataFrame(prediction_rows))
    atomic_csv(path / "confusion_matrix.csv", pd.DataFrame(confusion_rows))
    atomic_csv(path / "training_history.csv", history)
    atomic_json(path / "normalization_audit.json", trained.normalization_audit)
    atomic_json(path / "model_config.json", trained.model_config)
    atomic_json(path / "gate.json", gate_payload)
    atomic_json(
        path / "COMPLETE.json",
        {
            "task_key": task_key(session, seed, fold),
            "run_fingerprint": run_fp,
            "task_fingerprint": task_fp,
            "config_fingerprint": str(expected["config_fingerprint"]),
            "runtime_environment_fingerprint": str(
                expected["runtime_environment_fingerprint"]
            ),
            "completed_utc": utc_now(),
            "validated_files": [
                "result.json",
                "predictions.csv",
                "confusion_matrix.csv",
                "training_history.csv",
                "normalization_audit.json",
                "model_config.json",
                "gate.json",
            ],
        },
    )
    validate_completed_task(path, expected, run_fp, raise_on_error=True)


def read_all_validated_tasks(
    args: argparse.Namespace, plan: pd.DataFrame, run_fp: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    confusions: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    gates: list[dict[str, Any]] = []
    for expected in plan.to_dict(orient="records"):
        path = task_dir(
            args.output_dir,
            str(expected["session"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        validate_completed_task(path, expected, run_fp, raise_on_error=True)
        results.append(json.loads((path / "result.json").read_text(encoding="utf-8")))
        predictions.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        confusions.append(
            pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str})
        )
        histories.append(
            pd.read_csv(path / "training_history.csv", dtype={"session": str})
        )
        gates.append(json.loads((path / "gate.json").read_text(encoding="utf-8")))
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True),
        pd.concat(confusions, ignore_index=True),
        pd.concat(histories, ignore_index=True),
        pd.DataFrame(gates),
    )


def clean4_long_candidates(args: argparse.Namespace) -> list[Path]:
    return audit_utils.clean4_long_candidates(args)


def mamba_summary_candidates(args: argparse.Namespace) -> list[Path]:
    base = args.project_root / "outputs" / "mamba_visual_binary_v1"
    return [base / "mamba_summary.csv"]


def mamba_overfit_candidates(args: argparse.Namespace) -> list[Path]:
    base = args.project_root / "outputs" / "mamba_visual_binary_v1"
    return [base / "overfitting_summary.csv"]


def transformer_summary_candidates(args: argparse.Namespace) -> list[Path]:
    return audit_utils.transformer_summary_candidates(args)


def _first_existing(paths: list[Path], description: str) -> Path:
    selected = next((path for path in paths if path.exists()), None)
    if selected is None:
        raise FileNotFoundError(
            f"required existing formal {description} not found; checked {paths}"
        )
    return selected


def load_existing_comparison_baselines(args: argparse.Namespace) -> pd.DataFrame:
    """Read exactly four frozen baselines; never retrain them in this runner."""
    clean_path = _first_existing(clean4_long_candidates(args), "clean4 baseline table")
    clean = pd.read_csv(clean_path, dtype={"session": str})
    if "task" in clean.columns:
        clean = clean[clean["task"].astype(str).eq(TASK_NAME)]
    required_clean = {"session", "method", "seed", "balanced_accuracy", "accuracy"}
    if not required_clean.issubset(clean.columns):
        raise AssertionError("formal clean4 long table lacks required numeric columns")
    clean = clean[clean["method"].isin([LOCAL_BASELINE_NAME, "fcnn_meanpool"])]
    clean_rows: list[dict[str, Any]] = []
    for (session, method), group in clean.groupby(["session", "method"], sort=True):
        if group["seed"].nunique() != len(SEEDS):
            raise AssertionError(f"{session} {method}: expected exactly three formal seeds")
        clean_rows.append(
            {
                "session": str(session),
                "model": str(method),
                "model_display": DISPLAY_NAMES[str(method)],
                "mean_BA": float(group["balanced_accuracy"].astype(float).mean()),
                "std_BA": float(group["balanced_accuracy"].astype(float).std(ddof=1)),
                "mean_accuracy": float(group["accuracy"].astype(float).mean()),
                "n_seeds": int(group["seed"].nunique()),
                "source": str(clean_path),
                "retrained_by_this_runner": False,
            }
        )

    summary_specs = (
        (
            "spatial_mamba",
            _first_existing(mamba_summary_candidates(args), "Spatial Mamba summary"),
        ),
        (
            "cnn_factorized_transformer",
            _first_existing(
                transformer_summary_candidates(args), "Factorized Transformer summary"
            ),
        ),
    )
    summary_rows: list[dict[str, Any]] = []
    for model, path in summary_specs:
        frame = pd.read_csv(path, dtype={"session": str})
        required = {"session", "model", "mean_BA", "std_BA", "mean_accuracy"}
        if not required.issubset(frame.columns):
            raise AssertionError(f"formal {model} summary lacks required columns")
        selected = frame[frame["model"].astype(str).eq(model)]
        for row in selected.itertuples(index=False):
            summary_rows.append(
                {
                    "session": str(row.session),
                    "model": model,
                    "model_display": DISPLAY_NAMES[model],
                    "mean_BA": float(row.mean_BA),
                    "std_BA": float(row.std_BA),
                    "mean_accuracy": float(row.mean_accuracy),
                    "n_seeds": len(SEEDS),
                    "source": str(path),
                    "retrained_by_this_runner": False,
                }
            )
    combined = pd.concat(
        [pd.DataFrame(clean_rows), pd.DataFrame(summary_rows)], ignore_index=True
    )
    expected_pairs = {
        (session, model)
        for session in EXPECTED_SESSIONS
        for model in COMPARISON_BASELINES
    }
    observed_pairs = set(zip(combined["session"], combined["model"]))
    if observed_pairs != expected_pairs:
        raise AssertionError(
            f"formal comparison baseline coverage mismatch: missing={sorted(expected_pairs - observed_pairs)}, "
            f"unexpected={sorted(observed_pairs - expected_pairs)}"
        )
    return combined.sort_values(["session", "model"]).reset_index(drop=True)


def paired_comparison_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    rows: list[dict[str, Any]] = []
    for baseline in COMPARISON_BASELINES:
        if baseline not in pivot or MODEL_NAME not in pivot:
            raise AssertionError(f"comparison is incomplete for {baseline}")
        deltas = (
            pivot.loc[list(EXPECTED_SESSIONS), MODEL_NAME]
            - pivot.loc[list(EXPECTED_SESSIONS), baseline]
        )
        strong_delta = deltas.loc[list(STRONG_SESSIONS)]
        weak_delta = deltas.loc[list(WEAK_SESSIONS)]
        tolerance = 1e-12
        rows.append(
            {
                "comparison": f"{MODEL_NAME}_vs_{baseline}",
                "baseline": baseline,
                "n_sessions": len(deltas),
                "mean_delta_BA": float(deltas.mean()),
                "median_delta_BA": float(deltas.median()),
                "improved_sessions": int((deltas > tolerance).sum()),
                "tied_sessions": int((deltas.abs() <= tolerance).sum()),
                "worsened_sessions": int((deltas < -tolerance).sum()),
                "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(
                    deltas.to_numpy(float)
                ),
                "strong_session_mean_delta_BA": float(strong_delta.mean()),
                "strong_improved_tied_worsened": canonical_json(
                    {
                        "improved": int((strong_delta > tolerance).sum()),
                        "tied": int((strong_delta.abs() <= tolerance).sum()),
                        "worsened": int((strong_delta < -tolerance).sum()),
                    }
                ),
                "weak_session_mean_delta_BA": float(weak_delta.mean()),
                "weak_improved_tied_worsened": canonical_json(
                    {
                        "improved": int((weak_delta > tolerance).sum()),
                        "tied": int((weak_delta.abs() <= tolerance).sum()),
                        "worsened": int((weak_delta < -tolerance).sum()),
                    }
                ),
                "session_deltas_json": canonical_json(
                    {session: float(value) for session, value in deltas.items()}
                ),
            }
        )
    return pd.DataFrame(rows)


def build_proposed_overfit(
    history: pd.DataFrame, per_seed: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (session, seed), group in history.groupby(["session", "seed"], sort=True):
        by_epoch = (
            group.groupby("epoch", as_index=True)["train_accuracy"].mean().sort_index()
        )
        if list(by_epoch.index.astype(int)) != list(range(1, FORMAL_EPOCHS + 1)):
            raise AssertionError(f"history incomplete for {session} seed {seed}")
        oof = per_seed[
            per_seed["session"].astype(str).eq(str(session))
            & per_seed["seed"].astype(int).eq(int(seed))
        ]
        if len(oof) != 1:
            raise AssertionError(f"OOF row not unique for {session} seed {seed}")
        best_epoch = int(by_epoch.idxmax())
        oof_ba = float(oof.iloc[0]["balanced_accuracy"])
        best_train = float(by_epoch.loc[best_epoch])
        rows.append(
            {
                "session": str(session),
                "model": MODEL_NAME,
                "seed": int(seed),
                "final_train_accuracy": float(by_epoch.loc[FORMAL_EPOCHS]),
                "best_train_accuracy": best_train,
                "OOF_test_BA": oof_ba,
                "generalization_gap": best_train - oof_ba,
                "best_epoch": best_epoch,
                "selected_epoch": FORMAL_EPOCHS,
                "epoch_selection": "fixed_40_no_test_or_validation_selection",
                "n_folds": int(group["fold"].nunique()),
                "possible_severe_overfit": bool(best_train >= 0.95 and oof_ba <= 0.60),
            }
        )
    return pd.DataFrame(rows)


def load_pure_mamba_overfit(args: argparse.Namespace) -> pd.DataFrame:
    path = _first_existing(mamba_overfit_candidates(args), "Spatial Mamba overfit audit")
    frame = pd.read_csv(path, dtype={"session": str})
    required = {
        "session",
        "model",
        "seed",
        "final_train_accuracy",
        "best_train_accuracy",
        "OOF_test_BA",
        "best_epoch",
        "selected_epoch",
        "n_folds",
        "possible_severe_overfit",
    }
    if not required.issubset(frame.columns):
        raise AssertionError("Spatial Mamba overfit audit lacks required columns")
    frame = frame[frame["model"].astype(str).eq("spatial_mamba")].copy()
    if len(frame) != len(EXPECTED_SESSIONS) * len(SEEDS):
        raise AssertionError("Spatial Mamba overfit audit is not complete for 9x3")
    frame["generalization_gap"] = (
        frame["best_train_accuracy"].astype(float) - frame["OOF_test_BA"].astype(float)
    )
    return frame[
        [
            "session",
            "model",
            "seed",
            "final_train_accuracy",
            "best_train_accuracy",
            "OOF_test_BA",
            "generalization_gap",
            "best_epoch",
            "selected_epoch",
            "epoch_selection",
            "n_folds",
            "possible_severe_overfit",
        ]
    ]


def decision_rule_audit(
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    overfitting: pd.DataFrame,
) -> dict[str, Any]:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    temporal_delta = pivot[MODEL_NAME] - pivot[LOCAL_BASELINE_NAME]
    mamba_delta = pivot[MODEL_NAME] - pivot["spatial_mamba"]
    strong_temporal = temporal_delta.loc[list(STRONG_SESSIONS)]
    strong_mamba = mamba_delta.loc[list(STRONG_SESSIONS)]
    severe_flags = overfitting["possible_severe_overfit"].map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() == "true"
    )
    severe_by_model = (
        overfitting.assign(possible_severe_overfit=severe_flags)
        .groupby(["model", "session"])["possible_severe_overfit"]
        .any()
    )
    proposed_severe = int(severe_by_model.loc[MODEL_NAME].sum())
    mamba_severe = int(severe_by_model.loc["spatial_mamba"].sum())
    checks = {
        "mean_BA_above_temporal1d": bool(temporal_delta.mean() > 0.0),
        "at_least_6_of_9_non_decreasing_vs_temporal1d": bool(
            int((temporal_delta >= -1e-12).sum()) >= 6
        ),
        "strong_at_least_2_of_3_not_notably_down_vs_temporal1d": bool(
            int((strong_temporal >= -NOTABLE_DECLINE_TOLERANCE).sum()) >= 2
        ),
        "strong_mean_recovery_vs_spatial_mamba": bool(
            float(strong_mamba.mean()) >= STRONG_MAMBA_RECOVERY_THRESHOLD
        ),
        "strong_at_least_2_of_3_improve_vs_spatial_mamba": bool(
            int((strong_mamba > 1e-12).sum()) >= 2
        ),
        "severe_overfit_session_count_reduced_vs_spatial_mamba": bool(
            proposed_severe < mamba_severe
        ),
    }
    supports_continue = bool(all(checks.values()))
    return {
        "criteria_frozen_before_formal_results": True,
        "notable_decline_tolerance_BA": NOTABLE_DECLINE_TOLERANCE,
        "strong_mamba_recovery_threshold_BA": STRONG_MAMBA_RECOVERY_THRESHOLD,
        "checks": checks,
        "proposed_severe_overfit_sessions": proposed_severe,
        "spatial_mamba_severe_overfit_sessions": mamba_severe,
        "decision": (
            "supports_continue_mamba_route_to_manually_reviewed_multiscale_stage"
            if supports_continue
            else "does_not_support_continue_mamba_route; next candidate is lightweight_multiscale_cnn"
        ),
        "automatic_next_stage_started": False,
        "paired_table_fingerprint": fingerprint(
            paired.sort_values("baseline").to_dict(orient="records")
        ),
    }


def build_report(
    comparison: pd.DataFrame,
    paired: pd.DataFrame,
    gates: pd.DataFrame,
    overfitting: pd.DataFrame,
    decision: dict[str, Any],
) -> str:
    pivot = comparison.pivot(index="session", columns="model", values="mean_BA")
    gate_tagged = gates.assign(
        session_group=np.where(
            gates["session"].astype(str).isin(STRONG_SESSIONS), "strong", "weak"
        )
    )
    gate_group = gate_tagged.groupby("session_group")["final_alpha"].mean()
    lines = [
        "# Local + Global Residual Mamba proposed v1 report",
        "",
        "compute_environment = server",
        "",
        "本报告执行预先冻结的第一轮评价；没有根据测试结果修改 gate、Mamba、TCN 或训练参数。",
        "",
        "## Nine-session comparisons",
        "",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            f"- Proposed vs {DISPLAY_NAMES[row.baseline]}: mean ΔBA="
            f"{row.mean_delta_BA:+.4f}, median={row.median_delta_BA:+.4f}, "
            f"improved/tied/worsened={row.improved_sessions}/{row.tied_sessions}/"
            f"{row.worsened_sessions}, exact p={row.exact_two_sided_sign_flip_p:.4f}, "
            f"strong Δ={row.strong_session_mean_delta_BA:+.4f}, "
            f"weak Δ={row.weak_session_mean_delta_BA:+.4f}."
        )
    lines.extend(["", "## Strong sessions", ""])
    for session in STRONG_SESSIONS:
        lines.append(
            f"- {session}: Temporal1D={pivot.loc[session, LOCAL_BASELINE_NAME]:.4f}, "
            f"Spatial Mamba={pivot.loc[session, 'spatial_mamba']:.4f}, "
            f"Proposed={pivot.loc[session, MODEL_NAME]:.4f}."
        )
    lines.extend(
        [
            "",
            "## Gate and overfitting",
            "",
            f"- Strong-session mean final alpha={float(gate_group.get('strong', np.nan)):.4f}; "
            f"weak-session mean final alpha={float(gate_group.get('weak', np.nan)):.4f}. "
            "Gate comparison is exploratory only.",
            f"- Severe-overfit session count: proposed="
            f"{decision['proposed_severe_overfit_sessions']}, pure Spatial Mamba="
            f"{decision['spatial_mamba_severe_overfit_sessions']}.",
            "",
            "## Pre-registered decision",
            "",
            f"Decision: `{decision['decision']}`.",
            "",
            "本 runner 到此停止；不会自动运行 multi-scale、改变 gate 或扩展 Mamba。",
        ]
    )
    return "\n".join(lines) + "\n"


def aggregate_outputs(
    args: argparse.Namespace, plan: pd.DataFrame, identity: dict[str, Any]
) -> None:
    run_fp = fingerprint(identity)
    per_fold, predictions, confusions, history, gates = read_all_validated_tasks(
        args, plan, run_fp
    )
    per_fold = per_fold.sort_values(["session", "seed", "fold"]).reset_index(drop=True)
    predictions = predictions.sort_values(
        ["session", "seed", "fold", "sample_index"]
    ).reset_index(drop=True)
    confusions = confusions.sort_values(
        ["session", "seed", "fold", "true_label", "predicted_label"]
    ).reset_index(drop=True)
    history = history.sort_values(["session", "seed", "fold", "epoch"]).reset_index(
        drop=True
    )
    gates = gates.sort_values(["session", "seed", "fold"]).reset_index(drop=True)
    atomic_csv(args.output_dir / "proposed_per_fold.csv", per_fold)
    atomic_csv(args.output_dir / "proposed_predictions.csv", predictions)
    atomic_csv(args.output_dir / "proposed_confusion_matrices.csv", confusions)
    atomic_csv(args.output_dir / "proposed_training_history.csv", history)
    atomic_csv(args.output_dir / "gate_summary.csv", gates)

    seed_rows: list[dict[str, Any]] = []
    for (session, model, seed), group in predictions.groupby(
        ["session", "model", "seed"], sort=True
    ):
        source = per_fold[
            per_fold["session"].astype(str).eq(str(session))
            & per_fold["seed"].astype(int).eq(int(seed))
        ]
        if group["sample_index"].duplicated().any():
            raise AssertionError(f"duplicate OOF samples for {session} seed {seed}")
        expected_n = int(source["n_samples"].iloc[0])
        if len(group) != expected_n or set(group["sample_index"].astype(int)) != set(
            range(expected_n)
        ):
            raise AssertionError(f"incomplete OOF coverage for {session} seed {seed}")
        metrics = classification_metrics(
            group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int)
        )
        seed_rows.append(
            {
                "session": str(session),
                "model": str(model),
                "seed": int(seed),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "n_cycles": int(source["n_cycles"].iloc[0]),
                "n_samples": len(group),
                "n_folds": int(group["fold"].nunique()),
                "parameter_count": int(source["parameter_count"].iloc[0]),
            }
        )
    per_seed = pd.DataFrame(seed_rows).sort_values(["session", "seed"])
    if len(per_seed) != len(EXPECTED_SESSIONS) * len(SEEDS):
        raise AssertionError("proposed OOF per-seed coverage is incomplete")
    atomic_csv(args.output_dir / "proposed_per_seed.csv", per_seed)
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
    atomic_csv(args.output_dir / "proposed_summary.csv", summary)

    existing = load_existing_comparison_baselines(args)
    proposed_rows = summary.assign(
        model_display=MODEL_DISPLAY_NAME,
        n_seeds=len(SEEDS),
        source=str(args.output_dir / "proposed_summary.csv"),
        retrained_by_this_runner=True,
    )
    comparison = pd.concat([existing, proposed_rows], ignore_index=True).sort_values(
        ["session", "model"]
    )
    if len(comparison) != len(EXPECTED_SESSIONS) * (len(COMPARISON_BASELINES) + 1):
        raise AssertionError("model comparison row count is incomplete")
    atomic_csv(args.output_dir / "model_comparison.csv", comparison)
    paired = paired_comparison_rows(comparison)
    atomic_csv(args.output_dir / "paired_comparisons.csv", paired)

    strong = comparison[
        comparison["session"].isin(STRONG_SESSIONS)
        & comparison["model"].isin([LOCAL_BASELINE_NAME, "spatial_mamba", MODEL_NAME])
    ][["session", "model", "model_display", "mean_BA", "std_BA"]]
    if len(strong) != len(STRONG_SESSIONS) * 3:
        raise AssertionError("strong-session comparison is incomplete")
    atomic_csv(args.output_dir / "strong_session_comparison.csv", strong)

    proposed_overfit = build_proposed_overfit(history, per_seed)
    pure_overfit = load_pure_mamba_overfit(args)
    overfitting = pd.concat([pure_overfit, proposed_overfit], ignore_index=True).sort_values(
        ["session", "model", "seed"]
    )
    atomic_csv(args.output_dir / "overfitting_comparison.csv", overfitting)
    decision = decision_rule_audit(comparison, paired, overfitting)
    atomic_json(args.output_dir / "decision_rule_audit.json", decision)
    atomic_text(
        args.output_dir / "proposed_v1_report.md",
        build_report(comparison, paired, gates, overfitting, decision),
    )


def run_cuda_batch16_preflight(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size != 16:
        raise AssertionError("proposed v1 requires frozen batch size 16")
    device = torch.device(args.device if args.device != "auto" else "cuda")
    audit: dict[str, Any] = {}
    try:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        model = LocalGlobalResidualMambaClassifier().to(device)
        inputs = torch.zeros((16, 4, 1, 128, 501), dtype=torch.float32, device=device)
        targets = torch.arange(16, device=device, dtype=torch.long) % 2
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        initial_alpha = float(model.alpha.detach().cpu().item())
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        optimizer.step()
        if tuple(logits.shape) != (16, 2) or not bool(torch.isfinite(loss).item()):
            raise AssertionError("CUDA batch-16 preflight produced invalid output")
        audit = {
            "status": "pass",
            "formal_training_started": False,
            "device": str(device),
            "batch_size": 16,
            "input_shape": [16, 4, 1, 128, 501],
            "logits_shape": [16, 2],
            "loss_finite": True,
            "backward_success": True,
            "optimizer_step_success": True,
            "initial_alpha": initial_alpha,
            "post_step_alpha": float(model.alpha.detach().cpu().item()),
            **parameter_breakdown(model),
        }
        if audit["total_parameter_count"] != EXPECTED_FORMAL_PARAMETER_COUNT:
            raise AssertionError(
                "formal parameter count differs from frozen proposed v1 architecture"
            )
        del loss, logits, optimizer, targets, inputs, model
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "CUDA OOM during mandatory batch-16 preflight. No formal task started; "
                "do not change batch size automatically."
            ) from exc
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(args.output_dir / "audit" / "cuda_batch16_preflight.json", audit)
    print(
        f"PREFLIGHT PASS batch_size=16 parameters={audit['total_parameter_count']} "
        f"device={device}",
        flush=True,
    )
    return audit


def run_full(args: argparse.Namespace, identity: dict[str, Any]) -> None:
    if not args.review_approved:
        raise RuntimeError(
            "formal run is locked until external code review is approved; rerun with "
            "--review-approved only after the user gives a greenlight"
        )
    require_mamba_dependency()
    if not torch.cuda.is_available():
        raise RuntimeError("formal proposed v1 run requires CUDA")
    if args.device != "auto" and not args.device.startswith("cuda"):
        raise RuntimeError("formal proposed v1 run requires --device cuda or cuda:N")
    expected_env = Path("/data2/yuq1ngr/conda_envs/fus_mamba")
    executable = Path(sys.executable).resolve()
    if expected_env not in executable.parents:
        raise RuntimeError(
            f"formal run must use {expected_env}; current interpreter is {executable}"
        )
    invalid_sessions = sorted(
        set(str(value) for value in args.sessions) - set(EXPECTED_SESSIONS)
    )
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
        data, splits = audit_session(args, session)
        for expected in session_plan.to_dict(orient="records"):
            path = task_dir(
                args.output_dir,
                session,
                int(expected["seed"]),
                int(expected["fold"]),
            )
            valid, _ = validate_completed_task(path, expected, run_fp)
            if valid:
                print(
                    f"SKIP [{completed}/{total}] session={session} model={MODEL_NAME} "
                    f"fold={expected['fold']} seed={expected['seed']} device={args.device}",
                    flush=True,
                )
                continue
            train_idx, test_idx = splits[int(expected["fold"]) - 1]
            print(
                f"RUN  [{completed}/{total}] session={session} model={MODEL_NAME} "
                f"fold={expected['fold']} seed={expected['seed']} device={args.device}",
                flush=True,
            )
            write_fold_task(args, identity, expected, data, train_idx, test_idx)
            completed += 1
            print(
                f"DONE [{completed}/{total}] session={session} model={MODEL_NAME} "
                f"fold={expected['fold']} seed={expected['seed']} device={args.device}",
                flush=True,
            )
        del data
    status = update_status(args, plan, run_fp)
    if not bool(len(status) and status["status"].eq("complete").all()):
        print("PARTIAL RUN SAVED; rerun the identical screen command to resume", flush=True)
        return
    aggregate_outputs(args, plan, identity)
    missing = [
        name for name in REQUIRED_FINAL_OUTPUTS if not (args.output_dir / name).exists()
    ]
    if missing:
        raise AssertionError(f"finalization missing outputs: {missing}")
    atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "compute_environment": "server",
            "completed_utc": utc_now(),
            "run_fingerprint": run_fp,
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "runtime_environment_signature": identity["runtime_environment_signature"],
            "completed_tasks": len(plan),
            "total_tasks": len(plan),
            "existing_temporal_baseline_retrained": False,
            "required_outputs": list(REQUIRED_FINAL_OUTPUTS),
            "strict_task_revalidation_before_aggregation": True,
            "automatic_next_stage_started": False,
        },
    )
    print(f"FULL RUN COMPLETE: {args.output_dir / 'RUN_COMPLETE.json'}", flush=True)
    print("STOP: waiting for manual analysis; no next-stage model was started", flush=True)


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
