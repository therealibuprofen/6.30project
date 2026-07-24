#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Required before CUDA-backed deterministic linear algebra is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.linear import fit_predict_linear
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    TASK_CLASS_NAMES,
    class_count_dict,
    csv_json,
    cycle_text,
    dataset_audit_row,
    default_block_data_dir,
    load_block_sequence_session,
    split_manifest,
    task_run_dir_name,
)
from ultrasound_decoding.multiframe.evaluation import (
    CHANCE_LEVEL,
    completeness_report,
    confusion_rows,
    method_summary_table,
    metrics_with_flags,
    prediction_rows,
    seed_mean_summary,
    vs_singleframe_reference,
)
from ultrasound_decoding.multiframe.models import (
    LINEAR_METHODS,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MULTIFRAME_METHODS,
    NEURAL_METHODS,
    ORDER_SENSITIVE_METHODS,
    build_multiframe_model,
    count_trainable_parameters,
    model_shape_audit,
)
from ultrasound_decoding.multiframe.plotting import make_all_plots
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    order_sensitivity_for_trained_sequence_model,
    train_sequence_fold,
    train_single_frame_late_fusion_fold,
)


DEFAULT_METHODS = list(MULTIFRAME_METHODS)
DEFAULT_SEEDS = [0, 1, 2]
FIXED_SHUFFLE = "2,0,3,1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict block-level clean4 multiframe decoding benchmarks."
    )
    parser.add_argument(
        "--stage",
        default="dry-run",
        choices=["dry-run", "smoke", "benchmark", "aggregate-only"],
        help="dry-run audits only; smoke runs session 710/binary/fold1/seed0/2 epochs; benchmark runs requested jobs.",
    )
    parser.add_argument("--sessions", nargs="+", default=None, help="Sessions to run")
    parser.add_argument("--tasks", nargs="+", default=None, choices=EXPECTED_TASKS, help="Tasks to run")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "results" / "runs" / "multiframe")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Override task run directory name. Defaults to block_clean4_{task}_v1; smoke appends _smoke.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting files in the selected multiframe run.")
    return parser.parse_args()


def resolve_sessions_tasks(args: argparse.Namespace) -> tuple[list[str], list[str], list[int], int, int | None]:
    if args.stage == "smoke":
        sessions = ["710"] if args.sessions is None else [str(value) for value in args.sessions]
        tasks = ["binary"] if args.tasks is None else list(args.tasks)
        seeds = [0] if args.seeds == DEFAULT_SEEDS else [int(value) for value in args.seeds]
        max_epochs = min(int(args.max_epochs), 2)
        limit_folds = 1 if args.limit_folds is None else int(args.limit_folds)
    else:
        sessions = [str(value) for value in (args.sessions if args.sessions is not None else EXPECTED_SESSIONS)]
        tasks = list(args.tasks if args.tasks is not None else EXPECTED_TASKS)
        seeds = [int(value) for value in args.seeds]
        max_epochs = int(args.max_epochs)
        limit_folds = args.limit_folds
    return sessions, tasks, seeds, max_epochs, limit_folds


def task_run_dir(args: argparse.Namespace, task: str) -> Path:
    if args.run_name is not None:
        name = args.run_name
    else:
        name = task_run_dir_name(task)
        if args.stage == "smoke":
            name = f"{name}_smoke"
    return args.output_root / name


def ensure_output_run_is_safe(path: Path, output_root: Path) -> None:
    resolved_root = output_root.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents and resolved != resolved_root:
        raise ValueError(f"output path must be under {resolved_root}, got {resolved}")
    forbidden_parts = {"generalization", "interpretability", "temporal_windows", "spatial_filter_ablation"}
    if any(part in forbidden_parts for part in resolved.parts):
        raise ValueError(f"refusing to write multiframe outputs into existing non-multiframe run tree: {resolved}")


def json_dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def preprocess_blocks_flat4(X: np.ndarray) -> np.ndarray:
    if X.ndim != 4 or X.shape[1] != 4:
        raise ValueError(f"expected [N, 4, H, W], got {X.shape}")
    return np.arcsinh(X.astype(np.float64, copy=False)).reshape(len(X), -1)


def valid_fold(y_train: np.ndarray, y_test: np.ndarray) -> bool:
    return len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2


def senior_code_audit_rows() -> list[dict[str, Any]]:
    rows = []
    files = [
        Path("/Users/ibuprofen/Desktop/code2/dataset.py"),
        Path("/Users/ibuprofen/Desktop/code2/dataset_windows.py"),
        Path("/Users/ibuprofen/Desktop/code2/train_cnn.py"),
    ]
    facts = {
        "dataset.py": (
            "HDF5 feature arrays are loaded with f[key][:].flatten(); segments have length segment_length*100; "
            "dataset __getitem__ returns a 1D tensor [L]."
        ),
        "dataset_windows.py": (
            "HDF5 feature arrays are loaded with f[key][:].flatten(); fixed windows have length 100; "
            "dataset __getitem__ returns a 1D tensor [L]."
        ),
        "train_cnn.py": (
            "CNN1D.forward calls x.unsqueeze(1), so the Conv1d input is [B, 1, L]; "
            "the senior model is not a direct 2D fUS image sequence model."
        ),
    }
    for path in files:
        rows.append(
            {
                "file": str(path),
                "exists": bool(path.exists()),
                "observed_input_fact": facts[path.name],
                "adaptation_note": (
                    "This benchmark names cnn2d_temporal1d as a reference-to-senior 1D-CNN idea "
                    "adapted to fUS clean4 frame features, not an exact reproduction of the senior input pipeline."
                ),
            }
        )
    return rows


def parameter_audit_rows(methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        if method in LINEAR_METHODS:
            rows.append(
                {
                    "method": method,
                    "method_display": MODEL_DISPLAY_NAMES[method],
                    "method_description": MODEL_DESCRIPTIONS[method],
                    "model_parameters": 0,
                    "input_shape": "[B,4,128,501] flattened to [B,256512]",
                    "encoder_feature_dim": "",
                    "temporal_length": 4,
                    "temporal_adaptation": False,
                }
            )
            continue
        model = build_multiframe_model(method, n_classes=2)
        audit = model_shape_audit(method, n_classes=2)
        rows.append(
            {
                "method": method,
                "method_display": MODEL_DISPLAY_NAMES[method],
                "method_description": MODEL_DESCRIPTIONS[method],
                "model_parameters": int(count_trainable_parameters(model)),
                "input_shape": csv_json(list(audit.input_shape)),
                "encoder_feature_dim": int(audit.encoder_feature_dim),
                "temporal_length": int(audit.temporal_length),
                "output_shape": csv_json(list(audit.output_shape)),
                "temporal_conv_axis": audit.temporal_conv_axis,
                "temporal_adaptation": method == "cnn2d_temporal1d",
            }
        )
    return rows


def run_dry_run(args: argparse.Namespace) -> None:
    sessions, tasks, seeds, max_epochs, limit_folds = resolve_sessions_tasks(args)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    out_dir = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    plan_rows = []
    for task in tasks:
        for session in sessions:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            splits = grouped_cv_splits(data.groups, max_folds=args.max_folds)
            if limit_folds is not None:
                splits = splits[: int(limit_folds)]
            row = dataset_audit_row(data, max_folds=args.max_folds)
            row["planned_output_dir"] = str(task_run_dir(args, task) / f"session_{session}")
            audit_rows.append(row)
            n_linear_methods = sum(method in LINEAR_METHODS for method in args.methods)
            n_neural_methods = sum(method in NEURAL_METHODS for method in args.methods)
            plan_rows.append(
                {
                    "session": session,
                    "task": task,
                    "n_cycles": data.n_cycles,
                    "n_blocks": data.n_blocks,
                    "n_splits": len(splits),
                    "linear_fit_count": int(len(splits) * n_linear_methods),
                    "neural_training_run_count": int(len(splits) * n_neural_methods * len(seeds)),
                    "seeds": csv_json(seeds),
                    "max_epochs": int(max_epochs),
                    "batch_size_rule": "min(requested_batch_size, n_train_blocks), requested <= 16 by default",
                    "chance_level": CHANCE_LEVEL,
                }
            )
    pd.DataFrame(audit_rows).to_csv(out_dir / "stage0_block_data_audit.csv", index=False)
    pd.DataFrame(plan_rows).to_csv(out_dir / "stage0_training_plan.csv", index=False)
    pd.DataFrame(parameter_audit_rows(args.methods)).to_csv(out_dir / "stage0_model_parameter_audit.csv", index=False)
    pd.DataFrame(senior_code_audit_rows()).to_csv(out_dir / "stage0_senior_code_input_audit.csv", index=False)
    print(f"[dry-run] block data audit: {out_dir / 'stage0_block_data_audit.csv'}")
    print(f"[dry-run] training plan: {out_dir / 'stage0_training_plan.csv'}")
    print(f"[dry-run] model parameters: {out_dir / 'stage0_model_parameter_audit.csv'}")
    print(f"[dry-run] senior code audit: {out_dir / 'stage0_senior_code_input_audit.csv'}")


def split_cycle_info(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[str, str]:
    return cycle_text(groups[train_idx]), cycle_text(groups[test_idx])


def append_fold_outputs(
    *,
    session: str,
    task: str,
    method: str,
    seed: int,
    fold_i: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    data,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    n_components: int | None,
    model_parameters: int,
    final_training_loss: float | None,
    final_trained_epochs: int | None,
    device: str | None,
    fold_rows: list[dict[str, Any]],
    prediction_row_accumulator: list[dict[str, Any]],
    confusion_row_accumulator: list[dict[str, Any]],
) -> None:
    y_test = data.y[test_idx]
    metrics = metrics_with_flags(y_test, predictions)
    train_cycles, test_cycles = split_cycle_info(data.groups, train_idx, test_idx)
    fold_rows.append(
        {
            "session": str(session),
            "task": task,
            "method": method,
            "method_display": MODEL_DISPLAY_NAMES[method],
            "seed": int(seed),
            "fold": int(fold_i),
            "train_cycles": train_cycles,
            "test_cycles": test_cycles,
            "n_train_blocks": int(len(train_idx)),
            "n_test_blocks": int(len(test_idx)),
            "train_class_counts": csv_json(class_count_dict(data.y[train_idx], task)),
            "test_class_counts": csv_json(class_count_dict(data.y[test_idx], task)),
            "n_components": n_components,
            "model_parameters": int(model_parameters),
            "training_loss": final_training_loss,
            "final_trained_epochs": final_trained_epochs,
            "device": device,
            **metrics,
        }
    )
    prediction_row_accumulator.extend(
        prediction_rows(
            session=session,
            task=task,
            method=method,
            seed=seed,
            fold=fold_i,
            test_idx=test_idx,
            y_true=y_test,
            y_pred=predictions,
            probabilities=probabilities,
            metadata=data.metadata,
        )
    )
    confusion_row_accumulator.extend(
        confusion_rows(
            session=session,
            task=task,
            method=method,
            seed=seed,
            fold=fold_i,
            y_true=y_test,
            y_pred=predictions,
        )
    )


def session_completeness_rows(
    data,
    methods: list[str],
    seeds: list[int],
    master_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        method_seeds = [0] if method in LINEAR_METHODS else seeds
        for seed in method_seeds:
            pred_subset = predictions_df[
                (predictions_df["method"] == method)
                & (predictions_df["seed"].astype(int) == int(seed))
            ]
            master_subset = master_df[
                (master_df["method"] == method)
                & (master_df["seed"].astype(int) == int(seed))
            ]
            fold_subset = fold_df[
                (fold_df["method"] == method)
                & (fold_df["seed"].astype(int) == int(seed))
            ]
            per_block_counts = pred_subset.groupby("block_id").size() if not pred_subset.empty else pd.Series(dtype=int)
            rows.append(
                {
                    "session": data.session,
                    "task": data.task,
                    "method": method,
                    "seed": int(seed),
                    "expected_n_blocks": int(data.n_blocks),
                    "expected_n_folds": int(len(grouped_cv_splits(data.groups))),
                    "master_row_present": bool(len(master_subset) == 1),
                    "n_fold_rows": int(len(fold_subset)),
                    "n_prediction_rows": int(len(pred_subset)),
                    "prediction_rows_unique_blocks": int(pred_subset["block_id"].nunique()) if not pred_subset.empty else 0,
                    "all_test_blocks_predicted_once": bool(
                        len(pred_subset) == data.n_blocks
                        and pred_subset["block_id"].nunique() == data.n_blocks
                        and (not per_block_counts.empty)
                        and per_block_counts.eq(1).all()
                    ),
                    "has_nan_or_inf_metric": bool(
                        master_subset[["accuracy", "balanced_accuracy", "macro_f1"]]
                        .apply(pd.to_numeric, errors="coerce")
                        .isna()
                        .any()
                        .any()
                    )
                    if not master_subset.empty
                    else True,
                    "chance_level": CHANCE_LEVEL,
                }
            )
    return rows


def run_session_task(
    *,
    args: argparse.Namespace,
    session: str,
    task: str,
    run_dir: Path,
    seeds: list[int],
    max_epochs: int,
    limit_folds: int | None,
) -> None:
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    session_dir = run_dir / f"session_{session}"
    protected_outputs = [
        session_dir / "master_summary.csv",
        session_dir / "fold_summary.csv",
        session_dir / "predictions.csv",
        session_dir / "normalization_audit.csv",
    ]
    if any(path.exists() for path in protected_outputs) and not args.overwrite:
        print(f"[{task} session {session}] existing outputs found; skipping without --overwrite")
        return
    session_dir.mkdir(parents=True, exist_ok=True)
    data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
    splits = grouped_cv_splits(data.groups, max_folds=args.max_folds)
    if limit_folds is not None:
        splits = splits[: int(limit_folds)]
    split_df = split_manifest(session, task, data.y, data.groups, splits=splits, max_folds=args.max_folds)
    split_df.to_csv(session_dir / "split_manifest.csv", index=False)

    config_payload = {
        "session": str(session),
        "task": task,
        "block_data_path": str(data.source_h5_path),
        "metadata_path": str(data.source_metadata_path),
        "input_unit": "block",
        "input_shape": [4, 128, 501],
        "data_version": "block_sequences_v1_clean4",
        "cv_group": "cycle",
        "n_splits": len(splits),
        "max_folds": int(args.max_folds),
        "methods": args.methods,
        "seeds": seeds,
        "pca_variance": float(args.pca_variance),
        "linear_standardize": True,
        "deep_config": {
            "optimizer": "adamw",
            "lr": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(min(args.batch_size, 16)),
            "max_epochs": int(max_epochs),
            "loss": "cross_entropy",
            "dropout": 0.25,
            "data_augmentation": False,
            "epoch_selection": "fixed_epochs_no_test_fold_selection",
        },
        "temporal1d_description": MODEL_DESCRIPTIONS["cnn2d_temporal1d"],
        "senior_model_reproduction_claim": "not_exact_reproduction_temporal_adaptation",
    }
    json_dump(session_dir / "config.json", config_payload)

    X_flat = preprocess_blocks_flat4(data.X)
    classes = np.asarray(sorted(TASK_CLASS_NAMES[task]), dtype=np.int64)
    deep_config = DeepTrainingConfig(
        optimizer="adamw",
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(min(args.batch_size, 16)),
        max_epochs=int(max_epochs),
    )
    fold_rows: list[dict[str, Any]] = []
    prediction_row_accumulator: list[dict[str, Any]] = []
    fold_confusion_rows: list[dict[str, Any]] = []
    overall_confusion_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    training_history_rows: list[dict[str, Any]] = []
    linear_fit_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    true_pred_by_method_seed: dict[tuple[str, int], dict[str, list[int]]] = {}
    fold_counts_by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}

    for method in args.methods:
        method_seeds = [0] if method in LINEAR_METHODS else seeds
        for seed in method_seeds:
            key = (method, int(seed))
            true_pred_by_method_seed[key] = {"true": [], "pred": []}
            fold_counts_by_method_seed[key] = []
            for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                y_train = data.y[train_idx]
                y_test = data.y[test_idx]
                if not valid_fold(y_train, y_test):
                    raise AssertionError(f"{session} {task} fold {fold_i} has fewer than two classes")
                train_cycles, test_cycles = split_cycle_info(data.groups, train_idx, test_idx)
                if method in LINEAR_METHODS:
                    base_method = "pca_lda" if method == "pca_lda_flat4" else "cpca_lda"
                    pred, n_components = fit_predict_linear(
                        base_method,
                        X_flat[train_idx],
                        y_train,
                        X_flat[test_idx],
                        pca_variance=float(args.pca_variance),
                        standardize=True,
                    )
                    linear_fit_rows.append(
                        {
                            "session": str(session),
                            "task": task,
                            "method": method,
                            "seed": int(seed),
                            "fold": int(fold_i),
                            "train_cycles": train_cycles,
                            "test_cycles": test_cycles,
                            "transform_fit_scope": "train_fold_only",
                            "test_used_for_pca_or_cpca_fit": False,
                            "test_used_for_lda_fit": False,
                            "flatten_order": "time_position_0_1_2_3_then_row_major_pixels",
                            "standardizer_fit_scope": "train_fold_only",
                            "pca_variance": float(args.pca_variance),
                            "n_components": int(n_components),
                        }
                    )
                    append_fold_outputs(
                        session=session,
                        task=task,
                        method=method,
                        seed=seed,
                        fold_i=fold_i,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        data=data,
                        predictions=pred,
                        probabilities=None,
                        n_components=int(n_components),
                        model_parameters=0,
                        final_training_loss=None,
                        final_trained_epochs=0,
                        device=None,
                        fold_rows=fold_rows,
                        prediction_row_accumulator=prediction_row_accumulator,
                        confusion_row_accumulator=fold_confusion_rows,
                    )
                elif method == "single_frame_late_fusion":
                    result = train_single_frame_late_fusion_fold(
                        data.X[train_idx],
                        y_train,
                        data.X[test_idx],
                        classes,
                        session=session,
                        task=task,
                        fold=fold_i,
                        seed=seed,
                        train_cycles=train_cycles,
                        test_cycles=test_cycles,
                        config=deep_config,
                        device=args.device,
                    )
                    normalization_rows.append(result.normalization_audit)
                    for history_row in result.history:
                        training_history_rows.append(
                            {
                                "session": str(session),
                                "task": task,
                                "method": method,
                                "seed": int(seed),
                                "fold": int(fold_i),
                                "train_cycles": train_cycles,
                                "test_cycles": test_cycles,
                                **history_row,
                            }
                        )
                    append_fold_outputs(
                        session=session,
                        task=task,
                        method=method,
                        seed=seed,
                        fold_i=fold_i,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        data=data,
                        predictions=result.predictions,
                        probabilities=result.probabilities,
                        n_components=None,
                        model_parameters=result.model_parameters,
                        final_training_loss=result.final_training_loss,
                        final_trained_epochs=result.final_trained_epochs,
                        device=result.device,
                        fold_rows=fold_rows,
                        prediction_row_accumulator=prediction_row_accumulator,
                        confusion_row_accumulator=fold_confusion_rows,
                    )
                    pred = result.predictions
                else:
                    result = train_sequence_fold(
                        method,
                        data.X[train_idx],
                        y_train,
                        data.X[test_idx],
                        classes,
                        session=session,
                        task=task,
                        fold=fold_i,
                        seed=seed,
                        train_cycles=train_cycles,
                        test_cycles=test_cycles,
                        config=deep_config,
                        device=args.device,
                    )
                    normalization_rows.append(result.normalization_audit)
                    for history_row in result.history:
                        training_history_rows.append(
                            {
                                "session": str(session),
                                "task": task,
                                "method": method,
                                "seed": int(seed),
                                "fold": int(fold_i),
                                "train_cycles": train_cycles,
                                "test_cycles": test_cycles,
                                **history_row,
                            }
                        )
                    if method in ORDER_SENSITIVE_METHODS:
                        order_payload = order_sensitivity_for_trained_sequence_model(
                            result.model,
                            result.X_test_normalized,
                            y_test,
                            classes,
                            device=result.device,
                            batch_size=deep_config.batch_size,
                        )
                        if "fixed_shuffle_order_ba" in order_payload:
                            order_payload["shuffled_order_ba"] = order_payload["fixed_shuffle_order_ba"]
                            order_payload["shuffled_order_accuracy"] = order_payload["fixed_shuffle_order_accuracy"]
                            order_payload["shuffled_order_macro_f1"] = order_payload["fixed_shuffle_order_macro_f1"]
                        order_rows.append(
                            {
                                "session": str(session),
                                "task": task,
                                "method": method,
                                "seed": int(seed),
                                "fold": int(fold_i),
                                "test_cycles": test_cycles,
                                "n_test_blocks": int(len(test_idx)),
                                **order_payload,
                            }
                        )
                    append_fold_outputs(
                        session=session,
                        task=task,
                        method=method,
                        seed=seed,
                        fold_i=fold_i,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        data=data,
                        predictions=result.predictions,
                        probabilities=result.probabilities,
                        n_components=None,
                        model_parameters=result.model_parameters,
                        final_training_loss=result.final_training_loss,
                        final_trained_epochs=result.final_trained_epochs,
                        device=result.device,
                        fold_rows=fold_rows,
                        prediction_row_accumulator=prediction_row_accumulator,
                        confusion_row_accumulator=fold_confusion_rows,
                    )
                    pred = result.predictions

                true_pred_by_method_seed[key]["true"].extend(y_test.astype(int).tolist())
                true_pred_by_method_seed[key]["pred"].extend(np.asarray(pred).astype(int).tolist())
                fold_counts_by_method_seed[key].append(
                    {
                        "n_train_blocks": int(len(train_idx)),
                        "n_test_blocks": int(len(test_idx)),
                    }
                )

    master_rows = []
    for (method, seed), values in true_pred_by_method_seed.items():
        y_true_all = np.asarray(values["true"], dtype=np.int64)
        y_pred_all = np.asarray(values["pred"], dtype=np.int64)
        metrics = metrics_with_flags(y_true_all, y_pred_all)
        count_rows = fold_counts_by_method_seed[(method, seed)]
        model_parameters = 0
        matching_fold_rows = [
            row for row in fold_rows if row["method"] == method and int(row["seed"]) == int(seed)
        ]
        if matching_fold_rows:
            model_parameters = int(matching_fold_rows[0]["model_parameters"])
        master_rows.append(
            {
                "session": str(session),
                "task": task,
                "method": method,
                "method_display": MODEL_DISPLAY_NAMES[method],
                "seed": int(seed),
                "n_cycles": int(data.n_cycles),
                "n_blocks": int(data.n_blocks),
                "n_train_blocks_mean": float(np.mean([row["n_train_blocks"] for row in count_rows])),
                "n_test_blocks_mean": float(np.mean([row["n_test_blocks"] for row in count_rows])),
                "model_parameters": int(model_parameters),
                "training_loss": float(np.nanmean([row["training_loss"] for row in matching_fold_rows]))
                if matching_fold_rows and any(pd.notna(row["training_loss"]) for row in matching_fold_rows)
                else np.nan,
                "final_trained_epochs": int(np.nanmax([row["final_trained_epochs"] for row in matching_fold_rows]))
                if matching_fold_rows
                else 0,
                "n_test_predictions": int(len(y_true_all)),
                "chance_level": CHANCE_LEVEL,
                **metrics,
            }
        )
        overall_confusion_rows.extend(
            confusion_rows(
                session=session,
                task=task,
                method=method,
                seed=seed,
                fold=None,
                y_true=y_true_all,
                y_pred=y_pred_all,
            )
        )

    master_df = pd.DataFrame(master_rows)
    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_row_accumulator)
    confusion_df = pd.DataFrame(overall_confusion_rows + fold_confusion_rows)
    normalization_df = pd.DataFrame(normalization_rows)
    training_history_df = pd.DataFrame(training_history_rows)
    linear_fit_df = pd.DataFrame(linear_fit_rows)
    order_df = pd.DataFrame(order_rows)
    completeness_df = pd.DataFrame(
        session_completeness_rows(data, list(args.methods), seeds, master_df, fold_df, predictions_df)
    )

    master_df.to_csv(session_dir / "master_summary.csv", index=False)
    fold_df.to_csv(session_dir / "fold_summary.csv", index=False)
    predictions_df.to_csv(session_dir / "predictions.csv", index=False)
    confusion_df.to_csv(session_dir / "confusion_matrices.csv", index=False)
    normalization_df.to_csv(session_dir / "normalization_audit.csv", index=False)
    training_history_df.to_csv(session_dir / "training_history.csv", index=False)
    linear_fit_df.to_csv(session_dir / "linear_fit_audit.csv", index=False)
    order_df.to_csv(session_dir / "order_sensitivity.csv", index=False)
    completeness_df.to_csv(session_dir / "multiframe_completeness_report.csv", index=False)
    print(f"[{task} session {session}] wrote {session_dir}")


def read_existing_session_csvs(run_dir: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob(f"session_*/{filename}")):
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_task_outputs(
    *,
    run_dir: Path,
    task: str,
    sessions: list[str],
    methods: list[str],
    seeds: list[int],
) -> None:
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    master = read_existing_session_csvs(run_dir, "master_summary.csv")
    fold_summary = read_existing_session_csvs(run_dir, "fold_summary.csv")
    predictions = read_existing_session_csvs(run_dir, "predictions.csv")
    confusion = read_existing_session_csvs(run_dir, "confusion_matrices.csv")
    normalization = read_existing_session_csvs(run_dir, "normalization_audit.csv")
    order_df = read_existing_session_csvs(run_dir, "order_sensitivity.csv")
    session_completeness = read_existing_session_csvs(run_dir, "multiframe_completeness_report.csv")

    if not master.empty:
        master.to_csv(aggregate_dir / "multiframe_master_summary.csv", index=False)
        method_summary_table(master).to_csv(aggregate_dir / "multiframe_method_summary.csv", index=False)
        seed_mean_summary(master).to_csv(aggregate_dir / "multiframe_seed_mean_summary.csv", index=False)
        vs_singleframe_reference(master).to_csv(aggregate_dir / "multiframe_vs_singleframe_reference.csv", index=False)
    else:
        pd.DataFrame().to_csv(aggregate_dir / "multiframe_master_summary.csv", index=False)
        pd.DataFrame().to_csv(aggregate_dir / "multiframe_method_summary.csv", index=False)
        pd.DataFrame().to_csv(aggregate_dir / "multiframe_vs_singleframe_reference.csv", index=False)

    fold_summary.to_csv(aggregate_dir / "multiframe_fold_summary.csv", index=False)
    predictions.to_csv(aggregate_dir / "multiframe_predictions.csv", index=False)
    confusion.to_csv(aggregate_dir / "multiframe_confusion_matrices.csv", index=False)
    normalization.to_csv(aggregate_dir / "normalization_audit.csv", index=False)
    order_df.to_csv(aggregate_dir / "multiframe_order_sensitivity.csv", index=False)
    if not session_completeness.empty:
        completeness = session_completeness.copy()
    else:
        completeness = completeness_report(
            task=task,
            sessions=sessions,
            methods=methods,
            seeds=seeds,
            master=master,
            fold_summary=fold_summary,
            predictions=predictions,
        )
    completeness.to_csv(aggregate_dir / "multiframe_completeness_report.csv", index=False)

    plot_paths = make_all_plots(master, order_df, confusion, task, aggregate_dir)
    plot_manifest = pd.DataFrame({"path": [str(path) for path in plot_paths]})
    plot_manifest.to_csv(aggregate_dir / "multiframe_plot_manifest.csv", index=False)
    print(f"[aggregate {task}] wrote {aggregate_dir}")


def run_benchmark(args: argparse.Namespace) -> None:
    sessions, tasks, seeds, max_epochs, limit_folds = resolve_sessions_tasks(args)
    for task in tasks:
        run_dir = task_run_dir(args, task)
        ensure_output_run_is_safe(run_dir, args.output_root)
        run_dir.mkdir(parents=True, exist_ok=True)
        parameter_path = run_dir / "model_parameter_audit.csv"
        senior_path = run_dir / "senior_code_input_audit.csv"
        if args.overwrite or not parameter_path.exists():
            pd.DataFrame(parameter_audit_rows(list(args.methods))).to_csv(parameter_path, index=False)
        if args.overwrite or not senior_path.exists():
            pd.DataFrame(senior_code_audit_rows()).to_csv(senior_path, index=False)
        for session in sessions:
            run_session_task(
                args=args,
                session=session,
                task=task,
                run_dir=run_dir,
                seeds=seeds,
                max_epochs=max_epochs,
                limit_folds=limit_folds,
            )
        aggregate_task_outputs(
            run_dir=run_dir,
            task=task,
            sessions=sessions,
            methods=list(args.methods),
            seeds=seeds,
        )


def run_aggregate_only(args: argparse.Namespace) -> None:
    sessions, tasks, seeds, _, _ = resolve_sessions_tasks(args)
    for task in tasks:
        run_dir = task_run_dir(args, task)
        aggregate_task_outputs(
            run_dir=run_dir,
            task=task,
            sessions=sessions,
            methods=list(args.methods),
            seeds=seeds,
        )


def main() -> None:
    args = parse_args()
    if args.stage == "dry-run":
        run_dry_run(args)
    elif args.stage in {"smoke", "benchmark"}:
        run_benchmark(args)
    elif args.stage == "aggregate-only":
        run_aggregate_only(args)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
