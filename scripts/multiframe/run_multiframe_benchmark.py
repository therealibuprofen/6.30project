#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
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
    block_type_accuracy,
    completeness_report,
    confusion_rows,
    method_summary_table,
    metrics_with_flags,
    order_sensitivity_oof_summary,
    overfitting_audit_tables,
    prediction_rows,
    seed_mean_summary,
    vs_singleframe_reference,
)
from ultrasound_decoding.multiframe.models import (
    LATE_FUSION_METHODS,
    LINEAR_METHODS,
    METHOD_USES_TEMPORAL_ORDER,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    MULTIFRAME_METHODS,
    NEURAL_METHODS,
    ORDER_SENSITIVE_METHODS,
    SEQUENCE_DEEP_METHODS,
    build_multiframe_model,
    count_trainable_parameters,
    model_architecture_config,
    model_shape_audit,
)
from ultrasound_decoding.multiframe.plotting import make_all_plots
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    order_sensitivity_for_trained_sequence_model,
    save_fold_checkpoint,
    train_sequence_fold,
    train_single_frame_late_fusion_fold,
)


DEFAULT_METHODS = list(MULTIFRAME_METHODS)
DEFAULT_SEEDS = [0, 1, 2]
FIXED_SHUFFLE = "2,0,3,1"
CONFIG_VERSION = "multiframe_benchmark_v2_fcnn_checkpoints_order_predictions"


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
    parser.add_argument(
        "--reuse-compatible-results",
        action="store_true",
        help="Reuse existing session outputs only after checking split, data, optimizer, seed, and config compatibility.",
    )
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


def current_code_version() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_frame(value: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in value.columns]
    out = value[available].copy()
    for column in available:
        out[column] = out[column].astype(str)
    return out.sort_values(available).reset_index(drop=True)


def verify_compatible_existing_session(
    *,
    session_dir: Path,
    config_payload: dict[str, Any],
    split_df: pd.DataFrame,
    methods: list[str],
    seeds: list[int],
) -> None:
    differences: list[str] = []
    config_path = session_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"cannot reuse {session_dir}: missing config.json")
    existing = read_json(config_path)
    config_payload = json.loads(json.dumps(config_payload, ensure_ascii=False))

    for key in ["session", "task", "input_unit", "input_shape", "data_version", "cv_group", "max_folds"]:
        if existing.get(key) != config_payload.get(key):
            differences.append(f"{key}: existing={existing.get(key)!r} planned={config_payload.get(key)!r}")
    if existing.get("config_version") != config_payload.get("config_version"):
        differences.append(
            f"config_version: existing={existing.get('config_version')!r} planned={config_payload.get('config_version')!r}"
        )
    if list(existing.get("methods", [])) != list(config_payload.get("methods", [])):
        differences.append(f"methods: existing={existing.get('methods')!r} planned={config_payload.get('methods')!r}")
    if list(existing.get("seeds", [])) != list(config_payload.get("seeds", [])):
        differences.append(f"seeds: existing={existing.get('seeds')!r} planned={config_payload.get('seeds')!r}")
    existing_deep = existing.get("deep_config", {})
    planned_deep = config_payload.get("deep_config", {})
    for key in ["optimizer", "lr", "weight_decay", "batch_size", "max_epochs", "loss", "dropout"]:
        if existing_deep.get(key) != planned_deep.get(key):
            differences.append(f"deep_config.{key}: existing={existing_deep.get(key)!r} planned={planned_deep.get(key)!r}")
    if existing.get("model_architecture") != config_payload.get("model_architecture"):
        differences.append("model_architecture differs")

    split_path = session_dir / "split_manifest.csv"
    if not split_path.exists():
        differences.append("missing split_manifest.csv")
    else:
        existing_split = pd.read_csv(split_path)
        split_columns = ["session", "task", "fold", "train_cycles", "test_cycles", "n_train_blocks", "n_test_blocks"]
        if not _canonical_frame(existing_split, split_columns).equals(_canonical_frame(split_df, split_columns)):
            differences.append("cycle split manifest differs")

    master_path = session_dir / "master_summary.csv"
    fold_path = session_dir / "fold_summary.csv"
    pred_path = session_dir / "predictions.csv"
    for path in [master_path, fold_path, pred_path]:
        if not path.exists():
            differences.append(f"missing {path.name}")
    if master_path.exists() and fold_path.exists() and pred_path.exists():
        master = pd.read_csv(master_path)
        fold_summary = pd.read_csv(fold_path)
        predictions = pd.read_csv(pred_path)
        for method in methods:
            method_seeds = [0] if method in LINEAR_METHODS else seeds
            for seed in method_seeds:
                master_subset = master[(master["method"] == method) & (master["seed"].astype(int) == int(seed))]
                fold_subset = fold_summary[(fold_summary["method"] == method) & (fold_summary["seed"].astype(int) == int(seed))]
                pred_subset = predictions[(predictions["method"] == method) & (predictions["seed"].fillna(0).astype(int) == int(seed))]
                if len(master_subset) != 1:
                    differences.append(f"missing master row for {method} seed {seed}")
                if len(fold_subset) != len(split_df):
                    differences.append(f"fold count mismatch for {method} seed {seed}")
                expected_test_blocks = int(split_df["n_test_blocks"].sum())
                if len(pred_subset) != expected_test_blocks:
                    differences.append(f"prediction count mismatch for {method} seed {seed}")
    if differences:
        joined = "\n  - ".join(differences)
        raise ValueError(f"existing outputs in {session_dir} are not reuse-compatible:\n  - {joined}")


def sampling_time_audit_tables(project_dir: Path, data_dir: Path, sessions: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grid_to_blocks: dict[tuple[str, str], set[str]] = {}
    session_data = [
        load_block_sequence_session(project_dir, session, "binary", data_dir=data_dir)
        for session in sessions
    ]
    for data in session_data:
        for row_i, row in data.metadata.reset_index(drop=True).iterrows():
            signature = ",".join(f"{float(value):g}" for value in data.clean4_relative_time_s[row_i].tolist())
            grid_to_blocks.setdefault((str(data.session), signature), set()).add(str(row["block_name"]))

    for data in session_data:
        frame = data.metadata.reset_index(drop=True).copy()
        frame["_time_grid_signature"] = [
            ",".join(f"{float(value):g}" for value in data.clean4_relative_time_s[row_i].tolist())
            for row_i in range(len(frame))
        ]
        for (block_name, binary_label), group in frame.groupby(["block_name", "binary_label_name"], sort=True):
            signatures = sorted(group["_time_grid_signature"].astype(str).unique().tolist())
            all_times = [
                [float(value) for value in data.clean4_relative_time_s[int(idx)].tolist()]
                for idx in group.index.tolist()
            ]
            rows.append(
                {
                    "session": str(data.session),
                    "block_name": str(block_name),
                    "binary_label": str(binary_label),
                    "n_blocks": int(len(group)),
                    "unique_clean4_relative_time_s": csv_json(all_times[:1] if len(signatures) == 1 else sorted({tuple(v) for v in all_times})),
                    "min_relative_time_s": float(np.min(np.asarray(all_times, dtype=float))),
                    "max_relative_time_s": float(np.max(np.asarray(all_times, dtype=float))),
                    "time_grid_signature": signatures[0] if len(signatures) == 1 else csv_json(signatures),
                    "time_grid_shared_with_other_blocks": bool(
                        any(len(grid_to_blocks[(str(data.session), signature)]) > 1 for signature in signatures)
                    ),
                }
            )
    audit = pd.DataFrame(rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_version": "block_sequences_v1_clean4",
        "sessions": [str(session) for session in sessions],
        "interpretation": (
            "Observed grids are reported as nominal clean4 relative times. They are not automatically "
            "classified as label leakage; they reflect the original 4-second sampling grid and 30-second block boundary phase."
        ),
        "observed_time_grids_by_block_name": {},
    }
    if not audit.empty:
        for block_name, group in audit.groupby("block_name", sort=True):
            summary["observed_time_grids_by_block_name"][str(block_name)] = sorted(
                group["time_grid_signature"].astype(str).unique().tolist()
            )
    return audit, summary


def legacy_checkpoint_manifest(fold_summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session",
        "task",
        "method",
        "seed",
        "fold",
        "checkpoint_path",
        "checkpoint_sha256",
        "train_cycles",
        "test_cycles",
        "normalization_shape",
        "status",
    ]
    if fold_summary.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for _, row in fold_summary.iterrows():
        method = str(row["method"])
        if method not in NEURAL_METHODS:
            continue
        rows.append(
            {
                "session": str(row["session"]),
                "task": row["task"],
                "method": method,
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "checkpoint_path": "",
                "checkpoint_sha256": "",
                "train_cycles": row.get("train_cycles", ""),
                "test_cycles": row.get("test_cycles", ""),
                "normalization_shape": "",
                "status": "not_available_for_legacy_run",
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
        uses_temporal_order = bool(METHOD_USES_TEMPORAL_ORDER.get(method, False))
        if method in LINEAR_METHODS:
            rows.append(
                {
                    "method": method,
                    "method_display": MODEL_DISPLAY_NAMES[method],
                    "method_description": MODEL_DESCRIPTIONS[method],
                    "model_parameters": 0,
                    "input_shape": "[B,4,128,501] flattened to [B,256512]",
                    "encoder_feature_dim": "",
                    "frame_feature_dim": "",
                    "temporal_length": 4,
                    "block_temporal_length": 4,
                    "uses_temporal_order": uses_temporal_order,
                    "temporal_adaptation": uses_temporal_order,
                    "shared_frame_encoder_weights": False,
                    "uses_fcnn_paper_32": False,
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
                "frame_feature_dim": int(audit.frame_feature_dim or audit.encoder_feature_dim),
                "temporal_length": int(audit.temporal_length),
                "block_temporal_length": int(audit.temporal_length),
                "encoded_shape": csv_json(list(audit.encoded_shape)) if audit.encoded_shape is not None else "",
                "lstm_input_size": audit.lstm_input_size,
                "lstm_hidden_size": audit.lstm_hidden_size,
                "output_shape": csv_json(list(audit.output_shape)),
                "temporal_conv_axis": audit.temporal_conv_axis,
                "uses_temporal_order": uses_temporal_order,
                "temporal_adaptation": uses_temporal_order,
                "shared_frame_encoder_weights": method in SEQUENCE_DEEP_METHODS,
                "fcnn_frame_encoder_matches_official_single_frame_fcnn": method.startswith("fcnn"),
                "uses_fcnn_paper_32": False,
                "model_architecture": csv_json(model_architecture_config(method, n_classes=2)),
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
    parameter_rows = parameter_audit_rows(args.methods)
    pd.DataFrame(audit_rows).to_csv(out_dir / "stage0_block_data_audit.csv", index=False)
    pd.DataFrame(plan_rows).to_csv(out_dir / "stage0_training_plan.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(out_dir / "stage0_model_parameter_audit.csv", index=False)
    pd.DataFrame(senior_code_audit_rows()).to_csv(out_dir / "stage0_senior_code_input_audit.csv", index=False)
    implementation_lines = [
        "# Multiframe Implementation Audit",
        "",
        f"Created at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "- Static audit only; no model training is run in dry-run stage.",
        "- Existing `results/runs/multiframe/block_clean4_binary_v1/` outputs are not modified by dry-run.",
        "- Binary and stimulus_type data are loaded through `processed_data/block_sequences_v1/` and split by cycle.",
        "",
        "## FCNN Multiframe Methods",
        "",
        "- `fcnn_late_fusion`: official single-frame FCNN per frame, block probability = mean of frame probabilities.",
        "- `fcnn_meanpool`: shared FCNN frame encoder to 3D features, mean over time, `Linear(3, n_classes)`.",
        "- `fcnn_lstm`: shared FCNN frame encoder to 3D features, `LSTM(input_size=3, hidden_size=8)`, dropout, classifier.",
        "- `fcnn_paper_32` is not used by any multiframe FCNN method.",
        "",
        "## Model Parameter Audit",
        "",
    ]
    for row in parameter_rows:
        implementation_lines.append(
            f"- {row['method']}: parameters={row['model_parameters']}, "
            f"uses_temporal_order={row.get('uses_temporal_order')}, input={row.get('input_shape')}, output={row.get('output_shape', '')}"
        )
    (out_dir / "implementation_audit.md").write_text("\n".join(implementation_lines) + "\n", encoding="utf-8")
    print(f"[dry-run] block data audit: {out_dir / 'stage0_block_data_audit.csv'}")
    print(f"[dry-run] training plan: {out_dir / 'stage0_training_plan.csv'}")
    print(f"[dry-run] model parameters: {out_dir / 'stage0_model_parameter_audit.csv'}")
    print(f"[dry-run] senior code audit: {out_dir / 'stage0_senior_code_input_audit.csv'}")
    print(f"[dry-run] implementation audit: {out_dir / 'implementation_audit.md'}")


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
    data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
    splits = grouped_cv_splits(data.groups, max_folds=args.max_folds)
    if limit_folds is not None:
        splits = splits[: int(limit_folds)]
    split_df = split_manifest(session, task, data.y, data.groups, splits=splits, max_folds=args.max_folds)

    config_payload = {
        "config_version": CONFIG_VERSION,
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
        "class_mapping": TASK_CLASS_NAMES[task],
        "pca_variance": float(args.pca_variance),
        "linear_standardize": True,
        "normalization_protocol": "arcsinh_then_train_fold_all_frames_pixel_zscore",
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
        "model_architecture": {
            method: model_architecture_config(method, n_classes=len(TASK_CLASS_NAMES[task]))
            for method in args.methods
        },
        "temporal1d_description": MODEL_DESCRIPTIONS["cnn2d_temporal1d"],
        "senior_model_reproduction_claim": "not_exact_reproduction_temporal_adaptation",
    }

    protected_outputs = [
        session_dir / "master_summary.csv",
        session_dir / "fold_summary.csv",
        session_dir / "predictions.csv",
        session_dir / "normalization_audit.csv",
    ]
    if any(path.exists() for path in protected_outputs) and not args.overwrite:
        if args.reuse_compatible_results:
            verify_compatible_existing_session(
                session_dir=session_dir,
                config_payload=config_payload,
                split_df=split_df,
                methods=list(args.methods),
                seeds=seeds,
            )
            print(f"[{task} session {session}] reuse-compatible outputs found; reusing existing files")
            return
        raise FileExistsError(
            f"[{task} session {session}] existing outputs found in {session_dir}; "
            "pass --reuse-compatible-results to verify and reuse, or --overwrite to rerun."
        )

    session_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(session_dir / "split_manifest.csv", index=False)
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
    order_prediction_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    true_pred_by_method_seed: dict[tuple[str, int], dict[str, list[int]]] = {}
    fold_counts_by_method_seed: dict[tuple[str, int], list[dict[str, Any]]] = {}
    code_version = current_code_version()

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
                elif method in LATE_FUSION_METHODS:
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
                        method=method,
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
                    checkpoint_rows.append(
                        save_fold_checkpoint(
                            session_dir / "checkpoints" / method / f"seed_{seed}" / f"fold_{fold_i}" / "checkpoint.pt",
                            result,
                            classes=classes,
                            session=session,
                            task=task,
                            seed=seed,
                            fold=fold_i,
                            train_cycles=train_cycles,
                            test_cycles=test_cycles,
                            config=deep_config,
                            code_version=code_version,
                        )
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
                            session=session,
                            task=task,
                            method=method,
                            seed=seed,
                            fold=fold_i,
                            test_idx=test_idx,
                            metadata=data.metadata,
                            class_names=data.class_names,
                            include_prediction_rows=True,
                        )
                        order_prediction_rows.extend(order_payload.pop("prediction_rows", []))
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
                    checkpoint_rows.append(
                        save_fold_checkpoint(
                            session_dir / "checkpoints" / method / f"seed_{seed}" / f"fold_{fold_i}" / "checkpoint.pt",
                            result,
                            classes=classes,
                            session=session,
                            task=task,
                            seed=seed,
                            fold=fold_i,
                            train_cycles=train_cycles,
                            test_cycles=test_cycles,
                            config=deep_config,
                            code_version=code_version,
                        )
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
    order_predictions_df = pd.DataFrame(order_prediction_rows)
    order_oof_df = order_sensitivity_oof_summary(order_predictions_df)
    checkpoint_df = pd.DataFrame(checkpoint_rows)
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
    order_predictions_df.to_csv(session_dir / "order_sensitivity_predictions.csv", index=False)
    order_oof_df.to_csv(session_dir / "order_sensitivity_oof_summary.csv", index=False)
    checkpoint_df.to_csv(session_dir / "checkpoint_manifest.csv", index=False)
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
    data_dir: Path,
) -> None:
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    master = read_existing_session_csvs(run_dir, "master_summary.csv")
    fold_summary = read_existing_session_csvs(run_dir, "fold_summary.csv")
    predictions = read_existing_session_csvs(run_dir, "predictions.csv")
    confusion = read_existing_session_csvs(run_dir, "confusion_matrices.csv")
    normalization = read_existing_session_csvs(run_dir, "normalization_audit.csv")
    training_history = read_existing_session_csvs(run_dir, "training_history.csv")
    order_df = read_existing_session_csvs(run_dir, "order_sensitivity.csv")
    order_predictions = read_existing_session_csvs(run_dir, "order_sensitivity_predictions.csv")
    order_oof = read_existing_session_csvs(run_dir, "order_sensitivity_oof_summary.csv")
    checkpoint_manifest = read_existing_session_csvs(run_dir, "checkpoint_manifest.csv")
    session_completeness = read_existing_session_csvs(run_dir, "multiframe_completeness_report.csv")
    if checkpoint_manifest.empty:
        checkpoint_manifest = legacy_checkpoint_manifest(fold_summary)
    if order_oof.empty and not order_predictions.empty:
        order_oof = order_sensitivity_oof_summary(order_predictions)
    overfitting_audit, overfitting_summary = overfitting_audit_tables(fold_summary, training_history)
    block_type_df = block_type_accuracy(predictions)

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
    training_history.to_csv(aggregate_dir / "training_history.csv", index=False)
    order_df.to_csv(aggregate_dir / "multiframe_order_sensitivity.csv", index=False)
    order_predictions.to_csv(aggregate_dir / "order_sensitivity_predictions.csv", index=False)
    order_oof.to_csv(aggregate_dir / "order_sensitivity_oof_summary.csv", index=False)
    checkpoint_manifest.to_csv(aggregate_dir / "checkpoint_manifest.csv", index=False)
    overfitting_audit.to_csv(aggregate_dir / "overfitting_audit.csv", index=False)
    overfitting_summary.to_csv(aggregate_dir / "overfitting_method_summary.csv", index=False)
    block_type_df.to_csv(aggregate_dir / "block_type_accuracy.csv", index=False)
    sampling_audit, sampling_summary = sampling_time_audit_tables(PROJECT_DIR, data_dir, sessions)
    sampling_audit.to_csv(aggregate_dir / "sampling_time_audit.csv", index=False)
    json_dump(aggregate_dir / "sampling_time_audit_summary.json", sampling_summary)
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

    plot_paths = make_all_plots(
        master,
        order_df,
        confusion,
        task,
        aggregate_dir,
        overfitting_summary=overfitting_summary,
        block_type_df=block_type_df,
    )
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
            data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR),
        )


def run_aggregate_only(args: argparse.Namespace) -> None:
    sessions, tasks, seeds, _, _ = resolve_sessions_tasks(args)
    for task in tasks:
        run_dir = task_run_dir(args, task)
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(parameter_audit_rows(list(args.methods))).to_csv(run_dir / "model_parameter_audit.csv", index=False)
        aggregate_task_outputs(
            run_dir=run_dir,
            task=task,
            sessions=sessions,
            methods=list(args.methods),
            seeds=seeds,
            data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR),
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
