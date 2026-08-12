#!/usr/bin/env python3
"""Frozen SmallCNN masked-reconstruction within-session benchmark v1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
    split_manifest,
)
from ultrasound_decoding.multiframe.models import (
    CNN2DMeanPool,
    SmallCNNFrameEncoder,
    count_trainable_parameters,
    encoder_architecture_signature,
)
from ultrasound_decoding.multiframe.training import DeepTrainingConfig
from ultrasound_decoding.ssl_masked import (
    MASK_BLOCK_SIZE,
    MASK_RATIO,
    SSL_CONDITIONS,
    SSL_SEEDS,
    SSLPretrainingConfig,
    SmallCNNReconstructionDecoder,
    assert_within_session_scope,
    count_parameters,
    fixed_ssl_validation_cycles,
    load_full_cycle_frames,
    load_ssl_encoder_checkpoint,
    missing_formal_outputs,
    pretrain_masked_smallcnn,
    save_ssl_encoder_checkpoint,
    train_downstream_fold,
)
from ultrasound_decoding.ssl_reporting import (
    generalization_gap_summary,
    make_required_plots,
    paired_ssl_improvements,
    plot_single_reconstruction_qc,
    session_level_metrics,
    statistical_test_tables,
)


RUN_NAME = "ssl_masked_smallcnn_clean4_9sessions_v1"
HISTORICAL_RUN_NAMES = {
    "binary": "block_clean4_binary_all_models_9sessions_v1",
    "stimulus_type": "block_clean4_stimulus_type_all_models_9sessions_v1",
}
SUPERVISED_CONFIG = DeepTrainingConfig(
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=16,
    max_epochs=40,
    dropout=0.25,
    loss="cross_entropy",
)
SSL_CONFIG = SSLPretrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("audit", "baseline", "pilot", "benchmark", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument(
        "--historical-root",
        type=Path,
        default=PROJECT_DIR / "results" / "runs" / "multiframe",
    )
    parser.add_argument("--historical-tolerance", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def log(message: str, output_dir: Path) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def ensure_output_tree(output_dir: Path) -> None:
    for relative in (
        "audit",
        "pretraining/checkpoints",
        "downstream/training_curves",
        "summaries",
        "figures/reconstruction_qc",
        "figures/binary",
        "figures/stimulus_type",
        "figures/overfitting",
        "figures/seed_stability",
        "report",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    view = frame if columns is None else frame[columns]
    if view.empty:
        return "_No rows._"
    header = "| " + " | ".join(str(value) for value in view.columns) + " |"
    separator = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = []
    for _, row in view.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}" if np.isfinite(value) else "NA")
            else:
                values.append(str(value).replace("|", "\\|"))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _historical_session_dir(root: Path, task: str, session: str) -> Path:
    return root / HISTORICAL_RUN_NAMES[task] / f"session_{session}"


def _historical_aggregate_paths(root: Path, task: str) -> dict[str, Path]:
    aggregate = root / HISTORICAL_RUN_NAMES[task] / "aggregate"
    return {
        "folds": aggregate / "multiframe_all_models_fold_summary.csv",
        "predictions": aggregate / "multiframe_all_models_predictions.csv",
        "master": aggregate / "multiframe_all_models_master_long.csv",
        "overfitting": aggregate / "multiframe_all_models_overfitting_audit.csv",
        "parameters": aggregate / "multiframe_all_models_parameter_audit.csv",
        "completeness": aggregate / "multiframe_all_models_completeness_report.csv",
    }


def _fold_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["session", "task", "fold", "train_cycles", "test_cycles", "n_train_blocks", "n_test_blocks"]
    output = frame[columns].copy()
    for column in columns:
        output[column] = output[column].astype(str)
    return output.sort_values(columns).reset_index(drop=True)


def run_static_audit(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    ensure_output_tree(output_dir)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    architecture_model = CNN2DMeanPool(n_classes=2)
    encoder = architecture_model.encoder
    decoder_model = SmallCNNReconstructionDecoder()
    architecture = [
        "# SmallCNN Architecture Audit",
        "",
        f"- Source file: `{PROJECT_DIR / 'src/ultrasound_decoding/multiframe/models.py'}`",
        "- Class name: `SmallCNNFrameEncoder` (used by `CNN2DMeanPool`)",
        "- Encoder architecture: `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Conv2d -> BatchNorm2d -> ReLU -> AdaptiveAvgPool2d -> Flatten`",
        "- Channels: `1 -> 8 -> 16`",
        "- Kernel sizes: `(5,9)`, `(5,7)`",
        "- Convolution stride: `(1,1)` for both convolutions",
        "- Pooling: `MaxPool2d(2,4)`, then adaptive average pool to `(4,8)`",
        "- Final spatial feature map: `16 x 4 x 8`",
        "- Flattened feature dimension: `512`",
        f"- Encoder parameter count: `{count_parameters(encoder)}`",
        f"- Feature-mean classifier parameter count (2 classes): `{count_parameters(architecture_model)}`",
        f"- SSL-only decoder parameter count: `{count_parameters(decoder_model)}`",
        "- Decoder: `Conv2d(16,8,3,pad=1) -> ReLU -> bilinear upsample(128,501) -> Conv2d(8,1,3,pad=1)`",
        "- Decoder is discarded before downstream classification.",
        f"- Audited signature: `{encoder_architecture_signature()}`",
    ]
    (output_dir / "audit/smallcnn_architecture_audit.md").write_text("\n".join(architecture) + "\n", encoding="utf-8")

    preprocessing = [
        "# Frame Preprocessing Audit",
        "",
        "## Historical supervised SmallCNN feature-mean path",
        "",
        "- Source: `normalize_blocks_train_fold_only_with_stats` in `multiframe/training.py`.",
        "- Input dtype from clean4 HDF5: float32.",
        "- Transform: `arcsinh`, then pixel-wise z-score.",
        "- Statistics: fitted on the outer training blocks and their four clean4 frames only.",
        "- Epsilon: 1e-6 added to pixel-wise standard deviation.",
        "- Clipping: none. Resize: none. Spatial orientation change: none.",
        "",
        "## SSL path",
        "",
        "- Input: all 30 raw frames in each SSL-train cycle from `/full/X_padded`, excluding padding.",
        "- Transform family is frozen to the same `arcsinh` plus train-only pixel z-score representation.",
        "- SSL normalization statistics are fitted only on SSL-train cycles; SSL-val and outer-test cycles are excluded.",
        "- Masking occurs after preprocessing and fills masked pixels with zero.",
        "- No clipping, resize, CLAHE, gamma, histogram equalization, sharpening, vesselness, or geometric augmentation.",
        "- Downstream normalization remains the historical task-specific clean4 train-fold normalization so RANDOM_INIT is unchanged.",
    ]
    (output_dir / "audit/frame_preprocessing_audit.md").write_text("\n".join(preprocessing) + "\n", encoding="utf-8")

    fold_rows: list[dict[str, Any]] = []
    volume_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    historical_aggregate_folds: dict[str, pd.DataFrame | None] = {}
    for task in EXPECTED_TASKS:
        aggregate_fold_path = _historical_aggregate_paths(args.historical_root, task)["folds"]
        historical_aggregate_folds[task] = (
            pd.read_csv(aggregate_fold_path) if aggregate_fold_path.exists() else None
        )
    for session in EXPECTED_SESSIONS:
        task_data = {
            task: load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            for task in EXPECTED_TASKS
        }
        task_splits = {
            task: grouped_cv_splits(task_data[task].groups, max_folds=10)
            for task in EXPECTED_TASKS
        }
        binary_manifest = split_manifest(
            session, "binary", task_data["binary"].y, task_data["binary"].groups, splits=task_splits["binary"]
        )
        stimulus_manifest = split_manifest(
            session, "stimulus_type", task_data["stimulus_type"].y, task_data["stimulus_type"].groups,
            splits=task_splits["stimulus_type"],
        )
        task_partitions_equal = binary_manifest[["fold", "train_cycles", "test_cycles"]].equals(
            stimulus_manifest[["fold", "train_cycles", "test_cycles"]]
        )
        if not task_partitions_equal:
            raise AssertionError(f"{session}: binary and stimulus_type folds differ")
        for task, manifest in (("binary", binary_manifest), ("stimulus_type", stimulus_manifest)):
            historical_manifest_path = _historical_session_dir(args.historical_root, task, session) / "split_manifest.csv"
            if historical_manifest_path.exists():
                historical = pd.read_csv(historical_manifest_path)
            elif historical_aggregate_folds[task] is not None:
                aggregate = historical_aggregate_folds[task]
                assert aggregate is not None
                historical = aggregate[
                    (aggregate["session"].astype(str) == str(session))
                    & (aggregate["task"] == task)
                    & (aggregate["method"] == "cnn2d_meanpool")
                    & (aggregate["seed"].astype(int) == 0)
                ][
                    ["session", "task", "fold", "train_cycles", "test_cycles", "n_train_blocks", "n_test_blocks"]
                ].drop_duplicates()
                historical_manifest_path = _historical_aggregate_paths(args.historical_root, task)["folds"]
            else:
                historical = None
            for _, row in manifest.iterrows():
                match = None
                if historical is not None:
                    candidates = historical[historical["fold"].astype(int) == int(row["fold"])]
                    match = candidates.iloc[0] if len(candidates) == 1 else None
                fold_rows.append({
                    **row.to_dict(),
                    "historical_manifest_path": str(historical_manifest_path),
                    "historical_manifest_exists": bool(historical_manifest_path.exists()),
                    "train_cycles_match": bool(match is not None and str(match["train_cycles"]) == str(row["train_cycles"])),
                    "test_cycles_match": bool(match is not None and str(match["test_cycles"]) == str(row["test_cycles"])),
                    "status": "PASS" if match is not None and str(match["train_cycles"]) == str(row["train_cycles"]) and str(match["test_cycles"]) == str(row["test_cycles"]) else "MISSING_OR_MISMATCHED_HISTORICAL_REFERENCE",
                })
        for fold_i, (binary_train, binary_test) in enumerate(task_splits["binary"], start=1):
            binary = task_data["binary"]
            stimulus = task_data["stimulus_type"]
            train_cycles = np.unique(binary.groups[binary_train])
            test_cycles = np.unique(binary.groups[binary_test])
            ssl_train_cycles, ssl_val_cycles = fixed_ssl_validation_cycles(train_cycles)
            stimulus_train = np.flatnonzero(np.isin(stimulus.groups, train_cycles))
            stimulus_test = np.flatnonzero(np.isin(stimulus.groups, test_cycles))
            ssl_cycles = set(ssl_train_cycles.tolist()) | set(ssl_val_cycles.tolist())
            test_set = set(test_cycles.tolist())
            volume_rows.append({
                "session": session,
                "fold": fold_i,
                "n_train_cycles": int(len(train_cycles)),
                "n_test_cycles": int(len(test_cycles)),
                "train_cycles": cycle_text(train_cycles),
                "test_cycles": cycle_text(test_cycles),
                "ssl_train_cycles": cycle_text(ssl_train_cycles),
                "ssl_val_cycles": cycle_text(ssl_val_cycles),
                "n_ssl_train_frames": int(len(ssl_train_cycles) * 30),
                "n_ssl_val_frames": int(len(ssl_val_cycles) * 30),
                "n_binary_train_blocks": int(len(binary_train)),
                "n_binary_test_blocks": int(len(binary_test)),
                "n_stimulus_type_train_blocks": int(len(stimulus_train)),
                "n_stimulus_type_test_blocks": int(len(stimulus_test)),
                "task_folds_identical": task_partitions_equal,
                "ssl_encoder_cache_shared_across_tasks": task_partitions_equal,
            })
            leakage_rows.append({
                "session": session,
                "fold": fold_i,
                "outer_train_cycles": cycle_text(train_cycles),
                "outer_test_cycles": cycle_text(test_cycles),
                "ssl_train_cycles": cycle_text(ssl_train_cycles),
                "ssl_val_cycles": cycle_text(ssl_val_cycles),
                "ssl_test_cycle_overlap": cycle_text(sorted(ssl_cycles & test_set)),
                "normalization_fit_cycles": cycle_text(ssl_train_cycles),
                "test_used_for_ssl": False,
                "test_used_for_ssl_validation": False,
                "test_used_for_ssl_normalization": False,
                "within_session_only": True,
                "status": "PASS" if not (ssl_cycles & test_set) else "FAIL",
            })
    write_csv(output_dir / "audit/fold_reproduction.csv", fold_rows)
    write_csv(output_dir / "audit/ssl_data_volume.csv", volume_rows)
    write_csv(output_dir / "audit/ssl_leakage_audit.csv", leakage_rows)

    seed_rows = []
    for seed in SSL_SEEDS:
        for condition in SSL_CONDITIONS:
            seed_rows.append({
                "seed": int(seed),
                "condition": condition,
                "same_fold": True,
                "same_downstream_dataloader_seed": int(seed),
                "same_head_initialization_seed": int(seed),
                "ssl_mask_seed": int(seed) if condition != "RANDOM_INIT" else "not_applicable",
                "post_hoc_seed_added": False,
            })
    write_csv(output_dir / "audit/seed_audit.csv", seed_rows)

    config_lines = [
        "# Frozen Configuration",
        "",
        f"- Sessions: `{list(EXPECTED_SESSIONS)}`",
        f"- Tasks: `{list(EXPECTED_TASKS)}`",
        "- Data builder: `load_block_sequence_session` from the existing clean4 benchmark.",
        "- Fold generator: `grouped_cv_splits(groups, max_folds=10)`.",
        "- Input: exactly `4 x 128 x 501` clean-middle frames for downstream classification.",
        "- Temporal fusion: shared SmallCNN frame encoder, arithmetic feature mean, linear head.",
        f"- SSL: `{asdict(SSL_CONFIG)}`",
        f"- Downstream: `{asdict(SUPERVISED_CONFIG)}`",
        f"- Fixed comparison seeds: `{list(SSL_SEEDS)}`",
        f"- Conditions: `{list(SSL_CONDITIONS)}`",
        "- Historical code has fixed 40-epoch training and no early stopping; `best_epoch=40` records that frozen rule.",
        "- No registration, cross-session data, ROI, searchlight, new task, new frame selector, or new supervised architecture.",
    ]
    (output_dir / "audit/config_freeze.md").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    log("Stage 0 static audit completed", output_dir)


def _historical_prerequisites(args: argparse.Namespace) -> tuple[bool, list[dict[str, Any]]]:
    rows = []
    all_available = True
    for task in EXPECTED_TASKS:
        run_dir = args.historical_root / HISTORICAL_RUN_NAMES[task]
        aggregate_paths = _historical_aggregate_paths(args.historical_root, task)
        aggregate_missing = [str(path) for path in aggregate_paths.values() if not path.exists()]
        for session in EXPECTED_SESSIONS:
            session_dir = run_dir / f"session_{session}"
            per_session_required = [
                session_dir / "config.json",
                session_dir / "split_manifest.csv",
                session_dir / "master_summary.csv",
                session_dir / "fold_summary.csv",
                session_dir / "predictions.csv",
                session_dir / "training_history.csv",
            ]
            per_session_missing = [str(path) for path in per_session_required if not path.exists()]
            layout = "per_session" if not per_session_missing else "aggregate"
            missing = [] if layout == "per_session" else aggregate_missing
            available = not missing
            all_available &= available
            rows.append({
                "session": session,
                "task": task,
                "historical_run_dir": str(run_dir),
                "historical_layout": layout,
                "status": "PENDING_REPRODUCTION" if available else "MISSING_HISTORICAL_REFERENCE",
                "missing_files": json.dumps(missing),
                "sample_ids_match": False,
                "cycle_ids_match": False,
                "labels_match": False,
                "folds_match": False,
                "config_match": False,
                "metrics_reproduced": False,
                "gate_pass": False,
            })
    return all_available, rows


def _historic_train_accuracy(session_dir: Path, seed: int, fold: int) -> float:
    history = pd.read_csv(session_dir / "training_history.csv")
    subset = history[
        (history["method"] == "cnn2d_meanpool")
        & (history["seed"].astype(int) == int(seed))
        & (history["fold"].astype(int) == int(fold))
    ].sort_values("epoch")
    return float(subset.iloc[-1]["train_accuracy"]) if len(subset) else float("nan")


def write_stage1_stop_report(output_dir: Path, prerequisite_rows: list[dict[str, Any]]) -> None:
    inventory = pd.DataFrame(prerequisite_rows)
    missing_runs = sorted(inventory["historical_run_dir"].astype(str).unique())
    lines = [
        "# SSL v1 Stage 1 Stop Report",
        "",
        "Status: **STOPPED BEFORE SSL PRETRAINING**.",
        "",
        "The preregistered historical baseline gate could not run because the two named nine-session historical benchmark directories are absent. The pipeline did not substitute smoke outputs, the seven-session legacy binary run, or newly generated values for the missing historical references.",
        "",
        "Missing named runs:",
        "",
        *[f"- `{value}`" for value in missing_runs],
        "",
        "Consequences:",
        "",
        "- No masked-reconstruction training was started.",
        "- No SSL checkpoint was written.",
        "- No Random/Frozen/Finetune downstream comparison was run.",
        "- No BA, gap, sign-flip, two-task-corrected, scenario, or cross-session recommendation is reported.",
        "",
        "See `audit/historical_baseline_reproduction.csv` for the missing file inventory, and `audit/ssl_data_volume.csv` for the already-audited fold/cycle/frame plan.",
    ]
    path = output_dir / "report/stage1_stop_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_historical_baseline_reproduction(args: argparse.Namespace) -> bool:
    """Rerun the exact historical random-init feature-mean jobs before SSL."""
    output_path = args.output_dir / "audit/historical_baseline_reproduction.csv"
    available, prerequisite_rows = _historical_prerequisites(args)
    if not available:
        write_csv(output_path, prerequisite_rows)
        write_stage1_stop_report(args.output_dir, prerequisite_rows)
        log(
            "Historical baseline gate FAILED: the two named 9-session all-model historical runs are absent; SSL was not started",
            args.output_dir,
        )
        return False

    rows: list[dict[str, Any]] = []
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    for task in EXPECTED_TASKS:
        aggregate_paths = _historical_aggregate_paths(args.historical_root, task)
        aggregate_tables = {
            key: pd.read_csv(path) for key, path in aggregate_paths.items()
        }
        for session in EXPECTED_SESSIONS:
            session_dir = _historical_session_dir(args.historical_root, task, session)
            if session_dir.exists():
                config = json.loads((session_dir / "config.json").read_text(encoding="utf-8"))
                historical_folds = pd.read_csv(session_dir / "fold_summary.csv")
                historical_predictions = pd.read_csv(session_dir / "predictions.csv")
                historical_split = pd.read_csv(session_dir / "split_manifest.csv")
                historical_master = pd.read_csv(session_dir / "master_summary.csv")
                historical_overfitting = pd.read_csv(session_dir / "training_history.csv")
                layout = "per_session"
            else:
                historical_folds = aggregate_tables["folds"]
                historical_folds = historical_folds[
                    (historical_folds["session"].astype(str) == str(session))
                    & (historical_folds["task"] == task)
                ].copy()
                historical_predictions = aggregate_tables["predictions"]
                historical_predictions = historical_predictions[
                    (historical_predictions["session"].astype(str) == str(session))
                    & (historical_predictions["task"] == task)
                ].copy()
                historical_master = aggregate_tables["master"]
                historical_master = historical_master[
                    (historical_master["session"].astype(str) == str(session))
                    & (historical_master["task"] == task)
                ].copy()
                historical_overfitting = aggregate_tables["overfitting"]
                historical_overfitting = historical_overfitting[
                    (historical_overfitting["session"].astype(str) == str(session))
                    & (historical_overfitting["task"] == task)
                ].copy()
                parameter_row = aggregate_tables["parameters"]
                parameter_row = parameter_row[parameter_row["method"] == "cnn2d_meanpool"]
                meanpool_folds = historical_folds[historical_folds["method"] == "cnn2d_meanpool"]
                inferred_seeds = sorted(meanpool_folds["seed"].astype(int).unique().tolist())
                inferred_epochs = sorted(meanpool_folds["final_trained_epochs"].astype(int).unique().tolist())
                inferred_parameters = sorted(meanpool_folds["model_parameters"].astype(int).unique().tolist())
                config = {
                    "deep_config": asdict(SUPERVISED_CONFIG),
                    "config_evidence": {
                        "seeds": inferred_seeds,
                        "final_trained_epochs": inferred_epochs,
                        "model_parameters": inferred_parameters,
                        "parameter_audit_rows": int(len(parameter_row)),
                    },
                }
                historical_split = (
                    meanpool_folds[meanpool_folds["seed"].astype(int) == inferred_seeds[0]][
                        ["session", "task", "fold", "train_cycles", "test_cycles", "n_train_blocks", "n_test_blocks"]
                    ]
                    .drop_duplicates()
                    .sort_values("fold")
                    .reset_index(drop=True)
                )
                layout = "aggregate"
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            splits = grouped_cv_splits(data.groups, max_folds=10)
            current_split = split_manifest(session, task, data.y, data.groups, splits=splits)
            folds_match = _fold_key_frame(current_split).equals(_fold_key_frame(historical_split))
            method_predictions = historical_predictions[historical_predictions["method"] == "cnn2d_meanpool"]
            sample_ids_match = set(method_predictions["block_id"].astype(str)) == set(data.metadata["block_id"].astype(str))
            truth_map = method_predictions.drop_duplicates("block_id").set_index("block_id")["truth"].astype(int)
            current_truth = data.metadata.assign(truth=data.y).set_index("block_id")["truth"].astype(int)
            labels_match = truth_map.sort_index().equals(current_truth.sort_index())
            cycle_map = method_predictions.drop_duplicates("block_id").set_index("block_id")["cycle"].astype(int)
            current_cycles = data.metadata.set_index("block_id")["cycle"].astype(int)
            cycles_match = cycle_map.sort_index().equals(current_cycles.sort_index())
            deep = config.get("deep_config", {})
            historical_config = DeepTrainingConfig(
                optimizer=str(deep.get("optimizer", "adamw")),
                lr=float(deep.get("lr", 1e-3)),
                weight_decay=float(deep.get("weight_decay", 1e-3)),
                batch_size=int(deep.get("batch_size", 16)),
                max_epochs=int(deep.get("max_epochs", 40)),
                dropout=float(deep.get("dropout", 0.25)),
                loss=str(deep.get("loss", "cross_entropy")),
            )
            config_match = historical_config == SUPERVISED_CONFIG
            if layout == "aggregate":
                evidence = config["config_evidence"]
                config_match = bool(
                    config_match
                    and evidence["seeds"] == [0, 1, 2]
                    and evidence["final_trained_epochs"] == [40]
                    and evidence["model_parameters"] == [5938]
                    and evidence["parameter_audit_rows"] == 1
                )
            seeds = sorted(
                historical_folds[historical_folds["method"] == "cnn2d_meanpool"]["seed"].astype(int).unique()
            )
            for seed in seeds:
                reproduced_truth: list[int] = []
                reproduced_predictions: list[int] = []
                reproduced_train_accuracies: list[float] = []
                for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                    old = historical_folds[
                        (historical_folds["method"] == "cnn2d_meanpool")
                        & (historical_folds["seed"].astype(int) == seed)
                        & (historical_folds["fold"].astype(int) == fold_i)
                    ]
                    if len(old) != 1:
                        raise AssertionError(f"historical fold row missing or duplicated: {session} {task} {seed} {fold_i}")
                    result = train_downstream_fold(
                        "RANDOM_INIT", data, train_idx, test_idx,
                        fold=fold_i, seed=seed, pretrained_encoder_state=None,
                        config=historical_config, device=args.device,
                    )
                    reproduced_truth.extend(data.y[test_idx].astype(int).tolist())
                    reproduced_predictions.extend(result.test_predictions.astype(int).tolist())
                    reproduced_train_accuracies.append(
                        float(result.history[-1]["train_accuracy_minibatch"])
                    )
                reproduced_metrics = classification_metrics(
                    np.asarray(reproduced_truth, dtype=np.int64),
                    np.asarray(reproduced_predictions, dtype=np.int64),
                )
                old_master = historical_master[
                    (historical_master["method"] == "cnn2d_meanpool")
                    & (historical_master["seed"].astype(int) == int(seed))
                ]
                if len(old_master) != 1:
                    raise AssertionError(
                        f"historical master row missing or duplicated: {session} {task} {seed}"
                    )
                old_master = old_master.iloc[0]
                old_overfit = historical_overfitting[
                    (historical_overfitting["method"] == "cnn2d_meanpool")
                    & (historical_overfitting["seed"].astype(int) == int(seed))
                ]
                if layout == "per_session":
                    old_train_accuracy = float(
                        old_overfit.sort_values(["fold", "epoch"])
                        .groupby("fold").tail(1)["train_accuracy"].mean()
                    )
                else:
                    old_train_accuracy = float(old_overfit["final_train_accuracy"].mean())
                ba_delta = reproduced_metrics["balanced_accuracy"] - float(old_master["balanced_accuracy"])
                accuracy_delta = reproduced_metrics["accuracy"] - float(old_master["accuracy"])
                train_accuracy = float(np.mean(reproduced_train_accuracies))
                train_delta = train_accuracy - old_train_accuracy
                metric_match = bool(
                    abs(ba_delta) <= args.historical_tolerance
                    and abs(accuracy_delta) <= args.historical_tolerance
                    and abs(train_delta) <= args.historical_tolerance
                )
                rows.append({
                    "session": session,
                    "task": task,
                    "seed": seed,
                    "historical_layout": layout,
                    "n_folds": len(splits),
                    "sample_ids_match": sample_ids_match,
                    "cycle_ids_match": cycles_match,
                    "labels_match": labels_match,
                    "folds_match": folds_match,
                    "number_of_folds_match": len(splits) == historical_split["fold"].nunique(),
                    "config_match": config_match,
                    "historical_test_BA": float(old_master["balanced_accuracy"]),
                    "reproduced_test_BA": reproduced_metrics["balanced_accuracy"],
                    "test_BA_delta": ba_delta,
                    "historical_accuracy": float(old_master["accuracy"]),
                    "reproduced_accuracy": reproduced_metrics["accuracy"],
                    "accuracy_delta": accuracy_delta,
                    "historical_train_accuracy": old_train_accuracy,
                    "reproduced_train_accuracy": train_accuracy,
                    "train_accuracy_delta": train_delta,
                    "tolerance": float(args.historical_tolerance),
                    "metrics_reproduced": metric_match,
                    "status": "PASS" if all((sample_ids_match, cycles_match, labels_match, folds_match, config_match, metric_match)) else "FAIL",
                })
                write_csv(output_path, rows)
    frame = pd.DataFrame(rows)
    gate = bool(len(frame) and (frame["status"] == "PASS").all())
    frame["gate_pass"] = gate
    write_csv(output_path, frame)
    log(f"Historical baseline reproduction gate {'PASSED' if gate else 'FAILED'}", args.output_dir)
    return gate


def baseline_gate_passed(output_dir: Path) -> bool:
    path = output_dir / "audit/historical_baseline_reproduction.csv"
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    return bool(len(frame) and "gate_pass" in frame and frame["gate_pass"].astype(bool).all())


def _load_all_session_frames(args: argparse.Namespace, session: str, cycles: np.ndarray):
    return load_full_cycle_frames(
        PROJECT_DIR, session, cycles, data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR)
    )


def _pretrain_one(
    args: argparse.Namespace,
    *,
    session: str,
    fold_i: int,
    seed: int,
    all_frames,
    train_cycles: np.ndarray,
    test_cycles: np.ndarray,
    root: Path,
) -> Path:
    checkpoint_path = root / "checkpoints" / f"session_{session}" / f"fold_{fold_i}" / f"seed_{seed}.pt"
    if checkpoint_path.exists() and not args.overwrite:
        _encoder, payload = load_ssl_encoder_checkpoint(checkpoint_path)
        expected = {
            "session": session,
            "fold": fold_i,
            "seed": seed,
            "outer_test_cycles": [int(value) for value in test_cycles],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise AssertionError(f"incompatible cached SSL checkpoint: {checkpoint_path}")
        return checkpoint_path
    ssl_train_cycles, ssl_val_cycles = fixed_ssl_validation_cycles(train_cycles)
    train_mask = np.isin(all_frames.cycles, ssl_train_cycles)
    val_mask = np.isin(all_frames.cycles, ssl_val_cycles)
    if np.any(np.isin(all_frames.cycles[train_mask | val_mask], test_cycles)):
        raise AssertionError("test cycle entered SSL arrays")
    result = pretrain_masked_smallcnn(
        all_frames.frames[train_mask],
        all_frames.frames[val_mask] if np.any(val_mask) else None,
        seed=seed,
        config=SSL_CONFIG,
        device=args.device,
    )
    checkpoint_row = save_ssl_encoder_checkpoint(
        checkpoint_path,
        result,
        session=session,
        fold=fold_i,
        seed=seed,
        ssl_train_cycles=ssl_train_cycles,
        ssl_val_cycles=ssl_val_cycles,
        outer_test_cycles=test_cycles,
        config=SSL_CONFIG,
    )
    losses_path = root / "reconstruction_losses.csv"
    existing = pd.read_csv(losses_path) if losses_path.exists() else pd.DataFrame()
    loss_rows = pd.DataFrame([
        {"session": session, "fold": fold_i, "seed": seed, **row}
        for row in result.history
    ])
    write_csv(losses_path, pd.concat([existing, loss_rows], ignore_index=True))
    checkpoint_manifest_path = root / "checkpoint_manifest.csv"
    existing_manifest = pd.read_csv(checkpoint_manifest_path) if checkpoint_manifest_path.exists() else pd.DataFrame()
    write_csv(checkpoint_manifest_path, pd.concat([existing_manifest, pd.DataFrame([checkpoint_row])], ignore_index=True))
    if fold_i == 1 and seed == SSL_SEEDS[0]:
        sample_local = int(result.qc["sample_index"])
        qc_dir = root.parent / "figures/reconstruction_qc"
        qc_dir.mkdir(parents=True, exist_ok=True)
        qc_path = qc_dir / f"session_{session}_qc.npz"
        np.savez_compressed(
            qc_path,
            original=result.qc["original"],
            masked=result.qc["masked"],
            reconstruction=result.qc["reconstruction"],
            mask=result.qc["mask"],
            cycle=int(all_frames.cycles[np.flatnonzero(train_mask)[sample_local]]),
            original_frame_index=int(all_frames.original_frame_indices[np.flatnonzero(train_mask)[sample_local]]),
        )
        plot_single_reconstruction_qc(qc_path, qc_dir / f"session_{session}_reconstruction_qc.png")
    return checkpoint_path


def run_pilot(args: argparse.Namespace) -> None:
    if not baseline_gate_passed(args.output_dir):
        raise RuntimeError("historical baseline gate has not passed; refusing pilot SSL")
    session = "710"
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    data = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=data_dir)
    train_idx, test_idx = grouped_cv_splits(data.groups, max_folds=10)[0]
    all_frames = _load_all_session_frames(args, session, np.unique(data.groups))
    checkpoint = _pretrain_one(
        args,
        session=session,
        fold_i=1,
        seed=SSL_SEEDS[0],
        all_frames=all_frames,
        train_cycles=np.unique(data.groups[train_idx]),
        test_cycles=np.unique(data.groups[test_idx]),
        root=args.output_dir / "pilot/pretraining",
    )
    encoder, _payload = load_ssl_encoder_checkpoint(checkpoint)
    pilot_rows = []
    for condition in SSL_CONDITIONS:
        result = train_downstream_fold(
            condition,
            data,
            train_idx,
            test_idx,
            fold=1,
            seed=SSL_SEEDS[0],
            pretrained_encoder_state=None if condition == "RANDOM_INIT" else encoder.state_dict(),
            config=SUPERVISED_CONFIG,
            device=args.device,
        )
        pilot_rows.append(result.metrics)
    write_csv(args.output_dir / "pilot/pilot_fold_metrics.csv", pilot_rows)
    log("Stage 2 pilot completed without parameter tuning", args.output_dir)


def _prediction_rows(data, test_idx, result, condition: str, seed: int, fold_i: int) -> list[dict[str, Any]]:
    rows = []
    for local_i, sample_i in enumerate(test_idx):
        meta = data.metadata.iloc[int(sample_i)]
        rows.append({
            "session": data.session,
            "task": data.task,
            "fold": fold_i,
            "seed": seed,
            "condition": condition,
            "sample_i": int(sample_i),
            "sample_id": str(meta["block_id"]),
            "cycle": int(meta["cycle"]),
            "label": int(data.y[sample_i]),
            "prediction": int(result.test_predictions[local_i]),
            "probability_class_0": float(result.test_probabilities[local_i, 0]),
            "probability_class_1": float(result.test_probabilities[local_i, 1]),
        })
    return rows


def run_formal_benchmark(args: argparse.Namespace) -> None:
    if not baseline_gate_passed(args.output_dir):
        raise RuntimeError("historical baseline gate has not passed; refusing formal SSL")
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    for session in EXPECTED_SESSIONS:
        binary = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=data_dir)
        stimulus = load_block_sequence_session(PROJECT_DIR, session, "stimulus_type", data_dir=data_dir)
        splits = grouped_cv_splits(binary.groups, max_folds=10)
        stimulus_splits = grouped_cv_splits(stimulus.groups, max_folds=10)
        for (binary_train, binary_test), (stim_train, stim_test) in zip(splits, stimulus_splits):
            if not np.array_equal(np.unique(binary.groups[binary_test]), np.unique(stimulus.groups[stim_test])):
                raise AssertionError(f"{session}: task fold partitions differ")
        all_frames = _load_all_session_frames(args, session, np.unique(binary.groups))
        checkpoints: dict[tuple[int, int], Path] = {}
        for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
            for seed in SSL_SEEDS:
                log(f"pretrain session={session} fold={fold_i}/{len(splits)} seed={seed}", args.output_dir)
                checkpoints[(fold_i, seed)] = _pretrain_one(
                    args,
                    session=session,
                    fold_i=fold_i,
                    seed=seed,
                    all_frames=all_frames,
                    train_cycles=np.unique(binary.groups[train_idx]),
                    test_cycles=np.unique(binary.groups[test_idx]),
                    root=args.output_dir / "pretraining",
                )
        del all_frames
        for task, data, task_split in (("binary", binary, splits), ("stimulus_type", stimulus, stimulus_splits)):
            for fold_i, (train_idx, test_idx) in enumerate(task_split, start=1):
                for seed in SSL_SEEDS:
                    encoder, _payload = load_ssl_encoder_checkpoint(checkpoints[(fold_i, seed)])
                    for condition in SSL_CONDITIONS:
                        log(f"downstream session={session} task={task} fold={fold_i} seed={seed} condition={condition}", args.output_dir)
                        result = train_downstream_fold(
                            condition,
                            data,
                            train_idx,
                            test_idx,
                            fold=fold_i,
                            seed=seed,
                            pretrained_encoder_state=None if condition == "RANDOM_INIT" else encoder.state_dict(),
                            config=SUPERVISED_CONFIG,
                            device=args.device,
                        )
                        fold_rows.append(dict(result.metrics))
                        prediction_rows.extend(_prediction_rows(data, test_idx, result, condition, seed, fold_i))
                        history = pd.DataFrame([
                            {
                                "session": session, "task": task, "fold": fold_i,
                                "seed": seed, "condition": condition, **row,
                            }
                            for row in result.history
                        ])
                        curve_path = args.output_dir / "downstream/training_curves" / f"session_{session}_{task}_fold_{fold_i}_seed_{seed}_{condition}.csv"
                        write_csv(curve_path, history)
                        write_csv(args.output_dir / "downstream/fold_metrics.csv", fold_rows)
                        write_csv(args.output_dir / "downstream/predictions.csv", prediction_rows)
    predictions = pd.DataFrame(prediction_rows)
    for (session, task, seed, condition), group in predictions.groupby(
        ["session", "task", "seed", "condition"], sort=True
    ):
        # Reloading is deliberately avoided; expected sample coverage is also
        # encoded by uniqueness plus the task/session fold-metric block totals.
        metric_group = pd.DataFrame(fold_rows)
        metric_group = metric_group[
            (metric_group["session"].astype(str) == str(session))
            & (metric_group["task"] == task)
            & (metric_group["seed"].astype(int) == int(seed))
            & (metric_group["condition"] == condition)
        ]
        expected = int(metric_group["n_test_blocks"].sum())
        if len(group) != expected or group["sample_id"].nunique() != expected:
            raise AssertionError(
                f"OOF coverage failure: {session} {task} {seed} {condition}: "
                f"rows={len(group)}, unique={group['sample_id'].nunique()}, expected={expected}"
            )
    aggregate_outputs(args.output_dir)
    missing = missing_formal_outputs(args.output_dir)
    # pytest_output.txt is produced by the required external verification step,
    # immediately after the benchmark.  All other formal deliverables must
    # already exist here.
    missing_before_pytest = [value for value in missing if value != "pytest_output.txt"]
    if missing_before_pytest:
        raise AssertionError(f"formal output tree is incomplete: {missing_before_pytest}")
    log("Formal 9-session benchmark completed", args.output_dir)


def _scenario(tests: pd.DataFrame, improvements: pd.DataFrame) -> str:
    finetune = improvements.groupby("task")["delta_finetune_vs_random"].mean()
    frozen = improvements.groupby("task")["delta_frozen_vs_random"].mean()
    gap = improvements.groupby("task")["gap_reduction_finetune"].mean()
    weak = improvements[improvements["session"].astype(str).isin(("626", "628", "807", "813", "817", "822"))]
    if (finetune > 0).all() and (gap > 0).all():
        return "A: SSL明显改善泛化"
    if (frozen > 0).all() and (finetune < frozen).all():
        return "B: Frozen有效但finetune重新过拟合"
    if weak["delta_finetune_vs_random"].mean() <= 0 and improvements[improvements["session"].astype(str).isin(("708", "709", "710"))]["delta_finetune_vs_random"].mean() > 0:
        return "C: 只帮助强session"
    return "D: masked-reconstruction SSL无稳定收益"


def aggregate_outputs(output_dir: Path) -> None:
    fold_path = output_dir / "downstream/fold_metrics.csv"
    if not fold_path.exists():
        raise FileNotFoundError(fold_path)
    fold_metrics = pd.read_csv(fold_path)
    sessions = session_level_metrics(fold_metrics)
    improvements = paired_ssl_improvements(sessions)
    gaps = generalization_gap_summary(sessions, improvements)
    tests, correction = statistical_test_tables(improvements)
    write_csv(output_dir / "summaries/session_level_metrics.csv", sessions)
    write_csv(output_dir / "summaries/paired_ssl_improvements.csv", improvements)
    write_csv(output_dir / "summaries/generalization_gap_summary.csv", gaps)
    write_csv(output_dir / "summaries/statistical_tests.csv", tests)
    write_csv(output_dir / "summaries/two_task_correction.csv", correction)
    make_required_plots(output_dir, fold_metrics, sessions)
    scenario = _scenario(tests, improvements)
    weak = improvements[improvements["session"].astype(str).isin(("626", "628", "807", "813", "817", "822"))]
    report = [
        "# Frozen SmallCNN Masked-Reconstruction SSL Benchmark v1",
        "",
        "This is a within-session, train-cycle-only benchmark. Test-cycle frames were excluded from SSL training, validation, normalization, reconstruction QC, and checkpoint selection.",
        "",
        "## Session-level test BA",
        "",
        markdown_table(sessions, ["session", "task", "condition", "mean_test_BA", "std_across_seeds", "mean_train_test_BA_gap"]),
        "",
        "## Paired improvements",
        "",
        markdown_table(improvements),
        "",
        "## Exact session-level inference",
        "",
        markdown_table(tests),
        "",
        "## Two-task correction",
        "",
        markdown_table(correction),
        "",
        "## Historically difficult sessions",
        "",
        markdown_table(weak),
        "",
        "## Preregistered scenario classification",
        "",
        scenario,
        "",
        "The conclusion is limited to this masked-reconstruction formulation. No cross-session SSL was run.",
    ]
    (output_dir / "report/ssl_masked_pretraining_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    ensure_output_tree(args.output_dir)
    assert_within_session_scope(EXPECTED_SESSIONS)
    (args.output_dir / "run_command.txt").write_text(
        " ".join(shlex.quote(value) for value in sys.argv) + "\n", encoding="utf-8"
    )
    if args.stage in {"audit", "all"}:
        run_static_audit(args)
    if args.stage in {"baseline", "all"}:
        gate = run_historical_baseline_reproduction(args)
        if not gate:
            return 2
    if args.stage in {"pilot", "all"}:
        run_pilot(args)
    if args.stage in {"benchmark", "all"}:
        run_formal_benchmark(args)
    if args.stage == "aggregate":
        aggregate_outputs(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
