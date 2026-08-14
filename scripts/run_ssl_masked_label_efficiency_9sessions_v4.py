#!/usr/bin/env python3
"""Frozen within-session masked-SSL label-efficiency benchmark v4."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import encoder_architecture_signature
from ultrasound_decoding.ssl_label_efficiency_reporting_v4 import (
    classify_scenario,
    label_efficiency_aulc,
    label_fraction_to_target,
    low_label_summary,
    make_required_figures,
    planned_statistical_tests,
    seed_stability,
    session_label_efficiency,
)
from ultrasound_decoding.ssl_label_efficiency_v4 import (
    FROZEN_SSL_CONFIG,
    FROZEN_SUPERVISED_CONFIG,
    LABEL_FRACTIONS,
    LOW_LABEL_FRACTIONS,
    REQUIRED_FORMAL_OUTPUTS,
    STRONG_SESSIONS,
    V1_CONDITION_MAP,
    V4_CONDITIONS,
    V4_SEEDS,
    WEAK_SESSIONS,
    assert_formal_cuda,
    audit_v1_checkpoint,
    condition_encoder_state,
    label_class_balance_row,
    label_fraction_rows,
    labeled_sample_indices,
    missing_formal_outputs,
    nested_label_subsets,
    ordered_cycle_text,
)
from ultrasound_decoding.ssl_masked import train_downstream_fold


RUN_NAME = "ssl_masked_label_efficiency_9sessions_v4"
V1_RUN_NAME = "ssl_masked_smallcnn_clean4_9sessions_v1"
METRIC_KEY = ["session", "task", "fold", "seed", "condition", "label_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "formal", "aggregate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--v1-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V1_RUN_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(V4_SEEDS))
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_tree(output_dir: Path) -> None:
    for relative in (
        "audit", "downstream/jobs", "downstream/training_curves", "summaries",
        "figures", "report", "smoke",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def log(message: str, output_dir: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_log_server.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_csv(path: Path, value: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    frame.to_csv(path, index=False)


def _cycle_text_from_csv(value: Any) -> str:
    return cycle_text([int(item) for item in str(value).split(",") if str(item)])


def _manifest_sha(
    manifest: pd.DataFrame, *, session: str, fold: int, seed: int
) -> str | None:
    subset = manifest[
        (manifest["session"].astype(str) == str(session))
        & (manifest["fold"].astype(int) == int(fold))
        & (manifest["seed"].astype(int) == int(seed))
    ]
    if len(subset) != 1:
        return None
    return str(subset.iloc[0]["checkpoint_sha256"])


def _v1_checkpoint(args: argparse.Namespace, session: str, fold: int, seed: int) -> Path:
    return args.v1_output_dir / f"pretraining/checkpoints/session_{session}/fold_{fold}/seed_{seed}.pt"


def run_audit(args: argparse.Namespace) -> None:
    ensure_output_tree(args.output_dir)
    if tuple(map(int, args.seeds)) != tuple(V4_SEEDS):
        raise RuntimeError(f"v4 requires exactly the fixed seeds {V4_SEEDS}")
    v1_folds_path = args.v1_output_dir / "audit/fold_reproduction.csv"
    manifest_path = args.v1_output_dir / "pretraining/checkpoint_manifest.csv"
    if not v1_folds_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("complete v1 fold audit and checkpoint manifest are required")
    v1_folds = pd.read_csv(v1_folds_path)
    manifest = pd.read_csv(manifest_path)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)

    fold_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    v1_reuse_rows: list[dict[str, Any]] = []
    checkpoint_reuse_rows: list[dict[str, Any]] = []

    for session in EXPECTED_SESSIONS:
        task_data: dict[str, Any] = {}
        task_splits: dict[str, Any] = {}
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            splits = grouped_cv_splits(data.groups, max_folds=10)
            task_data[task] = data
            task_splits[task] = splits
            old = v1_folds[
                (v1_folds["session"].astype(str) == str(session)) & (v1_folds["task"] == task)
            ]
            for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                train_cycles = np.unique(data.groups[train_idx])
                test_cycles = np.unique(data.groups[test_idx])
                old_row = old[old["fold"].astype(int) == fold_i]
                if len(old_row) != 1:
                    raise AssertionError(f"v1 fold audit missing {session} {task} fold {fold_i}")
                old_train = _cycle_text_from_csv(old_row.iloc[0]["train_cycles"])
                old_test = _cycle_text_from_csv(old_row.iloc[0]["test_cycles"])
                current_train = cycle_text(train_cycles)
                current_test = cycle_text(test_cycles)
                status = "PASS" if (current_train == old_train and current_test == old_test) else "FAIL"
                fold_rows.append({
                    "session": str(session), "task": task, "fold": fold_i,
                    "current_train_cycle_ids": current_train,
                    "v1_train_cycle_ids": old_train,
                    "current_test_cycle_ids": current_test,
                    "v1_test_cycle_ids": old_test,
                    "train_cycles_match": current_train == old_train,
                    "test_cycles_match": current_test == old_test,
                    "n_train_samples": int(len(train_idx)),
                    "n_test_samples": int(len(test_idx)),
                    "status": status,
                })
                if status != "PASS":
                    raise AssertionError(f"outer fold mismatch for {session} {task} fold {fold_i}")

        binary_splits = task_splits["binary"]
        stimulus_splits = task_splits["stimulus_type"]
        if len(binary_splits) != len(stimulus_splits):
            raise AssertionError(f"task fold counts differ for session {session}")
        for fold_i, ((b_train, b_test), (s_train, s_test)) in enumerate(
            zip(binary_splits, stimulus_splits), start=1
        ):
            binary = task_data["binary"]
            stimulus = task_data["stimulus_type"]
            train_cycles = np.unique(binary.groups[b_train])
            test_cycles = np.unique(binary.groups[b_test])
            if not np.array_equal(train_cycles, np.unique(stimulus.groups[s_train])) or not np.array_equal(
                test_cycles, np.unique(stimulus.groups[s_test])
            ):
                raise AssertionError(f"task fold cycle identities differ for {session} fold {fold_i}")
            for seed in V4_SEEDS:
                counts, subsets = label_fraction_rows(
                    train_cycles, session=session, fold=fold_i, seed=seed
                )
                count_rows.extend(counts)
                subset_rows.extend(subsets)
                _permutation, subset_map = nested_label_subsets(
                    train_cycles, session=session, fold=fold_i, seed=seed
                )
                checkpoint = _v1_checkpoint(args, session, fold_i, seed)
                if not checkpoint.exists():
                    raise FileNotFoundError(f"v1 checkpoint required for strict 100% reproduction: {checkpoint}")
                reuse_row, payload = audit_v1_checkpoint(
                    checkpoint,
                    session=session,
                    fold=fold_i,
                    seed=seed,
                    outer_train_cycles=train_cycles,
                    outer_test_cycles=test_cycles,
                    manifest_sha256=_manifest_sha(manifest, session=session, fold=fold_i, seed=seed),
                )
                v1_reuse_rows.append(reuse_row)
                checkpoint_reuse_rows.append({
                    "session": str(session), "fold": fold_i, "seed": int(seed),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": reuse_row["checkpoint_sha256"],
                    "ssl_outer_train_cycle_ids": cycle_text(train_cycles),
                    "ssl_train_cycle_ids": cycle_text(payload["ssl_train_cycles"]),
                    "ssl_validation_cycle_ids": cycle_text(payload["ssl_val_cycles"]),
                    "label_fractions_reusing_checkpoint": "0.2,0.4,0.6,0.8,1.0",
                    "n_label_fractions_reusing_checkpoint": 5,
                    "one_checkpoint_for_all_fractions": True,
                    "status": "PASS",
                })
                for task in EXPECTED_TASKS:
                    data = task_data[task]
                    task_train_idx, task_test_idx = task_splits[task][fold_i - 1]
                    task_train_cycles = set(data.groups[task_train_idx].astype(int).tolist())
                    task_test_cycles = set(data.groups[task_test_idx].astype(int).tolist())
                    for fraction in LABEL_FRACTIONS:
                        labeled_cycles = subset_map[fraction]
                        balance_rows.append(label_class_balance_row(
                            data, fold=fold_i, seed=seed, label_fraction=fraction,
                            labeled_cycles=labeled_cycles,
                        ))
                        labeled_idx = labeled_sample_indices(data, labeled_cycles)
                        ssl_cycles = set(map(int, payload["ssl_train_cycles"])) | set(map(int, payload["ssl_val_cycles"]))
                        leakage_rows.append({
                            "task": task, "session": str(session), "fold": fold_i,
                            "seed": int(seed), "label_fraction": float(fraction),
                            "outer_train_cycle_ids": cycle_text(list(task_train_cycles)),
                            "outer_test_cycle_ids": cycle_text(list(task_test_cycles)),
                            "ssl_unlabeled_cycle_ids": cycle_text(list(ssl_cycles)),
                            "supervised_labeled_cycle_ids": cycle_text(labeled_cycles),
                            "ssl_uses_all_outer_train_cycles": ssl_cycles == task_train_cycles,
                            "n_ssl_unlabeled_frames": int(30 * len(ssl_cycles)),
                            "n_test_frames_seen_by_ssl": int(30 * len(ssl_cycles & task_test_cycles)),
                            "n_test_samples_seen_by_supervised_train": int(
                                np.sum(np.isin(data.groups[labeled_idx], list(task_test_cycles)))
                            ),
                            "normalization_fit_test_samples": 0,
                            "status": "PASS" if (
                                ssl_cycles == task_train_cycles
                                and not (ssl_cycles & task_test_cycles)
                                and set(map(int, labeled_cycles)).issubset(task_train_cycles)
                                and not (set(map(int, labeled_cycles)) & task_test_cycles)
                            ) else "FAIL",
                        })

    outputs = {
        "fold_identity_check.csv": fold_rows,
        "label_fraction_cycle_counts.csv": count_rows,
        "nested_label_subsets.csv": subset_rows,
        "label_class_balance.csv": balance_rows,
        "test_cycle_leakage.csv": leakage_rows,
        "ssl_checkpoint_reuse.csv": checkpoint_reuse_rows,
        "v1_checkpoint_reuse.csv": v1_reuse_rows,
    }
    for filename, rows in outputs.items():
        write_csv(args.output_dir / "audit" / filename, rows)
    if any(row["status"] != "PASS" for row in leakage_rows):
        raise AssertionError("test-cycle leakage audit failed")
    if any(row["status"] != "VALID" for row in balance_rows):
        invalid = [row for row in balance_rows if row["status"] != "VALID"]
        raise RuntimeError(f"INVALID_SINGLE_CLASS_TRAINING: {invalid[:3]}")
    freeze = [
        "# Frozen v4 configuration", "",
        f"- Sessions: `{list(EXPECTED_SESSIONS)}`",
        f"- Tasks: `{list(EXPECTED_TASKS)}`",
        f"- Conditions: `{list(V4_CONDITIONS)}`",
        f"- Label fractions: `{list(LABEL_FRACTIONS)}`",
        f"- Seeds: `{list(V4_SEEDS)}`",
        "- Label-budget unit: outer-training cycle; deterministic nested prefixes.",
        "- SSL pool: all and only target-session outer-training cycles; labels are not read.",
        "- SSL artifact: exact compatible v1 checkpoint, one per session/fold/seed, reused at all fractions.",
        f"- SSL configuration: `{asdict(FROZEN_SSL_CONFIG)}`",
        f"- Supervised configuration: `{asdict(FROZEN_SUPERVISED_CONFIG)}`",
        f"- SmallCNN encoder signature: `{encoder_architecture_signature()}`",
        "- Input: existing clean4 sample builder, four frames, shared encoder, feature mean, classifier.",
        "- The 100% rows are exact v1 artifact imports and are a replication/reference check.",
    ]
    (args.output_dir / "audit/config_freeze.md").write_text("\n".join(freeze) + "\n", encoding="utf-8")
    log("Static audit PASSED: folds, nested cycle budgets, balance, leakage, and v1 checkpoint reuse", args.output_dir)


def _fraction_tag(fraction: float) -> str:
    return f"fraction_{int(round(100 * float(fraction))):03d}"


def _job_dir(
    output_dir: Path,
    *,
    session: str,
    task: str,
    fold: int,
    seed: int,
    condition: str,
    fraction: float,
) -> Path:
    return (
        output_dir / "downstream/jobs" / f"session_{session}" / task
        / f"fold_{fold}" / f"seed_{seed}" / _fraction_tag(fraction) / condition
    )


def _curve_path(
    output_dir: Path,
    *,
    session: str,
    task: str,
    fold: int,
    seed: int,
    condition: str,
    fraction: float,
) -> Path:
    return output_dir / "downstream/training_curves" / (
        f"session_{session}_{task}_fold_{fold}_seed_{seed}_{_fraction_tag(fraction)}_{condition}.csv"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _job_complete(path: Path, curve_path: Path) -> bool:
    return (
        (path / "metrics.json").is_file()
        and (path / "predictions.csv").is_file()
        and curve_path.is_file()
    )


def _save_job(
    directory: Path,
    curve_path: Path,
    *,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    curve: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(directory / "metrics.json")
    predictions.to_csv(directory / "predictions.csv", index=False)
    curve.to_csv(curve_path, index=False)


def _prediction_frame(
    data: Any,
    test_idx: np.ndarray,
    result: Any,
    *,
    condition: str,
    seed: int,
    fold: int,
    fraction: float,
) -> pd.DataFrame:
    rows = []
    for local_i, sample_i in enumerate(test_idx):
        rows.append({
            "session": str(data.session),
            "task": data.task,
            "fold": int(fold),
            "seed": int(seed),
            "condition": condition,
            "label_fraction": float(fraction),
            "sample_i": int(sample_i),
            "sample_id": str(data.metadata.iloc[int(sample_i)]["block_id"]),
            "cycle": int(data.groups[sample_i]),
            "label": int(data.y[sample_i]),
            "prediction": int(result.test_predictions[local_i]),
            "probability_class_0": float(result.test_probabilities[local_i, 0]),
            "probability_class_1": float(result.test_probabilities[local_i, 1]),
            "source_artifact": "v4_low_label_training",
        })
    return pd.DataFrame(rows)


def _run_low_label_job(
    args: argparse.Namespace,
    *,
    data: Any,
    outer_train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold: int,
    seed: int,
    condition: str,
    fraction: float,
    labeled_cycles: np.ndarray,
    checkpoint: Path,
) -> None:
    directory = _job_dir(
        args.output_dir, session=data.session, task=data.task, fold=fold, seed=seed,
        condition=condition, fraction=fraction,
    )
    curve_path = _curve_path(
        args.output_dir, session=data.session, task=data.task, fold=fold, seed=seed,
        condition=condition, fraction=fraction,
    )
    if _job_complete(directory, curve_path) and not args.overwrite:
        return
    labeled_idx = labeled_sample_indices(data, labeled_cycles)
    if not set(labeled_idx.tolist()).issubset(set(outer_train_idx.tolist())):
        raise AssertionError("supervised label subset is not inside the outer training indices")
    classes = set(data.y[labeled_idx].astype(int).tolist())
    if classes != {0, 1}:
        raise RuntimeError("INVALID_SINGLE_CLASS_TRAINING")
    internal_condition = "RANDOM_INIT" if condition == "RANDOM_INIT" else "SSL_FINETUNE"
    state = condition_encoder_state(
        condition, None if condition == "RANDOM_INIT" else checkpoint
    )
    result = train_downstream_fold(
        internal_condition,
        data,
        labeled_idx,
        test_idx,
        fold=fold,
        seed=seed,
        pretrained_encoder_state=state,
        config=FROZEN_SUPERVISED_CONFIG,
        device=args.device,
    )
    metrics = dict(result.metrics)
    metrics.update({
        "session": str(data.session),
        "task": data.task,
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "label_fraction": float(fraction),
        "actual_label_cycle_fraction": float(len(labeled_cycles) / len(np.unique(data.groups[outer_train_idx]))),
        "n_label_cycles": int(len(labeled_cycles)),
        "labeled_cycles": cycle_text(labeled_cycles),
        "outer_train_cycles": cycle_text(data.groups[outer_train_idx]),
        "n_outer_train_cycles": int(len(np.unique(data.groups[outer_train_idx]))),
        "ssl_checkpoint": "" if condition == "RANDOM_INIT" else str(checkpoint),
        "run_status": "VALID",
        "source_artifact": "v4_low_label_training",
    })
    curve = pd.DataFrame([
        {
            "session": str(data.session), "task": data.task, "fold": int(fold),
            "seed": int(seed), "condition": condition, "label_fraction": float(fraction),
            "n_label_cycles": int(len(labeled_cycles)), **row,
        }
        for row in result.history
    ])
    predictions = _prediction_frame(
        data, test_idx, result, condition=condition, seed=seed, fold=fold, fraction=fraction
    )
    _save_job(directory, curve_path, metrics=metrics, predictions=predictions, curve=curve)


def _materialize_full_label_v1_jobs(args: argparse.Namespace) -> None:
    metrics_path = args.v1_output_dir / "downstream/fold_metrics.csv"
    predictions_path = args.v1_output_dir / "downstream/predictions.csv"
    curves_dir = args.v1_output_dir / "downstream/training_curves"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("v1 downstream metrics and predictions are required")
    v1_metrics = pd.read_csv(metrics_path)
    v1_predictions = pd.read_csv(predictions_path)
    selected = v1_metrics[
        v1_metrics["condition"].isin(V1_CONDITION_MAP)
        & v1_metrics["seed"].astype(int).isin(V4_SEEDS)
    ].copy()
    expected = 2 * len(V4_SEEDS) * int(
        selected[["session", "task", "fold"]].drop_duplicates().shape[0]
    )
    if len(selected) != expected:
        raise AssertionError(f"incomplete v1 100% metric rows: {len(selected)}/{expected}")
    for _, old in selected.iterrows():
        session = str(old["session"])
        task = str(old["task"])
        fold = int(old["fold"])
        seed = int(old["seed"])
        old_condition = str(old["condition"])
        condition = V1_CONDITION_MAP[old_condition]
        fraction = 1.0
        directory = _job_dir(
            args.output_dir, session=session, task=task, fold=fold, seed=seed,
            condition=condition, fraction=fraction,
        )
        curve_path = _curve_path(
            args.output_dir, session=session, task=task, fold=fold, seed=seed,
            condition=condition, fraction=fraction,
        )
        if _job_complete(directory, curve_path) and not args.overwrite:
            continue
        train_cycles = [int(value) for value in str(old["train_cycles"]).split(",")]
        metrics = old.to_dict()
        metrics.update({
            "session": session,
            "task": task,
            "fold": fold,
            "seed": seed,
            "condition": condition,
            "label_fraction": 1.0,
            "actual_label_cycle_fraction": 1.0,
            "n_label_cycles": len(train_cycles),
            "labeled_cycles": cycle_text(train_cycles),
            "outer_train_cycles": cycle_text(train_cycles),
            "n_outer_train_cycles": len(train_cycles),
            "ssl_checkpoint": "" if condition == "RANDOM_INIT" else str(_v1_checkpoint(args, session, fold, seed)),
            "run_status": "VALID",
            "source_artifact": "v1_exact_full_label_import",
        })
        predictions = v1_predictions[
            (v1_predictions["session"].astype(str) == session)
            & (v1_predictions["task"] == task)
            & (v1_predictions["fold"].astype(int) == fold)
            & (v1_predictions["seed"].astype(int) == seed)
            & (v1_predictions["condition"] == old_condition)
        ].copy()
        if len(predictions) != int(old["n_test_blocks"]):
            raise AssertionError("v1 prediction rows are incomplete for a 100% job")
        predictions["session"] = session
        predictions["condition"] = condition
        predictions["label_fraction"] = 1.0
        predictions["source_artifact"] = "v1_exact_full_label_import"
        old_curve = curves_dir / (
            f"session_{session}_{task}_fold_{fold}_seed_{seed}_{old_condition}.csv"
        )
        if not old_curve.exists():
            raise FileNotFoundError(old_curve)
        curve = pd.read_csv(old_curve)
        curve["session"] = session
        curve["condition"] = condition
        curve["label_fraction"] = 1.0
        curve["n_label_cycles"] = len(train_cycles)
        _save_job(directory, curve_path, metrics=metrics, predictions=predictions, curve=curve)


def consolidate_downstream(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for metrics_path in sorted((output_dir / "downstream/jobs").glob("**/metrics.json")):
        metric_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        prediction_path = metrics_path.parent / "predictions.csv"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        predictions.append(pd.read_csv(prediction_path))
    metrics = pd.DataFrame(metric_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    if len(metrics):
        metrics["session"] = metrics["session"].astype(str)
        metrics = metrics.sort_values(METRIC_KEY).reset_index(drop=True)
    if len(prediction_frame):
        prediction_frame["session"] = prediction_frame["session"].astype(str)
        prediction_frame = prediction_frame.sort_values(
            METRIC_KEY + ["sample_i"]
        ).reset_index(drop=True)
    write_csv(output_dir / "downstream/fold_metrics.csv", metrics)
    write_csv(output_dir / "downstream/predictions.csv", prediction_frame)
    return metrics, prediction_frame


def _full_label_reproduction(
    args: argparse.Namespace, metrics: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    v1_metrics = pd.read_csv(args.v1_output_dir / "downstream/fold_metrics.csv")
    v1_predictions = pd.read_csv(args.v1_output_dir / "downstream/predictions.csv")
    current = metrics[np.isclose(metrics["label_fraction"].astype(float), 1.0)].copy()
    numeric_columns = (
        "train_accuracy", "train_balanced_accuracy", "test_accuracy",
        "test_balanced_accuracy", "macro_F1", "ROC_AUC", "train_test_gap_BA",
    )
    rows: list[dict[str, Any]] = []
    for _, old in v1_metrics[v1_metrics["condition"].isin(V1_CONDITION_MAP)].iterrows():
        session = str(old["session"])
        task = str(old["task"])
        fold = int(old["fold"])
        seed = int(old["seed"])
        condition = V1_CONDITION_MAP[str(old["condition"])]
        match = current[
            (current["session"].astype(str) == session)
            & (current["task"] == task)
            & (current["fold"].astype(int) == fold)
            & (current["seed"].astype(int) == seed)
            & (current["condition"] == condition)
        ]
        if len(match) != 1:
            raise AssertionError("100% v4 row does not map one-to-one to v1")
        new = match.iloc[0]
        differences = {
            column: abs(float(new[column]) - float(old[column])) for column in numeric_columns
        }
        new_predictions = predictions[
            (predictions["session"].astype(str) == session)
            & (predictions["task"] == task)
            & (predictions["fold"].astype(int) == fold)
            & (predictions["seed"].astype(int) == seed)
            & (predictions["condition"] == condition)
            & np.isclose(predictions["label_fraction"].astype(float), 1.0)
        ].sort_values("sample_i")
        old_predictions = v1_predictions[
            (v1_predictions["session"].astype(str) == session)
            & (v1_predictions["task"] == task)
            & (v1_predictions["fold"].astype(int) == fold)
            & (v1_predictions["seed"].astype(int) == seed)
            & (v1_predictions["condition"] == str(old["condition"]))
        ].sort_values("sample_i")
        sample_ids_match = new_predictions["sample_id"].astype(str).tolist() == old_predictions["sample_id"].astype(str).tolist()
        labels_match = new_predictions["label"].astype(int).tolist() == old_predictions["label"].astype(int).tolist()
        folds_match = (
            _cycle_text_from_csv(new["train_cycles"]) == _cycle_text_from_csv(old["train_cycles"])
            and _cycle_text_from_csv(new["test_cycles"]) == _cycle_text_from_csv(old["test_cycles"])
        )
        best_epoch_match = int(new["best_epoch"]) == int(old["best_epoch"])
        max_difference = max(differences.values())
        passed = sample_ids_match and labels_match and folds_match and best_epoch_match and max_difference <= 1e-12
        rows.append({
            "session": session, "task": task, "fold": fold, "seed": seed,
            "condition": condition, "label_fraction": 1.0,
            "folds_match": folds_match,
            "sample_ids_match": sample_ids_match,
            "labels_match": labels_match,
            "best_epoch_match": best_epoch_match,
            "v1_test_BA": float(old["test_balanced_accuracy"]),
            "v4_test_BA": float(new["test_balanced_accuracy"]),
            "abs_test_BA_difference": differences["test_balanced_accuracy"],
            "v1_train_BA": float(old["train_balanced_accuracy"]),
            "v4_train_BA": float(new["train_balanced_accuracy"]),
            "abs_train_BA_difference": differences["train_balanced_accuracy"],
            "max_abs_metric_difference": max_difference,
            "tolerance": 1e-12,
            "source": "exact v1 full-label artifact import",
            "status": "PASS" if passed else "FAIL",
        })
    output = pd.DataFrame(rows)
    if len(output) != 984 or not (output["status"] == "PASS").all():
        failures = output[output["status"] != "PASS"]
        raise RuntimeError(f"100% v1 reproduction STOP: rows={len(output)}, failures={len(failures)}")
    write_csv(args.output_dir / "audit/full_label_v1_reproduction.csv", output)
    return output


def _markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    if frame.empty:
        return "_No rows._"
    columns = [str(value) for value in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("NA" if not np.isfinite(value) else f"{float(value):.5f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _validate_prediction_coverage(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    keys = ["session", "task", "seed", "condition", "label_fraction"]
    for key, group in metrics.groupby(keys, sort=True):
        expected = int(group["n_test_blocks"].astype(int).sum())
        selected = predictions
        for column, value in zip(keys, key):
            if column == "label_fraction":
                selected = selected[np.isclose(selected[column].astype(float), float(value))]
            else:
                selected = selected[selected[column].astype(str) == str(value)]
        if len(selected) != expected or selected["sample_id"].astype(str).nunique() != expected:
            raise AssertionError(
                f"OOF prediction coverage failed for {key}: rows={len(selected)}, "
                f"unique={selected['sample_id'].nunique()}, expected={expected}"
            )


def aggregate_outputs(args: argparse.Namespace) -> None:
    metrics, predictions = consolidate_downstream(args.output_dir)
    fold_audit = pd.read_csv(args.output_dir / "audit/fold_identity_check.csv")
    expected_rows = int(len(fold_audit) * len(V4_SEEDS) * len(V4_CONDITIONS) * len(LABEL_FRACTIONS))
    if len(metrics) != expected_rows:
        raise RuntimeError(f"aggregate STOP: incomplete fold metrics {len(metrics)}/{expected_rows}")
    if metrics.duplicated(METRIC_KEY).any():
        raise RuntimeError("aggregate STOP: duplicate metric keys")
    if set(metrics["condition"].unique()) != set(V4_CONDITIONS):
        raise RuntimeError("aggregate STOP: conditions differ from frozen v4 conditions")
    if set(metrics["seed"].astype(int).unique()) != set(V4_SEEDS):
        raise RuntimeError("aggregate STOP: seeds differ from fixed seeds")
    if set(np.round(metrics["label_fraction"].astype(float), 8)) != set(LABEL_FRACTIONS):
        raise RuntimeError("aggregate STOP: label fractions differ from frozen fractions")
    if not (metrics["run_status"] == "VALID").all():
        raise RuntimeError("aggregate STOP: invalid supervised run present")
    numeric = [
        "train_accuracy", "train_balanced_accuracy", "test_accuracy",
        "test_balanced_accuracy", "macro_F1", "ROC_AUC", "train_test_gap_BA",
    ]
    if not np.isfinite(metrics[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("aggregate STOP: non-finite formal metric")
    _validate_prediction_coverage(metrics, predictions)

    nested = pd.read_csv(args.output_dir / "audit/nested_label_subsets.csv")
    leakage = pd.read_csv(args.output_dir / "audit/test_cycle_leakage.csv")
    reuse = pd.read_csv(args.output_dir / "audit/ssl_checkpoint_reuse.csv")
    v1_reuse = pd.read_csv(args.output_dir / "audit/v1_checkpoint_reuse.csv")
    expected_fold_seed = int(fold_audit[fold_audit["task"] == "binary"].shape[0] * len(V4_SEEDS))
    if len(nested) != expected_fold_seed * len(LABEL_FRACTIONS) or not (nested["status"] == "PASS").all():
        raise RuntimeError("aggregate STOP: nested label-subset audit incomplete or failed")
    if not (leakage["status"] == "PASS").all() or leakage["n_test_frames_seen_by_ssl"].astype(int).sum() != 0:
        raise RuntimeError("aggregate STOP: target-test leakage detected")
    if len(reuse) != expected_fold_seed or not reuse["one_checkpoint_for_all_fractions"].astype(bool).all():
        raise RuntimeError("aggregate STOP: SSL checkpoint reuse audit incomplete")
    if len(v1_reuse) != expected_fold_seed or not v1_reuse["reused"].astype(bool).all():
        raise RuntimeError("aggregate STOP: incompatible v1 checkpoint reuse")
    expected_curves = expected_rows
    if len(list((args.output_dir / "downstream/training_curves").glob("*.csv"))) != expected_curves:
        raise RuntimeError("aggregate STOP: training curves incomplete")

    reproduction = _full_label_reproduction(args, metrics, predictions)
    session_table = session_label_efficiency(metrics)
    aulc = label_efficiency_aulc(session_table)
    low = low_label_summary(session_table)
    tests = planned_statistical_tests(aulc, low)
    targets = label_fraction_to_target(session_table)
    stability = seed_stability(metrics)
    write_csv(args.output_dir / "summaries/session_label_efficiency.csv", session_table)
    write_csv(args.output_dir / "summaries/label_efficiency_AULC.csv", aulc)
    write_csv(args.output_dir / "summaries/low_label_summary.csv", low)
    write_csv(args.output_dir / "summaries/planned_statistical_tests.csv", tests)
    write_csv(args.output_dir / "summaries/label_fraction_to_target.csv", targets)
    write_csv(args.output_dir / "summaries/seed_stability.csv", stability)
    make_required_figures(args.output_dir, session_table)

    scenario = classify_scenario(session_table, aulc)
    binary_table = session_table[session_table["task"] == "binary"]
    stimulus_table = session_table[session_table["task"] == "stimulus_type"]
    weak_low = low[(low["task"] == "binary") & low["session"].astype(str).isin(WEAK_SESSIONS)]
    strong_targets = targets[
        (targets["task"] == "binary") & targets["session"].astype(str).isin(STRONG_SESSIONS)
    ]
    report = [
        "# Masked SSL label-efficiency benchmark v4", "",
        "## Integrity gates", "",
        f"- Completed formal metric rows: `{len(metrics)}` / `{expected_rows}`.",
        f"- Sessions: `{sorted(metrics['session'].astype(str).unique())}`.",
        "- Outer folds are identical to v1; nested cycle subsets and zero target-test exposure passed.",
        f"- Exact 100% v1 reproduction rows: `{len(reproduction)}`; all passed at tolerance `1e-12`.",
        "- The same audited within-session masked checkpoint was reused across all five fractions.", "",
        "## Preregistered session-level tests", "", _markdown_table(tests), "",
        "## Binary label-efficiency table", "", _markdown_table(binary_table), "",
        "## Stimulus-type label-efficiency table", "", _markdown_table(stimulus_table), "",
        "## AULC", "", _markdown_table(aulc), "",
        "## Low-label descriptive results", "", _markdown_table(low), "",
        "### Historically difficult sessions", "", _markdown_table(weak_low), "",
        "## Fraction needed to reach 95% of Random-init 100% BA", "", _markdown_table(targets), "",
        "### Strong-session exploratory view", "", _markdown_table(strong_targets), "",
        "## Preregistered interpretation", "",
        f"- Scenario: **{scenario}**.",
        "- All five fractions are reported; no fraction was selected post hoc.",
        "- The 100% contrast is replication/reference only, not a new primary claim.",
    ]
    report_path = args.output_dir / "report/label_efficiency_report.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    missing = missing_formal_outputs(args.output_dir)
    if missing:
        raise RuntimeError(f"formal output completeness STOP: {missing}")
    log(f"Aggregation PASSED; preregistered scenario={scenario}", args.output_dir)


def run_smoke(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("the local smoke test is CPU-only")
    if int(args.smoke_epochs) < 1:
        raise ValueError("smoke epochs must be positive")
    run_audit(args)
    session = "626"
    task = "binary"
    fold_i = 1
    seed = V4_SEEDS[0]
    data = load_block_sequence_session(
        PROJECT_DIR, session, task,
        data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR),
    )
    train_idx, test_idx = grouped_cv_splits(data.groups, max_folds=10)[fold_i - 1]
    permutation, subsets = nested_label_subsets(
        np.unique(data.groups[train_idx]), session=session, fold=fold_i, seed=seed
    )
    checkpoint = _v1_checkpoint(args, session, fold_i, seed)
    smoke_config = replace(FROZEN_SUPERVISED_CONFIG, max_epochs=int(args.smoke_epochs))
    metric_rows: list[dict[str, Any]] = []
    checkpoint_paths: list[str] = []
    started = time.perf_counter()
    for fraction in (0.2, 1.0):
        labeled_cycles = subsets[fraction]
        labeled_idx = labeled_sample_indices(data, labeled_cycles)
        for condition in V4_CONDITIONS:
            state = condition_encoder_state(
                condition, None if condition == "RANDOM_INIT" else checkpoint
            )
            result = train_downstream_fold(
                "RANDOM_INIT" if condition == "RANDOM_INIT" else "SSL_FINETUNE",
                data,
                labeled_idx,
                test_idx,
                fold=fold_i,
                seed=seed,
                pretrained_encoder_state=state,
                config=smoke_config,
                device="cpu",
            )
            row = dict(result.metrics)
            row.update({
                "condition": condition,
                "label_fraction": fraction,
                "n_label_cycles": len(labeled_cycles),
                "labeled_cycles": cycle_text(labeled_cycles),
            })
            metric_rows.append(row)
            if condition == "WITHIN_MASKED_SSL_FT":
                checkpoint_paths.append(str(checkpoint.resolve()))
    smoke_metrics = pd.DataFrame(metric_rows)
    smoke_session = session_label_efficiency(
        smoke_metrics, fractions=(0.2, 1.0), require_nine_sessions=False
    )
    smoke_aulc = label_efficiency_aulc(smoke_session)
    if len(set(checkpoint_paths)) != 1:
        raise AssertionError("smoke fractions did not reuse one checkpoint")
    if set(permutation.tolist()) != set(subsets[1.0].tolist()):
        raise AssertionError("smoke 100% subset is incomplete")
    elapsed = time.perf_counter() - started
    lines = [
        "PASS: tiny local CPU smoke only; not a formal scientific result",
        f"session={session} task={task} fold={fold_i} seed={seed}",
        "fractions=0.2,1.0 conditions=RANDOM_INIT,WITHIN_MASKED_SSL_FT",
        f"smoke_epochs={smoke_config.max_epochs}",
        f"nested_permutation={ordered_cycle_text(permutation)}",
        f"fraction_0.2_cycles={cycle_text(subsets[0.2])}",
        f"fraction_1.0_cycles={cycle_text(subsets[1.0])}",
        f"checkpoint_reused_across_fractions={checkpoint_paths[0]}",
        f"metric_rows={len(smoke_metrics)}",
        f"session_summary_rows={len(smoke_session)}",
        f"aulc_rows={len(smoke_aulc)}",
        f"runtime_seconds={elapsed:.3f}",
        "schema_check=PASS",
    ]
    text = "\n".join(lines) + "\n"
    (args.output_dir / "smoke_test_local.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def _write_gpu_audit(args: argparse.Namespace, *, runtime_seconds: float | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal v4 requires CUDA; CPU fallback is forbidden")
    values = [
        f"gpu_name={torch.cuda.get_device_name(0)}",
        f"cuda_runtime={torch.version.cuda}",
        f"pytorch_version={torch.__version__}",
        f"pytorch_cuda_available={torch.cuda.is_available()}",
        f"device_count={torch.cuda.device_count()}",
    ]
    if runtime_seconds is not None:
        values.extend([
            f"peak_vram_bytes={torch.cuda.max_memory_allocated()}",
            f"runtime_seconds={runtime_seconds:.3f}",
        ])
    (args.output_dir / "audit/gpu_audit.txt").write_text("\n".join(values) + "\n", encoding="utf-8")


def run_formal(args: argparse.Namespace) -> None:
    assert_formal_cuda(args.device)
    if tuple(map(int, args.seeds)) != tuple(V4_SEEDS):
        raise RuntimeError(f"formal v4 requires exactly seeds {V4_SEEDS}")
    ensure_output_tree(args.output_dir)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _write_gpu_audit(args)
    log("Formal CUDA label-efficiency benchmark started", args.output_dir)
    run_audit(args)
    _materialize_full_label_v1_jobs(args)
    log("Materialized exact v1 100% reference jobs", args.output_dir)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    completed = 0
    for session in EXPECTED_SESSIONS:
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            splits = grouped_cv_splits(data.groups, max_folds=10)
            for fold_i, (outer_train_idx, test_idx) in enumerate(splits, start=1):
                train_cycles = np.unique(data.groups[outer_train_idx])
                for seed in V4_SEEDS:
                    _permutation, subsets = nested_label_subsets(
                        train_cycles, session=session, fold=fold_i, seed=seed
                    )
                    checkpoint = _v1_checkpoint(args, session, fold_i, seed)
                    for fraction in LABEL_FRACTIONS[:-1]:
                        for condition in V4_CONDITIONS:
                            _run_low_label_job(
                                args,
                                data=data,
                                outer_train_idx=outer_train_idx,
                                test_idx=test_idx,
                                fold=fold_i,
                                seed=seed,
                                condition=condition,
                                fraction=fraction,
                                labeled_cycles=subsets[fraction],
                                checkpoint=checkpoint,
                            )
                            completed += 1
                            if completed % 50 == 0:
                                log(f"low-label downstream jobs visited={completed}", args.output_dir)
    aggregate_outputs(args)
    runtime = time.perf_counter() - started
    _write_gpu_audit(args, runtime_seconds=runtime)
    log(f"Formal v4 completed in {runtime:.3f} seconds", args.output_dir)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.v1_output_dir = args.v1_output_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    ensure_output_tree(args.output_dir)
    if args.stage == "audit":
        run_audit(args)
    elif args.stage == "smoke":
        run_smoke(args)
    elif args.stage == "formal":
        run_formal(args)
    else:
        aggregate_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
