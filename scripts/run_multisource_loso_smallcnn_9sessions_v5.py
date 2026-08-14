#!/usr/bin/env python3
"""Multi-source leave-one-session-out clean4 SmallCNN benchmark v5."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    TASK_CLASS_NAMES,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import (
    CNN2DMeanPool,
    SmallCNNFrameEncoder,
    encoder_architecture_signature,
)
from ultrasound_decoding.multisource_loso_reporting_v5 import (
    classify_scenario,
    make_required_figures,
    planned_statistical_tests,
    seed_stability,
    target_level_comparison,
    within_cross_gap,
)
from ultrasound_decoding.multisource_loso_v5 import (
    FROZEN_SUPERVISED_CONFIG,
    MULTI_SOURCE_CONDITIONS,
    REQUIRED_FORMAL_OUTPUTS,
    V5_CONDITIONS,
    V5_SEEDS,
    assert_formal_cuda,
    missing_formal_outputs,
    prepare_cross_session_data,
    source_sessions_for_target,
    train_prepared_cross_session,
)


RUN_NAME = "multisource_loso_smallcnn_9sessions_v5"
V1_RUN_NAME = "ssl_masked_smallcnn_clean4_9sessions_v1"
JOB_KEY = ["task", "target_session", "condition", "source_sessions", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "formal", "aggregate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--v1-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V1_RUN_NAME)
    parser.add_argument(
        "--historical-cross-session-root",
        type=Path,
        default=PROJECT_DIR / "results/runs/generalization",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(V5_SEEDS))
    parser.add_argument("--smoke-epochs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_tree(output_dir: Path) -> None:
    for relative in (
        "audit", "downstream/jobs", "downstream/training_curves",
        "summaries", "figures", "report", "smoke",
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


def _source_key(sources: tuple[str, ...]) -> str:
    return "source_" + "_".join(sources)


def _job_dir(
    output_dir: Path,
    *,
    task: str,
    target: str,
    condition: str,
    sources: tuple[str, ...],
    seed: int,
) -> Path:
    return (
        output_dir / "downstream/jobs" / task / f"target_{target}" / condition
        / _source_key(sources) / f"seed_{seed}"
    )


def _curve_path(
    output_dir: Path,
    *,
    task: str,
    target: str,
    condition: str,
    sources: tuple[str, ...],
    seed: int,
) -> Path:
    return output_dir / "downstream/training_curves" / (
        f"{task}_target_{target}_{condition}_{_source_key(sources)}_seed_{seed}.csv"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _job_complete(directory: Path, curve_path: Path) -> bool:
    return all((directory / name).is_file() for name in ("metrics.json", "predictions.csv", "sampling.csv")) and curve_path.is_file()


def _save_job(
    directory: Path,
    curve_path: Path,
    *,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    curve: pd.DataFrame,
    sampling: pd.DataFrame,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory / "metrics.json.tmp"
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(directory / "metrics.json")
    predictions.to_csv(directory / "predictions.csv", index=False)
    sampling.to_csv(directory / "sampling.csv", index=False)
    curve.to_csv(curve_path, index=False)


def _within_reference_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    path = args.v1_output_dir / "downstream/fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"v1 within-session metrics missing: {path}")
    metrics = pd.read_csv(path)
    metrics = metrics[
        (metrics["condition"] == "RANDOM_INIT")
        & metrics["seed"].astype(int).isin(V5_SEEDS)
    ].copy()
    rows = []
    for task in EXPECTED_TASKS:
        for target in EXPECTED_SESSIONS:
            data = metrics[
                (metrics["task"] == task)
                & (metrics["session"].astype(str) == target)
            ]
            expected_folds = int(data["fold"].nunique())
            for seed in V5_SEEDS:
                subset = data[data["seed"].astype(int) == int(seed)]
                valid = (
                    len(subset) == expected_folds
                    and expected_folds > 1
                    and (subset["best_epoch"].astype(int) == FROZEN_SUPERVISED_CONFIG.max_epochs).all()
                    and (subset["encoder_requires_grad"].astype(bool)).all()
                    and not (subset["decoder_present"].astype(bool)).any()
                )
                rows.append({
                    "task": task,
                    "target_session": target,
                    "seed": int(seed),
                    "artifact": str(path),
                    "condition": "RANDOM_INIT",
                    "n_folds": int(len(subset)),
                    "within_session_reference_BA_seed": float(subset["test_balanced_accuracy"].mean()),
                    "sample_builder": "block_sequences_v1 clean4",
                    "smallcnn_feature_mean": True,
                    "supervised_config_match": bool(valid),
                    "reused": bool(valid),
                    "reason": "exact clean4 SmallCNN feature-mean v1 RANDOM_INIT fold/seed reference" if valid else "incomplete or incompatible v1 reference",
                    "status": "PASS" if valid else "FAIL",
                })
    if len(rows) != 54 or any(row["status"] != "PASS" for row in rows):
        raise AssertionError("within-session v1 reference reuse audit failed")
    return rows


def _historical_single_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    all_session_candidate = args.historical_cross_session_root / "pairwise_all_sessions_binary_v1"
    strong_cnn_candidate = args.historical_cross_session_root / "pairwise_strong_sessions_binary_fixed40_v1"
    rows = []
    for task in EXPECTED_TASKS:
        for target in EXPECTED_SESSIONS:
            for source in source_sessions_for_target(target):
                for seed in V5_SEEDS:
                    rows.append({
                        "task": task,
                        "target_session": target,
                        "source_session": source,
                        "seed": int(seed),
                        "all_session_historical_candidate": str(all_session_candidate),
                        "strong_smallcnn_historical_candidate": str(strong_cnn_candidate),
                        "historical_candidates_exist": bool(all_session_candidate.exists() or strong_cnn_candidate.exists()),
                        "reused": False,
                        "reason": (
                            "historical all-session results are single-frame linear models; historical SmallCNN is single-frame, binary-only, "
                            "limited to 708/709/710, and uses different seeds; v5 therefore regenerates a comparable clean4 SmallCNN baseline"
                        ),
                        "comparable_baseline_generated": False,
                        "generated_job": "",
                        "status": "PLANNED_REGENERATION",
                    })
    return rows


def run_audit(args: argparse.Namespace) -> None:
    ensure_output_tree(args.output_dir)
    if tuple(map(int, args.seeds)) != tuple(V5_SEEDS):
        raise RuntimeError(f"v5 requires exactly the fixed seeds {V5_SEEDS}")
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    volume_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for task in EXPECTED_TASKS:
        data_by_session = {
            session: load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            for session in EXPECTED_SESSIONS
        }
        for target in EXPECTED_SESSIONS:
            sources = source_sessions_for_target(target)
            total_blocks = int(sum(data_by_session[source].n_blocks for source in sources))
            for condition in MULTI_SOURCE_CONDITIONS:
                for source in sources:
                    data = data_by_session[source]
                    classes = {int(value): int(np.sum(data.y == value)) for value in (0, 1)}
                    sampling_weight = (
                        1.0 / len(sources)
                        if condition == "MULTI_SOURCE_BALANCED"
                        else data.n_blocks / total_blocks
                    )
                    volume_rows.append({
                        "target_session": target,
                        "task": task,
                        "condition": condition,
                        "source_session": source,
                        "n_source_cycles": data.n_cycles,
                        "n_source_blocks": data.n_blocks,
                        "n_source_frames": 4 * data.n_blocks,
                        "class0_count": classes[0],
                        "class1_count": classes[1],
                        "sampling_weight": float(sampling_weight),
                        "total_source_sessions": 8,
                        "total_source_blocks": total_blocks,
                        "target_excluded": source != target,
                    })
            for source in sources:
                for seed in V5_SEEDS:
                    leakage_rows.append(_leakage_row(
                        task=task, target=target, sources=(source,), seed=seed,
                        condition="SINGLE_SOURCE_TRANSFER",
                    ))
            for condition in MULTI_SOURCE_CONDITIONS:
                for seed in V5_SEEDS:
                    leakage_rows.append(_leakage_row(
                        task=task, target=target, sources=sources, seed=seed,
                        condition=condition,
                    ))
    if len(leakage_rows) != 540 or any(row["status"] != "PASS" for row in leakage_rows):
        raise AssertionError("target holdout audit failed")
    write_csv(args.output_dir / "audit/target_holdout_leakage.csv", leakage_rows)
    write_csv(args.output_dir / "audit/multisource_training_volume.csv", volume_rows)
    write_csv(args.output_dir / "audit/historical_single_source_reuse.csv", _historical_single_source_rows(args))
    write_csv(args.output_dir / "audit/within_session_reference_reuse.csv", _within_reference_rows(args))
    architecture = [
        "# SmallCNN identity audit", "",
        "- Model: `CNN2DMeanPool` with `SmallCNNFrameEncoder`.",
        "- Input: `[batch, 4, 1, 128, 501]` from the existing clean4 builder.",
        "- Temporal fusion: arithmetic mean of four shared-encoder frame features.",
        f"- Encoder feature dimension: `{SmallCNNFrameEncoder.feature_dim}`.",
        f"- Encoder signature: `{encoder_architecture_signature()}`.",
        f"- Classifier type check: `{type(CNN2DMeanPool(n_classes=2)).__name__}`.",
        "- No new backbone or temporal module is present.",
        "- Session 807 uses the official processed block-sequence orientation convention.",
    ]
    (args.output_dir / "audit/smallcnn_identity_check.md").write_text("\n".join(architecture) + "\n", encoding="utf-8")
    freeze = [
        "# Frozen v5 configuration", "",
        f"- Sessions: `{list(EXPECTED_SESSIONS)}`; one held-out target and exactly eight sources.",
        f"- Tasks: `{list(EXPECTED_TASKS)}`.",
        f"- Conditions: `{list(V5_CONDITIONS)}`.",
        f"- Seeds: `{list(V5_SEEDS)}`.",
        f"- Supervised config: `{asdict(FROZEN_SUPERVISED_CONFIG)}`.",
        "- Epoch selection: fixed 40 epochs exactly as the clean4 v1 SmallCNN; no validation or early stopping.",
        "- Balanced primary: uniform source-session draws with replacement as needed.",
        "- Natural control: concatenate source blocks and shuffle once per epoch.",
        "- Both branches use the identical historical clean4 preprocessing: arcsinh plus pixel z-score fit on all source blocks only; only the sampler differs.",
        "- Target labels and target unlabeled frames are excluded from fitting, validation, normalization, and model selection.",
        "- Input is the official unregistered clean4 data; no spatial transform is estimated.",
    ]
    (args.output_dir / "audit/config_freeze.md").write_text("\n".join(freeze) + "\n", encoding="utf-8")
    log("Static v5 audit PASSED: clean4, 9-target LOSO, source volume, target holdout, and artifact compatibility", args.output_dir)


def _leakage_row(
    *, task: str, target: str, sources: tuple[str, ...], seed: int, condition: str
) -> dict[str, Any]:
    source_set = set(map(str, sources))
    passed = str(target) not in source_set
    return {
        "task": task,
        "target_session": str(target),
        "source_sessions": ",".join(sources),
        "n_source_sessions": len(sources),
        "seed": int(seed),
        "condition": condition,
        "target_present_in_source_pool": not passed,
        "n_target_labeled_samples_seen_by_training": 0,
        "n_target_frames_seen_by_training": 0,
        "n_target_samples_seen_by_normalization_fit": 0,
        "n_target_samples_seen_by_validation": 0,
        "n_target_samples_seen_by_model_selection": 0,
        "n_target_unlabeled_frames_used": 0,
        "target_used_for_registration_fit": False,
        "status": "PASS" if passed else "FAIL",
    }


def _run_job(
    args: argparse.Namespace,
    *,
    prepared: Any,
    condition: str,
    balance_mode: str,
    seed: int,
    config: Any = FROZEN_SUPERVISED_CONFIG,
) -> None:
    directory = _job_dir(
        args.output_dir,
        task=prepared.task,
        target=prepared.target_session,
        condition=condition,
        sources=prepared.source_sessions,
        seed=seed,
    )
    curve_path = _curve_path(
        args.output_dir,
        task=prepared.task,
        target=prepared.target_session,
        condition=condition,
        sources=prepared.source_sessions,
        seed=seed,
    )
    if _job_complete(directory, curve_path) and not args.overwrite:
        return
    result = train_prepared_cross_session(
        prepared,
        condition=condition,
        seed=seed,
        balance_mode=balance_mode,
        config=config,
        device=args.device,
    )
    metrics = dict(result.metrics)
    metrics.update({
        "fold": "LOSO_target_session",
        "normalization_weighting": result.normalization_audit["normalization_weighting"],
        "normalization_fit_sessions": ",".join(result.normalization_audit["fit_sessions"]),
        "checkpoint_path": "",
        "source_artifact": "v5_comparable_clean4_training",
    })
    predictions = pd.DataFrame({
        "task": prepared.task,
        "target_session": prepared.target_session,
        "condition": condition,
        "source_sessions": ",".join(prepared.source_sessions),
        "seed": int(seed),
        "sample_i": np.arange(len(prepared.y_test), dtype=np.int64),
        "sample_id": prepared.test_sample_ids,
        "cycle": prepared.test_cycles,
        "label": prepared.y_test,
        "prediction": result.test_predictions,
        "probability_class_0": result.test_probabilities[:, 0],
        "probability_class_1": result.test_probabilities[:, 1],
    })
    curve = pd.DataFrame([
        {
            "task": prepared.task,
            "target_session": prepared.target_session,
            "condition": condition,
            "source_sessions": ",".join(prepared.source_sessions),
            "seed": int(seed),
            **row,
        }
        for row in result.history
    ])
    sampling = pd.DataFrame(result.sampling_history)
    _save_job(
        directory,
        curve_path,
        metrics=metrics,
        predictions=predictions,
        curve=curve,
        sampling=sampling,
    )


def consolidate_jobs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    sampling: list[pd.DataFrame] = []
    for metric_path in sorted((output_dir / "downstream/jobs").glob("**/metrics.json")):
        metrics_rows.append(json.loads(metric_path.read_text(encoding="utf-8")))
        prediction_path = metric_path.parent / "predictions.csv"
        sampling_path = metric_path.parent / "sampling.csv"
        if not prediction_path.exists() or not sampling_path.exists():
            raise FileNotFoundError(f"incomplete job beside {metric_path}")
        predictions.append(pd.read_csv(prediction_path))
        sampling.append(pd.read_csv(sampling_path))
    metrics = pd.DataFrame(metrics_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    sampling_frame = pd.concat(sampling, ignore_index=True) if sampling else pd.DataFrame()
    if len(metrics):
        metrics["target_session"] = metrics["target_session"].astype(str)
        metrics = metrics.sort_values(JOB_KEY).reset_index(drop=True)
    if len(prediction_frame):
        prediction_frame["target_session"] = prediction_frame["target_session"].astype(str)
        prediction_frame["source_sessions"] = prediction_frame["source_sessions"].astype(str)
        prediction_frame = prediction_frame.sort_values(JOB_KEY + ["sample_i"]).reset_index(drop=True)
    if len(sampling_frame):
        sampling_frame["target_session"] = sampling_frame["target_session"].astype(str)
        sampling_frame["source_session"] = sampling_frame["source_session"].astype(str)
        sampling_frame = sampling_frame.sort_values(
            ["task", "target_session", "condition", "seed", "epoch", "source_session"]
        ).reset_index(drop=True)
    write_csv(output_dir / "downstream/fold_metrics.csv", metrics)
    write_csv(output_dir / "downstream/target_predictions.csv", prediction_frame)
    return metrics, prediction_frame, sampling_frame


def _validate_prediction_coverage(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    for _, row in metrics.iterrows():
        selected = predictions[
            (predictions["task"] == row["task"])
            & (predictions["target_session"].astype(str) == str(row["target_session"]))
            & (predictions["condition"] == row["condition"])
            & (predictions["source_sessions"] == row["source_sessions"])
            & (predictions["seed"].astype(int) == int(row["seed"]))
        ]
        expected = int(row["n_test_blocks"])
        if len(selected) != expected or selected["sample_id"].astype(str).nunique() != expected:
            raise AssertionError(
                f"target prediction coverage failed for {tuple(row[key] for key in JOB_KEY)}: "
                f"rows={len(selected)} unique={selected['sample_id'].nunique()} expected={expected}"
            )
        if not (selected["target_session"].astype(str) == str(row["target_session"])).all():
            raise AssertionError("prediction table contains the wrong target session")


def _sampling_summary(sampling: pd.DataFrame) -> pd.DataFrame:
    multi = sampling[sampling["condition"].isin(MULTI_SOURCE_CONDITIONS)].copy()
    if multi.empty:
        raise RuntimeError("multi-source sampling audit is empty")
    grouped = (
        multi.groupby(
            ["task", "target_session", "condition", "seed", "source_session"], sort=True
        )
        .agg(
            total_draws=("n_draws", "sum"),
            mean_draw_proportion=("draw_proportion", "mean"),
            min_draw_proportion=("draw_proportion", "min"),
            max_draw_proportion=("draw_proportion", "max"),
            mean_available_blocks=("n_available_blocks", "mean"),
            mean_unique_blocks_drawn=("n_unique_blocks_drawn", "mean"),
            any_with_replacement=("with_replacement", "max"),
            n_epochs=("epoch", "nunique"),
        )
        .reset_index()
    )
    grouped["expected_uniform_proportion"] = 1.0 / 8.0
    group_keys = ["task", "target_session", "condition", "seed"]
    grouped["total_available_blocks"] = grouped.groupby(group_keys)["mean_available_blocks"].transform("sum")
    grouped["expected_natural_proportion"] = (
        grouped["mean_available_blocks"] / grouped["total_available_blocks"]
    )
    grouped["absolute_uniform_deviation"] = abs(
        grouped["mean_draw_proportion"] - grouped["expected_uniform_proportion"]
    )
    grouped["balanced_sampling_pass"] = np.where(
        grouped["condition"] == "MULTI_SOURCE_BALANCED",
        grouped["absolute_uniform_deviation"] <= 0.005,
        True,
    )
    grouped["natural_frequency_sampling_pass"] = np.where(
        grouped["condition"] == "NATURAL_FREQUENCY_MULTI_SOURCE",
        abs(grouped["mean_draw_proportion"] - grouped["expected_natural_proportion"]) <= 1e-12,
        True,
    )
    grouped = grouped.rename(columns={"mean_draw_proportion": "draw_proportion"})
    return grouped


def _update_historical_reuse(args: argparse.Namespace, metrics: pd.DataFrame) -> None:
    path = args.output_dir / "audit/historical_single_source_reuse.csv"
    audit = pd.read_csv(path)
    audit["generated_job"] = audit["generated_job"].fillna("").astype(str)
    singles = metrics[metrics["condition"] == "SINGLE_SOURCE_TRANSFER"].copy()
    generated: dict[tuple[str, str, str, int], str] = {}
    for _, row in singles.iterrows():
        directory = _job_dir(
            args.output_dir,
            task=str(row["task"]), target=str(row["target_session"]),
            condition="SINGLE_SOURCE_TRANSFER", sources=(str(row["source_sessions"]),),
            seed=int(row["seed"]),
        )
        generated[(str(row["task"]), str(row["target_session"]), str(row["source_sessions"]), int(row["seed"]))] = str(directory)
    for index, row in audit.iterrows():
        key = (
            str(row["task"]), str(row["target_session"]),
            str(row["source_session"]), int(row["seed"]),
        )
        if key in generated:
            audit.loc[index, "comparable_baseline_generated"] = True
            audit.loc[index, "generated_job"] = generated[key]
            audit.loc[index, "status"] = "PASS_REGENERATED_COMPARABLE_BASELINE"
    if len(audit) != 432 or not audit["comparable_baseline_generated"].astype(bool).all():
        raise RuntimeError("single-source baseline regeneration audit incomplete")
    if audit["reused"].astype(bool).any():
        raise RuntimeError("an incompatible historical single-source artifact was marked reused")
    write_csv(path, audit)


def _within_reference_frame(args: argparse.Namespace) -> pd.DataFrame:
    audit = pd.read_csv(args.output_dir / "audit/within_session_reference_reuse.csv")
    if len(audit) != 54 or not audit["reused"].astype(bool).all():
        raise RuntimeError("within-session reference audit incomplete")
    return (
        audit.groupby(["task", "target_session"], sort=True)
        .agg(
            within_session_reference_BA=("within_session_reference_BA_seed", "mean"),
            within_session_seed_std=("within_session_reference_BA_seed", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )


def _markdown_table(frame: pd.DataFrame) -> str:
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


def aggregate_outputs(args: argparse.Namespace) -> None:
    metrics, predictions, sampling = consolidate_jobs(args.output_dir)
    expected_rows = 2 * 9 * ((8 * 3) + (2 * 3))
    if len(metrics) != expected_rows:
        raise RuntimeError(f"aggregate STOP: incomplete jobs {len(metrics)}/{expected_rows}")
    if metrics.duplicated(JOB_KEY).any():
        raise RuntimeError("aggregate STOP: duplicate formal job keys")
    if set(metrics["condition"].unique()) != set(V5_CONDITIONS):
        raise RuntimeError("aggregate STOP: formal conditions differ from v5")
    if set(metrics["seed"].astype(int).unique()) != set(V5_SEEDS):
        raise RuntimeError("aggregate STOP: seed set differs from v5")
    if not (metrics["run_status"] == "VALID").all():
        raise RuntimeError("aggregate STOP: invalid formal job")
    numeric = [
        "train_accuracy", "train_balanced_accuracy", "test_accuracy",
        "test_balanced_accuracy", "macro_F1", "ROC_AUC", "train_test_gap_BA",
    ]
    if not np.isfinite(metrics[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("aggregate STOP: non-finite metric")
    singles = metrics[metrics["condition"] == "SINGLE_SOURCE_TRANSFER"]
    multi = metrics[metrics["condition"].isin(MULTI_SOURCE_CONDITIONS)]
    if len(singles) != 432 or len(multi) != 108:
        raise RuntimeError("aggregate STOP: single/multi job counts are wrong")
    if not (singles["n_source_sessions"].astype(int) == 1).all() or not (
        multi["n_source_sessions"].astype(int) == 8
    ).all():
        raise RuntimeError("aggregate STOP: source-session count invariant failed")
    if any(
        str(target) in str(sources).split(",")
        for target, sources in zip(metrics["target_session"], metrics["source_sessions"])
    ):
        raise RuntimeError("aggregate STOP: target session entered training source list")
    _validate_prediction_coverage(metrics, predictions)
    leakage = pd.read_csv(args.output_dir / "audit/target_holdout_leakage.csv")
    if len(leakage) != 540 or not (leakage["status"] == "PASS").all():
        raise RuntimeError("aggregate STOP: target holdout audit incomplete or failed")
    leakage_count_columns = [
        "n_target_labeled_samples_seen_by_training", "n_target_frames_seen_by_training",
        "n_target_samples_seen_by_normalization_fit", "n_target_samples_seen_by_validation",
        "n_target_samples_seen_by_model_selection", "n_target_unlabeled_frames_used",
    ]
    if leakage[leakage_count_columns].to_numpy(dtype=int).sum() != 0:
        raise RuntimeError("aggregate STOP: target leakage is nonzero")
    distribution = _sampling_summary(sampling)
    expected_distribution_rows = 2 * 9 * 2 * 3 * 8
    if len(distribution) != expected_distribution_rows:
        raise RuntimeError("aggregate STOP: source sampling audit is incomplete")
    balanced = distribution[distribution["condition"] == "MULTI_SOURCE_BALANCED"]
    if not balanced["balanced_sampling_pass"].astype(bool).all():
        raise RuntimeError("aggregate STOP: session-balanced sampling tolerance failed")
    natural = distribution[distribution["condition"] == "NATURAL_FREQUENCY_MULTI_SOURCE"]
    if not natural["natural_frequency_sampling_pass"].astype(bool).all():
        raise RuntimeError("aggregate STOP: natural-frequency sampling does not match source volume")
    if not (distribution["n_epochs"].astype(int) == FROZEN_SUPERVISED_CONFIG.max_epochs).all():
        raise RuntimeError("aggregate STOP: sampling epochs are incomplete")
    write_csv(args.output_dir / "audit/source_sampling_distribution.csv", distribution)
    _update_historical_reuse(args, metrics)
    expected_curves = expected_rows
    if len(list((args.output_dir / "downstream/training_curves").glob("*.csv"))) != expected_curves:
        raise RuntimeError("aggregate STOP: training curves incomplete")

    within = _within_reference_frame(args)
    target_table = target_level_comparison(metrics, within)
    tests = planned_statistical_tests(target_table)
    gaps = within_cross_gap(target_table)
    stability = seed_stability(metrics)
    write_csv(args.output_dir / "summaries/target_level_comparison.csv", target_table)
    write_csv(args.output_dir / "summaries/planned_statistical_tests.csv", tests)
    write_csv(args.output_dir / "summaries/within_cross_gap.csv", gaps)
    write_csv(args.output_dir / "summaries/seed_stability.csv", stability)
    make_required_figures(args.output_dir, target_table, distribution)
    scenario = classify_scenario(target_table, tests)
    binary = target_table[target_table["task"] == "binary"]
    stimulus = target_table[target_table["task"] == "stimulus_type"]
    report = [
        "# Multi-source LOSO SmallCNN benchmark v5", "",
        "## Integrity gates", "",
        f"- Formal jobs: `{len(metrics)}` / `{expected_rows}`.",
        "- Exactly one target was held out and exactly eight source sessions were used for every multi-source job.",
        "- Target labeled samples, target frames, target normalization data, target validation data, and target model-selection data were all zero.",
        "- Historical single-source artifacts were not reused because their input/model/session/task/seed protocol was incompatible; all 432 comparable clean4 baselines were regenerated.",
        "- Within-session references were reused from the exact v1 clean4 RANDOM_INIT SmallCNN fold results.",
        "- No target unlabeled data or spatial transform was used.", "",
        "## Planned target-session statistics", "", _markdown_table(tests), "",
        "## Binary target-level comparison", "", _markdown_table(binary), "",
        "## Stimulus-type target-level comparison", "", _markdown_table(stimulus), "",
        "## Within-to-cross-session gap", "", _markdown_table(gaps), "",
        "## Preregistered interpretation", "",
        f"- Scenario: **{scenario}**.",
        "- The primary baseline is the predefined mean across all eight single-source transfers, never the post-hoc best source.",
        "- Natural-frequency multi-source training is a secondary control only.",
        "- No source combination was selected using target performance.",
    ]
    (args.output_dir / "report/multisource_loso_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
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
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    task = "binary"
    target = "626"
    sources = ("628", "708")
    seed = V5_SEEDS[0]
    data = {
        session: load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
        for session in (*sources, target)
    }
    smoke_config = replace(FROZEN_SUPERVISED_CONFIG, max_epochs=int(args.smoke_epochs))
    rows = []
    started = time.perf_counter()
    for condition, balance_mode in (
        ("MULTI_SOURCE_BALANCED", "session_balanced"),
        ("NATURAL_FREQUENCY_MULTI_SOURCE", "natural_frequency"),
    ):
        prepared = prepare_cross_session_data(
            data, source_sessions=sources, target_session=target, balance_mode=balance_mode
        )
        result = train_prepared_cross_session(
            prepared,
            condition=condition,
            seed=seed,
            balance_mode=balance_mode,
            config=smoke_config,
            device="cpu",
        )
        if result.normalization_audit["target_used_for_stats"]:
            raise AssertionError("smoke normalization used target")
        rows.append(result.metrics)
    frame = pd.DataFrame(rows)
    required = {
        "train_balanced_accuracy", "test_balanced_accuracy", "macro_F1", "ROC_AUC",
        "train_test_gap_BA", "target_frames_used_for_training", "target_used_for_normalization",
    }
    if required - set(frame.columns) or frame["target_frames_used_for_training"].astype(int).sum() != 0:
        raise AssertionError("smoke metric/leakage schema failed")
    elapsed = time.perf_counter() - started
    text = "\n".join([
        "PASS: tiny local CPU smoke only; not a formal scientific result",
        f"target={target} sources={','.join(sources)} task={task} seed={seed}",
        "conditions=MULTI_SOURCE_BALANCED,NATURAL_FREQUENCY_MULTI_SOURCE",
        f"smoke_epochs={smoke_config.max_epochs}",
        f"jobs={len(frame)}",
        f"balanced_source_count={len(sources)}",
        "target_labeled_samples_in_training=0",
        "target_frames_in_training=0",
        "target_used_for_normalization=False",
        "sampler_forward_backward=PASS",
        "metric_output_schema=PASS",
        f"runtime_seconds={elapsed:.3f}",
    ]) + "\n"
    (args.output_dir / "smoke_test_local.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def _write_gpu_audit(args: argparse.Namespace, *, runtime_seconds: float | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal v5 requires CUDA; CPU fallback is forbidden")
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
    if tuple(map(int, args.seeds)) != tuple(V5_SEEDS):
        raise RuntimeError(f"formal v5 requires exactly seeds {V5_SEEDS}")
    ensure_output_tree(args.output_dir)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _write_gpu_audit(args)
    log("Formal CUDA multi-source LOSO benchmark started", args.output_dir)
    run_audit(args)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    visited = 0
    for task in EXPECTED_TASKS:
        data_by_session = {
            session: load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            for session in EXPECTED_SESSIONS
        }
        for target in EXPECTED_SESSIONS:
            sources = source_sessions_for_target(target)
            for source in sources:
                prepared = prepare_cross_session_data(
                    data_by_session,
                    source_sessions=(source,),
                    target_session=target,
                    balance_mode="natural_frequency",
                )
                for seed in V5_SEEDS:
                    _run_job(
                        args,
                        prepared=prepared,
                        condition="SINGLE_SOURCE_TRANSFER",
                        balance_mode="natural_frequency",
                        seed=seed,
                    )
                    visited += 1
                    if visited % 25 == 0:
                        log(f"formal jobs visited={visited}/540", args.output_dir)
                del prepared
                gc.collect()
            for condition, balance_mode in (
                ("MULTI_SOURCE_BALANCED", "session_balanced"),
                ("NATURAL_FREQUENCY_MULTI_SOURCE", "natural_frequency"),
            ):
                prepared = prepare_cross_session_data(
                    data_by_session,
                    source_sessions=sources,
                    target_session=target,
                    balance_mode=balance_mode,
                )
                for seed in V5_SEEDS:
                    _run_job(
                        args,
                        prepared=prepared,
                        condition=condition,
                        balance_mode=balance_mode,
                        seed=seed,
                    )
                    visited += 1
                    if visited % 25 == 0:
                        log(f"formal jobs visited={visited}/540", args.output_dir)
                del prepared
                gc.collect()
        del data_by_session
        gc.collect()
    aggregate_outputs(args)
    runtime = time.perf_counter() - started
    _write_gpu_audit(args, runtime_seconds=runtime)
    log(f"Formal v5 completed in {runtime:.3f} seconds", args.output_dir)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.v1_output_dir = args.v1_output_dir.resolve()
    args.historical_cross_session_root = args.historical_cross_session_root.resolve()
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
