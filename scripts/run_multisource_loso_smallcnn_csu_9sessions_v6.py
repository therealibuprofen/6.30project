#!/usr/bin/env python3
"""SmallCNN + CSU multi-source leave-one-session-out benchmark v6."""

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
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import (
    CNN2DMeanPool,
    SmallCNNFrameEncoder,
    encoder_architecture_signature,
)
from ultrasound_decoding.multisource_csu_reporting_v6 import (
    classify_scenario,
    make_required_figures,
    planned_statistical_tests,
    seed_stability,
    target_level_csu_comparison,
    within_cross_gap,
)
from ultrasound_decoding.multisource_csu_v6 import (
    CSU_ALPHA,
    CSU_EPSILON,
    CSU_INSERTION_POINT,
    CSU_OFFICIAL_COMMIT,
    CSU_OFFICIAL_REPOSITORY,
    CSU_PAPER_URL,
    CSU_PROBABILITY,
    CSU_PROJECTED_EIGENVALUE_FLOOR,
    CSU_SUPPLEMENT_URL,
    CSUCNN2DMeanPool,
    CSUSmallCNNFrameEncoder,
    FROZEN_SUPERVISED_CONFIG,
    REQUIRED_FORMAL_OUTPUTS,
    V6_CONDITIONS,
    V6_SEEDS,
    assert_formal_cuda,
    missing_formal_outputs,
    resolve_v5_artifact_dir,
    train_prepared_csu,
    v5_baseline_compatibility,
)
from ultrasound_decoding.multisource_loso_v5 import (
    prepare_cross_session_data,
    source_sessions_for_target,
    train_prepared_cross_session,
)


RUN_NAME = "multisource_loso_smallcnn_csu_9sessions_v6"
V5_RUN_NAME = "multisource_loso_smallcnn_9sessions_v5"
V1_RUN_NAME = "ssl_masked_smallcnn_clean4_9sessions_v1"
JOB_KEY = ["task", "target_session", "condition", "source_sessions", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "formal", "aggregate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--v5-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V5_RUN_NAME)
    parser.add_argument("--v1-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V1_RUN_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(V6_SEEDS))
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


def _job_dir(output_dir: Path, task: str, target: str, condition: str, seed: int) -> Path:
    return output_dir / "downstream/jobs" / task / f"target_{target}" / condition / f"seed_{seed}"


def _curve_path(output_dir: Path, task: str, target: str, condition: str, seed: int) -> Path:
    return output_dir / "downstream/training_curves" / (
        f"{task}_target_{target}_{condition}_seed_{seed}.csv"
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


def _job_complete(output_dir: Path, task: str, target: str, condition: str, seed: int) -> bool:
    directory = _job_dir(output_dir, task, target, condition, seed)
    required = [directory / "metrics.json", directory / "predictions.csv"]
    if condition == "MULTI_SOURCE_CSU":
        required.append(directory / "batch_domain_diversity.csv")
    return all(path.is_file() and path.stat().st_size > 0 for path in required) and _curve_path(
        output_dir, task, target, condition, seed
    ).is_file()


def _save_job(
    output_dir: Path,
    *,
    task: str,
    target: str,
    condition: str,
    seed: int,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    curve: pd.DataFrame,
    diversity: pd.DataFrame | None = None,
) -> None:
    directory = _job_dir(output_dir, task, target, condition, seed)
    curve_path = _curve_path(output_dir, task, target, condition, seed)
    directory.mkdir(parents=True, exist_ok=True)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory / "metrics.json.tmp"
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(directory / "metrics.json")
    predictions.to_csv(directory / "predictions.csv", index=False)
    curve.to_csv(curve_path, index=False)
    if diversity is not None:
        diversity.to_csv(directory / "batch_domain_diversity.csv", index=False)


def _expected_v5_curve(root: Path, task: str, target: str, seed: int) -> Path:
    sources = "_".join(source_sessions_for_target(target))
    return root / "downstream/training_curves" / (
        f"{task}_target_{target}_MULTI_SOURCE_BALANCED_source_{sources}_seed_{seed}.csv"
    )


def _load_v5_tables(args: argparse.Namespace) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    root = resolve_v5_artifact_dir(args.v5_output_dir)
    metrics = pd.read_csv(root / "downstream/fold_metrics.csv")
    predictions = pd.read_csv(root / "downstream/target_predictions.csv")
    metrics["target_session"] = metrics["target_session"].astype(str)
    predictions["target_session"] = predictions["target_session"].astype(str)
    return root, metrics, predictions


def audit_v5_baseline_reuse(
    args: argparse.Namespace, *, materialize: bool
) -> tuple[pd.DataFrame, set[tuple[str, str, int]]]:
    rows: list[dict[str, Any]] = []
    reused_keys: set[tuple[str, str, int]] = set()
    try:
        root, metrics, predictions = _load_v5_tables(args)
        architecture_text = (root / "audit/smallcnn_identity_check.md").read_text(encoding="utf-8")
        config_text = (root / "audit/config_freeze.md").read_text(encoding="utf-8")
        global_reason = []
        if "CNN2DMeanPool" not in architecture_text or "SmallCNNFrameEncoder" not in architecture_text:
            global_reason.append("v5 SmallCNN identity audit mismatch")
        if str(asdict(FROZEN_SUPERVISED_CONFIG)) not in config_text:
            global_reason.append("v5 supervised config audit mismatch")
        if str(list(V6_SEEDS)) not in config_text:
            global_reason.append("v5 seed audit mismatch")
    except (FileNotFoundError, RuntimeError, pd.errors.ParserError) as error:
        root = Path(args.v5_output_dir)
        metrics = pd.DataFrame()
        predictions = pd.DataFrame()
        global_reason = [f"v5 artifact unavailable: {error}"]

    for task in EXPECTED_TASKS:
        for target in EXPECTED_SESSIONS:
            sources = ",".join(source_sessions_for_target(target))
            for seed in V6_SEEDS:
                reason = list(global_reason)
                selected = pd.DataFrame()
                if not metrics.empty:
                    selected = metrics[
                        (metrics["task"] == task)
                        & (metrics["target_session"].astype(str) == target)
                        & (metrics["seed"].astype(int) == int(seed))
                        & (metrics["condition"] == "MULTI_SOURCE_BALANCED")
                    ]
                    if len(selected) != 1:
                        reason.append(f"expected one v5 baseline fold, found {len(selected)}")
                    else:
                        compatible, detail = v5_baseline_compatibility(
                            selected.iloc[0], task=task, target=target, seed=seed
                        )
                        if not compatible:
                            reason.append(detail)
                pred = pd.DataFrame()
                curve_path = _expected_v5_curve(root, task, target, seed)
                curve = pd.DataFrame()
                if len(selected) == 1 and not reason:
                    metric_row = selected.iloc[0]
                    pred = predictions[
                        (predictions["task"] == task)
                        & (predictions["target_session"].astype(str) == target)
                        & (predictions["seed"].astype(int) == int(seed))
                        & (predictions["condition"] == "MULTI_SOURCE_BALANCED")
                        & (predictions["source_sessions"].astype(str) == sources)
                    ].copy()
                    expected = int(metric_row["n_test_blocks"])
                    if len(pred) != expected or pred["sample_id"].astype(str).nunique() != expected:
                        reason.append("v5 target-prediction coverage mismatch")
                    if not curve_path.is_file():
                        reason.append("v5 training curve missing")
                    else:
                        curve = pd.read_csv(curve_path)
                        if len(curve) != FROZEN_SUPERVISED_CONFIG.max_epochs:
                            reason.append("v5 training curve epoch count mismatch")
                reusable = not reason and len(selected) == 1
                artifact = str(root / "downstream/fold_metrics.csv")
                rows.append({
                    "target": target,
                    "task": task,
                    "seed": int(seed),
                    "artifact_path": artifact,
                    "reused": bool(reusable),
                    "reason": "exact frozen v5 MULTI_SOURCE_BALANCED renamed MULTI_SOURCE_ERM" if reusable else "; ".join(reason),
                })
                if reusable:
                    key = (task, target, int(seed))
                    reused_keys.add(key)
                    if materialize and not (
                        _job_complete(args.output_dir, task, target, "MULTI_SOURCE_ERM", seed)
                        and not args.overwrite
                    ):
                        metric_dict = selected.iloc[0].to_dict()
                        metric_dict.update({
                            "condition": "MULTI_SOURCE_ERM",
                            "source_artifact": artifact,
                            "v5_baseline_reused": True,
                            "csu_alpha": np.nan,
                            "csu_probability": np.nan,
                            "csu_epsilon": np.nan,
                            "csu_insertion_point": "none",
                            "target_unlabeled_adaptation": False,
                        })
                        pred["condition"] = "MULTI_SOURCE_ERM"
                        curve["condition"] = "MULTI_SOURCE_ERM"
                        _save_job(
                            args.output_dir,
                            task=task,
                            target=target,
                            condition="MULTI_SOURCE_ERM",
                            seed=seed,
                            metrics=metric_dict,
                            predictions=pred,
                            curve=curve,
                        )
    audit = pd.DataFrame(rows)
    if len(audit) != 54:
        raise AssertionError("v5 reuse audit must contain exactly 54 task-target-seed rows")
    write_csv(args.output_dir / "audit/v5_baseline_reuse.csv", audit)
    return audit, reused_keys


def _leakage_rows() -> list[dict[str, Any]]:
    rows = []
    for task in EXPECTED_TASKS:
        for target in EXPECTED_SESSIONS:
            sources = source_sessions_for_target(target)
            for condition in V6_CONDITIONS:
                for seed in V6_SEEDS:
                    target_seen = target in sources
                    rows.append({
                        "target": target,
                        "task": task,
                        "seed": int(seed),
                        "condition": condition,
                        "source_sessions": ",".join(sources),
                        "n_source_sessions": len(sources),
                        "target_seen_in_training": bool(target_seen),
                        "target_seen_in_normalization": False,
                        "target_seen_in_validation": False,
                        "target_seen_in_model_selection": False,
                        "target_seen_in_csu_cache": False,
                        "target_unlabeled_adaptation": False,
                        "target_frames_seen": 0,
                        "registration_used": False,
                        "status": "FAIL" if target_seen else "PASS",
                    })
    return rows


def run_audit(args: argparse.Namespace) -> None:
    ensure_output_tree(args.output_dir)
    if tuple(map(int, args.seeds)) != V6_SEEDS:
        raise RuntimeError(f"v6 requires exactly the fixed seeds {V6_SEEDS}")
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    for task in EXPECTED_TASKS:
        for session in EXPECTED_SESSIONS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            if tuple(data.X.shape[1:]) != (4, 128, 501):
                raise AssertionError(f"{task} session {session} is not frozen clean4")
            if set(np.unique(data.y).astype(int).tolist()) != {0, 1}:
                raise AssertionError(f"{task} session {session} label definition changed")
    leakage = _leakage_rows()
    if len(leakage) != 108 or any(row["status"] != "PASS" for row in leakage):
        raise AssertionError("v6 target-holdout audit failed")
    write_csv(args.output_dir / "audit/target_holdout_leakage.csv", leakage)
    audit_v5_baseline_reuse(args, materialize=False)

    baseline = CNN2DMeanPool(n_classes=2, dropout=0.25, temporal_length=4)
    csu_model = CSUCNN2DMeanPool(n_classes=2, dropout=0.25, temporal_length=4)
    identity = [
        "# SmallCNN identity check", "",
        f"- Frozen encoder signature: `{encoder_architecture_signature()}`.",
        f"- ERM encoder feature dimension: `{baseline.encoder.feature_dim}`.",
        f"- CSU encoder feature dimension: `{csu_model.encoder.feature_dim}`.",
        "- Both conditions use four shared frame encodes, arithmetic feature mean, dropout 0.25, and one linear classifier.",
        "- CSU reuses the exact nine frozen SmallCNN layers; only their module grouping exposes the block-1 insertion boundary.",
        "- No LSTM, temporal convolution, attention, Transformer, Mamba, FCNN, or new fusion is present.",
        "- Input remains unregistered clean4; session 807 uses the frozen processed orientation convention.",
    ]
    (args.output_dir / "audit/smallcnn_identity_check.md").write_text(
        "\n".join(identity) + "\n", encoding="utf-8"
    )
    implementation = [
        "# CSU implementation audit", "",
        f"- Paper: {CSU_PAPER_URL}",
        f"- Supplementary pseudocode: {CSU_SUPPLEMENT_URL}",
        f"- Author repository: {CSU_OFFICIAL_REPOSITORY}",
        f"- Author repository commit inspected: `{CSU_OFFICIAL_COMMIT}`.",
        "- Official source mapped: `multi-domain-generalization/dassl/modeling/backbone/uresnet.py`, class `CorrelatedDistributionUncertainty`.",
        f"- Frozen canonical multi-domain classification parameters: `alpha={CSU_ALPHA}`, `p={CSU_PROBABILITY}`, `eps={CSU_EPSILON}`.",
        "- Style statistics: per-instance channel mean and unbiased spatial variance-derived standard deviation.",
        "- Covariance: batch-centered mean and standard-deviation matrices, each divided by batch feature-map count.",
        "- Decomposition: eigenvectors from `eigh(C * covariance + eps * I)` inside `no_grad`, matching the author code.",
        f"- Stabilization: projected eigenvalues clamp at `{CSU_PROJECTED_EIGENVALUE_FLOOR}` before square root; every intermediate and output is asserted finite.",
        "- Random sampling: one module-level Bernoulli gate per forward call; per-instance `Beta(alpha, alpha)` intensity; independent standard Gaussian vectors for mean and standard deviation transformed by the correlated square roots.",
        "- Gradient behavior: no eigenvector gradient; covariance-intensity path remains differentiable, matching the supplementary explanation.",
        "- Train/eval: stochastic CSU only in `train()`; `eval()` returns the input exactly and uses no target/session/class metadata.",
        "- Safety tightening: unlike the author's identity fallback after decomposition failure, v6 raises a hard error so no invalid formal step is silently continued.",
    ]
    (args.output_dir / "audit/csu_implementation_audit.md").write_text(
        "\n".join(implementation) + "\n", encoding="utf-8"
    )
    insertion = [
        "# CSU insertion point", "",
        f"- Frozen name: `{CSU_INSERTION_POINT}`.",
        "- Code path: input frame -> Conv2d(1,8,5x9) -> BatchNorm2d(8) -> ReLU -> MaxPool2d(2x4) -> CSU -> remaining SmallCNN.",
        "- Feature shape for a 128x501 frame: `[B_frames, 8, 64, 125]`.",
        "- Exactly one `CorrelatedStyleUncertainty` instance exists in `CSUSmallCNNFrameEncoder`.",
        "- No insertion-position search or target-based selection is implemented.",
    ]
    (args.output_dir / "audit/csu_insertion_point.md").write_text(
        "\n".join(insertion) + "\n", encoding="utf-8"
    )
    freeze = [
        "# Frozen v6 configuration", "",
        f"- Sessions: `{list(EXPECTED_SESSIONS)}`; strict eight-source/one-unseen-target LOSO.",
        f"- Tasks: `{list(EXPECTED_TASKS)}`.",
        f"- Conditions: `{list(V6_CONDITIONS)}`.",
        f"- Seeds: `{list(V6_SEEDS)}`.",
        f"- Supervised config inherited exactly from v5: `{asdict(FROZEN_SUPERVISED_CONFIG)}`.",
        f"- CSU config: `alpha={CSU_ALPHA}`, `p={CSU_PROBABILITY}`, `eps={CSU_EPSILON}`, insertion=`{CSU_INSERTION_POINT}`.",
        "- No CSU hyperparameter, strength, or insertion-position search is permitted.",
        "- Session-balanced supervised sampling is inherited unchanged from v5.",
        "- Fixed 40 epochs; no validation, patience, early stopping, class weighting, or target model selection.",
        "- No target labeled/unlabeled use, target adaptation, target SSL, TTA, or registration.",
    ]
    (args.output_dir / "audit/config_freeze.md").write_text(
        "\n".join(freeze) + "\n", encoding="utf-8"
    )
    log("Static v6 audit PASSED: official CSU, clean4, fixed SmallCNN insertion, LOSO holdout", args.output_dir)


def _predictions_frame(prepared: Any, condition: str, seed: int, result: Any) -> pd.DataFrame:
    return pd.DataFrame({
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


def _run_erm_job(args: argparse.Namespace, prepared: Any, seed: int, reason: str) -> None:
    if _job_complete(args.output_dir, prepared.task, prepared.target_session, "MULTI_SOURCE_ERM", seed) and not args.overwrite:
        return
    result = train_prepared_cross_session(
        prepared,
        condition="MULTI_SOURCE_BALANCED",
        seed=seed,
        balance_mode="session_balanced",
        config=FROZEN_SUPERVISED_CONFIG,
        device=args.device,
    )
    metrics = dict(result.metrics)
    metrics.update({
        "condition": "MULTI_SOURCE_ERM",
        "fold": "LOSO_target_session",
        "normalization_weighting": result.normalization_audit["normalization_weighting"],
        "normalization_fit_sessions": ",".join(result.normalization_audit["fit_sessions"]),
        "source_artifact": "v6_exact_ERM_rerun",
        "v5_baseline_reused": False,
        "v5_nonreuse_reason": reason,
        "target_unlabeled_adaptation": False,
        "csu_alpha": np.nan,
        "csu_probability": np.nan,
        "csu_epsilon": np.nan,
        "csu_insertion_point": "none",
    })
    curve = pd.DataFrame([
        {
            "task": prepared.task,
            "target_session": prepared.target_session,
            "condition": "MULTI_SOURCE_ERM",
            "source_sessions": ",".join(prepared.source_sessions),
            "seed": int(seed),
            **row,
        }
        for row in result.history
    ])
    _save_job(
        args.output_dir,
        task=prepared.task,
        target=prepared.target_session,
        condition="MULTI_SOURCE_ERM",
        seed=seed,
        metrics=metrics,
        predictions=_predictions_frame(prepared, "MULTI_SOURCE_ERM", seed, result),
        curve=curve,
    )


def _run_csu_job(
    args: argparse.Namespace,
    prepared: Any,
    seed: int,
    *,
    config: Any = FROZEN_SUPERVISED_CONFIG,
) -> Any:
    if _job_complete(args.output_dir, prepared.task, prepared.target_session, "MULTI_SOURCE_CSU", seed) and not args.overwrite:
        return None
    result = train_prepared_csu(prepared, seed=seed, config=config, device=args.device)
    metrics = dict(result.metrics)
    metrics.update({
        "fold": "LOSO_target_session",
        "normalization_weighting": result.normalization_audit["normalization_weighting"],
        "normalization_fit_sessions": ",".join(result.normalization_audit["fit_sessions"]),
        "source_artifact": "v6_author_aligned_CSU_training",
        "v5_baseline_reused": False,
    })
    curve = pd.DataFrame([
        {
            "task": prepared.task,
            "target_session": prepared.target_session,
            "condition": "MULTI_SOURCE_CSU",
            "source_sessions": ",".join(prepared.source_sessions),
            "seed": int(seed),
            **row,
        }
        for row in result.history
    ])
    _save_job(
        args.output_dir,
        task=prepared.task,
        target=prepared.target_session,
        condition="MULTI_SOURCE_CSU",
        seed=seed,
        metrics=metrics,
        predictions=_predictions_frame(prepared, "MULTI_SOURCE_CSU", seed, result),
        curve=curve,
        diversity=pd.DataFrame(result.batch_domain_diversity),
    )
    return result


def consolidate_jobs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    diversity: list[pd.DataFrame] = []
    for path in sorted((output_dir / "downstream/jobs").glob("**/metrics.json")):
        metric_rows.append(json.loads(path.read_text(encoding="utf-8")))
        prediction_path = path.parent / "predictions.csv"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"prediction file missing beside {path}")
        predictions.append(pd.read_csv(prediction_path))
        diversity_path = path.parent / "batch_domain_diversity.csv"
        if diversity_path.is_file():
            diversity.append(pd.read_csv(diversity_path))
    metrics = pd.DataFrame(metric_rows)
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    batch = pd.concat(diversity, ignore_index=True) if diversity else pd.DataFrame()
    if len(metrics):
        metrics["target_session"] = metrics["target_session"].astype(str)
        metrics = metrics.sort_values(JOB_KEY).reset_index(drop=True)
    if len(pred):
        pred["target_session"] = pred["target_session"].astype(str)
        pred["source_sessions"] = pred["source_sessions"].astype(str)
        pred = pred.sort_values(JOB_KEY + ["sample_i"]).reset_index(drop=True)
    if len(batch):
        batch["target"] = batch["target"].astype(str)
        batch = batch.sort_values(["task", "target", "seed", "epoch", "batch_index"]).reset_index(drop=True)
    write_csv(output_dir / "downstream/fold_metrics.csv", metrics)
    write_csv(output_dir / "downstream/target_predictions.csv", pred)
    write_csv(output_dir / "audit/csu_batch_domain_diversity.csv", batch)
    return metrics, pred, batch


def _within_reference(args: argparse.Namespace) -> pd.DataFrame:
    try:
        root = resolve_v5_artifact_dir(args.v5_output_dir)
        summary_path = root / "summaries/target_level_comparison.csv"
        summary = pd.read_csv(summary_path)
        output = summary[["task", "target_session", "within_session_reference_BA"]].copy()
    except (FileNotFoundError, RuntimeError, KeyError):
        path = args.v1_output_dir / "downstream/fold_metrics.csv"
        if not path.is_file():
            nested = list(args.v1_output_dir.glob("**/downstream/fold_metrics.csv"))
            if len(nested) != 1:
                raise FileNotFoundError("within-session v1 reference unavailable")
            path = nested[0]
        metrics = pd.read_csv(path)
        selected = metrics[
            (metrics["condition"] == "RANDOM_INIT")
            & metrics["seed"].astype(int).isin(V6_SEEDS)
        ].copy()
        selected = selected.rename(columns={"session": "target_session"})
        output = (
            selected.groupby(["task", "target_session"], sort=True)
            .agg(within_session_reference_BA=("test_balanced_accuracy", "mean"))
            .reset_index()
        )
    output["target_session"] = output["target_session"].astype(str)
    if len(output) != 18 or output.groupby("task")["target_session"].nunique().to_dict() != {
        "binary": 9, "stimulus_type": 9
    }:
        raise AssertionError("within-session reference does not cover all task-target combinations")
    return output


def _validate_predictions(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    for _, row in metrics.iterrows():
        selected = predictions[
            (predictions["task"] == row["task"])
            & (predictions["target_session"].astype(str) == str(row["target_session"]))
            & (predictions["condition"] == row["condition"])
            & (predictions["source_sessions"].astype(str) == str(row["source_sessions"]))
            & (predictions["seed"].astype(int) == int(row["seed"]))
        ]
        expected = int(row["n_test_blocks"])
        if len(selected) != expected or selected["sample_id"].astype(str).nunique() != expected:
            raise AssertionError(f"prediction coverage failed for {tuple(row[key] for key in JOB_KEY)}")
    paired_keys = ["task", "target_session", "seed", "sample_id"]
    erm = predictions[predictions["condition"] == "MULTI_SOURCE_ERM"][paired_keys + ["label"]].copy()
    csu = predictions[predictions["condition"] == "MULTI_SOURCE_CSU"][paired_keys + ["label"]].copy()
    paired = erm.merge(csu, on=paired_keys, how="outer", suffixes=("_ERM", "_CSU"), indicator=True)
    if not (paired["_merge"] == "both").all() or not (
        paired["label_ERM"].astype(int) == paired["label_CSU"].astype(int)
    ).all():
        raise AssertionError("ERM and CSU do not evaluate identical frozen target samples and labels")


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
    metrics, predictions, diversity = consolidate_jobs(args.output_dir)
    expected_rows = 2 * 9 * 2 * 3
    if len(metrics) != expected_rows:
        raise RuntimeError(f"aggregate STOP: incomplete jobs {len(metrics)}/{expected_rows}")
    if metrics.duplicated(JOB_KEY).any():
        raise RuntimeError("aggregate STOP: duplicate formal job keys")
    if set(metrics["condition"].unique()) != set(V6_CONDITIONS):
        raise RuntimeError("aggregate STOP: conditions differ from frozen v6")
    if set(metrics["seed"].astype(int).unique()) != set(V6_SEEDS):
        raise RuntimeError("aggregate STOP: seeds differ from frozen v6")
    if not (metrics["n_source_sessions"].astype(int) == 8).all():
        raise RuntimeError("aggregate STOP: every fold must have exactly eight sources")
    if not (metrics["run_status"] == "VALID").all():
        raise RuntimeError("aggregate STOP: invalid formal job")
    if any(
        str(target) in str(sources).split(",")
        for target, sources in zip(metrics["target_session"], metrics["source_sessions"])
    ):
        raise RuntimeError("aggregate STOP: target session entered a source pool")
    numeric = [
        "train_accuracy", "train_balanced_accuracy", "test_accuracy",
        "test_balanced_accuracy", "macro_F1", "ROC_AUC", "train_test_gap_BA",
    ]
    if not np.isfinite(metrics[numeric].to_numpy(dtype=float)).all():
        raise RuntimeError("aggregate STOP: non-finite formal metric")
    _validate_predictions(metrics, predictions)
    leakage = pd.read_csv(args.output_dir / "audit/target_holdout_leakage.csv")
    if len(leakage) != 108 or not (leakage["status"] == "PASS").all():
        raise RuntimeError("aggregate STOP: target leakage audit failed")
    if leakage["target_seen_in_training"].astype(bool).any() or leakage["target_frames_seen"].astype(int).sum() != 0:
        raise RuntimeError("aggregate STOP: nonzero target exposure")
    if diversity.empty or not set(diversity["target"].astype(str)).issubset(set(EXPECTED_SESSIONS)):
        raise RuntimeError("aggregate STOP: CSU batch-domain diversity audit missing")
    if (diversity["n_unique_source_sessions"].astype(int) < 1).any() or (
        diversity["n_unique_source_sessions"].astype(int) > 8
    ).any():
        raise RuntimeError("aggregate STOP: invalid CSU batch domain count")
    csu = metrics[metrics["condition"] == "MULTI_SOURCE_CSU"]
    if not (csu["csu_alpha"].astype(float) == CSU_ALPHA).all() or not (
        csu["csu_probability"].astype(float) == CSU_PROBABILITY
    ).all():
        raise RuntimeError("aggregate STOP: CSU frozen parameters changed")
    curves = list((args.output_dir / "downstream/training_curves").glob("*.csv"))
    if len(curves) != expected_rows:
        raise RuntimeError(f"aggregate STOP: training curves {len(curves)}/{expected_rows}")

    target_table = target_level_csu_comparison(metrics, _within_reference(args))
    tests = planned_statistical_tests(target_table)
    gaps = within_cross_gap(target_table)
    stability = seed_stability(metrics)
    write_csv(args.output_dir / "summaries/target_level_csu_comparison.csv", target_table)
    write_csv(args.output_dir / "summaries/planned_statistical_tests.csv", tests)
    write_csv(args.output_dir / "summaries/within_cross_gap.csv", gaps)
    write_csv(args.output_dir / "summaries/seed_stability.csv", stability)
    make_required_figures(args.output_dir, target_table, stability)
    scenario = classify_scenario(target_table, tests)
    binary = target_table[target_table["task"] == "binary"]
    stimulus = target_table[target_table["task"] == "stimulus_type"]
    strong = target_table[target_table["target_session"].isin(["708", "709", "710"])]
    weak = target_table[target_table["target_session"].isin(["626", "628", "807", "813", "817", "822"])]
    report = [
        "# SmallCNN + CSU multi-source domain generalization v6", "",
        "## Integrity gates", "",
        f"- Formal jobs: `{len(metrics)}` / `{expected_rows}`; all nine targets, two tasks, two conditions, and three paired seeds completed.",
        "- Every fold used exactly eight source sessions and zero target training, normalization, validation, model-selection, adaptation, or registration data.",
        f"- CSU was author-aligned and frozen at `alpha={CSU_ALPHA}`, `p={CSU_PROBABILITY}`, `eps={CSU_EPSILON}` after SmallCNN block 1.",
        "- ERM was directly reused from compatible v5 artifacts whenever possible; any incompatible/missing job was rerun with the exact frozen v5 trainer.", "",
        "## Planned target-session statistics", "", _markdown_table(tests), "",
        "## Binary target comparison", "", _markdown_table(binary), "",
        "## Stimulus-type target comparison", "", _markdown_table(stimulus), "",
        "## Strong within-session targets (descriptive only)", "", _markdown_table(strong), "",
        "## Weak within-session targets (descriptive only)", "", _markdown_table(weak), "",
        "## Preregistered interpretation", "",
        f"- Scenario: **{scenario}**.",
        "- Formal inference uses the nine target sessions, not folds, seeds, or samples, as independent units.",
        "- No post-hoc CSU tuning, insertion search, extra DG method, or target deletion was performed.",
    ]
    (args.output_dir / "report/csu_domain_generalization_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    missing = missing_formal_outputs(args.output_dir)
    if missing:
        raise RuntimeError(f"formal output completeness STOP: {missing}")
    log(f"Aggregation PASSED; preregistered scenario={scenario}", args.output_dir)


def run_smoke(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("the local v6 smoke test is CPU-only")
    if int(args.smoke_epochs) != 1:
        raise ValueError("v6 smoke is frozen to one tiny epoch")
    run_audit(args)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    task, target, sources, seed = "binary", "626", ("628", "708"), V6_SEEDS[0]
    data = {
        session: load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
        for session in (*sources, target)
    }
    prepared = prepare_cross_session_data(
        data, source_sessions=sources, target_session=target, balance_mode="session_balanced"
    )
    # At most eight blocks, stratified by source, gives a single frozen-size minibatch.
    keep = np.concatenate([
        np.flatnonzero(prepared.train_session_labels == source)[:4] for source in sources
    ])
    tiny = replace(
        prepared,
        X_train=prepared.X_train[keep],
        y_train=prepared.y_train[keep],
        train_session_labels=prepared.train_session_labels[keep],
        train_composite_groups=prepared.train_composite_groups[keep],
        train_sample_ids=prepared.train_sample_ids[keep],
        X_test=prepared.X_test[:4],
        y_test=prepared.y_test[:4],
        test_cycles=prepared.test_cycles[:4],
        test_sample_ids=prepared.test_sample_ids[:4],
    )
    started = time.perf_counter()
    result = train_prepared_csu(
        tiny,
        seed=seed,
        config=replace(FROZEN_SUPERVISED_CONFIG, max_epochs=1),
        device="cpu",
    )
    model = result.model
    probe = torch.from_numpy(tiny.X_test[:2, :, None, :, :])
    model.train()
    activated = False
    for _ in range(20):
        output = model(probe)
        if model.encoder.csu.last_applied:
            activated = True
            break
    if not activated or not bool(torch.isfinite(output).all()):
        raise AssertionError("canonical CSU did not activate finitely in smoke")
    model.eval()
    before = model.encoder.block1(probe.reshape(-1, 1, 128, 501))
    after = model.encoder.csu(before)
    if not torch.equal(before, after) or model.encoder.csu.last_applied:
        raise AssertionError("eval-mode CSU is not an exact bypass")
    checkpoint = args.output_dir / "smoke/csu_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = CSUCNN2DMeanPool(n_classes=2, dropout=0.25, temporal_length=4)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    restored.eval()
    with torch.no_grad():
        if not torch.equal(model(probe), restored(probe)):
            raise AssertionError("CSU checkpoint save/load changed eval output")
    if result.normalization_audit["target_used_for_stats"] or result.metrics["target_frames_used_for_training"] != 0:
        raise AssertionError("smoke target holdout failed")
    elapsed = time.perf_counter() - started
    text = "\n".join([
        "PASS: tiny local CPU smoke only; not a formal scientific result",
        f"target={target} sources={','.join(sources)} task={task} seed={seed}",
        "condition=MULTI_SOURCE_CSU",
        "training_steps=1",
        f"train_blocks={len(tiny.X_train)} target_probe_blocks={len(tiny.X_test)}",
        "csu_forward_backward=PASS",
        "finite_loss_gradients_parameters=PASS",
        "checkpoint_save_load=PASS",
        "train_mode_csu_can_activate=PASS",
        "eval_mode_exact_bypass=PASS",
        "target_labeled_samples_in_training=0",
        "target_frames_in_training=0",
        "target_used_for_normalization=False",
        "output_schema=PASS",
        f"runtime_seconds={elapsed:.3f}",
    ]) + "\n"
    (args.output_dir / "smoke_test_local.txt").write_text(text, encoding="utf-8")
    print(text, end="", flush=True)


def _write_gpu_audit(args: argparse.Namespace, runtime_seconds: float | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal v6 requires CUDA; CPU fallback is forbidden")
    values = [
        f"gpu_name={torch.cuda.get_device_name(0)}",
        f"cuda_runtime={torch.version.cuda}",
        f"pytorch_version={torch.__version__}",
        f"pytorch_cuda_available={torch.cuda.is_available()}",
        f"device_count={torch.cuda.device_count()}",
        f"actual_batch_size={FROZEN_SUPERVISED_CONFIG.batch_size}",
    ]
    if runtime_seconds is not None:
        values.extend([
            f"peak_vram_bytes={torch.cuda.max_memory_allocated()}",
            f"runtime_seconds={runtime_seconds:.3f}",
        ])
    (args.output_dir / "audit/gpu_audit.txt").write_text(
        "\n".join(values) + "\n", encoding="utf-8"
    )


def run_formal(args: argparse.Namespace) -> None:
    assert_formal_cuda(args.device)
    if tuple(map(int, args.seeds)) != V6_SEEDS:
        raise RuntimeError(f"formal v6 requires exactly seeds {V6_SEEDS}")
    ensure_output_tree(args.output_dir)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    _write_gpu_audit(args)
    log("Formal CUDA SmallCNN + CSU v6 started", args.output_dir)
    run_audit(args)
    reuse_audit, reused = audit_v5_baseline_reuse(args, materialize=True)
    nonreuse_reason = {
        (str(row.task), str(row.target), int(row.seed)): str(row.reason)
        for row in reuse_audit.itertuples()
        if not bool(row.reused)
    }
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    visited = 0
    for task in EXPECTED_TASKS:
        data_by_session = {
            session: load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            for session in EXPECTED_SESSIONS
        }
        for target in EXPECTED_SESSIONS:
            prepared = prepare_cross_session_data(
                data_by_session,
                source_sessions=source_sessions_for_target(target),
                target_session=target,
                balance_mode="session_balanced",
            )
            for seed in V6_SEEDS:
                key = (task, target, int(seed))
                if key not in reused:
                    _run_erm_job(args, prepared, seed, nonreuse_reason[key])
                _run_csu_job(args, prepared, seed)
                visited += 2
                if visited % 18 == 0:
                    log(f"formal jobs visited={visited}/108", args.output_dir)
            del prepared
            gc.collect()
        del data_by_session
        gc.collect()
    aggregate_outputs(args)
    runtime = time.perf_counter() - started
    _write_gpu_audit(args, runtime_seconds=runtime)
    log(f"Formal v6 completed in {runtime:.3f} seconds", args.output_dir)


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.v5_output_dir = args.v5_output_dir.resolve()
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
