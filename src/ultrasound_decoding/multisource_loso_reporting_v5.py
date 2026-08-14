"""Target-level inference and frozen figures for multi-source LOSO v5."""

from __future__ import annotations

from itertools import product
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ultrasound_decoding.multisource_loso_v5 import V5_CONDITIONS


os.environ.setdefault("MPLBACKEND", "Agg")

CONDITION_LABELS = {
    "SINGLE_SOURCE_TRANSFER": "Single-source mean",
    "MULTI_SOURCE_BALANCED": "Multi-source balanced",
    "NATURAL_FREQUENCY_MULTI_SOURCE": "Multi-source natural",
    "WITHIN_SESSION_REFERENCE": "Within-session reference",
}
COLORS = {
    "SINGLE_SOURCE_TRANSFER": "#777777",
    "MULTI_SOURCE_BALANCED": "#4c78a8",
    "NATURAL_FREQUENCY_MULTI_SOURCE": "#f58518",
    "WITHIN_SESSION_REFERENCE": "#54a24b",
}


def target_level_comparison(
    fold_metrics: pd.DataFrame, within_reference: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "task", "target_session", "source_sessions", "seed", "condition",
        "test_balanced_accuracy", "train_balanced_accuracy", "train_test_gap_BA",
    }
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"formal metrics missing columns: {sorted(missing)}")
    metrics = fold_metrics.copy()
    metrics["target_session"] = metrics["target_session"].astype(str)
    if set(metrics["condition"].unique()) != set(V5_CONDITIONS):
        raise AssertionError("formal metrics do not contain exactly the three v5 conditions")
    reference = within_reference.copy()
    reference["target_session"] = reference["target_session"].astype(str)
    rows: list[dict[str, Any]] = []
    for (task, target), group in metrics.groupby(["task", "target_session"], sort=True):
        single = group[group["condition"] == "SINGLE_SOURCE_TRANSFER"].copy()
        pair_means = (
            single.groupby("source_sessions", sort=True)["test_balanced_accuracy"].mean()
        )
        if len(pair_means) != 8:
            raise AssertionError(f"{task} target {target} does not have eight single-source baselines")
        balanced = group[group["condition"] == "MULTI_SOURCE_BALANCED"]
        natural = group[group["condition"] == "NATURAL_FREQUENCY_MULTI_SOURCE"]
        if len(balanced) != 3 or len(natural) != 3:
            raise AssertionError(f"{task} target {target} does not have three seeds per multi-source condition")
        ref = reference[
            (reference["task"] == task)
            & (reference["target_session"].astype(str) == str(target))
        ]
        if len(ref) != 1:
            raise AssertionError(f"within-session reference missing for {task} target {target}")
        single_mean = float(pair_means.mean())
        multi_ba = float(balanced["test_balanced_accuracy"].mean())
        natural_ba = float(natural["test_balanced_accuracy"].mean())
        within_ba = float(ref.iloc[0]["within_session_reference_BA"])
        rows.append({
            "task": task,
            "target_session": str(target),
            "single_source_mean_BA": single_mean,
            "single_source_median_BA": float(pair_means.median()),
            "single_source_best_BA": float(pair_means.max()),
            "single_source_worst_BA": float(pair_means.min()),
            "n_single_source_sessions": int(len(pair_means)),
            "multi_source_balanced_BA": multi_ba,
            "multi_source_natural_BA": natural_ba,
            "within_session_reference_BA": within_ba,
            "delta_multi_vs_single_mean": multi_ba - single_mean,
            "delta_natural_vs_single_mean": natural_ba - single_mean,
            "delta_balanced_vs_natural": multi_ba - natural_ba,
            "within_minus_single_gap": within_ba - single_mean,
            "within_minus_multi_gap": within_ba - multi_ba,
            "single_source_mean_train_BA": float(single["train_balanced_accuracy"].mean()),
            "multi_source_balanced_train_BA": float(balanced["train_balanced_accuracy"].mean()),
            "multi_source_natural_train_BA": float(natural["train_balanced_accuracy"].mean()),
            "single_source_mean_train_test_gap": float(single["train_test_gap_BA"].mean()),
            "multi_source_balanced_train_test_gap": float(balanced["train_test_gap_BA"].mean()),
            "multi_source_natural_train_test_gap": float(natural["train_test_gap_BA"].mean()),
        })
    output = pd.DataFrame(rows).sort_values(["task", "target_session"]).reset_index(drop=True)
    counts = output.groupby("task")["target_session"].nunique().to_dict()
    if counts != {"binary": 9, "stimulus_type": 9}:
        raise AssertionError(f"formal target summary requires nine targets per task: {counts}")
    return output


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or len(values) != 9 or not np.isfinite(values).all():
        raise ValueError("exact v5 inference requires nine finite target-session deltas")
    observed = abs(float(values.mean()))
    exceed = 0
    for signs in product((-1.0, 1.0), repeat=9):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
    return float(exceed / 512)


def holm_adjust(raw_p: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_p, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def planned_statistical_tests(target_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ("binary", "stimulus_type"):
        values = target_table[target_table["task"] == task].sort_values("target_session")
        if len(values) != 9 or values["target_session"].nunique() != 9:
            raise AssertionError("planned statistics use exactly nine target sessions")
        deltas = values["delta_multi_vs_single_mean"].to_numpy(dtype=float)
        rows.append({
            "task": task,
            "contrast": "MULTI_SOURCE_BALANCED_vs_mean_SINGLE_SOURCE_TRANSFER",
            "primary_unit": "target_session",
            "n_target_sessions": 9,
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "positive_target_count": int(np.sum(deltas > 0)),
            "zero_target_count": int(np.sum(deltas == 0)),
            "raw_p": exact_sign_flip_pvalue(deltas),
            "n_exact_sign_patterns": 512,
        })
    output = pd.DataFrame(rows)
    output["holm_corrected_p"] = holm_adjust(output["raw_p"].to_numpy(dtype=float))
    output["correction_family"] = "Holm across binary and stimulus_type"
    output["reject_fwer_0_05"] = output["holm_corrected_p"] <= 0.05
    return output


def within_cross_gap(target_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task", "target_session", "within_session_reference_BA",
        "single_source_mean_BA", "multi_source_balanced_BA", "multi_source_natural_BA",
        "within_minus_single_gap", "within_minus_multi_gap", "delta_multi_vs_single_mean",
    ]
    return target_table[columns].copy()


def seed_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = fold_metrics.copy()
    metrics["target_session"] = metrics["target_session"].astype(str)
    single = (
        metrics[metrics["condition"] == "SINGLE_SOURCE_TRANSFER"]
        .groupby(["task", "target_session", "seed"], sort=True)
        .agg(
            mean_across_single_sources_BA=("test_balanced_accuracy", "mean"),
            std_across_single_sources_BA=("test_balanced_accuracy", "std"),
            n_sources=("source_sessions", "nunique"),
        )
        .reset_index()
    )
    single["condition"] = "SINGLE_SOURCE_TRANSFER"
    single = single.rename(columns={"mean_across_single_sources_BA": "seed_target_BA"})
    multi = (
        metrics[metrics["condition"].isin(V5_CONDITIONS[1:])]
        .groupby(["task", "target_session", "seed", "condition"], sort=True)
        .agg(seed_target_BA=("test_balanced_accuracy", "mean"))
        .reset_index()
    )
    multi["std_across_single_sources_BA"] = np.nan
    multi["n_sources"] = 8
    columns = [
        "task", "target_session", "condition", "seed", "seed_target_BA",
        "std_across_single_sources_BA", "n_sources",
    ]
    return pd.concat([single[columns], multi[columns]], ignore_index=True).sort_values(
        ["task", "target_session", "condition", "seed"]
    ).reset_index(drop=True)


def classify_scenario(target_table: pd.DataFrame, tests: pd.DataFrame) -> str:
    binary = target_table[target_table["task"] == "binary"]
    mean_delta = float(binary["delta_multi_vs_single_mean"].mean())
    positive = int(np.sum(binary["delta_multi_vs_single_mean"] > 0))
    mean_multi = float(binary["multi_source_balanced_BA"].mean())
    binary_test = tests[tests["task"] == "binary"].iloc[0]
    if mean_delta < -0.01:
        return "M-D"
    if mean_delta > 0 and positive >= 5 and bool(binary_test["reject_fwer_0_05"]) and mean_multi > 0.55:
        return "M-A"
    if mean_delta > 0 and positive >= 5 and mean_multi <= 0.55:
        return "M-B"
    return "M-C"


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _target_bars(target_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    values = target_table[target_table["task"] == task].sort_values("target_session")
    x = np.arange(len(values))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width, values["single_source_mean_BA"], width, color=COLORS["SINGLE_SOURCE_TRANSFER"], label="Single-source mean")
    ax.bar(x, values["multi_source_balanced_BA"], width, color=COLORS["MULTI_SOURCE_BALANCED"], label="Multi-source balanced")
    ax.bar(x + width, values["within_session_reference_BA"], width, color=COLORS["WITHIN_SESSION_REFERENCE"], label="Within-session reference")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance")
    ax.set(xticks=x, xticklabels=values["target_session"], xlabel="Held-out target session", ylabel="Balanced Accuracy")
    ax.set_title("Binary strict unseen-session decoding" if task == "binary" else "Stimulus-type strict unseen-session decoding")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _delta_plot(target_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    values = target_table[target_table["task"] == task].sort_values("target_session")
    colors = ["#4c78a8" if value >= 0 else "#e45756" for value in values["delta_multi_vs_single_mean"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(values["target_session"], values["delta_multi_vs_single_mean"], color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set(xlabel="Held-out target session", ylabel="Multi-source balanced minus single-source mean BA")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _within_cross_gap(target_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    summary = target_table.groupby("task", sort=True)[
        ["single_source_mean_BA", "multi_source_balanced_BA", "within_session_reference_BA"]
    ].mean()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary))
    width = 0.25
    ax.bar(x - width, summary["single_source_mean_BA"], width, color=COLORS["SINGLE_SOURCE_TRANSFER"], label="Single-source mean")
    ax.bar(x, summary["multi_source_balanced_BA"], width, color=COLORS["MULTI_SOURCE_BALANCED"], label="Multi-source balanced")
    ax.bar(x + width, summary["within_session_reference_BA"], width, color=COLORS["WITHIN_SESSION_REFERENCE"], label="Within-session reference")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax.set(xticks=x, xticklabels=summary.index, ylabel="Mean target-session Balanced Accuracy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _sampling_plot(distribution: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    values = (
        distribution.groupby(["condition", "source_session"], sort=True)["draw_proportion"]
        .mean().reset_index()
    )
    sessions = sorted(values["source_session"].astype(str).unique(), key=int)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(sessions))
    width = 0.36
    for offset, condition in zip((-width / 2, width / 2), V5_CONDITIONS[1:]):
        group = values[values["condition"] == condition].set_index("source_session").reindex(sessions)
        ax.bar(x + offset, group["draw_proportion"], width, color=COLORS[condition], label=CONDITION_LABELS[condition])
    ax.set(xticks=x, xticklabels=sessions, xlabel="Source session", ylabel="Mean draw proportion")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def make_required_figures(
    output_dir: Path, target_table: pd.DataFrame, distribution: pd.DataFrame
) -> None:
    figures = output_dir / "figures"
    _target_bars(target_table, "binary", figures / "binary_target_level_cross_session.png")
    _target_bars(target_table, "stimulus_type", figures / "stimulus_type_target_level_cross_session.png")
    _delta_plot(target_table, "binary", figures / "binary_multi_minus_single_delta.png")
    _delta_plot(target_table, "stimulus_type", figures / "stimulus_type_multi_minus_single_delta.png")
    _within_cross_gap(target_table, figures / "within_vs_cross_session_gap.png")
    _sampling_plot(distribution, figures / "source_sampling_distribution.png")
