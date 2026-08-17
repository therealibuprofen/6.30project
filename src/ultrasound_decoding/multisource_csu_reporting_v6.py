"""Target-session inference and frozen figures for SmallCNN + CSU v6."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ultrasound_decoding.multisource_csu_v6 import V6_CONDITIONS, V6_SEEDS
from ultrasound_decoding.multisource_loso_reporting_v5 import (
    exact_sign_flip_pvalue,
    holm_adjust,
)


os.environ.setdefault("MPLBACKEND", "Agg")

COLORS = {
    "MULTI_SOURCE_ERM": "#777777",
    "MULTI_SOURCE_CSU": "#4c78a8",
    "WITHIN_SESSION_REFERENCE": "#54a24b",
    "NEGATIVE": "#e45756",
}


def target_level_csu_comparison(
    fold_metrics: pd.DataFrame, within_reference: pd.DataFrame
) -> pd.DataFrame:
    required = {
        "task", "target_session", "seed", "condition", "test_balanced_accuracy",
        "train_balanced_accuracy", "train_test_gap_BA", "n_source_sessions",
    }
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"v6 metrics missing columns: {sorted(missing)}")
    metrics = fold_metrics.copy()
    metrics["target_session"] = metrics["target_session"].astype(str)
    if set(metrics["condition"].unique()) != set(V6_CONDITIONS):
        raise AssertionError("formal v6 metrics must contain exactly ERM and CSU")
    reference = within_reference.copy()
    reference["target_session"] = reference["target_session"].astype(str)
    rows: list[dict[str, Any]] = []
    for (task, target), group in metrics.groupby(["task", "target_session"], sort=True):
        erm = group[group["condition"] == "MULTI_SOURCE_ERM"].sort_values("seed")
        csu = group[group["condition"] == "MULTI_SOURCE_CSU"].sort_values("seed")
        if len(erm) != 3 or len(csu) != 3:
            raise AssertionError(f"{task} target {target} requires three seeds for ERM and CSU")
        if tuple(erm["seed"].astype(int)) != V6_SEEDS or tuple(csu["seed"].astype(int)) != V6_SEEDS:
            raise AssertionError("ERM/CSU seed pairing differs from the frozen seed set")
        if not (group["n_source_sessions"].astype(int) == 8).all():
            raise AssertionError("v6 summary received a non-eight-source fold")
        ref = reference[
            (reference["task"] == task)
            & (reference["target_session"].astype(str) == str(target))
        ]
        if len(ref) != 1:
            raise AssertionError(f"within-session reference missing for {task} target {target}")
        erm_ba = float(erm["test_balanced_accuracy"].mean())
        csu_ba = float(csu["test_balanced_accuracy"].mean())
        within_ba = float(ref.iloc[0]["within_session_reference_BA"])
        rows.append({
            "task": task,
            "target_session": str(target),
            "MULTI_SOURCE_ERM_BA": erm_ba,
            "MULTI_SOURCE_CSU_BA": csu_ba,
            "delta_CSU_minus_ERM": csu_ba - erm_ba,
            "ERM_seed_std": float(erm["test_balanced_accuracy"].std(ddof=1)),
            "CSU_seed_std": float(csu["test_balanced_accuracy"].std(ddof=1)),
            "within_session_reference_BA": within_ba,
            "ERM_train_BA": float(erm["train_balanced_accuracy"].mean()),
            "CSU_train_BA": float(csu["train_balanced_accuracy"].mean()),
            "ERM_train_target_gap_BA": float(erm["train_test_gap_BA"].mean()),
            "CSU_train_target_gap_BA": float(csu["train_test_gap_BA"].mean()),
            "within_minus_ERM_gap": within_ba - erm_ba,
            "within_minus_CSU_gap": within_ba - csu_ba,
            "gap_reduction_CSU_vs_ERM": csu_ba - erm_ba,
        })
    output = pd.DataFrame(rows).sort_values(["task", "target_session"]).reset_index(drop=True)
    counts = output.groupby("task")["target_session"].nunique().to_dict()
    if counts != {"binary": 9, "stimulus_type": 9}:
        raise AssertionError(f"v6 requires all nine targets for both tasks: {counts}")
    return output


def planned_statistical_tests(target_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in ("binary", "stimulus_type"):
        values = target_table[target_table["task"] == task].sort_values("target_session")
        if len(values) != 9 or values["target_session"].nunique() != 9:
            raise AssertionError("v6 planned inference uses exactly nine target sessions")
        deltas = values["delta_CSU_minus_ERM"].to_numpy(dtype=float)
        rows.append({
            "task": task,
            "contrast": "MULTI_SOURCE_CSU_minus_MULTI_SOURCE_ERM",
            "primary_unit": "target_session",
            "n_target_sessions": 9,
            "mean_delta": float(np.mean(deltas)),
            "median_delta": float(np.median(deltas)),
            "positive_target_count": int(np.sum(deltas > 0)),
            "zero_target_count": int(np.sum(deltas == 0)),
            "negative_target_count": int(np.sum(deltas < 0)),
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
        "MULTI_SOURCE_ERM_BA", "MULTI_SOURCE_CSU_BA", "within_minus_ERM_gap",
        "within_minus_CSU_gap", "gap_reduction_CSU_vs_ERM",
    ]
    return target_table[columns].copy()


def seed_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = fold_metrics.copy()
    metrics["target_session"] = metrics["target_session"].astype(str)
    rows = (
        metrics.groupby(["task", "target_session", "condition"], sort=True)
        .agg(
            mean_target_BA=("test_balanced_accuracy", "mean"),
            seed_std=("test_balanced_accuracy", "std"),
            seed_min=("test_balanced_accuracy", "min"),
            seed_max=("test_balanced_accuracy", "max"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    if not (rows["n_seeds"].astype(int) == 3).all():
        raise AssertionError("seed-stability table requires exactly three seeds")
    return rows


def classify_scenario(target_table: pd.DataFrame, tests: pd.DataFrame) -> str:
    binary = target_table[target_table["task"] == "binary"]
    mean_delta = float(binary["delta_CSU_minus_ERM"].mean())
    positive = int(np.sum(binary["delta_CSU_minus_ERM"] > 0))
    mean_csu = float(binary["MULTI_SOURCE_CSU_BA"].mean())
    binary_test = tests[tests["task"] == "binary"].iloc[0]
    if mean_delta < -0.01:
        return "C-D"
    if mean_delta > 0 and positive >= 5 and bool(binary_test["reject_fwer_0_05"]) and mean_csu > 0.55:
        return "C-A"
    if mean_delta > 0 and positive >= 5 and mean_csu <= 0.55:
        return "C-B"
    return "C-C"


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
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    ax.bar(x - width, values["MULTI_SOURCE_ERM_BA"], width, color=COLORS["MULTI_SOURCE_ERM"], label="Multi-source ERM")
    ax.bar(x, values["MULTI_SOURCE_CSU_BA"], width, color=COLORS["MULTI_SOURCE_CSU"], label="Multi-source CSU")
    ax.bar(x + width, values["within_session_reference_BA"], width, color=COLORS["WITHIN_SESSION_REFERENCE"], label="Within-session reference")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance")
    ax.set(xticks=x, xticklabels=values["target_session"], xlabel="Held-out target session", ylabel="Balanced Accuracy")
    ax.set_title("Binary: CSU vs ERM" if task == "binary" else "Stimulus type: CSU vs ERM")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _delta_plot(target_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    values = target_table[target_table["task"] == task].sort_values("target_session")
    colors = [
        COLORS["MULTI_SOURCE_CSU"] if value >= 0 else COLORS["NEGATIVE"]
        for value in values["delta_CSU_minus_ERM"]
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(values["target_session"], values["delta_CSU_minus_ERM"], color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set(xlabel="Held-out target session", ylabel="CSU minus ERM BA")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _gap_plot(target_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    values = target_table.sort_values(["task", "target_session"])
    labels = [f"{row.task}:{row.target_session}" for row in values.itertuples()]
    x = np.arange(len(values))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(x - width / 2, values["within_minus_ERM_gap"], width, color=COLORS["MULTI_SOURCE_ERM"], label="Within − ERM")
    ax.bar(x + width / 2, values["within_minus_CSU_gap"], width, color=COLORS["MULTI_SOURCE_CSU"], label="Within − CSU")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xticks=x, xticklabels=labels, ylabel="Within-to-cross BA gap")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _seed_plot(stability: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    summary = (
        stability.groupby(["task", "condition"], sort=True)
        .agg(mean_seed_std=("seed_std", "mean"), max_seed_std=("seed_std", "max"))
        .reset_index()
    )
    labels = [f"{row.task}\n{row.condition.replace('MULTI_SOURCE_', '')}" for row in summary.itertuples()]
    colors = [COLORS[condition] for condition in summary["condition"]]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.bar(labels, summary["mean_seed_std"], color=colors)
    ax.scatter(labels, summary["max_seed_std"], color="black", marker="_", s=150, label="Maximum target SD")
    ax.set(ylabel="Across-seed BA standard deviation", title="Seed stability across held-out targets")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def make_required_figures(
    output_dir: Path, target_table: pd.DataFrame, stability: pd.DataFrame
) -> None:
    figures = output_dir / "figures"
    _target_bars(target_table, "binary", figures / "binary_csu_vs_erm_by_target.png")
    _target_bars(target_table, "stimulus_type", figures / "stimulus_type_csu_vs_erm_by_target.png")
    _delta_plot(target_table, "binary", figures / "binary_csu_delta.png")
    _delta_plot(target_table, "stimulus_type", figures / "stimulus_type_csu_delta.png")
    _gap_plot(target_table, figures / "within_cross_gap_csu.png")
    _seed_plot(stability, figures / "csu_seed_stability.png")
