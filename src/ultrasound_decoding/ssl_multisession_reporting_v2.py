"""Session-level inference and fixed figures for multi-session masked SSL v2."""

from __future__ import annotations

from itertools import product
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ultrasound_decoding.ssl_multisession_v2 import V2_CONDITIONS, WEAK_SESSIONS


os.environ.setdefault("MPLBACKEND", "Agg")


CONDITION_LABELS = {
    "RANDOM_INIT": "Random-init",
    "WITHIN_SSL_FT": "Within SSL",
    "OTHER_ONLY_SSL_FT": "Other-only SSL",
    "MULTI_SSL_FT": "Multi SSL",
}
CONDITION_COLORS = {
    "RANDOM_INIT": "#777777",
    "WITHIN_SSL_FT": "#4c78a8",
    "OTHER_ONLY_SSL_FT": "#f58518",
    "MULTI_SSL_FT": "#54a24b",
}


def session_level_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session", "task", "condition", "seed", "fold",
        "test_balanced_accuracy", "train_balanced_accuracy", "train_test_gap_BA",
    }
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold metrics missing columns: {sorted(missing)}")
    fold_metrics = fold_metrics.copy()
    fold_metrics["session"] = fold_metrics["session"].astype(str)
    if set(fold_metrics["condition"].unique()) != set(V2_CONDITIONS):
        raise AssertionError("formal fold metrics do not contain exactly four v2 conditions")
    aggregated = (
        fold_metrics.groupby(["task", "session", "condition"], sort=True)
        .agg(
            mean_test_BA=("test_balanced_accuracy", "mean"),
            mean_train_BA=("train_balanced_accuracy", "mean"),
            mean_gap=("train_test_gap_BA", "mean"),
            std_test_BA=("test_balanced_accuracy", "std"),
            n_fold_seed_values=("test_balanced_accuracy", "size"),
        )
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for (task, session), group in aggregated.groupby(["task", "session"], sort=True):
        values = group.set_index("condition")
        if set(values.index) != set(V2_CONDITIONS):
            raise AssertionError(f"incomplete conditions for {task} session {session}")
        ba = values["mean_test_BA"]
        train = values["mean_train_BA"]
        gap = values["mean_gap"]
        rows.append({
            "task": task,
            "session": str(session),
            "RANDOM_INIT_BA": float(ba["RANDOM_INIT"]),
            "WITHIN_SSL_FT_BA": float(ba["WITHIN_SSL_FT"]),
            "OTHER_ONLY_SSL_FT_BA": float(ba["OTHER_ONLY_SSL_FT"]),
            "MULTI_SSL_FT_BA": float(ba["MULTI_SSL_FT"]),
            "delta_within_vs_random": float(ba["WITHIN_SSL_FT"] - ba["RANDOM_INIT"]),
            "delta_other_vs_within": float(ba["OTHER_ONLY_SSL_FT"] - ba["WITHIN_SSL_FT"]),
            "delta_multi_vs_within": float(ba["MULTI_SSL_FT"] - ba["WITHIN_SSL_FT"]),
            "delta_multi_vs_random": float(ba["MULTI_SSL_FT"] - ba["RANDOM_INIT"]),
            "random_train_BA": float(train["RANDOM_INIT"]),
            "within_train_BA": float(train["WITHIN_SSL_FT"]),
            "other_train_BA": float(train["OTHER_ONLY_SSL_FT"]),
            "multi_train_BA": float(train["MULTI_SSL_FT"]),
            "random_gap": float(gap["RANDOM_INIT"]),
            "within_gap": float(gap["WITHIN_SSL_FT"]),
            "other_gap": float(gap["OTHER_ONLY_SSL_FT"]),
            "multi_gap": float(gap["MULTI_SSL_FT"]),
            "historically_difficult": str(session) in WEAK_SESSIONS,
        })
    output = pd.DataFrame(rows)
    counts = output.groupby("task")["session"].nunique().to_dict()
    if counts != {"binary": 9, "stimulus_type": 9}:
        raise AssertionError(f"session-level inference requires nine sessions per task: {counts}")
    return output


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional deltas required")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return float(exceed / total)


def holm_adjust(raw_p: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_p, dtype=float)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(values) - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def planned_statistical_tests(session_table: pd.DataFrame) -> pd.DataFrame:
    contrasts = (
        ("MULTI_SSL_FT_vs_WITHIN_SSL_FT", "delta_multi_vs_within"),
        ("OTHER_ONLY_SSL_FT_vs_WITHIN_SSL_FT", "delta_other_vs_within"),
    )
    rows: list[dict[str, Any]] = []
    for task in ("binary", "stimulus_type"):
        task_values = session_table[session_table["task"] == task].sort_values("session")
        if task_values["session"].nunique() != 9:
            raise AssertionError("planned tests use exactly nine session-level observations")
        for comparison, column in contrasts:
            deltas = task_values[column].to_numpy(dtype=float)
            rows.append({
                "task": task,
                "comparison": comparison,
                "primary_unit": "session",
                "n_sessions": int(len(deltas)),
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "positive_session_count": int(np.sum(deltas > 0)),
                "zero_session_count": int(np.sum(deltas == 0)),
                "raw_p": exact_sign_flip_pvalue(deltas),
                "n_exact_sign_patterns": int(2 ** len(deltas)),
            })
    output = pd.DataFrame(rows)
    output["corrected_p"] = holm_adjust(output["raw_p"].to_numpy())
    output["correction"] = "Holm across four preregistered tests"
    output["reject_fwer_0_05"] = output["corrected_p"] <= 0.05
    return output


def generalization_gap_summary(session_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "task", "session", "random_train_BA", "RANDOM_INIT_BA", "random_gap",
        "within_train_BA", "WITHIN_SSL_FT_BA", "within_gap",
        "other_train_BA", "OTHER_ONLY_SSL_FT_BA", "other_gap",
        "multi_train_BA", "MULTI_SSL_FT_BA", "multi_gap", "historically_difficult",
    ]
    return session_table[columns].copy()


def seed_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    fold_metrics = fold_metrics.copy()
    fold_metrics["session"] = fold_metrics["session"].astype(str)
    return (
        fold_metrics.groupby(["task", "session", "condition", "seed"], sort=True)
        .agg(
            mean_fold_test_BA=("test_balanced_accuracy", "mean"),
            std_across_folds=("test_balanced_accuracy", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_session_ba(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task].sort_values("session", key=lambda x: x.astype(int))
    sessions = subset["session"].astype(str).tolist()
    columns = [f"{condition}_BA" for condition in V2_CONDITIONS]
    x = np.arange(len(sessions))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (condition, column) in enumerate(zip(V2_CONDITIONS, columns)):
        ax.bar(x + (i - 1.5) * width, subset[column], width, label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, sessions)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Session-level mean test balanced accuracy")
    ax.set_title(f"{task}: multi-session masked SSL")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, path)


def plot_multi_delta(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task].sort_values("session", key=lambda x: x.astype(int))
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in subset["delta_multi_vs_within"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(subset["session"].astype(str), subset["delta_multi_vs_within"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("MULTI_SSL_FT - WITHIN_SSL_FT test BA")
    ax.set_title(f"{task}: added cross-session unlabeled data")
    _save(fig, path)


def plot_weak_sessions(session_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["historically_difficult"]].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, session in zip(axes.flat, WEAK_SESSIONS):
        values = subset[subset["session"].astype(str) == session]
        for task, marker in (("binary", "o"), ("stimulus_type", "s")):
            row = values[values["task"] == task].iloc[0]
            test = [row[f"{condition}_BA"] for condition in V2_CONDITIONS]
            ax.plot(range(4), test, marker=marker, label=task)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(f"Session {session}")
        ax.set_xticks(range(4), ["Random", "Within", "Other", "Multi"], rotation=25)
        ax.set_ylim(0, 1.03)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Historically difficult sessions: test BA")
    _save(fig, path)


def plot_gaps(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task].sort_values("session", key=lambda x: x.astype(int))
    sessions = subset["session"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for condition, column in zip(V2_CONDITIONS, ("random_gap", "within_gap", "other_gap", "multi_gap")):
        ax.plot(sessions, subset[column], marker="o", label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Train BA - test BA")
    ax.set_title(f"{task}: generalization gap")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_sampling_distribution(distribution: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    distribution = distribution.copy()
    distribution["source_session"] = distribution["source_session"].astype(str)
    values = (
        distribution.groupby(["condition", "source_session"], sort=True)["actual_proportion"]
        .mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sessions = sorted(values["source_session"].astype(str).unique(), key=int)
    x = np.arange(len(sessions))
    for offset, condition in enumerate(("OTHER_ONLY_SSL_FT", "MULTI_SSL_FT")):
        subset = values[values["condition"] == condition].set_index("source_session").reindex(sessions)
        ax.bar(x + (offset - 0.5) * 0.36, subset["actual_proportion"], 0.36, label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition])
    ax.set_xticks(x, sessions)
    ax.set_ylabel("Mean sampled proportion")
    ax.set_title("Session-balanced masked-SSL sampling audit")
    ax.legend()
    _save(fig, path)


def make_required_figures(output_dir: Path, fold_metrics: pd.DataFrame, session_table: pd.DataFrame, distribution: pd.DataFrame) -> None:
    plot_session_ba(session_table, "binary", output_dir / "figures/binary_multisession_ssl_BA.png")
    plot_session_ba(session_table, "stimulus_type", output_dir / "figures/stimulus_type_multisession_ssl_BA.png")
    plot_multi_delta(session_table, "binary", output_dir / "figures/binary_delta_multi_vs_within.png")
    plot_multi_delta(session_table, "stimulus_type", output_dir / "figures/stimulus_type_delta_multi_vs_within.png")
    plot_weak_sessions(session_table, output_dir / "figures/weak_sessions_multisession_ssl.png")
    plot_gaps(session_table, "binary", output_dir / "figures/train_test_gap_binary.png")
    plot_gaps(session_table, "stimulus_type", output_dir / "figures/train_test_gap_stimulus_type.png")
    plot_sampling_distribution(distribution, output_dir / "figures/session_sampling_distribution.png")


def classify_scenario(session_table: pd.DataFrame) -> str:
    binary = session_table[session_table["task"] == "binary"]
    multi_within = binary["delta_multi_vs_within"].to_numpy()
    other_random = (binary["OTHER_ONLY_SSL_FT_BA"] - binary["RANDOM_INIT_BA"]).to_numpy()
    multi_other = (binary["MULTI_SSL_FT_BA"] - binary["OTHER_ONLY_SSL_FT_BA"]).to_numpy()
    other_within = binary["delta_other_vs_within"].to_numpy()
    if np.mean(multi_within) > 0 and np.sum(multi_within > 0) >= 6:
        return "Scenario A: MULTI > WITHIN across multiple sessions"
    if np.mean(other_random) > 0 and abs(np.mean(other_within)) <= 0.01:
        return "Scenario B: OTHER_ONLY transfers and approaches WITHIN"
    if np.mean(multi_other) > 0 and abs(np.mean(other_random)) <= 0.01:
        return "Scenario C: target-session unlabeled adaptation is still required"
    if np.mean(other_within) < 0 and np.mean(multi_within) < 0 and np.sum((other_within < 0) & (multi_within < 0)) >= 6:
        return "Scenario E: consistent negative transfer relative to WITHIN"
    return "Scenario D: expanding simple masked reconstruction gives no clear gain over WITHIN"
