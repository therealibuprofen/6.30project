"""Session-level summaries, preregistered inference, and figures for v4."""

from __future__ import annotations

from itertools import product
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ultrasound_decoding.ssl_label_efficiency_v4 import (
    LABEL_FRACTIONS,
    LOW_LABEL_FRACTIONS,
    V4_CONDITIONS,
    WEAK_SESSIONS,
)


os.environ.setdefault("MPLBACKEND", "Agg")

DISPLAY = {
    "RANDOM_INIT": "Random-init",
    "WITHIN_MASKED_SSL_FT": "Masked SSL",
}
COLORS = {
    "RANDOM_INIT": "#777777",
    "WITHIN_MASKED_SSL_FT": "#4c78a8",
}


def _require_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "task", "session", "fold", "seed", "condition", "label_fraction",
        "test_balanced_accuracy", "train_balanced_accuracy", "train_test_gap_BA",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fold metrics missing columns: {sorted(missing)}")
    output = frame.copy()
    output["session"] = output["session"].astype(str)
    output["label_fraction"] = output["label_fraction"].astype(float)
    return output


def session_label_efficiency(
    fold_metrics: pd.DataFrame,
    *,
    fractions: Iterable[float] = LABEL_FRACTIONS,
    require_nine_sessions: bool = True,
) -> pd.DataFrame:
    metrics = _require_metrics(fold_metrics)
    expected_fractions = tuple(float(value) for value in fractions)
    if set(metrics["condition"].unique()) != set(V4_CONDITIONS):
        raise AssertionError("fold metrics must contain exactly the two frozen v4 conditions")
    aggregated = (
        metrics.groupby(["task", "session", "label_fraction", "condition"], sort=True)
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
    for (task, session, fraction), group in aggregated.groupby(
        ["task", "session", "label_fraction"], sort=True
    ):
        values = group.set_index("condition")
        if set(values.index) != set(V4_CONDITIONS):
            raise AssertionError(f"incomplete v4 conditions for {task} session {session} fraction {fraction}")
        random = values.loc["RANDOM_INIT"]
        masked = values.loc["WITHIN_MASKED_SSL_FT"]
        rows.append({
            "task": task,
            "session": str(session),
            "label_fraction": float(fraction),
            "Random_init_BA": float(random["mean_test_BA"]),
            "Masked_SSL_BA": float(masked["mean_test_BA"]),
            "delta_SSL_minus_Random": float(masked["mean_test_BA"] - random["mean_test_BA"]),
            "Random_train_BA": float(random["mean_train_BA"]),
            "SSL_train_BA": float(masked["mean_train_BA"]),
            "Random_gap": float(random["mean_gap"]),
            "SSL_gap": float(masked["mean_gap"]),
            "gap_reduction": float(random["mean_gap"] - masked["mean_gap"]),
            "n_fold_seed_values_per_condition": int(random["n_fold_seed_values"]),
        })
    output = pd.DataFrame(rows).sort_values(["task", "session", "label_fraction"]).reset_index(drop=True)
    for (task, session), group in output.groupby(["task", "session"], sort=True):
        observed = tuple(sorted(group["label_fraction"].astype(float).tolist()))
        if observed != tuple(sorted(expected_fractions)):
            raise AssertionError(f"incomplete label fractions for {task} session {session}: {observed}")
    if require_nine_sessions:
        counts = output.groupby("task")["session"].nunique().to_dict()
        if counts != {"binary": 9, "stimulus_type": 9}:
            raise AssertionError(f"formal summaries require nine sessions per task: {counts}")
    return output


def trapezoidal_aulc(x: np.ndarray, y: np.ndarray) -> float:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("AULC requires equal one-dimensional x/y arrays with at least two points")
    order = np.argsort(x_values, kind="stable")
    x_values = x_values[order]
    y_values = y_values[order]
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("AULC inputs must be finite")
    if np.any(np.diff(x_values) <= 0):
        raise ValueError("AULC fractions must be unique and increasing")
    return float(np.sum(np.diff(x_values) * (y_values[:-1] + y_values[1:]) / 2.0))


def label_efficiency_aulc(session_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task, session), group in session_table.groupby(["task", "session"], sort=True):
        group = group.sort_values("label_fraction")
        x = group["label_fraction"].to_numpy(dtype=float)
        random = trapezoidal_aulc(x, group["Random_init_BA"].to_numpy(dtype=float))
        masked = trapezoidal_aulc(x, group["Masked_SSL_BA"].to_numpy(dtype=float))
        rows.append({
            "task": task,
            "session": str(session),
            "AULC_RANDOM": random,
            "AULC_SSL": masked,
            "delta_AULC": masked - random,
            "n_curve_points": int(len(group)),
            "fraction_min": float(x.min()),
            "fraction_max": float(x.max()),
            "integration": "trapezoidal",
        })
    return pd.DataFrame(rows)


def low_label_summary(session_table: pd.DataFrame) -> pd.DataFrame:
    low = session_table[session_table["label_fraction"].isin(LOW_LABEL_FRACTIONS)].copy()
    rows: list[dict[str, Any]] = []
    for (task, session), group in low.groupby(["task", "session"], sort=True):
        observed = set(group["label_fraction"].astype(float))
        if observed != set(LOW_LABEL_FRACTIONS):
            raise AssertionError(f"20/40% low-label rows missing for {task} session {session}")
        random = float(group["Random_init_BA"].mean())
        masked = float(group["Masked_SSL_BA"].mean())
        rows.append({
            "task": task,
            "session": str(session),
            "low_label_fractions": "0.2,0.4",
            "Random_init_low_label_BA": random,
            "Masked_SSL_low_label_BA": masked,
            "low_label_delta": masked - random,
        })
    return pd.DataFrame(rows)


def exact_sign_flip_pvalue(deltas: np.ndarray) -> float:
    values = np.asarray(deltas, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite one-dimensional session deltas required")
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
    if values.ndim != 1 or not len(values) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm correction requires one-dimensional p-values in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def planned_statistical_tests(
    aulc: pd.DataFrame, low_label: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in ("binary", "stimulus_type"):
        for contrast, source, column in (
            ("AULC_SSL_vs_Random", aulc, "delta_AULC"),
            ("Low_label_20_40_SSL_vs_Random", low_label, "low_label_delta"),
        ):
            values = source[source["task"] == task].sort_values("session")
            if values["session"].astype(str).nunique() != 9 or len(values) != 9:
                raise AssertionError("planned inference requires exactly nine session-level observations")
            deltas = values[column].to_numpy(dtype=float)
            rows.append({
                "task": task,
                "contrast": contrast,
                "primary_unit": "session",
                "n_sessions": 9,
                "mean_delta": float(np.mean(deltas)),
                "median_delta": float(np.median(deltas)),
                "positive_sessions": int(np.sum(deltas > 0)),
                "zero_sessions": int(np.sum(deltas == 0)),
                "raw_p": exact_sign_flip_pvalue(deltas),
                "n_exact_sign_patterns": 512,
            })
    output = pd.DataFrame(rows)
    output["holm_corrected_p"] = holm_adjust(output["raw_p"].to_numpy(dtype=float))
    output["correction_family"] = "Holm across four preregistered tests"
    output["reject_fwer_0_05"] = output["holm_corrected_p"] <= 0.05
    return output


def label_fraction_to_target(session_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (task, session), group in session_table.groupby(["task", "session"], sort=True):
        full = group[np.isclose(group["label_fraction"], 1.0)]
        if len(full) != 1:
            raise AssertionError("each session/task must have exactly one 100% summary row")
        target = 0.95 * float(full.iloc[0]["Random_init_BA"])
        for condition, column in (
            ("RANDOM_INIT", "Random_init_BA"),
            ("WITHIN_MASKED_SSL_FT", "Masked_SSL_BA"),
        ):
            reached = group[group[column] >= target].sort_values("label_fraction")
            numeric = float(reached.iloc[0]["label_fraction"]) if len(reached) else np.nan
            rows.append({
                "task": task,
                "session": str(session),
                "condition": condition,
                "target_BA": target,
                "minimum_label_fraction": "NOT_REACHED" if not len(reached) else f"{numeric:.1f}",
                "minimum_label_fraction_numeric": numeric,
                "reached_target": bool(len(reached)),
            })
    return pd.DataFrame(rows)


def seed_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = _require_metrics(fold_metrics)
    return (
        metrics.groupby(["task", "session", "condition", "label_fraction", "seed"], sort=True)
        .agg(
            mean_fold_test_BA=("test_balanced_accuracy", "mean"),
            std_across_folds=("test_balanced_accuracy", "std"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )


def classify_scenario(session_table: pd.DataFrame, aulc: pd.DataFrame) -> str:
    binary = session_table[session_table["task"] == "binary"]
    mean_aulc = float(aulc[aulc["task"] == "binary"]["delta_AULC"].mean())
    low = float(binary[binary["label_fraction"].isin(LOW_LABEL_FRACTIONS)]["delta_SSL_minus_Random"].mean())
    full = float(binary[np.isclose(binary["label_fraction"], 1.0)]["delta_SSL_minus_Random"].mean())
    if low < -0.01:
        return "L-E"
    if full > 0 and low <= 0:
        return "L-C"
    if mean_aulc > 0 and low > full + 0.01:
        return "L-A"
    if mean_aulc > 0:
        return "L-B"
    return "L-D"


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _mean_curve(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for condition, column in (
        ("RANDOM_INIT", "Random_init_BA"),
        ("WITHIN_MASKED_SSL_FT", "Masked_SSL_BA"),
    ):
        summary = subset.groupby("label_fraction")[column].agg(["mean", "sem"]).reset_index()
        x = 100 * summary["label_fraction"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, label=DISPLAY[condition], color=COLORS[condition])
        ax.fill_between(x, y - sem, y + sem, alpha=0.18, color=COLORS[condition])
    ax.set(xlabel="Labeled outer-train cycles (%)", ylabel="Balanced accuracy", xticks=[20, 40, 60, 80, 100])
    ax.set_title("Binary label efficiency" if task == "binary" else "Stimulus-type label efficiency")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, path)


def _by_session(session_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == "binary"]
    sessions = sorted(subset["session"].astype(str).unique())
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True)
    for ax, session in zip(axes.flat, sessions):
        values = subset[subset["session"].astype(str) == session].sort_values("label_fraction")
        x = 100 * values["label_fraction"].to_numpy(dtype=float)
        ax.plot(x, values["Random_init_BA"], marker="o", color=COLORS["RANDOM_INIT"], label=DISPLAY["RANDOM_INIT"])
        ax.plot(x, values["Masked_SSL_BA"], marker="o", color=COLORS["WITHIN_MASKED_SSL_FT"], label=DISPLAY["WITHIN_MASKED_SSL_FT"])
        ax.set_title(f"Session {session}")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Labeled outer-train cycles (%)")
    fig.supylabel("Balanced accuracy")
    _save(fig, path)


def _advantage(session_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == "binary"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fractions = sorted(subset["label_fraction"].unique())
    values = [subset[np.isclose(subset["label_fraction"], fraction)]["delta_SSL_minus_Random"].to_numpy() for fraction in fractions]
    ax.boxplot(values, positions=100 * np.asarray(fractions), widths=10, showfliers=False)
    for fraction, deltas in zip(fractions, values):
        ax.scatter(np.full(len(deltas), 100 * fraction), deltas, color="#4c78a8", alpha=0.65, s=22)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xlabel="Labeled outer-train cycles (%)", ylabel="Masked SSL minus Random-init BA", xticks=[20, 40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _gap_curve(session_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == "binary"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for condition, column in (("RANDOM_INIT", "Random_gap"), ("WITHIN_MASKED_SSL_FT", "SSL_gap")):
        summary = subset.groupby("label_fraction")[column].agg(["mean", "sem"]).reset_index()
        x = 100 * summary["label_fraction"].to_numpy(dtype=float)
        y = summary["mean"].to_numpy(dtype=float)
        sem = summary["sem"].fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", color=COLORS[condition], label=DISPLAY[condition])
        ax.fill_between(x, y - sem, y + sem, color=COLORS[condition], alpha=0.18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Labeled outer-train cycles (%)", ylabel="Train BA minus test BA", xticks=[20, 40, 60, 80, 100])
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, path)


def _weak_sessions(session_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[
        (session_table["task"] == "binary") & session_table["session"].astype(str).isin(WEAK_SESSIONS)
    ]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for condition, column in (("RANDOM_INIT", "Random_init_BA"), ("WITHIN_MASKED_SSL_FT", "Masked_SSL_BA")):
        summary = subset.groupby("label_fraction")[column].mean().reset_index()
        ax.plot(100 * summary["label_fraction"], summary[column], marker="o", linewidth=2, color=COLORS[condition], label=DISPLAY[condition])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Chance")
    ax.set(xlabel="Labeled outer-train cycles (%)", ylabel="Mean balanced accuracy", xticks=[20, 40, 60, 80, 100])
    ax.set_title("Historically difficult sessions (descriptive)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    _save(fig, path)


def make_required_figures(output_dir: Path, session_table: pd.DataFrame) -> None:
    figures = output_dir / "figures"
    _mean_curve(session_table, "binary", figures / "binary_label_efficiency_mean_curve.png")
    _by_session(session_table, figures / "binary_label_efficiency_by_session.png")
    _advantage(session_table, figures / "binary_ssl_advantage_by_fraction.png")
    _gap_curve(session_table, figures / "binary_train_test_gap_by_fraction.png")
    _weak_sessions(session_table, figures / "weak_sessions_low_label.png")
    _mean_curve(session_table, "stimulus_type", figures / "stimulus_type_label_efficiency_mean_curve.png")
