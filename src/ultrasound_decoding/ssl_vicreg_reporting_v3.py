"""Session-level inference and frozen output figures for VICReg SSL v3."""

from __future__ import annotations

from itertools import product
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ultrasound_decoding.ssl_vicreg_v3 import V3_CONDITIONS, WEAK_SESSIONS


os.environ.setdefault("MPLBACKEND", "Agg")

CONDITION_LABELS = {
    "RANDOM_INIT": "Random-init",
    "WITHIN_MASKED_SSL_FT": "Within masked",
    "MULTI_MASKED_SSL_FT": "Multi masked",
    "WITHIN_VICREG_SSL_FT": "Within VICReg",
    "MULTI_VICREG_SSL_FT": "Multi VICReg",
}
CONDITION_COLORS = {
    "RANDOM_INIT": "#777777",
    "WITHIN_MASKED_SSL_FT": "#4c78a8",
    "MULTI_MASKED_SSL_FT": "#72b7b2",
    "WITHIN_VICREG_SSL_FT": "#f58518",
    "MULTI_VICREG_SSL_FT": "#54a24b",
}


def session_level_comparison(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session", "task", "condition", "seed", "fold",
        "test_balanced_accuracy", "train_balanced_accuracy", "train_test_gap_BA",
    }
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold metrics missing columns: {sorted(missing)}")
    values = fold_metrics.copy()
    values["session"] = values["session"].astype(str)
    if set(values["condition"].unique()) != set(V3_CONDITIONS):
        raise AssertionError("formal metrics do not contain exactly five v3 conditions")
    aggregated = (
        values.groupby(["task", "session", "condition"], sort=True)
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
        indexed = group.set_index("condition")
        if set(indexed.index) != set(V3_CONDITIONS):
            raise AssertionError(f"incomplete conditions for {task} session {session}")
        ba = indexed["mean_test_BA"]
        train = indexed["mean_train_BA"]
        gap = indexed["mean_gap"]
        rows.append({
            "task": task,
            "session": str(session),
            "RANDOM_INIT_BA": float(ba["RANDOM_INIT"]),
            "WITHIN_MASKED_SSL_FT_BA": float(ba["WITHIN_MASKED_SSL_FT"]),
            "MULTI_MASKED_SSL_FT_BA": float(ba["MULTI_MASKED_SSL_FT"]),
            "WITHIN_VICREG_SSL_FT_BA": float(ba["WITHIN_VICREG_SSL_FT"]),
            "MULTI_VICREG_SSL_FT_BA": float(ba["MULTI_VICREG_SSL_FT"]),
            "delta_within_vicreg_vs_within_masked": float(
                ba["WITHIN_VICREG_SSL_FT"] - ba["WITHIN_MASKED_SSL_FT"]
            ),
            "delta_multi_vicreg_vs_multi_masked": float(
                ba["MULTI_VICREG_SSL_FT"] - ba["MULTI_MASKED_SSL_FT"]
            ),
            "delta_multi_vicreg_vs_within_vicreg": float(
                ba["MULTI_VICREG_SSL_FT"] - ba["WITHIN_VICREG_SSL_FT"]
            ),
            "delta_multi_vicreg_vs_random": float(
                ba["MULTI_VICREG_SSL_FT"] - ba["RANDOM_INIT"]
            ),
            "random_train_BA": float(train["RANDOM_INIT"]),
            "within_masked_train_BA": float(train["WITHIN_MASKED_SSL_FT"]),
            "multi_masked_train_BA": float(train["MULTI_MASKED_SSL_FT"]),
            "within_vicreg_train_BA": float(train["WITHIN_VICREG_SSL_FT"]),
            "multi_vicreg_train_BA": float(train["MULTI_VICREG_SSL_FT"]),
            "random_gap": float(gap["RANDOM_INIT"]),
            "within_masked_gap": float(gap["WITHIN_MASKED_SSL_FT"]),
            "multi_masked_gap": float(gap["MULTI_MASKED_SSL_FT"]),
            "within_vicreg_gap": float(gap["WITHIN_VICREG_SSL_FT"]),
            "multi_vicreg_gap": float(gap["MULTI_VICREG_SSL_FT"]),
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
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def planned_statistical_tests(session_table: pd.DataFrame) -> pd.DataFrame:
    contrasts = (
        ("WITHIN_VICREG_SSL_FT_vs_WITHIN_MASKED_SSL_FT", "delta_within_vicreg_vs_within_masked"),
        ("MULTI_VICREG_SSL_FT_vs_MULTI_MASKED_SSL_FT", "delta_multi_vicreg_vs_multi_masked"),
        ("MULTI_VICREG_SSL_FT_vs_WITHIN_VICREG_SSL_FT", "delta_multi_vicreg_vs_within_vicreg"),
    )
    rows: list[dict[str, Any]] = []
    for task in ("binary", "stimulus_type"):
        task_values = session_table[session_table["task"] == task].sort_values("session")
        if task_values["session"].nunique() != 9:
            raise AssertionError("planned tests require nine session-level observations")
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
    output["correction"] = "Holm across six preregistered tests"
    output["reject_fwer_0_05"] = output["corrected_p"] <= 0.05
    return output


def generalization_gap_summary(session_table: pd.DataFrame) -> pd.DataFrame:
    return session_table[[
        "task", "session", "RANDOM_INIT_BA", "random_train_BA", "random_gap",
        "WITHIN_MASKED_SSL_FT_BA", "within_masked_train_BA", "within_masked_gap",
        "MULTI_MASKED_SSL_FT_BA", "multi_masked_train_BA", "multi_masked_gap",
        "WITHIN_VICREG_SSL_FT_BA", "within_vicreg_train_BA", "within_vicreg_gap",
        "MULTI_VICREG_SSL_FT_BA", "multi_vicreg_train_BA", "multi_vicreg_gap",
        "historically_difficult",
    ]].copy()


def seed_stability(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    values = fold_metrics.copy()
    values["session"] = values["session"].astype(str)
    return (
        values.groupby(["task", "session", "condition", "seed"], sort=True)
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


def plot_objective_comparison(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task].sort_values("session", key=lambda x: x.astype(int))
    sessions = subset["session"].astype(str).tolist()
    columns = [f"{condition}_BA" for condition in V3_CONDITIONS]
    x = np.arange(len(sessions))
    width = 0.16
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, (condition, column) in enumerate(zip(V3_CONDITIONS, columns)):
        ax.bar(
            x + (i - 2) * width, subset[column], width,
            label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition],
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, sessions)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Session-level mean test balanced accuracy")
    ax.set_title(f"{task}: masked reconstruction versus VICReg-style SSL")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, path)


def plot_vicreg_delta(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task].sort_values("session", key=lambda x: x.astype(int))
    sessions = subset["session"].astype(str).tolist()
    x = np.arange(len(sessions))
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 4.7))
    ax.bar(
        x - width / 2, subset["delta_within_vicreg_vs_within_masked"], width,
        label="Within: VICReg - masked", color="#f58518",
    )
    ax.bar(
        x + width / 2, subset["delta_multi_vicreg_vs_multi_masked"], width,
        label="Multi: VICReg - masked", color="#54a24b",
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x, sessions)
    ax.set_ylabel("Session-level test BA delta")
    ax.set_title(f"{task}: invariance objective versus reconstruction")
    ax.legend()
    _save(fig, path)


def plot_weak_sessions(session_table: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["historically_difficult"]]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, session in zip(axes.flat, WEAK_SESSIONS):
        values = subset[subset["session"].astype(str) == session]
        for task, marker in (("binary", "o"), ("stimulus_type", "s")):
            row = values[values["task"] == task].iloc[0]
            ax.plot(
                range(5), [row[f"{condition}_BA"] for condition in V3_CONDITIONS],
                marker=marker, label=task,
            )
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set_title(f"Session {session}")
        ax.set_xticks(range(5), ["Random", "W-Mask", "M-Mask", "W-VIC", "M-VIC"], rotation=30)
        ax.set_ylim(0, 1.03)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Historically difficult sessions: SSL objective comparison")
    _save(fig, path)


def plot_gaps(session_table: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_table[session_table["task"] == task].sort_values("session", key=lambda x: x.astype(int))
    sessions = subset["session"].astype(str).tolist()
    gap_columns = (
        "random_gap", "within_masked_gap", "multi_masked_gap", "within_vicreg_gap", "multi_vicreg_gap",
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for condition, column in zip(V3_CONDITIONS, gap_columns):
        ax.plot(sessions, subset[column], marker="o", label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Train BA - test BA")
    ax.set_title(f"{task}: generalization gap")
    ax.legend(ncol=3, fontsize=8)
    _save(fig, path)


def plot_augmentation_contact_sheet(qc_files: list[Path], path: Path) -> None:
    import matplotlib.pyplot as plt
    if len(qc_files) != 9:
        raise AssertionError(f"augmentation QC requires exactly nine sessions, got {len(qc_files)}")
    rows = []
    for file in sorted(qc_files, key=lambda value: int(value.stem.split("_")[1])):
        rows.append((file.stem.split("_")[1], np.load(file)))
    titles = ("Original", "View 1", "View 2", "|View 1 - original|", "|View 2 - original|")
    fig, axes = plt.subplots(9, 5, figsize=(17, 24), squeeze=False)
    for row_i, (session, data) in enumerate(rows):
        original = data["original"]
        images = (
            original, data["view1"], data["view2"],
            np.abs(data["view1"] - original), np.abs(data["view2"] - original),
        )
        lo, hi = np.percentile(original, [1, 99])
        diff_hi = max(float(np.percentile(images[3], 99)), float(np.percentile(images[4], 99)), 1e-6)
        for col_i, image in enumerate(images):
            axes[row_i, col_i].imshow(
                image, cmap="gray", aspect="auto",
                vmin=0 if col_i >= 3 else lo, vmax=diff_hi if col_i >= 3 else hi,
            )
            axes[row_i, col_i].axis("off")
            if row_i == 0:
                axes[row_i, col_i].set_title(titles[col_i], fontsize=10)
        axes[row_i, 0].set_ylabel(f"Session {session}")
    fig.suptitle("VICReg conservative augmentation QC (train-cycle frames only)", y=0.995)
    _save(fig, path)


def make_required_figures(output_dir: Path, session_table: pd.DataFrame) -> None:
    plot_objective_comparison(session_table, "binary", output_dir / "figures/binary_ssl_objective_comparison.png")
    plot_objective_comparison(session_table, "stimulus_type", output_dir / "figures/stimulus_type_ssl_objective_comparison.png")
    plot_vicreg_delta(session_table, "binary", output_dir / "figures/binary_vicreg_delta.png")
    plot_vicreg_delta(session_table, "stimulus_type", output_dir / "figures/stimulus_type_vicreg_delta.png")
    plot_weak_sessions(session_table, output_dir / "figures/weak_sessions_vicreg.png")
    plot_gaps(session_table, "binary", output_dir / "figures/train_test_gap_binary.png")
    plot_gaps(session_table, "stimulus_type", output_dir / "figures/train_test_gap_stimulus_type.png")
    plot_augmentation_contact_sheet(
        list((output_dir / "figures/augmentation_qc").glob("session_*_qc.npz")),
        output_dir / "figures/augmentation_qc/vicreg_augmentation_contact_sheet.png",
    )


def classify_scenario(session_table: pd.DataFrame) -> str:
    binary = session_table[session_table["task"] == "binary"]
    within = binary["delta_within_vicreg_vs_within_masked"].to_numpy()
    multi_masked = binary["delta_multi_vicreg_vs_multi_masked"].to_numpy()
    multi_scope = binary["delta_multi_vicreg_vs_within_vicreg"].to_numpy()
    if np.mean(multi_scope) > 0 and np.sum(multi_scope > 0) >= 6 and np.mean(multi_masked) > 0:
        return "Scenario V-C: multi-session VICReg uses session diversity positively"
    if np.mean(within) > 0 and np.sum(within > 0) >= 6 and np.mean(multi_masked) > 0:
        return "Scenario V-A: invariance SSL outperforms masked reconstruction"
    if np.mean(within) > 0 and np.sum(within > 0) >= 6 and abs(np.mean(multi_scope)) <= 0.01:
        return "Scenario V-B: VICReg helps, but added session diversity does not"
    if np.mean(within) < 0 and np.mean(multi_masked) < 0 and np.sum((within < 0) & (multi_masked < 0)) >= 6:
        return "Scenario V-E: current augmentation-induced invariance shows negative transfer"
    return "Scenario V-D: current conservative VICReg is not stably better than masked reconstruction"
