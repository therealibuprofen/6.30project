"""Aggregation, exact session-level inference, and figures for SSL v1."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ultrasound_decoding.ssl_masked import SSL_CONDITIONS, STRONG_SESSIONS, WEAK_SESSIONS


CONDITION_LABELS = {
    "RANDOM_INIT": "Random",
    "SSL_FROZEN": "SSL Frozen",
    "SSL_FINETUNE": "SSL Finetune",
}
CONDITION_COLORS = {
    "RANDOM_INIT": "#6b7280",
    "SSL_FROZEN": "#2563eb",
    "SSL_FINETUNE": "#dc2626",
}


def session_level_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session", "task", "condition", "seed", "fold", "test_balanced_accuracy",
        "train_balanced_accuracy", "train_test_gap_BA",
    }
    missing = required - set(fold_metrics.columns)
    if missing:
        raise ValueError(f"fold metrics missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for (session, task, condition), group in fold_metrics.groupby(
        ["session", "task", "condition"], sort=True
    ):
        seed_means = group.groupby("seed", sort=True)["test_balanced_accuracy"].mean()
        rows.append({
            "session": str(session),
            "task": str(task),
            "condition": str(condition),
            "n_folds": int(group["fold"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
            "mean_test_BA": float(group["test_balanced_accuracy"].mean()),
            "median_test_BA": float(group["test_balanced_accuracy"].median()),
            "std_across_seeds": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "mean_train_BA": float(group["train_balanced_accuracy"].mean()),
            "mean_train_test_BA_gap": float(group["train_test_gap_BA"].mean()),
            "historical_group": "strong" if str(session) in STRONG_SESSIONS else "difficult",
        })
    return pd.DataFrame(rows)


def paired_ssl_improvements(session_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = session_metrics.pivot(index=["session", "task"], columns="condition")
    missing = set(SSL_CONDITIONS) - set(pivot["mean_test_BA"].columns)
    if missing:
        raise ValueError(f"session summary missing conditions: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for session, task in pivot.index:
        test_ba = pivot.loc[(session, task), "mean_test_BA"]
        gaps = pivot.loc[(session, task), "mean_train_test_BA_gap"]
        rows.append({
            "session": str(session),
            "task": str(task),
            "random_test_BA": float(test_ba["RANDOM_INIT"]),
            "frozen_test_BA": float(test_ba["SSL_FROZEN"]),
            "finetune_test_BA": float(test_ba["SSL_FINETUNE"]),
            "delta_frozen_vs_random": float(test_ba["SSL_FROZEN"] - test_ba["RANDOM_INIT"]),
            "delta_finetune_vs_random": float(test_ba["SSL_FINETUNE"] - test_ba["RANDOM_INIT"]),
            "gap_reduction_frozen": float(gaps["RANDOM_INIT"] - gaps["SSL_FROZEN"]),
            "gap_reduction_finetune": float(gaps["RANDOM_INIT"] - gaps["SSL_FINETUNE"]),
        })
    return pd.DataFrame(rows)


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


def statistical_test_tables(improvements: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    delta_columns = {
        "SSL_FINETUNE_vs_RANDOM_INIT": "delta_finetune_vs_random",
        "SSL_FROZEN_vs_RANDOM_INIT": "delta_frozen_vs_random",
    }
    for task in ("binary", "stimulus_type"):
        task_df = improvements[improvements["task"] == task]
        if task_df["session"].astype(str).nunique() != 9:
            raise AssertionError(f"{task} inference requires exactly nine session-level values")
        for comparison, column in delta_columns.items():
            values = task_df[column].to_numpy(dtype=float)
            rows.append({
                "task": task,
                "comparison": comparison,
                "primary_unit": "session",
                "n_sessions": int(len(values)),
                "mean_delta": float(np.mean(values)),
                "median_delta": float(np.median(values)),
                "positive_session_count": int(np.sum(values > 0)),
                "zero_session_count": int(np.sum(values == 0)),
                "exact_sign_flip_raw_p": exact_sign_flip_pvalue(values),
            })
    tests = pd.DataFrame(rows)

    finetune = {
        task: improvements[improvements["task"] == task]
        .sort_values("session")["delta_finetune_vs_random"]
        .to_numpy(dtype=float)
        for task in ("binary", "stimulus_type")
    }
    observed = {task: abs(float(values.mean())) for task, values in finetune.items()}
    exceed = {task: 0 for task in finetune}
    total = 0
    n = len(finetune["binary"])
    for signs in product((-1.0, 1.0), repeat=n):
        sign_array = np.asarray(signs)
        permuted = {task: abs(float(np.mean(values * sign_array))) for task, values in finetune.items()}
        max_stat = max(permuted.values())
        for task in finetune:
            exceed[task] += int(max_stat >= observed[task] - 1e-15)
        total += 1
    correction_rows = []
    for task in ("binary", "stimulus_type"):
        raw = tests[
            (tests["task"] == task)
            & (tests["comparison"] == "SSL_FINETUNE_vs_RANDOM_INIT")
        ]["exact_sign_flip_raw_p"].iloc[0]
        correction_rows.append({
            "task": task,
            "comparison": "SSL_FINETUNE_vs_RANDOM_INIT",
            "correction": "two_task_exact_max_stat_shared_session_signs",
            "raw_p": float(raw),
            "two_task_corrected_p": float(exceed[task] / total),
            "n_exact_sign_patterns": int(total),
        })
    return tests, pd.DataFrame(correction_rows)


def generalization_gap_summary(session_metrics: pd.DataFrame, improvements: pd.DataFrame) -> pd.DataFrame:
    values = session_metrics[
        ["session", "task", "condition", "mean_train_BA", "mean_test_BA", "mean_train_test_BA_gap"]
    ].copy()
    values = values.merge(
        improvements[
            ["session", "task", "gap_reduction_frozen", "gap_reduction_finetune"]
        ],
        on=["session", "task"],
        how="left",
    )
    values["historically_difficult"] = values["session"].astype(str).isin(WEAK_SESSIONS)
    return values


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_session_ba(session_metrics: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_metrics[session_metrics["task"] == task].copy()
    sessions = sorted(subset["session"].astype(str).unique(), key=int)
    x = np.arange(len(sessions))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for offset, condition in enumerate(SSL_CONDITIONS):
        values = subset[subset["condition"] == condition].set_index("session").reindex(sessions)
        ax.bar(
            x + (offset - 1) * width,
            values["mean_test_BA"],
            width,
            yerr=values["std_across_seeds"],
            label=CONDITION_LABELS[condition],
            color=CONDITION_COLORS[condition],
            capsize=2,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Chance")
    ax.set_xticks(x, sessions)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test balanced accuracy")
    ax.set_title(f"{task}: frozen clean4 SmallCNN comparison")
    ax.legend(ncol=4, fontsize=8)
    _save_figure(fig, path)


def plot_train_test_gap(session_metrics: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_metrics[session_metrics["task"] == task].copy()
    sessions = sorted(subset["session"].astype(str).unique(), key=int)
    x = np.arange(len(sessions))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for condition in SSL_CONDITIONS:
        values = subset[subset["condition"] == condition].set_index("session").reindex(sessions)
        ax.plot(x, values["mean_train_test_BA_gap"], marker="o", label=CONDITION_LABELS[condition], color=CONDITION_COLORS[condition])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x, sessions)
    ax.set_ylabel("Train BA - test BA")
    ax.set_title(f"{task}: generalization gap")
    ax.legend()
    _save_figure(fig, path)


def plot_seed_stability(fold_metrics: pd.DataFrame, task: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    seed_summary = (
        fold_metrics[fold_metrics["task"] == task]
        .groupby(["session", "condition", "seed"], sort=True)["test_balanced_accuracy"]
        .mean()
        .reset_index()
    )
    sessions = sorted(seed_summary["session"].astype(str).unique(), key=int)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, condition in zip(axes, SSL_CONDITIONS):
        values = seed_summary[seed_summary["condition"] == condition]
        for seed, group in values.groupby("seed", sort=True):
            indexed = group.set_index("session").reindex(sessions)
            ax.plot(sessions, indexed["test_balanced_accuracy"], marker="o", label=str(seed), alpha=0.8)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(CONDITION_LABELS[condition])
        ax.tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("Mean fold test BA")
    axes[-1].legend(title="Seed", fontsize=7)
    fig.suptitle(f"{task}: fixed-seed stability")
    _save_figure(fig, path)


def plot_weak_overfitting(session_metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    subset = session_metrics[session_metrics["session"].astype(str).isin(WEAK_SESSIONS)].copy()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for ax, session in zip(axes.flat, WEAK_SESSIONS):
        values = subset[subset["session"].astype(str) == session]
        for task, marker in (("binary", "o"), ("stimulus_type", "s")):
            task_values = values[values["task"] == task].set_index("condition").reindex(SSL_CONDITIONS)
            ax.plot(
                range(3), task_values["mean_train_BA"], marker=marker, linestyle="--",
                label=f"{task} train" if session == WEAK_SESSIONS[0] else None,
            )
            ax.plot(
                range(3), task_values["mean_test_BA"], marker=marker, linestyle="-",
                label=f"{task} test" if session == WEAK_SESSIONS[0] else None,
            )
        ax.set_title(f"Session {session}")
        ax.set_xticks(range(3), ["Random", "Frozen", "Finetune"], rotation=25)
        ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8)
        ax.set_ylim(0, 1.05)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("Historically difficult sessions: train versus test BA")
    _save_figure(fig, path)


def plot_reconstruction_contact_sheet(qc_files: list[Path], path: Path) -> None:
    import matplotlib.pyplot as plt
    if not qc_files:
        return
    rows = []
    for file in sorted(qc_files, key=lambda value: int(value.stem.split("_")[1])):
        data = np.load(file)
        rows.append((file.stem.split("_")[1], data))
    fig, axes = plt.subplots(len(rows), 4, figsize=(14, 3 * len(rows)), squeeze=False)
    titles = ("Original", "Masked input", "Reconstruction", "Absolute error")
    for row_i, (session, data) in enumerate(rows):
        original = data["original"]
        images = (original, data["masked"], data["reconstruction"], np.abs(data["reconstruction"] - original))
        lo, hi = np.percentile(original, [1, 99])
        for col_i, image in enumerate(images):
            axes[row_i, col_i].imshow(image, cmap="gray" if col_i < 3 else "magma", vmin=lo if col_i < 3 else None, vmax=hi if col_i < 3 else None, aspect="auto")
            axes[row_i, col_i].set_axis_off()
            if row_i == 0:
                axes[row_i, col_i].set_title(titles[col_i])
        axes[row_i, 0].set_ylabel(session)
    fig.suptitle("Fixed train-cycle masked-reconstruction QC (epoch 50)")
    _save_figure(fig, path)


def plot_single_reconstruction_qc(qc_file: Path, path: Path) -> None:
    """Render one fixed train-cycle sample without checkpoint selection."""
    import matplotlib.pyplot as plt
    data = np.load(qc_file)
    original = data["original"]
    images = (
        original,
        data["masked"],
        data["reconstruction"],
        np.abs(data["reconstruction"] - original),
    )
    titles = ("Original", "Masked input", "Reconstruction", "Absolute error")
    lo, hi = np.percentile(original, [1, 99])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for index, (ax, image, title) in enumerate(zip(axes, images, titles)):
        ax.imshow(
            image,
            cmap="gray" if index < 3 else "magma",
            vmin=lo if index < 3 else None,
            vmax=hi if index < 3 else None,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_axis_off()
    session = qc_file.stem.split("_")[1]
    fig.suptitle(
        f"Session {session}: fixed train-cycle reconstruction "
        f"(cycle {int(data['cycle'])}, frame {int(data['original_frame_index'])})"
    )
    _save_figure(fig, path)


def make_required_plots(output_dir: Path, fold_metrics: pd.DataFrame, session_metrics: pd.DataFrame) -> None:
    plot_session_ba(session_metrics, "binary", output_dir / "figures/binary/binary_session_BA.png")
    plot_session_ba(session_metrics, "stimulus_type", output_dir / "figures/stimulus_type/stimulus_type_session_BA.png")
    plot_train_test_gap(session_metrics, "binary", output_dir / "figures/binary/binary_train_test_gap.png")
    plot_train_test_gap(session_metrics, "stimulus_type", output_dir / "figures/stimulus_type/stimulus_type_train_test_gap.png")
    plot_weak_overfitting(session_metrics, output_dir / "figures/overfitting/weak_session_overfitting.png")
    plot_seed_stability(fold_metrics, "binary", output_dir / "figures/seed_stability/seed_stability_binary.png")
    plot_seed_stability(fold_metrics, "stimulus_type", output_dir / "figures/seed_stability/seed_stability_stimulus_type.png")
    plot_reconstruction_contact_sheet(
        list((output_dir / "figures/reconstruction_qc").glob("session_*_qc.npz")),
        output_dir / "figures/reconstruction_qc/ssl_reconstruction_contact_sheet.png",
    )
