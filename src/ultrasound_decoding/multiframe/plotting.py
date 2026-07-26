from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import CHANCE_LEVEL, seed_mean_summary
from .models import MODEL_DISPLAY_NAMES


METHOD_ORDER = [
    "pca_lda_flat4",
    "cpca_lda_flat4",
    "cnn2d_meanpool",
    "cnn2d_lstm",
    "cnn2d_temporal1d",
    "single_frame_late_fusion",
    "fcnn_late_fusion",
    "fcnn_meanpool",
    "fcnn_lstm",
]
METHOD_COLORS = {
    "pca_lda_flat4": "#4C78A8",
    "cpca_lda_flat4": "#F58518",
    "cnn2d_meanpool": "#54A24B",
    "cnn2d_lstm": "#E45756",
    "cnn2d_temporal1d": "#72B7B2",
    "single_frame_late_fusion": "#B279A2",
    "fcnn_late_fusion": "#9D755D",
    "fcnn_meanpool": "#59A14F",
    "fcnn_lstm": "#EDC948",
}


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def save_png_pdf(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def plot_method_comparison(master: pd.DataFrame, task: str, out_dir: Path, stem: str) -> list[Path]:
    plt = _setup_matplotlib()
    summary = seed_mean_summary(master)
    summary = summary[summary["task"] == task].copy()
    if summary.empty:
        return []
    summary["session"] = summary["session"].astype(str)
    sessions = sorted(summary["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    x = np.arange(len(sessions), dtype=float)
    methods = [method for method in METHOD_ORDER if method in set(summary["method"])]
    width = min(0.12, 0.78 / max(len(methods), 1))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for method_i, method in enumerate(methods):
        subset = summary[summary["method"] == method].set_index("session")
        means = [float(subset.loc[session, "balanced_accuracy_mean"]) if session in subset.index else np.nan for session in sessions]
        stds = [float(subset.loc[session, "balanced_accuracy_std"]) if session in subset.index else 0.0 for session in sessions]
        offset = (method_i - (len(methods) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            color=METHOD_COLORS.get(method, "#777777"),
            label=MODEL_DISPLAY_NAMES[method],
            edgecolor="white",
            linewidth=0.4,
            yerr=stds if method not in {"pca_lda_flat4", "cpca_lda_flat4"} else None,
            capsize=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
    ax.axhline(CHANCE_LEVEL, color="#333333", linestyle="--", linewidth=1.0, label="Chance level")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.set_xlabel("Session")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"Block-level clean4 multiframe decoding: {task}")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_temporal_gain(master: pd.DataFrame, task: str, out_dir: Path, stem: str) -> list[Path]:
    plt = _setup_matplotlib()
    summary = seed_mean_summary(master)
    summary = summary[summary["task"] == task].copy()
    if summary.empty:
        return []
    summary["session"] = summary["session"].astype(str)
    pivot = summary.pivot_table(
        index="session",
        columns="method",
        values="balanced_accuracy_mean",
        aggfunc="mean",
    )
    required = {"cnn2d_meanpool", "cnn2d_lstm", "cnn2d_temporal1d"}
    if not required.issubset(set(pivot.columns)):
        return []
    sessions = sorted(pivot.index.astype(str).tolist(), key=lambda value: int(value))
    lstm_gain = [float(pivot.loc[session, "cnn2d_lstm"] - pivot.loc[session, "cnn2d_meanpool"]) for session in sessions]
    temporal_gain = [
        float(pivot.loc[session, "cnn2d_temporal1d"] - pivot.loc[session, "cnn2d_meanpool"])
        for session in sessions
    ]
    x = np.arange(len(sessions), dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - 0.18, lstm_gain, width=0.34, color=METHOD_COLORS["cnn2d_lstm"], label="CNN-LSTM minus mean-pool")
    ax.bar(
        x + 0.18,
        temporal_gain,
        width=0.34,
        color=METHOD_COLORS["cnn2d_temporal1d"],
        label="Temporal 1D-CNN minus mean-pool",
    )
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.set_xlabel("Session")
    ax.set_ylabel("Balanced Accuracy Gain")
    ax.set_title(f"Temporal aggregation gain: {task}")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_order_sensitivity(order_df: pd.DataFrame, task: str, out_dir: Path, stem: str) -> list[Path]:
    plt = _setup_matplotlib()
    df = order_df[order_df["task"] == task].copy() if not order_df.empty else pd.DataFrame()
    if df.empty:
        return []
    grouped = (
        df.groupby(["session", "method"], sort=True)[
            ["original_order_ba", "reverse_order_ba", "shuffled_order_ba"]
        ]
        .mean()
        .reset_index()
    )
    grouped["session"] = grouped["session"].astype(str)
    sessions = sorted(grouped["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    methods = [method for method in ["cnn2d_lstm", "cnn2d_temporal1d", "fcnn_lstm"] if method in set(grouped["method"])]
    if not methods:
        return []
    fig, axes = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.2), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    order_cols = [
        ("original_order_ba", "Original", "#4C78A8"),
        ("reverse_order_ba", "Reverse", "#F58518"),
        ("shuffled_order_ba", "Fixed shuffle", "#E45756"),
    ]
    x = np.arange(len(sessions), dtype=float)
    width = 0.24
    for ax, method in zip(axes, methods):
        subset = grouped[grouped["method"] == method].set_index("session")
        for i, (column, label, color) in enumerate(order_cols):
            values = [float(subset.loc[session, column]) if session in subset.index else np.nan for session in sessions]
            ax.bar(x + (i - 1) * width, values, width=width, color=color, label=label, edgecolor="white", linewidth=0.4)
        ax.axhline(CHANCE_LEVEL, color="#333333", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sessions)
        ax.set_xlabel("Session")
        ax.set_title(MODEL_DISPLAY_NAMES[method])
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0].set_ylabel("Balanced Accuracy")
    axes[-1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.suptitle(f"Order sensitivity without retraining: {task}", y=1.02)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_confusion_matrices(
    confusion_df: pd.DataFrame,
    task: str,
    out_dir: Path,
    stem: str,
    sessions: list[str] | None = None,
) -> list[Path]:
    plt = _setup_matplotlib()
    df = confusion_df[confusion_df["task"] == task].copy() if not confusion_df.empty else pd.DataFrame()
    if df.empty:
        return []
    if sessions is None:
        sessions = ["708", "709", "710"] if task == "binary" else sorted(df["session"].astype(str).unique().tolist())[:3]
    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    if not methods:
        return []
    df = df[df["session"].astype(str).isin([str(session) for session in sessions])]
    if df.empty:
        return []
    fig, axes = plt.subplots(len(sessions), len(methods), figsize=(2.15 * len(methods), 2.1 * len(sessions)))
    axes = np.asarray(axes).reshape(len(sessions), len(methods))
    class_names = sorted(set(df["truth"].astype(str).tolist()) | set(df["pred"].astype(str).tolist()))
    for row_i, session in enumerate(sessions):
        for col_i, method in enumerate(methods):
            ax = axes[row_i, col_i]
            subset = df[
                (df["session"].astype(str) == str(session))
                & (df["method"] == method)
                & (df["fold"].isna() if "fold" in df.columns else True)
            ]
            if subset.empty:
                subset = df[(df["session"].astype(str) == str(session)) & (df["method"] == method)]
            matrix = np.zeros((len(class_names), len(class_names)), dtype=int)
            for _, item in subset.iterrows():
                i = class_names.index(str(item["truth"]))
                j = class_names.index(str(item["pred"]))
                matrix[i, j] += int(item["count"])
            vmax = max(int(matrix.max()), 1)
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=8, color="#111111")
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=35, ha="right")
            ax.set_yticklabels(class_names)
            if row_i == 0:
                ax.set_title(MODEL_DISPLAY_NAMES[method], fontsize=8)
            if col_i == 0:
                ax.set_ylabel(f"{session}\nTruth")
            else:
                ax.set_yticklabels([])
            if row_i == len(sessions) - 1:
                ax.set_xlabel("Pred")
    fig.suptitle(f"Block-level confusion matrices: {task}", y=1.01)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_block_type_accuracy(block_type_df: pd.DataFrame, task: str, out_dir: Path, stem: str) -> list[Path]:
    plt = _setup_matplotlib()
    df = block_type_df[block_type_df["task"] == task].copy() if not block_type_df.empty else pd.DataFrame()
    if df.empty:
        return []
    df["session"] = df["session"].astype(str)
    sessions = sorted(df["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    block_cols = [
        ("grating_accuracy", "grating"),
        ("dot_accuracy", "dot"),
        ("stop_after_grating_accuracy", "stop"),
        ("static_accuracy", "static"),
    ]
    fig, axes = plt.subplots(len(methods), 1, figsize=(8.8, max(2.2, 1.25 * len(methods))), sharex=True)
    axes = np.asarray(axes).reshape(len(methods))
    x = np.arange(len(sessions), dtype=float)
    width = 0.18
    colors = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]
    for ax, method in zip(axes, methods):
        subset = df[df["method"] == method].set_index("session")
        for i, ((col, label), color) in enumerate(zip(block_cols, colors)):
            values = [float(subset.loc[session, col]) if session in subset.index else np.nan for session in sessions]
            ax.bar(x + (i - 1.5) * width, values, width=width, color=color, label=label, edgecolor="white", linewidth=0.4)
        ax.axhline(CHANCE_LEVEL, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(MODEL_DISPLAY_NAMES.get(method, method), rotation=0, ha="right", va="center")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(sessions)
    axes[-1].set_xlabel("Session")
    axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.suptitle(f"Block-type accuracy: {task}", y=1.01)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_generalization_gap(overfitting_summary: pd.DataFrame, task: str, out_dir: Path, stem: str) -> list[Path]:
    plt = _setup_matplotlib()
    df = overfitting_summary[overfitting_summary["task"] == task].copy() if not overfitting_summary.empty else pd.DataFrame()
    if df.empty:
        return []
    methods = [method for method in METHOD_ORDER if method in set(df["method"])]
    sessions = sorted(df["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    x = np.arange(len(methods), dtype=float)
    width = min(0.11, 0.8 / max(len(sessions), 1))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, session in enumerate(sessions):
        subset = df[df["session"].astype(str) == session].set_index("method")
        values = [float(subset.loc[method, "mean_generalization_gap"]) if method in subset.index else np.nan for method in methods]
        ax.bar(x + (i - (len(sessions) - 1) / 2.0) * width, values, width=width, label=session, linewidth=0.3)
    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(method, method) for method in methods], rotation=25, ha="right")
    ax.set_ylabel("Final train accuracy minus test BA")
    ax.set_title(f"Train-test diagnostic gap: {task}")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(title="Session", frameon=False, ncol=min(len(sessions), 7))
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_parameter_count_vs_test_ba(master: pd.DataFrame, task: str, out_dir: Path, stem: str) -> list[Path]:
    plt = _setup_matplotlib()
    df = master[master["task"] == task].copy() if not master.empty else pd.DataFrame()
    if df.empty or "model_parameters" not in df:
        return []
    summary = seed_mean_summary(df)
    if summary.empty:
        return []
    params = (
        df.groupby(["session", "task", "method"], sort=True)["model_parameters"]
        .first()
        .reset_index()
    )
    summary["session"] = summary["session"].astype(str)
    params["session"] = params["session"].astype(str)
    summary = summary.merge(params, on=["session", "task", "method"], how="left")
    summary["model_parameters"] = pd.to_numeric(summary["model_parameters"], errors="coerce")
    summary = summary.dropna(subset=["model_parameters"])
    if summary.empty:
        return []
    positive_params = summary.loc[summary["model_parameters"] > 0, "model_parameters"].astype(float)
    zero_anchor = max(1.0, float(positive_params.min()) / 10.0) if not positive_params.empty else 1.0
    if not positive_params.empty and any(np.isclose(positive_params, zero_anchor)):
        zero_anchor = max(0.1, zero_anchor / 10.0)
    summary["model_parameters_plot"] = summary["model_parameters"].astype(float).where(
        summary["model_parameters"].astype(float) > 0,
        zero_anchor,
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in [method for method in METHOD_ORDER if method in set(summary["method"])]:
        subset = summary[summary["method"] == method]
        ax.scatter(
            subset["model_parameters_plot"].astype(float),
            subset["balanced_accuracy_mean"].astype(float),
            s=34,
            color=METHOD_COLORS.get(method, "#777777"),
            label=MODEL_DISPLAY_NAMES.get(method, method),
            alpha=0.9,
        )
    ax.axhline(CHANCE_LEVEL, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    positive_tick_values = sorted(
        {
            float(value)
            for value in summary.loc[summary["model_parameters"] > 0, "model_parameters"].unique().tolist()
        }
    )
    tick_values = [zero_anchor, *positive_tick_values]
    tick_labels = ["0 (linear)", *[f"{int(value):,}" for value in positive_tick_values]]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def make_all_plots(
    master: pd.DataFrame,
    order_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    task: str,
    out_dir: Path,
    overfitting_summary: pd.DataFrame | None = None,
    block_type_df: pd.DataFrame | None = None,
) -> list[Path]:
    paths: list[Path] = []
    comparison_stem = (
        "multiframe_binary_method_comparison"
        if task == "binary"
        else "multiframe_stimulus_type_method_comparison"
    )
    paths.extend(plot_method_comparison(master, task, out_dir, comparison_stem))
    paths.extend(plot_temporal_gain(master, task, out_dir, "temporal_model_gain"))
    paths.extend(plot_order_sensitivity(order_df, task, out_dir, "order_sensitivity"))
    paths.extend(plot_confusion_matrices(confusion_df, task, out_dir, "multiframe_confusion_matrices"))
    if overfitting_summary is not None:
        paths.extend(plot_generalization_gap(overfitting_summary, task, out_dir, "generalization_gap"))
    if block_type_df is not None:
        paths.extend(plot_block_type_accuracy(block_type_df, task, out_dir, "block_type_accuracy"))
    paths.extend(plot_parameter_count_vs_test_ba(master, task, out_dir, "parameter_count_vs_test_ba"))
    return paths
