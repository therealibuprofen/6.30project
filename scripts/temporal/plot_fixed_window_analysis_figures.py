#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_DIR
    / "results"
    / "runs"
    / "temporal_windows"
    / "fixed_window_unified_v1"
    / "aggregate"
    / "fixed_window_master_summary.csv"
)
OUTPUT_DIR = PROJECT_DIR / "results" / "figures" / "fixed_window_analysis"

SESSIONS = ["708", "709", "710", "807", "813", "817", "822"]
TASKS = ["binary", "stimulus_type"]
TASK_TITLES = {"binary": "Binary", "stimulus_type": "Stimulus type"}
WINDOWS = [
    "k1_p0",
    "k1_p1",
    "k1_p2",
    "k1_p3",
    "k2_p0-1",
    "k2_p1-2",
    "k2_p2-3",
    "k3_p0-1-2",
    "k3_p1-2-3",
    "k4_p0-1-2-3",
]
SINGLE_FRAME_WINDOWS = ["k1_p0", "k1_p1", "k1_p2", "k1_p3"]
POSITIONS = [0, 1, 2, 3]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )


def load_plotting_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input summary not found: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    required = {"session", "task", "window_id", "window_start_position", "balanced_accuracy"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {INPUT_PATH}: {sorted(missing)}")
    df["session"] = df["session"].astype(str)
    expected_rows = len(SESSIONS) * len(TASKS) * len(WINDOWS)
    observed = df[df["session"].isin(SESSIONS) & df["task"].isin(TASKS) & df["window_id"].isin(WINDOWS)]
    if len(observed) != expected_rows:
        raise ValueError(f"Expected {expected_rows} plotting rows, found {len(observed)}")
    return observed.copy()


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [OUTPUT_DIR / f"{stem}.png", OUTPUT_DIR / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def ordered_heatmap_values(df: pd.DataFrame, task: str) -> pd.DataFrame:
    pivot = (
        df[df["task"] == task]
        .pivot(index="session", columns="window_id", values="balanced_accuracy")
        .reindex(index=SESSIONS, columns=WINDOWS)
    )
    if pivot.isna().any().any():
        raise ValueError(f"Missing heatmap values for {task}")
    return pivot


def plot_heatmap(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    im = None
    for ax, task, label in zip(axes, TASKS, ["A", "B"]):
        values = ordered_heatmap_values(df, task)
        im = ax.imshow(values.to_numpy(), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(WINDOWS)))
        ax.set_xticklabels(WINDOWS, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(SESSIONS)))
        ax.set_yticklabels(SESSIONS)
        ax.set_ylabel("Session")
        ax.set_title(TASK_TITLES[task], pad=8)
        panel_label(ax, label)
        ax.set_xticks(np.arange(-0.5, len(WINDOWS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(SESSIONS), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_i in range(values.shape[0]):
            for col_i in range(values.shape[1]):
                value = float(values.iloc[row_i, col_i])
                text_color = "white" if value < 0.55 else "black"
                ax.text(col_i, row_i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7.5)
    axes[-1].set_xlabel("Fixed window")
    fig.suptitle("Fixed-window decoding performance across sessions", fontsize=13, y=1.02)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Balanced accuracy")
    return save_figure(fig, "fixed_window_heatmap")


def single_frame_data(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df[(df["task"] == task) & (df["window_id"].isin(SINGLE_FRAME_WINDOWS))].copy()
    out["window_id"] = pd.Categorical(out["window_id"], SINGLE_FRAME_WINDOWS, ordered=True)
    out = out.sort_values(["session", "window_id"])
    if len(out) != len(SESSIONS) * len(SINGLE_FRAME_WINDOWS):
        raise ValueError(f"Missing single-frame values for {task}")
    return out


def plot_single_frame_curves(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True, constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for ax, task, label in zip(axes, TASKS, ["A", "B"]):
        task_df = single_frame_data(df, task)
        for session_i, session in enumerate(SESSIONS):
            session_df = task_df[task_df["session"] == session]
            ax.plot(
                POSITIONS,
                session_df["balanced_accuracy"].to_numpy(),
                color=colors[session_i % len(colors)],
                marker=markers[session_i],
                linewidth=1.4,
                markersize=4.5,
                label=session,
            )
        ax.axhline(0.5, color="0.25", linestyle="--", linewidth=1.0)
        ax.text(3.03, 0.505, "chance", ha="left", va="bottom", fontsize=8, color="0.25")
        ax.set_xticks(POSITIONS)
        ax.set_xticklabels([f"Position {position}" for position in POSITIONS])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Block-relative position")
        ax.set_title(TASK_TITLES[task], pad=8)
        panel_label(ax, label)
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Balanced accuracy")
    axes[1].legend(title="Session", frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Single-frame fixed-window decoding across block-relative positions", fontsize=13, y=1.04)
    fig.text(
        0.5,
        -0.02,
        "Positions 0-3 correspond to the four clean-middle sampling locations within each block.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, "fixed_window_single_frame_curves")


def plot_single_frame_cross_session_mean(df: pd.DataFrame) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True, constrained_layout=True)
    for ax, task, label in zip(axes, TASKS, ["A", "B"]):
        task_df = single_frame_data(df, task)
        summary = (
            task_df.groupby("window_start_position", sort=True)["balanced_accuracy"]
            .agg(["mean", "std"])
            .reindex(POSITIONS)
        )
        ax.errorbar(
            POSITIONS,
            summary["mean"].to_numpy(),
            yerr=summary["std"].to_numpy(),
            color="black",
            marker="o",
            linewidth=1.6,
            markersize=5,
            capsize=4,
        )
        ax.axhline(0.5, color="0.25", linestyle="--", linewidth=1.0)
        ax.text(3.03, 0.505, "chance", ha="left", va="bottom", fontsize=8, color="0.25")
        ax.set_xticks(POSITIONS)
        ax.set_xticklabels([f"Position {position}" for position in POSITIONS])
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Block-relative position")
        ax.set_title(TASK_TITLES[task], pad=8)
        panel_label(ax, label)
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Balanced accuracy")
    fig.suptitle("Session-equal mean single-frame decoding", fontsize=13, y=1.04)
    fig.text(
        0.5,
        -0.02,
        "Means and standard deviations are computed across the seven sessions with equal session weight.",
        ha="center",
        fontsize=9,
    )
    return save_figure(fig, "fixed_window_single_frame_cross_session_mean")


def write_plotting_readme(paths: list[Path]) -> Path:
    readme_path = OUTPUT_DIR / "plotting_readme.txt"
    all_paths = [*paths, readme_path]
    lines = [
        "Fixed-window analysis plotting notes",
        "====================================",
        "",
        "Input file:",
        f"- {INPUT_PATH}",
        "",
        "Main figure:",
        "- fixed_window_heatmap.png / fixed_window_heatmap.pdf",
        "  Two-panel heatmap of balanced_accuracy for binary and stimulus_type tasks.",
        "",
        "Secondary figure:",
        "- fixed_window_single_frame_curves.png / fixed_window_single_frame_curves.pdf",
        "  Single-frame windows k1_p0 through k1_p3, with one line per session.",
        "",
        "Supplementary figure:",
        "- fixed_window_single_frame_cross_session_mean.png / fixed_window_single_frame_cross_session_mean.pdf",
        "  Equal-session mean and standard deviation across the seven sessions.",
        "",
        "Axis notes:",
        "- Position 0, 1, 2, and 3 are block-relative positions corresponding to the four clean-middle sampling locations within each block.",
        "- Positions are not treated as exact timestamps.",
        "",
        "Metric:",
        "- All figures use balanced_accuracy.",
        "",
        "Output files:",
        *[f"- {path}" for path in all_paths],
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_plotting_data()
    plotting_data_path = OUTPUT_DIR / "fixed_window_plotting_data.csv"
    df[
        [
            "session",
            "task",
            "window_id",
            "window_size",
            "window_start_position",
            "position",
            "balanced_accuracy",
            "macro_f1",
            "n_samples",
            "time_mapping_status",
        ]
    ].to_csv(plotting_data_path, index=False)

    paths: list[Path] = [plotting_data_path]
    paths.extend(plot_heatmap(df))
    paths.extend(plot_single_frame_curves(df))
    paths.extend(plot_single_frame_cross_session_mean(df))
    paths.append(write_plotting_readme(paths))

    print(f"Read {len(df)} rows from {INPUT_PATH}")
    print(f"Sessions: {', '.join(SESSIONS)}")
    print(f"Tasks: {', '.join(TASKS)}")
    print("Generated files:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
