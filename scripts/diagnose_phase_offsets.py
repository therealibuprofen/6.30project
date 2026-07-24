#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MPLCONFIG_DIR = PROJECT_DIR / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultrasound_decoding.datasets import DEFAULT_ANALYSIS_LIMITS
from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.io import load_session_frames
from ultrasound_decoding.labels import STIMULUS_BLOCKS, is_clean_block_middle
from ultrasound_decoding.linear import fit_predict_linear, preprocess_frames


TASKS = ("binary", "stimulus_type")
PHASE_OFFSETS_S = tuple(float(offset) for offset in range(0, 29, 4))


@dataclass(frozen=True)
class OffsetFrameTimingLabel:
    index: int
    cycle: int
    center_s: float
    center_in_cycle_s: float
    block_start_s: float
    block_name: str
    binary_label: str


def infer_offset_label(
    index: int,
    phase_offset_s: float,
    group_seconds: float = 4.0,
    cycle_seconds: float = 120.0,
) -> OffsetFrameTimingLabel:
    """Infer labels after shifting the stimulus phase by phase_offset_s."""
    zero_based = index - 1
    start_s = zero_based * group_seconds
    center_s = start_s + group_seconds / 2.0
    shifted_center_s = center_s - phase_offset_s
    cycle = math.floor(shifted_center_s / cycle_seconds)
    center_in_cycle_s = shifted_center_s % cycle_seconds
    block_i = min(int(center_in_cycle_s // 30.0), len(STIMULUS_BLOCKS) - 1)
    block_start_s = block_i * 30.0
    block_name, binary_label = STIMULUS_BLOCKS[block_i]
    return OffsetFrameTimingLabel(
        index=index,
        cycle=cycle,
        center_s=center_s,
        center_in_cycle_s=center_in_cycle_s,
        block_start_s=block_start_s,
        block_name=block_name,
        binary_label=binary_label,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan label phase offsets with PCA+LDA and leave-one-cycle-out CV."
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Session folders under data/. Defaults to all numeric session directories.",
    )
    parser.add_argument(
        "--phase-offsets-s",
        nargs="+",
        type=float,
        default=list(PHASE_OFFSETS_S),
        help="Phase offsets in seconds to scan. Defaults to 0, 4, ..., 28.",
    )
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument("--frames-per-cycle", type=int, default=30)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument(
        "--analysis-limit",
        default="default",
        help=(
            "Use 'default' to match DEFAULT_ANALYSIS_LIMITS where present, "
            "'none' for no limits, or a range like 1:180 applied to all sessions."
        ),
    )
    return parser.parse_args()


def discover_sessions(root: Path) -> list[str]:
    sessions = [path.name for path in (root / "data").iterdir() if path.is_dir()]
    return sorted(sessions, key=lambda value: int(value) if value.isdigit() else value)


def analysis_limit_for_session(session: str, value: str) -> tuple[int, int] | None:
    if value == "default":
        return DEFAULT_ANALYSIS_LIMITS.get(session)
    if value.lower() in {"none", "null"}:
        return None
    lo, hi = value.split(":")
    return int(lo), int(hi)


def leave_one_cycle_out_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = []
    for group in np.unique(groups):
        test_mask = groups == group
        splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
    if len(splits) < 2:
        raise ValueError("Need at least two complete cycles for leave-one-cycle-out CV")
    return splits


def complete_cycles_for_offset(
    indices: np.ndarray,
    phase_offset_s: float,
    frames_per_cycle: int,
    analysis_limit: tuple[int, int] | None,
) -> set[int]:
    analysis_keep = np.ones(len(indices), dtype=bool)
    if analysis_limit is not None:
        lo, hi = analysis_limit
        analysis_keep &= (indices >= lo) & (indices <= hi)

    cycle_counts: dict[int, int] = {}
    for index, keep in zip(indices, analysis_keep):
        if not keep:
            continue
        cycle = infer_offset_label(int(index), phase_offset_s).cycle
        cycle_counts[cycle] = cycle_counts.get(cycle, 0) + 1
    return {cycle for cycle, count in cycle_counts.items() if count == frames_per_cycle}


def common_complete_cycles(
    indices: np.ndarray,
    phase_offsets_s: list[float],
    frames_per_cycle: int,
    analysis_limit: tuple[int, int] | None,
) -> set[int]:
    cycle_sets = [
        complete_cycles_for_offset(
            indices=indices,
            phase_offset_s=offset,
            frames_per_cycle=frames_per_cycle,
            analysis_limit=analysis_limit,
        )
        for offset in phase_offsets_s
    ]
    if not cycle_sets:
        return set()
    return set.intersection(*cycle_sets)


def build_offset_dataset(
    X_raw: np.ndarray,
    indices: np.ndarray,
    session: str,
    task: str,
    phase_offset_s: float,
    clean_margin_s: float,
    frames_per_cycle: int,
    analysis_limit: tuple[int, int] | None,
    allowed_cycles: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    analysis_keep = np.ones(len(indices), dtype=bool)
    if analysis_limit is not None:
        lo, hi = analysis_limit
        analysis_keep &= (indices >= lo) & (indices <= hi)

    labels = [infer_offset_label(int(index), phase_offset_s) for index in indices]
    meta = pd.DataFrame(
        {
            **asdict(label),
            "session": session,
            "task": task,
            "phase_offset_s": phase_offset_s,
            "after_analysis_limit": bool(keep),
            "complete_cycle": False,
            "clean_middle": False,
            "selected_before_task": False,
        }
        for label, keep in zip(labels, analysis_keep)
    )

    if allowed_cycles is None:
        complete_cycles = complete_cycles_for_offset(
            indices=indices,
            phase_offset_s=phase_offset_s,
            frames_per_cycle=frames_per_cycle,
            analysis_limit=analysis_limit,
        )
    else:
        complete_cycles = allowed_cycles

    complete_cycle_keep = meta["cycle"].isin(complete_cycles).to_numpy()
    meta["complete_cycle"] = complete_cycle_keep
    clean_keep = np.asarray(
        [is_clean_block_middle(label, clean_margin_s) for label in labels],
        dtype=bool,
    )
    meta["clean_middle"] = clean_keep

    keep = analysis_keep & complete_cycle_keep & clean_keep
    meta["selected_before_task"] = keep
    X = X_raw[keep]
    meta = meta.loc[keep].reset_index(drop=True)

    if task == "binary":
        y = meta["binary_label"].to_numpy()
    elif task == "stimulus_type":
        stim_keep = meta["block_name"].isin(["grating", "dot"]).to_numpy()
        X = X[stim_keep]
        meta = meta.loc[stim_keep].reset_index(drop=True)
        y = meta["block_name"].to_numpy()
    else:
        raise ValueError(f"Unknown task: {task}")

    groups = meta["cycle"].to_numpy(dtype=np.int64)
    return X, y, groups, meta


def valid_fold(y_train: np.ndarray, y_test: np.ndarray) -> bool:
    return len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2


def run_pca_lda_loco(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    pca_variance: float,
) -> dict[str, float]:
    X_flat = preprocess_frames(X)
    splits = leave_one_cycle_out_splits(groups)
    all_true: list[str] = []
    all_pred: list[str] = []

    for train_idx, test_idx in splits:
        y_train = y[train_idx]
        y_test = y[test_idx]
        if not valid_fold(y_train, y_test):
            continue
        pred, _ = fit_predict_linear(
            "pca_lda",
            X_flat[train_idx],
            y_train,
            X_flat[test_idx],
            pca_variance=pca_variance,
        )
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

    if not all_true:
        raise ValueError("No valid leave-one-cycle-out folds remained")
    return classification_metrics(np.asarray(all_true), np.asarray(all_pred))


def plot_balanced_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    sessions = sorted(df["session"].unique(), key=lambda value: int(value) if value.isdigit() else value)
    tasks = [task for task in TASKS if task in set(df["task"])]
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(6.2 * len(tasks), 4.8),
        dpi=160,
        sharey=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10")

    for ax, task in zip(axes[0], tasks):
        task_df = df[df["task"] == task]
        for i, session in enumerate(sessions):
            session_df = task_df[task_df["session"] == session].sort_values("phase_offset_s")
            if session_df.empty:
                continue
            ax.plot(
                session_df["phase_offset_s"],
                session_df["balanced_accuracy"],
                marker="o",
                linewidth=1.8,
                label=session,
                color=cmap(i % 10),
            )
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.1, label="Chance = 0.5")
        ax.set_title(task)
        ax.set_xlabel("Phase offset (s)")
        ax.set_xticks(sorted(df["phase_offset_s"].unique()))
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0][0].set_ylabel("Balanced accuracy")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 8), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sessions = args.sessions if args.sessions is not None else discover_sessions(PROJECT_DIR)
    out_dir = PROJECT_DIR / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for session in sessions:
        X_raw, indices = load_session_frames(PROJECT_DIR / "data" / session)
        analysis_limit = analysis_limit_for_session(session, args.analysis_limit)
        shared_cycles = common_complete_cycles(
            indices=indices,
            phase_offsets_s=args.phase_offsets_s,
            frames_per_cycle=args.frames_per_cycle,
            analysis_limit=analysis_limit,
        )
        if len(shared_cycles) < 2:
            raise ValueError(
                f"{session} has fewer than two cycles complete across all requested offsets"
            )
        for task in TASKS:
            for offset in args.phase_offsets_s:
                X, y, groups, _ = build_offset_dataset(
                    X_raw=X_raw,
                    indices=indices,
                    session=session,
                    task=task,
                    phase_offset_s=offset,
                    clean_margin_s=args.clean_margin_s,
                    frames_per_cycle=args.frames_per_cycle,
                    analysis_limit=analysis_limit,
                    allowed_cycles=shared_cycles,
                )
                metrics = run_pca_lda_loco(X, y, groups, args.pca_variance)
                row = {
                    "session": session,
                    "task": task,
                    "phase_offset_s": int(offset) if float(offset).is_integer() else offset,
                    "n_samples": int(len(X)),
                    "n_cycles": int(len(np.unique(groups))),
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{session} {task} offset={offset:g}s "
                    f"n={row['n_samples']} cycles={row['n_cycles']} "
                    f"balanced_accuracy={row['balanced_accuracy']:.3f}",
                    flush=True,
                )

    result_df = pd.DataFrame(
        rows,
        columns=[
            "session",
            "task",
            "phase_offset_s",
            "n_samples",
            "n_cycles",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    csv_path = out_dir / "phase_offset_scan.csv"
    fig_path = out_dir / "phase_offset_balanced_accuracy.png"
    result_df.to_csv(csv_path, index=False)
    plot_balanced_accuracy(result_df, fig_path)
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
