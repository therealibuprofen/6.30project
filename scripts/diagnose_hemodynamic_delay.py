#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
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
from ultrasound_decoding.labels import STIMULUS_BLOCKS, infer_visual_stimulus_label, is_clean_block_middle
from ultrasound_decoding.linear import fit_predict_linear, preprocess_frames


SESSION = "813"
TASK = "binary"
METHOD = "pca_lda"
DELAYS_S = (0.0, 4.0, 8.0, 12.0, 16.0)


@dataclass(frozen=True)
class TimingLabel:
    index: int
    cycle: int
    center_s: float
    center_in_cycle_s: float
    block_start_s: float
    block_name: str
    binary_label: str


def label_from_center_s(
    index: int,
    center_s: float,
    cycle_seconds: float = 120.0,
) -> TimingLabel:
    cycle = math.floor(center_s / cycle_seconds)
    center_in_cycle_s = center_s % cycle_seconds
    block_i = min(int(center_in_cycle_s // 30.0), len(STIMULUS_BLOCKS) - 1)
    block_start_s = block_i * 30.0
    block_name, binary_label = STIMULUS_BLOCKS[block_i]
    return TimingLabel(
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
        description=(
            "Scan hemodynamic delays for session 813 binary PCA+LDA with "
            "leave-one-cycle-out CV."
        )
    )
    parser.add_argument("--session", default=SESSION)
    parser.add_argument("--task", default=TASK, choices=[TASK])
    parser.add_argument("--method", default=METHOD, choices=[METHOD])
    parser.add_argument("--delay-s", nargs="+", type=float, default=list(DELAYS_S))
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument("--frames-per-cycle", type=int, default=30)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument(
        "--analysis-limit",
        default="default",
        help=(
            "Use 'default' to match DEFAULT_ANALYSIS_LIMITS where present, "
            "'none' for no limits, or a range like 1:180."
        ),
    )
    return parser.parse_args()


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


def complete_fus_cycles(
    indices: np.ndarray,
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
        cycle = infer_visual_stimulus_label(int(index)).cycle
        cycle_counts[cycle] = cycle_counts.get(cycle, 0) + 1
    return {cycle for cycle, count in cycle_counts.items() if count == frames_per_cycle}


def delayed_label_for_fus_index(index: int, delay_s: float) -> TimingLabel:
    fus_label = infer_visual_stimulus_label(index)
    return label_from_center_s(index=index, center_s=fus_label.center_s - delay_s)


def stays_within_same_block(index: int, delayed_label: TimingLabel) -> bool:
    fus_label = infer_visual_stimulus_label(index)
    return (
        fus_label.cycle == delayed_label.cycle
        and fus_label.block_start_s == delayed_label.block_start_s
    )


def build_delay_dataset(
    X_raw: np.ndarray,
    indices: np.ndarray,
    session: str,
    task: str,
    delay_s: float,
    clean_margin_s: float,
    complete_cycles: set[int],
    analysis_limit: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    analysis_keep = np.ones(len(indices), dtype=bool)
    if analysis_limit is not None:
        lo, hi = analysis_limit
        analysis_keep &= (indices >= lo) & (indices <= hi)

    fus_labels = [infer_visual_stimulus_label(int(index)) for index in indices]
    delayed_labels = [delayed_label_for_fus_index(int(index), delay_s) for index in indices]
    complete_cycle_keep = np.asarray(
        [label.cycle in complete_cycles for label in fus_labels],
        dtype=bool,
    )
    clean_keep = np.asarray(
        [is_clean_block_middle(label, clean_margin_s) for label in fus_labels],
        dtype=bool,
    )
    within_block_keep = np.asarray(
        [stays_within_same_block(int(index), label) for index, label in zip(indices, delayed_labels)],
        dtype=bool,
    )

    meta = pd.DataFrame(
        {
            "index": [label.index for label in fus_labels],
            "session": session,
            "task": task,
            "method": METHOD,
            "delay_s": delay_s,
            "cycle": [label.cycle for label in fus_labels],
            "fus_center_s": [label.center_s for label in fus_labels],
            "fus_center_in_cycle_s": [label.center_in_cycle_s for label in fus_labels],
            "fus_block_start_s": [label.block_start_s for label in fus_labels],
            "fus_block_name": [label.block_name for label in fus_labels],
            "stim_center_s": [label.center_s for label in delayed_labels],
            "stim_center_in_cycle_s": [label.center_in_cycle_s for label in delayed_labels],
            "stim_block_start_s": [label.block_start_s for label in delayed_labels],
            "block_name": [label.block_name for label in delayed_labels],
            "binary_label": [label.binary_label for label in delayed_labels],
            "after_analysis_limit": analysis_keep,
            "complete_cycle": complete_cycle_keep,
            "clean_middle": clean_keep,
            "within_same_30s_block": within_block_keep,
        }
    )

    keep = analysis_keep & complete_cycle_keep & clean_keep & within_block_keep
    meta["selected_before_task"] = keep
    X = X_raw[keep]
    meta = meta.loc[keep].reset_index(drop=True)

    if task != "binary":
        raise ValueError("This diagnostic is currently fixed to the binary task")
    y = meta["binary_label"].to_numpy()
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
            METHOD,
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
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    plot_df = df.sort_values("delay_s")
    ax.plot(
        plot_df["delay_s"],
        plot_df["balanced_accuracy"],
        marker="o",
        linewidth=2.0,
        color="#4C78A8",
    )
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.1, label="Chance = 0.5")
    ax.set_xlabel("Delay (s)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_xticks(sorted(df["delay_s"].unique()))
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Session 813 binary PCA+LDA delay scan")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_sample_table(meta: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "delay_s": meta["delay_s"].map(
                lambda value: int(value) if float(value).is_integer() else value
            ),
            "cycle": meta["cycle"].astype(int),
            "block": meta["block_name"],
            "original_frame_id": meta["index"].astype(int),
            "frame_center_time_s": meta["fus_center_s"],
            "label": meta["binary_label"],
        }
    )


def main() -> None:
    args = parse_args()
    out_dir = PROJECT_DIR / "reports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    X_raw, indices = load_session_frames(PROJECT_DIR / "data" / args.session)
    analysis_limit = analysis_limit_for_session(args.session, args.analysis_limit)
    shared_cycles = complete_fus_cycles(
        indices=indices,
        frames_per_cycle=args.frames_per_cycle,
        analysis_limit=analysis_limit,
    )
    if len(shared_cycles) < 2:
        raise ValueError(f"{args.session} has fewer than two complete fUS cycles")

    rows = []
    sample_tables = []
    for delay_s in args.delay_s:
        X, y, groups, meta = build_delay_dataset(
            X_raw=X_raw,
            indices=indices,
            session=args.session,
            task=args.task,
            delay_s=delay_s,
            clean_margin_s=args.clean_margin_s,
            complete_cycles=shared_cycles,
            analysis_limit=analysis_limit,
        )
        sample_tables.append(build_sample_table(meta))
        metrics = run_pca_lda_loco(X, y, groups, args.pca_variance)
        row = {
            "session": args.session,
            "task": args.task,
            "method": args.method,
            "delay_s": int(delay_s) if float(delay_s).is_integer() else delay_s,
            "n_samples": int(len(X)),
            "n_cycles": int(len(np.unique(groups))),
            **metrics,
        }
        rows.append(row)
        print(
            f"{args.session} {args.task} {args.method} delay={delay_s:g}s "
            f"n={row['n_samples']} cycles={row['n_cycles']} "
            f"balanced_accuracy={row['balanced_accuracy']:.3f}",
            flush=True,
        )

    result_df = pd.DataFrame(
        rows,
        columns=[
            "session",
            "task",
            "method",
            "delay_s",
            "n_samples",
            "n_cycles",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ],
    )
    csv_path = out_dir / "813_binary_delay_scan.csv"
    fig_path = out_dir / "813_binary_delay_scan.png"
    samples_path = out_dir / "813_binary_delay_scan_samples.csv"
    result_df.to_csv(csv_path, index=False)
    pd.concat(sample_tables, ignore_index=True).to_csv(samples_path, index=False)
    plot_balanced_accuracy(result_df, fig_path)
    print(f"Saved {csv_path}")
    print(f"Saved {samples_path}")
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()
