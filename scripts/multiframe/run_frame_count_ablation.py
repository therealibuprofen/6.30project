#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import (
    TASK_CLASS_NAMES,
    csv_json,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.evaluation import CHANCE_LEVEL, metrics_with_flags
from ultrasound_decoding.multiframe.models import MODEL_DISPLAY_NAMES
from ultrasound_decoding.multiframe.plotting import METHOD_COLORS, _setup_matplotlib, save_png_pdf
from ultrasound_decoding.multiframe.training import DeepTrainingConfig, train_sequence_fold


WINDOWS: list[tuple[str, tuple[int, ...]]] = [
    ("p0", (0,)),
    ("p1", (1,)),
    ("p2", (2,)),
    ("p3", (3,)),
    ("p0-1", (0, 1)),
    ("p1-2", (1, 2)),
    ("p2-3", (2, 3)),
    ("p0-1-2", (0, 1, 2)),
    ("p1-2-3", (1, 2, 3)),
    ("p0-1-2-3", (0, 1, 2, 3)),
]
DEFAULT_SESSIONS = ["708", "709", "710"]
DEFAULT_METHODS = ["cnn2d_meanpool", "cnn2d_lstm", "cnn2d_temporal1d", "fcnn_meanpool", "fcnn_lstm"]
DEFAULT_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frame-count/window ablations for multiframe binary decoding.")
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "results" / "runs" / "multiframe" / "frame_count_ablation_binary_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def valid_fold(y_train: np.ndarray, y_test: np.ndarray) -> bool:
    return len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2


def plot_frame_count(aggregate: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    if aggregate.empty:
        return []
    methods = [method for method in DEFAULT_METHODS if method in set(aggregate["method"])]
    sessions = sorted(aggregate["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    fig, axes = plt.subplots(len(methods), 1, figsize=(8, max(2.4, 2.0 * len(methods))), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(methods))
    for ax, method in zip(axes, methods):
        subset = aggregate[aggregate["method"] == method]
        for session in sessions:
            line = subset[subset["session"].astype(str) == session].sort_values("frame_count")
            ax.errorbar(
                line["frame_count"].astype(int),
                line["balanced_accuracy_mean"].astype(float),
                yerr=[
                    line["balanced_accuracy_mean"].astype(float) - line["balanced_accuracy_min"].astype(float),
                    line["balanced_accuracy_max"].astype(float) - line["balanced_accuracy_mean"].astype(float),
                ],
                marker="o",
                linewidth=1.2,
                capsize=2,
                label=session,
            )
        ax.axhline(CHANCE_LEVEL, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylabel(MODEL_DISPLAY_NAMES.get(method, method), rotation=0, ha="right", va="center")
        ax.grid(color="#DDDDDD", linewidth=0.6)
    axes[-1].set_xticks([1, 2, 3, 4])
    axes[-1].set_xlabel("Frame count")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(title="Session", frameon=False, ncol=min(len(sessions), 3), loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.suptitle("Frame-count ablation: binary multiframe decoding", y=1.01)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, "frame_count_ablation")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} exists and is not empty; pass --overwrite to rerun")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    config = DeepTrainingConfig(
        optimizer="adamw",
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        batch_size=int(min(args.batch_size, 16)),
        max_epochs=int(args.max_epochs),
    )
    classes = np.asarray(sorted(TASK_CLASS_NAMES["binary"]), dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    master_rows: list[dict[str, Any]] = []

    for session in [str(value) for value in args.sessions]:
        data = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=data_dir)
        splits = grouped_cv_splits(data.groups, max_folds=args.max_folds)
        for window_name, positions in WINDOWS:
            X_window = data.X[:, positions, :, :]
            for method in args.methods:
                for seed in [int(value) for value in args.seeds]:
                    y_true_all: list[int] = []
                    y_pred_all: list[int] = []
                    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                        y_train = data.y[train_idx]
                        y_test = data.y[test_idx]
                        if not valid_fold(y_train, y_test):
                            raise AssertionError(f"{session} binary fold {fold_i} has fewer than two classes")
                        train_cycles = cycle_text(data.groups[train_idx])
                        test_cycles = cycle_text(data.groups[test_idx])
                        result = train_sequence_fold(
                            method,
                            X_window[train_idx],
                            y_train,
                            X_window[test_idx],
                            classes,
                            session=session,
                            task="binary",
                            fold=fold_i,
                            seed=seed,
                            train_cycles=train_cycles,
                            test_cycles=test_cycles,
                            config=config,
                            device=args.device,
                        )
                        metrics = metrics_with_flags(y_test, result.predictions)
                        fold_rows.append(
                            {
                                "session": session,
                                "task": "binary",
                                "method": method,
                                "method_display": MODEL_DISPLAY_NAMES.get(method, method),
                                "seed": int(seed),
                                "fold": int(fold_i),
                                "window": window_name,
                                "frame_positions": csv_json(list(positions)),
                                "frame_count": int(len(positions)),
                                "train_cycles": train_cycles,
                                "test_cycles": test_cycles,
                                "n_train_blocks": int(len(train_idx)),
                                "n_test_blocks": int(len(test_idx)),
                                "model_parameters": int(result.model_parameters),
                                "training_loss": float(result.final_training_loss),
                                "final_trained_epochs": int(result.final_trained_epochs),
                                "device": result.device,
                                **metrics,
                            }
                        )
                        y_true_all.extend(y_test.astype(int).tolist())
                        y_pred_all.extend(np.asarray(result.predictions).astype(int).tolist())
                    master_metrics = metrics_with_flags(np.asarray(y_true_all, dtype=np.int64), np.asarray(y_pred_all, dtype=np.int64))
                    master_rows.append(
                        {
                            "session": session,
                            "task": "binary",
                            "method": method,
                            "method_display": MODEL_DISPLAY_NAMES.get(method, method),
                            "seed": int(seed),
                            "window": window_name,
                            "frame_positions": csv_json(list(positions)),
                            "frame_count": int(len(positions)),
                            "n_test_predictions": int(len(y_true_all)),
                            "chance_level": CHANCE_LEVEL,
                            **master_metrics,
                        }
                    )

    frame_window_results = pd.DataFrame(master_rows)
    fold_results = pd.DataFrame(fold_rows)
    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in frame_window_results.groupby(["session", "task", "method", "frame_count"], sort=True):
        session, task, method, frame_count = keys
        values = group["balanced_accuracy"].astype(float)
        aggregate_rows.append(
            {
                "session": str(session),
                "task": task,
                "method": method,
                "method_display": MODEL_DISPLAY_NAMES.get(method, method),
                "frame_count": int(frame_count),
                "n_windows": int(group["window"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "windows": csv_json(sorted(group["window"].astype(str).unique().tolist())),
                "balanced_accuracy_mean": float(values.mean()),
                "balanced_accuracy_min": float(values.min()),
                "balanced_accuracy_max": float(values.max()),
                "balanced_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "chance_level": CHANCE_LEVEL,
            }
        )
    frame_count_aggregate = pd.DataFrame(aggregate_rows)
    frame_window_results.to_csv(args.output_dir / "frame_window_results.csv", index=False)
    fold_results.to_csv(args.output_dir / "frame_window_fold_results.csv", index=False)
    frame_count_aggregate.to_csv(args.output_dir / "frame_count_aggregate.csv", index=False)
    plot_paths = plot_frame_count(frame_count_aggregate, args.output_dir)
    pd.DataFrame({"path": [str(path) for path in plot_paths]}).to_csv(args.output_dir / "frame_count_plot_manifest.csv", index=False)
    print(f"[frame count ablation] wrote {args.output_dir}")


if __name__ == "__main__":
    main()
