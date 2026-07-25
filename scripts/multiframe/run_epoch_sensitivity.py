#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.multiframe.evaluation import overfitting_audit_tables, seed_mean_summary
from ultrasound_decoding.multiframe.models import MODEL_DISPLAY_NAMES
from ultrasound_decoding.multiframe.plotting import METHOD_COLORS, _setup_matplotlib, save_png_pdf


DEFAULT_SESSIONS = ["710", "807", "813", "817", "822"]
DEFAULT_METHODS = ["cnn2d_lstm", "cnn2d_temporal1d", "fcnn_lstm"]
DEFAULT_EPOCHS = [10, 20, 40]
DEFAULT_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent epoch-sensitivity experiments for multiframe binary decoding.")
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--epochs", nargs="+", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--output-root", type=Path, default=PROJECT_DIR / "results" / "runs" / "multiframe" / "epoch_sensitivity_binary_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only aggregate existing epoch subruns.")
    return parser.parse_args()


def run_epoch(args: argparse.Namespace, epochs: int) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "multiframe" / "run_multiframe_benchmark.py"),
        "--stage",
        "benchmark",
        "--tasks",
        "binary",
        "--sessions",
        *[str(value) for value in args.sessions],
        "--methods",
        *args.methods,
        "--seeds",
        *[str(value) for value in args.seeds],
        "--max-epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--learning-rate",
        str(float(args.learning_rate)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--device",
        args.device,
        "--output-root",
        str(args.output_root),
        "--run-name",
        f"epoch_{int(epochs)}",
        "--reuse-compatible-results",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def read_epoch_csv(args: argparse.Namespace, epochs: int, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted((args.output_root / f"epoch_{int(epochs)}").glob(f"session_*/{filename}")):
        if path.exists() and path.stat().st_size > 0:
            df = pd.read_csv(path)
            df["epochs"] = int(epochs)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_epoch_sensitivity(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    if summary.empty:
        return []
    methods = [method for method in DEFAULT_METHODS if method in set(summary["method"])]
    sessions = sorted(summary["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    fig, axes = plt.subplots(len(methods), 1, figsize=(8, max(2.4, 2.0 * len(methods))), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(len(methods))
    for ax, method in zip(axes, methods):
        subset = summary[summary["method"] == method]
        for session in sessions:
            line = subset[subset["session"].astype(str) == session].sort_values("epochs")
            ax.errorbar(
                line["epochs"].astype(int),
                line["balanced_accuracy_mean"].astype(float),
                yerr=line["balanced_accuracy_std"].astype(float),
                marker="o",
                linewidth=1.2,
                capsize=2,
                label=session,
            )
        ax.axhline(0.5, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_ylabel(MODEL_DISPLAY_NAMES.get(method, method), rotation=0, ha="right", va="center")
        ax.grid(color="#DDDDDD", linewidth=0.6)
    axes[-1].set_xlabel("Max epochs")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].legend(title="Session", frameon=False, ncol=min(len(sessions), 5), loc="upper center", bbox_to_anchor=(0.5, 1.35))
    fig.suptitle("Epoch sensitivity: binary multiframe decoding", y=1.01)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, "epoch_sensitivity")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_run:
        for epochs in args.epochs:
            run_epoch(args, int(epochs))

    master = pd.concat([read_epoch_csv(args, epochs, "master_summary.csv") for epochs in args.epochs], ignore_index=True)
    fold_summary = pd.concat([read_epoch_csv(args, epochs, "fold_summary.csv") for epochs in args.epochs], ignore_index=True)
    training_history = pd.concat([read_epoch_csv(args, epochs, "training_history.csv") for epochs in args.epochs], ignore_index=True)
    seed_frames = []
    if not master.empty:
        for epochs in sorted(master["epochs"].astype(int).unique().tolist()):
            subset = master[master["epochs"].astype(int) == int(epochs)]
            epoch_summary = seed_mean_summary(subset)
            epoch_summary["epochs"] = int(epochs)
            seed_frames.append(epoch_summary)
    seed_summary = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    overfitting_audit, _overfitting_method = overfitting_audit_tables(fold_summary, training_history)
    args.output_root.joinpath("epoch_sensitivity_summary.csv").write_text(master.to_csv(index=False), encoding="utf-8")
    args.output_root.joinpath("epoch_sensitivity_seed_summary.csv").write_text(seed_summary.to_csv(index=False), encoding="utf-8")
    args.output_root.joinpath("epoch_sensitivity_train_test_gap.csv").write_text(overfitting_audit.to_csv(index=False), encoding="utf-8")
    plot_paths = plot_epoch_sensitivity(seed_summary, args.output_root)
    pd.DataFrame({"path": [str(path) for path in plot_paths]}).to_csv(args.output_root / "epoch_sensitivity_plot_manifest.csv", index=False)
    print(f"[epoch sensitivity] wrote {args.output_root}")


if __name__ == "__main__":
    main()
