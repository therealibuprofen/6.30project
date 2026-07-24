#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results" / "spatial_filter_ablation"
EXPERIMENTS = [
    ("no_filter", "none", 0),
    ("pillbox_r1", "pillbox", 1),
    ("pillbox_r2", "pillbox", 2),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-filter/radius-1/radius-2 spatial smoothing ablation."
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Session folders under data/. Defaults to every numeric data directory.",
    )
    parser.add_argument(
        "--task",
        default="stimulus_type",
        choices=["binary", "stimulus_type"],
        help="Decoding task to compare across spatial-filter settings.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca_lda"],
        choices=[
            "pca_lda",
            "cpca_lda",
            "cnn",
            "cnn_lstm",
            "fcnn",
            "fcnn_paper_32",
            "fus_lite_cnn",
        ],
        help="Decoder methods. The comparison plot uses the first method.",
    )
    parser.add_argument(
        "--analysis-limit",
        default="default",
        help=(
            "Inclusive frame range like 1:180. Defaults to 'default' so no_filter "
            "matches the established single-session baseline where available."
        ),
    )
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_DIR),
        help="Root directory for ablation outputs.",
    )
    return parser.parse_args()


def discover_sessions() -> list[str]:
    data_dir = PROJECT_DIR / "data"
    sessions = [path.name for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(sessions, key=int)


def run_one(args: argparse.Namespace, experiment: str, method: str, radius: int, session: str) -> None:
    output_base = Path(args.output_root) / experiment
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "run_single_session_decoding.py"),
        "--session",
        session,
        "--task",
        args.task,
        "--methods",
        *args.methods,
        "--clean-margin-s",
        str(args.clean_margin_s),
        "--window-size",
        str(args.window_size),
        "--max-folds",
        str(args.max_folds),
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--pca-variance",
        str(args.pca_variance),
        "--spatial-filter-method",
        method,
        "--spatial-filter-radius",
        str(radius),
        "--output-base",
        str(output_base),
    ]
    if args.analysis_limit is not None:
        cmd.extend(["--analysis-limit", args.analysis_limit])
    print(f"Running {experiment} session {session}...")
    completed = subprocess.run(cmd, cwd=PROJECT_DIR, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()


def collect_summary(args: argparse.Namespace, sessions: list[str]) -> pd.DataFrame:
    rows = []
    for experiment, method, radius in EXPERIMENTS:
        metrics_dir = Path(args.output_root) / experiment / "metrics"
        summary_dir = Path(args.output_root) / experiment / "summary"
        for session in sessions:
            stem = (
                f"{session}_{args.task}"
                if args.window_size == 1
                else f"{session}_{args.task}_window{args.window_size}"
            )
            overall_path = metrics_dir / f"{stem}_overall_metrics.csv"
            fold_path = metrics_dir / f"{stem}_fold_metrics.csv"
            summary_path = summary_dir / f"{stem}_summary.json"
            if not overall_path.exists():
                continue
            overall = pd.read_csv(overall_path)
            folds = pd.read_csv(fold_path) if fold_path.exists() else pd.DataFrame()
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
            for _, metric_row in overall.iterrows():
                method_name = str(metric_row["method"])
                method_folds = folds[folds["method"] == method_name] if not folds.empty else folds
                rows.append(
                    {
                        "experiment_name": experiment,
                        "session_id": session,
                        "method": method_name,
                        "spatial_filter_method": method,
                        "spatial_filter_radius": radius,
                        "accuracy": float(metric_row["accuracy"]),
                        "balanced_accuracy": float(metric_row["balanced_accuracy"]),
                        "n_train": float(method_folds["n_train"].mean()) if not method_folds.empty else np.nan,
                        "n_val": int(method_folds["n_test"].sum()) if not method_folds.empty else np.nan,
                        "n_test": int(metric_row["n_test_predictions"]),
                        "random_seed": int(summary.get("random_seed", args.seed)),
                    }
                )
    return pd.DataFrame(rows)


def plot_accuracy(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if df.empty:
        return
    try:
        mplconfig = PROJECT_DIR / ".cache" / "matplotlib"
        mplconfig.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped accuracy_comparison.png")
        return

    plot_method = args.methods[0]
    plot_df = df[df["method"] == plot_method].copy()
    order = [name for name, _, _ in EXPERIMENTS]
    x = np.arange(len(order), dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=160)
    for session, session_df in plot_df.groupby("session_id", sort=True):
        values = (
            session_df.set_index("experiment_name")
            .reindex(order)["accuracy"]
            .to_numpy(dtype=float)
        )
        ax.plot(x, values, color="#8A8A8A", linewidth=1.0, alpha=0.65)
        ax.scatter(x, values, s=28, color="#4C78A8", alpha=0.85, label=None)

    means = plot_df.groupby("experiment_name")["accuracy"].mean().reindex(order)
    ax.plot(x, means.to_numpy(dtype=float), color="#D62728", linewidth=2.4, marker="o", label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Decoding accuracy")
    ax.set_xlabel("Spatial filter setting")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path = Path(args.output_root) / "accuracy_comparison.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sessions = args.sessions if args.sessions else discover_sessions()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    (Path(args.output_root) / "ablation_config.json").write_text(
        json.dumps(
            {
                "sessions": sessions,
                "task": args.task,
                "methods": args.methods,
                "analysis_limit": args.analysis_limit,
                "seed": args.seed,
                "experiments": [
                    {"experiment_name": name, "spatial_filter_method": method, "spatial_filter_radius": radius}
                    for name, method, radius in EXPERIMENTS
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    for experiment, method, radius in EXPERIMENTS:
        for session in sessions:
            run_one(args, experiment, method, radius, session)

    summary = collect_summary(args, sessions)
    summary_path = Path(args.output_root) / "summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_accuracy(summary, args)

    print("Spatial filter ablation finished.")
    means = summary[summary["method"] == args.methods[0]].groupby("experiment_name")["accuracy"].mean()
    for experiment, _, _ in EXPERIMENTS:
        value = means.get(experiment, np.nan)
        print(f"{experiment} mean accuracy: {value:.4f}" if np.isfinite(value) else f"{experiment} mean accuracy: n/a")
    if not means.empty:
        print(f"Best setting: {str(means.idxmax())}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
