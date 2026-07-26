#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.multiframe.evaluation import seed_mean_summary
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
    parser.add_argument(
        "--repair-train-test-gap-only",
        action="store_true",
        help="Do not train and do not rewrite epoch_sensitivity_summary.csv or epoch_sensitivity_seed_summary.csv; only rebuild fixed train-test gap outputs.",
    )
    parser.add_argument("--expected-folds", type=int, default=10)
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


def read_epoch_configs(args: argparse.Namespace, epochs: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for session in [str(value) for value in args.sessions]:
        path = args.output_root / f"epoch_{int(epochs)}" / f"session_{session}" / "config.json"
        row: dict[str, Any] = {
            "session": session,
            "task": "binary",
            "epochs": int(epochs),
            "config_path": str(path),
            "config_exists": path.exists(),
            "config_max_epochs": np.nan,
            "config_matches_epochs": False,
        }
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            config_max_epochs = payload.get("deep_config", {}).get("max_epochs")
            row["task"] = payload.get("task", "binary")
            row["config_max_epochs"] = int(config_max_epochs) if config_max_epochs is not None else np.nan
            row["config_matches_epochs"] = config_max_epochs == int(epochs)
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_train_test_gap_table(fold_summary: pd.DataFrame, training_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session",
        "task",
        "method",
        "seed",
        "fold",
        "epochs",
        "final_epoch",
        "final_train_accuracy",
        "final_train_loss",
        "test_balanced_accuracy",
        "test_accuracy",
        "generalization_gap",
        "n_train_blocks",
        "n_test_blocks",
    ]
    if fold_summary.empty or training_history.empty:
        return pd.DataFrame(columns=columns)

    key_cols = ["session", "task", "method", "seed", "fold", "epochs"]
    fold_df = fold_summary.copy()
    hist = training_history.copy()
    for frame in [fold_df, hist]:
        frame["session"] = frame["session"].astype(str)
        frame["task"] = frame["task"].astype(str)
        frame["method"] = frame["method"].astype(str)
        for column in ["seed", "fold", "epochs"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    hist["epoch"] = pd.to_numeric(hist["epoch"], errors="coerce")
    hist = hist.dropna(subset=["epoch", "seed", "fold", "epochs"]).sort_values(key_cols + ["epoch"])
    final_history = hist.groupby(key_cols, sort=True, dropna=False).tail(1).copy()
    merged = final_history.merge(
        fold_df[
            key_cols
            + [
                "n_train_blocks",
                "n_test_blocks",
                "accuracy",
                "balanced_accuracy",
            ]
        ],
        on=key_cols,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(
        {
            "session": merged["session"].astype(str),
            "task": merged["task"],
            "method": merged["method"],
            "seed": merged["seed"].astype(int),
            "fold": merged["fold"].astype(int),
            "epochs": merged["epochs"].astype(int),
            "final_epoch": pd.to_numeric(merged["epoch"], errors="coerce").astype(int),
            "final_train_accuracy": pd.to_numeric(merged["train_accuracy"], errors="coerce"),
            "final_train_loss": pd.to_numeric(merged["train_loss"], errors="coerce"),
            "test_balanced_accuracy": pd.to_numeric(merged["balanced_accuracy"], errors="coerce"),
            "test_accuracy": pd.to_numeric(merged["accuracy"], errors="coerce"),
            "n_train_blocks": pd.to_numeric(merged["n_train_blocks"], errors="coerce").astype(int),
            "n_test_blocks": pd.to_numeric(merged["n_test_blocks"], errors="coerce").astype(int),
        }
    )
    out["generalization_gap"] = out["final_train_accuracy"] - out["test_balanced_accuracy"]
    return out[columns].sort_values(["session", "method", "epochs", "seed", "fold"]).reset_index(drop=True)


def fixed_gap_summary(gap: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session",
        "task",
        "method",
        "epochs",
        "mean_train_accuracy",
        "std_train_accuracy",
        "mean_test_ba",
        "std_test_ba",
        "mean_generalization_gap",
        "std_generalization_gap",
        "fraction_train_accuracy_above_0_95",
        "fraction_test_ba_below_0_5",
        "n_seeds",
        "n_folds",
    ]
    if gap.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (session, task, method, epochs), group in gap.groupby(["session", "task", "method", "epochs"], sort=True):
        rows.append(
            {
                "session": str(session),
                "task": task,
                "method": method,
                "epochs": int(epochs),
                "mean_train_accuracy": float(group["final_train_accuracy"].astype(float).mean()),
                "std_train_accuracy": float(group["final_train_accuracy"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_test_ba": float(group["test_balanced_accuracy"].astype(float).mean()),
                "std_test_ba": float(group["test_balanced_accuracy"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_generalization_gap": float(group["generalization_gap"].astype(float).mean()),
                "std_generalization_gap": float(group["generalization_gap"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
                "fraction_train_accuracy_above_0_95": float((group["final_train_accuracy"].astype(float) > 0.95).mean()),
                "fraction_test_ba_below_0_5": float((group["test_balanced_accuracy"].astype(float) < 0.5).mean()),
                "n_seeds": int(group["seed"].nunique()),
                "n_folds": int(group["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def completeness_report(
    *,
    gap: pd.DataFrame,
    fold_summary: pd.DataFrame,
    training_history: pd.DataFrame,
    configs: pd.DataFrame,
    sessions: list[str],
    methods: list[str],
    epochs_values: list[int],
    seeds: list[int],
    expected_folds: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_cols = ["session", "method", "epochs", "seed", "fold"]
    duplicate_count = int(gap.duplicated(key_cols).sum()) if not gap.empty else 0
    for frame in [fold_summary, training_history, configs, gap]:
        if not frame.empty and "session" in frame:
            frame["session"] = frame["session"].astype(str)
    for session in [str(value) for value in sessions]:
        for method in methods:
            for epochs in [int(value) for value in epochs_values]:
                config_subset = configs[
                    (configs["session"].astype(str) == session)
                    & (configs["epochs"].astype(int) == int(epochs))
                ] if not configs.empty else pd.DataFrame()
                config_max_epochs = (
                    sorted(config_subset["config_max_epochs"].dropna().astype(int).unique().tolist())
                    if not config_subset.empty and "config_max_epochs" in config_subset
                    else []
                )
                config_matches_epochs = (
                    bool(config_subset["config_matches_epochs"].astype(bool).all())
                    if not config_subset.empty and "config_matches_epochs" in config_subset
                    else False
                )
                for seed in [int(value) for value in seeds]:
                    subset = gap[
                        (gap["session"].astype(str) == session)
                        & (gap["method"] == method)
                        & (gap["epochs"].astype(int) == int(epochs))
                        & (gap["seed"].astype(int) == int(seed))
                    ] if not gap.empty else pd.DataFrame()
                    fold_subset = fold_summary[
                        (fold_summary["session"].astype(str) == session)
                        & (fold_summary["method"] == method)
                        & (fold_summary["epochs"].astype(int) == int(epochs))
                        & (fold_summary["seed"].astype(int) == int(seed))
                    ] if not fold_summary.empty else pd.DataFrame()
                    history_subset = training_history[
                        (training_history["session"].astype(str) == session)
                        & (training_history["method"] == method)
                        & (training_history["epochs"].astype(int) == int(epochs))
                        & (training_history["seed"].astype(int) == int(seed))
                    ] if not training_history.empty else pd.DataFrame()
                    present_folds = sorted(subset["fold"].astype(int).unique().tolist()) if not subset.empty else []
                    expected_fold_values = list(range(1, int(expected_folds) + 1))
                    missing_folds = [fold for fold in expected_fold_values if fold not in present_folds]
                    final_epoch_values = sorted(subset["final_epoch"].astype(int).unique().tolist()) if not subset.empty else []
                    rows.append(
                        {
                            "session": session,
                            "task": "binary",
                            "method": method,
                            "epochs": int(epochs),
                            "seed": int(seed),
                            "expected_folds": int(expected_folds),
                            "n_gap_rows": int(len(subset)),
                            "n_fold_summary_rows": int(len(fold_subset)),
                            "n_training_history_folds": int(history_subset["fold"].nunique()) if not history_subset.empty else 0,
                            "present_folds": json.dumps(present_folds),
                            "missing_folds": json.dumps(missing_folds),
                            "final_epoch_values": json.dumps(final_epoch_values),
                            "config_max_epochs": json.dumps(config_max_epochs),
                            "config_matches_epochs": config_matches_epochs,
                            "duplicate_key_count_total": duplicate_count,
                            "complete": bool(
                                len(subset) == int(expected_folds)
                                and not missing_folds
                                and final_epoch_values == [int(epochs)]
                                and config_matches_epochs
                            ),
                        }
                    )
    report = pd.DataFrame(rows)
    expected_total = int(len(sessions) * len(methods) * len(epochs_values) * len(seeds) * expected_folds)
    report["expected_total_fold_rows"] = expected_total
    report["observed_total_fold_rows"] = int(len(gap))
    return report


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
    if args.repair_train_test_gap_only:
        args.skip_run = True
    if not args.skip_run:
        for epochs in args.epochs:
            run_epoch(args, int(epochs))

    master = pd.concat([read_epoch_csv(args, epochs, "master_summary.csv") for epochs in args.epochs], ignore_index=True)
    fold_summary = pd.concat([read_epoch_csv(args, epochs, "fold_summary.csv") for epochs in args.epochs], ignore_index=True)
    training_history = pd.concat([read_epoch_csv(args, epochs, "training_history.csv") for epochs in args.epochs], ignore_index=True)
    configs = pd.concat([read_epoch_configs(args, epochs) for epochs in args.epochs], ignore_index=True)
    fixed_gap = fixed_train_test_gap_table(fold_summary, training_history)
    gap_summary = fixed_gap_summary(fixed_gap)
    complete = completeness_report(
        gap=fixed_gap,
        fold_summary=fold_summary,
        training_history=training_history,
        configs=configs,
        sessions=[str(value) for value in args.sessions],
        methods=list(args.methods),
        epochs_values=[int(value) for value in args.epochs],
        seeds=[int(value) for value in args.seeds],
        expected_folds=int(args.expected_folds),
    )
    fixed_gap.to_csv(args.output_root / "epoch_sensitivity_train_test_gap_fixed.csv", index=False)
    gap_summary.to_csv(args.output_root / "epoch_sensitivity_gap_summary.csv", index=False)
    complete.to_csv(args.output_root / "epoch_sensitivity_completeness_report.csv", index=False)
    duplicate_keys = int(
        fixed_gap.duplicated(["session", "method", "epochs", "seed", "fold"]).sum()
    ) if not fixed_gap.empty else 0
    expected_total = int(len(args.sessions) * len(args.methods) * len(args.epochs) * len(args.seeds) * int(args.expected_folds))
    incomplete_rows = int((~complete["complete"].astype(bool)).sum()) if not complete.empty else expected_total
    print(
        "[epoch sensitivity repair] "
        f"fixed_gap_rows={len(fixed_gap)} expected_rows={expected_total} "
        f"duplicate_keys={duplicate_keys} incomplete_combinations={incomplete_rows}"
    )
    if args.repair_train_test_gap_only:
        print(f"[epoch sensitivity repair] wrote fixed gap outputs under {args.output_root}")
        return

    seed_frames = []
    if not master.empty:
        for epochs in sorted(master["epochs"].astype(int).unique().tolist()):
            subset = master[master["epochs"].astype(int) == int(epochs)]
            epoch_summary = seed_mean_summary(subset)
            epoch_summary["epochs"] = int(epochs)
            seed_frames.append(epoch_summary)
    seed_summary = pd.concat(seed_frames, ignore_index=True) if seed_frames else pd.DataFrame()
    args.output_root.joinpath("epoch_sensitivity_summary.csv").write_text(master.to_csv(index=False), encoding="utf-8")
    args.output_root.joinpath("epoch_sensitivity_seed_summary.csv").write_text(seed_summary.to_csv(index=False), encoding="utf-8")
    args.output_root.joinpath("epoch_sensitivity_train_test_gap.csv").write_text(fixed_gap.to_csv(index=False), encoding="utf-8")
    plot_paths = plot_epoch_sensitivity(seed_summary, args.output_root)
    pd.DataFrame({"path": [str(path) for path in plot_paths]}).to_csv(args.output_root / "epoch_sensitivity_plot_manifest.csv", index=False)
    print(f"[epoch sensitivity] wrote {args.output_root}")


if __name__ == "__main__":
    main()
