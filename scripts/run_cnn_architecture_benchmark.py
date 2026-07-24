#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results" / "cnn_architecture_benchmark"
CNN_MODELS = ["fcnn", "fcnn_paper_32", "fus_lite_cnn"]
BASELINE_METHODS = ["pca_lda", "cpca_lda", "cnn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark lightweight single-frame 2D CNN architectures with cycle-wise CV."
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=None,
        help="Session folders under data/. Defaults to every numeric data directory.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["binary", "stimulus_type"],
        choices=["binary", "stimulus_type"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=CNN_MODELS,
        choices=CNN_MODELS,
        help="New CNN architectures to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument(
        "--analysis-limit",
        default=None,
        help="Optional inclusive frame range like 1:180, or 'default'.",
    )
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_DIR),
        help="Root directory for this benchmark. Defaults outside reports/decoding.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional child directory name under --output-root for this benchmark run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in the resolved benchmark output directory.",
    )
    parser.add_argument(
        "--include-existing-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include already-computed pca_lda/cpca_lda/cnn rows from reports/decoding when present.",
    )
    return parser.parse_args()


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "run"


def timestamp_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def resolve_output_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if args.run_id is not None:
        resolved = root / safe_run_id(args.run_id)
        if resolved.exists() and any(resolved.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Benchmark output directory already contains files: {resolved}. "
                "Use a different --run-id or pass --overwrite."
            )
        return resolved

    if root.exists() and any(root.iterdir()) and not args.overwrite:
        return root / timestamp_run_id()
    return root


def discover_sessions() -> list[str]:
    data_dir = PROJECT_DIR / "data"
    sessions = [path.name for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(sessions, key=int)


def run_one(args: argparse.Namespace, session: str, task: str) -> None:
    output_base = Path(args.output_root)
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "run_single_session_decoding.py"),
        "--session",
        session,
        "--task",
        task,
        "--methods",
        *args.models,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--max-folds",
        str(args.max_folds),
        "--max-epochs",
        str(args.max_epochs),
        "--clean-margin-s",
        str(args.clean_margin_s),
        "--device",
        args.device,
        "--output-base",
        str(output_base),
    ]
    if args.patience is not None:
        cmd.extend(["--patience", str(args.patience)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.learning_rate is not None:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    if args.weight_decay is not None:
        cmd.extend(["--weight-decay", str(args.weight_decay)])
    if args.analysis_limit is not None:
        cmd.extend(["--analysis-limit", args.analysis_limit])
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"Running {session} {task}: {', '.join(args.models)} seeds={args.seeds}")
    completed = subprocess.run(cmd, cwd=PROJECT_DIR, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    print(completed.stdout)


def load_overall_rows(root: Path, sessions: list[str], tasks: list[str], source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics_dir = root / "metrics"
    for session in sessions:
        for task in tasks:
            path = metrics_dir / f"{session}_{task}_overall_metrics.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                item = row.to_dict()
                item.update(
                    {
                        "session": session,
                        "task": task,
                        "source": source,
                        "seed": str(int(item["seed"]))
                        if "seed" in item and pd.notna(item["seed"])
                        else "",
                    }
                )
                rows.append(item)
    return rows


def collect_summary(args: argparse.Namespace, sessions: list[str]) -> pd.DataFrame:
    rows = load_overall_rows(Path(args.output_root), sessions, args.tasks, "cnn_architecture_benchmark")
    if args.include_existing_baselines:
        baseline_rows = load_overall_rows(
            PROJECT_DIR / "reports" / "decoding",
            sessions,
            args.tasks,
            "existing_reports",
        )
        rows.extend(row for row in baseline_rows if row.get("method") in BASELINE_METHODS)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["task", "session", "method", "seed", "source"] if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    sessions = args.sessions if args.sessions is not None else discover_sessions()
    output_root = resolve_output_root(args)
    args.output_root = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for session in sessions:
        for task in args.tasks:
            run_one(args, session, task)

    summary = collect_summary(args, sessions)
    summary_path = output_root / "summary.csv"
    summary.to_csv(summary_path, index=False)
    config_path = output_root / "benchmark_config.json"
    config_path.write_text(
        json.dumps(
            {
                "sessions": sessions,
                "tasks": args.tasks,
                "models": args.models,
                "seeds": args.seeds,
                "max_folds": args.max_folds,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "device": args.device,
                "clean_margin_s": args.clean_margin_s,
                "analysis_limit": args.analysis_limit,
                "include_existing_baselines": args.include_existing_baselines,
                "summary": str(summary_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Saved benchmark summary to {summary_path}")


if __name__ == "__main__":
    main()
