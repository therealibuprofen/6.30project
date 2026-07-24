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
RESULTS_DIR = PROJECT_DIR / "results" / "model_batch_test"
METHODS = ["pca_lda", "cpca_lda", "cnn", "fcnn"]
TASKS = ["binary", "stimulus_type"]
TORCH_SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pca_lda/cpca_lda/cnn/fcnn on all selected sessions and tasks "
            "through the single-session cycle-wise CV decoder."
        )
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
        default=TASKS,
        choices=TASKS,
        help="Tasks to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=TORCH_SEEDS,
        help="Seeds passed to torch methods; linear methods run once in the single-session script.",
    )
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=40)
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
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument(
        "--output-root",
        default=str(RESULTS_DIR),
        help="Root directory for batch outputs.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional child directory name under --output-root for this batch run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files in the resolved batch output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the single-session commands without running them.",
    )
    return parser.parse_args()


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "run"


def timestamp_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def discover_sessions() -> list[str]:
    data_dir = PROJECT_DIR / "data"
    sessions = [path.name for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(sessions, key=int)


def resolve_output_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if args.run_id is not None:
        resolved = root / safe_run_id(args.run_id)
        if resolved.exists() and any(resolved.iterdir()) and not args.overwrite and not args.dry_run:
            raise FileExistsError(
                f"Batch output directory already contains files: {resolved}. "
                "Use a different --run-id or pass --overwrite."
            )
        return resolved

    if root.exists() and any(root.iterdir()) and not args.overwrite and not args.dry_run:
        return root / timestamp_run_id()
    return root


def build_command(args: argparse.Namespace, output_root: Path, session: str, task: str) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_DIR / "scripts" / "run_single_session_decoding.py"),
        "--session",
        session,
        "--task",
        task,
        "--methods",
        *METHODS,
        "--seeds",
        *(str(seed) for seed in args.seeds),
        "--max-folds",
        str(args.max_folds),
        "--max-epochs",
        str(args.max_epochs),
        "--clean-margin-s",
        str(args.clean_margin_s),
        "--pca-variance",
        str(args.pca_variance),
        "--device",
        args.device,
        "--output-base",
        str(output_root),
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
    return cmd


def run_one(args: argparse.Namespace, output_root: Path, session: str, task: str) -> dict[str, object]:
    cmd = build_command(args, output_root, session, task)
    command_text = " ".join(cmd)
    if args.dry_run:
        print(command_text)
        return {"session": session, "task": task, "command": cmd, "status": "dry_run"}

    print(f"Running session={session} task={task} methods={','.join(METHODS)} seeds={args.seeds}")
    completed = subprocess.run(cmd, cwd=PROJECT_DIR, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    print(f"Finished session={session} task={task}")
    return {"session": session, "task": task, "command": cmd, "status": "completed"}


def load_rows(output_root: Path, sessions: list[str], tasks: list[str], filename_suffix: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    subdir = "metrics" if filename_suffix != "predictions" else "predictions"
    for session in sessions:
        for task in tasks:
            path = output_root / subdir / f"{session}_{task}_{filename_suffix}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df.insert(0, "task", task)
            df.insert(0, "session", session)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_config(
    args: argparse.Namespace,
    output_root: Path,
    sessions: list[str],
    commands: list[dict[str, object]],
) -> None:
    config = {
        "sessions": sessions,
        "tasks": args.tasks,
        "methods": METHODS,
        "torch_seeds": args.seeds,
        "max_folds": args.max_folds,
        "cv_group": "cycle",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "clean_margin_s": args.clean_margin_s,
        "analysis_limit": args.analysis_limit,
        "pca_variance": args.pca_variance,
        "single_session_entrypoint": str(PROJECT_DIR / "scripts" / "run_single_session_decoding.py"),
        "output_root": str(output_root),
        "commands": commands,
    }
    (output_root / "batch_test_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    sessions = args.sessions if args.sessions is not None else discover_sessions()
    output_root = resolve_output_root(args)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)

    commands = []
    for session in sessions:
        for task in args.tasks:
            commands.append(run_one(args, output_root, session, task))

    if args.dry_run:
        return

    overall = load_rows(output_root, sessions, args.tasks, "overall_metrics")
    fold_metrics = load_rows(output_root, sessions, args.tasks, "fold_metrics")
    predictions = load_rows(output_root, sessions, args.tasks, "predictions")
    overall.to_csv(output_root / "summary.csv", index=False)
    fold_metrics.to_csv(output_root / "fold_metrics_all.csv", index=False)
    predictions.to_csv(output_root / "predictions_all.csv", index=False)
    write_config(args, output_root, sessions, commands)
    print(f"Saved batch summary to {output_root / 'summary.csv'}")
    print(f"Saved batch fold metrics to {output_root / 'fold_metrics_all.csv'}")
    print(f"Saved batch predictions to {output_root / 'predictions_all.csv'}")


if __name__ == "__main__":
    main()
