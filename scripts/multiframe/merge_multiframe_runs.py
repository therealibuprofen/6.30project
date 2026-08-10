#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_multiframe_benchmark import legacy_checkpoint_manifest, parameter_audit_rows
from ultrasound_decoding.multiframe.evaluation import (
    block_type_accuracy,
    completeness_report,
    method_summary_table,
    order_sensitivity_oof_summary,
    overfitting_audit_tables,
    seed_mean_summary,
)
from ultrasound_decoding.multiframe.models import LINEAR_METHODS, METHOD_USES_TEMPORAL_ORDER, MODEL_DISPLAY_NAMES
from ultrasound_decoding.multiframe.plotting import (
    METHOD_COLORS,
    METHOD_ORDER,
    _setup_matplotlib,
    plot_block_type_accuracy,
    plot_generalization_gap,
    plot_method_comparison,
    plot_parameter_count_vs_test_ba,
    save_png_pdf,
)


CSV_FILENAMES = [
    "master_summary.csv",
    "fold_summary.csv",
    "predictions.csv",
    "confusion_matrices.csv",
    "normalization_audit.csv",
    "training_history.csv",
    "linear_fit_audit.csv",
    "order_sensitivity.csv",
    "order_sensitivity_predictions.csv",
    "order_sensitivity_oof_summary.csv",
    "checkpoint_manifest.csv",
    "multiframe_completeness_report.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge legacy multiframe and FCNN multiframe runs after compatibility checks.")
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--additional-run", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def session_dirs(run: Path) -> dict[str, Path]:
    out = {}
    for path in sorted(run.glob("session_*")):
        if path.is_dir():
            out[path.name.replace("session_", "")] = path
    return out


def run_dir_diagnostic(run: Path) -> str:
    if not run.exists():
        return f"{run} does not exist."
    if not run.is_dir():
        return f"{run} exists but is not a directory."
    children = sorted(path.name for path in run.iterdir()) if run.is_dir() else []
    preview = ", ".join(children[:12]) if children else "(empty)"
    if len(children) > 12:
        preview += ", ..."
    nested = [child for child in sorted(run.iterdir()) if child.is_dir() and session_dirs(child)]
    nested_text = ""
    if nested:
        nested_text = "\nNested run directories with session_* were found:\n  - " + "\n  - ".join(str(path) for path in nested)
    return (
        f"{run} exists but has no direct session_* directories. Immediate entries: {preview}."
        f"{nested_text}"
    )


def resolve_run_dir(run: Path, label: str) -> Path:
    direct_sessions = session_dirs(run)
    if direct_sessions:
        return run

    nested = [
        child
        for child in sorted(run.iterdir())
        if run.exists() and run.is_dir() and child.is_dir() and session_dirs(child)
    ] if run.exists() and run.is_dir() else []
    if len(nested) == 1:
        resolved = nested[0]
        print(f"[merge] {label} has no direct session_* directories; using nested run directory {resolved}")
        return resolved

    hint = run_dir_diagnostic(run)
    raise ValueError(
        f"{label} is not a valid multiframe run directory.\n"
        f"{hint}\n\n"
        "A valid run directory must contain session directories such as session_708/config.json.\n"
        "To inspect the server output, run:\n"
        f"  find {run} -maxdepth 2 -type d | sort\n\n"
        "If the FCNN benchmark has not been run yet, create the additional run first, for example:\n"
        "  python scripts/multiframe/run_multiframe_benchmark.py \\\n"
        "    --stage benchmark --tasks binary \\\n"
        "    --sessions 708 709 710 807 813 817 822 \\\n"
        "    --methods fcnn_late_fusion fcnn_meanpool fcnn_lstm \\\n"
        "    --seeds 0 1 2 --max-epochs 40 --batch-size 16 \\\n"
        "    --learning-rate 1e-3 --weight-decay 1e-3 --device auto \\\n"
        "    --run-name block_clean4_binary_fcnn_v1"
    )


def read_session_csvs(run: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in sorted(run.glob(f"session_*/{filename}")):
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                print(f"[merge] skipping empty CSV with no columns: {path}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_mapping(mapping: Any, task: str) -> dict[str, str]:
    if not mapping:
        if task == "binary":
            return {"0": "no_stimulus", "1": "stimulus"}
        if task == "stimulus_type":
            return {"0": "dot", "1": "grating"}
    return {str(key): str(value) for key, value in dict(mapping).items()}


def split_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = ["session", "task", "fold", "train_cycles", "test_cycles", "n_train_blocks", "n_test_blocks"]
    out = df[columns].copy()
    for column in columns:
        out[column] = out[column].astype(str)
    return out.sort_values(columns).reset_index(drop=True)


def normalization_protocol(run: Path) -> str:
    df = read_session_csvs(run, "normalization_audit.csv")
    if df.empty:
        return "not_available"
    cols = [column for column in ["transform", "statistics_scope"] if column in df]
    return json.dumps({column: sorted(df[column].astype(str).unique().tolist()) for column in cols}, sort_keys=True)


def config_value(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def read_run_configs(run: Path) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for session, session_dir in session_dirs(run).items():
        config_path = session_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"missing config.json for {run}/session_{session}")
        configs[str(session)] = read_json(config_path)
    return configs


def coverage_from_master(master: pd.DataFrame, run: Path, label: str) -> set[tuple[str, str]]:
    if master.empty:
        raise ValueError(f"{label} has no master_summary.csv rows under {run}")
    required = {"session", "method"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"{label} master_summary.csv is missing columns: {sorted(missing)}")
    frame = master.copy()
    frame["session"] = frame["session"].astype(str)
    frame["method"] = frame["method"].astype(str)
    return set(zip(frame["session"], frame["method"]))


def duplicate_count(frame: pd.DataFrame, columns: list[str]) -> int:
    if frame.empty or not set(columns).issubset(frame.columns):
        return 0
    subset = frame[columns].copy()
    for column in columns:
        subset[column] = subset[column].fillna("__NA__").astype(str)
    return int(subset.duplicated().sum())


def validate_merged_tables(frames: dict[str, pd.DataFrame]) -> None:
    checks = {
        "master_summary.csv": ["session", "task", "method", "seed"],
        "fold_summary.csv": ["session", "task", "method", "seed", "fold"],
        "predictions.csv": ["session", "task", "method", "seed", "fold", "block_id"],
        "training_history.csv": ["session", "task", "method", "seed", "fold", "epoch"],
        "order_sensitivity.csv": ["session", "task", "method", "seed", "fold"],
        "order_sensitivity_predictions.csv": ["session", "task", "method", "seed", "fold", "block_id", "order_condition"],
        "order_sensitivity_oof_summary.csv": ["session", "task", "method", "seed", "order_condition"],
    }
    errors = []
    for filename, columns in checks.items():
        count = duplicate_count(frames.get(filename, pd.DataFrame()), columns)
        if count:
            errors.append(f"{filename}: {count} duplicate rows for key {columns}")
    if errors:
        raise ValueError("merged result tables contain duplicate keys:\n  - " + "\n  - ".join(errors))


def check_compatibility(base_run: Path, additional_run: Path) -> tuple[list[str], str, list[str], list[int]]:
    base_sessions = session_dirs(base_run)
    add_sessions = session_dirs(additional_run)
    base_configs = read_run_configs(base_run)
    add_configs = read_run_configs(additional_run)
    sessions = sorted(set(base_sessions) | set(add_sessions), key=lambda value: int(value))
    errors: list[str] = []
    task_values: set[str] = set()
    seeds_values: set[tuple[int, ...]] = set()
    base_master = read_session_csvs(base_run, "master_summary.csv")
    add_master = read_session_csvs(additional_run, "master_summary.csv")
    base_coverage = coverage_from_master(base_master, base_run, "base-run")
    add_coverage = coverage_from_master(add_master, additional_run, "additional-run")
    overlap = sorted(base_coverage & add_coverage, key=lambda item: (int(item[0]), item[1]))
    if overlap:
        preview = ", ".join(f"{session}/{method}" for session, method in overlap[:20])
        if len(overlap) > 20:
            preview += ", ..."
        errors.append(f"both runs contain results for the same session/method: {preview}")

    for session in sessions:
        configs = []
        if session in base_configs:
            configs.append(("base-run", base_configs[session], base_sessions[session]))
        if session in add_configs:
            configs.append(("additional-run", add_configs[session], add_sessions[session]))
        for _label, cfg, _session_dir in configs:
            task_values.add(str(cfg.get("task")))
            seeds_values.add(tuple(int(value) for value in cfg.get("seeds", [])))
        if len(configs) == 2:
            _base_label, base_cfg, base_dir = configs[0]
            _add_label, add_cfg, add_dir = configs[1]
            for key in ["task", "input_shape", "data_version", "cv_group", "max_folds"]:
                if base_cfg.get(key) != add_cfg.get(key):
                    errors.append(f"session {session}: {key} differs ({base_cfg.get(key)!r} vs {add_cfg.get(key)!r})")
            for key in ["deep_config.optimizer", "deep_config.lr", "deep_config.weight_decay", "deep_config.batch_size", "deep_config.max_epochs", "deep_config.loss"]:
                if config_value(base_cfg, key) != config_value(add_cfg, key):
                    errors.append(f"session {session}: {key} differs ({config_value(base_cfg, key)!r} vs {config_value(add_cfg, key)!r})")
            if normalize_mapping(base_cfg.get("class_mapping"), str(base_cfg.get("task"))) != normalize_mapping(
                add_cfg.get("class_mapping"),
                str(add_cfg.get("task")),
            ):
                errors.append(f"session {session}: class mapping differs")
            if not split_frame(base_dir / "split_manifest.csv").equals(split_frame(add_dir / "split_manifest.csv")):
                errors.append(f"session {session}: split_manifest differs")

    if len(task_values) != 1:
        errors.append(f"task values differ: {sorted(task_values)}")
    if len(seeds_values) != 1:
        errors.append(f"seed lists differ: {sorted(seeds_values)}")
    if normalization_protocol(base_run) != normalization_protocol(additional_run):
        errors.append("normalization protocol differs")
    if errors:
        raise ValueError("runs are not merge-compatible:\n  - " + "\n  - ".join(errors))
    task = next(iter(task_values))
    seeds = list(next(iter(seeds_values)))
    methods = {method for _session, method in (base_coverage | add_coverage)}
    print(
        "[merge] coverage union: "
        f"sessions={sessions}, methods={sorted(methods, key=lambda method: METHOD_ORDER.index(method) if method in METHOD_ORDER else 999)}"
    )
    return sorted(methods, key=lambda method: METHOD_ORDER.index(method) if method in METHOD_ORDER else 999), task, sessions, seeds


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_selected_methods(master: pd.DataFrame, task: str, out_dir: Path, stem: str, methods: list[str], title: str) -> list[Path]:
    plt = _setup_matplotlib()
    summary = seed_mean_summary(master)
    summary = summary[(summary["task"] == task) & (summary["method"].isin(methods))].copy()
    if summary.empty:
        return []
    summary["session"] = summary["session"].astype(str)
    sessions = sorted(summary["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    methods = [method for method in methods if method in set(summary["method"])]
    x = np.arange(len(sessions), dtype=float)
    width = min(0.18, 0.78 / max(len(methods), 1))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, method in enumerate(methods):
        subset = summary[summary["method"] == method].set_index("session")
        means = [float(subset.loc[session, "balanced_accuracy_mean"]) if session in subset.index else np.nan for session in sessions]
        stds = [float(subset.loc[session, "balanced_accuracy_std"]) if session in subset.index else 0.0 for session in sessions]
        ax.bar(
            x + (i - (len(methods) - 1) / 2.0) * width,
            means,
            width=width,
            yerr=stds,
            color=METHOD_COLORS.get(method, "#777777"),
            edgecolor="white",
            linewidth=0.4,
            capsize=2,
            label=MODEL_DISPLAY_NAMES.get(method, method),
        )
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0, label="Chance level")
    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.set_xlabel("Session")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, stem)
    plt.close(fig)
    return paths


def plot_fcnn_order_sensitivity(order_oof: pd.DataFrame, task: str, out_dir: Path) -> list[Path]:
    plt = _setup_matplotlib()
    df = order_oof[(order_oof["task"] == task) & (order_oof["method"] == "fcnn_lstm")].copy() if not order_oof.empty else pd.DataFrame()
    if df.empty:
        return []
    grouped = df.groupby(["session", "order_condition"], sort=True)["balanced_accuracy"].mean().reset_index()
    grouped["session"] = grouped["session"].astype(str)
    sessions = sorted(grouped["session"].astype(str).unique().tolist(), key=lambda value: int(value))
    conditions = ["original", "reverse", "fixed_shuffle"]
    labels = {"original": "Original", "reverse": "Reverse", "fixed_shuffle": "Fixed shuffle"}
    colors = {"original": "#4C78A8", "reverse": "#F58518", "fixed_shuffle": "#E45756"}
    x = np.arange(len(sessions), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, condition in enumerate(conditions):
        subset = grouped[grouped["order_condition"] == condition].set_index("session")
        values = [float(subset.loc[session, "balanced_accuracy"]) if session in subset.index else np.nan for session in sessions]
        ax.bar(x + (i - 1) * width, values, width=width, color=colors[condition], edgecolor="white", linewidth=0.4, label=labels[condition])
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.set_xlabel("Session")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("FCNN-LSTM order sensitivity")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    paths = save_png_pdf(fig, out_dir, "order_sensitivity_fcnn_lstm")
    plt.close(fig)
    return paths


def write_analysis(path: Path, master: pd.DataFrame, overfitting_summary: pd.DataFrame, order_oof: pd.DataFrame) -> None:
    summary = seed_mean_summary(master)

    def delta(method_a: str, method_b: str) -> str:
        pivot = summary.pivot_table(index="session", columns="method", values="balanced_accuracy_mean", aggfunc="mean")
        if method_a not in pivot or method_b not in pivot:
            return "not available"
        values = (pivot[method_a] - pivot[method_b]).dropna()
        if values.empty:
            return "not available"
        return f"mean delta {float(values.mean()):.3f} across {len(values)} sessions"

    gap_note = "not available"
    if not overfitting_summary.empty:
        worst = overfitting_summary.sort_values("mean_generalization_gap", ascending=False).head(3)
        gap_note = "; ".join(
            f"{row.session}/{row.method}: {float(row.mean_generalization_gap):.3f}"
            for row in worst.itertuples(index=False)
        )
    order_note = "not available"
    if not order_oof.empty:
        subset = order_oof[order_oof["method"] == "fcnn_lstm"]
        if not subset.empty and "original_minus_reverse" in subset:
            order_note = f"FCNN-LSTM original-minus-reverse mean {float(subset['original_minus_reverse'].dropna().mean()):.3f}"

    lines = [
        "# Multiframe Binary Analysis",
        "",
        "This report is descriptive and does not claim statistical significance.",
        "",
        f"1. FCNN four-frame probability averaging vs single-frame/CNN late fusion: {delta('fcnn_late_fusion', 'single_frame_late_fusion')}.",
        f"2. FCNN mean-pool vs FCNN late fusion: {delta('fcnn_meanpool', 'fcnn_late_fusion')}.",
        f"3. FCNN-LSTM vs FCNN mean-pool: {delta('fcnn_lstm', 'fcnn_meanpool')}.",
        f"4. SmallCNN-LSTM vs FCNN-LSTM stability: compare seed std in `multiframe_all_models_seed_summary.csv`; mean BA delta is {delta('cnn2d_lstm', 'fcnn_lstm')}.",
        "5. The 3D FCNN bottleneck may limit temporal modeling capacity; this is an architectural diagnostic, not a causal conclusion.",
        "6. Strong and weak session layering should be read from the session-wise master table without hiding below-chance results.",
        f"7. Largest train-test diagnostic gaps: {gap_note}.",
        f"8. Explicit temporal methods vs order-invariant methods: CNN-LSTM/Temporal1D and FCNN-LSTM should be compared against their mean-pool counterparts; FCNN-LSTM delta is {delta('fcnn_lstm', 'fcnn_meanpool')}.",
        f"9. Order shuffling effects, including session 709 when present: {order_note}.",
        "10. Single-class predictions and abnormal folds are flagged in the fold, seed, completeness, and OOF order sensitivity tables.",
        "",
        "Legacy CNN results may not include `order_sensitivity_predictions.csv`; no old per-block order predictions are reconstructed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_run = resolve_run_dir(args.base_run.resolve(), "base-run")
    additional_run = resolve_run_dir(args.additional_run.resolve(), "additional-run")
    output_run = args.output_run.resolve()
    methods, task, sessions, seeds = check_compatibility(base_run, additional_run)
    if output_run.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_run} exists; pass --overwrite to replace the merged output")
        shutil.rmtree(output_run)
    aggregate_dir = output_run / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    frames = {filename: pd.concat([read_session_csvs(base_run, filename), read_session_csvs(additional_run, filename)], ignore_index=True) for filename in CSV_FILENAMES}
    validate_merged_tables(frames)
    if frames["checkpoint_manifest.csv"].empty:
        frames["checkpoint_manifest.csv"] = legacy_checkpoint_manifest(frames["fold_summary.csv"])
    else:
        legacy = legacy_checkpoint_manifest(read_session_csvs(base_run, "fold_summary.csv"))
        frames["checkpoint_manifest.csv"] = pd.concat([frames["checkpoint_manifest.csv"], legacy], ignore_index=True)
        frames["checkpoint_manifest.csv"] = frames["checkpoint_manifest.csv"].drop_duplicates(
            subset=["session", "task", "method", "seed", "fold"],
            keep="first",
        )
    if frames["order_sensitivity_oof_summary.csv"].empty and not frames["order_sensitivity_predictions.csv"].empty:
        frames["order_sensitivity_oof_summary.csv"] = order_sensitivity_oof_summary(frames["order_sensitivity_predictions.csv"])

    overfitting_audit, overfitting_summary = overfitting_audit_tables(frames["fold_summary.csv"], frames["training_history.csv"])
    block_type_df = block_type_accuracy(frames["predictions.csv"])
    master_wide = method_summary_table(frames["master_summary.csv"])
    seed_summary = seed_mean_summary(frames["master_summary.csv"])
    completeness = completeness_report(
        task=task,
        sessions=sessions,
        methods=methods,
        seeds=seeds,
        master=frames["master_summary.csv"],
        fold_summary=frames["fold_summary.csv"],
        predictions=frames["predictions.csv"],
    )
    parameter_audit = pd.DataFrame(parameter_audit_rows(methods))

    write_df(master_wide, aggregate_dir / "multiframe_all_models_master_summary.csv")
    write_df(frames["master_summary.csv"], aggregate_dir / "multiframe_all_models_master_long.csv")
    write_df(seed_summary, aggregate_dir / "multiframe_all_models_seed_summary.csv")
    write_df(frames["fold_summary.csv"], aggregate_dir / "multiframe_all_models_fold_summary.csv")
    write_df(frames["predictions.csv"], aggregate_dir / "multiframe_all_models_predictions.csv")
    write_df(overfitting_audit, aggregate_dir / "multiframe_all_models_overfitting_audit.csv")
    write_df(overfitting_summary, aggregate_dir / "multiframe_all_models_overfitting_method_summary.csv")
    write_df(parameter_audit, aggregate_dir / "multiframe_all_models_parameter_audit.csv")
    write_df(completeness, aggregate_dir / "multiframe_all_models_completeness_report.csv")
    write_df(block_type_df, aggregate_dir / "multiframe_all_models_block_type_accuracy.csv")
    write_df(frames["order_sensitivity.csv"], aggregate_dir / "multiframe_all_models_order_sensitivity.csv")
    write_df(frames["order_sensitivity_predictions.csv"], aggregate_dir / "order_sensitivity_predictions.csv")
    write_df(frames["order_sensitivity_oof_summary.csv"], aggregate_dir / "order_sensitivity_oof_summary.csv")
    write_df(frames["checkpoint_manifest.csv"], aggregate_dir / "checkpoint_manifest.csv")

    plot_paths = []
    plot_paths.extend(plot_method_comparison(frames["master_summary.csv"], task, aggregate_dir, "all_models_binary_comparison"))
    plot_paths.extend(
        plot_selected_methods(
            frames["master_summary.csv"],
            task,
            aggregate_dir,
            "cnn_vs_fcnn_temporal_comparison",
            ["cnn2d_lstm", "cnn2d_temporal1d", "fcnn_lstm"],
            "CNN vs FCNN temporal methods",
        )
    )
    plot_paths.extend(
        plot_selected_methods(
            frames["master_summary.csv"],
            task,
            aggregate_dir,
            "late_fusion_vs_meanpool_vs_lstm",
            ["single_frame_late_fusion", "fcnn_late_fusion", "fcnn_meanpool", "fcnn_lstm"],
            "Late fusion vs mean-pool vs LSTM",
        )
    )
    plot_paths.extend(plot_fcnn_order_sensitivity(frames["order_sensitivity_oof_summary.csv"], task, aggregate_dir))
    plot_paths.extend(plot_parameter_count_vs_test_ba(frames["master_summary.csv"], task, aggregate_dir, "parameter_count_vs_test_ba"))
    plot_paths.extend(plot_block_type_accuracy(block_type_df, task, aggregate_dir, "block_type_accuracy"))
    plot_paths.extend(plot_generalization_gap(overfitting_summary, task, aggregate_dir, "generalization_gap"))
    write_df(pd.DataFrame({"path": [str(path) for path in plot_paths]}), aggregate_dir / "multiframe_all_models_plot_manifest.csv")

    write_analysis(aggregate_dir / "multiframe_binary_analysis.md", frames["master_summary.csv"], overfitting_summary, frames["order_sensitivity_oof_summary.csv"])
    print(f"[merge] wrote {aggregate_dir}")
    print("[merge] temporal-order flags: " + json.dumps({method: METHOD_USES_TEMPORAL_ORDER.get(method, False) for method in methods}, sort_keys=True))


if __name__ == "__main__":
    main()
