#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.datasets import (
    DEFAULT_ANALYSIS_LIMITS,
    load_monkey_session,
    make_fixed_temporal_windows,
)
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.io import session_mat_files
from ultrasound_decoding.linear import fit_predict_linear, preprocess_frames
from ultrasound_decoding.preprocessing import SpatialFilterConfig


METHOD = "pca_lda"
FRAME_DURATION_S = 4.0
EXPECTED_SESSIONS = ["708", "709", "710", "807", "813", "817", "822"]
EXPECTED_TASKS = ["binary", "stimulus_type"]
WINDOW_SPECS: list[tuple[str, tuple[int, ...]]] = [
    ("k1_p0", (0,)),
    ("k1_p1", (1,)),
    ("k1_p2", (2,)),
    ("k1_p3", (3,)),
    ("k2_p0-1", (0, 1)),
    ("k2_p1-2", (1, 2)),
    ("k2_p2-3", (2, 3)),
    ("k3_p0-1-2", (0, 1, 2)),
    ("k3_p1-2-3", (1, 2, 3)),
    ("k4_p0-1-2-3", (0, 1, 2, 3)),
]


@dataclass(frozen=True)
class Experiment:
    session: str
    task: str
    window_id: str
    positions: tuple[int, ...]

    @property
    def window_size(self) -> int:
        return len(self.positions)

    @property
    def start_position(self) -> int:
        return int(self.positions[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unified fixed temporal-window PCA+LDA analysis."
    )
    parser.add_argument("--sessions", nargs="+", default=EXPECTED_SESSIONS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=EXPECTED_TASKS,
        choices=EXPECTED_TASKS,
        help="Tasks to compute in this invocation.",
    )
    parser.add_argument(
        "--aggregate-tasks",
        nargs="+",
        default=None,
        choices=EXPECTED_TASKS,
        help="Tasks to include when rebuilding aggregate outputs.",
    )
    parser.add_argument("--method", default=METHOD, choices=[METHOD])
    parser.add_argument("--window-spec", default="all", choices=["all"])
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument("--output-root", default="results/runs/temporal_windows")
    parser.add_argument("--run-name", default="fixed_window_unified_v1")
    parser.add_argument("--reuse-compatible-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only rebuild aggregate tables, plots, and findings from existing outputs.",
    )
    parser.add_argument(
        "--analysis-limit",
        default=None,
        help="Optional inclusive frame range like 1:180. Use 'default' for legacy limits.",
    )
    return parser.parse_args()


def parse_analysis_limit(session: str, value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if value == "default":
        return DEFAULT_ANALYSIS_LIMITS.get(session)
    lo, hi = value.split(":")
    return int(lo), int(hi)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def csv_readable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def data_fingerprint(project_dir: Path, session: str) -> dict[str, Any]:
    files = session_mat_files(project_dir / "data" / session)
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        piece = f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}\n"
        digest.update(piece.encode("utf-8"))
    return {
        "session": session,
        "n_mat_files": len(files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "sha256_names_sizes_mtimes": digest.hexdigest(),
    }


def experiment_dir(run_dir: Path, exp: Experiment) -> Path:
    return run_dir / exp.session / exp.task / exp.window_id


def output_paths(run_dir: Path, exp: Experiment) -> dict[str, Path]:
    base = experiment_dir(run_dir, exp)
    return {
        "dir": base,
        "summary": base / "summary.json",
        "fold_metrics": base / "fold_metrics.csv",
        "overall_metrics": base / "overall_metrics.csv",
        "predictions": base / "predictions.csv",
        "split_audit": base / "split_audit.csv",
        "config": base / "config.json",
        "confusion_matrix": base / "confusion_matrix.csv",
        "samples": base / "samples.csv",
    }


def expected_config(
    project_dir: Path,
    exp: Experiment,
    args: argparse.Namespace,
    data_version: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session": exp.session,
        "task": exp.task,
        "method": METHOD,
        "window_mode": "fixed",
        "window_id": exp.window_id,
        "window_size": exp.window_size,
        "within_block_positions": list(exp.positions),
        "fixed_window_start_position": exp.start_position,
        "clean_middle": True,
        "clean_margin_s": float(args.clean_margin_s),
        "trim_incomplete_cycles": True,
        "frames_per_cycle": 30,
        "frame_duration_s": FRAME_DURATION_S,
        "analysis_limit": parse_analysis_limit(exp.session, args.analysis_limit),
        "spatial_filter": SpatialFilterConfig(method="none", radius=0, mode="reflect").to_dict(),
        "pca_variance": float(args.pca_variance),
        "standardize_scope": "train_fold_only",
        "cv_group": "cycle",
        "max_folds": int(args.max_folds),
        "data_version": data_version,
        "time_mapping_status": "nominal",
        "code_entrypoint": str(Path(__file__).relative_to(project_dir)),
    }


def config_matches(path: Path, config: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return old == config


def complete_output_set(paths: dict[str, Path]) -> bool:
    required = ["summary", "fold_metrics", "overall_metrics", "predictions", "split_audit", "config"]
    return all(paths[name].exists() and paths[name].stat().st_size > 0 for name in required)


def legacy_candidates(project_dir: Path, exp: Experiment) -> list[Path]:
    names = [
        project_dir / "reports" / "fixed_temporal_window" / f"{exp.session}_{exp.task}_fixed_window_scan.csv",
        project_dir / "reports" / "temporal_window" / f"{exp.session}_{exp.task}_fixed_window_scan.csv",
    ]
    return [path for path in names if path.exists()]


def plan_experiments(
    project_dir: Path,
    run_dir: Path,
    experiments: list[Experiment],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], dict[str, Any]]]:
    fingerprints = {session: data_fingerprint(project_dir, session) for session in sorted({e.session for e in experiments})}
    configs: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows = []
    for exp in experiments:
        config = expected_config(project_dir, exp, args, fingerprints[exp.session])
        configs[(exp.session, exp.task, exp.window_id)] = config
        paths = output_paths(run_dir, exp)
        has_outputs = complete_output_set(paths)
        compatible = has_outputs and config_matches(paths["config"], config)
        legacy = legacy_candidates(project_dir, exp)
        if compatible and args.reuse_compatible_results:
            action = "reuse_compatible_result"
        elif has_outputs and not compatible:
            action = "blocked_incompatible_existing_output"
        else:
            action = "compute"
        rows.append(
            {
                "session": exp.session,
                "task": exp.task,
                "method": METHOD,
                "window_id": exp.window_id,
                "window_size": exp.window_size,
                "within_block_positions": csv_readable_json(list(exp.positions)),
                "existing_complete_output": bool(has_outputs),
                "compatible_existing_output": bool(compatible),
                "legacy_candidate_count": int(len(legacy)),
                "legacy_candidate_paths": ";".join(str(path) for path in legacy),
                "legacy_status": "incompatible_no_config" if legacy else "none",
                "planned_action": action,
            }
        )
    return pd.DataFrame(rows), configs


def class_count_dict(values: np.ndarray) -> dict[str, int]:
    series = pd.Series(values)
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return classification_metrics(np.asarray(y_true), np.asarray(y_pred))


def nominal_times(meta: pd.DataFrame) -> dict[str, Any]:
    if meta.empty:
        return {
            "nominal_time_start_s": np.nan,
            "nominal_time_end_s": np.nan,
            "nominal_time_center_s": np.nan,
            "nominal_time_start_s_values": [],
            "nominal_time_end_s_values": [],
            "nominal_time_center_s_values": [],
        }
    start_values = sorted((meta["window_start_offset_s"].astype(float) - FRAME_DURATION_S / 2.0).unique())
    end_values = sorted((meta["window_end_offset_s"].astype(float) + FRAME_DURATION_S / 2.0).unique())
    center_values = sorted(
        (
            (
                meta["window_start_offset_s"].astype(float)
                + meta["window_end_offset_s"].astype(float)
            )
            / 2.0
        ).unique()
    )
    start = float(np.mean(start_values))
    end = float(np.mean(end_values))
    return {
        "nominal_time_start_s": start,
        "nominal_time_end_s": end,
        "nominal_time_center_s": float((start + end) / 2.0),
        "nominal_time_start_s_values": [float(value) for value in start_values],
        "nominal_time_end_s_values": [float(value) for value in end_values],
        "nominal_time_center_s_values": [float(value) for value in center_values],
    }


def unique_float_list(values: pd.Series | np.ndarray | list[float]) -> list[float]:
    series = pd.Series(values).dropna().astype(float)
    return [float(value) for value in sorted(series.unique())]


def scalar_if_unique(values: list[float]) -> float:
    return float(values[0]) if len(values) == 1 else np.nan


def positions_for_window_id(window_id: str) -> tuple[int, ...]:
    lookup = dict(WINDOW_SPECS)
    if window_id not in lookup:
        raise KeyError(f"Unknown fixed-window id: {window_id}")
    return lookup[window_id]


def time_mapping_status(center_values: list[float]) -> str:
    return "nominal" if len(center_values) <= 1 else "block_dependent_nominal"


def block_time_rows_for_samples(
    session: str,
    task: str,
    window_id: str,
    samples: pd.DataFrame,
) -> list[dict[str, Any]]:
    positions = positions_for_window_id(window_id)
    rows = []
    for block_name, group in samples.groupby("block_name", sort=True):
        start_values = unique_float_list(group["window_start_offset_s"] - FRAME_DURATION_S / 2.0)
        end_values = unique_float_list(group["window_end_offset_s"] + FRAME_DURATION_S / 2.0)
        center_values = unique_float_list(
            (group["window_start_offset_s"] + group["window_end_offset_s"]) / 2.0
        )
        block_offset_values = unique_float_list(group["block_offset_s"])
        window_start_offset_values = unique_float_list(group["window_start_offset_s"])
        window_end_offset_values = unique_float_list(group["window_end_offset_s"])
        labels = (
            unique_float_list([])
            if "binary_label" not in group.columns
            else sorted(str(value) for value in group["binary_label"].dropna().unique())
        )
        label = ",".join(labels) if task == "binary" else str(block_name)
        rows.append(
            {
                "session": str(session),
                "task": str(task),
                "block_name": str(block_name),
                "label": label,
                "position": ",".join(str(value) for value in positions),
                "window_id": str(window_id),
                "nominal_start_s": scalar_if_unique(start_values),
                "nominal_end_s": scalar_if_unique(end_values),
                "nominal_center_s": scalar_if_unique(center_values),
                "n_samples": int(len(group)),
                "unique_center_values": csv_readable_json(center_values),
                "block_offset_s_values": csv_readable_json(block_offset_values),
                "window_start_offset_s_values": csv_readable_json(window_start_offset_values),
                "window_end_offset_s_values": csv_readable_json(window_end_offset_values),
            }
        )
    return rows


def read_block_time_mapping(run_dir: Path, sessions: list[str], tasks: list[str]) -> pd.DataFrame:
    rows = []
    for session in sessions:
        for task in tasks:
            for window_id, _positions in WINDOW_SPECS:
                samples_path = run_dir / session / task / window_id / "samples.csv"
                if not samples_path.exists():
                    continue
                samples = pd.read_csv(samples_path)
                rows.extend(block_time_rows_for_samples(session, task, window_id, samples))
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        return mapping
    center_by_position = (
        mapping.groupby(["session", "task", "position"])["unique_center_values"]
        .apply(lambda values: sorted({center for item in values for center in json.loads(item)}))
        .reset_index(name="position_center_values_across_blocks")
    )
    center_by_position["position_center_consistency"] = center_by_position[
        "position_center_values_across_blocks"
    ].apply(lambda values: "same_across_blocks" if len(values) <= 1 else "block_dependent")
    center_by_position["position_center_values_across_blocks"] = center_by_position[
        "position_center_values_across_blocks"
    ].apply(csv_readable_json)
    return mapping.merge(center_by_position, on=["session", "task", "position"], how="left")


def time_info_for_window(mapping: pd.DataFrame, session: str, task: str, window_id: str) -> dict[str, Any]:
    if mapping.empty:
        return {
            "nominal_time_start_s": np.nan,
            "nominal_time_end_s": np.nan,
            "nominal_time_center_s": np.nan,
            "time_mapping_status": "missing_time_mapping",
            "nominal_time_start_s_values": "[]",
            "nominal_time_end_s_values": "[]",
            "nominal_time_center_s_values": "[]",
            "block_specific_time_mapping": "[]",
        }
    subset = mapping[
        (mapping["session"].astype(str) == str(session))
        & (mapping["task"].astype(str) == str(task))
        & (mapping["window_id"] == window_id)
    ]
    if subset.empty:
        return {
            "nominal_time_start_s": np.nan,
            "nominal_time_end_s": np.nan,
            "nominal_time_center_s": np.nan,
            "time_mapping_status": "missing_time_mapping",
            "nominal_time_start_s_values": "[]",
            "nominal_time_end_s_values": "[]",
            "nominal_time_center_s_values": "[]",
            "block_specific_time_mapping": "[]",
        }
    start_values = unique_float_list(subset["nominal_start_s"])
    end_values = unique_float_list(subset["nominal_end_s"])
    center_values = unique_float_list(subset["nominal_center_s"])
    status = time_mapping_status(center_values)
    block_records = subset[
        [
            "block_name",
            "label",
            "position",
            "nominal_start_s",
            "nominal_end_s",
            "nominal_center_s",
            "n_samples",
            "unique_center_values",
        ]
    ].sort_values("block_name").to_dict("records")
    return {
        "nominal_time_start_s": scalar_if_unique(start_values),
        "nominal_time_end_s": scalar_if_unique(end_values),
        "nominal_time_center_s": scalar_if_unique(center_values),
        "time_mapping_status": status,
        "nominal_time_start_s_values": csv_readable_json(start_values),
        "nominal_time_end_s_values": csv_readable_json(end_values),
        "nominal_time_center_s_values": csv_readable_json(center_values),
        "block_specific_time_mapping": csv_readable_json(block_records),
    }


def validate_fixed_window_samples(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    base_meta: pd.DataFrame,
    exp: Experiment,
) -> None:
    if len(X) != len(y) or len(X) != len(groups) or len(X) != len(meta):
        raise AssertionError("Window arrays, labels, groups, and metadata lengths differ")
    if X.shape[1] != exp.window_size:
        raise AssertionError("Window tensor length does not match window_size")
    if not np.all(meta["window_size"].astype(int).to_numpy() == exp.window_size):
        raise AssertionError("Metadata window_size does not match requested window_size")
    if not np.all(meta["window_start_position"].astype(int).to_numpy() == exp.start_position):
        raise AssertionError("Metadata start position does not match requested positions")
    if meta.groupby(["cycle", "block_name"]).size().max() > 1:
        raise AssertionError("A cycle/block generated more than one fixed-window sample")

    by_index = base_meta.set_index("index", drop=False)
    for sample_i, row in meta.reset_index(drop=True).iterrows():
        indices = [int(value) for value in str(row["window_indices"]).split(",") if value]
        if len(indices) != exp.window_size:
            raise AssertionError("window_indices length does not match window_size")
        if not np.all(np.diff(indices) == 1):
            raise AssertionError("Window indices are not in fixed ascending temporal order")
        frame_rows = by_index.loc[indices]
        if isinstance(frame_rows, pd.Series):
            frame_rows = frame_rows.to_frame().T
        if frame_rows["cycle"].nunique() != 1:
            raise AssertionError("A fixed window crosses cycle boundaries")
        if frame_rows["block_name"].nunique() != 1:
            raise AssertionError("A fixed window crosses block boundaries")
        label_column = "binary_label" if exp.task == "binary" else "block_name"
        if frame_rows[label_column].nunique() != 1:
            raise AssertionError("A fixed window contains mixed labels")
        if int(frame_rows["cycle"].iloc[0]) != int(row["cycle"]):
            raise AssertionError("Window metadata cycle disagrees with frame metadata")
        if str(frame_rows["block_name"].iloc[0]) != str(row["block_name"]):
            raise AssertionError("Window metadata block disagrees with frame metadata")
        if str(frame_rows[label_column].iloc[0]) != str(y[sample_i]):
            raise AssertionError("Window label disagrees with frame metadata")

    expected_blocks_per_cycle = 4 if exp.task == "binary" else 2
    expected_samples = int(pd.Series(groups).nunique()) * expected_blocks_per_cycle
    if len(X) != expected_samples:
        raise AssertionError(
            f"n_samples={len(X)} does not equal expected {expected_samples}"
        )


def validate_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        train_cycles = set(groups[train_idx].tolist())
        test_cycles = set(groups[test_idx].tolist())
        if train_cycles & test_cycles:
            raise AssertionError(f"Fold {fold_i} has cycle leakage between train and test")
        if len(np.unique(y[train_idx])) < 2:
            raise AssertionError(f"Fold {fold_i} train set has fewer than two classes")
        if len(np.unique(y[test_idx])) < 2:
            raise AssertionError(f"Fold {fold_i} test set has fewer than two classes")


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    meta: pd.DataFrame,
    exp: Experiment,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    classes = np.unique(y)
    splits = grouped_cv_splits(groups, max_folds=args.max_folds)
    validate_splits(splits, y, groups)
    X_flat = preprocess_frames(X)

    all_true: list[Any] = []
    all_pred: list[Any] = []
    prediction_rows = []
    fold_rows = []
    split_rows = []
    assertions = [
        "fixed_windows_stay_within_one_block",
        "fixed_windows_stay_within_one_cycle",
        "fixed_windows_have_requested_length",
        "fixed_window_ids_have_global_positions",
        "cycle_group_cv_has_no_train_test_overlap",
        "linear_scaler_fit_on_train_fold_only",
        "pca_fit_on_train_fold_only",
        "lda_fit_on_train_fold_only",
        "test_data_transform_uses_train_fold_statistics",
        "multi_frame_flattening_uses_time_order_then_pixels",
        "one_window_per_cycle_block",
        "n_samples_matches_cycle_block_count",
        "incomplete_cycles_trimmed_before_windowing",
        "invalid_folds_raise_error",
        "compatible_reuse_requires_exact_config_match",
        "existing_outputs_are_not_overwritten",
    ]
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        pred, n_components = fit_predict_linear(
            METHOD,
            X_flat[train_idx],
            y[train_idx],
            X_flat[test_idx],
            pca_variance=args.pca_variance,
            standardize=True,
        )
        metrics = metric_row(y[test_idx], pred)
        train_cycles = sorted(int(value) for value in np.unique(groups[train_idx]))
        test_cycles = sorted(int(value) for value in np.unique(groups[test_idx]))
        fold_rows.append(
            {
                "session": exp.session,
                "task": exp.task,
                "method": METHOD,
                "window_id": exp.window_id,
                "fold": fold_i,
                "test_cycles": ",".join(str(value) for value in test_cycles),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_components": int(n_components),
                **metrics,
            }
        )
        split_rows.append(
            {
                "fold": fold_i,
                "train_cycles": ",".join(str(value) for value in train_cycles),
                "test_cycles": ",".join(str(value) for value in test_cycles),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_class_counts": csv_readable_json(class_count_dict(y[train_idx])),
                "test_class_counts": csv_readable_json(class_count_dict(y[test_idx])),
                "cycle_overlap_count": 0,
                "valid_fold": True,
            }
        )
        for sample_i, truth, predicted in zip(test_idx, y[test_idx], pred):
            sample = meta.loc[int(sample_i)]
            prediction_rows.append(
                {
                    "session": exp.session,
                    "task": exp.task,
                    "method": METHOD,
                    "window_id": exp.window_id,
                    "fold": fold_i,
                    "sample_i": int(sample_i),
                    "cycle": int(sample["cycle"]),
                    "block_name": str(sample["block_name"]),
                    "window_indices": str(sample["window_indices"]),
                    "truth": str(truth),
                    "pred": str(predicted),
                }
            )
        all_true.extend(y[test_idx].tolist())
        all_pred.extend(pred.tolist())

    overall = metric_row(np.asarray(all_true), np.asarray(all_pred))
    cm = confusion_matrix(np.asarray(all_true), np.asarray(all_pred), classes)
    overall_df = pd.DataFrame(
        [
            {
                "session": exp.session,
                "task": exp.task,
                "method": METHOD,
                "window_id": exp.window_id,
                "n_test_predictions": int(len(all_true)),
                **overall,
            }
        ]
    )
    confusion_rows = []
    for i, truth_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            confusion_rows.append(
                {
                    "truth": str(truth_class),
                    "pred": str(pred_class),
                    "count": int(cm[i, j]),
                }
            )
    details = {
        "classes": [str(value) for value in classes.tolist()],
        "confusion_matrix": cm.tolist(),
        "overall": overall,
        "n_splits": len(splits),
    }
    return (
        pd.DataFrame(fold_rows),
        overall_df,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(split_rows),
        assertions,
        details,
    )


def write_experiment_outputs(
    project_dir: Path,
    run_dir: Path,
    exp: Experiment,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    paths = output_paths(run_dir, exp)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if any(paths[name].exists() for name in ["summary", "fold_metrics", "overall_metrics", "predictions", "split_audit", "config"]):
        raise FileExistsError(
            f"Refusing to overwrite existing files in {paths['dir']}. "
            "Use --reuse-compatible-results for exact-compatible outputs or choose a new --run-name."
        )

    base_X, base_y, _, base_meta = load_monkey_session(
        project_dir,
        session=exp.session,
        task=exp.task,
        clean_middle=True,
        clean_margin_s=args.clean_margin_s,
        analysis_limit=parse_analysis_limit(exp.session, args.analysis_limit),
        trim_incomplete_cycles=True,
        window_size=1,
        spatial_filter=SpatialFilterConfig(method="none", radius=0, mode="reflect"),
    )
    X, y, groups, meta = make_fixed_temporal_windows(
        base_X,
        base_y,
        base_meta,
        exp.window_size,
        exp.start_position,
    )
    validate_fixed_window_samples(X, y, groups, meta, base_meta, exp)
    fold_df, overall_df, pred_df, split_df, assertions, cv_details = run_cv(X, y, groups, meta, exp, args)
    config_with_observed = {
        **config,
        "raw_loaded_shape": base_meta.attrs.get("selection_info", {}).get("raw_loaded_shape"),
        "data_shape_after_windowing": [int(value) for value in X.shape],
    }
    time_info = nominal_times(meta)
    classes = np.unique(y)
    cm = cv_details["confusion_matrix"]
    summary = {
        "session": exp.session,
        "task": exp.task,
        "method": METHOD,
        "window_id": exp.window_id,
        "window_size": exp.window_size,
        "within_block_positions": list(exp.positions),
        **time_info,
        "time_mapping_status": "nominal",
        "n_samples": int(len(X)),
        "n_cycles": int(len(np.unique(groups))),
        "class_counts": class_count_dict(y),
        "block_counts": {
            str(key): int(value)
            for key, value in meta["block_name"].value_counts().sort_index().items()
        },
        "clean_margin_s": float(args.clean_margin_s),
        "pca_variance": float(args.pca_variance),
        "cv_group": "cycle",
        "n_splits": int(cv_details["n_splits"]),
        "accuracy": float(cv_details["overall"]["accuracy"]),
        "balanced_accuracy": float(cv_details["overall"]["balanced_accuracy"]),
        "macro_f1": float(cv_details["overall"]["macro_f1"]),
        "classes": [str(value) for value in classes.tolist()],
        "confusion_matrix": cm,
        "source_type": "newly_computed",
        "source_path": str(paths["summary"]),
        "config_path": str(paths["config"]),
        "assertions_passed": assertions,
        "config": config_with_observed,
    }
    fold_df.to_csv(paths["fold_metrics"], index=False)
    overall_df.to_csv(paths["overall_metrics"], index=False)
    pred_df.to_csv(paths["predictions"], index=False)
    split_df.to_csv(paths["split_audit"], index=False)
    meta.to_csv(paths["samples"], index=False)
    pd.DataFrame(
        [
            {"truth": str(classes[i]), "pred": str(classes[j]), "count": int(cm[i][j])}
            for i in range(len(classes))
            for j in range(len(classes))
        ]
    ).to_csv(paths["confusion_matrix"], index=False)
    json_dump(paths["config"], config)
    json_dump(paths["summary"], summary)
    return summary


def reuse_summary(run_dir: Path, exp: Experiment) -> dict[str, Any]:
    summary_path = output_paths(run_dir, exp)["summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_type"] = "reused_compatible_result"
    summary["source_path"] = str(summary_path)
    return summary


def read_all_summaries(
    run_dir: Path,
    sessions: list[str],
    tasks: list[str],
    block_time_mapping: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for session in sessions:
        for task in tasks:
            for window_id, _positions in WINDOW_SPECS:
                path = run_dir / session / task / window_id / "summary.json"
                if not path.exists():
                    continue
                summary = json.loads(path.read_text(encoding="utf-8"))
                positions = summary["within_block_positions"]
                time_info = time_info_for_window(block_time_mapping, session, task, window_id)
                rows.append(
                    {
                        "session": str(summary["session"]),
                        "task": str(summary["task"]),
                        "method": str(summary["method"]),
                        "window_id": str(summary["window_id"]),
                        "window_size": int(summary["window_size"]),
                        "window_start_position": int(positions[0]),
                        "within_block_positions": csv_readable_json(positions),
                        "position": ",".join(str(value) for value in positions),
                        **time_info,
                        "n_samples": int(summary["n_samples"]),
                        "n_cycles": int(summary["n_cycles"]),
                        "accuracy": float(summary["accuracy"]),
                        "balanced_accuracy": float(summary["balanced_accuracy"]),
                        "macro_f1": float(summary["macro_f1"]),
                        "n_splits": int(summary["n_splits"]),
                        "source_type": str(summary.get("source_type", "newly_computed")),
                        "source_summary_path": str(path),
                    }
                )
    return pd.DataFrame(rows)


def session_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if master.empty:
        return pd.DataFrame(rows)
    for (session, task), group in master.groupby(["session", "task"], sort=True):
        ordered = group.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
        worst = group.sort_values(["balanced_accuracy", "macro_f1"], ascending=[True, True]).iloc[0]
        by_k = {}
        for k in [1, 2, 3, 4]:
            kg = group[group["window_size"] == k].sort_values(
                ["balanced_accuracy", "macro_f1"], ascending=[False, False]
            )
            by_k[k] = kg.iloc[0] if len(kg) else None
        rows.append(
            {
                "session": session,
                "task": task,
                "best_window_id": best["window_id"],
                "best_balanced_accuracy": float(best["balanced_accuracy"]),
                "best_macro_f1": float(best["macro_f1"]),
                "second_best_window_id": second["window_id"],
                "worst_window_id": worst["window_id"],
                "single_frame_best_window": by_k[1]["window_id"] if by_k[1] is not None else "",
                "two_frame_best_window": by_k[2]["window_id"] if by_k[2] is not None else "",
                "three_frame_best_window": by_k[3]["window_id"] if by_k[3] is not None else "",
                "four_frame_result": by_k[4]["window_id"] if by_k[4] is not None else "",
                "four_frame_balanced_accuracy": float(by_k[4]["balanced_accuracy"]) if by_k[4] is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cross_session_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if master.empty:
        return pd.DataFrame(rows)
    for (task, window_id), group in master.groupby(["task", "window_id"], sort=True):
        rows.append(
            {
                "task": task,
                "window_id": window_id,
                "window_size": int(group["window_size"].iloc[0]),
                "within_block_positions": group["within_block_positions"].iloc[0],
                "n_sessions": int(group["session"].nunique()),
                "balanced_accuracy_mean_unweighted": float(group["balanced_accuracy"].mean()),
                "balanced_accuracy_std_unweighted": float(group["balanced_accuracy"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "balanced_accuracy_median": float(group["balanced_accuracy"].median()),
                "macro_f1_mean_unweighted": float(group["macro_f1"].mean()),
                "number_of_sessions_above_chance": int((group["balanced_accuracy"] > 0.5).sum()),
                "number_of_sessions_below_chance": int((group["balanced_accuracy"] < 0.5).sum()),
                "min_session_score": float(group["balanced_accuracy"].min()),
                "max_session_score": float(group["balanced_accuracy"].max()),
            }
        )
    return pd.DataFrame(rows)


def completeness_report(
    run_dir: Path,
    sessions: list[str],
    tasks: list[str],
    master: pd.DataFrame,
    plan_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    key_set = {
        (row.session, row.task, row.window_id)
        for row in master[["session", "task", "window_id"]].itertuples(index=False)
    } if not master.empty else set()
    for session in sessions:
        for task in tasks:
            for window_id, positions in WINDOW_SPECS:
                exp = Experiment(session, task, window_id, positions)
                paths = output_paths(run_dir, exp)
                present = (session, task, window_id) in key_set
                config_status = "missing"
                method_status = "missing"
                missing_fold_count = np.nan
                invalid_fold_count = np.nan
                duplicate_run_count = 0
                if paths["summary"].exists():
                    try:
                        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
                        method_status = "ok" if summary.get("method") == METHOD else "wrong_method"
                        config_status = "present"
                    except json.JSONDecodeError:
                        config_status = "invalid_json"
                if paths["split_audit"].exists():
                    split_df = pd.read_csv(paths["split_audit"])
                    missing_fold_count = 0
                    invalid_fold_count = int((~split_df["valid_fold"].astype(bool)).sum()) if "valid_fold" in split_df else np.nan
                if paths["dir"].exists():
                    duplicate_run_count = max(0, len(list(paths["dir"].glob("summary*.json"))) - 1)
                planned_action = ""
                if plan_df is not None and not plan_df.empty:
                    match = plan_df[
                        (plan_df["session"].astype(str) == session)
                        & (plan_df["task"].astype(str) == task)
                        & (plan_df["window_id"] == window_id)
                    ]
                    if len(match):
                        planned_action = str(match.iloc[0]["planned_action"])
                rows.append(
                    {
                        "session": session,
                        "task": task,
                        "window_id": window_id,
                        "expected": True,
                        "present": bool(present),
                        "complete_file_set": bool(complete_output_set(paths)),
                        "missing_fold_count": missing_fold_count,
                        "duplicate_run_count": duplicate_run_count,
                        "config_status": config_status,
                        "method_status": method_status,
                        "invalid_fold_count": invalid_fold_count,
                        "planned_action": planned_action,
                    }
                )
    return pd.DataFrame(rows)


def save_single_frame_plot(master: pd.DataFrame, task: str, out_path: Path) -> None:
    df = master[(master["task"] == task) & (master["window_size"] == 1)].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=160)
    order = [spec[0] for spec in WINDOW_SPECS if len(spec[1]) == 1]
    for session, group in df.groupby("session", sort=True):
        group = group.set_index("window_id").loc[order].reset_index()
        ax.plot(
            group["window_start_position"],
            group["balanced_accuracy"],
            marker="o",
            linewidth=1.8,
            label=session,
        )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(single_frame_xtick_labels(df))
    ax.set_xlabel("position")
    ax.set_ylabel("balanced_accuracy")
    ax.set_title(f"{task} fixed single-frame windows")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Nominal time ranges are block-specific; different blocks can be offset by about 1-2 s.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def single_frame_xtick_labels(df: pd.DataFrame) -> list[str]:
    labels = []
    for position in [0, 1, 2, 3]:
        group = df[df["window_start_position"] == position]
        start_values = sorted(
            {
                value
                for item in group["nominal_time_start_s_values"].dropna()
                for value in json.loads(item)
            }
        )
        end_values = sorted(
            {
                value
                for item in group["nominal_time_end_s_values"].dropna()
                for value in json.loads(item)
            }
        )
        if start_values and end_values:
            label = (
                f"{position}\n"
                f"{format_values_for_tick(start_values)}-"
                f"{format_values_for_tick(end_values)}s"
            )
        else:
            label = str(position)
        labels.append(label)
    return labels


def format_values_for_tick(values: list[float]) -> str:
    formatted = [f"{value:g}" for value in values]
    return "/".join(formatted)


def save_single_frame_mean_plot(master: pd.DataFrame, task: str, out_path: Path) -> None:
    df = master[(master["task"] == task) & (master["window_size"] == 1)].copy()
    if df.empty:
        return
    grouped = (
        df.groupby(["window_id", "window_start_position"], sort=False)["balanced_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    order = [spec[0] for spec in WINDOW_SPECS if len(spec[1]) == 1]
    grouped["order"] = grouped["window_id"].map({value: i for i, value in enumerate(order)})
    grouped = grouped.sort_values("order")
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=160)
    ax.errorbar(
        grouped["window_start_position"],
        grouped["mean"],
        yerr=grouped["std"].fillna(0),
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(single_frame_xtick_labels(df))
    ax.set_xlabel("position")
    ax.set_ylabel("session-equal balanced_accuracy")
    ax.set_title(f"{task} single-frame session mean")
    ax.grid(True, axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.01,
        "Nominal time ranges are block-specific; different blocks can be offset by about 1-2 s.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path)
    plt.close(fig)


def save_heatmap(master: pd.DataFrame, task: str, out_path: Path) -> None:
    df = master[master["task"] == task].copy()
    if df.empty:
        return
    order = [spec[0] for spec in WINDOW_SPECS]
    pivot = df.pivot(index="session", columns="window_id", values="balanced_accuracy").reindex(columns=order)
    fig, ax = plt.subplots(figsize=(11, 4.3), dpi=160)
    im = ax.imshow(pivot.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"{task} fixed-window balanced_accuracy")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("balanced_accuracy")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_length_comparison(master: pd.DataFrame, task: str, out_path: Path) -> None:
    df = master[master["task"] == task].copy()
    if df.empty:
        return
    best = (
        df.sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
        .groupby(["session", "window_size"], sort=True)
        .head(1)
    )
    pivot = best.pivot(index="session", columns="window_size", values="balanced_accuracy").reindex(columns=[1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    width = 0.18
    x = np.arange(len(pivot.index))
    for offset_i, k in enumerate([1, 2, 3, 4]):
        ax.bar(x + (offset_i - 1.5) * width, pivot[k], width=width, label=f"K={k}")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("balanced_accuracy")
    ax.set_title(f"{task} best within each K; descriptive same-data selection")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_sample_count_plot(master: pd.DataFrame, task: str, out_path: Path) -> None:
    df = master[master["task"] == task].copy()
    if df.empty:
        return
    sample_df = df.groupby(["session", "window_size"], sort=True)["n_samples"].first().reset_index()
    pivot = sample_df.pivot(index="session", columns="window_size", values="n_samples").reindex(columns=[1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=160)
    width = 0.18
    x = np.arange(len(pivot.index))
    for offset_i, k in enumerate([1, 2, 3, 4]):
        ax.bar(x + (offset_i - 1.5) * width, pivot[k], width=width, label=f"K={k}")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("n_samples")
    ax.set_title(f"{task} one fixed-window sample per block")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_fold_test_count_plot(run_dir: Path, sessions: list[str], tasks: list[str], out_path: Path) -> None:
    rows = []
    for session in sessions:
        for task in tasks:
            for window_id, _positions in WINDOW_SPECS:
                path = run_dir / session / task / window_id / "split_audit.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                for row in df.itertuples(index=False):
                    rows.append(
                        {
                            "session": session,
                            "task": task,
                            "window_id": window_id,
                            "fold": int(row.fold),
                            "n_test": int(row.n_test),
                        }
                    )
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
    labels = []
    values = []
    for (session, task), group in df.groupby(["session", "task"], sort=True):
        labels.append(f"{session}\n{task}")
        values.append(group["n_test"].to_numpy())
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_ylabel("test samples per fold")
    ax.set_title("Fixed-window fold test sample counts")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_plots(run_dir: Path, aggregate_dir: Path, master: pd.DataFrame, sessions: list[str], tasks: list[str]) -> list[Path]:
    plot_paths = []
    for task in tasks:
        paths = [
            aggregate_dir / f"fixed_single_frame_{task}.png",
            aggregate_dir / f"fixed_single_frame_mean_{task}.png",
            aggregate_dir / f"fixed_window_heatmap_{task}.png",
            aggregate_dir / f"fixed_window_length_comparison_{task}.png",
            aggregate_dir / f"fixed_window_sample_counts_{task}.png",
        ]
        save_single_frame_plot(master, task, paths[0])
        save_single_frame_mean_plot(master, task, paths[1])
        save_heatmap(master, task, paths[2])
        save_length_comparison(master, task, paths[3])
        save_sample_count_plot(master, task, paths[4])
        plot_paths.extend([path for path in paths if path.exists()])
    fold_path = aggregate_dir / "fixed_window_fold_test_counts.png"
    save_fold_test_count_plot(run_dir, sessions, tasks, fold_path)
    if fold_path.exists():
        plot_paths.append(fold_path)
    return plot_paths


def label_stage(window_id: str) -> str:
    if window_id in {"k1_p0", "k2_p0-1", "k3_p0-1-2"}:
        return "early"
    if window_id in {"k1_p3", "k2_p2-3", "k3_p1-2-3"}:
        return "late"
    return "middle"


def write_findings(aggregate_dir: Path, master: pd.DataFrame, cross: pd.DataFrame) -> Path:
    path = aggregate_dir / "temporal_window_findings.md"
    lines = [
        "# Temporal Window Findings",
        "",
        "All findings are descriptive. The reported best windows were selected on the same data used for comparison, so they are follow-up candidates rather than unbiased final inputs.",
        "Timing is nominal. The aggregate tables preserve block-specific time lists; when a window has multiple block-dependent values, no averaged time is treated as a precise timestamp.",
        "",
    ]
    if master.empty:
        lines.append("No complete fixed-window results were available.")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    for task in sorted(master["task"].unique()):
        task_cross = cross[cross["task"] == task].sort_values(
            ["balanced_accuracy_mean_unweighted", "macro_f1_mean_unweighted"],
            ascending=[False, False],
        )
        if task_cross.empty:
            continue
        top = task_cross.iloc[0]
        single = task_cross[task_cross["window_size"] == 1].iloc[0]
        lines.extend(
            [
                f"## {task}",
                "",
                f"- Highest session-equal mean balanced accuracy: {top['window_id']} ({top['balanced_accuracy_mean_unweighted']:.3f}).",
                f"- Best single-frame window by session-equal mean: {single['window_id']} ({single['balanced_accuracy_mean_unweighted']:.3f}).",
            ]
        )
        status_counts = master[master["task"] == task]["time_mapping_status"].value_counts().to_dict()
        lines.append(
            "- Time mapping status: "
            + ", ".join(f"{status}={count}" for status, count in sorted(status_counts.items()))
            + "."
        )
        best_by_session = (
            master[master["task"] == task]
            .sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
            .groupby("session", sort=True)
            .head(1)
        )
        lines.append(
            "- Session-level best windows: "
            + ", ".join(
                f"{row.session}={row.window_id} ({row.balanced_accuracy:.3f})"
                for row in best_by_session.itertuples(index=False)
            )
            + "."
        )
        stage_counts = best_by_session["window_id"].map(label_stage).value_counts().to_dict()
        lines.append(
            "- Best-window stage counts: "
            + ", ".join(f"{stage}={count}" for stage, count in sorted(stage_counts.items()))
            + "."
        )
        low_sessions = ["807", "813", "817", "822"]
        low = best_by_session[best_by_session["session"].isin(low_sessions)]
        if len(low):
            recovered = low[low["balanced_accuracy"] > 0.6]
            lines.append(
                f"- Low-performance session check: {len(recovered)}/{len(low)} sessions exceeded 0.60 at their descriptive best window."
            )
        lines.append("")

    if "binary" in set(master["task"]):
        binary = master[master["task"] == "binary"]
        single = (
            binary[binary["window_size"] == 1]
            .groupby("window_id")["balanced_accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        top_single = single.index[0]
        early_mid_late = label_stage(top_single)
        lines.extend(
            [
                "## Questions",
                "",
                f"1. Binary decoding was strongest at the {early_mid_late} single-frame position by cross-session mean ({top_single}).",
            ]
        )
        subset = (
            binary[binary["session"].isin(["708", "709", "710"])]
            .sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
            .groupby("session")
            .head(1)
        )
        lines.append(
            "2. For 708/709/710, descriptive best windows were "
            + ", ".join(f"{row.session}={row.window_id}" for row in subset.itertuples(index=False))
            + "."
        )
        low = (
            binary[binary["session"].isin(["807", "813", "817", "822"])]
            .sort_values(["balanced_accuracy", "macro_f1"], ascending=[False, False])
            .groupby("session")
            .head(1)
        )
        lines.append(
            "3. Low-session improvement check: "
            + ", ".join(f"{row.session} best {row.window_id}={row.balanced_accuracy:.3f}" for row in low.itertuples(index=False))
            + "."
        )

    if not cross.empty:
        by_k = (
            cross.groupby(["task", "window_size"])["balanced_accuracy_mean_unweighted"]
            .agg(["mean", "std"])
            .reset_index()
        )
        lines.append(
            "4. Window-length stability, by mean cross-window session-equal score: "
            + "; ".join(
                f"{row['task']} K={int(row['window_size'])} mean={row['mean']:.3f}, sd={(row['std'] if pd.notna(row['std']) else 0):.3f}"
                for _, row in by_k.iterrows()
            )
            + "."
        )
        multi = master[master["window_size"] > 1]
        single = master[master["window_size"] == 1]
        wins = 0
        total = 0
        for key, single_group in single.groupby(["session", "task"]):
            multi_group = multi[(multi["session"] == key[0]) & (multi["task"] == key[1])]
            if len(multi_group):
                total += 1
                wins += float(multi_group["balanced_accuracy"].max()) > float(single_group["balanced_accuracy"].max())
        lines.append(f"5. Multi-frame windows beat the best single frame in {wins}/{total} session-task pairs.")
        if "stimulus_type" in set(master["task"]):
            stim_cross = cross[cross["task"] == "stimulus_type"].sort_values(
                ["balanced_accuracy_mean_unweighted", "balanced_accuracy_std_unweighted"],
                ascending=[False, True],
            )
            top_stim = stim_cross.iloc[0]
            lines.append(
                f"6. Stimulus_type's top descriptive window was {top_stim.window_id}, but stability should be judged from the session curves and heatmap."
            )
        recommended = (
            cross.assign(
                stability_score=lambda frame: frame["balanced_accuracy_mean_unweighted"]
                - 0.25 * frame["balanced_accuracy_std_unweighted"].fillna(0)
            )
            .sort_values(["stability_score", "balanced_accuracy_mean_unweighted"], ascending=[False, False])
        )
        one = recommended[recommended["window_size"] == 1].head(1)
        multi_rec = recommended[recommended["window_size"].isin([2, 3])].head(1)
        rec_ids = []
        if len(one):
            rec_ids.append(str(one.iloc[0]["window_id"]))
        if len(multi_rec) and str(multi_rec.iloc[0]["window_id"]) not in rec_ids:
            rec_ids.append(str(multi_rec.iloc[0]["window_id"]))
        lines.append("7. Fixed two- or three-frame CNN inputs are worth follow-up only as predefined validation candidates.")
        lines.append("8. Recommended follow-up CNN validation candidates: " + ", ".join(rec_ids[:2]) + ".")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_aggregates(
    project_dir: Path,
    run_dir: Path,
    sessions: list[str],
    tasks: list[str],
    plan_df: pd.DataFrame | None,
) -> dict[str, Any]:
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    block_time_mapping = read_block_time_mapping(run_dir, sessions, tasks)
    block_time_path = aggregate_dir / "fixed_window_block_time_mapping.csv"
    block_time_mapping.to_csv(block_time_path, index=False)
    master = read_all_summaries(run_dir, sessions, tasks, block_time_mapping)
    master_path = aggregate_dir / "fixed_window_master_summary.csv"
    master.to_csv(master_path, index=False)
    sess = session_summary(master)
    sess_path = aggregate_dir / "fixed_window_session_summary.csv"
    sess.to_csv(sess_path, index=False)
    cross = cross_session_summary(master)
    cross_path = aggregate_dir / "fixed_window_cross_session_summary.csv"
    cross.to_csv(cross_path, index=False)
    comp = completeness_report(run_dir, sessions, tasks, master, plan_df=plan_df)
    comp_path = aggregate_dir / "fixed_window_completeness_report.csv"
    comp.to_csv(comp_path, index=False)
    plot_paths = write_plots(run_dir, aggregate_dir, master, sessions, tasks)
    findings_path = write_findings(aggregate_dir, master, cross)
    return {
        "aggregate_dir": str(aggregate_dir),
        "master_summary": str(master_path),
        "session_summary": str(sess_path),
        "cross_session_summary": str(cross_path),
        "completeness_report": str(comp_path),
        "block_time_mapping": str(block_time_path),
        "findings": str(findings_path),
        "plots": [str(path) for path in plot_paths],
        "n_master_rows": int(len(master)),
        "n_block_time_rows": int(len(block_time_mapping)),
        "n_expected_rows": int(len(sessions) * len(tasks) * len(WINDOW_SPECS)),
        "n_missing": int((~comp["present"]).sum()) if not comp.empty else 0,
        "n_duplicate": int(comp["duplicate_run_count"].sum()) if not comp.empty else 0,
    }


def experiments_from_args(args: argparse.Namespace) -> list[Experiment]:
    expected_window_positions = {window_id: positions for window_id, positions in WINDOW_SPECS}
    if len(expected_window_positions) != len(WINDOW_SPECS):
        raise AssertionError("Duplicate window_id in WINDOW_SPECS")
    for window_id, positions in WINDOW_SPECS:
        expected = tuple(range(positions[0], positions[0] + len(positions)))
        if positions != expected:
            raise AssertionError(f"{window_id} is not a consecutive fixed window")
    return [
        Experiment(session=str(session), task=str(task), window_id=window_id, positions=positions)
        for session in args.sessions
        for task in args.tasks
        for window_id, positions in WINDOW_SPECS
    ]


def summarize_plan(plan_df: pd.DataFrame) -> dict[str, int]:
    counts = plan_df["planned_action"].value_counts().to_dict() if not plan_df.empty else {}
    return {
        "planned_new_runs": int(counts.get("compute", 0)),
        "reusable_compatible_results": int(counts.get("reuse_compatible_result", 0)),
        "incompatible_existing_results": int(counts.get("blocked_incompatible_existing_output", 0)),
        "missing_results": int((~plan_df["existing_complete_output"]).sum()) if not plan_df.empty else 0,
        "duplicate_results": 0,
        "legacy_incompatible_results": int((plan_df["legacy_candidate_count"] > 0).sum()) if not plan_df.empty else 0,
    }


def main() -> None:
    args = parse_args()
    project_dir = PROJECT_DIR
    run_dir = project_dir / args.output_root / args.run_name
    aggregate_tasks = args.aggregate_tasks if args.aggregate_tasks is not None else args.tasks
    if args.aggregate_only:
        aggregate = write_aggregates(project_dir, run_dir, args.sessions, aggregate_tasks, plan_df=None)
        print(json.dumps({"aggregate_only": True, "aggregate": aggregate}, indent=2, ensure_ascii=False))
        return

    experiments = experiments_from_args(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_df, configs = plan_experiments(project_dir, run_dir, experiments, args)
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    plan_path = aggregate_dir / ("dry_run_plan.csv" if args.dry_run else "run_plan.csv")
    plan_df.to_csv(plan_path, index=False)
    plan_summary = summarize_plan(plan_df)
    json_dump(aggregate_dir / ("dry_run_summary.json" if args.dry_run else "run_summary.json"), plan_summary)

    if args.dry_run:
        print(json.dumps({"plan": plan_summary, "plan_path": str(plan_path)}, indent=2))
        return

    if (plan_df["planned_action"] == "blocked_incompatible_existing_output").any():
        blocked = plan_df[plan_df["planned_action"] == "blocked_incompatible_existing_output"]
        raise FileExistsError(
            "Found incompatible existing outputs; refusing to overwrite: "
            + "; ".join(
                f"{row.session}/{row.task}/{row.window_id}" for row in blocked.itertuples(index=False)
            )
        )

    run_summaries = []
    for exp in experiments:
        key = (exp.session, exp.task, exp.window_id)
        action = plan_df[
            (plan_df["session"].astype(str) == exp.session)
            & (plan_df["task"].astype(str) == exp.task)
            & (plan_df["window_id"] == exp.window_id)
        ]["planned_action"].iloc[0]
        if action == "reuse_compatible_result":
            run_summaries.append(reuse_summary(run_dir, exp))
            print(f"reused {exp.session} {exp.task} {exp.window_id}")
            continue
        summary = write_experiment_outputs(project_dir, run_dir, exp, configs[key], args)
        run_summaries.append(summary)
        print(
            f"computed {exp.session} {exp.task} {exp.window_id}: "
            f"balanced_accuracy={summary['balanced_accuracy']:.4f}, macro_f1={summary['macro_f1']:.4f}"
        )

    aggregate = write_aggregates(project_dir, run_dir, args.sessions, aggregate_tasks, plan_df)
    result = {
        "computed_or_reused": len(run_summaries),
        "plan": plan_summary,
        "aggregate": aggregate,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
