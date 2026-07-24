#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import itertools
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "matplotlib-codex"))

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.datasets import load_monkey_session
from ultrasound_decoding.deep import fit_predict_torch, get_torch_model_defaults, torch_available
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.linear import fit_predict_linear, preprocess_frames
from ultrasound_decoding.preprocessing import SpatialFilterConfig


LINEAR_METHODS = {"pca_lda", "cpca_lda"}
NEURAL_METHODS = {"cnn", "fcnn_berthon2023"}
ALL_METHODS = ["pca_lda", "cpca_lda", "cnn", "fcnn_berthon2023"]
METHOD_TO_INTERNAL = {"fcnn_berthon2023": "fcnn"}
EXPECTED_CLASSES = {
    "binary": ["no_stimulus", "stimulus"],
    "stimulus_type": ["dot", "grating"],
}
EXPECTED_BLOCKS = {
    "binary": ["dot", "grating", "static", "stop_after_grating"],
    "stimulus_type": ["dot", "grating"],
}
EXPECTED_STRONG_BINARY_CYCLES = {"708": 6, "709": 22, "710": 18}
SOURCE_BALANCE_MODES = {"pooled_all", "session_balanced"}


@dataclass
class SessionData:
    session: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    classes: list[str]

    @property
    def n_samples(self) -> int:
        return int(len(self.y))

    @property
    def n_cycles(self) -> int:
        return int(len(np.unique(self.groups)))

    @property
    def sample_counts(self) -> dict[str, int]:
        return {self.session: self.n_samples}

    @property
    def cycle_counts(self) -> dict[str, int]:
        return {self.session: self.n_cycles}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict zero-shot cross-session generalization for monkey fUS decoding."
    )
    parser.add_argument("--sessions", nargs="+", default=["708", "709", "710"])
    parser.add_argument("--tasks", nargs="+", default=["binary"], choices=["binary", "stimulus_type"])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--protocol", choices=["pairwise", "loso"], default="pairwise")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument(
        "--epoch-selection",
        choices=["fixed", "source-inner-validation"],
        default="source-inner-validation",
    )
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=str(PROJECT_DIR / "results" / "runs" / "generalization"))
    parser.add_argument("--run-name", default="pairwise_strong_sessions_v1")
    parser.add_argument("--source-balance", choices=sorted(SOURCE_BALANCE_MODES), default="pooled_all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="Only rebuild aggregate audits from existing result files.")
    parser.add_argument(
        "--reuse-compatible-results",
        action="store_true",
        help="Reuse completed result directories with matching config instead of rerunning.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "run"


def method_internal(method: str) -> str:
    return METHOD_TO_INTERNAL.get(method, method)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def prediction_class_summary(pred: np.ndarray, classes: list[str]) -> tuple[dict[str, int], bool]:
    counts = {label: int(np.sum(pred == label)) for label in classes}
    observed = [label for label, count in counts.items() if count > 0]
    return counts, len(observed) == 1


def completed_status(prediction_is_single_class: bool) -> str:
    return "completed_degenerate_prediction" if prediction_is_single_class else "completed"


def _compatible_value(existing: Any, expected: Any) -> bool:
    if existing == expected:
        return True
    if isinstance(existing, (int, float)) and isinstance(expected, (int, float)):
        return float(existing) == float(expected)
    return False


def assert_reuse_config_compatible(exp_dir: Path, expected: dict[str, Any]) -> None:
    config_path = exp_dir / "config.json"
    if not config_path.exists():
        raise FileExistsError(f"Cannot reuse {exp_dir}: missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mismatches = []
    for key, expected_value in expected.items():
        existing_value = config.get(key)
        if not _compatible_value(existing_value, expected_value):
            mismatches.append(f"{key}: existing={existing_value!r} expected={expected_value!r}")
    if mismatches:
        raise FileExistsError(
            f"Cannot reuse incompatible result directory {exp_dir}: " + "; ".join(mismatches)
        )


def load_session(session: str, task: str, clean_margin_s: float) -> SessionData:
    spatial_filter = SpatialFilterConfig(method="none", radius=0, mode="reflect")
    X, y, groups, meta = load_monkey_session(
        PROJECT_DIR,
        session=session,
        task=task,
        clean_middle=True,
        clean_margin_s=clean_margin_s,
        analysis_limit=None,
        trim_incomplete_cycles=True,
        window_size=1,
        window_mode="sliding",
        fixed_window_start_position=None,
        spatial_filter=spatial_filter,
    )
    meta = meta.copy()
    meta["session"] = session
    classes = np.unique(y).tolist()
    validate_single_session(SessionData(session, X, y, groups, meta, classes), task, clean_margin_s)
    return SessionData(session, X, y, groups, meta, classes)


def validate_single_session(data: SessionData, task: str, clean_margin_s: float) -> None:
    if tuple(data.X.shape[1:]) != (128, 501):
        raise AssertionError(f"{data.session}: expected frame shape [128,501], got {data.X.shape[1:]}")
    if data.classes != EXPECTED_CLASSES[task]:
        raise AssertionError(f"{data.session}: classes {data.classes} != {EXPECTED_CLASSES[task]}")
    if data.meta.attrs.get("selection_info", {}).get("window_size") != 1:
        raise AssertionError(f"{data.session}: window_size is not 1")
    if data.meta.attrs.get("selection_info", {}).get("window_mode") != "sliding":
        raise AssertionError(f"{data.session}: window_mode is not sliding")
    if not bool(data.meta.attrs.get("selection_info", {}).get("clean_middle")):
        raise AssertionError(f"{data.session}: clean_middle is not true")
    if float(data.meta.attrs.get("selection_info", {}).get("clean_margin_s")) != float(clean_margin_s):
        raise AssertionError(f"{data.session}: clean_margin_s mismatch")
    if data.meta.attrs.get("selection_info", {}).get("analysis_limit") is not None:
        raise AssertionError(f"{data.session}: analysis_limit must be none")
    if data.meta.attrs.get("selection_info", {}).get("spatial_filter", {}).get("method") != "none":
        raise AssertionError(f"{data.session}: spatial_filter must be none")
    if np.isnan(data.X).any() or np.isinf(data.X).any():
        raise AssertionError(f"{data.session}: input contains NaN or Inf")

    expected_blocks = set(EXPECTED_BLOCKS[task])
    for cycle, rows in data.meta.groupby("cycle", sort=True):
        block_counts = rows.groupby("block_name").size().to_dict()
        if set(block_counts) != expected_blocks:
            raise AssertionError(f"{data.session} cycle {cycle}: block set {sorted(block_counts)} mismatch")
        bad = {block: int(n) for block, n in block_counts.items() if int(n) != 4}
        if bad:
            raise AssertionError(f"{data.session} cycle {cycle}: expected 4 samples per block, got {bad}")


def experiment_specs(sessions: list[str], protocol: str) -> list[tuple[list[str], str, str]]:
    if protocol == "pairwise":
        return [([src], tgt, f"{src}_to_{tgt}") for src, tgt in itertools.permutations(sessions, 2)]
    return [([src for src in sessions if src != tgt], tgt, f"target_{tgt}") for tgt in sessions]


def combine_sources(
    source_sessions: list[str],
    data_by_session: dict[str, SessionData],
    selected_cycles_by_session: dict[str, list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X_parts = []
    y_parts = []
    group_parts = []
    meta_parts = []
    for session in source_sessions:
        data = data_by_session[session]
        if selected_cycles_by_session is None:
            mask = np.ones(len(data.y), dtype=bool)
        else:
            selected = set(int(cycle) for cycle in selected_cycles_by_session[session])
            mask = np.asarray([int(cycle) in selected for cycle in data.groups], dtype=bool)
        X_parts.append(data.X[mask])
        y_parts.append(data.y[mask])
        groups = np.asarray([f"{session}_cycle{int(c)}" for c in data.groups[mask]], dtype=object)
        group_parts.append(groups)
        meta_part = data.meta.loc[mask].copy()
        meta_parts.append(meta_part)
    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(group_parts, axis=0),
        pd.concat(meta_parts, ignore_index=True),
    )


def cycle_sample_counts(meta: pd.DataFrame) -> dict[int, int]:
    return {int(cycle): int(n) for cycle, n in meta.groupby("cycle", sort=True).size().items()}


def assert_complete_binary_cycles(source_sessions: list[str], source_meta: pd.DataFrame) -> None:
    for (session, cycle), rows in source_meta.groupby(["session", "cycle"], sort=True):
        if int(len(rows)) != 16:
            raise AssertionError(f"{session} cycle {cycle}: expected whole 16-frame binary cycle, got {len(rows)}")


def select_balanced_cycles(
    source_sessions: list[str],
    data_by_session: dict[str, SessionData],
    seed: int,
) -> tuple[dict[str, list[int]], pd.DataFrame]:
    available = {
        session: sorted(int(cycle) for cycle in np.unique(data_by_session[session].groups))
        for session in source_sessions
    }
    n_balanced_cycles = min(len(cycles) for cycles in available.values())
    rng = np.random.default_rng(seed)
    selected: dict[str, list[int]] = {}
    rows = []
    for session in source_sessions:
        cycles = available[session]
        if len(cycles) == n_balanced_cycles:
            chosen = cycles
        else:
            chosen = sorted(int(cycle) for cycle in rng.choice(cycles, size=n_balanced_cycles, replace=False))
        selected[session] = chosen
        sample_counts = cycle_sample_counts(data_by_session[session].meta)
        if any(sample_counts[int(cycle)] != 16 for cycle in chosen):
            raise AssertionError(f"{session}: balanced linear selection included partial binary cycle")
        rows.append(
            {
                "repeat_seed": int(seed),
                "source_session": session,
                "available_cycles": "+".join(str(cycle) for cycle in cycles),
                "selected_cycles": "+".join(str(cycle) for cycle in chosen),
                "n_selected_cycles": int(len(chosen)),
                "n_selected_samples": int(sum(sample_counts[int(cycle)] for cycle in chosen)),
            }
        )
    return selected, pd.DataFrame(rows)


def validate_experiment(
    source_sessions: list[str],
    target_session: str,
    task: str,
    data_by_session: dict[str, SessionData],
    source_meta: pd.DataFrame,
    source_groups: np.ndarray,
) -> None:
    if target_session in source_sessions:
        raise AssertionError("target session appears in source sessions")
    if len(source_sessions) == 1 and source_sessions[0] == target_session:
        raise AssertionError("source_session must differ from target_session")
    target = data_by_session[target_session]
    for source in source_sessions:
        src = data_by_session[source]
        if src.classes != target.classes or src.classes != EXPECTED_CLASSES[task]:
            raise AssertionError(f"class mapping mismatch for {source}->{target_session}")
        if src.X.shape[1:] != target.X.shape[1:]:
            raise AssertionError(f"shape mismatch for {source}->{target_session}")
    if target_session in set(source_meta["session"].astype(str)):
        raise AssertionError("target session leaked into source meta")
    if any(str(group).startswith(f"{target_session}_cycle") for group in source_groups):
        raise AssertionError("target session leaked into source groups")
    selected_cycle_count = int(source_meta.groupby(["session", "cycle"], sort=True).ngroups)
    if len(set(source_groups.tolist())) != selected_cycle_count:
        raise AssertionError("composite (session, cycle) groups are not unique")


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}


def session_sample_counts(source_sessions: list[str], data_by_session: dict[str, SessionData]) -> dict[str, int]:
    return {session: data_by_session[session].n_samples for session in source_sessions}


def session_cycle_counts(source_sessions: list[str], data_by_session: dict[str, SessionData]) -> dict[str, int]:
    return {session: data_by_session[session].n_cycles for session in source_sessions}


def source_session_proportions(source_sessions: list[str], data_by_session: dict[str, SessionData]) -> dict[str, float]:
    counts = session_sample_counts(source_sessions, data_by_session)
    total = max(sum(counts.values()), 1)
    return {session: float(count / total) for session, count in counts.items()}


def split_audit_rows(
    source_sessions: list[str],
    target_session: str,
    protocol: str,
    source_groups: np.ndarray,
    metadata: dict[str, Any] | None,
    epoch_selection: str = "source-inner-validation",
) -> list[dict[str, Any]]:
    source_group_labels = list(dict.fromkeys(str(group) for group in source_groups))
    parse_session = lambda group: group.split("_cycle", 1)[0]
    if epoch_selection == "fixed":
        inner_train_sessions = sorted({parse_session(group) for group in source_group_labels})
        if target_session in inner_train_sessions:
            raise AssertionError("target session leaked into fixed training split")
        return [
            {
                "protocol": protocol,
                "source_sessions": "+".join(source_sessions),
                "target_session": target_session,
                "phase": "fixed_full_source_training",
                "inner_train_sessions": "+".join(inner_train_sessions),
                "inner_val_sessions": "",
                "inner_train_cycles": "+".join(source_group_labels),
                "inner_val_cycles": "",
                "target_in_inner_train": False,
                "target_in_inner_val": False,
                "source_cycle_overlap": False,
            }
        ]
    val_groups = []
    if metadata:
        val_groups = metadata.get("inner_validation", {}).get("val_cycles", []) or []
    val_group_set = set(str(group) for group in val_groups)
    inner_train_groups = [group for group in source_group_labels if group not in val_group_set]
    inner_train_sessions = sorted({parse_session(group) for group in inner_train_groups})
    inner_val_sessions = sorted({parse_session(group) for group in val_group_set})
    if target_session in inner_train_sessions or target_session in inner_val_sessions:
        raise AssertionError("target session leaked into neural split")
    overlap = set(inner_train_groups) & val_group_set
    if overlap:
        raise AssertionError(f"source cycle overlap between inner train and validation: {sorted(overlap)[:3]}")
    return [
        {
            "protocol": protocol,
            "source_sessions": "+".join(source_sessions),
            "target_session": target_session,
            "phase": "inner_epoch_selection" if val_groups else "fixed_epoch_no_inner_validation",
            "inner_train_sessions": "+".join(inner_train_sessions),
            "inner_val_sessions": "+".join(inner_val_sessions),
            "inner_train_cycles": "+".join(inner_train_groups),
            "inner_val_cycles": "+".join(str(group) for group in sorted(val_group_set)),
            "target_in_inner_train": False,
            "target_in_inner_val": False,
            "source_cycle_overlap": False,
        }
    ]


def target_cycle_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_meta: pd.DataFrame,
    base: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    pred = np.asarray(y_pred)
    for cycle, cycle_rows in target_meta.groupby("cycle", sort=True):
        idx = cycle_rows.index.to_numpy(dtype=np.int64)
        metrics = classification_metrics(y_true[idx], pred[idx])
        rows.append(
            {
                **base,
                "target_cycle": int(cycle),
                "n_samples": int(len(idx)),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def write_common_outputs(
    exp_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    overall: dict[str, Any],
    cycle_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    cm_df: pd.DataFrame,
    norm_df: pd.DataFrame,
    history_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_json(exp_dir / "config.json", config)
    write_json(exp_dir / "summary.json", summary)
    pd.DataFrame([overall]).to_csv(exp_dir / "overall_metrics.csv", index=False)
    cycle_df.to_csv(exp_dir / "target_cycle_metrics.csv", index=False)
    pred_df.to_csv(exp_dir / "predictions.csv", index=False)
    cm_df.to_csv(exp_dir / "confusion_matrix.csv", index=False)
    norm_df.to_csv(exp_dir / "normalization_audit.csv", index=False)
    history_df.to_csv(exp_dir / "training_history.csv", index=False)
    split_df.to_csv(exp_dir / "split_audit.csv", index=False)


def build_prediction_df(
    target: SessionData,
    pred: np.ndarray,
    base: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for sample_i, (truth, predicted) in enumerate(zip(target.y, pred)):
        meta_row = target.meta.iloc[sample_i]
        rows.append(
            {
                **base,
                "sample_i": int(sample_i),
                "target_index": int(meta_row["index"]),
                "target_cycle": int(meta_row["cycle"]),
                "block_name": str(meta_row["block_name"]),
                "truth": str(truth),
                "pred": str(predicted),
            }
        )
    if len(rows) != len(target.y):
        raise AssertionError("target sample prediction count mismatch")
    return pd.DataFrame(rows)


def build_confusion_df(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], base: dict[str, Any]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, np.asarray(classes))
    rows = []
    for i, truth in enumerate(classes):
        for j, pred in enumerate(classes):
            rows.append({**base, "truth": truth, "pred": pred, "count": int(cm[i, j])})
    return pd.DataFrame(rows)


def linear_normalization_audit(
    X_source_flat: np.ndarray,
    source_sessions: list[str],
    target_session: str,
    protocol: str,
    normalization_weighting: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol": protocol,
                "source_sessions": "+".join(source_sessions),
                "target_session": target_session,
                "phase": "linear_source_fit",
                "statistics_scope": "source_sessions_for_pca_cpca_lda",
                "normalization_weighting": normalization_weighting,
                "n_samples_used_for_stats": int(len(X_source_flat)),
                "target_used_for_stats": False,
                "mean_mean": float(X_source_flat.mean()),
                "std_mean": float(X_source_flat.std()),
                "nan_count": int(np.isnan(X_source_flat).sum()),
                "inf_count": int(np.isinf(X_source_flat).sum()),
            }
        ]
    )


def neural_normalization_audit(
    metadata: dict[str, Any],
    source_sessions: list[str],
    target_session: str,
    protocol: str,
) -> pd.DataFrame:
    rows = []
    for norm in metadata.get("normalization_by_phase", []):
        phase = norm.get("phase")
        if phase == "inner_epoch_selection":
            n_stats = metadata.get("inner_validation", {}).get("n_inner_train")
        else:
            n_stats = metadata.get("final_training", {}).get("n_final_train")
        rows.append(
            {
                "protocol": protocol,
                "source_sessions": "+".join(source_sessions),
                "target_session": target_session,
                "phase": phase,
                "statistics_scope": norm.get("statistics_scope"),
                "normalization_weighting": norm.get("normalization_weighting", metadata.get("normalization_weighting")),
                "n_samples_used_for_stats": int(n_stats or 0),
                "target_used_for_stats": bool(norm.get("target_used_for_stats", False)),
                "mean_mean": norm.get("mean_mean"),
                "std_mean": norm.get("std_mean"),
                "negative_variance_pixels_before_clamp": norm.get("negative_variance_pixels_before_clamp", 0),
                "nan_count": norm.get("train_input_quality", {}).get("nan_count", 0),
                "inf_count": norm.get("train_input_quality", {}).get("inf_count", 0),
            }
        )
    return pd.DataFrame(rows)


def raw_counts_payload(source_sessions: list[str], data_by_session: dict[str, SessionData]) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    raw_samples = session_sample_counts(source_sessions, data_by_session)
    raw_cycles = session_cycle_counts(source_sessions, data_by_session)
    total = max(sum(raw_samples.values()), 1)
    raw_props = {session: float(raw_samples[session] / total) for session in source_sessions}
    return raw_cycles, raw_samples, raw_props


def source_count_fields(
    source_sessions: list[str],
    data_by_session: dict[str, SessionData],
    effective_samples: dict[str, int],
    effective_cycles: dict[str, int] | None = None,
) -> dict[str, Any]:
    raw_cycles, raw_samples, raw_props = raw_counts_payload(source_sessions, data_by_session)
    total_effective = max(sum(int(v) for v in effective_samples.values()), 1)
    effective_props = {session: float(int(effective_samples.get(session, 0)) / total_effective) for session in source_sessions}
    if effective_cycles is None:
        effective_cycles = raw_cycles
    return {
        "raw_source_session_cycle_counts": raw_cycles,
        "raw_source_session_sample_counts": raw_samples,
        "raw_source_session_sample_proportions": raw_props,
        "effective_source_session_cycle_counts": effective_cycles,
        "effective_source_session_sample_counts": effective_samples,
        "effective_source_session_sample_proportions": effective_props,
        "source_session_cycle_counts": raw_cycles,
        "source_session_sample_counts": raw_samples,
        "source_session_sample_proportions": raw_props,
    }


def run_linear(
    method: str,
    source_sessions: list[str],
    target_session: str,
    protocol: str,
    task: str,
    combo_name: str,
    task_dir: Path,
    data_by_session: dict[str, SessionData],
    args: argparse.Namespace,
    repeat_seed: int | None = None,
) -> dict[str, Any]:
    if args.source_balance == "session_balanced":
        if repeat_seed is None:
            raise ValueError("session_balanced linear runs require repeat_seed")
        exp_dir = task_dir / combo_name / method / f"repeat{int(repeat_seed)}"
    else:
        exp_dir = task_dir / combo_name / method
    expected_config = {
        "protocol": protocol,
        "task": task,
        "source_sessions": "+".join(source_sessions),
        "target_session": target_session,
        "method": method,
        "seed": None,
        "repeat_seed": repeat_seed,
        "max_epochs": None,
        "epoch_selection": "not_applicable",
        "source_balance_mode": args.source_balance,
        "clean_margin_s": args.clean_margin_s,
        "window_size": 1,
        "window_mode": "sliding",
        "spatial_filter": "none",
        "normalization_protocol": "strict_inductive_source_only",
        "model_defaults": None,
    }
    if exp_dir.exists():
        if args.reuse_compatible_results and (exp_dir / "summary.json").exists():
            assert_reuse_config_compatible(exp_dir, expected_config)
            return json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        if not any(exp_dir.iterdir()):
            pass
        elif not (exp_dir / "summary.json").exists():
            raise FileExistsError(f"Output directory has partial files but no summary: {exp_dir}")
        else:
            raise FileExistsError(f"Output directory already exists: {exp_dir}")
    else:
        exp_dir.mkdir(parents=True, exist_ok=False)

    selected_cycles_by_session = None
    selection_df = pd.DataFrame()
    if args.source_balance == "session_balanced":
        selected_cycles_by_session, selection_df = select_balanced_cycles(source_sessions, data_by_session, int(repeat_seed))
        selection_df.insert(0, "method", method)
        selection_df.insert(0, "target_session", target_session)
    X_source, y_source, source_groups, source_meta = combine_sources(source_sessions, data_by_session, selected_cycles_by_session)
    target = data_by_session[target_session]
    validate_experiment(source_sessions, target_session, task, data_by_session, source_meta, source_groups)
    assert_complete_binary_cycles(source_sessions, source_meta)
    X_source_flat = preprocess_frames(X_source)
    X_target_flat = preprocess_frames(target.X)
    pred, n_components = fit_predict_linear(
        method_internal(method),
        X_source_flat,
        y_source,
        X_target_flat,
        pca_variance=args.pca_variance,
        standardize=False,
    )
    predicted_class_counts, prediction_is_single_class = prediction_class_summary(pred, target.classes)
    metrics = classification_metrics(target.y, pred)
    base = {
        "protocol": protocol,
        "task": task,
        "source_sessions": "+".join(source_sessions),
        "target_session": target_session,
        "method": method,
        "seed": "",
        "repeat_seed": "" if repeat_seed is None else int(repeat_seed),
        "source_balance_mode": args.source_balance,
    }
    cycle_df = target_cycle_metrics(target.y, pred, target.meta, base)
    cm_df = build_confusion_df(target.y, pred, target.classes, base)
    pred_df = build_prediction_df(target, pred, base)
    norm_df = linear_normalization_audit(
        X_source_flat,
        source_sessions,
        target_session,
        protocol,
        normalization_weighting="sample_weighted",
    )
    split_df = pd.DataFrame(
        [
            {
                "protocol": protocol,
                "source_sessions": "+".join(source_sessions),
                "target_session": target_session,
                "phase": "linear_source_fit",
                "inner_train_sessions": "+".join(source_sessions),
                "inner_val_sessions": "",
                "inner_val_cycles": "",
                "target_in_inner_train": False,
                "target_in_inner_val": False,
                "source_cycle_overlap": False,
            }
        ]
    )
    if bool(norm_df["target_used_for_stats"].any()):
        raise AssertionError("target was used for normalization statistics")
    cycle_ba = cycle_df["balanced_accuracy"]
    effective_samples = {
        session: int((source_meta["session"].astype(str) == session).sum())
        for session in source_sessions
    }
    effective_cycles = {
        session: int(source_meta.loc[source_meta["session"].astype(str) == session, "cycle"].nunique())
        for session in source_sessions
    }
    source_fields = source_count_fields(source_sessions, data_by_session, effective_samples, effective_cycles)
    summary = {
        **base,
        "seed": None,
        "repeat_seed": repeat_seed,
        "source_session": source_sessions[0] if len(source_sessions) == 1 else None,
        "n_source_sessions": len(source_sessions),
        "n_source_samples": int(len(y_source)),
        "n_target_samples": target.n_samples,
        "n_source_cycles": int(len(np.unique(source_groups))),
        "n_target_cycles": target.n_cycles,
        "source_class_counts": class_counts(y_source),
        "target_class_counts": class_counts(target.y),
        **source_fields,
        "raw_source_session_sample_counts": source_fields["raw_source_session_sample_counts"],
        "effective_source_session_draw_counts": effective_samples,
        "effective_source_session_draw_proportions": source_fields["effective_source_session_sample_proportions"],
        "sampling_strategy": "equal_cycle_subsampling" if args.source_balance == "session_balanced" else "pooled_all_no_resampling",
        "sampling_seed": repeat_seed,
        "samples_per_epoch": None,
        "best_epoch": None,
        "inner_validation_score": None,
        "final_trained_epochs": None,
        "n_components": int(n_components),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "target_cycle_ba_mean": float(cycle_ba.mean()),
        "target_cycle_ba_std": float(cycle_ba.std(ddof=1)) if len(cycle_ba) > 1 else 0.0,
        "target_cycle_ba_min": float(cycle_ba.min()),
        "target_cycle_ba_max": float(cycle_ba.max()),
        "normalization_protocol": "strict_inductive_source_only",
        "normalization_weighting": "sample_weighted",
        "target_used_for_stats": False,
        "epoch_selection": "not_applicable",
        "predicted_class_counts": predicted_class_counts,
        "prediction_is_single_class": bool(prediction_is_single_class),
        "config_path": str(exp_dir / "config.json"),
        "prediction_path": str(exp_dir / "predictions.csv"),
        "checkpoint_path": "",
        "status": completed_status(prediction_is_single_class),
        "error_message": "",
        "source_summary_path": str(exp_dir / "summary.json"),
    }
    config = {
        **base,
        "seed": None,
        "repeat_seed": repeat_seed,
        "internal_method": method_internal(method),
        "max_epochs": None,
        "epoch_selection": "not_applicable",
        "source_balance_mode": args.source_balance,
        "model_defaults": None,
        "pca_variance": args.pca_variance,
        "clean_middle": True,
        "clean_margin_s": args.clean_margin_s,
        "trim_incomplete_cycles": True,
        "analysis_limit": None,
        "spatial_filter": "none",
        "normalization_protocol": "strict_inductive_source_only",
        "window_size": 1,
        "window_mode": "sliding",
    }
    write_common_outputs(
        exp_dir,
        config,
        summary,
        summary,
        cycle_df,
        pred_df,
        cm_df,
        norm_df,
        pd.DataFrame(),
        split_df,
    )
    if args.source_balance == "session_balanced":
        selection_df.to_csv(exp_dir / "linear_balanced_cycle_selection.csv", index=False)
    return summary


def run_neural(
    method: str,
    seed: int,
    source_sessions: list[str],
    target_session: str,
    protocol: str,
    task: str,
    combo_name: str,
    task_dir: Path,
    data_by_session: dict[str, SessionData],
    args: argparse.Namespace,
) -> dict[str, Any]:
    exp_dir = task_dir / combo_name / method / f"seed{seed}"
    model_defaults = get_torch_model_defaults(method_internal(method)).__dict__
    expected_config = {
        "protocol": protocol,
        "task": task,
        "source_sessions": "+".join(source_sessions),
        "target_session": target_session,
        "method": method,
        "seed": int(seed),
        "repeat_seed": None,
        "max_epochs": args.max_epochs,
        "epoch_selection": args.epoch_selection,
        "source_balance_mode": args.source_balance,
        "clean_margin_s": args.clean_margin_s,
        "window_size": 1,
        "window_mode": "sliding",
        "spatial_filter": "none",
        "normalization_protocol": "strict_inductive_source_only",
        "model_defaults": model_defaults,
    }
    if exp_dir.exists():
        if args.reuse_compatible_results and (exp_dir / "summary.json").exists():
            assert_reuse_config_compatible(exp_dir, expected_config)
            return json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        if not any(exp_dir.iterdir()):
            pass
        elif not (exp_dir / "summary.json").exists():
            raise FileExistsError(f"Output directory has partial files but no summary: {exp_dir}")
        else:
            raise FileExistsError(f"Output directory already exists: {exp_dir}")
    else:
        exp_dir.mkdir(parents=True, exist_ok=False)
    if not torch_available():
        raise RuntimeError("PyTorch is not installed")
    if method == "fcnn_berthon2023" and method_internal(method) != "fcnn":
        raise AssertionError("FCNN alias did not resolve to the frozen fcnn implementation")

    X_source, y_source, source_groups, source_meta = combine_sources(source_sessions, data_by_session)
    target = data_by_session[target_session]
    validate_experiment(source_sessions, target_session, task, data_by_session, source_meta, source_groups)
    assert_complete_binary_cycles(source_sessions, source_meta)
    checkpoint_path = exp_dir / "checkpoint.pt"
    patience = args.max_epochs if args.epoch_selection == "source-inner-validation" else None
    result = fit_predict_torch(
        method_internal(method),
        X_source,
        y_source,
        target.X,
        classes=np.asarray(EXPECTED_CLASSES[task]),
        max_epochs=args.max_epochs,
        seed=int(seed),
        train_groups=source_groups,
        patience=patience,
        device=args.device,
        checkpoint_path=str(checkpoint_path),
        return_metadata=True,
        train_session_labels=source_meta["session"].astype(str).to_numpy(),
        source_balance_mode=args.source_balance,
    )
    pred = result.predictions
    metadata = result.metadata
    if metadata.get("final_training", {}).get("retrained_on_full_outer_train") is not True:
        raise AssertionError("final neural training did not retrain from a fresh full-source model")
    if metadata.get("method") != method_internal(method):
        raise AssertionError(f"wrong neural model executed: {metadata.get('method')}")
    predicted_class_counts, prediction_is_single_class = prediction_class_summary(pred, target.classes)
    metrics = classification_metrics(target.y, pred)
    base = {
        "protocol": protocol,
        "task": task,
        "source_sessions": "+".join(source_sessions),
        "target_session": target_session,
        "method": method,
        "seed": int(seed),
        "repeat_seed": "",
        "source_balance_mode": args.source_balance,
    }
    cycle_df = target_cycle_metrics(target.y, pred, target.meta, base)
    cm_df = build_confusion_df(target.y, pred, target.classes, base)
    pred_df = build_prediction_df(target, pred, base)
    norm_df = neural_normalization_audit(metadata, source_sessions, target_session, protocol)
    if norm_df.empty or bool(norm_df["target_used_for_stats"].any()):
        raise AssertionError("normalization audit failed target-used check")
    split_df = pd.DataFrame(
        split_audit_rows(
            source_sessions,
            target_session,
            protocol,
            source_groups,
            metadata,
            epoch_selection=args.epoch_selection,
        )
    )
    history_df = pd.DataFrame(metadata.get("training_history", []))
    if not history_df.empty:
        history_df.insert(0, "seed", int(seed))
        history_df.insert(0, "method", method)
    sampling_df = pd.DataFrame(metadata.get("training_session_sampling_audit", []))
    if not sampling_df.empty:
        sampling_df.insert(0, "seed", int(seed))
        sampling_df.insert(0, "method", method)
        sampling_df.insert(0, "target_session", target_session)
        sampling_df["draw_proportion"] = pd.to_numeric(sampling_df["draw_proportion"], errors="raise")
        final_sampling = sampling_df[sampling_df["phase"].astype(str).str.startswith("full_outer_train")]
        if args.source_balance == "session_balanced" and not final_sampling.empty:
            for epoch, epoch_rows in final_sampling.groupby("epoch", sort=True):
                if len(set(epoch_rows["source_session"].astype(str))) != len(source_sessions):
                    raise AssertionError(f"{combo_name} {method} seed={seed} epoch={epoch}: missing source in balanced sampler audit")
                max_delta = float((epoch_rows["draw_proportion"] - 0.5).abs().max())
                if max_delta > (1.0 / max(int(metadata.get("samples_per_epoch") or len(y_source)), 1)):
                    raise AssertionError(f"{combo_name} {method} seed={seed} epoch={epoch}: draw proportions not near 0.5")
    cycle_ba = cycle_df["balanced_accuracy"]
    raw_counts = source_count_fields(
        source_sessions,
        data_by_session,
        effective_samples={
            session: int(metadata.get("effective_source_session_draw_counts", {}).get(session, data_by_session[session].n_samples))
            for session in source_sessions
        },
    )
    summary = {
        **base,
        "source_session": source_sessions[0] if len(source_sessions) == 1 else None,
        "n_source_sessions": len(source_sessions),
        "n_source_samples": int(len(y_source)),
        "n_target_samples": target.n_samples,
        "n_source_cycles": int(len(np.unique(source_groups))),
        "n_target_cycles": target.n_cycles,
        "source_class_counts": class_counts(y_source),
        "target_class_counts": class_counts(target.y),
        **raw_counts,
        "raw_source_session_sample_counts": raw_counts["raw_source_session_sample_counts"],
        "effective_source_session_draw_counts": metadata.get("effective_source_session_draw_counts", {}),
        "effective_source_session_draw_proportions": metadata.get("effective_source_session_draw_proportions", {}),
        "sampling_strategy": metadata.get("sampling_strategy"),
        "sampling_seed": metadata.get("sampling_seed"),
        "samples_per_epoch": metadata.get("samples_per_epoch"),
        "best_epoch": metadata.get("best_epoch"),
        "inner_validation_score": metadata.get("best_val_balanced_accuracy"),
        "final_trained_epochs": metadata.get("final_trained_epochs"),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "target_cycle_ba_mean": float(cycle_ba.mean()),
        "target_cycle_ba_std": float(cycle_ba.std(ddof=1)) if len(cycle_ba) > 1 else 0.0,
        "target_cycle_ba_min": float(cycle_ba.min()),
        "target_cycle_ba_max": float(cycle_ba.max()),
        "normalization_protocol": "strict_inductive_source_only",
        "normalization_weighting": metadata.get("normalization_weighting"),
        "target_used_for_stats": False,
        "epoch_selection": args.epoch_selection,
        "predicted_class_counts": predicted_class_counts,
        "prediction_is_single_class": bool(prediction_is_single_class),
        "config_path": str(exp_dir / "config.json"),
        "prediction_path": str(exp_dir / "predictions.csv"),
        "checkpoint_path": str(checkpoint_path),
        "status": completed_status(prediction_is_single_class),
        "error_message": "",
        "source_summary_path": str(exp_dir / "summary.json"),
    }
    config = {
        **base,
        "internal_method": method_internal(method),
        "torch_defaults": model_defaults,
        "model_defaults": model_defaults,
        "max_epochs": args.max_epochs,
        "epoch_selection": args.epoch_selection,
        "source_balance_mode": args.source_balance,
        "patience_for_source_epoch_selection": patience,
        "device": args.device,
        "clean_middle": True,
        "clean_margin_s": args.clean_margin_s,
        "trim_incomplete_cycles": True,
        "analysis_limit": None,
        "spatial_filter": "none",
        "normalization_protocol": "strict_inductive_source_only",
        "window_size": 1,
        "window_mode": "sliding",
    }
    write_json(exp_dir / "config.json", config)
    write_json(exp_dir / "summary.json", summary)
    pd.DataFrame([summary]).to_csv(exp_dir / "overall_metrics.csv", index=False)
    cycle_df.to_csv(exp_dir / "target_cycle_metrics.csv", index=False)
    pred_df.to_csv(exp_dir / "predictions.csv", index=False)
    cm_df.to_csv(exp_dir / "confusion_matrix.csv", index=False)
    norm_df.to_csv(exp_dir / "normalization_audit.csv", index=False)
    history_df.to_csv(exp_dir / "training_history.csv", index=False)
    split_df.to_csv(exp_dir / "split_audit.csv", index=False)
    sampling_df.to_csv(exp_dir / "training_session_sampling_audit.csv", index=False)
    return summary


def dry_run(args: argparse.Namespace, run_dir: Path) -> None:
    total_neural = 0
    total_linear = 0
    for task in args.tasks:
        data = {session: load_session(session, task, args.clean_margin_s) for session in args.sessions}
        print(f"task={task} loaded_sessions")
        for session in args.sessions:
            expected_cycles = EXPECTED_STRONG_BINARY_CYCLES.get(session) if task == "binary" else None
            expected_note = "" if expected_cycles is None else f" expected_cycles={expected_cycles}"
            print(
                f"  session={session} actual_cycles={data[session].n_cycles} "
                f"actual_samples={data[session].n_samples}{expected_note}"
            )
        specs = experiment_specs(args.sessions, args.protocol)
        print(f"task={task} protocol={args.protocol} source_balance={args.source_balance} combinations={len(specs)}")
        for source_sessions, target_session, combo_name in specs:
            X_source, y_source, source_groups, source_meta = combine_sources(source_sessions, data)
            validate_experiment(source_sessions, target_session, task, data, source_meta, source_groups)
            target = data[target_session]
            raw_samples = session_sample_counts(source_sessions, data)
            raw_cycles = session_cycle_counts(source_sessions, data)
            raw_props = source_session_proportions(source_sessions, data)
            print(
                f"{combo_name}: source_samples={len(y_source)} target_samples={target.n_samples} "
                f"source_cycles={len(np.unique(source_groups))} target_cycles={target.n_cycles} "
                f"raw_source_cycles={raw_cycles} raw_source_samples={raw_samples} "
                f"raw_source_sample_proportions={raw_props} "
                f"methods={','.join(args.methods)} neural_seeds={','.join(map(str, args.seeds))}"
            )
            if args.source_balance == "session_balanced":
                n_balanced_cycles = min(data[session].n_cycles for session in source_sessions)
                balanced_samples = {session: int(n_balanced_cycles * 16) for session in source_sessions}
                samples_per_epoch = int(sum(raw_samples.values()))
                per_source_draws = {
                    session: int(samples_per_epoch / max(len(source_sessions), 1))
                    for session in source_sessions
                }
                print(
                    f"  linear_balanced_cycles_per_source={n_balanced_cycles} "
                    f"linear_selected_samples={balanced_samples}"
                )
                print(
                    f"  neural_samples_per_epoch={samples_per_epoch} "
                    f"neural_expected_draws_per_source={per_source_draws} "
                    "normalization_weighting=session_equal"
                )
        neural_methods = [method for method in args.methods if method in NEURAL_METHODS]
        linear_methods = [method for method in args.methods if method in LINEAR_METHODS]
        total_neural += len(specs) * len(neural_methods) * len(args.seeds)
        linear_repeats = len(args.seeds) if args.source_balance == "session_balanced" else 1
        total_linear += len(specs) * len(linear_methods) * linear_repeats
    print(f"estimated_neural_training_runs={total_neural}")
    print(f"estimated_linear_runs={total_linear}")
    print(f"output_dir={run_dir}")
    print("existing_compatible_results=checked only during non-dry-run with --reuse-compatible-results")
    print("conflicts_or_incomplete_results=none detected during dry-run path construction")


def summarize_and_plot(run_dir: Path, protocol: str, tasks: list[str], sessions: list[str]) -> None:
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in run_dir.glob("*/**/summary.json"):
        if "/aggregate/" in str(path):
            continue
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    master = pd.DataFrame(summaries)
    if master.empty:
        return
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "target_cycle_ba_mean", "target_cycle_ba_std"]
    if master[metric_cols].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise AssertionError("Completed summaries contain NaN or Inf metrics")
    master.to_csv(aggregate_dir / "cross_session_master_summary.csv", index=False)
    method_summary = (
        master.groupby(["protocol", "task", "source_sessions", "target_session", "method"], dropna=False)
        .agg(
            n_seeds=("seed", lambda x: int(pd.Series(x).replace("", np.nan).dropna().nunique()) or 1),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            target_cycle_ba_mean_across_seeds=("target_cycle_ba_mean", "mean"),
            target_cycle_ba_std_mean_across_seeds=("target_cycle_ba_std", "mean"),
            seed_std_of_target_cycle_ba_mean=(
                "target_cycle_ba_mean",
                lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0,
            ),
        )
        .reset_index()
    )
    method_summary.to_csv(aggregate_dir / "cross_session_method_summary.csv", index=False)

    target_rows = []
    for (proto, task, target, method), group in method_summary.groupby(["protocol", "task", "target_session", "method"], sort=True):
        source_means = group.groupby("source_sessions")["balanced_accuracy_mean"].mean().sort_values()
        target_rows.append(
            {
                "protocol": proto,
                "task": task,
                "target_session": target,
                "method": method,
                "mean_transfer_balanced_accuracy": float(source_means.mean()),
                "std_across_source_sessions": float(source_means.std(ddof=1)) if len(source_means) > 1 else 0.0,
                "best_source_session": str(source_means.index[-1]),
                "worst_source_session": str(source_means.index[0]),
                "number_of_source_sessions_above_chance": int((source_means > 0.5).sum()),
                "number_below_chance": int((source_means < 0.5).sum()),
            }
        )
    pd.DataFrame(target_rows).to_csv(aggregate_dir / "cross_session_target_summary.csv", index=False)

    source_rows = []
    pairwise_method = method_summary[method_summary["source_sessions"].astype(str).str.count(r"\+") == 0]
    for (proto, task, source, method), group in pairwise_method.groupby(["protocol", "task", "source_sessions", "method"], sort=True):
        target_means = group.groupby("target_session")["balanced_accuracy_mean"].mean().sort_values()
        source_rows.append(
            {
                "protocol": proto,
                "task": task,
                "source_session": source,
                "method": method,
                "mean_transfer_balanced_accuracy": float(target_means.mean()),
                "std_across_targets": float(target_means.std(ddof=1)) if len(target_means) > 1 else 0.0,
                "best_target": str(target_means.index[-1]),
                "worst_target": str(target_means.index[0]),
                "number_of_targets_above_chance": int((target_means > 0.5).sum()),
            }
        )
    pd.DataFrame(source_rows).to_csv(aggregate_dir / "cross_session_source_summary.csv", index=False)
    write_completeness_report(aggregate_dir, master, protocol, tasks, sessions)
    write_source_composition_audit(aggregate_dir, master)
    write_transfer_gap(aggregate_dir, method_summary)
    write_figures(aggregate_dir, method_summary, run_dir, protocol, tasks, sessions)
    write_findings(aggregate_dir, method_summary)
    write_neural_training_audit(aggregate_dir, run_dir)
    write_fixed40_comparison(aggregate_dir, run_dir, master)


def write_source_composition_audit(aggregate_dir: Path, master: pd.DataFrame) -> None:
    def parse_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return {}
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        return {}

    rows = []
    for _, summary in master.iterrows():
        source_sessions = [value for value in str(summary.get("source_sessions", "")).split("+") if value]
        raw_cycles = parse_payload(summary.get("raw_source_session_cycle_counts"))
        raw_samples = parse_payload(summary.get("raw_source_session_sample_counts"))
        raw_props = parse_payload(summary.get("raw_source_session_sample_proportions"))
        effective_cycles = parse_payload(summary.get("effective_source_session_cycle_counts"))
        effective_samples = summary.get("effective_source_session_draw_counts")
        if effective_samples is None or (isinstance(effective_samples, float) and np.isnan(effective_samples)):
            effective_samples = summary.get("effective_source_session_sample_counts", {})
        effective_samples = parse_payload(effective_samples)
        effective_props = summary.get("effective_source_session_draw_proportions")
        if effective_props is None or (isinstance(effective_props, float) and np.isnan(effective_props)):
            effective_props = summary.get("effective_source_session_sample_proportions", {})
        effective_props = parse_payload(effective_props)
        class_counts_payload = parse_payload(summary.get("source_class_counts", {}))
        seed_or_repeat = summary.get("repeat_seed")
        if pd.isna(seed_or_repeat) or seed_or_repeat == "":
            seed_or_repeat = summary.get("seed")
        for source in source_sessions:
            rows.append(
                {
                    "target_session": summary.get("target_session"),
                    "method": summary.get("method"),
                    "source_session": source,
                    "raw_n_cycles": int(raw_cycles.get(source, 0)),
                    "raw_n_samples": int(raw_samples.get(source, 0)),
                    "raw_sample_proportion": float(raw_props.get(source, 0.0)),
                    "effective_n_cycles": int(effective_cycles.get(source, 0)),
                    "effective_n_sample_draws": int(effective_samples.get(source, 0)),
                    "effective_sample_proportion": float(effective_props.get(source, 0.0)),
                    "source_balance_mode": summary.get("source_balance_mode", "pooled_all"),
                    "seed_or_repeat": "" if pd.isna(seed_or_repeat) else seed_or_repeat,
                    "class_counts": json.dumps(class_counts_payload, sort_keys=True),
                }
            )
    pd.DataFrame(rows).to_csv(aggregate_dir / "source_composition_audit.csv", index=False)


def write_completeness_report(aggregate_dir: Path, master: pd.DataFrame, protocol: str, tasks: list[str], sessions: list[str]) -> None:
    rows = []
    specs = experiment_specs(sessions, protocol)
    for task, (source_sessions, target, _), method in itertools.product(tasks, specs, ALL_METHODS):
        if method not in set(master["method"]):
            continue
        source_balance_values = set(master.get("source_balance_mode", pd.Series(["pooled_all"])).dropna().astype(str))
        source_balance = next(iter(source_balance_values)) if len(source_balance_values) == 1 else "pooled_all"
        expected_ids = [0, 1, 2] if (method in NEURAL_METHODS or source_balance == "session_balanced") else [None]
        for seed in expected_ids:
            if method in LINEAR_METHODS and source_balance == "session_balanced":
                seed_mask = pd.to_numeric(master.get("repeat_seed", pd.Series(dtype=float)), errors="coerce").eq(float(seed))
            else:
                seed_mask = (
                    master["seed"].isna()
                    if seed is None
                    else pd.to_numeric(master["seed"], errors="coerce").eq(float(seed))
                )
            mask = (
                (master["task"] == task)
                & (master["source_sessions"] == "+".join(source_sessions))
                & (master["target_session"].astype(str) == str(target))
                & (master["method"] == method)
                & seed_mask
            )
            observed_statuses = set(master.loc[mask, "status"]) if mask.any() else set()
            completed_statuses = {"completed", "completed_degenerate_prediction"}
            rows.append(
                {
                    "protocol": protocol,
                    "task": task,
                    "source_sessions": "+".join(source_sessions),
                    "target_session": target,
                    "method": method,
                    "seed_or_repeat": "" if seed is None else seed,
                    "present": bool(mask.any()),
                    "target_used_for_stats": bool(master.loc[mask, "target_used_for_stats"].astype(bool).any()) if mask.any() else False,
                    "has_nan_metric": bool(master.loc[mask, ["accuracy", "balanced_accuracy", "macro_f1"]].isna().any().any()) if mask.any() else False,
                    "status": (
                        "+".join(sorted(observed_statuses))
                        if mask.any() and observed_statuses.issubset(completed_statuses)
                        else "missing"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(aggregate_dir / "cross_session_completeness_report.csv", index=False)


def write_transfer_gap(aggregate_dir: Path, method_summary: pd.DataFrame) -> None:
    benchmark = PROJECT_DIR / "results" / "model_batch_test" / "benchmark" / "summary.csv"
    if not benchmark.exists():
        pd.DataFrame().to_csv(aggregate_dir / "transfer_gap_summary.csv", index=False)
        return
    within = pd.read_csv(benchmark)
    within["method"] = within["method"].replace({"fcnn": "fcnn_berthon2023"})
    within_mean = (
        within.groupby(["session", "task", "method"], as_index=False)
        .agg(within_target_session_ba=("balanced_accuracy", "mean"))
    )
    rows = []
    pairwise = method_summary[method_summary["source_sessions"].astype(str).str.count(r"\+") == 0]
    for _, row in pairwise.iterrows():
        match = within_mean[
            (within_mean["session"].astype(str) == str(row["target_session"]))
            & (within_mean["task"] == row["task"])
            & (within_mean["method"] == row["method"])
        ]
        if match.empty:
            continue
        within_ba = float(match.iloc[0]["within_target_session_ba"])
        cross_ba = float(row["balanced_accuracy_mean"])
        rows.append(
            {
                "task": row["task"],
                "source_session": row["source_sessions"],
                "target_session": row["target_session"],
                "method": row["method"],
                "within_target_session_ba": within_ba,
                "cross_session_ba": cross_ba,
                "transfer_gap_target_reference": within_ba - cross_ba,
            }
        )
    pd.DataFrame(rows).to_csv(aggregate_dir / "transfer_gap_summary.csv", index=False)


def write_figures(aggregate_dir: Path, method_summary: pd.DataFrame, run_dir: Path, protocol: str, tasks: list[str], sessions: list[str]) -> None:
    try:
        os.environ["MPLCONFIGDIR"] = str(aggregate_dir / "matplotlib_cache")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        (aggregate_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")
        return

    if protocol == "pairwise":
        for task in tasks:
            for method in sorted(method_summary["method"].unique()):
                subset = method_summary[(method_summary["task"] == task) & (method_summary["method"] == method)]
                matrix = pd.DataFrame(np.nan, index=sessions, columns=sessions)
                for _, row in subset.iterrows():
                    source = str(row["source_sessions"])
                    target = str(row["target_session"])
                    if source in matrix.index and target in matrix.columns:
                        matrix.loc[source, target] = row["balanced_accuracy_mean"]
                fig, ax = plt.subplots(figsize=(6, 5))
                im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(sessions)), sessions)
                ax.set_yticks(range(len(sessions)), sessions)
                ax.set_xlabel("target session")
                ax.set_ylabel("source session")
                ax.set_title(f"{task} {method} direct transfer BA")
                for i, source in enumerate(sessions):
                    for j, target in enumerate(sessions):
                        value = matrix.loc[source, target]
                        if pd.notna(value):
                            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.65 else "black")
                fig.colorbar(im, ax=ax, label="balanced accuracy")
                fig.tight_layout()
                fig.savefig(aggregate_dir / f"pairwise_{task}_{method}_heatmap.png", dpi=160)
                plt.close(fig)

        gap_path = aggregate_dir / "transfer_gap_summary.csv"
        if gap_path.exists():
            gap = pd.read_csv(gap_path)
            for task in tasks:
                task_gap = gap[gap["task"] == task]
                if task_gap.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4))
                labels = []
                within_vals = []
                cross_vals = []
                for (target, method), group in task_gap.groupby(["target_session", "method"], sort=True):
                    labels.append(f"{target}\n{method}")
                    within_vals.append(float(group["within_target_session_ba"].mean()))
                    cross_vals.append(float(group["cross_session_ba"].mean()))
                x = np.arange(len(labels))
                ax.bar(x - 0.18, within_vals, width=0.36, label="within-session CV")
                ax.bar(x + 0.18, cross_vals, width=0.36, label="pairwise transfer mean")
                ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
                ax.set_ylim(0, 1)
                ax.set_ylabel("balanced accuracy")
                ax.set_xticks(x, labels, rotation=45, ha="right")
                ax.legend()
                fig.tight_layout()
                fig.savefig(aggregate_dir / f"pairwise_{task}_transfer_gap.png", dpi=160)
                plt.close(fig)

        for task in tasks:
            cycle_files = list((run_dir / task).glob("*/*/target_cycle_metrics.csv"))
            frames = [pd.read_csv(path) for path in cycle_files]
            cycle_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if cycle_all.empty:
                continue
            for target in sessions:
                subset = cycle_all[cycle_all["target_session"].astype(str) == str(target)]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(8, 4))
                for (source, method), group in subset.groupby(["source_sessions", "method"], sort=True):
                    grouped = group.groupby("target_cycle")["balanced_accuracy"].mean().reset_index()
                    ax.plot(
                        grouped["target_cycle"],
                        grouped["balanced_accuracy"],
                        marker="o",
                        linewidth=1,
                        label=f"{source}->{target} {method}",
                    )
                ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
                ax.set_ylim(0, 1)
                ax.set_xlabel("target cycle")
                ax.set_ylabel("balanced accuracy")
                ax.set_title(f"{task} target-cycle stability: target {target}")
                ax.legend(fontsize=6, ncol=2)
                fig.tight_layout()
                fig.savefig(aggregate_dir / f"pairwise_{task}_target_{target}_cycle_stability.png", dpi=160)
                plt.close(fig)

        asym_rows = []
        for task in tasks:
            subset = method_summary[method_summary["task"] == task]
            for method in sorted(subset["method"].unique()):
                by_pair = {
                    (str(row["source_sessions"]), str(row["target_session"])): float(row["balanced_accuracy_mean"])
                    for _, row in subset[subset["method"] == method].iterrows()
                }
                for a, b in itertools.combinations(sessions, 2):
                    if (a, b) in by_pair and (b, a) in by_pair:
                        asym_rows.append(
                            {
                                "task": task,
                                "method": method,
                                "session_pair": f"{a}<->{b}",
                                "ba_a_to_b": by_pair[(a, b)],
                                "ba_b_to_a": by_pair[(b, a)],
                                "asymmetry_a_to_b_minus_b_to_a": by_pair[(a, b)] - by_pair[(b, a)],
                            }
                        )
        asym_df = pd.DataFrame(asym_rows)
        asym_df.to_csv(aggregate_dir / "pairwise_asymmetry.csv", index=False)
    elif protocol == "loso":
        for task in tasks:
            subset = method_summary[method_summary["task"] == task]
            if subset.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            width = 0.8 / max(len(subset["method"].unique()), 1)
            x = np.arange(len(sessions))
            for i, method in enumerate(sorted(subset["method"].unique())):
                vals = []
                for session in sessions:
                    match = subset[(subset["method"] == method) & (subset["target_session"].astype(str) == str(session))]
                    vals.append(float(match["balanced_accuracy_mean"].mean()) if not match.empty else np.nan)
                ax.bar(x + i * width, vals, width=width, label=method)
            ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
            ax.set_xticks(x + width, sessions)
            ax.set_ylim(0, 1)
            ax.set_ylabel("balanced accuracy")
            ax.set_xlabel("held-out target session")
            ax.legend()
            fig.tight_layout()
            fig.savefig(aggregate_dir / f"loso_{task}_balanced_accuracy.png", dpi=160)
            plt.close(fig)


def write_findings(aggregate_dir: Path, method_summary: pd.DataFrame) -> None:
    lines = [
        "# Cross-session findings",
        "",
        "Balanced accuracy is the primary metric. Pairwise direct transfer and LOSO are interpreted separately, and no target-session data were used for fitting, epoch selection, or normalization statistics in completed rows.",
        "",
    ]
    if method_summary.empty:
        lines.append("No completed runs were available.")
    else:
        top = method_summary.sort_values("balanced_accuracy_mean", ascending=False).head(10)
        lines.append("## Top completed transfers")
        for _, row in top.iterrows():
            lines.append(
                f"- {row['protocol']} {row['task']} {row['source_sessions']} -> {row['target_session']} "
                f"{row['method']}: BA={row['balanced_accuracy_mean']:.3f}"
            )
        near_chance = method_summary["balanced_accuracy_mean"].between(0.45, 0.55).mean()
        lines.extend(
            [
                "",
                "## Interpretation guardrails",
                f"- Fraction of method-level transfers in the 0.45-0.55 chance band: {near_chance:.2f}.",
                "- Values below 0.5 are not interpreted as reverse coding without a separate test.",
                "- CNN/fCNN conclusions should use seed means and standard deviations, not best single seeds.",
                "- If direct transfer is weak across methods, the first explanation to consider is session distribution shift, imaging-position difference, or spatial misalignment rather than a new model architecture.",
            ]
        )
    (aggregate_dir / "cross_session_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_prediction_counts(path: Path, classes: list[str]) -> tuple[dict[str, int], bool]:
    pred_df = pd.read_csv(path)
    counts = {label: int((pred_df["pred"] == label).sum()) for label in classes}
    return counts, sum(count > 0 for count in counts.values()) == 1


def _phase_epoch_count(history: pd.DataFrame, phase_prefix: str) -> int | None:
    if history.empty or "phase" not in history or "epoch" not in history:
        return None
    phase_rows = history[history["phase"].astype(str).str.startswith(phase_prefix)]
    if phase_rows.empty:
        return None
    return int(phase_rows["epoch"].max())


def collect_neural_training_audit(run_dir: Path) -> pd.DataFrame:
    rows = []
    for summary_path in sorted(run_dir.glob("binary/*/*/seed*/summary.json")):
        method = summary_path.parents[1].name
        if method not in NEURAL_METHODS:
            continue
        seed_dir = summary_path.parent
        combo_name = summary_path.parents[2].name
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "_to_" in combo_name:
            source_session, target_session = combo_name.split("_to_", 1)
        else:
            source_session = str(summary.get("source_sessions", ""))
            target_session = str(summary.get("target_session", "") or combo_name.replace("target_", ""))
        predictions_path = seed_dir / "predictions.csv"
        history_path = seed_dir / "training_history.csv"
        norm_path = seed_dir / "normalization_audit.csv"
        split_path = seed_dir / "split_audit.csv"

        predicted_counts, single_class = _read_prediction_counts(predictions_path, EXPECTED_CLASSES["binary"])
        history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
        norm = pd.read_csv(norm_path) if norm_path.exists() else pd.DataFrame()
        split = pd.read_csv(split_path) if split_path.exists() else pd.DataFrame()

        selection_epochs = summary.get("selection_trained_epochs")
        if selection_epochs is None:
            selection_epochs = _phase_epoch_count(history, "inner_epoch_selection")
        final_epochs = summary.get("final_trained_epochs")
        if final_epochs is None:
            final_epochs = _phase_epoch_count(history, "full_outer_train")
        target_used = bool(norm["target_used_for_stats"].astype(bool).any()) if "target_used_for_stats" in norm else None
        scopes = (
            ";".join(str(value) for value in norm["statistics_scope"].dropna().unique())
            if "statistics_scope" in norm
            else ""
        )
        inner_val_cycles = ""
        inner_train_cycles = ""
        if not split.empty:
            inner_val_cycles = str(split.get("inner_val_cycles", pd.Series([""])).fillna("").iloc[0])
            inner_train_cycles = str(split.get("inner_train_cycles", pd.Series([""])).fillna("").iloc[0])
        if not inner_train_cycles:
            n_source_cycles = int(summary.get("n_source_cycles") or 0)
            all_cycles = [f"{source_session}_cycle{i}" for i in range(n_source_cycles)]
            val_set = {value for value in inner_val_cycles.split("+") if value}
            inner_train_cycles = "+".join(cycle for cycle in all_cycles if cycle not in val_set)

        rows.append(
            {
                "source_session": source_session,
                "target_session": target_session,
                "method": method,
                "seed": int(summary.get("seed")),
                "best_epoch": summary.get("best_epoch"),
                "inner_validation_score": summary.get("inner_validation_score"),
                "selection_trained_epochs": selection_epochs if selection_epochs is not None else 0,
                "final_trained_epochs": final_epochs,
                "predicted_no_stimulus_count": predicted_counts.get("no_stimulus", 0),
                "predicted_stimulus_count": predicted_counts.get("stimulus", 0),
                "prediction_is_single_class": bool(single_class),
                "target_used_for_stats": target_used,
                "inner_train_cycles": inner_train_cycles,
                "inner_val_cycles": inner_val_cycles,
                "normalization_statistics_scope": scopes,
                "status": summary.get("status"),
            }
        )
    return pd.DataFrame(rows)


def write_neural_training_audit(aggregate_dir: Path, run_dir: Path) -> None:
    audit = collect_neural_training_audit(run_dir)
    if audit.empty:
        return
    audit.to_csv(aggregate_dir / "neural_training_audit.csv", index=False)
    best_epoch_one = int((pd.to_numeric(audit["best_epoch"], errors="coerce") == 1).sum())
    low_final = int((pd.to_numeric(audit["final_trained_epochs"], errors="coerce") <= 3).sum())
    single_class = int(audit["prediction_is_single_class"].astype(bool).sum())
    single_low = int(
        (
            audit["prediction_is_single_class"].astype(bool)
            & (pd.to_numeric(audit["final_trained_epochs"], errors="coerce") <= 3)
        ).sum()
    )
    target_stats_all_false = not bool(audit["target_used_for_stats"].fillna(False).astype(bool).any())
    lines = [
        "# Neural training audit findings",
        "",
        f"1. best_epoch=1 runs: {best_epoch_one} / {len(audit)}.",
        f"2. final_trained_epochs<=3 runs: {low_final} / {len(audit)}.",
        f"3. single-class target predictions: {single_class} / {len(audit)}.",
        (
            "4. Single-class predictions with final_trained_epochs<=3: "
            f"{single_low} / {single_class if single_class else 0}; "
            f"non-single low-epoch runs: {low_final - single_low}."
        ),
        f"5. target_used_for_stats all false: {target_stats_all_false}.",
    ]
    (aggregate_dir / "neural_training_audit_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fixed40_comparison(aggregate_dir: Path, run_dir: Path, fixed_master: pd.DataFrame) -> None:
    if "epoch_selection" not in fixed_master or set(fixed_master["epoch_selection"].dropna()) != {"fixed"}:
        return
    inner_dir = run_dir.parent / "pairwise_strong_sessions_v1"
    if not inner_dir.exists():
        return
    inner_summaries = []
    for path in sorted(inner_dir.glob("binary/*/*/seed*/summary.json")):
        method = path.parents[1].name
        if method in NEURAL_METHODS:
            row = json.loads(path.read_text(encoding="utf-8"))
            row["_summary_path"] = str(path)
            inner_summaries.append(row)
    inner = pd.DataFrame(inner_summaries)
    if inner.empty:
        return
    inner_audit = collect_neural_training_audit(inner_dir)
    fixed_audit = collect_neural_training_audit(run_dir)
    rows = []
    for _, fixed_row in fixed_master.iterrows():
        key_mask = (
            (inner["source_sessions"].astype(str) == str(fixed_row["source_sessions"]))
            & (inner["target_session"].astype(str) == str(fixed_row["target_session"]))
            & (inner["method"] == fixed_row["method"])
            & (pd.to_numeric(inner["seed"], errors="coerce") == float(fixed_row["seed"]))
        )
        if not key_mask.any():
            continue
        inner_row = inner.loc[key_mask].iloc[0]
        audit_mask = (
            (inner_audit["source_session"].astype(str) == str(fixed_row["source_sessions"]))
            & (inner_audit["target_session"].astype(str) == str(fixed_row["target_session"]))
            & (inner_audit["method"] == fixed_row["method"])
            & (pd.to_numeric(inner_audit["seed"], errors="coerce") == float(fixed_row["seed"]))
        )
        fixed_audit_mask = (
            (fixed_audit["source_session"].astype(str) == str(fixed_row["source_sessions"]))
            & (fixed_audit["target_session"].astype(str) == str(fixed_row["target_session"]))
            & (fixed_audit["method"] == fixed_row["method"])
            & (pd.to_numeric(fixed_audit["seed"], errors="coerce") == float(fixed_row["seed"]))
        )
        rows.append(
            {
                "source_session": fixed_row["source_sessions"],
                "target_session": fixed_row["target_session"],
                "method": fixed_row["method"],
                "seed": int(fixed_row["seed"]),
                "inner_selected_best_epoch": inner_row.get("best_epoch"),
                "inner_selection_ba": inner_row.get("balanced_accuracy"),
                "fixed40_ba": fixed_row.get("balanced_accuracy"),
                "inner_selection_macro_f1": inner_row.get("macro_f1"),
                "fixed40_macro_f1": fixed_row.get("macro_f1"),
                "inner_prediction_single_class": bool(inner_audit.loc[audit_mask, "prediction_is_single_class"].iloc[0])
                if audit_mask.any()
                else None,
                "fixed40_prediction_single_class": bool(fixed_audit.loc[fixed_audit_mask, "prediction_is_single_class"].iloc[0])
                if fixed_audit_mask.any()
                else None,
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(aggregate_dir / "fixed40_vs_inner_selection.csv", index=False)
    if comparison.empty:
        return
    direction_summary = (
        comparison.groupby(["source_session", "target_session", "method"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            inner_selection_ba_mean=("inner_selection_ba", "mean"),
            inner_selection_ba_std=("inner_selection_ba", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            fixed40_ba_mean=("fixed40_ba", "mean"),
            fixed40_ba_std=("fixed40_ba", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            ba_delta_fixed_minus_inner_mean=(
                "fixed40_ba",
                lambda x: 0.0,
            ),
        )
    )
    deltas = (
        comparison.assign(ba_delta_fixed_minus_inner=comparison["fixed40_ba"] - comparison["inner_selection_ba"])
        .groupby(["source_session", "target_session", "method"])["ba_delta_fixed_minus_inner"]
        .agg(["mean", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0])
        .reset_index()
    )
    deltas.columns = ["source_session", "target_session", "method", "ba_delta_fixed_minus_inner_mean", "ba_delta_fixed_minus_inner_std"]
    direction_summary = direction_summary.drop(columns=["ba_delta_fixed_minus_inner_mean"]).merge(
        deltas,
        on=["source_session", "target_session", "method"],
        how="left",
    )
    direction_summary.to_csv(aggregate_dir / "fixed40_vs_inner_selection_direction_summary.csv", index=False)


def rewrite_fixed_split_audits(run_dir: Path) -> None:
    for config_path in sorted(run_dir.glob("binary/*/*/seed*/config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("epoch_selection") != "fixed":
            continue
        seed_dir = config_path.parent
        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_sessions = str(summary.get("source_sessions", "")).split("+")
        cycle_counts = summary.get("source_session_cycle_counts") or {}
        source_groups = []
        for session in source_sessions:
            n_cycles = int(cycle_counts.get(str(session), 0))
            source_groups.extend(f"{session}_cycle{cycle_i}" for cycle_i in range(n_cycles))
        split_df = pd.DataFrame(
            split_audit_rows(
                source_sessions,
                str(summary.get("target_session")),
                str(summary.get("protocol")),
                np.asarray(source_groups, dtype=object),
                metadata=None,
                epoch_selection="fixed",
            )
        )
        split_df.to_csv(seed_dir / "split_audit.csv", index=False)


def _load_master(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "aggregate" / "cross_session_master_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _summary_by_target_method(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    return (
        master.groupby(["target_session", "method"], as_index=False)
        .agg(
            ba_mean=("balanced_accuracy", "mean"),
            ba_std=("balanced_accuracy", lambda x: float(pd.Series(x).std(ddof=1)) if len(x) > 1 else 0.0),
            prediction_collapse_rate=("prediction_is_single_class", lambda x: float(pd.Series(x).astype(bool).mean())),
        )
    )


def write_multisource_comparison(output_root: Path) -> None:
    pooled_dir = output_root / "loso_strong_sessions_pooled_fixed40_v1"
    balanced_dir = output_root / "loso_strong_sessions_balanced_fixed40_v1"
    pairwise_dir = output_root / "pairwise_strong_sessions_binary_fixed40_v1"
    pooled = _load_master(pooled_dir)
    balanced = _load_master(balanced_dir)
    if pooled.empty or balanced.empty:
        return

    comparison_dir = output_root / "loso_strong_sessions_comparison_v1"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    pooled_summary = _summary_by_target_method(pooled).rename(
        columns={
            "ba_mean": "pooled_all_ba_mean",
            "ba_std": "pooled_all_ba_std",
            "prediction_collapse_rate": "pooled_prediction_collapse_rate",
        }
    )
    balanced_summary = _summary_by_target_method(balanced).rename(
        columns={
            "ba_mean": "session_balanced_ba_mean",
            "ba_std": "session_balanced_ba_std",
            "prediction_collapse_rate": "balanced_prediction_collapse_rate",
        }
    )
    pooled_vs_balanced = pooled_summary.merge(balanced_summary, on=["target_session", "method"], how="outer")
    pooled_vs_balanced["balanced_minus_pooled"] = (
        pooled_vs_balanced["session_balanced_ba_mean"] - pooled_vs_balanced["pooled_all_ba_mean"]
    )
    pooled_vs_balanced.sort_values(["method", "target_session"]).to_csv(
        comparison_dir / "pooled_vs_session_balanced.csv",
        index=False,
    )

    pairwise_rows = []
    pairwise_summary_path = pairwise_dir / "aggregate" / "cross_session_method_summary.csv"
    pairwise_frames = []
    if pairwise_summary_path.exists():
        pairwise_frames.append(pd.read_csv(pairwise_summary_path))
    linear_pairwise_path = output_root / "pairwise_strong_sessions_v1" / "aggregate" / "cross_session_method_summary.csv"
    if linear_pairwise_path.exists():
        linear_pairwise = pd.read_csv(linear_pairwise_path)
        linear_pairwise = linear_pairwise[linear_pairwise["method"].isin(LINEAR_METHODS)]
        pairwise_frames.append(linear_pairwise)
    if pairwise_frames:
        pairwise = pd.concat(pairwise_frames, ignore_index=True)
        pairwise = pairwise[pairwise["task"].astype(str).eq("binary")]
        pairwise = pairwise[pairwise["source_sessions"].astype(str).str.count(r"\+") == 0]
        pooled_lookup = pooled_summary.set_index(["target_session", "method"])
        balanced_lookup = balanced_summary.set_index(["target_session", "method"])
        for (target, method), group in pairwise.groupby(["target_session", "method"], sort=True):
            group = group.copy()
            group["balanced_accuracy_mean"] = pd.to_numeric(group["balanced_accuracy_mean"], errors="coerce")
            best_row = group.sort_values("balanced_accuracy_mean", ascending=False).iloc[0]
            pooled_ba = pooled_lookup.loc[(target, method), "pooled_all_ba_mean"] if (target, method) in pooled_lookup.index else np.nan
            balanced_ba = (
                balanced_lookup.loc[(target, method), "session_balanced_ba_mean"]
                if (target, method) in balanced_lookup.index
                else np.nan
            )
            pairwise_rows.append(
                {
                    "target_session": target,
                    "method": method,
                    "best_pairwise_source": str(best_row["source_sessions"]),
                    "best_pairwise_ba": float(best_row["balanced_accuracy_mean"]),
                    "mean_pairwise_ba": float(group["balanced_accuracy_mean"].mean()),
                    "multisource_pooled_ba": float(pooled_ba) if pd.notna(pooled_ba) else np.nan,
                    "multisource_balanced_ba": float(balanced_ba) if pd.notna(balanced_ba) else np.nan,
                    "pooled_minus_best_pairwise": float(pooled_ba - best_row["balanced_accuracy_mean"]) if pd.notna(pooled_ba) else np.nan,
                    "balanced_minus_best_pairwise": float(balanced_ba - best_row["balanced_accuracy_mean"]) if pd.notna(balanced_ba) else np.nan,
                }
            )
    pairwise_vs = pd.DataFrame(pairwise_rows)
    pairwise_vs.to_csv(comparison_dir / "pairwise_vs_multisource.csv", index=False)
    write_multisource_figures(comparison_dir, pooled_vs_balanced, pairwise_vs)
    write_multisource_findings(comparison_dir, pooled_vs_balanced, pairwise_vs, pooled, balanced)


def write_multisource_figures(comparison_dir: Path, pooled_vs_balanced: pd.DataFrame, pairwise_vs: pd.DataFrame) -> None:
    try:
        os.environ["MPLCONFIGDIR"] = str(comparison_dir / "matplotlib_cache")
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        (comparison_dir / "plot_warning.txt").write_text(str(exc), encoding="utf-8")
        return

    methods = [method for method in ALL_METHODS if method in set(pooled_vs_balanced["method"])]
    targets = ["708", "709", "710"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * max(len(methods), 1), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, method in zip(axes, methods):
        subset = pooled_vs_balanced[pooled_vs_balanced["method"] == method].copy()
        subset["target_session"] = subset["target_session"].astype(str)
        x = np.arange(len(targets))
        pooled_vals = [float(subset.loc[subset["target_session"].eq(t), "pooled_all_ba_mean"].mean()) for t in targets]
        pooled_err = [float(subset.loc[subset["target_session"].eq(t), "pooled_all_ba_std"].mean()) for t in targets]
        bal_vals = [float(subset.loc[subset["target_session"].eq(t), "session_balanced_ba_mean"].mean()) for t in targets]
        bal_err = [float(subset.loc[subset["target_session"].eq(t), "session_balanced_ba_std"].mean()) for t in targets]
        ax.bar(x - 0.18, pooled_vals, width=0.36, yerr=pooled_err, capsize=3, label="pooled_all", color="#4c78a8")
        ax.bar(x + 0.18, bal_vals, width=0.36, yerr=bal_err, capsize=3, label="session_balanced", color="#f58518")
        ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
        ax.set_title(method)
        ax.set_xticks(x, targets)
        ax.set_xlabel("target session")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[0].set_ylabel("Balanced Accuracy")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(comparison_dir / "multisource_pooled_vs_balanced.png", dpi=300)
    fig.savefig(comparison_dir / "multisource_pooled_vs_balanced.pdf")
    plt.close(fig)

    if not pairwise_vs.empty:
        methods = [method for method in ALL_METHODS if method in set(pairwise_vs["method"])]
        fig, axes = plt.subplots(1, len(methods), figsize=(4 * max(len(methods), 1), 3.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, method in zip(axes, methods):
            subset = pairwise_vs[pairwise_vs["method"] == method].copy()
            subset["target_session"] = subset["target_session"].astype(str)
            x = np.arange(len(targets))
            pair_vals = [float(subset.loc[subset["target_session"].eq(t), "mean_pairwise_ba"].mean()) for t in targets]
            pooled_vals = [float(subset.loc[subset["target_session"].eq(t), "multisource_pooled_ba"].mean()) for t in targets]
            bal_vals = [float(subset.loc[subset["target_session"].eq(t), "multisource_balanced_ba"].mean()) for t in targets]
            ax.bar(x - 0.25, pair_vals, width=0.25, label="pairwise mean", color="#54a24b")
            ax.bar(x, pooled_vals, width=0.25, label="multisource pooled", color="#4c78a8")
            ax.bar(x + 0.25, bal_vals, width=0.25, label="multisource balanced", color="#f58518")
            ax.axhline(0.5, color="black", linewidth=1, linestyle="--")
            ax.set_title(method)
            ax.set_xticks(x, targets)
            ax.set_xlabel("target session")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color="#dddddd", linewidth=0.7)
        axes[0].set_ylabel("Balanced Accuracy")
        axes[-1].legend(frameon=False, fontsize=8)
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        fig.savefig(comparison_dir / "pairwise_vs_multisource.png", dpi=300)
        fig.savefig(comparison_dir / "pairwise_vs_multisource.pdf")
        plt.close(fig)


def write_multisource_findings(
    comparison_dir: Path,
    pooled_vs_balanced: pd.DataFrame,
    pairwise_vs: pd.DataFrame,
    pooled: pd.DataFrame,
    balanced: pd.DataFrame,
) -> None:
    lines = ["# Multisource comparison findings", ""]
    if not pairwise_vs.empty:
        pooled_better = int((pairwise_vs["pooled_minus_best_pairwise"] > 0).sum())
        balanced_better = int((pairwise_vs["balanced_minus_best_pairwise"] > 0).sum())
        lines.append(f"- Multisource pooled_all exceeded the best single-source pairwise result in {pooled_better}/{len(pairwise_vs)} target-method rows.")
        lines.append(f"- Multisource session_balanced exceeded the best single-source pairwise result in {balanced_better}/{len(pairwise_vs)} target-method rows.")
    delta = pooled_vs_balanced["balanced_minus_pooled"].dropna()
    if not delta.empty:
        best = pooled_vs_balanced.loc[pooled_vs_balanced["balanced_minus_pooled"].idxmax()]
        lines.append(f"- The largest balancing gain was {best['method']} target {best['target_session']}: {best['balanced_minus_pooled']:.3f} BA.")
        lines.append(f"- Mean balanced-minus-pooled BA across rows was {delta.mean():.3f}.")
    pooled_collapse = int(pooled["prediction_is_single_class"].astype(bool).sum()) if "prediction_is_single_class" in pooled else 0
    balanced_collapse = int(balanced["prediction_is_single_class"].astype(bool).sum()) if "prediction_is_single_class" in balanced else 0
    lines.append(f"- Single-class prediction runs: pooled_all {pooled_collapse}, session_balanced {balanced_collapse}.")
    target708 = pooled_vs_balanced[pooled_vs_balanced["target_session"].astype(str).eq("708")]
    if not target708.empty:
        lines.append(f"- Target 708 mean pooled BA={target708['pooled_all_ba_mean'].mean():.3f}; mean balanced BA={target708['session_balanced_ba_mean'].mean():.3f}.")
    near_chance_balanced = pooled_vs_balanced["session_balanced_ba_mean"].between(0.45, 0.55).mean()
    lines.append(f"- Fraction of balanced method-target means in the 0.45-0.55 band: {near_chance_balanced:.2f}; if high, sample imbalance alone is unlikely to explain the transfer limit.")
    lines.append("")
    lines.append("All statements above are computed from completed CSV summaries only; no model parameters were tuned from these comparisons.")
    (comparison_dir / "multisource_comparison_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.sessions = [str(session) for session in args.sessions]
    if args.source_balance == "session_balanced" and args.protocol != "loso":
        raise ValueError("session_balanced is implemented for multi-source LOSO experiments")
    if args.source_balance == "session_balanced" and len(args.sessions) < 3:
        raise ValueError("session_balanced LOSO requires at least three sessions")
    if args.epoch_selection == "fixed" and args.max_epochs != 40:
        raise ValueError("fixed strong-session experiments must use max_epochs=40")
    run_dir = Path(args.output_root) / safe_name(args.run_name)
    if args.audit_only:
        rewrite_fixed_split_audits(run_dir)
        summarize_and_plot(run_dir, args.protocol, args.tasks, args.sessions)
        write_multisource_comparison(Path(args.output_root))
        print(f"Rebuilt aggregate audits under {run_dir}")
        return
    if args.dry_run:
        dry_run(args, run_dir)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    for task in args.tasks:
        task_dir = run_dir / task
        data_by_session = {session: load_session(session, task, args.clean_margin_s) for session in args.sessions}
        for source_sessions, target_session, combo_name in experiment_specs(args.sessions, args.protocol):
            for method in args.methods:
                if method in LINEAR_METHODS:
                    linear_repeats = args.seeds if args.source_balance == "session_balanced" else [None]
                    for repeat_seed in linear_repeats:
                        suffix = "" if repeat_seed is None else f" repeat={int(repeat_seed)}"
                        print(f"Running {task} {args.protocol} {combo_name} {method}{suffix}")
                        all_summaries.append(
                            run_linear(
                                method,
                                source_sessions,
                                target_session,
                                args.protocol,
                                task,
                                combo_name,
                                task_dir,
                                data_by_session,
                                args,
                                repeat_seed=None if repeat_seed is None else int(repeat_seed),
                            )
                        )
                elif method in NEURAL_METHODS:
                    for seed in args.seeds:
                        print(f"Running {task} {args.protocol} {combo_name} {method} seed={seed}")
                        all_summaries.append(
                            run_neural(method, int(seed), source_sessions, target_session, args.protocol, task, combo_name, task_dir, data_by_session, args)
                        )
                    expected_seed_count = len(args.seeds)
                    observed = [
                        row for row in all_summaries
                        if row.get("task") == task
                        and row.get("source_sessions") == "+".join(source_sessions)
                        and str(row.get("target_session")) == str(target_session)
                        and row.get("method") == method
                    ]
                    if len(observed) != expected_seed_count:
                        raise AssertionError(f"{combo_name} {method}: expected {expected_seed_count} seeds, got {len(observed)}")
                else:
                    raise ValueError(f"Unsupported method: {method}")
    summarize_and_plot(run_dir, args.protocol, args.tasks, args.sessions)
    write_multisource_comparison(Path(args.output_root))
    print(f"Saved cross-session outputs under {run_dir}")


if __name__ == "__main__":
    main()
