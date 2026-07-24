from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .io import load_session_frames
from .labels import infer_visual_stimulus_label, is_clean_block_middle
from .preprocessing import SpatialFilterConfig, apply_spatial_filter


DEFAULT_ANALYSIS_LIMITS = {
    "709_early": (5, 305),
}


def load_monkey_session(
    root: Path,
    session: str,
    task: str = "binary",
    clean_middle: bool = True,
    clean_margin_s: float = 8.0,
    analysis_limit: tuple[int, int] | None = None,
    trim_incomplete_cycles: bool = True,
    frames_per_cycle: int = 30,
    window_size: int = 1,
    window_mode: Literal["sliding", "fixed"] = "sliding",
    fixed_window_start_position: int | None = None,
    spatial_filter: SpatialFilterConfig | dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load a single session and return frames, labels, cycle groups, metadata."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_mode not in {"sliding", "fixed"}:
        raise ValueError("window_mode must be 'sliding' or 'fixed'")
    if window_mode == "fixed" and fixed_window_start_position is None:
        raise ValueError("fixed_window_start_position is required when window_mode='fixed'")

    session_dir = root / "data" / session
    X, indices = load_session_frames(session_dir)
    raw_frame_shape = tuple(int(value) for value in X.shape)

    rows = []
    analysis_keep = np.ones(len(indices), dtype=bool)
    if analysis_limit is not None:
        lo, hi = analysis_limit
        analysis_keep &= (indices >= lo) & (indices <= hi)

    for index in indices:
        timing = infer_visual_stimulus_label(int(index))
        rows.append(
            {
                **asdict(timing),
                "after_analysis_limit": False,
                "complete_cycle": False,
                "clean_middle": False,
                "selected_before_task": False,
            }
        )

    meta = pd.DataFrame(rows)
    meta["after_analysis_limit"] = analysis_keep

    cycle_report_rows = []
    complete_cycles: set[int] = set()
    analysis_meta = meta.loc[analysis_keep]
    for cycle, cycle_rows in analysis_meta.groupby("cycle", sort=True):
        frame_count = int(len(cycle_rows))
        is_complete = frame_count == frames_per_cycle
        if is_complete:
            complete_cycles.add(int(cycle))
        cycle_report_rows.append(
            {
                "cycle": int(cycle),
                "n_frames_after_analysis_limit": frame_count,
                "frames_per_cycle_expected": int(frames_per_cycle),
                "complete_cycle": bool(is_complete),
                "first_index": int(cycle_rows["index"].min()),
                "last_index": int(cycle_rows["index"].max()),
            }
        )

    if trim_incomplete_cycles:
        complete_cycle_keep = meta["cycle"].isin(complete_cycles).to_numpy()
    else:
        complete_cycle_keep = np.ones(len(meta), dtype=bool)
        meta.loc[analysis_keep, "complete_cycle"] = True
    if trim_incomplete_cycles:
        meta["complete_cycle"] = complete_cycle_keep

    clean_keep = np.ones(len(meta), dtype=bool)
    if clean_middle:
        clean_keep = np.asarray(
            [is_clean_block_middle(infer_visual_stimulus_label(int(index)), clean_margin_s) for index in indices]
        )
    meta["clean_middle"] = clean_keep

    keep = analysis_keep & complete_cycle_keep & clean_keep
    meta["selected_before_task"] = keep
    selection_info = {
        "raw_frame_count": int(len(indices)),
        "analysis_limit": list(analysis_limit) if analysis_limit is not None else None,
        "frames_after_analysis_limit": int(analysis_keep.sum()),
        "trim_incomplete_cycles": bool(trim_incomplete_cycles),
        "frames_per_cycle_expected": int(frames_per_cycle),
        "complete_cycles_after_analysis_limit": sorted(complete_cycles),
        "n_complete_cycles_after_analysis_limit": int(len(complete_cycles)),
        "incomplete_cycles_after_analysis_limit": [
            row for row in cycle_report_rows if not row["complete_cycle"]
        ],
        "frames_dropped_incomplete_cycles": int((analysis_keep & ~complete_cycle_keep).sum())
        if trim_incomplete_cycles
        else 0,
        "indices_dropped_incomplete_cycles": [
            int(index) for index in indices[analysis_keep & ~complete_cycle_keep]
        ]
        if trim_incomplete_cycles
        else [],
        "clean_middle": bool(clean_middle),
        "clean_margin_s": float(clean_margin_s),
        "frames_after_complete_cycle_trim": int((analysis_keep & complete_cycle_keep).sum()),
        "frames_after_clean_middle": int(keep.sum()),
    }

    X = X[keep]
    meta = meta.loc[keep].reset_index(drop=True)

    if task == "binary":
        y = meta["binary_label"].to_numpy()
    elif task == "stimulus_type":
        stim_keep = meta["block_name"].isin(["grating", "dot"]).to_numpy()
        X = X[stim_keep]
        meta = meta.loc[stim_keep].reset_index(drop=True)
        y = meta["block_name"].to_numpy()
    else:
        raise ValueError("task must be 'binary' or 'stimulus_type'")

    selection_info["task"] = task
    selection_info["frames_after_task_filter"] = int(len(meta))
    cfg = spatial_filter if isinstance(spatial_filter, SpatialFilterConfig) else SpatialFilterConfig(
        method=str((spatial_filter or {}).get("method", "none")),
        radius=int((spatial_filter or {}).get("radius", 0)),
        mode=str((spatial_filter or {}).get("mode", "reflect")),
    )
    selection_info["raw_loaded_shape"] = list(raw_frame_shape)
    selection_info["pre_spatial_filter_shape"] = [int(value) for value in X.shape]
    selection_info["spatial_filter"] = cfg.to_dict()
    # Spatial-filter ablation purpose: test whether local spatial smoothing improves
    # visual-stimulus decoding. A pillbox filter is a circular local mean; it can
    # suppress single-voxel noise, but may blur fine-grained spatial differences,
    # so no_filter, radius=1, and radius=2 should be compared as an ablation study.
    X = apply_spatial_filter(X, cfg)
    meta["block_offset_s"] = meta["center_in_cycle_s"] - meta["block_start_s"]
    selection_info["window_size"] = int(window_size)
    selection_info["window_mode"] = window_mode
    selection_info["fixed_window_start_position"] = (
        int(fixed_window_start_position) if fixed_window_start_position is not None else None
    )
    if window_mode == "fixed":
        X, y, groups, meta = make_fixed_temporal_windows(
            X,
            y,
            meta,
            window_size,
            int(fixed_window_start_position),
        )
        selection_info["windows_after_temporal_windowing"] = int(len(meta))
    elif window_size > 1:
        X, y, groups, meta = make_temporal_windows(X, y, meta, window_size)
        selection_info["windows_after_temporal_windowing"] = int(len(meta))
    else:
        groups = meta["cycle"].to_numpy(dtype=np.int64)
        selection_info["windows_after_temporal_windowing"] = int(len(meta))

    meta.attrs["selection_info"] = selection_info
    meta.attrs["cycle_report"] = cycle_report_rows
    return X, y, groups, meta


def make_temporal_windows(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build consecutive within-cycle, within-block temporal windows."""
    if window_size == 1:
        return X, y, meta["cycle"].to_numpy(dtype=np.int64), meta
    if len(X) < window_size:
        empty_X = X[:0]
        empty_y = y[:0]
        empty_meta = meta.iloc[:0].copy().reset_index(drop=True)
        return empty_X, empty_y, np.asarray([], dtype=np.int64), empty_meta

    windows = []
    labels = []
    rows = []
    for _, block_rows in meta.groupby(["cycle", "block_name"], sort=False):
        ordered = block_rows.sort_values("index")
        row_positions = ordered.index.to_numpy(dtype=np.int64)
        frame_indices = ordered["index"].to_numpy(dtype=np.int64)
        for start in range(0, len(row_positions) - window_size + 1):
            pos = row_positions[start : start + window_size]
            idx = frame_indices[start : start + window_size]
            if not np.all(np.diff(idx) == 1):
                continue
            first = meta.loc[pos[0]]
            last = meta.loc[pos[-1]]
            windows.append(X[pos])
            labels.append(y[pos[-1]])
            row = last.to_dict()
            row.update(
                {
                    "window_size": int(window_size),
                    "window_start_index": int(first["index"]),
                    "window_end_index": int(last["index"]),
                    "window_indices": ",".join(str(int(value)) for value in idx),
                }
            )
            rows.append(row)

    if windows:
        X_window = np.stack(windows, axis=0)
        y_window = np.asarray(labels)
        window_meta = pd.DataFrame(rows).reset_index(drop=True)
        groups = window_meta["cycle"].to_numpy(dtype=np.int64)
    else:
        X_window = X[:0]
        y_window = y[:0]
        window_meta = meta.iloc[:0].copy().reset_index(drop=True)
        window_meta["window_size"] = pd.Series(dtype=np.int64)
        window_meta["window_start_index"] = pd.Series(dtype=np.int64)
        window_meta["window_end_index"] = pd.Series(dtype=np.int64)
        window_meta["window_indices"] = pd.Series(dtype=str)
        groups = np.asarray([], dtype=np.int64)
    return X_window, y_window, groups, window_meta


def _empty_window_result(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    empty_X = X[:0]
    empty_y = y[:0]
    empty_meta = meta.iloc[:0].copy().reset_index(drop=True)
    empty_meta["window_size"] = pd.Series(dtype=np.int64)
    empty_meta["window_start_index"] = pd.Series(dtype=np.int64)
    empty_meta["window_end_index"] = pd.Series(dtype=np.int64)
    empty_meta["window_indices"] = pd.Series(dtype=str)
    empty_meta["window_start_position"] = pd.Series(dtype=np.int64)
    empty_meta["window_start_offset_s"] = pd.Series(dtype=float)
    empty_meta["window_end_offset_s"] = pd.Series(dtype=float)
    empty_meta["window_start_in_cycle_s"] = pd.Series(dtype=float)
    empty_meta["window_end_in_cycle_s"] = pd.Series(dtype=float)
    return empty_X, empty_y, np.asarray([], dtype=np.int64), empty_meta


def fixed_temporal_window_start_positions(
    meta: pd.DataFrame,
    window_size: int,
) -> list[int]:
    """Return within-block window positions present for every cycle and block."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if meta.empty:
        return []
    if "block_offset_s" not in meta.columns:
        meta = meta.copy()
        meta["block_offset_s"] = meta["center_in_cycle_s"] - meta["block_start_s"]

    block_sets = meta.groupby("cycle")["block_name"].apply(lambda values: tuple(sorted(set(values))))
    if len(set(block_sets)) > 1:
        return []

    required_pairs = {
        (int(row.cycle), str(row.block_name))
        for row in meta[["cycle", "block_name"]].drop_duplicates().itertuples(index=False)
    }
    candidates_by_pair: list[set[int]] = []
    for _, block_rows in meta.groupby(["cycle", "block_name"], sort=False):
        ordered = block_rows.sort_values("index")
        frame_indices = ordered["index"].to_numpy(dtype=np.int64)
        starts = set()
        for start in range(0, len(ordered) - window_size + 1):
            idx = frame_indices[start : start + window_size]
            if not np.all(np.diff(idx) == 1):
                continue
            starts.add(int(start))
        candidates_by_pair.append(starts)

    if len(candidates_by_pair) != len(required_pairs):
        return []
    common = set.intersection(*candidates_by_pair) if candidates_by_pair else set()
    return sorted(common)


def make_fixed_temporal_windows(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    window_size: int,
    window_start_position: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build one same relative within-block window per cycle/block."""
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_start_position < 0:
        raise ValueError("window_start_position must be >= 0")
    if "block_offset_s" not in meta.columns:
        meta = meta.copy()
        meta["block_offset_s"] = meta["center_in_cycle_s"] - meta["block_start_s"]
    valid_positions = fixed_temporal_window_start_positions(meta, window_size)
    if window_start_position not in valid_positions:
        return _empty_window_result(X, y, meta, window_size)

    windows = []
    labels = []
    rows = []
    for _, block_rows in meta.groupby(["cycle", "block_name"], sort=False):
        ordered = block_rows.sort_values("index")
        start = int(window_start_position)
        stop = start + window_size
        if stop > len(ordered):
            return _empty_window_result(X, y, meta, window_size)

        pos = ordered.index.to_numpy(dtype=np.int64)[start:stop]
        idx = ordered["index"].to_numpy(dtype=np.int64)[start:stop]
        if len(pos) != window_size or not np.all(np.diff(idx) == 1):
            return _empty_window_result(X, y, meta, window_size)

        first = meta.loc[pos[0]]
        last = meta.loc[pos[-1]]
        windows.append(X[pos])
        labels.append(y[pos[-1]])
        row = last.to_dict()
        row.update(
                {
                    "window_size": int(window_size),
                    "window_start_position": int(window_start_position),
                    "window_start_index": int(first["index"]),
                    "window_end_index": int(last["index"]),
                    "window_indices": ",".join(str(int(value)) for value in idx),
                    "window_start_offset_s": float(first["block_offset_s"]),
                    "window_end_offset_s": float(last["block_offset_s"]),
                    "window_start_in_cycle_s": float(first["center_in_cycle_s"]),
                    "window_end_in_cycle_s": float(last["center_in_cycle_s"]),
                }
            )
        rows.append(row)

    if not windows:
        return _empty_window_result(X, y, meta, window_size)

    X_window = np.stack(windows, axis=0)
    y_window = np.asarray(labels)
    window_meta = pd.DataFrame(rows).reset_index(drop=True)
    groups = window_meta["cycle"].to_numpy(dtype=np.int64)
    return X_window, y_window, groups, window_meta
