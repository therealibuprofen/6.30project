"""Discrete-frame temporal response-latency feasibility analysis (v11).

The formal unit is a complete 30-frame cycle loaded from raw ``Data_SVD`` MAT
files.  Fixed 16x16 grid patches are used without ROI selection, registration,
temporal interpolation, HRF fitting, or decoder training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ultrasound_decoding.io import frame_index, session_mat_files


RUN_NAME = "temporal_response_latency_propagation_feasibility_9sessions_v11"
SESSIONS = ("626", "628", "708", "709", "710", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
IMAGE_SHAPE = (128, 501)
PATCH_SIZE = (16, 16)
PATCH_GRID_SHAPE = (8, 32)
CYCLE_SECONDS = 120.0
FRAMES_PER_CYCLE = 30
BLOCK_SECONDS = 30.0
FRAME_INTERVAL_SECONDS = CYCLE_SECONDS / FRAMES_PER_CYCLE
CONDITIONS = ("grating", "stop_after_grating", "dot", "static")
FIXED_ORIENTATIONS = {session: "identity" for session in SESSIONS}
FIXED_ORIENTATIONS["807"] = "flip_vertical"
ONSET_CRITERIA = {"PRIMARY_Z2": 2.0, "SENSITIVITY_Z1_5": 1.5}
CONSECUTIVE_FRAMES = 2
PEAK_WINDOW_SECONDS = (0.0, 20.0)
STABLE_DETECTION_RATE = 0.6
STABLE_ONSET_IQR = 1.0
N_SPLITS = 1000
MIN_COMMON_PATCHES = 5
RANDOM_SEED = 20260818
EPS = 1.0e-8

REQUIRED_OUTPUTS = (
    "audit/temporal_metadata_audit.csv",
    "audit/within_session_ba_reuse.csv",
    "audit/v9_metric_reuse.csv",
    "audit/config_freeze.md",
    "latency/patch_timecourses",
    "latency/onset_metrics.csv",
    "latency/peak_latency_metrics.csv",
    "latency/patch_latency_summary.csv",
    "latency/split_half_latency_metrics.csv",
    "summaries/session_latency_feasibility.csv",
    "summaries/latency_vs_withinBA_associations.csv",
    "summaries/v9_vs_v11_latency_association.csv",
    "summaries/feasibility_decision.csv",
    "figures/temporal_sampling_audit.png",
    "figures/stable_patch_fraction_by_session.png",
    "figures/GS_latency_maps_9sessions.png",
    "figures/DS_latency_maps_9sessions.png",
    "figures/latency_reproducibility_by_session.png",
    "figures/latency_stability_vs_within_BA.png",
    "figures/latency_diagnostic_overview.png",
    "figures/latency_maps",
    "report/temporal_latency_feasibility_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass(frozen=True)
class PatchSpec:
    patch_id: int
    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class TemporalSession:
    session: str
    frames: np.ndarray  # cycle x frame x height x width
    cycle_ids: np.ndarray
    frame_indices: np.ndarray  # cycle x frame
    center_in_cycle_seconds: np.ndarray  # cycle x frame
    conditions: np.ndarray  # cycle x frame strings
    audit: Mapping[str, Any]


@dataclass
class SessionLatencyResult:
    audit_row: dict[str, Any]
    onset_rows: list[dict[str, Any]]
    peak_rows: list[dict[str, Any]]
    patch_rows: list[dict[str, Any]]
    split_rows: list[dict[str, Any]]
    session_rows: list[dict[str, Any]]
    latency_maps: dict[str, np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_grid(
    image_shape: tuple[int, int] = IMAGE_SHAPE,
    patch_size: tuple[int, int] = PATCH_SIZE,
) -> list[PatchSpec]:
    height, width = map(int, image_shape)
    patch_h, patch_w = map(int, patch_size)
    if height <= 0 or width <= 0 or patch_h <= 0 or patch_w <= 0:
        raise ValueError("image and patch dimensions must be positive")
    specs: list[PatchSpec] = []
    patch_id = 0
    for row, y0 in enumerate(range(0, height, patch_h)):
        for col, x0 in enumerate(range(0, width, patch_w)):
            specs.append(PatchSpec(
                patch_id=patch_id, row=row, col=col,
                y0=y0, y1=min(y0 + patch_h, height),
                x0=x0, x1=min(x0 + patch_w, width),
            ))
            patch_id += 1
    return specs


def apply_fixed_orientation(frames: np.ndarray, orientation: str) -> np.ndarray:
    values = np.asarray(frames)
    if values.shape[-2:] != IMAGE_SHAPE:
        raise ValueError(f"expected trailing shape {IMAGE_SHAPE}, got {values.shape}")
    if orientation == "identity":
        return values
    if orientation == "flip_vertical":
        return np.flip(values, axis=-2).copy()
    raise ValueError(f"unsupported frozen orientation: {orientation}")


def _scalar_timestamp_candidates(handle: h5py.File) -> list[float]:
    candidates: list[float] = []
    tokens = ("timestamp", "time_seconds", "time_s", "acquisition_time")
    for key, value in handle.attrs.items():
        if any(token in str(key).lower() for token in tokens):
            array = np.asarray(value)
            if array.size == 1 and np.issubdtype(array.dtype, np.number):
                candidates.append(float(array.ravel()[0]))
    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and any(token in name.lower() for token in tokens):
            array = np.asarray(obj[()])
            if array.size == 1 and np.issubdtype(array.dtype, np.number):
                candidates.append(float(array.ravel()[0]))
    handle.visititems(visitor)
    return candidates


def infer_timeline(
    indices: np.ndarray,
    timestamps: np.ndarray | None = None,
    *,
    frame_interval_seconds: float = FRAME_INTERVAL_SECONDS,
) -> pd.DataFrame:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("frame indices must be a nonempty strictly increasing vector")
    if timestamps is None:
        time_values = (indices - 1).astype(float) * float(frame_interval_seconds)
        source = "INFERRED_FROM_FRAME_INDEX"
        precision = "APPROXIMATE_FRAME_TIME"
    else:
        time_values = np.asarray(timestamps, dtype=float)
        if time_values.shape != indices.shape or not np.isfinite(time_values).all() or np.any(np.diff(time_values) <= 0):
            raise ValueError("actual timestamps must be finite and strictly increasing")
        time_values = time_values - time_values[0]
        source = "ACTUAL_TIMESTAMP"
        precision = "ACTUAL_TIMESTAMP"
    rows = []
    for index, time_s in zip(indices, time_values):
        cycle = int((int(index) - 1) // FRAMES_PER_CYCLE)
        frame_in_cycle = int((int(index) - 1) % FRAMES_PER_CYCLE)
        center_in_cycle = float((frame_in_cycle + 0.5) * frame_interval_seconds)
        condition_i = min(int(center_in_cycle // BLOCK_SECONDS), 3)
        rows.append({
            "frame_index": int(index), "cycle": cycle, "frame_in_cycle": frame_in_cycle,
            "timestamp_seconds": float(time_s), "center_in_cycle_seconds": center_in_cycle,
            "condition": CONDITIONS[condition_i], "time_source": source,
            "time_precision": precision,
        })
    return pd.DataFrame(rows)


def complete_cycle_ids(indices: np.ndarray) -> list[int]:
    values = np.asarray(indices, dtype=np.int64)
    cycles = (values - 1) // FRAMES_PER_CYCLE
    complete = []
    for cycle in np.unique(cycles):
        expected = np.arange(cycle * FRAMES_PER_CYCLE + 1, (cycle + 1) * FRAMES_PER_CYCLE + 1)
        if np.array_equal(np.sort(values[cycles == cycle]), expected):
            complete.append(int(cycle))
    return complete


def load_temporal_session(
    data_root: Path,
    session: str,
    *,
    max_cycles: int | None = None,
) -> TemporalSession:
    session = str(session)
    if session not in SESSIONS:
        raise ValueError(f"unexpected session: {session}")
    files = session_mat_files(Path(data_root) / session)
    records: list[tuple[int, np.ndarray, float | None]] = []
    timestamp_fields_seen = False
    non_data_keys: set[str] = set()
    for path in files:
        with h5py.File(path, "r") as handle:
            if "Data_SVD" not in handle:
                raise KeyError(f"{path} lacks Data_SVD")
            frame = np.asarray(handle["Data_SVD"][:], dtype=np.float32)
            if frame.shape != IMAGE_SHAPE:
                raise ValueError(f"{path} shape {frame.shape} != {IMAGE_SHAPE}")
            non_data_keys.update(str(key) for key in handle.keys() if str(key) != "Data_SVD")
            candidates = _scalar_timestamp_candidates(handle)
            if len(candidates) > 1 and not np.allclose(candidates, candidates[0]):
                raise ValueError(f"ambiguous timestamp candidates in {path}")
            timestamp = candidates[0] if candidates else None
            timestamp_fields_seen = timestamp_fields_seen or bool(candidates)
        records.append((frame_index(path), frame, timestamp))
    records.sort(key=lambda item: item[0])
    indices = np.asarray([item[0] for item in records], dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) != 1):
        raise ValueError(f"session {session} contains duplicate or missing raw frame indices")
    timestamp_values: np.ndarray | None = None
    if timestamp_fields_seen:
        if any(item[2] is None for item in records):
            raise ValueError("timestamp field is present for only part of the session")
        timestamp_values = np.asarray([float(item[2]) for item in records], dtype=float)
    timeline = infer_timeline(indices, timestamp_values)
    complete = complete_cycle_ids(indices)
    if max_cycles is not None:
        complete = complete[: int(max_cycles)]
    if len(complete) < 2:
        raise ValueError(f"session {session} has fewer than two complete cycles")
    record_lookup = {index: frame for index, frame, _ in records}
    frame_cycles = []
    index_cycles = []
    center_cycles = []
    condition_cycles = []
    for cycle in complete:
        expected = np.arange(cycle * FRAMES_PER_CYCLE + 1, (cycle + 1) * FRAMES_PER_CYCLE + 1)
        frame_cycles.append(np.stack([record_lookup[int(index)] for index in expected]))
        cycle_rows = timeline[timeline["cycle"] == cycle].sort_values("frame_in_cycle")
        index_cycles.append(expected)
        center_cycles.append(cycle_rows["center_in_cycle_seconds"].to_numpy(float))
        condition_cycles.append(cycle_rows["condition"].astype(str).to_numpy())
    raw = np.stack(frame_cycles).astype(np.float32, copy=False)
    transformed = np.arcsinh(apply_fixed_orientation(raw, FIXED_ORIENTATIONS[session])).astype(np.float32, copy=False)
    if not np.isfinite(transformed).all():
        raise ValueError(f"session {session} contains non-finite transformed frames")
    if timestamp_values is None:
        intervals = np.full(max(len(indices) - 1, 1), FRAME_INTERVAL_SECONDS)
        interval_source = "INFERRED_FROM_FRAME_INDEX"
        precision = "APPROXIMATE_FRAME_TIME"
        timestamps_available = False
    else:
        intervals = np.diff(timestamp_values)
        interval_source = "ACTUAL_TIMESTAMP"
        precision = "ACTUAL_TIMESTAMP"
        timestamps_available = True
    counts = [FRAMES_PER_CYCLE for _ in complete]
    interval_mean = float(np.mean(intervals))
    interval_std = float(np.std(intervals, ddof=0))
    ratio_low = 5.0 / interval_mean
    ratio_high = 6.0 / interval_mean
    audit = {
        "session": session,
        "n_complete_cycles": len(complete),
        "frames_per_cycle_min": min(counts), "frames_per_cycle_max": max(counts),
        "frames_per_cycle_mode": FRAMES_PER_CYCLE,
        "frame_interval_seconds_mean": interval_mean,
        "frame_interval_seconds_std": interval_std,
        "frame_interval_source": interval_source,
        "time_precision": precision,
        "timestamps_available": timestamps_available,
        "timestamps_monotonic": bool(timestamp_values is None or np.all(np.diff(timestamp_values) > 0)),
        "block_boundary_source": "FIXED_PARADIGM_BOUNDARIES_USING_FRAME_CENTERS",
        "grating_start_time": 0.0, "stop_start_time": 30.0,
        "dot_start_time": 60.0, "static_start_time": 90.0,
        "timing_jitter_present": bool(interval_std > 1.0e-6),
        "temporal_resolution_seconds": interval_mean,
        "latency_resolution_ratio_5s_frames": ratio_low,
        "latency_resolution_ratio_6s_frames": ratio_high,
        "latency_resolution_ratio": f"{ratio_low:.6g}-{ratio_high:.6g}_sampled_frames",
        "usable_for_latency_analysis": True,
        "reason": (
            "APPROXIMATE_FRAME_TIME; NO_TRUE_PRE_GRATING_BASELINE; complete cycles and monotonic indices available"
            if timestamp_values is None else
            "ACTUAL_TIMESTAMP; NO_TRUE_PRE_GRATING_BASELINE; complete cycles available"
        ),
        "raw_source": str((Path(data_root) / session).resolve()),
        "raw_variable": "Data_SVD",
        "raw_total_frames": len(records),
        "discarded_incomplete_frames": len(records) - len(complete) * FRAMES_PER_CYCLE,
        "complete_cycle_ids": ",".join(map(str, complete)),
        "non_Data_SVD_keys": ",".join(sorted(non_data_keys)),
        "full_temporal_sequence_used": True,
        "clean4_used": False,
        "orientation": FIXED_ORIENTATIONS[session],
    }
    return TemporalSession(
        session=session, frames=transformed, cycle_ids=np.asarray(complete, dtype=np.int64),
        frame_indices=np.stack(index_cycles), center_in_cycle_seconds=np.stack(center_cycles),
        conditions=np.stack(condition_cycles), audit=audit,
    )


def extract_patch_timecourses(frames: np.ndarray, patches: Sequence[PatchSpec]) -> np.ndarray:
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim != 4 or values.shape[-2:] != IMAGE_SHAPE:
        raise ValueError("frames must be cycle x frame x 128 x 501")
    output = np.empty((*values.shape[:2], len(patches)), dtype=np.float32)
    for column, patch in enumerate(patches):
        output[..., column] = values[..., patch.y0:patch.y1, patch.x0:patch.x1].mean(axis=(-2, -1))
    if not np.isfinite(output).all():
        raise ValueError("patch time courses contain NaN/Inf")
    return output


def transition_windows(conditions: Sequence[str], transition: str) -> tuple[np.ndarray, np.ndarray, str]:
    names = np.asarray(conditions).astype(str)
    if transition == "GS":
        post = np.flatnonzero(names == "grating")
        if not len(post):
            raise ValueError("grating frames unavailable")
        return post[:1], post, "NO_TRUE_PRE_GRATING_BASELINE_EARLIEST_GRATING_FRAME_REFERENCE"
    if transition == "DS":
        baseline_available = np.flatnonzero(names == "stop_after_grating")
        post = np.flatnonzero(names == "dot")
        if not len(baseline_available) or not len(post):
            raise ValueError("stop/dot frames unavailable")
        baseline = baseline_available[-min(2, len(baseline_available)):]
        status = "STOP_LAST_2_FRAMES" if len(baseline) == 2 else "STOP_AVAILABLE_FRAMES_LT2"
        return baseline, post, status
    raise ValueError("transition must be GS or DS")


def safe_zscore(signal: np.ndarray, baseline: np.ndarray, eps: float = EPS) -> tuple[np.ndarray, float, float]:
    response = np.asarray(signal, dtype=float)
    reference = np.asarray(baseline, dtype=float)
    if not len(reference) or eps <= 0 or not np.isfinite(response).all() or not np.isfinite(reference).all():
        raise ValueError("invalid response/baseline for z score")
    mean = float(reference.mean())
    std = float(reference.std(ddof=0))
    z = (response - mean) / (std + float(eps))
    if not np.isfinite(z).all():
        raise ValueError("z score contains NaN/Inf")
    return z, mean, std


def detect_discrete_onset(
    z_values: Sequence[float], threshold: float, consecutive: int = CONSECUTIVE_FRAMES,
) -> tuple[int | None, int | None]:
    values = np.asarray(z_values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or threshold <= 0 or consecutive < 1:
        raise ValueError("invalid onset inputs")
    above = np.abs(values) >= float(threshold)
    for start in range(0, len(values) - int(consecutive) + 1):
        if bool(np.all(above[start:start + int(consecutive)])):
            direction = int(np.sign(values[start]))
            return int(start), direction
    return None, None


def peak_latency(
    z_values: Sequence[float], post_sample_times_seconds: Sequence[float],
    window_seconds: tuple[float, float] = PEAK_WINDOW_SECONDS,
) -> tuple[int, float]:
    values = np.asarray(z_values, dtype=float)
    times = np.asarray(post_sample_times_seconds, dtype=float)
    if values.shape != times.shape or values.ndim != 1 or not len(values):
        raise ValueError("peak values/times must be aligned vectors")
    keep = (times >= float(window_seconds[0])) & (times <= float(window_seconds[1]))
    if not keep.any():
        raise ValueError("no sampled frame lies inside the 0-20 s peak window")
    available = np.flatnonzero(keep)
    selected = int(available[np.argmax(np.abs(values[keep]))])
    return selected, float(times[selected])


def cycle_patch_latency_rows(
    session: TemporalSession,
    patch_signals: np.ndarray,
    patches: Sequence[PatchSpec],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    onset_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    for cycle_position, cycle in enumerate(session.cycle_ids):
        conditions = session.conditions[cycle_position]
        centers = session.center_in_cycle_seconds[cycle_position]
        for transition in ("GS", "DS"):
            baseline_idx, post_idx, baseline_status = transition_windows(conditions, transition)
            onset_s = 0.0 if transition == "GS" else 60.0
            post_times = centers[post_idx] - onset_s
            for patch_column, patch in enumerate(patches):
                signal = patch_signals[cycle_position, :, patch_column]
                z, baseline_mean, baseline_std = safe_zscore(signal[post_idx], signal[baseline_idx])
                peak_frame, peak_seconds = peak_latency(z, post_times)
                peak_rows.append({
                    "session": session.session, "cycle": int(cycle), "transition": transition,
                    "patch_id": patch.patch_id, "patch_row": patch.row, "patch_col": patch.col,
                    "peak_latency_frame": peak_frame, "peak_latency_seconds_sample_center": peak_seconds,
                    "peak_abs_z": float(abs(z[peak_frame])), "peak_signed_z": float(z[peak_frame]),
                    "peak_window_seconds": "0-20", "subframe_fitting_used": False,
                })
                for criterion, threshold in ONSET_CRITERIA.items():
                    onset, direction = detect_discrete_onset(z, threshold)
                    onset_rows.append({
                        "session": session.session, "cycle": int(cycle), "transition": transition,
                        "criterion": criterion, "abs_z_threshold": threshold,
                        "required_consecutive_frames": CONSECUTIVE_FRAMES,
                        "patch_id": patch.patch_id, "patch_row": patch.row, "patch_col": patch.col,
                        "baseline_mean": baseline_mean, "baseline_std": baseline_std,
                        "baseline_status": baseline_status, "baseline_frame_count": len(baseline_idx),
                        "onset_status": "DETECTED" if onset is not None else "NO_DETECTED_ONSET",
                        "onset_latency_frame": float(onset) if onset is not None else np.nan,
                        "onset_latency_seconds_sample_center": float(post_times[onset]) if onset is not None else np.nan,
                        "signed_response_direction": direction if direction is not None else np.nan,
                        "subframe_fitting_used": False, "temporal_interpolation_used": False,
                    })
    return onset_rows, peak_rows


def iqr(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, 75) - np.percentile(array, 25)) if len(array) else np.nan


def finite_median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if len(array) else np.nan


def summarize_patches(onsets: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    peak_group = peaks.groupby(["session", "transition", "patch_id"], sort=False)
    peak_lookup = {
        key: (float(group["peak_latency_frame"].median()), iqr(group["peak_latency_frame"]))
        for key, group in peak_group
    }
    rows = []
    keys = ["session", "transition", "criterion", "patch_id", "patch_row", "patch_col"]
    for key, group in onsets.groupby(keys, sort=False):
        session, transition, criterion, patch_id, patch_row, patch_col = key
        detected = group["onset_status"].astype(str) == "DETECTED"
        values = group.loc[detected, "onset_latency_frame"].to_numpy(float)
        detection_rate = float(detected.mean())
        onset_iqr = iqr(values)
        stable = bool(detection_rate >= STABLE_DETECTION_RATE and np.isfinite(onset_iqr) and onset_iqr <= STABLE_ONSET_IQR)
        median_peak, peak_iqr = peak_lookup[(session, transition, patch_id)]
        rows.append({
            "session": session, "transition": transition, "criterion": criterion,
            "patch_id": int(patch_id), "patch_row": int(patch_row), "patch_col": int(patch_col),
            "n_cycles": len(group), "n_detected": int(detected.sum()),
            "onset_detection_rate": detection_rate,
            "median_onset_frame": float(np.median(values)) if len(values) else np.nan,
            "IQR_onset_frame": onset_iqr,
            "median_peak_frame": median_peak, "IQR_peak_frame": peak_iqr,
            "latency_stable_patch": stable,
            "stable_detection_rate_threshold": STABLE_DETECTION_RATE,
            "stable_IQR_frame_threshold": STABLE_ONSET_IQR,
        })
    return pd.DataFrame(rows)


def four_neighbor_pairs(patches: Sequence[PatchSpec]) -> list[tuple[int, int]]:
    lookup = {(patch.row, patch.col): patch.patch_id for patch in patches}
    pairs = []
    for patch in patches:
        for neighbor in ((patch.row + 1, patch.col), (patch.row, patch.col + 1)):
            if neighbor in lookup:
                pairs.append((patch.patch_id, lookup[neighbor]))
    return pairs


def neighbor_summary(patch_summary: pd.DataFrame, patches: Sequence[PatchSpec]) -> dict[str, Any]:
    stable = patch_summary[patch_summary["latency_stable_patch"].astype(bool)]
    onset = dict(zip(stable["patch_id"].astype(int), stable["median_onset_frame"].astype(float)))
    differences = []
    directional = []
    for a, b in four_neighbor_pairs(patches):
        if a in onset and b in onset:
            diff = abs(onset[a] - onset[b])
            differences.append(diff)
            if np.isclose(diff, 1.0):
                directional.append(f"{a}_earlier_than_{b}" if onset[a] < onset[b] else f"{b}_earlier_than_{a}")
    values = np.asarray(differences, dtype=float)
    n = len(values)
    return {
        "n_stable_neighbor_pairs": n,
        "fraction_neighbor_pairs_diff_0": float(np.mean(np.isclose(values, 0))) if n else np.nan,
        "fraction_neighbor_pairs_diff_1": float(np.mean(np.isclose(values, 1))) if n else np.nan,
        "fraction_neighbor_pairs_diff_ge2": float(np.mean(values >= 2)) if n else np.nan,
        "directional_one_frame_pairs_exploratory": json.dumps(directional),
        "paths_constructed": False,
    }


def split_cycle_indices(n_cycles: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_cycles < 2:
        raise ValueError("split-half requires at least two cycles")
    rng = np.random.default_rng(int(seed))
    output = []
    for _ in range(int(n_splits)):
        order = rng.permutation(n_cycles)
        cut = n_cycles // 2
        output.append((np.sort(order[:cut]), np.sort(order[cut:])))
    return output


def split_half_latency(
    primary_onsets: pd.DataFrame,
    cycle_ids: Sequence[int],
    *,
    n_splits: int = N_SPLITS,
    min_common_patches: int = MIN_COMMON_PATCHES,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    cycles = np.asarray(cycle_ids, dtype=np.int64)
    patches = np.sort(primary_onsets["patch_id"].astype(int).unique())
    cycle_lookup = {int(cycle): row for row, cycle in enumerate(cycles)}
    patch_lookup = {int(patch): col for col, patch in enumerate(patches)}
    matrix = np.full((len(cycles), len(patches)), np.nan, dtype=float)
    for row in primary_onsets.itertuples():
        if str(row.onset_status) == "DETECTED":
            matrix[cycle_lookup[int(row.cycle)], patch_lookup[int(row.patch_id)]] = float(row.onset_latency_frame)
    rows = []
    for split, (a, b) in enumerate(split_cycle_indices(len(cycles), n_splits, seed), start=1):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_a = np.nanmedian(matrix[a], axis=0)
            median_b = np.nanmedian(matrix[b], axis=0)
        common = np.isfinite(median_a) & np.isfinite(median_b)
        n_common = int(common.sum())
        status = "PASS" if n_common >= int(min_common_patches) else "INSUFFICIENT_COMMON_PATCHES"
        rho = exact = within = np.nan
        if status == "PASS":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho = float(spearmanr(median_a[common], median_b[common]).statistic)
            differences = np.abs(median_a[common] - median_b[common])
            exact = float(np.mean(np.isclose(differences, 0)))
            within = float(np.mean(differences <= 1.0))
        rows.append({
            "split": split,
            "half_A_cycles": ",".join(map(str, cycles[a])),
            "half_B_cycles": ",".join(map(str, cycles[b])),
            "cycle_overlap": False,
            "n_common_patches": n_common, "status": status,
            "latency_map_split_half_rho": rho,
            "latency_map_exact_match": exact,
            "latency_map_within1frame": within,
            "minimum_common_patch_guard": int(min_common_patches),
        })
    return pd.DataFrame(rows)


def analyze_session(
    temporal: TemporalSession,
    *,
    patches: Sequence[PatchSpec] | None = None,
    n_splits: int = N_SPLITS,
    timecourse_dir: Path | None = None,
) -> SessionLatencyResult:
    patches = list(patches if patches is not None else patch_grid())
    signals = extract_patch_timecourses(temporal.frames, patches)
    if timecourse_dir is not None:
        Path(timecourse_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(timecourse_dir) / f"session_{temporal.session}_patch_timecourses.npz",
            patch_signal=signals, cycle_ids=temporal.cycle_ids,
            frame_indices=temporal.frame_indices,
            center_in_cycle_seconds=temporal.center_in_cycle_seconds,
            conditions=temporal.conditions,
            patch_id=np.asarray([patch.patch_id for patch in patches]),
            patch_row=np.asarray([patch.row for patch in patches]),
            patch_col=np.asarray([patch.col for patch in patches]),
            patch_bounds=np.asarray([[patch.y0, patch.y1, patch.x0, patch.x1] for patch in patches]),
        )
    onset_rows, peak_rows = cycle_patch_latency_rows(temporal, signals, patches)
    onset = pd.DataFrame(onset_rows)
    peak = pd.DataFrame(peak_rows)
    patch_summary = summarize_patches(onset, peak)
    split_rows: list[dict[str, Any]] = []
    session_rows = []
    maps: dict[str, np.ndarray] = {}
    for transition in ("GS", "DS"):
        transition_primary = onset[
            (onset["transition"] == transition) & (onset["criterion"] == "PRIMARY_Z2")
        ]
        splits = split_half_latency(
            transition_primary, temporal.cycle_ids, n_splits=n_splits,
            seed=RANDOM_SEED + int(temporal.session) + (0 if transition == "GS" else 10000),
        )
        splits.insert(0, "transition", transition)
        splits.insert(0, "session", temporal.session)
        split_rows.extend(splits.to_dict("records"))
        patch_primary = patch_summary[
            (patch_summary["transition"] == transition) & (patch_summary["criterion"] == "PRIMARY_Z2")
        ].copy()
        latency_map = np.full(PATCH_GRID_SHAPE, np.nan, dtype=float)
        stable = patch_primary[patch_primary["latency_stable_patch"].astype(bool)]
        for row in stable.itertuples():
            latency_map[int(row.patch_row), int(row.patch_col)] = float(row.median_onset_frame)
        maps[transition] = latency_map
        valid = splits[splits["status"] == "PASS"]
        neighbors = neighbor_summary(patch_primary, patches)
        session_rows.append({
            "session": temporal.session, "transition": transition,
            "n_cycles": len(temporal.cycle_ids), "n_patches": len(patch_primary),
            "fraction_stable_patches": float(patch_primary["latency_stable_patch"].mean()),
            "median_patch_detection_rate": float(patch_primary["onset_detection_rate"].median()),
            "median_patch_onset_IQR": finite_median(patch_primary["IQR_onset_frame"]),
            "median_patch_peak_IQR": finite_median(patch_primary["IQR_peak_frame"]),
            "latency_map_split_half_rho": finite_median(valid["latency_map_split_half_rho"]),
            "latency_map_exact_match": finite_median(valid["latency_map_exact_match"]),
            "latency_map_within1frame": finite_median(valid["latency_map_within1frame"]),
            "split_half_valid_fraction": float((splits["status"] == "PASS").mean()),
            "n_splits": n_splits,
            **neighbors,
        })
    return SessionLatencyResult(
        dict(temporal.audit), onset_rows, peak_rows, patch_summary.to_dict("records"),
        split_rows, session_rows, maps,
    )


def exact_spearman_permutation(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) != 9 or len(y_values) != 9 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        return {"rho": np.nan, "permutation_p_two_sided": np.nan, "n_permutations": 0, "status": "UNAVAILABLE"}
    rx = rankdata(x_values); ry = rankdata(y_values)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    observed = float(np.dot(rx, ry) / denominator) if denominator else np.nan
    if not np.isfinite(observed):
        return {"rho": np.nan, "permutation_p_two_sided": np.nan, "n_permutations": 0, "status": "UNAVAILABLE_CONSTANT_VECTOR"}
    extreme = 0; total = 0
    for permutation in itertools.permutations(range(9)):
        rho = float(np.dot(rx, ry[np.asarray(permutation)]) / denominator)
        extreme += int(abs(rho) >= abs(observed) - 1.0e-12)
        total += 1
    return {
        "rho": observed, "permutation_p_two_sided": float(extreme / total),
        "n_permutations": total, "status": "PASS_EXACT_9_FACTORIAL",
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return output
    selected = values[finite]
    if np.any((selected < 0) | (selected > 1)):
        raise ValueError("probabilities must lie in [0,1]")
    order = np.argsort(selected, kind="mergesort")
    adjusted_sorted = np.maximum.accumulate((len(selected) - np.arange(len(selected))) * selected[order])
    adjusted = np.empty_like(selected)
    adjusted[order] = np.clip(adjusted_sorted, 0, 1)
    output[finite] = adjusted
    return output


def resolve_v9_table(root: Path | None) -> Path | None:
    if root is None:
        return None
    candidates = (
        Path(root) / "summaries/session_spatial_diagnostic_table.csv",
        Path(root) / RUN_NAME.replace("temporal_response_latency_propagation_feasibility_9sessions_v11", "spatial_glm_contrast_reproducibility_9sessions_v9") / "summaries/session_spatial_diagnostic_table.csv",
        Path(root) / "spatial_glm_contrast_reproducibility_9sessions_v9/summaries/session_spatial_diagnostic_table.csv",
    )
    return next((path for path in candidates if path.is_file()), None)


def load_historical_metrics(root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = resolve_v9_table(root)
    if path is None:
        empty = pd.DataFrame({"session": list(SESSIONS)})
        ba_audit = empty.assign(within_session_BA=np.nan, artifact="", artifact_sha256="", status="MISSING_SAFE_DEGRADE")
        v9_audit = empty.assign(binary_RMS_standardized_effect=np.nan, binary_split_half_corr_median=np.nan, artifact="", artifact_sha256="", status="MISSING_SAFE_DEGRADE")
        return empty, ba_audit, v9_audit
    table = pd.read_csv(path, dtype={"session": str})
    required = {"session", "within_session_BA", "binary_RMS_standardized_effect", "binary_split_half_corr_median"}
    if not required.issubset(table.columns) or set(table["session"]) != set(SESSIONS):
        raise ValueError("v9 diagnostic table is incompatible with fixed nine sessions")
    digest = sha256_file(path)
    artifact = str(path.resolve())
    ba_audit = table[["session", "within_session_BA"]].copy()
    ba_audit["artifact"] = artifact; ba_audit["artifact_sha256"] = digest
    ba_audit["metric_provenance"] = "frozen v9 session diagnostic within_session_BA"
    ba_audit["status"] = "PASS"
    v9_audit = table[["session", "binary_RMS_standardized_effect", "binary_split_half_corr_median"]].copy()
    v9_audit["artifact"] = artifact; v9_audit["artifact_sha256"] = digest
    v9_audit["metric_provenance"] = "frozen v9 spatial diagnostic metrics"
    v9_audit["status"] = "PASS"
    return table, ba_audit, v9_audit


def planned_associations(session_summary: pd.DataFrame, historical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide = session_summary.pivot(index="session", columns="transition", values=[
        "fraction_stable_patches", "latency_map_within1frame", "median_patch_detection_rate"
    ])
    composite = pd.DataFrame({"session": wide.index.astype(str)})
    for metric in ("fraction_stable_patches", "latency_map_within1frame", "median_patch_detection_rate"):
        composite[metric] = wide[metric].mean(axis=1, skipna=True).to_numpy()
    if historical.empty or "within_session_BA" not in historical:
        composite["within_session_BA"] = np.nan
        composite["binary_split_half_corr_median"] = np.nan
    else:
        composite = composite.merge(historical[["session", "within_session_BA", "binary_split_half_corr_median"]], on="session", how="left")
    rows = []
    for metric in ("fraction_stable_patches", "latency_map_within1frame", "median_patch_detection_rate"):
        result = exact_spearman_permutation(composite["within_session_BA"], composite[metric])
        rows.append({
            "predictor": "within_session_BA", "outcome": f"GS_DS_mean_{metric}",
            "n_sessions": 9, "multiple_testing_family": "three_planned_within_BA_associations", **result,
        })
    associations = pd.DataFrame(rows)
    associations["holm_adjusted_p"] = holm_adjust(associations["permutation_p_two_sided"])
    v9_result = exact_spearman_permutation(
        composite["binary_split_half_corr_median"], composite["latency_map_within1frame"],
    )
    v9_assoc = pd.DataFrame([{
        "predictor": "v9_binary_split_half_corr_median",
        "outcome": "v11_GS_DS_mean_latency_map_within1frame",
        "analysis_role": "secondary_mechanistic_analysis", "included_in_primary_Holm_family": False,
        "n_sessions": 9, **v9_result,
    }])
    return composite, associations, v9_assoc


def feasibility_decision(session_summary: pd.DataFrame, temporal_audit: pd.DataFrame) -> pd.DataFrame:
    strong = session_summary[session_summary["session"].isin(STRONG_SESSIONS)]
    meets = strong.assign(meets=(strong["fraction_stable_patches"] >= 0.2) & (strong["latency_map_within1frame"] >= 0.7))
    both_count = int(meets.pivot(index="session", columns="transition", values="meets").fillna(False).all(axis=1).sum())
    any_meets = bool(meets["meets"].any())
    coarse = bool((temporal_audit["latency_resolution_ratio_6s_frames"] <= 1.5 + 1.0e-12).all())
    median_repro = float(session_summary["latency_map_within1frame"].median(skipna=True))
    if both_count >= 2:
        decision = "TEMPORAL_PROPAGATION_ANALYSIS_FEASIBLE"
        reason = "at least two strong sessions meet both GS and DS stable-patch and within-one-frame criteria"
    elif any_meets:
        decision = "PARTIALLY_FEASIBLE"
        reason = "stable latency is limited to a subset of sessions or transitions"
    elif coarse and (not np.isfinite(median_repro) or median_repro < 0.7):
        decision = "TEMPORAL_RESOLUTION_LIMITED"
        reason = "5-6 seconds spans at most 1.5 sampled frames and split-half latency stability is insufficient"
    else:
        decision = "NO_STABLE_LATENCY_STRUCTURE"
        reason = "most sessions lack stable, repeatable condition-associated response latency"
    return pd.DataFrame([{
        "decision": decision, "reason": reason,
        "strong_sessions_meeting_both_GS_DS": both_count,
        "any_strong_session_transition_meeting": any_meets,
        "five_to_six_seconds_at_most_1_5_frames": coarse,
        "median_latency_map_within1frame": median_repro,
        "propagation_proven": False,
        "recommendation": (
            "consider further discrete-frame session-specific spatial latency analysis"
            if decision in {"TEMPORAL_PROPAGATION_ANALYSIS_FEASIBLE", "PARTIALLY_FEASIBLE"}
            else "do not escalate to path or velocity modeling with the current data"
        ),
    }])


def expected_outputs(output_dir: Path) -> list[Path]:
    return [Path(output_dir) / relative for relative in REQUIRED_OUTPUTS]


def config_freeze_text(config_path: Path) -> str:
    return f"""# v11 configuration freeze

- Config: {config_path}
- Sessions: {', '.join(SESSIONS)}; none excluded
- Source: raw `data/session/*.mat:Data_SVD`, complete 30-frame cycles only; clean4 is forbidden
- Time: actual timestamp if present, otherwise explicit INFERRED_FROM_FRAME_INDEX / APPROXIMATE_FRAME_TIME
- Paradigm: 120 s cycle, fixed 30 s blocks; frame interval is audited from data metadata, never silently assumed
- Preprocessing: fixed session orientation, arcsinh, finite check
- Spatial representation: fixed non-overlapping 16x16 grid patches, edge pixels retained
- GS reference: earliest grating frame because no true pre-grating frame exists
- DS reference: last two available stop_after_grating frames
- Primary onset: first |z| >= 2 for two consecutive sampled frames; sensitivity: |z| >= 1.5 only
- Latency precision: integer sampled-frame index only; no interpolation, smoothing, curve fit, or sub-frame estimate
- Peak: sampled point with maximum |z| in 0-20 s post-onset
- Stable patch: detection rate >= 0.6 and onset IQR <= 1 frame
- Split half: {N_SPLITS} cycle splits; minimum {MIN_COMMON_PATCHES} common detected patches
- No external threshold, HRF search, registration, ROI selection, decoder training, path construction, or velocity estimation
- Temporal latency estimates are limited by frame sampling resolution.
"""


def _save_figure(fig: plt.Figure, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def make_figures(
    temporal_audit: pd.DataFrame,
    session_summary: pd.DataFrame,
    composite: pd.DataFrame,
    historical: pd.DataFrame,
    latency_maps: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: Path,
    sessions: Sequence[str],
) -> None:
    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sessions = tuple(map(str, sessions))
    audit = temporal_audit.set_index("session").reindex(sessions)
    x = np.arange(len(sessions))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].bar(x, audit["frames_per_cycle_mode"], color="#4c78a8")
    axes[0].set_title("Complete-cycle samples"); axes[0].set_ylabel("frames/cycle")
    axes[1].bar(x, audit["frame_interval_seconds_mean"], color="#72b7b2")
    axes[1].set_title("Audited frame interval"); axes[1].set_ylabel("seconds")
    axes[2].bar(x - 0.18, audit["latency_resolution_ratio_5s_frames"], 0.36, label="5 s")
    axes[2].bar(x + 0.18, audit["latency_resolution_ratio_6s_frames"], 0.36, label="6 s")
    axes[2].axhline(1.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("5-6 s in sampled frames"); axes[2].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, sessions, rotation=45)
    _save_figure(fig, figures / "temporal_sampling_audit.png")

    summary = session_summary.set_index(["session", "transition"])
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    width = 0.36
    for offset, transition, color in ((-width / 2, "GS", "#4c78a8"), (width / 2, "DS", "#f58518")):
        values = [summary.loc[(session, transition), "fraction_stable_patches"] for session in sessions]
        ax.bar(x + offset, values, width, label=transition, color=color)
    ax.axhline(0.2, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, sessions); ax.set_ylim(0, 1); ax.set_ylabel("stable patch fraction")
    ax.legend(frameon=False)
    _save_figure(fig, figures / "stable_patch_fraction_by_session.png")

    for transition in ("GS", "DS"):
        ncols = 3
        nrows = int(math.ceil(len(sessions) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.9 * nrows), constrained_layout=True, squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, session in zip(axes.flat, sessions):
            image = latency_maps[session][transition]
            shown = np.ma.masked_invalid(image)
            im = ax.imshow(shown, cmap="viridis", vmin=0, vmax=6, interpolation="nearest", aspect="auto")
            ax.set_title(f"{session} | {transition}")
            ax.set_xlabel("patch column"); ax.set_ylabel("patch row"); ax.set_axis_on()
            single, single_ax = plt.subplots(figsize=(8, 3.2), constrained_layout=True)
            single_im = single_ax.imshow(
                shown, cmap="viridis", vmin=0, vmax=6, interpolation="nearest", aspect="auto",
            )
            single_ax.set_title(f"Condition-associated response latency map: {session} {transition}")
            single_ax.set_xlabel("patch column"); single_ax.set_ylabel("patch row")
            single.colorbar(single_im, ax=single_ax, label="median onset sampled-frame index")
            _save_figure(
                single,
                figures / "latency_maps" / f"session_{session}_{transition}_condition_associated_latency.png",
            )
        fig.suptitle(f"Condition-associated response latency map: {transition} (stable patches only)")
        if len(sessions):
            fig.colorbar(im, ax=list(axes.flat), shrink=0.75, label="median onset sampled-frame index")
        _save_figure(fig, figures / f"{transition}_latency_maps_9sessions.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for transition, color in (("GS", "#4c78a8"), ("DS", "#f58518")):
        part = session_summary[session_summary["transition"] == transition].set_index("session").reindex(sessions)
        axes[0].plot(sessions, part["latency_map_within1frame"], marker="o", label=transition, color=color)
        axes[1].plot(sessions, part["latency_map_split_half_rho"], marker="o", label=transition, color=color)
    axes[0].axhline(0.7, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("split-half within-one-frame fraction")
    axes[1].set_ylabel("split-half Spearman rho")
    for ax in axes:
        ax.set_ylim(-0.1 if ax is axes[1] else 0, 1.02); ax.legend(frameon=False); ax.tick_params(axis="x", rotation=45)
    _save_figure(fig, figures / "latency_reproducibility_by_session.png")

    merged = composite.copy()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, metric in zip(axes, ("fraction_stable_patches", "latency_map_within1frame", "median_patch_detection_rate")):
        ax.scatter(merged["within_session_BA"], merged[metric], color="#4c78a8")
        for row in merged.itertuples():
            ax.annotate(str(row.session), (row.within_session_BA, getattr(row, metric)), fontsize=7)
        ax.set_xlabel("within-session BA"); ax.set_ylabel(f"GS/DS mean {metric}")
    _save_figure(fig, figures / "latency_stability_vs_within_BA.png")

    diagnostic = session_summary.pivot(index="session", columns="transition", values=[
        "fraction_stable_patches", "latency_map_within1frame", "median_patch_detection_rate"
    ]).reindex(sessions)
    hist = historical.set_index("session").reindex(sessions) if not historical.empty else pd.DataFrame(index=sessions)
    matrix = np.column_stack([
        hist.get("within_session_BA", pd.Series(np.nan, index=sessions)),
        hist.get("binary_split_half_corr_median", pd.Series(np.nan, index=sessions)),
        diagnostic[("fraction_stable_patches", "GS")], diagnostic[("fraction_stable_patches", "DS")],
        diagnostic[("latency_map_within1frame", "GS")], diagnostic[("latency_map_within1frame", "DS")],
        diagnostic[("median_patch_detection_rate", "GS")].add(diagnostic[("median_patch_detection_rate", "DS")]).div(2),
    ]).astype(float)
    labels = ["within BA", "v9 spatial repro", "GS stable", "DS stable", "GS within1", "DS within1", "median detection"]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=-0.2, vmax=1)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(sessions)), sessions)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, "NA" if not np.isfinite(value) else f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(value) and value < 0.35 else "black")
    ax.set_title("Discrete-frame latency feasibility diagnostic")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save_figure(fig, figures / "latency_diagnostic_overview.png")


def make_report(
    temporal_audit: pd.DataFrame,
    session_summary: pd.DataFrame,
    associations: pd.DataFrame,
    v9_association: pd.DataFrame,
    decision: pd.DataFrame,
) -> str:
    audit_columns = [
        "session", "n_complete_cycles", "frame_interval_seconds_mean", "frame_interval_source",
        "time_precision", "latency_resolution_ratio",
    ]
    summary_columns = [
        "session", "transition", "fraction_stable_patches", "median_patch_detection_rate",
        "latency_map_within1frame", "latency_map_split_half_rho", "split_half_valid_fraction",
    ]
    return "\n".join([
        "# Temporal response-latency feasibility v11", "",
        "This analysis estimates condition-associated response latency on the sampled-frame grid. It does not establish a causal spatial process.", "",
        "Temporal latency estimates are limited by frame sampling resolution.", "",
        "## Time-axis audit", "", "```text", temporal_audit[audit_columns].to_string(index=False), "```", "",
        "All sessions lacking an acquisition timestamp are explicitly marked INFERRED_FROM_FRAME_INDEX and APPROXIMATE_FRAME_TIME. No sub-frame interpolation was used.", "",
        "## Baseline and onset definition", "",
        "- GS has no true pre-grating sample; the earliest grating frame is the explicit reference.",
        "- DS uses the final two available stop_after_grating frames.",
        "- Primary onset is the first two consecutive sampled frames with absolute z at least 2; z=1.5 is sensitivity only.",
        "- Peak latency is the observed sample with maximum absolute z in 0-20 seconds.", "",
        "## Session feasibility", "", "```text", session_summary[summary_columns].to_string(index=False), "```", "",
        "## Planned within-session BA associations", "", "```text", associations.to_string(index=False), "```", "",
        "## Secondary v9 linkage", "", "```text", v9_association.to_string(index=False), "```", "",
        "## Decision", "", f"**{decision.iloc[0]['decision']}** — {decision.iloc[0]['reason']}.", "",
        f"Recommendation: {decision.iloc[0]['recommendation']}.", "",
        "No spatial path, physical velocity, anatomical correspondence, or cross-session alignment is inferred.", "",
    ])


def run_analysis(
    *,
    data_root: Path,
    output_dir: Path,
    config_path: Path,
    v9_root: Path | None,
    sessions: Sequence[str] = SESSIONS,
    max_cycles: int | None = None,
    patch_limit: int | None = None,
    n_splits: int = N_SPLITS,
    require_nine_sessions: bool = True,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    for directory in ("audit", "latency/patch_timecourses", "summaries", "figures", "report"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    sessions = tuple(map(str, sessions))
    if require_nine_sessions and sessions != SESSIONS:
        raise ValueError("formal v11 requires all fixed nine sessions in order")
    historical, ba_audit, v9_audit = load_historical_metrics(v9_root)
    all_results = []
    all_maps: dict[str, Mapping[str, np.ndarray]] = {}
    patches = patch_grid()
    if patch_limit is not None:
        patches = patches[: int(patch_limit)]
    for session in sessions:
        temporal = load_temporal_session(data_root, session, max_cycles=max_cycles)
        result = analyze_session(
            temporal, patches=patches, n_splits=n_splits,
            timecourse_dir=output_dir / "latency/patch_timecourses",
        )
        all_results.append(result); all_maps[session] = result.latency_maps
    temporal_audit = pd.DataFrame([result.audit_row for result in all_results])
    onsets = pd.DataFrame([row for result in all_results for row in result.onset_rows])
    peaks = pd.DataFrame([row for result in all_results for row in result.peak_rows])
    patch_summary = pd.DataFrame([row for result in all_results for row in result.patch_rows])
    split_summary = pd.DataFrame([row for result in all_results for row in result.split_rows])
    session_summary = pd.DataFrame([row for result in all_results for row in result.session_rows])
    if require_nine_sessions:
        composite, associations, v9_association = planned_associations(session_summary, historical)
        decision = feasibility_decision(session_summary, temporal_audit)
    else:
        composite = session_summary.groupby("session", as_index=False).agg(
            fraction_stable_patches=("fraction_stable_patches", "mean"),
            latency_map_within1frame=("latency_map_within1frame", "mean"),
            median_patch_detection_rate=("median_patch_detection_rate", "mean"),
        )
        if not historical.empty:
            composite = composite.merge(historical[["session", "within_session_BA", "binary_split_half_corr_median"]], on="session", how="left")
        associations = pd.DataFrame([{"status": "SMOKE_NOT_INFERENTIAL", "n_sessions": len(sessions)}])
        v9_association = associations.copy()
        decision = pd.DataFrame([{
            "decision": "SMOKE_NOT_SCIENTIFIC", "reason": "limited sessions/cycles/patches/splits",
            "recommendation": "run formal nine-session analysis before interpretation",
        }])
    temporal_audit.to_csv(output_dir / "audit/temporal_metadata_audit.csv", index=False)
    ba_audit[ba_audit["session"].isin(sessions)].to_csv(output_dir / "audit/within_session_ba_reuse.csv", index=False)
    v9_audit[v9_audit["session"].isin(sessions)].to_csv(output_dir / "audit/v9_metric_reuse.csv", index=False)
    (output_dir / "audit/config_freeze.md").write_text(config_freeze_text(config_path), encoding="utf-8")
    onsets.to_csv(output_dir / "latency/onset_metrics.csv", index=False)
    peaks.to_csv(output_dir / "latency/peak_latency_metrics.csv", index=False)
    patch_summary.to_csv(output_dir / "latency/patch_latency_summary.csv", index=False)
    split_summary.to_csv(output_dir / "latency/split_half_latency_metrics.csv", index=False)
    session_summary.to_csv(output_dir / "summaries/session_latency_feasibility.csv", index=False)
    associations.to_csv(output_dir / "summaries/latency_vs_withinBA_associations.csv", index=False)
    v9_association.to_csv(output_dir / "summaries/v9_vs_v11_latency_association.csv", index=False)
    decision.to_csv(output_dir / "summaries/feasibility_decision.csv", index=False)
    make_figures(temporal_audit, session_summary, composite, historical, all_maps, output_dir, sessions)
    (output_dir / "report/temporal_latency_feasibility_report.md").write_text(
        make_report(temporal_audit, session_summary, associations, v9_association, decision), encoding="utf-8",
    )
    scientific = [
        path for path in expected_outputs(output_dir)
        if path.name not in {"pytest_output_local.txt", "smoke_test_local.txt", "run_command_server.txt", "run_log_server.txt"}
    ]
    missing = [str(path) for path in scientific if not path.exists()]
    if missing:
        raise AssertionError(f"v11 output completeness failed: {missing}")
    return {
        "sessions": len(sessions), "n_cycles": int(temporal_audit["n_complete_cycles"].sum()),
        "n_patches_per_session": len(patches), "n_splits": n_splits,
        "decision": str(decision.iloc[0]["decision"]),
    }
