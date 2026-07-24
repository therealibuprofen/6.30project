#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.datasets import load_monkey_session
from ultrasound_decoding.io import frame_index, session_mat_files
from ultrasound_decoding.labels import STIMULUS_BLOCKS


TARGET_SESSIONS = ["708", "709", "710", "807", "813", "817", "822"]
EXPECTED_COMPLETE_CYCLES = {
    "708": 6,
    "709": 22,
    "710": 18,
    "807": 12,
    "813": 10,
    "817": 20,
    "822": 10,
}
EXPECTED_IMAGE_SHAPE = (128, 501)
FRAMES_PER_CYCLE = 30
CYCLE_SECONDS = 120.0
BLOCK_SECONDS = 30.0
BLOCK_NAMES = [name for name, _ in STIMULUS_BLOCKS]
BLOCK_ORDER = {name: order for order, name in enumerate(BLOCK_NAMES)}
BINARY_LABEL_INT = {"no_stimulus": 0, "stimulus": 1}
STIMULUS_TYPE_LABEL_INT = {"dot": 0, "grating": 1, "not_applicable": -1}
BLOCK_TO_STIMULUS_TYPE = {
    "dot": ("dot", 0),
    "grating": ("grating", 1),
    "stop_after_grating": ("not_applicable", -1),
    "static": ("not_applicable", -1),
}


@dataclass
class SessionBuild:
    session: str
    metadata: pd.DataFrame
    X_clean4: np.ndarray | None
    clean4_relative_time_s: np.ndarray
    clean4_original_frame_indices: np.ndarray
    X_full_padded: np.ndarray | None
    valid_mask: np.ndarray
    full_relative_time_s: np.ndarray
    full_original_frame_indices: np.ndarray
    n_frames_full: np.ndarray
    binary_labels: np.ndarray
    stimulus_type_labels: np.ndarray
    qc_row: dict[str, Any]
    manifest_row: dict[str, Any]
    errors: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export block-level fUS temporal sequences for CNN-LSTM/temporal CNN experiments."
    )
    parser.add_argument("--sessions", nargs="+", default=TARGET_SESSIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data/block_sequences_v1"))
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument("--compression", choices=["gzip", "lzf"], default="gzip")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate only; do not write HDF5 arrays.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing session HDF5/metadata outputs.")
    return parser.parse_args()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def shape_text(shape: tuple[int, ...] | list[int]) -> str:
    return "[" + ", ".join(str(int(value)) for value in shape) + "]"


def human_size(num_bytes: int | float) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def git_commit(project_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "not_available"
    return result.stdout.strip() or "not_available"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version_info(project_dir: Path, script_path: Path) -> dict[str, Any]:
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(project_dir),
        "export_script": str(script_path.relative_to(project_dir)),
        "export_script_sha256": file_sha256(script_path),
    }


def source_file_map(project_dir: Path, session: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for path in session_mat_files(project_dir / "data" / session):
        index = frame_index(path)
        if index in mapping:
            raise ValueError(f"Duplicate frame index {index} in session {session}")
        mapping[index] = str(path.relative_to(project_dir))
    return mapping


def block_labels(block_name: str) -> tuple[int, str, int, str]:
    binary_name = dict(STIMULUS_BLOCKS)[block_name]
    stimulus_type_name, stimulus_type_int = BLOCK_TO_STIMULUS_TYPE[block_name]
    return (
        BINARY_LABEL_INT[binary_name],
        binary_name,
        int(stimulus_type_int),
        stimulus_type_name,
    )


def class_count_text(values: list[str] | pd.Series | np.ndarray) -> str:
    counts = pd.Series(values).value_counts().sort_index().to_dict()
    return json_text({str(key): int(value) for key, value in counts.items()})


def load_benchmark_session_views(
    project_dir: Path,
    session: str,
    clean_margin_s: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    X_full, _, _, full_meta = load_monkey_session(
        project_dir,
        session=session,
        task="binary",
        clean_middle=False,
        clean_margin_s=clean_margin_s,
        analysis_limit=None,
        trim_incomplete_cycles=True,
        frames_per_cycle=FRAMES_PER_CYCLE,
        window_size=1,
        spatial_filter=None,
    )
    X_clean, _, _, clean_meta = load_monkey_session(
        project_dir,
        session=session,
        task="binary",
        clean_middle=True,
        clean_margin_s=clean_margin_s,
        analysis_limit=None,
        trim_incomplete_cycles=True,
        frames_per_cycle=FRAMES_PER_CYCLE,
        window_size=1,
        spatial_filter=None,
    )
    return X_full, full_meta, X_clean, clean_meta


def _sorted_block_rows(meta: pd.DataFrame, cycle: int, block_name: str) -> pd.DataFrame:
    rows = meta[(meta["cycle"] == cycle) & (meta["block_name"] == block_name)]
    return rows.sort_values(["center_s", "index"], kind="mergesort")


def _frame_positions(rows: pd.DataFrame) -> np.ndarray:
    return rows.index.to_numpy(dtype=np.int64)


def _frame_indices(rows: pd.DataFrame) -> list[int]:
    return [int(value) for value in rows["index"].to_numpy(dtype=np.int64)]


def _relative_times(rows: pd.DataFrame) -> list[float]:
    if "block_offset_s" in rows.columns:
        values = rows["block_offset_s"].to_numpy(dtype=float)
    else:
        values = rows["center_in_cycle_s"].to_numpy(dtype=float) - rows["block_start_s"].to_numpy(dtype=float)
    return [float(value) for value in values]


def estimate_uncompressed_bytes(n_blocks: int, t_max: int, image_shape: tuple[int, int]) -> int:
    height, width = image_shape
    image_bytes = n_blocks * (4 + t_max) * height * width * np.dtype(np.float32).itemsize
    small_arrays = n_blocks * (4 + t_max) * 24
    metadata_overhead = n_blocks * 2048
    return int(image_bytes + small_arrays + metadata_overhead)


def build_session(
    project_dir: Path,
    session: str,
    clean_margin_s: float,
    include_images: bool,
    compression: str,
    version: dict[str, Any],
) -> SessionBuild:
    X_full, full_meta, X_clean, clean_meta = load_benchmark_session_views(project_dir, session, clean_margin_s)
    full_info = full_meta.attrs.get("selection_info", {})
    clean_info = clean_meta.attrs.get("selection_info", {})
    source_map = source_file_map(project_dir, session)

    errors: list[str] = []
    if len(source_map) != int(full_info.get("raw_frame_count", len(source_map))):
        errors.append(
            "n_mat_files differs from load_session_frames raw_frame_count; a source file may have been skipped"
        )
    if X_full.ndim != 3:
        errors.append(f"full image array must be 3-D, got shape {X_full.shape}")
    image_shape = tuple(int(value) for value in X_full.shape[1:]) if X_full.ndim == 3 else (0, 0)
    if image_shape != EXPECTED_IMAGE_SHAPE:
        errors.append(f"image shape must be {EXPECTED_IMAGE_SHAPE}, got {image_shape}")

    if "block_offset_s" not in full_meta.columns:
        full_meta = full_meta.copy()
        full_meta["block_offset_s"] = full_meta["center_in_cycle_s"] - full_meta["block_start_s"]
    if "block_offset_s" not in clean_meta.columns:
        clean_meta = clean_meta.copy()
        clean_meta["block_offset_s"] = clean_meta["center_in_cycle_s"] - clean_meta["block_start_s"]

    complete_cycles = sorted(int(value) for value in full_meta["cycle"].unique())
    block_rows: list[dict[str, Any]] = []
    clean_sequences: list[np.ndarray] = []
    full_sequences: list[np.ndarray] = []
    clean_time_rows: list[list[float]] = []
    full_time_rows: list[list[float]] = []
    clean_index_rows: list[list[int]] = []
    full_index_rows: list[list[int]] = []
    binary_label_values: list[int] = []
    stimulus_type_label_values: list[int] = []

    time_order_errors = 0
    cross_block_errors = 0
    sequence_errors = 0
    clean_subset_errors = 0

    for cycle in complete_cycles:
        cycle_rows = full_meta[full_meta["cycle"] == cycle].sort_values(["center_in_cycle_s", "index"])
        observed_sequence = list(dict.fromkeys(cycle_rows["block_name"].astype(str)))
        if observed_sequence != BLOCK_NAMES:
            sequence_errors += 1
            errors.append(
                f"session {session} cycle {cycle} block order {observed_sequence} != {BLOCK_NAMES}"
            )

        for block_name in BLOCK_NAMES:
            full_rows = _sorted_block_rows(full_meta, cycle, block_name)
            clean_rows = _sorted_block_rows(clean_meta, cycle, block_name)
            full_indices = _frame_indices(full_rows)
            clean_indices = _frame_indices(clean_rows)
            full_times = _relative_times(full_rows)
            clean_times = _relative_times(clean_rows)

            if len(clean_indices) != 4:
                errors.append(
                    f"session {session} cycle {cycle} block {block_name} clean4 has {len(clean_indices)} frames"
                )
            if not set(clean_indices).issubset(set(full_indices)):
                clean_subset_errors += 1
                errors.append(
                    f"session {session} cycle {cycle} block {block_name} clean4 indices are not a subset of full indices"
                )
            if len(clean_times) > 1 and not np.all(np.diff(np.asarray(clean_times, dtype=float)) > 0):
                time_order_errors += 1
                errors.append(
                    f"session {session} cycle {cycle} block {block_name} clean4 relative times are not strictly increasing"
                )
            if len(full_times) > 1 and not np.all(np.diff(np.asarray(full_times, dtype=float)) > 0):
                time_order_errors += 1
                errors.append(
                    f"session {session} cycle {cycle} block {block_name} full relative times are not strictly increasing"
                )

            for rows, name in [(full_rows, "full"), (clean_rows, "clean4")]:
                if len(rows) == 0:
                    cross_block_errors += 1
                    errors.append(f"session {session} cycle {cycle} block {block_name} {name} has no frames")
                    continue
                if rows["cycle"].nunique() != 1 or rows["block_name"].nunique() != 1:
                    cross_block_errors += 1
                    errors.append(f"session {session} cycle {cycle} block {block_name} {name} crosses labels")
                if set(rows["binary_label"].astype(str)) != {dict(STIMULUS_BLOCKS)[block_name]}:
                    cross_block_errors += 1
                    errors.append(f"session {session} cycle {cycle} block {block_name} {name} has inconsistent labels")

            binary_int, binary_name, stimulus_type_int, stimulus_type_name = block_labels(block_name)
            block_start_time_s = float(cycle * CYCLE_SECONDS + BLOCK_ORDER[block_name] * BLOCK_SECONDS)
            block_end_time_s = float(block_start_time_s + BLOCK_SECONDS)
            block_id = f"session{session}_cycle{cycle:03d}_{block_name}"

            clean_sources = [source_map.get(index, "") for index in clean_indices]
            full_sources = [source_map.get(index, "") for index in full_indices]

            block_rows.append(
                {
                    "block_id": block_id,
                    "session": session,
                    "cycle": int(cycle),
                    "block_name": block_name,
                    "block_order_in_cycle": int(BLOCK_ORDER[block_name]),
                    "binary_label_int": int(binary_int),
                    "binary_label_name": binary_name,
                    "stimulus_type_label_int": int(stimulus_type_int),
                    "stimulus_type_label_name": stimulus_type_name,
                    "n_frames_clean4": int(len(clean_indices)),
                    "n_frames_full": int(len(full_indices)),
                    "clean4_original_frame_indices": json_text(clean_indices),
                    "full_original_frame_indices": json_text(full_indices),
                    "clean4_relative_time_s": json_text(clean_times),
                    "full_relative_time_s": json_text(full_times),
                    "clean4_source_files": json_text(clean_sources),
                    "full_source_files": json_text(full_sources),
                    "block_start_time_s": block_start_time_s,
                    "block_end_time_s": block_end_time_s,
                    "complete_cycle": True,
                    "clean_margin_s": float(clean_margin_s),
                    "image_height": int(image_shape[0]),
                    "image_width": int(image_shape[1]),
                }
            )
            clean_time_rows.append(clean_times)
            full_time_rows.append(full_times)
            clean_index_rows.append(clean_indices)
            full_index_rows.append(full_indices)
            binary_label_values.append(int(binary_int))
            stimulus_type_label_values.append(int(stimulus_type_int))
            if include_images:
                clean_sequences.append(X_clean[_frame_positions(clean_rows)].astype(np.float32, copy=False))
                full_sequences.append(X_full[_frame_positions(full_rows)].astype(np.float32, copy=False))

    metadata = pd.DataFrame(block_rows)
    n_blocks = int(len(metadata))
    t_max = int(max((len(values) for values in full_index_rows), default=0))
    height, width = image_shape

    clean4_relative_time_s = np.full((n_blocks, 4), np.nan, dtype=np.float32)
    clean4_original_frame_indices = np.full((n_blocks, 4), -1, dtype=np.int64)
    valid_mask = np.zeros((n_blocks, t_max), dtype=np.uint8)
    full_relative_time_s = np.full((n_blocks, t_max), np.nan, dtype=np.float32)
    full_original_frame_indices = np.full((n_blocks, t_max), -1, dtype=np.int64)
    n_frames_full = np.asarray([len(values) for values in full_index_rows], dtype=np.int64)

    for row_i, (times, indices) in enumerate(zip(clean_time_rows, clean_index_rows)):
        count = min(len(indices), 4)
        clean4_relative_time_s[row_i, :count] = np.asarray(times[:count], dtype=np.float32)
        clean4_original_frame_indices[row_i, :count] = np.asarray(indices[:count], dtype=np.int64)
    for row_i, (times, indices) in enumerate(zip(full_time_rows, full_index_rows)):
        count = len(indices)
        valid_mask[row_i, :count] = 1
        full_relative_time_s[row_i, :count] = np.asarray(times, dtype=np.float32)
        full_original_frame_indices[row_i, :count] = np.asarray(indices, dtype=np.int64)

    X_clean4 = None
    X_full_padded = None
    if include_images:
        if clean_sequences and all(sequence.shape == (4, height, width) for sequence in clean_sequences):
            X_clean4 = np.stack(clean_sequences, axis=0).astype(np.float32, copy=False)
        else:
            X_clean4 = np.empty((0, 4, height, width), dtype=np.float32)
        X_full_padded = np.zeros((n_blocks, t_max, height, width), dtype=np.float32)
        for row_i, sequence in enumerate(full_sequences):
            X_full_padded[row_i, : sequence.shape[0]] = sequence

    nan_count = int(np.isnan(X_full).sum())
    inf_count = int(np.isinf(X_full).sum())
    duplicate_block_ids = int(metadata["block_id"].duplicated().sum()) if not metadata.empty else 0
    if duplicate_block_ids:
        errors.append(f"session {session} has {duplicate_block_ids} duplicate block_id values")
    if nan_count:
        errors.append(f"session {session} contains {nan_count} NaN values in exported full frames")
    if inf_count:
        errors.append(f"session {session} contains {inf_count} Inf values in exported full frames")

    cycle_block_counts = metadata.groupby("cycle")["block_id"].size() if not metadata.empty else pd.Series(dtype=int)
    bad_cycle_block_counts = cycle_block_counts[cycle_block_counts != 4]
    if not bad_cycle_block_counts.empty:
        errors.append(f"session {session} cycles without exactly 4 blocks: {bad_cycle_block_counts.to_dict()}")

    binary_by_cycle = metadata.groupby(["cycle", "binary_label_name"]).size().unstack(fill_value=0) if not metadata.empty else pd.DataFrame()
    for cycle, counts in binary_by_cycle.iterrows():
        if int(counts.get("stimulus", 0)) != 2 or int(counts.get("no_stimulus", 0)) != 2:
            errors.append(f"session {session} cycle {int(cycle)} binary block counts are not 2/2")

    stimulus_by_cycle = metadata[metadata["block_name"].isin(["grating", "dot"])].groupby(["cycle", "block_name"]).size().unstack(fill_value=0) if not metadata.empty else pd.DataFrame()
    for cycle, counts in stimulus_by_cycle.iterrows():
        if int(counts.get("grating", 0)) != 1 or int(counts.get("dot", 0)) != 1:
            errors.append(f"session {session} cycle {int(cycle)} stimulus-type block counts are not 1/1")

    if include_images and X_full_padded is not None:
        mask_sums = valid_mask.sum(axis=1).astype(np.int64)
        if not np.array_equal(mask_sums, n_frames_full):
            errors.append(f"session {session} valid_mask sums do not match n_frames_full")
        for row_i, count in enumerate(n_frames_full):
            if count < t_max and np.any(X_full_padded[row_i, int(count) :] != 0):
                errors.append(f"session {session} block row {row_i} has nonzero padded image values")
                break
        if X_clean4 is not None and X_clean4.shape[:2] != (n_blocks, 4):
            errors.append(f"session {session} X_clean4 shape starts with {X_clean4.shape[:2]}, expected {(n_blocks, 4)}")
        if X_full_padded.shape[:2] != (n_blocks, t_max):
            errors.append(f"session {session} X_full_padded shape starts with {X_full_padded.shape[:2]}, expected {(n_blocks, t_max)}")

    expected_cycles = EXPECTED_COMPLETE_CYCLES.get(session)
    n_complete_cycles = int(metadata["cycle"].nunique()) if not metadata.empty else 0
    n_removed_incomplete_cycles = int(len(full_info.get("incomplete_cycles_after_analysis_limit", [])))
    n_removed_incomplete_frames = int(full_info.get("frames_dropped_incomplete_cycles", 0))
    estimated_bytes = estimate_uncompressed_bytes(n_blocks, t_max, image_shape)
    expected_blocks = int(n_complete_cycles * len(BLOCK_NAMES))

    qc_row = {
        "session": session,
        "n_raw_frames": int(full_info.get("raw_frame_count", 0)),
        "n_complete_cycles": n_complete_cycles,
        "n_removed_incomplete_cycles": n_removed_incomplete_cycles,
        "n_removed_incomplete_frames": n_removed_incomplete_frames,
        "n_blocks": n_blocks,
        "n_clean4_frames": int(metadata["n_frames_clean4"].sum()) if not metadata.empty else 0,
        "min_full_block_frames": int(n_frames_full.min()) if len(n_frames_full) else 0,
        "max_full_block_frames": int(n_frames_full.max()) if len(n_frames_full) else 0,
        "binary_class_counts": class_count_text(metadata["binary_label_name"]) if not metadata.empty else "{}",
        "stimulus_type_class_counts": class_count_text(metadata["stimulus_type_label_name"]) if not metadata.empty else "{}",
        "nan_count": nan_count,
        "inf_count": inf_count,
        "duplicate_block_ids": duplicate_block_ids,
        "time_order_errors": time_order_errors,
        "cross_block_errors": cross_block_errors,
        "status": "PASS" if not errors else "FAIL",
        "failed_checks": json_text(errors),
    }
    manifest_row = {
        "session": session,
        "h5_file": f"session_{session}_blocks.h5",
        "metadata_csv": f"session_{session}_block_metadata.csv",
        "n_raw_frames": qc_row["n_raw_frames"],
        "n_complete_cycles": n_complete_cycles,
        "expected_complete_cycles_for_audit": expected_cycles,
        "complete_cycle_count_matches_expected": bool(expected_cycles is None or expected_cycles == n_complete_cycles),
        "n_removed_incomplete_cycles": n_removed_incomplete_cycles,
        "n_removed_incomplete_frames": n_removed_incomplete_frames,
        "n_blocks": n_blocks,
        "expected_blocks_from_complete_cycles": expected_blocks,
        "clean4_shape": shape_text((n_blocks, 4, height, width)),
        "full_padded_shape": shape_text((n_blocks, t_max, height, width)),
        "t_max_full": t_max,
        "image_height": height,
        "image_width": width,
        "clean_margin_s": float(clean_margin_s),
        "compression": compression,
        "estimated_uncompressed_size_bytes": estimated_bytes,
        "estimated_uncompressed_size": human_size(estimated_bytes),
        "h5_size_bytes": "",
        "h5_size": "",
        "metadata_size_bytes": "",
        "metadata_size": "",
        "status": qc_row["status"],
        "generated_at": version["generated_at"],
        "git_commit": version["git_commit"],
        "export_script_sha256": version["export_script_sha256"],
    }
    return SessionBuild(
        session=session,
        metadata=metadata,
        X_clean4=X_clean4,
        clean4_relative_time_s=clean4_relative_time_s,
        clean4_original_frame_indices=clean4_original_frame_indices,
        X_full_padded=X_full_padded,
        valid_mask=valid_mask,
        full_relative_time_s=full_relative_time_s,
        full_original_frame_indices=full_original_frame_indices,
        n_frames_full=n_frames_full,
        binary_labels=np.asarray(binary_label_values, dtype=np.int8),
        stimulus_type_labels=np.asarray(stimulus_type_label_values, dtype=np.int8),
        qc_row=qc_row,
        manifest_row=manifest_row,
        errors=errors,
    )


def _compression_kwargs(compression: str) -> dict[str, Any]:
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    if compression == "lzf":
        return {"compression": "lzf"}
    raise ValueError(f"Unsupported compression: {compression}")


def write_string_dataset(group: h5py.Group, name: str, values: pd.Series | list[Any]) -> None:
    texts = ["" if pd.isna(value) else str(value) for value in list(values)]
    width = max(1, *(len(text.encode("utf-8")) for text in texts))
    data = np.asarray([text.encode("utf-8") for text in texts], dtype=f"S{width}")
    group.create_dataset(name, data=data)


def write_hdf5(build: SessionBuild, path: Path, compression: str, version: dict[str, Any]) -> None:
    if build.X_clean4 is None or build.X_full_padded is None:
        raise ValueError("Image arrays were not built; cannot write HDF5")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.h5", dir=path.parent)
    os.close(fd)
    Path(tmp_name).unlink(missing_ok=True)
    with h5py.File(tmp_name, "w") as handle:
        handle.attrs["format_version"] = "block_sequences_v1"
        handle.attrs["session"] = build.session
        handle.attrs["clean_middle"] = True
        handle.attrs["clean_margin_s"] = float(build.metadata["clean_margin_s"].iloc[0]) if not build.metadata.empty else np.nan
        handle.attrs["frames_per_cycle_expected"] = FRAMES_PER_CYCLE
        handle.attrs["cycle_seconds"] = CYCLE_SECONDS
        handle.attrs["block_seconds"] = BLOCK_SECONDS
        handle.attrs["time_mapping_status"] = "nominal_from_frame_index_4s_groups"
        handle.attrs["image_height"] = int(build.metadata["image_height"].iloc[0]) if not build.metadata.empty else 0
        handle.attrs["image_width"] = int(build.metadata["image_width"].iloc[0]) if not build.metadata.empty else 0
        handle.attrs["n_raw_frames"] = int(build.qc_row["n_raw_frames"])
        handle.attrs["n_complete_cycles"] = int(build.qc_row["n_complete_cycles"])
        handle.attrs["n_removed_incomplete_cycles"] = int(build.qc_row["n_removed_incomplete_cycles"])
        handle.attrs["n_removed_incomplete_frames"] = int(build.qc_row["n_removed_incomplete_frames"])
        handle.attrs["generated_at"] = version["generated_at"]
        handle.attrs["git_commit"] = version["git_commit"]
        handle.attrs["export_script"] = version["export_script"]
        handle.attrs["export_script_sha256"] = version["export_script_sha256"]

        kwargs = _compression_kwargs(compression)
        clean = handle.create_group("clean4")
        clean.create_dataset(
            "X",
            data=build.X_clean4,
            chunks=(1, build.X_clean4.shape[1], build.X_clean4.shape[2], build.X_clean4.shape[3]),
            **kwargs,
        )
        clean.create_dataset("relative_time_s", data=build.clean4_relative_time_s, **kwargs)
        clean.create_dataset("original_frame_indices", data=build.clean4_original_frame_indices, **kwargs)

        full = handle.create_group("full")
        full.create_dataset(
            "X_padded",
            data=build.X_full_padded,
            chunks=(1, build.X_full_padded.shape[1], build.X_full_padded.shape[2], build.X_full_padded.shape[3]),
            **kwargs,
        )
        full.create_dataset("valid_mask", data=build.valid_mask, **kwargs)
        full.create_dataset("relative_time_s", data=build.full_relative_time_s, **kwargs)
        full.create_dataset("original_frame_indices", data=build.full_original_frame_indices, **kwargs)
        full.create_dataset("n_frames", data=build.n_frames_full, **kwargs)

        labels = handle.create_group("labels")
        labels.create_dataset("binary", data=build.binary_labels, **kwargs)
        labels.create_dataset("stimulus_type", data=build.stimulus_type_labels, **kwargs)

        metadata = handle.create_group("metadata")
        write_string_dataset(metadata, "block_id", build.metadata["block_id"])
        metadata.create_dataset("cycle", data=build.metadata["cycle"].to_numpy(dtype=np.int64), **kwargs)
        write_string_dataset(metadata, "block_name", build.metadata["block_name"])
        metadata.create_dataset(
            "block_order_in_cycle", data=build.metadata["block_order_in_cycle"].to_numpy(dtype=np.int64), **kwargs
        )
        metadata.create_dataset(
            "binary_label_int", data=build.metadata["binary_label_int"].to_numpy(dtype=np.int8), **kwargs
        )
        write_string_dataset(metadata, "binary_label_name", build.metadata["binary_label_name"])
        metadata.create_dataset(
            "stimulus_type_label_int",
            data=build.metadata["stimulus_type_label_int"].to_numpy(dtype=np.int8),
            **kwargs,
        )
        write_string_dataset(metadata, "stimulus_type_label_name", build.metadata["stimulus_type_label_name"])
        metadata.create_dataset(
            "n_frames_clean4", data=build.metadata["n_frames_clean4"].to_numpy(dtype=np.int64), **kwargs
        )
        metadata.create_dataset("n_frames_full", data=build.metadata["n_frames_full"].to_numpy(dtype=np.int64), **kwargs)
        metadata.create_dataset(
            "block_start_time_s", data=build.metadata["block_start_time_s"].to_numpy(dtype=np.float32), **kwargs
        )
        metadata.create_dataset(
            "block_end_time_s", data=build.metadata["block_end_time_s"].to_numpy(dtype=np.float32), **kwargs
        )
        metadata.create_dataset(
            "complete_cycle", data=build.metadata["complete_cycle"].to_numpy(dtype=np.uint8), **kwargs
        )
        metadata.create_dataset(
            "clean_margin_s", data=build.metadata["clean_margin_s"].to_numpy(dtype=np.float32), **kwargs
        )
        metadata.create_dataset("image_height", data=build.metadata["image_height"].to_numpy(dtype=np.int64), **kwargs)
        metadata.create_dataset("image_width", data=build.metadata["image_width"].to_numpy(dtype=np.int64), **kwargs)
        for name in [
            "clean4_original_frame_indices",
            "full_original_frame_indices",
            "clean4_relative_time_s",
            "full_relative_time_s",
            "clean4_source_files",
            "full_source_files",
        ]:
            write_string_dataset(metadata, name, build.metadata[name])

    Path(tmp_name).replace(path)


def write_label_mapping(path: Path) -> None:
    mapping = {
        "binary": {
            "no_stimulus": 0,
            "stimulus": 1,
            "block_to_label_name": {
                "grating": "stimulus",
                "dot": "stimulus",
                "stop_after_grating": "no_stimulus",
                "static": "no_stimulus",
            },
            "block_to_label_int": {
                "grating": 1,
                "dot": 1,
                "stop_after_grating": 0,
                "static": 0,
            },
        },
        "stimulus_type": {
            "dot": 0,
            "grating": 1,
            "not_applicable": -1,
            "block_to_label_name": {
                "dot": "dot",
                "grating": "grating",
                "stop_after_grating": "not_applicable",
                "static": "not_applicable",
            },
            "block_to_label_int": {
                "dot": 0,
                "grating": 1,
                "stop_after_grating": -1,
                "static": -1,
            },
        },
    }
    path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_readme(output_dir: Path, manifest: pd.DataFrame, version: dict[str, Any], command_line: str) -> None:
    clean_margin_s = float(manifest["clean_margin_s"].iloc[0]) if not manifest.empty else 8.0
    table_lines = [
        "| Session | 完整cycle数 | block数 | clean4 shape | full padded shape | full每block最大帧数 | HDF5大小 | 删除的不完整帧数 |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in manifest.itertuples(index=False):
        h5_size = row.h5_size if str(row.h5_size) else "dry-run"
        table_lines.append(
            f"| {row.session} | {row.n_complete_cycles} | {row.n_blocks} | `{row.clean4_shape}` | "
            f"`{row.full_padded_shape}` | {row.t_max_full} | {h5_size} | {row.n_removed_incomplete_frames} |"
        )

    lines = [
        "# Block序列数据 v1",
        "",
        "本目录保存猴子fUS视觉刺激解码项目的block级多帧序列数据。这里把一个行为block整理为一个序列样本，便于导师检查、共享，以及后续CNN-LSTM和时序1D-CNN实验直接读取，不需要重新解析原始`.mat`文件名。",
        "",
        "## 原始数据",
        "",
        "原始数据来自`data/{session}/*.mat`。每个MATLAB v7.3 `.mat`文件本质上是HDF5容器，本次导出的图像来自其中的`Data_SVD`字段，并通过项目现有的`ultrasound_decoding.io.load_mat_svd`读取。图像统一保存为`float32`，单帧shape为`128 x 501`。",
        "",
        "当前metadata里没有独立事件日志或精确采集时间戳。因此时间字段保留现有benchmark使用的名义时间映射：每个`.mat`文件视作约4秒采集组，标签使用该组的中心时间；一个cycle为120秒，包含4个30秒block。",
        "",
        "## Cycle和Block定义",
        "",
        "一个完整cycle必须正好包含30帧，并按以下顺序包含4个block：`grating`、`stop_after_grating`、`dot`、`static`。导出前会使用当前单帧benchmark相同的完整cycle裁剪逻辑，删除不完整cycle。",
        "",
        "`block_id`稳定且全局唯一，格式为`session{session}_cycle{cycle:03d}_{block_name}`。cycle编号沿用现有标签代码中的0-based编号。",
        "",
        "## 两个数据版本",
        "",
        f"`clean4`是固定长度版本，用于需要每个block固定4帧输入的序列模型。该版本使用`clean_middle=true`和`clean_margin_s={clean_margin_s:g}`，每个block只保留中间4帧，并按名义采集时间从早到晚排序。",
        "",
        "`full`是完整30秒block版本。不同block实际包含的帧数可能不同，因此按session padding到`T_max`。`valid_mask=1`表示真实帧，padding位置图像为0，时间为`NaN`，原始帧编号为`-1`。",
        "",
        "## HDF5结构",
        "",
        "每个`session_{session}_blocks.h5`至少包含：",
        "",
        "```text",
        "/clean4/X                         [N_blocks, 4, 128, 501] float32",
        "/clean4/relative_time_s           [N_blocks, 4] float32",
        "/clean4/original_frame_indices    [N_blocks, 4] int64",
        "/full/X_padded                    [N_blocks, T_max, 128, 501] float32",
        "/full/valid_mask                  [N_blocks, T_max] uint8",
        "/full/relative_time_s             [N_blocks, T_max] float32",
        "/full/original_frame_indices      [N_blocks, T_max] int64",
        "/full/n_frames                    [N_blocks] int64",
        "/labels/binary                    [N_blocks] int8",
        "/labels/stimulus_type             [N_blocks] int8",
        "/metadata/block_id                [N_blocks] string",
        "/metadata/cycle                   [N_blocks] int64",
        "/metadata/block_name              [N_blocks] string",
        "```",
        "",
        "配套的`session_{session}_block_metadata.csv`每行对应一个block，包含block标签、来源`.mat`文件、原始帧编号、名义相对时间，以及名义block开始/结束时间。",
        "",
        "## 标签映射",
        "",
        "binary任务：`no_stimulus = 0`，`stimulus = 1`。其中`grating`和`dot`映射为`stimulus`；`stop_after_grating`和`static`映射为`no_stimulus`。",
        "",
        "stimulus_type任务：`dot = 0`，`grating = 1`。非刺激block不会从共享数据里删除，而是保留为`stimulus_type_label_int = -1`、`stimulus_type_label_name = not_applicable`，后续任务可自行过滤。",
        "",
        "## 读取示例",
        "",
        "Python:",
        "",
        "```python",
        "import h5py",
        "",
        "with h5py.File('processed_data/block_sequences_v1/session_708_blocks.h5', 'r') as h5:",
        "    X_clean4 = h5['/clean4/X'][:]             # [N_blocks, 4, 128, 501]",
        "    y_binary = h5['/labels/binary'][:]",
        "    cycles = h5['/metadata/cycle'][:]",
        "```",
        "",
        "MATLAB:",
        "",
        "```matlab",
        "X_clean4 = h5read('processed_data/block_sequences_v1/session_708_blocks.h5', '/clean4/X');",
        "valid_mask = h5read('processed_data/block_sequences_v1/session_708_blocks.h5', '/full/valid_mask');",
        "```",
        "",
        "后续建模时请用`cycle`做分组划分。不要随机拆分block或帧，因为同一个120秒cycle内的相邻时间点高度相关，随机拆分会把cycle相关信息泄漏到测试集。",
        "",
        "导出的block数据未进行全数据标准化。模型训练时应在每个训练fold中独立拟合归一化参数，并将其应用于对应测试fold，以避免数据泄漏。",
        "",
        "## 各Session数量",
        "",
        *table_lines,
        "",
        "## 生成文件",
        "",
        "`dataset_manifest.csv`汇总每个session的数据量和文件大小。`quality_control_report.csv`记录质量检查结果。`label_mapping.json`保存标签定义。`checksums.sha256`保存共享文件的SHA256校验值。",
        "",
        "常用命令：",
        "",
        "```bash",
        ".venv/bin/python scripts/data/export_block_sequences.py --dry-run",
        ".venv/bin/python scripts/data/export_block_sequences.py --sessions 708 --overwrite",
        ".venv/bin/python scripts/data/inspect_block_sequences.py --sessions 708 --preview-blocks 3",
        ".venv/bin/python scripts/data/export_block_sequences.py --sessions 708 709 710 807 813 817 822 --overwrite",
        ".venv/bin/python scripts/data/inspect_block_sequences.py --sessions 708 709 710 807 813 817 822",
        "```",
        "",
        f"本次生成命令：`{command_line}`",
        "",
        f"生成时间：`{version['generated_at']}`",
        "",
        f"Git commit：`{version['git_commit']}`",
        "",
        f"导出脚本SHA256：`{version['export_script_sha256']}`",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    paths = [
        path
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256" and not path.name.startswith(".")
    ]
    lines = [f"{file_sha256(path)}  {path.relative_to(output_dir)}" for path in paths]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifest_sizes(output_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = []
    for row in rows:
        item = dict(row)
        h5_path = output_dir / str(item["h5_file"])
        metadata_path = output_dir / str(item["metadata_csv"])
        if h5_path.exists():
            size = h5_path.stat().st_size
            item["h5_size_bytes"] = int(size)
            item["h5_size"] = human_size(size)
        if metadata_path.exists():
            size = metadata_path.stat().st_size
            item["metadata_size_bytes"] = int(size)
            item["metadata_size"] = human_size(size)
        updated.append(item)
    return updated


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    version = code_version_info(PROJECT_DIR, Path(__file__).resolve())
    command_line = " ".join(sys.argv)

    manifest_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []

    for session in args.sessions:
        print(f"[{session}] loading benchmark views")
        build = build_session(
            PROJECT_DIR,
            session=str(session),
            clean_margin_s=float(args.clean_margin_s),
            include_images=not args.dry_run,
            compression=args.compression,
            version=version,
        )
        qc_rows.append(build.qc_row)
        manifest_rows.append(build.manifest_row)
        print(
            f"[{session}] complete_cycles={build.qc_row['n_complete_cycles']} "
            f"blocks={build.qc_row['n_blocks']} clean4_frames={build.qc_row['n_clean4_frames']} "
            f"full_frames/block={build.qc_row['min_full_block_frames']}-{build.qc_row['max_full_block_frames']} "
            f"status={build.qc_row['status']}"
        )
        if build.errors:
            raise RuntimeError(f"Session {session} failed critical checks: {build.errors}")
        if args.dry_run:
            continue

        h5_path = output_dir / f"session_{session}_blocks.h5"
        metadata_path = output_dir / f"session_{session}_block_metadata.csv"
        if not args.overwrite and (h5_path.exists() or metadata_path.exists()):
            raise FileExistsError(f"{h5_path} or {metadata_path} exists; pass --overwrite to replace generated outputs")

        build.metadata.to_csv(metadata_path, index=False)
        write_hdf5(build, h5_path, args.compression, version)
        print(f"[{session}] wrote {h5_path} and {metadata_path}")

    manifest_rows = update_manifest_sizes(output_dir, manifest_rows)
    manifest = pd.DataFrame(manifest_rows)
    qc = pd.DataFrame(qc_rows)

    report_name = "dry_run_report.csv" if args.dry_run else "quality_control_report.csv"
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    qc.to_csv(output_dir / report_name, index=False)
    write_label_mapping(output_dir / "label_mapping.json")
    write_readme(output_dir, manifest, version, command_line)
    write_checksums(output_dir)

    print(f"Output directory: {output_dir}")
    print(f"Manifest: {output_dir / 'dataset_manifest.csv'}")
    print(f"Report: {output_dir / report_name}")
    print(f"README: {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
