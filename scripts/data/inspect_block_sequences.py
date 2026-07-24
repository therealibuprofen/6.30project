#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_block_sequences import (
    BLOCK_NAMES,
    EXPECTED_IMAGE_SHAPE,
    TARGET_SESSIONS,
    class_count_text,
    file_sha256,
    human_size,
    shape_text,
    write_checksums,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect exported block-sequence HDF5 files.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data/block_sequences_v1"))
    parser.add_argument("--sessions", nargs="+", default=TARGET_SESSIONS)
    parser.add_argument("--preview-blocks", type=int, default=0)
    parser.add_argument("--preview-dir", type=Path, default=Path("previews"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-name", default="quality_control_report.csv")
    return parser.parse_args()


def read_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset[:]
    if values.dtype.kind == "S":
        return [bytes(value).decode("utf-8") for value in values]
    return [str(value) for value in values]


def parse_json_list(value: Any) -> list[Any]:
    if pd.isna(value):
        return []
    return list(json.loads(str(value)))


def require_dataset(handle: h5py.File, path: str, errors: list[str]) -> h5py.Dataset | None:
    if path not in handle:
        errors.append(f"missing HDF5 dataset {path}")
        return None
    return handle[path]


def make_previews(
    h5_path: Path,
    metadata: pd.DataFrame,
    block_indices: np.ndarray,
    preview_dir: Path,
) -> list[str]:
    try:
        mpl_cache = Path("/private/tmp/codex_mpl_cache")
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for preview generation") from exc

    preview_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    with h5py.File(h5_path, "r") as handle:
        X = handle["/clean4/X"]
        times = handle["/clean4/relative_time_s"][:]
        indices = handle["/clean4/original_frame_indices"][:]
        for block_i in block_indices:
            sequence = X[int(block_i)]
            finite = sequence[np.isfinite(sequence)]
            if finite.size:
                vmin, vmax = np.percentile(finite, [1, 99])
            else:
                vmin, vmax = 0.0, 1.0
            if float(vmin) == float(vmax):
                vmin, vmax = None, None
            row = metadata.iloc[int(block_i)]
            fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
            for frame_i, ax in enumerate(axes):
                ax.imshow(sequence[frame_i], cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(f"{times[int(block_i), frame_i]:.0f}s / idx {int(indices[int(block_i), frame_i])}", fontsize=9)
                ax.axis("off")
            fig.suptitle(str(row["block_id"]), fontsize=11)
            out_path = preview_dir / f"{row['block_id']}_clean4.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved.append(str(out_path))
    return saved


def inspect_session(output_dir: Path, session: str, preview_blocks: int, preview_dir: Path, seed: int) -> dict[str, Any]:
    h5_path = output_dir / f"session_{session}_blocks.h5"
    metadata_path = output_dir / f"session_{session}_block_metadata.csv"
    errors: list[str] = []
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)
    with h5py.File(h5_path, "r") as handle:
        d_clean = require_dataset(handle, "/clean4/X", errors)
        d_clean_t = require_dataset(handle, "/clean4/relative_time_s", errors)
        d_clean_idx = require_dataset(handle, "/clean4/original_frame_indices", errors)
        d_full = require_dataset(handle, "/full/X_padded", errors)
        d_mask = require_dataset(handle, "/full/valid_mask", errors)
        d_full_t = require_dataset(handle, "/full/relative_time_s", errors)
        d_full_idx = require_dataset(handle, "/full/original_frame_indices", errors)
        d_n_frames = require_dataset(handle, "/full/n_frames", errors)
        d_y_binary = require_dataset(handle, "/labels/binary", errors)
        d_y_stim = require_dataset(handle, "/labels/stimulus_type", errors)
        d_block_id = require_dataset(handle, "/metadata/block_id", errors)
        d_cycle = require_dataset(handle, "/metadata/cycle", errors)
        d_block_name = require_dataset(handle, "/metadata/block_name", errors)
        if errors:
            raise RuntimeError(errors)

        assert d_clean is not None
        assert d_clean_t is not None
        assert d_clean_idx is not None
        assert d_full is not None
        assert d_mask is not None
        assert d_full_t is not None
        assert d_full_idx is not None
        assert d_n_frames is not None
        assert d_y_binary is not None
        assert d_y_stim is not None
        assert d_block_id is not None
        assert d_cycle is not None
        assert d_block_name is not None

        clean_shape = tuple(int(value) for value in d_clean.shape)
        full_shape = tuple(int(value) for value in d_full.shape)
        n_blocks = clean_shape[0]
        t_max = full_shape[1]
        clean_times = d_clean_t[:]
        clean_indices = d_clean_idx[:]
        valid_mask = d_mask[:]
        full_times = d_full_t[:]
        full_indices = d_full_idx[:]
        n_frames = d_n_frames[:].astype(np.int64)
        block_ids = read_strings(d_block_id)
        h5_cycles = d_cycle[:].astype(np.int64)
        h5_block_names = read_strings(d_block_name)
        y_binary = d_y_binary[:]
        y_stimulus = d_y_stim[:]

        if len(metadata) != n_blocks:
            errors.append(f"metadata rows {len(metadata)} != HDF5 N_blocks {n_blocks}")
        if clean_shape[1:] != (4, *EXPECTED_IMAGE_SHAPE):
            errors.append(f"clean4 shape {clean_shape} does not end with (4, 128, 501)")
        if full_shape[2:] != EXPECTED_IMAGE_SHAPE:
            errors.append(f"full shape {full_shape} does not end with (128, 501)")
        if d_clean.dtype != np.float32 or d_full.dtype != np.float32:
            errors.append("image datasets must be float32")
        if len(block_ids) != len(set(block_ids)):
            errors.append("duplicate block_id values in HDF5 metadata")
        if len(metadata) == n_blocks:
            if metadata["block_id"].astype(str).tolist() != block_ids:
                errors.append("metadata CSV block_id order differs from HDF5")
            if metadata["cycle"].to_numpy(dtype=np.int64).tolist() != h5_cycles.tolist():
                errors.append("metadata CSV cycle order differs from HDF5")
            if metadata["block_name"].astype(str).tolist() != h5_block_names:
                errors.append("metadata CSV block_name order differs from HDF5")
            if not np.array_equal(metadata["binary_label_int"].to_numpy(dtype=np.int8), y_binary):
                errors.append("metadata CSV binary labels differ from HDF5")
            if not np.array_equal(metadata["stimulus_type_label_int"].to_numpy(dtype=np.int8), y_stimulus):
                errors.append("metadata CSV stimulus_type labels differ from HDF5")

        time_order_errors = 0
        cross_block_errors = 0
        for row_i in range(n_blocks):
            if not np.all(np.diff(clean_times[row_i]) > 0):
                time_order_errors += 1
            count = int(n_frames[row_i])
            if count != int(valid_mask[row_i].sum()):
                errors.append(f"block {block_ids[row_i]} valid_mask sum does not match n_frames")
            if count > 1 and not np.all(np.diff(full_times[row_i, :count]) > 0):
                time_order_errors += 1
            if count < t_max:
                pad_slice = d_full[row_i, count:t_max]
                if pad_slice.size and np.any(pad_slice != 0):
                    errors.append(f"block {block_ids[row_i]} has nonzero padded images")
                if not np.all(valid_mask[row_i, count:t_max] == 0):
                    errors.append(f"block {block_ids[row_i]} has nonzero padded mask")
                if not np.all(np.isnan(full_times[row_i, count:t_max])):
                    errors.append(f"block {block_ids[row_i]} has non-NaN padded times")
                if not np.all(full_indices[row_i, count:t_max] == -1):
                    errors.append(f"block {block_ids[row_i]} has non--1 padded frame indices")
            clean_set = set(int(value) for value in clean_indices[row_i])
            full_set = set(int(value) for value in full_indices[row_i, :count])
            if not clean_set.issubset(full_set):
                errors.append(f"block {block_ids[row_i]} clean4 frames are not in full frame set")
            if len(metadata) == n_blocks:
                row = metadata.iloc[row_i]
                if str(row["session"]) != str(session):
                    cross_block_errors += 1
                if parse_json_list(row["clean4_original_frame_indices"]) != [int(value) for value in clean_indices[row_i]]:
                    errors.append(f"block {block_ids[row_i]} CSV clean4 indices differ from HDF5")
                if parse_json_list(row["full_original_frame_indices"]) != [
                    int(value) for value in full_indices[row_i, :count]
                ]:
                    errors.append(f"block {block_ids[row_i]} CSV full indices differ from HDF5")

        if time_order_errors:
            errors.append(f"{time_order_errors} blocks have non-increasing relative times")
        cycle_counts = metadata.groupby("cycle")["block_id"].size() if len(metadata) else pd.Series(dtype=int)
        if any(cycle_counts != 4):
            errors.append(f"cycles with block count != 4: {cycle_counts[cycle_counts != 4].to_dict()}")
        for cycle, cycle_rows in metadata.groupby("cycle", sort=True):
            ordered = cycle_rows.sort_values("block_order_in_cycle")["block_name"].astype(str).tolist()
            if ordered != BLOCK_NAMES:
                errors.append(f"cycle {int(cycle)} block order {ordered} != {BLOCK_NAMES}")
            binary_counts = cycle_rows["binary_label_name"].value_counts().to_dict()
            if int(binary_counts.get("stimulus", 0)) != 2 or int(binary_counts.get("no_stimulus", 0)) != 2:
                errors.append(f"cycle {int(cycle)} binary block counts are not 2/2")
            stim_counts = cycle_rows[cycle_rows["block_name"].isin(["grating", "dot"])]["block_name"].value_counts().to_dict()
            if int(stim_counts.get("grating", 0)) != 1 or int(stim_counts.get("dot", 0)) != 1:
                errors.append(f"cycle {int(cycle)} stimulus-type block counts are not 1/1")

        nan_count = 0
        inf_count = 0
        for row_i in range(n_blocks):
            full_seq = d_full[row_i, : int(n_frames[row_i])]
            nan_count += int(np.isnan(full_seq).sum())
            inf_count += int(np.isinf(full_seq).sum())
        if nan_count:
            errors.append(f"found {nan_count} NaN values in exported image datasets")
        if inf_count:
            errors.append(f"found {inf_count} Inf values in exported image datasets")

        selected_preview_paths: list[str] = []
        if preview_blocks > 0 and n_blocks > 0:
            rng = np.random.default_rng(seed)
            choices = rng.choice(n_blocks, size=min(preview_blocks, n_blocks), replace=False)
            selected_preview_paths = make_previews(h5_path, metadata, np.sort(choices), preview_dir)
            print(f"[{session}] preview blocks: {', '.join(metadata.iloc[int(i)]['block_id'] for i in np.sort(choices))}")

        row = {
            "session": session,
            "n_raw_frames": int(handle.attrs.get("n_raw_frames", 0)),
            "n_complete_cycles": int(metadata["cycle"].nunique()),
            "n_removed_incomplete_cycles": int(handle.attrs.get("n_removed_incomplete_cycles", 0)),
            "n_removed_incomplete_frames": int(handle.attrs.get("n_removed_incomplete_frames", 0)),
            "n_blocks": int(n_blocks),
            "n_clean4_frames": int(clean_shape[0] * clean_shape[1]),
            "min_full_block_frames": int(n_frames.min()) if len(n_frames) else 0,
            "max_full_block_frames": int(n_frames.max()) if len(n_frames) else 0,
            "binary_class_counts": class_count_text(metadata["binary_label_name"]) if len(metadata) else "{}",
            "stimulus_type_class_counts": class_count_text(metadata["stimulus_type_label_name"]) if len(metadata) else "{}",
            "nan_count": int(nan_count),
            "inf_count": int(inf_count),
            "duplicate_block_ids": int(len(block_ids) - len(set(block_ids))),
            "time_order_errors": int(time_order_errors),
            "cross_block_errors": int(cross_block_errors),
            "clean4_shape": shape_text(clean_shape),
            "full_padded_shape": shape_text(full_shape),
            "h5_size_bytes": int(h5_path.stat().st_size),
            "h5_size": human_size(h5_path.stat().st_size),
            "metadata_size_bytes": int(metadata_path.stat().st_size),
            "metadata_size": human_size(metadata_path.stat().st_size),
            "h5_sha256": file_sha256(h5_path),
            "preview_files": json.dumps(selected_preview_paths, ensure_ascii=False),
            "status": "PASS" if not errors else "FAIL",
            "failed_checks": json.dumps(errors, ensure_ascii=False),
        }
    return row


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    preview_dir = args.preview_dir
    if not preview_dir.is_absolute():
        preview_dir = output_dir / preview_dir

    rows = []
    for session in args.sessions:
        row = inspect_session(output_dir, str(session), int(args.preview_blocks), preview_dir, int(args.seed))
        rows.append(row)
        print(
            f"[{session}] {row['status']} clean4={row['clean4_shape']} "
            f"full={row['full_padded_shape']} h5={row['h5_size']}"
        )
        if row["status"] != "PASS":
            raise RuntimeError(f"Session {session} failed inspection: {row['failed_checks']}")

    report_path = output_dir / args.report_name
    pd.DataFrame(rows).to_csv(report_path, index=False)
    write_checksums(output_dir)
    print(f"Inspection report: {report_path}")
    print(f"Checksums: {output_dir / 'checksums.sha256'}")


if __name__ == "__main__":
    main()
