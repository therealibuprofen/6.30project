from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np


def natural_sort_key(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 10**12


def frame_index(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse frame index from {path}")
    return int(match.group(1))


def load_mat_svd(path: Path) -> np.ndarray:
    """Load the Data_SVD array from a MATLAB v7.3/HDF5 file."""
    with h5py.File(path, "r") as handle:
        if "Data_SVD" not in handle:
            raise KeyError(f"{path} does not contain Data_SVD")
        return handle["Data_SVD"][:].astype(np.float32, copy=False)


def session_mat_files(session_dir: Path) -> list[Path]:
    files = sorted(session_dir.glob("*.mat"), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No .mat files found in {session_dir}")
    return files


def load_session_frames(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load one session as [n_frames, height, width] plus numeric file indices."""
    files = session_mat_files(session_dir)
    frames: list[np.ndarray] = []
    indices: list[int] = []
    expected_shape: tuple[int, ...] | None = None
    for path in files:
        frame = load_mat_svd(path)
        if expected_shape is None:
            expected_shape = frame.shape
        if frame.shape != expected_shape:
            continue
        frames.append(frame)
        indices.append(frame_index(path))
    return np.stack(frames, axis=0), np.asarray(indices, dtype=np.int64)
