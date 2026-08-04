from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from ultrasound_decoding.cv import grouped_cv_splits


BLOCK_SEQUENCE_VERSION = "block_sequences_v1"
EXPECTED_BLOCK_SHAPE = (4, 128, 501)
EXPECTED_SESSIONS = ["626", "628", "708", "709", "710", "807", "813", "817", "822"]
EXPECTED_TASKS = ["binary", "stimulus_type"]
BLOCK_NAMES = ["grating", "stop_after_grating", "dot", "static"]
STIMULUS_BLOCK_NAMES = ["grating", "dot"]
TASK_CLASS_NAMES = {
    "binary": {0: "no_stimulus", 1: "stimulus"},
    "stimulus_type": {0: "dot", 1: "grating"},
}
TASK_RUN_DIR = {
    "binary": "block_clean4_binary_v1",
    "stimulus_type": "block_clean4_stimulus_type_v1",
}


@dataclass(frozen=True)
class BlockSequenceData:
    session: str
    task: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    metadata: pd.DataFrame
    clean4_relative_time_s: np.ndarray
    clean4_original_frame_indices: np.ndarray
    source_h5_path: Path
    source_metadata_path: Path

    @property
    def n_blocks(self) -> int:
        return int(len(self.X))

    @property
    def n_cycles(self) -> int:
        return int(len(np.unique(self.groups)))

    @property
    def class_names(self) -> dict[int, str]:
        return TASK_CLASS_NAMES[self.task]


def default_block_data_dir(project_dir: Path) -> Path:
    return project_dir / "processed_data" / BLOCK_SEQUENCE_VERSION


def task_run_dir_name(task: str) -> str:
    if task not in TASK_RUN_DIR:
        raise ValueError(f"Unknown task: {task}")
    return TASK_RUN_DIR[task]


def read_h5_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset[:]
    if values.dtype.kind == "S":
        return [bytes(value).decode("utf-8") for value in values]
    return [str(value) for value in values]


def parse_json_list(value: Any) -> list[Any]:
    if pd.isna(value):
        return []
    return list(json.loads(str(value)))


def label_column_for_task(task: str) -> str:
    if task == "binary":
        return "binary_label_int"
    if task == "stimulus_type":
        return "stimulus_type_label_int"
    raise ValueError("task must be 'binary' or 'stimulus_type'")


def label_name_column_for_task(task: str) -> str:
    if task == "binary":
        return "binary_label_name"
    if task == "stimulus_type":
        return "stimulus_type_label_name"
    raise ValueError("task must be 'binary' or 'stimulus_type'")


def load_block_sequence_session(
    project_dir: Path,
    session: str,
    task: str,
    data_dir: Path | None = None,
) -> BlockSequenceData:
    """Load fixed clean4 block sequences for one session and task."""
    if task not in EXPECTED_TASKS:
        raise ValueError("task must be 'binary' or 'stimulus_type'")
    session = str(session)
    base = data_dir if data_dir is not None else default_block_data_dir(project_dir)
    h5_path = base / f"session_{session}_blocks.h5"
    metadata_path = base / f"session_{session}_block_metadata.csv"
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    metadata = pd.read_csv(metadata_path)
    with h5py.File(h5_path, "r") as handle:
        X = handle["/clean4/X"][:]
        clean_times = handle["/clean4/relative_time_s"][:]
        clean_indices = handle["/clean4/original_frame_indices"][:]
        y_binary = handle["/labels/binary"][:].astype(np.int64)
        y_stimulus = handle["/labels/stimulus_type"][:].astype(np.int64)
        h5_cycles = handle["/metadata/cycle"][:].astype(np.int64)
        h5_block_names = read_h5_strings(handle["/metadata/block_name"])
        h5_block_ids = read_h5_strings(handle["/metadata/block_id"])

    if len(metadata) != len(X):
        raise AssertionError(f"metadata rows {len(metadata)} != HDF5 blocks {len(X)}")
    if tuple(X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(f"clean4 shape {X.shape} does not end with {EXPECTED_BLOCK_SHAPE}")
    if not np.array_equal(metadata["cycle"].to_numpy(dtype=np.int64), h5_cycles):
        raise AssertionError("metadata cycle order differs from HDF5")
    if metadata["block_name"].astype(str).tolist() != h5_block_names:
        raise AssertionError("metadata block_name order differs from HDF5")
    if metadata["block_id"].astype(str).tolist() != h5_block_ids:
        raise AssertionError("metadata block_id order differs from HDF5")
    if not np.array_equal(metadata["binary_label_int"].to_numpy(dtype=np.int64), y_binary):
        raise AssertionError("metadata binary labels differ from HDF5")
    if not np.array_equal(metadata["stimulus_type_label_int"].to_numpy(dtype=np.int64), y_stimulus):
        raise AssertionError("metadata stimulus_type labels differ from HDF5")

    metadata = metadata.copy()
    metadata["source_row_i"] = np.arange(len(metadata), dtype=np.int64)
    if task == "binary":
        keep = np.ones(len(metadata), dtype=bool)
        y = y_binary
    else:
        keep = y_stimulus >= 0
        y = y_stimulus[keep]

    task_metadata = metadata.loc[keep].reset_index(drop=True)
    data = BlockSequenceData(
        session=session,
        task=task,
        X=X[keep].astype(np.float32, copy=False),
        y=y.astype(np.int64, copy=False),
        groups=task_metadata["cycle"].to_numpy(dtype=np.int64),
        metadata=task_metadata,
        clean4_relative_time_s=clean_times[keep].astype(np.float32, copy=False),
        clean4_original_frame_indices=clean_indices[keep].astype(np.int64, copy=False),
        source_h5_path=h5_path,
        source_metadata_path=metadata_path,
    )
    validate_block_sequence_data(data)
    return data


def class_count_dict(y: np.ndarray, task: str) -> dict[str, int]:
    mapping = TASK_CLASS_NAMES[task]
    counts: dict[str, int] = {}
    for value, count in zip(*np.unique(y, return_counts=True)):
        counts[mapping.get(int(value), str(value))] = int(count)
    return counts


def csv_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def cycle_text(values: np.ndarray | list[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(np.unique(values).tolist()))


def validate_block_sequence_data(data: BlockSequenceData) -> None:
    X = data.X
    meta = data.metadata
    if len(X) != len(data.y) or len(X) != len(data.groups) or len(X) != len(meta):
        raise AssertionError("X, y, groups, and metadata lengths differ")
    if tuple(X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(f"each sample must have shape {EXPECTED_BLOCK_SHAPE}, got {X.shape}")
    if not np.isfinite(X).all():
        raise AssertionError("clean4 data contains NaN or Inf")
    if not np.all(meta["session"].astype(str).to_numpy() == str(data.session)):
        raise AssertionError("loaded samples include a different session")
    if not np.all(meta["n_frames_clean4"].astype(int).to_numpy() == 4):
        raise AssertionError("a clean4 sample does not contain exactly 4 frames")
    if not np.all(meta["image_height"].astype(int).to_numpy() == EXPECTED_BLOCK_SHAPE[1]):
        raise AssertionError("metadata image_height differs from clean4 shape")
    if not np.all(meta["image_width"].astype(int).to_numpy() == EXPECTED_BLOCK_SHAPE[2]):
        raise AssertionError("metadata image_width differs from clean4 shape")
    if data.clean4_relative_time_s.shape != (len(X), 4):
        raise AssertionError("relative_time_s must have shape [N, 4]")
    if data.clean4_original_frame_indices.shape != (len(X), 4):
        raise AssertionError("original_frame_indices must have shape [N, 4]")
    if not np.all(np.diff(data.clean4_relative_time_s, axis=1) > 0):
        raise AssertionError("clean4 frame times are not strictly increasing")
    if not np.all(np.diff(data.clean4_original_frame_indices, axis=1) > 0):
        raise AssertionError("clean4 original frame indices are not strictly increasing")

    for row_i, row in meta.reset_index(drop=True).iterrows():
        csv_indices = parse_json_list(row["clean4_original_frame_indices"])
        csv_times = parse_json_list(row["clean4_relative_time_s"])
        if csv_indices != [int(value) for value in data.clean4_original_frame_indices[row_i]]:
            raise AssertionError("metadata clean4 indices differ from HDF5 clean4 indices")
        if [float(value) for value in csv_times] != [
            float(value) for value in data.clean4_relative_time_s[row_i]
        ]:
            raise AssertionError("metadata clean4 times differ from HDF5 clean4 times")
        if str(row["block_id"]) != (
            f"session{data.session}_cycle{int(row['cycle']):03d}_{row['block_name']}"
        ):
            raise AssertionError("block_id does not match session/cycle/block metadata")

    full_meta = pd.read_csv(data.source_metadata_path)
    cycle_counts = full_meta.groupby("cycle")["block_id"].size()
    if any(cycle_counts != 4):
        raise AssertionError(f"binary source cycles do not all have 4 blocks: {cycle_counts.to_dict()}")
    for cycle, cycle_rows in full_meta.groupby("cycle", sort=True):
        ordered_blocks = cycle_rows.sort_values("block_order_in_cycle")["block_name"].astype(str).tolist()
        if ordered_blocks != BLOCK_NAMES:
            raise AssertionError(f"cycle {int(cycle)} block order is {ordered_blocks}, expected {BLOCK_NAMES}")
        binary_counts = cycle_rows["binary_label_name"].value_counts().to_dict()
        if int(binary_counts.get("stimulus", 0)) != 2 or int(binary_counts.get("no_stimulus", 0)) != 2:
            raise AssertionError(f"cycle {int(cycle)} binary counts are not 2/2")
        stim_counts = cycle_rows[cycle_rows["block_name"].isin(STIMULUS_BLOCK_NAMES)]["block_name"].value_counts()
        if int(stim_counts.get("grating", 0)) != 1 or int(stim_counts.get("dot", 0)) != 1:
            raise AssertionError(f"cycle {int(cycle)} stimulus_type counts are not 1/1")

    expected_blocks_per_cycle = 4 if data.task == "binary" else 2
    task_counts = meta.groupby("cycle")["block_id"].size()
    if any(task_counts != expected_blocks_per_cycle):
        raise AssertionError(
            f"{data.task} cycles do not all have {expected_blocks_per_cycle} block samples: "
            f"{task_counts.to_dict()}"
        )
    expected_n = int(data.n_cycles * expected_blocks_per_cycle)
    if len(data.X) != expected_n:
        raise AssertionError(f"{data.task} n_blocks={len(data.X)} != expected {expected_n}")


def split_manifest(
    session: str,
    task: str,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]] | None = None,
    max_folds: int = 10,
) -> pd.DataFrame:
    if splits is None:
        splits = grouped_cv_splits(groups, max_folds=max_folds)
    rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        train_cycles = np.unique(groups[train_idx])
        test_cycles = np.unique(groups[test_idx])
        overlap = set(train_cycles.tolist()) & set(test_cycles.tolist())
        if overlap:
            raise AssertionError(f"cycle leakage in fold {fold_i}: {sorted(overlap)}")
        rows.append(
            {
                "session": str(session),
                "task": task,
                "fold": int(fold_i),
                "train_cycles": cycle_text(train_cycles),
                "test_cycles": cycle_text(test_cycles),
                "n_train_blocks": int(len(train_idx)),
                "n_test_blocks": int(len(test_idx)),
                "train_class_counts": csv_json(class_count_dict(y[train_idx], task)),
                "test_class_counts": csv_json(class_count_dict(y[test_idx], task)),
            }
        )
    return pd.DataFrame(rows)


def dataset_audit_row(data: BlockSequenceData, max_folds: int = 10) -> dict[str, Any]:
    splits = grouped_cv_splits(data.groups, max_folds=max_folds)
    blocks_per_cycle = 4 if data.task == "binary" else 2
    return {
        "session": data.session,
        "task": data.task,
        "block_data_path": str(data.source_h5_path),
        "metadata_path": str(data.source_metadata_path),
        "X_blocks_shape": csv_json([int(value) for value in data.X.shape]),
        "n_cycles": int(data.n_cycles),
        "n_blocks": int(data.n_blocks),
        "blocks_per_cycle": int(blocks_per_cycle),
        "class_counts": csv_json(class_count_dict(data.y, data.task)),
        "n_splits": int(len(splits)),
        "split_protocol": "cycle_grouped_cv",
        "max_folds": int(max_folds),
    }
