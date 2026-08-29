from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn

from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.evaluate import confusion_matrix

from .dataset import EXPECTED_BLOCK_SHAPE, read_h5_strings
from .training import (
    DeepTrainingConfig,
    _train_epochs,
    blocks_to_frame_tensor,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


TASK_NAME = "block_identity_4class"
MODEL_NAME = "fcnn_fourclass_late_fusion"
MODEL_VERSION = "fcnn_fourclass_late_fusion_v1.0.0"
CLASS_NAMES = {
    0: "grating",
    1: "stop_after_grating",
    2: "dot",
    3: "static",
}
CLASS_TO_INDEX = {name: index for index, name in CLASS_NAMES.items()}
CLASSES = np.asarray(sorted(CLASS_NAMES), dtype=np.int64)
BLOCK_ORDER = [CLASS_NAMES[index] for index in CLASSES]
STIMULUS_CLASSES = (0, 2)
NONSTIMULUS_CLASSES = (1, 3)
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
CHANCE_LEVEL = 0.25
HISTORICAL_BINARY_PARAMETERS = 48_011
EXPECTED_FOURCLASS_PARAMETERS = 48_019
FRAMES_PER_BLOCK = 4


@dataclass(frozen=True)
class FourClassBlockData:
    session: str
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


@dataclass
class FourClassFoldResult:
    model: nn.Module
    history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    frame_probabilities: np.ndarray
    block_probabilities: np.ndarray
    predictions: np.ndarray
    train_frame_accuracy: float
    train_block_accuracy: float
    train_block_balanced_accuracy: float
    device: str


def architecture_config() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "input": "single_clean4_frame_[1,128,501]",
        "layers": [
            "MaxPool2d(kernel_size=2,stride=2)",
            "Flatten(16000)",
            "Linear(16000,3)",
            "ReLU",
            "Linear(3,4)",
        ],
        "fusion": "arithmetic_mean_of_four_frame_softmax_vectors",
        "classes": CLASS_NAMES,
    }


def build_model() -> nn.Module:
    return FCNN((EXPECTED_BLOCK_SHAPE[1], EXPECTED_BLOCK_SHAPE[2]), len(CLASSES))


def count_trainable_parameters(model: nn.Module | None = None) -> int:
    candidate = build_model() if model is None else model
    return int(sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad))


def parameter_audit() -> dict[str, int]:
    fourclass = count_trainable_parameters()
    return {
        "historical_binary_parameters": HISTORICAL_BINARY_PARAMETERS,
        "fourclass_parameters": fourclass,
        "delta_parameters": fourclass - HISTORICAL_BINARY_PARAMETERS,
    }


def load_fourclass_block_session(
    project_dir: Path,
    session: str,
    data_dir: Path | None = None,
) -> FourClassBlockData:
    """Load every clean4 block and derive the label only from block identity."""
    session = str(session)
    base = data_dir or project_dir / "processed_data" / "block_sequences_v1"
    h5_path = base / f"session_{session}_blocks.h5"
    metadata_path = base / f"session_{session}_block_metadata.csv"
    if not h5_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"missing clean4 source for session {session}")

    metadata = pd.read_csv(metadata_path)
    with h5py.File(h5_path, "r") as handle:
        X = handle["/clean4/X"][:]
        clean_times = handle["/clean4/relative_time_s"][:]
        clean_indices = handle["/clean4/original_frame_indices"][:]
        h5_cycles = handle["/metadata/cycle"][:].astype(np.int64)
        h5_block_names = read_h5_strings(handle["/metadata/block_name"])
        h5_block_ids = read_h5_strings(handle["/metadata/block_id"])

    if len(metadata) != len(X):
        raise AssertionError("metadata and clean4 block counts differ")
    names = metadata["block_name"].astype(str).tolist()
    if names != h5_block_names:
        raise AssertionError("metadata and HDF5 block identities differ")
    if metadata["block_id"].astype(str).tolist() != h5_block_ids:
        raise AssertionError("metadata and HDF5 block IDs differ")
    if not np.array_equal(metadata["cycle"].to_numpy(np.int64), h5_cycles):
        raise AssertionError("metadata and HDF5 cycle order differ")
    unknown = sorted(set(names) - set(CLASS_TO_INDEX))
    if unknown:
        raise AssertionError(f"unknown block identities: {unknown}")

    data = FourClassBlockData(
        session=session,
        X=X.astype(np.float32, copy=False),
        y=np.asarray([CLASS_TO_INDEX[name] for name in names], dtype=np.int64),
        groups=h5_cycles,
        metadata=metadata.copy(),
        clean4_relative_time_s=clean_times.astype(np.float32, copy=False),
        clean4_original_frame_indices=clean_indices.astype(np.int64, copy=False),
        source_h5_path=h5_path,
        source_metadata_path=metadata_path,
    )
    validate_fourclass_data(data)
    return data


def validate_fourclass_data(data: FourClassBlockData) -> None:
    if tuple(data.X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(f"expected clean4 shape {EXPECTED_BLOCK_SHAPE}, got {data.X.shape}")
    if len(data.X) != len(data.y) or len(data.X) != len(data.groups):
        raise AssertionError("X, labels, and groups have different lengths")
    if not np.isfinite(data.X).all():
        raise AssertionError("clean4 contains NaN or Inf")
    if set(np.unique(data.y).tolist()) != set(CLASSES.tolist()):
        raise AssertionError("the dataset must contain exactly four nonnegative classes")
    if data.clean4_relative_time_s.shape != (len(data.X), FRAMES_PER_BLOCK):
        raise AssertionError("clean4 times must have shape [N,4]")
    if data.clean4_original_frame_indices.shape != (len(data.X), FRAMES_PER_BLOCK):
        raise AssertionError("clean4 frame indices must have shape [N,4]")
    if not np.all(data.metadata["n_frames_clean4"].to_numpy(int) == FRAMES_PER_BLOCK):
        raise AssertionError("a block does not contain exactly four clean frames")
    if not np.all(data.metadata["complete_cycle"].astype(bool).to_numpy()):
        raise AssertionError("an incomplete cycle entered the formal dataset")

    for cycle, rows in data.metadata.groupby("cycle", sort=True):
        ordered = rows.sort_values("block_order_in_cycle")["block_name"].astype(str).tolist()
        if ordered != BLOCK_ORDER:
            raise AssertionError(f"cycle {int(cycle)} has block order {ordered}")
        labels = sorted(data.y[rows.index.to_numpy(int)].tolist())
        if labels != CLASSES.tolist():
            raise AssertionError(f"cycle {int(cycle)} does not contain one of every class")
    counts = np.bincount(data.y, minlength=len(CLASSES))
    if not np.all(counts == data.n_cycles):
        raise AssertionError(f"class balance is not exact: {counts.tolist()}")


def expand_training_frames(X: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    if X.ndim != 4 or X.shape[1] != FRAMES_PER_BLOCK or len(X) != len(y):
        raise ValueError("expected N clean4 blocks and N labels")
    return blocks_to_frame_tensor(X), np.repeat(np.asarray(y, dtype=np.int64), FRAMES_PER_BLOCK)


def late_fuse_probabilities(frame_probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(frame_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASSES):
        raise ValueError("frame probabilities must have shape [4N,4]")
    if len(probabilities) % FRAMES_PER_BLOCK:
        raise ValueError("each block must contribute exactly four frames")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
        raise ValueError("frame probability rows must be finite and sum to one")
    fused = probabilities.reshape(-1, FRAMES_PER_BLOCK, len(CLASSES)).mean(axis=1)
    if not np.allclose(fused.sum(1), 1.0, atol=1e-6):
        raise AssertionError("fused probability rows do not sum to one")
    return fused.astype(np.float32)


def fixed_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), CLASSES)
    recalls, f1s = [], []
    for index in range(len(CLASSES)):
        tp = float(matrix[index, index])
        recall = tp / max(float(matrix[index].sum()), 1.0)
        precision = tp / max(float(matrix[:, index].sum()), 1.0)
        recalls.append(recall)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    result = {
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
    }
    result.update({f"recall_{CLASS_NAMES[index]}": float(recalls[index]) for index in range(4)})
    return result


def collapsed_binary_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("four-class probabilities must have shape [N,4]")
    collapsed = np.column_stack(
        [values[:, NONSTIMULUS_CLASSES].sum(axis=1), values[:, STIMULUS_CLASSES].sum(axis=1)]
    )
    if not np.isfinite(collapsed).all() or not np.allclose(collapsed.sum(1), 1.0, atol=1e-6):
        raise AssertionError("collapsed probability rows must be finite and sum to one")
    return collapsed


def collapsed_binary_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if not set(np.unique(values)).issubset(set(CLASSES.tolist())):
        raise ValueError("labels outside the frozen four-class mapping")
    return np.isin(values, STIMULUS_CLASSES).astype(np.int64)


def binary_metrics_from_fourclass(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    truth = collapsed_binary_labels(y_true)
    pred = collapsed_binary_probabilities(probabilities).argmax(axis=1)
    matrix = confusion_matrix(truth, pred, np.asarray([0, 1], dtype=np.int64))
    recalls = [matrix[i, i] / max(matrix[i].sum(), 1) for i in range(2)]
    return {
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "balanced_accuracy": float(np.mean(recalls)),
    }


def coarse_error_audit(matrix: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(matrix, dtype=np.int64)
    if values.shape != (4, 4):
        raise ValueError("four-class confusion matrix must be 4x4")
    within_pairs = {(0, 2), (2, 0), (1, 3), (3, 1)}
    within = sum(int(values[i, j]) for i, j in within_pairs)
    total_errors = int(values.sum() - np.trace(values))
    cross = total_errors - within
    return {
        "within_coarse_error_count": within,
        "cross_coarse_error_count": cross,
        "within_coarse_error_fraction": float(within / total_errors) if total_errors else 0.0,
        "cross_coarse_error_fraction": float(cross / total_errors) if total_errors else 0.0,
    }


def normalized_confusion(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return np.divide(values, values.sum(axis=1, keepdims=True), out=np.zeros_like(values), where=values.sum(axis=1, keepdims=True) != 0)


def feasibility_gate(mean_ba: float, session_bas: np.ndarray, collapsed_mean_ba: float) -> dict[str, Any]:
    bas = np.asarray(session_bas, dtype=np.float64)
    conditions = {
        "mean_fourclass_ba_gte_0_35": bool(mean_ba >= 0.35),
        "at_least_6_of_9_sessions_gt_0_30": bool(np.sum(bas > 0.30) >= 6),
        "mean_collapsed_binary_ba_gte_0_55": bool(collapsed_mean_ba >= 0.55),
    }
    passed = all(conditions.values())
    return {
        "thresholds_frozen_before_results": True,
        "conditions": conditions,
        "sessions_gt_0_30": int(np.sum(bas > 0.30)),
        "decision": (
            "four_class_signal_sufficient_for_multitask_experiment"
            if passed
            else "four_class_signal_insufficient_for_multitask_experiment"
        ),
    }


def frozen_training_config(epochs: int = 40) -> DeepTrainingConfig:
    return DeepTrainingConfig(
        optimizer="adamw",
        lr=0.001,
        weight_decay=0.001,
        batch_size=16,
        max_epochs=int(epochs),
        dropout=0.0,
        loss="cross_entropy",
    )


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    session: str,
    seed: int,
    fold: int,
    train_cycles: str,
    test_cycles: str,
    config: DeepTrainingConfig | None = None,
    device: str = "auto",
) -> FourClassFoldResult:
    training = config or frozen_training_config()
    set_reproducible_seed(seed)
    torch_device = resolve_device(device)
    train_norm, test_norm, audit, mean, std = normalize_blocks_train_fold_only_with_stats(
        X_train,
        X_test,
        session=session,
        task=TASK_NAME,
        method=MODEL_NAME,
        seed=seed,
        fold=fold,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
    )
    audit.update(
        {
            "test_used_for_normalization_fit": False,
            "test_used_for_feature_scaling": False,
            "secondary_bottleneck_scaling": False,
        }
    )
    train_tensor, train_labels = expand_training_frames(train_norm, y_train)
    test_tensor = blocks_to_frame_tensor(test_norm)
    model = build_model().to(torch_device)
    history = _train_epochs(
        model,
        train_tensor,
        train_labels,
        config=training,
        seed=seed,
        device=torch_device,
        batch_size_reference=len(train_labels),
    )
    test_frame_probs = predict_probabilities(
        model, test_tensor, device=torch_device, batch_size=training.batch_size
    )
    block_probs = late_fuse_probabilities(test_frame_probs)
    train_frame_probs = predict_probabilities(
        model, train_tensor, device=torch_device, batch_size=training.batch_size
    )
    train_block_probs = late_fuse_probabilities(train_frame_probs)
    train_block_pred = train_block_probs.argmax(axis=1)
    train_metrics = fixed_class_metrics(y_train, train_block_pred)
    return FourClassFoldResult(
        model=model,
        history=history,
        normalization_audit=audit,
        normalization_mean=mean,
        normalization_std=std,
        frame_probabilities=test_frame_probs,
        block_probabilities=block_probs,
        predictions=block_probs.argmax(axis=1).astype(np.int64),
        train_frame_accuracy=float((train_frame_probs.argmax(axis=1) == train_labels).mean()),
        train_block_accuracy=float(train_metrics["accuracy"]),
        train_block_balanced_accuracy=float(train_metrics["balanced_accuracy"]),
        device=str(torch_device),
    )


def json_list(values: np.ndarray | list[Any]) -> str:
    return json.dumps(np.asarray(values).tolist(), ensure_ascii=False, separators=(",", ":"))
