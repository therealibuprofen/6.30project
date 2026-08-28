from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from ultrasound_decoding.deep import FCNN


MODEL_NAME = "fcnn_canonical_single_frame"
MODEL_IMPLEMENTATION_VERSION = "fcnn_canonical_single_frame_v1.0.0"
HISTORICAL_METHOD = "fcnn_late_fusion"
HISTORICAL_BASE_MODEL = "official_single_frame_FCNN"
NORMALIZATION_TRANSFORM = "arcsinh_then_train_pixel_zscore"
EXPECTED_IMAGE_SHAPE = (128, 501)
EXPECTED_NORMALIZATION_SHAPE = (1, 128, 501)
EXPECTED_INPUT_SHAPE = [4, 128, 501]
EXPECTED_PARAMETERS = 48_011
EXPECTED_EPOCH = 40
EXPECTED_CLASSES = [0, 1]
TEMPORAL_MIDPOINT_S = 15.0
TIE_ATOL = 1e-12


@dataclass(frozen=True)
class CanonicalFrameSelection:
    positions: np.ndarray
    relative_times_s: np.ndarray
    distances_to_midpoint_s: np.ndarray
    ties: np.ndarray


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_canonical_positions(
    relative_times_s: np.ndarray,
    *,
    midpoint_s: float = TEMPORAL_MIDPOINT_S,
) -> CanonicalFrameSelection:
    """Select the midpoint-nearest frame; exact ties go to the earlier time."""

    times = np.asarray(relative_times_s, dtype=np.float64)
    one_dimensional = times.ndim == 1
    if one_dimensional:
        times = times[None, :]
    if times.ndim != 2 or times.shape[1] < 1:
        raise ValueError("relative_times_s must have shape [N,T] or [T]")
    if not np.isfinite(times).all():
        raise ValueError("relative_times_s contains NaN or Inf")
    if np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("clean4 timestamps must be strictly increasing")

    distances = np.abs(times - float(midpoint_s))
    minima = distances.min(axis=1, keepdims=True)
    candidates = np.isclose(distances, minima, rtol=0.0, atol=TIE_ATOL)
    ties = candidates.sum(axis=1) > 1
    # Timestamps are strictly increasing, so argmax returns the earlier frame
    # whenever the two midpoint-nearest candidates are tied.
    positions = candidates.argmax(axis=1).astype(np.int64)
    rows = np.arange(len(times), dtype=np.int64)
    selected_times = times[rows, positions]
    selected_distances = distances[rows, positions]
    result = CanonicalFrameSelection(
        positions=positions,
        relative_times_s=selected_times,
        distances_to_midpoint_s=selected_distances,
        ties=ties.astype(bool),
    )
    if one_dimensional:
        return CanonicalFrameSelection(
            positions=result.positions[:1],
            relative_times_s=result.relative_times_s[:1],
            distances_to_midpoint_s=result.distances_to_midpoint_s[:1],
            ties=result.ties[:1],
        )
    return result


def select_canonical_frames(
    blocks: np.ndarray,
    relative_times_s: np.ndarray,
) -> tuple[np.ndarray, CanonicalFrameSelection]:
    """Reduce [N,T,H,W] blocks to exactly one canonical [N,H,W] frame."""

    values = np.asarray(blocks)
    times = np.asarray(relative_times_s)
    if values.ndim != 4:
        raise ValueError(f"blocks must have shape [N,T,H,W], got {values.shape}")
    if tuple(values.shape[-2:]) != EXPECTED_IMAGE_SHAPE:
        raise ValueError(f"unexpected image shape {values.shape[-2:]}")
    if times.shape != values.shape[:2]:
        raise ValueError("timestamp shape differs from block/frame dimensions")
    selection = select_canonical_positions(times)
    rows = np.arange(len(values), dtype=np.int64)
    frames = values[rows, selection.positions]
    if frames.shape != (len(values), *EXPECTED_IMAGE_SHAPE):
        raise AssertionError("canonical selection did not yield one frame per block")
    return frames.astype(np.float32, copy=False), selection


def apply_saved_normalization(
    frames: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    transform: str,
) -> np.ndarray:
    """Apply saved historical train-fold statistics without fitting anything."""

    values = np.asarray(frames)
    saved_mean = np.asarray(mean)
    saved_std = np.asarray(std)
    if values.ndim != 3 or tuple(values.shape[-2:]) != EXPECTED_IMAGE_SHAPE:
        raise ValueError("canonical frames must have shape [N,128,501]")
    if saved_mean.shape != EXPECTED_NORMALIZATION_SHAPE:
        raise ValueError(f"normalization mean shape is {saved_mean.shape}")
    if saved_std.shape != EXPECTED_NORMALIZATION_SHAPE:
        raise ValueError(f"normalization std shape is {saved_std.shape}")
    if transform != NORMALIZATION_TRANSFORM:
        raise ValueError(f"unexpected normalization transform: {transform}")
    if not (
        np.isfinite(values).all()
        and np.isfinite(saved_mean).all()
        and np.isfinite(saved_std).all()
    ):
        raise ValueError("canonical input or saved normalization is non-finite")
    if np.any(saved_std <= 0):
        raise ValueError("saved normalization std must be positive")
    transformed = np.arcsinh(values.astype(np.float32, copy=False))
    normalized = (
        transformed
        - saved_mean.astype(np.float32, copy=False)
    ) / saved_std.astype(np.float32, copy=False)
    if not np.isfinite(normalized).all():
        raise ValueError("normalized canonical frames are non-finite")
    return normalized.astype(np.float32, copy=False)


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    expected_session: str,
    expected_seed: int,
    expected_fold: int,
    expected_train_cycles: str,
    expected_test_cycles: str,
) -> dict[str, Any]:
    """Validate all scientific/provenance fields needed for reconstruction."""

    model_config = payload.get("model_config", {})
    checks = {
        "method": payload.get("method") == HISTORICAL_METHOD,
        "base_model": model_config.get("base_model") == HISTORICAL_BASE_MODEL,
        "late_fusion_training_lineage": bool(
            model_config.get("late_fusion_probability_average", False)
        ),
        "temporal_length": int(model_config.get("temporal_length", -1)) == 4,
        "session": str(payload.get("session")) == str(expected_session),
        "seed": int(payload.get("seed", -1)) == int(expected_seed),
        "fold": int(payload.get("fold", -1)) == int(expected_fold),
        "task": payload.get("task") == "binary",
        "classes": list(payload.get("classes", [])) == EXPECTED_CLASSES,
        "max_epochs": int(payload.get("max_epochs", -1)) == EXPECTED_EPOCH,
        "final_epoch": int(payload.get("final_epoch", -1)) == EXPECTED_EPOCH,
        "train_cycles": str(payload.get("train_cycles")) == expected_train_cycles,
        "test_cycles": str(payload.get("test_cycles")) == expected_test_cycles,
        "input_shape": list(payload.get("input_shape", [])) == EXPECTED_INPUT_SHAPE,
        "model_parameters": int(payload.get("model_parameters", -1))
        == EXPECTED_PARAMETERS,
        "normalization_transform": payload.get("normalization_transform")
        == NORMALIZATION_TRANSFORM,
        "normalization_mean_shape": tuple(
            np.asarray(payload.get("normalization_mean")).shape
        )
        == EXPECTED_NORMALIZATION_SHAPE,
        "normalization_std_shape": tuple(
            np.asarray(payload.get("normalization_std")).shape
        )
        == EXPECTED_NORMALIZATION_SHAPE,
        "code_version": bool(str(payload.get("code_version", "")).strip()),
    }
    state = payload.get("model_state_dict", {})
    expected_state_shapes = {
        "2.weight": (3, 16000),
        "2.bias": (3,),
        "4.weight": (2, 3),
        "4.bias": (2,),
    }
    checks["state_keys"] = set(state) == set(expected_state_shapes)
    checks["state_shapes"] = checks["state_keys"] and all(
        tuple(state[key].shape) == shape
        for key, shape in expected_state_shapes.items()
    )
    checks["state_parameters"] = checks["state_keys"] and sum(
        int(value.numel()) for value in state.values()
    ) == EXPECTED_PARAMETERS
    mean = np.asarray(payload.get("normalization_mean"))
    std = np.asarray(payload.get("normalization_std"))
    checks["normalization_finite"] = bool(
        np.isfinite(mean).all() and np.isfinite(std).all()
    )
    checks["normalization_std_positive"] = bool(
        std.shape == EXPECTED_NORMALIZATION_SHAPE and np.all(std > 0)
    )
    failures = sorted(name for name, valid in checks.items() if not valid)
    if failures:
        raise AssertionError(
            "historical checkpoint provenance mismatch: " + ", ".join(failures)
        )
    return {"valid": True, "checks": checks}


def load_validated_checkpoint(
    path: Path | str,
    *,
    expected_sha256: str,
    expected_session: str,
    expected_seed: int,
    expected_fold: int,
    expected_train_cycles: str,
    expected_test_cycles: str,
) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(path)
    observed_sha256 = file_sha256(checkpoint_path)
    if observed_sha256 != str(expected_sha256):
        raise AssertionError("checkpoint SHA256 differs from source manifest")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    audit = validate_checkpoint_payload(
        payload,
        expected_session=expected_session,
        expected_seed=expected_seed,
        expected_fold=expected_fold,
        expected_train_cycles=expected_train_cycles,
        expected_test_cycles=expected_test_cycles,
    )
    model = FCNN(input_shape=EXPECTED_IMAGE_SHAPE, n_classes=2)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(torch.device("cpu"))
    model.eval()
    if count_parameters(model) != EXPECTED_PARAMETERS:
        raise AssertionError("loaded canonical FCNN parameter count differs")
    audit.update(
        {
            "checkpoint_sha256": observed_sha256,
            "checkpoint_path": str(checkpoint_path),
            "code_version": str(payload["code_version"]),
        }
    )
    return model, payload, audit


def predict_single_frame_probabilities(
    model: nn.Module,
    normalized_frames: np.ndarray,
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """CPU-only inference; every model forward receives canonical frames only."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    frames = np.asarray(normalized_frames, dtype=np.float32)
    if frames.ndim != 3 or tuple(frames.shape[-2:]) != EXPECTED_IMAGE_SHAPE:
        raise ValueError("normalized frames must have shape [N,128,501]")
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError("canonical reconstruction is CPU-only")
    model.eval()
    tensor = torch.from_numpy(frames[:, None, :, :])
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensor), int(batch_size)):
            logits = model(tensor[start : start + int(batch_size)])
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise AssertionError("single-frame FCNN logits must have shape [B,2]")
            outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = (
        np.concatenate(outputs, axis=0)
        if outputs
        else np.empty((0, 2), dtype=np.float32)
    )
    if len(probabilities) != len(frames):
        raise AssertionError("one canonical frame did not yield one prediction")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("invalid canonical single-frame probabilities")
    return probabilities.astype(np.float32, copy=False)
