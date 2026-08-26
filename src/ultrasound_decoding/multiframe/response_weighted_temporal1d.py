from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

import numpy as np
from scipy.stats import rankdata

from .models import (
    CNN2DTemporal1D,
    count_trainable_parameters,
    model_architecture_config,
)
from .training import (
    DeepTrainingConfig,
    FoldTrainingResult,
    _train_epochs,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


RAW_VARIANT = "raw"
RESPONSE_WEIGHTED_VARIANT = "response_weighted"
INPUT_VARIANTS = (RAW_VARIANT, RESPONSE_WEIGHTED_VARIANT)
MODEL_NAME = "cnn2d_temporal1d"
MODEL_IMPLEMENTATION_VERSION = "response_weighted_temporal1d_v1.0.0"
RESPONSE_MAP_IMPLEMENTATION_VERSION = "training_cycle_presence_snr_rank_v1"
RESPONSE_EPS = 1e-6
BLOCK_NAMES = ("grating", "stop_after_grating", "dot", "static")


@dataclass(frozen=True)
class TrainingResponseMap:
    response_score: np.ndarray
    weight_map: np.ndarray
    cycle_contrasts: np.ndarray
    train_cycle_ids: tuple[int, ...]
    response_map_hash: str
    weight_map_hash: str
    eps: float


def _array_hash(
    array: np.ndarray,
    *,
    kind: str,
    train_cycle_ids: tuple[int, ...],
) -> str:
    contiguous = np.ascontiguousarray(array.astype("<f4", copy=False))
    header = json.dumps(
        {
            "implementation": RESPONSE_MAP_IMPLEMENTATION_VERSION,
            "kind": kind,
            "shape": list(contiguous.shape),
            "dtype": str(contiguous.dtype),
            "train_cycle_ids": list(train_cycle_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def response_map_cache_key(
    *,
    session: str,
    fold: int,
    train_cycle_ids: Sequence[int],
    relevant_source_sha256: str,
) -> str:
    """Return a fold-specific in-memory cache key; never a session-only key."""

    cycles = tuple(sorted({int(value) for value in train_cycle_ids}))
    if not cycles:
        raise ValueError("response-map cache key requires training cycles")
    payload = {
        "session": str(session),
        "fold": int(fold),
        "train_cycle_ids": list(cycles),
        "input_protocol": "clean4_arcsinh",
        "implementation": RESPONSE_MAP_IMPLEMENTATION_VERSION,
        "relevant_source_sha256": str(relevant_source_sha256),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def presence_contrast_from_block_images(
    grating: np.ndarray,
    stop_after_grating: np.ndarray,
    dot: np.ndarray,
    static: np.ndarray,
) -> np.ndarray:
    arrays = [np.asarray(value) for value in (grating, stop_after_grating, dot, static)]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("presence-contrast block images must have identical shapes")
    return 0.5 * (arrays[0] + arrays[2]) - 0.5 * (arrays[1] + arrays[3])


def build_training_cycle_presence_contrasts(
    X_train: np.ndarray,
    block_names_train: Sequence[str],
    cycle_ids_train: Sequence[int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build one G/D minus S/T contrast per training cycle.

    This API intentionally accepts only a training subset. It has no test-data,
    full-session-map, GLM-map, label, or held-out-index argument.
    """

    X = np.asarray(X_train)
    names = np.asarray(block_names_train, dtype=str)
    cycles = np.asarray(cycle_ids_train, dtype=np.int64)
    if X.ndim != 4:
        raise ValueError(f"expected training clean4 [N,T,H,W], got {X.shape}")
    if X.shape[1] != 4:
        raise ValueError(f"response map requires frozen clean4 T=4, got {X.shape}")
    if len(X) != len(names) or len(X) != len(cycles):
        raise ValueError("training blocks, block names, and cycle ids differ in length")
    if not np.isfinite(X).all():
        raise ValueError("training response-map input contains NaN or Inf")
    unique_cycles = tuple(sorted(int(value) for value in np.unique(cycles)))
    if len(unique_cycles) < 2:
        raise ValueError("stable response score requires at least two training cycles")

    # Reuse the formal deterministic transform and clean4 frames, then average
    # only the four frames belonging to each training block.
    block_images = np.arcsinh(X.astype(np.float32, copy=False)).mean(
        axis=1, dtype=np.float64
    )
    contrasts: list[np.ndarray] = []
    for cycle in unique_cycles:
        in_cycle = np.flatnonzero(cycles == cycle)
        lookup: dict[str, np.ndarray] = {}
        for block_name in BLOCK_NAMES:
            matches = in_cycle[names[in_cycle] == block_name]
            if len(matches) != 1:
                raise AssertionError(
                    f"training cycle {cycle} requires exactly one {block_name} block; "
                    f"observed={len(matches)}"
                )
            lookup[block_name] = block_images[int(matches[0])]
        contrast = presence_contrast_from_block_images(
            lookup["grating"],
            lookup["stop_after_grating"],
            lookup["dot"],
            lookup["static"],
        )
        contrasts.append(contrast)
    return np.stack(contrasts).astype(np.float32), unique_cycles


def response_score_from_cycle_contrasts(
    cycle_contrasts: np.ndarray, *, eps: float = RESPONSE_EPS
) -> np.ndarray:
    contrasts = np.asarray(cycle_contrasts)
    if contrasts.ndim != 3 or len(contrasts) < 2:
        raise ValueError(f"expected [C,H,W] with C>=2, got {contrasts.shape}")
    if not np.isfinite(contrasts).all():
        raise ValueError("cycle contrasts contain NaN or Inf")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be a positive finite scalar")
    values = contrasts.astype(np.float64, copy=False)
    response_mean = values.mean(axis=0)
    response_std = values.std(axis=0, ddof=0)
    score = np.abs(response_mean) / (response_std + float(eps))
    if not np.isfinite(score).all():
        raise ValueError("response score contains NaN or Inf")
    return score.astype(np.float32)


def response_score_to_soft_weight(response_score: np.ndarray) -> np.ndarray:
    """Map spatial response scores to average-tie percentile ranks in [0.5,1.5]."""

    score = np.asarray(response_score)
    if score.ndim != 2 or score.size < 2:
        raise ValueError(f"expected a nontrivial [H,W] response score, got {score.shape}")
    if not np.isfinite(score).all():
        raise ValueError("response score contains NaN or Inf")
    ranks = rankdata(score.reshape(-1), method="average")
    percentile = (ranks - 1.0) / float(score.size - 1)
    weight = (0.5 + percentile).reshape(score.shape).astype(np.float32)
    tolerance = 1e-6
    if float(weight.min()) < 0.5 - tolerance or float(weight.max()) > 1.5 + tolerance:
        raise AssertionError("soft response weight falls outside [0.5,1.5]")
    if not np.isclose(float(weight.mean()), 1.0, atol=2e-6):
        raise AssertionError("average-rank response weight must have spatial mean 1")
    return weight


def build_training_response_map(
    X_train: np.ndarray,
    block_names_train: Sequence[str],
    cycle_ids_train: Sequence[int],
    *,
    eps: float = RESPONSE_EPS,
) -> TrainingResponseMap:
    contrasts, train_cycles = build_training_cycle_presence_contrasts(
        X_train, block_names_train, cycle_ids_train
    )
    score = response_score_from_cycle_contrasts(contrasts, eps=eps)
    weight = response_score_to_soft_weight(score)
    return TrainingResponseMap(
        response_score=score,
        weight_map=weight,
        cycle_contrasts=contrasts,
        train_cycle_ids=train_cycles,
        response_map_hash=_array_hash(
            score, kind="response_score", train_cycle_ids=train_cycles
        ),
        weight_map_hash=_array_hash(
            weight, kind="soft_weight", train_cycle_ids=train_cycles
        ),
        eps=float(eps),
    )


def apply_response_weight(X_normalized: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    X = np.asarray(X_normalized)
    weight = np.asarray(weight_map)
    if X.ndim != 4 or weight.ndim != 2 or X.shape[-2:] != weight.shape:
        raise ValueError(f"expected [N,T,H,W] and matching [H,W], got {X.shape}, {weight.shape}")
    if not np.isfinite(X).all() or not np.isfinite(weight).all():
        raise ValueError("normalized input or response weight contains NaN or Inf")
    return (X * weight[None, None, :, :]).astype(np.float32, copy=False)


def preprocess_and_normalize_train_fold_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    input_variant: str,
    response_map: TrainingResponseMap | None,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Normalize exactly as baseline, then optionally apply one shared fixed W."""

    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown input variant: {input_variant!r}")
    train_norm, test_norm, audit, mean, std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_test,
            session=str(session),
            task="binary",
            method=input_variant,
            seed=int(seed),
            fold=int(fold),
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    audit.update(
        {
            "input_variant": input_variant,
            "response_weighting_after_normalization": (
                input_variant == RESPONSE_WEIGHTED_VARIANT
            ),
            "response_map_fit_scope": (
                "training_cycles_only"
                if input_variant == RESPONSE_WEIGHTED_VARIANT
                else None
            ),
        }
    )
    if input_variant == RAW_VARIANT:
        if response_map is not None:
            raise AssertionError("raw path must not receive a response map")
        audit.update(
            {
                "preprocessing_order": "clean4 -> arcsinh -> train_fold_pixel_zscore",
                "response_map_hash": None,
                "train_and_test_weight_map_hash": None,
            }
        )
        return train_norm, test_norm, audit, mean, std

    if response_map is None:
        raise AssertionError("response_weighted path requires one training-fold map")
    weighted_train = apply_response_weight(train_norm, response_map.weight_map)
    weighted_test = apply_response_weight(test_norm, response_map.weight_map)
    # Both branches above consume the same immutable map object. Record the one
    # deterministic hash twice and assert equality to prevent test-path refits.
    train_weight_hash = response_map.weight_map_hash
    test_weight_hash = response_map.weight_map_hash
    if train_weight_hash != test_weight_hash:
        raise AssertionError("train and test did not use the same response weight")
    audit.update(
        {
            "preprocessing_order": (
                "clean4 -> arcsinh -> train_fold_pixel_zscore -> "
                "training_fold_response_weight"
            ),
            "response_map_hash": response_map.response_map_hash,
            "weight_map_hash": response_map.weight_map_hash,
            "train_and_test_weight_map_hash": train_weight_hash,
            "test_data_used_to_fit_response_map": False,
        }
    )
    return weighted_train, weighted_test, audit, mean, std


def response_map_audit(
    response_map: TrainingResponseMap,
    *,
    session: str,
    fold: int,
    train_cycle_ids: Sequence[int],
    test_cycle_ids: Sequence[int],
    cache_key: str,
) -> dict[str, Any]:
    train_cycles = tuple(sorted({int(value) for value in train_cycle_ids}))
    test_cycles = tuple(sorted({int(value) for value in test_cycle_ids}))
    overlap = sorted(set(train_cycles) & set(test_cycles))
    if overlap:
        raise AssertionError(f"response-map train/test cycle overlap: {overlap}")
    if train_cycles != response_map.train_cycle_ids:
        raise AssertionError("response-map cycle provenance differs from outer train fold")
    score = response_map.response_score
    weight = response_map.weight_map
    return {
        "session": str(session),
        "fold": int(fold),
        "train_cycle_ids": list(train_cycles),
        "test_cycle_ids": list(test_cycles),
        "train_test_cycle_overlap": False,
        "construction_scope": "training_cycles_only",
        "full_session_glm_loaded": False,
        "response_map_shape": list(score.shape),
        "response_map_min": float(score.min()),
        "response_map_max": float(score.max()),
        "response_map_mean": float(score.mean()),
        "response_map_std": float(score.std()),
        "weight_map_min": float(weight.min()),
        "weight_map_max": float(weight.max()),
        "weight_map_mean": float(weight.mean()),
        "weight_map_std": float(weight.std()),
        "response_map_hash": response_map.response_map_hash,
        "weight_map_hash": response_map.weight_map_hash,
        "response_map_cache_key": str(cache_key),
        "response_map_implementation_version": RESPONSE_MAP_IMPLEMENTATION_VERSION,
        "eps": response_map.eps,
    }


def formal_architecture_config() -> dict[str, Any]:
    config = model_architecture_config(MODEL_NAME, n_classes=2, temporal_length=4)
    config.update(
        {
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "input_variant_is_outside_model": True,
            "raw_and_response_weighted_share_architecture": True,
        }
    )
    return config


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    input_variant: str,
    response_map: TrainingResponseMap | None,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    training_config: DeepTrainingConfig,
    device: str | None = "auto",
    workers: int = 0,
) -> FoldTrainingResult:
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_ready, X_test_ready, audit, norm_mean, norm_std = (
        preprocess_and_normalize_train_fold_only(
            X_train,
            X_test,
            input_variant=input_variant,
            response_map=response_map,
            session=session,
            fold=fold,
            seed=seed,
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_ready)
    test_tensor = blocks_to_sequence_tensor(X_test_ready)
    model = CNN2DTemporal1D(n_classes=len(classes), temporal_length=4).to(torch_device)
    parameters = count_trainable_parameters(model)
    history = _train_epochs(
        model,
        train_tensor,
        y_train_i,
        config=training_config,
        seed=seed,
        device=torch_device,
        batch_size_reference=len(X_train),
        num_workers=workers,
    )
    probabilities = predict_probabilities(
        model,
        test_tensor,
        device=torch_device,
        batch_size=training_config.batch_size,
        num_workers=workers,
    )
    predictions = classes[probabilities.argmax(axis=1)]
    model_config = formal_architecture_config()
    model_config["parameter_count"] = int(parameters)
    return FoldTrainingResult(
        method=input_variant,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=int(parameters),
        history=history,
        normalization_audit=audit,
        final_training_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        final_trained_epochs=len(history),
        device=str(torch_device),
        X_test_normalized=X_test_ready,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_config,
    )
