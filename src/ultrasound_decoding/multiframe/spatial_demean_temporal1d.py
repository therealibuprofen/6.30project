from __future__ import annotations

from typing import Any

import numpy as np

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
SPATIAL_DEMEAN_VARIANT = "spatial_demean"
INPUT_VARIANTS = (RAW_VARIANT, SPATIAL_DEMEAN_VARIANT)
MODEL_NAME = "cnn2d_temporal1d"
MODEL_IMPLEMENTATION_VERSION = "spatial_demean_temporal1d_v1.0.0"


def spatial_demean_per_frame(x: np.ndarray) -> np.ndarray:
    """Subtract each frame's own spatial mean from an [N,T,H,W] tensor.

    The reduction is strictly over H and W.  No sample, temporal, cycle, label,
    train-fold, or test-fold statistic enters this deterministic transform.
    """

    array = np.asarray(x)
    if array.ndim != 4:
        raise ValueError(f"expected [N,T,H,W], got {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"spatial demean requires floating input, got {array.dtype}")
    spatial_mean = array.mean(axis=(-2, -1), keepdims=True, dtype=array.dtype)
    return array - spatial_mean


def apply_input_variant_after_arcsinh(
    x_arcsinh: np.ndarray, input_variant: str
) -> np.ndarray:
    """Apply the only experimental variable after arcsinh and before z-score."""

    if input_variant == RAW_VARIANT:
        return x_arcsinh
    if input_variant == SPATIAL_DEMEAN_VARIANT:
        return spatial_demean_per_frame(x_arcsinh)
    raise ValueError(
        f"input_variant must be one of {INPUT_VARIANTS}, got {input_variant!r}"
    )


def preprocess_and_normalize_train_fold_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    input_variant: str,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Run the frozen clean4 preprocessing with one optional deterministic step.

    Frozen order:
      clean4 extraction (upstream loader)
      -> arcsinh (existing deterministic per-sample transform)
      -> per-frame spatial demean (spatial_demean only)
      -> pixel-wise z-score fitted on outer-train blocks/all four frames only

    The raw branch calls the existing reviewed normalization implementation
    directly so its numerical behavior remains exactly the formal baseline.
    """

    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown input variant: {input_variant!r}")
    common = dict(
        session=str(session),
        task="binary",
        method=str(input_variant),
        seed=int(seed),
        fold=int(fold),
        train_cycles=train_cycles,
        test_cycles=test_cycles,
    )
    if input_variant == RAW_VARIANT:
        train, test, audit, mean, std = normalize_blocks_train_fold_only_with_stats(
            X_train, X_test, **common
        )
        audit.update(
            {
                "input_variant": RAW_VARIANT,
                "preprocessing_order": (
                    "clean4 -> arcsinh -> train_fold_pixel_zscore"
                ),
                "spatial_demean_applied": False,
                "spatial_demean_axes": None,
                "spatial_demean_statistics_scope": None,
            }
        )
        return train, test, audit, mean, std

    if X_train.ndim != 4 or X_test.ndim != 4:
        raise ValueError(
            f"expected [N,T,H,W], got {X_train.shape} and {X_test.shape}"
        )
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("train and test block shapes differ")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("deep model input contains NaN or Inf values")

    # This is the only difference from raw: demean each arcsinh-transformed
    # frame independently before any train-fold-fitted statistic is computed.
    train_asinh = np.arcsinh(X_train.astype(np.float32, copy=False))
    test_asinh = np.arcsinh(X_test.astype(np.float32, copy=False))
    train_variant = apply_input_variant_after_arcsinh(
        train_asinh, SPATIAL_DEMEAN_VARIANT
    )
    test_variant = apply_input_variant_after_arcsinh(
        test_asinh, SPATIAL_DEMEAN_VARIANT
    )

    train_frames = train_variant.reshape(
        -1, train_variant.shape[-2], train_variant.shape[-1]
    ).astype(np.float64, copy=False)
    mean = train_frames.mean(axis=0, keepdims=True)
    std_raw = train_frames.std(axis=0, keepdims=True)
    std = std_raw + 1e-6
    train_norm = (train_variant - mean) / std
    test_norm = (test_variant - mean) / std
    if not np.isfinite(train_norm).all() or not np.isfinite(test_norm).all():
        raise ValueError("deep normalized data contains NaN or Inf values")

    audit = {
        **common,
        "phase": "outer_train_fold_only",
        "input_variant": SPATIAL_DEMEAN_VARIANT,
        "transform": "arcsinh_then_frame_spatial_demean_then_train_pixel_zscore",
        "preprocessing_order": (
            "clean4 -> arcsinh -> frame_spatial_demean -> "
            "train_fold_pixel_zscore"
        ),
        "spatial_demean_applied": True,
        "spatial_demean_axes": "H,W",
        "spatial_demean_statistics_scope": "one_sample_one_frame_only",
        "statistics_scope": "train_blocks_all_four_frames_only",
        "target_used_for_stats": False,
        "test_fold_used_for_fitted_statistics": False,
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "n_train_blocks": int(len(X_train)),
        "n_test_blocks": int(len(X_test)),
        "temporal_length": int(X_train.shape[1]),
        "n_train_frames_for_stats": int(len(X_train) * X_train.shape[1]),
        "n_test_frames_transformed": int(len(X_test) * X_test.shape[1]),
        "epsilon": 1e-6,
        "mean_mean": float(mean.mean()),
        "mean_std": float(mean.std()),
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_mean": float(std_raw.mean()),
        "std_std": float(std_raw.std()),
        "std_min": float(std_raw.min()),
        "std_max": float(std_raw.max()),
        "train_nan_count": int(np.isnan(X_train).sum()),
        "train_inf_count": int(np.isinf(X_train).sum()),
        "test_nan_count": int(np.isnan(X_test).sum()),
        "test_inf_count": int(np.isinf(X_test).sum()),
    }
    return (
        train_norm.astype(np.float32, copy=False),
        test_norm.astype(np.float32, copy=False),
        audit,
        mean.astype(np.float32, copy=True),
        std.astype(np.float32, copy=True),
    )


def formal_architecture_config() -> dict[str, Any]:
    config = model_architecture_config(MODEL_NAME, n_classes=2, temporal_length=4)
    config.update(
        {
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "input_variant_is_outside_model": True,
            "raw_and_spatial_demean_share_architecture": True,
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
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    training_config: DeepTrainingConfig,
    device: str | None = "auto",
    workers: int = 0,
) -> FoldTrainingResult:
    """Train the unchanged formal CNN2DTemporal1D for one paired task."""

    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, audit, norm_mean, norm_std = (
        preprocess_and_normalize_train_fold_only(
            X_train,
            X_test,
            input_variant=input_variant,
            session=session,
            fold=fold,
            seed=seed,
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    model = CNN2DTemporal1D(
        n_classes=len(classes), temporal_length=int(X_train_norm.shape[1])
    ).to(torch_device)
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
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_config,
    )
