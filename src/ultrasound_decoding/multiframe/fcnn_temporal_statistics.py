from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import FCNNMeanPool, count_trainable_parameters
from .training import (
    DeepTrainingConfig,
    _train_epochs,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


MEAN_ONLY_VARIANT = "mean_only"
MEAN_STD_VARIANT = "mean_std"
INPUT_VARIANTS = (MEAN_ONLY_VARIANT, MEAN_STD_VARIANT)
MODEL_NAME = "fcnn_bottleneck_temporal_statistics"
MODEL_IMPLEMENTATION_VERSION = "fcnn_mean_std_temporal_statistics_v1.0.0"
TEMPORAL_LENGTH = 4
BOTTLENECK_DIM = 3
STD_CORRECTION = 0


def build_bottleneck_temporal_statistics(
    representations: torch.Tensor, variant: str
) -> torch.Tensor:
    """Reduce [B, 4, 3] FCNN bottlenecks using frozen temporal statistics."""

    if representations.ndim != 3:
        raise ValueError(
            "expected FCNN bottlenecks [B, T, D], got "
            f"{tuple(representations.shape)}"
        )
    if int(representations.shape[1]) != TEMPORAL_LENGTH:
        raise ValueError(
            f"expected temporal length {TEMPORAL_LENGTH}, got "
            f"{int(representations.shape[1])}"
        )
    if int(representations.shape[2]) != BOTTLENECK_DIM:
        raise ValueError(
            f"expected bottleneck dimension {BOTTLENECK_DIM}, got "
            f"{int(representations.shape[2])}"
        )
    temporal_mean = representations.mean(dim=1)
    if variant == MEAN_ONLY_VARIANT:
        return temporal_mean
    if variant != MEAN_STD_VARIANT:
        raise ValueError(f"unknown temporal-statistics variant: {variant!r}")
    temporal_std = torch.std(
        representations, dim=1, correction=STD_CORRECTION
    )
    if bool((temporal_std < 0).any().item()):
        raise AssertionError("population temporal std contains a negative value")
    return torch.cat((temporal_mean, temporal_std), dim=1)


class FCNNMeanStd(FCNNMeanPool):
    """Formal FCNN mean-pool with population std appended before classification."""

    def __init__(
        self,
        n_classes: int,
        temporal_length: int = TEMPORAL_LENGTH,
        input_shape: tuple[int, int] = (128, 501),
    ) -> None:
        if int(temporal_length) != TEMPORAL_LENGTH:
            raise ValueError(
                f"v1 requires exactly {TEMPORAL_LENGTH} clean4 frames"
            )
        super().__init__(
            n_classes=n_classes,
            temporal_length=temporal_length,
            input_shape=input_shape,
        )
        # The inherited shared per-frame encoder is unchanged. Only the
        # classifier input expands from mean[3] to concat(mean, std)[6].
        self.classifier = nn.Linear(2 * self.encoder_feature_dim, n_classes)

    def forward_with_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        representations = self.encode_sequence(x)
        statistics = build_bottleneck_temporal_statistics(
            representations, MEAN_STD_VARIANT
        )
        mean, std = statistics.split(self.encoder_feature_dim, dim=1)
        return self.classifier(statistics), mean, std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _mean, _std = self.forward_with_statistics(x)
        return logits


def build_model(variant: str, n_classes: int = 2) -> nn.Module:
    if variant == MEAN_ONLY_VARIANT:
        # Instantiate the actual historical class: this is not a reimplementation.
        return FCNNMeanPool(
            n_classes=n_classes, temporal_length=TEMPORAL_LENGTH
        )
    if variant == MEAN_STD_VARIANT:
        return FCNNMeanStd(
            n_classes=n_classes, temporal_length=TEMPORAL_LENGTH
        )
    raise ValueError(f"unknown temporal-statistics variant: {variant!r}")


def architecture_config(variant: str, n_classes: int = 2) -> dict[str, Any]:
    model = build_model(variant, n_classes=n_classes)
    classifier_in = (
        BOTTLENECK_DIM
        if variant == MEAN_ONLY_VARIANT
        else 2 * BOTTLENECK_DIM
    )
    return {
        "method": MODEL_NAME,
        "variant": variant,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "input": "normalized clean4 [B,4,1,H,W]",
        "shared_frame_encoder": [
            ["MaxPool2d", [2, 2]],
            ["Flatten", None],
            ["Linear", [64 * 250, BOTTLENECK_DIM]],
            ["ReLU", None],
        ],
        "shared_frame_encoder_modified": False,
        "temporal_length": TEMPORAL_LENGTH,
        "bottleneck_shape": ["B", TEMPORAL_LENGTH, BOTTLENECK_DIM],
        "temporal_reduction": (
            "mean"
            if variant == MEAN_ONLY_VARIANT
            else "concat(mean,std_correction_0)"
        ),
        "classifier": ["Linear", [classifier_in, int(n_classes)]],
        "classifier_input_dim": classifier_in,
        "n_classes": int(n_classes),
        "trainable_parameters": count_trainable_parameters(model),
    }


def parameter_audit(n_classes: int = 2) -> dict[str, Any]:
    mean_only = build_model(MEAN_ONLY_VARIANT, n_classes=n_classes)
    mean_std = build_model(MEAN_STD_VARIANT, n_classes=n_classes)
    mean_only_parameters = count_trainable_parameters(mean_only)
    mean_std_parameters = count_trainable_parameters(mean_std)
    delta = mean_std_parameters - mean_only_parameters
    expected_delta = BOTTLENECK_DIM * int(n_classes)
    if delta != expected_delta:
        raise AssertionError(
            f"parameter delta {delta} differs from classifier-only delta "
            f"{expected_delta}"
        )
    if mean_only.encoder.state_dict().keys() != mean_std.encoder.state_dict().keys():
        raise AssertionError("FCNN encoder state structure differs between variants")
    return {
        "mean_only_trainable_parameters": mean_only_parameters,
        "mean_std_trainable_parameters": mean_std_parameters,
        "parameter_delta": delta,
        "parameter_delta_percentage": 100.0 * delta / mean_only_parameters,
        "delta_source": "classifier input dimension 3->6 only",
        "shared_frame_encoder_modified": False,
    }


def _summary(values: torch.Tensor, prefix: str) -> dict[str, Any]:
    array = values.detach().cpu().numpy()
    return {
        f"{prefix}_shape": list(array.shape),
        f"{prefix}_min": float(array.min()),
        f"{prefix}_max": float(array.max()),
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_std": float(array.std(ddof=0)),
        f"{prefix}_nan_count": int(np.isnan(array).sum()),
        f"{prefix}_inf_count": int(np.isinf(array).sum()),
    }


def bottleneck_statistics_audit(
    model: nn.Module,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    """Describe final-model bottleneck statistics without fitting any statistic."""

    model.eval()
    with torch.no_grad():
        representations = model.encode_sequence(tensor.to(device))
        mean = representations.mean(dim=1)
        std = torch.std(
            representations, dim=1, correction=STD_CORRECTION
        )
    result = {
        "split": split,
        "temporal_length": TEMPORAL_LENGTH,
        "bottleneck_dim": BOTTLENECK_DIM,
        "std_correction": STD_CORRECTION,
        "statistics_are_deterministic_reductions_not_fitted": True,
        "used_for_model_selection": False,
        "std_nonnegative": bool((std >= 0).all().item()),
        "std_all_zero": bool((std == 0).all().item()),
    }
    result.update(_summary(mean, "mean_channel"))
    result.update(_summary(std, "std_channel"))
    if (
        result["mean_channel_nan_count"]
        or result["mean_channel_inf_count"]
        or result["std_channel_nan_count"]
        or result["std_channel_inf_count"]
        or not result["std_nonnegative"]
    ):
        raise AssertionError("invalid FCNN bottleneck temporal statistics")
    return result


@dataclass
class TemporalStatisticsFoldResult:
    variant: str
    seed: int
    predictions: np.ndarray
    probabilities: np.ndarray
    model: nn.Module
    model_parameters: int
    history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    statistics_audit: dict[str, Any]
    final_training_loss: float
    final_trained_epochs: int
    device: str
    model_config: dict[str, Any]


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    variant: str,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    training_config: DeepTrainingConfig,
    device: str | None = "auto",
    workers: int = 0,
) -> TemporalStatisticsFoldResult:
    if variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown temporal-statistics variant: {variant!r}")
    set_reproducible_seed(seed)
    torch_device = resolve_device(device)
    train_norm, test_norm, normalization, _norm_mean, _norm_std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_test,
            session=str(session),
            task="binary",
            method=variant,
            seed=int(seed),
            fold=int(fold),
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    normalization.update(
        {
            "input_variant": variant,
            "preprocessing_order": (
                "clean4 -> per-frame arcsinh -> outer-train-fold all-frame "
                "pixel z-score -> unchanged shared FCNN encoder -> bottleneck "
                "temporal statistics"
            ),
            "temporal_statistics_space": "normalized-frame FCNN bottleneck",
            "temporal_statistics_are_fitted": False,
            "test_used_for_normalization_fit": False,
            "test_used_for_feature_scaling": False,
            "secondary_bottleneck_scaling": False,
        }
    )
    train_tensor = blocks_to_sequence_tensor(train_norm)
    test_tensor = blocks_to_sequence_tensor(test_norm)
    y_train_i = labels_to_class_indices(y_train, classes)
    model = build_model(variant, n_classes=len(classes)).to(torch_device)
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
    statistics_audit = {
        "session": str(session),
        "variant": variant,
        "seed": int(seed),
        "fold": int(fold),
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "test_used_for_fitted_statistics": False,
        "test_used_for_model_selection": False,
        "train": bottleneck_statistics_audit(
            model, train_tensor, device=torch_device, split="train"
        ),
        "test_diagnostic_only": bottleneck_statistics_audit(
            model, test_tensor, device=torch_device, split="test"
        ),
    }
    return TemporalStatisticsFoldResult(
        variant=variant,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=count_trainable_parameters(model),
        history=history,
        normalization_audit=normalization,
        statistics_audit=statistics_audit,
        final_training_loss=float(history[-1]["train_loss"]),
        final_trained_epochs=len(history),
        device=str(torch_device),
        model_config=architecture_config(variant, n_classes=len(classes)),
    )
