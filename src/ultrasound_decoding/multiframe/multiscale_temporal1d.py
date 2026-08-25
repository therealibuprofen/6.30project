from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ultrasound_decoding.deep import MPSCompatibleAdaptiveAvgPool2d

from .models import CNN2DTemporal1D, SmallCNNFrameEncoder, count_trainable_parameters
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


SINGLE_SCALE_MODEL_NAME = "same_backbone_single_scale"
MODEL_NAME = "multiscale_temporal1d"
MODEL_NAMES = (SINGLE_SCALE_MODEL_NAME, MODEL_NAME)
ENCODER_MODE_BY_MODEL_NAME = {
    SINGLE_SCALE_MODEL_NAME: "single_scale",
    MODEL_NAME: "multiscale",
}
MODEL_DISPLAY_NAMES = {
    SINGLE_SCALE_MODEL_NAME: "Same-backbone Single-scale CNN + Temporal 1D-CNN",
    MODEL_NAME: "Lightweight Multi-scale Spatial CNN + Temporal 1D-CNN",
}
MODEL_IMPLEMENTATION_VERSION = "multiscale_temporal1d_v1.0.0"
FORMAL_TEMPORAL_BASELINE_NAME = "cnn2d_temporal1d"
EXPECTED_PARAMETER_COUNTS = {
    SINGLE_SCALE_MODEL_NAME: 112_562,
    MODEL_NAME: 112_562,
    FORMAL_TEMPORAL_BASELINE_NAME: 115_890,
}


@dataclass(frozen=True)
class MultiScaleTemporal1DConfig:
    temporal_length: int = 4
    input_channels: int = 1
    first_stage_channels: int = 8
    second_stage_channels: int = 16
    branch_channels: int = 8
    first_kernel: tuple[int, int] = (5, 9)
    first_padding: tuple[int, int] = (2, 4)
    first_pool: tuple[int, int] = (2, 4)
    branch_kernel: tuple[int, int] = (3, 3)
    local_dilation: int = 1
    context_dilation: int = 2
    output_pool: tuple[int, int] = (4, 8)
    frame_feature_dim: int = 512
    temporal_channels: int = 64
    temporal_kernel_size: int = 3
    temporal_norm: str = "batchnorm"
    dropout: float = 0.25
    n_classes: int = 2


class ControlledScaleFrameEncoder(nn.Module):
    """One definition for the single-scale control and two-branch candidate."""

    feature_dim = 16 * 4 * 8

    def __init__(
        self,
        encoder_mode: str,
        config: MultiScaleTemporal1DConfig | None = None,
    ) -> None:
        super().__init__()
        if encoder_mode not in {"single_scale", "multiscale"}:
            raise ValueError("encoder_mode must be 'single_scale' or 'multiscale'")
        self.encoder_mode = encoder_mode
        self.config = config or MultiScaleTemporal1DConfig()
        cfg = self.config
        if (
            cfg.temporal_length != 4
            or cfg.input_channels != 1
            or cfg.first_stage_channels != 8
            or cfg.second_stage_channels != 16
            or cfg.branch_channels != 8
            or cfg.output_pool != (4, 8)
            or cfg.frame_feature_dim != self.feature_dim
        ):
            raise ValueError("v1 requires T4, channels 1/8/16, 8-channel branches, and 16x4x8 output")

        # This first stage is layer-for-layer identical to SmallCNNFrameEncoder.
        self.first_stage = nn.Sequential(
            nn.Conv2d(
                cfg.input_channels,
                cfg.first_stage_channels,
                kernel_size=cfg.first_kernel,
                padding=cfg.first_padding,
            ),
            nn.BatchNorm2d(cfg.first_stage_channels),
            nn.ReLU(),
            nn.MaxPool2d(cfg.first_pool),
        )
        if encoder_mode == "single_scale":
            self.single_scale = nn.Conv2d(
                cfg.first_stage_channels,
                cfg.second_stage_channels,
                kernel_size=cfg.branch_kernel,
                dilation=cfg.local_dilation,
                padding=cfg.local_dilation,
            )
        else:
            self.local_branch = nn.Conv2d(
                cfg.first_stage_channels,
                cfg.branch_channels,
                kernel_size=cfg.branch_kernel,
                dilation=cfg.local_dilation,
                padding=cfg.local_dilation,
            )
            self.context_branch = nn.Conv2d(
                cfg.first_stage_channels,
                cfg.branch_channels,
                kernel_size=cfg.branch_kernel,
                dilation=cfg.context_dilation,
                padding=cfg.context_dilation,
            )
        self.post_fusion = nn.Sequential(
            nn.BatchNorm2d(cfg.second_stage_channels),
            nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d(cfg.output_pool),
        )
        self.flatten = nn.Flatten()

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        first = self.first_stage(x)
        if self.encoder_mode == "single_scale":
            second = self.single_scale(first)
        else:
            second = torch.cat(
                (self.local_branch(first), self.context_branch(first)), dim=1
            )
        return self.post_fusion(second)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.forward_spatial(x))


class ControlledScaleTemporal1DClassifier(nn.Module):
    """Shared skeleton; only the frame encoder's second spatial stage varies."""

    def __init__(
        self,
        encoder_mode: str,
        config: MultiScaleTemporal1DConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MultiScaleTemporal1DConfig()
        self.encoder_mode = encoder_mode
        self.model_name = (
            SINGLE_SCALE_MODEL_NAME if encoder_mode == "single_scale" else MODEL_NAME
        )
        self.encoder = ControlledScaleFrameEncoder(encoder_mode, self.config)
        self.encoder_feature_dim = self.encoder.feature_dim
        self.temporal_length = self.config.temporal_length

        # Direct module reuse from the formal implementation. No temporal or
        # classifier layer is reconstructed or altered in this candidate.
        formal_reference = CNN2DTemporal1D(
            n_classes=self.config.n_classes,
            dropout=self.config.dropout,
            norm=self.config.temporal_norm,
            temporal_length=self.config.temporal_length,
        )
        self.temporal_conv = formal_reference.temporal_conv
        self.classifier = formal_reference.classifier

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B,T,1,H,W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.temporal_length or int(x.shape[2]) != 1:
            raise ValueError(
                f"expected T={self.temporal_length} and one image channel, got {tuple(x.shape)}"
            )
        batch, time, channels, height, width = x.shape
        features = self.encoder(x.reshape(batch * time, channels, height, width))
        return features.reshape(batch, time, self.encoder_feature_dim)

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        sequence = self.encode_sequence(x)
        temporal_input = sequence.transpose(1, 2)
        temporal_features = self.temporal_conv(temporal_input)
        shapes = {
            "input": tuple(int(value) for value in x.shape),
            "frame_sequence": tuple(int(value) for value in sequence.shape),
            "temporal_input": tuple(int(value) for value in temporal_input.shape),
            "temporal_features": tuple(int(value) for value in temporal_features.shape),
        }
        return temporal_features, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.forward_features(x)
        return self.classifier(features)

    def forward_with_shapes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        features, shapes = self.forward_features(x)
        logits = self.classifier(features)
        shapes["logits"] = tuple(int(value) for value in logits.shape)
        return logits, shapes


def build_model(
    model_name: str,
    config: MultiScaleTemporal1DConfig | None = None,
) -> ControlledScaleTemporal1DClassifier:
    if model_name not in ENCODER_MODE_BY_MODEL_NAME:
        raise ValueError(f"unknown model: {model_name}")
    return ControlledScaleTemporal1DClassifier(
        ENCODER_MODE_BY_MODEL_NAME[model_name], config=config
    )


def parameter_breakdown(model: ControlledScaleTemporal1DClassifier) -> dict[str, int]:
    first_stage = count_trainable_parameters(model.encoder.first_stage)
    if model.encoder_mode == "single_scale":
        local_branch = count_trainable_parameters(model.encoder.single_scale)
        context_branch = 0
    else:
        local_branch = count_trainable_parameters(model.encoder.local_branch)
        context_branch = count_trainable_parameters(model.encoder.context_branch)
    parts = {
        "first_stage_parameters": first_stage,
        "local_or_single_branch_parameters": local_branch,
        "context_branch_parameters": context_branch,
        "post_fusion_parameters": count_trainable_parameters(model.encoder.post_fusion),
        "temporal_1d_parameters": count_trainable_parameters(model.temporal_conv),
        "classifier_parameters": count_trainable_parameters(model.classifier),
    }
    parts["frame_encoder_parameters"] = (
        parts["first_stage_parameters"]
        + parts["local_or_single_branch_parameters"]
        + parts["context_branch_parameters"]
        + parts["post_fusion_parameters"]
    )
    parts["total_parameter_count"] = (
        parts["frame_encoder_parameters"]
        + parts["temporal_1d_parameters"]
        + parts["classifier_parameters"]
    )
    if parts["total_parameter_count"] != count_trainable_parameters(model):
        raise AssertionError("parameter breakdown does not sum")
    return parts


def formal_temporal1d_audit() -> dict[str, Any]:
    reference = CNN2DTemporal1D(n_classes=2, dropout=0.25, norm="batchnorm")
    return {
        "frame_encoder_class": "SmallCNNFrameEncoder",
        "frame_encoder_layers": [
            "Conv2d(1,8,kernel_size=(5,9),padding=(2,4),stride=1)",
            "BatchNorm2d(8)",
            "ReLU",
            "MaxPool2d(kernel_size=(2,4))",
            "Conv2d(8,16,kernel_size=(5,7),padding=(2,3),stride=1)",
            "BatchNorm2d(16)",
            "ReLU",
            "AdaptiveAvgPool2d((4,8))",
            "Flatten",
        ],
        "spatial_output_shape_per_frame": "[16,4,8]",
        "frame_feature_dim": SmallCNNFrameEncoder.feature_dim,
        "temporal_conv_repr": repr(reference.temporal_conv),
        "classifier_repr": repr(reference.classifier),
        "training": {
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "batch_size": 16,
            "max_epochs": 40,
            "dropout": 0.25,
            "loss": "cross_entropy",
            "epoch_selection": "fixed_epochs_no_test_fold_selection",
        },
        "parameter_count": count_trainable_parameters(reference),
    }


def architecture_config(
    model_name: str,
    config: MultiScaleTemporal1DConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MultiScaleTemporal1DConfig()
    mode = ENCODER_MODE_BY_MODEL_NAME[model_name]
    second_stage = (
        "Conv2d(8,16,3,padding=1,dilation=1)"
        if mode == "single_scale"
        else "parallel Conv2d(8,8,3,padding=1,dilation=1) + Conv2d(8,8,3,padding=2,dilation=2); concat"
    )
    return {
        "model": model_name,
        "model_display": MODEL_DISPLAY_NAMES[model_name],
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "encoder_mode": mode,
        "config": asdict(cfg),
        "shared_first_stage": "exact formal SmallCNN layers 0:4",
        "controlled_second_stage": second_stage,
        "fusion": "channel concatenation only" if mode == "multiscale" else "not applicable",
        "post_second_stage": "BatchNorm2d(16), ReLU, AdaptiveAvgPool2d((4,8)), Flatten",
        "spatial_output_shape": "[B*T,16,4,8]",
        "frame_feature_dim": 512,
        "temporal_head": "direct CNN2DTemporal1D.temporal_conv module reuse; unchanged",
        "classifier": "direct CNN2DTemporal1D.classifier module reuse; unchanged",
        "attention_present": False,
        "mamba_present": False,
        "transformer_present": False,
        "branch_gate_present": False,
        "residual_present": False,
        "expected_parameter_count": EXPECTED_PARAMETER_COUNTS[model_name],
    }


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    training_config: DeepTrainingConfig,
    model_name: str,
    architecture: MultiScaleTemporal1DConfig | None = None,
    device: str | None = "auto",
    workers: int = 0,
) -> FoldTrainingResult:
    cfg = architecture or MultiScaleTemporal1DConfig()
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_test,
            session=session,
            task="binary",
            method=model_name,
            seed=seed,
            fold=fold,
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    model = build_model(model_name, cfg).to(torch_device)
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
    breakdown = parameter_breakdown(model)
    model_config = architecture_config(model_name, cfg)
    model_config["parameter_breakdown"] = breakdown
    return FoldTrainingResult(
        method=model_name,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=breakdown["total_parameter_count"],
        history=history,
        normalization_audit=norm_audit,
        final_training_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        final_trained_epochs=len(history),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_config,
    )
