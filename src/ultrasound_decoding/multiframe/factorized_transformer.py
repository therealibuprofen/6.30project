from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import count_trainable_parameters
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


MODEL_NAME = "cnn_factorized_transformer"
MODEL_DISPLAY_NAME = "CNN Factorized Transformer"
MODEL_IMPLEMENTATION_VERSION = "cnn_factorized_transformer_v1.0.0"


@dataclass(frozen=True)
class FactorizedTransformerConfig:
    """Frozen lightweight v1 baseline architecture (not a proposed method)."""

    temporal_length: int = 4
    input_channels: int = 1
    stem_channels: tuple[int, int, int] = (16, 32, 64)
    pooled_height: int = 8
    pooled_width: int = 32
    d_model: int = 64
    num_heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 1
    dim_feedforward: int = 128
    dropout: float = 0.25
    n_classes: int = 2


class SharedCNNStem(nn.Module):
    """Apply exactly the same CNN weights to every clean4 frame."""

    def __init__(self, config: FactorizedTransformerConfig) -> None:
        super().__init__()
        c1, c2, c3 = config.stem_channels
        self.layers = nn.Sequential(
            nn.Conv2d(config.input_channels, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((config.pooled_height, config.pooled_width)),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.layers(frames)


class CNNFactorizedTransformer(nn.Module):
    """Factorized spatial/temporal attention baseline for four clean-middle frames.

    With only four temporal samples, full attention over all 1024 space-time
    tokens is unnecessary. The shared spatial encoder first models long-range
    within-frame dependencies, then the temporal encoder models the four ordered
    frame representations. This is a baseline design, not a methodological claim.
    """

    def __init__(self, config: FactorizedTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or FactorizedTransformerConfig()
        cfg = self.config
        if cfg.stem_channels[-1] != cfg.d_model:
            raise ValueError("frozen v1 requires the stem output width to equal d_model")
        self.stem = SharedCNNStem(cfg)

        # Row + column parameters are a learnable two-dimensional positional
        # embedding. They encode spatial coordinates only and contain no labels.
        self.spatial_row_position = nn.Parameter(
            torch.empty(1, cfg.pooled_height, 1, cfg.d_model)
        )
        self.spatial_column_position = nn.Parameter(
            torch.empty(1, 1, cfg.pooled_width, cfg.d_model)
        )
        spatial_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # One encoder instance processes B*T frames, so all four frames share
        # precisely the same spatial Transformer weights.
        self.spatial_transformer = nn.TransformerEncoder(
            spatial_layer, num_layers=cfg.spatial_layers
        )

        self.temporal_position = nn.Parameter(
            torch.empty(1, cfg.temporal_length, cfg.d_model)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            temporal_layer, num_layers=cfg.temporal_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_classes),
        )
        nn.init.trunc_normal_(self.spatial_row_position, std=0.02)
        nn.init.trunc_normal_(self.spatial_column_position, std=0.02)
        nn.init.trunc_normal_(self.temporal_position, std=0.02)

    def _canonical_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(f"expected [B,T,H,W] or [B,T,1,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.config.temporal_length or x.shape[2] != 1:
            raise ValueError(
                f"expected temporal length {self.config.temporal_length} and one channel, "
                f"got {tuple(x.shape)}"
            )
        return x

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        x = self._canonical_input(x)
        batch_size, temporal_length, _, height, width = x.shape
        frames = x.reshape(batch_size * temporal_length, 1, height, width)
        cnn = self.stem(frames)
        cnn_bt = cnn.reshape(
            batch_size,
            temporal_length,
            self.config.d_model,
            self.config.pooled_height,
            self.config.pooled_width,
        )
        spatial_grid = cnn.permute(0, 2, 3, 1)
        spatial_grid = (
            spatial_grid
            + self.spatial_row_position
            + self.spatial_column_position
        )
        spatial_tokens = spatial_grid.reshape(
            batch_size * temporal_length, -1, self.config.d_model
        )
        spatial_encoded = self.spatial_transformer(spatial_tokens)
        frame_representations = spatial_encoded.mean(dim=1).reshape(
            batch_size, temporal_length, self.config.d_model
        )
        temporal_input = frame_representations + self.temporal_position[:, :temporal_length]
        temporal_encoded = self.temporal_transformer(temporal_input)
        pooled = temporal_encoded.mean(dim=1)
        shapes = {
            "input": tuple(int(v) for v in x.shape),
            "cnn_output": tuple(int(v) for v in cnn_bt.shape),
            "spatial_tokens": tuple(int(v) for v in spatial_tokens.shape),
            "spatial_transformer_output": tuple(int(v) for v in spatial_encoded.shape),
            "temporal_transformer_input": tuple(int(v) for v in temporal_input.shape),
            "temporal_transformer_output": tuple(int(v) for v in temporal_encoded.shape),
            "pooled": tuple(int(v) for v in pooled.shape),
        }
        return pooled, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled, _ = self.forward_features(x)
        return self.classifier(pooled)

    def forward_with_shapes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        pooled, shapes = self.forward_features(x)
        logits = self.classifier(pooled)
        shapes["logits"] = tuple(int(v) for v in logits.shape)
        return logits, shapes


def architecture_config(
    config: FactorizedTransformerConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FactorizedTransformerConfig()
    return {
        "model": MODEL_NAME,
        "model_display": MODEL_DISPLAY_NAME,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "factorization_rationale": (
            "shared spatial attention per frame followed by four-token temporal attention; "
            "baseline only, not a methodological innovation"
        ),
        "config": asdict(cfg),
        "spatial_tokens_per_frame": cfg.pooled_height * cfg.pooled_width,
        "pre_norm": True,
        "causal_temporal_mask": False,
        "positional_embeddings": "learnable 2D row+column spatial and learnable length-4 temporal",
    }


def train_factorized_transformer_fold(
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
    architecture: FactorizedTransformerConfig | None = None,
    device: str | None = "auto",
    workers: int = 0,
) -> FoldTrainingResult:
    cfg = architecture or FactorizedTransformerConfig()
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_test,
            session=session,
            task="binary",
            method=MODEL_NAME,
            seed=seed,
            fold=fold,
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    model = CNNFactorizedTransformer(cfg).to(torch_device)
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
    return FoldTrainingResult(
        method=MODEL_NAME,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=count_trainable_parameters(model),
        history=history,
        normalization_audit=norm_audit,
        final_training_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        final_trained_epochs=len(history),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(v) for v in X_train.shape[1:]),
        model_config=architecture_config(cfg),
    )
