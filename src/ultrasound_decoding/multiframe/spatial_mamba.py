from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .factorized_transformer import (
    CNNFactorizedTransformer,
    FactorizedTransformerConfig,
)
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


MODEL_NAME = "spatial_mamba"
MODEL_DISPLAY_NAME = "Spatial Mamba"
MODEL_IMPLEMENTATION_VERSION = "spatial_mamba_v1.1.0"
TRANSFORMER_REFERENCE_PARAMETER_COUNT = 127_010
MAMBA_DEPENDENCY_MESSAGE = (
    "Mamba dependency is not installed. Use the dedicated server Mamba environment."
)

try:
    from mamba_ssm import Mamba as _OfficialMamba
except Exception as _mamba_import_error:  # Import may fail on a non-CUDA workstation.
    _OfficialMamba = None
    _MAMBA_IMPORT_ERROR = _mamba_import_error
else:
    _MAMBA_IMPORT_ERROR = None


def mamba_dependency_available() -> bool:
    return _OfficialMamba is not None


def require_mamba_dependency() -> None:
    if _OfficialMamba is None:
        detail = f" Original import error: {_MAMBA_IMPORT_ERROR}" if _MAMBA_IMPORT_ERROR else ""
        raise RuntimeError(MAMBA_DEPENDENCY_MESSAGE + detail)


@dataclass(frozen=True)
class SpatialMambaConfig:
    """Frozen controlled v1 architecture; no hyperparameter variants."""

    temporal_length: int = 4
    input_channels: int = 1
    stem_channels: tuple[int, int, int] = (16, 32, 64)
    pooled_height: int = 8
    pooled_width: int = 32
    d_model: int = 64
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    spatial_mamba_layers: int = 2
    temporal_heads: int = 4
    temporal_layers: int = 1
    temporal_dim_feedforward: int = 128
    dropout: float = 0.25
    n_classes: int = 2


def transformer_reference_config(config: SpatialMambaConfig) -> FactorizedTransformerConfig:
    return FactorizedTransformerConfig(
        temporal_length=config.temporal_length,
        input_channels=config.input_channels,
        stem_channels=config.stem_channels,
        pooled_height=config.pooled_height,
        pooled_width=config.pooled_width,
        d_model=config.d_model,
        num_heads=config.temporal_heads,
        spatial_layers=2,
        temporal_layers=config.temporal_layers,
        dim_feedforward=config.temporal_dim_feedforward,
        dropout=config.dropout,
        n_classes=config.n_classes,
    )


class BidirectionalSharedMambaLayer(nn.Module):
    """Bidirectional shared-weight spatial scan with a pre-norm residual block.

    The same official Mamba module processes row-major tokens in forward order
    and in reversed order. The backward output is reversed back before the 0.5
    average, avoiding two direction-specific parameter sets.
    """

    def __init__(self, config: SpatialMambaConfig) -> None:
        super().__init__()
        require_mamba_dependency()
        assert _OfficialMamba is not None
        self.norm = nn.LayerNorm(config.d_model)
        self.mamba = _OfficialMamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )
        # Match the frozen Transformer's 0.25 regularization without tuning.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        forward_output = self.mamba(normalized)
        backward_input = torch.flip(normalized, dims=(1,))
        backward_output = torch.flip(self.mamba(backward_input), dims=(1,))
        bidirectional = 0.5 * (forward_output + backward_output)
        return tokens + self.dropout(bidirectional)


class SpatialMambaClassifier(nn.Module):
    """Controlled Spatial-Mamba baseline with unchanged temporal/classifier path."""

    def __init__(self, config: SpatialMambaConfig | None = None) -> None:
        super().__init__()
        require_mamba_dependency()
        self.config = config or SpatialMambaConfig()
        cfg = self.config
        if cfg.pooled_height * cfg.pooled_width != 256:
            raise ValueError("frozen v1 requires an 8x32 grid with 256 spatial tokens")

        # Construct the reviewed Transformer reference once and reuse these exact
        # modules. Its spatial Transformer is intentionally not retained.
        reference = CNNFactorizedTransformer(transformer_reference_config(cfg))
        self.stem = reference.stem
        # Reuse the reviewed Transformer's exact learnable 2D row+column
        # positional parameterization. No additional flat [1,256,64] position
        # parameter is retained in Spatial-Mamba v1.1.
        self.spatial_row_position = reference.spatial_row_position
        self.spatial_column_position = reference.spatial_column_position
        self.temporal_position = reference.temporal_position
        self.temporal_transformer = reference.temporal_transformer
        self.classifier = reference.classifier

        self.spatial_mamba = nn.ModuleList(
            [BidirectionalSharedMambaLayer(cfg) for _ in range(cfg.spatial_mamba_layers)]
        )

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
        # Flatten the positioned [8,32] grid in fixed row-major order: width
        # changes fastest, exactly as in the controlled Transformer baseline.
        spatial_tokens = spatial_grid.contiguous().reshape(
            batch_size * temporal_length, 256, self.config.d_model
        )
        spatial_encoded = spatial_tokens
        for layer in self.spatial_mamba:
            spatial_encoded = layer(spatial_encoded)
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
            "spatial_mamba_output": tuple(int(v) for v in spatial_encoded.shape),
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


def parameter_breakdown(model: SpatialMambaClassifier) -> dict[str, int]:
    parts = {
        "cnn_stem_parameters": count_trainable_parameters(model.stem),
        "spatial_mamba_parameters": (
            int(model.spatial_row_position.numel())
            + int(model.spatial_column_position.numel())
            + count_trainable_parameters(model.spatial_mamba)
        ),
        "temporal_transformer_parameters": (
            int(model.temporal_position.numel())
            + count_trainable_parameters(model.temporal_transformer)
        ),
        "classifier_parameters": count_trainable_parameters(model.classifier),
    }
    parts["total_parameter_count"] = sum(parts.values())
    if parts["total_parameter_count"] != count_trainable_parameters(model):
        raise AssertionError("parameter breakdown does not sum to total trainable parameters")
    return parts


def architecture_config(config: SpatialMambaConfig | None = None) -> dict[str, Any]:
    cfg = config or SpatialMambaConfig()
    return {
        "model": MODEL_NAME,
        "model_display": MODEL_DISPLAY_NAME,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "baseline_claim": "controlled Spatial-Mamba baseline/backbone candidate; not a proposed model",
        "dependency": "official mamba_ssm.Mamba",
        "config": asdict(cfg),
        "spatial_tokens_per_frame": 256,
        "spatial_flatten_order": "row-major (width index changes fastest)",
        "spatial_scan": "bidirectional shared-weight spatial scan",
        "bidirectional_merge": "0.5 * (forward + reverse(backward))",
        "spatial_position": (
            "exact Transformer v1 learnable 2D row+column form: "
            "row [1,8,1,64] + column [1,1,32,64]; label-independent and shared"
        ),
        "spatial_position_parameter_count": (
            cfg.pooled_height * cfg.d_model + cfg.pooled_width * cfg.d_model
        ),
        "spatial_position_parameter_delta_vs_v1_0": (
            (cfg.pooled_height + cfg.pooled_width) * cfg.d_model
            - cfg.pooled_height * cfg.pooled_width * cfg.d_model
        ),
        "flat_spatial_position_parameter_present": False,
        "temporal_path": "exact modules reused from cnn_factorized_transformer v1",
        "causal_temporal_mask": False,
        "transformer_reference_parameter_count": TRANSFORMER_REFERENCE_PARAMETER_COUNT,
    }


def train_spatial_mamba_fold(
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
    architecture: SpatialMambaConfig | None = None,
    device: str | None = "auto",
    workers: int = 0,
) -> FoldTrainingResult:
    cfg = architecture or SpatialMambaConfig()
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train, X_test, session=session, task="binary", method=MODEL_NAME,
            seed=seed, fold=fold, train_cycles=train_cycles, test_cycles=test_cycles,
        )
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    model = SpatialMambaClassifier(cfg).to(torch_device)
    history = _train_epochs(
        model, train_tensor, y_train_i, config=training_config, seed=seed,
        device=torch_device, batch_size_reference=len(X_train), num_workers=workers,
    )
    probabilities = predict_probabilities(
        model, test_tensor, device=torch_device,
        batch_size=training_config.batch_size, num_workers=workers,
    )
    predictions = classes[probabilities.argmax(axis=1)]
    breakdown = parameter_breakdown(model)
    model_config = architecture_config(cfg)
    model_config["parameter_breakdown"] = breakdown
    return FoldTrainingResult(
        method=MODEL_NAME, seed=int(seed), predictions=predictions,
        probabilities=probabilities, model=model,
        model_parameters=breakdown["total_parameter_count"], history=history,
        normalization_audit=norm_audit,
        final_training_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        final_trained_epochs=len(history), device=str(torch_device),
        X_test_normalized=X_test_norm, normalization_mean=norm_mean,
        normalization_std=norm_std, normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(v) for v in X_train.shape[1:]), model_config=model_config,
    )
