from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


SBIND_ADAPTED_METHODS = ("sbind_noatt", "sbind")
SBIND_ADAPTED_DISPLAY_NAMES = {
    "sbind_noatt": "SBIND-adapted-NoAtt",
    "sbind": "SBIND-adapted",
}


@dataclass(frozen=True)
class SBINDAdaptedConfig:
    """Frozen v1 architecture for the supervised clean4 classification adaptation."""

    input_channels: int = 1
    encoder_channels: tuple[int, int, int] = (32, 32, 8)
    latent_channels: int = 8
    latent_height: int = 32
    latent_width: int = 32
    encoder_kernel_size: int = 5
    encoder_strides: tuple[int, int, int] = (2, 2, 1)
    recurrence_kernel_size: int = 3
    attention_patch_size: int = 8
    attention_heads: int = 8
    attention_embedding_dim: int = 256
    attention_reduced_channels: int = 2
    attention_initial_scale: float = 0.15
    attention_dropout: float = 0.0
    classifier_channels: int = 16
    classifier_conv_layers: int = 4
    classifier_hidden_units: int = 16
    dropout: float = 0.25
    negative_slope: float = 0.01


class _EncoderResidualBlock(nn.Module):
    """The residual second encoder layer present in the official tutorial config."""

    def __init__(self, channels: int, kernel_size: int, stride: int, negative_slope: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(channels)
        self.skip = nn.Conv2d(channels, channels, 1, stride=stride, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)) + self.skip(x))


class SBINDSpatialEncoder(nn.Module):
    """Official-style K mapping with one documented non-square-to-32x32 adaptation."""

    def __init__(self, config: SBINDAdaptedConfig) -> None:
        super().__init__()
        c1, c2, c3 = config.encoder_channels
        k = config.encoder_kernel_size
        p = k // 2
        s1, s2, s3 = config.encoder_strides
        self.first = nn.Sequential(
            nn.Conv2d(config.input_channels, c1, k, stride=s1, padding=p),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(config.negative_slope),
        )
        if c1 != c2:
            raise ValueError("frozen v1 residual encoder requires equal first and second channel counts")
        self.second = _EncoderResidualBlock(c1, k, s2, config.negative_slope)
        self.third = nn.Sequential(
            nn.Conv2d(c2, c3, k, stride=s3, padding=p),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(config.negative_slope),
        )
        self.spatial_adapter = nn.AdaptiveAvgPool2d((config.latent_height, config.latent_width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_adapter(self.third(self.second(self.first(x))))


class CompatibleScaledDotProductAttention(nn.Module):
    """Official MHA equations using APIs available in PyTorch 1.12.1.

    Official SBIND calls torch.nn.functional.scaled_dot_product_attention,
    which was introduced after the frozen server release. This module keeps the
    same qkv/projection parameterization and evaluates the standard equation with
    matmul, softmax, and dropout.
    """

    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        num_heads: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("attention embedding dimension must be divisible by heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(embedding_dim // num_heads)
        self.embedding_dim = int(embedding_dim)
        self.qkv = nn.Linear(token_dim, 3 * embedding_dim, bias=False)
        self.proj = nn.Linear(embedding_dim, output_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_tokens, _ = x.shape
        qkv = self.qkv(x).view(
            batch_size, n_tokens, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.softmax(scores, dim=-1)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        context = torch.matmul(weights, values)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, n_tokens, self.embedding_dim
        )
        return self.proj(context)


class SBINDImageSelfAttention(nn.Module):
    """Patch self-attention integrated into the recurrent transition."""

    def __init__(self, config: SBINDAdaptedConfig) -> None:
        super().__init__()
        patch = config.attention_patch_size
        reduced = config.attention_reduced_channels
        token_dim = reduced * patch * patch
        max_tokens = math.ceil(config.latent_height / patch) * math.ceil(config.latent_width / patch)
        self.patch_size = int(patch)
        self.reduce_channels = nn.Conv2d(config.latent_channels, reduced, kernel_size=1)
        self.restore_channels = nn.Conv2d(reduced, config.latent_channels, kernel_size=1)
        self.pos_embedding = nn.Parameter(torch.empty(1, max_tokens, 1))
        nn.init.uniform_(self.pos_embedding, -0.02, 0.02)
        self.norm1 = nn.LayerNorm(token_dim)
        self.attention = CompatibleScaledDotProductAttention(
            token_dim=token_dim,
            output_dim=token_dim,
            num_heads=config.attention_heads,
            embedding_dim=config.attention_embedding_dim,
            dropout=config.attention_dropout,
        )
        self.dropout = nn.Dropout(config.attention_dropout)
        self.norm2 = nn.LayerNorm(token_dim)
        self.scaling_factor = nn.Parameter(torch.tensor(config.attention_initial_scale))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = y.shape
        patch = self.patch_size
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        x = F.pad(y, (0, pad_w, 0, pad_h))
        x = self.reduce_channels(x)
        reduced_channels = int(x.shape[1])
        patches_h = int(x.shape[2] // patch)
        patches_w = int(x.shape[3] // patch)
        x = x.unfold(2, patch, patch).unfold(3, patch, patch)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(
            batch_size, patches_h * patches_w, reduced_channels * patch * patch
        )
        x = x + self.pos_embedding[:, : x.shape[1], :].expand_as(x)
        x = self.norm1(x)
        x = self.norm2(self.dropout(self.attention(x)))
        x = x.view(batch_size, patches_h, patches_w, reduced_channels, patch, patch)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            batch_size, reduced_channels, patches_h * patch, patches_w * patch
        )
        x = self.restore_channels(x[:, :, :height, :width])
        return y + self.scaling_factor * x


class SBINDConvRNNCell(nn.Module):
    """Behavior-relevant ConvRNN1 recurrence from Eq. A.1 of the paper."""

    def __init__(self, config: SBINDAdaptedConfig, use_attention: bool) -> None:
        super().__init__()
        k = config.recurrence_kernel_size
        self.local_transition = nn.Sequential(
            nn.Conv2d(config.latent_channels, config.latent_channels, k, padding=k // 2),
            nn.BatchNorm2d(config.latent_channels),
            nn.LeakyReLU(config.negative_slope),
        )
        self.global_attention: nn.Module
        self.global_attention = SBINDImageSelfAttention(config) if use_attention else nn.Identity()

    def forward(self, state: torch.Tensor, encoded_frame: torch.Tensor) -> torch.Tensor:
        transitioned = self.global_attention(self.local_transition(state))
        return transitioned + encoded_frame


class SBINDClassificationDecoder(nn.Module):
    """fUS-style behavior decoder D adapted to a two-logit block output."""

    def __init__(self, config: SBINDAdaptedConfig, n_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = config.latent_channels
        for _ in range(config.classifier_conv_layers):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        config.classifier_channels,
                        config.encoder_kernel_size,
                        stride=2,
                        padding=config.encoder_kernel_size // 2,
                    ),
                    nn.BatchNorm2d(config.classifier_channels),
                    nn.LeakyReLU(config.negative_slope),
                    nn.Dropout2d(config.dropout),
                ]
            )
            in_channels = config.classifier_channels
        self.convolutions = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.classifier_channels * 2 * 2, config.classifier_hidden_units),
            nn.LeakyReLU(config.negative_slope),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden_units, n_classes),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.convolutions(state))


class SBINDAdaptedClassifier(nn.Module):
    """SBIND-adapted classification baseline; this is not full SBIND."""

    def __init__(
        self,
        *,
        n_classes: int = 2,
        use_attention: bool = True,
        config: SBINDAdaptedConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SBINDAdaptedConfig()
        self.use_attention = bool(use_attention)
        self.encoder = SBINDSpatialEncoder(self.config)
        self.recurrence = SBINDConvRNNCell(self.config, use_attention=self.use_attention)
        self.decoder = SBINDClassificationDecoder(self.config, n_classes=n_classes)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5 or sequence.shape[2] != self.config.input_channels:
            raise ValueError(f"expected [B,T,1,H,W], got {tuple(sequence.shape)}")
        if sequence.shape[1] < 1:
            raise ValueError("sequence must contain at least one frame")
        batch_size, time_steps = int(sequence.shape[0]), int(sequence.shape[1])
        state = sequence.new_zeros(
            batch_size,
            self.config.latent_channels,
            self.config.latent_height,
            self.config.latent_width,
        )
        for time_i in range(time_steps):
            encoded = self.encoder(sequence[:, time_i])
            state = self.recurrence(state, encoded)
        return self.decoder(state)


def build_sbind_adapted_model(
    method: str,
    *,
    n_classes: int = 2,
    config: SBINDAdaptedConfig | None = None,
) -> SBINDAdaptedClassifier:
    if method not in SBIND_ADAPTED_METHODS:
        raise ValueError(f"unknown SBIND-adapted method: {method}")
    return SBINDAdaptedClassifier(
        n_classes=n_classes,
        use_attention=method == "sbind",
        config=config,
    )


def sbind_adapted_architecture_config(
    method: str,
    *,
    n_classes: int = 2,
    config: SBINDAdaptedConfig | None = None,
) -> dict[str, Any]:
    frozen = config or SBINDAdaptedConfig()
    return {
        "baseline_claim": "SBIND-adapted classification baseline; not full SBIND reproduction",
        "method": method,
        "display_name": SBIND_ADAPTED_DISPLAY_NAMES[method],
        "n_classes": int(n_classes),
        "attention_enabled": method == "sbind",
        "input_shape": [4, 1, 128, 501],
        "recurrent_update": "state_t_plus_1 = GlobalAttn(local_conv(state_t)) + K(frame_t)",
        "temporal_readout": "classification_from_final_updated_state",
        "spatial_size_adaptation": "AdaptiveAvgPool2d to 32x32 after official-style encoder",
        "pytorch_1_12_attention": "manual scaled dot-product equation matching official qkv/projection",
        "config": asdict(frozen),
    }


def train_sbind_adapted_fold(
    method: str,
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
    architecture_config: SBINDAdaptedConfig | None = None,
    device: str | None = "auto",
    workers: int = 0,
) -> FoldTrainingResult:
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_test,
            session=session,
            task="binary",
            method=method,
            seed=seed,
            fold=fold,
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    model = build_sbind_adapted_model(
        method, n_classes=len(classes), config=architecture_config
    ).to(torch_device)
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
    model_config = sbind_adapted_architecture_config(
        method, n_classes=len(classes), config=architecture_config
    )
    return FoldTrainingResult(
        method=method,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=count_trainable_parameters(model),
        history=history,
        normalization_audit=norm_audit,
        final_training_loss=float(history[-1]["train_loss"]),
        final_trained_epochs=int(len(history)),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_config,
    )
