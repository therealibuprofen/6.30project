from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .factorized_transformer import CNNFactorizedTransformer
from .models import CNN2DTemporal1D, count_trainable_parameters
from .spatial_mamba import (
    BidirectionalSharedMambaLayer,
    SpatialMambaConfig,
    transformer_reference_config,
)
from .training import (
    DeepTrainingConfig,
    FoldTrainingResult,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


LOCAL_ONLY_MODEL_NAME = "same_stem_local_temporal1d"
GLOBAL_ONLY_MODEL_NAME = "same_stem_global_mamba_temporal1d"
MODEL_NAME = "local_global_residual_mamba"
MODEL_DISPLAY_NAMES = {
    LOCAL_ONLY_MODEL_NAME: "Same-stem Local Temporal 1D-CNN",
    GLOBAL_ONLY_MODEL_NAME: "Same-stem Global Spatial Mamba + Temporal 1D-CNN",
    MODEL_NAME: "CNN + Gated Residual Spatial Mamba + Temporal 1D-CNN",
}
MODEL_DISPLAY_NAME = MODEL_DISPLAY_NAMES[MODEL_NAME]
MODEL_IMPLEMENTATION_VERSION = "local_global_residual_mamba_v1.1.0"
FUSION_MODES = ("local_only", "global_only", "gated_local_global")
MODEL_NAME_BY_FUSION_MODE = {
    "local_only": LOCAL_ONLY_MODEL_NAME,
    "global_only": GLOBAL_ONLY_MODEL_NAME,
    "gated_local_global": MODEL_NAME,
}
FUSION_MODE_BY_MODEL_NAME = {
    model_name: fusion_mode
    for fusion_mode, model_name in MODEL_NAME_BY_FUSION_MODE.items()
}
LOCAL_BASELINE_NAME = "cnn2d_temporal1d"
INITIAL_GATE_LOGIT = -2.0
EXPECTED_FORMAL_PARAMETER_COUNT = 116_579
EXPECTED_FORMAL_PARAMETER_COUNTS = {
    LOCAL_ONLY_MODEL_NAME: 48_482,
    GLOBAL_ONLY_MODEL_NAME: 116_578,
    MODEL_NAME: EXPECTED_FORMAL_PARAMETER_COUNT,
}


@dataclass(frozen=True)
class LocalGlobalResidualMambaConfig:
    """Frozen proposed-method v1 architecture; no variants or tuning knobs."""

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
    dropout: float = 0.25
    temporal_hidden_channels: int = 64
    temporal_kernel_size: int = 3
    temporal_norm: str = "batchnorm"
    initial_gate_logit: float = INITIAL_GATE_LOGIT
    n_classes: int = 2


def spatial_mamba_config(
    config: LocalGlobalResidualMambaConfig,
) -> SpatialMambaConfig:
    """Map v1 to the exact frozen Spatial-Mamba v1.1 spatial configuration."""
    return SpatialMambaConfig(
        temporal_length=config.temporal_length,
        input_channels=config.input_channels,
        stem_channels=config.stem_channels,
        pooled_height=config.pooled_height,
        pooled_width=config.pooled_width,
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        spatial_mamba_layers=config.spatial_mamba_layers,
        dropout=config.dropout,
        n_classes=config.n_classes,
    )


class LocalGlobalMambaTemporal1DClassifier(nn.Module):
    """One shared architecture definition for all three controlled fusion modes."""

    def __init__(
        self,
        fusion_mode: str,
        config: LocalGlobalResidualMambaConfig | None = None,
    ) -> None:
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}, got {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.model_name = MODEL_NAME_BY_FUSION_MODE[fusion_mode]
        self.config = config or LocalGlobalResidualMambaConfig()
        cfg = self.config
        if (
            cfg.temporal_length != 4
            or cfg.stem_channels != (16, 32, 64)
            or (cfg.pooled_height, cfg.pooled_width) != (8, 32)
            or cfg.d_model != 64
        ):
            raise ValueError("proposed v1 requires T=4, stem 16/32/64, 8x32, d_model=64")
        if cfg.temporal_hidden_channels != 64 or cfg.temporal_kernel_size != 3:
            raise ValueError("proposed v1 requires the frozen 64-channel kernel-3 temporal head")
        if cfg.temporal_norm != "batchnorm":
            raise ValueError("proposed v1 requires the formal Temporal 1D-CNN BatchNorm")

        mamba_cfg = spatial_mamba_config(cfg)
        transformer_reference = CNNFactorizedTransformer(
            transformer_reference_config(mamba_cfg)
        )
        self.stem = transformer_reference.stem
        if fusion_mode != "local_only":
            self.spatial_row_position = transformer_reference.spatial_row_position
            self.spatial_column_position = transformer_reference.spatial_column_position
            self.spatial_mamba = nn.ModuleList(
                [
                    BidirectionalSharedMambaLayer(mamba_cfg)
                    for _ in range(cfg.spatial_mamba_layers)
                ]
            )

        # The formal cnn2d_temporal1d receives 512-D SmallCNN frame features.
        # Proposed v1 produces 64-D spatially pooled fused maps, so only the
        # first Conv1d input width must be adapted. Every downstream temporal
        # layer and the classifier are directly transferred from the reviewed
        # formal implementation.
        temporal_reference = CNN2DTemporal1D(
            n_classes=cfg.n_classes,
            dropout=cfg.dropout,
            norm=cfg.temporal_norm,
            temporal_length=cfg.temporal_length,
        )
        reference_temporal_layers = list(temporal_reference.temporal_conv.children())
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(
                cfg.d_model,
                cfg.temporal_hidden_channels,
                kernel_size=cfg.temporal_kernel_size,
                padding=cfg.temporal_kernel_size // 2,
            ),
            *reference_temporal_layers[1:],
        )
        self.classifier = temporal_reference.classifier

        if fusion_mode == "gated_local_global":
            # Exactly one global scalar gate. sigmoid(-2) ~= 0.119 initially.
            self.gate_logit = nn.Parameter(
                torch.tensor(float(cfg.initial_gate_logit), dtype=torch.float32)
            )

    @property
    def effective_alpha(self) -> torch.Tensor:
        if self.fusion_mode == "local_only":
            return next(self.stem.parameters()).new_tensor(0.0)
        if self.fusion_mode == "global_only":
            return next(self.stem.parameters()).new_tensor(1.0)
        return torch.sigmoid(self.gate_logit)

    @property
    def alpha_is_trainable(self) -> bool:
        return self.fusion_mode == "gated_local_global"

    def _canonical_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(
                f"expected [B,T,H,W] or [B,T,1,H,W], got {tuple(x.shape)}"
            )
        if x.shape[1] != self.config.temporal_length or x.shape[2] != 1:
            raise ValueError(
                f"expected temporal length {self.config.temporal_length} and one channel, "
                f"got {tuple(x.shape)}"
            )
        return x

    def spatial_feature_maps(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return local, optional global, and selected maps as [B,T,64,8,32]."""
        x = self._canonical_input(x)
        batch_size, temporal_length, _, height, width = x.shape
        frames = x.reshape(batch_size * temporal_length, 1, height, width)
        local_flat = self.stem(frames)
        shape = (
            batch_size,
            temporal_length,
            self.config.d_model,
            self.config.pooled_height,
            self.config.pooled_width,
        )
        local_map = local_flat.reshape(shape)
        if self.fusion_mode == "local_only":
            return local_map, None, local_map

        positioned_grid = (
            local_flat.permute(0, 2, 3, 1)
            + self.spatial_row_position
            + self.spatial_column_position
        )
        global_tokens = positioned_grid.contiguous().reshape(
            batch_size * temporal_length,
            self.config.pooled_height * self.config.pooled_width,
            self.config.d_model,
        )
        for layer in self.spatial_mamba:
            global_tokens = layer(global_tokens)
        global_flat = global_tokens.reshape(
            batch_size * temporal_length,
            self.config.pooled_height,
            self.config.pooled_width,
            self.config.d_model,
        ).permute(0, 3, 1, 2).contiguous()
        global_map = global_flat.reshape(shape)
        if self.fusion_mode == "global_only":
            return local_map, global_map, global_map

        global_residual_flat = global_flat - local_flat
        fused_flat = local_flat + self.effective_alpha * global_residual_flat
        return local_map, global_map, fused_flat.reshape(shape)

    def forward_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        local, global_map, selected = self.spatial_feature_maps(x)
        frame_features = selected.mean(dim=(-2, -1))
        temporal_input = frame_features.transpose(1, 2)
        temporal_features = self.temporal_conv(temporal_input)
        shapes = {
            "input": tuple(int(value) for value in self._canonical_input(x).shape),
            "local_map": tuple(int(value) for value in local.shape),
            "selected_map": tuple(int(value) for value in selected.shape),
            "frame_features": tuple(int(value) for value in frame_features.shape),
            "temporal_input": tuple(int(value) for value in temporal_input.shape),
            "temporal_features": tuple(int(value) for value in temporal_features.shape),
        }
        if global_map is not None:
            shapes["global_map"] = tuple(int(value) for value in global_map.shape)
        if self.fusion_mode == "gated_local_global":
            shapes["fused_map"] = tuple(int(value) for value in selected.shape)
        return temporal_features, shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal_features, _ = self.forward_features(x)
        return self.classifier(temporal_features)

    def forward_with_shapes(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        temporal_features, shapes = self.forward_features(x)
        logits = self.classifier(temporal_features)
        shapes["logits"] = tuple(int(value) for value in logits.shape)
        return logits, shapes


class LocalGlobalResidualMambaClassifier(LocalGlobalMambaTemporal1DClassifier):
    """Backward-compatible name for the unchanged gated candidate."""

    def __init__(
        self, config: LocalGlobalResidualMambaConfig | None = None
    ) -> None:
        super().__init__("gated_local_global", config=config)

    @property
    def alpha(self) -> torch.Tensor:
        return self.effective_alpha


def parameter_breakdown(
    model: LocalGlobalMambaTemporal1DClassifier,
) -> dict[str, int]:
    parts = {
        "cnn_stem_parameters": count_trainable_parameters(model.stem),
        "spatial_position_parameters": (
            int(model.spatial_row_position.numel())
            + int(model.spatial_column_position.numel())
            if hasattr(model, "spatial_row_position")
            else 0
        ),
        "spatial_mamba_parameters": (
            count_trainable_parameters(model.spatial_mamba)
            if hasattr(model, "spatial_mamba")
            else 0
        ),
        "gate_parameters": (
            int(model.gate_logit.numel()) if hasattr(model, "gate_logit") else 0
        ),
        "temporal_1d_parameters": count_trainable_parameters(model.temporal_conv),
        "classifier_parameters": count_trainable_parameters(model.classifier),
    }
    parts["total_parameter_count"] = sum(parts.values())
    if parts["total_parameter_count"] != count_trainable_parameters(model):
        raise AssertionError("parameter breakdown does not sum to total trainable parameters")
    return parts


def architecture_config(
    config: LocalGlobalResidualMambaConfig | None = None,
    fusion_mode: str = "gated_local_global",
) -> dict[str, Any]:
    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
    cfg = config or LocalGlobalResidualMambaConfig()
    initial_alpha = float(torch.sigmoid(torch.tensor(cfg.initial_gate_logit)).item())
    model_name = MODEL_NAME_BY_FUSION_MODE[fusion_mode]
    uses_global = fusion_mode != "local_only"
    gate_trainable = fusion_mode == "gated_local_global"
    if fusion_mode == "local_only":
        fusion = "selected representation = F_local; fixed effective_alpha=0"
    elif fusion_mode == "global_only":
        fusion = "selected representation = F_global; fixed effective_alpha=1"
    else:
        fusion = "F_local + sigmoid(gate_logit) * (F_global - F_local)"
    return {
        "model": model_name,
        "model_display": MODEL_DISPLAY_NAMES[model_name],
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "fusion_mode": fusion_mode,
        "claim": "same-backbone mechanistic ablation" if not gate_trainable else "proposed-method first-round validation candidate",
        "config": asdict(cfg),
        "shared_backbone_signature": "SharedCNNStem -> spatial mean -> Temporal1D(64,64) -> classifier(64,2)",
        "local_backbone": (
            "exact reviewed SharedCNNStem: Conv-BN-GELU x3, "
            "AdaptiveAvgPool2d((8,32))"
        ),
        "local_feature_shape": "[B,4,64,8,32]",
        "uses_spatial_mamba_global_branch": uses_global,
        "spatial_position": (
            "exact Spatial Mamba v1.1 / Transformer 2D row [1,8,1,64] "
            "+ column [1,1,32,64]"
            if uses_global else "absent"
        ),
        "spatial_scan": (
            "exact two-layer bidirectional shared-weight Spatial Mamba v1.1"
            if uses_global else "absent"
        ),
        "spatial_flatten_order": "row-major (width index changes fastest)",
        "fusion": fusion,
        "gate_scope": (
            "one global trainable scalar; not session/channel/pixel/label conditioned"
            if gate_trainable else "no trainable gate"
        ),
        "alpha_is_trainable": gate_trainable,
        "fixed_effective_alpha": (
            0.0 if fusion_mode == "local_only" else 1.0 if fusion_mode == "global_only" else None
        ),
        "initial_gate_logit": float(cfg.initial_gate_logit) if gate_trainable else None,
        "initial_alpha": initial_alpha if gate_trainable else None,
        "frame_feature_reduction": "mean over selected 8x32 spatial grid -> [B,4,64]",
        "temporal_head": (
            "formal cnn2d_temporal1d topology: Conv1d-BN-ReLU-Conv1d-ReLU-"
            "AdaptiveAvgPool1d(1)-Flatten; only required input width adaptation 512->64"
        ),
        "temporal_first_conv": "Conv1d(64,64,kernel_size=3,padding=1)",
        "temporal_layers_after_first_conv_directly_reused": True,
        "classifier_directly_reused": True,
        "expected_formal_parameter_count_mamba_ssm_2_2_2": EXPECTED_FORMAL_PARAMETER_COUNTS[model_name],
        "temporal_transformer_present": False,
        "multiscale_present": False,
        "roi_glm_vascular_mask_present": False,
    }


def _train_epochs_with_control_history(
    model: LocalGlobalMambaTemporal1DClassifier,
    train_tensor: torch.Tensor,
    y_train_i: np.ndarray,
    *,
    config: DeepTrainingConfig,
    seed: int,
    device: torch.device,
    batch_size_reference: int,
    num_workers: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Frozen loop; log alpha by epoch only for the trainable gated model."""
    criterion = nn.CrossEntropyLoss()
    if config.optimizer.lower() != "adamw":
        raise ValueError("proposed v1 requires AdamW")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    batch_size = max(1, min(int(config.batch_size), int(batch_size_reference)))
    dataset = TensorDataset(train_tensor, torch.from_numpy(y_train_i))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=max(0, int(num_workers)),
    )
    initial_alpha = (
        float(model.effective_alpha.detach().cpu().item())
        if model.alpha_is_trainable else None
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            n_items = int(len(yb))
            total_loss += float(loss.detach().cpu().item()) * n_items
            total_correct += int(
                (logits.argmax(dim=1) == yb).sum().detach().cpu().item()
            )
            total_seen += n_items
        row = {
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(total_seen, 1)),
            "train_accuracy": float(total_correct / max(total_seen, 1)),
            "n_train_items": int(total_seen),
            "batch_size": int(batch_size),
        }
        if model.alpha_is_trainable:
            row["alpha"] = float(model.effective_alpha.detach().cpu().item())
        history.append(row)
    control_audit: dict[str, Any] = {
        "fusion_mode": model.fusion_mode,
        "alpha_is_trainable": model.alpha_is_trainable,
        "effective_alpha": float(model.effective_alpha.detach().cpu().item()),
    }
    if model.alpha_is_trainable:
        alphas = np.asarray([row["alpha"] for row in history], dtype=float)
        control_audit.update(
            {
                "initial_alpha": initial_alpha,
                "final_alpha": float(model.effective_alpha.detach().cpu().item()),
                "mean_alpha_last5_epochs": float(alphas[-5:].mean()),
            }
        )
    return history, control_audit


def _train_epochs_with_gate_history(
    model: LocalGlobalResidualMambaClassifier,
    train_tensor: torch.Tensor,
    y_train_i: np.ndarray,
    *,
    config: DeepTrainingConfig,
    seed: int,
    device: torch.device,
    batch_size_reference: int,
    num_workers: int = 0,
) -> tuple[list[dict[str, Any]], float]:
    """Compatibility wrapper for gated-v1 callers and tests."""
    history, audit = _train_epochs_with_control_history(
        model,
        train_tensor,
        y_train_i,
        config=config,
        seed=seed,
        device=device,
        batch_size_reference=batch_size_reference,
        num_workers=num_workers,
    )
    return history, float(audit["initial_alpha"])


def train_same_backbone_fusion_fold(
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
    architecture: LocalGlobalResidualMambaConfig | None = None,
    device: str | None = "auto",
    workers: int = 0,
) -> tuple[FoldTrainingResult, dict[str, Any]]:
    if model_name not in FUSION_MODE_BY_MODEL_NAME:
        raise ValueError(f"unknown same-backbone model: {model_name}")
    fusion_mode = FUSION_MODE_BY_MODEL_NAME[model_name]
    cfg = architecture or LocalGlobalResidualMambaConfig()
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
    model = LocalGlobalMambaTemporal1DClassifier(fusion_mode, cfg).to(torch_device)
    history, control_audit = _train_epochs_with_control_history(
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
    model_config = architecture_config(cfg, fusion_mode=fusion_mode)
    model_config["parameter_breakdown"] = breakdown
    result = FoldTrainingResult(
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
    return result, control_audit


def train_local_global_residual_mamba_fold(
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
    architecture: LocalGlobalResidualMambaConfig | None = None,
    device: str | None = "auto",
    workers: int = 0,
) -> tuple[FoldTrainingResult, dict[str, Any]]:
    """Compatibility wrapper preserving the gated candidate exactly."""
    return train_same_backbone_fusion_fold(
        X_train,
        y_train,
        X_test,
        classes,
        session=session,
        fold=fold,
        seed=seed,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
        training_config=training_config,
        model_name=MODEL_NAME,
        architecture=architecture,
        device=device,
        workers=workers,
    )
