from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ultrasound_decoding.deep import MPSCompatibleAdaptiveAvgPool2d, SmallCNN


LINEAR_METHODS = ("pca_lda_flat4", "cpca_lda_flat4")
SEQUENCE_DEEP_METHODS = ("cnn2d_meanpool", "cnn2d_lstm", "cnn2d_temporal1d")
LATE_FUSION_METHODS = ("single_frame_late_fusion",)
NEURAL_METHODS = (*SEQUENCE_DEEP_METHODS, *LATE_FUSION_METHODS)
ORDER_SENSITIVE_METHODS = ("cnn2d_lstm", "cnn2d_temporal1d")
MULTIFRAME_METHODS = (*LINEAR_METHODS, *SEQUENCE_DEEP_METHODS, *LATE_FUSION_METHODS)

MODEL_DISPLAY_NAMES = {
    "pca_lda_flat4": "PCA+LDA flat4",
    "cpca_lda_flat4": "cPCA+LDA flat4",
    "cnn2d_meanpool": "CNN mean-pool",
    "cnn2d_lstm": "CNN-LSTM",
    "cnn2d_temporal1d": "Temporal 1D-CNN",
    "single_frame_late_fusion": "Single-frame late fusion",
}

MODEL_DESCRIPTIONS = {
    "pca_lda_flat4": "clean4 frames flattened in fixed temporal order, then PCA+LDA",
    "cpca_lda_flat4": "clean4 frames flattened in fixed temporal order, then current class-contrastive PCA+LDA",
    "cnn2d_meanpool": "shared current SmallCNN encoder per frame, feature mean over the four frames, classifier",
    "cnn2d_lstm": "shared current SmallCNN encoder per frame, single-layer LSTM over the four frame features, classifier",
    "cnn2d_temporal1d": (
        "reference-to-senior 1D-CNN idea adapted to fUS clean4 frame features: "
        "shared 2D encoder, Conv1d along the four-frame temporal axis, temporal pooling, classifier"
    ),
    "single_frame_late_fusion": "current SmallCNN trained on train-cycle frames, four test-frame probabilities averaged per block",
}


@dataclass(frozen=True)
class ModelShapeAudit:
    method: str
    input_shape: tuple[int, ...]
    encoder_feature_dim: int
    temporal_length: int
    output_shape: tuple[int, ...]
    temporal_conv_axis: str | None = None


class SmallCNNFrameEncoder(nn.Module):
    """Current SmallCNN convolutional body without its dropout and linear head."""

    feature_dim = 16 * 4 * 8

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 9), padding=(2, 4)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(8, 16, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def encoder_architecture_signature() -> tuple[tuple[str, tuple[int, ...] | None], ...]:
    return (
        ("Conv2d", (1, 8, 5, 9)),
        ("BatchNorm2d", (8,)),
        ("ReLU", None),
        ("MaxPool2d", (2, 4)),
        ("Conv2d", (8, 16, 5, 7)),
        ("BatchNorm2d", (16,)),
        ("ReLU", None),
        ("AdaptiveAvgPool2d", (4, 8)),
        ("Flatten", None),
    )


class _SharedEncoderSequenceClassifier(nn.Module):
    temporal_length = 4

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.encoder = SmallCNNFrameEncoder()
        self.encoder_feature_dim = self.encoder.feature_dim
        self.n_classes = int(n_classes)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B, T, 1, H, W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.temporal_length:
            raise ValueError(f"expected temporal length 4, got {int(x.shape[1])}")
        if int(x.shape[2]) != 1:
            raise ValueError(f"expected singleton image channel, got {int(x.shape[2])}")
        b, t, c, h, w = x.shape
        z = self.encoder(x.reshape(b * t, c, h, w))
        return z.reshape(b, t, self.encoder_feature_dim)


class CNN2DMeanPool(_SharedEncoderSequenceClassifier):
    def __init__(self, n_classes: int, dropout: float = 0.25) -> None:
        super().__init__(n_classes=n_classes)
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(self.encoder_feature_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_sequence(x)
        return self.classifier(z.mean(dim=1))


class CNN2DLSTM(_SharedEncoderSequenceClassifier):
    def __init__(self, n_classes: int, hidden_size: int = 32, dropout: float = 0.25) -> None:
        super().__init__(n_classes=n_classes)
        self.hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.encoder_feature_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(self.hidden_size, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_sequence(x)
        out, _ = self.lstm(z)
        return self.classifier(out[:, -1, :])


class CNN2DTemporal1D(_SharedEncoderSequenceClassifier):
    """fUS clean4 temporal adaptation of the senior 1D-CNN idea."""

    def __init__(self, n_classes: int, dropout: float = 0.25, norm: str = "batchnorm") -> None:
        super().__init__(n_classes=n_classes)
        if norm == "batchnorm":
            norm_layer: nn.Module = nn.BatchNorm1d(64)
        elif norm == "groupnorm":
            norm_layer = nn.GroupNorm(8, 64)
        else:
            raise ValueError("norm must be 'batchnorm' or 'groupnorm'")
        self.temporal_axis = "time"
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.encoder_feature_dim, 64, kernel_size=3, padding=1),
            norm_layer,
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(64, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_sequence(x)
        z_time = z.transpose(1, 2)
        return self.classifier(self.temporal_conv(z_time))


def build_multiframe_model(method: str, n_classes: int) -> nn.Module:
    if method == "cnn2d_meanpool":
        return CNN2DMeanPool(n_classes=n_classes)
    if method == "cnn2d_lstm":
        return CNN2DLSTM(n_classes=n_classes)
    if method == "cnn2d_temporal1d":
        return CNN2DTemporal1D(n_classes=n_classes)
    if method == "single_frame_late_fusion":
        return SmallCNN(n_classes=n_classes)
    raise ValueError(f"Unknown neural multiframe method: {method}")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def model_shape_audit(method: str, n_classes: int = 2) -> ModelShapeAudit:
    model = build_multiframe_model(method, n_classes=n_classes)
    model.eval()
    if method == "single_frame_late_fusion":
        x = torch.zeros(2, 1, 128, 501)
        with torch.no_grad():
            y = model(x)
        return ModelShapeAudit(
            method=method,
            input_shape=tuple(x.shape),
            encoder_feature_dim=SmallCNNFrameEncoder.feature_dim,
            temporal_length=4,
            output_shape=tuple(y.shape),
        )
    x = torch.zeros(2, 4, 1, 128, 501)
    with torch.no_grad():
        y = model(x)
    temporal_axis = "time_T4" if method == "cnn2d_temporal1d" else None
    return ModelShapeAudit(
        method=method,
        input_shape=tuple(x.shape),
        encoder_feature_dim=SmallCNNFrameEncoder.feature_dim,
        temporal_length=4,
        output_shape=tuple(y.shape),
        temporal_conv_axis=temporal_axis,
    )
