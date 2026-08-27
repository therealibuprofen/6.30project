from __future__ import annotations

from dataclasses import dataclass
import os

# Required before CUDA-backed deterministic linear algebra is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from ultrasound_decoding.deep import FCNN, MPSCompatibleAdaptiveAvgPool2d, SmallCNN


LINEAR_METHODS = ("pca_lda_flat4", "cpca_lda_flat4")
SEQUENCE_DEEP_METHODS = (
    "cnn2d_meanpool",
    "cnn2d_lstm",
    "cnn2d_temporal1d",
    "fcnn_meanpool",
    "fcnn_lstm",
)
LATE_FUSION_METHODS = ("single_frame_late_fusion", "fcnn_late_fusion")
NEURAL_METHODS = (*SEQUENCE_DEEP_METHODS, *LATE_FUSION_METHODS)
ORDER_SENSITIVE_METHODS = ("cnn2d_lstm", "cnn2d_temporal1d", "fcnn_lstm")
MULTIFRAME_METHODS = (*LINEAR_METHODS, *SEQUENCE_DEEP_METHODS, *LATE_FUSION_METHODS)

METHOD_USES_TEMPORAL_ORDER = {
    "pca_lda_flat4": False,
    "cpca_lda_flat4": False,
    "cnn2d_meanpool": False,
    "cnn2d_lstm": True,
    "cnn2d_temporal1d": True,
    "single_frame_late_fusion": False,
    "fcnn_late_fusion": False,
    "fcnn_meanpool": False,
    "fcnn_lstm": True,
}

MODEL_DISPLAY_NAMES = {
    "pca_lda_flat4": "PCA+LDA flat4",
    "cpca_lda_flat4": "cPCA+LDA flat4",
    "cnn2d_meanpool": "CNN mean-pool",
    "cnn2d_lstm": "CNN-LSTM",
    "cnn2d_temporal1d": "Temporal 1D-CNN",
    "single_frame_late_fusion": "CNN late fusion",
    "fcnn_late_fusion": "FCNN late fusion",
    "fcnn_meanpool": "FCNN mean-pool",
    "fcnn_lstm": "FCNN-LSTM",
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
    "fcnn_late_fusion": (
        "current official single-frame FCNN trained on train-cycle frames, "
        "four test-frame probabilities averaged per block"
    ),
    "fcnn_meanpool": (
        "shared official FCNN frame encoder (MaxPool2d, Flatten, Linear to 3, ReLU), "
        "mean over the four 3D frame embeddings, classifier"
    ),
    "fcnn_lstm": (
        "shared official FCNN frame encoder (3D bottleneck), single-layer hidden_size=8 LSTM over "
        "the four frame embeddings, dropout, classifier"
    ),
}


@dataclass(frozen=True)
class ModelShapeAudit:
    method: str
    input_shape: tuple[int, ...]
    encoder_feature_dim: int
    temporal_length: int
    output_shape: tuple[int, ...]
    temporal_conv_axis: str | None = None
    encoded_shape: tuple[int, ...] | None = None
    frame_feature_dim: int | None = None
    lstm_input_size: int | None = None
    lstm_hidden_size: int | None = None


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_spatial_feature_map: bool = False,
    ) -> torch.Tensor:
        """Encode one frame, optionally retaining the audited 16 x 4 x 8 map.

        The default path deliberately remains the original ``self.layers(x)``
        call.  Apart from making old state dictionaries compatible, this keeps
        the supervised mean-pool forward numerically identical to the version
        that predates masked-reconstruction pretraining.
        """
        if return_spatial_feature_map:
            return self.layers[:-1](x)
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


class FCNNFrameEncoder(nn.Module):
    """Official single-frame FCNN up to the 3D bottleneck representation."""

    feature_dim = 3

    def __init__(self, input_shape: tuple[int, int] = (128, 501)) -> None:
        super().__init__()
        if len(input_shape) != 2:
            raise ValueError(f"fcnn frame encoder expects [H, W], got {input_shape}")
        pooled_h = int(input_shape[0]) // 2
        pooled_w = int(input_shape[1]) // 2
        if pooled_h < 1 or pooled_w < 1:
            raise ValueError(f"fcnn input is too small for 2x2 max pooling: {input_shape}")
        self.input_shape = (int(input_shape[0]), int(input_shape[1]))
        self.pooled_shape = (pooled_h, pooled_w)
        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(pooled_h * pooled_w, self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected single-frame tensor [B, 1, H, W], got {tuple(x.shape)}")
        if int(x.shape[1]) != 1:
            raise ValueError(f"expected singleton image channel, got {int(x.shape[1])}")
        return self.layers(x)


def fcnn_frame_encoder_architecture_signature(
    input_shape: tuple[int, int] = (128, 501),
) -> tuple[tuple[str, tuple[int, ...] | None], ...]:
    pooled_h = int(input_shape[0]) // 2
    pooled_w = int(input_shape[1]) // 2
    return (
        ("MaxPool2d", (2, 2)),
        ("Flatten", None),
        ("Linear", (pooled_h * pooled_w, 3)),
        ("ReLU", None),
    )


class _SharedEncoderSequenceClassifier(nn.Module):
    def __init__(self, n_classes: int, temporal_length: int = 4) -> None:
        super().__init__()
        if int(temporal_length) < 1:
            raise ValueError("temporal_length must be >= 1")
        self.temporal_length = int(temporal_length)
        self.encoder = SmallCNNFrameEncoder()
        self.encoder_feature_dim = self.encoder.feature_dim
        self.n_classes = int(n_classes)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B, T, 1, H, W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.temporal_length:
            raise ValueError(f"expected temporal length {self.temporal_length}, got {int(x.shape[1])}")
        if int(x.shape[2]) != 1:
            raise ValueError(f"expected singleton image channel, got {int(x.shape[2])}")
        b, t, c, h, w = x.shape
        z = self.encoder(x.reshape(b * t, c, h, w))
        return z.reshape(b, t, self.encoder_feature_dim)


class CNN2DMeanPool(_SharedEncoderSequenceClassifier):
    def __init__(self, n_classes: int, dropout: float = 0.25, temporal_length: int = 4) -> None:
        super().__init__(n_classes=n_classes, temporal_length=temporal_length)
        self.classifier = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(self.encoder_feature_dim, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_sequence(x)
        return self.classifier(z.mean(dim=1))


class CNN2DLSTM(_SharedEncoderSequenceClassifier):
    def __init__(self, n_classes: int, hidden_size: int = 32, dropout: float = 0.25, temporal_length: int = 4) -> None:
        super().__init__(n_classes=n_classes, temporal_length=temporal_length)
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

    def __init__(
        self,
        n_classes: int,
        dropout: float = 0.25,
        norm: str = "batchnorm",
        temporal_length: int = 4,
    ) -> None:
        super().__init__(n_classes=n_classes, temporal_length=temporal_length)
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

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return binary logits and the unchanged classifier-input embedding."""

        z = self.encode_sequence(x)
        z_time = z.transpose(1, 2)
        embedding = self.temporal_conv(z_time)
        return self.classifier(embedding), embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _embedding = self.forward_with_embedding(x)
        return logits


class _SharedFCNNSequenceClassifier(nn.Module):
    def __init__(
        self,
        n_classes: int,
        temporal_length: int = 4,
        input_shape: tuple[int, int] = (128, 501),
    ) -> None:
        super().__init__()
        if int(temporal_length) < 1:
            raise ValueError("temporal_length must be >= 1")
        self.temporal_length = int(temporal_length)
        self.encoder = FCNNFrameEncoder(input_shape=input_shape)
        self.encoder_feature_dim = self.encoder.feature_dim
        self.frame_feature_dim = self.encoder.feature_dim
        self.n_classes = int(n_classes)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected [B, T, 1, H, W], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.temporal_length:
            raise ValueError(f"expected temporal length {self.temporal_length}, got {int(x.shape[1])}")
        if int(x.shape[2]) != 1:
            raise ValueError(f"expected singleton image channel, got {int(x.shape[2])}")
        b, t, c, h, w = x.shape
        z = self.encoder(x.reshape(b * t, c, h, w))
        return z.reshape(b, t, self.encoder_feature_dim)


class FCNNMeanPool(_SharedFCNNSequenceClassifier):
    def __init__(
        self,
        n_classes: int,
        temporal_length: int = 4,
        input_shape: tuple[int, int] = (128, 501),
    ) -> None:
        super().__init__(n_classes=n_classes, temporal_length=temporal_length, input_shape=input_shape)
        self.classifier = nn.Linear(self.encoder_feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_sequence(x)
        return self.classifier(z.mean(dim=1))


class FCNNLSTM(_SharedFCNNSequenceClassifier):
    def __init__(
        self,
        n_classes: int,
        hidden_size: int = 8,
        dropout: float = 0.25,
        temporal_length: int = 4,
        input_shape: tuple[int, int] = (128, 501),
    ) -> None:
        super().__init__(n_classes=n_classes, temporal_length=temporal_length, input_shape=input_shape)
        self.hidden_size = int(hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.encoder_feature_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode_sequence(x)
        out, _ = self.lstm(z)
        return self.classifier(self.dropout(out[:, -1, :]))


def build_multiframe_model(method: str, n_classes: int, temporal_length: int = 4) -> nn.Module:
    if method == "cnn2d_meanpool":
        return CNN2DMeanPool(n_classes=n_classes, temporal_length=temporal_length)
    if method == "cnn2d_lstm":
        return CNN2DLSTM(n_classes=n_classes, temporal_length=temporal_length)
    if method == "cnn2d_temporal1d":
        return CNN2DTemporal1D(n_classes=n_classes, temporal_length=temporal_length)
    if method == "single_frame_late_fusion":
        return SmallCNN(n_classes=n_classes)
    if method == "fcnn_late_fusion":
        return FCNN(input_shape=(128, 501), n_classes=n_classes)
    if method == "fcnn_meanpool":
        return FCNNMeanPool(n_classes=n_classes, temporal_length=temporal_length)
    if method == "fcnn_lstm":
        return FCNNLSTM(n_classes=n_classes, temporal_length=temporal_length)
    raise ValueError(f"Unknown neural multiframe method: {method}")


def model_architecture_config(method: str, n_classes: int = 2, temporal_length: int = 4) -> dict[str, object]:
    if method == "single_frame_late_fusion":
        return {
            "method": method,
            "base_model": "SmallCNN",
            "late_fusion_probability_average": True,
            "temporal_length": int(temporal_length),
            "n_classes": int(n_classes),
        }
    if method == "fcnn_late_fusion":
        return {
            "method": method,
            "base_model": "official_single_frame_FCNN",
            "frame_encoder": fcnn_frame_encoder_architecture_signature(),
            "late_fusion_probability_average": True,
            "temporal_length": int(temporal_length),
            "n_classes": int(n_classes),
            "uses_fcnn_paper_32": False,
        }
    if method == "fcnn_meanpool":
        return {
            "method": method,
            "frame_encoder": fcnn_frame_encoder_architecture_signature(),
            "frame_feature_dim": 3,
            "aggregation": "mean_dim_1",
            "temporal_length": int(temporal_length),
            "n_classes": int(n_classes),
            "shared_frame_encoder_weights": True,
            "uses_fcnn_paper_32": False,
        }
    if method == "fcnn_lstm":
        return {
            "method": method,
            "frame_encoder": fcnn_frame_encoder_architecture_signature(),
            "frame_feature_dim": 3,
            "lstm_input_size": 3,
            "lstm_hidden_size": 8,
            "lstm_num_layers": 1,
            "bidirectional": False,
            "dropout_before_classifier": 0.25,
            "temporal_length": int(temporal_length),
            "n_classes": int(n_classes),
            "shared_frame_encoder_weights": True,
            "uses_fcnn_paper_32": False,
        }
    if method in {"cnn2d_meanpool", "cnn2d_lstm", "cnn2d_temporal1d"}:
        return {
            "method": method,
            "frame_encoder": encoder_architecture_signature(),
            "frame_feature_dim": SmallCNNFrameEncoder.feature_dim,
            "temporal_length": int(temporal_length),
            "n_classes": int(n_classes),
            "uses_temporal_order": METHOD_USES_TEMPORAL_ORDER[method],
        }
    if method in LINEAR_METHODS:
        return {
            "method": method,
            "input": "arcsinh_clean4_flat4",
            "temporal_length": int(temporal_length),
            "n_classes": int(n_classes),
        }
    raise ValueError(f"Unknown method: {method}")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def model_shape_audit(method: str, n_classes: int = 2) -> ModelShapeAudit:
    model = build_multiframe_model(method, n_classes=n_classes)
    model.eval()
    if method in {"single_frame_late_fusion", "fcnn_late_fusion"}:
        x = torch.zeros(2, 1, 128, 501)
        with torch.no_grad():
            y = model(x)
        feature_dim = 3 if method == "fcnn_late_fusion" else SmallCNNFrameEncoder.feature_dim
        return ModelShapeAudit(
            method=method,
            input_shape=tuple(x.shape),
            encoder_feature_dim=feature_dim,
            temporal_length=4,
            output_shape=tuple(y.shape),
            frame_feature_dim=feature_dim,
        )
    x = torch.zeros(2, 4, 1, 128, 501)
    with torch.no_grad():
        y = model(x)
    temporal_axis = "time_T4" if method == "cnn2d_temporal1d" else None
    encoded_shape = None
    lstm_input_size = None
    lstm_hidden_size = None
    if hasattr(model, "encode_sequence"):
        with torch.no_grad():
            encoded_shape = tuple(model.encode_sequence(x).shape)
    if hasattr(model, "lstm"):
        lstm_input_size = int(model.lstm.input_size)
        lstm_hidden_size = int(model.lstm.hidden_size)
    return ModelShapeAudit(
        method=method,
        input_shape=tuple(x.shape),
        encoder_feature_dim=int(getattr(model, "encoder_feature_dim", SmallCNNFrameEncoder.feature_dim)),
        temporal_length=4,
        output_shape=tuple(y.shape),
        temporal_conv_axis=temporal_axis,
        encoded_shape=encoded_shape,
        frame_feature_dim=int(getattr(model, "frame_feature_dim", getattr(model, "encoder_feature_dim", 0))),
        lstm_input_size=lstm_input_size,
        lstm_hidden_size=lstm_hidden_size,
    )
