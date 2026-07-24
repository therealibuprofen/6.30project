from __future__ import annotations

from dataclasses import dataclass, replace
import copy
import random

import numpy as np


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class TorchModelConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 40
    patience: int | None = None
    activation: str = "relu"
    normalization: str = "batchnorm"
    dropout: float = 0.25
    loss: str = "cross_entropy"


@dataclass
class TorchFitResult:
    predictions: np.ndarray
    metadata: dict[str, object]


MODEL_DEFAULTS: dict[str, TorchModelConfig] = {
    "cnn": TorchModelConfig(),
    "cnn_lstm": TorchModelConfig(),
    "fcnn": TorchModelConfig(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=16,
        max_epochs=40,
        patience=None,
        activation="relu",
        normalization="none",
        dropout=0.0,
    ),
    "fcnn_paper_32": TorchModelConfig(
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        batch_size=32,
        max_epochs=80,
        patience=12,
        activation="elu",
        normalization="none",
        dropout=0.0,
    ),
    "fus_lite_cnn": TorchModelConfig(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        max_epochs=80,
        patience=12,
        activation="elu",
        normalization="groupnorm",
        dropout=0.3,
    ),
}

TORCH_MODEL_NAMES = tuple(MODEL_DEFAULTS)


def get_torch_model_defaults(method: str) -> TorchModelConfig:
    if method not in MODEL_DEFAULTS:
        raise ValueError(f"Unknown torch method: {method}")
    return MODEL_DEFAULTS[method]


def resolve_torch_config(
    method: str,
    max_epochs: int | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    patience: int | None = None,
    activation: str | None = None,
    normalization: str | None = None,
    dropout: float | None = None,
    optimizer: str | None = None,
) -> TorchModelConfig:
    cfg = get_torch_model_defaults(method)
    resolved_epochs = max_epochs if max_epochs is not None else epochs
    overrides = {
        "max_epochs": resolved_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "activation": activation.lower() if activation is not None else None,
        "normalization": normalization.lower() if normalization is not None else None,
        "dropout": dropout,
        "optimizer": optimizer.lower() if optimizer is not None else None,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(cfg, **clean_overrides)


def _set_reproducible_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _resolve_device(device: str | None):
    import torch

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _activation(name: str):
    from torch import nn

    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


def _normalization(name: str, channels: int):
    from torch import nn

    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "batchnorm":
        return nn.BatchNorm2d(channels)
    if name == "groupnorm":
        groups_by_channels = {16: 4, 32: 8, 64: 8}
        groups = groups_by_channels.get(channels, min(8, channels))
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization: {name}")


def _conv_block(
    in_channels: int,
    out_channels: int,
    *,
    activation: str,
    normalization: str,
    bias: bool,
    pool: bool,
):
    from torch import nn

    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
        _normalization(normalization, out_channels),
        _activation(activation),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class MPSCompatibleAdaptiveAvgPool2d:
    def __new__(cls, output_size: int | tuple[int, int]):
        from torch import nn
        from torch.nn import functional as F

        class _MPSCompatibleAdaptiveAvgPool2d(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size = output_size
                self.pool = nn.AdaptiveAvgPool2d(output_size)

            def forward(self, x):
                if x.device.type != "mps":
                    return self.pool(x)

                out_h, out_w = (
                    (self.output_size, self.output_size)
                    if isinstance(self.output_size, int)
                    else self.output_size
                )
                in_h, in_w = x.shape[-2:]
                if in_h % out_h == 0 and in_w % out_w == 0:
                    return self.pool(x)

                return F.adaptive_avg_pool2d(x.cpu(), self.output_size).to(x.device)

        return _MPSCompatibleAdaptiveAvgPool2d()


class FCNNPaper32:
    def __new__(
        cls,
        n_classes: int,
        activation: str = "elu",
        normalization: str = "none",
        dropout: float = 0.0,
    ):
        from torch import nn

        class _FCNNPaper32(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = _conv_block(
                    1,
                    32,
                    activation=activation,
                    normalization=normalization,
                    bias=True,
                    pool=True,
                )
                self.conv2 = _conv_block(
                    32,
                    32,
                    activation=activation,
                    normalization=normalization,
                    bias=True,
                    pool=True,
                )
                self.conv3 = _conv_block(
                    32,
                    32,
                    activation=activation,
                    normalization=normalization,
                    bias=True,
                    pool=True,
                )
                self.conv4 = _conv_block(
                    32,
                    32,
                    activation=activation,
                    normalization=normalization,
                    bias=True,
                    pool=False,
                )
                self.last_conv = self.conv4
                head: list[nn.Module] = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]
                if dropout > 0:
                    head.append(nn.Dropout(float(dropout)))
                head.append(nn.Linear(32, n_classes))
                self.classifier = nn.Sequential(*head)

            def forward(self, x, return_features: bool = False):
                z = self.conv1(x)
                z = self.conv2(z)
                z = self.conv3(z)
                features = self.conv4(z)
                logits = self.classifier(features)
                if return_features:
                    return logits, features
                return logits

        return _FCNNPaper32()


class FUSLiteCNN:
    def __new__(
        cls,
        n_classes: int,
        activation: str = "elu",
        normalization: str = "groupnorm",
        dropout: float = 0.3,
    ):
        from torch import nn

        class _FUSLiteCNN(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = _conv_block(
                    1,
                    16,
                    activation=activation,
                    normalization=normalization,
                    bias=False,
                    pool=True,
                )
                self.conv2 = _conv_block(
                    16,
                    32,
                    activation=activation,
                    normalization=normalization,
                    bias=False,
                    pool=True,
                )
                self.conv3 = _conv_block(
                    32,
                    64,
                    activation=activation,
                    normalization=normalization,
                    bias=False,
                    pool=True,
                )
                self.conv4 = _conv_block(
                    64,
                    64,
                    activation=activation,
                    normalization=normalization,
                    bias=False,
                    pool=False,
                )
                self.last_conv = self.conv4
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(64, n_classes),
                )

            def forward(self, x, return_features: bool = False):
                z = self.conv1(x)
                z = self.conv2(z)
                z = self.conv3(z)
                features = self.conv4(z)
                logits = self.classifier(features)
                if return_features:
                    return logits, features
                return logits

        return _FUSLiteCNN()


class FCNN:
    def __new__(cls, input_shape: tuple[int, ...], n_classes: int):
        from torch import nn

        if len(input_shape) != 2:
            raise ValueError(f"fcnn expects single frames [H, W], got {input_shape}")
        pooled_h = input_shape[0] // 2
        pooled_w = input_shape[1] // 2
        if pooled_h < 1 or pooled_w < 1:
            raise ValueError(f"fcnn input is too small for 2x2 max pooling: {input_shape}")

        return nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(pooled_h * pooled_w, 3),
            nn.ReLU(),
            nn.Linear(3, n_classes),
        )


class SmallCNN:
    def __new__(cls, n_classes: int):
        from torch import nn

        return nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(5, 9), padding=(2, 4)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d((2, 4)),
            nn.Conv2d(8, 16, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            MPSCompatibleAdaptiveAvgPool2d((4, 8)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(16 * 4 * 8, n_classes),
        )


class SmallCNNLSTM:
    def __new__(cls, n_channels: int, n_classes: int):
        from torch import nn

        class _CNNLSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.temporal = nn.Sequential(
                    nn.Conv1d(n_channels, 64, kernel_size=9, padding=4),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                    nn.Conv1d(64, 64, kernel_size=7, padding=3),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                )
                self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
                self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(32, n_classes))

            def forward(self, x):
                z = self.temporal(x)
                z = z.transpose(1, 2)
                _, (hidden, _) = self.lstm(z)
                return self.classifier(hidden[-1])

        return _CNNLSTM()


def build_torch_model(method: str, n_classes: int, input_shape: tuple[int, ...], config: TorchModelConfig):
    if method == "cnn":
        return SmallCNN(n_classes=n_classes)
    if method == "cnn_lstm":
        if len(input_shape) != 2:
            raise ValueError(f"cnn_lstm expects single frames [H, W], got {input_shape}")
        return SmallCNNLSTM(n_channels=input_shape[0], n_classes=n_classes)
    if method == "fcnn":
        return FCNN(input_shape=input_shape, n_classes=n_classes)
    if method == "fcnn_paper_32":
        return FCNNPaper32(
            n_classes=n_classes,
            activation=config.activation,
            normalization=config.normalization,
            dropout=config.dropout,
        )
    if method == "fus_lite_cnn":
        return FUSLiteCNN(
            n_classes=n_classes,
            activation=config.activation,
            normalization=config.normalization,
            dropout=config.dropout,
        )
    raise ValueError(f"Unknown torch method: {method}")


def _balanced_accuracy_int(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for cls_i in range(n_classes):
        mask = y_true == cls_i
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(y_pred[mask] == cls_i)))
    return float(np.mean(recalls)) if recalls else 0.0


def _grouped_validation_indices(
    y: np.ndarray,
    groups: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None, list[object]]:
    if groups is None:
        return None, None, []
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        return None, None, []

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    max_val_groups = max(1, int(np.ceil(0.2 * len(unique_groups))))
    for n_val_groups in range(max_val_groups, len(unique_groups)):
        for start in range(len(shuffled)):
            val_groups = np.roll(shuffled, -start)[:n_val_groups]
            val_mask = np.isin(groups, val_groups)
            train_mask = ~val_mask
            if len(np.unique(y[train_mask])) >= 2 and len(np.unique(y[val_mask])) >= 2:
                return (
                    np.flatnonzero(train_mask),
                    np.flatnonzero(val_mask),
                    [value.item() if hasattr(value, "item") else value for value in np.sort(val_groups)],
                )
    return None, None, []


def _make_optimizer(model, config: TorchModelConfig):
    import torch

    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def _data_quality_summary(X: np.ndarray) -> dict[str, object]:
    reshaped = X.reshape(len(X), -1) if len(X) else X.reshape(0, int(np.prod(X.shape[1:])))
    finite_mask = np.isfinite(reshaped)
    return {
        "n_samples": int(len(X)),
        "nan_count": int(np.isnan(reshaped).sum()),
        "inf_count": int(np.isinf(reshaped).sum()),
        "nonfinite_count": int((~finite_mask).sum()),
        "all_zero_images": int(np.all(reshaped == 0, axis=1).sum()) if len(X) else 0,
    }


def _normalize_frames(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    statistics_scope: str,
    normalization_weighting: str = "sample_weighted",
    train_session_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_quality_raw = _data_quality_summary(X_train)
    test_quality_raw = _data_quality_summary(X_test)
    if train_quality_raw["nonfinite_count"] or test_quality_raw["nonfinite_count"]:
        raise ValueError("CNN input contains NaN or Inf values")

    X_train = np.arcsinh(X_train.astype(np.float32, copy=False))
    X_test = np.arcsinh(X_test.astype(np.float32, copy=False))
    epsilon = 1e-6
    negative_variance_pixels = 0
    source_weights: dict[str, float] = {}
    if normalization_weighting == "sample_weighted":
        mean = X_train.mean(axis=0, keepdims=True)
        std_raw = X_train.std(axis=0, keepdims=True)
    elif normalization_weighting == "session_equal":
        if train_session_labels is None:
            raise ValueError("session_equal normalization requires train_session_labels")
        labels = np.asarray(train_session_labels).astype(str)
        if len(labels) != len(X_train):
            raise ValueError("train_session_labels length does not match X_train")
        unique_sessions = sorted(np.unique(labels).tolist())
        if len(unique_sessions) < 2:
            raise ValueError("session_equal normalization requires at least two source sessions")
        means = []
        second_moments = []
        for session in unique_sessions:
            X_session = X_train[labels == session]
            if len(X_session) == 0:
                raise ValueError(f"session {session} has no samples for normalization")
            means.append(X_session.mean(axis=0, keepdims=True))
            second_moments.append(np.square(X_session).mean(axis=0, keepdims=True))
            source_weights[session] = 1.0 / len(unique_sessions)
        mean = np.mean(np.concatenate(means, axis=0), axis=0, keepdims=True)
        second_moment = np.mean(np.concatenate(second_moments, axis=0), axis=0, keepdims=True)
        variance = second_moment - np.square(mean)
        negative_variance_pixels = int((variance < 0).sum())
        variance = np.maximum(variance, 0.0)
        std_raw = np.sqrt(variance)
    else:
        raise ValueError(f"Unknown normalization_weighting: {normalization_weighting}")
    if not np.isfinite(mean).all() or not np.isfinite(std_raw).all():
        raise ValueError("Non-finite normalization statistics")
    std = std_raw + 1e-6
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    if not np.isfinite(X_train_norm).all() or not np.isfinite(X_test_norm).all():
        raise ValueError("Non-finite normalized CNN inputs")
    stats = {
        "transform": "arcsinh_then_train_pixel_zscore",
        "statistics_scope": statistics_scope,
        "normalization_weighting": normalization_weighting,
        "target_used_for_stats": False,
        "epsilon": 1e-6,
        "train_input_quality": train_quality_raw,
        "test_input_quality": test_quality_raw,
        "source_session_weights": source_weights,
        "negative_variance_pixels_before_clamp": negative_variance_pixels,
        "mean_mean": float(mean.mean()),
        "mean_std": float(mean.std()),
        "mean_min": float(mean.min()),
        "mean_max": float(mean.max()),
        "std_mean": float(std_raw.mean()),
        "std_std": float(std_raw.std()),
        "std_min": float(std_raw.min()),
        "std_max": float(std_raw.max()),
    }
    return X_train_norm, X_test_norm, stats


def _torch_frame_tensors(method: str, X_train: np.ndarray, X_test: np.ndarray):
    import torch

    if method in {"cnn", "fcnn", "fcnn_paper_32", "fus_lite_cnn"}:
        if X_train.ndim != 3:
            raise ValueError(f"{method} expects single frames [N, H, W], got {X_train.shape}")
        return torch.from_numpy(X_train[:, None, :, :]), torch.from_numpy(X_test[:, None, :, :])
    if method == "cnn_lstm":
        return torch.from_numpy(X_train), torch.from_numpy(X_test)
    raise ValueError(f"Unknown torch method: {method}")


def _predict_int(model, tensor, device, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(tensor), batch_size=max(1, batch_size), shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.asarray([], dtype=np.int64)


def _train_model_epochs(
    model,
    train_tensor,
    y_train_i: np.ndarray,
    train_indices: np.ndarray,
    *,
    device,
    config: TorchModelConfig,
    criterion,
    seed: int,
    n_epochs: int,
    phase: str,
    val_indices: np.ndarray | None = None,
    train_session_labels: np.ndarray | None = None,
    train_groups: np.ndarray | None = None,
    source_balance_mode: str = "pooled_all",
) -> tuple[
    list[dict[str, object]],
    int,
    int | None,
    float | None,
    dict[str, object] | None,
    list[dict[str, object]],
]:
    import torch

    if n_epochs < 1:
        raise ValueError("n_epochs must be >= 1")

    train_indices = np.asarray(train_indices, dtype=np.int64)
    if len(train_indices) == 0:
        raise ValueError("train_indices is empty")
    session_labels = None if train_session_labels is None else np.asarray(train_session_labels).astype(str)
    group_labels = None if train_groups is None else np.asarray(train_groups).astype(str)
    if session_labels is not None and len(session_labels) != len(y_train_i):
        raise ValueError("train_session_labels length does not match y_train")
    if group_labels is not None and len(group_labels) != len(y_train_i):
        raise ValueError("train_groups length does not match y_train")

    optimizer = _make_optimizer(model, config)
    history: list[dict[str, object]] = []
    sampling_audit: list[dict[str, object]] = []
    best_epoch: int | None = None
    best_val_balanced_accuracy: float | None = None
    best_state = None
    epochs_without_improvement = 0
    trained_epochs = 0
    batch_size = max(1, min(int(config.batch_size), len(train_indices)))

    def make_epoch_batches(epoch: int) -> list[np.ndarray]:
        rng = np.random.default_rng(seed + epoch * 1009)
        if source_balance_mode == "pooled_all":
            shuffled = rng.permutation(train_indices)
            return [shuffled[start : start + batch_size] for start in range(0, len(shuffled), batch_size)]
        if source_balance_mode != "session_balanced":
            raise ValueError(f"Unknown source_balance_mode: {source_balance_mode}")
        if session_labels is None:
            raise ValueError("session_balanced training requires train_session_labels")
        unique_sessions = sorted(np.unique(session_labels[train_indices]).tolist())
        if len(unique_sessions) < 2:
            raise ValueError("session_balanced training requires at least two source sessions")
        total_draws = len(train_indices)
        n_batches = int(np.ceil(total_draws / batch_size))
        by_session = {
            session: rng.permutation(train_indices[session_labels[train_indices] == session])
            for session in unique_sessions
        }
        pointers = {session: 0 for session in unique_sessions}
        batches: list[np.ndarray] = []
        draws_remaining = total_draws
        for _ in range(n_batches):
            current_batch_size = min(batch_size, draws_remaining)
            draws_remaining -= current_batch_size
            base_quota = current_batch_size // len(unique_sessions)
            remainder = current_batch_size % len(unique_sessions)
            batch_parts = []
            for session_i, session in enumerate(unique_sessions):
                quota = base_quota + (1 if session_i < remainder else 0)
                if quota == 0:
                    continue
                pool = by_session[session]
                selected = []
                while len(selected) < quota:
                    remaining = len(pool) - pointers[session]
                    take = min(quota - len(selected), remaining)
                    if take > 0:
                        selected.extend(pool[pointers[session] : pointers[session] + take].tolist())
                        pointers[session] += take
                    if len(selected) < quota:
                        by_session[session] = rng.permutation(train_indices[session_labels[train_indices] == session])
                        pool = by_session[session]
                        pointers[session] = 0
                batch_parts.extend(selected)
            batch = rng.permutation(np.asarray(batch_parts, dtype=np.int64))
            if len(unique_sessions) >= 2 and len(np.unique(session_labels[batch])) < 2:
                raise AssertionError("session_balanced batch did not include all source sessions")
            batches.append(batch)
        return batches

    for epoch in range(1, n_epochs + 1):
        trained_epochs = epoch
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_batches = make_epoch_batches(epoch)
        epoch_drawn_indices: list[int] = []
        for batch_indices in epoch_batches:
            xb = train_tensor[batch_indices].to(device)
            yb = torch.from_numpy(y_train_i[batch_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            batch_n = int(len(yb))
            total_loss += float(loss.detach().cpu().item()) * batch_n
            total_samples += batch_n
            epoch_drawn_indices.extend(int(i) for i in batch_indices)

        if source_balance_mode == "session_balanced" and session_labels is not None:
            unique_sessions = sorted(np.unique(session_labels[train_indices]).tolist())
        elif session_labels is not None:
            unique_sessions = sorted(np.unique(session_labels[train_indices]).tolist())
        else:
            unique_sessions = ["all_sources"]
        drawn = np.asarray(epoch_drawn_indices, dtype=np.int64)
        for session in unique_sessions:
            if session_labels is None:
                session_draws = drawn
                available = len(train_indices)
            else:
                session_draws = drawn[session_labels[drawn] == session]
                available = int(np.sum(session_labels[train_indices] == session))
            if group_labels is not None and len(session_draws):
                unique_cycles = int(len(np.unique(group_labels[session_draws])))
            else:
                unique_cycles = 0
            sampling_audit.append(
                {
                    "phase": phase,
                    "epoch": int(epoch),
                    "source_session": str(session),
                    "n_draws": int(len(session_draws)),
                    "draw_proportion": float(len(session_draws) / max(len(drawn), 1)),
                    "unique_samples_drawn": int(len(np.unique(session_draws))),
                    "unique_cycles_drawn": unique_cycles,
                    "with_replacement": bool(len(session_draws) > available),
                    "source_balance_mode": source_balance_mode,
                }
            )

        row: dict[str, object] = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "n_train": int(len(train_indices)),
        }
        if val_indices is not None:
            val_pred = _predict_int(model, train_tensor[val_indices], device, config.batch_size)
            val_score = _balanced_accuracy_int(y_train_i[val_indices], val_pred, len(np.unique(y_train_i)))
            row.update(
                {
                    "n_val": int(len(val_indices)),
                    "val_balanced_accuracy": val_score,
                }
            )
            if best_val_balanced_accuracy is None or val_score > best_val_balanced_accuracy:
                best_val_balanced_accuracy = val_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                row["is_best_epoch"] = True
            else:
                epochs_without_improvement += 1
                row["is_best_epoch"] = False
        history.append(row)

        if (
            val_indices is not None
            and config.patience is not None
            and epochs_without_improvement >= int(config.patience)
            ):
            break

    return history, trained_epochs, best_epoch, best_val_balanced_accuracy, best_state, sampling_audit


def fit_predict_torch(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    epochs: int | None = None,
    max_epochs: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    seed: int = 0,
    train_groups: np.ndarray | None = None,
    patience: int | None = None,
    activation: str | None = None,
    normalization: str | None = None,
    dropout: float | None = None,
    optimizer: str | None = None,
    device: str | None = "auto",
    checkpoint_path: str | None = None,
    return_metadata: bool = False,
    train_session_labels: np.ndarray | None = None,
    source_balance_mode: str = "pooled_all",
) -> np.ndarray | TorchFitResult:
    import torch
    from torch import nn

    torch_device = _resolve_device(device)
    config = resolve_torch_config(
        method,
        max_epochs=max_epochs,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        activation=activation,
        normalization=normalization,
        dropout=dropout,
        optimizer=optimizer,
    )
    class_to_int = {label: i for i, label in enumerate(classes)}
    y_train_i = np.asarray([class_to_int[label] for label in y_train], dtype=np.int64)
    if source_balance_mode not in {"pooled_all", "session_balanced"}:
        raise ValueError(f"Unknown source_balance_mode: {source_balance_mode}")
    session_labels = None if train_session_labels is None else np.asarray(train_session_labels).astype(str)
    if session_labels is not None and len(session_labels) != len(y_train_i):
        raise ValueError("train_session_labels length does not match y_train")
    normalization_weighting = "session_equal" if source_balance_mode == "session_balanced" else "sample_weighted"

    if config.patience is not None:
        inner_train_idx, val_idx, val_groups = _grouped_validation_indices(
            y_train_i,
            train_groups,
            seed=seed,
        )
    else:
        inner_train_idx, val_idx, val_groups = None, None, []
    use_early_stopping = config.patience is not None and val_idx is not None
    if inner_train_idx is None:
        inner_train_idx = np.arange(len(y_train_i), dtype=np.int64)

    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, object]] = []
    normalization_stats_by_phase: list[dict[str, object]] = []
    best_val_balanced_accuracy: float | None = None
    selection_trained_epochs = 0
    if use_early_stopping and val_idx is not None:
        _, selection_X_train, selection_normalization_stats = _normalize_frames(
            X_train[inner_train_idx],
            X_train,
            statistics_scope="inner_train_fold_only_for_epoch_selection",
            normalization_weighting=normalization_weighting,
            train_session_labels=session_labels[inner_train_idx] if session_labels is not None else None,
        )
        selection_normalization_stats["phase"] = "inner_epoch_selection"
        normalization_stats_by_phase.append(selection_normalization_stats)
        selection_train_tensor, _ = _torch_frame_tensors(method, selection_X_train, selection_X_train)

        _set_reproducible_seed(seed)
        selection_model = build_torch_model(
            method,
            len(classes),
            tuple(selection_X_train.shape[1:]),
            config,
        ).to(torch_device)
        (
            selection_history,
            selection_trained_epochs,
            selected_epoch,
            best_val_balanced_accuracy,
            _,
            selection_sampling_audit,
        ) = _train_model_epochs(
            selection_model,
            selection_train_tensor,
            y_train_i,
            inner_train_idx,
            device=torch_device,
            config=config,
            criterion=criterion,
            seed=seed,
            n_epochs=config.max_epochs,
            phase="inner_epoch_selection",
            val_indices=val_idx,
            train_session_labels=session_labels,
            train_groups=train_groups,
            source_balance_mode=source_balance_mode,
        )
        history.extend(selection_history)
        best_epoch = int(selected_epoch or selection_trained_epochs)
        final_phase = "full_outer_train_retrain_after_epoch_selection"
    else:
        selection_sampling_audit = []
        best_epoch = int(config.max_epochs)
        final_phase = "full_outer_train_fixed_epochs"

    X_train_final, X_test_final, final_normalization_stats = _normalize_frames(
        X_train,
        X_test,
        statistics_scope="outer_train_fold_only_for_final_training",
        normalization_weighting=normalization_weighting,
        train_session_labels=session_labels,
    )
    final_normalization_stats["phase"] = final_phase
    normalization_stats_by_phase.append(final_normalization_stats)
    train_tensor, test_tensor = _torch_frame_tensors(method, X_train_final, X_test_final)

    full_train_idx = np.arange(len(y_train_i), dtype=np.int64)
    _set_reproducible_seed(seed)
    model = build_torch_model(method, len(classes), tuple(X_train_final.shape[1:]), config).to(torch_device)
    final_history, final_trained_epochs, _, _, _, final_sampling_audit = _train_model_epochs(
        model,
        train_tensor,
        y_train_i,
        full_train_idx,
        device=torch_device,
        config=config,
        criterion=criterion,
        seed=seed,
        n_epochs=best_epoch,
        phase=final_phase,
        val_indices=None,
        train_session_labels=session_labels,
        train_groups=train_groups,
        source_balance_mode=source_balance_mode,
    )
    history.extend(final_history)
    sampling_audit = selection_sampling_audit + final_sampling_audit
    final_draws_by_session: dict[str, int] = {}
    final_draw_props_by_session: dict[str, float] = {}
    samples_per_epoch = int(len(full_train_idx))
    final_audit_rows = [row for row in final_sampling_audit if row.get("epoch") == 1]
    total_first_epoch_draws = sum(int(row.get("n_draws", 0)) for row in final_audit_rows)
    for row in final_audit_rows:
        source_session = str(row.get("source_session"))
        n_draws = int(row.get("n_draws", 0))
        final_draws_by_session[source_session] = n_draws
        final_draw_props_by_session[source_session] = float(n_draws / max(total_first_epoch_draws, 1))
    final_state = copy.deepcopy(model.state_dict())

    if checkpoint_path is not None:
        torch.save(
            {
                "method": method,
                "model_state_dict": final_state,
                "classes": classes.tolist(),
                "config": config.__dict__,
                "seed": int(seed),
                "best_epoch": int(best_epoch),
                "best_val_balanced_accuracy": best_val_balanced_accuracy,
                "selection_trained_epochs": int(selection_trained_epochs),
                "final_trained_epochs": int(final_trained_epochs),
                "retrained_on_full_outer_train": True,
                "normalization": final_normalization_stats,
                "normalization_by_phase": normalization_stats_by_phase,
                "training_history": history,
                "training_session_sampling_audit": sampling_audit,
                "source_balance_mode": source_balance_mode,
            },
            checkpoint_path,
        )

    pred_i = _predict_int(model, test_tensor, torch_device, config.batch_size)
    result = TorchFitResult(
        predictions=classes[pred_i],
        metadata={
            "method": method,
            "device": str(torch_device),
            "seed": int(seed),
            "config": config.__dict__,
            "source_balance_mode": source_balance_mode,
            "normalization_weighting": normalization_weighting,
            "sampling_strategy": (
                "session_balanced_equal_batch_draws"
                if source_balance_mode == "session_balanced"
                else "pooled_all_shuffle_once_per_epoch"
            ),
            "sampling_seed": int(seed),
            "samples_per_epoch": samples_per_epoch,
            "effective_source_session_draw_counts": final_draws_by_session,
            "effective_source_session_draw_proportions": final_draw_props_by_session,
            "inner_validation": {
                "enabled": bool(use_early_stopping),
                "strategy": "cycle_group_holdout" if use_early_stopping else "fixed_epochs_no_inner_validation",
                "val_cycles": val_groups,
                "n_inner_train": int(len(inner_train_idx)),
                "n_val": int(len(val_idx)) if val_idx is not None else 0,
                "monitor": "balanced_accuracy",
                "patience": config.patience,
            },
            "best_epoch": int(best_epoch),
            "selection_trained_epochs": int(selection_trained_epochs),
            "final_trained_epochs": int(final_trained_epochs),
            "trained_epochs": int(final_trained_epochs),
            "final_training": {
                "strategy": final_phase,
                "n_final_train": int(len(full_train_idx)),
                "retrained_on_full_outer_train": True,
            },
            "best_val_balanced_accuracy": best_val_balanced_accuracy,
            "normalization": final_normalization_stats,
            "normalization_by_phase": normalization_stats_by_phase,
            "training_history": history,
            "training_session_sampling_audit": sampling_audit,
        },
    )
    return result if return_metadata else result.predictions
