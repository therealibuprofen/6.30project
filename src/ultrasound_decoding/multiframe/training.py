from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import random
from typing import Any

# Required before CUDA-backed deterministic linear algebra is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ultrasound_decoding.evaluate import classification_metrics

from .models import build_multiframe_model, count_trainable_parameters, model_architecture_config


@dataclass(frozen=True)
class DeepTrainingConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 16
    max_epochs: int = 40
    dropout: float = 0.25
    loss: str = "cross_entropy"


@dataclass
class FoldTrainingResult:
    method: str
    seed: int
    predictions: np.ndarray
    probabilities: np.ndarray
    model: nn.Module
    model_parameters: int
    history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    final_training_loss: float
    final_trained_epochs: int
    device: str
    X_test_normalized: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    normalization_transform: str
    input_shape: tuple[int, ...]
    model_config: dict[str, Any]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


def resolve_device(device: str | None = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _make_optimizer(model: nn.Module, config: DeepTrainingConfig):
    name = config.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def _quality_summary(X: np.ndarray) -> dict[str, Any]:
    flat = X.reshape(len(X), -1) if len(X) else X.reshape(0, int(np.prod(X.shape[1:])))
    finite = np.isfinite(flat)
    return {
        "n_blocks": int(len(X)),
        "n_frames": int(len(X) * X.shape[1]) if X.ndim == 4 else int(len(X)),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "nonfinite_count": int((~finite).sum()),
        "all_zero_blocks": int(np.all(flat == 0, axis=1).sum()) if len(X) else 0,
    }


def normalize_blocks_train_fold_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    session: str,
    task: str,
    method: str,
    seed: int,
    fold: int,
    train_cycles: str,
    test_cycles: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply arcsinh and pixel-wise z-score using only train blocks and all their frames."""
    X_train_norm, X_test_norm, audit, _mean, _std = normalize_blocks_train_fold_only_with_stats(
        X_train,
        X_test,
        session=session,
        task=task,
        method=method,
        seed=seed,
        fold=fold,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
    )
    return X_train_norm, X_test_norm, audit


def normalize_blocks_train_fold_only_with_stats(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    session: str,
    task: str,
    method: str,
    seed: int,
    fold: int,
    train_cycles: str,
    test_cycles: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Return normalized arrays plus the actual pixel-wise mean/std arrays used for the fold."""
    if X_train.ndim != 4 or X_test.ndim != 4:
        raise ValueError(f"expected block tensors [N, T, H, W], got {X_train.shape} and {X_test.shape}")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("train and test block shapes differ")
    if X_train.shape[1] < 1:
        raise ValueError("expected at least one frame per block")

    train_quality = _quality_summary(X_train)
    test_quality = _quality_summary(X_test)
    if train_quality["nonfinite_count"] or test_quality["nonfinite_count"]:
        raise ValueError("deep model input contains NaN or Inf values")

    X_train_asinh = np.arcsinh(X_train.astype(np.float32, copy=False))
    X_test_asinh = np.arcsinh(X_test.astype(np.float32, copy=False))
    train_frames = X_train_asinh.reshape(-1, X_train_asinh.shape[-2], X_train_asinh.shape[-1]).astype(
        np.float64,
        copy=False,
    )
    mean = train_frames.mean(axis=0, keepdims=True)
    std_raw = train_frames.std(axis=0, keepdims=True)
    std = std_raw + 1e-6
    X_train_norm = (X_train_asinh - mean) / std
    X_test_norm = (X_test_asinh - mean) / std
    if not np.isfinite(X_train_norm).all() or not np.isfinite(X_test_norm).all():
        raise ValueError("deep normalized data contains NaN or Inf values")

    audit = {
        "session": str(session),
        "task": task,
        "method": method,
        "seed": int(seed),
        "fold": int(fold),
        "phase": "outer_train_fold_only",
        "transform": "arcsinh_then_train_pixel_zscore",
        "statistics_scope": "train_blocks_all_four_frames_only",
        "target_used_for_stats": False,
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
        "train_nan_count": int(train_quality["nan_count"]),
        "train_inf_count": int(train_quality["inf_count"]),
        "test_nan_count": int(test_quality["nan_count"]),
        "test_inf_count": int(test_quality["inf_count"]),
    }
    return (
        X_train_norm.astype(np.float32, copy=False),
        X_test_norm.astype(np.float32, copy=False),
        audit,
        mean.astype(np.float32, copy=True),
        std.astype(np.float32, copy=True),
    )


def blocks_to_sequence_tensor(X: np.ndarray) -> torch.Tensor:
    if X.ndim != 4 or X.shape[1] < 1:
        raise ValueError(f"expected [N, T, H, W] with T >= 1, got {X.shape}")
    return torch.from_numpy(X[:, :, None, :, :].astype(np.float32, copy=False))


def blocks_to_frame_tensor(X: np.ndarray) -> torch.Tensor:
    if X.ndim != 4 or X.shape[1] < 1:
        raise ValueError(f"expected [N, T, H, W] with T >= 1, got {X.shape}")
    frames = X.reshape(-1, X.shape[-2], X.shape[-1])
    return torch.from_numpy(frames[:, None, :, :].astype(np.float32, copy=False))


def labels_to_class_indices(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    lookup = {int(label): i for i, label in enumerate(classes)}
    return np.asarray([lookup[int(label)] for label in y], dtype=np.int64)


def _train_epochs(
    model: nn.Module,
    train_tensor: torch.Tensor,
    y_train_i: np.ndarray,
    *,
    config: DeepTrainingConfig,
    seed: int,
    device: torch.device,
    batch_size_reference: int,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, config)
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
            n = int(len(yb))
            total_loss += float(loss.detach().cpu().item()) * n
            total_correct += int((logits.argmax(dim=1) == yb).sum().detach().cpu().item())
            total_seen += n
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(total_loss / max(total_seen, 1)),
                "train_accuracy": float(total_correct / max(total_seen, 1)),
                "n_train_items": int(total_seen),
                "batch_size": int(batch_size),
            }
        )
    return history


def predict_probabilities(
    model: nn.Module,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
    )
    probs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
    return np.concatenate(probs, axis=0) if probs else np.zeros((0, 0), dtype=np.float32)


def train_sequence_fold(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    session: str,
    task: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    config: DeepTrainingConfig,
    device: str | None = "auto",
) -> FoldTrainingResult:
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = normalize_blocks_train_fold_only_with_stats(
        X_train,
        X_test,
        session=session,
        task=task,
        method=method,
        seed=seed,
        fold=fold,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    temporal_length = int(X_train_norm.shape[1])
    model = build_multiframe_model(method, n_classes=len(classes), temporal_length=temporal_length).to(torch_device)
    parameters = count_trainable_parameters(model)
    history = _train_epochs(
        model,
        train_tensor,
        y_train_i,
        config=config,
        seed=seed,
        device=torch_device,
        batch_size_reference=len(X_train),
    )
    probs = predict_probabilities(model, test_tensor, device=torch_device, batch_size=config.batch_size)
    pred_i = probs.argmax(axis=1)
    predictions = classes[pred_i]
    return FoldTrainingResult(
        method=method,
        seed=int(seed),
        predictions=predictions,
        probabilities=probs,
        model=model,
        model_parameters=parameters,
        history=history,
        normalization_audit=norm_audit,
        final_training_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        final_trained_epochs=int(len(history)),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_architecture_config(method, n_classes=len(classes), temporal_length=temporal_length),
    )


def train_single_frame_late_fusion_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    session: str,
    task: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    config: DeepTrainingConfig,
    device: str | None = "auto",
    method: str = "single_frame_late_fusion",
) -> FoldTrainingResult:
    if method not in {"single_frame_late_fusion", "fcnn_late_fusion"}:
        raise ValueError(f"late fusion training does not support method={method}")
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = normalize_blocks_train_fold_only_with_stats(
        X_train,
        X_test,
        session=session,
        task=task,
        method=method,
        seed=seed,
        fold=fold,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    temporal_length = int(X_train_norm.shape[1])
    y_train_frames_i = np.repeat(y_train_i, temporal_length)
    train_tensor = blocks_to_frame_tensor(X_train_norm)
    test_tensor = blocks_to_frame_tensor(X_test_norm)
    model = build_multiframe_model(method, n_classes=len(classes)).to(torch_device)
    parameters = count_trainable_parameters(model)
    history = _train_epochs(
        model,
        train_tensor,
        y_train_frames_i,
        config=config,
        seed=seed,
        device=torch_device,
        batch_size_reference=len(y_train_frames_i),
    )
    frame_probs = predict_probabilities(model, test_tensor, device=torch_device, batch_size=config.batch_size)
    block_probs = frame_probs.reshape(len(X_test), temporal_length, len(classes)).mean(axis=1)
    pred_i = block_probs.argmax(axis=1)
    predictions = classes[pred_i]
    return FoldTrainingResult(
        method=method,
        seed=int(seed),
        predictions=predictions,
        probabilities=block_probs,
        model=model,
        model_parameters=parameters,
        history=history,
        normalization_audit=norm_audit,
        final_training_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        final_trained_epochs=int(len(history)),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_architecture_config(method, n_classes=len(classes), temporal_length=temporal_length),
    )


def order_sensitivity_for_trained_sequence_model(
    model: nn.Module,
    X_test_normalized: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
    session: str | None = None,
    task: str | None = None,
    method: str | None = None,
    seed: int | None = None,
    fold: int | None = None,
    test_idx: np.ndarray | None = None,
    metadata=None,
    class_names: dict[int, str] | None = None,
    include_prediction_rows: bool = False,
) -> dict[str, Any]:
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    permutations = {
        "original": (0, 1, 2, 3),
        "reverse": (3, 2, 1, 0),
        "fixed_shuffle": (2, 0, 3, 1),
    }
    out: dict[str, Any] = {
        "reverse_permutation": "3,2,1,0",
        "shuffle_permutation": "2,0,3,1",
        "labels_modified": False,
    }
    prediction_rows: list[dict[str, Any]] = []
    if include_prediction_rows and (test_idx is None or metadata is None):
        raise ValueError("test_idx and metadata are required when include_prediction_rows=True")
    for name, order in permutations.items():
        X_perm = X_test_normalized[:, order, :, :]
        tensor = blocks_to_sequence_tensor(X_perm)
        probs = predict_probabilities(model, tensor, device=torch_device, batch_size=batch_size)
        pred = classes[probs.argmax(axis=1)]
        metrics = classification_metrics(y_test, pred)
        out[f"{name}_order_ba"] = float(metrics["balanced_accuracy"])
        out[f"{name}_order_accuracy"] = float(metrics["accuracy"])
        out[f"{name}_order_macro_f1"] = float(metrics["macro_f1"])
        out[f"{name}_prediction_is_single_class"] = bool(len(np.unique(pred)) == 1)
        if include_prediction_rows:
            assert test_idx is not None
            assert metadata is not None
            for local_i, sample_i in enumerate(test_idx):
                row = metadata.iloc[int(sample_i)]
                payload: dict[str, Any] = {
                    "session": str(session),
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "fold": fold,
                    "block_id": str(row["block_id"]),
                    "cycle": int(row["cycle"]),
                    "block_name": str(row["block_name"]),
                    "truth": int(y_test[local_i]),
                    "order_condition": name,
                    "permutation": ",".join(str(int(value)) for value in order),
                    "prediction": int(pred[local_i]),
                }
                for class_i, class_value in enumerate(classes):
                    class_label = (
                        class_names[int(class_value)]
                        if class_names is not None and int(class_value) in class_names
                        else f"class_{int(class_value)}"
                    )
                    payload[f"prob_{class_label}"] = float(probs[local_i, class_i])
                prediction_rows.append(payload)
    out["shuffled_order_ba"] = out["fixed_shuffle_order_ba"]
    out["shuffled_order_accuracy"] = out["fixed_shuffle_order_accuracy"]
    out["shuffled_order_macro_f1"] = out["fixed_shuffle_order_macro_f1"]
    out["shuffled_prediction_is_single_class"] = out["fixed_shuffle_prediction_is_single_class"]
    out["reverse_drop"] = float(out["original_order_ba"] - out["reverse_order_ba"])
    out["shuffle_drop"] = float(out["original_order_ba"] - out["shuffled_order_ba"])
    if include_prediction_rows:
        out["prediction_rows"] = prediction_rows
    return out


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_fold_checkpoint(
    path: Path,
    result: FoldTrainingResult,
    *,
    classes: np.ndarray,
    session: str,
    task: str,
    seed: int,
    fold: int,
    train_cycles: str,
    test_cycles: str,
    config: DeepTrainingConfig,
    code_version: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        key: value.detach().cpu()
        for key, value in result.model.state_dict().items()
    }
    payload = {
        "model_state_dict": state_dict,
        "method": result.method,
        "model_config": result.model_config,
        "model_parameters": int(result.model_parameters),
        "classes": [int(value) for value in classes.tolist()],
        "session": str(session),
        "task": task,
        "seed": int(seed),
        "fold": int(fold),
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "max_epochs": int(config.max_epochs),
        "final_epoch": int(result.final_trained_epochs),
        "normalization_mean": result.normalization_mean,
        "normalization_std": result.normalization_std,
        "normalization_transform": result.normalization_transform,
        "input_shape": [int(value) for value in result.input_shape],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version,
    }
    torch.save(payload, path)
    return {
        "session": str(session),
        "task": task,
        "method": result.method,
        "seed": int(seed),
        "fold": int(fold),
        "checkpoint_path": str(path),
        "checkpoint_sha256": checkpoint_sha256(path),
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "normalization_shape": str(list(result.normalization_mean.shape)),
        "status": "available",
    }


def load_multiframe_checkpoint(path: Path | str, map_location: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    method = str(payload["method"])
    classes = payload["classes"]
    temporal_length = int(payload.get("model_config", {}).get("temporal_length", 4))
    model = build_multiframe_model(method, n_classes=len(classes), temporal_length=temporal_length)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload
