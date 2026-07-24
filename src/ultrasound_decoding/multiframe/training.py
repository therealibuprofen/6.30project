from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ultrasound_decoding.evaluate import classification_metrics

from .models import build_multiframe_model, count_trainable_parameters


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
    """Apply arcsinh and pixel-wise z-score using only train blocks and all their four frames."""
    if X_train.ndim != 4 or X_test.ndim != 4:
        raise ValueError(f"expected block tensors [N, 4, H, W], got {X_train.shape} and {X_test.shape}")
    if X_train.shape[1:] != X_test.shape[1:]:
        raise ValueError("train and test block shapes differ")
    if X_train.shape[1] != 4:
        raise ValueError(f"expected four clean frames per block, got {X_train.shape[1]}")

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
        "n_train_frames_for_stats": int(len(X_train) * 4),
        "n_test_frames_transformed": int(len(X_test) * 4),
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
    return X_train_norm.astype(np.float32, copy=False), X_test_norm.astype(np.float32, copy=False), audit


def blocks_to_sequence_tensor(X: np.ndarray) -> torch.Tensor:
    if X.ndim != 4 or X.shape[1] != 4:
        raise ValueError(f"expected [N, 4, H, W], got {X.shape}")
    return torch.from_numpy(X[:, :, None, :, :].astype(np.float32, copy=False))


def blocks_to_frame_tensor(X: np.ndarray) -> torch.Tensor:
    if X.ndim != 4 or X.shape[1] != 4:
        raise ValueError(f"expected [N, 4, H, W], got {X.shape}")
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
) -> list[dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, config)
    batch_size = max(1, min(int(config.batch_size), int(batch_size_reference)))
    dataset = TensorDataset(train_tensor, torch.from_numpy(y_train_i))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
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
) -> np.ndarray:
    loader = DataLoader(TensorDataset(tensor), batch_size=max(1, int(batch_size)), shuffle=False)
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
    X_train_norm, X_test_norm, norm_audit = normalize_blocks_train_fold_only(
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
    model = build_multiframe_model(method, n_classes=len(classes)).to(torch_device)
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
) -> FoldTrainingResult:
    method = "single_frame_late_fusion"
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit = normalize_blocks_train_fold_only(
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
    y_train_frames_i = np.repeat(y_train_i, 4)
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
        batch_size_reference=len(X_train),
    )
    frame_probs = predict_probabilities(model, test_tensor, device=torch_device, batch_size=config.batch_size)
    block_probs = frame_probs.reshape(len(X_test), 4, len(classes)).mean(axis=1)
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
    )


def order_sensitivity_for_trained_sequence_model(
    model: nn.Module,
    X_test_normalized: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
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
    out["shuffled_order_ba"] = out["fixed_shuffle_order_ba"]
    out["shuffled_order_accuracy"] = out["fixed_shuffle_order_accuracy"]
    out["shuffled_order_macro_f1"] = out["fixed_shuffle_order_macro_f1"]
    out["shuffled_prediction_is_single_class"] = out["fixed_shuffle_prediction_is_single_class"]
    out["reverse_drop"] = float(out["original_order_ba"] - out["reverse_order_ba"])
    out["shuffle_drop"] = float(out["original_order_ba"] - out["shuffled_order_ba"])
    return out
