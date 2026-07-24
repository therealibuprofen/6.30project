from __future__ import annotations

from pathlib import Path

import numpy as np

from ultrasound_decoding.deep import _normalize_frames, _resolve_device
from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.interpretability.common import (
    CLASS_ORDER,
    IMAGE_SHAPE,
    checkpoint_path_for,
    display_model_name,
    load_torch_checkpoint_model,
)


def normalize_fold_inputs(X: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    X_train_norm, X_test_norm, stats = _normalize_frames(
        X[train_idx],
        X[test_idx],
        statistics_scope="outer_train_fold_only_for_interpretability",
        normalization_weighting="sample_weighted",
    )
    if tuple(X_train_norm.shape[1:]) != IMAGE_SHAPE or tuple(X_test_norm.shape[1:]) != IMAGE_SHAPE:
        raise ValueError("normalized fold input shape changed unexpectedly")
    return X_train_norm, X_test_norm, stats


def tensor_from_normalized_frames(X_norm: np.ndarray, device) :
    import torch

    if X_norm.ndim != 3:
        raise ValueError(f"expected normalized single frames [N,H,W], got {X_norm.shape}")
    return torch.from_numpy(X_norm[:, None, :, :].astype(np.float32, copy=False)).to(device)


def class_indices(y: np.ndarray, classes: np.ndarray = CLASS_ORDER) -> np.ndarray:
    lookup = {label: i for i, label in enumerate(classes)}
    return np.asarray([lookup[label] for label in y], dtype=np.int64)


def predict_logits_probabilities(model, tensor, batch_size: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    logits_parts = []
    model.eval()
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(tensor), batch_size=max(1, batch_size), shuffle=False):
            logits = model(xb)
            logits_parts.append(logits.detach().cpu())
    logits_t = torch.cat(logits_parts, dim=0) if logits_parts else torch.empty((0, 0))
    probs_t = torch.softmax(logits_t, dim=1)
    pred_i = probs_t.argmax(dim=1).numpy()
    return logits_t.numpy(), probs_t.numpy(), pred_i


def load_fold_model_and_inputs(
    *,
    project_dir: Path,
    benchmark_root: Path,
    session: str,
    task: str,
    model_name: str,
    seed: int,
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str = "auto",
):
    torch_device = _resolve_device(device)
    checkpoint_path = checkpoint_path_for(benchmark_root, session, task, model_name, seed, fold)
    model, checkpoint, config = load_torch_checkpoint_model(
        checkpoint_path,
        model_name,
        n_classes=len(CLASS_ORDER),
        input_shape=IMAGE_SHAPE,
        device=str(torch_device),
    )
    if checkpoint.get("classes") != CLASS_ORDER.tolist():
        raise ValueError(f"checkpoint classes differ from expected binary class order: {checkpoint.get('classes')}")
    X_train_norm, X_test_norm, norm_stats = normalize_fold_inputs(X, train_idx, test_idx)
    tensor = tensor_from_normalized_frames(X_test_norm, torch_device)
    y_test = y[test_idx]
    y_test_i = class_indices(y_test)
    return {
        "model": model,
        "device": torch_device,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "config": config,
        "X_train_norm": X_train_norm,
        "X_test_norm": X_test_norm,
        "test_tensor": tensor,
        "y_test": y_test,
        "y_test_i": y_test_i,
        "normalization": norm_stats,
        "display_model": display_model_name(model_name),
    }


def original_metrics_payload(
    *,
    model_name: str,
    seed: int,
    fold: int,
    checkpoint_path: Path,
    y_test: np.ndarray,
    pred_i: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray = CLASS_ORDER,
) -> dict[str, object]:
    pred = classes[pred_i]
    metrics = classification_metrics(y_test, pred)
    truth_i = class_indices(y_test, classes)
    true_probs = probabilities[np.arange(len(truth_i)), truth_i]
    return {
        "model": display_model_name(model_name),
        "seed": int(seed),
        "fold": int(fold),
        "checkpoint_path": str(checkpoint_path),
        "classes": classes.tolist(),
        "n_test_samples": int(len(y_test)),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "mean_true_class_probability": float(true_probs.mean()) if len(true_probs) else None,
    }

