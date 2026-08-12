"""Leakage-safe within-session masked-reconstruction pretraining for clean4.

This module intentionally imports the existing clean4 builder, grouped fold
generator, and SmallCNN frame encoder.  It contains no cross-session path and
does not read class labels while constructing SSL datasets/checkpoints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    BlockSequenceData,
    cycle_text,
    default_block_data_dir,
)
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


SSL_SEEDS = (20260812, 20260813, 20260814)
SSL_CONDITIONS = ("RANDOM_INIT", "SSL_FROZEN", "SSL_FINETUNE")
MASK_BLOCK_SIZE = (16, 16)
MASK_RATIO = 0.50
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")

REQUIRED_FORMAL_OUTPUTS = (
    "audit/historical_baseline_reproduction.csv",
    "audit/fold_reproduction.csv",
    "audit/smallcnn_architecture_audit.md",
    "audit/frame_preprocessing_audit.md",
    "audit/ssl_data_volume.csv",
    "audit/ssl_leakage_audit.csv",
    "audit/seed_audit.csv",
    "audit/config_freeze.md",
    "pretraining/reconstruction_losses.csv",
    "downstream/fold_metrics.csv",
    "downstream/predictions.csv",
    "summaries/session_level_metrics.csv",
    "summaries/paired_ssl_improvements.csv",
    "summaries/generalization_gap_summary.csv",
    "summaries/statistical_tests.csv",
    "summaries/two_task_correction.csv",
    "figures/binary/binary_session_BA.png",
    "figures/stimulus_type/stimulus_type_session_BA.png",
    "figures/binary/binary_train_test_gap.png",
    "figures/stimulus_type/stimulus_type_train_test_gap.png",
    "figures/overfitting/weak_session_overfitting.png",
    "figures/seed_stability/seed_stability_binary.png",
    "figures/seed_stability/seed_stability_stimulus_type.png",
    "figures/reconstruction_qc/ssl_reconstruction_contact_sheet.png",
    "report/ssl_masked_pretraining_report.md",
    "pytest_output.txt",
    "run_command.txt",
    "run_log.txt",
)


@dataclass(frozen=True)
class SSLPretrainingConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    mask_block_height: int = 16
    mask_block_width: int = 16
    mask_ratio: float = 0.50


@dataclass
class SSLFrameData:
    frames: np.ndarray
    cycles: np.ndarray
    original_frame_indices: np.ndarray
    source_h5_path: Path


@dataclass
class PretrainingResult:
    encoder: SmallCNNFrameEncoder
    decoder: nn.Module
    history: list[dict[str, Any]]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    actual_batch_size: int
    device: str
    qc: dict[str, np.ndarray | int]


@dataclass
class DownstreamResult:
    model: CNN2DMeanPool
    train_predictions: np.ndarray
    train_probabilities: np.ndarray
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    metrics: dict[str, float | int | str | bool]
    device: str


class SmallCNNReconstructionDecoder(nn.Module):
    """Small decoder used only during SSL; no skips, attention, or token path."""

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.output = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        x = self.project(spatial_features)
        x = F.interpolate(x, size=(128, 501), mode="bilinear", align_corners=False)
        return self.output(x)


class MaskedReconstructionSmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SmallCNNFrameEncoder()
        self.decoder = SmallCNNReconstructionDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = self.encoder(x, return_spatial_feature_map=True)
        return self.decoder(spatial)


def count_parameters(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def deterministic_block_mask(
    height: int,
    width: int,
    *,
    seed: int,
    epoch: int,
    sample_index: int,
    block_size: tuple[int, int] = MASK_BLOCK_SIZE,
    mask_ratio: float = MASK_RATIO,
) -> np.ndarray:
    """Return a reproducible boolean spatial block mask with natural edges."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be positive")
    if not 0.0 < float(mask_ratio) < 1.0:
        raise ValueError("mask_ratio must be between zero and one")
    bh, bw = (int(block_size[0]), int(block_size[1]))
    if bh < 1 or bw < 1:
        raise ValueError("mask block dimensions must be positive")
    n_rows = int(np.ceil(height / bh))
    n_cols = int(np.ceil(width / bw))
    n_blocks = n_rows * n_cols
    n_masked = int(round(float(mask_ratio) * n_blocks))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), int(sample_index)]))
    selected = rng.choice(n_blocks, size=n_masked, replace=False)
    mask = np.zeros((height, width), dtype=bool)
    for block_i in selected:
        row = int(block_i) // n_cols
        col = int(block_i) % n_cols
        mask[row * bh : min((row + 1) * bh, height), col * bw : min((col + 1) * bw, width)] = True
    return mask


def masked_pixel_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked pixels only, as preregistered."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    if mask.shape != target.shape:
        try:
            mask = torch.broadcast_to(mask, target.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to target") from exc
    weights = mask.to(dtype=prediction.dtype)
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0:
        raise ValueError("mask contains no masked pixels")
    return (weights * (prediction - target).square()).sum() / denominator


class MaskedFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, frames: np.ndarray, *, seed: int) -> None:
        if frames.ndim != 3 or tuple(frames.shape[1:]) != EXPECTED_BLOCK_SHAPE[1:]:
            raise ValueError(f"expected [N,128,501], got {frames.shape}")
        self.frames = frames.astype(np.float32, copy=False)
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(len(self.frames))

    def __getitem__(self, sample_index: int):
        target = self.frames[int(sample_index)]
        mask = deterministic_block_mask(
            target.shape[0],
            target.shape[1],
            seed=self.seed,
            epoch=self.epoch,
            sample_index=int(sample_index),
        )
        masked = target.copy()
        masked[mask] = 0.0
        return (
            torch.from_numpy(masked[None]),
            torch.from_numpy(target[None]),
            torch.from_numpy(mask[None]),
            torch.tensor(int(sample_index), dtype=torch.int64),
        )


def _read_full_frame_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read only unlabeled frame/cycle/index datasets from one clean4 export."""
    with h5py.File(path, "r") as handle:
        frames = handle["/full/X_padded"][:]
        valid = handle["/full/valid_mask"][:].astype(bool)
        indices = handle["/full/original_frame_indices"][:]
        cycles = handle["/metadata/cycle"][:].astype(np.int64)
    return frames, valid, indices, cycles


def load_full_cycle_frames(
    project_dir: Path,
    session: str,
    cycles: Iterable[int],
    *,
    data_dir: Path | None = None,
) -> SSLFrameData:
    """Load every original frame belonging to the requested within-session cycles."""
    requested = {int(value) for value in cycles}
    if not requested:
        raise ValueError("at least one cycle is required")
    path = (data_dir or default_block_data_dir(project_dir)) / f"session_{session}_blocks.h5"
    frames, valid, indices, block_cycles = _read_full_frame_arrays(path)
    output_frames: list[np.ndarray] = []
    output_cycles: list[int] = []
    output_indices: list[int] = []
    for row_i, cycle in enumerate(block_cycles.tolist()):
        if int(cycle) not in requested:
            continue
        for time_i in np.flatnonzero(valid[row_i]):
            output_frames.append(frames[row_i, int(time_i)])
            output_cycles.append(int(cycle))
            output_indices.append(int(indices[row_i, int(time_i)]))
    if set(output_cycles) != requested:
        raise AssertionError(f"requested cycles not fully found: requested={sorted(requested)}, found={sorted(set(output_cycles))}")
    keys = list(zip(output_cycles, output_indices))
    if len(keys) != len(set(keys)):
        raise AssertionError("a raw frame appears in more than one full block")
    counts = {cycle: output_cycles.count(cycle) for cycle in requested}
    if any(count != 30 for count in counts.values()):
        raise AssertionError(f"complete cycle does not contain exactly 30 raw frames: {counts}")
    order = np.lexsort((np.asarray(output_indices), np.asarray(output_cycles)))
    return SSLFrameData(
        frames=np.stack(output_frames, axis=0)[order].astype(np.float32, copy=False),
        cycles=np.asarray(output_cycles, dtype=np.int64)[order],
        original_frame_indices=np.asarray(output_indices, dtype=np.int64)[order],
        source_h5_path=path,
    )


def fixed_ssl_validation_cycles(train_cycles: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    """Use a fixed ~15% cycle-level split; small outer training sets have no SSL val."""
    cycles = np.asarray(sorted({int(value) for value in train_cycles}), dtype=np.int64)
    if len(cycles) < 5:
        return cycles, np.asarray([], dtype=np.int64)
    n_val = max(1, int(round(0.15 * len(cycles))))
    return cycles[:-n_val], cycles[-n_val:]


def fit_ssl_frame_normalizer(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if frames.ndim != 3 or not len(frames):
        raise ValueError("nonempty [N,H,W] frames required")
    transformed = np.arcsinh(frames.astype(np.float32, copy=False)).astype(np.float64, copy=False)
    mean = transformed.mean(axis=0, keepdims=True)
    std = transformed.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_ssl_frame_normalizer(frames: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = np.arcsinh(frames.astype(np.float32, copy=False))
    output = (transformed - mean) / std
    if not np.isfinite(output).all():
        raise ValueError("non-finite SSL frame after preprocessing")
    return output.astype(np.float32, copy=False)


def _ssl_loader(dataset: MaskedFrameDataset, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=max(1, min(int(batch_size), len(dataset))),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _run_pretraining_once(
    train_frames: np.ndarray,
    val_frames: np.ndarray | None,
    *,
    seed: int,
    config: SSLPretrainingConfig,
    batch_size: int,
    device: torch.device,
) -> PretrainingResult:
    set_reproducible_seed(seed)
    model = MaskedReconstructionSmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_dataset = MaskedFrameDataset(train_frames, seed=seed)
    val_dataset = MaskedFrameDataset(val_frames, seed=seed + 1_000_003) if val_frames is not None and len(val_frames) else None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config.epochs) + 1):
        train_dataset.set_epoch(epoch)
        loader = _ssl_loader(train_dataset, batch_size, seed + epoch, True)
        model.train()
        train_loss_sum = 0.0
        train_masked_pixels = 0
        for masked, target, mask, _sample_i in loader:
            masked = masked.to(device)
            target = target.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(masked)
            loss = masked_pixel_mse(prediction, target, mask)
            loss.backward()
            optimizer.step()
            pixels = int(mask.sum().detach().cpu())
            train_loss_sum += float(loss.detach().cpu()) * pixels
            train_masked_pixels += pixels
        val_loss = float("nan")
        if val_dataset is not None:
            val_dataset.set_epoch(epoch)
            model.eval()
            val_loss_sum = 0.0
            val_masked_pixels = 0
            with torch.no_grad():
                for masked, target, mask, _sample_i in _ssl_loader(val_dataset, batch_size, seed, False):
                    masked = masked.to(device)
                    target = target.to(device)
                    mask = mask.to(device)
                    loss = masked_pixel_mse(model(masked), target, mask)
                    pixels = int(mask.sum().detach().cpu())
                    val_loss_sum += float(loss.detach().cpu()) * pixels
                    val_masked_pixels += pixels
            val_loss = val_loss_sum / max(val_masked_pixels, 1)
        history.append(
            {
                "epoch": epoch,
                "train_reconstruction_loss": train_loss_sum / max(train_masked_pixels, 1),
                "ssl_val_reconstruction_loss": val_loss,
                "actual_batch_size": int(batch_size),
            }
        )
    train_dataset.set_epoch(int(config.epochs))
    masked, target, mask, sample_i = train_dataset[0]
    model.eval()
    with torch.no_grad():
        reconstruction = model(masked[None].to(device))[0].cpu().numpy()
    return PretrainingResult(
        encoder=model.encoder.cpu(),
        decoder=model.decoder.cpu(),
        history=history,
        normalization_mean=np.empty((0,), dtype=np.float32),
        normalization_std=np.empty((0,), dtype=np.float32),
        actual_batch_size=int(batch_size),
        device=str(device),
        qc={
            "sample_index": int(sample_i),
            "original": target.numpy()[0],
            "masked": masked.numpy()[0],
            "mask": mask.numpy()[0],
            "reconstruction": reconstruction[0],
        },
    )


def pretrain_masked_smallcnn(
    train_raw_frames: np.ndarray,
    val_raw_frames: np.ndarray | None,
    *,
    seed: int,
    config: SSLPretrainingConfig = SSLPretrainingConfig(),
    device: str | None = "auto",
) -> PretrainingResult:
    """Pretrain for the fixed epoch count, reducing only batch size on OOM."""
    mean, std = fit_ssl_frame_normalizer(train_raw_frames)
    train_frames = apply_ssl_frame_normalizer(train_raw_frames, mean, std)
    val_frames = (
        apply_ssl_frame_normalizer(val_raw_frames, mean, std)
        if val_raw_frames is not None and len(val_raw_frames)
        else None
    )
    torch_device = resolve_device(device)
    candidates = [value for value in (32, 16, 8) if value <= int(config.batch_size)]
    if int(config.batch_size) not in candidates:
        candidates.insert(0, int(config.batch_size))
    last_error: RuntimeError | None = None
    for batch_size in candidates:
        try:
            result = _run_pretraining_once(
                train_frames,
                val_frames,
                seed=seed,
                config=config,
                batch_size=batch_size,
                device=torch_device,
            )
            result.normalization_mean = mean
            result.normalization_std = std
            return result
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_error = exc
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
    assert last_error is not None
    raise last_error


def save_ssl_encoder_checkpoint(
    path: Path,
    result: PretrainingResult,
    *,
    session: str,
    fold: int,
    seed: int,
    ssl_train_cycles: Iterable[int],
    ssl_val_cycles: Iterable[int],
    outer_test_cycles: Iterable[int],
    config: SSLPretrainingConfig,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state_dict": {key: value.detach().cpu() for key, value in result.encoder.state_dict().items()},
        "encoder_class": "SmallCNNFrameEncoder",
        "session": str(session),
        "fold": int(fold),
        "seed": int(seed),
        "ssl_train_cycles": [int(value) for value in ssl_train_cycles],
        "ssl_val_cycles": [int(value) for value in ssl_val_cycles],
        "outer_test_cycles": [int(value) for value in outer_test_cycles],
        "normalization_mean": result.normalization_mean,
        "normalization_std": result.normalization_std,
        "pretraining_config": asdict(config),
        "actual_batch_size": int(result.actual_batch_size),
        "final_epoch": int(config.epochs),
        "decoder_discarded": True,
        "contains_labels": False,
    }
    forbidden = {key for key in payload if "label" in key.lower() or key.lower() in {"y", "targets"}}
    if forbidden != {"contains_labels"}:
        raise AssertionError(f"label-like fields entered SSL checkpoint: {sorted(forbidden)}")
    torch.save(payload, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"checkpoint_path": str(path), "checkpoint_sha256": digest, **{k: payload[k] for k in (
        "session", "fold", "seed", "ssl_train_cycles", "ssl_val_cycles", "outer_test_cycles",
        "actual_batch_size", "final_epoch", "decoder_discarded", "contains_labels",
    )}}


def load_ssl_encoder_checkpoint(path: Path | str) -> tuple[SmallCNNFrameEncoder, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("contains_labels") is not False:
        raise AssertionError("SSL checkpoint does not certify label-free construction")
    encoder = SmallCNNFrameEncoder()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    return encoder, payload


def configure_downstream_model(
    condition: str,
    *,
    n_classes: int,
    pretrained_encoder_state: dict[str, torch.Tensor] | None,
) -> CNN2DMeanPool:
    if condition not in SSL_CONDITIONS:
        raise ValueError(f"unknown downstream condition: {condition}")
    model = CNN2DMeanPool(n_classes=n_classes, temporal_length=4)
    if condition == "RANDOM_INIT":
        if pretrained_encoder_state is not None:
            raise ValueError("RANDOM_INIT must not receive an SSL checkpoint")
    else:
        if pretrained_encoder_state is None:
            raise ValueError(f"{condition} requires an SSL encoder checkpoint")
        model.encoder.load_state_dict(pretrained_encoder_state, strict=True)
    train_encoder = condition != "SSL_FROZEN"
    for parameter in model.encoder.parameters():
        parameter.requires_grad = train_encoder
    return model


def binary_roc_auc(y_true: np.ndarray, positive_probability: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(positive_probability, dtype=float)
    positive = scores[y == 1]
    negative = scores[y == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((comparisons + 0.5 * ties) / (len(positive) * len(negative)))


def _supervised_epoch_loop(
    model: nn.Module,
    train_tensor: torch.Tensor,
    y_train_i: np.ndarray,
    *,
    seed: int,
    config: DeepTrainingConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(train_tensor, torch.from_numpy(y_train_i))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=max(1, min(int(config.batch_size), len(dataset))),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        # A frozen representation includes BatchNorm running statistics.  Keep
        # the encoder in eval mode so downstream labels cannot update them.
        if hasattr(model, "encoder") and not any(
            parameter.requires_grad for parameter in model.encoder.parameters()
        ):
            model.encoder.eval()
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
            n = len(yb)
            total_loss += float(loss.detach().cpu()) * n
            total_correct += int((logits.argmax(1) == yb).sum().detach().cpu())
            total_seen += n
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy_minibatch": total_correct / max(total_seen, 1),
            "n_train_blocks": int(total_seen),
        })
    return history


def train_downstream_fold(
    condition: str,
    data: BlockSequenceData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    fold: int,
    seed: int,
    pretrained_encoder_state: dict[str, torch.Tensor] | None,
    config: DeepTrainingConfig = DeepTrainingConfig(),
    device: str | None = "auto",
) -> DownstreamResult:
    """Train the historical SmallCNN feature-mean downstream configuration."""
    set_reproducible_seed(seed)
    train_cycles = cycle_text(data.groups[train_idx])
    test_cycles = cycle_text(data.groups[test_idx])
    X_train, X_test, norm_audit, _mean, _std = normalize_blocks_train_fold_only_with_stats(
        data.X[train_idx],
        data.X[test_idx],
        session=data.session,
        task=data.task,
        method="cnn2d_meanpool",
        seed=seed,
        fold=fold,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
    )
    classes = np.asarray(sorted(TASK_CLASS_NAMES[data.task]), dtype=np.int64)
    y_train_i = labels_to_class_indices(data.y[train_idx], classes)
    train_tensor = blocks_to_sequence_tensor(X_train)
    test_tensor = blocks_to_sequence_tensor(X_test)
    torch_device = resolve_device(device)
    model = configure_downstream_model(
        condition,
        n_classes=len(classes),
        pretrained_encoder_state=pretrained_encoder_state,
    ).to(torch_device)
    history = _supervised_epoch_loop(
        model,
        train_tensor,
        y_train_i,
        seed=seed,
        config=config,
        device=torch_device,
    )
    train_probs = predict_probabilities(model, train_tensor, device=torch_device, batch_size=config.batch_size)
    test_probs = predict_probabilities(model, test_tensor, device=torch_device, batch_size=config.batch_size)
    train_pred = classes[train_probs.argmax(axis=1)]
    test_pred = classes[test_probs.argmax(axis=1)]
    train_metrics = classification_metrics(data.y[train_idx], train_pred)
    test_metrics = classification_metrics(data.y[test_idx], test_pred)
    metrics: dict[str, float | int | str | bool] = {
        "session": str(data.session),
        "task": data.task,
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "train_accuracy": train_metrics["accuracy"],
        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "macro_F1": test_metrics["macro_f1"],
        "ROC_AUC": binary_roc_auc(data.y[test_idx], test_probs[:, 1]),
        "best_epoch": int(config.max_epochs),
        "train_test_gap_accuracy": train_metrics["accuracy"] - test_metrics["accuracy"],
        "train_test_gap_BA": train_metrics["balanced_accuracy"] - test_metrics["balanced_accuracy"],
        "n_train_blocks": int(len(train_idx)),
        "n_test_blocks": int(len(test_idx)),
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "encoder_requires_grad": bool(any(parameter.requires_grad for parameter in model.encoder.parameters())),
        "decoder_present": bool(hasattr(model, "decoder")),
    }
    return DownstreamResult(
        model=model.cpu(),
        train_predictions=train_pred,
        train_probabilities=train_probs,
        test_predictions=test_pred,
        test_probabilities=test_probs,
        history=history,
        normalization_audit=norm_audit,
        metrics=metrics,
        device=str(torch_device),
    )


def checkpoint_has_no_labels(path: Path | str) -> bool:
    _encoder, payload = load_ssl_encoder_checkpoint(path)
    serialized_keys = json.dumps(sorted(payload.keys())).lower()
    return payload.get("contains_labels") is False and '"y"' not in serialized_keys and "target" not in serialized_keys


def assert_within_session_scope(sessions: Iterable[str]) -> None:
    values = [str(value) for value in sessions]
    invalid = sorted(set(values) - set(EXPECTED_SESSIONS))
    if invalid:
        raise ValueError(f"non-preregistered sessions: {invalid}")


def missing_formal_outputs(output_dir: Path) -> list[str]:
    """Return preregistered deliverables that are absent or empty."""
    missing = []
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    checkpoint_dir = output_dir / "pretraining/checkpoints"
    if not checkpoint_dir.is_dir() or not any(checkpoint_dir.rglob("*.pt")):
        missing.append("pretraining/checkpoints/**/*.pt")
    curve_dir = output_dir / "downstream/training_curves"
    if not curve_dir.is_dir() or not any(curve_dir.glob("*.csv")):
        missing.append("downstream/training_curves/*.csv")
    qc_dir = output_dir / "figures/reconstruction_qc"
    for session in EXPECTED_SESSIONS:
        if not (qc_dir / f"session_{session}_reconstruction_qc.png").is_file():
            missing.append(f"figures/reconstruction_qc/session_{session}_reconstruction_qc.png")
    return missing
