from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import (
    CNN2DTemporal1D,
    count_trainable_parameters,
    model_architecture_config,
)
from .training import (
    DeepTrainingConfig,
    _make_optimizer,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


RAW_VARIANT = "raw"
CYCLEMIX_VARIANT = "cyclemix"
INPUT_VARIANTS = (RAW_VARIANT, CYCLEMIX_VARIANT)
MODEL_NAME = "cnn2d_temporal1d"
MODEL_IMPLEMENTATION_VERSION = "cyclemix_temporal1d_v1.0.0"
CYCLEMIX_IMPLEMENTATION_VERSION = "same_block_cross_cycle_50_50_v1"
LAMBDA_MIX = 0.5
BLOCK_NAMES = ("grating", "stop_after_grating", "dot", "static")
BLOCK_TO_ID = {name: index for index, name in enumerate(BLOCK_NAMES)}


@dataclass(frozen=True)
class SameBlockCrossCyclePartnerPool:
    partners_by_anchor: tuple[np.ndarray, ...]
    block_ids: np.ndarray
    cycle_ids: np.ndarray
    train_cycle_ids: tuple[int, ...]


@dataclass
class CycleMixFoldResult:
    method: str
    seed: int
    predictions: np.ndarray
    probabilities: np.ndarray
    model: nn.Module
    model_parameters: int
    history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    mix_audit: dict[str, Any]
    final_training_loss: float
    final_trained_epochs: int
    device: str
    X_test_normalized: np.ndarray
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    normalization_transform: str
    input_shape: tuple[int, ...]
    model_config: dict[str, Any]


def encode_block_identities(block_names: Sequence[str]) -> np.ndarray:
    names = np.asarray(block_names, dtype=str)
    unknown = sorted(set(names.tolist()) - set(BLOCK_NAMES))
    if unknown:
        raise ValueError(f"unknown block identities: {unknown}")
    return np.asarray([BLOCK_TO_ID[name] for name in names], dtype=np.int64)


def build_same_block_cross_cycle_partner_pool(
    block_ids: Sequence[int],
    cycle_ids: Sequence[int],
) -> SameBlockCrossCyclePartnerPool:
    """Build legal partner indices from the current outer-training fold only."""

    blocks = np.asarray(block_ids, dtype=np.int64)
    cycles = np.asarray(cycle_ids, dtype=np.int64)
    if blocks.ndim != 1 or cycles.ndim != 1 or len(blocks) != len(cycles):
        raise ValueError("block and cycle ids must be aligned one-dimensional arrays")
    train_cycles = tuple(sorted(int(value) for value in np.unique(cycles)))
    if not train_cycles:
        raise ValueError("CycleMix partner pool cannot be empty")
    partners = []
    for anchor in range(len(blocks)):
        legal = np.flatnonzero(blocks == blocks[anchor])
        legal = legal[cycles[legal] != cycles[anchor]]
        if np.any(legal == anchor):
            raise AssertionError("CycleMix partner pool contains a self-pair")
        partners.append(legal.astype(np.int64, copy=False))
    return SameBlockCrossCyclePartnerPool(
        partners_by_anchor=tuple(partners),
        block_ids=blocks.copy(),
        cycle_ids=cycles.copy(),
        train_cycle_ids=train_cycles,
    )


def _partner_rng(
    *, seed: int, fold: int, epoch: int, batch_index: int
) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(
        [int(seed), int(fold), int(epoch), int(batch_index), 0xC1C1E]
    )
    return np.random.default_rng(seed_sequence)


def select_cyclemix_partners(
    anchor_indices: Sequence[int],
    partner_pool: SameBlockCrossCyclePartnerPool,
    *,
    seed: int,
    fold: int,
    epoch: int,
    batch_index: int,
) -> np.ndarray:
    """Select at most one reproducible legal partner per training anchor."""

    anchors = np.asarray(anchor_indices, dtype=np.int64)
    if anchors.ndim != 1:
        raise ValueError("anchor indices must be one-dimensional")
    if np.any(anchors < 0) or np.any(anchors >= len(partner_pool.block_ids)):
        raise IndexError("anchor index falls outside the training partner pool")
    rng = _partner_rng(
        seed=seed, fold=fold, epoch=epoch, batch_index=batch_index
    )
    selected = np.full(len(anchors), -1, dtype=np.int64)
    for local_index, anchor in enumerate(anchors):
        legal = partner_pool.partners_by_anchor[int(anchor)]
        if len(legal):
            selected[local_index] = int(legal[int(rng.integers(len(legal)))])
    return selected


def make_cyclemix_batch(
    normalized_train_tensor: torch.Tensor,
    y_train: torch.Tensor,
    anchor_indices: Sequence[int],
    partner_indices: Sequence[int],
    partner_pool: SameBlockCrossCyclePartnerPool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Create fixed 50:50 samples in normalized space, retaining only legal pairs."""

    anchors = np.asarray(anchor_indices, dtype=np.int64)
    partners = np.asarray(partner_indices, dtype=np.int64)
    if anchors.shape != partners.shape:
        raise ValueError("anchor and partner selections differ in shape")
    valid = partners >= 0
    valid_anchors = anchors[valid]
    valid_partners = partners[valid]
    if np.any(valid_partners >= len(partner_pool.block_ids)):
        raise IndexError("partner index falls outside the training pool")
    same_cycle_violations = int(
        np.sum(
            partner_pool.cycle_ids[valid_anchors]
            == partner_pool.cycle_ids[valid_partners]
        )
    )
    different_block_violations = int(
        np.sum(
            partner_pool.block_ids[valid_anchors]
            != partner_pool.block_ids[valid_partners]
        )
    )
    self_pair_violations = int(np.sum(valid_anchors == valid_partners))
    if same_cycle_violations or different_block_violations or self_pair_violations:
        raise AssertionError(
            "illegal CycleMix pair: "
            f"same_cycle={same_cycle_violations}, "
            f"different_block={different_block_violations}, "
            f"self_pair={self_pair_violations}"
        )
    if len(valid_anchors) == 0:
        empty_x = normalized_train_tensor[:0]
        empty_y = y_train[:0]
        return empty_x, empty_y, {
            "number_of_mixed_samples_generated": 0,
            "number_of_anchors_without_valid_partner": int(len(anchors)),
            "same_cycle_violation_count": 0,
            "different_block_violation_count": 0,
            "self_pair_violation_count": 0,
        }
    anchor_tensor = normalized_train_tensor[
        torch.from_numpy(valid_anchors).long()
    ]
    partner_tensor = normalized_train_tensor[
        torch.from_numpy(valid_partners).long()
    ]
    anchor_labels = y_train[torch.from_numpy(valid_anchors).long()]
    partner_labels = y_train[torch.from_numpy(valid_partners).long()]
    if not torch.equal(anchor_labels, partner_labels):
        raise AssertionError("same-block CycleMix partners have different hard labels")
    mixed = LAMBDA_MIX * anchor_tensor + (1.0 - LAMBDA_MIX) * partner_tensor
    return mixed, anchor_labels, {
        "number_of_mixed_samples_generated": int(len(valid_anchors)),
        "number_of_anchors_without_valid_partner": int(np.sum(~valid)),
        "same_cycle_violation_count": 0,
        "different_block_violation_count": 0,
        "self_pair_violation_count": 0,
    }


def preprocess_train_fold_only(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    input_variant: str,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Run the frozen real-sample normalization before any synthetic mixing."""

    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown input variant: {input_variant!r}")
    train_norm, test_norm, audit, mean, std = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_test,
            session=str(session),
            task="binary",
            method=input_variant,
            seed=int(seed),
            fold=int(fold),
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    audit.update(
        {
            "input_variant": input_variant,
            "preprocessing_order": (
                "clean4 -> arcsinh -> train_fold_pixel_zscore -> "
                "training_only_cyclemix"
                if input_variant == CYCLEMIX_VARIANT
                else "clean4 -> arcsinh -> train_fold_pixel_zscore"
            ),
            "normalization_fit_before_cyclemix": True,
            "synthetic_samples_used_for_normalization_fit": False,
            "test_mix_applied": False,
            "test_block_identity_used": False,
        }
    )
    return train_norm, test_norm, audit, mean, std


def _train_epochs_with_cyclemix(
    model: CNN2DTemporal1D,
    train_tensor: torch.Tensor,
    y_train_i: np.ndarray,
    partner_pool: SameBlockCrossCyclePartnerPool,
    *,
    input_variant: str,
    config: DeepTrainingConfig,
    seed: int,
    fold: int,
    device: torch.device,
    batch_size_reference: int,
    num_workers: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, config)
    batch_size = max(1, min(int(config.batch_size), int(batch_size_reference)))
    y_tensor = torch.from_numpy(y_train_i.astype(np.int64, copy=False))
    sample_indices = torch.arange(len(train_tensor), dtype=torch.int64)
    dataset = TensorDataset(train_tensor, y_tensor, sample_indices)
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
    examples: list[dict[str, Any]] = []
    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        total_loss_sum = 0.0
        total_optimization_items = 0
        real_correct = 0
        real_seen = 0
        mixed_correct = 0
        mixed_seen = 0
        number_of_mixed_samples = 0
        anchors_without_partner = 0
        same_cycle_violations = 0
        different_block_violations = 0
        self_pair_violations = 0
        number_of_batches = 0
        for batch_index, (xb, yb, anchor_indices) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            real_count = int(len(yb))
            if input_variant == CYCLEMIX_VARIANT:
                anchor_numpy = anchor_indices.numpy()
                partner_indices = select_cyclemix_partners(
                    anchor_numpy,
                    partner_pool,
                    seed=seed,
                    fold=fold,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                mixed_x, mixed_y, batch_audit = make_cyclemix_batch(
                    train_tensor,
                    y_tensor,
                    anchor_numpy,
                    partner_indices,
                    partner_pool,
                )
                augmented_x = torch.cat((xb, mixed_x), dim=0).to(device)
                augmented_y = torch.cat((yb, mixed_y), dim=0).to(device)
                for local_index, partner in enumerate(partner_indices):
                    if partner >= 0 and len(examples) < 8:
                        anchor = int(anchor_numpy[local_index])
                        examples.append(
                            {
                                "epoch": int(epoch),
                                "batch_index": int(batch_index),
                                "anchor_index": anchor,
                                "anchor_cycle": int(partner_pool.cycle_ids[anchor]),
                                "anchor_block_id": int(partner_pool.block_ids[anchor]),
                                "partner_index": int(partner),
                                "partner_cycle": int(
                                    partner_pool.cycle_ids[int(partner)]
                                ),
                                "partner_block_id": int(
                                    partner_pool.block_ids[int(partner)]
                                ),
                                "lambda_mix": LAMBDA_MIX,
                                "mixed_shape": list(mixed_x[0].shape),
                            }
                        )
            else:
                augmented_x = xb.to(device)
                augmented_y = yb.to(device)
                batch_audit = {
                    "number_of_mixed_samples_generated": 0,
                    "number_of_anchors_without_valid_partner": 0,
                    "same_cycle_violation_count": 0,
                    "different_block_violation_count": 0,
                    "self_pair_violation_count": 0,
                }
            logits = model(augmented_x)
            loss = criterion(logits, augmented_y)
            loss.backward()
            optimizer.step()
            optimization_count = int(len(augmented_y))
            mixed_count = int(
                batch_audit["number_of_mixed_samples_generated"]
            )
            total_loss_sum += float(loss.detach().cpu().item()) * optimization_count
            total_optimization_items += optimization_count
            real_targets = augmented_y[:real_count]
            real_correct += int(
                (logits[:real_count].argmax(dim=1) == real_targets)
                .sum()
                .detach()
                .cpu()
                .item()
            )
            real_seen += real_count
            if mixed_count:
                mixed_correct += int(
                    (
                        logits[real_count:].argmax(dim=1)
                        == augmented_y[real_count:]
                    )
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )
                mixed_seen += mixed_count
            number_of_mixed_samples += mixed_count
            anchors_without_partner += int(
                batch_audit["number_of_anchors_without_valid_partner"]
            )
            same_cycle_violations += int(
                batch_audit["same_cycle_violation_count"]
            )
            different_block_violations += int(
                batch_audit["different_block_violation_count"]
            )
            self_pair_violations += int(batch_audit["self_pair_violation_count"])
            number_of_batches += 1
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(
                    total_loss_sum / max(total_optimization_items, 1)
                ),
                "mean_training_loss": float(
                    total_loss_sum / max(total_optimization_items, 1)
                ),
                "train_accuracy": float(real_correct / max(real_seen, 1)),
                "final_train_accuracy_real": float(
                    real_correct / max(real_seen, 1)
                ),
                "mixed_train_accuracy": (
                    float(mixed_correct / mixed_seen) if mixed_seen else None
                ),
                "number_of_real_training_samples": int(real_seen),
                "number_of_mixed_samples_generated": int(
                    number_of_mixed_samples
                ),
                "mix_coverage": float(
                    number_of_mixed_samples / max(real_seen, 1)
                ),
                "number_of_anchors_without_valid_partner": int(
                    anchors_without_partner
                ),
                "same_cycle_violation_count": int(same_cycle_violations),
                "different_block_violation_count": int(
                    different_block_violations
                ),
                "self_pair_violation_count": int(self_pair_violations),
                "number_of_batches": int(number_of_batches),
                "n_train_items": int(real_seen),
                "n_optimization_items": int(total_optimization_items),
                "batch_size": int(batch_size),
                "lambda_mix": LAMBDA_MIX,
            }
        )
    return history, examples


def formal_architecture_config() -> dict[str, Any]:
    config = model_architecture_config(MODEL_NAME, n_classes=2, temporal_length=4)
    config.update(
        {
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "model_architecture_modified": False,
            "raw_and_cyclemix_share_architecture": True,
            "cyclemix_is_training_data_augmentation_only": True,
        }
    )
    return config


def train_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    *,
    block_names_train: Sequence[str],
    cycle_ids_train: Sequence[int],
    cycle_ids_test: Sequence[int],
    input_variant: str,
    session: str,
    fold: int,
    seed: int,
    train_cycles: str,
    test_cycles: str,
    training_config: DeepTrainingConfig,
    device: str | None = "auto",
    workers: int = 0,
) -> CycleMixFoldResult:
    train_cycle_array = np.asarray(cycle_ids_train, dtype=np.int64)
    test_cycle_array = np.asarray(cycle_ids_test, dtype=np.int64)
    if len(train_cycle_array) != len(X_train):
        raise ValueError("training cycle metadata does not align with X_train")
    if len(test_cycle_array) != len(X_test):
        raise ValueError("test cycle audit metadata does not align with X_test")
    train_cycle_set = tuple(sorted(int(value) for value in np.unique(train_cycle_array)))
    test_cycle_set = tuple(sorted(int(value) for value in np.unique(test_cycle_array)))
    overlap = sorted(set(train_cycle_set) & set(test_cycle_set))
    if overlap:
        raise AssertionError(f"outer train/test cycle overlap: {overlap}")
    block_ids = encode_block_identities(block_names_train)
    if len(block_ids) != len(X_train):
        raise ValueError("training block metadata does not align with X_train")
    partner_pool = build_same_block_cross_cycle_partner_pool(
        block_ids, train_cycle_array
    )
    if not set(partner_pool.cycle_ids.tolist()).issubset(set(train_cycle_set)):
        raise AssertionError("CycleMix partner pool includes a non-training cycle")

    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = (
        preprocess_train_fold_only(
            X_train,
            X_test,
            input_variant=input_variant,
            session=session,
            fold=fold,
            seed=seed,
            train_cycles=train_cycles,
            test_cycles=test_cycles,
        )
    )
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    y_train_i = labels_to_class_indices(y_train, classes)
    model = CNN2DTemporal1D(n_classes=len(classes), temporal_length=4).to(torch_device)
    parameters = count_trainable_parameters(model)
    history, examples = _train_epochs_with_cyclemix(
        model,
        train_tensor,
        y_train_i,
        partner_pool,
        input_variant=input_variant,
        config=training_config,
        seed=seed,
        fold=fold,
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
    total_real = int(sum(row["number_of_real_training_samples"] for row in history))
    total_mixed = int(
        sum(row["number_of_mixed_samples_generated"] for row in history)
    )
    mix_audit = {
        "session": str(session),
        "fold": int(fold),
        "seed": int(seed),
        "variant": input_variant,
        "train_cycle_ids": list(train_cycle_set),
        "test_cycle_ids": list(test_cycle_set),
        "train_test_cycle_overlap": False,
        "partner_pool_scope": "current_outer_training_samples_only",
        "test_samples_in_partner_pool": False,
        "test_block_identity_loaded_for_training": False,
        "test_mix_applied": False,
        "same_block_required": True,
        "different_cycle_required": True,
        "one_partner_per_anchor_maximum": True,
        "real_samples_retained": True,
        "hard_labels_used": True,
        "mixing_space": "after_train_fold_normalization",
        "lambda_mix": LAMBDA_MIX,
        "cyclemix_implementation_version": CYCLEMIX_IMPLEMENTATION_VERSION,
        "number_of_real_training_samples": total_real,
        "number_of_mixed_samples_generated": total_mixed,
        "mix_coverage": float(total_mixed / max(total_real, 1)),
        "number_of_anchors_without_valid_partner": int(
            sum(
                row["number_of_anchors_without_valid_partner"]
                for row in history
            )
        ),
        "same_cycle_violation_count": int(
            sum(row["same_cycle_violation_count"] for row in history)
        ),
        "different_block_violation_count": int(
            sum(row["different_block_violation_count"] for row in history)
        ),
        "self_pair_violation_count": int(
            sum(row["self_pair_violation_count"] for row in history)
        ),
        "partner_examples": examples,
    }
    model_config = formal_architecture_config()
    model_config["parameter_count"] = int(parameters)
    return CycleMixFoldResult(
        method=input_variant,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=int(parameters),
        history=history,
        normalization_audit=norm_audit,
        mix_audit=mix_audit,
        final_training_loss=float(history[-1]["train_loss"]),
        final_trained_epochs=len(history),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_config,
    )
