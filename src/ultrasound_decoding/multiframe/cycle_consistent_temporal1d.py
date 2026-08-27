from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
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
CYCLE_CONSISTENT_VARIANT = "cycle_consistent"
INPUT_VARIANTS = (RAW_VARIANT, CYCLE_CONSISTENT_VARIANT)
MODEL_NAME = "cnn2d_temporal1d"
MODEL_IMPLEMENTATION_VERSION = "cycle_consistent_temporal1d_v1.0.0"
CONSISTENCY_IMPLEMENTATION_VERSION = "same_block_cross_cycle_cosine_v1"
LAMBDA_CONSISTENCY = 0.1
BLOCK_NAMES = ("grating", "stop_after_grating", "dot", "static")
BLOCK_TO_ID = {name: index for index, name in enumerate(BLOCK_NAMES)}


@dataclass
class CycleConsistencyFoldResult:
    method: str
    seed: int
    predictions: np.ndarray
    probabilities: np.ndarray
    model: nn.Module
    model_parameters: int
    history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    pair_audit: dict[str, Any]
    representation_audit: dict[str, Any]
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


def build_positive_pair_mask(
    block_ids: torch.Tensor, cycle_ids: torch.Tensor
) -> torch.Tensor:
    """Return one upper-triangle entry per same-block, cross-cycle pair."""

    if block_ids.ndim != 1 or cycle_ids.ndim != 1:
        raise ValueError("block and cycle ids must be one-dimensional")
    if len(block_ids) != len(cycle_ids):
        raise ValueError("block and cycle ids differ in length")
    same_block = block_ids[:, None].eq(block_ids[None, :])
    different_cycle = cycle_ids[:, None].ne(cycle_ids[None, :])
    upper_triangle = torch.triu(
        torch.ones_like(same_block, dtype=torch.bool), diagonal=1
    )
    return same_block & different_cycle & upper_triangle


def assert_pair_cycles_are_training_only(
    cycle_ids: torch.Tensor,
    valid_pair_mask: torch.Tensor,
    allowed_train_cycle_ids: Sequence[int],
) -> None:
    allowed = {int(value) for value in allowed_train_cycle_ids}
    if not allowed:
        raise ValueError("allowed training cycle set is empty")
    if valid_pair_mask.shape != (len(cycle_ids), len(cycle_ids)):
        raise ValueError("pair mask shape differs from batch cycle ids")
    pair_indices = valid_pair_mask.nonzero(as_tuple=False)
    if len(pair_indices) == 0:
        observed = {int(value) for value in cycle_ids.detach().cpu().tolist()}
    else:
        observed = {
            int(value)
            for value in cycle_ids[pair_indices.reshape(-1)].detach().cpu().tolist()
        }
    if not observed.issubset(allowed):
        raise AssertionError(
            f"consistency pair uses non-training cycles: {sorted(observed-allowed)}"
        )


def cycle_consistency_loss(
    embedding: torch.Tensor,
    block_ids: torch.Tensor,
    cycle_ids: torch.Tensor,
    *,
    allowed_train_cycle_ids: Sequence[int] | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Mean 1-cosine loss over unique same-block, cross-cycle batch pairs."""

    if embedding.ndim != 2 or len(embedding) != len(block_ids):
        raise ValueError("embedding must be [B,D] and align with metadata")
    mask = build_positive_pair_mask(block_ids, cycle_ids)
    if allowed_train_cycle_ids is not None:
        assert_pair_cycles_are_training_only(
            cycle_ids, mask, allowed_train_cycle_ids
        )
    pair_indices = mask.nonzero(as_tuple=False)
    pair_count = int(len(pair_indices))
    if pair_count == 0:
        return embedding.sum() * 0.0, 0, mask
    normalized = F.normalize(embedding, p=2, dim=1)
    cosine = (normalized[pair_indices[:, 0]] * normalized[pair_indices[:, 1]]).sum(
        dim=1
    )
    return (1.0 - cosine).mean(), pair_count, mask


def total_training_loss(
    classification_loss: torch.Tensor,
    consistency_loss: torch.Tensor,
    *,
    lambda_consistency: float,
) -> torch.Tensor:
    if float(lambda_consistency) == 0.0:
        return classification_loss
    return classification_loss + float(lambda_consistency) * consistency_loss


def _train_epochs_with_cycle_consistency(
    model: CNN2DTemporal1D,
    train_tensor: torch.Tensor,
    y_train_i: np.ndarray,
    block_ids: np.ndarray,
    cycle_ids: np.ndarray,
    *,
    allowed_train_cycle_ids: Sequence[int],
    config: DeepTrainingConfig,
    seed: int,
    device: torch.device,
    batch_size_reference: int,
    lambda_consistency: float,
    num_workers: int = 0,
) -> list[dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, config)
    batch_size = max(1, min(int(config.batch_size), int(batch_size_reference)))
    dataset = TensorDataset(
        train_tensor,
        torch.from_numpy(y_train_i),
        torch.from_numpy(block_ids.astype(np.int64, copy=False)),
        torch.from_numpy(cycle_ids.astype(np.int64, copy=False)),
    )
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
        classification_loss_sum = 0.0
        consistency_loss_sum = 0.0
        total_loss_sum = 0.0
        total_correct = 0
        total_seen = 0
        number_of_batches = 0
        batches_with_valid_pairs = 0
        total_valid_positive_pairs = 0
        for xb, yb, block_batch, cycle_batch in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            block_batch = block_batch.to(device)
            cycle_batch = cycle_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            if float(lambda_consistency) == 0.0:
                logits = model(xb)
                consistency = logits.sum() * 0.0
                pair_mask = build_positive_pair_mask(block_batch, cycle_batch)
                assert_pair_cycles_are_training_only(
                    cycle_batch, pair_mask, allowed_train_cycle_ids
                )
                pair_count = int(pair_mask.sum().detach().cpu().item())
            else:
                logits, embedding = model.forward_with_embedding(xb)
                consistency, pair_count, _pair_mask = cycle_consistency_loss(
                    embedding,
                    block_batch,
                    cycle_batch,
                    allowed_train_cycle_ids=allowed_train_cycle_ids,
                )
            classification = criterion(logits, yb)
            loss = total_training_loss(
                classification,
                consistency,
                lambda_consistency=lambda_consistency,
            )
            loss.backward()
            optimizer.step()
            n = int(len(yb))
            classification_loss_sum += float(classification.detach().cpu().item()) * n
            consistency_loss_sum += float(consistency.detach().cpu().item()) * n
            total_loss_sum += float(loss.detach().cpu().item()) * n
            total_correct += int(
                (logits.argmax(dim=1) == yb).sum().detach().cpu().item()
            )
            total_seen += n
            number_of_batches += 1
            batches_with_valid_pairs += int(pair_count > 0)
            total_valid_positive_pairs += pair_count
        denominator = max(total_seen, 1)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(total_loss_sum / denominator),
                "classification_loss": float(classification_loss_sum / denominator),
                "consistency_loss": float(consistency_loss_sum / denominator),
                "total_loss": float(total_loss_sum / denominator),
                "train_accuracy": float(total_correct / denominator),
                "n_train_items": int(total_seen),
                "batch_size": int(batch_size),
                "number_of_batches": int(number_of_batches),
                "batches_with_valid_pairs": int(batches_with_valid_pairs),
                "valid_pair_fraction_of_batches": float(
                    batches_with_valid_pairs / max(number_of_batches, 1)
                ),
                "total_valid_positive_pairs": int(total_valid_positive_pairs),
                "mean_valid_pairs_per_batch": float(
                    total_valid_positive_pairs / max(number_of_batches, 1)
                ),
                "lambda_consistency": float(lambda_consistency),
            }
        )
    return history


def _training_representation_audit(
    model: CNN2DTemporal1D,
    train_tensor: torch.Tensor,
    block_ids: np.ndarray,
    cycle_ids: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    allowed_train_cycle_ids: Sequence[int],
) -> dict[str, Any]:
    embeddings: list[torch.Tensor] = []
    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
    )
    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            _logits, embedding = model.forward_with_embedding(xb.to(device))
            embeddings.append(embedding.detach().cpu())
    all_embeddings = torch.cat(embeddings, dim=0)
    block_tensor = torch.from_numpy(block_ids.astype(np.int64, copy=False))
    cycle_tensor = torch.from_numpy(cycle_ids.astype(np.int64, copy=False))
    loss, pair_count, _mask = cycle_consistency_loss(
        all_embeddings,
        block_tensor,
        cycle_tensor,
        allowed_train_cycle_ids=allowed_train_cycle_ids,
    )
    return {
        "diagnostic_only": True,
        "used_for_model_selection": False,
        "data_scope": "normalized_outer_training_samples_only",
        "test_embeddings_computed": False,
        "same_block_cross_cycle_pair_count": int(pair_count),
        "same_block_cross_cycle_mean_cosine_similarity": (
            float(1.0 - loss.item()) if pair_count else None
        ),
        "embedding_dimension": int(all_embeddings.shape[1]),
    }


def formal_architecture_config() -> dict[str, Any]:
    config = model_architecture_config(MODEL_NAME, n_classes=2, temporal_length=4)
    config.update(
        {
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "classifier_input_embedding_dimension": 64,
            "projection_head_added": False,
            "raw_and_cycle_consistent_share_architecture": True,
            "consistency_is_training_loss_only": True,
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
) -> CycleConsistencyFoldResult:
    if input_variant not in INPUT_VARIANTS:
        raise ValueError(f"unknown input variant: {input_variant!r}")
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

    lambda_consistency = (
        LAMBDA_CONSISTENCY if input_variant == CYCLE_CONSISTENT_VARIANT else 0.0
    )
    torch_device = resolve_device(device)
    set_reproducible_seed(seed)
    X_train_norm, X_test_norm, norm_audit, norm_mean, norm_std = (
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
    norm_audit.update(
        {
            "input_variant": input_variant,
            "preprocessing_order": "clean4 -> arcsinh -> train_fold_pixel_zscore",
            "consistency_regularization_scope": "training_loss_only",
            "test_block_identity_used": False,
            "test_cycle_metadata_used_for_model_input": False,
            "test_cycle_metadata_used_only_for_disjointness_assertion": True,
        }
    )
    y_train_i = labels_to_class_indices(y_train, classes)
    train_tensor = blocks_to_sequence_tensor(X_train_norm)
    test_tensor = blocks_to_sequence_tensor(X_test_norm)
    model = CNN2DTemporal1D(n_classes=len(classes), temporal_length=4).to(torch_device)
    parameters = count_trainable_parameters(model)
    history = _train_epochs_with_cycle_consistency(
        model,
        train_tensor,
        y_train_i,
        block_ids,
        train_cycle_array,
        allowed_train_cycle_ids=train_cycle_set,
        config=training_config,
        seed=seed,
        device=torch_device,
        batch_size_reference=len(X_train),
        lambda_consistency=lambda_consistency,
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
    pair_audit = {
        "session": str(session),
        "fold": int(fold),
        "seed": int(seed),
        "variant": input_variant,
        "train_cycle_ids": list(train_cycle_set),
        "test_cycle_ids": list(test_cycle_set),
        "train_test_cycle_overlap": False,
        "pair_pool_scope": "current_outer_training_samples_only",
        "same_block_required": True,
        "different_cycle_required": True,
        "upper_triangle_unique_pairs": True,
        "negatives_used": False,
        "test_samples_in_pair_pool": False,
        "test_block_identity_loaded_for_training": False,
        "consistency_implementation_version": CONSISTENCY_IMPLEMENTATION_VERSION,
        "lambda_consistency": float(lambda_consistency),
        "number_of_batches": int(sum(row["number_of_batches"] for row in history)),
        "batches_with_valid_pairs": int(
            sum(row["batches_with_valid_pairs"] for row in history)
        ),
        "total_valid_positive_pairs": int(
            sum(row["total_valid_positive_pairs"] for row in history)
        ),
    }
    pair_audit["valid_pair_fraction_of_batches"] = float(
        pair_audit["batches_with_valid_pairs"]
        / max(pair_audit["number_of_batches"], 1)
    )
    pair_audit["mean_valid_pairs_per_batch"] = float(
        pair_audit["total_valid_positive_pairs"]
        / max(pair_audit["number_of_batches"], 1)
    )
    representation_audit = _training_representation_audit(
        model,
        train_tensor,
        block_ids,
        train_cycle_array,
        device=torch_device,
        batch_size=training_config.batch_size,
        allowed_train_cycle_ids=train_cycle_set,
    )
    model_config = formal_architecture_config()
    model_config["parameter_count"] = int(parameters)
    return CycleConsistencyFoldResult(
        method=input_variant,
        seed=int(seed),
        predictions=predictions,
        probabilities=probabilities,
        model=model,
        model_parameters=int(parameters),
        history=history,
        normalization_audit=norm_audit,
        pair_audit=pair_audit,
        representation_audit=representation_audit,
        final_training_loss=float(history[-1]["total_loss"]),
        final_trained_epochs=len(history),
        device=str(torch_device),
        X_test_normalized=X_test_norm,
        normalization_mean=norm_mean,
        normalization_std=norm_std,
        normalization_transform=norm_audit["transform"],
        input_shape=tuple(int(value) for value in X_train.shape[1:]),
        model_config=model_config,
    )
