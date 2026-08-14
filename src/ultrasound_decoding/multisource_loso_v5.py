"""Strict source-only clean4 SmallCNN training for multi-source LOSO v5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    BlockSequenceData,
    cycle_text,
)
from ultrasound_decoding.multiframe.models import CNN2DMeanPool
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


V5_SEEDS = (20260812, 20260813, 20260814)
V5_CONDITIONS = (
    "SINGLE_SOURCE_TRANSFER",
    "MULTI_SOURCE_BALANCED",
    "NATURAL_FREQUENCY_MULTI_SOURCE",
)
MULTI_SOURCE_CONDITIONS = V5_CONDITIONS[1:]
BALANCE_MODES = ("session_balanced", "natural_frequency")
FROZEN_SUPERVISED_CONFIG = DeepTrainingConfig(
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=16,
    max_epochs=40,
    dropout=0.25,
    loss="cross_entropy",
)

REQUIRED_FORMAL_OUTPUTS = (
    "audit/smallcnn_identity_check.md",
    "audit/target_holdout_leakage.csv",
    "audit/source_sampling_distribution.csv",
    "audit/multisource_training_volume.csv",
    "audit/historical_single_source_reuse.csv",
    "audit/within_session_reference_reuse.csv",
    "audit/config_freeze.md",
    "audit/gpu_audit.txt",
    "downstream/fold_metrics.csv",
    "downstream/target_predictions.csv",
    "summaries/target_level_comparison.csv",
    "summaries/planned_statistical_tests.csv",
    "summaries/within_cross_gap.csv",
    "summaries/seed_stability.csv",
    "figures/binary_target_level_cross_session.png",
    "figures/stimulus_type_target_level_cross_session.png",
    "figures/binary_multi_minus_single_delta.png",
    "figures/stimulus_type_multi_minus_single_delta.png",
    "figures/within_vs_cross_session_gap.png",
    "figures/source_sampling_distribution.png",
    "report/multisource_loso_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass
class PreparedCrossSessionData:
    task: str
    source_sessions: tuple[str, ...]
    target_session: str
    balance_mode: str
    X_train: np.ndarray
    y_train: np.ndarray
    train_session_labels: np.ndarray
    train_composite_groups: np.ndarray
    train_sample_ids: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_cycles: np.ndarray
    test_sample_ids: np.ndarray
    normalization_audit: dict[str, Any]
    source_cycle_counts: dict[str, int]
    source_block_counts: dict[str, int]


@dataclass
class CrossSessionResult:
    model: CNN2DMeanPool
    train_predictions: np.ndarray
    train_probabilities: np.ndarray
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    history: list[dict[str, Any]]
    sampling_history: list[dict[str, Any]]
    normalization_audit: dict[str, Any]
    metrics: dict[str, Any]
    device: str


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


def source_sessions_for_target(
    target_session: str, sessions: Iterable[str] = EXPECTED_SESSIONS
) -> tuple[str, ...]:
    values = tuple(str(value) for value in sessions)
    target = str(target_session)
    if len(values) != 9 or set(values) != set(EXPECTED_SESSIONS):
        raise ValueError("formal LOSO requires exactly the nine frozen sessions")
    if target not in values:
        raise ValueError(f"unknown target session: {target}")
    sources = tuple(value for value in values if value != target)
    if len(sources) != 8 or target in sources:
        raise AssertionError("LOSO must contain exactly eight non-target sources")
    return sources


def validate_source_target_data(
    data_by_session: Mapping[str, BlockSequenceData],
    source_sessions: Iterable[str],
    target_session: str,
) -> tuple[str, ...]:
    sources = tuple(str(value) for value in source_sessions)
    target = str(target_session)
    if target in sources:
        raise AssertionError("target session entered the source training pool")
    if len(set(sources)) != len(sources) or not sources:
        raise ValueError("source sessions must be unique and nonempty")
    if (set(sources) | {target}) - set(map(str, data_by_session)):
        raise ValueError("source or target session data are missing")
    tasks = {data_by_session[value].task for value in (*sources, target)}
    if len(tasks) != 1:
        raise AssertionError("source and target tasks differ")
    for session in (*sources, target):
        data = data_by_session[session]
        if tuple(data.X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
            raise AssertionError(f"{session}: input is not clean4 {EXPECTED_BLOCK_SHAPE}")
        if set(np.unique(data.y).astype(int).tolist()) != {0, 1}:
            raise AssertionError(f"{session}: expected two frozen task classes")
    return sources


def _source_only_normalizer(
    X_train: np.ndarray,
    train_session_labels: np.ndarray,
    *,
    balance_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if balance_mode not in BALANCE_MODES:
        raise ValueError(f"unknown source balance mode: {balance_mode}")
    transformed = np.arcsinh(X_train.astype(np.float32, copy=False))
    labels = train_session_labels.astype(str)
    sessions = sorted(np.unique(labels).tolist(), key=int)
    if len(sessions) < 1:
        raise ValueError("source-only normalization received no sessions")
    # Keep the historical clean4 preprocessing exact in both formal branches.
    # Only the supervised draw distribution changes between the primary and
    # natural-frequency control, so their difference isolates the sampler.
    frames = transformed.reshape(-1, transformed.shape[-2], transformed.shape[-1]).astype(
        np.float64, copy=False
    )
    mean = frames.mean(axis=0, keepdims=True)
    second = np.square(frames).mean(axis=0, keepdims=True)
    weights = {session: float(np.mean(labels == session)) for session in sessions}
    weighting = "sample_frequency_weighted_source_only"
    variance = np.maximum(second - np.square(mean), 0.0)
    std_raw = np.sqrt(variance)
    std = std_raw + 1e-6
    audit = {
        "transform": "arcsinh_then_source_pixel_zscore",
        "statistics_scope": "source_sessions_only_all_clean4_frames",
        "normalization_weighting": weighting,
        "source_session_weights": weights,
        "fit_sessions": sessions,
        "target_used_for_stats": False,
        "target_unlabeled_used_for_stats": False,
        "epsilon": 1e-6,
        "mean_mean": float(mean.mean()),
        "std_mean": float(std_raw.mean()),
        "zero_variance_pixels": int(np.sum(std_raw == 0)),
    }
    return mean.astype(np.float32), std.astype(np.float32), audit


def prepare_cross_session_data(
    data_by_session: Mapping[str, BlockSequenceData],
    *,
    source_sessions: Iterable[str],
    target_session: str,
    balance_mode: str,
) -> PreparedCrossSessionData:
    sources = validate_source_target_data(data_by_session, source_sessions, target_session)
    target = str(target_session)
    source_arrays = [data_by_session[session].X for session in sources]
    X_train_raw = np.concatenate(source_arrays, axis=0).astype(np.float32, copy=False)
    y_train = np.concatenate([data_by_session[session].y for session in sources]).astype(np.int64)
    session_labels = np.concatenate([
        np.full(data_by_session[session].n_blocks, session, dtype=object) for session in sources
    ]).astype(str)
    composite_groups = np.concatenate([
        np.asarray([f"{session}_cycle{int(cycle)}" for cycle in data_by_session[session].groups], dtype=object)
        for session in sources
    ])
    train_ids = np.concatenate([
        data_by_session[session].metadata["block_id"].astype(str).to_numpy() for session in sources
    ])
    target_data = data_by_session[target]
    if bool(np.any(session_labels == target)):
        raise AssertionError("target session label entered training metadata")
    mean, std, audit = _source_only_normalizer(
        X_train_raw, session_labels, balance_mode=balance_mode
    )
    X_train = (np.arcsinh(X_train_raw) - mean) / std
    X_test = (np.arcsinh(target_data.X.astype(np.float32, copy=False)) - mean) / std
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("non-finite source-only normalized clean4 data")
    audit.update({
        "source_sessions": list(sources),
        "target_session": target,
        "n_source_blocks": int(len(X_train)),
        "n_target_blocks_transformed_only": int(len(X_test)),
        "target_labels_used_for_fit": False,
        "target_frames_used_for_fit": 0,
    })
    return PreparedCrossSessionData(
        task=target_data.task,
        source_sessions=sources,
        target_session=target,
        balance_mode=balance_mode,
        X_train=X_train.astype(np.float32, copy=False),
        y_train=y_train,
        train_session_labels=session_labels,
        train_composite_groups=composite_groups,
        train_sample_ids=train_ids.astype(str),
        X_test=X_test.astype(np.float32, copy=False),
        y_test=target_data.y.astype(np.int64, copy=False),
        test_cycles=target_data.groups.astype(np.int64, copy=False),
        test_sample_ids=target_data.metadata["block_id"].astype(str).to_numpy(),
        normalization_audit=audit,
        source_cycle_counts={session: data_by_session[session].n_cycles for session in sources},
        source_block_counts={session: data_by_session[session].n_blocks for session in sources},
    )


def epoch_draw_indices(
    session_labels: np.ndarray,
    *,
    seed: int,
    epoch: int,
    balance_mode: str,
) -> np.ndarray:
    labels = np.asarray(session_labels).astype(str)
    if not len(labels):
        raise ValueError("cannot sample an empty source pool")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch), 5]))
    if balance_mode == "natural_frequency":
        return rng.permutation(len(labels)).astype(np.int64)
    if balance_mode != "session_balanced":
        raise ValueError(f"unknown source balance mode: {balance_mode}")
    sessions = sorted(np.unique(labels).tolist(), key=int)
    if len(sessions) < 2:
        raise ValueError("session-balanced sampling requires at least two sources")
    total = len(labels)
    base, remainder = divmod(total, len(sessions))
    extra = set(rng.choice(sessions, size=remainder, replace=False).tolist()) if remainder else set()
    drawn: list[int] = []
    for session in sessions:
        required = base + int(session in extra)
        available = np.flatnonzero(labels == session)
        while required > 0:
            permutation = rng.permutation(available)
            take = min(required, len(permutation))
            drawn.extend(permutation[:take].astype(int).tolist())
            required -= take
    output = rng.permutation(np.asarray(drawn, dtype=np.int64))
    if len(output) != total:
        raise AssertionError("session-balanced sampler changed samples per epoch")
    return output


def sampling_distribution_rows(
    indices: np.ndarray,
    session_labels: np.ndarray,
    *,
    target_session: str,
    task: str,
    condition: str,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    labels = np.asarray(session_labels).astype(str)
    drawn = np.asarray(indices, dtype=np.int64)
    sessions = sorted(np.unique(labels).tolist(), key=int)
    rows = []
    for session in sessions:
        source_indices = drawn[labels[drawn] == session]
        available = int(np.sum(labels == session))
        rows.append({
            "target_session": str(target_session),
            "task": task,
            "condition": condition,
            "seed": int(seed),
            "epoch": int(epoch),
            "source_session": session,
            "n_draws": int(len(source_indices)),
            "draw_proportion": float(len(source_indices) / len(drawn)),
            "expected_uniform_proportion": float(1.0 / len(sessions)),
            "absolute_uniform_deviation": float(abs(len(source_indices) / len(drawn) - 1.0 / len(sessions))),
            "n_available_blocks": available,
            "n_unique_blocks_drawn": int(len(np.unique(source_indices))),
            "with_replacement": bool(len(source_indices) > available),
        })
    return rows


def train_prepared_cross_session(
    prepared: PreparedCrossSessionData,
    *,
    condition: str,
    seed: int,
    balance_mode: str,
    config: DeepTrainingConfig = FROZEN_SUPERVISED_CONFIG,
    device: str | None = "auto",
) -> CrossSessionResult:
    if condition not in V5_CONDITIONS:
        raise ValueError(f"unknown v5 condition: {condition}")
    if condition == "MULTI_SOURCE_BALANCED" and balance_mode != "session_balanced":
        raise ValueError("MULTI_SOURCE_BALANCED requires session-balanced sampling")
    if condition != "MULTI_SOURCE_BALANCED" and balance_mode != "natural_frequency":
        raise ValueError(f"{condition} requires natural-frequency sampling")
    if condition == "SINGLE_SOURCE_TRANSFER" and len(prepared.source_sessions) != 1:
        raise ValueError("single-source transfer requires exactly one source")
    if condition in MULTI_SOURCE_CONDITIONS and len(prepared.source_sessions) < 2:
        raise ValueError("multi-source condition requires at least two sources")
    if prepared.balance_mode != balance_mode:
        raise ValueError("prepared normalization mode and training sampling mode differ")
    if config != FROZEN_SUPERVISED_CONFIG and config.max_epochs > FROZEN_SUPERVISED_CONFIG.max_epochs:
        raise ValueError("training config exceeds the frozen supervised budget")

    set_reproducible_seed(seed)
    classes = np.asarray(sorted(TASK_CLASS_NAMES[prepared.task]), dtype=np.int64)
    y_train_i = labels_to_class_indices(prepared.y_train, classes)
    train_tensor = blocks_to_sequence_tensor(prepared.X_train)
    test_tensor = blocks_to_sequence_tensor(prepared.X_test)
    torch_device = resolve_device(device)
    model = CNN2DMeanPool(
        n_classes=len(classes), dropout=config.dropout, temporal_length=4
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, Any]] = []
    sampling_history: list[dict[str, Any]] = []
    batch_size = max(1, min(int(config.batch_size), len(train_tensor)))
    for epoch in range(1, int(config.max_epochs) + 1):
        indices = epoch_draw_indices(
            prepared.train_session_labels, seed=seed, epoch=epoch, balance_mode=balance_mode
        )
        sampling_history.extend(sampling_distribution_rows(
            indices,
            prepared.train_session_labels,
            target_session=prepared.target_session,
            task=prepared.task,
            condition=condition,
            seed=seed,
            epoch=epoch,
        ))
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            xb = train_tensor[batch].to(torch_device)
            yb = torch.from_numpy(y_train_i[batch]).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            n = len(batch)
            total_loss += float(loss.detach().cpu()) * n
            total_correct += int((logits.argmax(1) == yb).sum().detach().cpu())
            total_seen += n
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "train_accuracy_minibatch": total_correct / max(total_seen, 1),
            "n_draws": int(total_seen),
            "n_unique_source_blocks": int(len(np.unique(indices))),
            "source_balance_mode": balance_mode,
        })
    train_probs = predict_probabilities(model, train_tensor, device=torch_device, batch_size=config.batch_size)
    test_probs = predict_probabilities(model, test_tensor, device=torch_device, batch_size=config.batch_size)
    train_pred = classes[train_probs.argmax(axis=1)]
    test_pred = classes[test_probs.argmax(axis=1)]
    train_metrics = classification_metrics(prepared.y_train, train_pred)
    test_metrics = classification_metrics(prepared.y_test, test_pred)
    metrics = {
        "task": prepared.task,
        "target_session": prepared.target_session,
        "source_sessions": ",".join(prepared.source_sessions),
        "n_source_sessions": int(len(prepared.source_sessions)),
        "seed": int(seed),
        "condition": condition,
        "source_balance_mode": balance_mode,
        "train_accuracy": float(train_metrics["accuracy"]),
        "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "macro_F1": float(test_metrics["macro_f1"]),
        "ROC_AUC": binary_roc_auc(prepared.y_test, test_probs[:, 1]),
        "best_epoch": int(config.max_epochs),
        "train_test_gap_BA": float(train_metrics["balanced_accuracy"] - test_metrics["balanced_accuracy"]),
        "n_train_blocks": int(len(prepared.y_train)),
        "n_test_blocks": int(len(prepared.y_test)),
        "n_train_frames": int(4 * len(prepared.y_train)),
        "n_test_frames": int(4 * len(prepared.y_test)),
        "source_cycle_counts": ";".join(
            f"{session}:{prepared.source_cycle_counts[session]}" for session in prepared.source_sessions
        ),
        "test_cycles": cycle_text(prepared.test_cycles),
        "target_labels_used_for_training": False,
        "target_frames_used_for_training": 0,
        "target_used_for_normalization": False,
        "target_used_for_validation": False,
        "target_used_for_model_selection": False,
        "early_stopping": False,
        "run_status": "VALID",
    }
    return CrossSessionResult(
        model=model.cpu(),
        train_predictions=train_pred,
        train_probabilities=train_probs,
        test_predictions=test_pred,
        test_probabilities=test_probs,
        history=history,
        sampling_history=sampling_history,
        normalization_audit=dict(prepared.normalization_audit),
        metrics=metrics,
        device=str(torch_device),
    )


def assert_formal_cuda(device: str) -> torch.device:
    if str(device) != "cuda":
        raise RuntimeError("formal v5 requires --device cuda; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("formal v5 requires an available CUDA device; STOP")
    return torch.device("cuda")


def missing_formal_outputs(output_dir: Path) -> list[str]:
    missing = []
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = output_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative)
    curve_dir = output_dir / "downstream/training_curves"
    if not curve_dir.is_dir() or not any(curve_dir.glob("*.csv")):
        missing.append("downstream/training_curves/*.csv")
    return missing
