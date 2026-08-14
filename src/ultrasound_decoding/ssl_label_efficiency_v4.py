"""Frozen within-session masked-SSL label-efficiency utilities for clean4."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ultrasound_decoding.multiframe.dataset import BlockSequenceData, cycle_text
from ultrasound_decoding.multiframe.models import SmallCNNFrameEncoder
from ultrasound_decoding.multiframe.training import DeepTrainingConfig
from ultrasound_decoding.ssl_masked import (
    SSL_SEEDS,
    SSLPretrainingConfig,
    fixed_ssl_validation_cycles,
    load_ssl_encoder_checkpoint,
)


V4_SEEDS = SSL_SEEDS
LABEL_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
LOW_LABEL_FRACTIONS = (0.2, 0.4)
V4_CONDITIONS = ("RANDOM_INIT", "WITHIN_MASKED_SSL_FT")
V1_CONDITION_MAP = {
    "RANDOM_INIT": "RANDOM_INIT",
    "SSL_FINETUNE": "WITHIN_MASKED_SSL_FT",
}
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")

FROZEN_SUPERVISED_CONFIG = DeepTrainingConfig(
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=16,
    max_epochs=40,
    dropout=0.25,
    loss="cross_entropy",
)
FROZEN_SSL_CONFIG = SSLPretrainingConfig()

REQUIRED_FORMAL_OUTPUTS = (
    "audit/fold_identity_check.csv",
    "audit/label_fraction_cycle_counts.csv",
    "audit/nested_label_subsets.csv",
    "audit/label_class_balance.csv",
    "audit/test_cycle_leakage.csv",
    "audit/ssl_checkpoint_reuse.csv",
    "audit/v1_checkpoint_reuse.csv",
    "audit/full_label_v1_reproduction.csv",
    "audit/config_freeze.md",
    "audit/gpu_audit.txt",
    "downstream/fold_metrics.csv",
    "downstream/predictions.csv",
    "summaries/session_label_efficiency.csv",
    "summaries/label_efficiency_AULC.csv",
    "summaries/low_label_summary.csv",
    "summaries/planned_statistical_tests.csv",
    "summaries/label_fraction_to_target.csv",
    "summaries/seed_stability.csv",
    "figures/binary_label_efficiency_mean_curve.png",
    "figures/binary_label_efficiency_by_session.png",
    "figures/binary_ssl_advantage_by_fraction.png",
    "figures/binary_train_test_gap_by_fraction.png",
    "figures/weak_sessions_low_label.png",
    "figures/stimulus_type_label_efficiency_mean_curve.png",
    "report/label_efficiency_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


def round_half_up(value: float | Decimal) -> int:
    """Round a non-negative value with halves away from zero."""
    decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    if decimal_value < 0:
        raise ValueError("label-budget values must be non-negative")
    return int(decimal_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def n_label_cycles(label_fraction: float, n_train_cycles: int) -> int:
    fraction = float(label_fraction)
    if fraction not in LABEL_FRACTIONS:
        raise ValueError(f"label fraction must be one of {LABEL_FRACTIONS}")
    if int(n_train_cycles) < 1:
        raise ValueError("at least one outer training cycle is required")
    if fraction == 1.0:
        return int(n_train_cycles)
    return min(int(n_train_cycles), max(1, round_half_up(Decimal(str(fraction)) * int(n_train_cycles))))


def deterministic_cycle_permutation(
    train_cycles: Iterable[int], *, session: str, fold: int, seed: int
) -> np.ndarray:
    cycles = np.asarray(sorted({int(value) for value in train_cycles}), dtype=np.int64)
    if not len(cycles):
        raise ValueError("cannot permute an empty outer training set")
    sequence = np.random.SeedSequence([int(seed), int(session), int(fold), 4])
    return np.random.default_rng(sequence).permutation(cycles)


def nested_label_subsets(
    train_cycles: Iterable[int], *, session: str, fold: int, seed: int
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    """Return one deterministic cycle ordering and its five nested prefixes."""
    permutation = deterministic_cycle_permutation(
        train_cycles, session=session, fold=fold, seed=seed
    )
    subsets: dict[float, np.ndarray] = {}
    for fraction in LABEL_FRACTIONS:
        count = n_label_cycles(fraction, len(permutation))
        subsets[fraction] = np.sort(permutation[:count]).astype(np.int64, copy=False)
    validate_nested_subsets(permutation, subsets)
    return permutation, subsets


def validate_nested_subsets(
    permutation: np.ndarray, subsets: dict[float, np.ndarray]
) -> None:
    expected = set(LABEL_FRACTIONS)
    if set(subsets) != expected:
        raise AssertionError("nested subsets do not contain the five frozen fractions")
    outer_train = set(np.asarray(permutation, dtype=np.int64).tolist())
    previous: set[int] = set()
    for fraction in LABEL_FRACTIONS:
        current = set(np.asarray(subsets[fraction], dtype=np.int64).tolist())
        if not previous.issubset(current):
            raise AssertionError("label subsets are not nested")
        if not current.issubset(outer_train):
            raise AssertionError("label subset contains a non-training cycle")
        if len(current) != n_label_cycles(fraction, len(outer_train)):
            raise AssertionError("label subset has the wrong cycle count")
        previous = current
    if previous != outer_train:
        raise AssertionError("100% labels do not use every outer training cycle")


def label_fraction_rows(
    train_cycles: Iterable[int], *, session: str, fold: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    permutation, subsets = nested_label_subsets(
        train_cycles, session=session, fold=fold, seed=seed
    )
    count_rows: list[dict[str, Any]] = []
    subset_rows: list[dict[str, Any]] = []
    previous_count: int | None = None
    for fraction in LABEL_FRACTIONS:
        subset = subsets[fraction]
        count = int(len(subset))
        duplicate = previous_count == count
        common = {
            "session": str(session),
            "fold": int(fold),
            "seed": int(seed),
            "label_fraction": float(fraction),
            "n_train_cycles": int(len(permutation)),
            "n_label_cycles": count,
            "duplicate_cycle_count_with_previous_fraction": bool(duplicate),
        }
        count_rows.append(dict(common))
        subset_rows.append({
            **common,
            "permuted_train_cycle_ids": ordered_cycle_text(permutation),
            "labeled_cycle_ids": cycle_text(subset.tolist()),
            "subset_of_outer_train": bool(set(subset.tolist()).issubset(set(permutation.tolist()))),
            "nested_with_previous_fraction": True,
            "status": "PASS",
        })
        previous_count = count
    return count_rows, subset_rows


def labeled_sample_indices(data: BlockSequenceData, labeled_cycles: Iterable[int]) -> np.ndarray:
    """Select whole cycles; partial-cycle selections are a hard failure."""
    cycles = np.asarray(sorted({int(value) for value in labeled_cycles}), dtype=np.int64)
    indices = np.flatnonzero(np.isin(data.groups, cycles))
    selected_cycles = set(data.groups[indices].astype(int).tolist())
    if selected_cycles != set(cycles.tolist()):
        raise AssertionError("one or more requested labeled cycles have no samples")
    for cycle in cycles:
        all_cycle_indices = np.flatnonzero(data.groups == cycle)
        if not np.array_equal(all_cycle_indices, indices[np.isin(indices, all_cycle_indices)]):
            raise AssertionError("cycle-level selection split a cycle")
    return indices.astype(np.int64, copy=False)


def label_class_balance_row(
    data: BlockSequenceData,
    *,
    fold: int,
    seed: int,
    label_fraction: float,
    labeled_cycles: Iterable[int],
) -> dict[str, Any]:
    indices = labeled_sample_indices(data, labeled_cycles)
    counts = {int(label): int(np.sum(data.y[indices] == label)) for label in (0, 1)}
    valid = counts[0] > 0 and counts[1] > 0
    return {
        "task": data.task,
        "session": str(data.session),
        "fold": int(fold),
        "seed": int(seed),
        "label_fraction": float(label_fraction),
        "n_label_cycles": int(len(np.unique(data.groups[indices]))),
        "class_0_samples": counts[0],
        "class_1_samples": counts[1],
        "n_labeled_samples": int(len(indices)),
        "status": "VALID" if valid else "INVALID_SINGLE_CLASS_TRAINING",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_v1_checkpoint(
    path: Path,
    *,
    session: str,
    fold: int,
    seed: int,
    outer_train_cycles: Iterable[int],
    outer_test_cycles: Iterable[int],
    manifest_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every semantic and byte-level condition required for v1 reuse."""
    encoder, payload = load_ssl_encoder_checkpoint(path)
    train = np.asarray(sorted({int(value) for value in outer_train_cycles}), dtype=np.int64)
    test = np.asarray(sorted({int(value) for value in outer_test_cycles}), dtype=np.int64)
    expected_ssl_train, expected_ssl_val = fixed_ssl_validation_cycles(train)
    actual_ssl_train = np.asarray(sorted(map(int, payload.get("ssl_train_cycles", []))), dtype=np.int64)
    actual_ssl_val = np.asarray(sorted(map(int, payload.get("ssl_val_cycles", []))), dtype=np.int64)
    actual_test = np.asarray(sorted(map(int, payload.get("outer_test_cycles", []))), dtype=np.int64)
    actual_hash = file_sha256(path)
    checks = {
        "session_match": str(payload.get("session")) == str(session),
        "fold_match": int(payload.get("fold", -1)) == int(fold),
        "seed_match": int(payload.get("seed", -1)) == int(seed),
        "ssl_train_cycles_match": np.array_equal(actual_ssl_train, expected_ssl_train),
        "ssl_val_cycles_match": np.array_equal(actual_ssl_val, expected_ssl_val),
        "outer_train_pool_match": set(actual_ssl_train.tolist()) | set(actual_ssl_val.tolist()) == set(train.tolist()),
        "outer_test_cycles_match": np.array_equal(actual_test, test),
        "test_excluded_from_ssl": not bool(
            (set(actual_ssl_train.tolist()) | set(actual_ssl_val.tolist())) & set(test.tolist())
        ),
        "pretraining_config_match": payload.get("pretraining_config") == asdict(FROZEN_SSL_CONFIG),
        "architecture_match": payload.get("encoder_class") == "SmallCNNFrameEncoder"
        and isinstance(encoder, SmallCNNFrameEncoder),
        "normalization_shape_match": tuple(payload.get("normalization_mean", np.empty(0)).shape) == (1, 128, 501)
        and tuple(payload.get("normalization_std", np.empty(0)).shape) == (1, 128, 501),
        "decoder_discarded": payload.get("decoder_discarded") is True,
        "contains_labels_false": payload.get("contains_labels") is False,
        "final_epoch_match": int(payload.get("final_epoch", -1)) == FROZEN_SSL_CONFIG.epochs,
        "manifest_hash_match": manifest_sha256 is not None and actual_hash == str(manifest_sha256),
    }
    reused = all(checks.values())
    reason = (
        "exact v1 fold/seed/config/architecture/hash match; v1 SSL train+validation cycles equal outer TRAIN cycles"
        if reused
        else "reuse rejected: " + ",".join(key for key, passed in checks.items() if not passed)
    )
    row = {
        "session": str(session),
        "fold": int(fold),
        "seed": int(seed),
        "checkpoint": str(path),
        "checkpoint_sha256": actual_hash,
        "manifest_sha256": "" if manifest_sha256 is None else str(manifest_sha256),
        "reused": bool(reused),
        "reason": reason,
        **checks,
    }
    if not reused:
        raise AssertionError(reason)
    return row, payload


def condition_encoder_state(
    condition: str, checkpoint: Path | None
) -> dict[str, torch.Tensor] | None:
    if condition == "RANDOM_INIT":
        if checkpoint is not None:
            raise ValueError("RANDOM_INIT must not receive an SSL checkpoint")
        return None
    if condition != "WITHIN_MASKED_SSL_FT":
        raise ValueError(f"unknown v4 condition: {condition}")
    if checkpoint is None:
        raise ValueError("WITHIN_MASKED_SSL_FT requires a checkpoint")
    encoder, _payload = load_ssl_encoder_checkpoint(checkpoint)
    return encoder.state_dict()


def assert_formal_cuda(device: str) -> torch.device:
    if str(device) != "cuda":
        raise RuntimeError("formal v4 requires --device cuda; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("formal v4 requires an available CUDA device; STOP")
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


def serialized_cycle_list(value: Iterable[int]) -> str:
    return json.dumps([int(item) for item in value])


def ordered_cycle_text(value: Iterable[int]) -> str:
    """Serialize cycle IDs without sorting, preserving the nested-prefix order."""
    return ",".join(str(int(item)) for item in value)
