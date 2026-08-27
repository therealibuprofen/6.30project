"""Leakage-safe nested selection between the two frozen FCNN statistics variants.

This module deliberately has no API for reading the completed outer experiment.
Inner training, inner evaluation, and selection locking therefore cannot access
outer-test results through this dependency.  Outer-result reuse lives in the
separate ``adaptive_mean_std_outer_reuse`` module.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.evaluate import classification_metrics

from .fcnn_temporal_statistics import (
    INPUT_VARIANTS,
    MEAN_ONLY_VARIANT,
    MEAN_STD_VARIANT,
    MODEL_IMPLEMENTATION_VERSION,
    architecture_config,
    build_model,
)
from .training import (
    DeepTrainingConfig,
    _train_epochs,
    blocks_to_sequence_tensor,
    labels_to_class_indices,
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    resolve_device,
    set_reproducible_seed,
)


PROTOCOL_VERSION = "adaptive_mean_std_nestedcv_v1.0.0"
SELECTION_RULE_VERSION = "strict_inner_oof_ba_v1_tie_mean_only"
METRIC_IMPLEMENTATION_VERSION = "concatenated_oof_balanced_accuracy_v1"
NORMALIZATION_PROTOCOL = (
    "clean4 -> per-frame arcsinh -> inner-train-only all-frame pixel-wise "
    "z-score; epsilon=1e-6"
)
EXPECTED_OUTER_FOLDS = 82
EXPECTED_INNER_DEFINITIONS = 722
EXPECTED_UNIQUE_TRAIN_SETS = 425
EXPECTED_UNIQUE_TRAINING_JOBS = 2550
EXPECTED_LOGICAL_INNER_JOBS = 4332
EXPECTED_SELECTIONS = 246

TRAINING_CACHE_FILES = (
    "checkpoint.pt",
    "normalization.npz",
    "training_history.csv",
    "metadata.json",
)
EVALUATION_CACHE_FILES = ("predictions.csv", "metrics.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        if not value:
            return ()
        return tuple(sorted({int(item) for item in value.split(",")}))
    return tuple(sorted({int(item) for item in value}))


def ids_text(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in sorted({int(item) for item in values}))


def sample_ids_json(values: Sequence[str]) -> str:
    return canonical_json(sorted(str(value) for value in values))


def parse_sample_ids(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = list(value)
    return tuple(sorted(str(item) for item in parsed))


def select_variant(inner_ba_mean_only: float, inner_ba_mean_std: float) -> str:
    """Apply the frozen strict-greater rule; exact ties select Mean-only."""

    if not np.isfinite([inner_ba_mean_only, inner_ba_mean_std]).all():
        raise ValueError("inner OOF BA values must be finite")
    return (
        MEAN_STD_VARIANT
        if float(inner_ba_mean_std) > float(inner_ba_mean_only)
        else MEAN_ONLY_VARIANT
    )


def concatenated_oof_balanced_accuracy(
    predictions: pd.DataFrame, expected_sample_ids: Sequence[str]
) -> float:
    """Calculate BA once after exact OOF sample-coverage validation."""

    required = {"sample_id", "y_true", "y_pred"}
    if not required.issubset(predictions.columns):
        raise AssertionError(
            f"inner predictions lack {sorted(required-set(predictions.columns))}"
        )
    expected = tuple(sorted(str(value) for value in expected_sample_ids))
    observed = tuple(sorted(predictions["sample_id"].astype(str).tolist()))
    if predictions["sample_id"].astype(str).duplicated().any():
        raise AssertionError("inner OOF predictions contain duplicate sample IDs")
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise AssertionError(
            f"inner OOF sample coverage mismatch: missing={missing}, extra={extra}"
        )
    return float(
        classification_metrics(
            predictions["y_true"].to_numpy(int),
            predictions["y_pred"].to_numpy(int),
        )["balanced_accuracy"]
    )


def validate_outer_manifest(formal_task_plan: pd.DataFrame) -> pd.DataFrame:
    """Reduce the approved paired task plan to its exact 82 outer folds."""

    required = {
        "session",
        "variant",
        "seed",
        "fold",
        "train_cycles",
        "test_cycles",
        "n_train_samples",
        "n_test_samples",
    }
    if not required.issubset(formal_task_plan.columns):
        raise AssertionError(
            f"formal task plan lacks {sorted(required-set(formal_task_plan.columns))}"
        )
    frame = formal_task_plan.copy()
    frame["session"] = frame["session"].astype(str)
    if set(frame["variant"].astype(str)) != set(INPUT_VARIANTS):
        raise AssertionError("formal candidate coverage differs from approved variants")
    if set(pd.to_numeric(frame["seed"]).astype(int)) != {0, 1, 2}:
        raise AssertionError("formal seed coverage differs from 0/1/2")
    membership = [
        "session",
        "fold",
        "train_cycles",
        "test_cycles",
        "n_train_samples",
        "n_test_samples",
    ]
    outer = frame[membership].drop_duplicates().sort_values(
        ["session", "fold"]
    )
    if outer.duplicated(["session", "fold"]).any() or len(outer) != EXPECTED_OUTER_FOLDS:
        raise AssertionError("formal outer-fold membership is not exactly 82 folds")
    paired = frame.groupby(["session", "fold", "seed"])["variant"].nunique()
    if len(paired) != EXPECTED_SELECTIONS or not paired.eq(2).all():
        raise AssertionError("formal candidates are not paired on every fold and seed")
    return outer.reset_index(drop=True)


def enumerate_inner_splits(
    outer_manifest: pd.DataFrame,
    session_sample_ids: Mapping[str, Sequence[str]],
    session_groups: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Dynamically enumerate inner grouped CV without any outer-test data."""

    rows: list[dict[str, Any]] = []
    for outer in outer_manifest.itertuples(index=False):
        session = str(outer.session)
        groups = np.asarray(session_groups[session], dtype=np.int64)
        sample_ids = np.asarray(session_sample_ids[session], dtype=str)
        if len(groups) != len(sample_ids) or len(set(sample_ids.tolist())) != len(sample_ids):
            raise AssertionError(f"session {session}: sample identity mismatch")
        outer_train_cycles = parse_ids(str(outer.train_cycles))
        outer_test_cycles = parse_ids(str(outer.test_cycles))
        outer_train_mask = np.isin(groups, outer_train_cycles)
        outer_train_indices = np.flatnonzero(outer_train_mask)
        if set(groups[~outer_train_mask].tolist()) != set(outer_test_cycles):
            raise AssertionError(
                f"session {session} fold {outer.fold}: formal membership mismatch"
            )
        if len(outer_train_indices) != int(outer.n_train_samples) or int(
            (~outer_train_mask).sum()
        ) != int(outer.n_test_samples):
            raise AssertionError(
                f"session {session} fold {outer.fold}: formal sample count mismatch"
            )
        inner_relative = grouped_cv_splits(
            groups[outer_train_indices], max_folds=10
        )
        for inner_fold, (train_rel, val_rel) in enumerate(inner_relative, start=1):
            train_idx = outer_train_indices[train_rel]
            val_idx = outer_train_indices[val_rel]
            train_cycles = tuple(sorted(np.unique(groups[train_idx]).tolist()))
            val_cycles = tuple(sorted(np.unique(groups[val_idx]).tolist()))
            outer_train_set = set(outer_train_cycles)
            outer_test_set = set(outer_test_cycles)
            train_set = set(train_cycles)
            val_set = set(val_cycles)
            if outer_test_set & train_set:
                raise AssertionError("outer test cycle entered inner training")
            if outer_test_set & val_set:
                raise AssertionError("outer test cycle entered inner validation")
            if train_set & val_set:
                raise AssertionError("inner train/validation cycle overlap")
            if train_set | val_set != outer_train_set:
                raise AssertionError("inner train/validation do not cover outer train")
            rows.append(
                {
                    "session": session,
                    "outer_fold": int(outer.fold),
                    "inner_fold": int(inner_fold),
                    "outer_train_cycle_ids": ids_text(outer_train_cycles),
                    "outer_test_cycle_ids": ids_text(outer_test_cycles),
                    "inner_train_cycle_ids": ids_text(train_cycles),
                    "inner_val_cycle_ids": ids_text(val_cycles),
                    "inner_train_sample_ids": sample_ids_json(sample_ids[train_idx]),
                    "inner_val_sample_ids": sample_ids_json(sample_ids[val_idx]),
                    "outer_train_sample_ids": sample_ids_json(
                        sample_ids[outer_train_indices]
                    ),
                    "inner_train_sample_count": int(len(train_idx)),
                    "inner_val_sample_count": int(len(val_idx)),
                    "normalization_fit_cycle_ids": ids_text(train_cycles),
                    "outer_test_inner_train_overlap_count": 0,
                    "outer_test_inner_val_overlap_count": 0,
                    "inner_train_val_overlap_count": 0,
                    "inner_union_equals_outer_train": True,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["session", "outer_fold", "inner_fold"]
    ).reset_index(drop=True)
    if len(result) != EXPECTED_INNER_DEFINITIONS:
        raise AssertionError(
            f"inner split definitions {len(result)} != {EXPECTED_INNER_DEFINITIONS}"
        )
    unique_sets = result[["session", "inner_train_cycle_ids"]].drop_duplicates()
    if len(unique_sets) != EXPECTED_UNIQUE_TRAIN_SETS:
        raise AssertionError(
            f"unique inner training sets {len(unique_sets)} != "
            f"{EXPECTED_UNIQUE_TRAIN_SETS}"
        )
    return result


def build_training_cache_identity(
    *,
    session: str,
    train_sample_ids: Sequence[str],
    train_cycle_ids: Sequence[int],
    candidate: str,
    seed: int,
    dataset_source_hash: Mapping[str, str],
    session_manifest_hash: str,
    candidate_source_hashes: Mapping[str, str],
    protocol_fingerprint: str,
    runtime_fingerprint: str,
    training_config: DeepTrainingConfig,
) -> dict[str, Any]:
    if candidate not in INPUT_VARIANTS:
        raise ValueError(f"unknown candidate {candidate!r}")
    exact_sample_ids = sorted(str(value) for value in train_sample_ids)
    exact_cycle_ids = sorted({int(value) for value in train_cycle_ids})
    return {
        "cache_kind": "inner_training_model",
        "protocol_version": PROTOCOL_VERSION,
        "session": str(session),
        "exact_training_sample_ids": exact_sample_ids,
        "exact_training_cycle_ids": exact_cycle_ids,
        "dataset_source_sha256": dict(sorted(dataset_source_hash.items())),
        "formal_session_manifest_sha256": str(session_manifest_hash),
        "input_protocol": "clean4",
        "task": "binary_presence",
        "label_mapping": {"0": "no_stimulus", "1": "stimulus"},
        "candidate": str(candidate),
        "candidate_model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "candidate_architecture": architecture_config(candidate),
        "candidate_source_sha256": dict(sorted(candidate_source_hashes.items())),
        "normalization_protocol": NORMALIZATION_PROTOCOL,
        "normalization_fit_sample_ids": exact_sample_ids,
        "normalization_fit_cycle_ids": exact_cycle_ids,
        "training": asdict(training_config),
        "seed": int(seed),
        "initialization_policy": "set_reproducible_seed(seed)_before_build_model",
        "dataloader_rng_policy": "torch.Generator.manual_seed(seed); shuffle=True",
        "runtime_fingerprint": str(runtime_fingerprint),
        "protocol_fingerprint": str(protocol_fingerprint),
    }


def training_cache_key(identity: Mapping[str, Any]) -> str:
    return fingerprint(dict(identity))


def build_evaluation_cache_identity(
    *,
    training_key: str,
    session: str,
    parent_outer_fold: int,
    outer_seed: int,
    candidate: str,
    validation_sample_ids: Sequence[str],
    validation_cycle_ids: Sequence[int],
    current_outer_train_cycle_ids: Sequence[int],
    current_outer_test_cycle_ids: Sequence[int],
    protocol_fingerprint: str,
) -> dict[str, Any]:
    identity = {
        "cache_kind": "inner_validation_evaluation",
        "protocol_version": PROTOCOL_VERSION,
        "training_cache_key": str(training_key),
        "session": str(session),
        "parent_outer_fold": int(parent_outer_fold),
        "outer_seed": int(outer_seed),
        "candidate": str(candidate),
        "exact_validation_sample_ids": sorted(
            str(value) for value in validation_sample_ids
        ),
        "exact_validation_cycle_ids": sorted(
            {int(value) for value in validation_cycle_ids}
        ),
        "current_outer_train_cycle_ids": sorted(
            {int(value) for value in current_outer_train_cycle_ids}
        ),
        "current_outer_test_cycle_ids": sorted(
            {int(value) for value in current_outer_test_cycle_ids}
        ),
        "metric_implementation_version": METRIC_IMPLEMENTATION_VERSION,
        "protocol_fingerprint": str(protocol_fingerprint),
    }
    validate_parent_evaluation_access(
        identity,
        current_outer_train_cycle_ids=current_outer_train_cycle_ids,
        current_outer_test_cycle_ids=current_outer_test_cycle_ids,
    )
    return identity


def evaluation_cache_key(identity: Mapping[str, Any]) -> str:
    return fingerprint(dict(identity))


def validate_parent_evaluation_access(
    identity: Mapping[str, Any],
    *,
    current_outer_train_cycle_ids: Sequence[int],
    current_outer_test_cycle_ids: Sequence[int],
) -> None:
    allowed = {int(value) for value in current_outer_train_cycle_ids}
    forbidden = {int(value) for value in current_outer_test_cycle_ids}
    evaluated = {
        int(value) for value in identity["exact_validation_cycle_ids"]
    }
    if not evaluated <= allowed:
        raise PermissionError("evaluation cycles are outside current outer train")
    if evaluated & forbidden:
        raise PermissionError("current parent outer-test cycle entered evaluation")
    stored_allowed = {
        int(value) for value in identity["current_outer_train_cycle_ids"]
    }
    stored_forbidden = {
        int(value) for value in identity["current_outer_test_cycle_ids"]
    }
    if stored_allowed != allowed or stored_forbidden != forbidden:
        raise PermissionError("evaluation artifact belongs to another parent fold")


def _artifact_hashes(path: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: file_sha256(path / name) for name in names}


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def train_inner_cache(
    path: Path,
    identity: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation_probe: np.ndarray,
    *,
    device: str,
    workers: int,
) -> None:
    """Train one unique inner model using only its declared training samples."""

    candidate = str(identity["candidate"])
    seed = int(identity["seed"])
    config = DeepTrainingConfig(**dict(identity["training"]))
    set_reproducible_seed(seed)
    torch_device = resolve_device(device)
    train_norm, _probe_norm, audit, mean32, std32 = (
        normalize_blocks_train_fold_only_with_stats(
            X_train,
            X_validation_probe,
            session=str(identity["session"]),
            task="binary",
            method=candidate,
            seed=seed,
            fold=0,
            train_cycles=ids_text(identity["exact_training_cycle_ids"]),
            test_cycles="inner_validation_probe",
        )
    )
    audit["phase"] = "inner_train_fold_only"
    # Preserve the exact float64 statistics used by the approved normalizer so a
    # repeated cached model can transform another legal validation subset exactly.
    train_asinh = np.arcsinh(X_train.astype(np.float32, copy=False))
    frames64 = train_asinh.reshape(
        -1, train_asinh.shape[-2], train_asinh.shape[-1]
    ).astype(np.float64, copy=False)
    mean64 = frames64.mean(axis=0, keepdims=True)
    std64 = frames64.std(axis=0, keepdims=True) + 1e-6
    if not np.array_equal(mean64.astype(np.float32), mean32) or not np.array_equal(
        std64.astype(np.float32), std32
    ):
        raise AssertionError("cached normalization differs from approved normalizer")
    model = build_model(candidate, n_classes=2).to(torch_device)
    history = _train_epochs(
        model,
        blocks_to_sequence_tensor(train_norm),
        labels_to_class_indices(y_train, np.asarray([0, 1])),
        config=config,
        seed=seed,
        device=torch_device,
        batch_size_reference=len(X_train),
        num_workers=workers,
    )
    history_frame = pd.DataFrame(history)
    if len(history_frame) != int(config.max_epochs):
        raise AssertionError("inner training did not reach the fixed final epoch")
    training_key = training_cache_key(identity)
    metadata = {
        "identity": dict(identity),
        "training_cache_key": training_key,
        "candidate": candidate,
        "seed": seed,
        "normalization_fit_sample_ids": list(
            identity["normalization_fit_sample_ids"]
        ),
        "normalization_fit_cycle_ids": list(
            identity["normalization_fit_cycle_ids"]
        ),
        "normalization_validation_excluded": True,
        "normalization_outer_test_excluded": True,
        "normalization_audit": {
            key: value
            for key, value in audit.items()
            if not key.startswith("test_") and key != "test_cycles"
        },
        "final_epoch": int(config.max_epochs),
        "train_accuracy_epoch40": float(history_frame.iloc[-1]["train_accuracy"]),
        "final_training_loss": float(history_frame.iloc[-1]["train_loss"]),
        "created_at": utc_now(),
    }
    path.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        path / "checkpoint.pt",
        {
            "training_cache_key": training_key,
            "candidate": candidate,
            "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
            "model_config": architecture_config(candidate),
            "state_dict": model.state_dict(),
        },
    )
    _atomic_npz(path / "normalization.npz", mean=mean64, std=std64)
    atomic_csv(path / "training_history.csv", history_frame)
    atomic_json(path / "metadata.json", metadata)
    atomic_json(
        path / "COMPLETE.json",
        {
            "status": "complete",
            "training_cache_key": training_key,
            "artifact_sha256": _artifact_hashes(path, TRAINING_CACHE_FILES),
            "completed_at": utc_now(),
        },
    )
    valid, reason = validate_training_cache(path, identity, load_checkpoint=True)
    if not valid:
        raise AssertionError(f"new training cache failed validation: {reason}")


def validate_training_cache(
    path: Path,
    expected_identity: Mapping[str, Any],
    *,
    load_checkpoint: bool,
) -> tuple[bool, str]:
    expected_key = training_cache_key(expected_identity)
    required = ("COMPLETE.json", *TRAINING_CACHE_FILES)
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return False, f"missing {missing}"
    try:
        complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        history = pd.read_csv(path / "training_history.csv")
        with np.load(path / "normalization.npz") as arrays:
            mean = arrays["mean"]
            std = arrays["std"]
    except Exception as exc:
        return False, f"unreadable cache artifact: {exc}"
    if complete.get("status") != "complete":
        return False, "completion marker is not complete"
    if complete.get("training_cache_key") != expected_key:
        return False, "completion training key mismatch"
    if complete.get("artifact_sha256") != _artifact_hashes(path, TRAINING_CACHE_FILES):
        return False, "training artifact hash mismatch"
    if metadata.get("identity") != dict(expected_identity):
        return False, "training identity mismatch"
    if metadata.get("training_cache_key") != expected_key:
        return False, "metadata training key mismatch"
    epochs = int(expected_identity["training"]["max_epochs"])
    if len(history) != epochs or not np.array_equal(
        history["epoch"].to_numpy(int), np.arange(1, epochs + 1)
    ):
        return False, "training history is not complete"
    if not np.isfinite(history[["train_loss", "train_accuracy"]].to_numpy()).all():
        return False, "training history is non-finite"
    if mean.shape != std.shape or mean.ndim != 3 or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        return False, "normalization arrays are invalid"
    if metadata.get("normalization_fit_sample_ids") != expected_identity.get(
        "normalization_fit_sample_ids"
    ):
        return False, "normalization fit membership mismatch"
    if not bool(metadata.get("normalization_validation_excluded")) or not bool(
        metadata.get("normalization_outer_test_excluded")
    ):
        return False, "normalization exclusion assertion failed"
    if load_checkpoint:
        try:
            checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu")
            if checkpoint.get("training_cache_key") != expected_key:
                return False, "checkpoint training key mismatch"
            if checkpoint.get("candidate") != expected_identity["candidate"]:
                return False, "checkpoint candidate mismatch"
            model = build_model(str(expected_identity["candidate"]), n_classes=2)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
        except Exception as exc:
            return False, f"checkpoint is not loadable/compatible: {exc}"
    return True, "validated"


def load_training_cache(
    path: Path,
    expected_identity: Mapping[str, Any],
    *,
    device: str,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, torch.device]:
    valid, reason = validate_training_cache(path, expected_identity, load_checkpoint=True)
    if not valid:
        raise AssertionError(f"invalid training cache {path}: {reason}")
    torch_device = resolve_device(device)
    checkpoint = torch.load(path / "checkpoint.pt", map_location=torch_device)
    model = build_model(str(expected_identity["candidate"]), n_classes=2).to(
        torch_device
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    with np.load(path / "normalization.npz") as arrays:
        mean = arrays["mean"].copy()
        std = arrays["std"].copy()
    return model, mean, std, torch_device


def evaluate_inner_cache(
    path: Path,
    identity: Mapping[str, Any],
    *,
    model: torch.nn.Module,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    validation_sample_ids: Sequence[str],
    validation_cycles: Sequence[int],
    device: torch.device,
    batch_size: int,
    workers: int,
) -> None:
    validate_parent_evaluation_access(
        identity,
        current_outer_train_cycle_ids=identity["current_outer_train_cycle_ids"],
        current_outer_test_cycle_ids=identity["current_outer_test_cycle_ids"],
    )
    if tuple(sorted(str(value) for value in validation_sample_ids)) != tuple(
        identity["exact_validation_sample_ids"]
    ):
        raise AssertionError("runtime validation sample IDs differ from eval identity")
    if set(int(value) for value in validation_cycles) != set(
        identity["exact_validation_cycle_ids"]
    ):
        raise AssertionError("runtime validation cycles differ from eval identity")
    X_asinh = np.arcsinh(X_validation.astype(np.float32, copy=False))
    normalized = ((X_asinh - normalization_mean) / normalization_std).astype(
        np.float32, copy=False
    )
    if not np.isfinite(normalized).all():
        raise AssertionError("cached normalization produced non-finite validation data")
    probabilities = predict_probabilities(
        model,
        blocks_to_sequence_tensor(normalized),
        device=device,
        batch_size=int(batch_size),
        num_workers=workers,
    )
    predictions = probabilities.argmax(axis=1).astype(np.int64)
    metrics = classification_metrics(y_validation.astype(int), predictions)
    frame = pd.DataFrame(
        {
            "sample_id": [str(value) for value in validation_sample_ids],
            "cycle": np.asarray(validation_cycles, dtype=int),
            "y_true": y_validation.astype(int),
            "y_pred": predictions,
            "probability_0": probabilities[:, 0],
            "probability_1": probabilities[:, 1],
        }
    )
    eval_key = evaluation_cache_key(identity)
    payload = {
        "identity": dict(identity),
        "evaluation_cache_key": eval_key,
        "inner_val_BA": float(metrics["balanced_accuracy"]),
        "n_validation_samples": int(len(frame)),
        "created_at": utc_now(),
    }
    path.mkdir(parents=True, exist_ok=True)
    atomic_csv(path / "predictions.csv", frame)
    atomic_json(path / "metrics.json", payload)
    atomic_json(
        path / "COMPLETE.json",
        {
            "status": "complete",
            "evaluation_cache_key": eval_key,
            "artifact_sha256": _artifact_hashes(path, EVALUATION_CACHE_FILES),
            "completed_at": utc_now(),
        },
    )
    valid, reason = validate_evaluation_cache(
        path,
        identity,
        current_outer_train_cycle_ids=identity["current_outer_train_cycle_ids"],
        current_outer_test_cycle_ids=identity["current_outer_test_cycle_ids"],
    )
    if not valid:
        raise AssertionError(f"new evaluation cache failed validation: {reason}")


def validate_evaluation_cache(
    path: Path,
    expected_identity: Mapping[str, Any],
    *,
    current_outer_train_cycle_ids: Sequence[int],
    current_outer_test_cycle_ids: Sequence[int],
) -> tuple[bool, str]:
    expected_key = evaluation_cache_key(expected_identity)
    required = ("COMPLETE.json", *EVALUATION_CACHE_FILES)
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        return False, f"missing {missing}"
    try:
        complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path / "predictions.csv")
        validate_parent_evaluation_access(
            metrics["identity"],
            current_outer_train_cycle_ids=current_outer_train_cycle_ids,
            current_outer_test_cycle_ids=current_outer_test_cycle_ids,
        )
    except Exception as exc:
        return False, f"unreadable or forbidden evaluation: {exc}"
    if complete.get("status") != "complete":
        return False, "evaluation completion marker is not complete"
    if complete.get("evaluation_cache_key") != expected_key:
        return False, "evaluation completion key mismatch"
    if complete.get("artifact_sha256") != _artifact_hashes(path, EVALUATION_CACHE_FILES):
        return False, "evaluation artifact hash mismatch"
    if metrics.get("identity") != dict(expected_identity):
        return False, "evaluation identity mismatch"
    if metrics.get("evaluation_cache_key") != expected_key:
        return False, "evaluation metadata key mismatch"
    expected_ids = tuple(expected_identity["exact_validation_sample_ids"])
    observed_ids = tuple(sorted(predictions["sample_id"].astype(str).tolist()))
    if predictions["sample_id"].astype(str).duplicated().any() or observed_ids != expected_ids:
        return False, "evaluation sample coverage mismatch"
    metric = classification_metrics(
        predictions["y_true"].to_numpy(int), predictions["y_pred"].to_numpy(int)
    )["balanced_accuracy"]
    if not np.isclose(float(metrics.get("inner_val_BA", np.nan)), metric, atol=1e-12):
        return False, "stored validation BA differs from predictions"
    return True, "validated"


def build_selection_payload(
    *,
    session: str,
    outer_fold: int,
    seed: int,
    outer_train_cycle_ids: Sequence[int],
    outer_test_cycle_ids: Sequence[int],
    inner_ba_mean_only: float,
    inner_ba_mean_std: float,
    candidate_protocol_fingerprints: Mapping[str, str],
    split_fingerprint: str,
    normalization_protocol_fingerprint: str,
    inner_oof_prediction_hashes: Mapping[str, str],
    expected_outer_train_sample_ids: Sequence[str],
    observed_candidate_sample_ids: Mapping[str, Sequence[str]],
    protocol_fingerprint: str,
) -> dict[str, Any]:
    expected = tuple(sorted(str(value) for value in expected_outer_train_sample_ids))
    coverage: dict[str, Any] = {}
    for candidate in INPUT_VARIANTS:
        observed_values = [
            str(value) for value in observed_candidate_sample_ids[candidate]
        ]
        observed = tuple(sorted(observed_values))
        coverage[candidate] = {
            "expected_count": len(expected),
            "observed_count": len(observed_values),
            "unique_count": len(set(observed_values)),
            "complete_exactly_once": observed == expected
            and len(set(observed_values)) == len(observed_values),
        }
        if not coverage[candidate]["complete_exactly_once"]:
            raise AssertionError(f"{candidate} inner OOF coverage is incomplete")
    selected = select_variant(inner_ba_mean_only, inner_ba_mean_std)
    base = {
        "session": str(session),
        "outer_fold": int(outer_fold),
        "seed": int(seed),
        "selector_seed": int(seed),
        "outer_train_cycle_ids": sorted(
            {int(value) for value in outer_train_cycle_ids}
        ),
        "outer_test_cycle_ids": sorted(
            {int(value) for value in outer_test_cycle_ids}
        ),
        "inner_BA_mean_only": float(inner_ba_mean_only),
        "inner_BA_mean_std": float(inner_ba_mean_std),
        "delta_inner_BA": float(inner_ba_mean_std - inner_ba_mean_only),
        "tie": bool(float(inner_ba_mean_std) == float(inner_ba_mean_only)),
        "selected_variant": selected,
        "selection_rule_version": SELECTION_RULE_VERSION,
        "candidate_protocol_fingerprints": dict(
            sorted(candidate_protocol_fingerprints.items())
        ),
        "split_fingerprint": str(split_fingerprint),
        "normalization_protocol_fingerprint": str(
            normalization_protocol_fingerprint
        ),
        "inner_oof_prediction_hashes": dict(
            sorted(inner_oof_prediction_hashes.items())
        ),
        "inner_coverage_assertion": coverage,
        "protocol_fingerprint": str(protocol_fingerprint),
        "outer_result_read_before_selection": False,
        "created_at": utc_now(),
    }
    base["selection_artifact_hash"] = fingerprint(base)
    return base


def validate_selection_payload(
    payload: Mapping[str, Any], *, expected_protocol_fingerprint: str
) -> None:
    if payload.get("protocol_fingerprint") != expected_protocol_fingerprint:
        raise AssertionError("selection protocol fingerprint mismatch")
    if payload.get("selection_rule_version") != SELECTION_RULE_VERSION:
        raise AssertionError("selection rule version mismatch")
    if bool(payload.get("outer_result_read_before_selection", True)):
        raise AssertionError("selection claims outer result was read")
    expected_selected = select_variant(
        float(payload["inner_BA_mean_only"]),
        float(payload["inner_BA_mean_std"]),
    )
    if payload.get("selected_variant") != expected_selected:
        raise AssertionError("selected variant differs from frozen rule")
    base = dict(payload)
    observed_hash = base.pop("selection_artifact_hash", None)
    if observed_hash != fingerprint(base):
        raise AssertionError("selection artifact hash mismatch")
    coverage = payload.get("inner_coverage_assertion", {})
    if set(coverage) != set(INPUT_VARIANTS) or not all(
        bool(coverage[candidate].get("complete_exactly_once"))
        for candidate in INPUT_VARIANTS
    ):
        raise AssertionError("selection inner OOF coverage assertion failed")


def lock_selection(
    path: Path, payload: Mapping[str, Any], *, expected_protocol_fingerprint: str
) -> dict[str, Any]:
    """Create once; an existing differing/tampered artifact is never overwritten."""

    validate_selection_payload(
        payload, expected_protocol_fingerprint=expected_protocol_fingerprint
    )
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        validate_selection_payload(
            observed, expected_protocol_fingerprint=expected_protocol_fingerprint
        )
        if observed != dict(payload):
            raise RuntimeError("locked selection exists with different content")
        return observed
    atomic_json(path, dict(payload))
    observed = json.loads(path.read_text(encoding="utf-8"))
    validate_selection_payload(
        observed, expected_protocol_fingerprint=expected_protocol_fingerprint
    )
    return observed


def read_locked_selection(
    path: Path, *, expected_protocol_fingerprint: str
) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError("outer stage requires a locked selection artifact")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_selection_payload(
        payload, expected_protocol_fingerprint=expected_protocol_fingerprint
    )
    return payload
