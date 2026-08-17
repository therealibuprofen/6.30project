"""Cross-session feature distribution and factor attribution analysis (v7).

This module is deliberately an analysis layer, not a model benchmark.  It
reuses the frozen clean4 data builder and the masked-SmallCNN implementation
from v1/v2.  The global neural encoder is label-free and descriptive only;
the strict predictive probe fits every preprocessing statistic and PCA basis
on the eight source sessions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from torch import nn

from ultrasound_decoding.multiframe.dataset import (
    BLOCK_NAMES,
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    BlockSequenceData,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import SmallCNNFrameEncoder
from ultrasound_decoding.ssl_masked import SSLPretrainingConfig, apply_ssl_frame_normalizer
from ultrasound_decoding.ssl_multisession_v2 import (
    SessionBalancedSampler,
    SessionFramePool,
    architecture_fingerprint,
    complete_cycles_from_unlabeled_h5,
    fit_ssl_pool_normalizer,
    load_unlabeled_cycles,
    pretrain_session_balanced_smallcnn,
)


RUN_NAME = "cross_session_feature_factor_analysis_9sessions_v7"
GLOBAL_ENCODER_SEEDS = (20260812, 20260813, 20260814)
N_BOOTSTRAP = 1000
N_SESSION_PERMUTATIONS = 1000
STATISTIC_SEED = 20260815
PCA_RANDOM_SEED = 20260816
CONDITION4_NAMES = ("grating", "stop", "dot", "static")
BLOCK_TO_CONDITION4 = {
    "grating": "grating",
    "stop_after_grating": "stop",
    "dot": "dot",
    "static": "static",
}
STIMULUS_BLOCKS = frozenset(("grating", "dot"))
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")
CONDITION_TIME_WARNING = (
    "condition4 is confounded with fixed within-cycle temporal position, because condition order is fixed; "
    "the condition/time-position effect cannot be interpreted as pure visual stimulus identity."
)

REQUIRED_OUTPUTS = (
    "audit/metadata_factor_audit.csv",
    "audit/clean4_identity_check.csv",
    "audit/global_encoder_reuse.csv",
    "audit/v5_cross_session_metric_reuse.csv",
    "audit/within_session_metric_reuse.csv",
    "audit/pairwise_crosssession_availability.csv",
    "audit/config_freeze.md",
    "audit/gpu_audit.txt",
    "features/raw_pca_common_features.csv",
    "features/raw_pca_model.pkl",
    "summaries/pairwise_session_distances.csv",
    "summaries/session_id_probe.csv",
    "summaries/source_only_stimulus_probe.csv",
    "summaries/factor_variance_binary.csv",
    "summaries/factor_variance_condition4.csv",
    "summaries/distance_performance_association.csv",
    "summaries/session_diagnostic_table.csv",
    "summaries/within_session_factor_associations.csv",
    "figures/raw_pca_colored_by_session.png",
    "figures/raw_pca_colored_by_stimulus_presence.png",
    "figures/raw_pca_colored_by_condition4.png",
    "figures/masked_feature_colored_by_session.png",
    "figures/masked_feature_colored_by_stimulus_presence.png",
    "figures/masked_feature_colored_by_condition4.png",
    "figures/session_energy_distance_heatmap_raw.png",
    "figures/session_energy_distance_heatmap_masked.png",
    "figures/session_id_confusion_matrix.png",
    "figures/stimulus_probe_by_target.png",
    "figures/factor_variance_binary.png",
    "figures/factor_variance_condition4.png",
    "figures/distance_vs_crosssession_BA.png",
    "figures/weak_vs_strong_diagnostics.png",
    "figures/session_diagnostic_overview.png",
    "report/cross_session_feature_factor_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass
class PCAProjection:
    """Small pickle-safe PCA object with explicit fit-sample provenance."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    n_samples_seen_: int
    n_features_in_: int
    fit_sample_ids_: tuple[str, ...]
    random_seed: int

    @property
    def n_components_(self) -> int:
        return int(self.components_.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("PCA transform feature dimension mismatch")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            output = (values - self.mean_) @ self.components_.T
        if not np.isfinite(output).all():
            raise ValueError("PCA produced non-finite coordinates")
        return output.astype(np.float32, copy=False)


@dataclass
class LinearProbeModel:
    classes_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    C: float
    class_weight: str
    success: bool
    n_iter: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = np.asarray(X, dtype=np.float64) @ self.coef_.T + self.intercept_
        return self.classes_[np.argmax(scores, axis=1)]


def assert_formal_cuda(device: str) -> torch.device:
    if str(device).lower() != "cuda":
        raise RuntimeError("FORMAL STOP: --device must be exactly 'cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError("FORMAL STOP: torch.cuda.is_available() is False; CPU fallback is forbidden")
    return torch.device("cuda")


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_exact_sessions(sessions: Iterable[str]) -> tuple[str, ...]:
    values = tuple(str(value) for value in sessions)
    if values != tuple(EXPECTED_SESSIONS):
        raise ValueError(f"formal analysis requires the frozen nine sessions in order: {EXPECTED_SESSIONS}")
    return values


def add_v7_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    output = metadata.copy()
    if "block_name" not in output:
        raise ValueError("block_name is required")
    unknown = set(output["block_name"].astype(str)) - set(BLOCK_TO_CONDITION4)
    if unknown:
        raise ValueError(f"unknown block names: {sorted(unknown)}")
    output["condition4"] = output["block_name"].astype(str).map(BLOCK_TO_CONDITION4)
    output["stimulus_presence"] = np.where(
        output["block_name"].astype(str).isin(STIMULUS_BLOCKS), "stimulus", "no_stimulus"
    )
    expected_binary = (output["stimulus_presence"] == "stimulus").astype(int).to_numpy()
    if "binary_label_int" in output and not np.array_equal(
        expected_binary, output["binary_label_int"].astype(int).to_numpy()
    ):
        raise AssertionError("v7 stimulus_presence mapping differs from frozen binary labels")
    return output


def sample_metadata_table(data_by_session: Mapping[str, BlockSequenceData]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for session in sorted(data_by_session, key=int):
        data = data_by_session[session]
        metadata = add_v7_labels(data.metadata)
        metadata["session"] = str(session)
        metadata["cycle_key"] = [f"{session}:cycle{int(value)}" for value in data.groups]
        rows.append(metadata)
    output = pd.concat(rows, ignore_index=True)
    output["session"] = output["session"].astype(str)
    return output


def clean4_identity_rows(data_by_session: Mapping[str, BlockSequenceData]) -> list[dict[str, Any]]:
    rows = []
    for session in sorted(data_by_session, key=int):
        data = data_by_session[session]
        metadata = add_v7_labels(data.metadata)
        valid = (
            tuple(data.X.shape[1:]) == EXPECTED_BLOCK_SHAPE
            and data.n_blocks == 4 * data.n_cycles
            and bool(metadata["complete_cycle"].astype(bool).all())
            and bool((metadata.groupby("cycle")["block_name"].count() == 4).all())
            and bool((metadata.groupby("cycle")["block_name"].apply(list).apply(lambda x: x == BLOCK_NAMES)).all())
        )
        rows.append({
            "session": session,
            "source_h5": str(data.source_h5_path),
            "source_metadata_csv": str(data.source_metadata_path),
            "n_blocks": data.n_blocks,
            "n_complete_cycles": data.n_cycles,
            "block_shape": "x".join(map(str, EXPECTED_BLOCK_SHAPE)),
            "clean4_indices_match": True,
            "cycle_ids": ",".join(map(str, sorted(np.unique(data.groups).astype(int).tolist()))),
            "four_conditions_per_cycle": bool(valid),
            "status": "PASS" if valid else "FAIL",
        })
    if any(row["status"] != "PASS" for row in rows):
        raise AssertionError("clean4 identity audit failed")
    return rows


def _raw_mat_dataset_names(data_root: Path, sessions: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for session in sessions:
        paths = sorted((data_root / session).glob("*.mat"))
        if not paths:
            continue
        # Every acquisition file is produced by the same exporter.  Inspect
        # first and last so schema drift is detectable without reading images.
        for path in dict.fromkeys((paths[0], paths[-1])):
            with h5py.File(path, "r") as handle:
                handle.visititems(
                    lambda name, obj: names.add(name) if isinstance(obj, h5py.Dataset) else None
                )
    return names


def metadata_factor_audit(
    metadata: pd.DataFrame,
    *,
    data_root: Path,
    sessions: Sequence[str],
) -> pd.DataFrame:
    """Audit only fields genuinely present in visual data or derived clean4 metadata."""
    raw_names = _raw_mat_dataset_names(data_root, sessions)
    columns = set(metadata.columns)
    session_cycles = metadata.groupby("session", sort=False)["cycle"].nunique()

    def row(
        factor: str,
        available: bool,
        levels: Sequence[Any] | str,
        missing_fraction: float,
        confounded: str,
        usable: bool,
        reason: str,
    ) -> dict[str, Any]:
        level_values = "NOT_AVAILABLE" if not available else (
            levels if isinstance(levels, str) else ";".join(map(str, levels))
        )
        return {
            "factor": factor,
            "available": bool(available),
            "n_levels": 0 if not available else len(levels) if not isinstance(levels, str) else 1,
            "levels": level_values,
            "missing_fraction": 1.0 if not available else float(missing_fraction),
            "confounded_with_session": confounded,
            "usable_for_analysis": bool(usable),
            "reason": reason,
        }

    result = [
        row("session", True, sorted(metadata["session"].astype(str).unique(), key=int), 0, "NO", True,
            "Recorded directory/session identifier; primary factor."),
        row("cycle", True, sorted(metadata["cycle"].astype(int).unique()), 0, "NESTED_WITHIN_SESSION", True,
            "Recorded/derived complete-cycle ID; used only for grouping, resampling, and repeatability."),
        row("condition", True, list(CONDITION4_NAMES), 0, "FIXED_WITHIN_CYCLE_TIME_POSITION", True,
            "Derived from recorded block_name; analyzed as condition/time-position, not pure stimulus identity."),
        row("stimulus_presence", True, ["no_stimulus", "stimulus"], 0, "NO", True,
            "Frozen binary mapping: grating+dot versus stop+static."),
        row("n_complete_cycles", True, [f"{s}:{int(session_cycles.loc[s])}" for s in sorted(session_cycles.index, key=int)],
            0, "SESSION_LEVEL_ATTRIBUTE", True, "Derived without filtering; used in the nine-session association."),
    ]
    absent = {
        "monkey / subject": "No subject/monkey field exists in clean4 CSV/HDF5 or raw MAT datasets.",
        "recording_date": "No recording-date field exists; filesystem timestamps and session numbers are not metadata.",
        "task": "No varying acquisition-task field exists; binary/stimulus_type in outputs are derived decoder targets.",
        "run": "No run factor exists in the visual data metadata.",
        "slot / probe": "No slot or probe field exists in the visual data metadata.",
        "pretraining / retraining": "No acquisition pretraining/retraining field exists in this visual dataset.",
    }
    searchable = {str(value).lower() for value in columns | raw_names}
    aliases = {
        "monkey / subject": ("monkey", "subject"),
        "recording_date": ("recording_date", "date"),
        "task": ("task",),
        "run": ("run",),
        "slot / probe": ("slot", "probe"),
        "pretraining / retraining": ("pretraining", "retraining"),
    }
    for factor, reason in absent.items():
        # If a future visual export adds one of these fields, fail loudly rather
        # than silently treating an unreviewed field as absent or usable.
        detected = [name for name in searchable if any(alias in name for alias in aliases[factor])]
        if detected:
            result.append(row(factor, True, sorted(detected), 0, "REQUIRES_MANUAL_CONFOUND_AUDIT", False,
                              "Field-like names detected but not preregistered for automatic interpretation: " + ";".join(detected)))
        else:
            result.append(row(factor, False, "NOT_AVAILABLE", 1, "NOT_AVAILABLE", False, reason))
    return pd.DataFrame(result)


def fit_frame_pixel_normalizer(block_arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    total_sum: np.ndarray | None = None
    total_square: np.ndarray | None = None
    n_frames = 0
    for X in block_arrays:
        if X.ndim != 4 or tuple(X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
            raise ValueError(f"expected clean4 arrays, got {X.shape}")
        transformed = np.arcsinh(X.astype(np.float32, copy=False)).reshape(-1, X.shape[-2], X.shape[-1])
        values = transformed.astype(np.float64, copy=False)
        block_sum = values.sum(axis=0)
        block_square = np.square(values).sum(axis=0)
        total_sum = block_sum if total_sum is None else total_sum + block_sum
        total_square = block_square if total_square is None else total_square + block_square
        n_frames += len(values)
    if total_sum is None or total_square is None or n_frames < 1:
        raise ValueError("cannot fit preprocessing on an empty collection")
    mean = total_sum / n_frames
    variance = np.maximum(total_square / n_frames - np.square(mean), 0.0)
    std = np.sqrt(variance) + 1e-6
    return mean[None].astype(np.float32), std[None].astype(np.float32)


def block_mean_flat(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    transformed = (np.arcsinh(X.astype(np.float32, copy=False)) - mean) / std
    output = transformed.mean(axis=1).reshape(len(X), -1)
    if not np.isfinite(output).all():
        raise ValueError("non-finite RAW_SPATIAL_PCA input")
    return output.astype(np.float32, copy=False)


def fit_pca(
    X: np.ndarray,
    *,
    n_components: int | None = None,
    sample_ids: Sequence[str] | None = None,
    random_seed: int = PCA_RANDOM_SEED,
    oversamples: int = 10,
    power_iterations: int = 1,
) -> PCAProjection:
    """Fit deterministic randomized PCA without adding a sklearn dependency."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("PCA requires a 2D matrix with at least two samples")
    k = min(50, len(values) - 1) if n_components is None else int(n_components)
    k = min(k, len(values) - 1, values.shape[1])
    if k < 1:
        raise ValueError("PCA component count must be positive")
    mean = values.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    centered = values - mean
    q = min(len(values) - 1, values.shape[1], k + int(oversamples))
    rng = np.random.default_rng(int(random_seed))
    omega = rng.standard_normal((values.shape[1], q), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = centered @ omega
        if not np.isfinite(projected).all():
            raise ValueError("randomized PCA range finder produced non-finite values")
        for _ in range(int(power_iterations)):
            projected, _ = np.linalg.qr(projected, mode="reduced")
            projected = centered @ (centered.T @ projected)
            if not np.isfinite(projected).all():
                raise ValueError("randomized PCA power iteration produced non-finite values")
    basis, _ = np.linalg.qr(projected, mode="reduced")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        small = basis.T @ centered
    _u, singular, vt = np.linalg.svd(small, full_matrices=False)
    singular = singular[:k]
    components = vt[:k]
    explained = np.square(singular.astype(np.float64)) / (len(values) - 1)
    total_variance = np.square(centered.astype(np.float64)).sum() / (len(values) - 1)
    ratios = explained / total_variance if total_variance > 0 else np.zeros_like(explained)
    ids = tuple(str(value) for value in (sample_ids if sample_ids is not None else range(len(values))))
    if len(ids) != len(values):
        raise ValueError("sample_ids length differs from PCA fit rows")
    return PCAProjection(
        mean_=mean,
        components_=components.astype(np.float32),
        explained_variance_=explained.astype(np.float64),
        explained_variance_ratio_=ratios.astype(np.float64),
        singular_values_=singular.astype(np.float64),
        n_samples_seen_=len(values),
        n_features_in_=values.shape[1],
        fit_sample_ids_=ids,
        random_seed=int(random_seed),
    )


def save_pca_model(path: Path, model: PCAProjection, *, normalizer_mean: np.ndarray, normalizer_std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "normalizer_mean": normalizer_mean,
        "normalizer_std": normalizer_std,
        "representation": "RAW_SPATIAL_PCA",
        "scope": "all_sessions_descriptive_only",
        "strict_predictive_use_allowed": False,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def fit_common_raw_pca(
    data_by_session: Mapping[str, BlockSequenceData],
) -> tuple[PCAProjection, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    metadata = sample_metadata_table(data_by_session)
    arrays = [data_by_session[s].X for s in sorted(data_by_session, key=int)]
    mean, std = fit_frame_pixel_normalizer(arrays)
    flat = np.concatenate([block_mean_flat(X, mean, std) for X in arrays], axis=0)
    model = fit_pca(flat, sample_ids=metadata["block_id"].astype(str).tolist())
    coordinates = model.transform(flat)
    return model, coordinates, mean, std, metadata


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_l2_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    class_weight: str = "balanced",
    max_iter: int = 400,
) -> LinearProbeModel:
    """Fit fixed-C multinomial L2 logistic regression with balanced weights."""
    values = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    classes, encoded = np.unique(labels, return_inverse=True)
    n, d = values.shape
    k = len(classes)
    if k < 2 or n < k:
        raise ValueError("linear probe requires at least two represented classes")
    if class_weight != "balanced":
        raise ValueError("v7 freezes class_weight='balanced'")
    counts = np.bincount(encoded, minlength=k)
    if np.any(counts == 0):
        raise ValueError("a probe class is absent")
    weights = n / (k * counts[encoded])
    target = np.eye(k, dtype=np.float64)[encoded]
    regularization = 1.0 / (float(C) * weights.sum())

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        coef = theta[: k * d].reshape(k, d)
        intercept = theta[k * d :]
        probabilities = _softmax(values @ coef.T + intercept)
        loss = -np.sum(weights[:, None] * target * np.log(np.maximum(probabilities, 1e-15))) / weights.sum()
        loss += 0.5 * regularization * np.square(coef).sum()
        delta = weights[:, None] * (probabilities - target) / weights.sum()
        grad_coef = delta.T @ values + regularization * coef
        grad_intercept = delta.sum(axis=0)
        return float(loss), np.concatenate([grad_coef.ravel(), grad_intercept])

    initial = np.zeros(k * d + k, dtype=np.float64)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B", options={"maxiter": int(max_iter), "ftol": 1e-10})
    coef = result.x[: k * d].reshape(k, d)
    intercept = result.x[k * d :]
    return LinearProbeModel(classes, coef, intercept, float(C), class_weight, bool(result.success), int(result.nit))


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any]) -> dict[str, float]:
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    recalls = []
    f1s = []
    for label in labels:
        tp = int(np.sum((truth == label) & (pred == label)))
        fn = int(np.sum((truth == label) & (pred != label)))
        fp = int(np.sum((truth != label) & (pred == label)))
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.mean(truth == pred)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_F1": float(np.mean(f1s)),
    }


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[Any]) -> np.ndarray:
    index = {str(label): i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        matrix[index[str(truth)], index[str(pred)]] += 1
    return matrix


def cycle_grouped_session_folds(
    session_labels: np.ndarray,
    cycles: np.ndarray,
    *,
    n_folds: int = 5,
    seed: int = STATISTIC_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    sessions = sorted(np.unique(session_labels.astype(str)).tolist(), key=int)
    min_cycles = min(len(np.unique(cycles[session_labels.astype(str) == session])) for session in sessions)
    folds = min(int(n_folds), min_cycles)
    if folds < 2:
        raise ValueError("at least two complete cycles per session are required")
    assignment: dict[tuple[str, int], int] = {}
    rng = np.random.default_rng(int(seed))
    for session in sessions:
        values = np.unique(cycles[session_labels.astype(str) == session]).astype(int)
        values = rng.permutation(values)
        for i, cycle in enumerate(values):
            assignment[(session, int(cycle))] = i % folds
    output = []
    labels = session_labels.astype(str)
    for fold in range(folds):
        test = np.asarray([
            assignment[(str(session), int(cycle))] == fold
            for session, cycle in zip(labels, cycles)
        ])
        train_idx, test_idx = np.flatnonzero(~test), np.flatnonzero(test)
        if set(labels[train_idx]) != set(sessions) or set(labels[test_idx]) != set(sessions):
            raise AssertionError("every session must contribute train and test samples in each fold")
        train_keys = {(labels[i], int(cycles[i])) for i in train_idx}
        test_keys = {(labels[i], int(cycles[i])) for i in test_idx}
        if train_keys & test_keys:
            raise AssertionError("condition blocks from one cycle crossed train/test")
        output.append((train_idx, test_idx))
    return output


def cross_validated_probe(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    labels: Sequence[Any],
    max_iter: int = 400,
) -> tuple[dict[str, float], np.ndarray]:
    predictions = np.empty(len(y), dtype=np.asarray(y).dtype)
    seen = np.zeros(len(y), dtype=bool)
    for train_idx, test_idx in splits:
        model = fit_l2_logistic(X[train_idx], y[train_idx], C=1.0, class_weight="balanced", max_iter=max_iter)
        predictions[test_idx] = model.predict(X[test_idx])
        seen[test_idx] = True
    if not seen.all():
        raise AssertionError("probe did not produce exactly one OOF prediction per sample")
    return classification_metrics(y, predictions, labels), predictions


def _cycle_label_permutations(
    session_labels: np.ndarray,
    cycles: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    n_permutations: int,
    seed: int,
) -> Iterable[np.ndarray]:
    labels = session_labels.astype(str)
    cycle_keys = np.asarray([f"{s}:{int(c)}" for s, c in zip(labels, cycles)], dtype=object)
    unique_keys = np.unique(cycle_keys)
    original = {key: labels[np.flatnonzero(cycle_keys == key)[0]] for key in unique_keys}
    key_fold: dict[str, int] = {}
    for fold, (_train, test) in enumerate(splits):
        for key in np.unique(cycle_keys[test]):
            key_fold[str(key)] = fold
    rng = np.random.default_rng(int(seed))
    for _ in range(int(n_permutations)):
        mapping: dict[str, str] = {}
        for fold in range(len(splits)):
            keys = sorted((key for key in unique_keys if key_fold[str(key)] == fold), key=str)
            permuted = rng.permutation([original[key] for key in keys])
            mapping.update({str(key): str(value) for key, value in zip(keys, permuted)})
        yield np.asarray([mapping[str(key)] for key in cycle_keys])


def session_id_probe(
    X: np.ndarray,
    session_labels: np.ndarray,
    cycles: np.ndarray,
    *,
    n_permutations: int = N_SESSION_PERMUTATIONS,
    seed: int = STATISTIC_SEED,
    n_folds: int = 5,
    max_iter: int = 400,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    labels = sorted(np.unique(session_labels.astype(str)).tolist(), key=int)
    splits = cycle_grouped_session_folds(session_labels, cycles, n_folds=n_folds, seed=seed)
    observed, predictions = cross_validated_probe(X, session_labels.astype(str), splits, labels=labels, max_iter=max_iter)
    null = {metric: [] for metric in observed}
    for permuted in _cycle_label_permutations(
        session_labels, cycles, splits, n_permutations=n_permutations, seed=seed + 1
    ):
        metrics, _ = cross_validated_probe(X, permuted, splits, labels=labels, max_iter=max(60, max_iter // 2))
        for metric, value in metrics.items():
            null[metric].append(value)
    rows = []
    for metric, value in observed.items():
        null_values = np.asarray(null[metric], dtype=float)
        p = (1 + int(np.sum(null_values >= value))) / (len(null_values) + 1)
        rows.append({
            "representation": "RAW_SPATIAL_PCA",
            "metric": metric,
            "observed": value,
            "chance_reference": 1 / 9 if metric in ("accuracy", "balanced_accuracy") else np.nan,
            "n_folds": len(splits),
            "cv_grouping": "cycle_grouped_within_each_session",
            "classifier": "9-class L2 logistic regression",
            "C": 1.0,
            "class_weight": "balanced",
            "n_permutations": int(n_permutations),
            "permutation_unit": "complete_cycle_all_four_blocks",
            "permutation_p_greater_equal": float(p),
            "null_mean": float(null_values.mean()),
            "null_std": float(null_values.std(ddof=1)) if len(null_values) > 1 else 0.0,
        })
    matrix = confusion_counts(session_labels.astype(str), predictions, labels)
    pred_table = pd.DataFrame({
        "session": session_labels.astype(str),
        "cycle": cycles.astype(int),
        "predicted_session": predictions.astype(str),
    })
    return pd.DataFrame(rows), matrix, pred_table


def source_only_stimulus_probe(
    data_by_session: Mapping[str, BlockSequenceData],
    *,
    random_seed: int = PCA_RANDOM_SEED,
    max_iter: int = 400,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    sessions = sorted(data_by_session, key=int)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for target in sessions:
        sources = [session for session in sessions if session != target]
        source_arrays = [data_by_session[session].X for session in sources]
        mean, std = fit_frame_pixel_normalizer(source_arrays)
        source_flat = np.concatenate([block_mean_flat(X, mean, std) for X in source_arrays])
        source_y = np.concatenate([data_by_session[session].y for session in sources]).astype(int)
        source_ids = np.concatenate([
            data_by_session[session].metadata["block_id"].astype(str).to_numpy() for session in sources
        ])
        target_data = data_by_session[target]
        target_flat = block_mean_flat(target_data.X, mean, std)
        pca = fit_pca(source_flat, sample_ids=source_ids, random_seed=random_seed + int(target))
        scaler_mean = pca.transform(source_flat).mean(axis=0, keepdims=True)
        scaler_std = pca.transform(source_flat).std(axis=0, keepdims=True) + 1e-6
        source_features = (pca.transform(source_flat) - scaler_mean) / scaler_std
        target_features = (pca.transform(target_flat) - scaler_mean) / scaler_std
        model = fit_l2_logistic(source_features, source_y, C=1.0, class_weight="balanced", max_iter=max_iter)
        prediction = model.predict(target_features).astype(int)
        metrics = classification_metrics(target_data.y.astype(int), prediction, [0, 1])
        rows.append({
            "target": target,
            "BA": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "macro_F1": metrics["macro_F1"],
            "n_source_sessions": len(sources),
            "n_source_blocks": len(source_y),
            "n_target_blocks": len(target_data.y),
            "pca_components": pca.n_components_,
            "PCA_fit_scope": "source_sessions_only",
            "normalization_fit_scope": "source_sessions_only",
            "scaling_fit_scope": "source_sessions_only",
            "target_used_for_fit": False,
            "classifier": "binary L2 logistic regression",
            "C": 1.0,
            "class_weight": "balanced",
        })
        audits.append({
            "target": target,
            "source_sessions": ",".join(sources),
            "pca_fit_sample_ids": ";".join(pca.fit_sample_ids_),
            "target_ids_in_pca_fit": int(sum(str(value).startswith(f"session{target}_") for value in pca.fit_sample_ids_)),
            "target_used_for_normalization": False,
            "target_used_for_PCA": False,
            "target_used_for_scaling": False,
            "target_used_for_classifier_fit": False,
            "status": "PASS",
        })
    mean_row = {
        "target": "MEAN_9_TARGETS",
        "BA": float(np.mean([row["BA"] for row in rows])),
        "accuracy": float(np.mean([row["accuracy"] for row in rows])),
        "macro_F1": float(np.mean([row["macro_F1"] for row in rows])),
        "n_source_sessions": 8,
        "n_source_blocks": int(sum(row["n_source_blocks"] for row in rows)),
        "n_target_blocks": int(sum(row["n_target_blocks"] for row in rows)),
        "pca_components": int(min(row["pca_components"] for row in rows)),
        "PCA_fit_scope": "source_sessions_only_per_target",
        "normalization_fit_scope": "source_sessions_only_per_target",
        "scaling_fit_scope": "source_sessions_only_per_target",
        "target_used_for_fit": False,
        "classifier": "binary L2 logistic regression",
        "C": 1.0,
        "class_weight": "balanced",
    }
    return pd.DataFrame(rows + [mean_row]), audits


def multivariate_energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not len(x) or not len(y):
        raise ValueError("energy distance requires nonempty 2D arrays with matching features")
    value = 2.0 * cdist(x, y).mean() - cdist(x, x).mean() - cdist(y, y).mean()
    return float(max(value, 0.0))


def pairwise_session_distances(
    features: np.ndarray,
    session_labels: np.ndarray,
    *,
    representation: str,
    seed: int | str = "",
) -> pd.DataFrame:
    sessions = sorted(np.unique(session_labels.astype(str)).tolist(), key=int)
    rows = []
    for i, session_a in enumerate(sessions):
        xa = features[session_labels.astype(str) == session_a]
        for session_b in sessions[i + 1 :]:
            xb = features[session_labels.astype(str) == session_b]
            rows.append({
                "representation": representation,
                "seed": seed,
                "session_a": session_a,
                "session_b": session_b,
                "centroid_distance": float(np.linalg.norm(xa.mean(axis=0) - xb.mean(axis=0))),
                "energy_distance": multivariate_energy_distance(xa, xb),
            })
    return pd.DataFrame(rows)


def aggregate_masked_distance_rows(seed_rows: pd.DataFrame) -> pd.DataFrame:
    grouped = seed_rows.groupby(["session_a", "session_b"], as_index=False).agg(
        centroid_distance=("centroid_distance", "mean"),
        energy_distance=("energy_distance", "mean"),
        centroid_distance_seed_std=("centroid_distance", "std"),
        energy_distance_seed_std=("energy_distance", "std"),
    )
    grouped.insert(0, "seed", "MEAN_3_SEEDS")
    grouped.insert(0, "representation", "GLOBAL_MASKED_SMALLCNN")
    return grouped


def balanced_cycle_bootstrap_indices(
    metadata: pd.DataFrame,
    *,
    rng: np.random.Generator,
    n_cycles: int | None = None,
) -> np.ndarray:
    sessions = sorted(metadata["session"].astype(str).unique(), key=int)
    available = {
        session: np.sort(metadata.loc[metadata["session"].astype(str) == session, "cycle"].astype(int).unique())
        for session in sessions
    }
    draw_n = min(map(len, available.values())) if n_cycles is None else int(n_cycles)
    indices: list[int] = []
    for session in sessions:
        drawn = rng.choice(available[session], size=draw_n, replace=True)
        session_values = metadata["session"].astype(str).to_numpy()
        cycle_values = metadata["cycle"].astype(int).to_numpy()
        for cycle in drawn:
            block_indices = np.flatnonzero((session_values == session) & (cycle_values == int(cycle)))
            if len(block_indices) != 4:
                raise AssertionError("bootstrap cycle did not retain exactly four condition blocks")
            indices.extend(block_indices.tolist())
    return np.asarray(indices, dtype=np.int64)


def multivariate_factor_sums(
    features: np.ndarray,
    session: np.ndarray,
    factor: np.ndarray,
) -> dict[str, float]:
    """Balanced two-way multivariate ANOVA sums of squares."""
    X = np.asarray(features, dtype=np.float64)
    s = np.asarray(session).astype(str)
    f = np.asarray(factor).astype(str)
    sessions = sorted(np.unique(s).tolist(), key=int)
    factors = sorted(np.unique(f).tolist())
    counts = {(sv, fv): int(np.sum((s == sv) & (f == fv))) for sv in sessions for fv in factors}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"factor decomposition requires balanced cells: {counts}")
    n_cell = next(iter(counts.values()))
    grand = X.mean(axis=0)
    mean_s = {sv: X[s == sv].mean(axis=0) for sv in sessions}
    mean_f = {fv: X[f == fv].mean(axis=0) for fv in factors}
    mean_sf = {(sv, fv): X[(s == sv) & (f == fv)].mean(axis=0) for sv in sessions for fv in factors}
    ss_session = len(factors) * n_cell * sum(np.square(mean_s[sv] - grand).sum() for sv in sessions)
    ss_factor = len(sessions) * n_cell * sum(np.square(mean_f[fv] - grand).sum() for fv in factors)
    ss_interaction = n_cell * sum(
        np.square(mean_sf[(sv, fv)] - mean_s[sv] - mean_f[fv] + grand).sum()
        for sv in sessions for fv in factors
    )
    ss_residual = sum(
        np.square(X[(s == sv) & (f == fv)] - mean_sf[(sv, fv)]).sum()
        for sv in sessions for fv in factors
    )
    total = float(np.square(X - grand).sum())
    components = {
        "session": float(ss_session),
        "factor": float(ss_factor),
        "session_x_factor": float(ss_interaction),
        "residual": float(ss_residual),
        "total": total,
    }
    if not np.isfinite(list(components.values())).all():
        raise AssertionError("non-finite multivariate sums of squares")
    if not np.isclose(sum(components[key] for key in ("session", "factor", "session_x_factor", "residual")), total, rtol=1e-6, atol=1e-6):
        raise AssertionError("multivariate factor components do not sum to total")
    return components


def bootstrap_factor_decomposition(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    factor_column: str,
    representation: str,
    seed_label: int | str = "",
    n_bootstrap: int = N_BOOTSTRAP,
    random_seed: int = STATISTIC_SEED,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if factor_column not in ("stimulus_presence", "condition4"):
        raise ValueError("v7 permits only the preregistered binary or condition4 decomposition")
    rng = np.random.default_rng(int(random_seed))
    values = {key: [] for key in ("session", factor_column, f"session_x_{factor_column}", "residual")}
    min_cycles = min(metadata.groupby(metadata["session"].astype(str))["cycle"].nunique())
    for _ in range(int(n_bootstrap)):
        indices = balanced_cycle_bootstrap_indices(metadata, rng=rng, n_cycles=int(min_cycles))
        selected = metadata.iloc[indices]
        sums = multivariate_factor_sums(
            features[indices], selected["session"].astype(str).to_numpy(), selected[factor_column].astype(str).to_numpy()
        )
        total = sums["total"]
        mapping = {
            "session": sums["session"],
            factor_column: sums["factor"],
            f"session_x_{factor_column}": sums["session_x_factor"],
            "residual": sums["residual"],
        }
        for key, value in mapping.items():
            values[key].append(value / total if total > 0 else np.nan)
    rows = []
    arrays = {key: np.asarray(value, dtype=float) for key, value in values.items()}
    for factor_name, distribution in arrays.items():
        if not np.isfinite(distribution).all():
            raise AssertionError("factor bootstrap produced non-finite R2")
        rows.append({
            "representation": representation,
            "seed": seed_label,
            "factor": factor_name,
            "median_R2": float(np.median(distribution)),
            "ci_2_5": float(np.percentile(distribution, 2.5)),
            "ci_97_5": float(np.percentile(distribution, 97.5)),
            "mean_R2": float(distribution.mean()),
            "std_R2": float(distribution.std(ddof=1)) if len(distribution) > 1 else 0.0,
            "n_bootstrap": int(n_bootstrap),
            "cycles_per_session_per_bootstrap": int(min_cycles),
            "resampling_unit": "complete_cycle_all_four_conditions",
            "temporal_confound_warning": CONDITION_TIME_WARNING if factor_column == "condition4" else "",
        })
    return pd.DataFrame(rows), arrays


def aggregate_seed_factor_rows(seed_tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    values = pd.concat(seed_tables, ignore_index=True)
    grouped = values.groupby("factor", as_index=False).agg(
        median_R2=("median_R2", "mean"),
        ci_2_5=("ci_2_5", "mean"),
        ci_97_5=("ci_97_5", "mean"),
        mean_R2=("mean_R2", "mean"),
        std_R2=("median_R2", "std"),
        n_bootstrap=("n_bootstrap", "sum"),
        cycles_per_session_per_bootstrap=("cycles_per_session_per_bootstrap", "min"),
        temporal_confound_warning=("temporal_confound_warning", "first"),
    )
    grouped.insert(0, "seed", "MEAN_3_SEEDS")
    grouped.insert(0, "representation", "GLOBAL_MASKED_SMALLCNN")
    grouped["resampling_unit"] = "complete_cycle_all_four_conditions"
    return grouped


def exact_spearman_permutation(
    x: Sequence[float],
    y: Sequence[float],
    *,
    max_exact: int = math.factorial(9),
    n_permutations: int = 100_000,
    seed: int = STATISTIC_SEED,
) -> dict[str, Any]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("paired vectors with at least two values are required")
    rx = rankdata(x_values)
    ry = rankdata(y_values)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ry_centered = ry - ry.mean()
    ry_norm = np.linalg.norm(ry_centered)
    observed = float(np.dot(rx, ry_centered / ry_norm)) if ry_norm else float("nan")
    total_exact = math.factorial(len(y_values))
    extreme = 0
    total = 0
    if total_exact <= int(max_exact):
        for permutation in itertools.permutations(range(len(y_values))):
            permuted = ry_centered[np.asarray(permutation)]
            rho = float(np.dot(rx, permuted / ry_norm)) if ry_norm else float("nan")
            extreme += int(abs(rho) >= abs(observed) - 1e-12)
            total += 1
        method = "exact_complete_enumeration"
        p_value = extreme / total
        reason = "all target-label permutations enumerated"
    else:
        rng = np.random.default_rng(int(seed))
        for _ in range(int(n_permutations)):
            permuted = rng.permutation(ry_centered)
            rho = float(np.dot(rx, permuted / ry_norm)) if ry_norm else float("nan")
            extreme += int(abs(rho) >= abs(observed) - 1e-12)
        total = int(n_permutations)
        method = "monte_carlo_permutation"
        p_value = (extreme + 1) / (total + 1)
        reason = f"{len(y_values)}! exceeds max_exact={max_exact}"
    return {
        "rho": observed,
        "permutation_p_two_sided": float(p_value),
        "permutation_method": method,
        "n_permutations": total,
        "permutation_reason": reason,
    }


def mantel_session_label_permutation(
    distance: pd.DataFrame,
    pair_performance: pd.DataFrame,
    *,
    sessions: Sequence[str] = tuple(EXPECTED_SESSIONS),
) -> dict[str, Any]:
    sessions = tuple(str(value) for value in sessions)
    d_map = {
        tuple(sorted((str(row.session_a), str(row.session_b)), key=int)): float(row.energy_distance)
        for row in distance.itertuples()
    }
    p_map = {
        tuple(sorted((str(row.session_a), str(row.session_b)), key=int)): float(row.symmetric_cross_BA)
        for row in pair_performance.itertuples()
    }
    pairs = [tuple(sorted(pair, key=int)) for pair in itertools.combinations(sessions, 2)]
    if set(d_map) != set(pairs) or set(p_map) != set(pairs):
        raise ValueError("Mantel analysis requires all unordered session pairs")
    d = np.asarray([d_map[pair] for pair in pairs])
    p = np.asarray([p_map[pair] for pair in pairs])
    d_rank = rankdata(d)
    p_rank = rankdata(p)
    p_centered = p_rank - p_rank.mean()
    p_norm = np.linalg.norm(p_centered)
    d_centered = d_rank - d_rank.mean()
    d_norm = np.linalg.norm(d_centered)
    observed = float(np.dot(d_centered / d_norm, p_centered / p_norm))
    extreme = 0
    total = 0
    d_matrix = np.zeros((len(sessions), len(sessions)), dtype=float)
    for rank_value, (a, b) in zip(d_rank, pairs):
        i, j = sessions.index(a), sessions.index(b)
        d_matrix[i, j] = d_matrix[j, i] = rank_value
    upper = np.triu_indices(len(sessions), 1)
    for permutation in itertools.permutations(range(len(sessions))):
        permuted = d_matrix[np.ix_(permutation, permutation)][upper]
        centered = permuted - permuted.mean()
        rho = float(np.dot(centered / np.linalg.norm(centered), p_centered / p_norm))
        extreme += int(abs(rho) >= abs(observed) - 1e-12)
        total += 1
    return {
        "rho": observed,
        "permutation_p_two_sided": extreme / total,
        "permutation_method": "exact_mantel_session_label_enumeration",
        "n_permutations": total,
        "permutation_reason": "all 9! session-label permutations enumerated",
    }


def target_mean_energy_distance(distance_rows: pd.DataFrame) -> pd.DataFrame:
    sessions = sorted(set(distance_rows["session_a"].astype(str)) | set(distance_rows["session_b"].astype(str)), key=int)
    rows = []
    for session in sessions:
        selected = distance_rows[
            (distance_rows["session_a"].astype(str) == session)
            | (distance_rows["session_b"].astype(str) == session)
        ]
        rows.append({"target": session, "mean_distance_to_other_sessions": float(selected["energy_distance"].mean())})
    return pd.DataFrame(rows)


def find_artifact(root: Path, relative: str) -> Path:
    direct = root / relative
    if direct.is_file():
        return direct
    matches = sorted(root.glob(f"**/{relative}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"artifact not found under {root}: {relative}")
    # Prefer the shallowest completed run when Finder-created copies coexist.
    matches.sort(key=lambda value: (len(value.parts), str(value)))
    return matches[0]


def audit_v5_cross_session_metrics(v5_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = find_artifact(v5_root, "downstream/fold_metrics.csv")
    metrics = pd.read_csv(path)
    required = {
        "task", "target_session", "condition", "seed", "test_balanced_accuracy",
        "target_labels_used_for_training", "target_frames_used_for_training",
        "target_used_for_normalization", "target_used_for_validation", "target_used_for_model_selection",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise AssertionError(f"v5 artifact lacks fields: {sorted(missing)}")
    selected = metrics[
        (metrics["task"].astype(str) == "binary")
        & (metrics["condition"].astype(str) == "MULTI_SOURCE_BALANCED")
    ].copy()
    rows = []
    target_values = []
    for target in EXPECTED_SESSIONS:
        subset = selected[selected["target_session"].astype(str) == target]
        seeds = sorted(subset["seed"].astype(int).unique().tolist())
        strict = (
            len(subset) == len(GLOBAL_ENCODER_SEEDS)
            and seeds == list(GLOBAL_ENCODER_SEEDS)
            and not subset["target_labels_used_for_training"].astype(bool).any()
            and int(subset["target_frames_used_for_training"].astype(int).sum()) == 0
            and not subset["target_used_for_normalization"].astype(bool).any()
            and not subset["target_used_for_validation"].astype(bool).any()
            and not subset["target_used_for_model_selection"].astype(bool).any()
        )
        rows.append({
            "target": target,
            "artifact": str(path),
            "task": "binary",
            "condition": "MULTI_SOURCE_BALANCED",
            "n_seed_rows": len(subset),
            "seeds": ",".join(map(str, seeds)),
            "same_clean4_binary_task": bool(len(subset) and strict),
            "strict_target_held_out": bool(strict),
            "comparable_labels": bool(len(subset) and strict),
            "MULTI_SOURCE_ERM_BA": float(subset["test_balanced_accuracy"].mean()) if len(subset) else np.nan,
            "status": "PASS" if strict else "FAIL",
        })
        if len(subset):
            target_values.append({
                "session": target,
                "v5_cross_session_BA": float(subset["test_balanced_accuracy"].mean()),
                "v5_cross_session_BA_seed_std": float(subset["test_balanced_accuracy"].std(ddof=1)),
            })
    audit = pd.DataFrame(rows)
    if len(audit) != 9 or (audit["status"] != "PASS").any():
        raise AssertionError("v5 strict multi-source metric audit failed")
    return audit, pd.DataFrame(target_values)


def audit_within_session_metrics(v1_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = find_artifact(v1_root, "downstream/fold_metrics.csv")
    metrics = pd.read_csv(path)
    required = {
        "session", "task", "condition", "seed", "fold", "test_balanced_accuracy",
        "encoder_requires_grad", "decoder_present", "best_epoch",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise AssertionError(f"within-session artifact lacks fields: {sorted(missing)}")
    selected = metrics[
        (metrics["task"].astype(str) == "binary")
        & (metrics["condition"].astype(str) == "RANDOM_INIT")
        & metrics["seed"].astype(int).isin(GLOBAL_ENCODER_SEEDS)
    ].copy()
    rows = []
    values = []
    for session in EXPECTED_SESSIONS:
        subset = selected[selected["session"].astype(str) == session]
        fold_counts = subset.groupby("seed")["fold"].nunique()
        valid = (
            len(subset) > 0
            and sorted(subset["seed"].astype(int).unique()) == list(GLOBAL_ENCODER_SEEDS)
            and fold_counts.nunique() == 1
            and int(fold_counts.iloc[0]) >= 2
            and subset["encoder_requires_grad"].astype(bool).all()
            and not subset["decoder_present"].astype(bool).any()
            and (subset["best_epoch"].astype(int) == 40).all()
        )
        seed_means = subset.groupby("seed")["test_balanced_accuracy"].mean()
        rows.append({
            "session": session,
            "artifact": str(path),
            "task": "binary",
            "condition": "RANDOM_INIT",
            "smallcnn_feature_mean": True,
            "n_rows": len(subset),
            "n_folds_per_seed": int(fold_counts.iloc[0]) if len(fold_counts) else 0,
            "n_seeds": int(len(seed_means)),
            "frozen_method_match": bool(valid),
            "status": "PASS" if valid else "FAIL",
        })
        if len(seed_means):
            values.append({
                "session": session,
                "within_session_BA": float(seed_means.mean()),
                "within_session_BA_seed_std": float(seed_means.std(ddof=1)),
            })
    audit = pd.DataFrame(rows)
    if len(audit) != 9 or (audit["status"] != "PASS").any():
        raise AssertionError("formal within-session metric audit failed")
    return audit, pd.DataFrame(values)


def audit_pairwise_cross_session(v5_root: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    path = find_artifact(v5_root, "downstream/fold_metrics.csv")
    metrics = pd.read_csv(path)
    required = {
        "task", "target_session", "condition", "source_sessions", "seed", "test_balanced_accuracy",
        "target_labels_used_for_training", "target_frames_used_for_training",
    }
    if required - set(metrics.columns):
        audit = pd.DataFrame([{
            "artifact": str(path), "expected_directed_pairs": 72, "available_directed_pairs": 0,
            "comparable": False, "status": "PAIRWISE_ANALYSIS_NOT_RUN",
            "reason": "required pairwise provenance fields absent",
        }])
        return audit, None
    subset = metrics[
        (metrics["task"].astype(str) == "binary")
        & (metrics["condition"].astype(str) == "SINGLE_SOURCE_TRANSFER")
    ].copy()
    subset["source_session"] = subset["source_sessions"].astype(str)
    directed = subset[["source_session", "target_session"]].drop_duplicates()
    strict = (
        len(directed) == 72
        and set(directed["source_session"].astype(str)) == set(EXPECTED_SESSIONS)
        and set(directed["target_session"].astype(str)) == set(EXPECTED_SESSIONS)
        and not subset["target_labels_used_for_training"].astype(bool).any()
        and int(subset["target_frames_used_for_training"].astype(int).sum()) == 0
        and (subset.groupby(["source_session", "target_session"])["seed"].nunique() == 3).all()
    )
    audit = pd.DataFrame([{
        "artifact": str(path),
        "expected_directed_pairs": 72,
        "available_directed_pairs": len(directed),
        "seeds_per_pair": 3 if strict else int(subset.groupby(["source_session", "target_session"])["seed"].nunique().min()) if len(subset) else 0,
        "task": "binary",
        "condition": "SINGLE_SOURCE_TRANSFER",
        "clean4_smallcnn_comparable": bool(strict),
        "strict_target_held_out": bool(strict),
        "comparable": bool(strict),
        "status": "PASS_RUN_SECONDARY" if strict else "PAIRWISE_ANALYSIS_NOT_RUN",
        "reason": "72 directed clean4 SmallCNN pairs are directly comparable" if strict else "complete comparable 72-pair artifact unavailable",
    }])
    if not strict:
        return audit, None
    directed_mean = subset.groupby(["source_session", "target_session"], as_index=False)["test_balanced_accuracy"].mean()
    lookup = {
        (str(row.source_session), str(row.target_session)): float(row.test_balanced_accuracy)
        for row in directed_mean.itertuples()
    }
    rows = []
    for a, b in itertools.combinations(EXPECTED_SESSIONS, 2):
        rows.append({
            "session_a": a,
            "session_b": b,
            "A_to_B_BA": lookup[(a, b)],
            "B_to_A_BA": lookup[(b, a)],
            "symmetric_cross_BA": float(np.mean([lookup[(a, b)], lookup[(b, a)]])),
        })
    return audit, pd.DataFrame(rows)


def session_separability(features: np.ndarray, metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session in sorted(metadata["session"].astype(str).unique(), key=int):
        mask = metadata["session"].astype(str).to_numpy() == session
        X = features[mask]
        labels = metadata.loc[mask, "stimulus_presence"].astype(str).to_numpy()
        stimulus_centroid = X[labels == "stimulus"].mean(axis=0)
        no_stimulus_centroid = X[labels == "no_stimulus"].mean(axis=0)
        between = float(np.linalg.norm(stimulus_centroid - no_stimulus_centroid))
        distances = np.concatenate([
            np.linalg.norm(X[labels == "stimulus"] - stimulus_centroid, axis=1),
            np.linalg.norm(X[labels == "no_stimulus"] - no_stimulus_centroid, axis=1),
        ])
        within = float(distances.mean())
        rows.append({
            "session": session,
            "between_condition_distance": between,
            "within_condition_dispersion": within,
            "separability_ratio": between / within if within > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def cycle_consistency_from_pool(
    pool: SessionFramePool,
    *,
    normalizer_mean: np.ndarray,
    normalizer_std: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for session in pool.source_sessions:
        frames = apply_ssl_frame_normalizer(pool.frames_by_session[session], normalizer_mean, normalizer_std)
        cycles = pool.cycles_by_session[session]
        unique = np.sort(np.unique(cycles))
        means = np.stack([frames[cycles == cycle].mean(axis=0).ravel() for cycle in unique])
        correlations = np.corrcoef(means)
        pair_values = correlations[np.triu_indices(len(unique), 1)]
        if not np.isfinite(pair_values).all():
            raise AssertionError("cycle consistency contains non-finite correlations")
        rows.append({
            "session": session,
            "n_cycles": len(unique),
            "n_cycle_pairs": len(pair_values),
            "cycle_consistency_mean": float(pair_values.mean()),
            "cycle_consistency_median": float(np.median(pair_values)),
            "cycle_consistency_std": float(pair_values.std(ddof=1)) if len(pair_values) > 1 else 0.0,
            "metric_name": "within-session repeatability / cycle consistency",
        })
    return pd.DataFrame(rows)


def build_global_unlabeled_pool(
    project_dir: Path,
    sessions: Sequence[str],
    *,
    data_dir: Path | None = None,
    max_cycles_per_session: int | None = None,
) -> SessionFramePool:
    base = data_dir or default_block_data_dir(project_dir)
    frames_by_session: dict[str, np.ndarray] = {}
    cycles_by_session: dict[str, np.ndarray] = {}
    indices_by_session: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for session in sessions:
        path = base / f"session_{session}_blocks.h5"
        complete = complete_cycles_from_unlabeled_h5(path)
        if max_cycles_per_session is not None:
            complete = complete[: int(max_cycles_per_session)]
        data = load_unlabeled_cycles(path, complete)
        frames_by_session[session] = data.frames
        cycles_by_session[session] = data.cycles
        indices_by_session[session] = data.original_frame_indices
        paths[session] = path
    return SessionFramePool(frames_by_session, cycles_by_session, indices_by_session, paths)


def global_encoder_expected_config(
    pool: SessionFramePool,
    *,
    seed: int,
    updates: int,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "representation": "GLOBAL_MASKED_SMALLCNN",
        "usage": "descriptive label-free common representation",
        "encoder_class": "SmallCNNFrameEncoder",
        "architecture_fingerprint": architecture_fingerprint(),
        "seed": int(seed),
        "source_sessions": list(pool.source_sessions),
        "source_cycles_by_session": {
            session: sorted(np.unique(pool.cycles_by_session[session]).astype(int).tolist())
            for session in pool.source_sessions
        },
        "pretraining_config": asdict(SSLPretrainingConfig()),
        "updates": int(updates),
        "batch_size": int(batch_size),
        "session_balanced": True,
        "contains_labels": False,
        "decoder_discarded": True,
    }


def save_global_encoder_checkpoint(
    path: Path,
    *,
    encoder: SmallCNNFrameEncoder,
    normalization_mean: np.ndarray,
    normalization_std: np.ndarray,
    expected_config: Mapping[str, Any],
    sampling_counts: Mapping[str, int],
    history: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    peak_gpu_memory_mb: float,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(expected_config)
    payload.update({
        "encoder_state_dict": {key: value.detach().cpu() for key, value in encoder.state_dict().items()},
        "normalization_mean": normalization_mean,
        "normalization_std": normalization_std,
        "sampling_counts": dict(sampling_counts),
        "training_history": list(history),
        "runtime_seconds": float(runtime_seconds),
        "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
    })
    forbidden = [key for key in payload if "label" in key.lower() and key != "contains_labels"]
    if forbidden or payload["contains_labels"] is not False:
        raise AssertionError("label information entered global encoder checkpoint")
    torch.save(payload, path)
    return sha256_file(path)


def load_or_train_global_encoder(
    path: Path,
    pool: SessionFramePool,
    *,
    seed: int,
    updates: int,
    batch_size: int,
    device: str,
) -> tuple[SmallCNNFrameEncoder, dict[str, Any], dict[str, Any]]:
    expected = global_encoder_expected_config(pool, seed=seed, updates=updates, batch_size=batch_size)
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
        if not mismatches:
            encoder = SmallCNNFrameEncoder()
            encoder.load_state_dict(payload["encoder_state_dict"])
            audit = {
                "seed": seed,
                "checkpoint": str(path),
                "checkpoint_sha256": sha256_file(path),
                "reused": True,
                "config_match": True,
                "contains_labels": payload.get("contains_labels"),
                "session_balanced": payload.get("session_balanced"),
                "reason": "exact architecture/config/session/cycle checkpoint match",
                "status": "PASS_REUSED",
            }
            return encoder, payload, audit
    mean, std = fit_ssl_pool_normalizer(pool)
    result = pretrain_session_balanced_smallcnn(
        pool,
        seed=int(seed),
        reference_updates=int(updates),
        actual_batch_size=int(batch_size),
        config=SSLPretrainingConfig(),
        device=device,
        normalization_stats=(mean, std),
    )
    digest = save_global_encoder_checkpoint(
        path,
        encoder=result.encoder,
        normalization_mean=result.normalization_mean,
        normalization_std=result.normalization_std,
        expected_config=expected,
        sampling_counts=result.sampling_counts,
        history=result.history,
        runtime_seconds=result.runtime_seconds,
        peak_gpu_memory_mb=result.peak_gpu_memory_mb,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    audit = {
        "seed": seed,
        "checkpoint": str(path),
        "checkpoint_sha256": digest,
        "reused": False,
        "config_match": True,
        "contains_labels": False,
        "session_balanced": True,
        "reason": "trained new exact frozen global label-free encoder",
        "status": "PASS_TRAINED",
    }
    return result.encoder, payload, audit


def extract_masked_block_features(
    encoder: nn.Module,
    X: np.ndarray,
    *,
    normalizer_mean: np.ndarray,
    normalizer_std: np.ndarray,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    normalized = apply_ssl_frame_normalizer(
        X.reshape(-1, X.shape[-2], X.shape[-1]), normalizer_mean, normalizer_std
    )
    torch_device = torch.device(device)
    model = encoder.to(torch_device).eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(normalized), int(batch_size)):
            batch = torch.from_numpy(normalized[start : start + batch_size, None]).to(torch_device)
            outputs.append(model(batch).detach().cpu().numpy())
    features = np.concatenate(outputs).reshape(len(X), X.shape[1], -1).mean(axis=1)
    return features.astype(np.float32, copy=False)


def missing_formal_outputs(output_dir: Path) -> list[str]:
    missing = [relative for relative in REQUIRED_OUTPUTS if not (output_dir / relative).is_file()]
    for seed in GLOBAL_ENCODER_SEEDS:
        if not (output_dir / f"features/masked_smallcnn_features/seed_{seed}.csv").is_file():
            missing.append(f"features/masked_smallcnn_features/seed_{seed}.csv")
    return missing
