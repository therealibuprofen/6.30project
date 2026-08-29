from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_frame_tensor,
    labels_to_class_indices,
    set_reproducible_seed,
)


MODEL_NAME = "fcnn_cycle_calibrated_late_fusion"
MODEL_VERSION = "fcnn_cycle_calibrated_late_fusion_v1.0.0"
HISTORICAL_METHOD = "fcnn_late_fusion"
N_INNER_FOLDS = 3
FRAMES_PER_BLOCK = 4
IMAGE_SHAPE = (128, 501)
CLASSES = np.asarray([0, 1], dtype=np.int64)
FORMAL_TRAINING_CONFIG = DeepTrainingConfig(
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=16,
    max_epochs=40,
    dropout=0.25,
    loss="cross_entropy",
)
TEMPERATURE_LOG_BOUNDS = (-20.0, 20.0)
ECE_BINS = 10

STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")

FROZEN_GATE = {
    "A_overall_ba_delta_min": 0.005,
    "B_strong_ba_delta_min": -0.01,
    "C_weak_ba_delta_min": 0.01,
    "D_overall_ece_ratio_max": 0.80,
}


@dataclass(frozen=True)
class InnerCycleSplit:
    inner_fold: int
    train_cycles: tuple[int, ...]
    validation_cycles: tuple[int, ...]


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    log_temperature: float
    pre_nll: float
    post_nll: float
    pre_ece: float
    post_ece: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    objective: str = "cross_entropy_nll"
    parameterization: str = "temperature=exp(log_temperature)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def parse_cycle_ids(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [] if not value.strip() else [int(part) for part in value.split(",")]
    else:
        values = [int(item) for item in value]
    result = tuple(sorted(set(values)))
    if len(result) != len(values):
        raise ValueError("cycle IDs must be unique")
    return result


def cycle_text(values: Iterable[int]) -> str:
    return ",".join(str(value) for value in parse_cycle_ids(values))


def build_inner_cycle_splits(
    outer_train_cycles: str | Iterable[int],
    outer_test_cycles: str | Iterable[int],
    *,
    n_splits: int = N_INNER_FOLDS,
) -> tuple[InnerCycleSplit, ...]:
    """Build deterministic cycle-grouped cross-fitting folds inside outer train."""

    train = parse_cycle_ids(outer_train_cycles)
    test = parse_cycle_ids(outer_test_cycles)
    if int(n_splits) != N_INNER_FOLDS:
        raise ValueError("formal CCLF uses exactly three inner folds")
    if set(train) & set(test):
        raise AssertionError("outer train and outer test cycles overlap")
    if len(train) < n_splits:
        raise ValueError("outer training set has fewer cycles than inner folds")

    validation_parts = [
        tuple(int(value) for value in part.tolist())
        for part in np.array_split(np.asarray(train, dtype=np.int64), n_splits)
    ]
    splits: list[InnerCycleSplit] = []
    for inner_fold, validation in enumerate(validation_parts, start=1):
        inner_train = tuple(value for value in train if value not in set(validation))
        if not validation or not inner_train:
            raise AssertionError("an inner fold is empty")
        if set(inner_train) & set(validation):
            raise AssertionError("inner train/validation cycles overlap")
        if (set(inner_train) | set(validation)) != set(train):
            raise AssertionError("inner fold does not partition outer train cycles")
        if (set(inner_train) | set(validation)) & set(test):
            raise AssertionError("outer-test cycle entered inner calibration")
        splits.append(
            InnerCycleSplit(
                inner_fold=inner_fold,
                train_cycles=inner_train,
                validation_cycles=validation,
            )
        )
    observed = [cycle for split in splits for cycle in split.validation_cycles]
    if sorted(observed) != list(train) or len(observed) != len(set(observed)):
        raise AssertionError("each outer-training cycle must be inner-held-out once")
    return tuple(splits)


def fit_inner_train_normalization(
    inner_train_blocks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Fit arcsinh pixel z-score using only inner-training frames."""

    blocks = np.asarray(inner_train_blocks, dtype=np.float32)
    if blocks.ndim != 4 or blocks.shape[1:] != (FRAMES_PER_BLOCK, *IMAGE_SHAPE):
        raise ValueError("inner training blocks must have shape [N,4,128,501]")
    if len(blocks) == 0 or not np.isfinite(blocks).all():
        raise ValueError("inner training blocks must be nonempty and finite")
    transformed = np.arcsinh(blocks)
    frames = transformed.reshape(-1, *IMAGE_SHAPE).astype(np.float64, copy=False)
    mean = frames.mean(axis=0, keepdims=True).astype(np.float32)
    std = (frames.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise AssertionError("inner normalization is invalid")
    normalization_fingerprint = fingerprint(
        {
            "transform": "arcsinh_then_inner_train_pixel_zscore",
            "mean_sha256": hashlib.sha256(mean.tobytes(order="C")).hexdigest(),
            "std_sha256": hashlib.sha256(std.tobytes(order="C")).hexdigest(),
            "n_train_blocks": int(len(blocks)),
            "n_train_frames": int(len(blocks) * FRAMES_PER_BLOCK),
        }
    )
    return mean, std, normalization_fingerprint


def apply_inner_normalization(
    blocks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    values = np.asarray(blocks, dtype=np.float32)
    saved_mean = np.asarray(mean, dtype=np.float32)
    saved_std = np.asarray(std, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:] != (FRAMES_PER_BLOCK, *IMAGE_SHAPE):
        raise ValueError("blocks must have shape [N,4,128,501]")
    if saved_mean.shape != (1, *IMAGE_SHAPE) or saved_std.shape != (1, *IMAGE_SHAPE):
        raise ValueError("normalization arrays must have shape [1,128,501]")
    if np.any(saved_std <= 0):
        raise ValueError("normalization std must be positive")
    normalized = (np.arcsinh(values) - saved_mean) / saved_std
    if not np.isfinite(normalized).all():
        raise AssertionError("normalized frames are non-finite")
    return normalized.astype(np.float32, copy=False)


def build_inner_cache_key(
    *,
    session: str,
    outer_fold: int,
    outer_seed: int,
    outer_train_cycles: Sequence[int],
    inner_fold: int,
    inner_train_cycles: Sequence[int],
    inner_validation_cycles: Sequence[int],
    source_hash: str,
    protocol_hash: str,
    normalization_fingerprint: str,
    training_config: Mapping[str, Any],
) -> str:
    required_strings = (source_hash, protocol_hash, normalization_fingerprint)
    if any(not str(value).strip() for value in required_strings):
        raise ValueError("cache identity hashes must be nonempty")
    return fingerprint(
        {
            "session": str(session),
            "outer_fold": int(outer_fold),
            "outer_seed": int(outer_seed),
            "outer_train_cycles": list(parse_cycle_ids(outer_train_cycles)),
            "inner_fold": int(inner_fold),
            "inner_train_cycles": list(parse_cycle_ids(inner_train_cycles)),
            "inner_validation_cycles": list(parse_cycle_ids(inner_validation_cycles)),
            "source_hash": str(source_hash),
            "protocol_hash": str(protocol_hash),
            "normalization_fingerprint": str(normalization_fingerprint),
            "model": "MaxPool2d-Flatten-Linear16000x3-ReLU-Linear3x2",
            "training_config": dict(training_config),
        }
    )


def predict_raw_logits(
    model: torch.nn.Module,
    normalized_blocks: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int,
) -> np.ndarray:
    torch_device = device if isinstance(device, torch.device) else torch.device(device)
    blocks = np.asarray(normalized_blocks, dtype=np.float32)
    if blocks.ndim != 4 or blocks.shape[1:] != (FRAMES_PER_BLOCK, *IMAGE_SHAPE):
        raise ValueError("normalized blocks must have shape [N,4,128,501]")
    frames = blocks.reshape(-1, *IMAGE_SHAPE)
    tensor = torch.from_numpy(frames[:, None, :, :])
    logits: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tensor), int(batch_size)):
            value = model(tensor[start : start + int(batch_size)].to(torch_device))
            if value.ndim != 2 or value.shape[1] != 2:
                raise AssertionError("FCNN must emit two logits per frame")
            logits.append(value.detach().cpu().numpy())
    result = np.concatenate(logits, axis=0) if logits else np.empty((0, 2), dtype=np.float32)
    if result.shape != (len(blocks) * FRAMES_PER_BLOCK, 2):
        raise AssertionError("raw-logit frame coverage is incomplete")
    if not np.isfinite(result).all():
        raise AssertionError("raw logits are non-finite")
    return result.astype(np.float64, copy=False)


def train_inner_fcnn(
    inner_train_blocks: np.ndarray,
    inner_train_labels: np.ndarray,
    validation_blocks: np.ndarray,
    *,
    seed: int,
    device: str,
    training_config: DeepTrainingConfig = FORMAL_TRAINING_CONFIG,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    """Train one historical-protocol FCNN and return held-out raw frame logits."""

    if str(device) not in {"cpu", "cuda"} and not str(device).startswith("cuda:"):
        raise ValueError("device must be cpu or cuda")
    mean, std, normalization_fp = fit_inner_train_normalization(inner_train_blocks)
    train_norm = apply_inner_normalization(inner_train_blocks, mean, std)
    validation_norm = apply_inner_normalization(validation_blocks, mean, std)
    set_reproducible_seed(int(seed))
    torch_device = torch.device(device)
    model = FCNN(input_shape=IMAGE_SHAPE, n_classes=2).to(torch_device)
    train_tensor = blocks_to_frame_tensor(train_norm)
    train_labels = labels_to_class_indices(
        np.repeat(np.asarray(inner_train_labels, dtype=np.int64), FRAMES_PER_BLOCK),
        CLASSES,
    )
    # Use the same fixed optimizer/loss/batching loop as the historical runner.
    from ultrasound_decoding.multiframe.training import _train_epochs

    history = _train_epochs(
        model,
        train_tensor,
        train_labels,
        config=training_config,
        seed=int(seed),
        device=torch_device,
        batch_size_reference=len(train_labels),
    )
    logits = predict_raw_logits(
        model,
        validation_norm,
        device=torch_device,
        batch_size=training_config.batch_size,
    )
    return logits, history, normalization_fp


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("logits must be a finite [N,2] array")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
    ):
        raise AssertionError("softmax probabilities are invalid")
    return probabilities


def top_label_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = ECE_BINS,
) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.shape != (len(truth), 2) or len(truth) == 0:
        raise ValueError("probabilities/labels have invalid shape")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == truth
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    total = 0.0
    for index in range(int(n_bins)):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence >= lower) & (
            confidence <= upper if index == int(n_bins) - 1 else confidence < upper
        )
        if selected.any():
            total += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(total)


def multiclass_nll(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.shape != (len(truth), 2) or len(truth) == 0:
        raise ValueError("probabilities/labels have invalid shape")
    selected = np.clip(probs[np.arange(len(truth)), truth], 1e-15, 1.0)
    return float(-np.log(selected).mean())


def binary_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if probs.shape != (len(truth), 2) or len(truth) == 0:
        raise ValueError("probabilities/labels have invalid shape")
    return float(np.mean((probs[:, 1] - truth) ** 2))


def _temperature_objective(log_temperature: np.ndarray, logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    value = float(log_temperature[0])
    inverse_temperature = float(np.exp(-value))
    scaled = logits * inverse_temperature
    rows = np.arange(len(labels))
    loss = float(np.mean(logsumexp(scaled, axis=1) - scaled[rows, labels]))
    probabilities = softmax_probabilities(scaled)
    expected_scaled_logit = np.sum(probabilities * scaled, axis=1)
    gradient = float(np.mean(scaled[rows, labels] - expected_scaled_logit))
    return loss, np.asarray([gradient], dtype=np.float64)


def fit_scalar_temperature(logits: np.ndarray, labels: np.ndarray) -> TemperatureFit:
    """Fit one positive scalar temperature by deterministic frame-NLL minimization."""

    values = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or values.shape != (len(truth), 2) or len(truth) == 0:
        raise ValueError("logits/labels have invalid shape")
    if not np.isfinite(values).all() or not set(np.unique(truth)).issubset({0, 1}):
        raise ValueError("temperature inputs must be finite binary observations")
    result = minimize(
        fun=lambda x: _temperature_objective(x, values, truth),
        x0=np.asarray([0.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[TEMPERATURE_LOG_BOUNDS],
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 1000, "maxls": 50},
    )
    if not bool(result.success):
        raise AssertionError(
            f"temperature NLL optimizer failed: status={result.status} {result.message}"
        )
    log_temperature = float(result.x[0])
    temperature = float(np.exp(log_temperature))
    if not np.isfinite(temperature) or temperature <= 0:
        raise AssertionError("temperature optimizer returned invalid T")
    pre = softmax_probabilities(values)
    post = calibrated_frame_probabilities(values, temperature)
    pre_nll = multiclass_nll(pre, truth)
    post_nll = multiclass_nll(post, truth)
    if post_nll > pre_nll + 1e-10:
        raise AssertionError("NLL-only optimizer worsened its fitted objective")
    return TemperatureFit(
        temperature=temperature,
        log_temperature=log_temperature,
        pre_nll=pre_nll,
        post_nll=post_nll,
        pre_ece=top_label_ece(pre, truth),
        post_ece=top_label_ece(post, truth),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),
    )


def calibrated_frame_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    return softmax_probabilities(np.asarray(logits, dtype=np.float64) / float(temperature))


def equal_four_frame_probability_mean(frame_probabilities: np.ndarray) -> np.ndarray:
    """The only formal fusion rule: arithmetic mean of exactly four frame probabilities."""

    values = np.asarray(frame_probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (FRAMES_PER_BLOCK, 2):
        raise ValueError("frame probabilities must have shape [N,4,2]")
    if not np.isfinite(values).all() or not np.allclose(values.sum(axis=2), 1.0, atol=1e-10):
        raise ValueError("frame probabilities are invalid")
    result = np.mean(values, axis=1)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-10):
        raise AssertionError("equal four-frame mean is invalid")
    return result


def assert_complete_inner_oof(
    source_indices: np.ndarray,
    expected_source_indices: np.ndarray,
    source_cycles: np.ndarray,
    validation_cycle_by_row: np.ndarray,
) -> None:
    observed = np.asarray(source_indices, dtype=np.int64)
    expected = np.asarray(expected_source_indices, dtype=np.int64)
    cycles = np.asarray(source_cycles, dtype=np.int64)
    validation = np.asarray(validation_cycle_by_row, dtype=np.int64)
    if len(observed) != len(expected) * FRAMES_PER_BLOCK:
        raise AssertionError("inner OOF frame coverage count differs")
    unique, counts = np.unique(observed, return_counts=True)
    if not np.array_equal(np.sort(unique), np.sort(expected)) or not np.all(counts == FRAMES_PER_BLOCK):
        raise AssertionError("each outer-training block must have exactly four inner OOF frames")
    if len(cycles) != len(observed) or len(validation) != len(observed):
        raise AssertionError("inner OOF membership arrays differ in length")
    if not np.array_equal(cycles, validation):
        raise AssertionError("an inner OOF frame was not predicted by its cycle-held-out model")


def frame_calibration_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    predicted = probs.argmax(axis=1)
    correct = (predicted == truth).astype(np.int64)
    confidence = probs.max(axis=1)
    auroc = float("nan") if len(np.unique(correct)) < 2 else float(roc_auc_score(correct, confidence))
    ap = float("nan") if len(np.unique(correct)) < 2 else float(average_precision_score(correct, confidence))
    return {
        "n_frames": int(len(truth)),
        "ece": top_label_ece(probs, truth),
        "brier": binary_brier(probs, truth),
        "nll": multiclass_nll(probs, truth),
        "frame_accuracy": float(correct.mean()),
        "confidence_correctness_auroc": auroc,
        "confidence_correctness_average_precision": ap,
    }


def block_balanced_accuracy(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(balanced_accuracy_score(np.asarray(truth, dtype=int), np.asarray(prediction, dtype=int)))


def evaluate_frozen_gate(
    *,
    baseline_overall_ba: float,
    cclf_overall_ba: float,
    baseline_strong_ba: float,
    cclf_strong_ba: float,
    baseline_weak_ba: float,
    cclf_weak_ba: float,
    baseline_overall_ece: float,
    cclf_overall_ece: float,
) -> dict[str, Any]:
    comparison_tolerance = 1e-12
    values = {
        "A": float(cclf_overall_ba - baseline_overall_ba)
        >= FROZEN_GATE["A_overall_ba_delta_min"] - comparison_tolerance,
        "B": float(cclf_strong_ba - baseline_strong_ba)
        >= FROZEN_GATE["B_strong_ba_delta_min"] - comparison_tolerance,
        "C": float(cclf_weak_ba - baseline_weak_ba)
        >= FROZEN_GATE["C_weak_ba_delta_min"] - comparison_tolerance,
        "D": float(cclf_overall_ece)
        <= FROZEN_GATE["D_overall_ece_ratio_max"] * float(baseline_overall_ece)
        + comparison_tolerance,
    }
    return {
        "thresholds": dict(FROZEN_GATE),
        "observed": {
            "overall_ba_delta": float(cclf_overall_ba - baseline_overall_ba),
            "strong_ba_delta": float(cclf_strong_ba - baseline_strong_ba),
            "weak_ba_delta": float(cclf_weak_ba - baseline_weak_ba),
            "overall_ece_ratio": float(cclf_overall_ece / baseline_overall_ece),
        },
        "passes": values,
        "decision": (
            "supports_cycle_calibrated_late_fusion"
            if all(values.values())
            else "does_not_support_cycle_calibrated_late_fusion"
        ),
    }


def exact_paired_sign_flip_test(deltas: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("session deltas must be a finite vector")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 1 << len(values)
    for mask in range(total):
        signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(len(values))])
        exceed += int(abs(float(np.mean(values * signs))) >= observed - 1e-15)
    return {
        "n_sessions": int(len(values)),
        "observed_mean_delta": float(values.mean()),
        "two_sided_p_value": float(exceed / total),
        "enumerated_sign_flips": int(total),
    }
