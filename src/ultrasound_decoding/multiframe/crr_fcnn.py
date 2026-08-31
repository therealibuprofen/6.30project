from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import BLOCK_NAMES, EXPECTED_SESSIONS
from ultrasound_decoding.multiframe.training import set_reproducible_seed


MODEL_VERSION = "crr_fcnn_screening_v1.0.0"
MODELS = ("block_ce_fcnn", "crr_fcnn")
SEED = 0
IMAGE_SHAPE = (128, 501)
FRAMES_PER_BLOCK = 4
BLOCKS_PER_CYCLE = 4
EXPECTED_PARAMETERS = 48_011
STIMULUS_CLASS = 1
NONSTIMULUS_CLASS = 0
BLOCK_LABELS = {
    "grating": STIMULUS_CLASS,
    "stop_after_grating": NONSTIMULUS_CLASS,
    "dot": STIMULUS_CLASS,
    "static": NONSTIMULUS_CLASS,
}
RANKING_PAIRS = (
    ("grating", "stop_after_grating"),
    ("grating", "static"),
    ("dot", "stop_after_grating"),
    ("dot", "static"),
)
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
FROZEN_OPTIMIZATION = {
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-3,
    "epochs": 40,
    "early_stopping": False,
    "seed": SEED,
}
FROZEN_GATE = {
    "A_mean_delta_minimum_inclusive": 0.010,
    "B_session_wins_minimum_inclusive": 5,
    "C_minimum_leave_one_session_out_delta_exclusive": 0.0,
    "D_crr_mean_greater_than_historical": True,
}


@dataclass
class FoldModelResult:
    model_name: str
    probabilities: np.ndarray
    predictions: np.ndarray
    history: list[dict[str, Any]]
    initial_state_sha256: str
    cycle_order_sha256: str
    final_train_balanced_accuracy: float
    final_test_balanced_accuracy: float


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def validate_complete_cycle_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Validate exactly one authoritative row for each frozen block in every cycle."""

    required = {"session", "cycle", "block_name", "binary_label_int", "block_id"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise AssertionError(f"cycle metadata lacks required columns: {missing}")
    rows: list[dict[str, Any]] = []
    for (session, cycle), group in metadata.groupby(["session", "cycle"], sort=True):
        counts = group["block_name"].astype(str).value_counts().to_dict()
        exact = all(int(counts.get(name, 0)) == 1 for name in BLOCK_NAMES)
        extras = sorted(set(counts) - set(BLOCK_NAMES))
        label_matches = all(
            int(row.binary_label_int) == BLOCK_LABELS[str(row.block_name)]
            for row in group.itertuples(index=False)
            if str(row.block_name) in BLOCK_LABELS
        )
        valid = bool(len(group) == BLOCKS_PER_CYCLE and exact and not extras and label_matches)
        rows.append(
            {
                "session": str(session),
                "cycle": int(cycle),
                **{f"n_{name}": int(counts.get(name, 0)) for name in BLOCK_NAMES},
                "n_rows": int(len(group)),
                "extra_block_names": json.dumps(extras),
                "binary_labels_match_mapping": bool(label_matches),
                "status": "PASS" if valid else "FAIL",
            }
        )
        if not valid:
            raise AssertionError(
                f"session {session} cycle {cycle} is not exactly one of each frozen block"
            )
    if not rows:
        raise AssertionError("cycle metadata is empty")
    return pd.DataFrame(rows)


def validate_outer_split(train_cycles: Iterable[int], test_cycles: Iterable[int]) -> None:
    train = {int(value) for value in train_cycles}
    test = {int(value) for value in test_cycles}
    if not train or not test:
        raise AssertionError("outer train and test cycles must both be non-empty")
    overlap = sorted(train & test)
    if overlap:
        raise AssertionError(f"outer train/test cycle leakage: {overlap}")


def fit_train_only_normalization(train_blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(train_blocks, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != FRAMES_PER_BLOCK:
        raise ValueError("training blocks must have shape [N,4,H,W]")
    if not np.isfinite(values).all():
        raise ValueError("training blocks contain NaN or Inf")
    transformed = np.arcsinh(values)
    frames = transformed.reshape(-1, *transformed.shape[-2:]).astype(np.float64, copy=False)
    mean = frames.mean(axis=0, keepdims=True)
    std = frames.std(axis=0, keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_normalization(
    blocks: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    values = np.asarray(blocks, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != FRAMES_PER_BLOCK:
        raise ValueError("blocks must have shape [N,4,H,W]")
    expected = (1, *values.shape[-2:])
    if np.asarray(mean).shape != expected or np.asarray(std).shape != expected:
        raise ValueError(f"normalization statistics must have shape {expected}")
    if not np.isfinite(values).all() or np.any(np.asarray(std) <= 0):
        raise ValueError("normalization input/statistics are invalid")
    result = (np.arcsinh(values) - mean) / std
    if not np.isfinite(result).all():
        raise AssertionError("normalization produced non-finite values")
    return result.astype(np.float32, copy=False)


def fuse_frame_logits(frame_logits: torch.Tensor) -> torch.Tensor:
    """Softmax every frame, then equally average four probability vectors."""

    if frame_logits.ndim != 3 or frame_logits.shape[1:] != (FRAMES_PER_BLOCK, 2):
        raise ValueError("frame logits must have shape [N,4,2]")
    return torch.softmax(frame_logits, dim=-1).mean(dim=1)


def block_classification_loss(
    block_probabilities: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    if block_probabilities.ndim != 2 or block_probabilities.shape[1] != 2:
        raise ValueError("block probabilities must have shape [N,2]")
    labels = labels.to(dtype=torch.long, device=block_probabilities.device)
    if labels.shape != (len(block_probabilities),):
        raise ValueError("one label is required per block")
    selected = block_probabilities[
        torch.arange(len(labels), device=block_probabilities.device), labels
    ]
    return -torch.log(selected.clamp_min(1e-8)).mean()


def cycle_ranking_loss(
    probabilities_by_block: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Mean softplus ranking loss over the four frozen within-cycle pairs."""

    if set(probabilities_by_block) != set(BLOCK_NAMES):
        raise ValueError("ranking requires exactly all four frozen cycle blocks")
    evidence: dict[str, torch.Tensor] = {}
    for name in BLOCK_NAMES:
        probability = probabilities_by_block[name]
        if probability.shape != (2,):
            raise ValueError("each fused block probability must have shape [2]")
        evidence[name] = (
            torch.log(probability[STIMULUS_CLASS].clamp_min(1e-8))
            - torch.log(probability[NONSTIMULUS_CLASS].clamp_min(1e-8))
        )
    losses = [
        nn.functional.softplus(-(evidence[stimulus] - evidence[nonstimulus]))
        for stimulus, nonstimulus in RANKING_PAIRS
    ]
    if len(losses) != 4:
        raise AssertionError("CRR must use exactly four ranking pairs")
    return torch.stack(losses).mean()


def complete_cycle_loss(
    frame_logits: torch.Tensor, model_name: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    if frame_logits.shape != (BLOCKS_PER_CYCLE, FRAMES_PER_BLOCK, 2):
        raise ValueError("one optimizer step must contain four blocks x four frames")
    probabilities = fuse_frame_logits(frame_logits)
    labels = torch.tensor(
        [BLOCK_LABELS[name] for name in BLOCK_NAMES],
        dtype=torch.long,
        device=probabilities.device,
    )
    classification = block_classification_loss(probabilities, labels)
    ranking = cycle_ranking_loss(
        {name: probabilities[index] for index, name in enumerate(BLOCK_NAMES)}
    )
    total = classification if model_name == "block_ce_fcnn" else classification + ranking
    return total, classification, ranking


def deterministic_cycle_orders(
    cycle_ids: Iterable[int], *, seed: int, epochs: int
) -> list[list[int]]:
    cycles = np.asarray(sorted({int(value) for value in cycle_ids}), dtype=np.int64)
    if len(cycles) < 1 or int(epochs) < 1:
        raise ValueError("at least one cycle and one epoch are required")
    generator = np.random.default_rng(int(seed))
    return [generator.permutation(cycles).astype(int).tolist() for _ in range(int(epochs))]


def cycle_order_sha256(orders: list[list[int]]) -> str:
    encoded = json.dumps(orders, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initialized_fcnn(seed: int = SEED, *, image_shape: tuple[int, int] = IMAGE_SHAPE) -> nn.Module:
    set_reproducible_seed(int(seed))
    model = FCNN(input_shape=image_shape, n_classes=2)
    if image_shape == IMAGE_SHAPE and count_parameters(model) != EXPECTED_PARAMETERS:
        raise AssertionError("FCNN parameter count differs from frozen 48,011")
    return model


def _cycle_block_indices(
    groups: np.ndarray, metadata: pd.DataFrame
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    names = metadata["block_name"].astype(str).to_numpy()
    for cycle in sorted(np.unique(groups).astype(int).tolist()):
        indices = np.flatnonzero(groups == cycle)
        by_name = {names[index]: int(index) for index in indices}
        if set(by_name) != set(BLOCK_NAMES) or len(indices) != BLOCKS_PER_CYCLE:
            raise AssertionError(f"training cycle {cycle} is incomplete")
        result[cycle] = np.asarray([by_name[name] for name in BLOCK_NAMES], dtype=np.int64)
    return result


def predict_block_probabilities(
    model: nn.Module,
    normalized_blocks: np.ndarray,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    blocks = np.asarray(normalized_blocks, dtype=np.float32)
    if blocks.ndim != 4 or blocks.shape[1:] != (FRAMES_PER_BLOCK, *IMAGE_SHAPE):
        raise ValueError("normalized clean4 blocks must have shape [N,4,128,501]")
    torch_device = torch.device(device)
    frames = torch.from_numpy(blocks.reshape(-1, 1, *IMAGE_SHAPE))
    pieces: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frames), int(batch_size)):
            pieces.append(model(frames[start : start + int(batch_size)].to(torch_device)).cpu())
    logits = torch.cat(pieces, dim=0).reshape(len(blocks), FRAMES_PER_BLOCK, 2)
    probabilities = fuse_frame_logits(logits).numpy()
    if probabilities.shape != (len(blocks), 2):
        raise AssertionError("block prediction coverage is incomplete")
    return probabilities


def train_cycle_model(
    X_train_normalized: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    metadata_train: pd.DataFrame,
    X_test_normalized: np.ndarray,
    y_test: np.ndarray,
    *,
    model_name: str,
    cycle_orders: list[list[int]],
    seed: int = SEED,
    device: str = "cpu",
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
) -> FoldModelResult:
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    model = initialized_fcnn(seed).to(torch.device(device))
    initial_hash = model_state_sha256(model)
    order_hash = cycle_order_sha256(cycle_orders)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    cycle_indices = _cycle_block_indices(groups_train, metadata_train.reset_index(drop=True))
    expected_cycles = set(cycle_indices)
    labels = np.asarray(y_train, dtype=np.int64)
    history: list[dict[str, Any]] = []
    for epoch, order in enumerate(cycle_orders, start=1):
        if len(order) != len(expected_cycles) or set(order) != expected_cycles:
            raise AssertionError("each epoch must use every outer-training cycle exactly once")
        model.train()
        total_loss = total_cls = total_rank = 0.0
        for cycle in order:
            indices = cycle_indices[int(cycle)]
            expected_labels = np.asarray([BLOCK_LABELS[name] for name in BLOCK_NAMES])
            if not np.array_equal(labels[indices], expected_labels):
                raise AssertionError("cycle labels differ from the frozen binary mapping")
            blocks = torch.from_numpy(X_train_normalized[indices]).to(torch.device(device))
            frame_tensor = blocks.reshape(BLOCKS_PER_CYCLE * FRAMES_PER_BLOCK, 1, *IMAGE_SHAPE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frame_tensor).reshape(BLOCKS_PER_CYCLE, FRAMES_PER_BLOCK, 2)
            loss, classification, ranking = complete_cycle_loss(logits, model_name)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_cls += float(classification.detach().cpu())
            total_rank += float(ranking.detach().cpu())
        denominator = len(order)
        history.append(
            {
                "epoch": int(epoch),
                "mean_total_loss": total_loss / denominator,
                "mean_classification_loss": total_cls / denominator,
                "mean_ranking_loss_diagnostic": total_rank / denominator,
                "optimizer_steps": int(denominator),
                "cycles_seen_once": True,
            }
        )
    train_probabilities = predict_block_probabilities(
        model, X_train_normalized, device=device
    )
    test_probabilities = predict_block_probabilities(model, X_test_normalized, device=device)
    train_predictions = train_probabilities.argmax(axis=1).astype(np.int64)
    test_predictions = test_probabilities.argmax(axis=1).astype(np.int64)
    return FoldModelResult(
        model_name=model_name,
        probabilities=test_probabilities,
        predictions=test_predictions,
        history=history,
        initial_state_sha256=initial_hash,
        cycle_order_sha256=order_hash,
        final_train_balanced_accuracy=classification_metrics(labels, train_predictions)[
            "balanced_accuracy"
        ],
        final_test_balanced_accuracy=classification_metrics(
            np.asarray(y_test, dtype=np.int64), test_predictions
        )["balanced_accuracy"],
    )


def build_screening_plan(reference_plan: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session", "seed", "fold", "train_cycles", "test_cycles",
        "n_train_samples", "n_test_samples",
    }
    if not required.issubset(reference_plan.columns):
        raise AssertionError("historical reference plan lacks required columns")
    base = reference_plan.copy()
    base["session"] = base["session"].astype(str)
    base = base[base["seed"].astype(int).eq(SEED)].copy()
    base = base.sort_values(["session", "fold"]).reset_index(drop=True)
    if len(base) != 82 or set(base["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("formal historical seed-0 plan must contain exactly 82 folds")
    rows: list[dict[str, Any]] = []
    for row in base.to_dict("records"):
        for model_name in MODELS:
            rows.append(
                {
                    "session": str(row["session"]),
                    "seed": SEED,
                    "fold": int(row["fold"]),
                    "model": model_name,
                    "n_train_blocks": int(row["n_train_samples"]),
                    "n_test_blocks": int(row["n_test_samples"]),
                    "train_cycles": str(row["train_cycles"]),
                    "test_cycles": str(row["test_cycles"]),
                    "task_key": f"{row['session']}:{SEED}:{int(row['fold'])}:{model_name}",
                }
            )
    result = pd.DataFrame(rows)
    counts = result.groupby("model").size().to_dict()
    if counts != {"block_ce_fcnn": 82, "crr_fcnn": 82} or len(result) != 164:
        raise AssertionError("screening plan must be exactly 82 Block-CE + 82 CRR")
    return result


def session_oof_balanced_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "model", "truth", "pred", "block_id"}
    if not required.issubset(predictions.columns):
        raise AssertionError("OOF predictions lack required columns")
    rows = []
    for (session, model_name), group in predictions.groupby(["session", "model"], sort=True):
        if group["block_id"].duplicated().any():
            raise AssertionError("a held-out block occurs more than once in session OOF predictions")
        metrics = classification_metrics(
            group["truth"].to_numpy(dtype=np.int64),
            group["pred"].to_numpy(dtype=np.int64),
        )
        rows.append(
            {
                "session": str(session),
                "model": str(model_name),
                "oof_balanced_accuracy": metrics["balanced_accuracy"],
                "n_oof_blocks": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def historical_seed0_session_ba(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"session", "seed", "truth", "pred", "block_id"}
    if not required.issubset(predictions.columns):
        raise AssertionError("historical predictions lack required columns")
    selected = predictions[predictions["seed"].astype(int).eq(SEED)].copy()
    selected["session"] = selected["session"].astype(str)
    if set(selected["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("historical seed-0 predictions do not cover all nine sessions")
    rows = []
    for session, group in selected.groupby("session", sort=True):
        if group["block_id"].duplicated().any():
            raise AssertionError("historical OOF block occurs more than once")
        ba = classification_metrics(
            group["truth"].to_numpy(dtype=np.int64),
            group["pred"].to_numpy(dtype=np.int64),
        )["balanced_accuracy"]
        rows.append(
            {
                "session": str(session),
                "historical_fcnn_latefusion_seed0_BA": ba,
                "historical_n_oof_blocks": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def evaluate_screening_gate(per_session_summary: pd.DataFrame) -> dict[str, Any]:
    required = {
        "session",
        "historical_fcnn_latefusion_seed0_BA",
        "block_ce_fcnn_seed0_BA",
        "crr_fcnn_seed0_BA",
    }
    if not required.issubset(per_session_summary.columns):
        raise AssertionError("per-session summary lacks frozen screening inputs")
    table = per_session_summary.copy()
    table["session"] = table["session"].astype(str)
    if len(table) != 9 or set(table["session"]) != set(EXPECTED_SESSIONS):
        raise AssertionError("screening gate requires exactly the nine frozen sessions")
    deltas = (
        table["crr_fcnn_seed0_BA"].to_numpy(float)
        - table["block_ce_fcnn_seed0_BA"].to_numpy(float)
    )
    mean_delta = float(deltas.mean())
    wins = int((deltas > 0).sum())
    leave_one_out = {
        str(table.iloc[index]["session"]): float(np.delete(deltas, index).mean())
        for index in range(len(table))
    }
    minimum_loso = float(min(leave_one_out.values()))
    crr_mean = float(table["crr_fcnn_seed0_BA"].mean())
    block_ce_mean = float(table["block_ce_fcnn_seed0_BA"].mean())
    historical_mean = float(table["historical_fcnn_latefusion_seed0_BA"].mean())
    passes = {
        "A": bool(mean_delta >= FROZEN_GATE["A_mean_delta_minimum_inclusive"]),
        "B": bool(wins >= FROZEN_GATE["B_session_wins_minimum_inclusive"]),
        "C": bool(
            minimum_loso
            > FROZEN_GATE["C_minimum_leave_one_session_out_delta_exclusive"]
        ),
        "D": bool(crr_mean > historical_mean),
    }
    all_pass = all(passes.values())
    return {
        "thresholds": dict(FROZEN_GATE),
        "observed": {
            "nine_session_mean_crr_minus_blockce": mean_delta,
            "sessions_crr_greater_than_blockce": wins,
            "leave_one_session_out_mean_deltas": leave_one_out,
            "minimum_leave_one_session_out_mean_delta": minimum_loso,
            "historical_fcnn_latefusion_seed0_mean_BA": historical_mean,
            "block_ce_fcnn_seed0_mean_BA": block_ce_mean,
            "crr_fcnn_seed0_mean_BA": crr_mean,
        },
        "criteria": {name: {"passed": passed} for name, passed in passes.items()},
        "all_four_criteria_passed": all_pass,
        "decision": (
            "supports_full_evaluation_cycle_relative_ranking_fcnn"
            if all_pass
            else "does_not_support_cycle_relative_ranking_fcnn"
        ),
        "controlling_criteria": ["A", "B", "C", "D"],
        "p_value_computed": False,
    }
