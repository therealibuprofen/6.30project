from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, pearsonr, rankdata, spearmanr
from sklearn.metrics import balanced_accuracy_score

from ultrasound_decoding.multiframe.cycle_calibrated_late_fusion import (
    FRAMES_PER_BLOCK,
    build_inner_cycle_splits,
    equal_four_frame_probability_mean,
    softmax_probabilities,
)
from ultrasound_decoding.multiframe.dataset import BlockSequenceData, TASK_CLASS_NAMES


OUTPUT_VERSION = "dqi_dot_vs_grating_validation_v1"
TASK_NAME = "dot_vs_grating"
HISTORICAL_TASK_NAME = "stimulus_type"
CLASS_NAMES = {0: "dot", 1: "grating"}
CHANCE_BALANCED_ACCURACY = 0.5
SESSIONS = ("626", "628", "708", "709", "710", "807", "813", "817", "822")
SEEDS = (0, 1, 2)
EXPECTED_FOLDS = 82
EXPECTED_OUTER_TASKS = 246
N_INNER_FOLDS = 3
EXPECTED_INNER_TRAININGS = EXPECTED_OUTER_TASKS * N_INNER_FOLDS
FROZEN_CONFIRMATORY_GATE = {
    "A_session_spearman_min": 0.75,
    "B_exact_two_sided_permutation_p_max": 0.05,
    "C_loo_median_spearman_min": 0.65,
    "C_loo_min_spearman_strictly_above": 0.30,
}


@dataclass(frozen=True)
class ExactPermutationResult:
    observed: float
    extreme: int
    total: int
    two_sided_p: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "extreme_permutations": self.extreme,
            "total_permutations": self.total,
            "two_sided_exact_p": self.two_sided_p,
            "definition": "count(abs(rho_perm)>=abs(rho_observed))/9!; exhaustive",
        }


def validate_authoritative_mapping() -> None:
    if TASK_CLASS_NAMES.get(HISTORICAL_TASK_NAME) != CLASS_NAMES:
        raise AssertionError("authoritative stimulus_type mapping is not dot=0, grating=1")


def validate_dot_vs_grating_data(data: BlockSequenceData) -> None:
    validate_authoritative_mapping()
    if data.task != HISTORICAL_TASK_NAME:
        raise AssertionError("dot-vs-grating data must use the historical stimulus_type loader")
    names = data.metadata["block_name"].astype(str)
    if set(names) != {"dot", "grating"}:
        raise AssertionError("dot-vs-grating data contains stop/static or lacks a class")
    expected = names.map({"dot": 0, "grating": 1}).to_numpy(dtype=np.int64)
    if not np.array_equal(expected, np.asarray(data.y, dtype=np.int64)):
        raise AssertionError("dot-vs-grating labels differ from authoritative mapping")
    counts = data.metadata.groupby("cycle")["block_name"].value_counts().unstack(fill_value=0)
    if set(counts.columns) != {"dot", "grating"}:
        raise AssertionError("cycle block-name coverage differs")
    if not counts["dot"].eq(1).all() or not counts["grating"].eq(1).all():
        raise AssertionError("each cycle must contain exactly one dot and one grating block")
    if not data.metadata.groupby("cycle")["block_id"].size().eq(2).all():
        raise AssertionError("each dot-vs-grating cycle must contain exactly two blocks")


def block_predictions_from_frame_logits(frame_logits: pd.DataFrame) -> pd.DataFrame:
    required = {
        "session", "outer_seed", "outer_fold", "inner_fold", "source_index",
        "block_id", "cycle", "frame_position", "truth", "logit_dot", "logit_grating",
    }
    if not required.issubset(frame_logits.columns):
        raise ValueError(f"inner OOF logits lack columns {sorted(required - set(frame_logits.columns))}")
    if frame_logits[["source_index", "frame_position"]].duplicated().any():
        raise AssertionError("inner OOF frame prediction is duplicated")
    frames = frame_logits.copy()
    frames[["prob_dot", "prob_grating"]] = softmax_probabilities(
        frames[["logit_dot", "logit_grating"]].to_numpy(dtype=np.float64)
    )
    rows: list[dict[str, Any]] = []
    for block_id, group in frames.groupby("block_id", sort=True):
        ordered = group.sort_values("frame_position")
        if (
            len(ordered) != FRAMES_PER_BLOCK
            or ordered["frame_position"].astype(int).tolist() != list(range(FRAMES_PER_BLOCK))
            or ordered["truth"].nunique() != 1
            or ordered["cycle"].nunique() != 1
            or ordered["inner_fold"].nunique() != 1
            or ordered["source_index"].nunique() != 1
        ):
            raise AssertionError(f"block {block_id} is not an exact four-frame OOF unit")
        probabilities = equal_four_frame_probability_mean(
            ordered[["prob_dot", "prob_grating"]].to_numpy(dtype=np.float64)[None, :, :]
        )[0]
        rows.append(
            {
                "session": str(ordered.iloc[0]["session"]),
                "outer_seed": int(ordered.iloc[0]["outer_seed"]),
                "outer_fold": int(ordered.iloc[0]["outer_fold"]),
                "inner_fold": int(ordered.iloc[0]["inner_fold"]),
                "source_index": int(ordered.iloc[0]["source_index"]),
                "block_id": str(block_id),
                "cycle": int(ordered.iloc[0]["cycle"]),
                "truth": int(ordered.iloc[0]["truth"]),
                "prob_dot": float(probabilities[0]),
                "prob_grating": float(probabilities[1]),
                "prediction": int(np.argmax(probabilities)),
                "n_frames_fused": FRAMES_PER_BLOCK,
                "fusion": "arithmetic_mean_of_four_raw_frame_probabilities",
            }
        )
    result = pd.DataFrame(rows).sort_values("source_index").reset_index(drop=True)
    if result["source_index"].duplicated().any() or result["block_id"].duplicated().any():
        raise AssertionError("an outer-training block has multiple inner OOF predictions")
    return result


def concatenated_oof_balanced_accuracy(block_predictions: pd.DataFrame) -> float:
    required = {"truth", "prediction", "block_id"}
    if not required.issubset(block_predictions.columns) or block_predictions.empty:
        raise ValueError("block predictions are empty or incomplete")
    if block_predictions["block_id"].duplicated().any():
        raise AssertionError("Q_dec requires each outer-training block exactly once")
    truth = block_predictions["truth"].to_numpy(dtype=np.int64)
    prediction = block_predictions["prediction"].to_numpy(dtype=np.int64)
    if set(np.unique(truth)) != {0, 1}:
        raise AssertionError("Q_dec requires both dot and grating")
    return float(balanced_accuracy_score(truth, prediction))


def mean_inner_fold_ba_diagnostic(block_predictions: pd.DataFrame) -> float:
    values = []
    for _, group in block_predictions.groupby("inner_fold", sort=True):
        values.append(float(balanced_accuracy_score(group["truth"], group["prediction"])))
    if len(values) != N_INNER_FOLDS:
        raise AssertionError("inner-fold diagnostic requires exactly three folds")
    return float(np.mean(values))


def build_inner_manifest(task_plan: pd.DataFrame, protocol_hash: str, source_hash: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task in task_plan.sort_values(["session", "seed", "fold"]).itertuples(index=False):
        for split in build_inner_cycle_splits(task.outer_train_cycles, task.outer_test_cycles):
            rows.append(
                {
                    "task_key": str(task.task_key),
                    "task_fingerprint": str(task.task_fingerprint),
                    "task": TASK_NAME,
                    "session": str(task.session),
                    "outer_seed": int(task.seed),
                    "outer_fold": int(task.fold),
                    "outer_train_cycles": str(task.outer_train_cycles),
                    "outer_test_cycles": str(task.outer_test_cycles),
                    "inner_fold": int(split.inner_fold),
                    "inner_train_cycles": ",".join(map(str, split.train_cycles)),
                    "inner_validation_cycles": ",".join(map(str, split.validation_cycles)),
                    "n_inner_train_blocks": 2 * len(split.train_cycles),
                    "n_inner_validation_blocks": 2 * len(split.validation_cycles),
                    "n_inner_train_frames": 8 * len(split.train_cycles),
                    "n_inner_validation_frames": 8 * len(split.validation_cycles),
                    "inner_model_training_seed": int(task.seed),
                    "protocol_hash": str(protocol_hash),
                    "source_hash": str(source_hash),
                    "outer_test_used": False,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_INNER_TRAININGS:
        raise AssertionError("inner training count is not 738")
    if result[["task_key", "inner_fold"]].duplicated().any():
        raise AssertionError("inner split identities are duplicated")
    return result


def finite_pearson(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if len(a) != len(b) or len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def finite_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if len(a) != len(b) or len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(a, b).statistic)


def exact_spearman_permutation(q: Iterable[float], target: Iterable[float]) -> ExactPermutationResult:
    q_rank = rankdata(np.asarray(list(q), dtype=np.float64), method="average")
    target_rank = rankdata(np.asarray(list(target), dtype=np.float64), method="average")
    if len(q_rank) != 9 or len(target_rank) != 9:
        raise ValueError("confirmatory exact test requires exactly nine sessions")
    q_centered = q_rank - q_rank.mean()
    target_centered = target_rank - target_rank.mean()
    denominator = float(np.linalg.norm(q_centered) * np.linalg.norm(target_centered))
    if denominator == 0:
        raise ValueError("exact Spearman is undefined for a constant vector")
    observed = float(np.dot(q_centered, target_centered) / denominator)
    extreme = 0
    total = 0
    for permutation in itertools.permutations(range(9)):
        statistic = float(np.dot(q_centered, target_centered[list(permutation)]) / denominator)
        total += 1
        if abs(statistic) + 1e-15 >= abs(observed):
            extreme += 1
    if total != math.factorial(9):
        raise AssertionError("exact permutation count differs from 9!")
    return ExactPermutationResult(observed, extreme, total, float(extreme / total))


def leave_one_session_out(session_summary: pd.DataFrame, q_column: str, target_column: str) -> pd.DataFrame:
    if len(session_summary) != 9 or session_summary["session"].duplicated().any():
        raise ValueError("LOSO requires exactly nine unique sessions")
    rows = []
    for held_out in session_summary["session"].astype(str):
        remaining = session_summary[~session_summary["session"].astype(str).eq(held_out)]
        rows.append(
            {
                "held_out_session": held_out,
                "n_remaining_sessions": 8,
                "spearman_rho": finite_spearman(remaining[q_column], remaining[target_column]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_confirmatory_gate(
    *, session_spearman: float, exact_p: float, loo_median: float, loo_minimum: float
) -> dict[str, Any]:
    observed_values = np.asarray(
        [session_spearman, exact_p, loo_median, loo_minimum], dtype=np.float64
    )
    finite = bool(np.isfinite(observed_values).all())
    passes = {
        "A": finite and float(session_spearman) >= FROZEN_CONFIRMATORY_GATE["A_session_spearman_min"],
        "B": finite and float(exact_p) <= FROZEN_CONFIRMATORY_GATE["B_exact_two_sided_permutation_p_max"],
        "C": (
            finite
            and
            float(loo_median) >= FROZEN_CONFIRMATORY_GATE["C_loo_median_spearman_min"]
            and float(loo_minimum) > FROZEN_CONFIRMATORY_GATE["C_loo_min_spearman_strictly_above"]
        ),
    }
    return {
        "thresholds": dict(FROZEN_CONFIRMATORY_GATE),
        "observed": {
            "session_spearman": float(session_spearman),
            "exact_two_sided_permutation_p": float(exact_p),
            "loo_median_spearman": float(loo_median),
            "loo_minimum_spearman": float(loo_minimum),
        },
        "passes": passes,
        "all_observed_statistics_finite": finite,
        "decision": (
            "supports_cross_task_DQI_validation_dot_vs_grating"
            if all(passes.values())
            else "does_not_support_cross_task_DQI_validation_dot_vs_grating"
        ),
    }


def cross_task_relationship_matrix(session_summary: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Q_presence_session", "BA_presence_session", "Q_DG_session", "BA_DG_session"
    }
    if len(session_summary) != 9 or not required.issubset(session_summary.columns):
        raise ValueError("cross-task matrix requires nine sessions and four frozen columns")
    relationships = (
        ("Q_presence_to_BA_presence", "Q_presence_session", "BA_presence_session", "discovery_reference"),
        ("Q_DG_to_BA_DG", "Q_DG_session", "BA_DG_session", "primary_confirmatory"),
        ("Q_presence_to_BA_DG", "Q_presence_session", "BA_DG_session", "cross_task_secondary"),
        ("Q_DG_to_BA_presence", "Q_DG_session", "BA_presence_session", "cross_task_secondary"),
        ("Q_presence_to_Q_DG", "Q_presence_session", "Q_DG_session", "quality_to_quality_secondary"),
    )
    rows = []
    for name, left, right, role in relationships:
        rows.append(
            {
                "relationship": name,
                "left": left,
                "right": right,
                "n_sessions": 9,
                "pearson_r": finite_pearson(session_summary[left], session_summary[right]),
                "spearman_rho": finite_spearman(session_summary[left], session_summary[right]),
                "role": role,
                "enters_confirmatory_gate": role == "primary_confirmatory",
            }
        )
    return pd.DataFrame(rows)
