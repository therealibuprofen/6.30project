from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.block_ce_full_eval import (
    ALL_SEEDS,
    EXPECTED_BLOCKS_PER_SEED,
    EXPECTED_FOLDS,
    EXPECTED_NEW_TRAININGS,
    STABILITY_THRESHOLDS,
    build_new_training_plan,
    evaluate_stability,
    session_seed_balanced_accuracy,
    validate_prediction_identity_alignment,
    validate_seed0_reuse,
)
from ultrasound_decoding.multiframe.crr_fcnn import (
    apply_normalization,
    fit_train_only_normalization,
    fuse_frame_logits,
    validate_outer_split,
)
from ultrasound_decoding.multiframe.dataset import EXPECTED_SESSIONS


FOLD_COUNTS = dict(zip(EXPECTED_SESSIONS, [8, 8, 6, 10, 10, 10, 10, 10, 10]))
BLOCK_COUNTS = dict(zip(EXPECTED_SESSIONS, [32, 32, 24, 88, 72, 48, 40, 80, 40]))


def reference_plan() -> pd.DataFrame:
    rows = []
    for seed in ALL_SEEDS:
        for session, folds in FOLD_COUNTS.items():
            for fold in range(1, folds + 1):
                rows.append(
                    {
                        "session": session,
                        "seed": seed,
                        "fold": fold,
                        "n_train_samples": 8,
                        "n_test_samples": 4,
                        "train_cycles": "1,2",
                        "test_cycles": "0",
                    }
                )
    return pd.DataFrame(rows)


def seed0_prediction_table() -> pd.DataFrame:
    rows = []
    for session in EXPECTED_SESSIONS:
        folds = FOLD_COUNTS[session]
        count = BLOCK_COUNTS[session]
        for index in range(count):
            truth = index % 2
            rows.append(
                {
                    "session": session,
                    "seed": 0,
                    "fold": index % folds + 1,
                    "model": "block_ce_fcnn",
                    "block_id": f"session{session}_block{index:03d}",
                    "cycle": index // 4,
                    "block_name": "grating" if truth else "static",
                    "truth": truth,
                    "pred": truth,
                }
            )
    table = pd.DataFrame(rows)
    assert len(table) == EXPECTED_BLOCKS_PER_SEED
    return table


def test_plan_contains_only_seed1_and_seed2_new_trainings() -> None:
    plan = build_new_training_plan(reference_plan())
    assert set(plan["seed"]) == {1, 2}
    assert set(plan["training_action"]) == {"train_new"}
    assert not (plan["seed"] == 0).any()


def test_new_training_count_is_exactly_164() -> None:
    plan = build_new_training_plan(reference_plan())
    assert len(plan) == EXPECTED_NEW_TRAININGS == 164
    assert plan.groupby("seed").size().to_dict() == {1: EXPECTED_FOLDS, 2: EXPECTED_FOLDS}


def test_seed0_is_reused_without_retraining_and_matches_screening_ba() -> None:
    blockce = seed0_prediction_table()
    historical = blockce.drop(columns="model").copy()
    recomputed = session_seed_balanced_accuracy(
        blockce, value_name="block_ce_fcnn_seed0_BA"
    )
    screening = recomputed[["session", "block_ce_fcnn_seed0_BA"]]
    audit = validate_seed0_reuse(
        blockce,
        screening,
        historical,
        source_path="outputs/crr_fcnn_screening_v1",
    )
    assert audit["status"] == "PASS"
    assert audit["seed0_retrained"] is False
    assert audit["training_action"] == "reuse_predictions_only"
    assert audit["fold_count"] == 82
    assert audit["heldout_block_count"] == 456
    assert audit["maximum_absolute_session_BA_difference"] == 0.0


def test_outer_train_test_cycles_are_disjoint() -> None:
    validate_outer_split([0, 1, 2], [3])
    with pytest.raises(AssertionError, match="leakage"):
        validate_outer_split([0, 1], [1, 2])


def test_normalization_statistics_use_only_outer_training_frames() -> None:
    rng = np.random.default_rng(12)
    train = rng.normal(size=(3, 4, 2, 3)).astype(np.float32)
    test = rng.normal(size=(2, 4, 2, 3)).astype(np.float32)
    mean_before, std_before = fit_train_only_normalization(train)
    first = apply_normalization(test, mean_before, std_before)
    mutated_test = test + 9999.0
    mean_after, std_after = fit_train_only_normalization(train)
    second = apply_normalization(mutated_test, mean_after, std_after)
    assert np.array_equal(mean_before, mean_after)
    assert np.array_equal(std_before, std_after)
    assert not np.array_equal(first, second)


def test_block_fusion_is_equal_mean_of_four_frame_softmax_probabilities() -> None:
    logits = torch.tensor(
        [[[2.0, -1.0], [0.0, 1.0], [3.0, 2.0], [-2.0, 4.0]]], dtype=torch.float64
    )
    expected = torch.stack([torch.softmax(frame, dim=0) for frame in logits[0]]).mean(0)
    observed = fuse_frame_logits(logits)[0]
    assert torch.equal(observed, expected)
    assert not torch.allclose(observed, torch.softmax(logits.mean(1), dim=-1)[0])


def test_session_ba_concatenates_all_oof_blocks_not_mean_fold_ba() -> None:
    predictions = pd.DataFrame(
        {
            "session": ["708"] * 6,
            "seed": [1] * 6,
            "fold": [1, 1, 2, 2, 2, 2],
            "block_id": [f"b{i}" for i in range(6)],
            "truth": [0, 1, 0, 0, 0, 1],
            "pred": [0, 1, 1, 1, 1, 1],
        }
    )
    observed = session_seed_balanced_accuracy(predictions, value_name="BA").iloc[0]["BA"]
    mean_fold = np.mean(
        [
            classification_metrics(group.truth.to_numpy(), group.pred.to_numpy())[
                "balanced_accuracy"
            ]
            for _, group in predictions.groupby("fold")
        ]
    )
    assert observed == pytest.approx(0.625)
    assert mean_fold == pytest.approx(0.75)
    assert observed != mean_fold


def test_historical_and_blockce_identity_alignment_is_exact() -> None:
    seed0 = seed0_prediction_table()
    historical = seed0.drop(columns="model").copy()
    audit = validate_prediction_identity_alignment(historical, seed0, seeds=(0,))
    assert audit["status"] == "PASS"
    assert audit["aligned_rows"] == 456
    changed = seed0.copy()
    changed.loc[0, "block_id"] = "wrong"
    with pytest.raises(AssertionError, match="identities differ"):
        validate_prediction_identity_alignment(historical, changed, seeds=(0,))


def stability_inputs(delta: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_session = pd.DataFrame(
        {
            "session": EXPECTED_SESSIONS,
            "historical_3seed_mean_BA": [0.60] * 9,
            "blockce_3seed_mean_BA": [0.60 + delta] * 9,
            "delta_blockce_vs_historical": [delta] * 9,
            "p_value": [1.0] * 9,
            "unrelated_metric": [-100.0] * 9,
        }
    )
    seed_level = pd.DataFrame(
        {
            "seed": [0, 1, 2],
            "historical_9session_mean_BA": [0.60] * 3,
            "blockce_9session_mean_BA": [0.62] * 3,
            "delta_blockce_vs_historical": [0.02] * 3,
            "blockce_better": [True, True, True],
        }
    )
    return per_session, seed_level


def test_only_abcd_control_stability_decision() -> None:
    per_session, seed_level = stability_inputs()
    result = evaluate_stability(per_session, seed_level)
    assert result["thresholds"] == STABILITY_THRESHOLDS
    assert result["controlling_criteria"] == ["A", "B", "C", "D"]
    assert all(item["passed"] for item in result["criteria"].values())
    assert result["p_value_controls_decision"] is False
    assert result["decision"] == "supports_block_level_training_as_new_baseline"

    failed_sessions, failed_seeds = stability_inputs(delta=-0.01)
    failed_seeds["blockce_better"] = False
    failed = evaluate_stability(failed_sessions, failed_seeds)
    assert failed["decision"] == "does_not_support_block_level_training_as_new_baseline"
