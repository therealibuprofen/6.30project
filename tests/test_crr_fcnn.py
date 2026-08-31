from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.crr_fcnn import (
    BLOCK_NAMES,
    EXPECTED_PARAMETERS,
    FROZEN_GATE,
    RANKING_PAIRS,
    apply_normalization,
    build_screening_plan,
    complete_cycle_loss,
    count_parameters,
    cycle_order_sha256,
    cycle_ranking_loss,
    deterministic_cycle_orders,
    evaluate_screening_gate,
    fit_train_only_normalization,
    fuse_frame_logits,
    initialized_fcnn,
    model_state_sha256,
    predict_block_probabilities,
    session_oof_balanced_accuracy,
    validate_complete_cycle_metadata,
    validate_outer_split,
)
from ultrasound_decoding.multiframe.dataset import EXPECTED_SESSIONS


def complete_metadata(n_cycles: int = 2) -> pd.DataFrame:
    labels = {"grating": 1, "stop_after_grating": 0, "dot": 1, "static": 0}
    return pd.DataFrame(
        [
            {
                "session": "708",
                "cycle": cycle,
                "block_name": name,
                "binary_label_int": labels[name],
                "block_id": f"session708_cycle{cycle:03d}_{name}",
            }
            for cycle in range(n_cycles)
            for name in BLOCK_NAMES
        ]
    )


def test_every_cycle_has_exactly_one_of_each_frozen_block() -> None:
    audit = validate_complete_cycle_metadata(complete_metadata())
    assert len(audit) == 2
    assert set(audit["status"]) == {"PASS"}
    duplicate = pd.concat([complete_metadata(), complete_metadata().iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="exactly one"):
        validate_complete_cycle_metadata(duplicate)


def test_outer_train_and_test_cycles_are_disjoint() -> None:
    validate_outer_split([0, 1, 2], [3])
    with pytest.raises(AssertionError, match="leakage"):
        validate_outer_split([0, 1], [1, 2])


def test_outer_test_pixels_cannot_affect_normalization() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(3, 4, 3, 5)).astype(np.float32)
    test = rng.normal(size=(2, 4, 3, 5)).astype(np.float32)
    mean_before, std_before = fit_train_only_normalization(train)
    normalized_before = apply_normalization(test, mean_before, std_before)
    mutated_test = test + 10000.0
    mean_after, std_after = fit_train_only_normalization(train)
    _normalized_after = apply_normalization(mutated_test, mean_after, std_after)
    assert np.array_equal(mean_before, mean_after)
    assert np.array_equal(std_before, std_after)
    assert not np.array_equal(normalized_before, _normalized_after)


def test_four_frame_fusion_is_equal_mean_of_frame_softmax_probabilities() -> None:
    logits = torch.tensor(
        [[[2.0, -1.0], [0.0, 1.0], [3.0, 2.0], [-2.0, 4.0]]], dtype=torch.float64
    )
    expected = torch.stack([torch.softmax(frame, dim=0) for frame in logits[0]]).mean(0)
    observed = fuse_frame_logits(logits)[0]
    assert torch.equal(observed, expected)
    assert not torch.allclose(observed, torch.softmax(logits.mean(1), dim=-1)[0])


def test_block_inference_does_not_require_other_cycle_blocks() -> None:
    rng = np.random.default_rng(7)
    blocks = rng.normal(size=(2, 4, 128, 501)).astype(np.float32)
    model = initialized_fcnn(0)
    alone = predict_block_probabilities(model, blocks[:1])
    with_unrelated_block = predict_block_probabilities(model, blocks)
    assert np.allclose(alone[0], with_unrelated_block[0], atol=1e-7, rtol=0.0)


def test_ranking_loss_decreases_when_stimulus_evidence_increases() -> None:
    initial = {
        "grating": torch.tensor([0.5, 0.5]),
        "stop_after_grating": torch.tensor([0.5, 0.5]),
        "dot": torch.tensor([0.5, 0.5]),
        "static": torch.tensor([0.5, 0.5]),
    }
    improved = {
        "grating": torch.tensor([0.1, 0.9]),
        "stop_after_grating": torch.tensor([0.9, 0.1]),
        "dot": torch.tensor([0.2, 0.8]),
        "static": torch.tensor([0.8, 0.2]),
    }
    assert cycle_ranking_loss(improved) < cycle_ranking_loss(initial)


def test_ranking_loss_uses_exactly_four_frozen_pairs() -> None:
    assert RANKING_PAIRS == (
        ("grating", "stop_after_grating"),
        ("grating", "static"),
        ("dot", "stop_after_grating"),
        ("dot", "static"),
    )
    logits = torch.zeros(4, 4, 2)
    total_ce, classification, ranking = complete_cycle_loss(logits, "block_ce_fcnn")
    total_crr, classification_crr, ranking_crr = complete_cycle_loss(logits, "crr_fcnn")
    assert torch.equal(total_ce, classification)
    assert torch.equal(classification, classification_crr)
    assert torch.equal(ranking, ranking_crr)
    assert torch.equal(total_crr, classification + ranking)
    assert ranking.item() == pytest.approx(np.log(2.0))


def test_matched_models_begin_from_identical_frozen_fcnn_weights() -> None:
    first = initialized_fcnn(0)
    second = initialized_fcnn(0)
    assert model_state_sha256(first) == model_state_sha256(second)
    assert count_parameters(first) == count_parameters(second) == EXPECTED_PARAMETERS
    assert list(first.state_dict()) == ["2.weight", "2.bias", "4.weight", "4.bias"]


def test_matched_models_use_identical_seeded_cycle_order() -> None:
    first = deterministic_cycle_orders(range(8), seed=0, epochs=40)
    second = deterministic_cycle_orders(range(8), seed=0, epochs=40)
    assert first == second
    assert cycle_order_sha256(first) == cycle_order_sha256(second)
    assert all(sorted(epoch) == list(range(8)) for epoch in first)


def test_session_metric_concatenates_oof_blocks_instead_of_mean_fold_ba() -> None:
    predictions = pd.DataFrame(
        {
            "session": ["708"] * 6,
            "model": ["crr_fcnn"] * 6,
            "fold": [1, 1, 2, 2, 2, 2],
            "block_id": [f"b{i}" for i in range(6)],
            "truth": [0, 1, 0, 0, 0, 1],
            "pred": [0, 1, 1, 1, 1, 1],
        }
    )
    observed = session_oof_balanced_accuracy(predictions).iloc[0]["oof_balanced_accuracy"]
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


def test_screening_plan_is_exactly_82_blockce_plus_82_crr() -> None:
    fold_counts = dict(zip(EXPECTED_SESSIONS, [8, 8, 6, 10, 10, 10, 10, 10, 10]))
    rows = []
    for session, count in fold_counts.items():
        for fold in range(1, count + 1):
            rows.append(
                {
                    "session": session,
                    "seed": 0,
                    "fold": fold,
                    "train_cycles": "1,2",
                    "test_cycles": "0",
                    "n_train_samples": 8,
                    "n_test_samples": 4,
                }
            )
    plan = build_screening_plan(pd.DataFrame(rows))
    assert len(plan) == 164
    assert plan.groupby("model").size().to_dict() == {
        "block_ce_fcnn": 82,
        "crr_fcnn": 82,
    }
    assert set(plan["seed"]) == {0}


def test_only_frozen_criteria_abcd_control_screening_decision() -> None:
    table = pd.DataFrame(
        {
            "session": EXPECTED_SESSIONS,
            "historical_fcnn_latefusion_seed0_BA": [0.60] * 9,
            "block_ce_fcnn_seed0_BA": [0.60] * 9,
            "crr_fcnn_seed0_BA": [0.62] * 9,
            "p_value": [1.0] * 9,
            "unrelated_metric": [-999.0] * 9,
        }
    )
    result = evaluate_screening_gate(table)
    assert result["controlling_criteria"] == ["A", "B", "C", "D"]
    assert result["thresholds"] == FROZEN_GATE
    assert all(value["passed"] for value in result["criteria"].values())
    assert result["p_value_computed"] is False
    assert result["decision"] == "supports_full_evaluation_cycle_relative_ranking_fcnn"
    failed = table.copy()
    failed["crr_fcnn_seed0_BA"] = 0.605
    assert (
        evaluate_screening_gate(failed)["decision"]
        == "does_not_support_cycle_relative_ranking_fcnn"
    )
