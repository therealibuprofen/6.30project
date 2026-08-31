from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.multiframe.cycle_calibrated_late_fusion import (
    FORMAL_TRAINING_CONFIG, build_inner_cache_key, build_task_inner_cache_key,
)
from ultrasound_decoding.multiframe.dqi_dot_vs_grating import (
    TASK_NAME, block_predictions_from_frame_logits,
    concatenated_oof_balanced_accuracy, cross_task_relationship_matrix,
    evaluate_confirmatory_gate, exact_spearman_permutation, leave_one_session_out,
    mean_inner_fold_ba_diagnostic, validate_authoritative_mapping,
)


def _frame_rows() -> pd.DataFrame:
    rows = []
    # Deliberately unequal inner-fold sizes: fold 1 is wrong (2 blocks), folds 2/3 correct (6 blocks).
    for source, (inner, truth, correct) in enumerate(((1, 0, False), (1, 1, False), (2, 0, True), (2, 1, True), (2, 0, True), (2, 1, True), (3, 0, True), (3, 1, True))):
        for pos in range(4):
            predicted = truth if correct else 1 - truth
            logits = (5.0, -5.0) if predicted == 0 else (-5.0, 5.0)
            rows.append({"session": "626", "outer_seed": 0, "outer_fold": 1, "inner_fold": inner, "source_index": source, "block_id": f"b{source}", "cycle": source + 1, "frame_position": pos, "truth": truth, "logit_dot": logits[0], "logit_grating": logits[1]})
    return pd.DataFrame(rows)


def test_mapping_and_four_frame_probability_fusion_and_concatenated_q() -> None:
    validate_authoritative_mapping()
    blocks = block_predictions_from_frame_logits(_frame_rows())
    assert len(blocks) == 8
    assert blocks.n_frames_fused.eq(4).all()
    assert np.allclose(blocks[["prob_dot", "prob_grating"]].sum(axis=1), 1.0)
    q = concatenated_oof_balanced_accuracy(blocks)
    diagnostic = mean_inner_fold_ba_diagnostic(blocks)
    assert q == pytest.approx(0.75)
    assert diagnostic == pytest.approx(2 / 3)
    assert q != diagnostic


def test_task_explicit_cache_identity_cannot_collide_with_binary_identity() -> None:
    common = dict(session="626", outer_fold=1, outer_seed=0, outer_train_cycles=(1, 2, 3), inner_fold=1, inner_train_cycles=(2, 3), inner_validation_cycles=(1,), source_hash="source", protocol_hash="protocol", normalization_fingerprint="normalization", training_config=vars(FORMAL_TRAINING_CONFIG))
    binary = build_inner_cache_key(**common)
    dot_grating = build_task_inner_cache_key(task=TASK_NAME, **common)
    assert binary != dot_grating


def test_exact_permutation_loso_gate_and_cross_task_matrix() -> None:
    q = np.arange(9, dtype=float)
    exact = exact_spearman_permutation(q, q)
    assert exact.total == 362880
    assert exact.observed == pytest.approx(1.0)
    assert exact.two_sided_p == pytest.approx(2 / 362880)
    summary = pd.DataFrame({"session": [str(x) for x in range(9)], "Q_DG_session": q, "BA_DG_session": q, "Q_presence_session": q[::-1], "BA_presence_session": q})
    loso = leave_one_session_out(summary, "Q_DG_session", "BA_DG_session")
    assert len(loso) == 9
    assert loso.spearman_rho.eq(1.0).all()
    decision = evaluate_confirmatory_gate(session_spearman=1.0, exact_p=exact.two_sided_p, loo_median=1.0, loo_minimum=1.0)
    assert decision["decision"] == "supports_cross_task_DQI_validation_dot_vs_grating"
    matrix = cross_task_relationship_matrix(summary)
    assert set(matrix.relationship) == {"Q_presence_to_BA_presence", "Q_DG_to_BA_DG", "Q_presence_to_BA_DG", "Q_DG_to_BA_presence", "Q_presence_to_Q_DG"}
    assert matrix.enters_confirmatory_gate.sum() == 1


def test_oof_rejects_duplicate_frame_or_not_four_frames() -> None:
    rows = _frame_rows()
    with pytest.raises(AssertionError):
        block_predictions_from_frame_logits(pd.concat([rows, rows.iloc[[0]]], ignore_index=True))
    with pytest.raises(AssertionError):
        block_predictions_from_frame_logits(rows.iloc[1:].copy())
