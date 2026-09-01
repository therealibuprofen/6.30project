from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.multiframe.dataset import BlockSequenceData
from ultrasound_decoding.multiframe.within_session_cycle_drift import (
    INTERPRETATION_RULE,
    PRIMARY_BLOCK_TYPES,
    build_primary_templates,
    cycle_template,
    flip_invariance_audit,
    fold_train_test_drift,
    median_pairwise_spatial_correlation,
    pixelwise_training_reference,
    reconstruct_historical_decoder,
    spatial_correlation,
    spatial_zscore,
    summarize_session_drift,
)
import ultrasound_decoding.multiframe.within_session_cycle_drift as drift_module


BLOCK_NAMES = ("grating", "stop_after_grating", "dot", "static")


def synthetic_data(session: str = "708", n_cycles: int = 3) -> BlockSequenceData:
    rows = []
    blocks = []
    labels = []
    groups = []
    base = np.arange(24, dtype=np.float32).reshape(4, 2, 3) + 1.0
    for cycle in range(n_cycles):
        for block_index, name in enumerate(BLOCK_NAMES):
            frame_block = np.stack(
                [base[frame] + cycle * (block_index + 1) + frame * 0.1 for frame in range(4)],
                axis=0,
            )
            blocks.append(frame_block)
            labels.append(1 if name in {"grating", "dot"} else 0)
            groups.append(cycle)
            rows.append(
                {
                    "session": session,
                    "cycle": cycle,
                    "block_name": name,
                    "block_id": f"session{session}_cycle{cycle:03d}_{name}",
                }
            )
    X = np.asarray(blocks, dtype=np.float32)
    return BlockSequenceData(
        session=session,
        task="binary",
        X=X,
        y=np.asarray(labels, dtype=np.int64),
        groups=np.asarray(groups, dtype=np.int64),
        metadata=pd.DataFrame(rows),
        clean4_relative_time_s=np.tile(np.arange(4), (len(X), 1)).astype(np.float32),
        clean4_original_frame_indices=np.tile(np.arange(4), (len(X), 1)).astype(np.int64),
        source_h5_path=Path("unused.h5"),
        source_metadata_path=Path("unused.csv"),
    )


def test_primary_analysis_uses_only_stop_and_static() -> None:
    bundle = build_primary_templates(synthetic_data())
    assert {block_type for _, block_type in bundle.templates} == set(PRIMARY_BLOCK_TYPES)
    assert set(bundle.metrics["block_type"]) == {"stop_after_grating", "static"}
    assert "grating" not in set(bundle.metrics["block_type"])
    assert "dot" not in set(bundle.metrics["block_type"])


def test_stop_and_static_are_summarized_separately_then_equal_weighted() -> None:
    bundle = build_primary_templates(synthetic_data())
    _pairs, summary = summarize_session_drift(bundle)
    expected = np.mean(
        [summary["stop_spatial_stability"], summary["static_spatial_stability"]]
    )
    assert summary["background_spatial_stability"] == pytest.approx(expected)


def test_cycle_template_is_mean_of_four_arcsinh_frames() -> None:
    clean4 = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    assert np.allclose(cycle_template(clean4), np.arcsinh(clean4).mean(axis=0))


def test_spatial_correlation_uses_within_image_zscore() -> None:
    image = np.arange(12, dtype=np.float64).reshape(3, 4)
    transformed = image * 7.0 + 50.0
    assert spatial_zscore(image).mean() == pytest.approx(0.0, abs=1e-12)
    assert spatial_correlation(image, transformed) == pytest.approx(1.0, abs=1e-12)


def test_session_stability_is_median_pairwise_cycle_correlation() -> None:
    images = np.asarray(
        [
            [[0.0, 1.0], [2.0, 4.0]],
            [[0.0, 2.0], [1.0, 3.0]],
            [[3.0, 1.0], [0.0, 2.0]],
        ]
    )
    manual = np.median(
        [
            spatial_correlation(images[0], images[1]),
            spatial_correlation(images[0], images[2]),
            spatial_correlation(images[1], images[2]),
        ]
    )
    assert median_pairwise_spatial_correlation(images) == pytest.approx(manual)


def test_primary_drift_is_one_minus_background_stability() -> None:
    _pairs, summary = summarize_session_drift(build_primary_templates(synthetic_data()))
    assert summary["background_spatial_drift"] == pytest.approx(
        1.0 - summary["background_spatial_stability"]
    )
    assert summary["primary_metric"] == "background_spatial_drift"


def test_fold_reference_uses_only_outer_training_cycles() -> None:
    bundle = build_primary_templates(synthetic_data())
    observed = pixelwise_training_reference(
        bundle.templates, [0, 1], "stop_after_grating"
    )
    expected = np.median(
        np.stack(
            [
                bundle.templates[(0, "stop_after_grating")],
                bundle.templates[(1, "stop_after_grating")],
            ]
        ),
        axis=0,
    )
    assert np.array_equal(observed, expected)


def test_mutating_test_cycle_pixels_cannot_change_training_reference() -> None:
    bundle = build_primary_templates(synthetic_data())
    before = pixelwise_training_reference(bundle.templates, [0, 1], "static")
    bundle.templates[(2, "static")][:] += 10000.0
    after = pixelwise_training_reference(bundle.templates, [0, 1], "static")
    assert np.array_equal(before, after)


def test_fold_drift_equally_averages_test_cycles_times_stop_static() -> None:
    bundle = build_primary_templates(synthetic_data(n_cycles=4))
    row = fold_train_test_drift(
        bundle, fold=1, training_cycles=[0, 1], test_cycles=[2, 3]
    )
    references = {
        name: pixelwise_training_reference(bundle.templates, [0, 1], name)
        for name in PRIMARY_BLOCK_TYPES
    }
    values = [
        spatial_correlation(references[name], bundle.templates[(cycle, name)])
        for cycle in [2, 3]
        for name in PRIMARY_BLOCK_TYPES
    ]
    assert row["fold_background_similarity"] == pytest.approx(np.mean(values))
    assert row["fold_train_test_drift"] == pytest.approx(1.0 - np.mean(values))
    assert row["training_reference_uses_test_cycles"] is False


def historical_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for seed in [0, 1, 2]:
        for fold, (truth, pred) in enumerate(
            [([0, 1], [0, 1]), ([0, 0, 0, 1], [1, 1, 1, 1])], start=1
        ):
            for index, (y, p) in enumerate(zip(truth, pred)):
                rows.append(
                    {
                        "session": "708",
                        "seed": seed,
                        "fold": fold,
                        "block_id": f"s{seed}_f{fold}_b{index}",
                        "truth": y,
                        "pred": p,
                    }
                )
    predictions = pd.DataFrame(rows)
    saved = pd.DataFrame(
        {"session": ["708"] * 3, "seed": [0, 1, 2], "late_fusion_BA": [0.625] * 3}
    )
    return predictions, saved


def test_historical_session_ba_concatenates_oof_then_means_seeds() -> None:
    predictions, saved = historical_tables()
    formal, _fold, audit = reconstruct_historical_decoder(predictions, saved)
    assert formal.iloc[0]["formal_session_FCNN_latefusion_BA"] == pytest.approx(0.625)
    assert audit["session_metric"].startswith("concatenate all OOF blocks")


def test_fold_ba_is_averaged_across_three_seeds() -> None:
    predictions, saved = historical_tables()
    _formal, folds, _audit = reconstruct_historical_decoder(predictions, saved)
    assert folds.set_index("fold").loc[1, "fold_FCNN_latefusion_BA_seedavg"] == 1.0
    assert folds.set_index("fold").loc[2, "fold_FCNN_latefusion_BA_seedavg"] == 0.5


def test_uniform_vertical_flip_preserves_807_primary_drift() -> None:
    audit = flip_invariance_audit(build_primary_templates(synthetic_data("807", 4)))
    assert audit["status"] == "PASS"
    assert audit["absolute_difference"] <= audit["tolerance"]
    assert audit["raw_data_modified"] is False


def test_original_data_array_is_never_modified() -> None:
    data = synthetic_data()
    before = data.X.copy()
    _bundle = build_primary_templates(data)
    assert np.array_equal(data.X, before)


def test_analysis_module_has_no_training_optimizer_or_checkpoint_code() -> None:
    source = inspect.getsource(drift_module)
    assert "import torch" not in source
    assert "torch.optim" not in source
    assert "optimizer.step" not in source
    assert "torch.save" not in source
    assert "model_state_dict" not in source


def test_primary_metric_target_and_interpretation_rule_are_prelocked() -> None:
    assert PRIMARY_BLOCK_TYPES == ("stop_after_grating", "static")
    assert INTERPRETATION_RULE["candidate"]["session_spearman_maximum"] == -0.5
    assert INTERPRETATION_RULE["candidate"][
        "negative_within_session_fold_spearman_minimum"
    ] == 5
    _pairs, summary = summarize_session_drift(build_primary_templates(synthetic_data()))
    assert summary["primary_metric"] == "background_spatial_drift"
