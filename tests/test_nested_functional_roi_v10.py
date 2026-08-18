from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import (
    BLOCK_NAMES,
    BlockSequenceData,
    load_block_sequence_session,
)
from ultrasound_decoding.nested_functional_roi_v10 import (
    FIXED_ORIENTATIONS,
    MODELS,
    REQUIRED_OUTPUTS,
    ROI_RULES,
    SESSIONS,
    cycle_response_maps,
    expected_outputs,
    load_v9_metrics,
    mask_overlap,
    orient_clean4,
    roi_flat4_features,
    roi_mean4_features,
    run_session,
    top_fraction_mask,
    training_z_map,
    whole_brain_flat4_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed_data/block_sequences_v1"
IMAGE_SHAPE = (128, 501)


@pytest.fixture(scope="module")
def synthetic_data() -> BlockSequenceData:
    n_cycles = 3
    rows: list[dict[str, object]] = []
    samples: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[int] = []
    base = np.linspace(-0.5, 0.5, np.prod(IMAGE_SHAPE), dtype=np.float32).reshape(IMAGE_SHAPE)
    offsets = {"grating": 2.0, "stop_after_grating": 0.2, "dot": 1.4, "static": 0.0}
    for cycle in range(n_cycles):
        for order, name in enumerate(BLOCK_NAMES):
            signal = offsets[name] + (0.05 + 0.02 * cycle) * base
            frames = np.stack([signal + 0.01 * frame for frame in range(4)]).astype(np.float32)
            samples.append(frames)
            labels.append(int(name in {"grating", "dot"}))
            groups.append(cycle)
            rows.append({
                "session": "708", "cycle": cycle, "block_name": name,
                "block_id": f"session708_cycle{cycle:03d}_{name}",
                "block_order_in_cycle": order,
            })
    n = len(samples)
    return BlockSequenceData(
        session="708", task="binary", X=np.stack(samples),
        y=np.asarray(labels, dtype=np.int64), groups=np.asarray(groups, dtype=np.int64),
        metadata=pd.DataFrame(rows),
        clean4_relative_time_s=np.tile(np.arange(4, dtype=np.float32), (n, 1)),
        clean4_original_frame_indices=np.tile(np.arange(4, dtype=np.int64), (n, 1)),
        source_h5_path=Path("synthetic.h5"), source_metadata_path=Path("synthetic.csv"),
    )


@pytest.fixture(scope="module")
def mean_only_result(synthetic_data: BlockSequenceData):
    return run_session(
        synthetic_data,
        models=("roi_mean4_top10_rlda",),
        roi_rules={"functional_roi_top10": 0.10},
        max_folds=3,
    )


def test_01_fixed_nine_session_manifest() -> None:
    assert SESSIONS == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_real_clean4_block_loader_is_frozen() -> None:
    data = load_block_sequence_session(PROJECT_ROOT, "626", "binary", data_dir=DATA_DIR)
    assert data.X.ndim == 4
    assert data.X.shape[1:] == (4, 128, 501)
    assert len(data.X) == 4 * data.n_cycles
    assert set(data.metadata["block_name"]) == set(BLOCK_NAMES)


def test_03_cycle_grouping_has_no_train_test_overlap(synthetic_data: BlockSequenceData) -> None:
    splits = grouped_cv_splits(synthetic_data.groups, max_folds=3)
    assert len(splits) == 3
    for train, test in splits:
        assert set(synthetic_data.groups[train]).isdisjoint(set(synthetic_data.groups[test]))


def test_04_each_fold_roi_uses_only_training_cycles(mean_only_result) -> None:
    for row in mean_only_result.roi_audit_rows:
        assert row["roi_fit_cycles"] == row["train_cycles"]


def test_05_test_cycles_do_not_participate_in_roi(mean_only_result) -> None:
    for row in mean_only_result.roi_audit_rows:
        assert set(row["roi_fit_cycles"].split(",")).isdisjoint(row["test_cycles"].split(","))
        assert row["test_cycles_used_for_roi"] is False


def test_06_cycle_contrast_is_exact_raw_clean4_formula(synthetic_data: BlockSequenceData) -> None:
    maps, cycles = cycle_response_maps(
        synthetic_data.X, synthetic_data.metadata, synthetic_data.groups, [0],
    )
    rows = synthetic_data.metadata[synthetic_data.groups == 0]
    block = {
        row.block_name: synthetic_data.X[int(index)].mean(axis=0)
        for index, row in rows.iterrows()
    }
    expected = 0.5 * (block["grating"] + block["dot"]) - 0.5 * (
        block["stop_after_grating"] + block["static"]
    )
    assert cycles.tolist() == [0]
    assert np.allclose(maps[0], expected)


def test_07_training_z_map_formula() -> None:
    maps = np.stack([
        np.full(IMAGE_SHAPE, 1.0),
        np.full(IMAGE_SHAPE, 3.0),
        np.full(IMAGE_SHAPE, 5.0),
    ])
    expected = maps.mean(axis=0) / (maps.std(axis=0, ddof=0) + 1.0e-8)
    assert np.allclose(training_z_map(maps), expected)


def test_08_top10_exact_pixel_count() -> None:
    z_map = np.arange(np.prod(IMAGE_SHAPE), dtype=float).reshape(IMAGE_SHAPE)
    assert int(top_fraction_mask(z_map, 0.10).sum()) == math.ceil(z_map.size * 0.10)


def test_09_top20_exact_pixel_count() -> None:
    z_map = np.arange(np.prod(IMAGE_SHAPE), dtype=float).reshape(IMAGE_SHAPE)
    assert int(top_fraction_mask(z_map, 0.20).sum()) == math.ceil(z_map.size * 0.20)


def test_10_top10_is_nested_inside_top20_without_retuning() -> None:
    z_map = np.arange(np.prod(IMAGE_SHAPE), dtype=float).reshape(IMAGE_SHAPE)
    top10 = top_fraction_mask(z_map, ROI_RULES["functional_roi_top10"])
    top20 = top_fraction_mask(z_map, ROI_RULES["functional_roi_top20"])
    assert np.all(top20[top10])


def test_11_roi_mean4_feature_dimension(synthetic_data: BlockSequenceData) -> None:
    mask = np.zeros(IMAGE_SHAPE, dtype=bool)
    mask[:2, :3] = True
    features = roi_mean4_features(synthetic_data.X, mask)
    assert features.shape == (len(synthetic_data.X), 4)


def test_12_roi_flat4_feature_dimension_and_time_order(synthetic_data: BlockSequenceData) -> None:
    mask = np.zeros(IMAGE_SHAPE, dtype=bool)
    mask[0, :3] = True
    features = roi_flat4_features(synthetic_data.X, mask)
    assert features.shape == (len(synthetic_data.X), 4 * 3)
    expected = np.asarray(synthetic_data.X[0], dtype=np.float64)[:, mask].reshape(-1)
    assert np.array_equal(features[0], expected)


def test_13_whole_brain_baseline_input_is_flat4(synthetic_data: BlockSequenceData) -> None:
    features = whole_brain_flat4_features(synthetic_data.X)
    assert features.shape == (len(synthetic_data.X), 4 * 128 * 501)
    assert np.array_equal(features[0], np.asarray(synthetic_data.X[0], dtype=np.float64).reshape(-1))


def test_14_oof_predictions_cover_each_sample_once(mean_only_result, synthetic_data: BlockSequenceData) -> None:
    rows = pd.DataFrame(mean_only_result.oof_rows)
    assert len(rows) == len(synthetic_data.X)
    assert rows["sample_index"].nunique() == len(synthetic_data.X)
    assert not rows.duplicated(["model", "sample_index"]).any()


def test_15_class_balance_audit_is_exact(mean_only_result) -> None:
    audit = mean_only_result.class_balance_row
    assert audit["n_no_stimulus"] == audit["n_stimulus"]
    assert audit["class_ratio_stimulus"] == 0.5
    assert audit["class_balance_exact_1_to_1"] is True
    assert mean_only_result.summary_rows[0]["accuracy"] == mean_only_result.summary_rows[0]["balanced_accuracy"]


def test_16_roi_overlap_metrics_are_exact() -> None:
    a = np.array([[1, 1], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [1, 0]], dtype=bool)
    jaccard, dice = mask_overlap(a, b)
    assert jaccard == pytest.approx(1 / 3)
    assert dice == pytest.approx(1 / 2)


def test_17_missing_v9_summary_degrades_safely(tmp_path: Path) -> None:
    table, audit = load_v9_metrics(tmp_path / "does_not_exist")
    assert table.empty
    assert audit.iloc[0]["status"] == "MISSING_SAFE_DEGRADE"


def test_18_session_807_uses_only_frozen_vertical_flip() -> None:
    assert FIXED_ORIENTATIONS["807"] == "flip_vertical"
    array = np.arange(4 * 128 * 501, dtype=float).reshape(1, 4, 128, 501)
    assert np.array_equal(orient_clean4(array, "flip_vertical"), array[:, :, ::-1, :])


def test_19_user_drawn_roi_is_never_used(mean_only_result) -> None:
    assert all(row["user_drawn_roi_used"] is False for row in mean_only_result.roi_audit_rows)
    source = inspect.getsource(run_session)
    assert "candidate_roi" not in source and "roi_annotation" not in source


def test_20_full_session_roi_or_v9_map_is_never_used(mean_only_result) -> None:
    assert all(row["full_session_roi_used"] is False for row in mean_only_result.roi_audit_rows)
    assert all(row["v9_map_used_for_roi"] is False for row in mean_only_result.roi_audit_rows)


def test_21_no_cross_session_transfer_or_registration(mean_only_result) -> None:
    assert all(row["cross_session_transfer"] is False for row in mean_only_result.roi_audit_rows)
    assert all(row["registration_used"] is False for row in mean_only_result.roi_audit_rows)


def test_22_required_output_manifest_is_complete(tmp_path: Path) -> None:
    paths = expected_outputs(tmp_path)
    assert len(paths) == len(REQUIRED_OUTPUTS)
    assert len(set(paths)) == len(paths)
    required = {
        "within_session_roi_decoding_summary.csv", "fold_level_roi_results.csv",
        "roi_stability_summary.csv", "roi_gain_vs_v9_metrics.csv",
        "functional_roi_overview_top10.png", "functional_roi_overview_top20.png",
        "within_session_binary_roi_vs_wholebrain.png", "roi_gain_top10_by_session.png",
        "roi_size_stability_by_session.png", "roi_gain_vs_v9_metrics.png",
        "nested_functional_roi_decoding_report.md", "pytest_output_local.txt",
        "smoke_test_local.txt", "run_command_server.txt", "run_log_server.txt",
    }
    assert required.issubset({path.name for path in paths})


def test_23_only_preregistered_five_models_and_two_roi_rules() -> None:
    assert MODELS == (
        "whole_brain_clean4_flat4_pca_lda", "roi_mean4_top10_rlda",
        "roi_flat4_top10_pca_lda", "roi_mean4_top20_rlda", "roi_flat4_top20_pca_lda",
    )
    assert ROI_RULES == {"functional_roi_top10": 0.10, "functional_roi_top20": 0.20}


def test_24_roi_selection_audit_forbids_morphology(mean_only_result) -> None:
    assert all(row["morphology_used"] is False for row in mean_only_result.roi_audit_rows)
