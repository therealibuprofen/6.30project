from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.cross_session_feature_factor_v7 import exact_spearman_permutation
from ultrasound_decoding.spatial_glm_reproducibility_reporting_v9 import make_report
from ultrasound_decoding.spatial_glm_reproducibility_v9 import (
    CONDITION_ORDER,
    CONTRAST_WEIGHTS,
    EXPECTED_SESSIONS,
    FDR_ALPHA,
    FIXED_ORDER_WARNING,
    FIXED_ORIENTATIONS,
    GD_WARNING,
    IMAGE_SHAPE,
    N_BOOTSTRAP,
    N_SPLITS,
    analyze_session,
    apply_fixed_orientation,
    benjamini_hochberg,
    block_mean_clean4,
    bootstrap_gs_ds_concordance,
    build_diagnostic_table,
    cycle_contrast_maps,
    design_contrast,
    fit_pixelwise_glm,
    holm_adjust,
    load_session_block_images,
    load_v8_metrics,
    load_within_session_ba,
    planned_ba_associations,
    random_cycle_halves,
    repeated_measures_design,
    required_output_paths,
    safe_cosine,
    safe_pearson,
    split_half_reproducibility,
    v8_v9_stability_association,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"
V7_DIR = PROJECT_DIR / "outputs/cross_session_feature_factor_analysis_9sessions_v7"
V8_DIR = PROJECT_DIR / "outputs/session_centered_stimulus_vector_alignment_9sessions_v8"


def _synthetic_images(n_cycles: int = 5) -> np.ndarray:
    rng = np.random.default_rng(17)
    images = rng.normal(0, 0.2, size=(n_cycles, 4, *IMAGE_SHAPE))
    images += np.arange(n_cycles, dtype=float)[:, None, None, None] * 0.4
    images[:, 0] += 2.0
    images[:, 1] += 0.2
    images[:, 2] += 1.1
    images[:, 3] -= 0.1
    return images


def _summary_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = list(EXPECTED_SESSIONS)
    index = np.arange(9, dtype=float)
    glm = pd.DataFrame({
        "session": np.repeat(sessions, 4),
        "contrast": [name for _ in sessions for name in CONTRAST_WEIGHTS],
        "n_cycles": np.repeat(np.arange(6, 15), 4),
        "RMS_effect": np.repeat(1 + index, 4),
        "RMS_standardized_effect": np.repeat(0.1 + index / 10, 4),
        "n_fdr_pixels": np.repeat(index.astype(int), 4),
        "fdr_fraction": np.repeat(index / 1000, 4),
    })
    split = pd.DataFrame({
        "session": np.repeat(sessions, 4),
        "contrast": [name for _ in sessions for name in CONTRAST_WEIGHTS],
        "split_half_corr_median": np.repeat(-0.4 + index / 10, 4),
        "split_half_corr_2.5pct": np.repeat(-0.5 + index / 10, 4),
        "split_half_corr_97.5pct": np.repeat(-0.3 + index / 10, 4),
    })
    concordance = pd.DataFrame({
        "session": sessions,
        "GS_DS_spatial_corr": -0.3 + index / 10,
        "GS_DS_spatial_cosine": -0.2 + index / 10,
    })
    ba = pd.DataFrame({
        "session": sessions,
        "within_session_BA": 0.4 + index / 20,
        "within_session_BA_seed_std": np.full(9, 0.02),
    })
    v8 = pd.DataFrame({
        "session": sessions,
        "v8_stimulus_vector_magnitude": 10 + index,
        "v8_split_half_vector_stability": -0.2 + index / 10,
    })
    return glm, split, concordance, ba, v8


def test_frozen_sessions_data_and_config_exist() -> None:
    assert list(EXPECTED_SESSIONS) == ["626", "628", "708", "709", "710", "807", "813", "817", "822"]
    for session in EXPECTED_SESSIONS:
        assert (DATA_DIR / f"session_{session}_blocks.h5").is_file()
        assert (DATA_DIR / f"session_{session}_block_metadata.csv").is_file()
    config = json.loads((PROJECT_DIR / "configs/spatial_glm_contrast_reproducibility_9sessions_v9.json").read_text())
    assert config["sessions"] == list(EXPECTED_SESSIONS)
    assert config["n_splits"] == config["n_bootstrap"] == 1000


def test_clean4_is_reduced_to_one_block_observation_not_four_frame_samples() -> None:
    data = load_session_block_images(PROJECT_DIR, DATA_DIR, "626", max_cycles=2)
    assert data.images.shape == (2, 4, 128, 501)
    assert data.source.X.shape[1] == 4
    expected = np.arcsinh(data.source.X[0].astype(np.float32)).mean(axis=0, dtype=np.float64)
    assert np.allclose(data.images[0, 0], expected)
    assert len(data.cycle_ids) == 2


def test_every_cycle_has_exact_fixed_four_condition_order() -> None:
    for session in EXPECTED_SESSIONS:
        metadata = pd.read_csv(DATA_DIR / f"session_{session}_block_metadata.csv")
        orders = metadata.groupby("cycle", sort=True)["block_name"].apply(list)
        assert orders.apply(lambda value: value == list(CONDITION_ORDER)).all()
        assert (metadata.groupby("cycle").size() == 4).all()
        assert (metadata["n_frames_clean4"] == 4).all()


def test_807_uses_only_confirmed_fixed_vertical_orientation() -> None:
    assert FIXED_ORIENTATIONS["807"] == "flip_vertical"
    assert all(FIXED_ORIENTATIONS[s] == "identity" for s in EXPECTED_SESSIONS if s != "807")
    data = load_session_block_images(PROJECT_DIR, DATA_DIR, "807", max_cycles=2)
    unflipped = np.arcsinh(data.source.X[0].astype(np.float32)).mean(axis=0, dtype=np.float64)
    assert np.allclose(data.images[0, 0], np.flip(unflipped, axis=0))
    test_image = np.arange(np.prod(IMAGE_SHAPE)).reshape(IMAGE_SHAPE)
    assert np.array_equal(apply_fixed_orientation(test_image, "flip_vertical"), test_image[::-1])


def test_cycle_fixed_condition_design_is_full_rank_and_not_continuous_cycle() -> None:
    design, names = repeated_measures_design(6)
    assert design.shape == (24, 9)
    assert np.linalg.matrix_rank(design) == 9
    assert names[:2] == ("intercept", "cycle_1")
    assert not any(name == "cycle_continuous" for name in names)
    for cycle in range(6):
        assert np.all(design[4 * cycle : 4 * cycle + 4, 0] == 1)


def test_all_preregistered_condition_contrasts_and_design_vectors() -> None:
    expected = {
        "STIM_PRESENCE": [0.5, -0.5, 0.5, -0.5],
        "GS": [1, -1, 0, 0],
        "DS": [0, 0, 1, -1],
        "GD": [1, 0, -1, 0],
    }
    for name, weights in expected.items():
        assert np.array_equal(CONTRAST_WEIGHTS[name], weights)
        vector = design_contrast(weights, 5)
        assert vector.shape == (8,)
        assert np.all(vector[:-3] == 0)
        assert np.array_equal(vector[-3:], weights[:3])


def test_pixelwise_glm_maps_are_finite_correct_shape_and_effect_matches_cycles() -> None:
    images = _synthetic_images(5)
    maps, rank, residual_df = fit_pixelwise_glm(images)
    assert rank == 8
    assert residual_df == 12
    for name, result in maps.items():
        for value in (result.effect, result.standard_error, result.t_map, result.p_map, result.q_map, result.standardized):
            assert value.shape == IMAGE_SHAPE
            assert np.isfinite(value).all()
        assert result.fdr_mask.shape == IMAGE_SHAPE
        expected = cycle_contrast_maps(images, CONTRAST_WEIGHTS[name]).mean(axis=0)
        assert np.allclose(result.effect, expected)


def test_cycle_level_contrast_maps_preserve_complete_cycles() -> None:
    images = np.zeros((3, 4, *IMAGE_SHAPE), dtype=float)
    images[:, 0] = 4
    images[:, 1] = 1
    images[:, 2] = 2
    images[:, 3] = -1
    binary = cycle_contrast_maps(images, CONTRAST_WEIGHTS["STIM_PRESENCE"])
    gs = cycle_contrast_maps(images, CONTRAST_WEIGHTS["GS"])
    ds = cycle_contrast_maps(images, CONTRAST_WEIGHTS["DS"])
    gd = cycle_contrast_maps(images, CONTRAST_WEIGHTS["GD"])
    assert binary.shape == (3, *IMAGE_SHAPE)
    assert np.all(binary == 3)
    assert np.all(gs == 3)
    assert np.all(ds == 3)
    assert np.all(gd == 2)


def test_bh_fdr_is_correct_and_zero_pixel_case_is_safe() -> None:
    p = np.asarray([0.01, 0.04, 0.03, 0.20])
    q, mask = benjamini_hochberg(p)
    assert np.allclose(q, [0.04, 0.0533333333333, 0.0533333333333, 0.20])
    assert np.array_equal(mask, [True, False, False, False])
    q_zero, mask_zero = benjamini_hochberg(np.ones(10))
    assert np.all(q_zero == 1)
    assert not mask_zero.any()
    assert FDR_ALPHA == 0.05


def test_split_half_is_cycle_only_handles_odd_counts_and_formal_count_is_exact() -> None:
    rng = np.random.default_rng(5)
    cycles = np.arange(5)
    sizes = {len(random_cycle_halves(cycles, rng)[0]) for _ in range(100)}
    assert sizes == {2, 3}
    maps = np.random.default_rng(2).normal(size=(5, 4, 7))
    summary = split_half_reproducibility(maps, n_splits=N_SPLITS, seed=9)
    assert summary["n_splits"] == 1000
    assert summary["half_a_min_n"] == 2
    assert summary["half_a_max_n"] == 3
    assert summary["split_unit"] == "complete_cycle"


def test_safe_pearson_and_cosine_handle_constant_or_zero_maps() -> None:
    assert safe_pearson(np.ones(8), np.ones(8)) == 0.0
    assert safe_cosine(np.zeros(8), np.ones(8)) == 0.0
    assert safe_pearson(np.arange(8), np.arange(8)) == pytest.approx(1.0)
    assert safe_cosine(np.arange(8), np.arange(8)) == pytest.approx(1.0)


def test_gs_ds_bootstrap_is_paired_within_session_and_formal_count_frozen() -> None:
    gs = np.random.default_rng(1).normal(size=(4, 3, 5))
    ds = gs + np.random.default_rng(2).normal(scale=0.01, size=gs.shape)
    summary = bootstrap_gs_ds_concordance(gs, ds, n_bootstrap=11, seed=4)
    assert summary["n_bootstrap"] == 11
    assert summary["resampling_unit"] == "complete_cycle_paired_GS_DS"
    assert summary["bootstrap_corr_median"] > 0.99
    assert N_BOOTSTRAP == 1000


def test_v7_within_ba_reuse_is_exact_smallcnn_feature_mean_artifact() -> None:
    values, audit, root = load_within_session_ba(V7_DIR)
    assert values["session"].tolist() == list(EXPECTED_SESSIONS)
    assert np.isfinite(values["within_session_BA"]).all()
    assert (audit["source_model"] == "SmallCNN feature-mean").all()
    assert not audit["recomputed_or_reselected"].any()
    assert "smoke" not in str(root)


def test_v8_metric_reuse_selects_formal_raw_spatial_pca_rows() -> None:
    values, audit, root = load_v8_metrics(V8_DIR)
    assert values["session"].tolist() == list(EXPECTED_SESSIONS)
    assert list(values.columns) == [
        "session", "v8_stimulus_vector_magnitude", "v8_split_half_vector_stability"
    ]
    assert (audit["representation"] == "RAW_SPATIAL_PCA").all()
    assert (audit["task"] == "stimulus_presence").all()
    assert not audit["recomputed_or_reselected"].any()
    assert "smoke" not in str(root)


def test_exact_nine_session_permutation_and_holm_family() -> None:
    x = np.arange(9, dtype=float)
    result = exact_spearman_permutation(x, x)
    assert result["rho"] == pytest.approx(1.0)
    assert result["n_permutations"] == 362880
    assert result["permutation_method"] == "exact_complete_enumeration"
    assert np.allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    frames = _summary_frames()
    diagnostic = build_diagnostic_table(*frames)
    associations = planned_ba_associations(diagnostic)
    assert len(associations) == 3
    assert set(associations["planned_family_label"]) == {"effect_magnitude", "reproducibility", "GS_DS_concordance"}
    assert (associations["multiple_testing_family"] == "three_planned_BA_associations_only").all()


def test_v8_v9_stability_is_secondary_and_outside_holm() -> None:
    diagnostic = build_diagnostic_table(*_summary_frames())
    result = v8_v9_stability_association(diagnostic)
    assert len(result) == 1
    assert result.iloc[0]["analysis_role"] == "secondary_mechanistic_association"
    assert not bool(result.iloc[0]["included_in_primary_Holm_family"])
    assert result.iloc[0]["n_permutations"] == 362880


def test_diagnostic_patterns_are_descriptive_median_split_without_subgroup_test() -> None:
    diagnostic = build_diagnostic_table(*_summary_frames())
    assert set(diagnostic["spatial_diagnostic_pattern"]).issubset({"S1", "S2", "S3", "S4"})
    assert (diagnostic["threshold_rule"] == "9-session median split; descriptive only; no subgroup test").all()
    assert diagnostic["session"].tolist() == list(EXPECTED_SESSIONS)


def test_forbidden_branches_and_cross_session_pixel_map_correlation_are_absent() -> None:
    source = inspect.getsource(__import__(
        "ultrasound_decoding.spatial_glm_reproducibility_v9", fromlist=["dummy"]
    ))
    assert "def cross_session_map_corr" not in source
    assert "def train_decoder" not in source
    assert "def select_roi" not in source
    assert "def hrf_search" not in source


def test_fixed_order_warning_and_exploratory_gd_warning_are_mandatory() -> None:
    assert "fixed within-cycle temporal position" in FIXED_ORDER_WARNING
    assert "EXPLORATORY" in GD_WARNING
    assert "TEMPORAL-POSITION CONFOUNDED" in GD_WARNING
    frames = _summary_frames()
    diagnostic = build_diagnostic_table(*frames)
    associations = planned_ba_associations(diagnostic)
    report = make_report(diagnostic, frames[0].assign(n_fdr_pixels=0), associations, v8_v9_stability_association(diagnostic))
    assert "No HRF parameter search was performed in v9." in report
    assert "not anatomically registered" in report
    assert "FDR masks are not used to select ROIs" in report
    assert GD_WARNING in report


def test_formal_defaults_and_output_schema_are_complete() -> None:
    assert N_SPLITS == 1000
    assert N_BOOTSTRAP == 1000
    paths = set(required_output_paths())
    required = {
        "audit/clean4_identity_check.csv",
        "audit/within_session_ba_reuse.csv",
        "audit/v8_metric_reuse.csv",
        "glm/pixelwise_glm_summary.csv",
        "reproducibility/split_half_metrics.csv",
        "reproducibility/gs_ds_concordance.csv",
        "summaries/session_spatial_diagnostic_table.csv",
        "figures/binary_spatial_maps_9sessions.png",
        "figures/spatial_diagnostic_overview.png",
        "report/spatial_glm_reproducibility_report.md",
    }
    assert required.issubset(paths)
    for session in EXPECTED_SESSIONS:
        for contrast in CONTRAST_WEIGHTS:
            assert f"glm/contrast_maps/session_{session}_{contrast}.npy" in paths
            assert f"glm/standard_error_maps/session_{session}_{contrast}.npy" in paths
            assert f"glm/fdr_masks/session_{session}_{contrast}.npy" in paths


def test_session_analysis_reports_zero_fdr_safely_and_no_frame_replication() -> None:
    data = load_session_block_images(PROJECT_DIR, DATA_DIR, "626", max_cycles=2)
    result = analyze_session(data, n_splits=3, n_bootstrap=3)
    assert all(row["n_observations"] == 4 * data.n_cycles for row in result.glm_rows)
    for row in result.glm_rows:
        if row["n_fdr_pixels"] == 0:
            assert np.isnan(row["mean_abs_effect_FDR_pixels"])
    assert result.concordance_row["comparison_scope"] == "within_session_only_not_anatomically_registered"
