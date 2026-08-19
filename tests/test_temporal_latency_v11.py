from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ultrasound_decoding.temporal_latency_v11 as v11
from ultrasound_decoding.temporal_latency_v11 import (
    CONDITIONS,
    FRAME_INTERVAL_SECONDS,
    IMAGE_SHAPE,
    ONSET_CRITERIA,
    REQUIRED_OUTPUTS,
    SESSIONS,
    complete_cycle_ids,
    config_freeze_text,
    detect_discrete_onset,
    exact_spearman_permutation,
    expected_outputs,
    extract_patch_timecourses,
    four_neighbor_pairs,
    holm_adjust,
    infer_timeline,
    load_temporal_session,
    make_report,
    neighbor_summary,
    patch_grid,
    peak_latency,
    safe_zscore,
    split_half_latency,
    summarize_patches,
    transition_windows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@pytest.fixture(scope="module")
def real_session_626():
    return load_temporal_session(DATA_ROOT, "626", max_cycles=2)


def test_01_fixed_nine_sessions_and_raw_directories_exist() -> None:
    assert SESSIONS == ("626", "628", "708", "709", "710", "807", "813", "817", "822")
    assert all((DATA_ROOT / session).is_dir() for session in SESSIONS)


def test_02_complete_cycle_full_temporal_sequence_is_30_frames(real_session_626) -> None:
    assert real_session_626.frames.shape == (2, 30, 128, 501)
    assert real_session_626.frame_indices.shape == (2, 30)
    assert complete_cycle_ids(np.arange(1, 65)) == [0, 1]


def test_03_actual_timestamps_must_be_monotonic() -> None:
    indices = np.arange(1, 5)
    timeline = infer_timeline(indices, np.array([10.0, 14.1, 18.0, 22.2]))
    assert timeline["timestamp_seconds"].is_monotonic_increasing
    assert (timeline["time_source"] == "ACTUAL_TIMESTAMP").all()
    with pytest.raises(ValueError, match="strictly increasing"):
        infer_timeline(indices, np.array([0.0, 4.0, 3.0, 8.0]))


def test_04_frame_interval_is_audited_correctly(real_session_626) -> None:
    audit = real_session_626.audit
    assert audit["frame_interval_seconds_mean"] == pytest.approx(120 / 30)
    assert audit["frame_interval_seconds_std"] == 0
    assert audit["temporal_resolution_seconds"] == pytest.approx(FRAME_INTERVAL_SECONDS)


def test_05_approximate_time_is_explicit(real_session_626) -> None:
    audit = real_session_626.audit
    assert audit["frame_interval_source"] == "INFERRED_FROM_FRAME_INDEX"
    assert audit["time_precision"] == "APPROXIMATE_FRAME_TIME"
    assert audit["timestamps_available"] is False


def test_06_fixed_block_boundaries_follow_frame_centers(real_session_626) -> None:
    names = real_session_626.conditions[0].tolist()
    assert [names.count(condition) for condition in CONDITIONS] == [7, 8, 7, 8]
    assert real_session_626.audit["grating_start_time"] == 0
    assert real_session_626.audit["dot_start_time"] == 60


def test_07_full_temporal_input_is_not_clean4(real_session_626) -> None:
    assert real_session_626.audit["full_temporal_sequence_used"] is True
    assert real_session_626.audit["clean4_used"] is False
    assert "processed_data" not in real_session_626.audit["raw_source"]
    assert real_session_626.frames.shape[1] != 4


def test_08_patch_grid_is_fixed_16_by_16_with_edge_retained() -> None:
    patches = patch_grid()
    assert len(patches) == 8 * 32
    assert (patches[0].y1 - patches[0].y0, patches[0].x1 - patches[0].x0) == (16, 16)
    assert patches[-1].x1 == 501
    assert patches[-1].x1 - patches[-1].x0 == 5


def test_09_patch_mean_is_pixelwise_mean() -> None:
    frames = np.zeros((1, 1, *IMAGE_SHAPE), dtype=np.float32)
    frames[0, 0, :16, :16] = np.arange(256, dtype=np.float32).reshape(16, 16)
    output = extract_patch_timecourses(frames, patch_grid()[:1])
    assert output.shape == (1, 1, 1)
    assert output[0, 0, 0] == pytest.approx(np.arange(256).mean())


def test_10_GS_onset_frames_are_located() -> None:
    conditions = np.array(["grating"] * 7 + ["stop_after_grating"] * 8 + ["dot"] * 7 + ["static"] * 8)
    baseline, post, status = transition_windows(conditions, "GS")
    assert baseline.tolist() == [0]
    assert post.tolist() == list(range(7))
    assert status.startswith("NO_TRUE_PRE_GRATING_BASELINE")


def test_11_DS_onset_frames_are_located() -> None:
    conditions = np.array(["grating"] * 7 + ["stop_after_grating"] * 8 + ["dot"] * 7 + ["static"] * 8)
    baseline, post, status = transition_windows(conditions, "DS")
    assert baseline.tolist() == [13, 14]
    assert post.tolist() == list(range(15, 22))
    assert status == "STOP_LAST_2_FRAMES"


def test_12_baseline_windows_are_frozen() -> None:
    conditions = np.array(["grating"] * 7 + ["stop_after_grating"] * 8 + ["dot"] * 7 + ["static"] * 8)
    gs_base, _, _ = transition_windows(conditions, "GS")
    ds_base, _, _ = transition_windows(conditions, "DS")
    assert len(gs_base) == 1
    assert len(ds_base) == 2


def test_13_z_score_is_finite_with_zero_baseline_std() -> None:
    z, mean, std = safe_zscore(np.array([1.0, 2.0]), np.array([1.0]))
    assert mean == 1 and std == 0
    assert np.isfinite(z).all()
    assert z[0] == 0


def test_14_primary_onset_requires_two_consecutive_frames() -> None:
    assert detect_discrete_onset([0, 2.1, 0, 2.2, 2.3], 2.0) == (3, 1)
    assert detect_discrete_onset([0, -2.1, -2.2, 0], 2.0) == (1, -1)


def test_15_no_detected_onset_is_none_not_zero_or_max() -> None:
    onset, direction = detect_discrete_onset([0, 2.1, 0, 1.9], 2.0)
    assert onset is None and direction is None


def test_16_latency_estimator_is_discrete_without_subframe_fit() -> None:
    onset, _ = detect_discrete_onset([0, 2.1, 2.2], 2.0)
    assert isinstance(onset, int)
    source = inspect.getsource(detect_discrete_onset)
    assert all(term not in source for term in ("spline", "interp1d", "curve_fit", "cubic"))


def test_17_peak_window_is_fixed_zero_to_twenty_seconds() -> None:
    frame, seconds = peak_latency([0, 1, 3, 2, 4, 99], [2, 6, 10, 14, 18, 22])
    assert frame == 4 and seconds == 18


def test_18_latency_stable_patch_rule_is_exact() -> None:
    onset_rows = []
    peak_rows = []
    for patch_id, values in ((0, [1, 1, 2, np.nan, 1]), (1, [1, np.nan, np.nan, np.nan, np.nan])):
        for cycle, value in enumerate(values):
            onset_rows.append({
                "session": "708", "transition": "GS", "criterion": "PRIMARY_Z2",
                "patch_id": patch_id, "patch_row": 0, "patch_col": patch_id,
                "onset_status": "DETECTED" if np.isfinite(value) else "NO_DETECTED_ONSET",
                "onset_latency_frame": value,
            })
            peak_rows.append({
                "session": "708", "transition": "GS", "patch_id": patch_id,
                "peak_latency_frame": cycle % 2,
            })
    summary = summarize_patches(pd.DataFrame(onset_rows), pd.DataFrame(peak_rows)).set_index("patch_id")
    assert summary.loc[0, "onset_detection_rate"] == 0.8
    assert bool(summary.loc[0, "latency_stable_patch"])
    assert not bool(summary.loc[1, "latency_stable_patch"])


def _split_input(n_patches: int = 6) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "cycle": cycle, "patch_id": patch, "onset_status": "DETECTED",
            "onset_latency_frame": float((cycle + patch) % 3),
        }
        for cycle in range(4) for patch in range(n_patches)
    ])


def test_19_split_half_is_strictly_cycle_grouped() -> None:
    splits = split_half_latency(_split_input(), [0, 1, 2, 3], n_splits=5, seed=7)
    for row in splits.itertuples():
        a = set(map(int, row.half_A_cycles.split(",")))
        b = set(map(int, row.half_B_cycles.split(",")))
        assert a.isdisjoint(b)
        assert a | b == {0, 1, 2, 3}


def test_20_minimum_common_patch_guard_is_enforced() -> None:
    splits = split_half_latency(_split_input(4), [0, 1, 2, 3], n_splits=3, min_common_patches=5)
    assert (splits["status"] == "INSUFFICIENT_COMMON_PATCHES").all()
    assert splits["latency_map_split_half_rho"].isna().all()


def test_21_four_neighbor_adjacency_only() -> None:
    patches = patch_grid((32, 32), (16, 16))
    assert set(four_neighbor_pairs(patches)) == {(0, 1), (0, 2), (1, 3), (2, 3)}
    summary = pd.DataFrame({
        "patch_id": [0, 1, 2, 3], "latency_stable_patch": [True] * 4,
        "median_onset_frame": [0, 1, 2, 1],
    })
    result = neighbor_summary(summary, patches)
    assert result["n_stable_neighbor_pairs"] == 4
    assert result["paths_constructed"] is False


def test_22_report_does_not_output_physical_speed() -> None:
    source = inspect.getsource(make_report).lower()
    assert "mm/s" not in source
    assert "propagation speed" not in source


def test_23_exact_nine_session_permutation_is_complete() -> None:
    result = exact_spearman_permutation(np.arange(9), np.arange(9))
    assert result["rho"] == pytest.approx(1.0)
    assert result["n_permutations"] == math.factorial(9)
    assert result["permutation_p_two_sided"] == pytest.approx(2 / math.factorial(9))


def test_24_holm_correction_is_monotone_and_exact() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_25_external_unknown_threshold_is_not_used() -> None:
    assert ONSET_CRITERIA == {"PRIMARY_Z2": 2.0, "SENSITIVITY_Z1_5": 1.5}
    assert 0.4 not in ONSET_CRITERIA.values()


def test_26_no_HRF_search_or_fit() -> None:
    source = inspect.getsource(v11)
    assert "from ultrasound_decoding.hrf" not in source
    assert "curve_fit" not in source


def test_27_no_spatial_registration() -> None:
    source = inspect.getsource(v11)
    assert "from ultrasound_decoding.spatial_registration" not in source
    assert "registration_applied" not in source


def test_28_no_ROI_selection() -> None:
    source = inspect.getsource(v11)
    assert "roi_annotation" not in source
    assert "candidate_roi" not in source


def test_29_no_decoder_is_imported_or_trained() -> None:
    source = inspect.getsource(v11)
    assert "fit_predict" not in source
    assert "from ultrasound_decoding.linear" not in source
    assert "from ultrasound_decoding.deep" not in source


def test_30_output_manifest_is_complete(tmp_path: Path) -> None:
    paths = expected_outputs(tmp_path)
    assert len(paths) == len(REQUIRED_OUTPUTS) == len(set(paths))
    required = {
        "temporal_metadata_audit.csv", "onset_metrics.csv", "peak_latency_metrics.csv",
        "patch_latency_summary.csv", "split_half_latency_metrics.csv",
        "session_latency_feasibility.csv", "latency_vs_withinBA_associations.csv",
        "v9_vs_v11_latency_association.csv", "feasibility_decision.csv",
        "temporal_sampling_audit.png", "GS_latency_maps_9sessions.png",
        "DS_latency_maps_9sessions.png", "latency_diagnostic_overview.png",
        "temporal_latency_feasibility_report.md", "run_command_server.txt",
    }
    assert required.issubset({path.name for path in paths})
    freeze = config_freeze_text(Path("config.json"))
    assert "clean4 is forbidden" in freeze
    assert "no interpolation" in freeze
