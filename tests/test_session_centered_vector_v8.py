from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cross_session_feature_factor_v7 import EXPECTED_SESSIONS, GLOBAL_ENCODER_SEEDS, exact_spearman_permutation
from ultrasound_decoding.session_centered_vector_v8 import (
    FORMAL_MODE,
    GD_WARNING,
    REQUIRED_OUTPUTS,
    TRANSDUCTIVE_WARNING,
    _cycle_block_indices,
    centered_feature_frame,
    centroids_long_table,
    contrast_vector,
    cosine_matrix,
    descriptive_pattern_scores,
    load_and_audit_v7_features,
    loso_probe,
    missing_outputs,
    pairwise_transfer_audit,
    pairwise_vector_cosines,
    safe_cosine,
    session_centroids,
    session_contrast_vectors,
    vector_stability,
)


V7_DIR = PROJECT_DIR / "outputs/cross_session_feature_factor_analysis_9sessions_v7"
V5_DIR = PROJECT_DIR / "outputs/multisource_loso_smallcnn_9sessions_v5"
DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"


def _metadata(sessions=EXPECTED_SESSIONS, n_cycles: int = 4) -> pd.DataFrame:
    rows = []
    names = ("grating", "stop_after_grating", "dot", "static")
    for session in sessions:
        for cycle in range(n_cycles):
            for order, block in enumerate(names):
                rows.append({
                    "block_id": f"session{session}_cycle{cycle:03d}_{block}",
                    "session": str(session), "cycle": cycle, "cycle_key": f"{session}:cycle{cycle}",
                    "block_name": block, "condition4": ("stop" if block == "stop_after_grating" else block),
                    "stimulus_presence": "stimulus" if block in ("grating", "dot") else "no_stimulus",
                    "block_order_in_cycle": order,
                })
    return pd.DataFrame(rows)


def _features(metadata: pd.DataFrame, n_features: int = 8) -> np.ndarray:
    rng = np.random.default_rng(20260817)
    values = rng.normal(size=(len(metadata), n_features))
    values += metadata["session"].astype(int).to_numpy()[:, None] / 20
    values += (metadata["stimulus_presence"] == "stimulus").astype(float).to_numpy()[:, None] * np.linspace(0.1, 0.8, n_features)
    return values


def test_01_all_nine_sessions_are_fixed() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_v7_features_are_reused_exactly() -> None:
    root, artifacts, audit = load_and_audit_v7_features(V7_DIR, processed_data_dir=DATA_DIR)
    assert root.name == "cross_session_feature_factor_analysis_9sessions_v7"
    assert len(artifacts) == 4
    assert set(key[1] for key in artifacts if key[1] is not None) == set(GLOBAL_ENCODER_SEEDS)
    assert (audit["status"] == "PASS").all()
    assert audit["reused_without_refit_or_extraction"].astype(bool).all()


def test_03_session_centroid_is_correct() -> None:
    X = np.asarray([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [14.0, 24.0]])
    _centered, centroids = session_centroids(X, ["626", "626", "628", "628"])
    assert np.allclose(centroids["626"], [2, 3])
    assert np.allclose(centroids["628"], [12, 22])


def test_04_centered_session_means_are_zero() -> None:
    metadata = _metadata(("626", "628"), 3)
    centered, _ = session_centroids(_features(metadata), metadata["session"])
    for session in ("626", "628"):
        assert np.allclose(centered[metadata["session"].to_numpy() == session].mean(axis=0), 0, atol=1e-12)


def test_05_stimulus_vector_is_invariant_to_session_centering() -> None:
    metadata = _metadata(("626",), 4)
    X = _features(metadata)
    centered, _ = session_centroids(X, metadata["session"])
    before = contrast_vector(X, metadata["stimulus_presence"])
    after = contrast_vector(centered, metadata["stimulus_presence"])
    assert np.allclose(before, after, atol=1e-12)


def test_06_session_probe_calls_exact_v7_protocol() -> None:
    import ultrasound_decoding.session_centered_vector_v8 as module
    source = inspect.getsource(module.session_id_before_after)
    assert "session_id_probe" in source
    assert "n_permutations=n_permutations" in source
    assert "n_folds=n_folds" in source


def test_07_centering_accepts_no_labels() -> None:
    assert tuple(inspect.signature(session_centroids).parameters) == ("features", "sessions")


def test_08_transductive_centering_is_explicitly_marked() -> None:
    assert "MECHANISTIC TRANSDUCTIVE CONTROL" in TRANSDUCTIVE_WARNING
    assert "not strict unseen-session" in TRANSDUCTIVE_WARNING


def test_09_target_centroid_does_not_read_target_label() -> None:
    source = inspect.getsource(loso_probe)
    centering_line = next(line for line in source.splitlines() if "session_centroids" in line)
    assert "labels" not in centering_line


def test_10_transductive_result_does_not_overwrite_strict_loso() -> None:
    source = (PROJECT_DIR / "scripts/run_session_centered_stimulus_vector_alignment_9sessions_v8.py").read_text()
    assert "V7_STRICT_SOURCE_ONLY_PCA_LOSO" in source
    assert 'columns={"BA": "uncentered_BA"}' in source
    assert 'rename(columns={"BA": "centered_BA"})' in source


def test_11_primary_binary_mapping_is_exact() -> None:
    metadata = _metadata(("626",), 1)
    assert metadata["stimulus_presence"].tolist() == ["stimulus", "no_stimulus", "stimulus", "no_stimulus"]


def test_12_bootstrap_is_cycle_level() -> None:
    metadata = _metadata(("626",), 4)
    result = vector_stability(_features(metadata), metadata, session="626", n_bootstrap=5, n_split_half=5)
    assert result["resampling_unit"] == "complete_cycle_all_four_blocks"


def test_13_cycle_resampling_retains_four_blocks() -> None:
    metadata = _metadata(("626",), 4)
    indices = _cycle_block_indices(metadata, "626")
    assert all(len(value) == 4 for value in indices.values())


def test_14_split_half_is_cycle_level() -> None:
    metadata = _metadata(("626",), 4)
    result = vector_stability(_features(metadata), metadata, session="626", n_bootstrap=5, n_split_half=5)
    assert result["split_unit"] == "complete_cycle"


def test_15_cosine_is_finite_and_bounded() -> None:
    value = safe_cosine(np.asarray([1.0, 2.0]), np.asarray([-3.0, 4.0]))
    assert np.isfinite(value) and -1 <= value <= 1


def test_16_zero_norm_cosine_is_safe() -> None:
    assert safe_cosine(np.zeros(3), np.ones(3)) == 0.0
    assert safe_cosine(np.zeros(3), np.zeros(3)) == 0.0


def test_17_pairwise_cosine_matrix_is_symmetric() -> None:
    vectors = {session: np.asarray([i + 1.0, 1.0]) for i, session in enumerate(EXPECTED_SESSIONS)}
    pairs = pairwise_vector_cosines(vectors, representation="RAW_SPATIAL_PCA")
    matrix = cosine_matrix(pairs)
    assert np.allclose(matrix, matrix.T)


def test_18_pairwise_cosine_diagonal_is_one() -> None:
    vectors = {session: np.asarray([i + 1.0, 1.0]) for i, session in enumerate(EXPECTED_SESSIONS)}
    assert np.allclose(np.diag(cosine_matrix(pairwise_vector_cosines(vectors, representation="RAW"))), 1)


def test_19_exactly_36_unordered_pairs() -> None:
    vectors = {session: np.asarray([i + 1.0, 1.0]) for i, session in enumerate(EXPECTED_SESSIONS)}
    pairs = pairwise_vector_cosines(vectors, representation="RAW")
    assert len(pairs) == 36
    assert (pairs["session_a"] != pairs["session_b"]).all()


def test_20_strong_weak_grouping_is_descriptive_only() -> None:
    table = pd.DataFrame({
        "session": EXPECTED_SESSIONS, "normalized_vector_norm": np.arange(9),
        "v7_separability_ratio": np.arange(9), "bootstrap_vector_stability": np.arange(9),
        "split_half_vector_stability": np.arange(9), "mean_vector_cosine_to_other_sessions": np.arange(9),
    })
    output = descriptive_pattern_scores(table)
    assert output["pattern_assignment_scope"].eq("DESCRIPTIVE_RANK_BASED_NO_SUBGROUP_TEST").all()


def test_21_historical_pairwise_ba_artifact_passes_audit() -> None:
    audit, pairs = pairwise_transfer_audit(V5_DIR)
    assert audit.iloc[0]["status"] == "PASS_RUN_SECONDARY"
    assert len(pairs) == 36
    assert not audit.iloc[0]["retrained_for_v8"]


def test_22_mantel_style_session_label_permutation_is_used() -> None:
    import ultrasound_decoding.session_centered_vector_v8 as module
    source = inspect.getsource(module.vector_alignment_transfer_association)
    assert "mantel_session_label_permutation" in source


def test_23_exact_nine_session_permutation_implementation_is_reused() -> None:
    result = exact_spearman_permutation(range(5), range(5))
    assert result["permutation_method"] == "exact_complete_enumeration"
    assert result["n_permutations"] == 120


def test_24_gd_is_marked_exploratory_temporal_confound() -> None:
    assert GD_WARNING.startswith("EXPLORATORY:")
    assert "fixed within-cycle temporal position" in GD_WARNING


def test_25_no_new_model_or_encoder_training() -> None:
    core = (PROJECT_DIR / "src/ultrasound_decoding/session_centered_vector_v8.py").read_text().lower()
    script = (PROJECT_DIR / "scripts/run_session_centered_stimulus_vector_alignment_9sessions_v8.py").read_text().lower()
    assert "pretrain_session_balanced_smallcnn" not in core + script
    assert "torch.optim" not in core + script
    assert "load_or_train_global_encoder" not in core + script


def test_26_no_csu() -> None:
    source = (PROJECT_DIR / "src/ultrasound_decoding/session_centered_vector_v8.py").read_text().lower()
    assert "multisource_csu" not in source


def test_27_no_registration() -> None:
    source = (PROJECT_DIR / "src/ultrasound_decoding/session_centered_vector_v8.py").read_text().lower()
    assert "spatial_registration" not in source and "rigid_registration" not in source


def test_28_no_roi_execution() -> None:
    source = (PROJECT_DIR / "scripts/run_session_centered_stimulus_vector_alignment_9sessions_v8.py").read_text().lower()
    assert "roi_decoding" not in source and "candidate_roi" not in source


def test_29_no_glm_or_searchlight_execution() -> None:
    source = (PROJECT_DIR / "scripts/run_session_centered_stimulus_vector_alignment_9sessions_v8.py").read_text().lower()
    assert "glm_full_timeseries" not in source and "interpretability.searchlight" not in source


def test_30_output_completeness_contract() -> None:
    assert FORMAL_MODE == "CPU_STATS_ONLY"
    assert len(REQUIRED_OUTPUTS) >= 30
    assert "figures/session_vector_diagnostic_overview.png" in REQUIRED_OUTPUTS
    assert missing_outputs(Path("/not/a/v8/output"))


def test_31_loso_centered_probe_is_label_free_for_centroid() -> None:
    metadata = _metadata(("626", "628", "708"), 3)
    table = loso_probe(_features(metadata), metadata, center_by_session=True)
    assert len(table) == 3
    assert not table["target_centroid_uses_labels"].astype(bool).any()
    assert table["analysis_type"].eq("TRANSDUCTIVE_UNSUPERVISED_CENTERING").all()


def test_32_stability_outputs_are_finite() -> None:
    metadata = _metadata(("626",), 4)
    result = vector_stability(_features(metadata), metadata, session="626", n_bootstrap=10, n_split_half=10)
    values = [result["bootstrap_vector_stability"], result["split_half_vector_stability"]]
    assert np.isfinite(values).all()
