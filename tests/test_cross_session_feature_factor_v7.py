from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cross_session_feature_factor_v7 import (
    CONDITION4_NAMES,
    CONDITION_TIME_WARNING,
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    GLOBAL_ENCODER_SEEDS,
    N_BOOTSTRAP,
    REQUIRED_OUTPUTS,
    STATISTIC_SEED,
    add_v7_labels,
    assert_formal_cuda,
    audit_pairwise_cross_session,
    audit_v5_cross_session_metrics,
    audit_within_session_metrics,
    balanced_cycle_bootstrap_indices,
    bootstrap_factor_decomposition,
    classification_metrics,
    clean4_identity_rows,
    cycle_consistency_from_pool,
    cycle_grouped_session_folds,
    exact_spearman_permutation,
    fit_l2_logistic,
    fit_pca,
    metadata_factor_audit,
    missing_formal_outputs,
    multivariate_energy_distance,
    multivariate_factor_sums,
    load_torch_checkpoint_compat,
    pairwise_session_distances,
    sample_metadata_table,
    source_only_stimulus_probe,
)
from ultrasound_decoding.multiframe.dataset import load_block_sequence_session
from ultrasound_decoding.ssl_multisession_v2 import SessionBalancedSampler, SessionFramePool


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"
V5_DIR = PROJECT_DIR / "outputs/multisource_loso_smallcnn_9sessions_v5"
V1_DIR = PROJECT_DIR / "outputs/ssl_masked_smallcnn_clean4_9sessions_v1"


def _balanced_metadata(sessions=EXPECTED_SESSIONS, n_cycles: int = 2) -> pd.DataFrame:
    rows = []
    names = ("grating", "stop_after_grating", "dot", "static")
    for session in sessions:
        for cycle in range(n_cycles):
            for order, name in enumerate(names):
                rows.append({
                    "block_id": f"session{session}_cycle{cycle:03d}_{name}",
                    "session": str(session), "cycle": cycle, "cycle_key": f"{session}:cycle{cycle}",
                    "block_name": name, "block_order_in_cycle": order,
                    "binary_label_int": int(name in ("grating", "dot")),
                })
    return add_v7_labels(pd.DataFrame(rows))


def _feature_fixture(n_cycles: int = 2, n_features: int = 6):
    metadata = _balanced_metadata(n_cycles=n_cycles)
    rng = np.random.default_rng(22)
    features = rng.normal(size=(len(metadata), n_features))
    features += metadata["session"].astype(int).to_numpy()[:, None] / 100
    return features, metadata


def _small_pool(sessions=("626", "628"), n_cycles: int = 3) -> SessionFramePool:
    rng = np.random.default_rng(3)
    frames_by_session = {}
    cycles_by_session = {}
    indices_by_session = {}
    paths = {}
    for i, session in enumerate(sessions):
        cycles = np.repeat(np.arange(n_cycles), 3)
        frames_by_session[session] = rng.normal(i, 1, size=(len(cycles), 4, 5)).astype(np.float32)
        cycles_by_session[session] = cycles
        indices_by_session[session] = np.arange(len(cycles))
        paths[session] = Path(f"session_{session}.h5")
    return SessionFramePool(frames_by_session, cycles_by_session, indices_by_session, paths)


def test_01_all_nine_sessions_are_fixed_and_present() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")
    assert all((DATA_DIR / f"session_{session}_blocks.h5").is_file() for session in EXPECTED_SESSIONS)


def test_02_clean4_sample_matches_frozen_pipeline() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    assert tuple(data.X.shape[1:]) == EXPECTED_BLOCK_SHAPE
    assert data.clean4_original_frame_indices.shape == (data.n_blocks, 4)
    assert clean4_identity_rows({"708": data})[0]["status"] == "PASS"


def test_03_complete_cycle_ids_and_counts_are_exact() -> None:
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=DATA_DIR)
        assert data.n_blocks == 4 * data.n_cycles
        assert set(data.groups) == set(range(data.n_cycles))


def test_04_stimulus_presence_mapping_is_frozen_binary_mapping() -> None:
    metadata = _balanced_metadata(("626",), 1)
    assert metadata["stimulus_presence"].tolist() == ["stimulus", "no_stimulus", "stimulus", "no_stimulus"]


def test_05_condition4_mapping_is_exact() -> None:
    metadata = _balanced_metadata(("626",), 1)
    assert tuple(metadata["condition4"]) == CONDITION4_NAMES


def test_06_descriptive_and_predictive_pca_paths_are_distinct() -> None:
    import ultrasound_decoding.cross_session_feature_factor_v7 as module
    common = inspect.getsource(module.fit_common_raw_pca)
    predictive = inspect.getsource(module.source_only_stimulus_probe)
    assert "sample_metadata_table" in common
    assert "sources = [session for session in sessions if session != target]" in predictive
    assert "strict_predictive_use_allowed" in inspect.getsource(module.save_pca_model)


def test_07_predictive_pca_fit_ids_are_source_only() -> None:
    data = {
        session: load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=DATA_DIR)
        for session in ("626", "628")
    }
    # Keep this test small while exercising the real clean4 tensors.
    for session in data:
        keep = np.isin(data[session].groups, [0, 1])
        data[session] = data[session].__class__(
            session=data[session].session, task=data[session].task,
            X=data[session].X[keep], y=data[session].y[keep], groups=data[session].groups[keep],
            metadata=data[session].metadata.loc[keep].reset_index(drop=True),
            clean4_relative_time_s=data[session].clean4_relative_time_s[keep],
            clean4_original_frame_indices=data[session].clean4_original_frame_indices[keep],
            source_h5_path=data[session].source_h5_path, source_metadata_path=data[session].source_metadata_path,
        )
    _table, audit = source_only_stimulus_probe(data, max_iter=50)
    assert all(row["target_ids_in_pca_fit"] == 0 for row in audit)


def test_08_target_never_enters_normalization_pca_scaling_or_classifier() -> None:
    source = inspect.getsource(source_only_stimulus_probe)
    assert '"target_used_for_normalization": False' in source
    assert '"target_used_for_PCA": False' in source
    assert '"target_used_for_scaling": False' in source
    assert '"target_used_for_classifier_fit": False' in source


def test_09_global_encoder_input_loader_does_not_read_labels() -> None:
    import ultrasound_decoding.ssl_multisession_v2 as v2
    source = inspect.getsource(v2.load_unlabeled_cycles).lower()
    assert "/labels" not in source and "stimulus" not in source and "block_name" not in source


def test_10_global_sampler_is_session_balanced() -> None:
    pool = _small_pool(EXPECTED_SESSIONS, 2)
    samples = SessionBalancedSampler(pool, seed=4).sample(90_000)
    counts = pd.Series([session for session, _ in samples]).value_counts(normalize=True)
    assert np.max(np.abs(counts.to_numpy() - 1 / 9)) < 0.01


def test_11_three_global_encoder_seeds_are_fixed() -> None:
    assert GLOBAL_ENCODER_SEEDS == (20260812, 20260813, 20260814)


def test_12_energy_distance_is_symmetric() -> None:
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(9, 3))
    assert multivariate_energy_distance(x, y) == pytest.approx(multivariate_energy_distance(y, x))


def test_13_energy_distance_diagonal_is_zero() -> None:
    x = np.random.default_rng(2).normal(size=(8, 4))
    assert multivariate_energy_distance(x, x) == pytest.approx(0.0, abs=1e-12)


def test_14_session_probe_folds_are_cycle_grouped() -> None:
    _features, metadata = _feature_fixture(3)
    folds = cycle_grouped_session_folds(metadata["session"].to_numpy(), metadata["cycle"].to_numpy(), n_folds=3)
    assert len(folds) == 3


def test_15_four_blocks_from_one_cycle_never_cross_train_test() -> None:
    _features, metadata = _feature_fixture(3)
    labels, cycles = metadata["session"].to_numpy(), metadata["cycle"].to_numpy()
    for train, test in cycle_grouped_session_folds(labels, cycles, n_folds=3):
        train_keys = set(zip(labels[train], cycles[train]))
        test_keys = set(zip(labels[test], cycles[test]))
        assert not train_keys & test_keys


def test_16_binary_factor_mapping_matches_frozen_labels() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "709", "binary", data_dir=DATA_DIR)
    metadata = add_v7_labels(data.metadata)
    assert np.array_equal((metadata["stimulus_presence"] == "stimulus").astype(int), data.y)


def test_17_bootstrap_has_equal_cycle_draws_per_session() -> None:
    metadata = _balanced_metadata(n_cycles=3)
    indices = balanced_cycle_bootstrap_indices(metadata, rng=np.random.default_rng(5))
    selected = metadata.iloc[indices]
    assert selected.groupby("session").size().nunique() == 1


def test_18_bootstrap_retains_all_four_conditions() -> None:
    metadata = _balanced_metadata(n_cycles=3)
    indices = balanced_cycle_bootstrap_indices(metadata, rng=np.random.default_rng(6))
    assert len(indices) == len(EXPECTED_SESSIONS) * 3 * 4
    assert set(metadata.iloc[indices]["condition4"]) == set(CONDITION4_NAMES)


def test_19_factor_r2_components_are_finite() -> None:
    features, metadata = _feature_fixture(2)
    sums = multivariate_factor_sums(features, metadata["session"], metadata["stimulus_presence"])
    assert np.isfinite(list(sums.values())).all()


def test_20_factor_components_sum_to_total() -> None:
    features, metadata = _feature_fixture(2)
    sums = multivariate_factor_sums(features, metadata["session"], metadata["condition4"])
    assert sum(sums[key] for key in ("session", "factor", "session_x_factor", "residual")) == pytest.approx(sums["total"])


def test_21_condition4_output_has_temporal_confound_warning() -> None:
    features, metadata = _feature_fixture(2)
    table, _ = bootstrap_factor_decomposition(
        features, metadata, factor_column="condition4", representation="RAW_SPATIAL_PCA", n_bootstrap=3
    )
    assert table["temporal_confound_warning"].str.contains("fixed within-cycle temporal position").all()
    assert "pure visual stimulus identity" in CONDITION_TIME_WARNING


def test_22_exact_permutation_statistic_enumerates_all_labels() -> None:
    result = exact_spearman_permutation([1, 2, 3, 4], [1, 2, 3, 4])
    assert result["n_permutations"] == math.factorial(4)
    assert result["permutation_p_two_sided"] == pytest.approx(2 / 24)


def test_23_no_nine_point_multivariable_regression() -> None:
    source = (PROJECT_DIR / "scripts/run_cross_session_feature_factor_analysis_9sessions_v7.py").read_text().lower()
    assert "multivariable regression" not in source
    assert "linearregression" not in source


def test_24_cycle_consistency_uses_pairwise_cycle_correlations() -> None:
    pool = _small_pool(("626", "628"), 3)
    mean = np.zeros((1, 4, 5), dtype=np.float32)
    std = np.ones((1, 4, 5), dtype=np.float32)
    table = cycle_consistency_from_pool(pool, normalizer_mean=mean, normalizer_std=std)
    assert (table["n_cycle_pairs"] == 3).all()
    assert table["cycle_consistency_mean"].between(-1, 1).all()


def test_25_session_diagnostic_never_names_data_quality_score() -> None:
    core = (PROJECT_DIR / "src/ultrasound_decoding/cross_session_feature_factor_v7.py").read_text().lower()
    reporting = (PROJECT_DIR / "src/ultrasound_decoding/cross_session_feature_factor_reporting_v7.py").read_text().lower()
    assert "data quality score" not in core + reporting


def test_26_v5_artifact_audit_has_nine_strict_targets() -> None:
    audit, values = audit_v5_cross_session_metrics(V5_DIR)
    assert len(audit) == len(values) == 9
    assert values["session"].astype(str).tolist() == list(EXPECTED_SESSIONS)
    assert (audit["status"] == "PASS").all()


def test_27_no_csu_training_or_import() -> None:
    source = (PROJECT_DIR / "src/ultrasound_decoding/cross_session_feature_factor_v7.py").read_text().lower()
    assert "multisource_csu" not in source and "train_csu" not in source


def test_28_no_registration_code() -> None:
    source = (PROJECT_DIR / "src/ultrasound_decoding/cross_session_feature_factor_v7.py").read_text().lower()
    assert "from ultrasound_decoding.spatial_registration" not in source
    assert "from ultrasound_decoding.reference_rigid_registration" not in source


def test_29_no_roi_analysis() -> None:
    source = (PROJECT_DIR / "scripts/run_cross_session_feature_factor_analysis_9sessions_v7.py").read_text().lower()
    assert "roi_decoding" not in source and "candidate_roi" not in source


def test_30_no_glm_or_searchlight_analysis() -> None:
    source = (PROJECT_DIR / "scripts/run_cross_session_feature_factor_analysis_9sessions_v7.py").read_text().lower()
    assert "glm_full_timeseries" not in source and "interpretability.searchlight" not in source


def test_31_formal_gpu_guard_rejects_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="must be exactly 'cuda'"):
        assert_formal_cuda("cpu")
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        assert_formal_cuda("cuda")


def test_32_output_completeness_contract_and_artifact_audits() -> None:
    assert len(REQUIRED_OUTPUTS) >= 35
    assert "figures/session_diagnostic_overview.png" in REQUIRED_OUTPUTS
    assert "report/cross_session_feature_factor_report.md" in REQUIRED_OUTPUTS
    within, values = audit_within_session_metrics(V1_DIR)
    pair_audit, pair_values = audit_pairwise_cross_session(V5_DIR)
    assert len(within) == len(values) == 9 and (within["status"] == "PASS").all()
    assert pair_audit.iloc[0]["status"] == "PASS_RUN_SECONDARY"
    assert pair_values is not None and len(pair_values) == 36
    assert missing_formal_outputs(Path("/definitely/not/a/v7/output"))


def test_33_metadata_audit_marks_absent_fields_not_available() -> None:
    data = {
        session: load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=DATA_DIR)
        for session in EXPECTED_SESSIONS
    }
    metadata = sample_metadata_table(data)
    audit = metadata_factor_audit(metadata, data_root=PROJECT_DIR / "data", sessions=EXPECTED_SESSIONS)
    unavailable = audit[~audit["available"]]
    assert {"monkey / subject", "recording_date", "task", "run", "slot / probe", "pretraining / retraining"} <= set(unavailable["factor"])
    assert (unavailable["levels"] == "NOT_AVAILABLE").all()


def test_34_fixed_probe_is_l2_c1_balanced() -> None:
    rng = np.random.default_rng(STATISTIC_SEED)
    X = rng.normal(size=(40, 4))
    y = np.repeat([0, 1], 20)
    model = fit_l2_logistic(X, y, C=1.0, class_weight="balanced", max_iter=80)
    metrics = classification_metrics(y, model.predict(X), [0, 1])
    assert model.C == 1.0 and model.class_weight == "balanced"
    assert set(metrics) == {"accuracy", "balanced_accuracy", "macro_F1"}


def test_35_pairwise_distance_schema_has_no_diagonal_rows() -> None:
    features, metadata = _feature_fixture(2)
    rows = pairwise_session_distances(features, metadata["session"].to_numpy(), representation="RAW_SPATIAL_PCA")
    assert len(rows) == 36
    assert (rows["session_a"] != rows["session_b"]).all()


def test_36_checkpoint_loader_falls_back_for_old_pytorch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        if "weights_only" in kwargs:
            raise TypeError("'weights_only' is an invalid keyword argument for Unpickler()")
        return {"status": "old_torch_payload_loaded"}

    monkeypatch.setattr("torch.load", fake_load)
    payload = load_torch_checkpoint_compat(Path("checkpoint.pt"))
    assert payload == {"status": "old_torch_payload_loaded"}
    assert calls == [
        (Path("checkpoint.pt"), {"map_location": "cpu", "weights_only": False}),
        (Path("checkpoint.pt"), {"map_location": "cpu"}),
    ]


def test_37_checkpoint_loader_does_not_mask_other_type_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load(_path, **_kwargs):
        raise TypeError("unrelated checkpoint type failure")

    monkeypatch.setattr("torch.load", fake_load)
    with pytest.raises(TypeError, match="unrelated"):
        load_torch_checkpoint_compat(Path("checkpoint.pt"))
