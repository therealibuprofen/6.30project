from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    BlockSequenceData,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.multisource_loso_reporting_v5 import (
    exact_sign_flip_pvalue,
    holm_adjust,
    make_required_figures,
    planned_statistical_tests,
    target_level_comparison,
)
from ultrasound_decoding.multisource_loso_v5 import (
    FROZEN_SUPERVISED_CONFIG,
    REQUIRED_FORMAL_OUTPUTS,
    V5_CONDITIONS,
    V5_SEEDS,
    assert_formal_cuda,
    epoch_draw_indices,
    missing_formal_outputs,
    prepare_cross_session_data,
    sampling_distribution_rows,
    source_sessions_for_target,
    train_prepared_cross_session,
    validate_source_target_data,
)


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"


def _synthetic_data(session: str, task: str = "binary", offset: float = 0.0) -> BlockSequenceData:
    n = 4
    X = np.empty((n, 4, 128, 501), dtype=np.float32)
    for index in range(n):
        X[index].fill(offset + index * 0.01)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    metadata = pd.DataFrame({
        "block_id": [f"session{session}_cycle{groups[i]:03d}_block{i}" for i in range(n)],
        "cycle": groups,
    })
    return BlockSequenceData(
        session=session,
        task=task,
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        clean4_relative_time_s=np.tile(np.arange(4, dtype=np.float32), (n, 1)),
        clean4_original_frame_indices=np.tile(np.arange(4, dtype=np.int64), (n, 1)),
        source_h5_path=Path(f"session_{session}_blocks.h5"),
        source_metadata_path=Path(f"session_{session}_block_metadata.csv"),
    )


def _complete_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    within = []
    for task_i, task in enumerate(EXPECTED_TASKS):
        for target_i, target in enumerate(EXPECTED_SESSIONS):
            sources = source_sessions_for_target(target)
            for source_i, source in enumerate(sources):
                for seed in V5_SEEDS:
                    ba = 0.46 + 0.002 * source_i + 0.001 * task_i
                    rows.append({
                        "task": task, "target_session": target,
                        "source_sessions": source, "n_source_sessions": 1,
                        "seed": seed, "condition": "SINGLE_SOURCE_TRANSFER",
                        "test_balanced_accuracy": ba, "train_balanced_accuracy": 0.8,
                        "train_test_gap_BA": 0.8 - ba,
                    })
            for condition, extra in (
                ("MULTI_SOURCE_BALANCED", 0.04),
                ("NATURAL_FREQUENCY_MULTI_SOURCE", 0.02),
            ):
                for seed in V5_SEEDS:
                    ba = 0.47 + extra + 0.001 * task_i
                    rows.append({
                        "task": task, "target_session": target,
                        "source_sessions": ",".join(sources), "n_source_sessions": 8,
                        "seed": seed, "condition": condition,
                        "test_balanced_accuracy": ba, "train_balanced_accuracy": 0.8,
                        "train_test_gap_BA": 0.8 - ba,
                    })
            within.append({
                "task": task, "target_session": target,
                "within_session_reference_BA": 0.7 + 0.001 * target_i,
            })
    return pd.DataFrame(rows), pd.DataFrame(within)


def test_01_all_nine_sessions_are_frozen() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_each_target_has_exactly_eight_sources() -> None:
    for target in EXPECTED_SESSIONS:
        sources = source_sessions_for_target(target)
        assert len(sources) == 8
        assert target not in sources
        assert set(sources) | {target} == set(EXPECTED_SESSIONS)


def test_03_exactly_nine_distinct_loso_targets() -> None:
    pools = {target: source_sessions_for_target(target) for target in EXPECTED_SESSIONS}
    assert len(pools) == 9
    assert all(len(set(sources)) == 8 for sources in pools.values())


def test_04_target_in_source_pool_is_hard_failure() -> None:
    data = {session: _synthetic_data(session) for session in ("626", "628")}
    with pytest.raises(AssertionError, match="target session"):
        validate_source_target_data(data, ("626", "628"), "626")


def test_05_clean4_and_both_old_task_definitions_are_reused() -> None:
    for task, blocks_per_cycle in (("binary", 4), ("stimulus_type", 2)):
        data = load_block_sequence_session(PROJECT_DIR, "807", task, data_dir=DATA_DIR)
        assert data.X.shape[1:] == (4, 128, 501)
        assert set(data.metadata.groupby("cycle").size()) == {blocks_per_cycle}
        assert set(data.y) == {0, 1}


def test_06_prepared_pool_has_zero_target_training_samples() -> None:
    data = {session: _synthetic_data(session, offset=index) for index, session in enumerate(("626", "628", "708"))}
    prepared = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    assert set(prepared.train_session_labels) == {"628", "708"}
    assert "626" not in prepared.train_session_labels
    assert len(prepared.source_sessions) == 2
    assert prepared.normalization_audit["target_frames_used_for_fit"] == 0


def test_07_target_data_cannot_change_fitted_source_normalization() -> None:
    data = {session: _synthetic_data(session, offset=index) for index, session in enumerate(("626", "628", "708"))}
    first = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    changed = dict(data)
    changed["626"] = replace(data["626"], X=np.full_like(data["626"].X, 999.0))
    second = prepare_cross_session_data(
        changed, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    assert np.array_equal(first.X_train, second.X_train)
    assert first.normalization_audit["mean_mean"] == second.normalization_audit["mean_mean"]
    assert first.normalization_audit["target_used_for_stats"] is False
    natural = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="natural_frequency"
    )
    assert np.array_equal(first.X_train, natural.X_train)
    assert first.normalization_audit["normalization_weighting"] == "sample_frequency_weighted_source_only"


def test_08_session_balanced_sampler_is_uniform_with_unequal_sources() -> None:
    labels = np.asarray(["626"] * 8 + ["628"] * 24, dtype=object)
    drawn = epoch_draw_indices(labels, seed=V5_SEEDS[0], epoch=1, balance_mode="session_balanced")
    counts = {session: int(np.sum(labels[drawn] == session)) for session in ("626", "628")}
    assert counts == {"626": 16, "628": 16}
    assert len(drawn) == len(labels)


def test_09_natural_frequency_sampler_uses_each_block_once() -> None:
    labels = np.asarray(["626"] * 8 + ["628"] * 24, dtype=object)
    drawn = epoch_draw_indices(labels, seed=V5_SEEDS[0], epoch=1, balance_mode="natural_frequency")
    assert sorted(drawn.tolist()) == list(range(len(labels)))
    assert np.sum(labels[drawn] == "626") == 8
    assert np.sum(labels[drawn] == "628") == 24


def test_10_sampling_audit_reports_source_proportions() -> None:
    labels = np.asarray(["626"] * 8 + ["628"] * 24, dtype=object)
    drawn = epoch_draw_indices(labels, seed=1, epoch=1, balance_mode="session_balanced")
    rows = sampling_distribution_rows(
        drawn, labels, target_session="708", task="binary",
        condition="MULTI_SOURCE_BALANCED", seed=1, epoch=1,
    )
    assert len(rows) == 2
    assert all(row["draw_proportion"] == pytest.approx(0.5) for row in rows)


def test_11_smallcnn_feature_mean_architecture_is_identical() -> None:
    model = CNN2DMeanPool(n_classes=2, dropout=0.25, temporal_length=4)
    assert isinstance(model.encoder, SmallCNNFrameEncoder)
    assert model.encoder.feature_dim == 512
    assert model.temporal_length == 4


def test_12_conditions_seeds_and_supervised_config_are_frozen() -> None:
    assert V5_CONDITIONS == (
        "SINGLE_SOURCE_TRANSFER", "MULTI_SOURCE_BALANCED", "NATURAL_FREQUENCY_MULTI_SOURCE",
    )
    assert V5_SEEDS == (20260812, 20260813, 20260814)
    assert (
        FROZEN_SUPERVISED_CONFIG.optimizer,
        FROZEN_SUPERVISED_CONFIG.lr,
        FROZEN_SUPERVISED_CONFIG.weight_decay,
        FROZEN_SUPERVISED_CONFIG.batch_size,
        FROZEN_SUPERVISED_CONFIG.max_epochs,
        FROZEN_SUPERVISED_CONFIG.dropout,
    ) == ("adamw", 1e-3, 1e-3, 16, 40, 0.25)


def test_13_tiny_forward_backward_and_metric_schema() -> None:
    data = {session: _synthetic_data(session, offset=index) for index, session in enumerate(("626", "628", "708"))}
    prepared = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    result = train_prepared_cross_session(
        prepared,
        condition="MULTI_SOURCE_BALANCED",
        seed=V5_SEEDS[0],
        balance_mode="session_balanced",
        config=replace(FROZEN_SUPERVISED_CONFIG, max_epochs=1),
        device="cpu",
    )
    assert len(result.history) == 1
    assert result.metrics["target_frames_used_for_training"] == 0
    assert result.metrics["target_used_for_validation"] is False
    assert np.isfinite(result.metrics["test_balanced_accuracy"])


def test_14_condition_balance_mode_mismatch_stops() -> None:
    data = {session: _synthetic_data(session, offset=index) for index, session in enumerate(("626", "628", "708"))}
    prepared = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="natural_frequency"
    )
    with pytest.raises(ValueError, match="requires session-balanced"):
        train_prepared_cross_session(
            prepared, condition="MULTI_SOURCE_BALANCED", seed=1,
            balance_mode="natural_frequency", config=replace(FROZEN_SUPERVISED_CONFIG, max_epochs=1), device="cpu",
        )


def test_15_target_level_summary_uses_mean_of_eight_predefined_sources() -> None:
    metrics, within = _complete_metrics()
    table = target_level_comparison(metrics, within)
    assert len(table) == 18
    assert (table["n_single_source_sessions"] == 8).all()
    expected = np.mean([0.46 + 0.002 * index for index in range(8)])
    assert table[table["task"] == "binary"].iloc[0]["single_source_mean_BA"] == pytest.approx(expected)


def test_16_statistics_use_target_session_as_unit() -> None:
    metrics, within = _complete_metrics()
    tests = planned_statistical_tests(target_level_comparison(metrics, within))
    assert len(tests) == 2
    assert (tests["primary_unit"] == "target_session").all()
    assert (tests["n_target_sessions"] == 9).all()


def test_17_exact_sign_flip_has_512_patterns() -> None:
    assert exact_sign_flip_pvalue(np.ones(9)) == pytest.approx(2 / 512)


def test_18_two_task_holm_correction_is_valid() -> None:
    raw = np.asarray([0.01, 0.04])
    corrected = holm_adjust(raw)
    assert np.all(corrected >= raw)
    assert corrected.tolist() == pytest.approx([0.02, 0.04])


def test_19_required_figures_render(tmp_path: Path) -> None:
    metrics, within = _complete_metrics()
    table = target_level_comparison(metrics, within)
    sampling = []
    for condition in V5_CONDITIONS[1:]:
        for source in EXPECTED_SESSIONS:
            sampling.append({"condition": condition, "source_session": source, "draw_proportion": 0.125})
    make_required_figures(tmp_path, table, pd.DataFrame(sampling))
    assert len(list((tmp_path / "figures").glob("*.png"))) == 6


def test_20_formal_run_refuses_cpu_and_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="--device cuda"):
        assert_formal_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        assert_formal_cuda("cuda")


def test_21_output_completeness(tmp_path: Path) -> None:
    assert set(REQUIRED_FORMAL_OUTPUTS).issubset(set(missing_formal_outputs(tmp_path)))
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    curve = tmp_path / "downstream/training_curves/one.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("epoch,loss\n1,1\n", encoding="utf-8")
    assert missing_formal_outputs(tmp_path) == []


def test_22_core_has_no_spatial_alignment_or_target_adaptation_imports() -> None:
    import ultrasound_decoding.multisource_loso_v5 as module
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("spatial_registration", "rigid_registration", "affine", "target ssl", "feature alignment"):
        assert forbidden not in source


def test_23_no_validation_or_early_stopping_path_in_frozen_trainer() -> None:
    source = inspect.getsource(train_prepared_cross_session).lower()
    assert "val_idx" not in source
    assert "patience" not in source
    assert FROZEN_SUPERVISED_CONFIG.max_epochs == 40
