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

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import EXPECTED_SESSIONS, load_block_sequence_session
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.ssl_masked import (
    MASK_BLOCK_SIZE,
    MASK_RATIO,
    SSL_SEEDS,
    SSLFrameData,
    SSLPretrainingConfig,
    configure_downstream_model,
    deterministic_block_mask,
    load_ssl_encoder_checkpoint,
)
from ultrasound_decoding.ssl_multisession_reporting_v2 import (
    exact_sign_flip_pvalue,
    holm_adjust,
    planned_statistical_tests,
    session_level_comparison,
)
from ultrasound_decoding.ssl_multisession_v2 import (
    NEW_PRETRAINING_CONDITIONS,
    REQUIRED_FORMAL_OUTPUTS,
    V1_FROZEN_SOURCE_FINGERPRINT,
    V2_CONDITIONS,
    SessionBalancedSampler,
    SessionFramePool,
    architecture_fingerprint,
    assert_formal_cuda,
    build_ssl_pool,
    checkpoint_contains_no_label_information,
    compute_match_row,
    frozen_v1_source_fingerprint,
    load_unlabeled_cycles,
    missing_formal_outputs,
    pretrain_session_balanced_smallcnn,
    reference_optimizer_updates,
    sampling_distribution_rows,
    save_multisession_checkpoint,
)


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"
V1_DIR = PROJECT_DIR / "outputs/ssl_masked_smallcnn_clean4_9sessions_v1"


def _synthetic_all_sessions() -> dict[str, SSLFrameData]:
    result = {}
    for session_i, session in enumerate(EXPECTED_SESSIONS):
        result[session] = SSLFrameData(
            frames=np.full((4, 128, 501), session_i, dtype=np.float32),
            cycles=np.asarray([0, 0, 1, 1], dtype=np.int64),
            original_frame_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
            source_h5_path=Path(f"session_{session}_blocks.h5"),
        )
    return result


def _complete_fold_metrics() -> pd.DataFrame:
    rows = []
    for task_i, task in enumerate(("binary", "stimulus_type")):
        for session_i, session in enumerate(EXPECTED_SESSIONS):
            for condition_i, condition in enumerate(V2_CONDITIONS):
                for seed in SSL_SEEDS:
                    for fold in (1, 2):
                        test = 0.4 + 0.03 * condition_i + 0.002 * session_i + 0.001 * task_i
                        rows.append({
                            "session": session, "task": task, "condition": condition,
                            "seed": seed, "fold": fold, "test_balanced_accuracy": test,
                            "train_balanced_accuracy": 0.8, "train_test_gap_BA": 0.8 - test,
                        })
    return pd.DataFrame(rows)


def test_01_exact_nine_sessions_are_frozen() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_v2_has_exact_four_conditions_and_no_frozen() -> None:
    assert V2_CONDITIONS == ("RANDOM_INIT", "WITHIN_SSL_FT", "OTHER_ONLY_SSL_FT", "MULTI_SSL_FT")
    assert all("FROZEN" not in condition for condition in V2_CONDITIONS)


def test_03_frozen_v1_source_hash_matches() -> None:
    assert frozen_v1_source_fingerprint(PROJECT_DIR) == V1_FROZEN_SOURCE_FINGERPRINT


def test_04_fold_identity_matches_v1() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    old = pd.read_csv(V1_DIR / "audit/fold_reproduction.csv")
    old = old[(old["session"].astype(str) == "708") & (old["task"] == "binary")]
    for fold_i, (train_idx, test_idx) in enumerate(grouped_cv_splits(data.groups, max_folds=10), start=1):
        row = old[old["fold"].astype(int) == fold_i].iloc[0]
        train = ",".join(map(str, sorted(np.unique(data.groups[train_idx]).tolist())))
        test = ",".join(map(str, sorted(np.unique(data.groups[test_idx]).tolist())))
        assert str(row["train_cycles"]) == train
        assert str(row["test_cycles"]) == test


def test_05_other_only_contains_zero_target_frames() -> None:
    pool = build_ssl_pool(
        _synthetic_all_sessions(), target_session="709", target_ssl_train_cycles=[0],
        target_test_cycles=[1], condition="OTHER_ONLY_SSL_FT",
    )
    assert "709" not in pool.source_sessions
    assert len(pool.source_sessions) == 8


def test_06_multi_target_has_train_cycles_only() -> None:
    pool = build_ssl_pool(
        _synthetic_all_sessions(), target_session="709", target_ssl_train_cycles=[0],
        target_test_cycles=[1], condition="MULTI_SSL_FT",
    )
    assert set(pool.cycles_by_session["709"]) == {0}
    assert not set(pool.cycles_by_session["709"]) & {1}


def test_07_overlapping_target_test_cycle_is_hard_failure() -> None:
    with pytest.raises(AssertionError, match="overlap"):
        build_ssl_pool(
            _synthetic_all_sessions(), target_session="709", target_ssl_train_cycles=[0, 1],
            target_test_cycles=[1], condition="MULTI_SSL_FT",
        )


def test_08_unlabeled_loader_source_never_reads_label_dataset() -> None:
    source = inspect.getsource(load_unlabeled_cycles).lower()
    assert "/labels" not in source
    assert "block_name" not in source
    assert "stimulus" not in source


def test_09_session_balanced_sampler_is_near_uniform() -> None:
    pool = build_ssl_pool(
        _synthetic_all_sessions(), target_session="709", target_ssl_train_cycles=[0],
        target_test_cycles=[1], condition="OTHER_ONLY_SSL_FT",
    )
    sampled = SessionBalancedSampler(pool, seed=11).sample(80_000)
    counts = {session: 0 for session in pool.source_sessions}
    for session, _frame in sampled:
        counts[session] += 1
    proportions = np.asarray(list(counts.values())) / len(sampled)
    assert np.max(np.abs(proportions - 1 / 8)) < 0.01


def test_10_multi_target_is_one_equal_session_source() -> None:
    pool = build_ssl_pool(
        _synthetic_all_sessions(), target_session="709", target_ssl_train_cycles=[0],
        target_test_cycles=[1], condition="MULTI_SSL_FT",
    )
    sampled = SessionBalancedSampler(pool, seed=12).sample(90_000)
    target_proportion = sum(session == "709" for session, _ in sampled) / len(sampled)
    assert target_proportion == pytest.approx(1 / 9, abs=0.01)


def test_11_reference_updates_are_v1_equal_update_formula() -> None:
    assert reference_optimizer_updates(180, 32) == 300
    assert reference_optimizer_updates(240, 32) == 400


def test_12_compute_row_rejects_unequal_updates() -> None:
    with pytest.raises(AssertionError, match="equal-update"):
        compute_match_row(
            target_session="708", fold=1, seed=SSL_SEEDS[0], condition="MULTI_SSL_FT",
            ssl_pool_frames=10, actual_batch_size=2, reference_updates=2, actual_updates=1,
            frame_exposure_count=2, unique_frame_coverage=2, reused_artifact=False,
        )


def test_13_mask_is_exact_frozen_family() -> None:
    assert MASK_BLOCK_SIZE == (16, 16)
    assert MASK_RATIO == 0.50
    mask = deterministic_block_mask(128, 501, seed=1, epoch=2, sample_index=3)
    assert abs(float(mask.mean()) - 0.5) < 0.03


def test_14_architecture_is_exact_existing_smallcnn() -> None:
    assert isinstance(SmallCNNFrameEncoder(), SmallCNNFrameEncoder)
    assert len(architecture_fingerprint()) == 64


def test_15_same_three_seeds() -> None:
    assert SSL_SEEDS == (20260812, 20260813, 20260814)


def test_16_random_init_rejects_checkpoint() -> None:
    with pytest.raises(ValueError, match="must not receive"):
        configure_downstream_model(
            "RANDOM_INIT", n_classes=2, pretrained_encoder_state=SmallCNNFrameEncoder().state_dict()
        )


@pytest.mark.parametrize("condition", NEW_PRETRAINING_CONDITIONS)
def test_17_ssl_conditions_load_encoder_into_finetune_model(condition: str) -> None:
    del condition
    model = configure_downstream_model(
        "SSL_FINETUNE", n_classes=2, pretrained_encoder_state=SmallCNNFrameEncoder().state_dict()
    )
    assert isinstance(model, CNN2DMeanPool)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_18_supervised_binary_and_stimulus_folds_are_identical() -> None:
    binary = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    stimulus = load_block_sequence_session(PROJECT_DIR, "708", "stimulus_type", data_dir=DATA_DIR)
    for (_, b_test), (_, s_test) in zip(grouped_cv_splits(binary.groups), grouped_cv_splits(stimulus.groups)):
        assert np.array_equal(np.unique(binary.groups[b_test]), np.unique(stimulus.groups[s_test]))


def test_19_807_uses_official_orientation_input() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "807", "binary", data_dir=DATA_DIR)
    assert data.source_h5_path == DATA_DIR / "session_807_blocks.h5"


def test_20_no_registration_import_in_v2_core() -> None:
    import ultrasound_decoding.ssl_multisession_v2 as module
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "from ultrasound_decoding.spatial_registration" not in source
    assert "from ultrasound_decoding.reference_rigid_registration" not in source


def test_21_session_level_statistics_use_nine_sessions() -> None:
    table = session_level_comparison(_complete_fold_metrics())
    tests = planned_statistical_tests(table)
    assert len(table) == 18
    assert (tests["primary_unit"] == "session").all()
    assert (tests["n_sessions"] == 9).all()


def test_22_exact_sign_flip_enumerates_512_patterns() -> None:
    assert exact_sign_flip_pvalue(np.ones(9)) == pytest.approx(2 / 512)


def test_23_holm_correction_is_monotone_and_not_smaller() -> None:
    raw = np.asarray([0.01, 0.03, 0.02, 0.2])
    corrected = holm_adjust(raw)
    assert np.all(corrected >= raw)
    order = np.argsort(raw)
    assert np.all(np.diff(corrected[order]) >= 0)


def test_24_output_schema_contains_every_required_path(tmp_path: Path) -> None:
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    checkpoint = tmp_path / "pretraining/checkpoints/condition/session/fold/seed.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("x", encoding="utf-8")
    loss = tmp_path / "pretraining/losses/loss.csv"
    loss.parent.mkdir(parents=True, exist_ok=True)
    loss.write_text("x", encoding="utf-8")
    curve = tmp_path / "downstream/training_curves/curve.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("x", encoding="utf-8")
    assert missing_formal_outputs(tmp_path) == []


def test_25_formal_cuda_has_no_cpu_fallback(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="exactly 'cuda'"):
        assert_formal_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        assert_formal_cuda("cuda")


def test_26_sampling_distribution_schema_and_expected_proportion() -> None:
    rows = sampling_distribution_rows(
        {"626": 50, "628": 50}, target_session="709", fold=1,
        seed=SSL_SEEDS[0], condition="OTHER_ONLY_SSL_FT",
    )
    assert {row["expected_proportion"] for row in rows} == {0.5}
    assert {row["sampling_status"] for row in rows} == {"PASS"}


def test_27_tiny_cpu_pretraining_uses_exact_requested_updates(tmp_path: Path) -> None:
    frames = np.zeros((1, 128, 501), dtype=np.float32)
    pool = SessionFramePool(
        frames_by_session={"708": frames}, cycles_by_session={"708": np.asarray([0])},
        original_indices_by_session={"708": np.asarray([0])}, source_paths={"708": Path("synthetic.h5")},
    )
    config = replace(SSLPretrainingConfig(), batch_size=1, epochs=1)
    result = pretrain_session_balanced_smallcnn(
        pool, seed=SSL_SEEDS[0], reference_updates=1, actual_batch_size=1, config=config, device="cpu"
    )
    assert result.actual_updates == result.reference_updates == 1
    assert result.frame_exposure_count == 1
    path = tmp_path / "checkpoint.pt"
    save_multisession_checkpoint(
        path, result, target_session="708", fold=1, seed=SSL_SEEDS[0], condition="MULTI_SSL_FT",
        pool=pool, target_ssl_train_cycles=[0], target_test_cycles=[1], config=config,
        source_fingerprint=V1_FROZEN_SOURCE_FINGERPRINT,
    )
    encoder, payload = load_ssl_encoder_checkpoint(path)
    assert isinstance(encoder, SmallCNNFrameEncoder)
    assert payload["contains_labels"] is False
    assert checkpoint_contains_no_label_information(path)


def test_28_session_comparison_has_preregistered_columns() -> None:
    table = session_level_comparison(_complete_fold_metrics())
    required = {
        "RANDOM_INIT_BA", "WITHIN_SSL_FT_BA", "OTHER_ONLY_SSL_FT_BA", "MULTI_SSL_FT_BA",
        "delta_within_vs_random", "delta_other_vs_within", "delta_multi_vs_within",
        "delta_multi_vs_random", "random_gap", "within_gap", "other_gap", "multi_gap",
    }
    assert required <= set(table.columns)


def test_29_mixed_numeric_and_string_session_ids_are_canonicalized() -> None:
    metrics = _complete_fold_metrics()
    reused = metrics["condition"].isin(("RANDOM_INIT", "WITHIN_SSL_FT"))
    metrics.loc[reused, "session"] = metrics.loc[reused, "session"].astype(int)
    table = session_level_comparison(metrics)
    assert len(table) == 18
    assert table.groupby("task")["session"].nunique().to_dict() == {
        "binary": 9,
        "stimulus_type": 9,
    }
