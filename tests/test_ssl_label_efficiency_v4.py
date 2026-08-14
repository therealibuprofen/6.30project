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
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.ssl_label_efficiency_reporting_v4 import (
    exact_sign_flip_pvalue,
    holm_adjust,
    label_efficiency_aulc,
    low_label_summary,
    planned_statistical_tests,
    session_label_efficiency,
    trapezoidal_aulc,
)
from ultrasound_decoding.ssl_label_efficiency_v4 import (
    FROZEN_SSL_CONFIG,
    FROZEN_SUPERVISED_CONFIG,
    LABEL_FRACTIONS,
    REQUIRED_FORMAL_OUTPUTS,
    V4_CONDITIONS,
    V4_SEEDS,
    assert_formal_cuda,
    audit_v1_checkpoint,
    condition_encoder_state,
    file_sha256,
    label_class_balance_row,
    label_fraction_rows,
    labeled_sample_indices,
    missing_formal_outputs,
    n_label_cycles,
    nested_label_subsets,
    ordered_cycle_text,
    round_half_up,
)
from ultrasound_decoding.ssl_masked import MASK_BLOCK_SIZE, MASK_RATIO, SSL_SEEDS


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"
V1_DIR = PROJECT_DIR / "outputs/ssl_masked_smallcnn_clean4_9sessions_v1"


def _complete_metrics() -> pd.DataFrame:
    rows = []
    for task_i, task in enumerate(EXPECTED_TASKS):
        for session_i, session in enumerate(EXPECTED_SESSIONS):
            for fraction in LABEL_FRACTIONS:
                for condition in V4_CONDITIONS:
                    advantage = 0.03 if condition == "WITHIN_MASKED_SSL_FT" else 0.0
                    test = 0.48 + 0.2 * fraction + advantage + 0.001 * task_i + 0.0001 * session_i
                    rows.append({
                        "session": session, "task": task, "fold": 1,
                        "seed": V4_SEEDS[0], "condition": condition,
                        "label_fraction": fraction,
                        "test_balanced_accuracy": test,
                        "train_balanced_accuracy": 0.80,
                        "train_test_gap_BA": 0.80 - test,
                    })
    return pd.DataFrame(rows)


def test_01_all_nine_sessions_are_frozen() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_two_tasks_use_existing_clean4_builder() -> None:
    for task, blocks_per_cycle in (("binary", 4), ("stimulus_type", 2)):
        data = load_block_sequence_session(PROJECT_DIR, "708", task, data_dir=DATA_DIR)
        assert data.X.shape[1:] == (4, 128, 501)
        assert set(data.metadata.groupby("cycle").size()) == {blocks_per_cycle}


def test_03_all_outer_folds_match_v1() -> None:
    old = pd.read_csv(V1_DIR / "audit/fold_reproduction.csv")
    for session in EXPECTED_SESSIONS:
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=DATA_DIR)
            subset = old[(old["session"].astype(str) == session) & (old["task"] == task)]
            for fold_i, (train_idx, test_idx) in enumerate(grouped_cv_splits(data.groups), start=1):
                row = subset[subset["fold"].astype(int) == fold_i].iloc[0]
                train = ",".join(map(str, sorted(np.unique(data.groups[train_idx]).tolist())))
                test = ",".join(map(str, sorted(np.unique(data.groups[test_idx]).tolist())))
                assert train == str(row["train_cycles"])
                assert test == str(row["test_cycles"])


def test_04_exact_fractions_conditions_and_seeds() -> None:
    assert LABEL_FRACTIONS == (0.2, 0.4, 0.6, 0.8, 1.0)
    assert V4_CONDITIONS == ("RANDOM_INIT", "WITHIN_MASKED_SSL_FT")
    assert V4_SEEDS == SSL_SEEDS == (20260812, 20260813, 20260814)


def test_05_round_half_up_is_not_bankers_rounding() -> None:
    assert round_half_up(2.5) == 3
    assert round_half_up(3.5) == 4
    assert n_label_cycles(0.2, 5) == 1
    assert n_label_cycles(1.0, 5) == 5


def test_06_preregistered_cycle_counts_are_exact() -> None:
    assert [n_label_cycles(fraction, 5) for fraction in LABEL_FRACTIONS] == [1, 2, 3, 4, 5]
    assert [n_label_cycles(fraction, 7) for fraction in LABEL_FRACTIONS] == [1, 3, 4, 6, 7]


def test_07_nested_subsets_are_deterministic_prefixes() -> None:
    permutation, subsets = nested_label_subsets(range(7), session="626", fold=1, seed=V4_SEEDS[0])
    permutation_two, subsets_two = nested_label_subsets(range(7), session="626", fold=1, seed=V4_SEEDS[0])
    assert np.array_equal(permutation, permutation_two)
    for fraction in LABEL_FRACTIONS:
        assert np.array_equal(subsets[fraction], subsets_two[fraction])
    for smaller, larger in zip(LABEL_FRACTIONS[:-1], LABEL_FRACTIONS[1:]):
        assert set(subsets[smaller]).issubset(set(subsets[larger]))


def test_08_label_subsets_only_contain_outer_train_cycles() -> None:
    _permutation, subsets = nested_label_subsets([1, 2, 3, 4], session="626", fold=2, seed=V4_SEEDS[1])
    assert all(set(values).issubset({1, 2, 3, 4}) for values in subsets.values())
    assert all(0 not in values for values in subsets.values())


def test_09_fraction_rows_keep_duplicate_cycle_counts() -> None:
    counts, subsets = label_fraction_rows([1, 2], session="626", fold=1, seed=V4_SEEDS[0])
    assert len(counts) == len(subsets) == 5
    assert any(row["duplicate_cycle_count_with_previous_fraction"] for row in counts)
    permutation, _ = nested_label_subsets([1, 2], session="626", fold=1, seed=V4_SEEDS[0])
    assert subsets[0]["permuted_train_cycle_ids"] == ordered_cycle_text(permutation)


def test_10_full_fraction_uses_every_train_cycle() -> None:
    _permutation, subsets = nested_label_subsets([9, 3, 7], session="708", fold=3, seed=V4_SEEDS[2])
    assert set(subsets[1.0]) == {3, 7, 9}


def test_11_cycle_selection_never_splits_a_cycle() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    cycle = int(np.unique(data.groups)[0])
    selected = labeled_sample_indices(data, [cycle])
    assert len(selected) == 4
    assert set(data.groups[selected]) == {cycle}


def test_12_class_balance_audit_is_valid_for_each_task() -> None:
    for task, expected_per_class in (("binary", 2), ("stimulus_type", 1)):
        data = load_block_sequence_session(PROJECT_DIR, "708", task, data_dir=DATA_DIR)
        cycle = int(np.unique(data.groups)[0])
        row = label_class_balance_row(
            data, fold=1, seed=V4_SEEDS[0], label_fraction=0.2, labeled_cycles=[cycle]
        )
        assert row["status"] == "VALID"
        assert row["class_0_samples"] == row["class_1_samples"] == expected_per_class


def test_13_single_class_subset_is_marked_invalid_without_swapping() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    altered = replace(data, y=np.zeros_like(data.y))
    cycle = int(np.unique(altered.groups)[0])
    row = label_class_balance_row(
        altered, fold=1, seed=V4_SEEDS[0], label_fraction=0.2, labeled_cycles=[cycle]
    )
    assert row["status"] == "INVALID_SINGLE_CLASS_TRAINING"
    assert row["class_1_samples"] == 0


def test_14_random_init_never_accepts_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="must not receive"):
        condition_encoder_state("RANDOM_INIT", Path("not_used.pt"))
    assert condition_encoder_state("RANDOM_INIT", None) is None


def test_15_masked_condition_requires_and_loads_checkpoint() -> None:
    checkpoint = V1_DIR / "pretraining/checkpoints/session_626/fold_1/seed_20260812.pt"
    with pytest.raises(ValueError, match="requires"):
        condition_encoder_state("WITHIN_MASKED_SSL_FT", None)
    state = condition_encoder_state("WITHIN_MASKED_SSL_FT", checkpoint)
    model = CNN2DMeanPool(n_classes=2, temporal_length=4)
    model.encoder.load_state_dict(state, strict=True)
    assert isinstance(model.encoder, SmallCNNFrameEncoder)


def test_16_checkpoint_reuse_audit_covers_full_outer_train_pool() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "626", "binary", data_dir=DATA_DIR)
    train_idx, test_idx = grouped_cv_splits(data.groups)[0]
    checkpoint = V1_DIR / "pretraining/checkpoints/session_626/fold_1/seed_20260812.pt"
    row, payload = audit_v1_checkpoint(
        checkpoint, session="626", fold=1, seed=V4_SEEDS[0],
        outer_train_cycles=np.unique(data.groups[train_idx]),
        outer_test_cycles=np.unique(data.groups[test_idx]),
        manifest_sha256=file_sha256(checkpoint),
    )
    assert row["reused"] and row["manifest_hash_match"]
    assert set(payload["ssl_train_cycles"]) | set(payload["ssl_val_cycles"]) == set(np.unique(data.groups[train_idx]))
    assert not (set(payload["ssl_train_cycles"]) | set(payload["ssl_val_cycles"])) & set(np.unique(data.groups[test_idx]))


def test_17_masked_ssl_method_and_supervised_config_are_frozen() -> None:
    assert MASK_BLOCK_SIZE == (16, 16) and MASK_RATIO == 0.50
    assert FROZEN_SSL_CONFIG.epochs == 50 and FROZEN_SSL_CONFIG.lr == 1e-3
    assert (
        FROZEN_SUPERVISED_CONFIG.optimizer,
        FROZEN_SUPERVISED_CONFIG.lr,
        FROZEN_SUPERVISED_CONFIG.weight_decay,
        FROZEN_SUPERVISED_CONFIG.batch_size,
        FROZEN_SUPERVISED_CONFIG.max_epochs,
        FROZEN_SUPERVISED_CONFIG.dropout,
    ) == ("adamw", 1e-3, 1e-3, 16, 40, 0.25)


def test_18_aulc_uses_trapezoidal_integration() -> None:
    assert trapezoidal_aulc(np.asarray([0.2, 0.6, 1.0]), np.asarray([0.5, 0.7, 0.9])) == pytest.approx(0.56)


def test_19_session_summary_keeps_all_fractions_and_deltas() -> None:
    table = session_label_efficiency(_complete_metrics())
    assert len(table) == 2 * 9 * 5
    assert set(table["label_fraction"]) == set(LABEL_FRACTIONS)
    assert np.allclose(table["delta_SSL_minus_Random"], 0.03)


def test_20_aulc_and_low_label_summaries_are_session_level() -> None:
    table = session_label_efficiency(_complete_metrics())
    aulc = label_efficiency_aulc(table)
    low = low_label_summary(table)
    assert len(aulc) == len(low) == 18
    assert np.allclose(aulc["delta_AULC"], 0.8 * 0.03)
    assert np.allclose(low["low_label_delta"], 0.03)


def test_21_planned_statistics_use_nine_sessions_and_four_tests() -> None:
    table = session_label_efficiency(_complete_metrics())
    tests = planned_statistical_tests(label_efficiency_aulc(table), low_label_summary(table))
    assert len(tests) == 4
    assert (tests["primary_unit"] == "session").all()
    assert (tests["n_sessions"] == 9).all()


def test_22_exact_sign_flip_enumerates_session_patterns() -> None:
    assert exact_sign_flip_pvalue(np.ones(9)) == pytest.approx(2 / 512)


def test_23_holm_is_monotone_and_never_smaller_than_raw() -> None:
    raw = np.asarray([0.01, 0.03, 0.02, 0.2])
    corrected = holm_adjust(raw)
    assert np.all(corrected >= raw)
    order = np.argsort(raw)
    assert np.all(np.diff(corrected[order]) >= 0)


def test_24_no_disallowed_method_or_spatial_pipeline_in_v4_core() -> None:
    import ultrasound_decoding.ssl_label_efficiency_v4 as module
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("vicreg", "simclr", "byol", "dino", "spatial_registration", "rigid_registration", "searchlight"):
        assert forbidden not in source


def test_25_formal_run_refuses_cpu_and_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="--device cuda"):
        assert_formal_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        assert_formal_cuda("cuda")


def test_26_output_completeness_detects_and_accepts_required_tree(tmp_path: Path) -> None:
    assert set(REQUIRED_FORMAL_OUTPUTS).issubset(set(missing_formal_outputs(tmp_path)))
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    curve = tmp_path / "downstream/training_curves/smoke.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("epoch,loss\n1,1\n", encoding="utf-8")
    assert missing_formal_outputs(tmp_path) == []


def test_27_full_label_v1_artifacts_cover_both_conditions_and_fixed_seeds() -> None:
    metrics = pd.read_csv(V1_DIR / "downstream/fold_metrics.csv")
    selected = metrics[metrics["condition"].isin(("RANDOM_INIT", "SSL_FINETUNE"))]
    assert len(selected) == 984
    assert set(selected["seed"].astype(int)) == set(V4_SEEDS)
    assert set(selected["task"]) == set(EXPECTED_TASKS)


def test_28_cycle_permutation_seed_controls_model_and_loader_seed_contract() -> None:
    source = inspect.getsource(nested_label_subsets)
    signature = inspect.signature(nested_label_subsets)
    assert "seed" in signature.parameters
    assert "deterministic_cycle_permutation" in source


def test_29_same_checkpoint_is_declared_for_all_five_fraction_rows() -> None:
    counts, _subsets = label_fraction_rows(range(7), session="626", fold=1, seed=V4_SEEDS[0])
    assert len(counts) == 5
    assert {row["seed"] for row in counts} == {V4_SEEDS[0]}
