from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.multiframe.dataset import EXPECTED_BLOCK_SHAPE
from ultrasound_decoding.multiframe.models import CNN2DTemporal1D, count_trainable_parameters
from ultrasound_decoding.multiframe.response_weighted_temporal1d import (
    RAW_VARIANT,
    RESPONSE_WEIGHTED_VARIANT,
    TrainingResponseMap,
    build_training_response_map,
    presence_contrast_from_block_images,
    preprocess_and_normalize_train_fold_only,
    response_map_cache_key,
    response_score_from_cycle_contrasts,
    response_score_to_soft_weight,
)
from ultrasound_decoding.multiframe.training import (
    normalize_blocks_train_fold_only_with_stats,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
BLOCK_NAMES = np.asarray(["grating", "stop_after_grating", "dot", "static"])


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_response_weighted_temporal1d.py"
    spec = importlib.util.spec_from_file_location(
        "run_response_weighted_temporal1d", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_fold(n_cycles: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return complete four-block clean4 cycles with spatially distinct stability."""

    names = np.tile(BLOCK_NAMES, n_cycles)
    cycles = np.repeat(np.arange(1, n_cycles + 1), 4)
    X = np.zeros((len(names), 4, 2, 3), dtype=np.float32)
    for cycle_i in range(n_cycles):
        base = cycle_i * 4
        stimulus = np.asarray(
            [
                [3.0, 3.0 if cycle_i % 2 == 0 else -3.0, 1.0 + cycle_i],
                [-2.0, 0.2 * cycle_i, 0.5],
            ],
            dtype=np.float32,
        )
        X[base] = stimulus
        X[base + 2] = stimulus
    return X, names, cycles


def normalization_kwargs() -> dict[str, object]:
    return {
        "session": "synthetic",
        "fold": 1,
        "seed": 0,
        "train_cycles": "1,2,3,4",
        "test_cycles": "5",
    }


def test_presence_contrast_formula_returns_two() -> None:
    observed = presence_contrast_from_block_images(
        np.asarray([[3.0]]),
        np.asarray([[1.0]]),
        np.asarray([[3.0]]),
        np.asarray([[1.0]]),
    )
    assert observed.item() == pytest.approx(2.0)


def test_stable_response_scores_above_sign_flipping_response() -> None:
    contrasts = np.asarray(
        [[[2.0, 2.0]], [[2.1, -2.0]], [[1.9, 2.0]], [[2.0, -2.0]]]
    )
    score = response_score_from_cycle_contrasts(contrasts)
    assert score[0, 0] > score[0, 1]


def test_stable_negative_response_has_high_score() -> None:
    contrasts = np.asarray(
        [[[-2.0, 2.0]], [[-2.1, -2.0]], [[-1.9, 2.0]], [[-2.0, -2.0]]]
    )
    score = response_score_from_cycle_contrasts(contrasts)
    assert score[0, 0] > score[0, 1]
    assert score[0, 0] > 10.0


def test_soft_weight_range_and_spatial_mean_with_average_ties() -> None:
    score = np.asarray([[0.0, 1.0, 1.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    weight = response_score_to_soft_weight(score)
    assert float(weight.min()) >= 0.5 - 1e-6
    assert float(weight.max()) <= 1.5 + 1e-6
    assert float(weight.mean()) == pytest.approx(1.0, abs=2e-6)
    assert weight[0, 1] == weight[0, 2]


def test_changing_test_cycle_cannot_change_training_response_map() -> None:
    X_train, names_train, cycles_train = synthetic_fold()
    rng = np.random.default_rng(4)
    X_test = rng.normal(size=(4, 4, 2, 3)).astype(np.float32)
    full_before = np.concatenate([X_train, X_test])
    full_after = full_before.copy()
    full_after[len(X_train) :] += 1_000_000.0
    before = build_training_response_map(
        full_before[: len(X_train)], names_train, cycles_train
    )
    after = build_training_response_map(
        full_after[: len(X_train)], names_train, cycles_train
    )
    assert np.array_equal(before.weight_map, after.weight_map)
    assert before.response_map_hash == after.response_map_hash
    assert before.weight_map_hash == after.weight_map_hash


def test_changing_training_cycle_changes_weight_map() -> None:
    X, names, cycles = synthetic_fold()
    before = build_training_response_map(X, names, cycles)
    changed = X.copy()
    cycle_four_grating = int(np.flatnonzero((cycles == 4) & (names == "grating"))[0])
    changed[cycle_four_grating, :, 0, 0] = -100.0
    changed[cycle_four_grating, :, 1, 2] = 100.0
    after = build_training_response_map(changed, names, cycles)
    assert not np.array_equal(before.weight_map, after.weight_map)
    assert before.response_map_hash != after.response_map_hash


def test_weighting_occurs_after_train_only_normalization() -> None:
    rng = np.random.default_rng(5)
    train = rng.normal(size=(8, 4, 2, 3)).astype(np.float32)
    test = rng.normal(size=(4, 4, 2, 3)).astype(np.float32)
    weight = response_score_to_soft_weight(
        np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    )
    response_map = TrainingResponseMap(
        response_score=np.arange(6, dtype=np.float32).reshape(2, 3),
        weight_map=weight,
        cycle_contrasts=np.zeros((2, 2, 3), dtype=np.float32),
        train_cycle_ids=(1, 2),
        response_map_hash="response-hash",
        weight_map_hash="weight-hash",
        eps=1e-6,
    )
    kwargs = normalization_kwargs()
    observed = preprocess_and_normalize_train_fold_only(
        train,
        test,
        input_variant=RESPONSE_WEIGHTED_VARIANT,
        response_map=response_map,
        **kwargs,
    )
    reference = normalize_blocks_train_fold_only_with_stats(
        train, test, task="binary", method=RESPONSE_WEIGHTED_VARIANT, **kwargs
    )
    assert np.array_equal(observed[0], reference[0] * weight[None, None])
    assert np.array_equal(observed[1], reference[1] * weight[None, None])
    assert observed[2]["response_weighting_after_normalization"] is True
    assert observed[2]["test_data_used_to_fit_response_map"] is False


def test_raw_path_is_bitwise_existing_formal_normalization() -> None:
    rng = np.random.default_rng(6)
    train = rng.normal(size=(8, 4, 2, 3)).astype(np.float32)
    test = rng.normal(size=(4, 4, 2, 3)).astype(np.float32)
    kwargs = normalization_kwargs()
    observed = preprocess_and_normalize_train_fold_only(
        train, test, input_variant=RAW_VARIANT, response_map=None, **kwargs
    )
    reference = normalize_blocks_train_fold_only_with_stats(
        train, test, task="binary", method=RAW_VARIANT, **kwargs
    )
    for observed_array, reference_array in zip(
        (observed[0], observed[1], observed[3], observed[4]),
        (reference[0], reference[1], reference[3], reference[4]),
    ):
        assert np.array_equal(observed_array, reference_array)
    assert observed[2]["response_weighting_after_normalization"] is False


def test_cache_key_binds_fold_cycles_version_and_source() -> None:
    common = {"session": "710", "train_cycle_ids": [1, 2], "relevant_source_sha256": "abc"}
    key = response_map_cache_key(fold=1, **common)
    assert key != response_map_cache_key(fold=2, **common)
    assert key != response_map_cache_key(
        session="710", fold=1, train_cycle_ids=[1, 3], relevant_source_sha256="abc"
    )
    assert key != response_map_cache_key(
        session="710", fold=1, train_cycle_ids=[1, 2], relevant_source_sha256="def"
    )


def test_temporal1d_architecture_and_exact_sign_flip_are_unchanged() -> None:
    assert count_trainable_parameters(CNN2DTemporal1D(2, temporal_length=4)) == 115890
    assert load_runner().exact_two_sided_sign_flip(np.ones(9)) == pytest.approx(2 / 512)


def test_real_clean4_response_map_is_training_fold_only_when_data_present(
    tmp_path: Path,
) -> None:
    data_dir = PROJECT_DIR / "processed_data/block_sequences_v1"
    if not (data_dir / "session_710_blocks.h5").is_file():
        pytest.skip("real clean4 data are unavailable")
    runner = load_runner()
    args = SimpleNamespace(
        project_root=PROJECT_DIR,
        data_dir=data_dir,
        output_dir=tmp_path,
        formal_fold_run_dir=PROJECT_DIR / "outputs/multiscale_temporal1d_v1",
    )
    data, splits = runner.audit_session(args, "710")
    train_idx, test_idx = splits[0]
    response_map = build_training_response_map(
        data.X[train_idx],
        data.metadata.iloc[train_idx]["block_name"].astype(str).to_numpy(),
        data.groups[train_idx],
    )
    assert tuple(data.X.shape[1:]) == EXPECTED_BLOCK_SHAPE
    assert response_map.response_score.shape == EXPECTED_BLOCK_SHAPE[-2:]
    assert set(response_map.train_cycle_ids) == set(data.groups[train_idx])
    assert set(response_map.train_cycle_ids).isdisjoint(set(data.groups[test_idx]))


def test_summary_uses_epoch_40_final_train_accuracy_and_required_schema() -> None:
    runner = load_runner()
    fold_rows, prediction_rows, history_rows = [], [], []
    truth = np.asarray([0, 0, 1, 1])
    for session in runner.EXPECTED_SESSIONS:
        for variant in runner.INPUT_VARIANTS:
            predicted = truth if variant == RESPONSE_WEIGHTED_VARIANT else np.asarray([0, 1, 1, 0])
            for seed in runner.SEEDS:
                for fold in (1, 2):
                    fold_rows.append(
                        {"session": session, "variant": variant, "seed": seed, "fold": fold, "n_samples": 4, "n_cycles": 2}
                    )
                for sample_index, (y_true, y_pred) in enumerate(zip(truth, predicted)):
                    prediction_rows.append(
                        {"session": session, "variant": variant, "seed": seed, "fold": 1 if sample_index < 2 else 2, "sample_index": sample_index, "y_true": y_true, "y_pred": y_pred}
                    )
                for fold in (1, 2):
                    for epoch in range(1, runner.FORMAL_EPOCHS + 1):
                        history_rows.append(
                            {"session": session, "variant": variant, "seed": seed, "fold": fold, "epoch": epoch, "train_accuracy": 0.99 if epoch == 20 else 0.5 + 0.001 * epoch + 0.01 * fold}
                        )
    seed_summary = runner.build_seed_summary(
        pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), pd.DataFrame(history_rows)
    )
    assert {"mean_oof_BA", "final_train_accuracy", "train_test_gap"}.issubset(seed_summary)
    assert seed_summary.iloc[0]["final_train_accuracy"] == pytest.approx(0.555)
    assert seed_summary.iloc[0]["train_test_gap"] == pytest.approx(0.055)
    session_summary = runner.build_session_summary(seed_summary)
    overall, paired, decision = runner.build_overall_and_decision(session_summary)
    assert overall.iloc[0]["median_delta_BA"] == pytest.approx(0.5)
    assert paired.iloc[0]["exact_two_sided_sign_flip_p"] == pytest.approx(2 / 512)
    assert decision["decision"] == "supports_continue_response_guided_route"
