from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ultrasound_decoding.multiframe.dataset import EXPECTED_BLOCK_SHAPE
from ultrasound_decoding.multiframe.models import CNN2DTemporal1D, count_trainable_parameters
from ultrasound_decoding.multiframe.spatial_demean_temporal1d import (
    RAW_VARIANT,
    SPATIAL_DEMEAN_VARIANT,
    apply_input_variant_after_arcsinh,
    preprocess_and_normalize_train_fold_only,
    spatial_demean_per_frame,
)
from ultrasound_decoding.multiframe.training import (
    normalize_blocks_train_fold_only_with_stats,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_spatial_demean_temporal1d.py"
    spec = importlib.util.spec_from_file_location("run_spatial_demean_temporal1d", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalization_kwargs() -> dict[str, object]:
    return {
        "session": "synthetic",
        "fold": 1,
        "seed": 0,
        "train_cycles": "1,2",
        "test_cycles": "3",
    }


def test_spatial_demean_makes_every_frame_spatial_mean_zero_and_preserves_shape() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 4, 9, 11)).astype(np.float32)
    result = spatial_demean_per_frame(x)
    assert result.shape == x.shape
    assert result.dtype == x.dtype
    assert np.allclose(result.mean(axis=(-2, -1)), 0.0, atol=2e-7)


def test_raw_variant_is_identity_and_does_not_modify_input() -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(2, 4, 5, 7)).astype(np.float32)
    before = x.copy()
    result = apply_input_variant_after_arcsinh(x, RAW_VARIANT)
    assert result is x
    assert np.array_equal(result, before)


def test_spatial_demean_never_mixes_samples_or_time_frames() -> None:
    base = np.arange(30, dtype=np.float32).reshape(5, 6)
    offsets = np.asarray([[1.0, 10.0, 100.0], [1000.0, -5.0, 0.25]], dtype=np.float32)
    x = np.stack(
        [np.stack([base + offsets[n, t] for t in range(3)]) for n in range(2)]
    )
    expected = base - base.mean()
    result = spatial_demean_per_frame(x)
    assert np.allclose(result, expected[None, None, :, :], atol=1e-5)

    changed = x.copy()
    changed[1, 2, 0, 0] += 12345.0
    changed_result = spatial_demean_per_frame(changed)
    unaffected = np.ones((2, 3), dtype=bool)
    unaffected[1, 2] = False
    assert np.array_equal(changed_result[unaffected], result[unaffected])


def test_raw_preprocessing_is_bitwise_existing_formal_normalization() -> None:
    rng = np.random.default_rng(9)
    train = rng.normal(size=(6, 4, 5, 7)).astype(np.float32)
    test = rng.normal(size=(2, 4, 5, 7)).astype(np.float32)
    kwargs = normalization_kwargs()
    observed = preprocess_and_normalize_train_fold_only(
        train, test, input_variant=RAW_VARIANT, **kwargs
    )
    reference = normalize_blocks_train_fold_only_with_stats(
        train,
        test,
        task="binary",
        method=RAW_VARIANT,
        **kwargs,
    )
    assert np.array_equal(observed[0], reference[0])
    assert np.array_equal(observed[1], reference[1])
    assert np.array_equal(observed[3], reference[3])
    assert np.array_equal(observed[4], reference[4])
    assert observed[2]["preprocessing_order"] == (
        "clean4 -> arcsinh -> train_fold_pixel_zscore"
    )


def test_demean_is_after_arcsinh_before_train_only_zscore_and_test_is_not_fit() -> None:
    rng = np.random.default_rng(10)
    train = rng.normal(size=(5, 4, 6, 8)).astype(np.float32)
    test = rng.normal(size=(3, 4, 6, 8)).astype(np.float32)
    kwargs = normalization_kwargs()
    observed = preprocess_and_normalize_train_fold_only(
        train, test, input_variant=SPATIAL_DEMEAN_VARIANT, **kwargs
    )
    manual_train = spatial_demean_per_frame(np.arcsinh(train))
    manual_test = spatial_demean_per_frame(np.arcsinh(test))
    frames = manual_train.reshape(-1, 6, 8).astype(np.float64)
    mean = frames.mean(axis=0, keepdims=True)
    std = frames.std(axis=0, keepdims=True) + 1e-6
    assert np.allclose(observed[0], (manual_train - mean) / std, atol=2e-6)
    assert np.allclose(observed[1], (manual_test - mean) / std, atol=2e-6)
    assert observed[2]["target_used_for_stats"] is False
    assert observed[2]["spatial_demean_statistics_scope"] == "one_sample_one_frame_only"

    replaced_test = test.copy()
    replaced_test[1:] += 1_000_000.0
    rerun = preprocess_and_normalize_train_fold_only(
        train, replaced_test, input_variant=SPATIAL_DEMEAN_VARIANT, **kwargs
    )
    assert np.array_equal(rerun[0], observed[0])
    assert np.array_equal(rerun[3], observed[3])
    assert np.array_equal(rerun[4], observed[4])
    assert np.array_equal(rerun[1][0], observed[1][0])


def test_formal_temporal1d_architecture_is_unchanged() -> None:
    model = CNN2DTemporal1D(n_classes=2, temporal_length=4)
    assert count_trainable_parameters(model) == 115890


def test_clean4_fold_and_normalization_pipeline_is_intact_when_data_present(
    tmp_path: Path,
) -> None:
    data_dir = PROJECT_DIR / "processed_data/block_sequences_v1"
    if not (data_dir / "session_710_blocks.h5").is_file():
        pytest.skip("real clean4 data are unavailable")
    runner = load_runner()
    args = SimpleNamespace(
        project_root=PROJECT_DIR,
        data_dir=data_dir,
        benchmark_root=(
            PROJECT_DIR
            / "results/runs/multiframe/block_clean4_binary_v1"
        ),
        output_dir=tmp_path,
        formal_fold_run_dir=PROJECT_DIR / "outputs/multiscale_temporal1d_v1",
    )
    data, splits = runner.audit_session(args, "710")
    assert tuple(data.X.shape[1:]) == EXPECTED_BLOCK_SHAPE
    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert set(data.groups[train_idx]).isdisjoint(set(data.groups[test_idx]))
    train_idx, test_idx = splits[0]
    raw = preprocess_and_normalize_train_fold_only(
        data.X[train_idx],
        data.X[test_idx],
        input_variant=RAW_VARIANT,
        session="710",
        fold=1,
        seed=0,
        train_cycles="formal",
        test_cycles="formal",
    )
    assert raw[0].shape == data.X[train_idx].shape
    assert raw[1].shape == data.X[test_idx].shape
    assert raw[2]["phase"] == "outer_train_fold_only"


def test_exact_sign_flip_reuses_project_512_pattern_implementation() -> None:
    runner = load_runner()
    assert runner.exact_two_sided_sign_flip(np.ones(9)) == pytest.approx(2 / 512)


def test_required_summary_schema_and_historical_final_epoch_overfit_definition() -> None:
    runner = load_runner()
    fold_rows, prediction_rows, history_rows = [], [], []
    truth = np.asarray([0, 0, 1, 1])
    for session in runner.EXPECTED_SESSIONS:
        for variant in runner.INPUT_VARIANTS:
            predicted = truth if variant == SPATIAL_DEMEAN_VARIANT else np.asarray([0, 1, 1, 0])
            for seed in runner.SEEDS:
                for fold in (1, 2):
                    fold_rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": fold,
                            "n_samples": 4,
                            "n_cycles": 2,
                            "train_accuracy": 0.99,
                        }
                    )
                for sample_index, (y_true, y_pred) in enumerate(zip(truth, predicted)):
                    prediction_rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1 if sample_index < 2 else 2,
                            "sample_index": sample_index,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }
                    )
                for fold in (1, 2):
                    for epoch in range(1, runner.FORMAL_EPOCHS + 1):
                        history_rows.append(
                            {
                                "session": session,
                                "variant": variant,
                                "seed": seed,
                                "fold": fold,
                                "epoch": epoch,
                                "train_accuracy": (
                                    0.95 + 0.01 * fold
                                    if epoch == 20
                                    else 0.5 + 0.001 * epoch + 0.01 * fold
                                ),
                            }
                        )
    seed_summary = runner.build_seed_summary(
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(history_rows),
    )
    assert {
        "session",
        "variant",
        "seed",
        "mean_oof_BA",
        "train_accuracy",
        "train_test_gap",
    }.issubset(seed_summary.columns)
    # Epoch 20 is deliberately higher (.96/.97), but the historical formal
    # audit uses fixed epoch 40 (.55/.56), averaged across folds.
    assert seed_summary.iloc[0]["train_accuracy"] == pytest.approx(0.555)
    assert seed_summary.iloc[0]["train_test_gap"] == pytest.approx(0.055)

    session_summary = runner.build_session_summary(seed_summary)
    assert list(session_summary.columns) == [
        "session",
        "raw_BA",
        "spatial_demean_BA",
        "delta_BA",
        "raw_train_acc",
        "demean_train_acc",
        "raw_gap",
        "demean_gap",
        "delta_gap",
    ]
    overall, paired, decision = runner.build_overall_and_decision(session_summary)
    assert overall.iloc[0]["overall_delta_BA"] == pytest.approx(0.5)
    assert paired.iloc[0]["exact_two_sided_sign_flip_p"] == pytest.approx(2 / 512)
    assert decision["decision"] == "supports_continue_spatial_pattern_route"
