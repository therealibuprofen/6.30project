from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ultrasound_decoding.multiframe.fcnn_temporal_statistics import (
    BOTTLENECK_DIM,
    MEAN_ONLY_VARIANT,
    MEAN_STD_VARIANT,
    STD_CORRECTION,
    FCNNMeanStd,
    architecture_config,
    build_bottleneck_temporal_statistics,
    build_model,
    parameter_audit,
)
from ultrasound_decoding.multiframe.models import FCNNMeanPool
from ultrasound_decoding.multiframe.training import (
    normalize_blocks_train_fold_only_with_stats,
    predict_probabilities,
    set_reproducible_seed,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_runner():
    path = (
        PROJECT_DIR
        / "scripts/baselines/run_fcnn_mean_std_temporal_statistics.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_fcnn_mean_std_temporal_statistics", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _representations(values: list[float], batch: int = 1) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 4, 1)
    return tensor.expand(batch, 4, BOTTLENECK_DIM).clone()


def test_temporal_mean_is_exact() -> None:
    observed = build_bottleneck_temporal_statistics(
        _representations([1, 2, 3, 4]), MEAN_ONLY_VARIANT
    )
    assert tuple(observed.shape) == (1, 3)
    assert torch.equal(observed, torch.full((1, 3), 2.5))


def test_population_temporal_std_is_exact() -> None:
    observed = build_bottleneck_temporal_statistics(
        _representations([1, 2, 3, 4]), MEAN_STD_VARIANT
    )
    expected = np.sqrt(1.25)
    assert STD_CORRECTION == 0
    assert torch.allclose(
        observed[:, 3:], torch.full((1, 3), expected), atol=1e-7
    )


def test_mean_only_is_the_historical_fcnn_meanpool_class_and_logits() -> None:
    set_reproducible_seed(17)
    historical = FCNNMeanPool(n_classes=2, temporal_length=4).eval()
    current = build_model(MEAN_ONLY_VARIANT, n_classes=2).eval()
    assert type(current) is FCNNMeanPool
    current.load_state_dict(historical.state_dict())
    x = torch.randn(2, 4, 1, 128, 501)
    with torch.no_grad():
        historical_logits = historical(x)
        current_logits = current(x)
        z = historical.encode_sequence(x)
        historical_mean = z.mean(dim=1)
        new_mean = build_bottleneck_temporal_statistics(
            z, MEAN_ONLY_VARIANT
        )
    assert torch.equal(new_mean, historical_mean)
    assert torch.equal(current_logits, historical_logits)


def test_mean_std_shape_is_six_dimensional() -> None:
    observed = build_bottleneck_temporal_statistics(
        torch.randn(5, 4, 3), MEAN_STD_VARIANT
    )
    assert tuple(observed.shape) == (5, 6)


def test_std_is_nonnegative() -> None:
    observed = build_bottleneck_temporal_statistics(
        torch.randn(9, 4, 3), MEAN_STD_VARIANT
    )
    assert bool((observed[:, 3:] >= 0).all())


def test_constant_temporal_bottleneck_has_zero_std() -> None:
    z = torch.randn(7, 1, 3).expand(7, 4, 3).clone()
    observed = build_bottleneck_temporal_statistics(z, MEAN_STD_VARIANT)
    assert torch.equal(observed[:, 3:], torch.zeros(7, 3))


def test_changing_test_fold_cannot_change_training_normalization_statistics() -> None:
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(3, 4, 5, 7)).astype(np.float32)
    x_test_a = np.zeros((2, 4, 5, 7), dtype=np.float32)
    x_test_b = np.full((2, 4, 5, 7), 1e6, dtype=np.float32)
    left = normalize_blocks_train_fold_only_with_stats(
        x_train,
        x_test_a,
        session="x",
        task="binary",
        method=MEAN_ONLY_VARIANT,
        seed=0,
        fold=1,
        train_cycles="1,2,3",
        test_cycles="4",
    )
    right = normalize_blocks_train_fold_only_with_stats(
        x_train,
        x_test_b,
        session="x",
        task="binary",
        method=MEAN_ONLY_VARIANT,
        seed=0,
        fold=1,
        train_cycles="1,2,3",
        test_cycles="4",
    )
    assert np.array_equal(left[0], right[0])
    assert np.array_equal(left[3], right[3])
    assert np.array_equal(left[4], right[4])
    assert left[2]["target_used_for_stats"] is False
    assert left[2]["statistics_scope"] == "train_blocks_all_four_frames_only"


def test_mean_only_forward_and_architecture_are_formal_fcnn() -> None:
    model = build_model(MEAN_ONLY_VARIANT, n_classes=2).eval()
    assert type(model) is FCNNMeanPool
    assert model.classifier.in_features == 3
    assert model.classifier.out_features == 2
    with torch.no_grad():
        logits = model(torch.zeros(2, 4, 1, 128, 501))
    assert tuple(logits.shape) == (2, 2)
    config = architecture_config(MEAN_ONLY_VARIANT)
    assert config["temporal_reduction"] == "mean"
    assert config["shared_frame_encoder_modified"] is False


def test_mean_std_only_changes_classifier_input_dimension() -> None:
    mean_only = build_model(MEAN_ONLY_VARIANT)
    mean_std = build_model(MEAN_STD_VARIANT)
    assert isinstance(mean_std, FCNNMeanStd)
    assert str(mean_only.encoder) == str(mean_std.encoder)
    assert list(mean_only.encoder.state_dict()) == list(mean_std.encoder.state_dict())
    for key in mean_only.encoder.state_dict():
        assert (
            mean_only.encoder.state_dict()[key].shape
            == mean_std.encoder.state_dict()[key].shape
        )
    assert mean_only.classifier.in_features == 3
    assert mean_std.classifier.in_features == 6
    assert mean_only.classifier.out_features == mean_std.classifier.out_features == 2


def test_parameter_delta_is_classifier_only() -> None:
    audit = parameter_audit()
    assert audit["mean_only_trainable_parameters"] == 48011
    assert audit["mean_std_trainable_parameters"] == 48017
    assert audit["parameter_delta"] == 6
    assert audit["parameter_delta_percentage"] == pytest.approx(
        100.0 * 6 / 48011
    )
    assert audit["delta_source"] == "classifier input dimension 3->6 only"
    assert audit["shared_frame_encoder_modified"] is False


def test_paired_seed_produces_identical_encoder_initialization() -> None:
    set_reproducible_seed(23)
    mean_only = build_model(MEAN_ONLY_VARIANT)
    set_reproducible_seed(23)
    mean_std = build_model(MEAN_STD_VARIANT)
    for key, value in mean_only.encoder.state_dict().items():
        assert torch.equal(value, mean_std.encoder.state_dict()[key])
    x = torch.randn(2, 4, 1, 128, 501)
    with torch.no_grad():
        mean_only_z = mean_only.encode_sequence(x)
        mean_std_z = mean_std.encode_sequence(x)
        mean_only_mean = build_bottleneck_temporal_statistics(
            mean_only_z, MEAN_ONLY_VARIANT
        )
        mean_std_mean = build_bottleneck_temporal_statistics(
            mean_std_z, MEAN_STD_VARIANT
        )[:, :BOTTLENECK_DIM]
    assert torch.equal(mean_only_z, mean_std_z)
    assert torch.equal(mean_only_mean, mean_std_mean)


def test_test_inference_requires_only_x() -> None:
    model = build_model(MEAN_STD_VARIANT).eval()
    probabilities = predict_probabilities(
        model,
        torch.randn(2, 4, 1, 128, 501),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_formal_task_plan_is_exactly_492_and_paired() -> None:
    runner = load_runner()
    fold_counts = {session: 10 for session in runner.EXPECTED_SESSIONS}
    fold_counts.update({"626": 8, "628": 8, "708": 6})
    assert sum(fold_counts.values()) == 82
    rows = []
    for session, n_folds in fold_counts.items():
        for variant in runner.INPUT_VARIANTS:
            for seed in runner.SEEDS:
                for fold in range(1, n_folds + 1):
                    rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": fold,
                            "n_test_samples": 4,
                            "train_cycles": "1,2",
                            "test_cycles": "3",
                            "task_key": f"{session}:{variant}:{seed}:{fold}",
                            "task_fingerprint": f"fp-{session}-{variant}-{seed}-{fold}",
                        }
                    )
    counts = runner.validate_task_plan(pd.DataFrame(rows))
    assert counts["number_of_folds"] == 82
    assert counts["expected_total_tasks"] == 492


def test_seed_summary_uses_fixed_epoch_40_not_best_epoch() -> None:
    runner = load_runner()
    per_fold, predictions, history = [], [], []
    for session in runner.EXPECTED_SESSIONS:
        for variant in runner.INPUT_VARIANTS:
            for seed in runner.SEEDS:
                per_fold.append(
                    {
                        "session": session,
                        "variant": variant,
                        "seed": seed,
                        "fold": 1,
                    }
                )
                predictions.extend(
                    [
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1,
                            "y_true": 0,
                            "y_pred": 0,
                        },
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1,
                            "y_true": 1,
                            "y_pred": 1,
                        },
                    ]
                )
                history.extend(
                    [
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1,
                            "epoch": 1,
                            "train_accuracy": 0.99,
                        },
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1,
                            "epoch": 40,
                            "train_accuracy": 0.75,
                        },
                    ]
                )
    summary = runner.build_seed_summary(
        pd.DataFrame(per_fold), pd.DataFrame(predictions), pd.DataFrame(history)
    )
    assert summary["final_epoch"].eq(40).all()
    assert np.allclose(summary["final_train_accuracy"], 0.75)
    assert np.allclose(summary["train_test_gap"], -0.25)


def test_corrupt_or_missing_task_is_not_resume_complete(tmp_path: Path) -> None:
    runner = load_runner()
    expected = {
        "session": "626",
        "variant": MEAN_ONLY_VARIANT,
        "seed": 0,
        "fold": 1,
        "n_test_samples": 4,
        "train_cycles": "1,2",
        "test_cycles": "3",
        "config_fingerprint": "config",
        "runtime_environment_fingerprint": "runtime",
    }
    valid, reason = runner.validate_completed_task(tmp_path, expected, "run")
    assert valid is False
    assert "missing files" in reason
    (tmp_path / "COMPLETE.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    valid, reason = runner.validate_completed_task(tmp_path, expected, "run")
    assert valid is False
    assert "missing files" in reason


def test_decision_rule_uses_exact_sign_flip_and_registered_label() -> None:
    runner = load_runner()
    rows = []
    for index, session in enumerate(runner.EXPECTED_SESSIONS):
        baseline = 0.6
        delta = 0.03 if session in runner.WEAK_SESSIONS else 0.0
        rows.append(
            {
                "session": session,
                "mean_only_BA": baseline,
                "mean_std_BA": baseline + delta,
                "delta_BA": delta,
            }
        )
    _overall, paired, decision = runner.build_overall_and_decision(
        pd.DataFrame(rows)
    )
    assert 0.0 <= paired.iloc[0]["exact_two_sided_sign_flip_p"] <= 1.0
    assert decision["decision"] in {
        "supports_continue_temporal_statistics_route",
        "does_not_support_temporal_statistics_route",
    }
    assert decision["automatic_next_stage_started"] is False
