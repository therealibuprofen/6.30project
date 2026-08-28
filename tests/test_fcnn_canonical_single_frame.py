from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn

from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.multiframe.canonical_single_frame import (
    EXPECTED_IMAGE_SHAPE,
    EXPECTED_PARAMETERS,
    NORMALIZATION_TRANSFORM,
    apply_saved_normalization,
    count_parameters,
    predict_single_frame_probabilities,
    reconstruct_late_fusion_probabilities,
    select_canonical_frames,
    select_canonical_positions,
    validate_checkpoint_payload,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_fcnn_canonical_single_frame.py"
    spec = importlib.util.spec_from_file_location(
        "run_fcnn_canonical_single_frame", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_payload(
    *, session: str = "708", seed: int = 0, fold: int = 1
) -> dict:
    model = FCNN(input_shape=EXPECTED_IMAGE_SHAPE, n_classes=2)
    return {
        "method": "fcnn_late_fusion",
        "model_config": {
            "base_model": "official_single_frame_FCNN",
            "late_fusion_probability_average": True,
            "temporal_length": 4,
        },
        "model_parameters": EXPECTED_PARAMETERS,
        "model_state_dict": model.state_dict(),
        "classes": [0, 1],
        "session": session,
        "task": "binary",
        "seed": seed,
        "fold": fold,
        "train_cycles": "1,2,3",
        "test_cycles": "0",
        "max_epochs": 40,
        "final_epoch": 40,
        "normalization_mean": np.zeros(
            (1, *EXPECTED_IMAGE_SHAPE), dtype=np.float32
        ),
        "normalization_std": np.ones(
            (1, *EXPECTED_IMAGE_SHAPE), dtype=np.float32
        ),
        "normalization_transform": NORMALIZATION_TRANSFORM,
        "input_shape": [4, *EXPECTED_IMAGE_SHAPE],
        "code_version": "test-commit",
    }


def formal_plan_rows() -> list[dict]:
    runner = load_runner()
    fold_counts = {session: 10 for session in runner.EXPECTED_SESSIONS}
    fold_counts.update({"626": 8, "628": 8, "708": 6})
    cycle_counts = {
        "626": 8,
        "628": 8,
        "708": 6,
        "709": 22,
        "710": 18,
        "807": 12,
        "813": 10,
        "817": 20,
        "822": 10,
    }
    rows = []
    for session in runner.EXPECTED_SESSIONS:
        n_folds = fold_counts[session]
        for seed in runner.SEEDS:
            for fold in range(1, n_folds + 1):
                base_test_cycles = cycle_counts[session] // n_folds
                extra = cycle_counts[session] % n_folds
                n_test_cycles = base_test_cycles + int(fold <= extra)
                rows.append(
                    {
                        "session": session,
                        "variant": "mean_only",
                        "seed": seed,
                        "fold": fold,
                        "n_train_samples": 4
                        * (cycle_counts[session] - n_test_cycles),
                        "n_test_samples": 4 * n_test_cycles,
                        "train_cycles": "1,2,3",
                        "test_cycles": "0",
                    }
                )
    return rows


def test_timestamp_midpoint_nearest_is_exact() -> None:
    result = select_canonical_positions(np.asarray([2.0, 9.0, 14.5, 22.0]))
    assert result.positions.tolist() == [2]
    assert result.relative_times_s.tolist() == [14.5]
    assert result.distances_to_midpoint_s.tolist() == [0.5]
    assert result.ties.tolist() == [False]


def test_timestamp_tie_chooses_earlier() -> None:
    result = select_canonical_positions(np.asarray([10.0, 20.0]))
    assert result.positions.tolist() == [0]
    assert result.relative_times_s.tolist() == [10.0]
    assert result.ties.tolist() == [True]


def test_real_clean4_grids_select_expected_midpoint_frame() -> None:
    cases = [
        ([10.0, 14.0, 18.0, 22.0], 1, 14.0),
        ([8.0, 12.0, 16.0, 20.0], 2, 16.0),
    ]
    for times, position, timestamp in cases:
        result = select_canonical_positions(np.asarray(times))
        assert result.positions.tolist() == [position]
        assert result.relative_times_s.tolist() == [timestamp]


def test_canonical_selection_api_cannot_read_label_prediction_or_ba() -> None:
    parameters = set(inspect.signature(select_canonical_positions).parameters)
    assert parameters == {"relative_times_s", "midpoint_s"}
    source = inspect.getsource(select_canonical_positions)
    assert "label" not in source
    assert "prediction" not in source
    assert "balanced_accuracy" not in source


def test_checkpoint_payload_is_exact_historical_single_frame_fcnn() -> None:
    payload = valid_payload()
    audit = validate_checkpoint_payload(
        payload,
        expected_session="708",
        expected_seed=0,
        expected_fold=1,
        expected_train_cycles="1,2,3",
        expected_test_cycles="0",
    )
    assert audit["valid"] is True
    model = FCNN(input_shape=EXPECTED_IMAGE_SHAPE, n_classes=2)
    assert count_parameters(model) == EXPECTED_PARAMETERS


def test_checkpoint_membership_mismatch_stops() -> None:
    try:
        validate_checkpoint_payload(
            valid_payload(),
            expected_session="708",
            expected_seed=0,
            expected_fold=1,
            expected_train_cycles="9",
            expected_test_cycles="0",
        )
    except AssertionError as exc:
        assert "train_cycles" in str(exc)
    else:
        raise AssertionError("membership mismatch did not stop")


def test_saved_normalization_is_applied_without_fitting_test_data() -> None:
    frames = np.zeros((2, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    mean = np.ones((1, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    std = np.full((1, *EXPECTED_IMAGE_SHAPE), 2.0, dtype=np.float32)
    observed = apply_saved_normalization(
        frames, mean, std, transform=NORMALIZATION_TRANSFORM
    )
    assert np.array_equal(observed, np.full_like(observed, -0.5))
    altered_test = np.full_like(frames, 1000.0)
    _ = apply_saved_normalization(
        altered_test, mean, std, transform=NORMALIZATION_TRANSFORM
    )
    assert np.array_equal(mean, np.ones_like(mean))
    assert np.array_equal(std, np.full_like(std, 2.0))
    assert set(inspect.signature(apply_saved_normalization).parameters) == {
        "frames",
        "mean",
        "std",
        "transform",
    }


class ForwardAuditModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls: list[tuple[tuple[int, ...], bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls.append((tuple(x.shape), self.training, torch.is_grad_enabled()))
        score = x[:, 0, 0, 0] + self.anchor
        return torch.stack((-score, score), dim=1)


def test_model_eval_no_grad_and_one_prediction_per_block() -> None:
    model = ForwardAuditModel().train()
    frames = np.zeros((3, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    probabilities = predict_single_frame_probabilities(model, frames, batch_size=2)
    assert probabilities.shape == (3, 2)
    assert model.training is False
    assert all(shape[1:] == (1, *EXPECTED_IMAGE_SHAPE) for shape, _, _ in model.calls)
    assert all(training is False for _, training, _ in model.calls)
    assert all(grad_enabled is False for _, _, grad_enabled in model.calls)


def test_single_frame_path_forwards_only_selected_canonical_frames() -> None:
    blocks = np.zeros((2, 4, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    blocks[:, 0] = -100.0
    blocks[:, 1] = 1.0
    blocks[:, 2] = 2.0
    blocks[:, 3] = 100.0
    times = np.asarray(
        [[10.0, 14.0, 18.0, 22.0], [8.0, 12.0, 16.0, 20.0]]
    )
    frames, selection = select_canonical_frames(blocks, times)
    assert selection.positions.tolist() == [1, 2]
    assert np.all(frames[0] == 1.0)
    assert np.all(frames[1] == 2.0)
    model = ForwardAuditModel()
    _ = predict_single_frame_probabilities(model, frames, batch_size=2)
    assert model.calls == [((2, 1, *EXPECTED_IMAGE_SHAPE), False, False)]


def test_single_frame_module_has_no_late_fusion_averaging_call() -> None:
    source = inspect.getsource(predict_single_frame_probabilities)
    assert "late_fusion" not in source
    assert "reshape" not in source
    assert "mean(" not in source
    assert "majority" not in source


def test_reconstructed_late_fusion_matches_historical_probability_average() -> None:
    model = ForwardAuditModel()
    blocks = np.zeros((2, 4, *EXPECTED_IMAGE_SHAPE), dtype=np.float32)
    scores = np.asarray([[-2.0, -1.0, 1.0, 2.0], [0.0, 0.5, 1.5, 3.0]])
    blocks[:, :, 0, 0] = scores
    observed = reconstruct_late_fusion_probabilities(
        model, blocks, batch_size=3
    )
    logits = np.stack((-scores, scores), axis=-1)
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    expected = (exponentials / exponentials.sum(axis=-1, keepdims=True)).mean(
        axis=1
    )
    assert np.allclose(observed, expected, atol=1e-7, rtol=1e-7)


def late_fusion_reference_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for sample_i, block_name, truth, probability in (
        (0, "grating", 1, (0.2, 0.8)),
        (1, "static", 0, (0.7, 0.3)),
    ):
        rows.append(
            {
                "session": "626",
                "seed": 0,
                "fold": 1,
                "sample_i": sample_i,
                "block_id": f"session626_cycle000_{block_name}",
                "cycle": 0,
                "block_name": block_name,
                "truth": truth,
                "pred": int(probability[1] > probability[0]),
                "prob_no_stimulus": probability[0],
                "prob_stimulus": probability[1],
            }
        )
    historical = pd.DataFrame(rows)
    reconstructed = historical.copy()
    reconstructed["source_index"] = [0, 3]
    return reconstructed, historical


def test_historical_late_fusion_probability_mismatch_stops_summary() -> None:
    runner = load_runner()
    reconstructed, historical = late_fusion_reference_frames()
    reconstructed.loc[0, "prob_stimulus"] += 1e-3
    _audit, summary = runner.build_late_fusion_reconstruction_audit(
        reconstructed,
        historical,
        expected_blocks=2,
        expected_tasks=1,
    )
    assert summary["status"] == "FAIL"
    with np.testing.assert_raises(AssertionError):
        runner.require_late_fusion_reconstruction_pass(summary)


def test_meanpool_sample_or_truth_mismatch_stops() -> None:
    runner = load_runner()
    canonical = pd.DataFrame(
        {
            "session": ["626", "626"],
            "seed": [0, 0],
            "fold": [1, 1],
            "source_index": [0, 3],
            "cycle": [0, 0],
            "block_name": ["grating", "static"],
            "y_true": [1, 0],
        }
    )
    meanpool = canonical.copy()
    meanpool.loc[1, "y_true"] = 1
    with np.testing.assert_raises(AssertionError):
        runner.validate_meanpool_sample_identity(
            canonical,
            meanpool,
            expected_blocks=2,
            expected_session_seed_groups=1,
        )


def valid_meanpool_protocol_payloads() -> tuple[dict, dict]:
    complete = {
        "status": "complete",
        "completed_tasks": 492,
        "expected_tasks": 492,
        "total_tasks": 492,
        "number_of_sessions": 9,
        "number_of_variants": 2,
        "number_of_seeds": 3,
        "number_of_folds": 82,
        "model_implementation_version": (
            "fcnn_mean_std_temporal_statistics_v1.0.0"
        ),
    }
    experiment = {
        "output_version": "fcnn_mean_std_temporal_statistics_v1",
        "task": "binary_presence",
        "class_mapping": {"0": "no_stimulus", "1": "stimulus"},
        "input_protocol": "clean4",
        "raw_input_shape": [4, 128, 501],
        "sessions": ["626", "628", "708", "709", "710", "807", "813", "817", "822"],
        "seeds": [0, 1, 2],
        "variants": ["mean_only", "mean_std"],
        "architectures": {
            "mean_only": {
                "method": "fcnn_bottleneck_temporal_statistics",
                "variant": "mean_only",
                "model_implementation_version": (
                    "fcnn_mean_std_temporal_statistics_v1.0.0"
                ),
                "temporal_length": 4,
                "temporal_reduction": "mean",
                "trainable_parameters": 48011,
            }
        },
        "training": {"max_epochs": 40},
        "epoch_selection": "fixed 40 epochs; no validation or early stopping",
        "cv": "exact formal clean4 cycle-grouped folds, max_folds=10",
        "normalization": (
            "pixel z-score fit on outer-training blocks and all four real frames only"
        ),
        "preprocessing": (
            "clean4 -> per-frame arcsinh -> outer-train-fold all-frame "
            "pixel z-score -> unchanged shared FCNN frame encoder -> "
            "bottleneck temporal statistics"
        ),
        "test_used_for_normalization": False,
        "test_used_for_feature_scaling": False,
        "test_used_for_model_selection": False,
        "test_used_for_early_stopping": False,
    }
    config = {
        "model_implementation_version": (
            "fcnn_mean_std_temporal_statistics_v1.0.0"
        ),
        "experiment_config": experiment,
    }
    return complete, config


def test_meanpool_scientific_protocol_mismatch_stops() -> None:
    runner = load_runner()
    complete, config = valid_meanpool_protocol_payloads()
    runner.validate_meanpool_protocol_payloads(complete, config)
    config["experiment_config"]["input_protocol"] = "not_clean4"
    with np.testing.assert_raises(AssertionError):
        runner.validate_meanpool_protocol_payloads(complete, config)


def test_direct_framework_import_is_in_source_provenance() -> None:
    runner = load_runner()
    relative = {
        str(path.resolve().relative_to(PROJECT_DIR.resolve()))
        for path in runner.source_paths(PROJECT_DIR)
    }
    assert "scripts/baselines/run_multiscale_temporal1d.py" in relative


def test_formal_task_plan_is_246_and_exactly_maps_current_outer_plan() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = Path(temporary) / "meanpool"
        run_dir.mkdir()
        pd.DataFrame(formal_plan_rows()).to_csv(
            run_dir / "task_plan.csv", index=False
        )
        args = SimpleNamespace(meanpool_run_dir=run_dir)
        plan = runner.load_formal_task_plan(args)
        assert len(plan) == 246
        assert plan[["session", "seed", "fold"]].drop_duplicates().shape[0] == 246
        assert plan[["session", "fold"]].drop_duplicates().shape[0] == 82
        assert plan["task_key"].nunique() == 246


def test_checkpoint_coverage_preflight_requires_246_valid_tasks() -> None:
    runner = load_runner()
    plan = pd.DataFrame(formal_plan_rows())
    plan["task_key"] = plan.apply(
        lambda row: f"{row.session}:{row.seed}:{row.fold}", axis=1
    )
    with tempfile.TemporaryDirectory() as temporary:
        placeholder = Path(temporary) / "checkpoint.pt"
        placeholder.write_bytes(b"read-only-placeholder")
        source_rows = lambda _args, _session: {
            (seed, fold): {"checkpoint_sha256": "sha"}
            for seed in runner.SEEDS
            for fold in range(1, 11)
        }
        fake_load = lambda path, **kwargs: (
            nn.Linear(1, 1),
            {
                "code_version": "source",
                "method": "fcnn_late_fusion",
                "model_config": {"base_model": "official_single_frame_FCNN"},
                "model_parameters": 48011,
                "final_epoch": 40,
                "train_cycles": kwargs["expected_train_cycles"],
                "test_cycles": kwargs["expected_test_cycles"],
                "normalization_transform": NORMALIZATION_TRANSFORM,
                "normalization_mean": np.zeros(
                    (1, *EXPECTED_IMAGE_SHAPE), dtype=np.float32
                ),
            },
            {"checkpoint_sha256": "sha"},
        )
        with mock.patch.object(runner, "source_checkpoint_rows", source_rows), mock.patch.object(
            runner, "checkpoint_path_for_task", lambda *_args: placeholder
        ), mock.patch.object(runner, "load_validated_checkpoint", fake_load):
            manifest = runner.validate_all_checkpoints(SimpleNamespace(), plan)
        assert len(manifest) == 246
        assert manifest["valid"].all()
        assert manifest["task_key"].nunique() == 246


def test_oof_ba_is_computed_after_concatenating_fold_predictions() -> None:
    runner = load_runner()
    rows = []
    truth = [0, 1, 0, 0, 0, 1]
    prediction = [0, 1, 1, 1, 1, 1]
    for session in runner.EXPECTED_SESSIONS:
        for seed in runner.SEEDS:
            for value_true, value_pred in zip(truth, prediction):
                rows.append(
                    {
                        "session": session,
                        "seed": seed,
                        "truth": value_true,
                        "pred": value_pred,
                    }
                )
    summary = runner.reference_seed_ba(
        pd.DataFrame(rows),
        truth_column="truth",
        prediction_column="pred",
        method_name="test",
    )
    # Fold-wise BA would be mean([1.0, 0.5]) = 0.75. Concatenated OOF is
    # mean(class-0 recall=1/4, class-1 recall=2/2) = 0.625.
    assert np.isclose(summary.iloc[0]["test_BA"], 0.625)
    assert not np.isclose(summary.iloc[0]["test_BA"], 0.75)


def test_exact_sign_flip_reuses_all_512_patterns() -> None:
    runner = load_runner()
    assert np.isclose(runner.exact_two_sided_sign_flip(np.ones(9)), 2 / 512)


def test_run_complete_requires_full_246_of_246() -> None:
    runner = load_runner()
    complete = {
        "status": "complete",
        "completed_tasks": 246,
        "expected_tasks": 246,
        "number_of_folds": 82,
        "number_of_sessions": 9,
        "number_of_seeds": 3,
        "test_block_predictions": 1368,
        "expected_test_block_predictions": 1368,
        "train_diagnostic_block_predictions": 11640,
        "expected_train_diagnostic_block_predictions": 11640,
        "canonical_block_forwards": 13008,
        "expected_canonical_block_forwards": 13008,
        "late_fusion_reconstructed_block_predictions": 1368,
        "expected_late_fusion_reconstructed_block_predictions": 1368,
        "late_fusion_verification_frame_forwards": 5472,
        "expected_late_fusion_verification_frame_forwards": 5472,
        "total_model_frame_forwards": 18480,
        "expected_total_model_frame_forwards": 18480,
        "late_fusion_reconstruction_status": "PASS",
        "meanpool_sample_identity_status": "PASS",
        "training_performed": False,
        "device": "cpu",
    }
    assert runner.validate_run_complete_payload(complete) is True
    complete["completed_tasks"] = 245
    assert runner.validate_run_complete_payload(complete) is False


def test_runner_contains_no_training_entrypoint() -> None:
    runner = load_runner()
    source = inspect.getsource(runner)
    assert "train_fold(" not in source
    assert "optimizer.step(" not in source
    assert "loss.backward(" not in source
    assert "--review-approved" in source


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
