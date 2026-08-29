from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.multiframe.fcnn_fourclass_late_fusion import (
    BLOCK_ORDER,
    CHANCE_LEVEL,
    CLASS_NAMES,
    CLASSES,
    EXPECTED_FOURCLASS_PARAMETERS,
    HISTORICAL_BINARY_PARAMETERS,
    TASK_NAME,
    architecture_config,
    binary_metrics_from_fourclass,
    build_model,
    coarse_error_audit,
    collapsed_binary_labels,
    collapsed_binary_probabilities,
    expand_training_frames,
    feasibility_gate,
    fixed_class_metrics,
    late_fuse_probabilities,
    load_fourclass_block_session,
    parameter_audit,
)
from ultrasound_decoding.multiframe.training import normalize_blocks_train_fold_only_with_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_ROOT / "scripts/baselines/run_fcnn_fourclass_late_fusion.py"
    spec = importlib.util.spec_from_file_location("run_fcnn_fourclass_late_fusion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_708():
    return load_fourclass_block_session(PROJECT_ROOT, "708")


def test_01_class_mapping_is_exact() -> None:
    assert TASK_NAME == "block_identity_4class"
    assert CLASS_NAMES == {0: "grating", 1: "stop_after_grating", 2: "dot", 3: "static"}


def test_02_real_data_contains_four_classes_and_no_negative_label(real_708) -> None:
    assert set(real_708.y.tolist()) == {0, 1, 2, 3}
    assert np.all(real_708.y >= 0)
    assert len(real_708.y) == len(real_708.metadata)


def test_03_every_real_cycle_has_each_class_once(real_708) -> None:
    for cycle in np.unique(real_708.groups):
        assert sorted(real_708.y[real_708.groups == cycle].tolist()) == [0, 1, 2, 3]


def test_04_real_class_balance_is_exact(real_708) -> None:
    assert np.array_equal(np.bincount(real_708.y, minlength=4), np.full(4, real_708.n_cycles))


def test_05_fcnn_output_dimension_is_four() -> None:
    assert build_model()(torch.zeros(2, 1, 128, 501)).shape == (2, 4)


def test_06_encoder_matches_historical_fcnn_exactly() -> None:
    historical = FCNN((128, 501), 2)
    current = build_model()
    assert [type(layer) for layer in historical[:-1]] == [type(layer) for layer in current[:-1]]
    assert historical[2].in_features == current[2].in_features == 16_000
    assert historical[2].out_features == current[2].out_features == 3


def test_07_parameter_delta_is_only_eight_output_parameters() -> None:
    assert parameter_audit() == {
        "historical_binary_parameters": HISTORICAL_BINARY_PARAMETERS,
        "fourclass_parameters": EXPECTED_FOURCLASS_PARAMETERS,
        "delta_parameters": 8,
    }
    assert build_model()[-1].weight.numel() + build_model()[-1].bias.numel() == 16


def test_08_training_expands_n_blocks_to_four_n_frames() -> None:
    X = np.zeros((3, 4, 5, 7), dtype=np.float32)
    frames, labels = expand_training_frames(X, np.asarray([0, 2, 3]))
    assert frames.shape == (12, 1, 5, 7)
    assert labels.tolist() == [0] * 4 + [2] * 4 + [3] * 4


def test_09_grouped_split_never_divides_cycle_frames(real_708) -> None:
    for train_idx, test_idx in grouped_cv_splits(real_708.groups):
        assert not (set(real_708.groups[train_idx]) & set(real_708.groups[test_idx]))


def test_10_outer_cycles_match_formal_reference_for_session_708(real_708) -> None:
    runner = load_runner()
    reference = pd.read_csv(PROJECT_ROOT / "outputs/fcnn_mean_std_temporal_statistics_v1/task_plan.csv", dtype={"session": str})
    reference = reference[reference["variant"] == "mean_only"]
    for fold, (train_idx, test_idx) in enumerate(grouped_cv_splits(real_708.groups), start=1):
        runner.assert_reference_fold_match(reference, "708", fold, runner.cycle_text(real_708.groups[train_idx]), runner.cycle_text(real_708.groups[test_idx]))


def _normalize(train: np.ndarray, test: np.ndarray):
    return normalize_blocks_train_fold_only_with_stats(
        train,
        test,
        session="x",
        task=TASK_NAME,
        method="fcnn_fourclass_late_fusion",
        seed=0,
        fold=1,
        train_cycles="0",
        test_cycles="1",
    )


def test_11_normalization_matches_outer_training_arcsinh_statistics() -> None:
    train = np.arange(2 * 4 * 3 * 2, dtype=np.float32).reshape(2, 4, 3, 2)
    test = np.ones((1, 4, 3, 2), dtype=np.float32)
    _, _, audit, mean, std = _normalize(train, test)
    expected = np.arcsinh(train).reshape(-1, 3, 2).astype(np.float64)
    assert np.allclose(mean, expected.mean(0, keepdims=True))
    assert np.allclose(std, expected.std(0, keepdims=True) + 1e-6)
    assert audit["target_used_for_stats"] is False


def test_12_changing_test_pixels_cannot_change_normalization_statistics() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(2, 4, 3, 2)).astype(np.float32)
    left = _normalize(train, np.zeros((1, 4, 3, 2), dtype=np.float32))
    right = _normalize(train, np.full((1, 4, 3, 2), 9999, dtype=np.float32))
    assert np.array_equal(left[3], right[3])
    assert np.array_equal(left[4], right[4])


def test_13_four_softmax_vectors_are_averaged_exactly() -> None:
    probabilities = np.asarray([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]])
    assert np.allclose(late_fuse_probabilities(probabilities), [[0.25] * 4])


def test_14_late_fusion_emits_one_prediction_per_block() -> None:
    probabilities = np.tile(np.asarray([[0.6, 0.2, 0.1, 0.1]]), (12, 1))
    assert late_fuse_probabilities(probabilities).shape == (3, 4)


def test_15_primary_oof_ba_is_concatenated_not_fold_ba_mean() -> None:
    y1 = np.asarray([0, 1, 2, 3])
    p1 = y1.copy()
    y2 = np.tile(y1, 2)
    p2 = np.asarray([0, 1, 0, 0, 0, 1, 0, 0])
    fold_mean = np.mean([fixed_class_metrics(y1, p1)["balanced_accuracy"], fixed_class_metrics(y2, p2)["balanced_accuracy"]])
    concatenated = fixed_class_metrics(np.r_[y1, y2], np.r_[p1, p2])["balanced_accuracy"]
    assert concatenated != fold_mean
    assert concatenated == pytest.approx(2 / 3)


def test_16_binary_collapse_mapping_is_exact() -> None:
    assert collapsed_binary_labels(np.asarray([0, 1, 2, 3])).tolist() == [1, 0, 1, 0]
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.4]])
    assert np.allclose(collapsed_binary_probabilities(probabilities), [[0.6, 0.4]])


def test_17_collapsed_probabilities_sum_to_one() -> None:
    probabilities = np.asarray([[0.4, 0.1, 0.4, 0.1], [0.0, 0.2, 0.3, 0.5]])
    assert np.allclose(collapsed_binary_probabilities(probabilities).sum(1), 1.0)


def test_18_within_and_cross_coarse_confusion_are_categorized() -> None:
    matrix = np.zeros((4, 4), dtype=int)
    matrix[0, 2] = 3
    matrix[1, 3] = 2
    matrix[0, 1] = 4
    matrix[3, 2] = 1
    audit = coarse_error_audit(matrix)
    assert audit["within_coarse_error_count"] == 5
    assert audit["cross_coarse_error_count"] == 5
    assert audit["within_coarse_error_fraction"] == 0.5


def _valid_task(tmp_path: Path):
    runner = load_runner()
    expected = {
        "session": "708", "seed": 0, "fold": 1, "train_cycles": "1", "test_cycles": "0",
        "dataset_fingerprint": "dataset", "split_fingerprint": "split", "task_fingerprint": "task", "run_fingerprint": "run",
        "task_key": "708:0:1", "n_test_blocks": 4, "n_test_frames": 16,
    }
    path = tmp_path / "task"
    path.mkdir()
    exact = {key: expected[key] for key in ("session", "seed", "fold", "train_cycles", "test_cycles", "dataset_fingerprint", "split_fingerprint", "task_fingerprint", "run_fingerprint")}
    result_payload = {**exact, "task": TASK_NAME, "model_name": runner.MODEL_NAME, "model_version": runner.MODEL_VERSION, "class_mapping": CLASS_NAMES}
    runner.framework.atomic_json(path / "result.json", result_payload)
    normalization = {"phase": "outer_train_fold_only", "target_used_for_stats": False, "test_used_for_normalization_fit": False}
    runner.framework.atomic_json(path / "normalization_audit.json", normalization)
    pd.DataFrame({"session": ["708"] * 40, "epoch": range(1, 41)}).to_csv(path / "training_history.csv", index=False)
    block_probs = np.eye(4)
    predictions = pd.DataFrame({
        "session": ["708"] * 4, "cycle_id": [0] * 4,
        "block_id": [f"b{i}" for i in range(4)], "block_name": BLOCK_ORDER,
        "true_label": np.arange(4), "pred_label": np.arange(4),
        **{f"prob_{name}": block_probs[:, index] for index, name in CLASS_NAMES.items()},
    })
    predictions.to_csv(path / "predictions.csv", index=False)
    frames = pd.DataFrame({
        "session": ["708"] * 16, "block_id": np.repeat(predictions["block_id"], 4), "frame_position": np.tile(np.arange(4), 4),
        **{f"prob_{name}": np.repeat(block_probs[:, index], 4) for index, name in CLASS_NAMES.items()},
    })
    frames.to_csv(path / "frame_predictions.csv", index=False)
    checkpoint = {
        **exact, "task": TASK_NAME, "model_name": runner.MODEL_NAME, "model_version": runner.MODEL_VERSION,
        "class_mapping": CLASS_NAMES, "classes": CLASSES.tolist(), "epoch": 40,
        "optimizer_config": vars(runner.frozen_training_config()), "model_state_dict": build_model().state_dict(),
        "normalization_mean": np.zeros((1, 128, 501), dtype=np.float32), "normalization_std": np.ones((1, 128, 501), dtype=np.float32),
        "normalization_config": normalization, "source_protocol_fingerprint": "run", "dataset_manifest_fingerprint": "dataset",
    }
    torch.save(checkpoint, path / "checkpoint.pt")
    runner.framework.atomic_json(path / "COMPLETE.json", {"status": "complete", "task_key": expected["task_key"], "artifact_sha256": runner.task_artifact_hashes(path)})
    assert runner.validate_completed_task(path, expected)[0]
    return runner, path, expected, checkpoint


@pytest.mark.parametrize("mutation", ["class_mapping", "output_head", "fold_membership"])
def test_19_resume_rejects_mapping_head_or_fold_drift(tmp_path: Path, mutation: str) -> None:
    runner, path, expected, checkpoint = _valid_task(tmp_path)
    if mutation == "class_mapping":
        checkpoint["class_mapping"] = {0: "wrong"}
        torch.save(checkpoint, path / "checkpoint.pt")
    elif mutation == "output_head":
        checkpoint["model_state_dict"] = FCNN((128, 501), 2).state_dict()
        torch.save(checkpoint, path / "checkpoint.pt")
    else:
        predictions = pd.read_csv(path / "predictions.csv")
        predictions["cycle_id"] = 9
        predictions.to_csv(path / "predictions.csv", index=False)
    runner.framework.atomic_json(path / "COMPLETE.json", {"status": "complete", "task_key": expected["task_key"], "artifact_sha256": runner.task_artifact_hashes(path)})
    assert not runner.validate_completed_task(path, expected)[0]


def test_20_run_complete_requires_exactly_246_valid_checkpoints() -> None:
    runner = load_runner()
    manifest = pd.DataFrame({"validation": ["PASS"] * 246})
    predictions = pd.DataFrame({"true_label": [0, 1, 2, 3]})
    runner.assert_run_completion_ready(246, manifest, predictions, {"status": "PASS"})
    with pytest.raises(AssertionError, match="246/246"):
        runner.assert_run_completion_ready(245, manifest, predictions, {"status": "PASS"})


def test_21_feasibility_gate_thresholds_and_tokens_are_frozen() -> None:
    sufficient = feasibility_gate(0.35, np.asarray([0.31] * 6 + [0.30] * 3), 0.55)
    insufficient = feasibility_gate(0.349, np.asarray([0.31] * 9), 0.9)
    assert sufficient["decision"] == "four_class_signal_sufficient_for_multitask_experiment"
    assert insufficient["decision"] == "four_class_signal_insufficient_for_multitask_experiment"


def test_22_architecture_contains_no_unapproved_model_or_selection() -> None:
    config = architecture_config()
    assert config["layers"] == ["MaxPool2d(kernel_size=2,stride=2)", "Flatten(16000)", "Linear(16000,3)", "ReLU", "Linear(3,4)"]
    assert config["fusion"] == "arithmetic_mean_of_four_frame_softmax_vectors"


def test_23_plan_constants_are_strict() -> None:
    runner = load_runner()
    config = runner.protocol_config()
    assert (len(config["sessions"]), config["expected_folds"], len(config["seeds"]), config["expected_tasks"], len(config["class_mapping"]), config["chance_level"]) == (9, 82, 3, 246, 4, CHANCE_LEVEL)


def test_24_binary_diagnostic_uses_probabilities_without_retraining() -> None:
    probabilities = np.eye(4)
    metrics = binary_metrics_from_fourclass(np.arange(4), probabilities)
    assert metrics == {"accuracy": 1.0, "balanced_accuracy": 1.0}
