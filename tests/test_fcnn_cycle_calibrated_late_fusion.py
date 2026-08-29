from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch

from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.multiframe.cycle_calibrated_late_fusion import (
    FORMAL_TRAINING_CONFIG,
    FROZEN_GATE,
    IMAGE_SHAPE,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    apply_inner_normalization,
    assert_complete_inner_oof,
    build_inner_cache_key,
    build_inner_cycle_splits,
    calibrated_frame_probabilities,
    equal_four_frame_probability_mean,
    evaluate_frozen_gate,
    fit_inner_train_normalization,
    fit_scalar_temperature,
    softmax_probabilities,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_fcnn_cycle_calibrated_late_fusion.py"
    spec = importlib.util.spec_from_file_location("run_fcnn_cycle_calibrated_late_fusion", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_blocks(seed: int, n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 4, *IMAGE_SHAPE)).astype(np.float32)


def test_historical_outer_architecture_and_training_protocol_are_frozen() -> None:
    model = FCNN(input_shape=IMAGE_SHAPE, n_classes=2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 48_011
    assert tuple(FORMAL_TRAINING_CONFIG.__dict__.values())[:5] == (
        "adamw", 0.001, 0.001, 16, 40
    )
    runner = load_runner()
    protocol = runner.formal_protocol()
    assert protocol["outer_model"]["training_performed"] is False
    assert protocol["outer_model"]["weights_updated"] is False
    assert protocol["expected_inner_trainings"] == 738


def test_baseline_exact_reconstruction_uses_frame_softmax_then_equal_mean() -> None:
    logits = np.asarray(
        [[2.0, -1.0], [0.0, 1.0], [-2.0, 0.5], [1.0, 1.0],
         [-1.0, 3.0], [2.0, 0.0], [0.2, 0.3], [-0.4, 1.2]]
    )
    frame_probs = softmax_probabilities(logits)
    observed = equal_four_frame_probability_mean(frame_probs.reshape(2, 4, 2))
    expected = np.stack([frame_probs[:4].mean(axis=0), frame_probs[4:].mean(axis=0)])
    assert np.array_equal(observed, expected)


def test_formal_plan_implies_246_outer_tasks_and_738_inner_trainings() -> None:
    fold_counts = {"626": 8, "628": 8, "708": 6, "709": 10, "710": 10, "807": 10, "813": 10, "817": 10, "822": 10}
    assert sum(fold_counts.values()) == 82
    outer_tasks = sum(value * 3 for value in fold_counts.values())
    assert outer_tasks == 246
    assert outer_tasks * 3 == 738
    assert "validate_historical_checkpoint_coverage" in inspect.getsource(
        load_runner().build_plan
    )


def test_inner_splits_only_use_outer_training_cycles() -> None:
    splits = build_inner_cycle_splits([0, 1, 2, 3, 4, 5, 6], [7])
    for split in splits:
        assert not ({7} & set(split.train_cycles))
        assert not ({7} & set(split.validation_cycles))
        assert set(split.train_cycles) | set(split.validation_cycles) == set(range(7))


def test_inner_cycle_group_isolation_and_exact_crossfit() -> None:
    splits = build_inner_cycle_splits(range(8), [8, 9])
    heldout = []
    for split in splits:
        assert not (set(split.train_cycles) & set(split.validation_cycles))
        heldout.extend(split.validation_cycles)
    assert sorted(heldout) == list(range(8))
    assert len(heldout) == len(set(heldout))


def test_inner_normalization_is_fit_on_inner_train_only() -> None:
    train = small_blocks(1, 2)
    validation = small_blocks(2, 2)
    mean_a, std_a, fp_a = fit_inner_train_normalization(train)
    changed_validation = validation + 1e6
    mean_b, std_b, fp_b = fit_inner_train_normalization(train)
    _ = apply_inner_normalization(changed_validation, mean_b, std_b)
    assert np.array_equal(mean_a, mean_b)
    assert np.array_equal(std_a, std_b)
    assert fp_a == fp_b


def test_outer_test_pixel_or_label_mutation_cannot_affect_temperature() -> None:
    logits = np.asarray([[2.0, -1.0], [1.0, 0.0], [-0.5, 1.0], [-1.0, 2.0]])
    labels = np.asarray([0, 0, 1, 1])
    outer_test_pixels = np.zeros((2, 4, 2, 2), dtype=np.float32)
    outer_test_labels = np.asarray([0, 1])
    before = fit_scalar_temperature(logits, labels).temperature
    outer_test_pixels[:] = 9999.0
    outer_test_labels[:] = 1 - outer_test_labels
    after = fit_scalar_temperature(logits, labels).temperature
    assert before == after
    assert set(inspect.signature(fit_scalar_temperature).parameters) == {"logits", "labels"}


def test_each_outer_training_frame_has_exactly_one_inner_oof_prediction() -> None:
    source = np.repeat(np.asarray([2, 4, 8]), 4)
    cycles = np.repeat(np.asarray([0, 1, 2]), 4)
    assert_complete_inner_oof(source, np.asarray([2, 4, 8]), cycles, cycles.copy())
    duplicate = source.copy()
    duplicate[-1] = 4
    try:
        assert_complete_inner_oof(duplicate, np.asarray([2, 4, 8]), cycles, cycles)
    except AssertionError:
        pass
    else:
        raise AssertionError("duplicate inner OOF frame was accepted")


def test_scalar_temperature_is_positive_and_nll_nonincreasing() -> None:
    logits = np.asarray([[8.0, -8.0], [6.0, -6.0], [-5.0, 5.0], [4.0, -4.0]])
    labels = np.asarray([0, 1, 1, 0])
    result = fit_scalar_temperature(logits, labels)
    assert np.isscalar(result.temperature)
    assert result.temperature > 0
    assert result.post_nll <= result.pre_nll + 1e-10
    assert result.objective == "cross_entropy_nll"


def test_temperature_fitting_uses_nll_only_and_has_no_ba_or_test_inputs() -> None:
    source = inspect.getsource(fit_scalar_temperature)
    parameters = set(inspect.signature(fit_scalar_temperature).parameters)
    assert parameters == {"logits", "labels"}
    assert "balanced_accuracy" not in source
    assert "test" not in source
    assert "_temperature_objective" in source
    assert "multiclass_nll" in source


def test_t_equals_one_exactly_reproduces_baseline_probabilities() -> None:
    logits = np.asarray([[0.1, 0.9], [5.0, -2.0], [-1.2, 0.4]])
    assert np.array_equal(
        calibrated_frame_probabilities(logits, 1.0),
        softmax_probabilities(logits),
    )


def test_calibrated_probabilities_are_finite_and_sum_to_one() -> None:
    logits = np.asarray([[1000.0, -1000.0], [-1000.0, 1000.0], [0.0, 0.0]])
    probabilities = calibrated_frame_probabilities(logits, 2.5)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_four_frame_fusion_is_strict_equal_arithmetic_average() -> None:
    values = np.asarray([[[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]])
    observed = equal_four_frame_probability_mean(values)
    assert np.array_equal(observed, np.asarray([[0.55, 0.45]]))
    assert set(inspect.signature(equal_four_frame_probability_mean).parameters) == {"frame_probabilities"}


def test_fusion_has_no_confidence_entropy_margin_attention_or_gating() -> None:
    source = inspect.getsource(equal_four_frame_probability_mean)
    for forbidden in ("confidence", "entropy", "margin", "attention", "gate", "weight"):
        assert forbidden not in source.lower()


def test_fusion_has_no_timestamp_frame_position_or_block_type_input() -> None:
    parameters = set(inspect.signature(equal_four_frame_probability_mean).parameters)
    assert parameters == {"frame_probabilities"}
    source = inspect.getsource(equal_four_frame_probability_mean).lower()
    for forbidden in ("timestamp", "relative_time", "frame_position", "block_type", "block_name"):
        assert forbidden not in source


def test_oof_ba_is_computed_after_concatenation_not_mean_fold_ba() -> None:
    runner = load_runner()
    predictions = pd.DataFrame(
        {
            "session": ["626"] * 10,
            "seed": [0] * 10,
            "fold": [1] * 2 + [2] * 8,
            "truth": [0, 1] + [0, 0, 0, 0, 1, 1, 1, 1],
            "baseline_pred": [0, 1] + [0, 0, 0, 0, 0, 0, 0, 0],
            "cclf_pred": [0, 1] + [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    seed_summary, _ = runner.summarize_ba(predictions)
    assert seed_summary.iloc[0]["baseline_ba"] == 0.60
    assert seed_summary.iloc[0]["cclf_ba"] == 1.0
    assert seed_summary.iloc[0]["baseline_ba"] != (1.0 + 0.5) / 2.0


def test_frozen_gate_requires_all_four_conditions() -> None:
    passed = evaluate_frozen_gate(
        baseline_overall_ba=0.60, cclf_overall_ba=0.605,
        baseline_strong_ba=0.80, cclf_strong_ba=0.79,
        baseline_weak_ba=0.45, cclf_weak_ba=0.46,
        baseline_overall_ece=0.20, cclf_overall_ece=0.16,
    )
    assert all(passed["passes"].values())
    assert passed["decision"] == "supports_cycle_calibrated_late_fusion"
    failed = evaluate_frozen_gate(
        baseline_overall_ba=0.60, cclf_overall_ba=0.6049,
        baseline_strong_ba=0.80, cclf_strong_ba=0.79,
        baseline_weak_ba=0.45, cclf_weak_ba=0.46,
        baseline_overall_ece=0.20, cclf_overall_ece=0.16,
    )
    assert failed["passes"]["A"] is False
    assert failed["decision"] == "does_not_support_cycle_calibrated_late_fusion"


def test_strong_and_weak_sessions_are_fixed() -> None:
    assert STRONG_SESSIONS == ("708", "709", "710")
    assert WEAK_SESSIONS == ("626", "628", "807", "813", "817", "822")
    assert not (set(STRONG_SESSIONS) & set(WEAK_SESSIONS))


def cache_key(outer_fold: int, normalization: str = "norm-a") -> str:
    return build_inner_cache_key(
        session="708",
        outer_fold=outer_fold,
        outer_seed=0,
        outer_train_cycles=[1, 2, 3, 4, 5],
        inner_fold=1,
        inner_train_cycles=[3, 4, 5],
        inner_validation_cycles=[1, 2],
        source_hash="source",
        protocol_hash="protocol",
        normalization_fingerprint=normalization,
        training_config=vars(FORMAL_TRAINING_CONFIG),
    )


def test_cache_is_isolated_by_parent_outer_fold_and_normalization() -> None:
    assert cache_key(1) != cache_key(2)
    assert cache_key(1, "norm-a") != cache_key(1, "norm-b")


def test_formal_source_hash_set_contains_all_direct_runtime_dependencies() -> None:
    runner = load_runner()
    observed = {str(path.relative_to(PROJECT_DIR)) for path in runner.source_paths(PROJECT_DIR)}
    required = {
        "scripts/baselines/run_fcnn_cycle_calibrated_late_fusion.py",
        "scripts/baselines/run_multiscale_temporal1d.py",
        "src/ultrasound_decoding/multiframe/cycle_calibrated_late_fusion.py",
        "src/ultrasound_decoding/multiframe/canonical_single_frame.py",
        "src/ultrasound_decoding/multiframe/training.py",
        "src/ultrasound_decoding/multiframe/models.py",
        "src/ultrasound_decoding/multiframe/dataset.py",
        "src/ultrasound_decoding/deep.py",
        "src/ultrasound_decoding/evaluate.py",
        "configs/fcnn_cycle_calibrated_late_fusion_v1.json",
        "docs/fcnn_cycle_calibrated_late_fusion_v1.md",
    }
    assert required == observed


def test_aggregate_hashes_cover_every_required_output_except_run_complete() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in runner.REQUIRED_RUN_OUTPUTS:
            (root / name).write_text(name, encoding="utf-8")
        hashes = runner.aggregate_artifact_sha256(root)
        assert set(hashes) == set(runner.REQUIRED_RUN_OUTPUTS)
        assert "RUN_COMPLETE.json" not in hashes
        valid, reason = runner.validate_aggregate_artifact_integrity(root, {"aggregate_artifact_sha256": hashes})
        assert valid, reason


def test_status_integrity_validation_detects_corrupted_final_artifact() -> None:
    runner = load_runner()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name in runner.REQUIRED_RUN_OUTPUTS:
            (root / name).write_text(name, encoding="utf-8")
        complete = {"aggregate_artifact_sha256": runner.aggregate_artifact_sha256(root)}
        (root / "predictions.csv").write_text("corrupted", encoding="utf-8")
        valid, reason = runner.validate_aggregate_artifact_integrity(root, complete)
        assert valid is False
        assert "predictions.csv" in reason
        assert "integrity-failed" in inspect.getsource(runner.run_status)


def test_review_gate_blocks_formal_full_without_explicit_approval() -> None:
    runner = load_runner()
    args = type("Args", (), {"review_approved": False})()
    try:
        runner.run_full(args)
    except RuntimeError as exc:
        assert "review-approved" in str(exc)
    else:
        raise AssertionError("formal full was not blocked before code review")


def test_no_inner_or_outer_checkpoint_is_a_required_output() -> None:
    runner = load_runner()
    assert not any(name.endswith(".pt") for name in runner.REQUIRED_TASK_FILES)
    assert not any(name.endswith(".pt") for name in runner.REQUIRED_RUN_OUTPUTS)


def test_protocol_records_complete_provenance_and_strict_resume_fields() -> None:
    runner = load_runner()
    source = inspect.getsource(runner)
    for required in (
        "git_head", "source_hashes", "run_fingerprint", "task_fingerprint",
        "historical_checkpoint_sha256", "aggregate_artifact_sha256",
        "normalization_fingerprint", "cache_key",
    ):
        assert required in source


def test_config_matches_frozen_gate_and_equal_weights() -> None:
    config = json.loads(
        (PROJECT_DIR / "configs/fcnn_cycle_calibrated_late_fusion_v1.json").read_text(encoding="utf-8")
    )
    assert config["frozen_gate"] == FROZEN_GATE
    assert config["fusion"]["weights"] == [0.25, 0.25, 0.25, 0.25]
    assert config["inner_cross_fit"]["expected_trainings"] == 738
