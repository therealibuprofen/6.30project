from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
for search_path in (PROJECT_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ultrasound_decoding.multiframe.models import (
    CNN2DTemporal1D,
    SmallCNNFrameEncoder,
    count_trainable_parameters,
)
from ultrasound_decoding.multiframe.multiscale_temporal1d import (
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_TEMPORAL_BASELINE_NAME,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    MODEL_NAMES,
    SINGLE_SCALE_MODEL_NAME,
    ControlledScaleFrameEncoder,
    architecture_config,
    build_model,
    formal_temporal1d_audit,
    parameter_breakdown,
)
from ultrasound_decoding.multiframe.training import (
    normalize_blocks_train_fold_only_with_stats,
)


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_multiscale_temporal1d.py"
    spec = importlib.util.spec_from_file_location("run_multiscale_temporal1d", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_temporal1d_source_audit_is_exact() -> None:
    audit = formal_temporal1d_audit()
    reference = CNN2DTemporal1D(n_classes=2, dropout=0.25, norm="batchnorm")
    assert SmallCNNFrameEncoder.feature_dim == 16 * 4 * 8 == 512
    assert audit["frame_feature_dim"] == 512
    assert audit["spatial_output_shape_per_frame"] == "[16,4,8]"
    assert audit["temporal_conv_repr"] == repr(reference.temporal_conv)
    assert audit["classifier_repr"] == repr(reference.classifier)
    first = reference.temporal_conv[0]
    assert isinstance(first, nn.Conv1d)
    assert (first.in_channels, first.out_channels, first.kernel_size, first.padding) == (
        512,
        64,
        (3,),
        (1,),
    )
    assert audit["training"] == {
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.001,
        "batch_size": 16,
        "max_epochs": 40,
        "dropout": 0.25,
        "loss": "cross_entropy",
        "epoch_selection": "fixed_epochs_no_test_fold_selection",
    }


def test_first_stage_matches_formal_smallcnn_and_modes_are_controlled() -> None:
    formal = SmallCNNFrameEncoder()
    single = ControlledScaleFrameEncoder("single_scale")
    multi = ControlledScaleFrameEncoder("multiscale")
    assert repr(single.first_stage) == repr(multi.first_stage)
    assert repr(single.first_stage) == repr(formal.layers[:4])
    assert isinstance(single.single_scale, nn.Conv2d)
    assert single.single_scale.in_channels == 8
    assert single.single_scale.out_channels == 16
    assert single.single_scale.kernel_size == (3, 3)
    assert single.single_scale.dilation == (1, 1)
    assert multi.local_branch.out_channels == 8
    assert multi.local_branch.dilation == (1, 1)
    assert multi.local_branch.padding == (1, 1)
    assert multi.context_branch.out_channels == 8
    assert multi.context_branch.dilation == (2, 2)
    assert multi.context_branch.padding == (2, 2)
    assert not any(
        token in repr(multi).lower()
        for token in ("attention", "mamba", "transformer", "linear(in_features=16")
    )


def test_two_models_reuse_identical_unchanged_temporal_head_and_classifier() -> None:
    reference = CNN2DTemporal1D(n_classes=2, dropout=0.25, norm="batchnorm")
    single = build_model(SINGLE_SCALE_MODEL_NAME)
    multi = build_model(MODEL_NAME)
    assert repr(single.temporal_conv) == repr(multi.temporal_conv)
    assert repr(single.classifier) == repr(multi.classifier)
    assert repr(single.temporal_conv) == repr(reference.temporal_conv)
    assert repr(single.classifier) == repr(reference.classifier)
    assert single.encoder_feature_dim == multi.encoder_feature_dim == 512
    assert type(single) is type(multi)


def test_spatial_and_full_shapes_forward_backward() -> None:
    torch.manual_seed(0)
    frames = torch.randn(8, 1, 128, 501)
    blocks = frames.reshape(2, 4, 1, 128, 501)
    for model_name in MODEL_NAMES:
        model = build_model(model_name)
        spatial = model.encoder.forward_spatial(frames)
        assert spatial.shape == (8, 16, 4, 8)
        encoded = model.encode_sequence(blocks)
        assert encoded.shape == (2, 4, 512)
        logits, shapes = model.forward_with_shapes(blocks)
        assert logits.shape == (2, 2)
        assert shapes == {
            "input": (2, 4, 1, 128, 501),
            "frame_sequence": (2, 4, 512),
            "temporal_input": (2, 512, 4),
            "temporal_features": (2, 64),
            "logits": (2, 2),
        }
        nn.CrossEntropyLoss()(logits, torch.tensor([0, 1])).backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_parameter_counts_are_frozen_and_branches_are_naturally_matched() -> None:
    assert MODEL_IMPLEMENTATION_VERSION == "multiscale_temporal1d_v1.0.0"
    assert EXPECTED_PARAMETER_COUNTS == {
        SINGLE_SCALE_MODEL_NAME: 112_562,
        MODEL_NAME: 112_562,
        FORMAL_TEMPORAL_BASELINE_NAME: 115_890,
    }
    single = parameter_breakdown(build_model(SINGLE_SCALE_MODEL_NAME))
    multi = parameter_breakdown(build_model(MODEL_NAME))
    assert single["frame_encoder_parameters"] == multi["frame_encoder_parameters"] == 1584
    assert single["context_branch_parameters"] == 0
    assert single["local_or_single_branch_parameters"] == 1168
    assert multi["local_or_single_branch_parameters"] == 584
    assert multi["context_branch_parameters"] == 584
    assert single["temporal_1d_parameters"] == multi["temporal_1d_parameters"] == 110848
    assert single["classifier_parameters"] == multi["classifier_parameters"] == 130
    assert count_trainable_parameters(CNN2DTemporal1D(2)) == 115890


def test_architecture_audit_freezes_only_receptive_field_difference() -> None:
    single = architecture_config(SINGLE_SCALE_MODEL_NAME)
    multi = architecture_config(MODEL_NAME)
    shared_keys = {
        "shared_first_stage",
        "post_second_stage",
        "spatial_output_shape",
        "frame_feature_dim",
        "temporal_head",
        "classifier",
        "expected_parameter_count",
    }
    for key in shared_keys:
        assert single[key] == multi[key]
    assert single["encoder_mode"] == "single_scale"
    assert multi["encoder_mode"] == "multiscale"
    assert "dilation=2" in multi["controlled_second_stage"]
    assert single["branch_gate_present"] is multi["branch_gate_present"] is False
    assert single["mamba_present"] is multi["mamba_present"] is False


def test_train_only_normalization_is_independent_of_test_values() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(3, 4, 5, 7)).astype(np.float32)
    test_a = rng.normal(size=(2, 4, 5, 7)).astype(np.float32)
    test_b = test_a + 10000.0
    common = dict(
        session="710",
        task="binary",
        method=MODEL_NAME,
        seed=0,
        fold=1,
        train_cycles="1,2",
        test_cycles="3",
    )
    train_a, _, audit_a, mean_a, std_a = normalize_blocks_train_fold_only_with_stats(
        train, test_a, **common
    )
    train_b, _, audit_b, mean_b, std_b = normalize_blocks_train_fold_only_with_stats(
        train, test_b, **common
    )
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(std_a, std_b)
    assert audit_a["target_used_for_stats"] is audit_b["target_used_for_stats"] is False


def test_runner_freezes_protocol_models_and_review_lock() -> None:
    runner = load_runner()
    config = runner.frozen_experiment_config(16)
    assert config["sessions"] == list(runner.EXPECTED_SESSIONS)
    assert config["seeds"] == [0, 1, 2]
    assert config["mechanistic_models"] == list(MODEL_NAMES)
    assert config["training"] == {
        "optimizer": "adamw",
        "lr": 0.001,
        "weight_decay": 0.001,
        "batch_size": 16,
        "max_epochs": 40,
        "dropout": 0.25,
        "loss": "cross_entropy",
    }
    assert config["automatic_next_stage"] is False
    source = (PROJECT_DIR / "scripts/baselines/run_multiscale_temporal1d.py").read_text()
    assert "for model_name in MODEL_NAMES" in source
    with pytest.raises(RuntimeError, match="code review is approved"):
        runner.run_full(SimpleNamespace(review_approved=False), {})


def test_task_plan_contains_two_models_and_492_formal_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    fold_counts = {
        "626": 8,
        "628": 8,
        "708": 6,
        "709": 10,
        "710": 10,
        "807": 10,
        "813": 10,
        "817": 10,
        "822": 10,
    }

    def fake_audit(args, session):
        n_folds = fold_counts[session]
        data = SimpleNamespace(n_cycles=n_folds, n_blocks=n_folds * 4)
        splits = [
            (np.asarray([1, 2]), np.asarray([0])) for _ in range(n_folds)
        ]
        return data, splits

    monkeypatch.setattr(runner, "audit_session", fake_audit)
    identity = {
        "experiment_config": runner.frozen_experiment_config(16),
        "runtime_environment_signature": {"python_version": "test"},
    }
    args = SimpleNamespace(output_dir=tmp_path)
    plan = runner.build_task_plan(args, identity)
    assert sum(fold_counts.values()) == 82
    assert len(plan) == 82 * 2 * 3 == 492
    assert set(plan["model"]) == set(MODEL_NAMES)
    assert set(plan["seed"]) == {0, 1, 2}
    assert plan["task_key"].nunique() == len(plan)


def _write_valid_task(path: Path, runner, expected: dict[str, object]) -> None:
    path.mkdir(parents=True)
    model_name = str(expected["model"])
    run_fp = "run-fingerprint"
    task_fp = runner.task_fingerprint(run_fp, expected)
    identity = {"session": "626", "model": model_name, "seed": 0, "fold": 1}
    pd.DataFrame(
        {
            **{key: [value] * 4 for key, value in identity.items()},
            "sample_index": range(4),
            "block_id": [f"block_{index}" for index in range(4)],
            "cycle": [0] * 4,
            "block_name": ["grating", "static", "dot", "stop_after_grating"],
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 0, 1, 1],
            "probability_0": [1.0, 1.0, 0.0, 0.0],
            "probability_1": [0.0, 0.0, 1.0, 1.0],
        }
    ).to_csv(path / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                **identity,
                "true_label": truth,
                "predicted_label": prediction,
                "count": 2 if truth == prediction else 0,
            }
            for truth in (0, 1)
            for prediction in (0, 1)
        ]
    ).to_csv(path / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        {
            **{key: [value] * 40 for key, value in identity.items()},
            "epoch": range(1, 41),
            "train_loss": np.linspace(0.7, 0.2, 40),
            "train_accuracy": np.linspace(0.5, 1.0, 40),
        }
    ).to_csv(path / "training_history.csv", index=False)
    config = architecture_config(model_name)
    config["parameter_breakdown"] = parameter_breakdown(build_model(model_name))
    (path / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "normalization_audit.json").write_text(
        json.dumps(
            {
                "session": "626",
                "method": model_name,
                "seed": 0,
                "fold": 1,
                "phase": "outer_train_fold_only",
                "target_used_for_stats": False,
            }
        ),
        encoding="utf-8",
    )
    shared = {
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": expected["config_fingerprint"],
        "runtime_environment_fingerprint": expected["runtime_environment_fingerprint"],
    }
    (path / "result.json").write_text(
        json.dumps(
            {
                **shared,
                **identity,
                "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
                "n_test_samples": 4,
                "balanced_accuracy": 1.0,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "parameter_count": EXPECTED_PARAMETER_COUNTS[model_name],
                "actual_batch_size": 16,
                "trained_epochs": 40,
            }
        ),
        encoding="utf-8",
    )
    (path / "COMPLETE.json").write_text(
        json.dumps(
            {
                **shared,
                "task_key": runner.task_key("626", model_name, 0, 1),
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_resume_validation_is_strict_and_mode_specific(
    tmp_path: Path, model_name: str
) -> None:
    runner = load_runner()
    expected: dict[str, object] = {
        "session": "626",
        "model": model_name,
        "seed": 0,
        "fold": 1,
        "n_test_samples": 4,
        "config_fingerprint": "config",
        "runtime_environment_fingerprint": "runtime",
        "batch_size": 16,
    }
    path = tmp_path / model_name
    _write_valid_task(path, runner, expected)
    assert runner.validate_completed_task(path, expected, "run-fingerprint") == (
        True,
        "validated",
    )
    result = json.loads((path / "result.json").read_text())
    result["model_implementation_version"] = "tampered"
    (path / "result.json").write_text(json.dumps(result))
    valid, reason = runner.validate_completed_task(path, expected, "run-fingerprint")
    assert valid is False
    assert "implementation version" in reason


def _comparison_frame(runner) -> pd.DataFrame:
    rows = []
    for session in runner.EXPECTED_SESSIONS:
        values = {
            SINGLE_SCALE_MODEL_NAME: 0.60,
            MODEL_NAME: 0.62,
            FORMAL_TEMPORAL_BASELINE_NAME: 0.61,
            "fcnn_meanpool": 0.60,
            "cnn_factorized_transformer": 0.59,
            "spatial_mamba": 0.58,
            "local_global_residual_mamba": 0.59,
        }
        rows.extend(
            {"session": session, "model": model, "mean_BA": value}
            for model, value in values.items()
        )
    return pd.DataFrame(rows)


def test_mechanistic_and_external_comparisons_are_separate() -> None:
    runner = load_runner()
    comparison = _comparison_frame(runner)
    mechanism = runner.pairwise_rows(
        comparison,
        ((MODEL_NAME, SINGLE_SCALE_MODEL_NAME),),
        "mechanistic_same_backbone",
    )
    external = runner.pairwise_rows(
        comparison,
        tuple((MODEL_NAME, baseline) for baseline in runner.EXTERNAL_BASELINES),
        "external_baseline",
    )
    assert len(mechanism) == 1
    assert len(external) == 5
    assert mechanism.iloc[0]["mean_delta_BA"] == pytest.approx(0.02)
    assert mechanism.iloc[0]["improved_sessions"] == 9
    assert set(external["baseline"]) == set(runner.EXTERNAL_BASELINES)


def test_decision_rule_is_pre_registered_and_stops() -> None:
    runner = load_runner()
    comparison = _comparison_frame(runner)
    paired = pd.concat(
        [
            runner.pairwise_rows(
                comparison,
                ((MODEL_NAME, SINGLE_SCALE_MODEL_NAME),),
                "mechanistic_same_backbone",
            ),
            runner.pairwise_rows(
                comparison,
                tuple((MODEL_NAME, baseline) for baseline in runner.EXTERNAL_BASELINES),
                "external_baseline",
            ),
        ],
        ignore_index=True,
    )
    overfit = pd.DataFrame(
        [
            {
                "session": session,
                "model": model,
                "seed": seed,
                "possible_severe_overfit": (
                    model in {"spatial_mamba", "local_global_residual_mamba"}
                    and session in {"626", "628"}
                ),
            }
            for session in runner.EXPECTED_SESSIONS
            for model in (
                MODEL_NAME,
                "cnn_factorized_transformer",
                "spatial_mamba",
                "local_global_residual_mamba",
            )
            for seed in runner.SEEDS
        ]
    )
    audit = runner.decision_rule_audit(comparison, overfit, paired)
    assert all(audit["checks"].values())
    assert audit["automatic_next_stage_started"] is False


def _formal_summary(
    path: Path, model: str, sessions: tuple[str, ...], mean_ba: float
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "session": session,
                "model": model,
                "mean_BA": mean_ba,
                "std_BA": 0.01,
                "mean_accuracy": mean_ba - 0.01,
            }
            for session in sessions
        ]
    ).to_csv(path, index=False)
    return path


def _completed_gated_v1_1(project_root: Path, status: str = "complete") -> Path:
    run_dir = project_root / "outputs/local_global_residual_mamba_v1_1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "RUN_COMPLETE.json").write_text(
        json.dumps({"status": status}) + "\n", encoding="utf-8"
    )
    return run_dir


def _patch_non_gated_external_sources(
    runner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_rows = []
    for session in runner.EXPECTED_SESSIONS:
        for method in (FORMAL_TEMPORAL_BASELINE_NAME, "fcnn_meanpool"):
            for seed in runner.SEEDS:
                clean_rows.append(
                    {
                        "session": session,
                        "method": method,
                        "seed": seed,
                        "balanced_accuracy": 0.6,
                        "accuracy": 0.59,
                        "source": "formal_clean4.csv",
                    }
                )
    monkeypatch.setattr(
        runner.prior_runner,
        "load_clean4_formal_seed_records",
        lambda args: pd.DataFrame(clean_rows),
    )
    mamba = _formal_summary(
        tmp_path / "mamba.csv", "spatial_mamba", runner.EXPECTED_SESSIONS, 0.6
    )
    transformer = _formal_summary(
        tmp_path / "transformer.csv",
        "cnn_factorized_transformer",
        runner.EXPECTED_SESSIONS,
        0.6,
    )
    monkeypatch.setattr(
        runner.prior_runner,
        "mamba_summary_candidates",
        lambda args: [mamba],
    )
    monkeypatch.setattr(
        runner.prior_runner,
        "transformer_summary_candidates",
        lambda args: [transformer],
    )


def test_external_loader_uses_completed_gated_v1_1_not_legacy_v1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load_runner()
    _patch_non_gated_external_sources(runner, tmp_path, monkeypatch)
    project_root = tmp_path / "project"
    old_path = _formal_summary(
        project_root
        / "outputs/local_global_residual_mamba_v1/proposed_summary.csv",
        "local_global_residual_mamba",
        runner.EXPECTED_SESSIONS,
        0.57,
    )
    gated_run_dir = _completed_gated_v1_1(project_root)
    formal_path = _formal_summary(
        gated_run_dir / "proposed_summary.csv",
        "local_global_residual_mamba",
        runner.EXPECTED_SESSIONS,
        0.59,
    )
    frame = runner.load_external_baselines(
        SimpleNamespace(project_root=project_root, benchmark_root=tmp_path)
    )
    assert len(frame) == 9 * 5
    assert set(frame["model"]) == set(runner.EXTERNAL_BASELINES)
    assert not frame["retrained_by_this_runner"].any()
    gated = frame[frame["model"].eq("local_global_residual_mamba")]
    assert np.allclose(gated["mean_BA"], 0.59)
    assert set(gated["source"]) == {str(formal_path)}
    assert str(old_path) not in set(gated["source"])
    assert set(gated["source_run_complete"]) == {
        str(gated_run_dir / "RUN_COMPLETE.json")
    }
    assert set(gated["source_run_status"]) == {"complete"}


@pytest.mark.parametrize("manifest_status", [None, "failed"])
def test_gated_v1_1_missing_or_invalid_completion_never_falls_back_to_legacy(
    tmp_path: Path, manifest_status: Optional[str]
) -> None:
    runner = load_runner()
    project_root = tmp_path / "project"
    _formal_summary(
        project_root
        / "outputs/local_global_residual_mamba_v1/proposed_summary.csv",
        "local_global_residual_mamba",
        runner.EXPECTED_SESSIONS,
        0.57,
    )
    formal_run_dir = project_root / "outputs/local_global_residual_mamba_v1_1"
    _formal_summary(
        formal_run_dir / "proposed_summary.csv",
        "local_global_residual_mamba",
        runner.EXPECTED_SESSIONS,
        0.59,
    )
    if manifest_status is not None:
        _completed_gated_v1_1(project_root, status=manifest_status)
    args = SimpleNamespace(project_root=project_root)
    error = FileNotFoundError if manifest_status is None else AssertionError
    with pytest.raises(error, match="RUN_COMPLETE|completion"):
        runner.validate_gated_mamba_formal_run(args)


def test_complex_overfit_uses_completed_gated_v1_1_provenance(
    tmp_path: Path
) -> None:
    runner = load_runner()
    project_root = tmp_path / "project"
    old_dir = project_root / "outputs/local_global_residual_mamba_v1"
    old_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"session": "626", "model": "local_global_residual_mamba", "seed": 0}]
    ).to_csv(old_dir / "overfitting_comparison.csv", index=False)

    formal_run_dir = _completed_gated_v1_1(project_root)
    mamba_rows = [
        {
            "session": session,
            "model": model,
            "seed": seed,
            "possible_severe_overfit": False,
        }
        for session in runner.EXPECTED_SESSIONS
        for model in ("spatial_mamba", "local_global_residual_mamba")
        for seed in runner.SEEDS
    ]
    formal_overfit = formal_run_dir / "overfitting_comparison.csv"
    pd.DataFrame(mamba_rows).to_csv(formal_overfit, index=False)
    transformer_path = (
        project_root
        / "outputs/transformer_visual_binary_v1/overfitting_summary.csv"
    )
    transformer_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "session": session,
                "model": "cnn_factorized_transformer",
                "seed": seed,
                "possible_severe_overfit": False,
            }
            for session in runner.EXPECTED_SESSIONS
            for seed in runner.SEEDS
        ]
    ).to_csv(transformer_path, index=False)

    frame = runner.load_complex_overfit(SimpleNamespace(project_root=project_root))
    gated_and_spatial = frame[
        frame["model"].isin(["spatial_mamba", "local_global_residual_mamba"])
    ]
    assert set(gated_and_spatial["source"]) == {str(formal_overfit)}
    assert set(gated_and_spatial["source_run_status"]) == {"complete"}
