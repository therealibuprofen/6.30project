from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.multiframe import spatial_mamba
from ultrasound_decoding.multiframe.local_global_residual_mamba import (
    EXPECTED_FORMAL_PARAMETER_COUNT,
    INITIAL_GATE_LOGIT,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    LocalGlobalResidualMambaClassifier,
    LocalGlobalResidualMambaConfig,
    _train_epochs_with_gate_history,
    architecture_config,
    parameter_breakdown,
    spatial_mamba_config,
)
from ultrasound_decoding.multiframe.models import CNN2DTemporal1D
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    normalize_blocks_train_fold_only_with_stats,
)


def load_runner():
    path = PROJECT_DIR / "scripts" / "baselines" / "run_local_global_residual_mamba.py"
    spec = importlib.util.spec_from_file_location("run_local_global_residual_mamba", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMamba(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.projection(x)


def make_model(monkeypatch: pytest.MonkeyPatch) -> LocalGlobalResidualMambaClassifier:
    monkeypatch.setattr(spatial_mamba, "_OfficialMamba", FakeMamba)
    return LocalGlobalResidualMambaClassifier()


def test_architecture_is_single_frozen_candidate() -> None:
    audit = architecture_config()
    assert audit["model"] == MODEL_NAME
    assert MODEL_IMPLEMENTATION_VERSION == "local_global_residual_mamba_v1.0.0"
    assert audit["model_implementation_version"] == MODEL_IMPLEMENTATION_VERSION
    assert audit["fusion"] == "F_local + sigmoid(gate_logit) * (F_global - F_local)"
    assert audit["gate_scope"].startswith("one global trainable scalar")
    assert audit["initial_gate_logit"] == -2.0
    assert audit["initial_alpha"] == pytest.approx(0.11920292)
    assert audit["temporal_layers_after_first_conv_directly_reused"] is True
    assert audit["classifier_directly_reused"] is True
    assert audit["expected_formal_parameter_count_mamba_ssm_2_2_2"] == 116_579
    assert EXPECTED_FORMAL_PARAMETER_COUNT == 116_579
    assert EXPECTED_FORMAL_PARAMETER_COUNT == 23520 + 2560 + 65536 + 1 + 24832 + 130
    assert audit["temporal_transformer_present"] is False
    assert audit["multiscale_present"] is False
    config = LocalGlobalResidualMambaConfig()
    assert config.d_model == 64
    assert config.d_state == 16
    assert config.d_conv == 4
    assert config.expand == 2
    assert config.spatial_mamba_layers == 2


def test_spatial_config_exactly_matches_frozen_mamba_v11() -> None:
    config = spatial_mamba_config(LocalGlobalResidualMambaConfig())
    assert config.d_model == 64
    assert config.d_state == 16
    assert config.d_conv == 4
    assert config.expand == 2
    assert config.spatial_mamba_layers == 2
    assert config.stem_channels == (16, 32, 64)
    assert (config.pooled_height, config.pooled_width) == (8, 32)


def test_exact_2d_positions_single_gate_and_temporal_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = make_model(monkeypatch)
    assert model.spatial_row_position.shape == (1, 8, 1, 64)
    assert model.spatial_column_position.shape == (1, 1, 32, 64)
    assert not hasattr(model, "spatial_position")
    assert model.gate_logit.shape == torch.Size([])
    assert float(model.alpha.detach()) == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())
    gate_names = [name for name, _ in model.named_parameters() if "gate" in name]
    assert gate_names == ["gate_logit"]

    reference = CNN2DTemporal1D(n_classes=2, dropout=0.25, norm="batchnorm")
    first = model.temporal_conv[0]
    assert isinstance(first, nn.Conv1d)
    assert (first.in_channels, first.out_channels, first.kernel_size, first.padding) == (
        64,
        64,
        (3,),
        (1,),
    )
    assert repr(model.temporal_conv[1:]) == repr(reference.temporal_conv[1:])
    assert repr(model.classifier) == repr(reference.classifier)


def test_fusion_formula_shapes_backward_and_parameter_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(0)
    model = make_model(monkeypatch)
    x = torch.randn(2, 4, 128, 501)
    local, global_map, fused = model.spatial_feature_maps(x)
    assert local.shape == global_map.shape == fused.shape == (2, 4, 64, 8, 32)
    torch.testing.assert_close(fused, local + model.alpha * (global_map - local))
    logits, shapes = model.forward_with_shapes(x)
    assert shapes == {
        "input": (2, 4, 1, 128, 501),
        "local_map": (2, 4, 64, 8, 32),
        "global_map": (2, 4, 64, 8, 32),
        "fused_map": (2, 4, 64, 8, 32),
        "frame_features": (2, 4, 64),
        "temporal_input": (2, 64, 4),
        "temporal_features": (2, 64),
        "logits": (2, 2),
    }
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.gate_logit.grad is not None
    assert torch.isfinite(model.gate_logit.grad)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    breakdown = parameter_breakdown(model)
    assert breakdown["gate_parameters"] == 1
    assert breakdown["spatial_position_parameters"] == (8 + 32) * 64
    assert breakdown["total_parameter_count"] == sum(
        value for key, value in breakdown.items() if key != "total_parameter_count"
    )


class TinyGateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_logit = nn.Parameter(torch.tensor(INITIAL_GATE_LOGIT))
        self.linear = nn.Linear(3, 2)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.alpha * self.linear(x)


def test_training_loop_records_every_epoch_alpha() -> None:
    torch.manual_seed(0)
    model = TinyGateModel()
    tensor = torch.randn(6, 3)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    history, initial = _train_epochs_with_gate_history(
        model,  # type: ignore[arg-type]
        tensor,
        labels,
        config=DeepTrainingConfig(max_epochs=3, batch_size=2),
        seed=0,
        device=torch.device("cpu"),
        batch_size_reference=6,
        num_workers=0,
    )
    assert initial == pytest.approx(torch.sigmoid(torch.tensor(-2.0)).item())
    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert all(0.0 < row["alpha"] < 1.0 for row in history)


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
    assert audit_a["target_used_for_stats"] is False
    assert audit_b["target_used_for_stats"] is False


def test_runner_freezes_protocol_fingerprint_and_review_lock(tmp_path: Path) -> None:
    runner = load_runner()
    config = runner.frozen_experiment_config(16)
    assert config["sessions"] == ["626", "628", "708", "709", "710", "807", "813", "817", "822"]
    assert config["seeds"] == [0, 1, 2]
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
    base = {
        "experiment_config": config,
        "runtime_environment_signature": {"torch_version": "2.1.2+cu118"},
        "git_commit": "abc",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "model_source_sha256": "m",
        "runner_source_sha256": "r",
        "transitive_project_source_sha256": {"dependency.py": "one"},
    }
    changed = dict(
        base, runtime_environment_signature={"torch_version": "different"}
    )
    assert runner.fingerprint(base) != runner.fingerprint(changed)
    signature = runner.runtime_environment_signature()
    assert "h5py_version" in signature
    assert isinstance(signature["h5py_version"], str) and signature["h5py_version"]
    args = SimpleNamespace(review_approved=False)
    with pytest.raises(RuntimeError, match="code review is approved"):
        runner.run_full(args, base)


def test_paired_comparisons_include_strong_and_weak_deltas() -> None:
    runner = load_runner()
    rows = []
    for index, session in enumerate(runner.EXPECTED_SESSIONS):
        proposed = 0.50 + index * 0.01
        for model in (*runner.COMPARISON_BASELINES, MODEL_NAME):
            rows.append(
                {
                    "session": session,
                    "model": model,
                    "mean_BA": proposed if model == MODEL_NAME else proposed - 0.01,
                }
            )
    paired = runner.paired_comparison_rows(pd.DataFrame(rows))
    assert set(paired["baseline"]) == set(runner.COMPARISON_BASELINES)
    assert np.allclose(paired["mean_delta_BA"], 0.01)
    assert np.allclose(paired["strong_session_mean_delta_BA"], 0.01)
    assert np.allclose(paired["weak_session_mean_delta_BA"], 0.01)
    assert (paired["improved_sessions"] == 9).all()


def test_decision_rule_is_pre_registered_and_does_not_start_next_stage() -> None:
    runner = load_runner()
    rows = []
    for session in runner.EXPECTED_SESSIONS:
        temporal = 0.60
        mamba = 0.55 if session in runner.STRONG_SESSIONS else 0.59
        proposed = 0.63
        values = {
            runner.LOCAL_BASELINE_NAME: temporal,
            "spatial_mamba": mamba,
            "cnn_factorized_transformer": 0.58,
            "fcnn_meanpool": 0.59,
            MODEL_NAME: proposed,
        }
        for model, value in values.items():
            rows.append({"session": session, "model": model, "mean_BA": value})
    comparison = pd.DataFrame(rows)
    paired = runner.paired_comparison_rows(comparison)
    overfit_rows = []
    for session in runner.EXPECTED_SESSIONS:
        for model in ("spatial_mamba", MODEL_NAME):
            for seed in runner.SEEDS:
                overfit_rows.append(
                    {
                        "session": session,
                        "model": model,
                        "seed": seed,
                        "possible_severe_overfit": (
                            model == "spatial_mamba" and session in {"626", "628"}
                        ),
                    }
                )
    audit = runner.decision_rule_audit(
        comparison, paired, pd.DataFrame(overfit_rows)
    )
    assert all(audit["checks"].values())
    assert audit["decision"].startswith("supports_continue_mamba_route")
    assert audit["automatic_next_stage_started"] is False


def _write_valid_task(path: Path, runner, expected: dict[str, object]) -> None:
    path.mkdir(parents=True)
    run_fp = "run-fingerprint"
    task_fp = runner.task_fingerprint(run_fp, expected)
    initial = float(torch.sigmoid(torch.tensor(-2.0)).item())
    alphas = np.linspace(initial + 0.0001, initial + 0.004, 40)
    predictions = pd.DataFrame(
        {
            "session": ["626"] * 4,
            "model": [MODEL_NAME] * 4,
            "seed": [0] * 4,
            "fold": [1] * 4,
            "sample_index": range(4),
            "block_id": [f"block_{index}" for index in range(4)],
            "cycle": [0] * 4,
            "block_name": ["grating", "static", "dot", "stop_after_grating"],
            "y_true": [0, 0, 1, 1],
            "y_pred": [0, 0, 1, 1],
            "probability_0": [1.0, 1.0, 0.0, 0.0],
            "probability_1": [0.0, 0.0, 1.0, 1.0],
        }
    )
    predictions.to_csv(path / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "session": "626",
                "model": MODEL_NAME,
                "seed": 0,
                "fold": 1,
                "true_label": truth,
                "predicted_label": pred,
                "count": 2 if truth == pred else 0,
            }
            for truth in (0, 1)
            for pred in (0, 1)
        ]
    ).to_csv(path / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        {
            "session": ["626"] * 40,
            "model": [MODEL_NAME] * 40,
            "seed": [0] * 40,
            "fold": [1] * 40,
            "epoch": range(1, 41),
            "train_loss": np.linspace(0.7, 0.2, 40),
            "train_accuracy": np.linspace(0.5, 1.0, 40),
            "alpha": alphas,
        }
    ).to_csv(path / "training_history.csv", index=False)
    breakdown = {
        "cnn_stem_parameters": 23520,
        "spatial_position_parameters": 2560,
        "spatial_mamba_parameters": 65536,
        "gate_parameters": 1,
        "temporal_1d_parameters": 24832,
        "classifier_parameters": 130,
        "total_parameter_count": 116579,
    }
    model_config = architecture_config()
    model_config["parameter_breakdown"] = breakdown
    (path / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
    normalization = {
        "session": "626",
        "method": MODEL_NAME,
        "seed": 0,
        "fold": 1,
        "phase": "outer_train_fold_only",
        "target_used_for_stats": False,
    }
    (path / "normalization_audit.json").write_text(
        json.dumps(normalization), encoding="utf-8"
    )
    gate = {
        "session": "626",
        "model": MODEL_NAME,
        "seed": 0,
        "fold": 1,
        "initial_alpha": initial,
        "final_alpha": float(alphas[-1]),
        "mean_alpha_last5_epochs": float(alphas[-5:].mean()),
    }
    (path / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    shared = {
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": expected["config_fingerprint"],
        "runtime_environment_fingerprint": expected["runtime_environment_fingerprint"],
    }
    result = {
        **shared,
        "session": "626",
        "model": MODEL_NAME,
        "seed": 0,
        "fold": 1,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "n_test_samples": 4,
        "balanced_accuracy": 1.0,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "parameter_count": 116579,
        **breakdown,
        "actual_batch_size": 16,
        "trained_epochs": 40,
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    complete = {**shared, "task_key": "626:local_global_residual_mamba:0:1"}
    (path / "COMPLETE.json").write_text(json.dumps(complete), encoding="utf-8")


def test_resume_validation_rejects_gate_history_tampering(tmp_path: Path) -> None:
    runner = load_runner()
    expected: dict[str, object] = {
        "session": "626",
        "model": MODEL_NAME,
        "seed": 0,
        "fold": 1,
        "n_test_samples": 4,
        "config_fingerprint": "config",
        "runtime_environment_fingerprint": "runtime",
        "batch_size": 16,
    }
    task = tmp_path / "task"
    _write_valid_task(task, runner, expected)
    assert runner.validate_completed_task(task, expected, "run-fingerprint") == (
        True,
        "validated",
    )
    gate = json.loads((task / "gate.json").read_text())
    gate["final_alpha"] += 0.1
    (task / "gate.json").write_text(json.dumps(gate))
    valid, reason = runner.validate_completed_task(task, expected, "run-fingerprint")
    assert valid is False
    assert "final alpha" in reason


def test_existing_formal_baselines_are_read_not_retrained() -> None:
    runner = load_runner()
    args = SimpleNamespace(
        project_root=PROJECT_DIR,
        benchmark_root=(
            PROJECT_DIR / "results" / "runs" / "multiframe" / "block_clean4_binary_v1"
        ),
    )
    try:
        frame = runner.load_existing_comparison_baselines(args)
    except FileNotFoundError as exc:
        pytest.skip(f"formal result tables intentionally unavailable: {exc}")
    assert len(frame) == 9 * 4
    assert set(frame["model"]) == set(runner.COMPARISON_BASELINES)
    assert not frame["retrained_by_this_runner"].any()
