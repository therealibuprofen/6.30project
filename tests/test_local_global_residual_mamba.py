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
    EXPECTED_FORMAL_PARAMETER_COUNTS,
    FUSION_MODE_BY_MODEL_NAME,
    GLOBAL_ONLY_MODEL_NAME,
    INITIAL_GATE_LOGIT,
    LOCAL_ONLY_MODEL_NAME,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    LocalGlobalMambaTemporal1DClassifier,
    LocalGlobalResidualMambaClassifier,
    LocalGlobalResidualMambaConfig,
    _train_epochs_with_control_history,
    architecture_config,
    parameter_breakdown,
    spatial_mamba_config,
)
from ultrasound_decoding.multiframe.models import (
    CNN2DTemporal1D,
    SmallCNNFrameEncoder,
)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def patch_mamba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spatial_mamba, "_OfficialMamba", FakeMamba)


def test_architecture_version_modes_and_frozen_mamba_config() -> None:
    assert MODEL_IMPLEMENTATION_VERSION == "local_global_residual_mamba_v1.1.0"
    assert EXPECTED_FORMAL_PARAMETER_COUNT == 116_579
    assert EXPECTED_FORMAL_PARAMETER_COUNTS == {
        LOCAL_ONLY_MODEL_NAME: 48_482,
        GLOBAL_ONLY_MODEL_NAME: 116_578,
        MODEL_NAME: 116_579,
    }
    assert EXPECTED_FORMAL_PARAMETER_COUNT == 23520 + 2560 + 65536 + 1 + 24832 + 130
    assert set(FUSION_MODE_BY_MODEL_NAME) == {
        LOCAL_ONLY_MODEL_NAME,
        GLOBAL_ONLY_MODEL_NAME,
        MODEL_NAME,
    }
    for model_name, fusion_mode in FUSION_MODE_BY_MODEL_NAME.items():
        audit = architecture_config(fusion_mode=fusion_mode)
        assert audit["model"] == model_name
        assert audit["fusion_mode"] == fusion_mode
        assert audit["model_implementation_version"] == MODEL_IMPLEMENTATION_VERSION
        assert audit["expected_formal_parameter_count_mamba_ssm_2_2_2"] == (
            EXPECTED_FORMAL_PARAMETER_COUNTS[model_name]
        )
        assert audit["temporal_transformer_present"] is False
        assert audit["multiscale_present"] is False

    config = spatial_mamba_config(LocalGlobalResidualMambaConfig())
    assert (config.d_model, config.d_state, config.d_conv, config.expand) == (64, 16, 4, 2)
    assert config.spatial_mamba_layers == 2
    assert config.stem_channels == (16, 32, 64)
    assert (config.pooled_height, config.pooled_width) == (8, 32)


def test_old_temporal1d_fact_is_512_not_2048() -> None:
    assert SmallCNNFrameEncoder.feature_dim == 16 * 4 * 8 == 512
    old_formal = CNN2DTemporal1D(n_classes=2, dropout=0.25, norm="batchnorm")
    first = old_formal.temporal_conv[0]
    assert isinstance(first, nn.Conv1d)
    assert (first.in_channels, first.out_channels, first.kernel_size, first.padding) == (
        512,
        64,
        (3,),
        (1,),
    )
    assert "512->64" in architecture_config()["temporal_head"]
    assert "2048" not in json.dumps(architecture_config())


def test_three_modes_share_one_skeleton_and_only_expected_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_mamba(monkeypatch)
    local = LocalGlobalMambaTemporal1DClassifier("local_only")
    global_only = LocalGlobalMambaTemporal1DClassifier("global_only")
    gated = LocalGlobalMambaTemporal1DClassifier("gated_local_global")

    assert repr(local.stem) == repr(global_only.stem) == repr(gated.stem)
    assert repr(local.temporal_conv) == repr(global_only.temporal_conv) == repr(gated.temporal_conv)
    assert repr(local.classifier) == repr(global_only.classifier) == repr(gated.classifier)
    assert not hasattr(local, "spatial_mamba")
    assert not hasattr(local, "spatial_row_position")
    assert not hasattr(local, "gate_logit")
    assert hasattr(global_only, "spatial_mamba")
    assert not hasattr(global_only, "gate_logit")
    assert repr(global_only.spatial_mamba) == repr(gated.spatial_mamba)
    assert gated.gate_logit.shape == torch.Size([])
    assert [name for name, _ in gated.named_parameters() if "gate" in name] == [
        "gate_logit"
    ]
    assert float(local.effective_alpha) == 0.0
    assert float(global_only.effective_alpha) == 1.0
    assert float(gated.effective_alpha.detach()) == pytest.approx(
        torch.sigmoid(torch.tensor(-2.0)).item()
    )


def test_controlled_representations_shapes_and_gated_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_mamba(monkeypatch)
    x = torch.randn(2, 4, 128, 501)
    local_model = LocalGlobalMambaTemporal1DClassifier("local_only")
    global_model = LocalGlobalMambaTemporal1DClassifier("global_only")
    gated_model = LocalGlobalResidualMambaClassifier()

    local_map, no_global, local_selected = local_model.spatial_feature_maps(x)
    assert no_global is None
    torch.testing.assert_close(local_selected, local_map)
    global_local, global_map, global_selected = global_model.spatial_feature_maps(x)
    assert global_map is not None
    torch.testing.assert_close(global_selected, global_map)
    gated_local, gated_global, gated_selected = gated_model.spatial_feature_maps(x)
    assert gated_global is not None
    torch.testing.assert_close(
        gated_selected,
        gated_local + gated_model.alpha * (gated_global - gated_local),
    )
    assert local_map.shape == global_local.shape == gated_local.shape == (
        2,
        4,
        64,
        8,
        32,
    )
    for model in (local_model, global_model, gated_model):
        logits, shapes = model.forward_with_shapes(x)
        assert logits.shape == (2, 2)
        assert shapes["frame_features"] == (2, 4, 64)
        assert shapes["temporal_input"] == (2, 64, 4)
        assert shapes["temporal_features"] == (2, 64)


def test_positions_temporal_reuse_and_parameter_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_mamba(monkeypatch)
    reference = CNN2DTemporal1D(n_classes=2, dropout=0.25, norm="batchnorm")
    models = {
        name: LocalGlobalMambaTemporal1DClassifier(mode)
        for name, mode in FUSION_MODE_BY_MODEL_NAME.items()
    }
    for model in models.values():
        first = model.temporal_conv[0]
        assert isinstance(first, nn.Conv1d)
        assert (first.in_channels, first.out_channels) == (64, 64)
        assert repr(model.temporal_conv[1:]) == repr(reference.temporal_conv[1:])
        assert repr(model.classifier) == repr(reference.classifier)
    for name in (GLOBAL_ONLY_MODEL_NAME, MODEL_NAME):
        model = models[name]
        assert model.spatial_row_position.shape == (1, 8, 1, 64)
        assert model.spatial_column_position.shape == (1, 1, 32, 64)
        assert not hasattr(model, "spatial_position")
    assert parameter_breakdown(models[LOCAL_ONLY_MODEL_NAME])["spatial_mamba_parameters"] == 0
    assert parameter_breakdown(models[GLOBAL_ONLY_MODEL_NAME])["gate_parameters"] == 0
    assert parameter_breakdown(models[MODEL_NAME])["gate_parameters"] == 1


class TinyControlModel(nn.Module):
    def __init__(self, fusion_mode: str) -> None:
        super().__init__()
        self.fusion_mode = fusion_mode
        if fusion_mode == "gated_local_global":
            self.gate_logit = nn.Parameter(torch.tensor(INITIAL_GATE_LOGIT))
        self.linear = nn.Linear(3, 2)

    @property
    def alpha_is_trainable(self) -> bool:
        return self.fusion_mode == "gated_local_global"

    @property
    def effective_alpha(self) -> torch.Tensor:
        if self.fusion_mode == "local_only":
            return self.linear.weight.new_tensor(0.0)
        if self.fusion_mode == "global_only":
            return self.linear.weight.new_tensor(1.0)
        return torch.sigmoid(self.gate_logit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * (1.0 + self.effective_alpha)


def test_history_records_alpha_only_for_trainable_gate() -> None:
    tensor = torch.randn(6, 3)
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    for mode in ("local_only", "global_only", "gated_local_global"):
        history, audit = _train_epochs_with_control_history(
            TinyControlModel(mode),  # type: ignore[arg-type]
            tensor,
            labels,
            config=DeepTrainingConfig(max_epochs=3, batch_size=2),
            seed=0,
            device=torch.device("cpu"),
            batch_size_reference=6,
            num_workers=0,
        )
        assert [row["epoch"] for row in history] == [1, 2, 3]
        if mode == "gated_local_global":
            assert all(0.0 < row["alpha"] < 1.0 for row in history)
            assert audit["alpha_is_trainable"] is True
            assert {"initial_alpha", "final_alpha", "mean_alpha_last5_epochs"}.issubset(audit)
        else:
            assert all("alpha" not in row for row in history)
            assert audit == {
                "fusion_mode": mode,
                "alpha_is_trainable": False,
                "effective_alpha": 0.0 if mode == "local_only" else 1.0,
            }


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


def test_runner_freezes_three_model_protocol_and_review_lock() -> None:
    runner = load_runner()
    config = runner.frozen_experiment_config(16)
    assert config["sessions"] == list(runner.EXPECTED_SESSIONS)
    assert config["seeds"] == [0, 1, 2]
    assert config["mechanistic_models"] == list(runner.MECHANISTIC_MODELS)
    assert config["external_baselines_reused"] == list(runner.COMPARISON_BASELINES)
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
    changed = dict(base, runtime_environment_signature={"torch_version": "different"})
    assert runner.fingerprint(base) != runner.fingerprint(changed)
    assert "h5py_version" in runner.runtime_environment_signature()
    with pytest.raises(RuntimeError, match="code review is approved"):
        runner.run_full(SimpleNamespace(review_approved=False), base)


def _synthetic_comparison(runner) -> pd.DataFrame:
    rows = []
    for session in runner.EXPECTED_SESSIONS:
        values = {
            runner.LOCAL_ONLY_MODEL_NAME: 0.60,
            runner.GLOBAL_ONLY_MODEL_NAME: 0.61,
            runner.MODEL_NAME: 0.63,
            runner.LOCAL_BASELINE_NAME: 0.60,
            "spatial_mamba": 0.55 if session in runner.STRONG_SESSIONS else 0.59,
            "cnn_factorized_transformer": 0.58,
            "fcnn_meanpool": 0.59,
        }
        rows.extend(
            {"session": session, "model": model, "mean_BA": value}
            for model, value in values.items()
        )
    return pd.DataFrame(rows)


def test_mechanistic_and_external_sign_flip_tables_are_separate() -> None:
    runner = load_runner()
    comparison = _synthetic_comparison(runner)
    mechanistic = runner.mechanistic_comparison_rows(comparison)
    external = runner.paired_comparison_rows(comparison)
    assert len(mechanistic) == 3
    assert set(mechanistic["comparison_type"]) == {"mechanistic_same_backbone"}
    assert set(external["baseline"]) == set(runner.COMPARISON_BASELINES)
    assert set(external["comparison_type"]) == {"external_baseline"}
    required = {
        "mean_delta_BA",
        "median_delta_BA",
        "improved_sessions",
        "tied_sessions",
        "worsened_sessions",
        "exact_two_sided_sign_flip_p",
        "strong_session_mean_delta_BA",
        "weak_session_mean_delta_BA",
    }
    assert required.issubset(mechanistic.columns)


def test_decision_rule_uses_mechanistic_controls_and_stops() -> None:
    runner = load_runner()
    comparison = _synthetic_comparison(runner)
    paired = pd.concat(
        [
            runner.mechanistic_comparison_rows(comparison),
            runner.paired_comparison_rows(comparison),
        ],
        ignore_index=True,
    )
    overfit_rows = []
    for session in runner.EXPECTED_SESSIONS:
        for model in (*runner.MECHANISTIC_MODELS, "spatial_mamba"):
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
    gates = pd.DataFrame(
        [
            {
                "session": session,
                "model": MODEL_NAME,
                "seed": seed,
                "fold": 1,
                "final_alpha": 0.2,
            }
            for session in runner.EXPECTED_SESSIONS
            for seed in runner.SEEDS
        ]
    )
    audit = runner.decision_rule_audit(
        comparison, paired, pd.DataFrame(overfit_rows), gates
    )
    assert all(audit["checks"].values())
    assert audit["decision"].startswith("supports_continue_mamba_route")
    assert audit["automatic_next_stage_started"] is False


def _write_valid_task(
    path: Path, runner, expected: dict[str, object], model_name: str
) -> None:
    path.mkdir(parents=True)
    run_fp = "run-fingerprint"
    task_fp = runner.task_fingerprint(run_fp, expected)
    mode = FUSION_MODE_BY_MODEL_NAME[model_name]
    gated = model_name == MODEL_NAME
    initial = float(torch.sigmoid(torch.tensor(-2.0)).item())
    alphas = np.linspace(initial + 0.0001, initial + 0.004, 40)
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
                "predicted_label": pred,
                "count": 2 if truth == pred else 0,
            }
            for truth in (0, 1)
            for pred in (0, 1)
        ]
    ).to_csv(path / "confusion_matrix.csv", index=False)
    history_data: dict[str, object] = {
        **{key: [value] * 40 for key, value in identity.items()},
        "epoch": range(1, 41),
        "train_loss": np.linspace(0.7, 0.2, 40),
        "train_accuracy": np.linspace(0.5, 1.0, 40),
    }
    if gated:
        history_data["alpha"] = alphas
    pd.DataFrame(history_data).to_csv(path / "training_history.csv", index=False)
    breakdowns = {
        LOCAL_ONLY_MODEL_NAME: (0, 0, 0),
        GLOBAL_ONLY_MODEL_NAME: (2560, 65536, 0),
        MODEL_NAME: (2560, 65536, 1),
    }
    position, mamba, gate_count = breakdowns[model_name]
    breakdown = {
        "cnn_stem_parameters": 23520,
        "spatial_position_parameters": position,
        "spatial_mamba_parameters": mamba,
        "gate_parameters": gate_count,
        "temporal_1d_parameters": 24832,
        "classifier_parameters": 130,
        "total_parameter_count": EXPECTED_FORMAL_PARAMETER_COUNTS[model_name],
    }
    model_config = architecture_config(fusion_mode=mode)
    model_config["parameter_breakdown"] = breakdown
    (path / "model_config.json").write_text(json.dumps(model_config), encoding="utf-8")
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
    effective = float(alphas[-1]) if gated else (0.0 if mode == "local_only" else 1.0)
    control = {
        **identity,
        "fusion_mode": mode,
        "alpha_is_trainable": gated,
        "effective_alpha": effective,
    }
    if gated:
        control.update(
            {
                "initial_alpha": initial,
                "final_alpha": float(alphas[-1]),
                "mean_alpha_last5_epochs": float(alphas[-5:].mean()),
            }
        )
    (path / "gate.json").write_text(json.dumps(control), encoding="utf-8")
    shared = {
        "run_fingerprint": run_fp,
        "task_fingerprint": task_fp,
        "config_fingerprint": expected["config_fingerprint"],
        "runtime_environment_fingerprint": expected["runtime_environment_fingerprint"],
    }
    result = {
        **shared,
        **identity,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "n_test_samples": 4,
        "balanced_accuracy": 1.0,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "parameter_count": EXPECTED_FORMAL_PARAMETER_COUNTS[model_name],
        **breakdown,
        "actual_batch_size": 16,
        "trained_epochs": 40,
    }
    (path / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (path / "COMPLETE.json").write_text(
        json.dumps({**shared, "task_key": runner.task_key("626", model_name, 0, 1)}),
        encoding="utf-8",
    )


@pytest.mark.parametrize("model_name", [LOCAL_ONLY_MODEL_NAME, GLOBAL_ONLY_MODEL_NAME, MODEL_NAME])
def test_resume_validation_is_mode_specific(tmp_path: Path, model_name: str) -> None:
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
    task = tmp_path / model_name
    _write_valid_task(task, runner, expected, model_name)
    assert runner.validate_completed_task(task, expected, "run-fingerprint") == (
        True,
        "validated",
    )
    if model_name == MODEL_NAME:
        gate = json.loads((task / "gate.json").read_text())
        gate["final_alpha"] += 0.1
        (task / "gate.json").write_text(json.dumps(gate))
        valid, reason = runner.validate_completed_task(task, expected, "run-fingerprint")
        assert valid is False and "final alpha" in reason


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
