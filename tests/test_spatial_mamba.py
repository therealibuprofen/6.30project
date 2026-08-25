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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import load_block_sequence_session, split_manifest
from ultrasound_decoding.multiframe.factorized_transformer import CNNFactorizedTransformer
from ultrasound_decoding.multiframe import spatial_mamba
from ultrasound_decoding.multiframe.spatial_mamba import (
    MAMBA_DEPENDENCY_MESSAGE,
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    TRANSFORMER_REFERENCE_PARAMETER_COUNT,
    BidirectionalSharedMambaLayer,
    SpatialMambaClassifier,
    SpatialMambaConfig,
    architecture_config,
    mamba_dependency_available,
    parameter_breakdown,
    transformer_reference_config,
)
from ultrasound_decoding.multiframe.training import normalize_blocks_train_fold_only_with_stats


def load_runner():
    path = PROJECT_DIR / "scripts" / "baselines" / "run_mamba_visual_binary.py"
    spec = importlib.util.spec_from_file_location("run_mamba_visual_binary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMamba(nn.Module):
    """Minimal deterministic mixer for testing scan wiring, not a Mamba substitute."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, d_model, bias=False)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.projection(x)


def test_project_import_and_config_do_not_require_optional_dependency() -> None:
    payload = architecture_config()
    assert payload["model"] == MODEL_NAME
    assert payload["model_implementation_version"] == MODEL_IMPLEMENTATION_VERSION
    assert payload["dependency"] == "official mamba_ssm.Mamba"
    assert payload["spatial_scan"] == "bidirectional shared-weight spatial scan"
    assert payload["config"]["d_model"] == 64
    assert payload["config"]["d_state"] == 16
    assert payload["config"]["d_conv"] == 4
    assert payload["config"]["expand"] == 2
    assert payload["config"]["spatial_mamba_layers"] == 2


def test_missing_dependency_error_is_explicit_when_unavailable() -> None:
    if mamba_dependency_available():
        pytest.skip("mamba_ssm is installed; missing-dependency branch is not applicable")
    with pytest.raises(RuntimeError, match="Mamba dependency is not installed") as exc:
        SpatialMambaClassifier()
    assert MAMBA_DEPENDENCY_MESSAGE in str(exc.value)
    assert "dedicated server Mamba environment" in str(exc.value)


def test_bidirectional_scan_uses_one_shared_mamba_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(spatial_mamba, "_OfficialMamba", FakeMamba)
    layer = BidirectionalSharedMambaLayer(SpatialMambaConfig(dropout=0.0))
    assert isinstance(layer.mamba, FakeMamba)
    assert sum(1 for module in layer.modules() if isinstance(module, FakeMamba)) == 1
    tokens = torch.randn(2, 256, 64)
    output = layer(tokens)
    assert output.shape == tokens.shape
    assert layer.mamba.calls == 2


def test_stem_temporal_and_classifier_are_direct_reviewed_module_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spatial_mamba, "_OfficialMamba", FakeMamba)
    config = SpatialMambaConfig()
    model = SpatialMambaClassifier(config)
    reference = CNNFactorizedTransformer(transformer_reference_config(config))
    assert type(model.stem) is type(reference.stem)
    assert repr(model.stem) == repr(reference.stem)
    assert repr(model.temporal_transformer) == repr(reference.temporal_transformer)
    assert repr(model.classifier) == repr(reference.classifier)
    assert model.temporal_position.shape == reference.temporal_position.shape == (1, 4, 64)
    assert model.spatial_position.shape == (1, 256, 64)


def test_fake_mamba_full_shapes_backward_and_parameter_breakdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spatial_mamba, "_OfficialMamba", FakeMamba)
    torch.manual_seed(0)
    model = SpatialMambaClassifier()
    x = torch.randn(2, 4, 128, 501)
    y = torch.tensor([0, 1], dtype=torch.long)
    logits, shapes = model.forward_with_shapes(x)
    assert shapes == {
        "input": (2, 4, 1, 128, 501),
        "cnn_output": (2, 4, 64, 8, 32),
        "spatial_tokens": (8, 256, 64),
        "spatial_mamba_output": (8, 256, 64),
        "temporal_transformer_input": (2, 4, 64),
        "temporal_transformer_output": (2, 4, 64),
        "pooled": (2, 64),
        "logits": (2, 2),
    }
    loss = nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    breakdown = parameter_breakdown(model)
    assert breakdown["total_parameter_count"] == sum(
        value for key, value in breakdown.items() if key != "total_parameter_count"
    )


@pytest.mark.skipif(
    not mamba_dependency_available() or not torch.cuda.is_available(),
    reason="official mamba_ssm CUDA dependency unavailable; Mamba-specific forward/backward SKIPPED",
)
def test_official_mamba_specific_forward_backward_when_dependency_available() -> None:
    model = SpatialMambaClassifier().cuda()
    x = torch.randn(1, 4, 128, 501, device="cuda")
    logits = model(x)
    loss = logits.square().mean()
    loss.backward()
    assert logits.shape == (1, 2)
    assert torch.isfinite(loss)


def test_train_only_normalization_is_independent_of_test_values() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(size=(3, 4, 5, 7)).astype(np.float32)
    test_a = rng.normal(size=(2, 4, 5, 7)).astype(np.float32)
    test_b = test_a + 10_000.0
    common = dict(
        session="710", task="binary", method=MODEL_NAME, seed=0, fold=1,
        train_cycles="1,2", test_cycles="3",
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


def test_runner_config_fingerprint_and_fixed_batch_policy() -> None:
    runner = load_runner()
    assert runner.ALLOWED_BATCH_SIZES == (16,)
    config = runner.frozen_experiment_config(16)
    assert config["training"]["batch_size"] == 16
    assert config["training"]["max_epochs"] == 40
    assert config["seeds"] == [0, 1, 2]
    base = {
        "experiment_config": config, "git_commit": "abc",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "model_source_sha256": "m", "runner_source_sha256": "r",
        "transitive_project_source_sha256": {"dependency.py": "one"},
    }
    changed = dict(base, transitive_project_source_sha256={"dependency.py": "two"})
    assert runner.fingerprint(base) != runner.fingerprint(changed)


def test_manifest_dtype_normalization_and_exact_sign_flip() -> None:
    runner = load_runner()
    base = pd.DataFrame(
        [{
            "session": "626", "task": "binary", "fold": 1,
            "train_cycles": "1,2", "test_cycles": "0", "n_train_blocks": 8,
            "n_test_blocks": 4, "train_class_counts": '{"no_stimulus": 4, "stimulus": 4}',
            "test_class_counts": '{"no_stimulus": 2, "stimulus": 2}',
        }]
    )
    inferred = base.copy()
    inferred["session"] = 626
    inferred["test_cycles"] = 0
    assert runner.canonical_manifest(base).equals(runner.canonical_manifest(inferred))
    inferred["n_test_blocks"] = 8
    assert not runner.canonical_manifest(base).equals(runner.canonical_manifest(inferred))
    assert runner.exact_two_sided_sign_flip(np.ones(3)) == pytest.approx(0.25)


def _write_valid_task_artifacts(path: Path, runner, expected: dict[str, object]) -> None:
    run_fingerprint = "run-fingerprint"
    task_fingerprint = runner.task_fingerprint(run_fingerprint, expected)
    predictions = pd.DataFrame(
        {
            "session": ["626"] * 4, "model": [MODEL_NAME] * 4,
            "seed": [0] * 4, "fold": [1] * 4, "sample_index": range(4),
            "block_id": [f"block_{i}" for i in range(4)], "cycle": [0] * 4,
            "block_name": ["grating", "static", "dot", "stop_after_grating"],
            "y_true": [0, 0, 1, 1], "y_pred": [0, 0, 1, 1],
            "probability_0": [1.0, 1.0, 0.0, 0.0],
            "probability_1": [0.0, 0.0, 1.0, 1.0],
        }
    )
    predictions.to_csv(path / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
                "true_label": truth, "predicted_label": pred,
                "count": 2 if truth == pred else 0,
            }
            for truth in (0, 1) for pred in (0, 1)
        ]
    ).to_csv(path / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        {
            "session": ["626"] * 40, "model": [MODEL_NAME] * 40,
            "seed": [0] * 40, "fold": [1] * 40, "epoch": range(1, 41),
            "train_loss": np.linspace(1.0, 0.1, 40),
            "train_accuracy": np.linspace(0.5, 1.0, 40),
        }
    ).to_csv(path / "training_history.csv", index=False)
    (path / "normalization_audit.json").write_text(
        json.dumps(
            {"session": "626", "method": MODEL_NAME, "seed": 0, "fold": 1,
             "phase": "outer_train_fold_only", "target_used_for_stats": False}
        )
    )
    breakdown = {
        "cnn_stem_parameters": 10, "spatial_mamba_parameters": 20,
        "temporal_transformer_parameters": 30, "classifier_parameters": 40,
        "total_parameter_count": 100,
    }
    model_config = architecture_config()
    model_config["parameter_breakdown"] = breakdown
    (path / "model_config.json").write_text(json.dumps(model_config))
    result = {
        "run_fingerprint": run_fingerprint, "task_fingerprint": task_fingerprint,
        "config_fingerprint": "config-fingerprint",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
        "n_test_samples": 4, "balanced_accuracy": 1.0, "accuracy": 1.0,
        "macro_f1": 1.0, "trained_epochs": 40, "actual_batch_size": 16,
        "parameter_count": 100, **breakdown,
    }
    (path / "result.json").write_text(json.dumps(result))
    (path / "COMPLETE.json").write_text(
        json.dumps(
            {
                "task_key": "626:spatial_mamba:0:1", "run_fingerprint": run_fingerprint,
                "task_fingerprint": task_fingerprint, "config_fingerprint": "config-fingerprint",
            }
        )
    )


def test_strict_completed_task_revalidation(tmp_path: Path) -> None:
    runner = load_runner()
    expected = {
        "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
        "n_test_samples": 4, "config_fingerprint": "config-fingerprint", "batch_size": 16,
    }
    _write_valid_task_artifacts(tmp_path, runner, expected)
    valid, reason = runner.validate_completed_task(tmp_path, expected, "run-fingerprint")
    assert valid is True, reason
    result = json.loads((tmp_path / "result.json").read_text())
    result["spatial_mamba_parameters"] += 1
    (tmp_path / "result.json").write_text(json.dumps(result))
    valid, reason = runner.validate_completed_task(tmp_path, expected, "run-fingerprint")
    assert valid is False
    assert "parameter field mismatch" in reason


def test_existing_baseline_merge_includes_formal_transformer_rows(tmp_path: Path) -> None:
    runner = load_runner()
    clean_path = tmp_path / "outputs" / "block_clean4_binary_all_models_9sessions_v1" / "aggregate"
    clean_path.mkdir(parents=True)
    old_models = [
        "pca_lda_flat4", "cpca_lda_flat4", "fcnn_meanpool",
        "cnn2d_meanpool", "cnn2d_lstm", "cnn2d_temporal1d",
    ]
    clean_rows = [
        {"session": session, "task": "binary", "method": model, "seed": seed,
         "balanced_accuracy": 0.5, "accuracy": 0.5}
        for session in runner.EXPECTED_SESSIONS for model in old_models for seed in (0, 1, 2)
    ]
    pd.DataFrame(clean_rows).to_csv(clean_path / "multiframe_all_models_master_long.csv", index=False)
    sbind_path = tmp_path / "outputs" / "sbind_visual_binary_v1"
    sbind_path.mkdir(parents=True)
    pd.DataFrame(
        [{"session": session, "model": model, "mean_BA": 0.5, "std_BA": 0.0, "mean_accuracy": 0.5}
         for session in runner.EXPECTED_SESSIONS for model in ("sbind_noatt", "sbind")]
    ).to_csv(sbind_path / "sbind_summary.csv", index=False)
    transformer_path = tmp_path / "outputs" / "transformer_visual_binary_v1"
    transformer_path.mkdir(parents=True)
    pd.DataFrame(
        [{"session": session, "model": "cnn_factorized_transformer", "mean_BA": 0.59,
          "std_BA": 0.01, "mean_accuracy": 0.59} for session in runner.EXPECTED_SESSIONS]
    ).to_csv(transformer_path / "transformer_summary.csv", index=False)
    args = SimpleNamespace(project_root=tmp_path, benchmark_root=tmp_path / "missing")
    merged = runner.load_existing_baselines(args)
    assert len(merged) == len(runner.EXPECTED_SESSIONS) * 9
    assert set(merged["model"]) == set(runner.EXISTING_BASELINE_DISPLAY) - {MODEL_NAME}


def test_transformer_reference_parameter_count_is_audited() -> None:
    reference = CNNFactorizedTransformer()
    observed = sum(parameter.numel() for parameter in reference.parameters() if parameter.requires_grad)
    assert observed == TRANSFORMER_REFERENCE_PARAMETER_COUNT == 127_010


def test_real_clean4_shape_and_formal_fold_identity_when_data_present() -> None:
    data_path = PROJECT_DIR / "processed_data" / "block_sequences_v1" / "session_710_blocks.h5"
    candidates = [
        PROJECT_DIR / "outputs" / "block_clean4_binary_all_models_9sessions_v1"
        / "session_710" / "split_manifest.csv",
        PROJECT_DIR / "results" / "runs" / "multiframe" / "block_clean4_binary_v1"
        / "session_710" / "split_manifest.csv",
    ]
    manifest_path = next((path for path in candidates if path.exists()), None)
    if not data_path.exists() or manifest_path is None:
        pytest.skip("real clean4 session 710 data or formal fold manifest unavailable")
    runner = load_runner()
    data = load_block_sequence_session(PROJECT_DIR, "710", "binary")
    splits = grouped_cv_splits(data.groups, max_folds=10)
    regenerated = split_manifest("710", "binary", data.y, data.groups, splits=splits)
    historical = pd.read_csv(manifest_path)
    assert data.X.shape[1:] == (4, 128, 501)
    assert runner.canonical_manifest(regenerated).equals(runner.canonical_manifest(historical))
    for train_idx, test_idx in splits:
        assert not np.intersect1d(data.groups[train_idx], data.groups[test_idx]).size
