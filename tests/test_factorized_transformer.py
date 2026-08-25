from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import load_block_sequence_session, split_manifest
from ultrasound_decoding.multiframe.factorized_transformer import (
    MODEL_IMPLEMENTATION_VERSION,
    MODEL_NAME,
    CNNFactorizedTransformer,
    FactorizedTransformerConfig,
    architecture_config,
)
from ultrasound_decoding.multiframe.training import normalize_blocks_train_fold_only_with_stats


def load_runner():
    path = PROJECT_DIR / "scripts" / "baselines" / "run_transformer_visual_binary.py"
    spec = importlib.util.spec_from_file_location("run_transformer_visual_binary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_architecture_shapes_forward_backward_and_update() -> None:
    torch.manual_seed(0)
    model = CNNFactorizedTransformer()
    x = torch.randn(2, 4, 128, 501)
    y = torch.tensor([0, 1], dtype=torch.long)
    parameter = next(model.parameters())
    before = parameter.detach().clone()
    logits, shapes = model.forward_with_shapes(x)
    assert shapes == {
        "input": (2, 4, 1, 128, 501),
        "cnn_output": (2, 4, 64, 8, 32),
        "spatial_tokens": (8, 256, 64),
        "spatial_transformer_output": (8, 256, 64),
        "temporal_transformer_input": (2, 4, 64),
        "temporal_transformer_output": (2, 4, 64),
        "pooled": (2, 64),
        "logits": (2, 2),
    }
    loss = torch.nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, parameter.detach())


def test_spatial_weights_are_shared_and_positions_are_two_dimensional() -> None:
    config = FactorizedTransformerConfig()
    model = CNNFactorizedTransformer(config)
    assert model.spatial_row_position.shape == (1, 8, 1, 64)
    assert model.spatial_column_position.shape == (1, 1, 32, 64)
    assert model.temporal_position.shape == (1, 4, 64)
    assert len(model.spatial_transformer.layers) == 2
    assert len(model.temporal_transformer.layers) == 1
    assert not hasattr(model, "frame_specific_spatial_transformers")
    assert model.stem.layers[-1].output_size == (8, 32)


def test_architecture_config_is_frozen_and_explicit() -> None:
    payload = architecture_config()
    assert payload["model"] == MODEL_NAME
    assert payload["model_implementation_version"] == MODEL_IMPLEMENTATION_VERSION
    assert payload["spatial_tokens_per_frame"] == 256
    assert payload["pre_norm"] is True
    assert payload["causal_temporal_mask"] is False
    assert payload["config"]["d_model"] == 64
    assert payload["config"]["num_heads"] == 4
    assert payload["config"]["dropout"] == 0.25


def test_train_only_normalization_is_invariant_to_test_values() -> None:
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


def test_manifest_dtype_normalization_and_true_mismatch_detection() -> None:
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
    mismatched = inferred.copy()
    mismatched["n_test_blocks"] = 8
    assert not runner.canonical_manifest(base).equals(runner.canonical_manifest(mismatched))


def test_identity_fingerprint_changes_with_commit_or_model_version() -> None:
    runner = load_runner()
    base = {
        "experiment_config": {"batch_size": 16},
        "git_commit": "abc", "model_implementation_version": "v1",
        "model_source_sha256": "m", "runner_source_sha256": "r",
    }
    changed_commit = dict(base, git_commit="def")
    changed_version = dict(base, model_implementation_version="v2")
    assert runner.fingerprint(base) != runner.fingerprint(changed_commit)
    assert runner.fingerprint(base) != runner.fingerprint(changed_version)


def test_exact_sign_flip_is_two_sided_and_exact() -> None:
    runner = load_runner()
    assert runner.exact_two_sided_sign_flip(np.ones(3)) == pytest.approx(0.25)
    assert runner.exact_two_sided_sign_flip(-np.ones(3)) == pytest.approx(0.25)


def test_completed_task_requires_artifacts_not_just_complete_marker(tmp_path: Path) -> None:
    runner = load_runner()
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"task_key": "626:x:0:1"}))
    expected = {
        "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
        "n_test_samples": 4,
    }
    valid, reason = runner.validate_completed_task(tmp_path, expected, "run")
    assert valid is False
    assert "missing/empty files" in reason


def test_completed_task_is_revalidated_and_rejects_fingerprint_corruption(tmp_path: Path) -> None:
    runner = load_runner()
    run_fingerprint = "run-fingerprint"
    expected = {
        "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
        "n_test_samples": 4, "config_fingerprint": "config-fingerprint", "batch_size": 16,
    }
    task_fingerprint = runner.task_fingerprint(run_fingerprint, expected)
    y_true = np.asarray([0, 0, 1, 1])
    y_pred = y_true.copy()
    predictions = pd.DataFrame(
        {
            "session": ["626"] * 4, "model": [MODEL_NAME] * 4,
            "seed": [0] * 4, "fold": [1] * 4, "sample_index": range(4),
            "block_id": [f"block_{i}" for i in range(4)], "cycle": [0] * 4,
            "block_name": ["grating", "static", "dot", "stop_after_grating"],
            "y_true": y_true, "y_pred": y_pred,
            "probability_0": [1.0, 1.0, 0.0, 0.0],
            "probability_1": [0.0, 0.0, 1.0, 1.0],
        }
    )
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
                "true_label": truth, "predicted_label": pred,
                "count": 2 if truth == pred else 0,
            }
            for truth in (0, 1) for pred in (0, 1)
        ]
    ).to_csv(tmp_path / "confusion_matrix.csv", index=False)
    pd.DataFrame(
        {
            "session": ["626"] * 40, "model": [MODEL_NAME] * 40,
            "seed": [0] * 40, "fold": [1] * 40, "epoch": range(1, 41),
            "train_loss": np.linspace(1.0, 0.1, 40),
            "train_accuracy": np.linspace(0.5, 1.0, 40),
        }
    ).to_csv(tmp_path / "training_history.csv", index=False)
    (tmp_path / "normalization_audit.json").write_text(
        json.dumps(
            {
                "session": "626", "method": MODEL_NAME, "seed": 0, "fold": 1,
                "phase": "outer_train_fold_only", "target_used_for_stats": False,
            }
        )
    )
    (tmp_path / "model_config.json").write_text(json.dumps(architecture_config()))
    result = {
        "run_fingerprint": run_fingerprint, "task_fingerprint": task_fingerprint,
        "config_fingerprint": "config-fingerprint",
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "session": "626", "model": MODEL_NAME, "seed": 0, "fold": 1,
        "n_test_samples": 4, "balanced_accuracy": 1.0, "accuracy": 1.0,
        "macro_f1": 1.0, "trained_epochs": 40, "actual_batch_size": 16,
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "COMPLETE.json").write_text(
        json.dumps(
            {
                "task_key": "626:cnn_factorized_transformer:0:1",
                "run_fingerprint": run_fingerprint,
                "task_fingerprint": task_fingerprint,
                "config_fingerprint": "config-fingerprint",
            }
        )
    )
    valid, reason = runner.validate_completed_task(tmp_path, expected, run_fingerprint)
    assert valid is True, reason
    result["task_fingerprint"] = "corrupted"
    (tmp_path / "result.json").write_text(json.dumps(result))
    valid, reason = runner.validate_completed_task(tmp_path, expected, run_fingerprint)
    assert valid is False
    assert "task fingerprint mismatch" in reason


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
    assert data.X.shape[1:] == (4, 128, 501)
    splits = grouped_cv_splits(data.groups, max_folds=10)
    regenerated = split_manifest("710", "binary", data.y, data.groups, splits=splits)
    historical = pd.read_csv(manifest_path)
    assert runner.canonical_manifest(regenerated).equals(runner.canonical_manifest(historical))
    for train_idx, test_idx in splits:
        assert not np.intersect1d(data.groups[train_idx], data.groups[test_idx]).size
