from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.multiframe.sbind_adapted import (
    SBINDAdaptedConfig,
    SBINDImageSelfAttention,
    build_sbind_adapted_model,
    sbind_adapted_architecture_config,
)


def test_both_controlled_variants_forward_backward_and_update() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 1, 32, 40)
    y = torch.tensor([0, 1], dtype=torch.long)
    for method in ("sbind_noatt", "sbind"):
        model = build_sbind_adapted_model(method)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        parameter = next(model.parameters())
        before = parameter.detach().clone()
        logits = model(x)
        assert logits.shape == (2, 2)
        loss = torch.nn.CrossEntropyLoss()(logits, y)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert not torch.equal(before, parameter.detach())
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )


def test_attention_uses_frozen_paper_patch_shape() -> None:
    config = SBINDAdaptedConfig()
    attention = SBINDImageSelfAttention(config)
    x = torch.randn(2, config.latent_channels, config.latent_height, config.latent_width)
    y = attention(x)
    assert y.shape == x.shape
    assert attention.pos_embedding.shape == (1, 16, 1)


def test_noatt_is_strict_attention_removal() -> None:
    noatt = build_sbind_adapted_model("sbind_noatt")
    full = build_sbind_adapted_model("sbind")
    assert not any("global_attention.attention" in name for name, _ in noatt.named_parameters())
    assert any("global_attention.attention" in name for name, _ in full.named_parameters())
    assert sum(p.numel() for p in full.parameters()) > sum(p.numel() for p in noatt.parameters())


def test_architecture_claim_is_explicitly_adapted() -> None:
    config = sbind_adapted_architecture_config("sbind")
    assert "not full SBIND" in config["baseline_claim"]
    assert config["attention_enabled"] is True
    assert config["input_shape"] == [4, 1, 128, 501]
    assert config["config"]["attention_heads"] == 8
    assert config["config"]["attention_embedding_dim"] == 256


def test_manual_attention_path_does_not_require_torch2_fused_api() -> None:
    source = (SRC_DIR / "ultrasound_decoding" / "multiframe" / "sbind_adapted.py").read_text()
    assert "F.scaled_dot_product_attention" not in source
    assert "torch.matmul(queries, keys.transpose" in source


def test_real_clean4_shape_and_formal_fold_identity_when_data_present() -> None:
    data_path = PROJECT_DIR / "processed_data" / "block_sequences_v1" / "session_710_blocks.h5"
    manifest_path = (
        PROJECT_DIR / "results" / "runs" / "multiframe" / "block_clean4_binary_v1"
        / "session_710" / "split_manifest.csv"
    )
    if not data_path.exists() or not manifest_path.exists():
        pytest.skip("real clean4 data or formal fold manifest unavailable")
    import pandas as pd
    from ultrasound_decoding.cv import grouped_cv_splits
    from ultrasound_decoding.multiframe.dataset import load_block_sequence_session, split_manifest

    data = load_block_sequence_session(PROJECT_DIR, "710", "binary")
    assert data.X.shape[1:] == (4, 128, 501)
    splits = grouped_cv_splits(data.groups, max_folds=10)
    regenerated = split_manifest("710", "binary", data.y, data.groups, splits=splits)
    historical = pd.read_csv(manifest_path)
    historical["session"] = historical["session"].astype(str)
    regenerated["session"] = regenerated["session"].astype(str)
    assert regenerated.equals(historical)
    for train_idx, test_idx in splits:
        assert not np.intersect1d(data.groups[train_idx], data.groups[test_idx]).size
