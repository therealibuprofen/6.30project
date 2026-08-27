from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ultrasound_decoding.multiframe.cyclemix_temporal1d import (
    CYCLEMIX_VARIANT,
    LAMBDA_MIX,
    RAW_VARIANT,
    _train_epochs_with_cyclemix,
    build_same_block_cross_cycle_partner_pool,
    make_cyclemix_batch,
    preprocess_train_fold_only,
    select_cyclemix_partners,
)
from ultrasound_decoding.multiframe.models import CNN2DTemporal1D
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    _train_epochs,
    predict_probabilities,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_cyclemix_temporal1d.py"
    spec = importlib.util.spec_from_file_location("run_cyclemix_temporal1d", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool(blocks=(0, 0, 0, 0), cycles=(1, 2, 3, 4)):
    return build_same_block_cross_cycle_partner_pool(blocks, cycles)


def test_same_block_different_cycle_can_pair() -> None:
    pool = _pool((0, 0), (1, 2))
    assert pool.partners_by_anchor[0].tolist() == [1]
    assert pool.partners_by_anchor[1].tolist() == [0]


def test_same_block_same_cycle_is_forbidden() -> None:
    pool = _pool((0, 0), (1, 1))
    assert not len(pool.partners_by_anchor[0])
    assert not len(pool.partners_by_anchor[1])


def test_different_block_is_forbidden() -> None:
    pool = _pool((0, 2), (1, 2))
    assert not len(pool.partners_by_anchor[0])
    assert not len(pool.partners_by_anchor[1])


def test_changing_test_cycle_cannot_change_training_partner_pool() -> None:
    train_blocks = np.array([0, 0, 2, 2])
    train_cycles = np.array([1, 2, 1, 2])
    before = _pool(train_blocks, train_cycles)
    _test_cycles_before = np.array([3, 3, 3, 3])
    after = _pool(train_blocks, train_cycles)
    _test_cycles_after = np.array([99, 100, 101, 102])
    assert before.train_cycle_ids == after.train_cycle_ids == (1, 2)
    for left, right in zip(before.partners_by_anchor, after.partners_by_anchor):
        assert np.array_equal(left, right)


def test_fixed_50_50_mix_is_arithmetic_mean() -> None:
    pool = _pool((0, 0), (1, 2))
    x = torch.tensor([[[[[2.0]]]], [[[[4.0]]]]])
    y = torch.tensor([1, 1])
    mixed, labels, audit = make_cyclemix_batch(x, y, [0], [1], pool)
    assert LAMBDA_MIX == 0.5
    assert torch.equal(mixed, torch.tensor([[[[[3.0]]]]]))
    assert labels.tolist() == [1]
    assert audit["number_of_mixed_samples_generated"] == 1


def test_hard_label_must_match_for_same_block_partner() -> None:
    pool = _pool((0, 0), (1, 2))
    with pytest.raises(AssertionError, match="different hard labels"):
        make_cyclemix_batch(torch.zeros(2, 1), torch.tensor([0, 1]), [0], [1], pool)


def test_mix_occurs_after_real_train_only_normalization() -> None:
    x_train = np.array([0.0, 2.0], dtype=np.float32).reshape(2, 1, 1, 1)
    x_test = np.array([1000.0], dtype=np.float32).reshape(1, 1, 1, 1)
    train_norm, _, audit, _, _ = preprocess_train_fold_only(
        x_train,
        x_test,
        input_variant=CYCLEMIX_VARIANT,
        session="x",
        fold=1,
        seed=0,
        train_cycles="1,2",
        test_cycles="3",
    )
    pool = _pool((0, 0), (1, 2))
    mixed, _, _ = make_cyclemix_batch(
        torch.from_numpy(train_norm[:, :, None]),
        torch.tensor([1, 1]),
        [0],
        [1],
        pool,
    )
    assert float(mixed.mean()) == pytest.approx(0.0, abs=1e-6)
    assert audit["normalization_fit_before_cyclemix"] is True
    assert audit["synthetic_samples_used_for_normalization_fit"] is False
    assert audit["test_mix_applied"] is False


def test_no_valid_partner_returns_empty_mix_and_keeps_real_path() -> None:
    pool = _pool((0, 0), (1, 1))
    selected = select_cyclemix_partners(
        [0, 1], pool, seed=0, fold=1, epoch=1, batch_index=0
    )
    mixed, labels, audit = make_cyclemix_batch(
        torch.tensor([[2.0], [4.0]]),
        torch.tensor([1, 1]),
        [0, 1],
        selected,
        pool,
    )
    assert selected.tolist() == [-1, -1]
    assert len(mixed) == len(labels) == 0
    assert audit["number_of_anchors_without_valid_partner"] == 2


def test_partner_selection_is_reproducible_for_same_coordinates() -> None:
    pool = _pool()
    kwargs = dict(seed=7, fold=3, epoch=9, batch_index=2)
    first = select_cyclemix_partners([0, 1, 2, 3], pool, **kwargs)
    second = select_cyclemix_partners([0, 1, 2, 3], pool, **kwargs)
    assert np.array_equal(first, second)


def test_different_seeds_can_differ_but_all_pairs_remain_legal() -> None:
    pool = _pool((0,) * 8, tuple(range(1, 9)))
    anchors = np.arange(8)
    selections = [
        select_cyclemix_partners(
            anchors, pool, seed=seed, fold=1, epoch=1, batch_index=0
        )
        for seed in range(8)
    ]
    assert any(not np.array_equal(selections[0], item) for item in selections[1:])
    for selected in selections:
        assert np.all(selected != anchors)
        assert np.all(pool.cycle_ids[selected] != pool.cycle_ids[anchors])
        assert np.all(pool.block_ids[selected] == pool.block_ids[anchors])


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


def test_raw_training_path_is_exactly_historical_path() -> None:
    torch.manual_seed(11)
    historical = TinyClassifier()
    raw = TinyClassifier()
    raw.load_state_dict(historical.state_dict())
    x = torch.arange(18, dtype=torch.float32).reshape(6, 3) / 10
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    config = DeepTrainingConfig(batch_size=2, max_epochs=3)
    historical_history = _train_epochs(
        historical,
        x,
        y,
        config=config,
        seed=5,
        device=torch.device("cpu"),
        batch_size_reference=6,
    )
    raw_history, examples = _train_epochs_with_cyclemix(
        raw,
        x,
        y,
        _pool((0,) * 6, (1, 2, 3, 4, 5, 6)),
        input_variant=RAW_VARIANT,
        config=config,
        seed=5,
        fold=1,
        device=torch.device("cpu"),
        batch_size_reference=6,
    )
    assert examples == []
    for expected, observed in zip(historical_history, raw_history):
        assert observed["epoch"] == expected["epoch"]
        assert observed["train_loss"] == expected["train_loss"]
        assert observed["train_accuracy"] == expected["train_accuracy"]
        assert observed["n_train_items"] == expected["n_train_items"]
    for key, value in historical.state_dict().items():
        assert torch.equal(value, raw.state_dict()[key])


def test_test_inference_requires_only_x() -> None:
    model = CNN2DTemporal1D(n_classes=2, temporal_length=4).eval()
    probabilities = predict_probabilities(
        model,
        torch.randn(2, 4, 1, 32, 64),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_formal_task_plan_must_be_exactly_492_and_paired() -> None:
    runner = load_runner()
    rows = []
    fold_counts = {session: (6 if session == "708" else 10) for session in runner.EXPECTED_SESSIONS}
    fold_counts["626"] = 8
    fold_counts["628"] = 8
    fold_counts["709"] = 10
    fold_counts["710"] = 10
    fold_counts["807"] = 10
    fold_counts["813"] = 10
    fold_counts["817"] = 10
    fold_counts["822"] = 10
    assert sum(fold_counts.values()) == 82
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
    assert counts["expected_total_tasks"] == 492


def test_seed_summary_uses_epoch_40_real_accuracy_not_best_epoch() -> None:
    runner = load_runner()
    per_fold_rows, prediction_rows, history_rows = [], [], []
    for session in runner.EXPECTED_SESSIONS:
        for variant in runner.INPUT_VARIANTS:
            for seed in runner.SEEDS:
                per_fold_rows.append(
                    {
                        "session": session,
                        "variant": variant,
                        "seed": seed,
                        "fold": 1,
                        "n_samples": 4,
                        "n_cycles": 1,
                        "mean_mix_coverage": 1.0 if variant == CYCLEMIX_VARIANT else 0.0,
                    }
                )
                for sample in range(4):
                    prediction_rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1,
                            "sample_index": sample,
                            "y_true": sample % 2,
                            "y_pred": sample % 2,
                        }
                    )
                for epoch in range(1, 41):
                    history_rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1,
                            "epoch": epoch,
                            "train_accuracy": 0.99 if epoch == 1 else 0.60,
                        }
                    )
    summary = runner.build_seed_summary(
        pd.DataFrame(per_fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(history_rows),
    )
    assert np.allclose(summary["final_train_accuracy"], 0.60)
    assert np.allclose(summary["train_test_gap"], -0.40)


def test_exact_sign_flip_is_exact_enumeration() -> None:
    runner = load_runner()
    assert runner.exact_two_sided_sign_flip(np.ones(9)) == pytest.approx(2 / 512)
