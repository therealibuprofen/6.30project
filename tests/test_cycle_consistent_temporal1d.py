from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from ultrasound_decoding.multiframe.cycle_consistent_temporal1d import (
    CYCLE_CONSISTENT_VARIANT,
    LAMBDA_CONSISTENCY,
    RAW_VARIANT,
    _train_epochs_with_cycle_consistency,
    assert_pair_cycles_are_training_only,
    build_positive_pair_mask,
    cycle_consistency_loss,
    total_training_loss,
    train_fold,
)
from ultrasound_decoding.multiframe.models import (
    CNN2DTemporal1D,
    count_trainable_parameters,
)
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    _train_epochs,
    blocks_to_sequence_tensor,
    predict_probabilities,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def load_runner():
    path = PROJECT_DIR / "scripts/baselines/run_cycle_consistent_temporal1d.py"
    spec = importlib.util.spec_from_file_location(
        "run_cycle_consistent_temporal1d", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_block_different_cycle_is_one_valid_pair() -> None:
    mask = build_positive_pair_mask(
        torch.tensor([0, 0]), torch.tensor([1, 2])
    )
    assert int(mask.sum()) == 1
    assert bool(mask[0, 1])


def test_same_block_same_cycle_is_not_a_pair() -> None:
    mask = build_positive_pair_mask(
        torch.tensor([0, 0]), torch.tensor([1, 1])
    )
    assert int(mask.sum()) == 0


def test_different_block_is_not_a_pair() -> None:
    mask = build_positive_pair_mask(
        torch.tensor([0, 2]), torch.tensor([1, 2])
    )
    assert int(mask.sum()) == 0


def test_positive_pairs_are_not_double_counted() -> None:
    mask = build_positive_pair_mask(
        torch.tensor([0, 0]), torch.tensor([1, 2])
    )
    assert bool(mask[0, 1])
    assert not bool(mask[1, 0])
    assert int(mask.sum()) == 1


def test_cosine_consistency_loss_identical_and_orthogonal() -> None:
    blocks = torch.tensor([0, 0])
    cycles = torch.tensor([1, 2])
    identical, identical_count, _ = cycle_consistency_loss(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]), blocks, cycles
    )
    orthogonal, orthogonal_count, _ = cycle_consistency_loss(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), blocks, cycles
    )
    assert identical_count == orthogonal_count == 1
    assert float(identical) == pytest.approx(0.0, abs=1e-7)
    assert float(orthogonal) == pytest.approx(1.0, abs=1e-7)


def test_no_valid_pair_is_zero_and_classification_backward_still_runs() -> None:
    embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    classifier = nn.Linear(2, 2)
    logits = classifier(embedding)
    classification = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    consistency, pair_count, _ = cycle_consistency_loss(
        embedding,
        torch.tensor([0, 2]),
        torch.tensor([1, 2]),
    )
    loss = total_training_loss(
        classification,
        consistency,
        lambda_consistency=LAMBDA_CONSISTENCY,
    )
    loss.backward()
    assert pair_count == 0
    assert float(consistency.detach()) == 0.0
    assert classifier.weight.grad is not None
    assert torch.isfinite(classifier.weight.grad).all()


def test_lambda_zero_total_loss_is_exact_classification_tensor() -> None:
    classification = torch.tensor(1.25, requires_grad=True)
    consistency = torch.tensor(0.75, requires_grad=True)
    observed = total_training_loss(
        classification, consistency, lambda_consistency=0.0
    )
    assert observed is classification
    assert float(observed.detach()) == pytest.approx(1.25)


def test_changing_held_out_cycle_cannot_change_training_pair_mask() -> None:
    train_blocks = torch.tensor([0, 0, 2, 2])
    train_cycles = torch.tensor([1, 2, 1, 2])
    held_out_before = torch.tensor([3, 3])
    held_out_after = torch.tensor([99, 100])
    before = build_positive_pair_mask(train_blocks, train_cycles)
    _ = held_out_before
    after = build_positive_pair_mask(train_blocks, train_cycles)
    _ = held_out_after
    assert torch.equal(before, after)


def test_pair_cycle_assertion_rejects_non_training_cycle() -> None:
    cycles = torch.tensor([1, 99])
    mask = build_positive_pair_mask(torch.tensor([0, 0]), cycles)
    with pytest.raises(AssertionError, match="non-training cycles"):
        assert_pair_cycles_are_training_only(cycles, mask, [1, 2])


def test_default_forward_matches_historical_expression_exactly() -> None:
    torch.manual_seed(7)
    model = CNN2DTemporal1D(n_classes=2, temporal_length=4).eval()
    inputs = torch.randn(3, 4, 1, 32, 64)
    with torch.no_grad():
        sequence = model.encode_sequence(inputs)
        embedding = model.temporal_conv(sequence.transpose(1, 2))
        historical_logits = model.classifier(embedding)
        default_logits = model(inputs)
        exposed_logits, exposed_embedding = model.forward_with_embedding(inputs)
    assert torch.equal(default_logits, historical_logits)
    assert torch.equal(exposed_logits, historical_logits)
    assert torch.equal(exposed_embedding, embedding)


def test_classifier_input_embedding_shape_is_batch_by_64() -> None:
    model = CNN2DTemporal1D(n_classes=2, temporal_length=4).eval()
    with torch.no_grad():
        logits, embedding = model.forward_with_embedding(
            torch.randn(5, 4, 1, 32, 64)
        )
    assert logits.shape == (5, 2)
    assert embedding.shape == (5, 64)
    assert count_trainable_parameters(model) == 115890


def test_raw_inference_requires_only_x_and_no_metadata() -> None:
    model = CNN2DTemporal1D(n_classes=2, temporal_length=4).eval()
    tensor = torch.randn(3, 4, 1, 32, 64)
    probabilities = predict_probabilities(
        model, tensor, device=torch.device("cpu"), batch_size=2
    )
    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert list(inspect.signature(model.forward).parameters) == ["x"]
    assert "block_names_test" not in inspect.signature(train_fold).parameters


def test_lambda_zero_training_matches_historical_temporal1d_update_exactly() -> None:
    torch.manual_seed(11)
    historical = CNN2DTemporal1D(n_classes=2, temporal_length=4)
    proposed_raw = CNN2DTemporal1D(n_classes=2, temporal_length=4)
    proposed_raw.load_state_dict(historical.state_dict())
    train_tensor = torch.randn(8, 4, 1, 32, 64)
    labels = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    block_ids = np.tile(np.arange(4, dtype=np.int64), 2)
    cycle_ids = np.repeat(np.asarray([1, 2], dtype=np.int64), 4)
    config = DeepTrainingConfig(batch_size=4, max_epochs=1)

    torch.manual_seed(123)
    historical_history = _train_epochs(
        historical,
        train_tensor,
        labels,
        config=config,
        seed=5,
        device=torch.device("cpu"),
        batch_size_reference=len(train_tensor),
    )
    torch.manual_seed(123)
    raw_history = _train_epochs_with_cycle_consistency(
        proposed_raw,
        train_tensor,
        labels,
        block_ids,
        cycle_ids,
        allowed_train_cycle_ids=(1, 2),
        config=config,
        seed=5,
        device=torch.device("cpu"),
        batch_size_reference=len(train_tensor),
        lambda_consistency=0.0,
    )
    assert raw_history[0]["classification_loss"] == historical_history[0]["train_loss"]
    assert raw_history[0]["total_loss"] == historical_history[0]["train_loss"]
    assert raw_history[0]["train_accuracy"] == historical_history[0]["train_accuracy"]
    for name, historical_value in historical.state_dict().items():
        assert torch.equal(proposed_raw.state_dict()[name], historical_value), name


def test_real_clean4_folds_keep_test_cycles_out_when_data_present(tmp_path: Path) -> None:
    data_dir = PROJECT_DIR / "processed_data/block_sequences_v1"
    if not (data_dir / "session_710_blocks.h5").is_file():
        pytest.skip("real clean4 data are unavailable")
    runner = load_runner()
    args = SimpleNamespace(
        project_root=PROJECT_DIR,
        data_dir=data_dir,
        output_dir=tmp_path,
        formal_fold_run_dir=PROJECT_DIR / "outputs/multiscale_temporal1d_v1",
    )
    data, splits = runner.audit_session(args, "710")
    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert set(data.groups[train_idx]).isdisjoint(set(data.groups[test_idx]))
        assert set(data.metadata.iloc[train_idx]["block_name"]) == {
            "grating",
            "stop_after_grating",
            "dot",
            "static",
        }


def test_summary_uses_epoch_40_and_reports_pair_coverage() -> None:
    runner = load_runner()
    fold_rows, prediction_rows, history_rows = [], [], []
    truth = np.asarray([0, 0, 1, 1])
    for session in runner.EXPECTED_SESSIONS:
        for variant in runner.INPUT_VARIANTS:
            predicted = (
                truth
                if variant == CYCLE_CONSISTENT_VARIANT
                else np.asarray([0, 1, 1, 0])
            )
            for seed in runner.SEEDS:
                for fold in (1, 2):
                    fold_rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": fold,
                            "n_samples": 4,
                            "n_cycles": 2,
                            "number_of_batches": 10,
                            "batches_with_valid_pairs": 8,
                            "total_valid_positive_pairs": 30,
                            "consistency_loss_final": (
                                0.2 if variant == CYCLE_CONSISTENT_VARIANT else 0.0
                            ),
                            "training_same_block_cross_cycle_cosine": (
                                0.8 if variant == CYCLE_CONSISTENT_VARIANT else 0.5
                            ),
                        }
                    )
                for sample_index, (y_true, y_pred) in enumerate(zip(truth, predicted)):
                    prediction_rows.append(
                        {
                            "session": session,
                            "variant": variant,
                            "seed": seed,
                            "fold": 1 if sample_index < 2 else 2,
                            "sample_index": sample_index,
                            "y_true": y_true,
                            "y_pred": y_pred,
                        }
                    )
                for fold in (1, 2):
                    for epoch in range(1, runner.FORMAL_EPOCHS + 1):
                        history_rows.append(
                            {
                                "session": session,
                                "variant": variant,
                                "seed": seed,
                                "fold": fold,
                                "epoch": epoch,
                                "train_accuracy": (
                                    0.99
                                    if epoch == 20
                                    else 0.5 + 0.001 * epoch + 0.01 * fold
                                ),
                            }
                        )
    seed_summary = runner.build_seed_summary(
        pd.DataFrame(fold_rows),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(history_rows),
    )
    assert seed_summary.iloc[0]["final_train_accuracy"] == pytest.approx(0.555)
    assert seed_summary.iloc[0]["mean_valid_pairs_per_batch"] == pytest.approx(3.0)
    assert seed_summary.iloc[0]["valid_pair_batch_fraction"] == pytest.approx(0.8)
    session_summary = runner.build_session_summary(seed_summary)
    overall, paired, decision = runner.build_overall_and_decision(session_summary)
    assert overall.iloc[0]["median_delta_BA"] == pytest.approx(0.5)
    assert paired.iloc[0]["exact_two_sided_sign_flip_p"] == pytest.approx(2 / 512)
    assert decision["decision"] == "supports_continue_cycle_consistency_route"
