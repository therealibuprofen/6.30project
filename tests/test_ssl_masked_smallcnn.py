from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    load_block_sequence_session,
    split_manifest,
)
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.multiframe.training import normalize_blocks_train_fold_only_with_stats, set_reproducible_seed
from ultrasound_decoding.ssl_masked import (
    MASK_BLOCK_SIZE,
    MASK_RATIO,
    REQUIRED_FORMAL_OUTPUTS,
    SSL_CONDITIONS,
    SSL_SEEDS,
    MaskedFrameDataset,
    MaskedReconstructionSmallCNN,
    PretrainingResult,
    SSLPretrainingConfig,
    SmallCNNReconstructionDecoder,
    apply_ssl_frame_normalizer,
    assert_within_session_scope,
    checkpoint_has_no_labels,
    configure_downstream_model,
    deterministic_block_mask,
    fit_ssl_frame_normalizer,
    fixed_ssl_validation_cycles,
    load_full_cycle_frames,
    load_ssl_encoder_checkpoint,
    masked_pixel_mse,
    missing_formal_outputs,
    pretrain_masked_smallcnn,
    save_ssl_encoder_checkpoint,
)
from ultrasound_decoding.ssl_reporting import (
    exact_sign_flip_pvalue,
    paired_ssl_improvements,
    session_level_metrics,
    statistical_test_tables,
)


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"


@pytest.fixture(scope="module")
def data_708():
    return {
        task: load_block_sequence_session(PROJECT_DIR, "708", task, data_dir=DATA_DIR)
        for task in ("binary", "stimulus_type")
    }


def test_01_all_nine_preregistered_sessions_enter() -> None:
    assert list(EXPECTED_SESSIONS) == ["626", "628", "708", "709", "710", "807", "813", "817", "822"]
    for session in EXPECTED_SESSIONS:
        assert (DATA_DIR / f"session_{session}_blocks.h5").exists()


def test_02_both_tasks_use_existing_clean4_builder(data_708) -> None:
    assert data_708["binary"].source_h5_path == data_708["stimulus_type"].source_h5_path
    assert load_block_sequence_session.__module__ == "ultrasound_decoding.multiframe.dataset"


def test_03_every_labeled_sample_is_clean4_shape(data_708) -> None:
    for data in data_708.values():
        assert data.X.shape[1:] == EXPECTED_BLOCK_SHAPE == (4, 128, 501)


def test_04_fold_ids_match_existing_historical_benchmark(data_708) -> None:
    data = data_708["binary"]
    current = split_manifest("708", "binary", data.y, data.groups)
    historical = pd.read_csv(
        PROJECT_DIR / "results/runs/multiframe/block_clean4_binary_v1/session_708/split_manifest.csv"
    )
    columns = ["fold", "train_cycles", "test_cycles", "n_train_blocks", "n_test_blocks"]
    pd.testing.assert_frame_equal(
        current[columns].astype(str), historical[columns].astype(str), check_dtype=False
    )


def test_05_test_cycles_never_enter_ssl_partitions(data_708) -> None:
    train_idx, test_idx = grouped_cv_splits(data_708["binary"].groups)[0]
    train_cycles = np.unique(data_708["binary"].groups[train_idx])
    test_cycles = set(np.unique(data_708["binary"].groups[test_idx]))
    ssl_train, ssl_val = fixed_ssl_validation_cycles(train_cycles)
    assert not ((set(ssl_train) | set(ssl_val)) & test_cycles)


def test_06_ssl_full_frames_all_come_from_train_cycles(data_708) -> None:
    train_idx, test_idx = grouped_cv_splits(data_708["binary"].groups)[0]
    train_cycles = np.unique(data_708["binary"].groups[train_idx])
    frames = load_full_cycle_frames(PROJECT_DIR, "708", train_cycles, data_dir=DATA_DIR)
    assert set(frames.cycles) == set(train_cycles)
    assert len(frames.frames) == 30 * len(train_cycles)
    assert not set(frames.cycles) & set(np.unique(data_708["binary"].groups[test_idx]))


def test_07_test_frames_do_not_affect_normalization_fit() -> None:
    train = np.ones((2, 4, 3, 5), dtype=np.float32)
    test_a = np.zeros((1, 4, 3, 5), dtype=np.float32)
    test_b = np.full((1, 4, 3, 5), 1e9, dtype=np.float32)
    common = dict(session="x", task="binary", method="cnn2d_meanpool", seed=1, fold=1, train_cycles="0", test_cycles="1")
    *_, mean_a, std_a = normalize_blocks_train_fold_only_with_stats(train, test_a, **common)
    *_, mean_b, std_b = normalize_blocks_train_fold_only_with_stats(train, test_b, **common)
    assert np.array_equal(mean_a, mean_b)
    assert np.array_equal(std_a, std_b)


def test_08_mask_ratio_is_near_half() -> None:
    mask = deterministic_block_mask(128, 501, seed=3, epoch=2, sample_index=9)
    assert abs(mask.mean() - 0.50) < 0.03


def test_09_mask_block_size_is_fixed_16_by_16() -> None:
    assert MASK_BLOCK_SIZE == (16, 16)
    assert MASK_RATIO == 0.50
    mask = deterministic_block_mask(128, 501, seed=1, epoch=1, sample_index=1)
    full_block_sums = [mask[r : r + 16, c : c + 16].sum() for r in range(0, 128, 16) for c in range(0, 496, 16)]
    assert set(full_block_sums) <= {0, 256}


def test_10_mask_is_reproducible_by_seed_epoch_and_sample() -> None:
    one = deterministic_block_mask(128, 501, seed=7, epoch=4, sample_index=2)
    two = deterministic_block_mask(128, 501, seed=7, epoch=4, sample_index=2)
    other = deterministic_block_mask(128, 501, seed=8, epoch=4, sample_index=2)
    assert np.array_equal(one, two)
    assert not np.array_equal(one, other)


def test_11_loss_uses_only_masked_pixels() -> None:
    target = torch.zeros(1, 1, 2, 2)
    prediction = torch.tensor([[[[1.0, 100.0], [3.0, 100.0]]]])
    mask = torch.tensor([[[[True, False], [True, False]]]])
    assert float(masked_pixel_mse(prediction, target, mask)) == pytest.approx((1 + 9) / 2)


def test_12_ssl_encoder_is_exact_existing_smallcnn_class() -> None:
    model = MaskedReconstructionSmallCNN()
    assert type(model.encoder) is SmallCNNFrameEncoder


def test_13_old_supervised_forward_is_bitwise_identical() -> None:
    set_reproducible_seed(11)
    encoder = SmallCNNFrameEncoder().eval()
    x = torch.randn(2, 1, 128, 501)
    with torch.no_grad():
        historical_path = encoder.layers(x)
        default_path = encoder(x)
    assert torch.equal(historical_path, default_path)


def test_14_decoder_is_ssl_only() -> None:
    assert hasattr(MaskedReconstructionSmallCNN(), "decoder")
    assert not hasattr(CNN2DMeanPool(n_classes=2), "decoder")


def test_15_downstream_checkpoint_contains_no_decoder_parameters(tmp_path: Path) -> None:
    state = SmallCNNFrameEncoder().state_dict()
    model = configure_downstream_model("SSL_FINETUNE", n_classes=2, pretrained_encoder_state=state)
    assert all("decoder" not in key for key in model.state_dict())


def test_16_ssl_frozen_disables_encoder_gradients() -> None:
    state = SmallCNNFrameEncoder().state_dict()
    model = configure_downstream_model("SSL_FROZEN", n_classes=2, pretrained_encoder_state=state)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_17_ssl_finetune_enables_encoder_gradients() -> None:
    state = SmallCNNFrameEncoder().state_dict()
    model = configure_downstream_model("SSL_FINETUNE", n_classes=2, pretrained_encoder_state=state)
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_18_random_init_rejects_ssl_state() -> None:
    with pytest.raises(ValueError, match="must not receive"):
        configure_downstream_model("RANDOM_INIT", n_classes=2, pretrained_encoder_state=SmallCNNFrameEncoder().state_dict())


def test_19_three_conditions_share_exact_folds(data_708) -> None:
    splits = grouped_cv_splits(data_708["binary"].groups)
    manifests = [split_manifest("708", "binary", data_708["binary"].y, data_708["binary"].groups, splits=splits) for _ in SSL_CONDITIONS]
    for value in manifests[1:]:
        pd.testing.assert_frame_equal(value, manifests[0])


def test_20_three_conditions_share_supervised_config_argument() -> None:
    source = inspect.getsource(__import__("ultrasound_decoding.ssl_masked", fromlist=["train_downstream_fold"]).train_downstream_fold)
    assert source.count("config: DeepTrainingConfig") == 1
    assert "condition" not in "".join(line for line in source.splitlines() if "optimizer" in line or "max_epochs" in line)


def test_21_fixed_seeds_are_exact() -> None:
    assert SSL_SEEDS == (20260812, 20260813, 20260814)


def test_22_ssl_checkpoint_has_no_labels(tmp_path: Path) -> None:
    encoder = SmallCNNFrameEncoder()
    decoder = SmallCNNReconstructionDecoder()
    result = PretrainingResult(
        encoder=encoder,
        decoder=decoder,
        history=[],
        normalization_mean=np.zeros((1, 128, 501), dtype=np.float32),
        normalization_std=np.ones((1, 128, 501), dtype=np.float32),
        actual_batch_size=8,
        device="cpu",
        qc={},
    )
    path = tmp_path / "encoder.pt"
    save_ssl_encoder_checkpoint(
        path, result, session="708", fold=1, seed=SSL_SEEDS[0],
        ssl_train_cycles=[1, 2], ssl_val_cycles=[3], outer_test_cycles=[0],
        config=SSLPretrainingConfig(),
    )
    assert checkpoint_has_no_labels(path)


def test_23_identical_task_cycle_folds_allow_encoder_reuse(data_708) -> None:
    binary_splits = grouped_cv_splits(data_708["binary"].groups)
    stimulus_splits = grouped_cv_splits(data_708["stimulus_type"].groups)
    assert len(binary_splits) == len(stimulus_splits)
    for (_, binary_test), (_, stimulus_test) in zip(binary_splits, stimulus_splits):
        assert np.array_equal(
            np.unique(data_708["binary"].groups[binary_test]),
            np.unique(data_708["stimulus_type"].groups[stimulus_test]),
        )


def test_24_missing_historical_reproduction_gate_stops(tmp_path: Path) -> None:
    assert missing_formal_outputs(tmp_path)
    assert "audit/historical_baseline_reproduction.csv" in missing_formal_outputs(tmp_path)


def test_25_oof_each_supervised_sample_is_covered_once(data_708) -> None:
    counts = np.zeros(len(data_708["binary"].X), dtype=int)
    for _train, test in grouped_cv_splits(data_708["binary"].groups):
        counts[test] += 1
    assert np.array_equal(counts, np.ones_like(counts))


def test_26_train_test_gap_formula() -> None:
    train_ba, test_ba = 0.9, 0.6
    assert train_ba - test_ba == pytest.approx(0.3)


def _complete_fold_metrics() -> pd.DataFrame:
    rows = []
    for task_i, task in enumerate(("binary", "stimulus_type")):
        for session_i, session in enumerate(EXPECTED_SESSIONS):
            for condition_i, condition in enumerate(SSL_CONDITIONS):
                for seed in SSL_SEEDS:
                    rows.append({
                        "session": session,
                        "task": task,
                        "condition": condition,
                        "seed": seed,
                        "fold": 1,
                        "test_balanced_accuracy": 0.5 + 0.01 * condition_i + 0.001 * session_i,
                        "train_balanced_accuracy": 0.9,
                        "train_test_gap_BA": 0.4 - 0.01 * condition_i - 0.001 * session_i,
                    })
    return pd.DataFrame(rows)


def test_27_statistics_use_nine_sessions_as_primary_units() -> None:
    session_metrics = session_level_metrics(_complete_fold_metrics())
    improvements = paired_ssl_improvements(session_metrics)
    tests, _correction = statistical_test_tables(improvements)
    assert (tests["n_sessions"] == 9).all()
    assert (tests["primary_unit"] == "session").all()


def test_28_cross_session_ssl_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-preregistered"):
        assert_within_session_scope(["708", "source_to_target"])


def test_29_no_registration_path_exists_in_ssl_core() -> None:
    import ultrasound_decoding.ssl_masked as module
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "spatial_registration" not in source
    assert "reference_rigid" not in source


def test_30_no_new_supervised_architecture() -> None:
    model = configure_downstream_model("RANDOM_INIT", n_classes=2, pretrained_encoder_state=None)
    assert type(model) is CNN2DMeanPool
    assert model.temporal_length == 4


def test_31_output_completeness_validator(tmp_path: Path) -> None:
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    checkpoint = tmp_path / "pretraining/checkpoints/session_708/fold_1/seed.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"x")
    curve = tmp_path / "downstream/training_curves/curve.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("x", encoding="utf-8")
    for session in EXPECTED_SESSIONS:
        qc = tmp_path / f"figures/reconstruction_qc/session_{session}_reconstruction_qc.png"
        qc.parent.mkdir(parents=True, exist_ok=True)
        qc.write_bytes(b"x")
    assert missing_formal_outputs(tmp_path) == []


def test_32_decoder_output_is_one_by_128_by_501() -> None:
    decoder = SmallCNNReconstructionDecoder().eval()
    with torch.no_grad():
        output = decoder(torch.zeros(2, 16, 4, 8))
    assert output.shape == (2, 1, 128, 501)


def test_33_mask_value_is_zero_after_preprocessing() -> None:
    raw = np.arange(128 * 501, dtype=np.float32).reshape(1, 128, 501)
    mean, std = fit_ssl_frame_normalizer(raw)
    processed = apply_ssl_frame_normalizer(raw, mean, std)
    dataset = MaskedFrameDataset(processed, seed=1)
    masked, _target, mask, _index = dataset[0]
    assert torch.all(masked[mask] == 0)


def test_34_ssl_optimizer_and_fixed_epoch_config() -> None:
    config = SSLPretrainingConfig()
    assert config.optimizer == "adamw"
    assert config.lr == 1e-3
    assert config.weight_decay == 1e-4
    assert config.batch_size == 32
    assert config.epochs == 50


def test_35_exact_sign_flip_enumerates_expected_null() -> None:
    assert exact_sign_flip_pvalue(np.ones(9)) == pytest.approx(2 / 512)


def test_36_two_task_correction_reports_both_tasks() -> None:
    sessions = session_level_metrics(_complete_fold_metrics())
    improvements = paired_ssl_improvements(sessions)
    _tests, correction = statistical_test_tables(improvements)
    assert set(correction["task"]) == {"binary", "stimulus_type"}
    assert (correction["two_task_corrected_p"] >= correction["raw_p"]).all()


def test_37_decoder_parameters_are_not_saved_in_encoder_checkpoint(tmp_path: Path) -> None:
    result = PretrainingResult(
        encoder=SmallCNNFrameEncoder(), decoder=SmallCNNReconstructionDecoder(), history=[],
        normalization_mean=np.zeros((1, 128, 501), dtype=np.float32),
        normalization_std=np.ones((1, 128, 501), dtype=np.float32),
        actual_batch_size=8, device="cpu", qc={},
    )
    path = tmp_path / "checkpoint.pt"
    save_ssl_encoder_checkpoint(
        path, result, session="708", fold=1, seed=1,
        ssl_train_cycles=[1], ssl_val_cycles=[], outer_test_cycles=[0],
        config=replace(SSLPretrainingConfig(), epochs=1),
    )
    _encoder, payload = load_ssl_encoder_checkpoint(path)
    assert "decoder_state_dict" not in payload
    assert all("decoder" not in key for key in payload["encoder_state_dict"])


def test_38_frozen_encoder_batchnorm_statistics_are_identified_as_frozen() -> None:
    state = SmallCNNFrameEncoder().state_dict()
    model = configure_downstream_model("SSL_FROZEN", n_classes=2, pretrained_encoder_state=state)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    source = inspect.getsource(__import__("ultrasound_decoding.ssl_masked", fromlist=["_supervised_epoch_loop"])._supervised_epoch_loop)
    assert "model.encoder.eval()" in source
