from __future__ import annotations

from dataclasses import replace
import inspect
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
from ultrasound_decoding.multiframe.dataset import EXPECTED_SESSIONS, load_block_sequence_session
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.ssl_masked import SSLFrameData, SSL_SEEDS, configure_downstream_model
from ultrasound_decoding.ssl_multisession_v2 import assert_formal_cuda, reference_optimizer_updates
from ultrasound_decoding.ssl_vicreg_reporting_v3 import (
    planned_statistical_tests,
    session_level_comparison,
)
from ultrasound_decoding.ssl_vicreg_v3 import (
    REQUIRED_FORMAL_OUTPUTS,
    V3_CONDITIONS,
    VICREG_SEEDS,
    VICRegAugmentationConfig,
    VICRegConfig,
    VICRegProjector,
    VICRegSmallCNN,
    build_vicreg_pool,
    checkpoint_contains_no_labels_or_projector,
    conservative_vicreg_augmentation,
    missing_formal_outputs,
    off_diagonal,
    pretrain_vicreg_smallcnn,
    save_vicreg_encoder_checkpoint,
    validate_vicreg_checkpoint,
    vicreg_loss,
    vicreg_loss_components,
)


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"
V1_DIR = PROJECT_DIR / "outputs/ssl_masked_smallcnn_clean4_9sessions_v1"
V2_DIR = PROJECT_DIR / "outputs/ssl_multisession_masked_smallcnn_9sessions_v2/ssl_multisession_masked_smallcnn_9sessions_v2"


def _all_sessions(frames_per_session: int = 4) -> dict[str, SSLFrameData]:
    output = {}
    for session_i, session in enumerate(EXPECTED_SESSIONS):
        cycles = np.arange(frames_per_session, dtype=np.int64) % 2
        output[session] = SSLFrameData(
            frames=np.full((frames_per_session, 128, 501), session_i + 1, dtype=np.float32),
            cycles=cycles,
            original_frame_indices=np.arange(frames_per_session, dtype=np.int64),
            source_h5_path=Path(f"session_{session}_blocks.h5"),
        )
    return output


def _complete_metrics() -> pd.DataFrame:
    rows = []
    for task_i, task in enumerate(("binary", "stimulus_type")):
        for session_i, session in enumerate(EXPECTED_SESSIONS):
            for condition_i, condition in enumerate(V3_CONDITIONS):
                for seed in VICREG_SEEDS:
                    for fold in (1, 2):
                        test = 0.40 + 0.025 * condition_i + 0.002 * session_i + 0.001 * task_i
                        rows.append({
                            "session": session, "task": task, "condition": condition,
                            "seed": seed, "fold": fold, "test_balanced_accuracy": test,
                            "train_balanced_accuracy": 0.8, "train_test_gap_BA": 0.8 - test,
                        })
    return pd.DataFrame(rows)


def test_01_all_nine_sessions_are_fixed() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_exact_five_formal_conditions() -> None:
    assert V3_CONDITIONS == (
        "RANDOM_INIT", "WITHIN_MASKED_SSL_FT", "MULTI_MASKED_SSL_FT",
        "WITHIN_VICREG_SSL_FT", "MULTI_VICREG_SSL_FT",
    )
    assert "OTHER_ONLY_VICREG_SSL_FT" not in V3_CONDITIONS


def test_03_old_folds_match_v1_and_v2() -> None:
    data = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    v1 = pd.read_csv(V1_DIR / "audit/fold_reproduction.csv")
    v2 = pd.read_csv(V2_DIR / "audit/fold_identity_check.csv")
    for fold_i, (train_idx, test_idx) in enumerate(grouped_cv_splits(data.groups), start=1):
        train = ",".join(map(str, sorted(np.unique(data.groups[train_idx]).tolist())))
        test = ",".join(map(str, sorted(np.unique(data.groups[test_idx]).tolist())))
        one = v1[(v1["session"].astype(str) == "708") & (v1["task"] == "binary") & (v1["fold"] == fold_i)].iloc[0]
        two = v2[(v2["session"].astype(str) == "708") & (v2["task"] == "binary") & (v2["fold"] == fold_i)].iloc[0]
        assert train == str(one["train_cycles"]) == str(two["current_train_cycle_ids"])
        assert test == str(one["test_cycles"]) == str(two["current_test_cycle_ids"])


def test_04_within_target_test_zero_leakage() -> None:
    pool = build_vicreg_pool(
        _all_sessions(), target_session="709", target_train_cycles=[0],
        target_test_cycles=[1], condition="WITHIN_VICREG_SSL_FT",
    )
    assert pool.source_sessions == ("709",)
    assert set(pool.cycles_by_session["709"]) == {0}


def test_05_multi_target_test_zero_leakage() -> None:
    pool = build_vicreg_pool(
        _all_sessions(), target_session="709", target_train_cycles=[0],
        target_test_cycles=[1], condition="MULTI_VICREG_SSL_FT",
    )
    assert len(pool.source_sessions) == 9
    assert set(pool.cycles_by_session["709"]) == {0}


def test_06_same_smallcnn_backbone() -> None:
    assert type(VICRegSmallCNN().encoder) is SmallCNNFrameEncoder


def test_07_projector_architecture_is_exact() -> None:
    layers = VICRegProjector().layers
    assert isinstance(layers[0], torch.nn.Linear) and (layers[0].in_features, layers[0].out_features) == (512, 256)
    assert isinstance(layers[1], torch.nn.BatchNorm1d)
    assert isinstance(layers[2], torch.nn.ReLU)
    assert isinstance(layers[3], torch.nn.Linear) and (layers[3].in_features, layers[3].out_features) == (256, 256)


def test_08_projector_is_ssl_only_and_downstream_has_none() -> None:
    assert hasattr(VICRegSmallCNN(), "projector")
    assert not hasattr(CNN2DMeanPool(n_classes=2), "projector")


def test_09_two_views_use_independent_randomness() -> None:
    frames = torch.linspace(-1, 1, 128 * 501).reshape(1, 1, 128, 501)
    view1 = conservative_vicreg_augmentation(frames, seed=100)
    view2 = conservative_vicreg_augmentation(frames, seed=101)
    assert not torch.equal(view1, view2)


def test_10_augmentation_is_reproducible_per_seed() -> None:
    frames = torch.randn(2, 1, 16, 20)
    one = conservative_vicreg_augmentation(frames, seed=5)
    two = conservative_vicreg_augmentation(frames, seed=5)
    assert torch.equal(one, two)


def test_11_no_flip_rotation_crop_or_affine_path() -> None:
    source = inspect.getsource(conservative_vicreg_augmentation).lower()
    for forbidden in ("flip", "rot", "crop", "affine", "elastic", "perspective", "translate"):
        assert forbidden not in source


def test_12_gain_range_is_frozen() -> None:
    config = VICRegAugmentationConfig()
    assert (config.gain_probability, config.gain_min, config.gain_max) == (0.8, 0.9, 1.1)


def test_13_offset_range_is_frozen() -> None:
    config = VICRegAugmentationConfig()
    assert (config.offset_probability, config.offset_min, config.offset_max) == (0.8, -0.05, 0.05)


def test_14_noise_range_is_frozen() -> None:
    config = VICRegAugmentationConfig()
    assert (config.noise_probability, config.noise_sigma_min, config.noise_sigma_max) == (0.8, 0.0, 0.03)


def test_15_blur_range_is_frozen() -> None:
    config = VICRegAugmentationConfig()
    assert (config.blur_probability, config.blur_sigma_min, config.blur_sigma_max) == (0.3, 0.1, 0.6)


def test_16_invariance_is_mse() -> None:
    z1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    z2 = torch.tensor([[1.0, 1.0], [2.0, 1.0]])
    invariance, _variance, _covariance = vicreg_loss_components(z1, z2)
    assert float(invariance) == pytest.approx(float(torch.nn.functional.mse_loss(z1, z2)))


def test_17_variance_penalty_is_standard_relu_one_minus_std() -> None:
    z = torch.zeros(8, 4)
    _invariance, variance, _covariance = vicreg_loss_components(z, z)
    assert float(variance) == pytest.approx(0.99, abs=1e-6)


def test_18_covariance_penalizes_off_diagonal_only() -> None:
    diagonal = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
    assert off_diagonal(diagonal).square().sum() == 0
    z = torch.eye(4)
    _invariance, _variance, covariance = vicreg_loss_components(z, z)
    centered = z - z.mean(0)
    matrix = centered.T @ centered / 3
    expected = 2 * off_diagonal(matrix).square().sum() / 4
    assert float(covariance) == pytest.approx(float(expected))


def test_19_vicreg_weights_are_25_25_1() -> None:
    config = VICRegConfig()
    assert (config.invariance_weight, config.variance_weight, config.covariance_weight) == (25.0, 25.0, 1.0)
    z1, z2 = torch.randn(8, 4), torch.randn(8, 4)
    total, parts = vicreg_loss(z1, z2, config=config)
    expected = 25 * parts["invariance"] + 25 * parts["variance"] + parts["covariance"]
    assert torch.equal(total, expected)


def test_20_three_fixed_seeds_match_prior_benchmarks() -> None:
    assert VICREG_SEEDS == SSL_SEEDS == (20260812, 20260813, 20260814)


def test_21_equal_optimizer_update_formula_reused() -> None:
    assert reference_optimizer_updates(180, 32) == 300


def test_22_multi_session_sampler_is_balanced() -> None:
    from ultrasound_decoding.ssl_multisession_v2 import SessionBalancedSampler
    pool = build_vicreg_pool(
        _all_sessions(), target_session="709", target_train_cycles=[0],
        target_test_cycles=[1], condition="MULTI_VICREG_SSL_FT",
    )
    values = SessionBalancedSampler(pool, seed=7).sample(90_000)
    proportions = {session: 0 for session in pool.source_sessions}
    for session, _ in values:
        proportions[session] += 1
    assert max(abs(count / len(values) - 1 / 9) for count in proportions.values()) < 0.01


def test_23_random_and_masked_prior_artifacts_are_complete() -> None:
    v1 = pd.read_csv(V1_DIR / "downstream/fold_metrics.csv")
    v2 = pd.read_csv(V2_DIR / "downstream/fold_metrics.csv")
    assert len(v1[v1["condition"] == "RANDOM_INIT"]) == 492
    assert len(v1[v1["condition"] == "SSL_FINETUNE"]) == 492
    assert len(v2[v2["condition"] == "MULTI_SSL_FT"]) == 492


def test_24_no_registration_in_vicreg_core() -> None:
    import ultrasound_decoding.ssl_vicreg_v3 as module
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    assert "spatial_registration" not in source
    assert "rigid_registration" not in source


def test_25_binary_and_stimulus_folds_remain_equal() -> None:
    binary = load_block_sequence_session(PROJECT_DIR, "708", "binary", data_dir=DATA_DIR)
    stimulus = load_block_sequence_session(PROJECT_DIR, "708", "stimulus_type", data_dir=DATA_DIR)
    for (_, binary_test), (_, stimulus_test) in zip(grouped_cv_splits(binary.groups), grouped_cv_splits(stimulus.groups)):
        assert np.array_equal(np.unique(binary.groups[binary_test]), np.unique(stimulus.groups[stimulus_test]))


def test_26_batch_size_below_eight_stops() -> None:
    pool = build_vicreg_pool(
        _all_sessions(8), target_session="709", target_train_cycles=[0],
        target_test_cycles=[1], condition="WITHIN_VICREG_SSL_FT",
    )
    with pytest.raises(RuntimeError, match="below 8"):
        pretrain_vicreg_smallcnn(pool, seed=1, reference_updates=1, config=replace(VICRegConfig(), batch_size=4), device="cpu")


def test_27_formal_cuda_unavailable_stops_without_cpu_fallback(monkeypatch) -> None:
    with pytest.raises(RuntimeError, match="exactly 'cuda'"):
        assert_formal_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        assert_formal_cuda("cuda")


def test_28_random_init_rejects_ssl_checkpoint() -> None:
    with pytest.raises(ValueError, match="must not receive"):
        configure_downstream_model("RANDOM_INIT", n_classes=2, pretrained_encoder_state=SmallCNNFrameEncoder().state_dict())


def test_29_ssl_conditions_finetune_whole_encoder() -> None:
    model = configure_downstream_model("SSL_FINETUNE", n_classes=2, pretrained_encoder_state=SmallCNNFrameEncoder().state_dict())
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_30_session_level_statistics_use_nine_sessions() -> None:
    table = session_level_comparison(_complete_metrics())
    tests = planned_statistical_tests(table)
    assert len(table) == 18
    assert len(tests) == 6
    assert (tests["primary_unit"] == "session").all()
    assert (tests["n_sessions"] == 9).all()


def test_31_output_completeness_schema(tmp_path: Path) -> None:
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    checkpoint = tmp_path / "pretraining/checkpoints/condition/session/fold/seed.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("x", encoding="utf-8")
    curve = tmp_path / "downstream/training_curves/curve.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("x", encoding="utf-8")
    assert missing_formal_outputs(tmp_path) == []


def test_32_tiny_vicreg_checkpoint_discards_projector(tmp_path: Path) -> None:
    frames = np.zeros((8, 128, 501), dtype=np.float32)
    all_frames = _all_sessions(8)
    all_frames["708"] = SSLFrameData(
        frames=frames, cycles=np.zeros(8, dtype=np.int64),
        original_frame_indices=np.arange(8), source_h5_path=Path("synthetic.h5"),
    )
    pool = build_vicreg_pool(
        all_frames, target_session="708", target_train_cycles=[0],
        target_test_cycles=[1], condition="WITHIN_VICREG_SSL_FT",
    )
    result = pretrain_vicreg_smallcnn(
        pool, seed=1, reference_updates=1, config=replace(VICRegConfig(), batch_size=8), device="cpu"
    )
    path = tmp_path / "vicreg.pt"
    fingerprint = "test-fingerprint"
    save_vicreg_encoder_checkpoint(
        path, result, target_session="708", fold=1, seed=1, condition="WITHIN_VICREG_SSL_FT",
        pool=pool, target_train_cycles=[0], target_test_cycles=[1],
        config=replace(VICRegConfig(), batch_size=8), augmentation_config=VICRegAugmentationConfig(),
        implementation_fingerprint=fingerprint,
    )
    payload = validate_vicreg_checkpoint(
        path, target_session="708", fold=1, seed=1, condition="WITHIN_VICREG_SSL_FT",
        reference_updates=1, implementation_fingerprint=fingerprint,
        source_sessions=pool.source_sessions, target_train_cycles=[0], target_test_cycles=[1],
        config=replace(VICRegConfig(), batch_size=8), augmentation_config=VICRegAugmentationConfig(),
    )
    assert payload["projector_discarded"] is True
    assert payload["contains_projector_state"] is False
    assert all("projector" not in key for key in payload["encoder_state_dict"])
    assert checkpoint_contains_no_labels_or_projector(path)
