from __future__ import annotations

from dataclasses import asdict, replace
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

from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    BlockSequenceData,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import CNN2DMeanPool, SmallCNNFrameEncoder
from ultrasound_decoding.multisource_csu_reporting_v6 import (
    make_required_figures,
    planned_statistical_tests,
    seed_stability,
    target_level_csu_comparison,
)
from ultrasound_decoding.multisource_csu_v6 import (
    CSU_ALPHA,
    CSU_EPSILON,
    CSU_INSERTION_POINT,
    CSU_OFFICIAL_COMMIT,
    CSU_PROBABILITY,
    CSU_PROJECTED_EIGENVALUE_FLOOR,
    CSUCNN2DMeanPool,
    CSUSmallCNNFrameEncoder,
    CorrelatedStyleUncertainty,
    FROZEN_SUPERVISED_CONFIG,
    REQUIRED_FORMAL_OUTPUTS,
    V6_CONDITIONS,
    V6_SEEDS,
    assert_formal_cuda,
    missing_formal_outputs,
    resolve_v5_artifact_dir,
    train_prepared_csu,
    v5_baseline_compatibility,
)
from ultrasound_decoding.multisource_loso_reporting_v5 import holm_adjust
from ultrasound_decoding.multisource_loso_v5 import (
    epoch_draw_indices,
    prepare_cross_session_data,
    source_sessions_for_target,
)


DATA_DIR = PROJECT_DIR / "processed_data/block_sequences_v1"
RUNNER_PATH = PROJECT_DIR / "scripts/run_multisource_loso_smallcnn_csu_9sessions_v6.py"


def _synthetic_data(session: str, task: str = "binary", offset: float = 0.0) -> BlockSequenceData:
    n = 4
    rng = np.random.default_rng(int(session) + int(offset * 10))
    X = rng.normal(offset, 0.2, size=(n, 4, 128, 501)).astype(np.float32)
    y = np.asarray([0, 1, 0, 1], dtype=np.int64)
    groups = np.asarray([0, 0, 1, 1], dtype=np.int64)
    metadata = pd.DataFrame({
        "block_id": [f"session{session}_cycle{groups[i]:03d}_block{i}" for i in range(n)],
        "cycle": groups,
    })
    return BlockSequenceData(
        session=session,
        task=task,
        X=X,
        y=y,
        groups=groups,
        metadata=metadata,
        clean4_relative_time_s=np.tile(np.arange(4, dtype=np.float32), (n, 1)),
        clean4_original_frame_indices=np.tile(np.arange(4, dtype=np.int64), (n, 1)),
        source_h5_path=Path(f"session_{session}_blocks.h5"),
        source_metadata_path=Path(f"session_{session}_block_metadata.csv"),
    )


def _complete_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    within = []
    for task_i, task in enumerate(EXPECTED_TASKS):
        for target_i, target in enumerate(EXPECTED_SESSIONS):
            sources = ",".join(source_sessions_for_target(target))
            for condition, extra in (("MULTI_SOURCE_ERM", 0.0), ("MULTI_SOURCE_CSU", 0.03)):
                for seed_i, seed in enumerate(V6_SEEDS):
                    ba = 0.49 + extra + 0.002 * task_i + 0.001 * seed_i + 0.0001 * target_i
                    rows.append({
                        "task": task,
                        "target_session": target,
                        "source_sessions": sources,
                        "n_source_sessions": 8,
                        "seed": seed,
                        "condition": condition,
                        "test_balanced_accuracy": ba,
                        "train_balanced_accuracy": 0.8,
                        "train_test_gap_BA": 0.8 - ba,
                    })
            within.append({
                "task": task,
                "target_session": target,
                "within_session_reference_BA": 0.70 + 0.001 * target_i,
            })
    return pd.DataFrame(rows), pd.DataFrame(within)


def _compatible_v5_row(target: str = "626", task: str = "binary", seed: int = V6_SEEDS[0]) -> dict[str, object]:
    return {
        "task": task,
        "target_session": target,
        "source_sessions": ",".join(source_sessions_for_target(target)),
        "n_source_sessions": 8,
        "seed": seed,
        "condition": "MULTI_SOURCE_BALANCED",
        "source_balance_mode": "session_balanced",
        "normalization_weighting": "sample_frequency_weighted_source_only",
        "best_epoch": 40,
        "early_stopping": False,
        "fold": "LOSO_target_session",
        "run_status": "VALID",
        "target_frames_used_for_training": 0,
        "target_labels_used_for_training": False,
        "target_used_for_normalization": False,
        "target_used_for_validation": False,
        "target_used_for_model_selection": False,
    }


def test_01_all_nine_targets_are_frozen() -> None:
    assert tuple(EXPECTED_SESSIONS) == ("626", "628", "708", "709", "710", "807", "813", "817", "822")


def test_02_every_target_has_exactly_eight_sources() -> None:
    assert len({target: source_sessions_for_target(target) for target in EXPECTED_SESSIONS}) == 9
    for target in EXPECTED_SESSIONS:
        sources = source_sessions_for_target(target)
        assert len(sources) == 8
        assert target not in sources


def test_03_target_has_zero_training_and_normalization_exposure() -> None:
    data = {session: _synthetic_data(session, offset=i) for i, session in enumerate(("626", "628", "708"))}
    prepared = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    assert set(prepared.train_session_labels) == {"628", "708"}
    assert prepared.normalization_audit["target_frames_used_for_fit"] == 0
    changed = dict(data)
    changed["626"] = replace(data["626"], X=np.full_like(data["626"].X, 999.0))
    second = prepare_cross_session_data(
        changed, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    assert np.array_equal(prepared.X_train, second.X_train)


def test_04_clean4_and_labels_are_identical_to_v5_builder() -> None:
    for task, blocks_per_cycle in (("binary", 4), ("stimulus_type", 2)):
        data = load_block_sequence_session(PROJECT_DIR, "807", task, data_dir=DATA_DIR)
        assert data.X.shape[1:] == (4, 128, 501)
        assert set(data.metadata.groupby("cycle").size()) == {blocks_per_cycle}
        assert set(data.y) == {0, 1}


def test_05_smallcnn_body_and_mean_fusion_are_unchanged() -> None:
    erm = CNN2DMeanPool(n_classes=2, dropout=0.25, temporal_length=4)
    csu = CSUCNN2DMeanPool(n_classes=2, dropout=0.25, temporal_length=4)
    frozen_types = [type(layer).__name__ for layer in erm.encoder.layers]
    csu_types = [type(layer).__name__ for layer in csu.encoder.block1] + [
        type(layer).__name__ for layer in csu.encoder.remaining
    ]
    assert frozen_types == csu_types
    assert erm.encoder.feature_dim == csu.encoder.feature_dim == 512
    assert erm.temporal_length == csu.temporal_length == 4
    assert [type(layer) for layer in erm.classifier] == [type(layer) for layer in csu.classifier]


def test_06_v5_baseline_exact_artifact_is_reusable() -> None:
    compatible, reason = v5_baseline_compatibility(
        _compatible_v5_row(), task="binary", target="626", seed=V6_SEEDS[0]
    )
    assert compatible
    assert "exact frozen" in reason
    bad = _compatible_v5_row()
    bad["source_balance_mode"] = "natural_frequency"
    assert v5_baseline_compatibility(bad, task="binary", target="626", seed=V6_SEEDS[0])[0] is False


def test_07_nested_v5_artifact_layout_resolves(tmp_path: Path) -> None:
    nested = tmp_path / "download" / "multisource_loso_smallcnn_9sessions_v5"
    metrics = nested / "downstream/fold_metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("condition\nMULTI_SOURCE_BALANCED\n", encoding="utf-8")
    assert resolve_v5_artifact_dir(tmp_path) == nested


def test_08_csu_input_output_shape_is_identical() -> None:
    module = CorrelatedStyleUncertainty(p=1.0)
    module.train()
    x = torch.randn(5, 8, 7, 9)
    assert module(x).shape == x.shape


def test_09_csu_api_cannot_receive_label_or_session_id() -> None:
    parameters = list(inspect.signature(CorrelatedStyleUncertainty.forward).parameters)
    assert parameters == ["self", "x"]
    source = inspect.getsource(CorrelatedStyleUncertainty.forward).lower()
    assert "label" not in source
    assert "session" not in source


def test_10_train_mode_can_activate_csu() -> None:
    module = CorrelatedStyleUncertainty(p=1.0)
    module.train()
    x = torch.randn(6, 8, 5, 7)
    output = module(x)
    assert module.last_applied
    assert module.application_count == 1
    assert not torch.equal(output, x)


def test_11_eval_mode_is_an_exact_deterministic_bypass() -> None:
    module = CorrelatedStyleUncertainty(p=1.0)
    module.eval()
    x = torch.randn(6, 8, 5, 7)
    first = module(x)
    second = module(x)
    assert torch.equal(first, x)
    assert torch.equal(second, x)
    assert first.data_ptr() == x.data_ptr()
    assert module.last_applied is False


def test_12_csu_numerical_output_and_gradients_are_finite() -> None:
    module = CorrelatedStyleUncertainty(p=1.0)
    module.train()
    x = torch.randn(5, 8, 6, 7, requires_grad=True)
    loss = module(x).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_13_rank_deficient_covariance_decomposition_is_stable() -> None:
    module = CorrelatedStyleUncertainty(p=1.0)
    module.train()
    x = torch.ones(4, 8, 4, 4, requires_grad=True)
    output = module(x)
    assert torch.isfinite(output).all()
    output.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_14_nonfinite_input_is_a_hard_failure() -> None:
    module = CorrelatedStyleUncertainty(p=1.0)
    x = torch.randn(4, 8, 4, 4)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="numerical STOP"):
        module(x)


def test_15_official_default_csu_config_is_frozen() -> None:
    module = CorrelatedStyleUncertainty()
    assert (module.alpha, module.p, module.eps) == (0.3, 0.5, 1e-6)
    assert (CSU_ALPHA, CSU_PROBABILITY, CSU_EPSILON) == (0.3, 0.5, 1e-6)
    assert CSU_PROJECTED_EIGENVALUE_FLOOR == 1e-12
    assert CSU_OFFICIAL_COMMIT == "17e948728cad633a218bfd9467f97e80521da1ce"


def test_16_insertion_point_is_exactly_once_after_block1() -> None:
    encoder = CSUSmallCNNFrameEncoder()
    assert CSU_INSERTION_POINT == "after_smallcnn_block1"
    assert encoder.insertion_feature_shape == (8, 64, 125)
    assert sum(isinstance(module, CorrelatedStyleUncertainty) for module in encoder.modules()) == 1
    with torch.no_grad():
        features = encoder.block1(torch.zeros(1, 1, 128, 501))
    assert features.shape == (1, 8, 64, 125)


def test_17_no_insertion_or_csu_hyperparameter_search_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8").lower()
    for forbidden in ("--alpha", "--csu-p", "--insertion", "insertion-search", "optuna"):
        assert forbidden not in source


def test_18_session_balanced_sampler_is_exactly_v5_sampler() -> None:
    labels = np.asarray(["626"] * 8 + ["628"] * 24, dtype=object)
    first = epoch_draw_indices(labels, seed=V6_SEEDS[0], epoch=1, balance_mode="session_balanced")
    second = epoch_draw_indices(labels, seed=V6_SEEDS[0], epoch=1, balance_mode="session_balanced")
    assert np.array_equal(first, second)
    assert {session: int(np.sum(labels[first] == session)) for session in ("626", "628")} == {
        "626": 16, "628": 16
    }


def test_19_seeds_conditions_and_supervision_are_frozen() -> None:
    assert V6_SEEDS == (20260812, 20260813, 20260814)
    assert V6_CONDITIONS == ("MULTI_SOURCE_ERM", "MULTI_SOURCE_CSU")
    assert asdict(FROZEN_SUPERVISED_CONFIG) == {
        "optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-3,
        "batch_size": 16, "max_epochs": 40, "dropout": 0.25,
        "loss": "cross_entropy",
    }


def test_20_tiny_csu_training_has_holdout_and_batch_audit_schema() -> None:
    data = {session: _synthetic_data(session, offset=i) for i, session in enumerate(("626", "628", "708"))}
    prepared = prepare_cross_session_data(
        data, source_sessions=("628", "708"), target_session="626", balance_mode="session_balanced"
    )
    result = train_prepared_csu(
        prepared,
        seed=V6_SEEDS[0],
        config=replace(FROZEN_SUPERVISED_CONFIG, max_epochs=1),
        device="cpu",
    )
    assert result.metrics["target_frames_used_for_training"] == 0
    assert result.metrics["target_used_for_normalization"] is False
    assert result.metrics["target_unlabeled_adaptation"] is False
    assert len(result.batch_domain_diversity) == 1
    assert set(result.batch_domain_diversity[0]) >= {
        "target", "task", "seed", "epoch", "batch_index", "batch_size",
        "n_unique_source_sessions",
    }


def test_21_no_target_adaptation_or_registration_in_core() -> None:
    import ultrasound_decoding.multisource_csu_v6 as module
    source = Path(module.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("spatial_registration", "rigid_registration", "test-time adaptation", "target ssl"):
        assert forbidden not in source


def test_22_statistics_use_nine_target_sessions_and_holm() -> None:
    metrics, within = _complete_metrics()
    target = target_level_csu_comparison(metrics, within)
    tests = planned_statistical_tests(target)
    assert len(target) == 18
    assert (tests["primary_unit"] == "target_session").all()
    assert (tests["n_target_sessions"] == 9).all()
    assert (tests["n_exact_sign_patterns"] == 512).all()
    assert np.all(tests["holm_corrected_p"] >= tests["raw_p"])
    assert holm_adjust(np.asarray([0.01, 0.04])).tolist() == pytest.approx([0.02, 0.04])


def test_23_target_summary_has_required_csu_fields() -> None:
    metrics, within = _complete_metrics()
    target = target_level_csu_comparison(metrics, within)
    assert set(target.columns) >= {
        "MULTI_SOURCE_ERM_BA", "MULTI_SOURCE_CSU_BA", "delta_CSU_minus_ERM",
        "ERM_seed_std", "CSU_seed_std", "within_session_reference_BA",
    }
    assert np.allclose(target["delta_CSU_minus_ERM"], 0.03)


def test_24_all_six_required_figures_render(tmp_path: Path) -> None:
    metrics, within = _complete_metrics()
    target = target_level_csu_comparison(metrics, within)
    stability = seed_stability(metrics)
    make_required_figures(tmp_path, target, stability)
    assert {path.name for path in (tmp_path / "figures").glob("*.png")} == {
        "binary_csu_vs_erm_by_target.png", "stimulus_type_csu_vs_erm_by_target.png",
        "binary_csu_delta.png", "stimulus_type_csu_delta.png",
        "within_cross_gap_csu.png", "csu_seed_stability.png",
    }


def test_25_formal_run_refuses_cpu_and_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(RuntimeError, match="--device cuda"):
        assert_formal_cuda("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        assert_formal_cuda("cuda")


def test_26_output_completeness_contract(tmp_path: Path) -> None:
    assert set(REQUIRED_FORMAL_OUTPUTS).issubset(set(missing_formal_outputs(tmp_path)))
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    curve = tmp_path / "downstream/training_curves/one.csv"
    curve.parent.mkdir(parents=True, exist_ok=True)
    curve.write_text("epoch,loss\n1,1\n", encoding="utf-8")
    assert missing_formal_outputs(tmp_path) == []
