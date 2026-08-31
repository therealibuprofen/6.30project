from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from scripts.baselines import run_dqi_dot_vs_grating_validation as runner
from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.multiframe.cycle_calibrated_late_fusion import (
    FORMAL_TRAINING_CONFIG, build_inner_cache_key, build_task_inner_cache_key,
)
from ultrasound_decoding.multiframe.dqi_dot_vs_grating import (
    TASK_NAME, block_predictions_from_frame_logits, build_inner_manifest,
    concatenated_oof_balanced_accuracy, cross_task_relationship_matrix,
    evaluate_confirmatory_gate, exact_spearman_permutation, leave_one_session_out,
    mean_inner_fold_ba_diagnostic, validate_authoritative_mapping,
)


def _frame_rows() -> pd.DataFrame:
    rows = []
    # Deliberately unequal inner-fold sizes: fold 1 is wrong (2 blocks), folds 2/3 correct (6 blocks).
    for source, (inner, truth, correct) in enumerate(((1, 0, False), (1, 1, False), (2, 0, True), (2, 1, True), (2, 0, True), (2, 1, True), (3, 0, True), (3, 1, True))):
        for pos in range(4):
            predicted = truth if correct else 1 - truth
            logits = (5.0, -5.0) if predicted == 0 else (-5.0, 5.0)
            rows.append({"session": "626", "outer_seed": 0, "outer_fold": 1, "inner_fold": inner, "source_index": source, "block_id": f"b{source}", "cycle": source + 1, "frame_position": pos, "truth": truth, "logit_dot": logits[0], "logit_grating": logits[1]})
    return pd.DataFrame(rows)


def test_mapping_and_four_frame_probability_fusion_and_concatenated_q() -> None:
    validate_authoritative_mapping()
    blocks = block_predictions_from_frame_logits(_frame_rows())
    assert len(blocks) == 8
    assert blocks.n_frames_fused.eq(4).all()
    assert np.allclose(blocks[["prob_dot", "prob_grating"]].sum(axis=1), 1.0)
    q = concatenated_oof_balanced_accuracy(blocks)
    diagnostic = mean_inner_fold_ba_diagnostic(blocks)
    assert q == pytest.approx(0.75)
    assert diagnostic == pytest.approx(2 / 3)
    assert q != diagnostic


def test_task_explicit_cache_identity_cannot_collide_with_binary_identity() -> None:
    common = dict(session="626", outer_fold=1, outer_seed=0, outer_train_cycles=(1, 2, 3), inner_fold=1, inner_train_cycles=(2, 3), inner_validation_cycles=(1,), source_hash="source", protocol_hash="protocol", normalization_fingerprint="normalization", training_config=vars(FORMAL_TRAINING_CONFIG))
    binary = build_inner_cache_key(**common)
    dot_grating = build_task_inner_cache_key(task=TASK_NAME, **common)
    assert binary != dot_grating


def test_exact_permutation_loso_gate_and_cross_task_matrix() -> None:
    q = np.arange(9, dtype=float)
    exact = exact_spearman_permutation(q, q)
    assert exact.total == 362880
    assert exact.observed == pytest.approx(1.0)
    assert exact.two_sided_p == pytest.approx(2 / 362880)
    summary = pd.DataFrame({"session": [str(x) for x in range(9)], "Q_DG_session": q, "BA_DG_session": q, "Q_presence_session": q[::-1], "BA_presence_session": q})
    loso = leave_one_session_out(summary, "Q_DG_session", "BA_DG_session")
    assert len(loso) == 9
    assert loso.spearman_rho.eq(1.0).all()
    decision = evaluate_confirmatory_gate(session_spearman=1.0, exact_p=exact.two_sided_p, loo_median=1.0, loo_minimum=1.0)
    assert decision["decision"] == "supports_cross_task_DQI_validation_dot_vs_grating"
    matrix = cross_task_relationship_matrix(summary)
    assert set(matrix.relationship) == {"Q_presence_to_BA_presence", "Q_DG_to_BA_DG", "Q_presence_to_BA_DG", "Q_DG_to_BA_presence", "Q_presence_to_Q_DG"}
    assert matrix.enters_confirmatory_gate.sum() == 1


def test_oof_rejects_duplicate_frame_or_not_four_frames() -> None:
    rows = _frame_rows()
    with pytest.raises(AssertionError):
        block_predictions_from_frame_logits(pd.concat([rows, rows.iloc[[0]]], ignore_index=True))
    with pytest.raises(AssertionError):
        block_predictions_from_frame_logits(rows.iloc[1:].copy())


def test_plan_and_q_phase_do_not_call_historical_target_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("historical target prediction reference was semantically loaded")

    monkeypatch.setattr(runner, "load_historical_prediction_reference", forbidden)
    quality = pd.DataFrame(
        [
            {"session": session, "seed": seed, "fold": fold, "Q_DG_concatenated_inner_oof_block_BA": 0.5 + 0.01 * fold}
            for session in ("626", "628")
            for fold in (1, 2)
            for seed in (0, 1, 2)
        ]
    )
    fold_q, session_q = runner.summarize_training_only_quality(quality)
    assert len(fold_q) == 4
    assert len(session_q) == 2


def test_quality_frozen_guard_precedes_target_reference_load(tmp_path: pytest.TempPathFactory) -> None:
    output = tmp_path / "output"
    aggregate = tmp_path / "aggregate"
    output.mkdir()
    aggregate.mkdir()
    reference = pd.DataFrame(
        [{"session": "626", "task": "stimulus_type", "method": "fcnn_late_fusion", "seed": 0, "fold": 1, "block_id": f"b{i}", "truth": i % 2, "pred": i % 2, "prob_dot": 0.8 if i % 2 == 0 else 0.2, "prob_grating": 0.2 if i % 2 == 0 else 0.8} for i in range(684)]
    )
    reference.to_csv(aggregate / "multiframe_all_models_predictions.csv", index=False)
    args = type("Args", (), {"output_dir": output, "historical_aggregate_dir": aggregate})()
    with pytest.raises(RuntimeError, match="locked until Q_DG is frozen"):
        runner.load_historical_prediction_reference(args, require_quality_frozen=True)
    quality = output / "session_training_only_quality.csv"
    pd.DataFrame({"session": ["626"], "Q_DG_session": [0.5]}).to_csv(quality, index=False)
    runner.framework.atomic_json(output / "QUALITY_FROZEN.json", {"status": "frozen_before_target_reference_load", "session_quality_sha256": runner.framework.file_sha256(quality)})
    loaded = runner.load_historical_prediction_reference(args, require_quality_frozen=True)
    assert len(loaded) == 684


def test_historical_outer_inference_is_explicitly_cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    model = torch.nn.Linear(1, 1)
    payload = {"normalization_mean": np.zeros((1, 128, 501), np.float32), "normalization_std": np.ones((1, 128, 501), np.float32)}
    monkeypatch.setattr(runner, "load_validated_checkpoint", lambda *args, **kwargs: (model, payload, {}))
    monkeypatch.setattr(runner, "_indices", lambda data, row: (np.array([], dtype=int), np.array([0], dtype=int)))
    seen: dict[str, str] = {}
    def fake_predict(model: torch.nn.Module, blocks: np.ndarray, *, device: str, batch_size: int) -> np.ndarray:
        seen["device"] = str(device)
        assert {parameter.device.type for parameter in model.parameters()} == {"cpu"}
        return np.zeros((4, 2), dtype=float)
    monkeypatch.setattr(runner, "predict_raw_logits", fake_predict)
    data = type("Data", (), {"X": np.zeros((1, 4, 128, 501), np.float32), "y": np.array([0]), "metadata": pd.DataFrame({"block_id": ["b0"]})})()
    row = {"session": "626", "seed": 0, "fold": 1, "outer_train_cycles": "1", "outer_test_cycles": "2", "historical_checkpoint_path": "unused", "historical_checkpoint_sha256": "sha"}
    args = type("Args", (), {"device": "cuda", "inference_batch_size": 64})()
    runner._historical_outer(args, row, data)
    assert seen["device"] == "cpu"


def _reconstruction_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    reconstructed = pd.DataFrame({"session": ["626", "626"], "seed": [0, 0], "fold": [1, 1], "block_id": ["dot", "grating"], "truth": [0, 1], "prediction": [0, 1], "prob_dot": [0.8, 0.2], "prob_grating": [0.2, 0.8]})
    reference = reconstructed.rename(columns={"prediction": "pred"}).copy()
    return reconstructed, reference


@pytest.mark.parametrize("column,value", [("truth", 1), ("pred", 1), ("prob_dot", 0.1)])
def test_historical_reconstruction_mismatch_stops(column: str, value: float) -> None:
    reconstructed, reference = _reconstruction_frames()
    reference.loc[0, column] = value
    audit, _ = runner.historical_reconstruction_audit(reconstructed, reference, expected_rows=2)
    assert audit["status"] == "FAIL"
    with pytest.raises(AssertionError):
        runner.require_reconstruction_pass(audit)


def test_validation_mutation_is_real_and_cannot_change_train_normalization() -> None:
    rng = np.random.default_rng(4)
    blocks = rng.normal(size=(3, 4, 128, 501)).astype(np.float32)
    audit = runner.validation_mutation_normalization_audit(blocks, np.array([0, 1]), np.array([2]))
    assert audit["validation_pixels_actually_changed"]
    assert audit["train_pixels_unchanged"]
    assert audit["normalization_arrays_unchanged"]
    assert audit["normalization_fingerprint_before"] == audit["normalization_fingerprint_after"]


def test_planning_invariants_are_246_outer_tasks_and_738_inner_splits() -> None:
    rows = []
    for task_i in range(246):
        rows.append({"task_key": f"t{task_i}", "task_fingerprint": f"fp{task_i}", "session": "626", "seed": task_i % 3, "fold": task_i // 3 + 1, "outer_train_cycles": "1,2,3", "outer_test_cycles": "4"})
    manifest = build_inner_manifest(pd.DataFrame(rows), "protocol", "source")
    assert len(rows) == 246
    assert len(manifest) == 738
    assert manifest.groupby("task_key").inner_fold.nunique().eq(3).all()


def test_formal_session_target_is_seed_oof_ba_then_mean() -> None:
    rows = []
    # Seed BAs are 1.0, 0.5 and 0.0; formal session target must be 0.5.
    predictions = ((0, 1), (0, 0), (1, 0))
    for seed, (pred0, pred1) in enumerate(predictions):
        rows.extend([{"session": "626", "seed": seed, "truth": 0, "prediction": pred0}, {"session": "626", "seed": seed, "truth": 1, "prediction": pred1}])
    seed_ba, session = runner.session_target_from_oof(pd.DataFrame(rows))
    assert seed_ba.BA_DG_seed_OOF.tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert session.iloc[0].BA_DG_session == pytest.approx(0.5)


def test_nonfinite_loso_cannot_pass_gate() -> None:
    gate = evaluate_confirmatory_gate(session_spearman=0.9, exact_p=0.01, loo_median=np.nan, loo_minimum=0.8)
    assert not gate["all_observed_statistics_finite"]
    assert gate["decision"] == "does_not_support_cross_task_DQI_validation_dot_vs_grating"
    assert not any(gate["passes"].values())


def _valid_dg_checkpoint_payload() -> dict[str, object]:
    model = FCNN(input_shape=(128, 501), n_classes=2)
    return {
        "method": "fcnn_late_fusion",
        "model_config": {"base_model": "official_single_frame_FCNN", "late_fusion_probability_average": True, "temporal_length": 4},
        "model_parameters": 48_011,
        "model_state_dict": model.state_dict(),
        "classes": [0, 1],
        "session": "626",
        "task": "stimulus_type",
        "seed": 0,
        "fold": 1,
        "train_cycles": "1,2,3",
        "test_cycles": "4",
        "max_epochs": 40,
        "final_epoch": 40,
        "normalization_mean": np.zeros((1, 128, 501), dtype=np.float32),
        "normalization_std": np.ones((1, 128, 501), dtype=np.float32),
        "normalization_transform": "arcsinh_then_train_pixel_zscore",
        "input_shape": [4, 128, 501],
        "code_version": "test",
    }


@pytest.mark.parametrize("failure_mode", ["missing", "corrupt", "metadata_mismatch"])
def test_checkpoint_preflight_failure_prevents_phase1(tmp_path: Path, failure_mode: str) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    if failure_mode == "corrupt":
        checkpoint.write_bytes(b"not a torch checkpoint")
    elif failure_mode == "metadata_mismatch":
        payload = _valid_dg_checkpoint_payload()
        payload["session"] = "wrong"
        torch.save(payload, checkpoint)
    expected_sha = runner.framework.file_sha256(checkpoint) if checkpoint.is_file() else "missing"
    plan = pd.DataFrame([{"task_key": "626:0:1", "session": "626", "seed": 0, "fold": 1, "outer_train_cycles": "1,2,3", "outer_test_cycles": "4", "historical_checkpoint_path": str(checkpoint), "historical_checkpoint_sha256": expected_sha}])
    args = type("Args", (), {"output_dir": tmp_path / "output"})()
    with pytest.raises(RuntimeError, match="preflight failed before Phase 1"):
        runner.historical_checkpoint_preflight(args, plan, expected_count=1)
    audit = runner.json.loads((args.output_dir / "historical_checkpoint_preflight.json").read_text())
    assert audit["status"] == "FAIL"
    assert audit["metadata_match"] < 1


def test_missing_required_aggregate_prevents_completion(tmp_path: Path) -> None:
    for name in runner.REQUIRED_RUN_OUTPUTS:
        if name != "outer_target_predictions.csv":
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("placeholder")
    with pytest.raises(FileNotFoundError, match="outer_target_predictions.csv"):
        runner.aggregate_artifact_sha256(tmp_path)


def test_status_rejects_posthoc_frozen_q_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in runner.REQUIRED_RUN_OUTPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder")
    quality = tmp_path / "session_training_only_quality.csv"
    quality.write_text("session,Q_DG_session\n626,0.5\n")
    runner.framework.atomic_json(tmp_path / "QUALITY_FROZEN.json", {"status": "frozen_before_target_reference_load", "session_quality_sha256": runner.framework.file_sha256(quality)})
    hashes = runner.aggregate_artifact_sha256(tmp_path)
    runner.framework.atomic_json(tmp_path / "RUN_COMPLETE.json", {"status": "complete", "aggregate_artifact_sha256": hashes})
    runner.validate_completed_run_integrity(tmp_path, runner.json.loads((tmp_path / "RUN_COMPLETE.json").read_text()))
    quality.write_text("session,Q_DG_session\n626,0.9\n")
    monkeypatch.setattr(runner, "load_strict_plan", lambda args: (pd.DataFrame(), pd.DataFrame(), {}))
    args = type("Args", (), {"output_dir": tmp_path})()
    with pytest.raises(AssertionError, match="frozen-Q integrity failed"):
        runner.run_status(args)
