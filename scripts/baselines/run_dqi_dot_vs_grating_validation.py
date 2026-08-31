#!/usr/bin/env python3
"""DQI / Q_dec cross-task confirmatory validation (dot vs grating).

The formal stage is deliberately two phase: all Q_dec caches are completed and
frozen before any historical outer-test prediction is read or reconstructed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

PROJECT_DIR = Path(__file__).resolve().parents[2]
for item in (PROJECT_DIR, PROJECT_DIR / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from scripts.baselines import run_multiscale_temporal1d as framework
from ultrasound_decoding.multiframe.canonical_single_frame import (
    NORMALIZATION_TRANSFORM, apply_saved_normalization, load_validated_checkpoint,
)
from ultrasound_decoding.multiframe.cycle_calibrated_late_fusion import (
    FORMAL_TRAINING_CONFIG, FRAMES_PER_BLOCK, assert_complete_inner_oof,
    build_task_inner_cache_key, cycle_text, fingerprint,
    fit_inner_train_normalization, predict_raw_logits, softmax_probabilities,
    train_inner_fcnn,
)
from ultrasound_decoding.multiframe.dataset import default_block_data_dir, load_block_sequence_session
from ultrasound_decoding.multiframe.dqi_dot_vs_grating import (
    CLASS_NAMES, EXPECTED_FOLDS, EXPECTED_INNER_TRAININGS, EXPECTED_OUTER_TASKS,
    HISTORICAL_TASK_NAME, N_INNER_FOLDS, OUTPUT_VERSION, SESSIONS, SEEDS, TASK_NAME,
    block_predictions_from_frame_logits, build_inner_manifest,
    concatenated_oof_balanced_accuracy, cross_task_relationship_matrix,
    evaluate_confirmatory_gate, exact_spearman_permutation, finite_pearson,
    finite_spearman, leave_one_session_out, mean_inner_fold_ba_diagnostic,
    validate_authoritative_mapping, validate_dot_vs_grating_data,
)

BASELINE_ATOL = 2e-6
HISTORICAL_OUTER_INFERENCE_DEVICE = "cpu"
REQUIRED_TASK_FILES = ("result.json", "inner_split_manifest.csv", "inner_training_summary.csv", "inner_oof_logits.csv", "inner_oof_predictions.csv", "leakage_audit.json")
REQUIRED_RUN_OUTPUTS = ("DQI_DOT_VS_GRATING_VALIDATION.md", "config.json", "runtime_fingerprint.json", "task_plan.csv", "inner_split_manifest.csv", "historical_fcnn_checkpoint_manifest.csv", "inner_oof_manifest.csv", "outer_task_quality.csv", "inner_oof_predictions.csv", "outer_fold_seed_averaged.csv", "within_session_fold_relationship.csv", "session_quality_summary.csv", "QUALITY_FROZEN.json", "outer_target_predictions.csv", "outer_target_reconstruction_audit.json", "loo_robustness.csv", "cross_task_relationship_matrix.csv", "statistical_audit.json", "provenance_audit.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DQI / Q_dec dot-vs-grating confirmatory validation")
    p.add_argument("--stage", choices=("plan", "sanity", "full", "status"), required=True)
    p.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / OUTPUT_VERSION)
    p.add_argument("--historical-aggregate-dir", type=Path, default=PROJECT_DIR / "results/runs/multiframe/block_clean4_stimulus_type_all_models_9sessions_v1/aggregate")
    p.add_argument("--binary-plan-dir", type=Path, default=PROJECT_DIR / "outputs/fcnn_canonical_single_frame_v1")
    p.add_argument("--presence-audit-dir", type=Path, default=PROJECT_DIR / "outputs/training_only_decodability_audit_v1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--inference-batch-size", type=int, default=64)
    p.add_argument("--sanity-epochs", type=int, default=1)
    p.add_argument("--review-approved", action="store_true")
    return p.parse_args()


def resolve_args(a: argparse.Namespace) -> argparse.Namespace:
    for name in ("project_root", "output_dir", "historical_aggregate_dir", "binary_plan_dir", "presence_audit_dir"):
        setattr(a, name, getattr(a, name).resolve())
    a.data_dir = a.data_dir.resolve() if a.data_dir else default_block_data_dir(a.project_root).resolve()
    return a


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_head(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def formal_protocol() -> dict[str, Any]:
    return {"output_version": OUTPUT_VERSION, "task": TASK_NAME, "historical_task": HISTORICAL_TASK_NAME, "mapping": CLASS_NAMES, "sessions": list(SESSIONS), "seeds": list(SEEDS), "outer_folds": EXPECTED_FOLDS, "outer_tasks": EXPECTED_OUTER_TASKS, "chance_ba": .5, "outer_model": {"reuse_historical_checkpoints": True, "new_outer_final_trainings": 0, "architecture": "MaxPool2d(2)->Flatten(16000)->Linear(3)->ReLU->Linear(2)", "parameters": 48011, "epochs": 40}, "inner_cross_fit": {"n_folds": 3, "group": "cycle", "required_new_trainings": EXPECTED_INNER_TRAININGS, "outer_test_access": False, "metric": "concatenate inner-OOF blocks then BA"}, "inner_training": asdict(FORMAL_TRAINING_CONFIG), "normalization": "arcsinh_then_inner_train_pixel_zscore", "fusion": "raw_logits -> softmax -> equal arithmetic mean of four frame probabilities", "confirmatory_gate": {"A_rho_min": .75, "B_exact_p_max": .05, "C_loo_median_min": .65, "C_loo_min_strictly_above": .30}, "automatic_next_stage": False}


def source_paths(root: Path) -> list[Path]:
    return [Path(__file__).resolve(), root / "src/ultrasound_decoding/multiframe/dqi_dot_vs_grating.py", root / "src/ultrasound_decoding/multiframe/cycle_calibrated_late_fusion.py", root / "src/ultrasound_decoding/multiframe/canonical_single_frame.py", root / "src/ultrasound_decoding/multiframe/training.py", root / "src/ultrasound_decoding/multiframe/models.py", root / "src/ultrasound_decoding/multiframe/dataset.py", root / "src/ultrasound_decoding/deep.py", root / "src/ultrasound_decoding/evaluate.py", root / "scripts/baselines/run_multiscale_temporal1d.py", root / "configs/dqi_dot_vs_grating_validation_v1.json", root / "docs/dqi_dot_vs_grating_validation_v1.md"]


def source_identity(a: argparse.Namespace) -> dict[str, Any]:
    paths = source_paths(a.project_root)
    missing = [str(x) for x in paths if not x.is_file()]
    if missing: raise FileNotFoundError(f"formal source provenance missing {missing}")
    hashes = {str(x.relative_to(a.project_root)): framework.file_sha256(x) for x in paths}
    return {"protocol": formal_protocol(), "source_hashes": hashes, "git_head": git_head(a.project_root), "runtime": framework.runtime_environment_signature()}


def _csv(path: Path) -> pd.DataFrame:
    if not path.is_file(): raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"session": str})


def historical_checkpoint_manifest(a: argparse.Namespace) -> pd.DataFrame:
    """Read checkpoint metadata only; this is legal before Q is frozen."""

    manifest = _csv(a.historical_aggregate_dir / "checkpoint_manifest.csv")
    required = {"session", "seed", "fold", "method", "status", "checkpoint_path", "checkpoint_sha256", "train_cycles", "test_cycles"}
    if not required.issubset(manifest): raise AssertionError("historical checkpoint manifest schema changed")
    fcnn = manifest[(manifest.method.eq("fcnn_late_fusion")) & manifest.status.eq("available") & manifest.checkpoint_path.notna() & manifest.checkpoint_path.astype(str).ne("")].copy()
    fcnn[["session", "seed", "fold"]] = fcnn[["session", "seed", "fold"]].astype({"session": str, "seed": int, "fold": int})
    fcnn = fcnn.sort_values(["session", "seed", "fold"]).reset_index(drop=True)
    if len(fcnn) != EXPECTED_OUTER_TASKS or fcnn[["session", "seed", "fold"]].duplicated().any(): raise AssertionError("must have exactly 246 available unique DG FCNN checkpoints")
    return fcnn


def load_historical_prediction_reference(
    a: argparse.Namespace, *, require_quality_frozen: bool
) -> pd.DataFrame:
    """First semantic target-table read, guarded by frozen-Q metadata in formal full."""

    if require_quality_frozen:
        frozen_path = a.output_dir / "QUALITY_FROZEN.json"
        quality_path = a.output_dir / "session_training_only_quality.csv"
        if not frozen_path.is_file() or not quality_path.is_file():
            raise RuntimeError("historical target reference is locked until Q_DG is frozen")
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if (
            frozen.get("status") != "frozen_before_target_reference_load"
            or frozen.get("session_quality_sha256") != framework.file_sha256(quality_path)
        ):
            raise AssertionError("QUALITY_FROZEN guard or session-Q SHA256 is invalid")
    predictions = _csv(a.historical_aggregate_dir / "multiframe_all_models_predictions.csv")
    predictions = predictions[predictions.method.eq("fcnn_late_fusion")].copy()
    predictions["session"] = predictions.session.astype(str)
    if len(predictions) != 684 or predictions[["session", "seed", "block_id"]].duplicated().any(): raise AssertionError("historical DG FCNN predictions must be 684 unique session/seed/block rows")
    if set(predictions.task.astype(str)) != {HISTORICAL_TASK_NAME} or not {"truth", "pred", "prob_dot", "prob_grating"}.issubset(predictions): raise AssertionError("historical DG FCNN predictions lack required task/probability fields")
    return predictions


def dataset_identity(a: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for session in SESSIONS:
        data = load_block_sequence_session(a.project_root, session, HISTORICAL_TASK_NAME, data_dir=a.data_dir)
        validate_dot_vs_grating_data(data)
        result[session] = {"n_blocks": int(len(data.y)), "cycles": sorted(map(int, np.unique(data.groups))), "h5_sha256": framework.file_sha256(data.source_h5_path), "metadata_sha256": framework.file_sha256(data.source_metadata_path)}
    return result


def build_plan(a: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validate_authoritative_mapping()
    fcnn = historical_checkpoint_manifest(a)
    binary = _csv(a.binary_plan_dir / "task_plan.csv")
    binary[["session", "seed", "fold"]] = binary[["session", "seed", "fold"]].astype({"session": str, "seed": int, "fold": int})
    left = fcnn.sort_values(["session", "seed", "fold"]); right = binary.sort_values(["session", "seed", "fold"])
    cols = ["session", "seed", "fold", "train_cycles", "test_cycles"]
    if not set(cols).issubset(right): raise AssertionError("binary task plan lacks outer-fold identity")
    pd.testing.assert_frame_equal(left[cols].reset_index(drop=True), right[cols].reset_index(drop=True), check_dtype=False)
    identity = source_identity(a); identity["datasets"] = dataset_identity(a); identity["historical_aggregate"] = {"checkpoint_manifest_sha256": framework.file_sha256(a.historical_aggregate_dir / "checkpoint_manifest.csv"), "predictions_sha256": framework.file_sha256(a.historical_aggregate_dir / "multiframe_all_models_predictions.csv")}
    run_fp = fingerprint(identity); rows = []
    for r in left.itertuples(index=False):
        train = cycle_text(str(r.train_cycles).split(",")); test = cycle_text(str(r.test_cycles).split(","))
        row = {"task_key": f"{r.session}:{r.seed}:{r.fold}", "session": str(r.session), "seed": int(r.seed), "fold": int(r.fold), "outer_train_cycles": train, "outer_test_cycles": test, "n_outer_train_blocks": 2 * len(train.split(",")), "n_outer_test_blocks": 2 * len(test.split(",")), "historical_checkpoint_path": str(r.checkpoint_path), "historical_checkpoint_sha256": str(r.checkpoint_sha256), "run_fingerprint": run_fp, "git_head": identity["git_head"]}
        row["task_fingerprint"] = fingerprint(row); rows.append(row)
    plan = pd.DataFrame(rows).sort_values(["session", "seed", "fold"]).reset_index(drop=True)
    inner = build_inner_manifest(plan, fingerprint(formal_protocol()), fingerprint(identity["source_hashes"]))
    expected_target_blocks = int(plan.n_outer_test_blocks.sum())
    expected_inner_oof_blocks = int(plan.n_outer_train_blocks.sum())
    if expected_target_blocks != 684 or expected_inner_oof_blocks != 5820:
        raise AssertionError("historical DG target/inner OOF block counts drifted")
    return plan, inner, {"identity": identity, "run_fingerprint": run_fp, "expected_target_blocks": expected_target_blocks, "expected_inner_oof_blocks": expected_inner_oof_blocks}


def write_plan(a: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    plan, inner, meta = build_plan(a); a.output_dir.mkdir(parents=True, exist_ok=True)
    fcnn = historical_checkpoint_manifest(a)
    framework.atomic_json(a.output_dir / "config.json", {**formal_protocol(), "run_fingerprint": meta["run_fingerprint"]})
    framework.atomic_text(a.output_dir / "DQI_DOT_VS_GRATING_VALIDATION.md", (a.project_root / "docs/dqi_dot_vs_grating_validation_v1.md").read_text(encoding="utf-8"))
    framework.atomic_json(a.output_dir / "runtime_fingerprint.json", {**meta["identity"], "run_fingerprint": meta["run_fingerprint"], "source_hashes_authoritative": True})
    framework.atomic_csv(a.output_dir / "task_plan.csv", plan); framework.atomic_csv(a.output_dir / "inner_split_manifest.csv", inner); framework.atomic_csv(a.output_dir / "historical_fcnn_checkpoint_manifest.csv", fcnn)
    framework.atomic_json(a.output_dir / "PLAN_COMPLETE.json", {"status": "complete", "created_utc": now(), "sessions": len(SESSIONS), "outer_tasks": len(plan), "folds": EXPECTED_FOLDS, "seeds": len(SEEDS), "dot_grating_blocks": int(sum(v["n_blocks"] for v in meta["identity"]["datasets"].values())), "outer_target_block_predictions": meta["expected_target_blocks"], "inner_oof_block_predictions": meta["expected_inner_oof_blocks"], "historical_checkpoints_expected": EXPECTED_OUTER_TASKS, "historical_checkpoints_found": len(fcnn), "planned_inner_trainings": len(inner), "historical_legal_inner_oof_artifacts": 0, "new_inner_trainings_required": len(inner), "outer_final_model_trainings": 0, "formal_training_started": False, "run_fingerprint": meta["run_fingerprint"]})
    print(f"PLAN COMPLETE outer_tasks={len(plan)} inner_trainings={len(inner)} historical_inner_oof=0 formal_started=False", flush=True)
    return plan, inner, meta


def load_strict_plan(a: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    plan, inner, meta = build_plan(a)
    for name in ("config.json", "runtime_fingerprint.json", "task_plan.csv", "inner_split_manifest.csv", "historical_fcnn_checkpoint_manifest.csv", "PLAN_COMPLETE.json"):
        if not (a.output_dir / name).is_file(): raise RuntimeError(f"plan artifact absent: {name}; run --stage plan")
    pd.testing.assert_frame_equal(_csv(a.output_dir / "task_plan.csv"), plan, check_dtype=False)
    pd.testing.assert_frame_equal(_csv(a.output_dir / "inner_split_manifest.csv"), inner, check_dtype=False)
    runtime = json.loads((a.output_dir / "runtime_fingerprint.json").read_text())
    if runtime.get("run_fingerprint") != meta["run_fingerprint"] or runtime.get("source_hashes") != meta["identity"]["source_hashes"]: raise AssertionError("plan provenance changed")
    return plan, inner, meta


def _indices(data: Any, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_cycles = np.array([int(x) for x in str(row["outer_train_cycles"]).split(",")]); test_cycles = np.array([int(x) for x in str(row["outer_test_cycles"]).split(",")])
    train, test = np.flatnonzero(np.isin(data.groups, train_cycles)), np.flatnonzero(np.isin(data.groups, test_cycles))
    if len(train) != row["n_outer_train_blocks"] or len(test) != row["n_outer_test_blocks"] or set(data.groups[train]) & set(data.groups[test]): raise AssertionError("outer fold membership drift")
    return train, test


def _task_dir(base: Path, row: dict[str, Any]) -> Path:
    return base / "tasks" / f"session_{row['session']}" / f"seed_{int(row['seed'])}" / f"fold_{int(row['fold']):02d}"


def _task_hashes(path: Path) -> dict[str, str]:
    return {n: framework.file_sha256(path / n) for n in REQUIRED_TASK_FILES}


def validation_mutation_normalization_audit(
    blocks: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
) -> dict[str, Any]:
    """Actually mutate validation pixels in a full copy, then refit from train indices."""

    values = np.asarray(blocks, dtype=np.float32)
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if not len(train) or not len(validation) or set(train) & set(validation):
        raise ValueError("mutation audit requires disjoint nonempty train/validation indices")
    mean, std, before = fit_inner_train_normalization(values[train])
    mutated = values.copy()
    mutated[validation] = mutated[validation] + np.float32(1000.0)
    if np.array_equal(mutated[validation], values[validation]):
        raise AssertionError("validation mutation was not applied")
    after_mean, after_std, after = fit_inner_train_normalization(mutated[train])
    unchanged = bool(
        before == after
        and np.array_equal(mean, after_mean)
        and np.array_equal(std, after_std)
    )
    return {
        "validation_pixels_actually_changed": True,
        "train_pixels_unchanged": bool(np.array_equal(mutated[train], values[train])),
        "normalization_arrays_unchanged": unchanged,
        "normalization_fingerprint_before": before,
        "normalization_fingerprint_after": after,
    }


def write_q_task(a: argparse.Namespace, row: dict[str, Any], inner: pd.DataFrame, *, training_config: Any = FORMAL_TRAINING_CONFIG, destination_base: Path | None = None) -> Path:
    root = destination_base or a.output_dir; dst = _task_dir(root, row); dst.mkdir(parents=True, exist_ok=True)
    data = load_block_sequence_session(a.project_root, str(row["session"]), HISTORICAL_TASK_NAME, data_dir=a.data_dir); validate_dot_vs_grating_data(data)
    outer_train, _ = _indices(data, row); task_inner = inner[inner.task_key.eq(str(row["task_key"]))].sort_values("inner_fold")
    if len(task_inner) != N_INNER_FOLDS: raise AssertionError("task has not exactly three inner splits")
    runtime = json.loads((a.output_dir / "runtime_fingerprint.json").read_text()); source_hash = fingerprint(runtime["source_hashes"]); protocol_hash = fingerprint(formal_protocol())
    frames, sums, split_rows = [], [], []
    for split in task_inner.to_dict("records"):
        tr_cycles = [int(x) for x in str(split["inner_train_cycles"]).split(",")]; va_cycles = [int(x) for x in str(split["inner_validation_cycles"]).split(",")]
        tr = outer_train[np.isin(data.groups[outer_train], tr_cycles)]; va = outer_train[np.isin(data.groups[outer_train], va_cycles)]
        if set(data.groups[tr]) & set(data.groups[va]) or set(data.groups[va]) & set(data.groups[np.setdiff1d(np.arange(len(data.y)), outer_train)]): raise AssertionError("inner cycle isolation/outer-test exclusion failed")
        mean, std, norm_fp = fit_inner_train_normalization(data.X[tr])
        mutation = validation_mutation_normalization_audit(data.X, tr, va)
        if not mutation["train_pixels_unchanged"] or not mutation["normalization_arrays_unchanged"] or mutation["normalization_fingerprint_before"] != norm_fp:
            raise AssertionError("validation mutation changed train-only normalization")
        key = build_task_inner_cache_key(task=TASK_NAME, session=str(row["session"]), outer_fold=int(row["fold"]), outer_seed=int(row["seed"]), outer_train_cycles=[int(x) for x in str(row["outer_train_cycles"]).split(",")], inner_fold=int(split["inner_fold"]), inner_train_cycles=tr_cycles, inner_validation_cycles=va_cycles, source_hash=source_hash, protocol_hash=protocol_hash, normalization_fingerprint=norm_fp, training_config=asdict(training_config))
        logits, history, seen_fp = train_inner_fcnn(data.X[tr], data.y[tr], data.X[va], seed=int(row["seed"]), device=str(a.device), training_config=training_config)
        if seen_fp != norm_fp: raise AssertionError("training normalization fingerprint differs")
        logits = logits.reshape(len(va), FRAMES_PER_BLOCK, 2)
        for i, idx in enumerate(va):
            md = data.metadata.iloc[int(idx)]
            for position in range(FRAMES_PER_BLOCK): frames.append({"session": str(row["session"]), "outer_seed": int(row["seed"]), "outer_fold": int(row["fold"]), "inner_fold": int(split["inner_fold"]), "source_index": int(idx), "block_id": str(md.block_id), "cycle": int(data.groups[idx]), "frame_position": position, "truth": int(data.y[idx]), "logit_dot": float(logits[i, position, 0]), "logit_grating": float(logits[i, position, 1]), "heldout_cycle": int(data.groups[idx]), "cache_key": key})
        sums.append({"task_key": row["task_key"], "inner_fold": int(split["inner_fold"]), "n_train_blocks": len(tr), "n_validation_blocks": len(va), "trained_epochs": len(history), "final_train_loss": float(history[-1]["train_loss"]), "normalization_fingerprint": norm_fp, "cache_key": key, "outer_test_used": False})
        split_rows.append({**split, "normalization_fingerprint": norm_fp, "cache_key": key, **mutation})
    oof = pd.DataFrame(frames).sort_values(["source_index", "frame_position"]).reset_index(drop=True)
    assert_complete_inner_oof(oof.source_index.to_numpy(int), outer_train, oof.cycle.to_numpy(int), oof.heldout_cycle.to_numpy(int))
    block = block_predictions_from_frame_logits(oof); q = concatenated_oof_balanced_accuracy(block)
    result = {"task_key": row["task_key"], "task_fingerprint": row["task_fingerprint"], "run_fingerprint": row["run_fingerprint"], "Q_DG_concatenated_inner_oof_block_BA": q, "mean_inner_fold_ba_diagnostic": mean_inner_fold_ba_diagnostic(block), "metric_definition": "concatenate all inner-OOF block predictions then balanced accuracy", "outer_test_read": False, "trained_epochs": int(training_config.max_epochs)}
    framework.atomic_json(dst / "result.json", result); framework.atomic_csv(dst / "inner_split_manifest.csv", pd.DataFrame(split_rows)); framework.atomic_csv(dst / "inner_training_summary.csv", pd.DataFrame(sums)); framework.atomic_csv(dst / "inner_oof_logits.csv", oof); framework.atomic_csv(dst / "inner_oof_predictions.csv", block); framework.atomic_json(dst / "leakage_audit.json", {"outer_test_used": False, "inner_cycle_grouped": True, "validation_pixels_actually_changed": True, "train_pixels_unchanged": True, "validation_mutation_normalization_unchanged": True})
    framework.atomic_json(dst / "COMPLETE.json", {"status": "complete", "task_fingerprint": row["task_fingerprint"], "artifact_sha256": _task_hashes(dst)})
    return dst


def _historical_outer(a: argparse.Namespace, row: dict[str, Any], data: Any) -> pd.DataFrame:
    _, test = _indices(data, row)
    model, payload, _ = load_validated_checkpoint(Path(row["historical_checkpoint_path"]), expected_sha256=str(row["historical_checkpoint_sha256"]), expected_session=str(row["session"]), expected_seed=int(row["seed"]), expected_fold=int(row["fold"]), expected_train_cycles=str(row["outer_train_cycles"]), expected_test_cycles=str(row["outer_test_cycles"]), expected_task=HISTORICAL_TASK_NAME)
    model_devices = {parameter.device.type for parameter in model.parameters()}
    if model_devices != {HISTORICAL_OUTER_INFERENCE_DEVICE}:
        raise RuntimeError(f"historical checkpoint model must remain CPU-only, got {model_devices}")
    normalized = apply_saved_normalization(data.X[test].reshape(-1, 128, 501), payload["normalization_mean"], payload["normalization_std"], transform=NORMALIZATION_TRANSFORM).reshape(len(test), 4, 128, 501)
    probs = softmax_probabilities(predict_raw_logits(model, normalized, device=HISTORICAL_OUTER_INFERENCE_DEVICE, batch_size=a.inference_batch_size)).reshape(len(test), 4, 2).mean(1)
    return pd.DataFrame({"session": str(row["session"]), "seed": int(row["seed"]), "fold": int(row["fold"]), "block_id": data.metadata.iloc[test].block_id.astype(str).to_numpy(), "truth": data.y[test].astype(int), "prob_dot": probs[:,0], "prob_grating": probs[:,1], "prediction": probs.argmax(1)})


def historical_reconstruction_audit(
    reconstructed: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    expected_rows: int,
    tolerance: float = BASELINE_ATOL,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Align exact sample identities and audit truth, predictions, and probabilities."""

    keys = ["session", "seed", "block_id"]
    if "fold" in reconstructed.columns and "fold" in reference.columns:
        keys.insert(2, "fold")
    for frame in (reconstructed, reference):
        missing = set(keys) - set(frame.columns)
        if missing:
            raise ValueError(f"reconstruction input lacks identity columns {sorted(missing)}")
    duplicate_identity_count = int(reconstructed.duplicated(keys).sum() + reference.duplicated(keys).sum())
    identity = reconstructed[keys].merge(reference[keys], on=keys, how="outer", indicator=True)
    identity_mismatch_count = int(identity["_merge"].ne("both").sum() + duplicate_identity_count)
    merged = reconstructed.merge(reference, on=keys, how="inner", suffixes=("_reconstructed", "_historical"), validate="one_to_one" if duplicate_identity_count == 0 else "many_to_many")
    truth_mismatch_count = int((merged["truth_reconstructed"].astype(int) != merged["truth_historical"].astype(int)).sum()) if len(merged) else int(expected_rows)
    prediction_mismatch_count = int((merged["prediction"].astype(int) != merged["pred"].astype(int)).sum()) if len(merged) else int(expected_rows)
    probability_columns = ["prob_dot", "prob_grating"]
    differences = [np.abs(merged[f"{name}_reconstructed"].to_numpy(float) - merged[f"{name}_historical"].to_numpy(float)) for name in probability_columns]
    max_probability_abs_diff = float(np.max(np.concatenate(differences))) if len(merged) else None
    row_count_valid = len(reconstructed) == len(reference) == len(merged) == int(expected_rows)
    probability_valid = max_probability_abs_diff is not None and max_probability_abs_diff <= float(tolerance)
    status = "PASS" if row_count_valid and identity_mismatch_count == 0 and truth_mismatch_count == 0 and prediction_mismatch_count == 0 and probability_valid else "FAIL"
    return ({"n_rows": int(len(merged)), "identity_keys": keys, "identity_mismatch_count": identity_mismatch_count, "truth_mismatch_count": truth_mismatch_count, "prediction_mismatch_count": prediction_mismatch_count, "max_probability_abs_diff": max_probability_abs_diff, "tolerance": float(tolerance), "status": status}, merged)


def require_reconstruction_pass(audit: dict[str, Any]) -> None:
    if audit.get("status") != "PASS":
        raise AssertionError(f"historical DG reconstruction audit failed: {audit}")


def session_target_from_oof(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute one OOF BA per session/seed, then arithmetic mean the three seeds."""

    rows = [{"session": str(session), "seed": int(seed), "BA_DG_seed_OOF": float(balanced_accuracy_score(group.truth, group.prediction))} for (session, seed), group in targets.groupby(["session", "seed"], sort=True)]
    seed_ba = pd.DataFrame(rows)
    if seed_ba[["session", "seed"]].duplicated().any() or not seed_ba.groupby("session").seed.nunique().eq(len(SEEDS)).all():
        raise AssertionError("formal target requires exactly three seed-level OOF BAs per session")
    session = seed_ba.groupby("session", as_index=False).BA_DG_seed_OOF.mean().rename(columns={"BA_DG_seed_OOF": "BA_DG_session"})
    return seed_ba, session


def run_sanity(a: argparse.Namespace) -> None:
    if str(a.device) != "cpu": raise ValueError("sanity is CPU-only; pass --device cpu")
    if a.sanity_epochs < 1: raise ValueError("sanity epochs must be >= 1")
    plan, inner, _ = load_strict_plan(a); row = plan.iloc[0].to_dict(); cfg = replace(FORMAL_TRAINING_CONFIG, max_epochs=int(a.sanity_epochs))
    sanity_root = a.output_dir / "sanity"; task = write_q_task(a, row, inner, training_config=cfg, destination_base=sanity_root)
    data = load_block_sequence_session(a.project_root, str(row["session"]), HISTORICAL_TASK_NAME, data_dir=a.data_dir); validate_dot_vs_grating_data(data)
    reconstructed = _historical_outer(a, row, data)
    historical = load_historical_prediction_reference(a, require_quality_frozen=False)
    ref = historical[(historical.session.astype(str).eq(str(row["session"]))) & (historical.seed.astype(int).eq(int(row["seed"]))) & (historical.fold.astype(int).eq(int(row["fold"]))) & historical.block_id.astype(str).isin(reconstructed.block_id)].reset_index(drop=True)
    audit, _ = historical_reconstruction_audit(reconstructed, ref, expected_rows=len(reconstructed))
    require_reconstruction_pass(audit)
    framework.atomic_json(a.output_dir / "SANITY_COMPLETE.json", {"status": "complete", "created_utc": now(), "device": "cpu", "sanity_epochs": a.sanity_epochs, "trained_inner_models": 3, "outer_final_model_trainings": 0, "task_dir": str(task), "historical_inference_device": HISTORICAL_OUTER_INFERENCE_DEVICE, "historical_reconstruction": audit, "formal_full_started": False})
    print("SANITY COMPLETE cpu inner_models=3 historical_reconstruction=PASS formal_full_started=False", flush=True)


def _complete_task(path: Path, row: dict[str, Any]) -> bool:
    try:
        c = json.loads((path / "COMPLETE.json").read_text())
        if not (c.get("status") == "complete" and c.get("task_fingerprint") == row["task_fingerprint"] and c.get("artifact_sha256") == _task_hashes(path)):
            return False
        result = json.loads((path / "result.json").read_text())
        splits, oof = _csv(path / "inner_split_manifest.csv"), _csv(path / "inner_oof_predictions.csv")
        if result.get("task_fingerprint") != row["task_fingerprint"] or result.get("run_fingerprint") != row["run_fingerprint"] or result.get("trained_epochs") != 40:
            return False
        if len(splits) != N_INNER_FOLDS or splits.cache_key.duplicated().any() or not splits.outer_test_used.eq(False).all():
            return False
        if len(oof) != int(row["n_outer_train_blocks"]) or oof.block_id.duplicated().any() or not oof.n_frames_fused.eq(FRAMES_PER_BLOCK).all():
            return False
        return abs(float(result["Q_DG_concatenated_inner_oof_block_BA"]) - concatenated_oof_balanced_accuracy(oof)) < 1e-12
    except Exception: return False


def summarize_training_only_quality(
    quality: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-average within outer fold, then fold-average to one Q per session."""

    required = {"session", "seed", "fold", "Q_DG_concatenated_inner_oof_block_BA"}
    if not required.issubset(quality.columns) or quality[["session", "seed", "fold"]].duplicated().any():
        raise AssertionError("training-only task-quality table is incomplete or duplicated")
    if not quality.groupby(["session", "fold"]).seed.nunique().eq(len(SEEDS)).all():
        raise AssertionError("each outer fold requires exactly three seed Q values")
    fold_q = quality.groupby(["session", "fold"], as_index=False).Q_DG_concatenated_inner_oof_block_BA.mean().rename(columns={"Q_DG_concatenated_inner_oof_block_BA": "Q_DG_seed_averaged"})
    session_q = fold_q.groupby("session", as_index=False).Q_DG_seed_averaged.mean().rename(columns={"Q_DG_seed_averaged": "Q_DG_session"})
    direct_q = quality.groupby("session", as_index=False).Q_DG_concatenated_inner_oof_block_BA.mean().rename(columns={"Q_DG_concatenated_inner_oof_block_BA": "direct_mean_all_fold_seed_Q"})
    session_q = session_q.merge(direct_q, on="session", validate="one_to_one")
    if not np.allclose(session_q.Q_DG_session, session_q.direct_mean_all_fold_seed_Q, atol=1e-12):
        raise AssertionError("seed-within-fold Q aggregation is not equivalent to direct mean")
    return fold_q, session_q


def freeze_training_only_quality(
    a: argparse.Namespace, plan: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate completed Phase-1 tasks and atomically freeze session Q values."""

    task_results = [json.loads((_task_dir(a.output_dir, row) / "result.json").read_text()) for row in plan.to_dict("records")]
    quality = pd.DataFrame(task_results).merge(plan[["task_key", "session", "seed", "fold"]], on="task_key", validate="one_to_one")
    inner_blocks = pd.concat([_csv(_task_dir(a.output_dir, row) / "inner_oof_predictions.csv") for row in plan.to_dict("records")], ignore_index=True)
    completed_inner = pd.concat([_csv(_task_dir(a.output_dir, row) / "inner_split_manifest.csv") for row in plan.to_dict("records")], ignore_index=True)
    if len(completed_inner) != EXPECTED_INNER_TRAININGS or len(inner_blocks) != 5820:
        raise AssertionError("formal inner OOF coverage/count drift")
    framework.atomic_csv(a.output_dir / "outer_task_quality.csv", quality)
    framework.atomic_csv(a.output_dir / "inner_oof_manifest.csv", completed_inner)
    framework.atomic_csv(a.output_dir / "inner_oof_predictions.csv", inner_blocks)
    fold_q, session_q = summarize_training_only_quality(quality)
    quality_path = a.output_dir / "session_training_only_quality.csv"
    framework.atomic_csv(quality_path, session_q)
    framework.atomic_json(a.output_dir / "QUALITY_FROZEN.json", {"status": "frozen_before_target_reference_load", "created_utc": now(), "session_quality_sha256": framework.file_sha256(quality_path), "target_reference_loader_guard": "requires this file and exact session-Q SHA256", "outer_test_read_before_freeze": False})
    return quality, fold_q, session_q


def run_full(a: argparse.Namespace) -> None:
    if not a.review_approved: raise PermissionError("formal full requires explicit --review-approved")
    plan, inner, _ = load_strict_plan(a)
    # Phase 1: only outer-training data are accessed.
    for row in plan.to_dict("records"):
        path = _task_dir(a.output_dir, row)
        if not _complete_task(path, row): write_q_task(a, row, inner)
    quality, fold_q, session_q = freeze_training_only_quality(a, plan)
    # Phase 2 starts only after Q is frozen: exact historical outer checkpoint reconstruction.
    targets = []
    for row in plan.to_dict("records"):
        data = load_block_sequence_session(a.project_root, str(row["session"]), HISTORICAL_TASK_NAME, data_dir=a.data_dir); validate_dot_vs_grating_data(data); targets.append(_historical_outer(a, row, data))
    targets = pd.concat(targets, ignore_index=True)
    reference = load_historical_prediction_reference(a, require_quality_frozen=True)
    reconstruction_audit, _ = historical_reconstruction_audit(targets, reference, expected_rows=684)
    framework.atomic_json(a.output_dir / "outer_target_reconstruction_audit.json", reconstruction_audit)
    require_reconstruction_pass(reconstruction_audit)
    framework.atomic_csv(a.output_dir / "outer_target_predictions.csv", targets)
    seed_ba, target_session = session_target_from_oof(targets)
    summary = session_q.merge(target_session, on="session", validate="one_to_one")
    fold_ba_rows = [{"session": str(session), "fold": int(fold), "BA_DG_seed_averaged": float(np.mean([balanced_accuracy_score(g.truth, g.prediction) for _, g in group.groupby("seed", sort=True)]))} for (session, fold), group in targets.groupby(["session", "fold"], sort=True)]
    fold_ba = pd.DataFrame(fold_ba_rows)
    outer_fold = fold_q.merge(fold_ba, on=["session", "fold"], validate="one_to_one")
    framework.atomic_csv(a.output_dir / "outer_fold_seed_averaged.csv", outer_fold)
    within_rows = [{"session": str(session), "n_folds": len(group), "pearson_r": finite_pearson(group.Q_DG_seed_averaged, group.BA_DG_seed_averaged), "spearman_rho": finite_spearman(group.Q_DG_seed_averaged, group.BA_DG_seed_averaged), "role": "DESCRIPTIVE_ONLY"} for session, group in outer_fold.groupby("session", sort=True)]
    framework.atomic_csv(a.output_dir / "within_session_fold_relationship.csv", pd.DataFrame(within_rows))
    rho, pearson = finite_spearman(summary.Q_DG_session, summary.BA_DG_session), finite_pearson(summary.Q_DG_session, summary.BA_DG_session)
    exact = exact_spearman_permutation(summary.Q_DG_session, summary.BA_DG_session); loso = leave_one_session_out(summary, "Q_DG_session", "BA_DG_session")
    gate = evaluate_confirmatory_gate(session_spearman=rho, exact_p=exact.two_sided_p, loo_median=float(loso.spearman_rho.median()), loo_minimum=float(loso.spearman_rho.min()))
    presence = _csv(a.presence_audit_dir / "session_quality_summary.csv").rename(columns={"Q_session":"Q_presence_session", "formal_session_FCNN_latefusion_BA":"BA_presence_session"})
    summary = summary.merge(presence[["session", "Q_presence_session", "BA_presence_session"]], on="session", validate="one_to_one")
    framework.atomic_csv(a.output_dir / "session_quality_summary.csv", summary); framework.atomic_csv(a.output_dir / "loo_robustness.csv", loso); framework.atomic_csv(a.output_dir / "cross_task_relationship_matrix.csv", cross_task_relationship_matrix(summary))
    framework.atomic_json(a.output_dir / "statistical_audit.json", {"primary": {"spearman_rho": rho, "pearson_r": pearson, "exact_permutation": exact.as_dict(), "gate": gate}, "chance_BA": .5, "Q_DG_distribution": {"mean": float(summary.Q_DG_session.mean()), "sd": float(summary.Q_DG_session.std(ddof=1)), "minimum": float(summary.Q_DG_session.min()), "median": float(summary.Q_DG_session.median()), "maximum": float(summary.Q_DG_session.max())}, "formal_DG_BA_distribution": {"mean": float(summary.BA_DG_session.mean()), "sd": float(summary.BA_DG_session.std(ddof=1)), "minimum": float(summary.BA_DG_session.min()), "median": float(summary.BA_DG_session.median()), "maximum": float(summary.BA_DG_session.max())}, "descriptive_only": {"fold_level_pearson": finite_pearson(outer_fold.Q_DG_seed_averaged, outer_fold.BA_DG_seed_averaged), "fold_level_spearman": finite_spearman(outer_fold.Q_DG_seed_averaged, outer_fold.BA_DG_seed_averaged), "within_session": "not confirmatory"}})
    framework.atomic_json(a.output_dir / "provenance_audit.json", {"quality_frozen_before_outer_target_read": True, "outer_final_model_trainings": 0, "historical_outer_run_path": str(a.historical_aggregate_dir), "checkpoint_manifest_sha256": framework.file_sha256(a.historical_aggregate_dir / "checkpoint_manifest.csv"), "historical_aggregate_sha256": framework.file_sha256(a.historical_aggregate_dir / "multiframe_all_models_predictions.csv"), "reconstructed_prediction_sha256": framework.file_sha256(a.output_dir / "outer_target_predictions.csv"), "inner_split_manifest_sha256": framework.file_sha256(a.output_dir / "inner_oof_manifest.csv"), "inner_oof_asset_sha256": framework.file_sha256(a.output_dir / "inner_oof_predictions.csv"), "formal_target_frozen_sha256": framework.file_sha256(a.output_dir / "outer_target_predictions.csv"), "presence_Q_source_sha256": framework.file_sha256(a.presence_audit_dir / "session_quality_summary.csv"), "historical_outer_checkpoint_reconstruction": "PASS", "mapping": CLASS_NAMES})
    hashes = {name: framework.file_sha256(a.output_dir / name) for name in REQUIRED_RUN_OUTPUTS if (a.output_dir / name).is_file()}
    framework.atomic_json(a.output_dir / "RUN_COMPLETE.json", {"status": "complete", "created_utc": now(), "aggregate_artifact_sha256": hashes, "gate": gate})


def run_status(a: argparse.Namespace) -> None:
    plan, _, _ = load_strict_plan(a); complete = a.output_dir / "RUN_COMPLETE.json"
    if complete.is_file():
        record = json.loads(complete.read_text()); expected = record.get("aggregate_artifact_sha256", {}); bad = [n for n, h in expected.items() if not (a.output_dir / n).is_file() or framework.file_sha256(a.output_dir / n) != h]
        if bad: raise AssertionError(f"RUN INVALID aggregate integrity failed: {bad}")
        print("STATUS complete integrity=PASS", flush=True); return
    done = sum(_complete_task(_task_dir(a.output_dir, r), r) for r in plan.to_dict("records"))
    print(f"STATUS planned={len(plan)} completed_Q_tasks={done} formal_complete=False", flush=True)


def main() -> None:
    a = resolve_args(parse_args())
    if a.stage == "plan": write_plan(a)
    elif a.stage == "sanity": run_sanity(a)
    elif a.stage == "full": run_full(a)
    else: run_status(a)


if __name__ == "__main__": main()
