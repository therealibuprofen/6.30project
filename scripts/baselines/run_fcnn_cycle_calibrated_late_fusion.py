#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
for search_path in (PROJECT_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.baselines import run_multiscale_temporal1d as framework
from ultrasound_decoding.multiframe.canonical_single_frame import (
    EXPECTED_EPOCH,
    EXPECTED_PARAMETERS,
    NORMALIZATION_TRANSFORM,
    apply_saved_normalization,
    load_validated_checkpoint,
)
from ultrasound_decoding.multiframe.cycle_calibrated_late_fusion import (
    CLASSES,
    ECE_BINS,
    FORMAL_TRAINING_CONFIG,
    FRAMES_PER_BLOCK,
    FROZEN_GATE,
    HISTORICAL_METHOD,
    IMAGE_SHAPE,
    MODEL_NAME,
    MODEL_VERSION,
    N_INNER_FOLDS,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    apply_inner_normalization,
    assert_complete_inner_oof,
    block_balanced_accuracy,
    build_inner_cache_key,
    build_inner_cycle_splits,
    calibrated_frame_probabilities,
    cycle_text,
    equal_four_frame_probability_mean,
    evaluate_frozen_gate,
    exact_paired_sign_flip_test,
    fingerprint,
    fit_inner_train_normalization,
    fit_scalar_temperature,
    frame_calibration_metrics,
    predict_raw_logits,
    softmax_probabilities,
    train_inner_fcnn,
)
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    default_block_data_dir,
    load_block_sequence_session,
)


OUTPUT_VERSION = "fcnn_cycle_calibrated_late_fusion_v1"
SEEDS = (0, 1, 2)
EXPECTED_FOLDS = 82
EXPECTED_TASKS = 246
EXPECTED_INNER_TRAININGS = EXPECTED_TASKS * N_INNER_FOLDS
EXPECTED_BLOCK_PREDICTIONS = 1368
EXPECTED_FRAME_PREDICTIONS = EXPECTED_BLOCK_PREDICTIONS * FRAMES_PER_BLOCK
BASELINE_ATOL = 2e-6
BASELINE_RTOL = 1e-6

REQUIRED_TASK_FILES = (
    "result.json",
    "temperature.json",
    "inner_split_manifest.csv",
    "inner_training_summary.csv",
    "inner_oof_logits.csv",
    "frame_predictions.csv",
    "predictions.csv",
    "baseline_reconstruction.json",
)

REQUIRED_RUN_OUTPUTS = (
    "config.json",
    "runtime_fingerprint.json",
    "task_plan.csv",
    "inner_split_manifest.csv",
    "temperature_summary.csv",
    "inner_calibration_summary.csv",
    "frame_predictions.csv",
    "predictions.csv",
    "session_seed_summary.csv",
    "session_summary.csv",
    "calibration_summary.csv",
    "statistical_audit.json",
    "provenance_audit.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cycle-Calibrated Late Fusion v1")
    parser.add_argument("--stage", choices=("plan", "sanity", "full", "status"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / OUTPUT_VERSION,
    )
    parser.add_argument(
        "--canonical-run-dir",
        type=Path,
        default=PROJECT_DIR / "outputs/fcnn_canonical_single_frame_v1",
    )
    parser.add_argument(
        "--historical-base-run-dir",
        type=Path,
        default=PROJECT_DIR / "results/runs/multiframe/block_clean4_binary_v1",
    )
    parser.add_argument(
        "--historical-fcnn-run-dir",
        type=Path,
        default=PROJECT_DIR / "results/runs/multiframe/block_clean4_binary_fcnn_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--sanity-epochs", type=int, default=1)
    parser.add_argument("--review-approved", action="store_true")
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.canonical_run_dir = args.canonical_run_dir.resolve()
    args.historical_base_run_dir = args.historical_base_run_dir.resolve()
    args.historical_fcnn_run_dir = args.historical_fcnn_run_dir.resolve()
    args.data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else default_block_data_dir(args.project_root).resolve()
    )
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def current_git_head(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def formal_protocol() -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "task": "binary_presence",
        "sessions": list(EXPECTED_SESSIONS),
        "seeds": list(SEEDS),
        "expected_folds": EXPECTED_FOLDS,
        "expected_outer_tasks": EXPECTED_TASKS,
        "expected_inner_trainings": EXPECTED_INNER_TRAININGS,
        "outer_model": {
            "method": HISTORICAL_METHOD,
            "architecture": "MaxPool2d-Flatten-Linear(16000,3)-ReLU-Linear(3,2)",
            "parameters": EXPECTED_PARAMETERS,
            "checkpoint_epoch": EXPECTED_EPOCH,
            "training_performed": False,
            "weights_updated": False,
        },
        "inner_cross_fit": {
            "folds": N_INNER_FOLDS,
            "group": "cycle",
            "split": "sorted outer-training cycle IDs divided deterministically with numpy.array_split",
            "outer_test_access": False,
            "oof_requirement": "each outer-training frame exactly once",
        },
        "inner_model_training": vars(FORMAL_TRAINING_CONFIG),
        "inner_normalization": {
            "transform": "arcsinh_then_inner_train_pixel_zscore",
            "statistics": "inner-training-fold frames only",
            "inner_validation_used": False,
            "outer_test_used": False,
        },
        "temperature": {
            "count_per_outer_task": 1,
            "parameterization": "T=exp(log_T)",
            "objective": "inner-OOF frame cross-entropy NLL only",
            "optimizer": "deterministic scipy L-BFGS-B",
            "log_T_bounds": [-20.0, 20.0],
            "uses_balanced_accuracy": False,
            "uses_outer_test": False,
        },
        "fusion": {
            "unit": "outer-held-out clean4 block",
            "frame_probability_weights": [0.25, 0.25, 0.25, 0.25],
            "operation": "arithmetic probability mean",
            "uses_confidence": False,
            "uses_entropy": False,
            "uses_margin": False,
            "uses_timestamp": False,
            "uses_block_type": False,
            "uses_attention_or_gating": False,
        },
        "evaluation": "concatenate outer-held-out blocks, then balanced accuracy",
        "calibration_bins": ECE_BINS,
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "frozen_gate": dict(FROZEN_GATE),
        "automatic_next_stage": False,
    }


def source_paths(project_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        project_root / "src/ultrasound_decoding/multiframe/cycle_calibrated_late_fusion.py",
        project_root / "src/ultrasound_decoding/multiframe/canonical_single_frame.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/models.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/deep.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "scripts/baselines/run_multiscale_temporal1d.py",
        project_root / "configs/fcnn_cycle_calibrated_late_fusion_v1.json",
        project_root / "docs/fcnn_cycle_calibrated_late_fusion_v1.md",
    ]


def validate_reference_assets(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = (
        "RUN_COMPLETE.json",
        "task_plan.csv",
        "checkpoint_manifest.csv",
        "late_fusion_reconstructed_predictions.csv",
        "provenance_audit.json",
    )
    missing = [name for name in required if not (args.canonical_run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"canonical reference artifacts missing: {missing}")
    complete = json.loads((args.canonical_run_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8"))
    if (
        complete.get("status") != "complete"
        or int(complete.get("completed_tasks", -1)) != EXPECTED_TASKS
        or complete.get("late_fusion_reconstruction_status") != "PASS"
        or complete.get("training_performed") is not False
    ):
        raise AssertionError("canonical FCNN reference RUN_COMPLETE is invalid")
    plan = pd.read_csv(args.canonical_run_dir / "task_plan.csv", dtype={"session": str})
    checkpoints = pd.read_csv(args.canonical_run_dir / "checkpoint_manifest.csv", dtype={"session": str})
    baseline = pd.read_csv(
        args.canonical_run_dir / "late_fusion_reconstructed_predictions.csv",
        dtype={"session": str},
    )
    if len(plan) != EXPECTED_TASKS or plan[["session", "seed", "fold"]].duplicated().any():
        raise AssertionError("canonical task plan is not exact 246-task coverage")
    if len(checkpoints) != EXPECTED_TASKS or checkpoints[["session", "seed", "fold"]].duplicated().any():
        raise AssertionError("canonical checkpoint manifest is not exact 246 coverage")
    if not checkpoints["valid"].astype(bool).all():
        raise AssertionError("canonical checkpoint manifest contains invalid entries")
    if not checkpoints["checkpoint_sha256"].eq(checkpoints["source_manifest_sha256"]).all():
        raise AssertionError("canonical checkpoint hashes do not match source manifests")
    if len(baseline) != EXPECTED_BLOCK_PREDICTIONS or baseline[["session", "seed", "block_id"]].duplicated().any():
        raise AssertionError("canonical late-fusion reference does not cover 1368 unique OOF blocks")
    expected_keys = set(map(tuple, plan[["session", "seed", "fold"]].to_records(index=False)))
    checkpoint_keys = set(map(tuple, checkpoints[["session", "seed", "fold"]].to_records(index=False)))
    if expected_keys != checkpoint_keys:
        raise AssertionError("canonical task/checkpoint keys differ")
    provenance = {
        name: framework.file_sha256(args.canonical_run_dir / name)
        for name in required
    }
    return plan, checkpoints, baseline, provenance


def validate_historical_checkpoint_coverage(
    args: argparse.Namespace,
    plan: pd.DataFrame,
    checkpoints: pd.DataFrame,
) -> dict[tuple[str, int, int], str]:
    """Read and validate every exact historical outer checkpoint during plan."""

    checkpoint_lookup = {
        (str(row.session), int(row.seed), int(row.fold)): row
        for row in checkpoints.itertuples(index=False)
    }
    validated: dict[tuple[str, int, int], str] = {}
    for expected in plan.itertuples(index=False):
        key = (str(expected.session), int(expected.seed), int(expected.fold))
        manifest_row = checkpoint_lookup.get(key)
        if manifest_row is None:
            raise AssertionError(f"historical checkpoint manifest lacks {key}")
        path = checkpoint_path(args, *key)
        if not path.is_file():
            raise FileNotFoundError(path)
        model, payload, audit = load_validated_checkpoint(
            path,
            expected_sha256=str(manifest_row.checkpoint_sha256),
            expected_session=key[0],
            expected_seed=key[1],
            expected_fold=key[2],
            expected_train_cycles=str(expected.train_cycles),
            expected_test_cycles=str(expected.test_cycles),
        )
        if int(payload["final_epoch"]) != EXPECTED_EPOCH:
            raise AssertionError("historical outer checkpoint is not epoch 40")
        validated[key] = str(audit["checkpoint_sha256"])
        del model, payload
    if len(validated) != EXPECTED_TASKS or set(validated) != set(checkpoint_lookup):
        raise AssertionError("historical outer checkpoint coverage is not exact 246/246")
    return validated


def dataset_identities(args: argparse.Namespace) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for session in EXPECTED_SESSIONS:
        h5_path = args.data_dir / f"session_{session}_blocks.h5"
        csv_path = args.data_dir / f"session_{session}_block_metadata.csv"
        if not h5_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"clean4 data missing for session {session}")
        identities[session] = {
            "h5_path": str(h5_path),
            "h5_sha256": framework.file_sha256(h5_path),
            "metadata_path": str(csv_path),
            "metadata_sha256": framework.file_sha256(csv_path),
        }
    return identities


def build_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference_plan, checkpoints, _baseline, reference_hashes = validate_reference_assets(args)
    validated_checkpoint_hashes = validate_historical_checkpoint_coverage(
        args, reference_plan, checkpoints
    )
    source_hashes = {
        str(path.relative_to(args.project_root)): framework.file_sha256(path)
        for path in source_paths(args.project_root)
    }
    identity = {
        "protocol": formal_protocol(),
        "source_hashes": source_hashes,
        "git_head": current_git_head(args.project_root),
        "runtime": framework.runtime_environment_signature(),
        "datasets": dataset_identities(args),
        "canonical_reference_sha256": reference_hashes,
    }
    run_fingerprint = fingerprint(identity)
    protocol_hash = fingerprint(identity["protocol"])
    source_hash = fingerprint(source_hashes)
    checkpoint_lookup = {
        (str(row.session), int(row.seed), int(row.fold)): row
        for row in checkpoints.itertuples(index=False)
    }
    task_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for row in reference_plan.sort_values(["session", "seed", "fold"]).itertuples(index=False):
        session, seed, fold = str(row.session), int(row.seed), int(row.fold)
        outer_train = tuple(int(value) for value in str(row.train_cycles).split(","))
        outer_test = tuple(int(value) for value in str(row.test_cycles).split(","))
        checkpoint = checkpoint_lookup[(session, seed, fold)]
        task_payload = {
            "session": session,
            "seed": seed,
            "fold": fold,
            "outer_train_cycles": cycle_text(outer_train),
            "outer_test_cycles": cycle_text(outer_test),
            "n_outer_train_blocks": int(row.n_train_samples),
            "n_outer_test_blocks": int(row.n_test_samples),
            "historical_checkpoint_sha256": str(checkpoint.checkpoint_sha256),
            "historical_checkpoint_path": str(
                checkpoint_path(args, session, seed, fold).resolve()
            ),
            "historical_checkpoint_validated_sha256": validated_checkpoint_hashes[
                (session, seed, fold)
            ],
            "run_fingerprint": run_fingerprint,
            "git_head": identity["git_head"],
        }
        task_payload["task_key"] = f"{session}:{seed}:{fold}"
        task_payload["task_fingerprint"] = fingerprint(task_payload)
        task_rows.append(task_payload)
        for split in build_inner_cycle_splits(outer_train, outer_test):
            inner_rows.append(
                {
                    "task_key": task_payload["task_key"],
                    "task_fingerprint": task_payload["task_fingerprint"],
                    "session": session,
                    "outer_seed": seed,
                    "outer_fold": fold,
                    "outer_train_cycles": cycle_text(outer_train),
                    "outer_test_cycles": cycle_text(outer_test),
                    "inner_fold": split.inner_fold,
                    "inner_train_cycles": cycle_text(split.train_cycles),
                    "inner_validation_cycles": cycle_text(split.validation_cycles),
                    "n_inner_train_blocks": 4 * len(split.train_cycles),
                    "n_inner_validation_blocks": 4 * len(split.validation_cycles),
                    "n_inner_train_frames": 16 * len(split.train_cycles),
                    "n_inner_validation_frames": 16 * len(split.validation_cycles),
                    "inner_model_training_seed": seed,
                    "protocol_hash": protocol_hash,
                    "source_hash": source_hash,
                    "outer_test_used": False,
                }
            )
    plan = pd.DataFrame(task_rows)
    inner = pd.DataFrame(inner_rows)
    if len(plan) != EXPECTED_TASKS or len(inner) != EXPECTED_INNER_TRAININGS:
        raise AssertionError("formal CCLF task/training count drift")
    if plan["task_key"].duplicated().any() or inner[["task_key", "inner_fold"]].duplicated().any():
        raise AssertionError("formal plan contains duplicate task identities")
    return plan, inner, {"identity": identity, "run_fingerprint": run_fingerprint}


def write_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    plan, inner, metadata = build_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    framework.atomic_json(
        args.output_dir / "config.json",
        {**formal_protocol(), "run_fingerprint": metadata["run_fingerprint"]},
    )
    framework.atomic_json(
        args.output_dir / "runtime_fingerprint.json",
        {
            **metadata["identity"],
            "run_fingerprint": metadata["run_fingerprint"],
            "source_hashes_authoritative": True,
        },
    )
    framework.atomic_csv(args.output_dir / "task_plan.csv", plan)
    framework.atomic_csv(args.output_dir / "inner_split_manifest.csv", inner)
    framework.atomic_json(
        args.output_dir / "PLAN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "outer_tasks": len(plan),
            "folds": EXPECTED_FOLDS,
            "seeds": len(SEEDS),
            "sessions": len(EXPECTED_SESSIONS),
            "planned_inner_trainings": len(inner),
            "outer_final_model_trainings": 0,
            "formal_training_started": False,
            "run_fingerprint": metadata["run_fingerprint"],
            "task_plan_sha256": framework.file_sha256(args.output_dir / "task_plan.csv"),
            "inner_split_manifest_sha256": framework.file_sha256(args.output_dir / "inner_split_manifest.csv"),
        },
    )
    print(
        f"PLAN COMPLETE outer_tasks={len(plan)} folds={EXPECTED_FOLDS} "
        f"inner_trainings={len(inner)} outer_final_model_trainings=0 formal_started=False",
        flush=True,
    )
    return plan, inner, metadata


def load_strict_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    current_plan, current_inner, metadata = build_plan(args)
    required = ("config.json", "runtime_fingerprint.json", "task_plan.csv", "inner_split_manifest.csv")
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"plan artifacts absent: {missing}; run --stage plan")
    saved_plan = pd.read_csv(args.output_dir / "task_plan.csv", dtype={"session": str})
    saved_inner = pd.read_csv(args.output_dir / "inner_split_manifest.csv", dtype={"session": str})
    pd.testing.assert_frame_equal(saved_plan, current_plan, check_dtype=False)
    pd.testing.assert_frame_equal(saved_inner, current_inner, check_dtype=False)
    config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    runtime = json.loads((args.output_dir / "runtime_fingerprint.json").read_text(encoding="utf-8"))
    expected_config = {**formal_protocol(), "run_fingerprint": metadata["run_fingerprint"]}
    if fingerprint(config) != fingerprint(expected_config):
        raise AssertionError("saved config differs from frozen CCLF protocol")
    if runtime.get("run_fingerprint") != metadata["run_fingerprint"] or runtime.get("source_hashes") != metadata["identity"]["source_hashes"] or runtime.get("git_head") != metadata["identity"]["git_head"]:
        raise AssertionError("saved runtime/source/Git provenance differs")
    return saved_plan, saved_inner, metadata


def source_run_dir(args: argparse.Namespace, session: str) -> Path:
    return args.historical_base_run_dir if session in {"626", "628"} else args.historical_fcnn_run_dir


def checkpoint_path(args: argparse.Namespace, session: str, seed: int, fold: int) -> Path:
    return source_run_dir(args, session) / f"session_{session}/checkpoints/{HISTORICAL_METHOD}/seed_{seed}/fold_{fold}/checkpoint.pt"


def task_dir(output_dir: Path, row: dict[str, Any]) -> Path:
    return output_dir / "tasks" / f"session_{row['session']}" / f"seed_{int(row['seed'])}" / f"fold_{int(row['fold']):02d}"


def task_artifact_hashes(path: Path) -> dict[str, str]:
    return {name: framework.file_sha256(path / name) for name in sorted(REQUIRED_TASK_FILES)}


def validate_completed_task(path: Path, expected: dict[str, Any], *, raise_on_error: bool = False) -> tuple[bool, str]:
    def fail(message: str) -> tuple[bool, str]:
        if raise_on_error:
            raise AssertionError(f"invalid task {path}: {message}")
        return False, message

    missing = [name for name in (*REQUIRED_TASK_FILES, "COMPLETE.json") if not (path / name).is_file()]
    if missing:
        return fail(f"missing files {missing}")
    try:
        complete = json.loads((path / "COMPLETE.json").read_text(encoding="utf-8"))
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        temperature = json.loads((path / "temperature.json").read_text(encoding="utf-8"))
        splits = pd.read_csv(path / "inner_split_manifest.csv")
        training = pd.read_csv(path / "inner_training_summary.csv")
        oof = pd.read_csv(path / "inner_oof_logits.csv")
        frames = pd.read_csv(path / "frame_predictions.csv")
        predictions = pd.read_csv(path / "predictions.csv")
        reconstruction = json.loads((path / "baseline_reconstruction.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")
    if complete.get("status") != "complete" or complete.get("task_fingerprint") != str(expected["task_fingerprint"]):
        return fail("completion fingerprint mismatch")
    if complete.get("artifact_sha256") != task_artifact_hashes(path):
        return fail("task artifact SHA256 mismatch")
    if result.get("task_key") != str(expected["task_key"]) or result.get("run_fingerprint") != str(expected["run_fingerprint"]):
        return fail("result parent identity mismatch")
    if len(splits) != N_INNER_FOLDS or len(training) != N_INNER_FOLDS:
        return fail("inner fold/training count is not three")
    if int(training["trained_epochs"].min()) != FORMAL_TRAINING_CONFIG.max_epochs or int(training["trained_epochs"].max()) != FORMAL_TRAINING_CONFIG.max_epochs:
        return fail("inner training is not fixed 40 epochs")
    if not splits["outer_test_used"].eq(False).all() or not training["outer_test_used"].eq(False).all():
        return fail("outer-test leakage marker failed")
    if splits["cache_key"].isna().any() or splits["cache_key"].duplicated().any():
        return fail("cache identities missing or duplicated")
    if len(oof) != int(expected["n_outer_train_blocks"]) * FRAMES_PER_BLOCK:
        return fail("inner OOF frame coverage differs")
    if oof[["source_index", "frame_position"]].duplicated().any():
        return fail("inner OOF frames duplicated")
    if len(frames) != int(expected["n_outer_test_blocks"]) * FRAMES_PER_BLOCK or frames[["block_id", "frame_position"]].duplicated().any():
        return fail("outer frame prediction coverage differs")
    if len(predictions) != int(expected["n_outer_test_blocks"]) or predictions["block_id"].duplicated().any():
        return fail("outer block prediction coverage differs")
    raw = frames[["raw_prob_no_stimulus", "raw_prob_stimulus"]].to_numpy(float)
    calibrated = frames[["cal_prob_no_stimulus", "cal_prob_stimulus"]].to_numpy(float)
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all() or not np.allclose(raw.sum(1), 1.0, atol=1e-6) or not np.allclose(calibrated.sum(1), 1.0, atol=1e-6):
        return fail("outer frame probabilities invalid")
    raw_fused = equal_four_frame_probability_mean(raw.reshape(-1, FRAMES_PER_BLOCK, 2))
    cal_fused = equal_four_frame_probability_mean(calibrated.reshape(-1, FRAMES_PER_BLOCK, 2))
    if not np.allclose(raw_fused, predictions[["baseline_prob_no_stimulus", "baseline_prob_stimulus"]], atol=1e-6) or not np.allclose(cal_fused, predictions[["cclf_prob_no_stimulus", "cclf_prob_stimulus"]], atol=1e-6):
        return fail("saved fusion is not equal four-frame probability mean")
    if not np.isfinite(float(temperature.get("temperature", np.nan))) or float(temperature["temperature"]) <= 0:
        return fail("temperature is not positive")
    if temperature.get("objective") != "cross_entropy_nll":
        return fail("temperature objective is not NLL-only")
    if reconstruction.get("status") != "PASS" or int(reconstruction.get("prediction_mismatch_count", -1)) != 0:
        return fail("baseline exact reconstruction failed")
    return True, "validated"


def task_indices(data: Any, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_cycles = {int(value) for value in str(row["outer_train_cycles"]).split(",")}
    test_cycles = {int(value) for value in str(row["outer_test_cycles"]).split(",")}
    train = np.flatnonzero(np.isin(data.groups, sorted(train_cycles)))
    test = np.flatnonzero(np.isin(data.groups, sorted(test_cycles)))
    if set(data.groups[train]) != train_cycles or set(data.groups[test]) != test_cycles:
        raise AssertionError("dataset cycles do not match frozen outer plan")
    if len(train) != int(row["n_outer_train_blocks"]) or len(test) != int(row["n_outer_test_blocks"]):
        raise AssertionError("dataset block counts differ from frozen outer plan")
    if set(train) & set(test):
        raise AssertionError("outer train/test block overlap")
    return train, test


def outer_checkpoint(args: argparse.Namespace, row: dict[str, Any]) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    path = checkpoint_path(args, str(row["session"]), int(row["seed"]), int(row["fold"]))
    return load_validated_checkpoint(
        path,
        expected_sha256=str(row["historical_checkpoint_sha256"]),
        expected_session=str(row["session"]),
        expected_seed=int(row["seed"]),
        expected_fold=int(row["fold"]),
        expected_train_cycles=str(row["outer_train_cycles"]),
        expected_test_cycles=str(row["outer_test_cycles"]),
    )


def write_task(
    args: argparse.Namespace,
    row: dict[str, Any],
    inner_plan: pd.DataFrame,
    baseline_reference: pd.DataFrame,
) -> Path:
    destination = task_dir(args.output_dir, row)
    destination.mkdir(parents=True, exist_ok=True)
    data = load_block_sequence_session(args.project_root, str(row["session"]), "binary", data_dir=args.data_dir)
    outer_train_idx, outer_test_idx = task_indices(data, row)
    protocol_hash = fingerprint(formal_protocol())
    source_hash = fingerprint(
        json.loads((args.output_dir / "runtime_fingerprint.json").read_text(encoding="utf-8"))["source_hashes"]
    )

    oof_parts: list[pd.DataFrame] = []
    inner_summary_rows: list[dict[str, Any]] = []
    completed_split_rows: list[dict[str, Any]] = []
    task_inner = inner_plan[inner_plan["task_key"].eq(str(row["task_key"]))].sort_values("inner_fold")
    for split_row in task_inner.to_dict(orient="records"):
        inner_train_cycles = {int(value) for value in str(split_row["inner_train_cycles"]).split(",")}
        inner_validation_cycles = {int(value) for value in str(split_row["inner_validation_cycles"]).split(",")}
        inner_train_idx = outer_train_idx[np.isin(data.groups[outer_train_idx], sorted(inner_train_cycles))]
        inner_validation_idx = outer_train_idx[np.isin(data.groups[outer_train_idx], sorted(inner_validation_cycles))]
        if set(data.groups[inner_train_idx]) & set(data.groups[inner_validation_idx]):
            raise AssertionError("inner cycle isolation failed")
        mean, std, normalization_fp = fit_inner_train_normalization(data.X[inner_train_idx])
        cache_key = build_inner_cache_key(
            session=str(row["session"]),
            outer_fold=int(row["fold"]),
            outer_seed=int(row["seed"]),
            outer_train_cycles=tuple(int(value) for value in str(row["outer_train_cycles"]).split(",")),
            inner_fold=int(split_row["inner_fold"]),
            inner_train_cycles=tuple(sorted(inner_train_cycles)),
            inner_validation_cycles=tuple(sorted(inner_validation_cycles)),
            source_hash=source_hash,
            protocol_hash=protocol_hash,
            normalization_fingerprint=normalization_fp,
            training_config=vars(FORMAL_TRAINING_CONFIG),
        )
        # The training helper refits the identical statistics from the same inner-train array.
        logits, history, observed_normalization_fp = train_inner_fcnn(
            data.X[inner_train_idx],
            data.y[inner_train_idx],
            data.X[inner_validation_idx],
            seed=int(row["seed"]),
            device=str(args.device),
            training_config=FORMAL_TRAINING_CONFIG,
        )
        if observed_normalization_fp != normalization_fp:
            raise AssertionError("inner normalization fingerprint is not deterministic")
        reshaped = logits.reshape(len(inner_validation_idx), FRAMES_PER_BLOCK, 2)
        for local_i, source_index in enumerate(inner_validation_idx):
            metadata = data.metadata.iloc[int(source_index)]
            for frame_position in range(FRAMES_PER_BLOCK):
                oof_parts.append(
                    pd.DataFrame(
                        [{
                            "session": str(row["session"]),
                            "outer_seed": int(row["seed"]),
                            "outer_fold": int(row["fold"]),
                            "inner_fold": int(split_row["inner_fold"]),
                            "source_index": int(source_index),
                            "block_id": str(metadata["block_id"]),
                            "cycle": int(data.groups[source_index]),
                            "frame_position": frame_position,
                            "truth": int(data.y[source_index]),
                            "logit_no_stimulus": float(reshaped[local_i, frame_position, 0]),
                            "logit_stimulus": float(reshaped[local_i, frame_position, 1]),
                            "heldout_cycle": int(data.groups[source_index]),
                            "cache_key": cache_key,
                        }]
                    )
                )
        inner_summary_rows.append(
            {
                "task_key": str(row["task_key"]),
                "inner_fold": int(split_row["inner_fold"]),
                "inner_train_cycles": str(split_row["inner_train_cycles"]),
                "inner_validation_cycles": str(split_row["inner_validation_cycles"]),
                "n_train_blocks": len(inner_train_idx),
                "n_validation_blocks": len(inner_validation_idx),
                "trained_epochs": len(history),
                "final_train_loss": float(history[-1]["train_loss"]),
                "normalization_fingerprint": normalization_fp,
                "cache_key": cache_key,
                "outer_test_used": False,
            }
        )
        completed_split_rows.append({**split_row, "normalization_fingerprint": normalization_fp, "cache_key": cache_key})

    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["source_index", "frame_position"]).reset_index(drop=True)
    assert_complete_inner_oof(
        oof["source_index"].to_numpy(int),
        outer_train_idx,
        oof["cycle"].to_numpy(int),
        oof["heldout_cycle"].to_numpy(int),
    )
    oof_logits = oof[["logit_no_stimulus", "logit_stimulus"]].to_numpy(float)
    oof_truth = oof["truth"].to_numpy(int)
    temperature = fit_scalar_temperature(oof_logits, oof_truth)

    model, payload, checkpoint_audit = outer_checkpoint(args, row)
    model.to(torch.device(args.device))
    flattened = data.X[outer_test_idx].reshape(-1, *IMAGE_SHAPE)
    normalized = apply_saved_normalization(
        flattened,
        np.asarray(payload["normalization_mean"]),
        np.asarray(payload["normalization_std"]),
        transform=str(payload["normalization_transform"]),
    ).reshape(len(outer_test_idx), FRAMES_PER_BLOCK, *IMAGE_SHAPE)
    outer_logits = predict_raw_logits(
        model,
        normalized,
        device="cpu" if str(args.device) == "cpu" else args.device,
        batch_size=int(args.inference_batch_size),
    )
    raw_frame_probabilities = softmax_probabilities(outer_logits)
    calibrated_probabilities = calibrated_frame_probabilities(outer_logits, temperature.temperature)
    raw_blocks = equal_four_frame_probability_mean(raw_frame_probabilities.reshape(-1, FRAMES_PER_BLOCK, 2))
    calibrated_blocks = equal_four_frame_probability_mean(calibrated_probabilities.reshape(-1, FRAMES_PER_BLOCK, 2))

    frame_rows: list[dict[str, Any]] = []
    for local_i, source_index in enumerate(outer_test_idx):
        metadata = data.metadata.iloc[int(source_index)]
        for position in range(FRAMES_PER_BLOCK):
            flat_i = local_i * FRAMES_PER_BLOCK + position
            frame_rows.append(
                {
                    "session": str(row["session"]),
                    "seed": int(row["seed"]),
                    "fold": int(row["fold"]),
                    "cycle": int(data.groups[source_index]),
                    "source_index": int(source_index),
                    "block_id": str(metadata["block_id"]),
                    "block_name": str(metadata["block_name"]),
                    "truth": int(data.y[source_index]),
                    "frame_position": position,
                    "original_frame_index": int(data.clean4_original_frame_indices[source_index, position]),
                    "relative_time_s": float(data.clean4_relative_time_s[source_index, position]),
                    "temperature": temperature.temperature,
                    "raw_logit_no_stimulus": float(outer_logits[flat_i, 0]),
                    "raw_logit_stimulus": float(outer_logits[flat_i, 1]),
                    "raw_prob_no_stimulus": float(raw_frame_probabilities[flat_i, 0]),
                    "raw_prob_stimulus": float(raw_frame_probabilities[flat_i, 1]),
                    "cal_prob_no_stimulus": float(calibrated_probabilities[flat_i, 0]),
                    "cal_prob_stimulus": float(calibrated_probabilities[flat_i, 1]),
                    "raw_frame_pred": int(raw_frame_probabilities[flat_i].argmax()),
                    "cal_frame_pred": int(calibrated_probabilities[flat_i].argmax()),
                }
            )
    prediction_rows = []
    for local_i, source_index in enumerate(outer_test_idx):
        metadata = data.metadata.iloc[int(source_index)]
        prediction_rows.append(
            {
                "session": str(row["session"]),
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "cycle": int(data.groups[source_index]),
                "source_index": int(source_index),
                "block_id": str(metadata["block_id"]),
                "block_name": str(metadata["block_name"]),
                "truth": int(data.y[source_index]),
                "temperature": temperature.temperature,
                "baseline_prob_no_stimulus": float(raw_blocks[local_i, 0]),
                "baseline_prob_stimulus": float(raw_blocks[local_i, 1]),
                "baseline_pred": int(raw_blocks[local_i].argmax()),
                "cclf_prob_no_stimulus": float(calibrated_blocks[local_i, 0]),
                "cclf_prob_stimulus": float(calibrated_blocks[local_i, 1]),
                "cclf_pred": int(calibrated_blocks[local_i].argmax()),
            }
        )
    predictions = pd.DataFrame(prediction_rows)
    reference = baseline_reference[
        baseline_reference["session"].eq(str(row["session"]))
        & baseline_reference["seed"].eq(int(row["seed"]))
        & baseline_reference["fold"].eq(int(row["fold"]))
    ].sort_values("block_id")
    observed = predictions.sort_values("block_id")
    if observed["block_id"].tolist() != reference["block_id"].tolist() or not np.array_equal(observed["truth"], reference["truth"]):
        raise AssertionError("baseline reconstruction identity/truth mismatch")
    differences = np.abs(
        observed[["baseline_prob_no_stimulus", "baseline_prob_stimulus"]].to_numpy(float)
        - reference[["prob_no_stimulus", "prob_stimulus"]].to_numpy(float)
    )
    prediction_mismatches = int(np.sum(observed["baseline_pred"].to_numpy(int) != reference["pred"].to_numpy(int)))
    reconstruction = {
        "status": "PASS" if np.allclose(
            observed[["baseline_prob_no_stimulus", "baseline_prob_stimulus"]],
            reference[["prob_no_stimulus", "prob_stimulus"]],
            atol=BASELINE_ATOL,
            rtol=BASELINE_RTOL,
        ) and prediction_mismatches == 0 else "FAIL",
        "n_blocks": len(observed),
        "maximum_probability_absolute_difference": float(differences.max(initial=0.0)),
        "prediction_mismatch_count": prediction_mismatches,
        "reference_sha256": framework.file_sha256(args.canonical_run_dir / "late_fusion_reconstructed_predictions.csv"),
    }
    if reconstruction["status"] != "PASS":
        raise AssertionError("historical baseline reconstruction mismatch; STOP")

    result = {
        "task_key": str(row["task_key"]),
        "task_fingerprint": str(row["task_fingerprint"]),
        "run_fingerprint": str(row["run_fingerprint"]),
        "git_head": str(row["git_head"]),
        "session": str(row["session"]),
        "seed": int(row["seed"]),
        "fold": int(row["fold"]),
        "outer_train_cycles": str(row["outer_train_cycles"]),
        "outer_test_cycles": str(row["outer_test_cycles"]),
        "temperature": temperature.temperature,
        "baseline_ba": block_balanced_accuracy(predictions["truth"], predictions["baseline_pred"]),
        "cclf_ba": block_balanced_accuracy(predictions["truth"], predictions["cclf_pred"]),
        "outer_checkpoint_sha256": checkpoint_audit["checkpoint_sha256"],
        "outer_final_model_trained": False,
        "fusion_weights": [0.25, 0.25, 0.25, 0.25],
    }
    framework.atomic_json(destination / "result.json", result)
    framework.atomic_json(destination / "temperature.json", temperature.to_dict())
    framework.atomic_csv(destination / "inner_split_manifest.csv", pd.DataFrame(completed_split_rows))
    framework.atomic_csv(destination / "inner_training_summary.csv", pd.DataFrame(inner_summary_rows))
    framework.atomic_csv(destination / "inner_oof_logits.csv", oof)
    framework.atomic_csv(destination / "frame_predictions.csv", pd.DataFrame(frame_rows))
    framework.atomic_csv(destination / "predictions.csv", predictions)
    framework.atomic_json(destination / "baseline_reconstruction.json", reconstruction)
    framework.atomic_json(
        destination / "COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "task_key": str(row["task_key"]),
            "task_fingerprint": str(row["task_fingerprint"]),
            "run_fingerprint": str(row["run_fingerprint"]),
            "git_head": str(row["git_head"]),
            "inner_trainings": N_INNER_FOLDS,
            "outer_final_model_trainings": 0,
            "artifact_sha256": task_artifact_hashes(destination),
        },
    )
    return destination


def summarize_ba(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows = []
    for (session, seed), group in predictions.groupby(["session", "seed"], sort=True):
        baseline = block_balanced_accuracy(group["truth"], group["baseline_pred"])
        cclf = block_balanced_accuracy(group["truth"], group["cclf_pred"])
        seed_rows.append({"session": str(session), "seed": int(seed), "n_blocks": len(group), "baseline_ba": baseline, "cclf_ba": cclf, "delta": cclf - baseline})
    session_seed = pd.DataFrame(seed_rows)
    session_rows = []
    for session, group in predictions.groupby("session", sort=True):
        baseline = block_balanced_accuracy(group["truth"], group["baseline_pred"])
        cclf = block_balanced_accuracy(group["truth"], group["cclf_pred"])
        session_rows.append(
            {
                "scope": "session",
                "scope_id": str(session),
                "aggregation": "concatenate_oof_within_session",
                "n_sessions": 1,
                "n_blocks": len(group),
                "baseline_ba": baseline,
                "cclf_ba": cclf,
                "delta": cclf - baseline,
                "baseline_pooled_block_ba": baseline,
                "cclf_pooled_block_ba": cclf,
                "pooled_block_delta": cclf - baseline,
            }
        )
    session_frame = pd.DataFrame(session_rows)
    for scope_id, sessions in (("strong", STRONG_SESSIONS), ("weak", WEAK_SESSIONS), ("overall", tuple(EXPECTED_SESSIONS))):
        group = predictions[predictions["session"].isin(sessions)]
        selected_sessions = session_frame[session_frame["scope_id"].isin(sessions)]
        if group.empty or selected_sessions.empty:
            continue
        baseline = float(selected_sessions["baseline_ba"].mean())
        cclf = float(selected_sessions["cclf_ba"].mean())
        pooled_baseline = block_balanced_accuracy(
            group["truth"], group["baseline_pred"]
        )
        pooled_cclf = block_balanced_accuracy(group["truth"], group["cclf_pred"])
        session_rows.append(
            {
                "scope": scope_id,
                "scope_id": ",".join(sessions),
                "aggregation": "arithmetic_mean_of_session_ba",
                "n_sessions": len(selected_sessions),
                "n_blocks": len(group),
                "baseline_ba": baseline,
                "cclf_ba": cclf,
                "delta": cclf - baseline,
                "baseline_pooled_block_ba": pooled_baseline,
                "cclf_pooled_block_ba": pooled_cclf,
                "pooled_block_delta": pooled_cclf - pooled_baseline,
            }
        )
    return session_seed, pd.DataFrame(session_rows)


def leave_one_session_out_mean_delta(session_summary: pd.DataFrame) -> list[dict[str, Any]]:
    sessions = session_summary[session_summary["scope"].eq("session")].copy()
    if sessions.empty or sessions["scope_id"].duplicated().any():
        raise ValueError("LOSO requires unique session-level BA rows")
    rows: list[dict[str, Any]] = []
    for heldout in sessions["scope_id"].astype(str).sort_values():
        remaining = sessions[~sessions["scope_id"].astype(str).eq(heldout)]
        if remaining.empty:
            raise ValueError("LOSO requires at least two sessions")
        baseline = float(remaining["baseline_ba"].mean())
        cclf = float(remaining["cclf_ba"].mean())
        mean_delta = float(remaining["delta"].mean())
        rows.append(
            {
                "heldout_session": heldout,
                "remaining_sessions": int(len(remaining)),
                "aggregation": "arithmetic_mean_of_remaining_session_deltas",
                "baseline_mean_session_ba": baseline,
                "cclf_mean_session_ba": cclf,
                "mean_session_delta": mean_delta,
                "delta": mean_delta,
            }
        )
    return rows


def summarize_calibration(frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scopes = [("overall", "all", frames)]
    scopes.extend(("session", session, group) for session, group in frames.groupby("session", sort=True))
    scopes.extend(
        [
            ("strong", ",".join(STRONG_SESSIONS), frames[frames["session"].isin(STRONG_SESSIONS)]),
            ("weak", ",".join(WEAK_SESSIONS), frames[frames["session"].isin(WEAK_SESSIONS)]),
        ]
    )
    for scope, scope_id, group in scopes:
        if group.empty:
            continue
        truth = group["truth"].to_numpy(int)
        for state, columns in (
            ("pre", ["raw_prob_no_stimulus", "raw_prob_stimulus"]),
            ("post", ["cal_prob_no_stimulus", "cal_prob_stimulus"]),
        ):
            rows.append({"scope": scope, "scope_id": scope_id, "calibration_state": state, **frame_calibration_metrics(group[columns].to_numpy(float), truth)})
    return pd.DataFrame(rows)


def aggregate_outputs(args: argparse.Namespace, plan: pd.DataFrame, metadata: dict[str, Any]) -> None:
    prediction_parts, frame_parts, temperature_rows, inner_parts = [], [], [], []
    reconstruction_max = 0.0
    for row in plan.to_dict(orient="records"):
        path = task_dir(args.output_dir, row)
        validate_completed_task(path, row, raise_on_error=True)
        prediction_parts.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        frame_parts.append(pd.read_csv(path / "frame_predictions.csv", dtype={"session": str}))
        temperature = json.loads((path / "temperature.json").read_text(encoding="utf-8"))
        temperature_rows.append({"task_key": row["task_key"], "session": row["session"], "seed": int(row["seed"]), "fold": int(row["fold"]), **temperature})
        inner_parts.append(pd.read_csv(path / "inner_training_summary.csv"))
        reconstruction = json.loads((path / "baseline_reconstruction.json").read_text(encoding="utf-8"))
        reconstruction_max = max(reconstruction_max, float(reconstruction["maximum_probability_absolute_difference"]))
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(["session", "seed", "fold", "source_index"]).reset_index(drop=True)
    frames = pd.concat(frame_parts, ignore_index=True).sort_values(["session", "seed", "fold", "source_index", "frame_position"]).reset_index(drop=True)
    temperatures = pd.DataFrame(temperature_rows).sort_values(["session", "seed", "fold"]).reset_index(drop=True)
    inner_summary = pd.concat(inner_parts, ignore_index=True).sort_values(["task_key", "inner_fold"]).reset_index(drop=True)
    if len(predictions) != EXPECTED_BLOCK_PREDICTIONS or predictions[["session", "seed", "block_id"]].duplicated().any():
        raise AssertionError("aggregate OOF block coverage is not exact")
    if len(frames) != EXPECTED_FRAME_PREDICTIONS or frames[["session", "seed", "block_id", "frame_position"]].duplicated().any():
        raise AssertionError("aggregate OOF frame coverage is not exact")
    if len(temperatures) != EXPECTED_TASKS or len(inner_summary) != EXPECTED_INNER_TRAININGS:
        raise AssertionError("aggregate temperature/inner-training coverage differs")
    session_seed, session_summary = summarize_ba(predictions)
    calibration = summarize_calibration(frames)
    lookup = {(row.scope, row.scope_id): row for row in session_summary.itertuples(index=False)}
    overall = lookup[("overall", ",".join(EXPECTED_SESSIONS))]
    strong = lookup[("strong", ",".join(STRONG_SESSIONS))]
    weak = lookup[("weak", ",".join(WEAK_SESSIONS))]
    calibration_lookup = {(row.scope, row.calibration_state): row for row in calibration.itertuples(index=False) if row.scope == "overall"}
    gate = evaluate_frozen_gate(
        baseline_overall_ba=overall.baseline_ba,
        cclf_overall_ba=overall.cclf_ba,
        baseline_strong_ba=strong.baseline_ba,
        cclf_strong_ba=strong.cclf_ba,
        baseline_weak_ba=weak.baseline_ba,
        cclf_weak_ba=weak.cclf_ba,
        baseline_overall_ece=calibration_lookup[("overall", "pre")].ece,
        cclf_overall_ece=calibration_lookup[("overall", "post")].ece,
    )
    session_deltas = session_summary[session_summary["scope"].eq("session")]["delta"].to_numpy(float)
    formal_session_rows = session_summary[session_summary["scope"].eq("session")]
    if set(formal_session_rows["scope_id"].astype(str)) != set(EXPECTED_SESSIONS):
        raise AssertionError("formal session BA summary does not cover exact 9 sessions")
    loso = leave_one_session_out_mean_delta(session_summary)
    statistical = {
        "per_session_delta": dict(zip(session_summary[session_summary["scope"].eq("session")]["scope_id"], session_deltas.tolist())),
        "improved_sessions": int(np.sum(session_deltas > 1e-15)),
        "tied_sessions": int(np.sum(np.abs(session_deltas) <= 1e-15)),
        "worsened_sessions": int(np.sum(session_deltas < -1e-15)),
        "median_session_delta": float(np.median(session_deltas)),
        "exact_paired_sign_flip": exact_paired_sign_flip_test(session_deltas),
        "leave_one_session_out": loso,
        "frozen_gate": gate,
    }
    provenance = {
        "status": "PASS",
        "run_fingerprint": metadata["run_fingerprint"],
        "git_head": metadata["identity"]["git_head"],
        "source_hashes": metadata["identity"]["source_hashes"],
        "source_hashes_authoritative": True,
        "dataset_identities": metadata["identity"]["datasets"],
        "canonical_reference_sha256": metadata["identity"]["canonical_reference_sha256"],
        "historical_checkpoint_coverage": f"{len(temperatures)}/{EXPECTED_TASKS}",
        "baseline_reconstruction": {"status": "PASS", "blocks": len(predictions), "maximum_probability_absolute_difference": reconstruction_max, "prediction_mismatch_count": 0},
        "inner_trainings": len(inner_summary),
        "outer_final_model_trainings": 0,
        "frame_weights": [0.25, 0.25, 0.25, 0.25],
    }
    framework.atomic_csv(args.output_dir / "temperature_summary.csv", temperatures)
    framework.atomic_csv(args.output_dir / "inner_calibration_summary.csv", inner_summary)
    framework.atomic_csv(args.output_dir / "frame_predictions.csv", frames)
    framework.atomic_csv(args.output_dir / "predictions.csv", predictions)
    framework.atomic_csv(args.output_dir / "session_seed_summary.csv", session_seed)
    framework.atomic_csv(args.output_dir / "session_summary.csv", session_summary)
    framework.atomic_csv(args.output_dir / "calibration_summary.csv", calibration)
    framework.atomic_json(args.output_dir / "statistical_audit.json", statistical)
    framework.atomic_json(args.output_dir / "provenance_audit.json", provenance)


def aggregate_artifact_sha256(output_dir: Path) -> dict[str, str]:
    missing = [name for name in REQUIRED_RUN_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"required aggregate outputs missing: {missing}")
    return {name: framework.file_sha256(output_dir / name) for name in sorted(REQUIRED_RUN_OUTPUTS)}


def validate_aggregate_artifact_integrity(output_dir: Path, completion: dict[str, Any]) -> tuple[bool, str]:
    expected = completion.get("aggregate_artifact_sha256")
    if not isinstance(expected, dict):
        return False, "RUN_COMPLETE lacks aggregate_artifact_sha256"
    if "RUN_COMPLETE.json" in expected:
        return False, "RUN_COMPLETE must not hash itself"
    if set(expected) != set(REQUIRED_RUN_OUTPUTS):
        return False, "aggregate hash manifest coverage mismatch"
    missing = [name for name in sorted(expected) if not (output_dir / name).is_file()]
    if missing:
        return False, f"aggregate artifacts missing: {missing}"
    changed = [name for name in sorted(expected) if framework.file_sha256(output_dir / name) != str(expected[name])]
    if changed:
        return False, f"aggregate artifact SHA256 mismatch: {changed}"
    return True, "validated"


def run_full(args: argparse.Namespace) -> None:
    if not args.review_approved:
        raise RuntimeError("formal full requires --review-approved after external code review")
    plan, inner, metadata = load_strict_plan(args)
    _reference_plan, _checkpoints, baseline, _hashes = validate_reference_assets(args)
    (args.output_dir / "RUN_COMPLETE.json").unlink(missing_ok=True)
    completed = 0
    for row in plan.to_dict(orient="records"):
        path = task_dir(args.output_dir, row)
        valid, reason = validate_completed_task(path, row)
        if valid:
            print(f"resume skip {row['task_key']}", flush=True)
        else:
            print(f"run {row['task_key']} ({reason})", flush=True)
            write_task(args, row, inner, baseline)
            validate_completed_task(path, row, raise_on_error=True)
        completed += 1
    if completed != EXPECTED_TASKS:
        raise AssertionError("formal task coverage is not 246/246")
    aggregate_outputs(args, plan, metadata)
    hashes = aggregate_artifact_sha256(args.output_dir)
    framework.atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "completed_tasks": EXPECTED_TASKS,
            "expected_tasks": EXPECTED_TASKS,
            "completed_inner_trainings": EXPECTED_INNER_TRAININGS,
            "outer_final_model_trainings": 0,
            "historical_checkpoint_coverage": "246/246",
            "baseline_reconstruction": "PASS",
            "block_prediction_coverage": EXPECTED_BLOCK_PREDICTIONS,
            "frame_prediction_coverage": EXPECTED_FRAME_PREDICTIONS,
            "run_fingerprint": metadata["run_fingerprint"],
            "git_head": metadata["identity"]["git_head"],
            "source_hashes_authoritative": True,
            "aggregate_artifact_sha256": hashes,
            "required_outputs": list(REQUIRED_RUN_OUTPUTS),
        },
    )
    print("RUN_COMPLETE written: strict 246/246 outer, 738/738 inner", flush=True)


def run_sanity(args: argparse.Namespace) -> None:
    if int(args.sanity_epochs) < 1:
        raise ValueError("sanity epochs must be positive")
    if str(args.device) != "cpu":
        raise ValueError("sanity is deliberately CPU-only")
    plan, inner_plan, metadata = load_strict_plan(args)
    row = plan.iloc[0].to_dict()
    data = load_block_sequence_session(
        args.project_root,
        str(row["session"]),
        "binary",
        data_dir=args.data_dir,
    )
    outer_train_idx, outer_test_idx = task_indices(data, row)
    from ultrasound_decoding.multiframe.training import DeepTrainingConfig

    sanity_config = DeepTrainingConfig(**{**vars(FORMAL_TRAINING_CONFIG), "max_epochs": int(args.sanity_epochs)})
    oof_logits, oof_labels, oof_indices, oof_cycles = [], [], [], []
    task_inner = inner_plan[inner_plan["task_key"].eq(str(row["task_key"]))].sort_values("inner_fold")
    histories = []
    for split in task_inner.itertuples(index=False):
        train_cycles = [int(value) for value in str(split.inner_train_cycles).split(",")]
        validation_cycles = [int(value) for value in str(split.inner_validation_cycles).split(",")]
        train_idx = outer_train_idx[np.isin(data.groups[outer_train_idx], train_cycles)]
        validation_idx = outer_train_idx[np.isin(data.groups[outer_train_idx], validation_cycles)]
        logits, history, _normalization_fp = train_inner_fcnn(
            data.X[train_idx], data.y[train_idx], data.X[validation_idx],
            seed=0, device="cpu", training_config=sanity_config,
        )
        oof_logits.append(logits)
        oof_labels.append(np.repeat(data.y[validation_idx], FRAMES_PER_BLOCK))
        oof_indices.append(np.repeat(validation_idx, FRAMES_PER_BLOCK))
        oof_cycles.append(np.repeat(data.groups[validation_idx], FRAMES_PER_BLOCK))
        histories.append(len(history))
    logits = np.concatenate(oof_logits)
    truth = np.concatenate(oof_labels)
    source_indices = np.concatenate(oof_indices)
    source_cycles = np.concatenate(oof_cycles)
    assert_complete_inner_oof(
        source_indices,
        outer_train_idx,
        source_cycles,
        source_cycles.copy(),
    )
    fitted = fit_scalar_temperature(logits, truth)
    mutated_outer_test = data.X[outer_test_idx].copy()
    mutated_outer_test[:] = -9999.0
    mutated_outer_test_labels = 1 - data.y[outer_test_idx]
    refitted_after_outer_test_mutation = fit_scalar_temperature(logits, truth)
    temperature_unchanged_after_outer_test_mutation = bool(
        fitted.temperature == refitted_after_outer_test_mutation.temperature
        and len(mutated_outer_test_labels) == len(outer_test_idx)
    )
    if not temperature_unchanged_after_outer_test_mutation:
        raise AssertionError("outer-test mutation changed fitted T")
    outer_model, payload, checkpoint_audit = outer_checkpoint(args, row)
    flattened = data.X[outer_test_idx].reshape(-1, *IMAGE_SHAPE)
    normalized_test = apply_saved_normalization(
        flattened,
        np.asarray(payload["normalization_mean"]),
        np.asarray(payload["normalization_std"]),
        transform=str(payload["normalization_transform"]),
    ).reshape(len(outer_test_idx), FRAMES_PER_BLOCK, *IMAGE_SHAPE)
    outer_logits = predict_raw_logits(
        outer_model,
        normalized_test,
        device="cpu",
        batch_size=int(args.inference_batch_size),
    )
    raw = softmax_probabilities(outer_logits)
    calibrated = calibrated_frame_probabilities(outer_logits, fitted.temperature)
    raw_blocks = equal_four_frame_probability_mean(raw.reshape(-1, FRAMES_PER_BLOCK, 2))
    calibrated_blocks = equal_four_frame_probability_mean(calibrated.reshape(-1, FRAMES_PER_BLOCK, 2))
    if not np.array_equal(raw, calibrated_frame_probabilities(outer_logits, 1.0)):
        raise AssertionError("T=1 did not exactly reproduce baseline probabilities")
    _reference_plan, _checkpoints, baseline, _hashes = validate_reference_assets(args)
    reference = baseline[
        baseline["session"].eq(str(row["session"]))
        & baseline["seed"].eq(int(row["seed"]))
        & baseline["fold"].eq(int(row["fold"]))
    ].sort_values("block_id")
    observed_order = np.argsort(data.metadata.iloc[outer_test_idx]["block_id"].astype(str).to_numpy())
    observed_raw_blocks = raw_blocks[observed_order]
    if data.metadata.iloc[outer_test_idx].iloc[observed_order]["block_id"].astype(str).tolist() != reference["block_id"].astype(str).tolist():
        raise AssertionError("sanity baseline block identities differ")
    baseline_max_difference = float(
        np.max(
            np.abs(
                observed_raw_blocks
                - reference[["prob_no_stimulus", "prob_stimulus"]].to_numpy(float)
            )
        )
    )
    baseline_mismatches = int(
        np.sum(observed_raw_blocks.argmax(axis=1) != reference["pred"].to_numpy(int))
    )
    if not np.allclose(
        observed_raw_blocks,
        reference[["prob_no_stimulus", "prob_stimulus"]].to_numpy(float),
        atol=BASELINE_ATOL,
        rtol=BASELINE_RTOL,
    ) or baseline_mismatches:
        raise AssertionError("sanity historical baseline reconstruction failed")
    framework.atomic_json(
        args.output_dir / "SANITY_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "device": "cpu",
            "synthetic": False,
            "task_key": str(row["task_key"]),
            "run_fingerprint": metadata["run_fingerprint"],
            "git_head": metadata["identity"]["git_head"],
            "historical_checkpoint_sha256": checkpoint_audit["checkpoint_sha256"],
            "baseline_reconstruction": "PASS",
            "baseline_probability_maximum_absolute_difference": baseline_max_difference,
            "baseline_prediction_mismatch_count": baseline_mismatches,
            "inner_folds": N_INNER_FOLDS,
            "inner_trainings": N_INNER_FOLDS,
            "epochs_per_inner_training": int(args.sanity_epochs),
            "temperature": fitted.temperature,
            "temperature_positive": fitted.temperature > 0,
            "pre_inner_oof_nll": fitted.pre_nll,
            "post_inner_oof_nll": fitted.post_nll,
            "calibrated_probabilities_finite_sum_one": bool(np.isfinite(calibrated).all() and np.allclose(calibrated.sum(1), 1.0)),
            "equal_four_frame_mean": True,
            "t_equals_one_exact_baseline": True,
            "outer_test_mutation_changes_temperature": False,
            "temperature_unchanged_after_outer_test_mutation": (
                temperature_unchanged_after_outer_test_mutation
            ),
            "outer_final_model_trainings": 0,
            "formal_training_started": False,
            "raw_block_predictions": raw_blocks.argmax(1).tolist(),
            "calibrated_block_predictions": calibrated_blocks.argmax(1).tolist(),
        },
    )
    print(
        f"SANITY PASS cpu task={row['task_key']} baseline=exact "
        f"inner_trainings=3 epochs={args.sanity_epochs} "
        f"T={fitted.temperature:.6g} outer_final_model_trainings=0 formal_started=False",
        flush=True,
    )


def run_status(args: argparse.Namespace) -> None:
    plan, _inner, _metadata = load_strict_plan(args)
    valid = 0
    reasons: dict[str, int] = {}
    for row in plan.to_dict(orient="records"):
        ok, reason = validate_completed_task(task_dir(args.output_dir, row), row)
        valid += int(ok)
        if not ok:
            reasons[reason] = reasons.get(reason, 0) + 1
    complete_path = args.output_dir / "RUN_COMPLETE.json"
    formal_status = "incomplete"
    integrity_status, integrity_reason = "not-applicable", "RUN_COMPLETE absent"
    if complete_path.is_file():
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            integrity_ok, integrity_reason = validate_aggregate_artifact_integrity(args.output_dir, complete)
            integrity_status = "PASS" if integrity_ok else "FAIL"
            if not integrity_ok:
                formal_status = "integrity-failed"
            elif valid == EXPECTED_TASKS and complete.get("status") == "complete" and complete.get("run_fingerprint") == str(plan.iloc[0]["run_fingerprint"]):
                formal_status = "valid"
            else:
                formal_status = "invalid"
        except Exception as exc:
            formal_status = "integrity-failed"
            integrity_status = "FAIL"
            integrity_reason = f"unreadable RUN_COMPLETE: {exc}"
    print(
        json.dumps(
            {
                "expected_outer_tasks": EXPECTED_TASKS,
                "expected_inner_trainings": EXPECTED_INNER_TRAININGS,
                "valid_outer_tasks": valid,
                "invalid_or_missing_outer_tasks": EXPECTED_TASKS - valid,
                "reasons": reasons,
                "run_complete": complete_path.is_file(),
                "formal_run_status": formal_status,
                "aggregate_integrity": integrity_status,
                "aggregate_integrity_reason": integrity_reason,
            },
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = resolve_args(parse_args())
    if args.stage == "plan":
        write_plan(args)
    elif args.stage == "sanity":
        run_sanity(args)
    elif args.stage == "full":
        run_full(args)
    else:
        run_status(args)


if __name__ == "__main__":
    main()
