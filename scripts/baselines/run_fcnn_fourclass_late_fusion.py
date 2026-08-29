#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_multiscale_temporal1d as framework
from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.evaluate import confusion_matrix
from ultrasound_decoding.multiframe.fcnn_fourclass_late_fusion import (
    BLOCK_ORDER,
    CHANCE_LEVEL,
    CLASS_NAMES,
    CLASSES,
    EXPECTED_FOURCLASS_PARAMETERS,
    FRAMES_PER_BLOCK,
    MODEL_NAME,
    MODEL_VERSION,
    STRONG_SESSIONS,
    TASK_NAME,
    WEAK_SESSIONS,
    architecture_config,
    binary_metrics_from_fourclass,
    build_model,
    coarse_error_audit,
    collapsed_binary_probabilities,
    collapsed_binary_labels,
    feasibility_gate,
    fixed_class_metrics,
    frozen_training_config,
    json_list,
    load_fourclass_block_session,
    normalized_confusion,
    parameter_audit,
    train_fold,
)


EXPECTED_SESSIONS = ("626", "628", "708", "709", "710", "807", "813", "817", "822")
EXPECTED_CYCLES = {"626": 8, "628": 8, "708": 6, "709": 22, "710": 18, "807": 12, "813": 10, "817": 20, "822": 10}
SEEDS = (0, 1, 2)
EXPECTED_FOLDS = 82
EXPECTED_TASKS = 246
FORMAL_EPOCHS = 40
FORMAL_BATCH_SIZE = 16
REFERENCE_VARIANT = "mean_only"
REQUIRED_TASK_FILES = (
    "checkpoint.pt",
    "result.json",
    "predictions.csv",
    "frame_predictions.csv",
    "training_history.csv",
    "normalization_audit.json",
)
REQUIRED_RUN_OUTPUTS = (
    "config.json",
    "runtime_fingerprint.json",
    "task_plan.csv",
    "split_manifest.csv",
    "checkpoint_manifest.csv",
    "normalization_audit.csv",
    "training_history.csv",
    "frame_predictions.csv",
    "predictions.csv",
    "fold_summary.csv",
    "session_seed_summary.csv",
    "session_summary.csv",
    "confusion_summary.csv",
    "binary_collapse_summary.csv",
    "statistical_audit.json",
    "provenance_audit.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict four-class FCNN late-fusion v1")
    parser.add_argument("--stage", choices=("plan", "sanity", "full", "status"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fcnn_fourclass_late_fusion_v1"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--fold-reference-dir", type=Path, default=Path("outputs/fcnn_mean_std_temporal_statistics_v1"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sanity-session", default="708")
    parser.add_argument("--sanity-epochs", type=int, default=1)
    parser.add_argument("--review-approved", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.resolve()
    for name in ("output_dir", "fold_reference_dir"):
        value = getattr(args, name)
        setattr(args, name, value.resolve() if value.is_absolute() else (args.project_root / value).resolve())
    if args.data_dir is None:
        args.data_dir = args.project_root / "processed_data" / "block_sequences_v1"
    else:
        args.data_dir = args.data_dir.resolve() if args.data_dir.is_absolute() else (args.project_root / args.data_dir).resolve()
    return args


def protocol_config() -> dict[str, Any]:
    return {
        "experiment": "fcnn_fourclass_late_fusion_v1",
        "task": TASK_NAME,
        "class_mapping": CLASS_NAMES,
        "class_order": BLOCK_ORDER,
        "chance_level": CHANCE_LEVEL,
        "sessions": list(EXPECTED_SESSIONS),
        "expected_cycles": EXPECTED_CYCLES,
        "seeds": list(SEEDS),
        "outer_cv": "exact_cycle_membership_from_fcnn_mean_std_temporal_statistics_v1_mean_only",
        "expected_folds": EXPECTED_FOLDS,
        "expected_tasks": EXPECTED_TASKS,
        "input": "existing_clean4_four_frames_per_block",
        "normalization": "arcsinh_then_pixel_zscore_fit_on_outer_training_frames_only",
        "model": architecture_config(),
        "training": vars(frozen_training_config(FORMAL_EPOCHS)),
        "primary_metric": "concatenated_block_level_oof_balanced_accuracy_per_session_seed",
        "feasibility_gate": {
            "fourclass_9session_mean_ba_gte": 0.35,
            "minimum_sessions": 6,
            "session_ba_strictly_gt": 0.30,
            "collapsed_binary_9session_mean_ba_gte": 0.55,
        },
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "early_stopping": False,
        "hyperparameter_search": False,
    }


def read_declared_checksums(data_dir: Path) -> dict[str, str]:
    path = data_dir / "checksums.sha256"
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.strip()] = digest
    return entries


def formal_source_paths(project_root: Path) -> list[Path]:
    return [
        project_root / "src/ultrasound_decoding/multiframe/fcnn_fourclass_late_fusion.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/deep.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "src/ultrasound_decoding/cv.py",
        project_root / "scripts/baselines/run_multiscale_temporal1d.py",
        Path(__file__).resolve(),
    ]


def current_git_head(project_root: Path) -> str:
    value = framework.git_text(project_root, "rev-parse", "HEAD")
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise AssertionError(f"unable to resolve exact Git HEAD: {value!r}")
    return value


def source_identities(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    declared = read_declared_checksums(args.data_dir)
    datasets: dict[str, Any] = {}
    for session in EXPECTED_SESSIONS:
        names = [f"session_{session}_blocks.h5", f"session_{session}_block_metadata.csv"]
        missing = [name for name in names if name not in declared]
        if missing:
            raise AssertionError(f"dataset checksum manifest missing {missing}")
        actual = {name: framework.file_sha256(args.data_dir / name) for name in names}
        if any(actual[name] != declared[name] for name in names):
            raise AssertionError(f"dataset checksum mismatch for session {session}")
        datasets[session] = {"declared_sha256": {name: declared[name] for name in names}, "actual_sha256": actual}
    source_paths = formal_source_paths(args.project_root)
    source_hashes = {str(path.relative_to(args.project_root)): framework.file_sha256(path) for path in source_paths}
    return datasets, source_hashes


def reference_plan(args: argparse.Namespace) -> pd.DataFrame:
    complete_path = args.fold_reference_dir / "RUN_COMPLETE.json"
    plan_path = args.fold_reference_dir / "task_plan.csv"
    if not complete_path.is_file() or not plan_path.is_file():
        raise FileNotFoundError("formal fold reference is incomplete")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or int(complete.get("number_of_folds", -1)) != EXPECTED_FOLDS:
        raise AssertionError("formal fold reference RUN_COMPLETE is invalid")
    plan = pd.read_csv(plan_path, dtype={"session": str})
    plan = plan[plan["variant"].astype(str) == REFERENCE_VARIANT].copy()
    required = {"session", "seed", "fold", "train_cycles", "test_cycles"}
    if not required.issubset(plan.columns) or len(plan) != EXPECTED_TASKS:
        raise AssertionError("formal fold reference does not contain 246 mean-only tasks")
    return plan


def cycle_text(values: np.ndarray) -> str:
    return ",".join(str(int(value)) for value in sorted(np.unique(values).tolist()))


def assert_reference_fold_match(
    reference: pd.DataFrame,
    session: str,
    fold: int,
    train_cycles: str,
    test_cycles: str,
) -> None:
    rows = reference[(reference["session"].astype(str) == str(session)) & (reference["fold"].astype(int) == int(fold))]
    if len(rows) != len(SEEDS) or set(rows["seed"].astype(int)) != set(SEEDS):
        raise AssertionError(f"reference coverage mismatch for {session} fold {fold}")
    if set(rows["train_cycles"].astype(str)) != {train_cycles} or set(rows["test_cycles"].astype(str)) != {test_cycles}:
        raise AssertionError(f"STOP: exact formal fold membership mismatch for {session} fold {fold}")


def build_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    datasets, source_hashes = source_identities(args)
    reference = reference_plan(args)
    split_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        data = load_fourclass_block_session(args.project_root, session, args.data_dir)
        if data.n_cycles != EXPECTED_CYCLES[session]:
            raise AssertionError(f"session {session} cycle count drift")
        splits = grouped_cv_splits(data.groups, max_folds=10)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_cycles = cycle_text(data.groups[train_idx])
            test_cycles = cycle_text(data.groups[test_idx])
            if set(data.groups[train_idx]) & set(data.groups[test_idx]):
                raise AssertionError("cycle leaked across the outer fold")
            assert_reference_fold_match(reference, session, fold, train_cycles, test_cycles)
            dataset_fp = framework.fingerprint(datasets[session])
            split_fp = framework.fingerprint({"session": session, "fold": fold, "train_cycles": train_cycles, "test_cycles": test_cycles})
            split_rows.append({
                "session": session,
                "fold": fold,
                "train_cycles": train_cycles,
                "test_cycles": test_cycles,
                "n_train_blocks": len(train_idx),
                "n_train_frame_samples": len(train_idx) * FRAMES_PER_BLOCK,
                "n_test_blocks": len(test_idx),
                "n_test_frames": len(test_idx) * FRAMES_PER_BLOCK,
                "dataset_fingerprint": dataset_fp,
                "split_fingerprint": split_fp,
                "formal_reference_exact_match": True,
            })
            for seed in SEEDS:
                task_rows.append({
                    "session": session,
                    "seed": seed,
                    "fold": fold,
                    "train_cycles": train_cycles,
                    "test_cycles": test_cycles,
                    "n_train_blocks": len(train_idx),
                    "n_train_frame_samples": len(train_idx) * FRAMES_PER_BLOCK,
                    "n_test_blocks": len(test_idx),
                    "n_test_frames": len(test_idx) * FRAMES_PER_BLOCK,
                    "dataset_fingerprint": dataset_fp,
                    "split_fingerprint": split_fp,
                    "task_key": f"{session}:{seed}:{fold}",
                })
        del data
    split_manifest = pd.DataFrame(split_rows)
    plan = pd.DataFrame(task_rows)
    if len(split_manifest) != EXPECTED_FOLDS or len(plan) != EXPECTED_TASKS:
        raise AssertionError("formal fold/task total drift")
    if plan["task_key"].duplicated().any():
        raise AssertionError("duplicate task keys")
    identity = {
        "protocol": protocol_config(),
        "dataset_identities": datasets,
        "source_hashes": source_hashes,
        "git_head": current_git_head(args.project_root),
        "fold_reference": {
            "directory": str(args.fold_reference_dir),
            "task_plan_sha256": framework.file_sha256(args.fold_reference_dir / "task_plan.csv"),
            "run_complete_sha256": framework.file_sha256(args.fold_reference_dir / "RUN_COMPLETE.json"),
        },
        "runtime": framework.runtime_environment_signature(),
    }
    run_fp = framework.fingerprint(identity)
    config_fp = framework.fingerprint(identity["protocol"])
    runtime_fp = framework.fingerprint(identity["runtime"])
    plan["run_fingerprint"] = run_fp
    plan["config_fingerprint"] = config_fp
    plan["runtime_fingerprint"] = runtime_fp
    plan["git_head"] = identity["git_head"]
    plan["task_fingerprint"] = [framework.fingerprint({"run_fingerprint": run_fp, **row}) for row in task_rows]
    totals = {
        "sessions": len(EXPECTED_SESSIONS),
        "folds": len(split_manifest),
        "seeds": len(SEEDS),
        "models": 1,
        "expected_tasks": len(plan),
        "classes": len(CLASSES),
        "chance": CHANCE_LEVEL,
        "expected_training_blocks": int(plan["n_train_blocks"].sum()),
        "expected_training_frame_samples": int(plan["n_train_frame_samples"].sum()),
        "expected_heldout_blocks": int(plan["n_test_blocks"].sum()),
        "expected_heldout_frames": int(plan["n_test_frames"].sum()),
    }
    return plan, split_manifest, {"identity": identity, "totals": totals}


def write_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    plan, splits, metadata = build_plan(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    framework.atomic_json(args.output_dir / "config.json", {**metadata["identity"]["protocol"], "run_fingerprint": plan.iloc[0]["run_fingerprint"]})
    framework.atomic_json(args.output_dir / "runtime_fingerprint.json", {
        "runtime": metadata["identity"]["runtime"],
        "runtime_fingerprint": plan.iloc[0]["runtime_fingerprint"],
        "source_hashes": metadata["identity"]["source_hashes"],
        "dataset_identities": metadata["identity"]["dataset_identities"],
        "fold_reference": metadata["identity"]["fold_reference"],
        "git_head": metadata["identity"]["git_head"],
        "run_fingerprint": plan.iloc[0]["run_fingerprint"],
    })
    framework.atomic_csv(args.output_dir / "task_plan.csv", plan)
    framework.atomic_csv(args.output_dir / "split_manifest.csv", splits)
    framework.atomic_json(args.output_dir / "PLAN_COMPLETE.json", {"status": "complete", "created_utc": utc_now(), **metadata["totals"], "run_fingerprint": plan.iloc[0]["run_fingerprint"]})
    for key, value in metadata["totals"].items():
        print(f"{key}: {value}", flush=True)
    return plan, splits, metadata


def load_strict_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    current_plan, current_splits, current_metadata = build_plan(args)
    plan_path = args.output_dir / "task_plan.csv"
    split_path = args.output_dir / "split_manifest.csv"
    if not plan_path.is_file() or not split_path.is_file():
        raise RuntimeError("plan artifacts are absent; run --stage plan first")
    saved_plan = pd.read_csv(plan_path, dtype={"session": str})
    saved_splits = pd.read_csv(split_path, dtype={"session": str})
    pd.testing.assert_frame_equal(saved_plan, current_plan, check_dtype=False)
    pd.testing.assert_frame_equal(saved_splits, current_splits, check_dtype=False)
    config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    runtime = json.loads((args.output_dir / "runtime_fingerprint.json").read_text(encoding="utf-8"))
    expected_run_fp = str(current_plan.iloc[0]["run_fingerprint"])
    expected_config = {**current_metadata["identity"]["protocol"], "run_fingerprint": expected_run_fp}
    if framework.fingerprint(config) != framework.fingerprint(expected_config):
        raise AssertionError("saved config.json differs from the frozen protocol")
    if runtime.get("run_fingerprint") != expected_run_fp or runtime.get("source_hashes") != current_metadata["identity"]["source_hashes"] or runtime.get("dataset_identities") != current_metadata["identity"]["dataset_identities"] or runtime.get("git_head") != current_metadata["identity"]["git_head"]:
        raise AssertionError("saved runtime/source/dataset fingerprints differ")
    return saved_plan, saved_splits, current_metadata


def task_dir(output_dir: Path, row: dict[str, Any]) -> Path:
    return output_dir / "tasks" / f"session_{row['session']}" / f"seed_{int(row['seed'])}" / f"fold_{int(row['fold']):02d}"


def task_artifact_hashes(path: Path) -> dict[str, str]:
    return {name: framework.file_sha256(path / name) for name in REQUIRED_TASK_FILES}


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
        normalization = json.loads((path / "normalization_audit.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        frames = pd.read_csv(path / "frame_predictions.csv", dtype={"session": str})
        history = pd.read_csv(path / "training_history.csv", dtype={"session": str})
        checkpoint = load_checkpoint(path / "checkpoint.pt")
    except Exception as exc:
        return fail(f"unreadable artifact: {exc}")
    expected_key = str(expected["task_key"])
    if complete.get("status") != "complete" or complete.get("task_key") != expected_key:
        return fail("completion identity mismatch")
    if complete.get("artifact_sha256") != task_artifact_hashes(path):
        return fail("artifact hashes mismatch")
    exact_fields = ("session", "seed", "fold", "train_cycles", "test_cycles", "dataset_fingerprint", "split_fingerprint", "task_fingerprint", "run_fingerprint", "git_head")
    for payload_name, payload in (("result", result), ("checkpoint", checkpoint)):
        for field in exact_fields:
            expected_value = int(expected[field]) if field in {"seed", "fold"} else str(expected[field])
            observed_value = int(payload.get(field, -1)) if field in {"seed", "fold"} else str(payload.get(field))
            if observed_value != expected_value:
                return fail(f"{payload_name} {field} mismatch")
    if checkpoint.get("task") != TASK_NAME or checkpoint.get("model_name") != MODEL_NAME or checkpoint.get("model_version") != MODEL_VERSION:
        return fail("checkpoint task/model provenance mismatch")
    if result.get("task") != TASK_NAME or result.get("model_name") != MODEL_NAME or result.get("model_version") != MODEL_VERSION:
        return fail("result task/model provenance mismatch")
    if checkpoint.get("class_mapping") != CLASS_NAMES or checkpoint.get("classes") != CLASSES.tolist():
        return fail("checkpoint class mapping mismatch")
    if result.get("class_mapping") != {str(key): value for key, value in CLASS_NAMES.items()} and result.get("class_mapping") != CLASS_NAMES:
        return fail("result class mapping mismatch")
    if int(checkpoint.get("epoch", -1)) != FORMAL_EPOCHS:
        return fail("checkpoint is not epoch 40")
    if checkpoint.get("optimizer_config") != vars(frozen_training_config()):
        return fail("checkpoint optimizer config mismatch")
    try:
        model = build_model()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except Exception as exc:
        return fail(f"checkpoint/output head is not loadable: {exc}")
    if model[-1].out_features != 4:
        return fail("output head is not four-class")
    norm_mean = np.asarray(checkpoint.get("normalization_mean"))
    norm_std = np.asarray(checkpoint.get("normalization_std"))
    if tuple(norm_mean.shape) != (1, 128, 501) or tuple(norm_std.shape) != (1, 128, 501):
        return fail("normalization arrays have wrong shape")
    if not np.isfinite(norm_mean).all() or not np.isfinite(norm_std).all() or not np.all(norm_std > 0):
        return fail("normalization arrays are invalid")
    if checkpoint.get("normalization_config") != normalization or checkpoint.get("source_protocol_fingerprint") != str(expected["run_fingerprint"]) or checkpoint.get("dataset_manifest_fingerprint") != str(expected["dataset_fingerprint"]):
        return fail("checkpoint normalization/source fingerprint mismatch")
    if normalization.get("phase") != "outer_train_fold_only" or normalization.get("target_used_for_stats") is not False or normalization.get("test_used_for_normalization_fit") is not False:
        return fail("normalization audit failed")
    if len(history) != FORMAL_EPOCHS or not np.array_equal(history["epoch"].to_numpy(int), np.arange(1, FORMAL_EPOCHS + 1)):
        return fail("training history is not fixed epochs 1..40")
    if len(predictions) != int(expected["n_test_blocks"]) or predictions["block_id"].duplicated().any():
        return fail("held-out block coverage is incomplete or duplicated")
    if len(frames) != int(expected["n_test_frames"]) or frames[["block_id", "frame_position"]].duplicated().any():
        return fail("held-out frame coverage is incomplete or duplicated")
    if set(predictions["cycle_id"].astype(int)) != {int(value) for value in str(expected["test_cycles"]).split(",")}:
        return fail("prediction fold membership mismatch")
    if set(predictions["block_name"].astype(str)) != set(BLOCK_ORDER):
        return fail("prediction class coverage mismatch")
    expected_truth = predictions["block_name"].map({name: index for index, name in CLASS_NAMES.items()})
    if expected_truth.isna().any() or not np.array_equal(expected_truth.to_numpy(int), predictions["true_label"].to_numpy(int)):
        return fail("prediction label/name mapping mismatch")
    prob_cols = [f"prob_{name}" for name in BLOCK_ORDER]
    values = predictions[prob_cols].to_numpy(float)
    frame_values = frames[prob_cols].to_numpy(float)
    if not np.isfinite(values).all() or not np.allclose(values.sum(1), 1.0, atol=1e-6):
        return fail("block probabilities invalid")
    if not np.isfinite(frame_values).all() or not np.allclose(frame_values.sum(1), 1.0, atol=1e-6):
        return fail("frame probabilities invalid")
    fused = frame_values.reshape(-1, FRAMES_PER_BLOCK, 4).mean(axis=1)
    if not np.allclose(fused, values, atol=1e-6):
        return fail("saved late fusion is inconsistent")
    if not np.array_equal(values.argmax(axis=1), predictions["pred_label"].to_numpy(int)):
        return fail("saved block prediction is not probability argmax")
    for block_id, group in frames.groupby("block_id", sort=False):
        if sorted(group["frame_position"].astype(int).tolist()) != list(range(FRAMES_PER_BLOCK)):
            return fail(f"frame positions incomplete for {block_id}")
    return True, "validated"


def indices_for_row(data: Any, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    train_cycles = {int(value) for value in str(row["train_cycles"]).split(",")}
    test_cycles = {int(value) for value in str(row["test_cycles"]).split(",")}
    if train_cycles & test_cycles or train_cycles | test_cycles != set(data.groups.tolist()):
        raise AssertionError("task cycles are not an exact non-overlapping partition")
    return np.flatnonzero(np.isin(data.groups, sorted(train_cycles))), np.flatnonzero(np.isin(data.groups, sorted(test_cycles)))


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_task(args: argparse.Namespace, row: dict[str, Any], *, epochs: int = FORMAL_EPOCHS, destination: Path | None = None) -> Path:
    data = load_fourclass_block_session(args.project_root, str(row["session"]), args.data_dir)
    train_idx, test_idx = indices_for_row(data, row)
    result = train_fold(
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        session=str(row["session"]),
        seed=int(row["seed"]),
        fold=int(row["fold"]),
        train_cycles=str(row["train_cycles"]),
        test_cycles=str(row["test_cycles"]),
        config=frozen_training_config(epochs),
        device=args.device,
    )
    path = destination or task_dir(args.output_dir, row)
    path.mkdir(parents=True, exist_ok=True)
    metadata = data.metadata.iloc[test_idx].reset_index(drop=True)
    probabilities = result.block_probabilities
    predictions = pd.DataFrame({
        "session": str(row["session"]),
        "seed": int(row["seed"]),
        "fold": int(row["fold"]),
        "cycle_id": data.groups[test_idx],
        "block_id": metadata["block_id"].astype(str),
        "block_name": metadata["block_name"].astype(str),
        "true_label": data.y[test_idx],
        "pred_label": result.predictions,
        **{f"prob_{name}": probabilities[:, index] for index, name in CLASS_NAMES.items()},
        "clean4_original_frame_indices": [json_list(values) for values in data.clean4_original_frame_indices[test_idx]],
        "clean4_relative_time_s": [json_list(values) for values in data.clean4_relative_time_s[test_idx]],
    })
    frame_rows = []
    for block_i, source_i in enumerate(test_idx):
        for position in range(FRAMES_PER_BLOCK):
            probability = result.frame_probabilities[block_i * FRAMES_PER_BLOCK + position]
            frame_rows.append({
                "session": str(row["session"]),
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "cycle_id": int(data.groups[source_i]),
                "block_id": str(data.metadata.iloc[source_i]["block_id"]),
                "block_name": str(data.metadata.iloc[source_i]["block_name"]),
                "true_label": int(data.y[source_i]),
                "frame_position": position,
                "original_frame_index": int(data.clean4_original_frame_indices[source_i, position]),
                "relative_time_s": float(data.clean4_relative_time_s[source_i, position]),
                **{f"prob_{name}": float(probability[index]) for index, name in CLASS_NAMES.items()},
            })
    history = pd.DataFrame(result.history)
    history.insert(0, "fold", int(row["fold"]))
    history.insert(0, "seed", int(row["seed"]))
    history.insert(0, "session", str(row["session"]))
    metrics = fixed_class_metrics(data.y[test_idx], result.predictions)
    result_payload = {
        **{key: int(row[key]) if key in {"seed", "fold"} else str(row[key]) for key in ("session", "seed", "fold", "train_cycles", "test_cycles", "dataset_fingerprint", "split_fingerprint", "task_fingerprint", "run_fingerprint", "git_head")},
        "task": TASK_NAME,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "class_mapping": CLASS_NAMES,
        "classes": CLASSES.tolist(),
        "trainable_parameters": parameter_audit()["fourclass_parameters"],
        "epoch": int(epochs),
        "n_train_blocks": len(train_idx),
        "n_train_frame_samples": len(train_idx) * FRAMES_PER_BLOCK,
        "n_test_blocks": len(test_idx),
        "n_test_frames": len(test_idx) * FRAMES_PER_BLOCK,
        "frame_level_train_accuracy": result.train_frame_accuracy,
        "block_level_train_accuracy": result.train_block_accuracy,
        "block_level_train_balanced_accuracy": result.train_block_balanced_accuracy,
        "heldout_block_metrics": metrics,
        "device": result.device,
    }
    checkpoint = {
        **{key: result_payload[key] for key in ("session", "seed", "fold", "train_cycles", "test_cycles", "dataset_fingerprint", "split_fingerprint", "task_fingerprint", "run_fingerprint", "git_head")},
        "task": TASK_NAME,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "classes": CLASSES.tolist(),
        "class_mapping": CLASS_NAMES,
        "architecture_config": architecture_config(),
        "model_state_dict": {key: value.detach().cpu() for key, value in result.model.state_dict().items()},
        "normalization_mean": result.normalization_mean,
        "normalization_std": result.normalization_std,
        "normalization_config": result.normalization_audit,
        "epoch": int(epochs),
        "optimizer_config": vars(frozen_training_config(epochs)),
        "source_protocol_fingerprint": str(row["run_fingerprint"]),
        "dataset_manifest_fingerprint": str(row["dataset_fingerprint"]),
    }
    framework.atomic_csv(path / "predictions.csv", predictions)
    framework.atomic_csv(path / "frame_predictions.csv", pd.DataFrame(frame_rows))
    framework.atomic_csv(path / "training_history.csv", history)
    framework.atomic_json(path / "normalization_audit.json", result.normalization_audit)
    atomic_torch_save(path / "checkpoint.pt", checkpoint)
    framework.atomic_json(path / "result.json", result_payload)
    if epochs == FORMAL_EPOCHS and destination is None:
        framework.atomic_json(path / "COMPLETE.json", {
            "status": "complete",
            "task_key": str(row["task_key"]),
            "completed_utc": utc_now(),
            "artifact_sha256": task_artifact_hashes(path),
        })
    return path


def run_sanity(args: argparse.Namespace) -> None:
    plan, _, _ = load_strict_plan(args)
    candidates = plan[(plan["session"] == str(args.sanity_session)) & (plan["seed"] == 0) & (plan["fold"] == 1)]
    if len(candidates) != 1:
        raise AssertionError("sanity task selection is not unique")
    row = candidates.iloc[0].to_dict()
    path = write_task(args, row, epochs=int(args.sanity_epochs), destination=args.output_dir / "sanity")
    result = json.loads((path / "result.json").read_text(encoding="utf-8"))
    framework.atomic_json(path / "SANITY_COMPLETE.json", {
        "status": "complete",
        "formal_task_marked_complete": False,
        "epochs": int(args.sanity_epochs),
        "session": str(row["session"]),
        "seed": int(row["seed"]),
        "fold": int(row["fold"]),
        "heldout_block_metrics": result["heldout_block_metrics"],
        "completed_utc": utc_now(),
    })
    print(f"sanity complete: {path}", flush=True)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    return fixed_class_metrics(frame["true_label"].to_numpy(int), frame["pred_label"].to_numpy(int))


def confusion_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, str, pd.DataFrame]] = [("aggregate", "all", predictions)]
    scopes.extend(("session", str(session), group) for session, group in predictions.groupby("session", sort=True))
    for scope, scope_id, group in scopes:
        matrix = confusion_matrix(group["true_label"].to_numpy(int), group["pred_label"].to_numpy(int), CLASSES)
        normalized = normalized_confusion(matrix)
        coarse = coarse_error_audit(matrix)
        for true_index, true_name in CLASS_NAMES.items():
            for pred_index, pred_name in CLASS_NAMES.items():
                rows.append({
                    "scope": scope,
                    "scope_id": scope_id,
                    "true_label": true_index,
                    "true_class": true_name,
                    "pred_label": pred_index,
                    "pred_class": pred_name,
                    "count": int(matrix[true_index, pred_index]),
                    "row_normalized": float(normalized[true_index, pred_index]),
                    **coarse,
                })
    return pd.DataFrame(rows)


def load_descriptive_binary_comparator(
    args: argparse.Namespace,
    source_dir: Path | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    directory = source_dir or args.project_root / "outputs/fcnn_canonical_single_frame_v1"
    paths = {
        "session_summary.csv": directory / "session_summary.csv",
        "RUN_COMPLETE.json": directory / "RUN_COMPLETE.json",
        "provenance_audit.json": directory / "provenance_audit.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"canonical descriptive binary comparator missing {missing}")
    complete = json.loads(paths["RUN_COMPLETE.json"].read_text(encoding="utf-8"))
    provenance = json.loads(paths["provenance_audit.json"].read_text(encoding="utf-8"))
    if complete.get("status") != "complete" or int(complete.get("number_of_sessions", -1)) != len(EXPECTED_SESSIONS):
        raise AssertionError("canonical binary comparator RUN_COMPLETE is invalid")
    if provenance.get("status") != "validated":
        raise AssertionError("canonical binary comparator provenance is not validated")
    frame = pd.read_csv(paths["session_summary.csv"], dtype={"session": str})
    required_columns = {"session", "late_fusion_BA"}
    if not required_columns.issubset(frame.columns):
        raise AssertionError("canonical binary comparator summary columns are incomplete")
    sessions = frame["session"].astype(str)
    if len(frame) != len(EXPECTED_SESSIONS) or sessions.duplicated().any() or set(sessions) != set(EXPECTED_SESSIONS):
        raise AssertionError("canonical binary comparator must contain exactly 9 unique formal sessions")
    values = pd.to_numeric(frame["late_fusion_BA"], errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise AssertionError("canonical binary comparator contains non-finite BA")
    comparison = dict(zip(sessions.tolist(), values.tolist()))
    source_provenance = {
        "source": "outputs/fcnn_canonical_single_frame_v1",
        "source_directory": str(directory),
        "source_artifact_sha256": {name: framework.file_sha256(path) for name, path in sorted(paths.items())},
        "session_count": len(comparison),
        "sessions": sorted(comparison),
        "role": "descriptive_only",
        "used_for_training": False,
        "used_for_model_selection": False,
        "used_for_feasibility_gate": False,
        "canonical_provenance_status": "validated",
    }
    return comparison, source_provenance


def aggregate_outputs(args: argparse.Namespace, plan: pd.DataFrame, metadata: dict[str, Any]) -> None:
    prediction_frames, frame_frames, history_frames = [], [], []
    normalization_rows, fold_rows, checkpoint_rows = [], [], []
    for row in plan.to_dict(orient="records"):
        path = task_dir(args.output_dir, row)
        validate_completed_task(path, row, raise_on_error=True)
        predictions = pd.read_csv(path / "predictions.csv", dtype={"session": str})
        frames = pd.read_csv(path / "frame_predictions.csv", dtype={"session": str})
        history = pd.read_csv(path / "training_history.csv", dtype={"session": str})
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        normalization = json.loads((path / "normalization_audit.json").read_text(encoding="utf-8"))
        prediction_frames.append(predictions)
        frame_frames.append(frames)
        history_frames.append(history)
        normalization_rows.append(normalization)
        fold_rows.append({
            "session": str(row["session"]), "seed": int(row["seed"]), "fold": int(row["fold"]),
            **result["heldout_block_metrics"],
            "frame_level_train_accuracy": result["frame_level_train_accuracy"],
            "block_level_train_accuracy": result["block_level_train_accuracy"],
            "block_level_train_balanced_accuracy": result["block_level_train_balanced_accuracy"],
            "n_train_blocks": result["n_train_blocks"], "n_test_blocks": result["n_test_blocks"],
        })
        checkpoint_rows.append({
            "task_key": row["task_key"], "session": str(row["session"]), "seed": int(row["seed"]), "fold": int(row["fold"]),
            "checkpoint_path": str((path / "checkpoint.pt").relative_to(args.output_dir)),
            "checkpoint_sha256": framework.file_sha256(path / "checkpoint.pt"), "validation": "PASS",
        })
    predictions = pd.concat(prediction_frames, ignore_index=True)
    frame_predictions = pd.concat(frame_frames, ignore_index=True)
    histories = pd.concat(history_frames, ignore_index=True)
    probability_columns = [f"prob_{name}" for name in BLOCK_ORDER]
    expected_prediction_rows = int(plan["n_test_blocks"].sum())
    if len(predictions) != expected_prediction_rows or len(frame_predictions) != expected_prediction_rows * FRAMES_PER_BLOCK:
        raise AssertionError("aggregate prediction coverage mismatch")
    if predictions[["session", "seed", "block_id"]].duplicated().any():
        raise AssertionError("duplicate OOF block prediction within session/seed")
    if not np.isfinite(predictions[probability_columns].to_numpy(float)).all() or not np.allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError("aggregate probabilities are invalid")

    session_seed_rows, binary_seed_rows = [], []
    for (session, seed), group in predictions.groupby(["session", "seed"], sort=True):
        data = load_fourclass_block_session(args.project_root, str(session), args.data_dir)
        if len(group) != data.n_blocks or set(group["block_id"].astype(str)) != set(data.metadata["block_id"].astype(str)):
            raise AssertionError(f"OOF block coverage mismatch for {session} seed {seed}")
        metrics = prediction_metrics(group)
        session_seed_rows.append({"session": str(session), "seed": int(seed), "n_oof_blocks": len(group), **metrics})
        binary = binary_metrics_from_fourclass(group["true_label"].to_numpy(int), group[probability_columns].to_numpy(float))
        binary_seed_rows.append({"scope": "session_seed", "session": str(session), "seed": int(seed), "n_oof_blocks": len(group), **binary})
    session_seed_summary = pd.DataFrame(session_seed_rows)
    session_rows = []
    for session, group in session_seed_summary.groupby("session", sort=True):
        row: dict[str, Any] = {"session": str(session), "n_seeds": len(group)}
        for metric in ("balanced_accuracy", "accuracy", "macro_f1", *[f"recall_{name}" for name in BLOCK_ORDER]):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=0))
        session_rows.append(row)
    session_summary = pd.DataFrame(session_rows)

    binary_seed = pd.DataFrame(binary_seed_rows)
    historical, comparator_provenance = load_descriptive_binary_comparator(args)
    binary_rows = binary_seed_rows.copy()
    for session, group in binary_seed.groupby("session", sort=True):
        binary_rows.append({
            "scope": "session", "session": str(session), "seed": "all", "n_oof_blocks": int(group["n_oof_blocks"].iloc[0]),
            "balanced_accuracy": float(group["balanced_accuracy"].mean()), "accuracy": float(group["accuracy"].mean()),
            "seed_sd": float(group["balanced_accuracy"].std(ddof=0)), "historical_binary_fcnn_late_fusion_ba": historical.get(str(session), np.nan),
        })
    binary_session = pd.DataFrame(binary_rows)
    only_sessions = binary_session[binary_session["scope"] == "session"]
    for scope, sessions in (("overall", EXPECTED_SESSIONS), ("strong", STRONG_SESSIONS), ("weak", WEAK_SESSIONS)):
        selected = only_sessions[only_sessions["session"].isin(sessions)]
        binary_rows.append({
            "scope": scope, "session": "all", "seed": "all", "n_oof_blocks": int(selected["n_oof_blocks"].sum()),
            "balanced_accuracy": float(selected["balanced_accuracy"].mean()), "accuracy": float(selected["accuracy"].mean()),
            "seed_sd": np.nan, "historical_binary_fcnn_late_fusion_ba": float(selected["historical_binary_fcnn_late_fusion_ba"].mean()),
        })
    binary_summary = pd.DataFrame(binary_rows)
    session_bas = session_summary["balanced_accuracy_mean"].to_numpy(float)
    collapsed_mean = float(binary_summary.loc[binary_summary["scope"] == "overall", "balanced_accuracy"].iloc[0])
    mean_ba = float(session_bas.mean())
    statistical = {
        "primary_metric": "mean_across_9_sessions_of_session_seed_concatenated_block_level_OOF_BA",
        "fourclass_9session_mean_ba": mean_ba,
        "fourclass_9session_median_ba": float(np.median(session_bas)),
        "mean_ba_minus_chance": mean_ba - CHANCE_LEVEL,
        "sessions_gt_0_25": int(np.sum(session_bas > 0.25)),
        "sessions_gt_0_30": int(np.sum(session_bas > 0.30)),
        "sessions_gt_0_35": int(np.sum(session_bas > 0.35)),
        "strong_mean_ba": float(session_summary[session_summary["session"].isin(STRONG_SESSIONS)]["balanced_accuracy_mean"].mean()),
        "weak_mean_ba": float(session_summary[session_summary["session"].isin(WEAK_SESSIONS)]["balanced_accuracy_mean"].mean()),
        "collapsed_binary_9session_mean_ba": collapsed_mean,
        "feasibility_gate": feasibility_gate(mean_ba, session_bas, collapsed_mean),
        "descriptive_binary_comparison_used_for_training_or_selection": False,
        "descriptive_binary_comparator": comparator_provenance,
    }
    provenance = {
        "status": "PASS",
        "formal_reference_exact_match": True,
        "task_key_exact": True,
        "checkpoint_loadable_and_metadata_exact": True,
        "checkpoint_hashes_valid": True,
        "prediction_coverage_complete": True,
        "no_duplicate_block_predictions_within_session_seed": True,
        "normalization_audit": "PASS",
        "class_mapping_exact": CLASS_NAMES,
        "git_head": metadata["identity"]["git_head"],
        "source_hashes": metadata["identity"]["source_hashes"],
        "dataset_identities": metadata["identity"]["dataset_identities"],
        "descriptive_binary_comparator": comparator_provenance,
    }
    framework.atomic_csv(args.output_dir / "checkpoint_manifest.csv", pd.DataFrame(checkpoint_rows))
    framework.atomic_csv(args.output_dir / "normalization_audit.csv", pd.DataFrame(normalization_rows))
    framework.atomic_csv(args.output_dir / "training_history.csv", histories)
    framework.atomic_csv(args.output_dir / "frame_predictions.csv", frame_predictions)
    framework.atomic_csv(args.output_dir / "predictions.csv", predictions)
    framework.atomic_csv(args.output_dir / "fold_summary.csv", pd.DataFrame(fold_rows))
    framework.atomic_csv(args.output_dir / "session_seed_summary.csv", session_seed_summary)
    framework.atomic_csv(args.output_dir / "session_summary.csv", session_summary)
    framework.atomic_csv(args.output_dir / "confusion_summary.csv", confusion_rows(predictions))
    framework.atomic_csv(args.output_dir / "binary_collapse_summary.csv", binary_summary)
    framework.atomic_json(args.output_dir / "statistical_audit.json", statistical)
    framework.atomic_json(args.output_dir / "provenance_audit.json", provenance)


def aggregate_artifact_sha256(output_dir: Path) -> dict[str, str]:
    missing = [name for name in REQUIRED_RUN_OUTPUTS if not (output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"required aggregate artifacts missing: {missing}")
    return {name: framework.file_sha256(output_dir / name) for name in sorted(REQUIRED_RUN_OUTPUTS)}


def validate_aggregate_artifact_integrity(
    output_dir: Path,
    completion: dict[str, Any],
) -> tuple[bool, str]:
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
    plan, _, metadata = load_strict_plan(args)
    (args.output_dir / "RUN_COMPLETE.json").unlink(missing_ok=True)
    completed = 0
    for row in plan.to_dict(orient="records"):
        path = task_dir(args.output_dir, row)
        valid, reason = validate_completed_task(path, row)
        if valid:
            completed += 1
            print(f"resume skip {row['task_key']}", flush=True)
            continue
        print(f"run {row['task_key']} ({reason})", flush=True)
        write_task(args, row)
        validate_completed_task(path, row, raise_on_error=True)
        completed += 1
    if completed != EXPECTED_TASKS:
        raise AssertionError("formal task coverage is not 246/246")
    aggregate_outputs(args, plan, metadata)
    missing = [name for name in REQUIRED_RUN_OUTPUTS if not (args.output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"required formal outputs missing: {missing}")
    checkpoint_manifest = pd.read_csv(args.output_dir / "checkpoint_manifest.csv")
    provenance = json.loads((args.output_dir / "provenance_audit.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.output_dir / "predictions.csv", dtype={"session": str})
    assert_run_completion_ready(completed, checkpoint_manifest, predictions, provenance)
    aggregate_hashes = aggregate_artifact_sha256(args.output_dir)
    framework.atomic_json(args.output_dir / "RUN_COMPLETE.json", {
        "status": "complete", "completed_utc": utc_now(), "completed_tasks": EXPECTED_TASKS, "expected_tasks": EXPECTED_TASKS,
        "number_of_sessions": len(EXPECTED_SESSIONS), "number_of_folds": EXPECTED_FOLDS, "number_of_seeds": len(SEEDS),
        "checkpoint_coverage": "246/246", "prediction_coverage_complete": True, "no_duplicate_block_predictions": True,
        "normalization_audit": "PASS", "provenance_audit": "PASS", "fourclass_labels_exact": True,
        "probability_audit": "PASS", "collapsed_binary_audit": "complete", "run_fingerprint": str(plan.iloc[0]["run_fingerprint"]),
        "git_head": str(plan.iloc[0]["git_head"]),
        "source_hashes_authoritative": True,
        "aggregate_artifact_sha256": aggregate_hashes,
        "required_outputs": list(REQUIRED_RUN_OUTPUTS),
    })
    print("RUN_COMPLETE written: strict 246/246", flush=True)


def assert_run_completion_ready(
    completed_tasks: int,
    checkpoint_manifest: pd.DataFrame,
    predictions: pd.DataFrame,
    provenance: dict[str, Any],
) -> None:
    if int(completed_tasks) != EXPECTED_TASKS:
        raise AssertionError("formal task coverage is not 246/246")
    if len(checkpoint_manifest) != EXPECTED_TASKS or not checkpoint_manifest["validation"].eq("PASS").all():
        raise AssertionError("checkpoint coverage is not 246/246 PASS")
    if provenance.get("status") != "PASS" or set(predictions["true_label"].astype(int)) != set(CLASSES.tolist()):
        raise AssertionError("final provenance or four-class coverage audit failed")


def run_status(args: argparse.Namespace) -> None:
    plan, _, _ = load_strict_plan(args)
    counts = {"valid": 0, "invalid_or_missing": 0}
    reasons: dict[str, int] = {}
    for row in plan.to_dict(orient="records"):
        valid, reason = validate_completed_task(task_dir(args.output_dir, row), row)
        key = "valid" if valid else "invalid_or_missing"
        counts[key] += 1
        if not valid:
            reasons[reason] = reasons.get(reason, 0) + 1
    complete_path = args.output_dir / "RUN_COMPLETE.json"
    integrity_status, integrity_reason = "not-applicable", "RUN_COMPLETE absent"
    formal_run_status = "incomplete"
    if complete_path.is_file():
        try:
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            integrity_valid, integrity_reason = validate_aggregate_artifact_integrity(args.output_dir, complete)
            integrity_status = "PASS" if integrity_valid else "FAIL"
            metadata_valid = (
                complete.get("status") == "complete"
                and complete.get("git_head") == str(plan.iloc[0]["git_head"])
                and complete.get("run_fingerprint") == str(plan.iloc[0]["run_fingerprint"])
            )
            if not integrity_valid:
                formal_run_status = "integrity-failed"
            elif counts["valid"] == EXPECTED_TASKS and metadata_valid:
                formal_run_status = "valid"
            else:
                formal_run_status = "invalid"
        except Exception as exc:
            integrity_status = "FAIL"
            integrity_reason = f"unreadable RUN_COMPLETE: {exc}"
            formal_run_status = "integrity-failed"
    print(json.dumps({
        "expected": EXPECTED_TASKS,
        **counts,
        "reasons": reasons,
        "run_complete": complete_path.is_file(),
        "formal_run_status": formal_run_status,
        "aggregate_integrity": integrity_status,
        "aggregate_integrity_reason": integrity_reason,
    }, indent=2), flush=True)


def main() -> None:
    args = resolve_paths(parse_args())
    audit = parameter_audit()
    if audit != {"historical_binary_parameters": 48_011, "fourclass_parameters": EXPECTED_FOURCLASS_PARAMETERS, "delta_parameters": 8}:
        raise AssertionError(f"parameter audit failed: {audit}")
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
