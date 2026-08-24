#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
    split_manifest,
)
from ultrasound_decoding.multiframe.sbind_adapted import (
    SBIND_ADAPTED_DISPLAY_NAMES,
    SBIND_ADAPTED_METHODS,
    SBINDAdaptedConfig,
    build_sbind_adapted_model,
    sbind_adapted_architecture_config,
    train_sbind_adapted_fold,
)
from ultrasound_decoding.multiframe.training import DeepTrainingConfig, blocks_to_sequence_tensor


OUTPUT_VERSION = "sbind_visual_binary_v1"
TASK_NAME = "binary"
SEEDS = (0, 1, 2)
MAX_FOLDS = 10
REQUIRED_FINAL_OUTPUTS = (
    "sbind_summary.csv",
    "sbind_per_seed.csv",
    "sbind_per_fold.csv",
    "sbind_predictions.csv",
    "sbind_confusion_matrices.csv",
    "sbind_vs_existing_baselines.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed clean4 SBIND-adapted within-session binary baseline."
    )
    parser.add_argument("--stage", choices=("sanity", "full", "status"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--sessions", nargs="+", default=list(EXPECTED_SESSIONS))
    parser.add_argument("--models", nargs="+", choices=SBIND_ADAPTED_METHODS, default=list(SBIND_ADAPTED_METHODS))
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / OUTPUT_VERSION,
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=PROJECT_DIR / "results" / "runs" / "multiframe" / "block_clean4_binary_v1",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git_text(project_root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout.strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def environment_payload(device: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    gpu_names = []
    if cuda_available:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    resolved = str(device)
    if device == "auto":
        resolved = "cuda" if cuda_available else "cpu"
    return {
        "compute_environment": "server" if resolved.startswith("cuda") else "local_sanity_or_cpu",
        "created_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "gpu_names": gpu_names,
        "requested_device": device,
        "resolved_device": resolved,
    }


def frozen_payload(batch_size: int) -> dict[str, Any]:
    architecture = SBINDAdaptedConfig(dropout=0.25)
    training = DeepTrainingConfig(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=int(batch_size),
        max_epochs=40,
        dropout=0.25,
        loss="cross_entropy",
    )
    return {
        "output_version": OUTPUT_VERSION,
        "baseline_claim": "SBIND-adapted classification baseline; not full SBIND reproduction",
        "sessions": list(EXPECTED_SESSIONS),
        "task": TASK_NAME,
        "class_mapping": TASK_CLASS_NAMES[TASK_NAME],
        "stimulus_blocks": ["grating", "dot"],
        "non_stimulus_blocks": ["stop_after_grating", "static"],
        "input_unit": "one block",
        "input_shape": list(EXPECTED_BLOCK_SHAPE),
        "data_builder": "load_block_sequence_session (audited clean4; unchanged indices)",
        "cv": "grouped_cv_splits by cycle, max_folds=10",
        "normalization": "arcsinh_then_train_pixel_zscore; train fold only",
        "metrics": ["balanced_accuracy", "accuracy", "macro_f1"],
        "seeds": list(SEEDS),
        "models": list(SBIND_ADAPTED_METHODS),
        "training": training.__dict__,
        "architecture": {
            method: sbind_adapted_architecture_config(
                method, n_classes=2, config=architecture
            )
            for method in SBIND_ADAPTED_METHODS
        },
        "epoch_selection": "fixed 40 epochs; no validation/test early stopping or model selection",
        "test_used_for_training_or_tuning": False,
    }


def write_run_metadata(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing_config = args.output_dir / "config.json"
    if existing_config.exists():
        observed = json.loads(existing_config.read_text())
        if fingerprint(observed) != fingerprint(payload):
            has_formal_tasks = (args.output_dir / "task_plan.csv").exists() or (
                args.output_dir / "tasks"
            ).exists()
            if has_formal_tasks:
                raise RuntimeError(
                    "existing formal output uses a different config; use a new output directory"
                )
    atomic_json(args.output_dir / "config.json", payload)
    atomic_json(args.output_dir / "environment.json", environment_payload(args.device))
    command_text = shlex.join(sys.argv) + "\n"
    atomic_text(args.output_dir / "command.txt", command_text)
    atomic_text(args.output_dir / f"{args.stage}_command.txt", command_text)
    git_state = {
        "commit": git_text(args.project_root, "rev-parse", "HEAD"),
        "branch": git_text(args.project_root, "branch", "--show-current"),
        "changed_files": git_text(args.project_root, "status", "--short").splitlines(),
        "diff_stat": git_text(args.project_root, "diff", "--stat"),
    }
    atomic_json(args.output_dir / "git_state.json", git_state)


def audit_session(args: argparse.Namespace, session: str) -> tuple[Any, list[tuple[np.ndarray, np.ndarray]]]:
    data_dir = args.data_dir or default_block_data_dir(args.project_root)
    data = load_block_sequence_session(args.project_root, session, TASK_NAME, data_dir=data_dir)
    if tuple(data.X.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(f"session {session}: expected clean4 {EXPECTED_BLOCK_SHAPE}, got {data.X.shape}")
    splits = grouped_cv_splits(data.groups, max_folds=MAX_FOLDS)
    current = split_manifest(session, TASK_NAME, data.y, data.groups, splits=splits, max_folds=MAX_FOLDS)
    manifest_candidates = [
        args.benchmark_root / f"session_{session}" / "split_manifest.csv",
        args.project_root / "outputs" / "block_clean4_binary_all_models_9sessions_v1"
        / f"session_{session}" / "split_manifest.csv",
        args.project_root / "results" / "runs" / "multiframe"
        / "block_clean4_binary_all_models_v1" / f"session_{session}" / "split_manifest.csv",
        args.project_root / "outputs" / "frame_count_ablation_v1" / "parts"
        / f"session_{session}" / "k_4" / "pca_lda_flat4" / "seed_0" / "split_manifest.csv",
    ]
    historical_path = next((path for path in manifest_candidates if path.exists()), None)
    if historical_path is None:
        raise FileNotFoundError(
            "verified clean4 split manifest is required; checked: "
            + ", ".join(str(path) for path in manifest_candidates)
        )
    historical = pd.read_csv(historical_path)
    common = [
        "session", "task", "fold", "train_cycles", "test_cycles",
        "n_train_blocks", "n_test_blocks", "train_class_counts", "test_class_counts",
    ]
    historical = historical[common].copy()
    current = current[common].copy()
    historical["session"] = historical["session"].astype(str)
    current["session"] = current["session"].astype(str)
    if not current.equals(historical):
        raise AssertionError(f"session {session}: regenerated folds differ from formal clean4 manifest")
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        overlap = set(data.groups[train_idx].tolist()) & set(data.groups[test_idx].tolist())
        if overlap:
            raise AssertionError(f"session {session} fold {fold_i}: cycle leakage {sorted(overlap)}")
    audit_dir = args.output_dir / "audit" / f"session_{session}"
    atomic_csv(audit_dir / "split_manifest.csv", current)
    atomic_json(
        audit_dir / "dataset.json",
        {
            "session": session,
            "source_h5": str(data.source_h5_path),
            "source_metadata": str(data.source_metadata_path),
            "shape": list(data.X.shape),
            "n_cycles": data.n_cycles,
            "n_samples": data.n_blocks,
            "class_counts": {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(data.y, return_counts=True))
            },
            "clean4_indices_reused_without_redefinition": True,
            "formal_manifest_identical": True,
            "formal_manifest_source": str(historical_path),
        },
    )
    return data, splits


def task_dir(output_dir: Path, session: str, model: str, seed: int, fold: int) -> Path:
    return output_dir / "tasks" / f"session_{session}" / model / f"seed_{seed}" / f"fold_{fold:02d}"


def task_key(session: str, model: str, seed: int, fold: int) -> str:
    return f"{session}:{model}:{seed}:{fold}"


def valid_completed_task(path: Path, expected: pd.Series, run_fingerprint: str) -> bool:
    complete_path = path / "COMPLETE.json"
    result_path = path / "result.json"
    prediction_path = path / "predictions.csv"
    confusion_path = path / "confusion_matrix.csv"
    history_path = path / "training_history.csv"
    normalization_path = path / "normalization_audit.json"
    required = (complete_path, result_path, prediction_path, confusion_path, history_path, normalization_path)
    if not all(item.exists() and item.stat().st_size > 0 for item in required):
        return False
    try:
        complete = json.loads(complete_path.read_text())
        result = json.loads(result_path.read_text())
        predictions = pd.read_csv(prediction_path)
        confusion = pd.read_csv(confusion_path)
        history = pd.read_csv(history_path)
    except Exception:
        return False
    expected_key = task_key(str(expected.session), str(expected.model), int(expected.seed), int(expected.fold))
    return bool(
        complete.get("task_key") == expected_key
        and complete.get("run_fingerprint") == run_fingerprint
        and result.get("run_fingerprint") == run_fingerprint
        and int(result.get("n_test_samples", -1)) == int(expected.n_test_samples)
        and len(predictions) == int(expected.n_test_samples)
        and int(confusion["count"].sum()) == int(expected.n_test_samples)
        and len(history) == 40
        and np.isfinite(predictions[["probability_0", "probability_1"]].to_numpy()).all()
    )


def build_task_plan(args: argparse.Namespace, payload: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        data, splits = audit_session(args, session)
        dataset_rows.append(
            {
                "session": session,
                "n_cycles": data.n_cycles,
                "n_samples": data.n_blocks,
                "n_folds": len(splits),
                "input_shape": canonical_json(list(data.X.shape[1:])),
                "formal_clean4_fold_match": True,
            }
        )
        for model in SBIND_ADAPTED_METHODS:
            for seed in SEEDS:
                for fold, (_, test_idx) in enumerate(splits, start=1):
                    rows.append(
                        {
                            "session": session,
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "n_test_samples": len(test_idx),
                            "task_key": task_key(session, model, seed, fold),
                        }
                    )
        del data
    plan = pd.DataFrame(rows)
    atomic_csv(args.output_dir / "task_plan.csv", plan)
    atomic_csv(args.output_dir / "dataset_and_fold_audit.csv", pd.DataFrame(dataset_rows))
    atomic_json(
        args.output_dir / "task_plan_metadata.json",
        {
            "run_fingerprint": fingerprint(payload),
            "total_tasks": len(plan),
            "task_definition": "session x model x seed x fold",
            "created_utc": utc_now(),
        },
    )
    return plan


def load_or_build_task_plan(args: argparse.Namespace, payload: dict[str, Any]) -> pd.DataFrame:
    plan_path = args.output_dir / "task_plan.csv"
    metadata_path = args.output_dir / "task_plan_metadata.json"
    run_fingerprint = fingerprint(payload)
    if plan_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("existing task plan uses a different frozen config; choose a new output directory")
        return pd.read_csv(plan_path, dtype={"session": str})
    return build_task_plan(args, payload)


def update_status(args: argparse.Namespace, plan: pd.DataFrame, run_fingerprint: str) -> pd.DataFrame:
    rows = []
    for row in plan.itertuples(index=False):
        path = task_dir(args.output_dir, str(row.session), str(row.model), int(row.seed), int(row.fold))
        rows.append(
            {
                "session": str(row.session),
                "model": str(row.model),
                "seed": int(row.seed),
                "fold": int(row.fold),
                "n_test_samples": int(row.n_test_samples),
                "status": "complete" if valid_completed_task(path, pd.Series(row._asdict()), run_fingerprint) else "pending",
                "task_dir": str(path),
            }
        )
    status = pd.DataFrame(rows)
    atomic_csv(args.output_dir / "run_status.csv", status)
    completed = int((status["status"] == "complete").sum())
    total = int(len(status))
    print(f"STATUS completed={completed} pending={total - completed} total={total}", flush=True)
    if completed < total:
        pending = status[status["status"] != "complete"]
        print("NEXT_PENDING " + ", ".join(pending["task_dir"].head(5).tolist()), flush=True)
    else:
        print(f"COMPLETE_MARKER {args.output_dir / 'RUN_COMPLETE.json'}", flush=True)
    return status


def select_balanced_indices(y: np.ndarray, candidates: Iterable[int], per_class: int) -> np.ndarray:
    selected: list[int] = []
    candidates_array = np.asarray(list(candidates), dtype=np.int64)
    for label in sorted(np.unique(y[candidates_array]).tolist()):
        selected.extend(candidates_array[y[candidates_array] == label][:per_class].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def run_sanity(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.workers != 0:
        raise ValueError("local sanity requires --workers 0")
    data, splits = audit_session(args, "710")
    train_idx, test_idx = splits[0]
    tiny_train = select_balanced_indices(data.y, train_idx, per_class=4)
    tiny_test = select_balanced_indices(data.y, test_idx, per_class=2)
    if len(tiny_test) < 2:
        tiny_test = np.asarray(test_idx[: min(4, len(test_idx))], dtype=np.int64)
    config = SBINDAdaptedConfig(dropout=0.25)
    classes = np.asarray([0, 1], dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for model_name in SBIND_ADAPTED_METHODS:
        model = build_sbind_adapted_model(model_name, n_classes=2, config=config).to(args.device)
        x = blocks_to_sequence_tensor(data.X[tiny_train[:2]]).to(args.device)
        y = torch.from_numpy(data.y[tiny_train[:2]].astype(np.int64)).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        before = next(model.parameters()).detach().clone()
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        parameter_changed = not torch.equal(before, next(model.parameters()).detach())
        if tuple(logits.shape) != (len(x), 2) or not torch.isfinite(loss) or not parameter_changed:
            raise AssertionError(f"{model_name}: forward/backward sanity failed")
        fit_config = DeepTrainingConfig(
            optimizer="adamw", lr=1e-3, weight_decay=1e-3,
            batch_size=min(4, len(tiny_train)), max_epochs=2, dropout=0.25,
        )
        result = train_sbind_adapted_fold(
            model_name,
            data.X[tiny_train],
            data.y[tiny_train],
            data.X[tiny_test],
            classes,
            session="710",
            fold=1,
            seed=0,
            train_cycles=cycle_text(data.groups[tiny_train]),
            test_cycles=cycle_text(data.groups[tiny_test]),
            training_config=fit_config,
            architecture_config=config,
            device=args.device,
            workers=0,
        )
        if len(result.history) != 2 or not np.isfinite(result.probabilities).all():
            raise AssertionError(f"{model_name}: tiny two-epoch fit failed")
        rows.append(
            {
                "session": "710",
                "model": model_name,
                "input_shape": canonical_json(list(data.X.shape[1:])),
                "formal_fold_manifest_identical": True,
                "forward_shape": canonical_json(list(logits.shape)),
                "finite_loss": bool(torch.isfinite(loss).item()),
                "backward_success": True,
                "parameter_changed": parameter_changed,
                "tiny_epochs": len(result.history),
                "tiny_train_samples": len(tiny_train),
                "tiny_test_samples": len(tiny_test),
                "finite_probabilities": bool(np.isfinite(result.probabilities).all()),
                "debug_only_not_formal": True,
            }
        )
        del result, model, x, y
    sanity_dir = args.output_dir / "sanity"
    atomic_csv(sanity_dir / "sanity_results.csv", pd.DataFrame(rows))
    atomic_json(
        sanity_dir / "SANITY_COMPLETE.json",
        {
            "completed_utc": utc_now(),
            "run_fingerprint": fingerprint(payload),
            "session": "710",
            "formal_results": False,
            "models": list(SBIND_ADAPTED_METHODS),
            "checks_passed": True,
        },
    )
    print(f"SANITY PASS: {sanity_dir / 'SANITY_COMPLETE.json'}", flush=True)


def write_fold_task(
    args: argparse.Namespace,
    payload: dict[str, Any],
    data: Any,
    model_name: str,
    seed: int,
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    run_fingerprint = fingerprint(payload)
    path = task_dir(args.output_dir, data.session, model_name, seed, fold)
    train_cycles = cycle_text(data.groups[train_idx])
    test_cycles = cycle_text(data.groups[test_idx])
    training_dict = payload["training"]
    training_config = DeepTrainingConfig(**training_dict)
    result = train_sbind_adapted_fold(
        model_name,
        data.X[train_idx],
        data.y[train_idx],
        data.X[test_idx],
        np.asarray([0, 1], dtype=np.int64),
        session=data.session,
        fold=fold,
        seed=seed,
        train_cycles=train_cycles,
        test_cycles=test_cycles,
        training_config=training_config,
        architecture_config=SBINDAdaptedConfig(dropout=0.25),
        device=args.device,
        workers=args.workers,
    )
    metrics = classification_metrics(data.y[test_idx], result.predictions)
    prediction_rows = []
    for local_i, sample_i in enumerate(test_idx):
        meta = data.metadata.iloc[int(sample_i)]
        prediction_rows.append(
            {
                "session": data.session,
                "model": model_name,
                "model_display": SBIND_ADAPTED_DISPLAY_NAMES[model_name],
                "seed": seed,
                "fold": fold,
                "sample_index": int(sample_i),
                "block_id": str(meta["block_id"]),
                "cycle": int(data.groups[sample_i]),
                "block_name": str(meta["block_name"]),
                "y_true": int(data.y[sample_i]),
                "y_pred": int(result.predictions[local_i]),
                "probability_0": float(result.probabilities[local_i, 0]),
                "probability_1": float(result.probabilities[local_i, 1]),
            }
        )
    cm = confusion_matrix(data.y[test_idx], result.predictions, np.asarray([0, 1]))
    confusion_rows = [
        {
            "session": data.session, "model": model_name, "seed": seed, "fold": fold,
            "true_label": true_label, "predicted_label": pred_label,
            "count": int(cm[true_label, pred_label]), "scope": "fold",
        }
        for true_label in (0, 1) for pred_label in (0, 1)
    ]
    history = pd.DataFrame(result.history)
    history.insert(0, "fold", fold)
    history.insert(0, "seed", seed)
    history.insert(0, "model", model_name)
    history.insert(0, "session", data.session)
    result_payload = {
        "run_fingerprint": run_fingerprint,
        "session": data.session,
        "model": model_name,
        "model_display": SBIND_ADAPTED_DISPLAY_NAMES[model_name],
        "seed": seed,
        "fold": fold,
        "n_cycles": data.n_cycles,
        "n_samples": data.n_blocks,
        "n_train_samples": len(train_idx),
        "n_test_samples": len(test_idx),
        "train_cycles": train_cycles,
        "test_cycles": test_cycles,
        "balanced_accuracy": metrics["balanced_accuracy"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "model_parameters": result.model_parameters,
        "final_training_loss": result.final_training_loss,
        "trained_epochs": result.final_trained_epochs,
        "device": result.device,
    }
    atomic_json(path / "result.json", result_payload)
    atomic_csv(path / "predictions.csv", pd.DataFrame(prediction_rows))
    atomic_csv(path / "confusion_matrix.csv", pd.DataFrame(confusion_rows))
    atomic_csv(path / "training_history.csv", history)
    atomic_json(path / "normalization_audit.json", result.normalization_audit)
    atomic_json(path / "model_config.json", result.model_config)
    atomic_json(
        path / "COMPLETE.json",
        {
            "task_key": task_key(data.session, model_name, seed, fold),
            "run_fingerprint": run_fingerprint,
            "completed_utc": utc_now(),
            "validated_files": [
                "result.json", "predictions.csv", "confusion_matrix.csv",
                "training_history.csv", "normalization_audit.json", "model_config.json",
            ],
        },
    )


def read_task_artifacts(args: argparse.Namespace, plan: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    confusions: list[pd.DataFrame] = []
    for row in plan.itertuples(index=False):
        path = task_dir(args.output_dir, str(row.session), str(row.model), int(row.seed), int(row.fold))
        if not (path / "COMPLETE.json").exists():
            continue
        results.append(json.loads((path / "result.json").read_text()))
        predictions.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        confusions.append(pd.read_csv(path / "confusion_matrix.csv", dtype={"session": str}))
    return (
        pd.DataFrame(results),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.concat(confusions, ignore_index=True) if confusions else pd.DataFrame(),
    )


def locate_existing_summary(args: argparse.Namespace) -> Path:
    candidates = [
        args.project_root / "outputs" / "block_clean4_binary_all_models_9sessions_v1" / "aggregate" / "multiframe_all_models_master_long.csv",
        args.project_root / "results" / "runs" / "multiframe" / "block_clean4_binary_all_models_v1" / "aggregate" / "multiframe_all_models_master_long.csv",
        args.benchmark_root / "aggregate" / "multiframe_master_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no existing clean4 aggregate summary found; old baselines are required read-only")


def aggregate_outputs(args: argparse.Namespace, plan: pd.DataFrame) -> None:
    per_fold, predictions, confusions = read_task_artifacts(args, plan)
    if per_fold.empty:
        return
    atomic_csv(args.output_dir / "sbind_per_fold.csv", per_fold.sort_values(["session", "model", "seed", "fold"]))
    atomic_csv(args.output_dir / "sbind_predictions.csv", predictions.sort_values(["session", "model", "seed", "fold", "sample_index"]))
    atomic_csv(args.output_dir / "sbind_confusion_matrices.csv", confusions.sort_values(["session", "model", "seed", "fold", "true_label", "predicted_label"]))

    seed_rows: list[dict[str, Any]] = []
    for (session, model, seed), group in predictions.groupby(["session", "model", "seed"], sort=True):
        metrics = classification_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        source = per_fold[
            (per_fold["session"].astype(str) == str(session))
            & (per_fold["model"] == model)
            & (per_fold["seed"] == seed)
        ]
        seed_rows.append(
            {
                "session": str(session), "model": model, "seed": int(seed),
                "balanced_accuracy": metrics["balanced_accuracy"],
                "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                "n_cycles": int(source["n_cycles"].iloc[0]),
                "n_samples": len(group), "n_folds": group["fold"].nunique(),
            }
        )
    per_seed = pd.DataFrame(seed_rows)
    atomic_csv(args.output_dir / "sbind_per_seed.csv", per_seed)
    summary = (
        per_seed.groupby(["session", "model"], as_index=False)
        .agg(
            mean_BA=("balanced_accuracy", "mean"),
            std_BA=("balanced_accuracy", "std"),
            mean_accuracy=("accuracy", "mean"),
            n_cycles=("n_cycles", "first"),
            n_samples=("n_samples", "first"),
        )
        .sort_values(["session", "model"])
    )
    atomic_csv(args.output_dir / "sbind_summary.csv", summary)

    old_path = locate_existing_summary(args)
    old = pd.read_csv(old_path, dtype={"session": str})
    old = old[old.get("task", "binary") == "binary"].copy() if "task" in old else old.copy()
    old_rows = []
    if {"session", "method", "balanced_accuracy", "accuracy"}.issubset(old.columns):
        for (session, method), group in old.groupby(["session", "method"], sort=True):
            old_rows.append(
                {
                    "session": str(session), "model": str(method),
                    "model_display": str(group["method_display"].iloc[0]) if "method_display" in group else str(method),
                    "source": str(old_path),
                    "mean_BA": float(group["balanced_accuracy"].mean()),
                    "std_BA": float(group["balanced_accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
                    "mean_accuracy": float(group["accuracy"].mean()),
                    "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(group),
                }
            )
    new_rows = [
        {
            "session": str(row.session), "model": str(row.model),
            "model_display": SBIND_ADAPTED_DISPLAY_NAMES[str(row.model)],
            "source": str(args.output_dir / "sbind_summary.csv"),
            "mean_BA": float(row.mean_BA), "std_BA": float(row.std_BA),
            "mean_accuracy": float(row.mean_accuracy), "n_seeds": len(SEEDS),
        }
        for row in summary.itertuples(index=False)
    ]
    atomic_csv(args.output_dir / "sbind_vs_existing_baselines.csv", pd.DataFrame(old_rows + new_rows))


def run_full(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--stage full requested CUDA, but torch.cuda.is_available() is False")
    invalid_sessions = sorted(set(map(str, args.sessions)) - set(EXPECTED_SESSIONS))
    if invalid_sessions:
        raise ValueError(f"unknown sessions: {invalid_sessions}")
    run_fingerprint = fingerprint(payload)
    plan = load_or_build_task_plan(args, payload)
    selected = plan[
        plan["session"].astype(str).isin([str(value) for value in args.sessions])
        & plan["model"].isin(args.models)
    ]
    status = update_status(args, plan, run_fingerprint)
    completed = int((status["status"] == "complete").sum())
    total = len(plan)
    for session in args.sessions:
        session = str(session)
        session_plan = selected[selected["session"].astype(str) == session]
        if session_plan.empty:
            continue
        data, splits = audit_session(args, session)
        for row in session_plan.itertuples(index=False):
            path = task_dir(args.output_dir, session, str(row.model), int(row.seed), int(row.fold))
            if valid_completed_task(path, pd.Series(row._asdict()), run_fingerprint):
                print(
                    f"SKIP [{completed}/{total}] session={session} model={row.model} "
                    f"fold={row.fold} seed={row.seed} device={args.device}", flush=True
                )
                continue
            train_idx, test_idx = splits[int(row.fold) - 1]
            print(
                f"RUN  [{completed}/{total}] session={session} model={row.model} "
                f"fold={row.fold} seed={row.seed} device={args.device}", flush=True
            )
            write_fold_task(
                args, payload, data, str(row.model), int(row.seed), int(row.fold), train_idx, test_idx
            )
            completed += 1
            print(
                f"DONE [{completed}/{total}] session={session} model={row.model} "
                f"fold={row.fold} seed={row.seed} device={args.device}", flush=True
            )
        del data
    status = update_status(args, plan, run_fingerprint)
    aggregate_outputs(args, plan)
    all_complete = bool(len(status) and (status["status"] == "complete").all())
    if all_complete:
        missing = [name for name in REQUIRED_FINAL_OUTPUTS if not (args.output_dir / name).exists()]
        if missing:
            raise AssertionError(f"completed tasks but missing aggregate outputs: {missing}")
        marker = {
            "status": "complete",
            "compute_environment": "server",
            "completed_utc": utc_now(),
            "run_fingerprint": run_fingerprint,
            "completed_tasks": len(plan),
            "total_tasks": len(plan),
            "required_outputs": list(REQUIRED_FINAL_OUTPUTS),
        }
        atomic_json(args.output_dir / "RUN_COMPLETE.json", marker)
        print(f"FULL RUN COMPLETE: {args.output_dir / 'RUN_COMPLETE.json'}", flush=True)


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    if args.batch_size < 1 or args.batch_size > 16:
        raise ValueError("--batch-size must be between 1 and the frozen maximum 16")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    payload = frozen_payload(args.batch_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_run_metadata(args, payload)
    if args.stage == "sanity":
        run_sanity(args, payload)
    elif args.stage == "full":
        run_full(args, payload)
    else:
        plan_path = args.output_dir / "task_plan.csv"
        if not plan_path.exists():
            print(f"NOT STARTED: no task plan at {plan_path}")
            return
        plan = pd.read_csv(plan_path, dtype={"session": str})
        update_status(args, plan, fingerprint(payload))


if __name__ == "__main__":
    main()
