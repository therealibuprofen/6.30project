#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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

from ultrasound_decoding.multiframe.block_ce_full_eval import (
    ALL_SEEDS,
    EXPECTED_BLOCKS_PER_SEED,
    EXPECTED_FOLDS,
    EXPECTED_NEW_TRAININGS,
    MODEL_NAME,
    MODEL_VERSION,
    NEW_SEEDS,
    REUSED_SEED,
    STABILITY_THRESHOLDS,
    build_evaluation_summaries,
    build_new_training_plan,
    evaluate_stability,
    validate_prediction_identity_alignment,
    validate_seed0_reuse,
)
from ultrasound_decoding.multiframe.crr_fcnn import (
    EXPECTED_PARAMETERS,
    FROZEN_OPTIMIZATION,
    apply_normalization,
    deterministic_cycle_orders,
    fit_train_only_normalization,
    train_cycle_model,
    validate_complete_cycle_metadata,
    validate_outer_split,
)
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    default_block_data_dir,
    load_block_sequence_session,
)


OUTPUT_VERSION = "block_ce_fcnn_full_eval_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Block-CE FCNN three-seed full evaluation v1")
    parser.add_argument("--stage", choices=("plan", "sanity", "full", "status"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs" / OUTPUT_VERSION
    )
    parser.add_argument(
        "--seed0-screening-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "crr_fcnn_screening_v1",
    )
    parser.add_argument(
        "--historical-reference-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "fcnn_canonical_single_frame_v1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sanity-epochs", type=int, default=1)
    parser.add_argument("--review-approved", action="store_true")
    return parser.parse_args()


def resolve_seed0_run_dir(path: Path) -> Path:
    candidates = [path, path / "crr_fcnn_screening_v1"]
    complete = [candidate for candidate in candidates if (candidate / "RUN_COMPLETE.json").is_file()]
    if len(complete) != 1:
        raise FileNotFoundError(
            f"expected exactly one completed seed-0 screening directory under {path}; found {complete}"
        )
    return complete[0].resolve()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.historical_reference_dir = args.historical_reference_dir.resolve()
    args.seed0_screening_dir = resolve_seed0_run_dir(args.seed0_screening_dir.resolve())
    args.data_dir = (
        args.data_dir.resolve()
        if args.data_dir is not None
        else default_block_data_dir(args.project_root).resolve()
    )
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def git_value(project_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=project_root, check=True, text=True, capture_output=True
    ).stdout.strip()


def parse_cycles(value: Any) -> list[int]:
    text = str(value).strip()
    return [] if not text else [int(item) for item in text.split(",")]


def formal_protocol() -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "scientific_question": (
            "whether block-level fused-probability CE stably improves held-out-cycle "
            "decoding over historical frame-level CE"
        ),
        "evaluation_type": "full descriptive evaluation after seed-0 screening; not confirmatory",
        "task": "visual_stimulus_presence_binary",
        "sessions": list(EXPECTED_SESSIONS),
        "all_evaluation_seeds": list(ALL_SEEDS),
        "reused_seed": REUSED_SEED,
        "new_training_seeds": list(NEW_SEEDS),
        "expected_new_folds_per_seed": EXPECTED_FOLDS,
        "expected_new_trainings": EXPECTED_NEW_TRAININGS,
        "seed0_training_in_this_run": False,
        "crr_training_in_this_run": False,
        "architecture": {
            "layers": "MaxPool2d(2)->Flatten(16000)->Linear(16000,3)->ReLU->Linear(3,2)",
            "parameters": EXPECTED_PARAMETERS,
            "modified": False,
        },
        "input": "frozen clean4 four middle frames",
        "outer_cv": "exact historical cycle-grouped folds; no repartition",
        "normalization": {
            "transform": "arcsinh_then_train_pixel_zscore",
            "statistics": "outer-training frames only",
            "outer_test_used": False,
        },
        "training_unit": "one complete four-block cycle per optimizer step",
        "block_fusion_train_and_test": (
            "softmax independently per frame then equal arithmetic mean of four probabilities"
        ),
        "loss": "mean -log(clamp(P_fused(true_class), min=1e-8)) over four cycle blocks",
        "optimization": dict(FROZEN_OPTIMIZATION),
        "evaluation": "concatenate all held-out blocks within session and seed, then BA",
        "session_aggregation": "mean three seed BAs within session",
        "overall_aggregation": "unweighted mean of nine session three-seed mean BAs",
        "stability_thresholds": dict(STABILITY_THRESHOLDS),
        "p_value_controls_decision": False,
        "hyperparameter_search": False,
        "next_decoder_design": False,
    }


def runtime_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    sources = [
        Path(__file__).resolve(),
        args.project_root / "src/ultrasound_decoding/multiframe/block_ce_full_eval.py",
        args.project_root / "src/ultrasound_decoding/multiframe/crr_fcnn.py",
        args.project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        args.project_root / "src/ultrasound_decoding/multiframe/training.py",
        args.project_root / "src/ultrasound_decoding/deep.py",
    ]
    return {
        "created_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "git_head": git_value(args.project_root, "rev-parse", "HEAD"),
        "git_branch": git_value(args.project_root, "branch", "--show-current"),
        "source_hashes": {str(path): file_sha256(path) for path in sources},
    }


def load_reference_assets(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    historical_plan_path = args.historical_reference_dir / "task_plan.csv"
    historical_predictions_path = (
        args.historical_reference_dir / "late_fusion_reconstructed_predictions.csv"
    )
    seed0_predictions_path = args.seed0_screening_dir / "per_fold_predictions.csv"
    seed0_summary_path = args.seed0_screening_dir / "per_session_summary.csv"
    run_complete_path = args.seed0_screening_dir / "RUN_COMPLETE.json"
    required = [
        historical_plan_path,
        historical_predictions_path,
        seed0_predictions_path,
        seed0_summary_path,
        run_complete_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required reference artifacts are missing: {missing}")
    run_complete = json.loads(run_complete_path.read_text(encoding="utf-8"))
    if run_complete.get("status") != "complete" or int(run_complete.get("seed", -1)) != 0:
        raise AssertionError("seed-0 screening source is not a completed seed-0 run")
    historical_plan = pd.read_csv(historical_plan_path, dtype={"session": str})
    historical_predictions = pd.read_csv(
        historical_predictions_path, dtype={"session": str}
    )
    seed0_all = pd.read_csv(seed0_predictions_path, dtype={"session": str})
    seed0_predictions = seed0_all[seed0_all["model"].eq(MODEL_NAME)].copy()
    seed0_summary = pd.read_csv(seed0_summary_path, dtype={"session": str})
    audit = validate_seed0_reuse(
        seed0_predictions,
        seed0_summary,
        historical_predictions,
        source_path=str(args.seed0_screening_dir),
    )
    audit["source_RUN_COMPLETE_sha256"] = file_sha256(run_complete_path)
    audit["source_predictions_sha256"] = file_sha256(seed0_predictions_path)
    audit["source_session_summary_sha256"] = file_sha256(seed0_summary_path)
    return historical_plan, historical_predictions, seed0_predictions, seed0_summary, audit


def build_plan_assets(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    historical_plan, historical_predictions, _seed0, _summary, seed0_audit = (
        load_reference_assets(args)
    )
    plan = build_new_training_plan(historical_plan)
    base_plan = plan.drop_duplicates(["session", "seed", "fold"])
    metadata_hashes: dict[str, str] = {}
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(
            args.project_root, session, "binary", data_dir=args.data_dir
        )
        authoritative = pd.read_csv(data.source_metadata_path, dtype={"session": str})
        validate_complete_cycle_metadata(authoritative)
        metadata_hashes[session] = file_sha256(data.source_metadata_path)
        for row in base_plan[base_plan["session"].eq(session)].itertuples(index=False):
            train_cycles = parse_cycles(row.train_cycles)
            test_cycles = parse_cycles(row.test_cycles)
            validate_outer_split(train_cycles, test_cycles)
            if set(train_cycles) | set(test_cycles) != set(np.unique(data.groups).astype(int)):
                raise AssertionError("planned outer split does not cover all authoritative cycles")
            if int(row.n_train_blocks) != int(np.isin(data.groups, train_cycles).sum()):
                raise AssertionError("planned training block count differs from data")
            if int(row.n_test_blocks) != int(np.isin(data.groups, test_cycles).sum()):
                raise AssertionError("planned test block count differs from data")
    historical_tasks = historical_predictions[["session", "seed", "fold"]].drop_duplicates()
    if len(historical_tasks) != EXPECTED_FOLDS * len(ALL_SEEDS):
        raise AssertionError("historical comparator lacks complete 246-fold-task coverage")
    provenance = {
        "seed0_reuse_validation": "PASS",
        "seed0_retrained": False,
        "new_training_seeds": list(NEW_SEEDS),
        "crr_in_scope": False,
        "historical_comparator_retrained": False,
        "historical_prediction_rows": int(len(historical_predictions)),
        "historical_fold_tasks": int(len(historical_tasks)),
        "historical_predictions_sha256": file_sha256(
            args.historical_reference_dir / "late_fusion_reconstructed_predictions.csv"
        ),
        "historical_plan_sha256": file_sha256(
            args.historical_reference_dir / "task_plan.csv"
        ),
        "authoritative_cycle_metadata_validation": "PASS",
        "outer_split_validation": "PASS",
        "metadata_hashes": metadata_hashes,
        "formal_new_training_started": False,
    }
    return plan, seed0_audit, provenance


def write_plan(args: argparse.Namespace) -> pd.DataFrame:
    plan, seed0_audit, provenance = build_plan_assets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "config.json", formal_protocol())
    atomic_json(args.output_dir / "runtime_fingerprint.json", runtime_fingerprint(args))
    atomic_csv(args.output_dir / "task_plan.csv", plan)
    atomic_json(args.output_dir / "seed0_reuse_audit.json", seed0_audit)
    atomic_json(args.output_dir / "provenance_audit.json", provenance)
    atomic_json(
        args.output_dir / "PLAN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "planned_new_trainings": len(plan),
            "seed1_trainings": int((plan["seed"] == 1).sum()),
            "seed2_trainings": int((plan["seed"] == 2).sum()),
            "seed0_trainings": 0,
            "crr_trainings": 0,
            "formal_new_training_started": False,
            "task_plan_sha256": file_sha256(args.output_dir / "task_plan.csv"),
        },
    )
    print(
        "PLAN PASS: seed1=82 + seed2=82 = 164 new Block-CE trainings; "
        "seed0 reused, CRR excluded, full not run",
        flush=True,
    )
    return plan


def load_strict_plan(args: argparse.Namespace) -> pd.DataFrame:
    required = [
        args.output_dir / "config.json",
        args.output_dir / "runtime_fingerprint.json",
        args.output_dir / "task_plan.csv",
        args.output_dir / "seed0_reuse_audit.json",
        args.output_dir / "provenance_audit.json",
        args.output_dir / "PLAN_COMPLETE.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"plan outputs missing; run --stage plan: {missing}")
    current, current_seed0, _provenance = build_plan_assets(args)
    saved = pd.read_csv(args.output_dir / "task_plan.csv", dtype={"session": str})
    pd.testing.assert_frame_equal(saved, current, check_dtype=False)
    saved_seed0 = json.loads(
        (args.output_dir / "seed0_reuse_audit.json").read_text(encoding="utf-8")
    )
    if saved_seed0 != current_seed0:
        raise AssertionError("saved seed-0 reuse audit differs from current source validation")
    config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    if config != formal_protocol():
        raise AssertionError("saved config differs from frozen full-evaluation protocol")
    return saved


def train_one_task(
    args: argparse.Namespace, row: dict[str, Any], *, epochs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(row["seed"])
    if seed not in NEW_SEEDS:
        raise AssertionError("this run may train only seeds 1 and 2")
    session = str(row["session"])
    fold = int(row["fold"])
    train_cycles = parse_cycles(row["train_cycles"])
    test_cycles = parse_cycles(row["test_cycles"])
    validate_outer_split(train_cycles, test_cycles)
    data = load_block_sequence_session(
        args.project_root, session, "binary", data_dir=args.data_dir
    )
    train_indices = np.flatnonzero(np.isin(data.groups, train_cycles))
    test_indices = np.flatnonzero(np.isin(data.groups, test_cycles))
    mean, std = fit_train_only_normalization(data.X[train_indices])
    X_train = apply_normalization(data.X[train_indices], mean, std)
    X_test = apply_normalization(data.X[test_indices], mean, std)
    orders = deterministic_cycle_orders(train_cycles, seed=seed, epochs=int(epochs))
    result = train_cycle_model(
        X_train,
        data.y[train_indices],
        data.groups[train_indices],
        data.metadata.iloc[train_indices].reset_index(drop=True),
        X_test,
        data.y[test_indices],
        model_name=MODEL_NAME,
        cycle_orders=orders,
        seed=seed,
        device=args.device,
        lr=float(FROZEN_OPTIMIZATION["lr"]),
        weight_decay=float(FROZEN_OPTIMIZATION["weight_decay"]),
    )
    predictions = []
    for local_index, source_index in enumerate(test_indices):
        metadata = data.metadata.iloc[int(source_index)]
        predictions.append(
            {
                "session": session,
                "seed": seed,
                "fold": fold,
                "model": MODEL_NAME,
                "source_index": int(source_index),
                "block_id": str(metadata["block_id"]),
                "cycle": int(data.groups[source_index]),
                "block_name": str(metadata["block_name"]),
                "truth": int(data.y[source_index]),
                "pred": int(result.predictions[local_index]),
                "prob_no_stimulus": float(result.probabilities[local_index, 0]),
                "prob_stimulus": float(result.probabilities[local_index, 1]),
                "prediction_origin": "new_training",
            }
        )
    final = result.history[-1]
    training = pd.DataFrame(
        [
            {
                "session": session,
                "seed": seed,
                "fold": fold,
                "model": MODEL_NAME,
                "trained_in_this_run": True,
                "epochs": int(epochs),
                "optimizer_steps": int(
                    sum(item["optimizer_steps"] for item in result.history)
                ),
                "cycles_per_epoch": len(train_cycles),
                "initial_state_sha256": result.initial_state_sha256,
                "cycle_order_sha256": result.cycle_order_sha256,
                "final_mean_total_loss": float(final["mean_total_loss"]),
                "final_mean_classification_loss": float(
                    final["mean_classification_loss"]
                ),
                "final_train_BA": result.final_train_balanced_accuracy,
                "fold_test_BA_diagnostic": result.final_test_balanced_accuracy,
                "train_test_gap_diagnostic": (
                    result.final_train_balanced_accuracy
                    - result.final_test_balanced_accuracy
                ),
            }
        ]
    )
    return pd.DataFrame(predictions), training


def run_sanity(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("minimal sanity is deliberately CPU-only")
    if int(args.sanity_epochs) < 1:
        raise ValueError("sanity epochs must be positive")
    plan = load_strict_plan(args)
    row = plan[plan["seed"].eq(1)].iloc[0].to_dict()
    predictions, training = train_one_task(args, row, epochs=int(args.sanity_epochs))
    sanity_dir = args.output_dir / "sanity"
    atomic_csv(sanity_dir / "per_fold_predictions.csv", predictions)
    atomic_csv(sanity_dir / "training_summary.csv", training)
    atomic_json(
        args.output_dir / "SANITY_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "device": "cpu",
            "task": f"{row['session']}:1:{int(row['fold'])}",
            "model": MODEL_NAME,
            "seed": 1,
            "trainings": 1,
            "epochs": int(args.sanity_epochs),
            "finite_probabilities": bool(
                np.isfinite(predictions[["prob_no_stimulus", "prob_stimulus"]]).all().all()
            ),
            "probabilities_sum_to_one": bool(
                np.allclose(
                    predictions[["prob_no_stimulus", "prob_stimulus"]].sum(axis=1),
                    1.0,
                )
            ),
            "seed0_retrained": False,
            "crr_trained": False,
            "formal_new_training_started": False,
        },
    )
    print(
        f"SANITY PASS cpu task={row['session']}:1:{int(row['fold'])} "
        f"model={MODEL_NAME} epochs={args.sanity_epochs} seed0_retrained=False "
        "CRR=False full=False",
        flush=True,
    )


def task_directory(output_dir: Path, session: str, seed: int, fold: int) -> Path:
    return (
        output_dir
        / "tasks"
        / f"session_{session}"
        / f"seed_{seed}"
        / f"fold_{fold:02d}"
    )


def run_full(args: argparse.Namespace) -> None:
    if not args.review_approved:
        raise RuntimeError("formal 164-training evaluation requires --review-approved")
    plan = load_strict_plan(args).sort_values(["seed", "session", "fold"])
    for row in plan.to_dict("records"):
        directory = task_directory(
            args.output_dir, str(row["session"]), int(row["seed"]), int(row["fold"])
        )
        complete_path = directory / "COMPLETE.json"
        if complete_path.is_file():
            continue
        predictions, training = train_one_task(
            args, row, epochs=int(FROZEN_OPTIMIZATION["epochs"])
        )
        atomic_csv(directory / "per_fold_predictions.csv", predictions)
        atomic_csv(directory / "training_summary.csv", training)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "session": str(row["session"]),
                "seed": int(row["seed"]),
                "fold": int(row["fold"]),
                "model": MODEL_NAME,
                "epochs": int(FROZEN_OPTIMIZATION["epochs"]),
            },
        )
        print(
            f"COMPLETE session={row['session']} seed={int(row['seed'])} "
            f"fold={int(row['fold'])}",
            flush=True,
        )
    prediction_tables = []
    training_tables = []
    for row in plan.to_dict("records"):
        directory = task_directory(
            args.output_dir, str(row["session"]), int(row["seed"]), int(row["fold"])
        )
        prediction_tables.append(
            pd.read_csv(directory / "per_fold_predictions.csv", dtype={"session": str})
        )
        training_tables.append(
            pd.read_csv(directory / "training_summary.csv", dtype={"session": str})
        )
    new_predictions = pd.concat(prediction_tables, ignore_index=True)
    training = pd.concat(training_tables, ignore_index=True)
    if len(training) != EXPECTED_NEW_TRAININGS or set(training["seed"].astype(int)) != set(
        NEW_SEEDS
    ):
        raise AssertionError("formal training summary must contain only 164 seed1/seed2 tasks")
    (
        _historical_plan,
        historical_predictions,
        seed0_predictions,
        _seed0_summary,
        seed0_audit,
    ) = load_reference_assets(args)
    seed0_predictions = seed0_predictions.copy()
    seed0_predictions["prediction_origin"] = "reused_crr_fcnn_screening_v1"
    blockce_predictions = pd.concat(
        [seed0_predictions, new_predictions], ignore_index=True, sort=False
    )
    identity_audit = validate_prediction_identity_alignment(
        historical_predictions, blockce_predictions, seeds=ALL_SEEDS
    )
    per_seed_session, per_session, seed_level = build_evaluation_summaries(
        historical_predictions, blockce_predictions
    )
    stability = evaluate_stability(per_session, seed_level)
    atomic_csv(args.output_dir / "per_fold_predictions.csv", blockce_predictions)
    atomic_csv(args.output_dir / "training_summary.csv", training)
    atomic_csv(args.output_dir / "per_seed_session_summary.csv", per_seed_session)
    atomic_csv(args.output_dir / "per_session_summary.csv", per_session)
    atomic_csv(args.output_dir / "seed_level_summary.csv", seed_level)
    atomic_json(args.output_dir / "stability_assessment.json", stability)
    provenance = json.loads(
        (args.output_dir / "provenance_audit.json").read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "formal_new_training_started": True,
            "formal_new_training_complete": True,
            "new_training_count": EXPECTED_NEW_TRAININGS,
            "seed0_reuse_audit": seed0_audit,
            "three_seed_identity_alignment": identity_audit,
            "crr_trained": False,
        }
    )
    atomic_json(args.output_dir / "provenance_audit.json", provenance)
    atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "new_trainings": EXPECTED_NEW_TRAININGS,
            "seed0_trainings": 0,
            "crr_trainings": 0,
            "blockce_prediction_rows": int(len(blockce_predictions)),
            "decision": stability["decision"],
        },
    )
    print(
        f"RUN COMPLETE new_trainings=164 seed0_retrained=False CRR=False "
        f"decision={stability['decision']}",
        flush=True,
    )


def run_status(args: argparse.Namespace) -> None:
    for name in (
        "PLAN_COMPLETE.json",
        "SANITY_COMPLETE.json",
        "RUN_COMPLETE.json",
        "stability_assessment.json",
    ):
        path = args.output_dir / name
        print(f"{name}: {'present' if path.is_file() else 'absent'}")
    complete = list(
        (args.output_dir / "tasks").glob("session_*/seed_*/fold_*/COMPLETE.json")
    )
    print(f"formal_new_trainings_complete: {len(complete)}/{EXPECTED_NEW_TRAININGS}")


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
