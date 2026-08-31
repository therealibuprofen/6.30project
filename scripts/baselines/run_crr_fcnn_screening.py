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

from ultrasound_decoding.multiframe.crr_fcnn import (
    EXPECTED_PARAMETERS,
    FROZEN_GATE,
    FROZEN_OPTIMIZATION,
    MODEL_VERSION,
    MODELS,
    SEED,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    apply_normalization,
    build_screening_plan,
    deterministic_cycle_orders,
    evaluate_screening_gate,
    fit_train_only_normalization,
    historical_seed0_session_ba,
    session_oof_balanced_accuracy,
    train_cycle_model,
    validate_complete_cycle_metadata,
    validate_outer_split,
)
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    default_block_data_dir,
    load_block_sequence_session,
)


OUTPUT_VERSION = "crr_fcnn_screening_v1"
EXPECTED_FOLDS = 82
EXPECTED_TRAININGS = 164


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CRR-FCNN v1 seed-0 feasibility screening")
    parser.add_argument("--stage", choices=("plan", "sanity", "full", "status"), required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_DIR / "outputs" / OUTPUT_VERSION
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


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.project_root = args.project_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.historical_reference_dir = args.historical_reference_dir.resolve()
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


def formal_protocol() -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "model_version": MODEL_VERSION,
        "scientific_goal": "cycle-relative ranking supervision feasibility",
        "task": "visual_stimulus_presence_binary",
        "sessions": list(EXPECTED_SESSIONS),
        "seed": SEED,
        "models": list(MODELS),
        "expected_outer_folds_per_model": EXPECTED_FOLDS,
        "expected_trainings": EXPECTED_TRAININGS,
        "architecture": {
            "layers": "MaxPool2d(2)->Flatten(16000)->Linear(16000,3)->ReLU->Linear(3,2)",
            "parameters": EXPECTED_PARAMETERS,
            "modified": False,
        },
        "input": "frozen clean4 four middle frames",
        "normalization": {
            "transform": "arcsinh_then_train_pixel_zscore",
            "statistics": "outer-training frames only",
            "epsilon": 1e-6,
            "outer_test_used": False,
        },
        "training_unit": "one complete four-block cycle per optimizer step",
        "block_fusion": "softmax independently per frame then equal arithmetic mean of four probabilities",
        "classification_loss": "mean negative log fused true-class probability over four blocks",
        "ranking_pairs": [
            "grating>stop_after_grating",
            "grating>static",
            "dot>stop_after_grating",
            "dot>static",
        ],
        "ranking_loss": "mean softplus(-(r_stimulus-r_nonstimulus)) over exactly four pairs",
        "lambda_rank": 1.0,
        "margin": 0.0,
        "optimization": dict(FROZEN_OPTIMIZATION),
        "matched_controls": {
            "identical_initial_weights": True,
            "identical_cycle_order": True,
            "only_difference": "presence or absence of L_rank",
        },
        "evaluation": "concatenate all outer-held-out blocks within session then balanced accuracy",
        "historical_comparator": "reuse validated seed-0 FCNN late-fusion OOF predictions; no retraining",
        "strong_sessions_descriptive_only": list(STRONG_SESSIONS),
        "weak_sessions_descriptive_only": list(WEAK_SESSIONS),
        "screening_gate": dict(FROZEN_GATE),
        "automatic_three_seed_run": False,
    }


def runtime_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    source_paths = [
        Path(__file__).resolve(),
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
        "device_requested": str(args.device),
        "git_head": git_value(args.project_root, "rev-parse", "HEAD"),
        "git_branch": git_value(args.project_root, "branch", "--show-current"),
        "source_hashes": {str(path): file_sha256(path) for path in source_paths},
    }


def parse_cycles(value: Any) -> list[int]:
    text = str(value).strip()
    return [] if not text else [int(item) for item in text.split(",")]


def load_reference_assets(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    plan_path = args.historical_reference_dir / "task_plan.csv"
    prediction_path = args.historical_reference_dir / "late_fusion_reconstructed_predictions.csv"
    summary_path = args.historical_reference_dir / "session_seed_summary.csv"
    required = [plan_path, prediction_path, summary_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"validated historical reference assets missing: {missing}")
    reference_plan = pd.read_csv(plan_path, dtype={"session": str})
    historical_predictions = pd.read_csv(prediction_path, dtype={"session": str})
    reconstructed = historical_seed0_session_ba(historical_predictions)
    saved = pd.read_csv(summary_path, dtype={"session": str})
    saved = saved[saved["seed"].astype(int).eq(SEED)][["session", "late_fusion_BA"]]
    checked = reconstructed.merge(saved, on="session", validate="one_to_one")
    maximum_difference = float(
        np.max(
            np.abs(
                checked["historical_fcnn_latefusion_seed0_BA"].to_numpy(float)
                - checked["late_fusion_BA"].to_numpy(float)
            )
        )
    )
    if maximum_difference > 1e-12:
        raise AssertionError("historical seed-0 session OOF BA differs from saved summary")
    audit = {
        "historical_predictions_path": str(prediction_path),
        "historical_predictions_sha256": file_sha256(prediction_path),
        "historical_plan_path": str(plan_path),
        "historical_plan_sha256": file_sha256(plan_path),
        "seed0_recomputed_against_saved_summary": "PASS",
        "maximum_absolute_BA_difference": maximum_difference,
        "historical_retrained": False,
    }
    return reference_plan, historical_predictions, audit


def build_plan_assets(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reference_plan, historical_predictions, reference_audit = load_reference_assets(args)
    plan = build_screening_plan(reference_plan)
    audits: list[pd.DataFrame] = []
    dataset_hashes: dict[str, dict[str, str]] = {}
    base_plan = plan.drop_duplicates(["session", "fold"])
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(
            args.project_root, session, "binary", data_dir=args.data_dir
        )
        authoritative = pd.read_csv(data.source_metadata_path, dtype={"session": str})
        audits.append(validate_complete_cycle_metadata(authoritative))
        dataset_hashes[session] = {
            "h5_sha256": file_sha256(data.source_h5_path),
            "metadata_sha256": file_sha256(data.source_metadata_path),
        }
        session_plan = base_plan[base_plan["session"].eq(session)]
        for row in session_plan.itertuples(index=False):
            train_cycles = parse_cycles(row.train_cycles)
            test_cycles = parse_cycles(row.test_cycles)
            validate_outer_split(train_cycles, test_cycles)
            if set(train_cycles) | set(test_cycles) != set(np.unique(data.groups).astype(int)):
                raise AssertionError(f"session {session} fold {row.fold} does not cover all cycles")
            if int(row.n_train_blocks) != int(np.isin(data.groups, train_cycles).sum()):
                raise AssertionError("formal plan training block count differs from clean4 data")
            if int(row.n_test_blocks) != int(np.isin(data.groups, test_cycles).sum()):
                raise AssertionError("formal plan test block count differs from clean4 data")
    cycle_audit = pd.concat(audits, ignore_index=True)
    historical_summary = historical_seed0_session_ba(historical_predictions)
    provenance = {
        **reference_audit,
        "authoritative_cycle_metadata_validation": "PASS",
        "outer_fold_identity_validation": "PASS",
        "outer_cycle_disjointness_validation": "PASS",
        "dataset_hashes": dataset_hashes,
        "formal_screening_started": False,
        "three_seed_experiment_started": False,
    }
    return plan, cycle_audit, historical_summary, provenance


def write_plan(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plan, cycle_audit, historical_summary, provenance = build_plan_assets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "config.json", formal_protocol())
    atomic_json(args.output_dir / "runtime_fingerprint.json", runtime_fingerprint(args))
    atomic_csv(args.output_dir / "task_plan.csv", plan)
    atomic_csv(args.output_dir / "cycle_structure_audit.csv", cycle_audit)
    atomic_json(args.output_dir / "provenance_audit.json", provenance)
    atomic_json(
        args.output_dir / "PLAN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "planned_trainings": len(plan),
            "block_ce_fcnn_trainings": int((plan["model"] == "block_ce_fcnn").sum()),
            "crr_fcnn_trainings": int((plan["model"] == "crr_fcnn").sum()),
            "outer_folds_per_model": EXPECTED_FOLDS,
            "sessions": len(EXPECTED_SESSIONS),
            "seed": SEED,
            "formal_screening_started": False,
            "three_seed_experiment_started": False,
            "task_plan_sha256": file_sha256(args.output_dir / "task_plan.csv"),
        },
    )
    print(
        "PLAN PASS: 82 block_ce_fcnn + 82 crr_fcnn = 164 trainings; formal screening not run",
        flush=True,
    )
    return plan, cycle_audit, historical_summary


def load_strict_plan(args: argparse.Namespace) -> pd.DataFrame:
    required = [
        args.output_dir / "config.json",
        args.output_dir / "runtime_fingerprint.json",
        args.output_dir / "task_plan.csv",
        args.output_dir / "cycle_structure_audit.csv",
        args.output_dir / "provenance_audit.json",
        args.output_dir / "PLAN_COMPLETE.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"plan outputs missing; run --stage plan: {missing}")
    current, cycle_audit, _historical, _provenance = build_plan_assets(args)
    saved = pd.read_csv(args.output_dir / "task_plan.csv", dtype={"session": str})
    pd.testing.assert_frame_equal(saved, current, check_dtype=False)
    saved_cycle = pd.read_csv(
        args.output_dir / "cycle_structure_audit.csv", dtype={"session": str}
    )
    pd.testing.assert_frame_equal(saved_cycle, cycle_audit, check_dtype=False)
    config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    if config != formal_protocol():
        raise AssertionError("saved config differs from frozen CRR protocol")
    return saved


def fold_pair(
    args: argparse.Namespace, row: dict[str, Any], *, epochs: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    orders = deterministic_cycle_orders(train_cycles, seed=SEED, epochs=int(epochs))
    prediction_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    paired_results = []
    for model_name in MODELS:
        result = train_cycle_model(
            X_train,
            data.y[train_indices],
            data.groups[train_indices],
            data.metadata.iloc[train_indices].reset_index(drop=True),
            X_test,
            data.y[test_indices],
            model_name=model_name,
            cycle_orders=orders,
            seed=SEED,
            device=args.device,
            lr=float(FROZEN_OPTIMIZATION["lr"]),
            weight_decay=float(FROZEN_OPTIMIZATION["weight_decay"]),
        )
        paired_results.append(result)
        for local_index, source_index in enumerate(test_indices):
            metadata = data.metadata.iloc[int(source_index)]
            prediction_rows.append(
                {
                    "session": session,
                    "seed": SEED,
                    "fold": fold,
                    "model": model_name,
                    "source_index": int(source_index),
                    "block_id": str(metadata["block_id"]),
                    "cycle": int(data.groups[source_index]),
                    "block_name": str(metadata["block_name"]),
                    "truth": int(data.y[source_index]),
                    "pred": int(result.predictions[local_index]),
                    "prob_no_stimulus": float(result.probabilities[local_index, 0]),
                    "prob_stimulus": float(result.probabilities[local_index, 1]),
                }
            )
        final = result.history[-1]
        training_rows.append(
            {
                "session": session,
                "seed": SEED,
                "fold": fold,
                "model": model_name,
                "epochs": int(epochs),
                "optimizer_steps": int(sum(item["optimizer_steps"] for item in result.history)),
                "cycles_per_epoch": len(train_cycles),
                "initial_state_sha256": result.initial_state_sha256,
                "cycle_order_sha256": result.cycle_order_sha256,
                "final_mean_total_loss": float(final["mean_total_loss"]),
                "final_mean_classification_loss": float(final["mean_classification_loss"]),
                "final_mean_ranking_loss_diagnostic": float(final["mean_ranking_loss_diagnostic"]),
                "final_train_BA": result.final_train_balanced_accuracy,
                "fold_test_BA_diagnostic": result.final_test_balanced_accuracy,
                "train_test_gap_diagnostic": (
                    result.final_train_balanced_accuracy - result.final_test_balanced_accuracy
                ),
            }
        )
    if paired_results[0].initial_state_sha256 != paired_results[1].initial_state_sha256:
        raise AssertionError("matched Block-CE and CRR initial weights differ")
    if paired_results[0].cycle_order_sha256 != paired_results[1].cycle_order_sha256:
        raise AssertionError("matched Block-CE and CRR cycle orders differ")
    return pd.DataFrame(prediction_rows), pd.DataFrame(training_rows)


def run_sanity(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise ValueError("the requested minimal sanity check is CPU-only")
    if int(args.sanity_epochs) < 1:
        raise ValueError("sanity epochs must be positive")
    plan = load_strict_plan(args)
    row = plan.drop_duplicates(["session", "fold"]).iloc[0].to_dict()
    predictions, training = fold_pair(args, row, epochs=int(args.sanity_epochs))
    sanity_dir = args.output_dir / "sanity"
    atomic_csv(sanity_dir / "per_fold_predictions.csv", predictions)
    atomic_csv(sanity_dir / "training_summary.csv", training)
    identical_weights = training["initial_state_sha256"].nunique() == 1
    identical_orders = training["cycle_order_sha256"].nunique() == 1
    atomic_json(
        args.output_dir / "SANITY_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "device": "cpu",
            "task": f"{row['session']}:0:{int(row['fold'])}",
            "models": list(MODELS),
            "trainings": 2,
            "epochs_per_model": int(args.sanity_epochs),
            "identical_initial_weights": bool(identical_weights),
            "identical_cycle_order": bool(identical_orders),
            "finite_probabilities": bool(
                np.isfinite(predictions[["prob_no_stimulus", "prob_stimulus"]]).all().all()
            ),
            "probabilities_sum_to_one": bool(
                np.allclose(
                    predictions[["prob_no_stimulus", "prob_stimulus"]].sum(axis=1), 1.0
                )
            ),
            "formal_screening_started": False,
            "three_seed_experiment_started": False,
        },
    )
    print(
        f"SANITY PASS cpu task={row['session']}:0:{int(row['fold'])} "
        f"models=2 epochs={args.sanity_epochs} matched_weights={identical_weights} "
        f"matched_order={identical_orders} formal_screening=False",
        flush=True,
    )


def task_directory(output_dir: Path, session: str, fold: int) -> Path:
    return output_dir / "tasks" / f"session_{session}" / f"fold_{fold:02d}"


def run_full(args: argparse.Namespace) -> None:
    if not args.review_approved:
        raise RuntimeError("formal 164-training screening requires --review-approved")
    plan = load_strict_plan(args)
    base_plan = plan.drop_duplicates(["session", "fold"]).sort_values(["session", "fold"])
    for row in base_plan.to_dict("records"):
        directory = task_directory(args.output_dir, str(row["session"]), int(row["fold"]))
        complete_path = directory / "COMPLETE.json"
        if complete_path.is_file():
            continue
        predictions, training = fold_pair(
            args, row, epochs=int(FROZEN_OPTIMIZATION["epochs"])
        )
        atomic_csv(directory / "per_fold_predictions.csv", predictions)
        atomic_csv(directory / "training_summary.csv", training)
        atomic_json(
            complete_path,
            {
                "status": "complete",
                "session": str(row["session"]),
                "seed": SEED,
                "fold": int(row["fold"]),
                "models": list(MODELS),
                "trainings": 2,
                "epochs": int(FROZEN_OPTIMIZATION["epochs"]),
            },
        )
        print(f"COMPLETE session={row['session']} fold={int(row['fold'])}", flush=True)
    prediction_tables = []
    training_tables = []
    for row in base_plan.to_dict("records"):
        directory = task_directory(args.output_dir, str(row["session"]), int(row["fold"]))
        prediction_tables.append(
            pd.read_csv(directory / "per_fold_predictions.csv", dtype={"session": str})
        )
        training_tables.append(
            pd.read_csv(directory / "training_summary.csv", dtype={"session": str})
        )
    predictions = pd.concat(prediction_tables, ignore_index=True)
    training = pd.concat(training_tables, ignore_index=True)
    if len(training) != EXPECTED_TRAININGS:
        raise AssertionError("formal training summary does not contain 164 tasks")
    oof = session_oof_balanced_accuracy(predictions)
    wide = oof.pivot(index="session", columns="model", values="oof_balanced_accuracy").reset_index()
    wide = wide.rename(
        columns={
            "block_ce_fcnn": "block_ce_fcnn_seed0_BA",
            "crr_fcnn": "crr_fcnn_seed0_BA",
        }
    )
    _reference_plan, historical_predictions, _audit = load_reference_assets(args)
    historical = historical_seed0_session_ba(historical_predictions)
    summary = historical.merge(wide, on="session", validate="one_to_one")
    summary["delta_crr_vs_blockce"] = (
        summary["crr_fcnn_seed0_BA"] - summary["block_ce_fcnn_seed0_BA"]
    )
    summary["delta_crr_vs_historical"] = (
        summary["crr_fcnn_seed0_BA"]
        - summary["historical_fcnn_latefusion_seed0_BA"]
    )
    gate = evaluate_screening_gate(summary)
    gate["descriptive"] = {
        "strong3_crr_mean_BA": float(
            summary[summary["session"].isin(STRONG_SESSIONS)]["crr_fcnn_seed0_BA"].mean()
        ),
        "weak6_crr_mean_BA": float(
            summary[summary["session"].isin(WEAK_SESSIONS)]["crr_fcnn_seed0_BA"].mean()
        ),
    }
    atomic_csv(args.output_dir / "per_fold_predictions.csv", predictions)
    atomic_csv(args.output_dir / "training_summary.csv", training)
    atomic_csv(args.output_dir / "per_session_summary.csv", summary)
    atomic_json(args.output_dir / "screening_gate.json", gate)
    provenance = json.loads(
        (args.output_dir / "provenance_audit.json").read_text(encoding="utf-8")
    )
    provenance["formal_screening_started"] = True
    provenance["formal_screening_complete"] = True
    provenance["three_seed_experiment_started"] = False
    atomic_json(args.output_dir / "provenance_audit.json", provenance)
    atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "trainings": len(training),
            "outer_folds_per_model": EXPECTED_FOLDS,
            "seed": SEED,
            "decision": gate["decision"],
            "three_seed_experiment_started": False,
        },
    )
    print(f"RUN COMPLETE trainings=164 decision={gate['decision']}", flush=True)


def run_status(args: argparse.Namespace) -> None:
    for name in (
        "PLAN_COMPLETE.json", "SANITY_COMPLETE.json", "RUN_COMPLETE.json", "screening_gate.json"
    ):
        path = args.output_dir / name
        print(f"{name}: {'present' if path.is_file() else 'absent'}")
    complete = list((args.output_dir / "tasks").glob("session_*/fold_*/COMPLETE.json"))
    print(f"formal_fold_pairs_complete: {len(complete)}/{EXPECTED_FOLDS}")


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
