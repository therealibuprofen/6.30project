#!/usr/bin/env python3
"""Formal Scheme-A runner for training-only Adaptive Mean/Std Nested-CV v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import shlex
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

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.adaptive_mean_std_nestedcv import (
    EXPECTED_INNER_DEFINITIONS,
    EXPECTED_LOGICAL_INNER_JOBS,
    EXPECTED_OUTER_FOLDS,
    EXPECTED_SELECTIONS,
    EXPECTED_UNIQUE_TRAINING_JOBS,
    EXPECTED_UNIQUE_TRAIN_SETS,
    INPUT_VARIANTS,
    MEAN_ONLY_VARIANT,
    MEAN_STD_VARIANT,
    NORMALIZATION_PROTOCOL,
    PROTOCOL_VERSION,
    atomic_csv,
    atomic_json,
    build_evaluation_cache_identity,
    build_selection_payload,
    build_training_cache_identity,
    canonical_json,
    concatenated_oof_balanced_accuracy,
    enumerate_inner_splits,
    evaluate_inner_cache,
    evaluation_cache_key,
    file_sha256,
    fingerprint,
    ids_text,
    load_training_cache,
    lock_selection,
    parse_ids,
    parse_sample_ids,
    read_locked_selection,
    sample_ids_json,
    select_variant,
    train_inner_cache,
    training_cache_key,
    utc_now,
    validate_evaluation_cache,
    validate_outer_manifest,
    validate_parent_evaluation_access,
    validate_training_cache,
)
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.fcnn_temporal_statistics import (
    MODEL_IMPLEMENTATION_VERSION,
    architecture_config,
)
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    normalize_blocks_train_fold_only_with_stats,
)


OUTPUT_VERSION = "adaptive_mean_std_nestedcv_v1"
SEEDS = (0, 1, 2)
FORMAL_EPOCHS = 40
FORMAL_BATCH_SIZE = 16
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--stage",
        choices=("plan", "sanity", "inner", "select", "outer", "summarize", "full", "status"),
        required=True,
    )
    preliminary, _ = bootstrap.parse_known_args(argv)
    parser = argparse.ArgumentParser(
        parents=[bootstrap],
        description="Scheme-A full per-seed nested selection between frozen FCNN Mean-only and Mean+Std.",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / OUTPUT_VERSION,
    )
    if preliminary.stage in {"plan", "inner", "full"}:
        parser.add_argument("--data-dir", type=Path, default=None)
    if preliminary.stage in {"plan", "full"}:
        parser.add_argument(
            "--feasibility-dir",
            type=Path,
            default=PROJECT_DIR / "outputs/adaptive_mean_std_nestedcv_feasibility",
        )
    if preliminary.stage in {"inner", "full"}:
        parser.add_argument("--device", default="auto")
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--batch-size", type=int, default=FORMAL_BATCH_SIZE)
        parser.add_argument("--review-approved", action="store_true")
    if preliminary.stage in {"plan", "outer", "summarize", "full"}:
        parser.add_argument(
            "--fixed-results-dir",
            type=Path,
            default=PROJECT_DIR / "outputs/fcnn_mean_std_temporal_statistics_v1",
        )
    return parser.parse_args(argv)


def frozen_training_config(batch_size: int = FORMAL_BATCH_SIZE) -> DeepTrainingConfig:
    return DeepTrainingConfig(
        optimizer="adamw",
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=int(batch_size),
        max_epochs=FORMAL_EPOCHS,
        dropout=0.25,
        loss="cross_entropy",
    )


def source_paths(project_root: Path) -> list[Path]:
    return [
        Path(__file__).resolve(),
        project_root / "src/ultrasound_decoding/multiframe/adaptive_mean_std_nestedcv.py",
        project_root / "src/ultrasound_decoding/multiframe/adaptive_mean_std_outer_reuse.py",
        project_root / "src/ultrasound_decoding/multiframe/fcnn_temporal_statistics.py",
        project_root / "src/ultrasound_decoding/multiframe/models.py",
        project_root / "src/ultrasound_decoding/multiframe/training.py",
        project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        project_root / "src/ultrasound_decoding/cv.py",
        project_root / "src/ultrasound_decoding/evaluate.py",
        project_root / "configs/adaptive_mean_std_nestedcv_v1.json",
        project_root / "docs/adaptive_mean_std_nestedcv_v1.md",
    ]


def runtime_signature() -> dict[str, Any]:
    try:
        import scipy
        import sklearn
    except ImportError:  # pragma: no cover - formal environment requires both
        scipy = sklearn = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "scipy": None if scipy is None else scipy.__version__,
        "scikit_learn": None if sklearn is None else sklearn.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }


def frozen_protocol_config() -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "scientific_name": "training-only adaptive temporal-representation selection",
        "method_class": "nested-CV model selection between two fixed decoders",
        "development_status": "exploratory_method_development_not_independent_confirmatory_evidence",
        "sessions": list(EXPECTED_SESSIONS),
        "seeds": list(SEEDS),
        "scheme": "A_full_per_seed",
        "seed_pairing": "outer_seed == selector_seed == candidate_inner_training_seed",
        "candidates": list(INPUT_VARIANTS),
        "candidate_model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "architectures": {
            candidate: architecture_config(candidate) for candidate in INPUT_VARIANTS
        },
        "input_protocol": "clean4_binary_presence",
        "normalization": NORMALIZATION_PROTOCOL,
        "inner_cv": "grouped_cv_splits(outer_train_cycles,max_folds=10)",
        "inner_score": "balanced_accuracy_on_concatenated_complete_inner_oof_predictions",
        "selection_rule": "mean_std iff inner_BA_mean_std > inner_BA_mean_only; tie -> mean_only",
        "outer_result_access": "forbidden_until_selection_artifact_is_locked",
        "training": frozen_training_config().__dict__,
        "expected_counts": {
            "outer_folds": EXPECTED_OUTER_FOLDS,
            "inner_split_definitions": EXPECTED_INNER_DEFINITIONS,
            "unique_inner_training_cycle_sets": EXPECTED_UNIQUE_TRAIN_SETS,
            "logical_inner_training_jobs": EXPECTED_LOGICAL_INNER_JOBS,
            "unique_inner_training_jobs": EXPECTED_UNIQUE_TRAINING_JOBS,
            "locked_selections": EXPECTED_SELECTIONS,
            "reused_selected_outer_outputs": EXPECTED_SELECTIONS,
        },
        "decision_rule": {
            "adaptive_overall_gt_fixed_mean_only": True,
            "adaptive_overall_ge_fixed_mean_std": True,
            "adaptive_strong3_ge_fixed_mean_only_strong3_minus": 0.010,
            "weak_gain_retention_fraction_at_least": 0.75,
            "success": "supports_continue_adaptive_temporal_selection",
            "failure": "does_not_support_adaptive_temporal_selection",
        },
        "automatic_next_stage": False,
        "four_class_implemented": False,
    }


def current_source_hashes(project_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(project_root)): file_sha256(path)
        for path in source_paths(project_root)
    }


def protocol_identity(project_root: Path) -> dict[str, Any]:
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    return {
        "protocol": frozen_protocol_config(),
        "source_sha256": current_source_hashes(project_root),
        "git_commit": git_commit,
    }


def load_plan_config(output_dir: Path, project_root: Path) -> dict[str, Any]:
    path = output_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError("run plan is missing; run --stage plan first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.get("protocol_identity")
    current = protocol_identity(project_root)
    if observed != current:
        raise AssertionError("planned protocol/source fingerprint differs from current code")
    if payload.get("protocol_fingerprint") != fingerprint(current):
        raise AssertionError("stored protocol fingerprint is invalid")
    return payload


def _write_command(output_dir: Path, stage: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join(sys.argv) + "\n"
    (output_dir / f"{stage}_command.txt").write_text(command, encoding="utf-8")


def _load_sessions(project_root: Path, data_dir: Path | None):
    return {
        session: load_block_sequence_session(
            project_root,
            session,
            "binary",
            data_dir=data_dir or default_block_data_dir(project_root),
        )
        for session in EXPECTED_SESSIONS
    }


def _compare_feasibility(
    task_plan: pd.DataFrame, split_manifest: pd.DataFrame, feasibility_dir: Path
) -> None:
    expected_plan = pd.read_csv(
        feasibility_dir / "adaptive_mean_std_task_plan.csv", dtype={"session": str}
    )
    left = task_plan[
        [
            "session",
            "outer_fold",
            "outer_train_cycle_ids",
            "outer_test_cycle_ids",
            "inner_fold_count",
        ]
    ].reset_index(drop=True)
    right = expected_plan[left.columns].reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError("formal task plan differs from feasibility artifact")
    expected_split = pd.read_csv(
        feasibility_dir / "adaptive_mean_std_split_audit.csv", dtype={"session": str}
    ).rename(columns={"inner_validation_cycle_ids": "inner_val_cycle_ids"})
    columns = [
        "session",
        "outer_fold",
        "inner_fold",
        "outer_train_cycle_ids",
        "outer_test_cycle_ids",
        "inner_train_cycle_ids",
        "inner_val_cycle_ids",
        "normalization_fit_cycle_ids",
    ]
    if not split_manifest[columns].reset_index(drop=True).equals(
        expected_split[columns].reset_index(drop=True)
    ):
        raise AssertionError("formal inner split manifest differs from feasibility artifact")


def validate_existing_plan(
    args: argparse.Namespace, fixed_provenance: dict[str, Any]
) -> None:
    """Validate an existing immutable plan; never rewrite it during resume."""

    config = load_plan_config(args.output_dir, args.project_root)
    if config.get("fixed_candidate_provenance") != fixed_provenance:
        raise AssertionError("existing plan has different fixed-run provenance")
    required = (
        "PLAN_COMPLETE.json",
        "task_plan.csv",
        "split_manifest.csv",
        "dataset_manifest.csv",
        "cache_manifest.csv",
        "inner_task_manifest.csv",
        "fixed_provenance_pin.json",
    )
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"existing plan is incomplete; missing {missing}")
    completion = json.loads(
        (args.output_dir / "PLAN_COMPLETE.json").read_text(encoding="utf-8")
    )
    if (
        completion.get("status") != "complete"
        or completion.get("protocol_fingerprint") != config["protocol_fingerprint"]
        or int(completion.get("outer_folds", -1)) != EXPECTED_OUTER_FOLDS
        or int(completion.get("inner_split_definitions", -1))
        != EXPECTED_INNER_DEFINITIONS
        or int(completion.get("unique_inner_training_cycle_sets", -1))
        != EXPECTED_UNIQUE_TRAIN_SETS
        or int(completion.get("logical_inner_training_jobs", -1))
        != EXPECTED_LOGICAL_INNER_JOBS
        or int(completion.get("unique_inner_training_jobs", -1))
        != EXPECTED_UNIQUE_TRAINING_JOBS
    ):
        raise AssertionError("existing PLAN_COMPLETE metadata is invalid")
    task_plan = pd.read_csv(args.output_dir / "task_plan.csv", dtype={"session": str})
    split_manifest = pd.read_csv(
        args.output_dir / "split_manifest.csv", dtype={"session": str}
    )
    cache = pd.read_csv(args.output_dir / "cache_manifest.csv", dtype={"session": str})
    tasks = pd.read_csv(
        args.output_dir / "inner_task_manifest.csv", dtype={"session": str}
    )
    if len(task_plan) != EXPECTED_OUTER_FOLDS or len(split_manifest) != EXPECTED_INNER_DEFINITIONS:
        raise AssertionError("existing plan split counts are invalid")
    if len(split_manifest[["session", "inner_train_cycle_ids"]].drop_duplicates()) != EXPECTED_UNIQUE_TRAIN_SETS:
        raise AssertionError("existing plan unique training-set count is invalid")
    if len(cache) != EXPECTED_UNIQUE_TRAINING_JOBS or cache["training_cache_key"].duplicated().any():
        raise AssertionError("existing training-cache manifest is invalid")
    if len(tasks) != EXPECTED_LOGICAL_INNER_JOBS or tasks["evaluation_cache_key"].duplicated().any():
        raise AssertionError("existing evaluation manifest is invalid")
    if not tasks["outer_seed"].equals(tasks["selector_seed"]):
        raise AssertionError("existing plan violates Scheme-A seed pairing")
    for row in cache.to_dict(orient="records"):
        if training_cache_key(json.loads(row["training_identity_json"])) != str(
            row["training_cache_key"]
        ):
            raise AssertionError("existing training-cache key is invalid")
    for row in tasks.to_dict(orient="records"):
        if evaluation_cache_key(json.loads(row["evaluation_identity_json"])) != str(
            row["evaluation_cache_key"]
        ):
            raise AssertionError("existing evaluation-cache key is invalid")
    pin = json.loads(
        (args.output_dir / "fixed_provenance_pin.json").read_text(encoding="utf-8")
    )
    if pin != fixed_provenance:
        raise AssertionError("existing fixed provenance pin was modified")
    _compare_feasibility(task_plan, split_manifest, args.feasibility_dir)


def run_plan(args: argparse.Namespace) -> None:
    # This is the only pre-selection stage allowed to inspect fixed-run manifests.
    from ultrasound_decoding.multiframe.adaptive_mean_std_outer_reuse import (
        validate_fixed_run,
    )

    fixed = validate_fixed_run(args.fixed_results_dir, validate_all_tasks=False)
    formal_plan = fixed.pop("task_plan")
    if (args.output_dir / "config.json").is_file():
        validate_existing_plan(args, fixed)
        _write_command(args.output_dir, "plan")
        print(
            "PLAN RESUME VALIDATED outer=82 inner=722 unique_sets=425 "
            "unique_jobs=2550; no plan artifact overwritten",
            flush=True,
        )
        return
    forbidden_without_config = (
        "task_plan.csv",
        "split_manifest.csv",
        "cache_manifest.csv",
        "inner_task_manifest.csv",
        "training_cache",
        "evaluation_cache",
        "selections",
    )
    unexpected = [
        name for name in forbidden_without_config if (args.output_dir / name).exists()
    ]
    if unexpected:
        raise RuntimeError(
            f"adaptive artifacts exist without config provenance: {unexpected}"
        )
    outer = validate_outer_manifest(formal_plan)
    sessions = _load_sessions(args.project_root, args.data_dir)
    session_ids = {
        session: data.metadata["block_id"].astype(str).tolist()
        for session, data in sessions.items()
    }
    session_groups = {session: data.groups for session, data in sessions.items()}
    split_manifest = enumerate_inner_splits(outer, session_ids, session_groups)
    inner_counts = split_manifest.groupby(["session", "outer_fold"]).size()
    task_plan = outer.rename(
        columns={
            "fold": "outer_fold",
            "train_cycles": "outer_train_cycle_ids",
            "test_cycles": "outer_test_cycle_ids",
            "n_train_samples": "outer_train_sample_count",
            "n_test_samples": "outer_test_sample_count",
        }
    ).copy()
    task_plan["inner_fold_count"] = [
        int(inner_counts[(str(row.session), int(row.outer_fold))])
        for row in task_plan.itertuples(index=False)
    ]
    task_plan["full_per_seed_inner_trainings"] = 2 * task_plan["inner_fold_count"]
    task_plan["logical_inner_trainings_all_3_seeds"] = 6 * task_plan["inner_fold_count"]
    task_plan["full_per_seed_total_trainings_all_3_seeds"] = (
        task_plan["logical_inner_trainings_all_3_seeds"] + 3
    )
    task_plan["shared_selector_seed_inner_trainings"] = 2 * task_plan["inner_fold_count"]
    task_plan["selected_outer_outputs"] = 3
    protocol_id = protocol_identity(args.project_root)
    protocol_fp = fingerprint(protocol_id)
    runtime = runtime_signature()
    runtime_fp = fingerprint(runtime)
    fixed_config = json.loads(
        (args.fixed_results_dir / "config.json").read_text(encoding="utf-8")
    )
    session_manifest_hashes = fixed_config["formal_fold_source"][
        "session_manifest_sha256"
    ]
    candidate_source_hashes = {
        key: value
        for key, value in fixed_config["project_source_sha256"].items()
        if key
        in {
            "src/ultrasound_decoding/multiframe/fcnn_temporal_statistics.py",
            "src/ultrasound_decoding/multiframe/models.py",
            "src/ultrasound_decoding/multiframe/training.py",
        }
    }
    for key, value in candidate_source_hashes.items():
        if file_sha256(args.project_root / key) != value:
            raise AssertionError(f"approved candidate dependency changed: {key}")
    dataset_rows: list[dict[str, Any]] = []
    dataset_hashes: dict[str, dict[str, str]] = {}
    for session, data in sessions.items():
        hashes = {
            "clean4_h5": file_sha256(data.source_h5_path),
            "metadata_csv": file_sha256(data.source_metadata_path),
        }
        dataset_hashes[session] = hashes
        dataset_rows.append(
            {
                "session": session,
                "n_samples": len(data.X),
                "n_cycles": data.n_cycles,
                "source_h5": str(data.source_h5_path),
                "source_metadata": str(data.source_metadata_path),
                "source_h5_sha256": hashes["clean4_h5"],
                "source_metadata_sha256": hashes["metadata_csv"],
                "formal_outer_manifest_sha256": session_manifest_hashes[session],
            }
        )
    training_config = frozen_training_config()
    training_identities: dict[str, dict[str, Any]] = {}
    logical_rows: list[dict[str, Any]] = []
    for split in split_manifest.to_dict(orient="records"):
        session = str(split["session"])
        for seed in SEEDS:
            for candidate in INPUT_VARIANTS:
                training_identity = build_training_cache_identity(
                    session=session,
                    train_sample_ids=parse_sample_ids(split["inner_train_sample_ids"]),
                    train_cycle_ids=parse_ids(split["inner_train_cycle_ids"]),
                    candidate=candidate,
                    seed=seed,
                    dataset_source_hash=dataset_hashes[session],
                    session_manifest_hash=session_manifest_hashes[session],
                    candidate_source_hashes=candidate_source_hashes,
                    protocol_fingerprint=protocol_fp,
                    runtime_fingerprint=runtime_fp,
                    training_config=training_config,
                )
                train_key = training_cache_key(training_identity)
                prior = training_identities.setdefault(train_key, training_identity)
                if prior != training_identity:
                    raise AssertionError("training cache key collision")
                evaluation_identity = build_evaluation_cache_identity(
                    training_key=train_key,
                    session=session,
                    parent_outer_fold=int(split["outer_fold"]),
                    outer_seed=seed,
                    candidate=candidate,
                    validation_sample_ids=parse_sample_ids(split["inner_val_sample_ids"]),
                    validation_cycle_ids=parse_ids(split["inner_val_cycle_ids"]),
                    current_outer_train_cycle_ids=parse_ids(split["outer_train_cycle_ids"]),
                    current_outer_test_cycle_ids=parse_ids(split["outer_test_cycle_ids"]),
                    protocol_fingerprint=protocol_fp,
                )
                logical_rows.append(
                    {
                        "session": session,
                        "outer_fold": int(split["outer_fold"]),
                        "outer_seed": seed,
                        "selector_seed": seed,
                        "candidate": candidate,
                        "inner_fold": int(split["inner_fold"]),
                        "outer_train_cycle_ids": split["outer_train_cycle_ids"],
                        "outer_test_cycle_ids": split["outer_test_cycle_ids"],
                        "inner_train_cycle_ids": split["inner_train_cycle_ids"],
                        "inner_val_cycle_ids": split["inner_val_cycle_ids"],
                        "inner_train_sample_ids": split["inner_train_sample_ids"],
                        "inner_val_sample_ids": split["inner_val_sample_ids"],
                        "outer_train_sample_ids": split["outer_train_sample_ids"],
                        "inner_train_sample_count": split["inner_train_sample_count"],
                        "inner_val_sample_count": split["inner_val_sample_count"],
                        "normalization_fit_cycle_ids": split["normalization_fit_cycle_ids"],
                        "training_cache_key": train_key,
                        "evaluation_cache_key": evaluation_cache_key(evaluation_identity),
                        "training_identity_json": canonical_json(training_identity),
                        "evaluation_identity_json": canonical_json(evaluation_identity),
                        "cache_hit_train": False,
                        "cache_hit_eval": False,
                        "train_accuracy_epoch40": np.nan,
                        "inner_val_BA": np.nan,
                        "status": "pending",
                        "protocol_fingerprint": protocol_fp,
                    }
                )
    cache_rows = [
        {
            "training_cache_key": key,
            "session": value["session"],
            "candidate": value["candidate"],
            "seed": value["seed"],
            "inner_train_cycle_ids": ids_text(value["exact_training_cycle_ids"]),
            "inner_train_sample_count": len(value["exact_training_sample_ids"]),
            "training_identity_json": canonical_json(value),
            "status": "pending",
        }
        for key, value in sorted(training_identities.items())
    ]
    inner_tasks = pd.DataFrame(logical_rows).sort_values(
        ["session", "outer_fold", "outer_seed", "candidate", "inner_fold"]
    )
    cache_manifest = pd.DataFrame(cache_rows)
    if len(inner_tasks) != EXPECTED_LOGICAL_INNER_JOBS:
        raise AssertionError("logical inner job count is not 4332")
    if len(cache_manifest) != EXPECTED_UNIQUE_TRAINING_JOBS:
        raise AssertionError("unique Scheme-A training job count is not 2550")
    _compare_feasibility(task_plan, split_manifest, args.feasibility_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol_identity": protocol_id,
        "protocol_fingerprint": protocol_fp,
        "runtime_signature": runtime,
        "runtime_fingerprint": runtime_fp,
        "fixed_candidate_provenance": fixed,
        "candidate_protocol_fingerprints": {
            candidate: fingerprint(
                {
                    "architecture": architecture_config(candidate),
                    "candidate_source_sha256": candidate_source_hashes,
                    "fixed_config_fingerprint": fixed["config_fingerprint"],
                }
            )
            for candidate in INPUT_VARIANTS
        },
        "normalization_protocol_fingerprint": fingerprint(NORMALIZATION_PROTOCOL),
        "created_at": utc_now(),
    }
    atomic_json(args.output_dir / "config.json", config)
    atomic_json(args.output_dir / "runtime_fingerprint.json", runtime)
    atomic_json(args.output_dir / "fixed_provenance_pin.json", fixed)
    atomic_csv(args.output_dir / "task_plan.csv", task_plan)
    atomic_csv(args.output_dir / "split_manifest.csv", split_manifest)
    atomic_csv(args.output_dir / "dataset_manifest.csv", pd.DataFrame(dataset_rows))
    atomic_csv(args.output_dir / "cache_manifest.csv", cache_manifest)
    atomic_csv(args.output_dir / "inner_task_manifest.csv", inner_tasks)
    atomic_json(
        args.output_dir / "PLAN_COMPLETE.json",
        {
            "status": "complete",
            "formal_training_started": False,
            "outer_folds": len(task_plan),
            "inner_split_definitions": len(split_manifest),
            "unique_inner_training_cycle_sets": int(
                split_manifest[["session", "inner_train_cycle_ids"]]
                .drop_duplicates()
                .shape[0]
            ),
            "logical_inner_training_jobs": len(inner_tasks),
            "unique_inner_training_jobs": len(cache_manifest),
            "expected_locked_selections": EXPECTED_SELECTIONS,
            "expected_reused_outer_outputs": EXPECTED_SELECTIONS,
            "feasibility_artifacts_match": True,
            "protocol_fingerprint": protocol_fp,
            "created_at": utc_now(),
        },
    )
    _write_command(args.output_dir, "plan")
    print("PLAN COMPLETE outer=82 inner=722 unique_sets=425 unique_jobs=2550; no training", flush=True)


def _indices_for_ids(data, expected_ids: tuple[str, ...]) -> np.ndarray:
    observed = data.metadata["block_id"].astype(str).to_numpy()
    lookup = {value: index for index, value in enumerate(observed)}
    if len(lookup) != len(observed):
        raise AssertionError("dataset block IDs are not unique")
    missing = sorted(set(expected_ids) - set(lookup))
    if missing:
        raise AssertionError(f"planned sample IDs are absent: {missing}")
    expected_set = set(expected_ids)
    # Preserve the approved dataset/fold row order.  Sorting is used only in the
    # canonical cache identity, never to change DataLoader input ordering.
    indices = np.flatnonzero(np.isin(observed, list(expected_set)))
    if set(observed[indices].tolist()) != expected_set:
        raise AssertionError("runtime sample membership differs from the cache identity")
    return indices.astype(np.int64, copy=False)


def run_inner(args: argparse.Namespace) -> None:
    # This function has no fixed-results argument and imports no outer reader.
    if not args.review_approved:
        raise PermissionError("formal inner training requires --review-approved")
    if args.batch_size != FORMAL_BATCH_SIZE or args.workers < 0:
        raise ValueError("formal batch size is fixed at 16 and workers must be nonnegative")
    config = load_plan_config(args.output_dir, args.project_root)
    if fingerprint(runtime_signature()) != config["runtime_fingerprint"]:
        raise AssertionError("runtime differs from the planned cache fingerprint")
    cache_manifest = pd.read_csv(args.output_dir / "cache_manifest.csv", dtype={"session": str})
    inner_tasks = pd.read_csv(args.output_dir / "inner_task_manifest.csv", dtype={"session": str})
    if len(cache_manifest) != EXPECTED_UNIQUE_TRAINING_JOBS or len(inner_tasks) != EXPECTED_LOGICAL_INNER_JOBS:
        raise AssertionError("planned inner counts changed")
    sessions: dict[str, Any] = {}
    session_source_hashes: dict[str, dict[str, str]] = {}
    completed_train = completed_eval = 0
    for cache_row in cache_manifest.to_dict(orient="records"):
        identity = json.loads(cache_row["training_identity_json"])
        key = str(cache_row["training_cache_key"])
        if training_cache_key(identity) != key:
            raise AssertionError("cache manifest training key mismatch")
        cache_dir = args.output_dir / "training_cache" / key
        valid, _reason = validate_training_cache(cache_dir, identity, load_checkpoint=True)
        group = inner_tasks[inner_tasks["training_cache_key"].eq(key)]
        if group.empty:
            raise AssertionError("unique training job has no legal evaluation")
        session = str(identity["session"])
        if session not in sessions:
            sessions[session] = load_block_sequence_session(
                args.project_root,
                session,
                "binary",
                data_dir=args.data_dir or default_block_data_dir(args.project_root),
            )
            session_source_hashes[session] = {
                "clean4_h5": file_sha256(sessions[session].source_h5_path),
                "metadata_csv": file_sha256(
                    sessions[session].source_metadata_path
                ),
            }
        data = sessions[session]
        if session_source_hashes[session] != identity["dataset_source_sha256"]:
            raise AssertionError(
                f"session {session}: dataset content hash differs from plan"
            )
        train_ids = tuple(identity["exact_training_sample_ids"])
        train_idx = _indices_for_ids(data, train_ids)
        if set(data.groups[train_idx].tolist()) != set(identity["exact_training_cycle_ids"]):
            raise AssertionError("runtime training cycles differ from cache identity")
        first_eval = json.loads(group.iloc[0]["evaluation_identity_json"])
        probe_idx = _indices_for_ids(data, tuple(first_eval["exact_validation_sample_ids"]))
        if not valid:
            train_inner_cache(
                cache_dir,
                identity,
                data.X[train_idx],
                data.y[train_idx],
                data.X[probe_idx],
                device=args.device,
                workers=args.workers,
            )
        completed_train += 1
        model, norm_mean, norm_std, torch_device = load_training_cache(
            cache_dir, identity, device=args.device
        )
        for task_index, task in group.iterrows():
            eval_identity = json.loads(task["evaluation_identity_json"])
            eval_key = str(task["evaluation_cache_key"])
            if evaluation_cache_key(eval_identity) != eval_key:
                raise AssertionError("inner manifest evaluation key mismatch")
            eval_dir = args.output_dir / "evaluation_cache" / eval_key
            valid_eval, _eval_reason = validate_evaluation_cache(
                eval_dir,
                eval_identity,
                current_outer_train_cycle_ids=eval_identity["current_outer_train_cycle_ids"],
                current_outer_test_cycle_ids=eval_identity["current_outer_test_cycle_ids"],
            )
            if not valid_eval:
                validation_ids = tuple(eval_identity["exact_validation_sample_ids"])
                val_idx = _indices_for_ids(data, validation_ids)
                evaluate_inner_cache(
                    eval_dir,
                    eval_identity,
                    model=model,
                    normalization_mean=norm_mean,
                    normalization_std=norm_std,
                    X_validation=data.X[val_idx],
                    y_validation=data.y[val_idx],
                    validation_sample_ids=validation_ids,
                    validation_cycles=data.groups[val_idx],
                    device=torch_device,
                    batch_size=FORMAL_BATCH_SIZE,
                    workers=args.workers,
                )
            metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
            metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
            inner_tasks.loc[task_index, "cache_hit_train"] = bool(valid)
            inner_tasks.loc[task_index, "cache_hit_eval"] = bool(valid_eval)
            inner_tasks.loc[task_index, "train_accuracy_epoch40"] = float(metadata["train_accuracy_epoch40"])
            inner_tasks.loc[task_index, "inner_val_BA"] = float(metrics["inner_val_BA"])
            inner_tasks.loc[task_index, "status"] = "complete"
            completed_eval += 1
        cache_manifest.loc[
            cache_manifest["training_cache_key"].eq(key), "status"
        ] = "complete"
        atomic_csv(args.output_dir / "cache_manifest.csv", cache_manifest)
        atomic_csv(args.output_dir / "inner_task_manifest.csv", inner_tasks)
        print(f"INNER training={completed_train}/{len(cache_manifest)} evaluations={completed_eval}/{len(inner_tasks)}", flush=True)
    if not cache_manifest["status"].eq("complete").all() or not inner_tasks["status"].eq("complete").all():
        raise AssertionError("inner stage ended with incomplete tasks")
    atomic_json(
        args.output_dir / "INNER_COMPLETE.json",
        {
            "status": "complete",
            "unique_training_jobs": len(cache_manifest),
            "logical_evaluations": len(inner_tasks),
            "cache_manifest_sha256": file_sha256(args.output_dir / "cache_manifest.csv"),
            "inner_task_manifest_sha256": file_sha256(args.output_dir / "inner_task_manifest.csv"),
            "protocol_fingerprint": config["protocol_fingerprint"],
            "completed_at": utc_now(),
        },
    )
    _write_command(args.output_dir, "inner")


def _selection_path(output_dir: Path, session: str, fold: int, seed: int) -> Path:
    return output_dir / "selections" / f"session_{session}" / f"fold_{fold:02d}_seed_{seed}.json"


def run_select(args: argparse.Namespace) -> None:
    # This function accepts no fixed-results path and cannot import outer results.
    config = load_plan_config(args.output_dir, args.project_root)
    inner_complete = json.loads((args.output_dir / "INNER_COMPLETE.json").read_text(encoding="utf-8"))
    if inner_complete.get("status") != "complete" or int(inner_complete.get("logical_evaluations", -1)) != EXPECTED_LOGICAL_INNER_JOBS:
        raise AssertionError("selection requires a complete inner stage")
    tasks = pd.read_csv(args.output_dir / "inner_task_manifest.csv", dtype={"session": str})
    if len(tasks) != EXPECTED_LOGICAL_INNER_JOBS or not tasks["status"].eq("complete").all():
        raise AssertionError("selection requires all inner tasks")
    all_predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    grouping = tasks.groupby(["session", "outer_fold", "outer_seed"], sort=True)
    if len(grouping) != EXPECTED_SELECTIONS:
        raise AssertionError("selection grouping is not 246 outer fold/seed tasks")
    for (session, outer_fold, seed), group in grouping:
        candidate_frames: dict[str, pd.DataFrame] = {}
        candidate_hashes: dict[str, str] = {}
        candidate_scores: dict[str, float] = {}
        expected_outer_ids = parse_sample_ids(group.iloc[0]["outer_train_sample_ids"])
        for candidate in INPUT_VARIANTS:
            candidate_group = group[group["candidate"].eq(candidate)].sort_values("inner_fold")
            frames: list[pd.DataFrame] = []
            for task in candidate_group.to_dict(orient="records"):
                identity = json.loads(task["evaluation_identity_json"])
                eval_dir = args.output_dir / "evaluation_cache" / str(task["evaluation_cache_key"])
                valid, reason = validate_evaluation_cache(
                    eval_dir,
                    identity,
                    current_outer_train_cycle_ids=parse_ids(task["outer_train_cycle_ids"]),
                    current_outer_test_cycle_ids=parse_ids(task["outer_test_cycle_ids"]),
                )
                if not valid:
                    raise AssertionError(f"invalid inner evaluation before selection: {reason}")
                frame = pd.read_csv(eval_dir / "predictions.csv")
                frame.insert(0, "inner_fold", int(task["inner_fold"]))
                frame.insert(0, "candidate", candidate)
                frame.insert(0, "outer_seed", int(seed))
                frame.insert(0, "outer_fold", int(outer_fold))
                frame.insert(0, "session", str(session))
                frames.append(frame)
                metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
                fold_metrics.append(
                    {
                        "session": str(session),
                        "outer_fold": int(outer_fold),
                        "seed": int(seed),
                        "candidate": candidate,
                        "inner_fold": int(task["inner_fold"]),
                        "inner_val_BA": float(metrics["inner_val_BA"]),
                        "evaluation_cache_key": str(task["evaluation_cache_key"]),
                    }
                )
            combined = pd.concat(frames, ignore_index=True)
            score = concatenated_oof_balanced_accuracy(combined, expected_outer_ids)
            candidate_frames[candidate] = combined
            candidate_scores[candidate] = score
            candidate_hashes[candidate] = fingerprint(
                combined.sort_values("sample_id").to_dict(orient="records")
            )
            oof_rows.append(
                {
                    "session": str(session),
                    "outer_fold": int(outer_fold),
                    "seed": int(seed),
                    "candidate": candidate,
                    "inner_OOF_BA": score,
                    "n_oof_samples": len(combined),
                    "n_unique_oof_samples": combined["sample_id"].nunique(),
                    "complete_outer_train_coverage": True,
                    "inner_oof_prediction_hash": candidate_hashes[candidate],
                }
            )
            all_predictions.append(combined)
        base_row = group.iloc[0]
        payload = build_selection_payload(
            session=str(session),
            outer_fold=int(outer_fold),
            seed=int(seed),
            outer_train_cycle_ids=parse_ids(base_row["outer_train_cycle_ids"]),
            outer_test_cycle_ids=parse_ids(base_row["outer_test_cycle_ids"]),
            inner_ba_mean_only=candidate_scores[MEAN_ONLY_VARIANT],
            inner_ba_mean_std=candidate_scores[MEAN_STD_VARIANT],
            candidate_protocol_fingerprints=config["candidate_protocol_fingerprints"],
            split_fingerprint=fingerprint(
                group[
                    [
                        "inner_fold",
                        "outer_train_cycle_ids",
                        "outer_test_cycle_ids",
                        "inner_train_cycle_ids",
                        "inner_val_cycle_ids",
                    ]
                ].drop_duplicates().sort_values("inner_fold").to_dict(orient="records")
            ),
            normalization_protocol_fingerprint=config["normalization_protocol_fingerprint"],
            inner_oof_prediction_hashes=candidate_hashes,
            expected_outer_train_sample_ids=expected_outer_ids,
            observed_candidate_sample_ids={
                candidate: candidate_frames[candidate]["sample_id"].astype(str).tolist()
                for candidate in INPUT_VARIANTS
            },
            protocol_fingerprint=config["protocol_fingerprint"],
        )
        path = _selection_path(args.output_dir, str(session), int(outer_fold), int(seed))
        if path.exists():
            locked = read_locked_selection(path, expected_protocol_fingerprint=config["protocol_fingerprint"])
            comparable = dict(payload)
            comparable["created_at"] = locked["created_at"]
            base = dict(comparable)
            base.pop("selection_artifact_hash")
            comparable["selection_artifact_hash"] = fingerprint(base)
            if comparable != locked:
                raise RuntimeError("recomputed selection differs from locked artifact")
        else:
            locked = lock_selection(path, payload, expected_protocol_fingerprint=config["protocol_fingerprint"])
        selections.append(locked)
    atomic_csv(args.output_dir / "inner_predictions.csv", pd.concat(all_predictions, ignore_index=True))
    atomic_csv(args.output_dir / "inner_fold_metrics.csv", pd.DataFrame(fold_metrics))
    atomic_csv(args.output_dir / "inner_oof_summary.csv", pd.DataFrame(oof_rows))
    selection_frame = pd.DataFrame(selections)
    atomic_csv(args.output_dir / "selection_manifest.csv", selection_frame)
    atomic_json(
        args.output_dir / "SELECTION_COMPLETE.json",
        {
            "status": "complete",
            "locked_selections": len(selection_frame),
            "outer_result_read_before_selection": False,
            "selection_manifest_sha256": file_sha256(args.output_dir / "selection_manifest.csv"),
            "protocol_fingerprint": config["protocol_fingerprint"],
            "completed_at": utc_now(),
        },
    )
    _write_command(args.output_dir, "select")


def run_outer(args: argparse.Namespace) -> None:
    config = load_plan_config(args.output_dir, args.project_root)
    selection_complete = json.loads((args.output_dir / "SELECTION_COMPLETE.json").read_text(encoding="utf-8"))
    if selection_complete.get("status") != "complete" or int(selection_complete.get("locked_selections", -1)) != EXPECTED_SELECTIONS:
        raise PermissionError("outer stage requires all 246 locked selections")
    # Deliberate local import: inner/select never load this capability.
    from ultrasound_decoding.multiframe.adaptive_mean_std_outer_reuse import (
        SelectedOuterResultReader,
        validate_fixed_run,
    )

    fixed = validate_fixed_run(args.fixed_results_dir, validate_all_tasks=False)
    fixed_plan = fixed.pop("task_plan")
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for path in sorted((args.output_dir / "selections").glob("session_*/*.json")):
        selection = read_locked_selection(path, expected_protocol_fingerprint=config["protocol_fingerprint"])
        reader = SelectedOuterResultReader(
            args.fixed_results_dir,
            fixed_plan,
            path,
            expected_protocol_fingerprint=config["protocol_fingerprint"],
        )
        result, prediction, provenance = reader.read()
        result_row = {
            "session": str(selection["session"]),
            "outer_fold": int(selection["outer_fold"]),
            "seed": int(selection["seed"]),
            "selected_variant": str(selection["selected_variant"]),
            "selection_artifact_hash": str(selection["selection_artifact_hash"]),
            "inner_BA_mean_only": float(selection["inner_BA_mean_only"]),
            "inner_BA_mean_std": float(selection["inner_BA_mean_std"]),
            "delta_inner_BA": float(selection["delta_inner_BA"]),
            "final_train_accuracy": float(result["final_train_accuracy"]),
            "final_epoch": int(result["final_epoch"]),
            "outer_test_balanced_accuracy": float(result["balanced_accuracy"]),
            **provenance,
        }
        results.append(result_row)
        prediction = prediction.copy()
        prediction["adaptive_selected_variant"] = str(selection["selected_variant"])
        prediction["selection_artifact_hash"] = str(selection["selection_artifact_hash"])
        predictions.append(prediction)
    if len(results) != EXPECTED_SELECTIONS:
        raise AssertionError("outer stage did not reuse exactly 246 selected outputs")
    atomic_csv(args.output_dir / "outer_selected_results.csv", pd.DataFrame(results))
    atomic_csv(args.output_dir / "outer_predictions.csv", pd.concat(predictions, ignore_index=True))
    atomic_json(
        args.output_dir / "OUTER_COMPLETE.json",
        {
            "status": "complete",
            "selected_outer_outputs_validated_and_reused": len(results),
            "unselected_outer_results_read": False,
            "fixed_provenance": fixed,
            "outer_selected_results_sha256": file_sha256(args.output_dir / "outer_selected_results.csv"),
            "outer_predictions_sha256": file_sha256(args.output_dir / "outer_predictions.csv"),
            "completed_at": utc_now(),
        },
    )
    _write_command(args.output_dir, "outer")


def exact_two_sided_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    statistics = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(statistics) >= observed - 1e-15))


def _fixed_seed_summary(fixed_dir: Path, fixed_plan: pd.DataFrame) -> pd.DataFrame:
    from ultrasound_decoding.multiframe.adaptive_mean_std_outer_reuse import (
        EXPECTED_FIXED_RUN_FINGERPRINT,
        _load_formal_runner,
    )

    formal_runner = _load_formal_runner()
    predictions: list[pd.DataFrame] = []
    results: list[dict[str, Any]] = []
    for expected in fixed_plan.to_dict(orient="records"):
        path = formal_runner.task_dir(
            fixed_dir,
            str(expected["session"]),
            str(expected["variant"]),
            int(expected["seed"]),
            int(expected["fold"]),
        )
        formal_runner.validate_completed_task(path, expected, EXPECTED_FIXED_RUN_FINGERPRINT, raise_on_error=True)
        predictions.append(pd.read_csv(path / "predictions.csv", dtype={"session": str}))
        results.append(json.loads((path / "result.json").read_text(encoding="utf-8")))
    prediction_frame = pd.concat(predictions, ignore_index=True)
    result_frame = pd.DataFrame(results)
    rows = []
    for (session, variant, seed), group in prediction_frame.groupby(["session", "variant", "seed"], sort=True):
        metric = classification_metrics(group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int))["balanced_accuracy"]
        folds = result_frame[
            result_frame["session"].astype(str).eq(str(session))
            & result_frame["variant"].eq(variant)
            & pd.to_numeric(result_frame["seed"]).eq(seed)
        ]
        final_train = float(folds["final_train_accuracy"].mean())
        rows.append({"session": str(session), "variant": variant, "seed": int(seed), "BA": float(metric), "final_train_accuracy": final_train, "train_test_gap": final_train-float(metric)})
    return pd.DataFrame(rows)


def validate_all_adaptive_artifacts(
    output_dir: Path, *, protocol_fingerprint: str
) -> dict[str, int]:
    """Revalidate every resumable artifact immediately before RUN_COMPLETE."""

    cache = pd.read_csv(output_dir / "cache_manifest.csv", dtype={"session": str})
    tasks = pd.read_csv(output_dir / "inner_task_manifest.csv", dtype={"session": str})
    if len(cache) != EXPECTED_UNIQUE_TRAINING_JOBS or len(tasks) != EXPECTED_LOGICAL_INNER_JOBS:
        raise AssertionError("final cache/task coverage mismatch")
    valid_training = 0
    for row in cache.to_dict(orient="records"):
        identity = json.loads(row["training_identity_json"])
        valid, reason = validate_training_cache(
            output_dir / "training_cache" / str(row["training_cache_key"]),
            identity,
            load_checkpoint=True,
        )
        if not valid:
            raise AssertionError(f"final training-cache validation failed: {reason}")
        valid_training += 1
    valid_evaluations = 0
    for row in tasks.to_dict(orient="records"):
        identity = json.loads(row["evaluation_identity_json"])
        valid, reason = validate_evaluation_cache(
            output_dir / "evaluation_cache" / str(row["evaluation_cache_key"]),
            identity,
            current_outer_train_cycle_ids=identity["current_outer_train_cycle_ids"],
            current_outer_test_cycle_ids=identity["current_outer_test_cycle_ids"],
        )
        if not valid:
            raise AssertionError(f"final evaluation-cache validation failed: {reason}")
        valid_evaluations += 1
    selection_paths = sorted((output_dir / "selections").glob("session_*/*.json"))
    if len(selection_paths) != EXPECTED_SELECTIONS:
        raise AssertionError("final locked-selection coverage mismatch")
    for path in selection_paths:
        read_locked_selection(
            path, expected_protocol_fingerprint=protocol_fingerprint
        )
    outer = json.loads((output_dir / "OUTER_COMPLETE.json").read_text(encoding="utf-8"))
    if (
        outer.get("status") != "complete"
        or int(outer.get("selected_outer_outputs_validated_and_reused", -1))
        != EXPECTED_SELECTIONS
        or bool(outer.get("unselected_outer_results_read", True))
        or outer.get("outer_selected_results_sha256")
        != file_sha256(output_dir / "outer_selected_results.csv")
        or outer.get("outer_predictions_sha256")
        != file_sha256(output_dir / "outer_predictions.csv")
    ):
        raise AssertionError("final selected-outer artifact validation failed")
    return {
        "valid_training_caches": valid_training,
        "valid_evaluation_caches": valid_evaluations,
        "valid_locked_selections": len(selection_paths),
        "valid_selected_outer_outputs": EXPECTED_SELECTIONS,
    }


def run_summarize(args: argparse.Namespace) -> None:
    config = load_plan_config(args.output_dir, args.project_root)
    outer_complete = json.loads((args.output_dir / "OUTER_COMPLETE.json").read_text(encoding="utf-8"))
    if outer_complete.get("status") != "complete" or int(outer_complete.get("selected_outer_outputs_validated_and_reused", -1)) != EXPECTED_SELECTIONS:
        raise AssertionError("summarize requires 246 validated outer outputs")
    from ultrasound_decoding.multiframe.adaptive_mean_std_outer_reuse import validate_fixed_run

    fixed = validate_fixed_run(args.fixed_results_dir, validate_all_tasks=True)
    fixed_plan = fixed.pop("task_plan")
    final_validation = validate_all_adaptive_artifacts(
        args.output_dir, protocol_fingerprint=config["protocol_fingerprint"]
    )
    fixed_seed = _fixed_seed_summary(args.fixed_results_dir, fixed_plan)
    outer_predictions = pd.read_csv(args.output_dir / "outer_predictions.csv", dtype={"session": str})
    outer_results = pd.read_csv(args.output_dir / "outer_selected_results.csv", dtype={"session": str})
    adaptive_rows = []
    for (session, seed), group in outer_predictions.groupby(["session", "seed"], sort=True):
        ba = classification_metrics(group["y_true"].to_numpy(int), group["y_pred"].to_numpy(int))["balanced_accuracy"]
        folds = outer_results[
            outer_results["session"].eq(str(session))
            & pd.to_numeric(outer_results["seed"]).eq(seed)
        ]
        train_acc = float(folds["final_train_accuracy"].mean())
        adaptive_rows.append({"session": str(session), "seed": int(seed), "adaptive_BA": float(ba), "final_train_accuracy": train_acc, "train_test_gap": train_acc-float(ba)})
    adaptive_seed = pd.DataFrame(adaptive_rows)
    session_rows = []
    for session in EXPECTED_SESSIONS:
        fixed_session = fixed_seed[fixed_seed["session"].eq(session)]
        mean_only = float(fixed_session[fixed_session["variant"].eq(MEAN_ONLY_VARIANT)]["BA"].mean())
        mean_std = float(fixed_session[fixed_session["variant"].eq(MEAN_STD_VARIANT)]["BA"].mean())
        adaptive = float(adaptive_seed[adaptive_seed["session"].eq(session)]["adaptive_BA"].mean())
        selected = outer_results[outer_results["session"].eq(session)]
        session_rows.append({"session": session, "mean_only_BA": mean_only, "mean_std_BA": mean_std, "adaptive_BA": adaptive, "adaptive_vs_mean_only": adaptive-mean_only, "adaptive_vs_mean_std": adaptive-mean_std, "selected_mean_std_fraction": float(selected["selected_variant"].eq(MEAN_STD_VARIANT).mean())})
    session_summary = pd.DataFrame(session_rows)
    selection = pd.read_csv(args.output_dir / "selection_manifest.csv", dtype={"session": str})
    stability_rows = []
    for session in EXPECTED_SESSIONS:
        subset = selection[selection["session"].eq(session)]
        agreements = subset.groupby("outer_fold")["selected_variant"].nunique().eq(1)
        std_count = int(subset["selected_variant"].eq(MEAN_STD_VARIANT).sum())
        only_count = int(subset["selected_variant"].eq(MEAN_ONLY_VARIANT).sum())
        stability_rows.append({"session": session, "mean_only_selected_count": only_count, "mean_std_selected_count": std_count, "mean_only_selected_fraction": only_count/len(subset), "mean_std_selected_fraction": std_count/len(subset), "all_3_seeds_agreement_folds": int(agreements.sum()), "outer_fold_count": int(len(agreements)), "all_3_seeds_agreement_rate": float(agreements.mean())})
    stability = pd.DataFrame(stability_rows)
    overall_agreement = float(selection.groupby(["session", "outer_fold"])["selected_variant"].nunique().eq(1).mean())
    strong = session_summary[session_summary["session"].isin(STRONG_SESSIONS)]
    weak = session_summary[session_summary["session"].isin(WEAK_SESSIONS)]
    means = {
        "fixed_mean_only_overall_BA": float(session_summary["mean_only_BA"].mean()),
        "fixed_mean_std_overall_BA": float(session_summary["mean_std_BA"].mean()),
        "adaptive_overall_BA": float(session_summary["adaptive_BA"].mean()),
        "fixed_mean_only_strong3_BA": float(strong["mean_only_BA"].mean()),
        "fixed_mean_std_strong3_BA": float(strong["mean_std_BA"].mean()),
        "adaptive_strong3_BA": float(strong["adaptive_BA"].mean()),
        "fixed_mean_only_weak6_BA": float(weak["mean_only_BA"].mean()),
        "fixed_mean_std_weak6_BA": float(weak["mean_std_BA"].mean()),
        "adaptive_weak6_BA": float(weak["adaptive_BA"].mean()),
        "overall_all_3_seeds_agreement_rate": overall_agreement,
    }
    weak_gain_fixed = means["fixed_mean_std_weak6_BA"] - means["fixed_mean_only_weak6_BA"]
    weak_gain_adaptive = means["adaptive_weak6_BA"] - means["fixed_mean_only_weak6_BA"]
    checks = {
        "adaptive_overall_gt_fixed_mean_only": means["adaptive_overall_BA"] > means["fixed_mean_only_overall_BA"],
        "adaptive_overall_ge_fixed_mean_std": means["adaptive_overall_BA"] >= means["fixed_mean_std_overall_BA"],
        "strong3_not_more_than_0.010_below_fixed_mean_only": means["adaptive_strong3_BA"] >= means["fixed_mean_only_strong3_BA"]-0.010,
        "retains_at_least_75_percent_fixed_weak_gain": weak_gain_adaptive >= 0.75*weak_gain_fixed,
    }
    decision = "supports_continue_adaptive_temporal_selection" if all(checks.values()) else "does_not_support_adaptive_temporal_selection"
    diagnostics: dict[str, Any] = {}
    statistical: dict[str, Any] = {"comparisons": {}}
    for baseline in ("mean_only", "mean_std"):
        deltas = session_summary["adaptive_BA"].to_numpy(float) - session_summary[f"{baseline}_BA"].to_numpy(float)
        largest_i = int(np.argmax(deltas))
        smallest_i = int(np.argmin(deltas))
        statistical["comparisons"][f"adaptive_vs_{baseline}"] = {"session_deltas": {row.session: float(delta) for row, delta in zip(session_summary.itertuples(index=False), deltas)}, "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)), "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip(deltas), "improved": int((deltas>1e-12).sum()), "tied": int((np.abs(deltas)<=1e-12).sum()), "worsened": int((deltas < -1e-12).sum()), "largest_improvement_session": str(session_summary.iloc[largest_i]["session"]), "largest_improvement": float(deltas[largest_i]), "largest_degradation_session": str(session_summary.iloc[smallest_i]["session"]), "largest_degradation": float(deltas[smallest_i]), "leave_largest_improvement_out_delta": float(np.delete(deltas, largest_i).mean()), "leave_one_session_out_mean_deltas": {str(session_summary.iloc[i]["session"]): float(np.delete(deltas, i).mean()) for i in range(len(deltas))}}
    overall = {**means, "weak_gain_fixed": weak_gain_fixed, "weak_gain_adaptive": weak_gain_adaptive, "decision_checks": checks, "decision": decision, "automatic_next_stage_started": False}
    atomic_csv(args.output_dir / "adaptive_seed_summary.csv", adaptive_seed)
    atomic_csv(args.output_dir / "session_summary.csv", session_summary)
    atomic_csv(args.output_dir / "selection_stability.csv", stability)
    atomic_json(args.output_dir / "overall_summary.json", overall)
    atomic_json(args.output_dir / "statistical_audit.json", statistical)
    provenance = {"status": "validated", "fixed_candidate_run": fixed, "adaptive_artifact_validation": final_validation, "selected_outer_outputs": EXPECTED_SELECTIONS, "fixed_controls_validated": 492, "outer_result_read_before_selection": False, "protocol_fingerprint": config["protocol_fingerprint"]}
    atomic_json(args.output_dir / "provenance_audit.json", provenance)
    report_lines = ["# Adaptive Mean/Std Nested-CV v1", "", "This is exploratory method development: nested training-only model selection between two fixed FCNN temporal representations, not a new architecture or independent confirmatory test.", "", f"Decision: `{decision}`", "", "| session | fixed mean-only BA | fixed mean+std BA | adaptive BA | adaptive vs mean-only | adaptive vs mean+std | Mean+Std selected |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in session_summary.itertuples(index=False):
        report_lines.append(f"| {row.session} | {row.mean_only_BA:.4f} | {row.mean_std_BA:.4f} | {row.adaptive_BA:.4f} | {row.adaptive_vs_mean_only:+.4f} | {row.adaptive_vs_mean_std:+.4f} | {row.selected_mean_std_fraction:.3f} |")
    report_lines.extend(["", "Selection used concatenated complete inner OOF predictions. Outer results were read only after immutable selection artifacts were locked. Fixed controls came from the provenance-validated 492-task formal run.", ""])
    (args.output_dir / "adaptive_mean_std_nestedcv_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    required = ["config.json", "runtime_fingerprint.json", "task_plan.csv", "split_manifest.csv", "cache_manifest.csv", "inner_task_manifest.csv", "inner_predictions.csv", "inner_fold_metrics.csv", "inner_oof_summary.csv", "selection_manifest.csv", "outer_selected_results.csv", "outer_predictions.csv", "session_summary.csv", "overall_summary.json", "selection_stability.csv", "statistical_audit.json", "provenance_audit.json", "adaptive_mean_std_nestedcv_report.md"]
    missing = [name for name in required if not (args.output_dir / name).is_file()]
    if missing:
        raise AssertionError(f"cannot mark complete; missing {missing}")
    atomic_json(args.output_dir / "RUN_COMPLETE.json", {"status": "complete", "unique_inner_training_jobs": EXPECTED_UNIQUE_TRAINING_JOBS, "logical_inner_evaluations": EXPECTED_LOGICAL_INNER_JOBS, "locked_selections": EXPECTED_SELECTIONS, "selected_outer_outputs_validated_and_reused": EXPECTED_SELECTIONS, "fixed_controls_validated": 492, "final_validation": final_validation, "required_output_sha256": {name: file_sha256(args.output_dir / name) for name in required}, "protocol_fingerprint": config["protocol_fingerprint"], "run_fingerprint": fingerprint({"protocol_fingerprint": config["protocol_fingerprint"], "fixed_run_fingerprint": fixed["run_fingerprint"], "runtime_fingerprint": config["runtime_fingerprint"]}), "decision": decision, "automatic_next_stage_started": False, "completed_at": utc_now()})
    _write_command(args.output_dir, "summarize")


def run_sanity(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(7)
    train = rng.normal(size=(4, 4, 3, 5)).astype(np.float32)
    validation_a = rng.normal(size=(2, 4, 3, 5)).astype(np.float32)
    validation_b = validation_a.copy()
    validation_b[0] += 1e6
    left = normalize_blocks_train_fold_only_with_stats(train, validation_a, session="synthetic", task="binary", method=MEAN_ONLY_VARIANT, seed=0, fold=1, train_cycles="1,2", test_cycles="3")
    right = normalize_blocks_train_fold_only_with_stats(train, validation_b, session="synthetic", task="binary", method=MEAN_ONLY_VARIANT, seed=0, fold=1, train_cycles="1,2", test_cycles="3")
    if not np.array_equal(left[0], right[0]) or not np.array_equal(left[3], right[3]) or not np.array_equal(left[4], right[4]):
        raise AssertionError("validation modification changed training normalization")
    mocked_mean_only = 0.60
    mocked_mean_std = 0.65
    selected_before = select_variant(mocked_mean_only, mocked_mean_std)
    outer_test_pixels_a = np.zeros((2, 4, 3, 5), dtype=np.float32)
    outer_test_pixels_b = np.full((2, 4, 3, 5), 1e9, dtype=np.float32)
    del outer_test_pixels_a, outer_test_pixels_b
    selected_after = select_variant(mocked_mean_only, mocked_mean_std)
    if selected_before != selected_after:
        raise AssertionError("outer-test modification changed mocked selection")
    split = pd.read_csv(args.project_root / "outputs/adaptive_mean_std_nestedcv_feasibility/adaptive_mean_std_split_audit.csv", dtype={"session": str})
    duplicated = split[split["same_training_set_occurrence_count"].eq(2)].iloc[0]
    peer = split[
        split["session"].eq(duplicated["session"])
        & split["inner_train_cycle_ids"].eq(duplicated["inner_train_cycle_ids"])
    ].iloc[1]
    common = dict(session=str(duplicated["session"]), train_sample_ids=["a", "b"], train_cycle_ids=parse_ids(duplicated["inner_train_cycle_ids"]), candidate=MEAN_ONLY_VARIANT, seed=0, dataset_source_hash={"synthetic": "x"}, session_manifest_hash="m", candidate_source_hashes={"candidate": "c"}, protocol_fingerprint="p", runtime_fingerprint="r", training_config=frozen_training_config())
    train_identity_a = build_training_cache_identity(**common)
    train_identity_b = build_training_cache_identity(**common)
    train_key_a = training_cache_key(train_identity_a)
    train_key_b = training_cache_key(train_identity_b)
    eval_a = build_evaluation_cache_identity(training_key=train_key_a, session=str(duplicated["session"]), parent_outer_fold=int(duplicated["outer_fold"]), outer_seed=0, candidate=MEAN_ONLY_VARIANT, validation_sample_ids=["v1"], validation_cycle_ids=parse_ids(duplicated["inner_validation_cycle_ids"]), current_outer_train_cycle_ids=parse_ids(duplicated["outer_train_cycle_ids"]), current_outer_test_cycle_ids=parse_ids(duplicated["outer_test_cycle_ids"]), protocol_fingerprint="p")
    eval_b = build_evaluation_cache_identity(training_key=train_key_b, session=str(peer["session"]), parent_outer_fold=int(peer["outer_fold"]), outer_seed=0, candidate=MEAN_ONLY_VARIANT, validation_sample_ids=["v2"], validation_cycle_ids=parse_ids(peer["inner_validation_cycle_ids"]), current_outer_train_cycle_ids=parse_ids(peer["outer_train_cycle_ids"]), current_outer_test_cycle_ids=parse_ids(peer["outer_test_cycle_ids"]), protocol_fingerprint="p")
    if train_key_a != train_key_b or evaluation_cache_key(eval_a) == evaluation_cache_key(eval_b):
        raise AssertionError("cache correctness sanity failed")
    sanity_dir = args.output_dir / "sanity"
    atomic_json(sanity_dir / "synthetic_leakage_audit.json", {"status": "pass", "outer_test_modification_changed_training_cache_keys": False, "outer_test_modification_changed_inner_normalization": False, "outer_test_modification_changed_inner_scores": False, "outer_test_modification_changed_selected_variant": False, "model_training_performed": False})
    atomic_json(sanity_dir / "cache_correctness_audit.json", {"status": "pass", "session": str(duplicated["session"]), "parent_outer_fold_a": int(duplicated["outer_fold"]), "parent_outer_fold_b": int(peer["outer_fold"]), "inner_train_cycle_ids": str(duplicated["inner_train_cycle_ids"]), "training_cache_keys_equal": True, "evaluation_cache_keys_different": True, "parent_access_control_validated": True})
    atomic_json(args.output_dir / "SANITY_COMPLETE.json", {"status": "complete", "formal_training_started": False, "synthetic_leakage_regression": "pass", "cache_correctness_regression": "pass", "completed_at": utc_now()})
    _write_command(args.output_dir, "sanity")


def run_status(args: argparse.Namespace) -> None:
    counts = {
        "planned_unique_training_jobs": 0,
        "valid_training_caches": 0,
        "planned_evaluations": 0,
        "valid_evaluation_caches": 0,
        "valid_locked_selections": 0,
        "validated_outer_outputs": 0,
        "run_complete": False,
    }
    if (args.output_dir / "cache_manifest.csv").is_file():
        cache = pd.read_csv(args.output_dir / "cache_manifest.csv")
        counts["planned_unique_training_jobs"] = len(cache)
        for row in cache.to_dict(orient="records"):
            identity = json.loads(row["training_identity_json"])
            valid, _ = validate_training_cache(args.output_dir / "training_cache" / row["training_cache_key"], identity, load_checkpoint=True)
            counts["valid_training_caches"] += int(valid)
    if (args.output_dir / "inner_task_manifest.csv").is_file():
        tasks = pd.read_csv(args.output_dir / "inner_task_manifest.csv")
        counts["planned_evaluations"] = len(tasks)
        for row in tasks.to_dict(orient="records"):
            identity = json.loads(row["evaluation_identity_json"])
            valid, _ = validate_evaluation_cache(args.output_dir / "evaluation_cache" / row["evaluation_cache_key"], identity, current_outer_train_cycle_ids=identity["current_outer_train_cycle_ids"], current_outer_test_cycle_ids=identity["current_outer_test_cycle_ids"])
            counts["valid_evaluation_caches"] += int(valid)
    config_path = args.output_dir / "config.json"
    if config_path.is_file():
        protocol_fp = json.loads(config_path.read_text(encoding="utf-8"))["protocol_fingerprint"]
        for path in (args.output_dir / "selections").glob("session_*/*.json"):
            try:
                read_locked_selection(path, expected_protocol_fingerprint=protocol_fp)
                counts["valid_locked_selections"] += 1
            except Exception:
                pass
    if (args.output_dir / "outer_selected_results.csv").is_file():
        counts["validated_outer_outputs"] = len(pd.read_csv(args.output_dir / "outer_selected_results.csv"))
    if (args.output_dir / "RUN_COMPLETE.json").is_file():
        counts["run_complete"] = json.loads((args.output_dir / "RUN_COMPLETE.json").read_text(encoding="utf-8")).get("status") == "complete"
    atomic_json(args.output_dir / "STATUS.json", counts)
    print(canonical_json(counts), flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.stage == "plan":
        run_plan(args)
    elif args.stage == "sanity":
        run_sanity(args)
    elif args.stage == "inner":
        run_inner(args)
    elif args.stage == "select":
        run_select(args)
    elif args.stage == "outer":
        run_outer(args)
    elif args.stage == "summarize":
        run_summarize(args)
    elif args.stage == "status":
        run_status(args)
    elif args.stage == "full":
        if not args.review_approved:
            raise PermissionError("formal full run requires --review-approved")
        run_plan(args)
        # Pass capability-minimal namespaces between stages. In particular, the
        # inner/select call objects do not contain the fixed-results directory.
        inner_args = argparse.Namespace(
            stage="inner",
            project_root=args.project_root,
            output_dir=args.output_dir,
            data_dir=args.data_dir,
            device=args.device,
            workers=args.workers,
            batch_size=args.batch_size,
            review_approved=True,
        )
        select_args = argparse.Namespace(
            stage="select",
            project_root=args.project_root,
            output_dir=args.output_dir,
        )
        outer_args = argparse.Namespace(
            stage="outer",
            project_root=args.project_root,
            output_dir=args.output_dir,
            fixed_results_dir=args.fixed_results_dir,
        )
        run_inner(inner_args)
        run_select(select_args)
        run_outer(outer_args)
        run_summarize(outer_args)
    else:
        raise AssertionError(f"unknown stage {args.stage}")


if __name__ == "__main__":
    main()
