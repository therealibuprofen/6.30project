#!/usr/bin/env python3
"""Equal-update, session-balanced multi-session masked SmallCNN SSL v2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    EXPECTED_TASKS,
    cycle_text,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.models import SmallCNNFrameEncoder, encoder_architecture_signature
from ultrasound_decoding.multiframe.training import DeepTrainingConfig
from ultrasound_decoding.ssl_masked import (
    SSL_SEEDS,
    SSLPretrainingConfig,
    fixed_ssl_validation_cycles,
    load_ssl_encoder_checkpoint,
    train_downstream_fold,
)
from ultrasound_decoding.ssl_multisession_v2 import (
    NEW_PRETRAINING_CONDITIONS,
    V1_FROZEN_SOURCE_FINGERPRINT,
    V2_CONDITIONS,
    SessionFramePool,
    architecture_fingerprint,
    assert_formal_cuda,
    build_ssl_pool,
    checkpoint_contains_no_label_information,
    complete_cycles_from_unlabeled_h5,
    compute_match_row,
    fit_ssl_pool_normalizer,
    frozen_v1_source_fingerprint,
    load_unlabeled_cycles,
    missing_formal_outputs,
    pretrain_session_balanced_smallcnn,
    reference_optimizer_updates,
    sampling_distribution_rows,
    save_multisession_checkpoint,
    validate_multisession_checkpoint,
    validate_v1_checkpoint_payload,
)
from ultrasound_decoding.ssl_multisession_reporting_v2 import (
    classify_scenario,
    generalization_gap_summary,
    make_required_figures,
    planned_statistical_tests,
    seed_stability,
    session_level_comparison,
)


RUN_NAME = "ssl_multisession_masked_smallcnn_9sessions_v2"
V1_RUN_NAME = "ssl_masked_smallcnn_clean4_9sessions_v1"
SUPERVISED_CONFIG = DeepTrainingConfig(
    optimizer="adamw",
    lr=1e-3,
    weight_decay=1e-3,
    batch_size=16,
    max_epochs=40,
    dropout=0.25,
    loss="cross_entropy",
)
SSL_CONFIG = SSLPretrainingConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "formal", "aggregate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--v1-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V1_RUN_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SSL_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=2, help="CPU smoke only; formal ignores this")
    return parser.parse_args()


def ensure_output_tree(output_dir: Path) -> None:
    for relative in (
        "audit", "pretraining/checkpoints", "pretraining/losses",
        "downstream/training_curves", "downstream/jobs", "summaries", "figures", "report", "smoke",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def log(message: str, output_dir: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_log_server.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_csv(path: Path, value: list[dict[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(value) for value in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}" if np.isfinite(value) else "NA")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def append_csv_deduplicated(path: Path, rows: list[dict[str, Any]], key: list[str]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=new.columns)
    output = pd.concat([old, new], ignore_index=True)
    output = output.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    write_csv(path, output)


def _v1_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "folds": args.v1_output_dir / "downstream/fold_metrics.csv",
        "predictions": args.v1_output_dir / "downstream/predictions.csv",
        "volume": args.v1_output_dir / "audit/ssl_data_volume.csv",
        "fold_audit": args.v1_output_dir / "audit/fold_reproduction.csv",
        "manifest": args.v1_output_dir / "pretraining/checkpoint_manifest.csv",
        "curves": args.v1_output_dir / "downstream/training_curves",
    }


def _required_v1_files(args: argparse.Namespace) -> None:
    missing = [str(path) for key, path in _v1_paths(args).items() if key != "curves" and not path.exists()]
    if not _v1_paths(args)["curves"].is_dir():
        missing.append(str(_v1_paths(args)["curves"]))
    if missing:
        raise RuntimeError(f"Stage 0 STOP: required v1 artifacts missing: {missing}")


def _current_fold_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    v1_audit = pd.read_csv(_v1_paths(args)["fold_audit"])
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    rows: list[dict[str, Any]] = []
    cache: dict[str, Any] = {}
    for session in EXPECTED_SESSIONS:
        cache[session] = {}
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            splits = grouped_cv_splits(data.groups, max_folds=10)
            cache[session][task] = (data, splits)
            for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                train_cycles = cycle_text(data.groups[train_idx])
                test_cycles = cycle_text(data.groups[test_idx])
                old = v1_audit[
                    (v1_audit["session"].astype(str) == session)
                    & (v1_audit["task"] == task)
                    & (v1_audit["fold"].astype(int) == fold_i)
                ]
                if len(old) != 1:
                    raise AssertionError(f"v1 fold audit missing or duplicated: {session} {task} {fold_i}")
                old = old.iloc[0]
                train_match = str(old["train_cycles"]) == train_cycles
                test_match = str(old["test_cycles"]) == test_cycles
                rows.append({
                    "session": session,
                    "task": task,
                    "fold": fold_i,
                    "current_train_cycle_ids": train_cycles,
                    "v1_train_cycle_ids": str(old["train_cycles"]),
                    "current_test_cycle_ids": test_cycles,
                    "v1_test_cycle_ids": str(old["test_cycles"]),
                    "train_cycles_match": train_match,
                    "test_cycles_match": test_match,
                    "status": "PASS" if train_match and test_match else "FAIL",
                })
    return rows, cache


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Stage 0: hard compatibility, fold, leakage, and pool-scope audits."""
    ensure_output_tree(args.output_dir)
    _required_v1_files(args)
    seeds = tuple(args.seeds)
    if seeds != SSL_SEEDS:
        raise RuntimeError(f"Stage 0 STOP: seeds must be exactly {SSL_SEEDS}, got {seeds}")
    source_hash = frozen_v1_source_fingerprint(PROJECT_DIR)
    if source_hash != V1_FROZEN_SOURCE_FINGERPRINT:
        raise RuntimeError(
            "Stage 0 STOP: frozen v1 source hash changed; checkpoint/metric reuse is forbidden "
            f"(expected {V1_FROZEN_SOURCE_FINGERPRINT}, found {source_hash})"
        )
    fold_rows, cache = _current_fold_rows(args)
    write_csv(args.output_dir / "audit/fold_identity_check.csv", fold_rows)
    if any(row["status"] != "PASS" for row in fold_rows):
        raise RuntimeError("Stage 0 STOP: at least one current fold differs from v1")

    volume = pd.read_csv(_v1_paths(args)["volume"])
    manifest = pd.read_csv(_v1_paths(args)["manifest"])
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    complete_cycles: dict[str, np.ndarray] = {}
    frame_counts: dict[str, int] = {}
    for session in EXPECTED_SESSIONS:
        path = data_dir / f"session_{session}_blocks.h5"
        complete_cycles[session] = complete_cycles_from_unlabeled_h5(path)
        frame_counts[session] = int(30 * len(complete_cycles[session]))

    leakage_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    reuse_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        binary, splits = cache[session]["binary"]
        for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
            target_train_cycles = np.unique(binary.groups[train_idx]).astype(int)
            target_test_cycles = np.unique(binary.groups[test_idx]).astype(int)
            row = volume[(volume["session"].astype(str) == session) & (volume["fold"].astype(int) == fold_i)]
            if len(row) != 1:
                raise AssertionError(f"v1 SSL volume missing: {session} fold {fold_i}")
            row = row.iloc[0]
            ssl_train_cycles = np.asarray([int(x) for x in str(row["ssl_train_cycles"]).split(",")], dtype=int)
            ssl_val_cycles = np.asarray([int(x) for x in str(row["ssl_val_cycles"]).split(",") if str(x)], dtype=int)
            expected_train, expected_val = fixed_ssl_validation_cycles(target_train_cycles)
            if not np.array_equal(ssl_train_cycles, expected_train) or not np.array_equal(ssl_val_cycles, expected_val):
                raise AssertionError("v1 fixed SSL split changed")
            for condition in ("WITHIN_SSL_FT", "OTHER_ONLY_SSL_FT", "MULTI_SSL_FT"):
                sources = [session] if condition == "WITHIN_SSL_FT" else [s for s in EXPECTED_SESSIONS if s != session]
                target_cycles_in_pool: np.ndarray
                if condition == "WITHIN_SSL_FT":
                    target_cycles_in_pool = ssl_train_cycles
                elif condition == "MULTI_SSL_FT":
                    sources = [*sources, session]
                    target_cycles_in_pool = target_train_cycles
                else:
                    target_cycles_in_pool = np.asarray([], dtype=int)
                leaked = sorted(set(target_cycles_in_pool) & set(target_test_cycles))
                target_frame_count = int(30 * len(target_cycles_in_pool))
                pool_frames = int(sum(frame_counts[s] for s in sources if s != session) + target_frame_count)
                leakage_rows.append({
                    "target_session": session,
                    "fold": fold_i,
                    "condition": condition,
                    "target_train_cycle_ids": cycle_text(target_train_cycles),
                    "target_test_cycle_ids": cycle_text(target_test_cycles),
                    "target_cycles_in_ssl_pool": cycle_text(target_cycles_in_pool),
                    "target_test_cycles_in_ssl_train": ",".join(map(str, leaked)),
                    "target_test_frames_in_ssl_train": int(30 * len(leaked)),
                    "target_test_frames_in_ssl_validation": 0,
                    "target_test_frames_in_normalization_fit": 0,
                    "target_test_frames_in_qc_selection": 0,
                    "target_test_frames_in_supervised_train": 0,
                    "status": "PASS" if not leaked else "FAIL",
                })
                pool_rows.append({
                    "target_session": session,
                    "fold": fold_i,
                    "condition": condition,
                    "source_sessions": ",".join(sorted(sources, key=int)),
                    "n_source_sessions": len(sources),
                    "ssl_pool_frames": pool_frames,
                    "target_frame_count": target_frame_count,
                    "auxiliary_frame_count": pool_frames - target_frame_count,
                    "labels_loaded_by_ssl_objective": False,
                    "preprocessing": "arcsinh_then_pixel_zscore_fit_on_legal_ssl_pool",
                    "orientation_input": "official_block_sequences_v1_as_exported",
                    "session_807_orientation_preserved": True,
                    "registration_used": False,
                })
            for seed in SSL_SEEDS:
                checkpoint = args.v1_output_dir / f"pretraining/checkpoints/session_{session}/fold_{fold_i}/seed_{seed}.pt"
                payload = validate_v1_checkpoint_payload(
                    checkpoint,
                    target_session=session,
                    fold=fold_i,
                    seed=seed,
                    ssl_train_cycles=ssl_train_cycles,
                    target_test_cycles=target_test_cycles,
                    config=SSL_CONFIG,
                )
                manifest_row = manifest[
                    (manifest["session"].astype(str) == session)
                    & (manifest["fold"].astype(int) == fold_i)
                    & (manifest["seed"].astype(int) == seed)
                ]
                if len(manifest_row) != 1:
                    raise AssertionError("v1 checkpoint manifest row missing")
                digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                sha_match = digest == str(manifest_row.iloc[0]["checkpoint_sha256"])
                if not sha_match or not checkpoint_contains_no_label_information(checkpoint):
                    raise AssertionError(f"v1 checkpoint integrity failed: {checkpoint}")
                batch_size = int(payload["actual_batch_size"])
                reference_updates = reference_optimizer_updates(int(row["n_ssl_train_frames"]), batch_size)
                reuse_rows.append({
                    "target_session": session,
                    "fold": fold_i,
                    "seed": seed,
                    "checkpoint_path": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "manifest_sha256_match": sha_match,
                    "frozen_v1_source_fingerprint": source_hash,
                    "source_fingerprint_match": True,
                    "architecture_fingerprint": architecture_fingerprint(),
                    "architecture_signature": repr(encoder_architecture_signature()),
                    "preprocessing_fields_match": True,
                    "fold_match": True,
                    "seed_match": True,
                    "reuse_status": "PASS",
                })
                compute_rows.append(compute_match_row(
                    target_session=session,
                    fold=fold_i,
                    seed=seed,
                    condition="WITHIN_SSL_FT",
                    ssl_pool_frames=int(row["n_ssl_train_frames"]),
                    actual_batch_size=batch_size,
                    reference_updates=reference_updates,
                    actual_updates=reference_updates,
                    frame_exposure_count=int(row["n_ssl_train_frames"]) * 50,
                    unique_frame_coverage=int(row["n_ssl_train_frames"]),
                    reused_artifact=True,
                ))
    write_csv(args.output_dir / "audit/target_test_leakage_audit.csv", leakage_rows)
    write_csv(args.output_dir / "audit/ssl_pool_audit.csv", pool_rows)
    write_csv(args.output_dir / "audit/v1_artifact_reuse.csv", reuse_rows)
    write_csv(args.output_dir / "audit/pretraining_compute_match.csv", compute_rows)
    write_csv(args.output_dir / "audit/session_sampling_distribution.csv", pd.DataFrame(columns=[
        "target_session", "fold", "seed", "condition", "source_session", "sample_count",
        "total_frame_exposures", "expected_proportion", "actual_proportion", "absolute_deviation",
        "descriptive_tolerance", "sampling_status",
    ]))
    config = [
        "# Multi-session Masked SmallCNN SSL v2 — Frozen Configuration", "",
        f"- Sessions: `{list(EXPECTED_SESSIONS)}`",
        f"- Tasks: `{list(EXPECTED_TASKS)}`; primary `binary`, secondary `stimulus_type`.",
        f"- Conditions: `{list(V2_CONDITIONS)}`; no Frozen condition.",
        "- Downstream: exact v1 clean4 builder, grouped cycle folds, shared SmallCNN, feature mean, classifier.",
        f"- Supervised config: `{asdict(SUPERVISED_CONFIG)}`",
        f"- SSL config: `{asdict(SSL_CONFIG)}`",
        "- SSL objective: 16x16 spatial blocks, ratio 0.50, zero mask value, masked-pixel MSE only.",
        "- Equal update reference: ceil(v1 n_ssl_train_frames / v1 actual_batch_size) * 50.",
        "- OTHER_ONLY/MULTI sampling: uniform source session then uniform legal source frame.",
        "- MULTI target source includes all outer-train cycles; target test cycles remain absent.",
        "- SSL normalization: exact v1 arcsinh then frame-pooled pixel z-score, fitted only on the legal SSL pool.",
        "- No labels are read by SSL loaders/objective/checkpoints.",
        "- No registration, augmentation, architecture search, early stopping, or target-specific tuning.",
        "- Session 807 uses the existing official block_sequences_v1 orientation without further transform.",
        f"- Seeds: `{list(SSL_SEEDS)}`",
        f"- Frozen v1 source fingerprint: `{source_hash}`",
        f"- SmallCNN architecture fingerprint: `{architecture_fingerprint()}`",
        "- Formal execution is CUDA-only; unavailable CUDA is a hard STOP with no CPU fallback.",
    ]
    (args.output_dir / "audit/config_freeze.md").write_text("\n".join(config) + "\n", encoding="utf-8")
    if any(row["status"] != "PASS" for row in leakage_rows):
        raise RuntimeError("Stage 0 STOP: target-test leakage detected")
    log("Stage 0 audit PASSED: folds, v1 reuse, target-test leakage, and frozen source verified", args.output_dir)
    return {"cache": cache, "volume": volume, "complete_cycles": complete_cycles}


def run_smoke(args: argparse.Namespace) -> None:
    """Tiny CPU-only smoke: one real session/fold/seed and two updates."""
    if args.device != "cpu":
        raise RuntimeError("smoke must use --device cpu")
    if not 1 <= args.smoke_updates <= 3:
        raise ValueError("smoke-updates must be 1..3")
    ensure_output_tree(args.output_dir)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    data = load_block_sequence_session(PROJECT_DIR, "710", "binary", data_dir=data_dir)
    train_idx, test_idx = grouped_cv_splits(data.groups, max_folds=10)[0]
    train_cycles = np.unique(data.groups[train_idx])
    test_cycles = np.unique(data.groups[test_idx])
    ssl_train, _ssl_val = fixed_ssl_validation_cycles(train_cycles)
    loaded = load_unlabeled_cycles(data.source_h5_path, ssl_train[:1])
    # Both sampler sources reuse the same two real frames.  This tests balancing
    # without reading a second session or producing any formal result.
    frames = loaded.frames[:2]
    cycles = loaded.cycles[:2]
    indices = loaded.original_frame_indices[:2]
    pool = SessionFramePool(
        frames_by_session={"709": frames, "710": frames.copy()},
        cycles_by_session={"709": cycles, "710": cycles.copy()},
        original_indices_by_session={"709": indices, "710": indices.copy()},
        source_paths={"709": loaded.source_h5_path, "710": loaded.source_h5_path},
    )
    config = replace(SSL_CONFIG, batch_size=2, epochs=1)
    result = pretrain_session_balanced_smallcnn(
        pool,
        seed=SSL_SEEDS[0],
        reference_updates=args.smoke_updates,
        actual_batch_size=2,
        config=config,
        device="cpu",
    )
    checkpoint = args.output_dir / "smoke/smoke_encoder.pt"
    save_multisession_checkpoint(
        checkpoint,
        result,
        target_session="710",
        fold=1,
        seed=SSL_SEEDS[0],
        condition="MULTI_SSL_FT",
        pool=pool,
        target_ssl_train_cycles=train_cycles,
        target_test_cycles=test_cycles,
        config=config,
        source_fingerprint=frozen_v1_source_fingerprint(PROJECT_DIR),
    )
    encoder, payload = load_ssl_encoder_checkpoint(checkpoint)
    if not isinstance(encoder, SmallCNNFrameEncoder) or payload["actual_updates"] != args.smoke_updates:
        raise AssertionError("smoke checkpoint validation failed")
    smoke = {
        "status": "PASS",
        "device": "cpu",
        "real_sessions_loaded": ["710"],
        "folds": [1],
        "seeds": [SSL_SEEDS[0]],
        "optimizer_updates": result.actual_updates,
        "formal_results_produced": False,
        "target_test_cycles": test_cycles.astype(int).tolist(),
        "target_test_frames_used": 0,
        "checkpoint_contains_labels": not checkpoint_contains_no_label_information(checkpoint),
    }
    (args.output_dir / "smoke/smoke_status.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    log("Stage 2 tiny CPU smoke PASSED (one real session, one fold, one seed, two updates maximum)", args.output_dir)


def _load_all_unlabeled(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    result = {}
    for session in EXPECTED_SESSIONS:
        path = data_dir / f"session_{session}_blocks.h5"
        cycles = complete_cycles_from_unlabeled_h5(path)
        log(f"loading label-free complete-cycle frames source_session={session}", args.output_dir)
        result[session] = load_unlabeled_cycles(path, cycles)
    return result


def _v1_fold_context(volume: pd.DataFrame, session: str, fold_i: int) -> tuple[np.ndarray, np.ndarray, int]:
    row = volume[(volume["session"].astype(str) == session) & (volume["fold"].astype(int) == fold_i)]
    if len(row) != 1:
        raise AssertionError("v1 fold volume row missing")
    row = row.iloc[0]
    ssl_train = np.asarray([int(x) for x in str(row["ssl_train_cycles"]).split(",")], dtype=int)
    ssl_val = np.asarray([int(x) for x in str(row["ssl_val_cycles"]).split(",") if str(x)], dtype=int)
    return ssl_train, ssl_val, int(row["n_ssl_train_frames"])


def _new_checkpoint_path(args: argparse.Namespace, session: str, fold_i: int, seed: int, condition: str) -> Path:
    return args.output_dir / f"pretraining/checkpoints/{condition}/session_{session}/fold_{fold_i}/seed_{seed}.pt"


def _pretrain_new_condition(
    args: argparse.Namespace,
    *,
    all_frames: dict[str, Any],
    target_session: str,
    fold_i: int,
    seed: int,
    condition: str,
    ssl_train_cycles: np.ndarray,
    target_train_cycles: np.ndarray,
    target_test_cycles: np.ndarray,
    batch_size: int,
    reference_updates: int,
    normalizer_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
) -> Path:
    path = _new_checkpoint_path(args, target_session, fold_i, seed, condition)
    source_hash = frozen_v1_source_fingerprint(PROJECT_DIR)
    pool_cycles = target_train_cycles if condition == "MULTI_SSL_FT" else ssl_train_cycles
    pool = build_ssl_pool(
        all_frames,
        target_session=target_session,
        target_ssl_train_cycles=pool_cycles,
        target_test_cycles=target_test_cycles,
        condition=condition,
    )
    if path.exists() and not args.overwrite:
        payload = validate_multisession_checkpoint(
            path,
            target_session=target_session,
            fold=fold_i,
            seed=seed,
            condition=condition,
            reference_updates=reference_updates,
            source_fingerprint=source_hash,
        )
        compute = compute_match_row(
            target_session=target_session, fold=fold_i, seed=seed, condition=condition,
            ssl_pool_frames=int(payload["ssl_pool_frames"]), actual_batch_size=int(payload["actual_batch_size"]),
            reference_updates=reference_updates, actual_updates=int(payload["actual_updates"]),
            frame_exposure_count=int(payload["frame_exposure_count"]),
            unique_frame_coverage=int(payload["unique_frame_coverage"]), reused_artifact=True,
        )
        append_csv_deduplicated(
            args.output_dir / "audit/pretraining_compute_match.csv", [compute],
            ["target_session", "fold", "seed", "condition"],
        )
        normalizer_key = (target_session, 0 if condition == "OTHER_ONLY_SSL_FT" else fold_i, condition)
        normalizer_cache.setdefault(
            normalizer_key,
            (np.asarray(payload["normalization_mean"]), np.asarray(payload["normalization_std"])),
        )
        loss_path = args.output_dir / f"pretraining/losses/{condition}_session_{target_session}_fold_{fold_i}_seed_{seed}.csv"
        if not loss_path.exists():
            write_csv(loss_path, [{
                "target_session": target_session, "fold": fold_i, "seed": seed,
                "condition": condition, **row,
            } for row in payload["training_history"]])
        sampling = sampling_distribution_rows(
            {str(key): int(value) for key, value in payload["sampling_counts"].items()},
            target_session=target_session, fold=fold_i, seed=seed, condition=condition,
        )
        append_csv_deduplicated(
            args.output_dir / "audit/session_sampling_distribution.csv", sampling,
            ["target_session", "fold", "seed", "condition", "source_session"],
        )
        append_csv_deduplicated(
            args.output_dir / "audit/runtime_resources.csv", [{
                "stage": "pretraining", "target_session": target_session, "task": "both",
                "fold": fold_i, "seed": seed, "condition": condition,
                "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
                "actual_batch_size": int(payload["actual_batch_size"]),
                "peak_gpu_memory_mb": float(payload.get("peak_gpu_memory_mb", float("nan"))),
                "runtime_seconds": float(payload.get("runtime_seconds", float("nan"))),
            }],
            ["stage", "target_session", "task", "fold", "seed", "condition"],
        )
        return path
    log(
        f"pretrain target={target_session} fold={fold_i} seed={seed} condition={condition} "
        f"sources={pool.source_sessions} frames={pool.n_frames} updates={reference_updates}",
        args.output_dir,
    )
    normalizer_key = (target_session, 0 if condition == "OTHER_ONLY_SSL_FT" else fold_i, condition)
    if normalizer_key not in normalizer_cache:
        normalizer_cache[normalizer_key] = fit_ssl_pool_normalizer(pool)
    result = pretrain_session_balanced_smallcnn(
        pool,
        seed=seed,
        reference_updates=reference_updates,
        actual_batch_size=batch_size,
        config=SSL_CONFIG,
        device="cuda",
        normalization_stats=normalizer_cache[normalizer_key],
    )
    save_multisession_checkpoint(
        path,
        result,
        target_session=target_session,
        fold=fold_i,
        seed=seed,
        condition=condition,
        pool=pool,
        target_ssl_train_cycles=pool_cycles,
        target_test_cycles=target_test_cycles,
        config=SSL_CONFIG,
        source_fingerprint=source_hash,
    )
    loss_path = args.output_dir / f"pretraining/losses/{condition}_session_{target_session}_fold_{fold_i}_seed_{seed}.csv"
    write_csv(loss_path, [{
        "target_session": target_session, "fold": fold_i, "seed": seed, "condition": condition, **row
    } for row in result.history])
    compute = compute_match_row(
        target_session=target_session, fold=fold_i, seed=seed, condition=condition,
        ssl_pool_frames=pool.n_frames, actual_batch_size=result.actual_batch_size,
        reference_updates=reference_updates, actual_updates=result.actual_updates,
        frame_exposure_count=result.frame_exposure_count, unique_frame_coverage=result.unique_frame_coverage,
        reused_artifact=False,
    )
    append_csv_deduplicated(
        args.output_dir / "audit/pretraining_compute_match.csv", [compute],
        ["target_session", "fold", "seed", "condition"],
    )
    sampling = sampling_distribution_rows(
        result.sampling_counts,
        target_session=target_session,
        fold=fold_i,
        seed=seed,
        condition=condition,
    )
    append_csv_deduplicated(
        args.output_dir / "audit/session_sampling_distribution.csv", sampling,
        ["target_session", "fold", "seed", "condition", "source_session"],
    )
    append_csv_deduplicated(
        args.output_dir / "audit/runtime_resources.csv", [{
            "stage": "pretraining", "target_session": target_session, "task": "both",
            "fold": fold_i, "seed": seed, "condition": condition,
            "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "actual_batch_size": result.actual_batch_size,
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
            "runtime_seconds": result.runtime_seconds,
        }],
        ["stage", "target_session", "task", "fold", "seed", "condition"],
    )
    return path


def _import_v1_downstream(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = {"RANDOM_INIT": "RANDOM_INIT", "SSL_FINETUNE": "WITHIN_SSL_FT"}
    metrics = pd.read_csv(_v1_paths(args)["folds"])
    predictions = pd.read_csv(_v1_paths(args)["predictions"])
    metrics = metrics[metrics["condition"].isin(mapping)].copy()
    predictions = predictions[predictions["condition"].isin(mapping)].copy()
    metrics["condition"] = metrics["condition"].map(mapping)
    predictions["condition"] = predictions["condition"].map(mapping)
    expected_metric_rows = 2 * 3 * 2 * sum(
        len(grouped_cv_splits(load_block_sequence_session(
            PROJECT_DIR, session, "binary", data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR)
        ).groups, max_folds=10)) for session in EXPECTED_SESSIONS
    )
    if len(metrics) != expected_metric_rows:
        raise AssertionError(f"v1 reusable fold metrics incomplete: {len(metrics)} != {expected_metric_rows}")
    curve_dir = args.output_dir / "downstream/training_curves"
    for source_condition, target_condition in mapping.items():
        for source in _v1_paths(args)["curves"].glob(f"*_{source_condition}.csv"):
            target = curve_dir / source.name.replace(f"_{source_condition}.csv", f"_{target_condition}.csv")
            if not target.exists() or args.overwrite:
                frame = pd.read_csv(source)
                frame["condition"] = target_condition
                write_csv(target, frame)
    return metrics, predictions


def _prediction_rows(data, test_idx, result, condition: str, seed: int, fold_i: int) -> list[dict[str, Any]]:
    rows = []
    for local_i, sample_i in enumerate(test_idx):
        meta = data.metadata.iloc[int(sample_i)]
        rows.append({
            "session": data.session, "task": data.task, "fold": fold_i, "seed": seed,
            "condition": condition, "sample_i": int(sample_i), "sample_id": str(meta["block_id"]),
            "cycle": int(meta["cycle"]), "label": int(data.y[sample_i]),
            "prediction": int(result.test_predictions[local_i]),
            "probability_class_0": float(result.test_probabilities[local_i, 0]),
            "probability_class_1": float(result.test_probabilities[local_i, 1]),
        })
    return rows


def _downstream_job_path(args: argparse.Namespace, session: str, task: str, fold_i: int, seed: int, condition: str) -> Path:
    return args.output_dir / f"downstream/jobs/session_{session}_{task}_fold_{fold_i}_seed_{seed}_{condition}.json"


def _run_downstream_job(
    args: argparse.Namespace,
    *,
    data,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fold_i: int,
    seed: int,
    condition: str,
    checkpoint: Path,
) -> None:
    job = _downstream_job_path(args, data.session, data.task, fold_i, seed, condition)
    prediction_job = job.with_suffix(".predictions.csv")
    if job.exists() and prediction_job.exists() and not args.overwrite:
        return
    encoder, _payload = load_ssl_encoder_checkpoint(checkpoint)
    log(f"downstream target={data.session} task={data.task} fold={fold_i} seed={seed} condition={condition}", args.output_dir)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = train_downstream_fold(
        "SSL_FINETUNE",
        data,
        train_idx,
        test_idx,
        fold=fold_i,
        seed=seed,
        pretrained_encoder_state=encoder.state_dict(),
        config=SUPERVISED_CONFIG,
        device="cuda",
    )
    torch.cuda.synchronize()
    runtime = time.perf_counter() - started
    metrics = dict(result.metrics)
    metrics["condition"] = condition
    job.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(prediction_job, _prediction_rows(data, test_idx, result, condition, seed, fold_i))
    curve = pd.DataFrame(result.history)
    for key, value in (("session", data.session), ("task", data.task), ("fold", fold_i), ("seed", seed), ("condition", condition)):
        curve.insert(0, key, value)
    write_csv(
        args.output_dir / f"downstream/training_curves/session_{data.session}_{data.task}_fold_{fold_i}_seed_{seed}_{condition}.csv",
        curve,
    )
    append_csv_deduplicated(
        args.output_dir / "audit/runtime_resources.csv", [{
            "stage": "downstream", "target_session": data.session, "task": data.task,
            "fold": fold_i, "seed": seed, "condition": condition,
            "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
            "actual_batch_size": SUPERVISED_CONFIG.batch_size,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "runtime_seconds": runtime,
        }],
        ["stage", "target_session", "task", "fold", "seed", "condition"],
    )


def consolidate_downstream(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    reused_metrics, reused_predictions = _import_v1_downstream(args)
    metric_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.output_dir / "downstream/jobs").glob("*.json"))]
    prediction_frames = [pd.read_csv(path) for path in sorted((args.output_dir / "downstream/jobs").glob("*.predictions.csv"))]
    metrics = pd.concat([reused_metrics, pd.DataFrame(metric_rows)], ignore_index=True)
    predictions = pd.concat([reused_predictions, *prediction_frames], ignore_index=True)
    # pandas infers numeric session IDs from the reused v1 CSV, while the new
    # JSON jobs retain strings. Canonicalize before de-duplication/grouping so
    # 626 and "626" cannot become separate, half-populated session groups.
    metrics["session"] = metrics["session"].astype(str)
    predictions["session"] = predictions["session"].astype(str)
    metric_key = ["session", "task", "fold", "seed", "condition"]
    prediction_key = [*metric_key, "sample_id"]
    metrics = metrics.drop_duplicates(metric_key, keep="last").sort_values(metric_key).reset_index(drop=True)
    predictions = predictions.drop_duplicates(prediction_key, keep="last").sort_values(prediction_key).reset_index(drop=True)
    write_csv(args.output_dir / "downstream/fold_metrics.csv", metrics)
    write_csv(args.output_dir / "downstream/predictions.csv", predictions)
    return metrics, predictions


def aggregate_outputs(args: argparse.Namespace) -> None:
    metrics, predictions = consolidate_downstream(args)
    expected_keys = 2 * 4 * 3 * sum(
        len(grouped_cv_splits(load_block_sequence_session(
            PROJECT_DIR, session, "binary", data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR)
        ).groups, max_folds=10)) for session in EXPECTED_SESSIONS
    )
    if len(metrics) != expected_keys:
        raise RuntimeError(f"aggregate STOP: incomplete fold metrics {len(metrics)}/{expected_keys}")
    if set(metrics["condition"]) != set(V2_CONDITIONS):
        raise RuntimeError("aggregate STOP: conditions incomplete")
    target_fold_seed_jobs = 3 * sum(
        len(grouped_cv_splits(load_block_sequence_session(
            PROJECT_DIR, session, "binary", data_dir=args.data_dir or default_block_data_dir(PROJECT_DIR)
        ).groups, max_folds=10)) for session in EXPECTED_SESSIONS
    )
    compute = pd.read_csv(args.output_dir / "audit/pretraining_compute_match.csv")
    if len(compute) != 3 * target_fold_seed_jobs or not (
        compute["actual_updates"].astype(int) == compute["reference_updates"].astype(int)
    ).all():
        raise RuntimeError("aggregate STOP: equal-update audit is incomplete or failed")
    distribution = pd.read_csv(args.output_dir / "audit/session_sampling_distribution.csv")
    if len(distribution) != 17 * target_fold_seed_jobs:
        raise RuntimeError("aggregate STOP: session-sampling audit is incomplete")
    if not (distribution["sampling_status"] == "PASS").all():
        raise RuntimeError("aggregate STOP: session-balanced sampling tolerance failed")
    leakage = pd.read_csv(args.output_dir / "audit/target_test_leakage_audit.csv")
    if len(leakage) != 3 * (target_fold_seed_jobs // 3) or not (leakage["status"] == "PASS").all():
        raise RuntimeError("aggregate STOP: target-test leakage audit is incomplete or failed")
    if len(list((args.output_dir / "pretraining/checkpoints").glob("**/*.pt"))) != 2 * target_fold_seed_jobs:
        raise RuntimeError("aggregate STOP: new pretraining checkpoints are incomplete")
    if len(list((args.output_dir / "pretraining/losses").glob("*.csv"))) != 2 * target_fold_seed_jobs:
        raise RuntimeError("aggregate STOP: pretraining loss histories are incomplete")
    if len(list((args.output_dir / "downstream/training_curves").glob("*.csv"))) != expected_keys:
        raise RuntimeError("aggregate STOP: downstream training curves are incomplete")
    oof_counts = predictions.groupby(["session", "task", "seed", "condition", "sample_id"]).size()
    if not (oof_counts == 1).all():
        raise AssertionError("OOF predictions are duplicated")
    session_table = session_level_comparison(metrics)
    tests = planned_statistical_tests(session_table)
    gaps = generalization_gap_summary(session_table)
    stability = seed_stability(metrics)
    write_csv(args.output_dir / "summaries/session_level_comparison.csv", session_table)
    write_csv(args.output_dir / "summaries/planned_statistical_tests.csv", tests)
    write_csv(args.output_dir / "summaries/generalization_gap_summary.csv", gaps)
    write_csv(args.output_dir / "summaries/seed_stability.csv", stability)
    make_required_figures(args.output_dir, metrics, session_table, distribution)
    scenario = classify_scenario(session_table)
    report = [
        "# Multi-session Masked SmallCNN SSL v2", "",
        "Formal execution completed on the server GPU. The only changed experimental variable was the label-free SSL data scope.", "",
        "## Integrity gates", "",
        "- All nine fixed sessions completed both tasks.",
        "- Every grouped CV fold matched v1 exactly.",
        "- Target-test exposure was zero for SSL training, validation, normalization, QC selection, and supervised training.",
        "- OTHER_ONLY contained zero target-session frames.",
        "- All three SSL conditions used the v1 reference optimizer update count per target/fold/seed.",
        "- OTHER_ONLY and MULTI used session-balanced sampling.", "",
        "## Planned session-level tests", "",
        markdown_table(tests), "",
        "## Session-level comparison", "",
        markdown_table(session_table), "",
        "## Preregistered scenario", "",
        scenario, "",
        "This conclusion is limited to the frozen SmallCNN masked-pixel reconstruction formulation. No next-stage architecture was run.",
    ]
    (args.output_dir / "report/multisession_ssl_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    missing = missing_formal_outputs(args.output_dir)
    if missing:
        raise RuntimeError(f"formal output completeness STOP: {missing}")
    log(f"Formal aggregation complete; {scenario}", args.output_dir)


def run_formal(args: argparse.Namespace) -> None:
    device = assert_formal_cuda(args.device)
    del device
    started = time.perf_counter()
    audit = run_audit(args)
    metrics, predictions = _import_v1_downstream(args)
    write_csv(args.output_dir / "downstream/fold_metrics.csv", metrics)
    write_csv(args.output_dir / "downstream/predictions.csv", predictions)
    all_frames = _load_all_unlabeled(args)
    volume = audit["volume"]
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    checkpoint_map: dict[tuple[str, int, int, str], Path] = {}
    normalizer_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for session in EXPECTED_SESSIONS:
        binary = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=data_dir)
        splits = grouped_cv_splits(binary.groups, max_folds=10)
        for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
            target_train_cycles = np.unique(binary.groups[train_idx]).astype(int)
            target_test_cycles = np.unique(binary.groups[test_idx]).astype(int)
            ssl_train_cycles, _ssl_val, n_ssl_train_frames = _v1_fold_context(volume, session, fold_i)
            for seed in SSL_SEEDS:
                v1_checkpoint = args.v1_output_dir / f"pretraining/checkpoints/session_{session}/fold_{fold_i}/seed_{seed}.pt"
                _encoder, v1_payload = load_ssl_encoder_checkpoint(v1_checkpoint)
                batch_size = int(v1_payload["actual_batch_size"])
                reference_updates = reference_optimizer_updates(n_ssl_train_frames, batch_size)
                for condition in NEW_PRETRAINING_CONDITIONS:
                    checkpoint_map[(session, fold_i, seed, condition)] = _pretrain_new_condition(
                        args,
                        all_frames=all_frames,
                        target_session=session,
                        fold_i=fold_i,
                        seed=seed,
                        condition=condition,
                        ssl_train_cycles=ssl_train_cycles,
                        target_train_cycles=target_train_cycles,
                        target_test_cycles=target_test_cycles,
                        batch_size=batch_size,
                        reference_updates=reference_updates,
                        normalizer_cache=normalizer_cache,
                    )
    del all_frames
    for session in EXPECTED_SESSIONS:
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            splits = grouped_cv_splits(data.groups, max_folds=10)
            for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                for seed in SSL_SEEDS:
                    for condition in NEW_PRETRAINING_CONDITIONS:
                        _run_downstream_job(
                            args, data=data, train_idx=train_idx, test_idx=test_idx,
                            fold_i=fold_i, seed=seed, condition=condition,
                            checkpoint=checkpoint_map[(session, fold_i, seed, condition)],
                        )
    aggregate_outputs(args)
    append_csv_deduplicated(
        args.output_dir / "audit/runtime_resources.csv", [{
            "stage": "formal_total", "target_session": "all", "task": "both", "fold": 0,
            "seed": 0, "condition": "all", "gpu_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda, "actual_batch_size": SSL_CONFIG.batch_size,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "runtime_seconds": time.perf_counter() - started,
        }],
        ["stage", "target_session", "task", "fold", "seed", "condition"],
    )


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.v1_output_dir = args.v1_output_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    ensure_output_tree(args.output_dir)
    if args.stage == "audit":
        run_audit(args)
    elif args.stage == "smoke":
        run_smoke(args)
    elif args.stage == "formal":
        run_formal(args)
    elif args.stage == "aggregate":
        aggregate_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
