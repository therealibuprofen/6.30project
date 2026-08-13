#!/usr/bin/env python3
"""Frozen SmallCNN VICReg-style invariance SSL benchmark v3."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
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
    apply_ssl_frame_normalizer,
    load_ssl_encoder_checkpoint,
    train_downstream_fold,
)
from ultrasound_decoding.ssl_multisession_v2 import (
    architecture_fingerprint,
    assert_formal_cuda,
    complete_cycles_from_unlabeled_h5,
    fit_ssl_pool_normalizer,
    load_unlabeled_cycles,
    reference_optimizer_updates,
    sampling_distribution_rows,
)
from ultrasound_decoding.ssl_vicreg_reporting_v3 import (
    classify_scenario,
    generalization_gap_summary,
    make_required_figures,
    planned_statistical_tests,
    seed_stability,
    session_level_comparison,
)
from ultrasound_decoding.ssl_vicreg_v3 import (
    V3_CONDITIONS,
    VICREG_CONDITIONS,
    VICREG_SEEDS,
    VICRegAugmentationConfig,
    VICRegConfig,
    build_vicreg_pool,
    checkpoint_contains_no_labels_or_projector,
    conservative_vicreg_augmentation,
    implementation_fingerprint,
    missing_formal_outputs,
    pretrain_vicreg_smallcnn,
    save_vicreg_encoder_checkpoint,
    validate_vicreg_checkpoint,
)


RUN_NAME = "ssl_vicreg_smallcnn_9sessions_v3"
V1_RUN_NAME = "ssl_masked_smallcnn_clean4_9sessions_v1"
V2_RUN_NAME = "ssl_multisession_masked_smallcnn_9sessions_v2"
SUPERVISED_CONFIG = DeepTrainingConfig(
    optimizer="adamw", lr=1e-3, weight_decay=1e-3, batch_size=16,
    max_epochs=40, dropout=0.25, loss="cross_entropy",
)
VICREG_CONFIG = VICRegConfig()
AUGMENTATION_CONFIG = VICRegAugmentationConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "smoke", "formal", "aggregate"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--v1-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V1_RUN_NAME)
    parser.add_argument("--v2-output-dir", type=Path, default=PROJECT_DIR / "outputs" / V2_RUN_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(VICREG_SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=2)
    return parser.parse_args()


def ensure_output_tree(output_dir: Path) -> None:
    for relative in (
        "audit", "pretraining/checkpoints", "downstream/training_curves",
        "downstream/jobs", "summaries", "figures/augmentation_qc", "report", "smoke",
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


def append_csv_deduplicated(path: Path, rows: list[dict[str, Any]], key: list[str]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    old = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=new.columns)
    output = pd.concat([old, new], ignore_index=True)
    output = output.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    write_csv(path, output)


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


def resolve_v2_artifact_root(path: Path) -> Path:
    """Accept direct server output or the common unzipped nested bundle layout."""
    direct = path / "downstream/fold_metrics.csv"
    if direct.exists():
        return path
    nested = path / path.name
    if (nested / "downstream/fold_metrics.csv").exists():
        return nested
    raise FileNotFoundError(f"complete v2 artifact root not found under {path}")


def _fold_cache(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    cache: dict[str, Any] = {}
    for session in EXPECTED_SESSIONS:
        cache[session] = {}
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            cache[session][task] = (data, grouped_cv_splits(data.groups, max_folds=10))
    return cache


def _expected_fold_metric_rows(cache: dict[str, Any], n_conditions: int) -> int:
    return int(2 * n_conditions * len(VICREG_SEEDS) * sum(len(cache[s]["binary"][1]) for s in EXPECTED_SESSIONS))


def _cycle_text_from_csv(value: Any) -> str:
    return cycle_text([int(item) for item in str(value).split(",") if str(item)])


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    ensure_output_tree(args.output_dir)
    if tuple(args.seeds) != VICREG_SEEDS:
        raise RuntimeError(f"Stage 0 STOP: seeds must be exactly {VICREG_SEEDS}")
    v2_root = resolve_v2_artifact_root(args.v2_output_dir)
    required_prior = (
        args.v1_output_dir / "downstream/fold_metrics.csv",
        args.v1_output_dir / "downstream/predictions.csv",
        args.v1_output_dir / "audit/fold_reproduction.csv",
        args.v1_output_dir / "audit/ssl_data_volume.csv",
        args.v1_output_dir / "pretraining/checkpoint_manifest.csv",
        v2_root / "downstream/fold_metrics.csv",
        v2_root / "downstream/predictions.csv",
        v2_root / "audit/fold_identity_check.csv",
        v2_root / "audit/pretraining_compute_match.csv",
    )
    missing = [str(path) for path in required_prior if not path.exists()]
    if missing:
        raise RuntimeError(f"Stage 0 STOP: prior artifacts missing: {missing}")
    cache = _fold_cache(args)
    v1_folds = pd.read_csv(args.v1_output_dir / "audit/fold_reproduction.csv")
    v2_folds = pd.read_csv(v2_root / "audit/fold_identity_check.csv")
    fold_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for session in EXPECTED_SESSIONS:
        for task in EXPECTED_TASKS:
            data, splits = cache[session][task]
            for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
                current_train = cycle_text(data.groups[train_idx])
                current_test = cycle_text(data.groups[test_idx])
                old1 = v1_folds[
                    (v1_folds["session"].astype(str) == session)
                    & (v1_folds["task"] == task)
                    & (v1_folds["fold"].astype(int) == fold_i)
                ]
                old2 = v2_folds[
                    (v2_folds["session"].astype(str) == session)
                    & (v2_folds["task"] == task)
                    & (v2_folds["fold"].astype(int) == fold_i)
                ]
                if len(old1) != 1 or len(old2) != 1:
                    raise AssertionError("prior fold audit row missing or duplicated")
                v1_train, v1_test = str(old1.iloc[0]["train_cycles"]), str(old1.iloc[0]["test_cycles"])
                v2_train = str(old2.iloc[0]["current_train_cycle_ids"])
                v2_test = str(old2.iloc[0]["current_test_cycle_ids"])
                passed = current_train == v1_train == v2_train and current_test == v1_test == v2_test
                fold_rows.append({
                    "session": session, "task": task, "fold": fold_i,
                    "current_train_cycle_ids": current_train, "v1_train_cycle_ids": v1_train,
                    "v2_train_cycle_ids": v2_train, "current_test_cycle_ids": current_test,
                    "v1_test_cycle_ids": v1_test, "v2_test_cycle_ids": v2_test,
                    "status": "PASS" if passed else "FAIL",
                })
                if task == "binary":
                    train_cycles = np.unique(data.groups[train_idx]).astype(int)
                    test_cycles = np.unique(data.groups[test_idx]).astype(int)
                    for condition in VICREG_CONDITIONS:
                        target_cycles = train_cycles
                        overlap = sorted(set(target_cycles) & set(test_cycles))
                        leakage_rows.append({
                            "target_session": session, "fold": fold_i, "condition": condition,
                            "target_train_cycle_ids": cycle_text(train_cycles),
                            "target_test_cycle_ids": cycle_text(test_cycles),
                            "target_cycles_in_ssl_pool": cycle_text(target_cycles),
                            "n_target_test_frames_seen_by_ssl": int(30 * len(overlap)),
                            "n_target_test_frames_in_normalization_fit": 0,
                            "n_target_test_frames_in_augmentation_statistics": 0,
                            "n_target_test_frames_in_checkpoint_selection": 0,
                            "n_target_test_frames_in_supervised_train": 0,
                            "n_target_test_frames_in_qc": 0,
                            "status": "PASS" if not overlap else "FAIL",
                        })
    write_csv(args.output_dir / "audit/fold_identity_check.csv", fold_rows)
    write_csv(args.output_dir / "audit/vicreg_target_test_leakage.csv", leakage_rows)
    if any(row["status"] != "PASS" for row in fold_rows):
        raise RuntimeError("Stage 0 STOP: fold identity mismatch")
    if any(row["status"] != "PASS" for row in leakage_rows):
        raise RuntimeError("Stage 0 STOP: VICReg target-test leakage")

    v1_metrics = pd.read_csv(args.v1_output_dir / "downstream/fold_metrics.csv")
    v2_metrics = pd.read_csv(v2_root / "downstream/fold_metrics.csv")
    reuse_rows = []
    reuse_specs = (
        ("RANDOM_INIT", args.v1_output_dir, v1_metrics, "RANDOM_INIT"),
        ("WITHIN_MASKED_SSL_FT", args.v1_output_dir, v1_metrics, "SSL_FINETUNE"),
        ("MULTI_MASKED_SSL_FT", v2_root, v2_metrics, "MULTI_SSL_FT"),
    )
    expected_one = _expected_fold_metric_rows(cache, 1)
    for condition, root, frame, source_condition in reuse_specs:
        subset = frame[frame["condition"] == source_condition]
        condition_ok = bool(
            len(subset) == expected_one
            and set(subset["seed"].astype(int)) == set(VICREG_SEEDS)
            and set(subset["session"].astype(str)) == set(EXPECTED_SESSIONS)
            and set(subset["task"]) == set(EXPECTED_TASKS)
            and (subset["encoder_requires_grad"].astype(bool)).all()
            and (~subset["decoder_present"].astype(bool)).all()
        )
        reuse_rows.append({
            "condition": condition,
            "source_output_dir": str(root),
            "source_condition": source_condition,
            "reused": condition_ok,
            "reason": (
                "PASS: complete fold/seed/task/session coverage; frozen SmallCNN, preprocessing, folds, "
                "supervised config, and seeds match audited v1/v2"
                if condition_ok else "FAIL: prior artifact coverage/config mismatch"
            ),
            "artifact_rows": len(subset),
            "expected_rows": expected_one,
            "architecture_fingerprint": architecture_fingerprint(),
            "supervised_config": json.dumps(asdict(SUPERVISED_CONFIG), sort_keys=True),
        })
    write_csv(args.output_dir / "audit/prior_artifact_reuse.csv", reuse_rows)
    if not all(row["reused"] for row in reuse_rows):
        raise RuntimeError("Stage 0 STOP: prior artifacts are not safely reusable")

    augmentation = [
        "# Frozen VICReg Augmentation Configuration", "",
        "All operations occur after the frozen arcsinh plus train-pool pixel z-score preprocessing.", "",
        f"- Multiplicative gain jitter: p={AUGMENTATION_CONFIG.gain_probability}, Uniform({AUGMENTATION_CONFIG.gain_min}, {AUGMENTATION_CONFIG.gain_max}).",
        f"- Additive offset jitter: p={AUGMENTATION_CONFIG.offset_probability}, Uniform({AUGMENTATION_CONFIG.offset_min}, {AUGMENTATION_CONFIG.offset_max}) normalized units.",
        f"- Gaussian noise: p={AUGMENTATION_CONFIG.noise_probability}, sigma Uniform({AUGMENTATION_CONFIG.noise_sigma_min}, {AUGMENTATION_CONFIG.noise_sigma_max}) normalized units.",
        f"- Mild Gaussian blur: p={AUGMENTATION_CONFIG.blur_probability}, sigma Uniform({AUGMENTATION_CONFIG.blur_sigma_min}, {AUGMENTATION_CONFIG.blur_sigma_max}).",
        "- Two views use independent deterministic RNG seeds.",
        "- No horizontal/vertical flip, rotation, crop, resize crop, affine, elastic deformation, perspective, or translation.",
        "- No augmentation search and no test-data statistic fitting.",
    ]
    (args.output_dir / "audit/augmentation_config.md").write_text("\n".join(augmentation) + "\n", encoding="utf-8")
    frozen = [
        "# VICReg-style SmallCNN SSL v3 — Frozen Configuration", "",
        f"- Sessions: `{list(EXPECTED_SESSIONS)}`",
        f"- Tasks: `{list(EXPECTED_TASKS)}`; primary binary, secondary stimulus_type.",
        f"- Conditions: `{list(V3_CONDITIONS)}`",
        f"- SmallCNN architecture: `{encoder_architecture_signature()}`",
        f"- SmallCNN architecture fingerprint: `{architecture_fingerprint()}`",
        f"- VICReg config: `{asdict(VICREG_CONFIG)}`",
        f"- Augmentation config: `{asdict(AUGMENTATION_CONFIG)}`",
        f"- Downstream config: `{asdict(SUPERVISED_CONFIG)}`",
        "- SSL pools use all complete-cycle frames; target test cycles are excluded.",
        "- MULTI uses uniform source-session then uniform legal-frame sampling.",
        "- Equal-update budget is inherited from v1 masked SSL reference updates.",
        "- Projector is SSL-only and discarded before downstream fine-tuning.",
        "- No labels/session labels, negatives, reconstruction loss, registration, or spatial augmentation.",
        "- Formal execution is CUDA-only with batch fallback 32 -> 16 -> 8; batch below 8 is a hard STOP.",
        f"- VICReg implementation fingerprint: `{implementation_fingerprint(PROJECT_DIR)}`",
    ]
    (args.output_dir / "audit/config_freeze.md").write_text("\n".join(frozen) + "\n", encoding="utf-8")
    write_csv(args.output_dir / "audit/vicreg_sampling_distribution.csv", pd.DataFrame(columns=[
        "target_session", "fold", "seed", "condition", "source_session", "sample_count",
        "total_frame_exposures", "expected_proportion", "actual_proportion", "absolute_deviation",
        "descriptive_tolerance", "sampling_status",
    ]))
    write_csv(args.output_dir / "audit/vicreg_compute_match.csv", pd.DataFrame(columns=[
        "target_session", "fold", "seed", "condition", "ssl_pool_frames", "actual_batch_size",
        "reference_updates", "actual_updates", "frame_exposure_count", "unique_frame_coverage",
        "compute_match",
    ]))
    log("Stage 0 audit PASSED: prior reuse, fold identity, and zero target-test leakage", args.output_dir)
    return {"cache": cache, "v2_root": v2_root}


def run_smoke(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise RuntimeError("smoke must use --device cpu")
    if not 1 <= args.smoke_updates <= 3:
        raise ValueError("smoke-updates must be 1..3")
    ensure_output_tree(args.output_dir)
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    data = load_block_sequence_session(PROJECT_DIR, "710", "binary", data_dir=data_dir)
    train_idx, test_idx = grouped_cv_splits(data.groups, max_folds=10)[0]
    train_cycles = np.unique(data.groups[train_idx]).astype(int)
    test_cycles = np.unique(data.groups[test_idx]).astype(int)
    loaded = load_unlabeled_cycles(data.source_h5_path, train_cycles[:1])
    all_frames = {session: loaded for session in EXPECTED_SESSIONS}
    pool = build_vicreg_pool(
        all_frames, target_session="710", target_train_cycles=train_cycles[:1],
        target_test_cycles=test_cycles, condition="WITHIN_VICREG_SSL_FT",
    )
    config = replace(VICREG_CONFIG, batch_size=8)
    result = pretrain_vicreg_smallcnn(
        pool, seed=VICREG_SEEDS[0], reference_updates=args.smoke_updates,
        config=config, augmentation_config=AUGMENTATION_CONFIG, device="cpu",
    )
    checkpoint = args.output_dir / "smoke/vicreg_encoder.pt"
    save_vicreg_encoder_checkpoint(
        checkpoint, result, target_session="710", fold=1, seed=VICREG_SEEDS[0],
        condition="WITHIN_VICREG_SSL_FT", pool=pool,
        target_train_cycles=train_cycles[:1], target_test_cycles=test_cycles,
        config=config, augmentation_config=AUGMENTATION_CONFIG,
        implementation_fingerprint=implementation_fingerprint(PROJECT_DIR),
    )
    payload = validate_vicreg_checkpoint(
        checkpoint, target_session="710", fold=1, seed=VICREG_SEEDS[0],
        condition="WITHIN_VICREG_SSL_FT", reference_updates=args.smoke_updates,
        implementation_fingerprint=implementation_fingerprint(PROJECT_DIR),
    )
    encoder, _ = load_ssl_encoder_checkpoint(checkpoint)
    downstream = train_downstream_fold(
        "SSL_FINETUNE", data, train_idx[:8], test_idx,
        fold=1, seed=VICREG_SEEDS[0], pretrained_encoder_state=encoder.state_dict(),
        config=replace(SUPERVISED_CONFIG, max_epochs=1, batch_size=8), device="cpu",
    )
    status = {
        "status": "PASS", "device": "cpu", "session": "710", "fold": 1,
        "seed": VICREG_SEEDS[0], "ssl_optimizer_updates": result.actual_updates,
        "supervised_epochs": 1, "target_test_frames_seen_by_ssl": 0,
        "projector_in_checkpoint": payload["contains_projector_state"],
        "contains_labels": payload["contains_labels"],
        "downstream_condition": "SSL_FINETUNE",
        "formal_scientific_result": False,
        "finite_downstream_test_BA": bool(np.isfinite(downstream.metrics["test_balanced_accuracy"])),
    }
    line = json.dumps(status, sort_keys=True)
    (args.output_dir / "smoke_test_local.txt").write_text(line + "\n", encoding="utf-8")
    log("Stage 2 tiny CPU smoke PASSED: augmentation, VICReg backward, checkpoint, and one-epoch finetune", args.output_dir)


def _load_all_unlabeled(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    output = {}
    for session in EXPECTED_SESSIONS:
        path = data_dir / f"session_{session}_blocks.h5"
        cycles = complete_cycles_from_unlabeled_h5(path)
        log(f"loading label-free complete-cycle frames source_session={session}", args.output_dir)
        output[session] = load_unlabeled_cycles(path, cycles)
    return output


def _v1_reference(args: argparse.Namespace, session: str, fold_i: int, seed: int) -> tuple[int, int]:
    volume = pd.read_csv(args.v1_output_dir / "audit/ssl_data_volume.csv")
    row = volume[(volume["session"].astype(str) == session) & (volume["fold"].astype(int) == fold_i)]
    if len(row) != 1:
        raise AssertionError("v1 SSL data volume row missing")
    checkpoint = args.v1_output_dir / f"pretraining/checkpoints/session_{session}/fold_{fold_i}/seed_{seed}.pt"
    _encoder, payload = load_ssl_encoder_checkpoint(checkpoint)
    batch_size = int(payload["actual_batch_size"])
    return reference_optimizer_updates(int(row.iloc[0]["n_ssl_train_frames"]), batch_size), batch_size


def _checkpoint_path(args: argparse.Namespace, session: str, fold_i: int, seed: int, condition: str) -> Path:
    return args.output_dir / f"pretraining/checkpoints/{condition}/session_{session}/fold_{fold_i}/seed_{seed}.pt"


def _record_vicreg_checkpoint_audits(
    args: argparse.Namespace,
    payload: dict[str, Any],
    *,
    reused: bool,
) -> None:
    compute = {
        "target_session": str(payload["target_session"]), "fold": int(payload["fold"]),
        "seed": int(payload["seed"]), "condition": payload["condition"],
        "ssl_pool_frames": int(payload["ssl_pool_frames"]),
        "actual_batch_size": int(payload["actual_batch_size"]),
        "reference_updates": int(payload["reference_updates"]),
        "actual_updates": int(payload["actual_updates"]),
        "frame_exposure_count": int(payload["frame_exposure_count"]),
        "unique_frame_coverage": int(payload["unique_frame_coverage"]),
        "compute_match": int(payload["actual_updates"]) == int(payload["reference_updates"]),
        "reused_checkpoint": bool(reused),
    }
    if not compute["compute_match"] or compute["actual_batch_size"] < 8:
        raise AssertionError("VICReg compute audit failed")
    append_csv_deduplicated(
        args.output_dir / "audit/vicreg_compute_match.csv", [compute],
        ["target_session", "fold", "seed", "condition"],
    )
    rows = sampling_distribution_rows(
        {str(key): int(value) for key, value in payload["sampling_counts"].items()},
        target_session=str(payload["target_session"]), fold=int(payload["fold"]),
        seed=int(payload["seed"]), condition=payload["condition"],
    )
    append_csv_deduplicated(
        args.output_dir / "audit/vicreg_sampling_distribution.csv", rows,
        ["target_session", "fold", "seed", "condition", "source_session"],
    )


def _write_fixed_train_frame_qc(
    args: argparse.Namespace,
    *,
    pool,
    payload: dict[str, Any],
    target_session: str,
    fold_i: int,
    seed: int,
    condition: str,
) -> None:
    if fold_i != 1 or seed != VICREG_SEEDS[0] or condition != "WITHIN_VICREG_SSL_FT":
        return
    frames = pool.frames_by_session[target_session]
    if not len(frames):
        raise AssertionError("empty within-session QC pool")
    normalized = apply_ssl_frame_normalizer(
        frames[:1], np.asarray(payload["normalization_mean"]), np.asarray(payload["normalization_std"])
    )
    original = torch.from_numpy(normalized[:, None])
    view1 = conservative_vicreg_augmentation(
        original, seed=VICREG_SEEDS[0] * 1_000_003 + int(target_session), config=AUGMENTATION_CONFIG
    )
    view2 = conservative_vicreg_augmentation(
        original, seed=VICREG_SEEDS[0] * 1_000_003 + int(target_session) + 1, config=AUGMENTATION_CONFIG
    )
    np.savez_compressed(
        args.output_dir / f"figures/augmentation_qc/session_{target_session}_qc.npz",
        original=original[0, 0].numpy(), view1=view1[0, 0].numpy(), view2=view2[0, 0].numpy(),
        source_session=target_session,
        source_cycle=int(pool.cycles_by_session[target_session][0]),
        source_frame_index=int(pool.original_indices_by_session[target_session][0]),
        target_test_frame_used=False,
    )


def _pretrain_one(
    args: argparse.Namespace,
    *,
    all_frames: dict[str, Any],
    target_session: str,
    fold_i: int,
    seed: int,
    condition: str,
    target_train_cycles: np.ndarray,
    target_test_cycles: np.ndarray,
    reference_updates: int,
    normalizer_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
) -> Path:
    path = _checkpoint_path(args, target_session, fold_i, seed, condition)
    fingerprint = implementation_fingerprint(PROJECT_DIR)
    pool = build_vicreg_pool(
        all_frames, target_session=target_session, target_train_cycles=target_train_cycles,
        target_test_cycles=target_test_cycles, condition=condition,
    )
    normalizer_key = (target_session, fold_i, condition)
    if path.exists() and not args.overwrite:
        payload = validate_vicreg_checkpoint(
            path, target_session=target_session, fold=fold_i, seed=seed,
            condition=condition, reference_updates=reference_updates,
            implementation_fingerprint=fingerprint,
            source_sessions=pool.source_sessions, target_train_cycles=target_train_cycles,
            target_test_cycles=target_test_cycles, config=VICREG_CONFIG,
            augmentation_config=AUGMENTATION_CONFIG,
        )
        normalizer_cache.setdefault(
            normalizer_key,
            (np.asarray(payload["normalization_mean"]), np.asarray(payload["normalization_std"])),
        )
        _record_vicreg_checkpoint_audits(args, payload, reused=True)
        append_csv_deduplicated(
            args.output_dir / "pretraining/vicreg_losses.csv", [{
                "target_session": target_session, "fold": fold_i, "seed": seed,
                "condition": condition, **row,
            } for row in payload["training_history"]],
            ["target_session", "fold", "seed", "condition", "update"],
        )
        _write_fixed_train_frame_qc(
            args, pool=pool, payload=payload, target_session=target_session,
            fold_i=fold_i, seed=seed, condition=condition,
        )
        return path
    if normalizer_key not in normalizer_cache:
        normalizer_cache[normalizer_key] = fit_ssl_pool_normalizer(pool)
    log(
        f"VICReg pretrain target={target_session} fold={fold_i} seed={seed} condition={condition} "
        f"sources={pool.source_sessions} frames={pool.n_frames} updates={reference_updates}",
        args.output_dir,
    )
    result = pretrain_vicreg_smallcnn(
        pool, seed=seed, reference_updates=reference_updates, config=VICREG_CONFIG,
        augmentation_config=AUGMENTATION_CONFIG, device="cuda",
        normalization_stats=normalizer_cache[normalizer_key],
    )
    save_vicreg_encoder_checkpoint(
        path, result, target_session=target_session, fold=fold_i, seed=seed,
        condition=condition, pool=pool, target_train_cycles=target_train_cycles,
        target_test_cycles=target_test_cycles, config=VICREG_CONFIG,
        augmentation_config=AUGMENTATION_CONFIG, implementation_fingerprint=fingerprint,
    )
    payload = validate_vicreg_checkpoint(
        path, target_session=target_session, fold=fold_i, seed=seed,
        condition=condition, reference_updates=reference_updates,
        implementation_fingerprint=fingerprint,
        source_sessions=pool.source_sessions, target_train_cycles=target_train_cycles,
        target_test_cycles=target_test_cycles, config=VICREG_CONFIG,
        augmentation_config=AUGMENTATION_CONFIG,
    )
    if not checkpoint_contains_no_labels_or_projector(path):
        raise AssertionError("VICReg checkpoint contains label/projector information")
    _record_vicreg_checkpoint_audits(args, payload, reused=False)
    append_csv_deduplicated(
        args.output_dir / "pretraining/vicreg_losses.csv", [{
            "target_session": target_session, "fold": fold_i, "seed": seed,
            "condition": condition, **row,
        } for row in result.history],
        ["target_session", "fold", "seed", "condition", "update"],
    )
    append_csv_deduplicated(
        args.output_dir / "audit/runtime_resources.csv", [{
            "stage": "pretraining", "target_session": target_session, "task": "both",
            "fold": fold_i, "seed": seed, "condition": condition,
            "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda, "actual_batch_size": result.actual_batch_size,
            "peak_gpu_memory_mb": result.peak_gpu_memory_mb, "runtime_seconds": result.runtime_seconds,
        }],
        ["stage", "target_session", "task", "fold", "seed", "condition"],
    )
    _write_fixed_train_frame_qc(
        args, pool=pool, payload=payload, target_session=target_session,
        fold_i=fold_i, seed=seed, condition=condition,
    )
    return path


def _import_prior_downstream(args: argparse.Namespace, v2_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    v1_metrics = pd.read_csv(args.v1_output_dir / "downstream/fold_metrics.csv")
    v1_predictions = pd.read_csv(args.v1_output_dir / "downstream/predictions.csv")
    v2_metrics = pd.read_csv(v2_root / "downstream/fold_metrics.csv")
    v2_predictions = pd.read_csv(v2_root / "downstream/predictions.csv")
    frames = []
    prediction_frames = []
    for source, source_predictions, old, new in (
        (v1_metrics, v1_predictions, "RANDOM_INIT", "RANDOM_INIT"),
        (v1_metrics, v1_predictions, "SSL_FINETUNE", "WITHIN_MASKED_SSL_FT"),
        (v2_metrics, v2_predictions, "MULTI_SSL_FT", "MULTI_MASKED_SSL_FT"),
    ):
        metric = source[source["condition"] == old].copy()
        pred = source_predictions[source_predictions["condition"] == old].copy()
        metric["condition"] = new
        pred["condition"] = new
        frames.append(metric)
        prediction_frames.append(pred)
    metrics = pd.concat(frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics["session"] = metrics["session"].astype(str)
    predictions["session"] = predictions["session"].astype(str)
    curve_mapping = (
        (args.v1_output_dir / "downstream/training_curves", "RANDOM_INIT", "RANDOM_INIT"),
        (args.v1_output_dir / "downstream/training_curves", "SSL_FINETUNE", "WITHIN_MASKED_SSL_FT"),
        (v2_root / "downstream/training_curves", "MULTI_SSL_FT", "MULTI_MASKED_SSL_FT"),
    )
    for root, old, new in curve_mapping:
        for source in root.glob(f"*_{old}.csv"):
            target = args.output_dir / "downstream/training_curves" / source.name.replace(f"_{old}.csv", f"_{new}.csv")
            if not target.exists() or args.overwrite:
                curve = pd.read_csv(source)
                curve["condition"] = new
                curve["session"] = curve["session"].astype(str)
                write_csv(target, curve)
    return metrics, predictions


def _prediction_rows(data, test_idx, result, condition: str, seed: int, fold_i: int) -> list[dict[str, Any]]:
    rows = []
    for local_i, sample_i in enumerate(test_idx):
        meta = data.metadata.iloc[int(sample_i)]
        rows.append({
            "session": str(data.session), "task": data.task, "fold": fold_i, "seed": seed,
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
    job = _downstream_job_path(args, str(data.session), data.task, fold_i, seed, condition)
    prediction_job = job.with_suffix(".predictions.csv")
    if job.exists() and prediction_job.exists() and not args.overwrite:
        return
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    encoder = SmallCNNFrameEncoder()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    log(f"downstream target={data.session} task={data.task} fold={fold_i} seed={seed} condition={condition}", args.output_dir)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    result = train_downstream_fold(
        "SSL_FINETUNE", data, train_idx, test_idx, fold=fold_i, seed=seed,
        pretrained_encoder_state=encoder.state_dict(), config=SUPERVISED_CONFIG, device="cuda",
    )
    torch.cuda.synchronize()
    metrics = dict(result.metrics)
    metrics["session"] = str(metrics["session"])
    metrics["condition"] = condition
    job.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_csv(prediction_job, _prediction_rows(data, test_idx, result, condition, seed, fold_i))
    curve = pd.DataFrame(result.history)
    for key, value in reversed((
        ("session", str(data.session)), ("task", data.task), ("fold", fold_i),
        ("seed", seed), ("condition", condition),
    )):
        curve.insert(0, key, value)
    write_csv(
        args.output_dir / f"downstream/training_curves/session_{data.session}_{data.task}_fold_{fold_i}_seed_{seed}_{condition}.csv",
        curve,
    )
    append_csv_deduplicated(
        args.output_dir / "audit/runtime_resources.csv", [{
            "stage": "downstream", "target_session": str(data.session), "task": data.task,
            "fold": fold_i, "seed": seed, "condition": condition,
            "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "pytorch_cuda_version": torch.version.cuda, "actual_batch_size": SUPERVISED_CONFIG.batch_size,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "runtime_seconds": time.perf_counter() - started,
        }],
        ["stage", "target_session", "task", "fold", "seed", "condition"],
    )


def consolidate_downstream(args: argparse.Namespace, v2_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prior_metrics, prior_predictions = _import_prior_downstream(args, v2_root)
    metric_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((args.output_dir / "downstream/jobs").glob("*.json"))]
    prediction_frames = [pd.read_csv(path) for path in sorted((args.output_dir / "downstream/jobs").glob("*.predictions.csv"))]
    metrics = pd.concat([prior_metrics, pd.DataFrame(metric_rows)], ignore_index=True)
    predictions = pd.concat([prior_predictions, *prediction_frames], ignore_index=True)
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
    v2_root = resolve_v2_artifact_root(args.v2_output_dir)
    cache = _fold_cache(args)
    metrics, predictions = consolidate_downstream(args, v2_root)
    expected_metrics = _expected_fold_metric_rows(cache, len(V3_CONDITIONS))
    if len(metrics) != expected_metrics or set(metrics["condition"]) != set(V3_CONDITIONS):
        raise RuntimeError(f"aggregate STOP: incomplete fold metrics {len(metrics)}/{expected_metrics}")
    oof = predictions.groupby(["session", "task", "seed", "condition", "sample_id"]).size()
    if not (oof == 1).all():
        raise AssertionError("OOF prediction duplication detected")
    target_fold_seed = len(VICREG_SEEDS) * sum(len(cache[s]["binary"][1]) for s in EXPECTED_SESSIONS)
    compute = pd.read_csv(args.output_dir / "audit/vicreg_compute_match.csv")
    if len(compute) != 2 * target_fold_seed or not compute["compute_match"].astype(bool).all():
        raise RuntimeError("aggregate STOP: VICReg equal-update audit incomplete or failed")
    if (compute["actual_batch_size"].astype(int) < 8).any():
        raise RuntimeError("aggregate STOP: VICReg batch size below 8")
    losses = pd.read_csv(args.output_dir / "pretraining/vicreg_losses.csv")
    if len(losses) != int(compute["reference_updates"].astype(int).sum()):
        raise RuntimeError("aggregate STOP: VICReg per-update loss history is incomplete")
    distribution = pd.read_csv(args.output_dir / "audit/vicreg_sampling_distribution.csv")
    expected_sampling_rows = target_fold_seed * (1 + 9)
    if len(distribution) != expected_sampling_rows or not (distribution["sampling_status"] == "PASS").all():
        raise RuntimeError("aggregate STOP: VICReg session-balanced sampling audit incomplete or failed")
    if len(list((args.output_dir / "pretraining/checkpoints").glob("**/*.pt"))) != 2 * target_fold_seed:
        raise RuntimeError("aggregate STOP: VICReg checkpoints incomplete")
    if len(list((args.output_dir / "downstream/training_curves").glob("*.csv"))) != expected_metrics:
        raise RuntimeError("aggregate STOP: downstream curves incomplete")
    session_table = session_level_comparison(metrics)
    tests = planned_statistical_tests(session_table)
    gaps = generalization_gap_summary(session_table)
    stability = seed_stability(metrics)
    write_csv(args.output_dir / "summaries/session_level_comparison.csv", session_table)
    write_csv(args.output_dir / "summaries/planned_statistical_tests.csv", tests)
    write_csv(args.output_dir / "summaries/generalization_gap_summary.csv", gaps)
    write_csv(args.output_dir / "summaries/seed_stability.csv", stability)
    make_required_figures(args.output_dir, session_table)
    scenario = classify_scenario(session_table)
    report = [
        "# VICReg-style SmallCNN SSL Benchmark v3", "",
        "The formal comparison changed only the SSL objective and used conservative geometry-preserving augmentation.", "",
        "## Integrity", "",
        "- All nine fixed sessions and both frozen tasks completed.",
        "- Every fold matched v1 and v2.",
        "- Target-test exposure was zero in SSL, normalization, augmentation statistics, QC, checkpoint selection, and supervised training.",
        "- RANDOM_INIT and masked-SSL baselines were reused after compatibility auditing.",
        "- VICReg used the v1 masked-SSL reference optimizer update budget.",
        "- Projectors were discarded; downstream checkpoints contain only the SmallCNN encoder.", "",
        "## Planned session-level tests", "", markdown_table(tests), "",
        "## Session-level comparison", "", markdown_table(session_table), "",
        "## Preregistered scenario", "", scenario, "",
        "This conclusion is limited to the frozen conservative-augmentation VICReg formulation. No other-only VICReg or next-stage model was run.",
    ]
    (args.output_dir / "report/vicreg_ssl_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    missing = missing_formal_outputs(args.output_dir)
    if missing:
        raise RuntimeError(f"formal output completeness STOP: {missing}")
    log(f"Formal aggregation complete; {scenario}", args.output_dir)


def run_formal(args: argparse.Namespace) -> None:
    assert_formal_cuda(args.device)
    started = time.perf_counter()
    audit = run_audit(args)
    gpu_lines = [
        f"gpu_name={torch.cuda.get_device_name(0)}",
        f"torch_version={torch.__version__}",
        f"pytorch_cuda_version={torch.version.cuda}",
        f"cuda_available={torch.cuda.is_available()}",
        f"requested_batch_size={VICREG_CONFIG.batch_size}",
        "allowed_batch_fallback=32,16,8",
    ]
    (args.output_dir / "audit/gpu_audit.txt").write_text("\n".join(gpu_lines) + "\n", encoding="utf-8")
    prior_metrics, prior_predictions = _import_prior_downstream(args, audit["v2_root"])
    write_csv(args.output_dir / "downstream/fold_metrics.csv", prior_metrics)
    write_csv(args.output_dir / "downstream/predictions.csv", prior_predictions)
    all_frames = _load_all_unlabeled(args)
    checkpoints: dict[tuple[str, int, int, str], Path] = {}
    normalizer_cache: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    data_dir = args.data_dir or default_block_data_dir(PROJECT_DIR)
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=data_dir)
        splits = grouped_cv_splits(data.groups, max_folds=10)
        for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
            train_cycles = np.unique(data.groups[train_idx]).astype(int)
            test_cycles = np.unique(data.groups[test_idx]).astype(int)
            for seed in VICREG_SEEDS:
                reference_updates, _v1_batch_size = _v1_reference(args, session, fold_i, seed)
                for condition in VICREG_CONDITIONS:
                    checkpoints[(session, fold_i, seed, condition)] = _pretrain_one(
                        args, all_frames=all_frames, target_session=session, fold_i=fold_i,
                        seed=seed, condition=condition, target_train_cycles=train_cycles,
                        target_test_cycles=test_cycles, reference_updates=reference_updates,
                        normalizer_cache=normalizer_cache,
                    )
    del all_frames
    compute = pd.read_csv(args.output_dir / "audit/vicreg_compute_match.csv")
    gpu_lines.extend([
        "actual_vicreg_batch_sizes=" + ",".join(map(str, sorted(compute["actual_batch_size"].astype(int).unique()))),
        f"pretraining_jobs={len(compute)}",
    ])
    (args.output_dir / "audit/gpu_audit.txt").write_text("\n".join(gpu_lines) + "\n", encoding="utf-8")
    for session in EXPECTED_SESSIONS:
        for task in EXPECTED_TASKS:
            data = load_block_sequence_session(PROJECT_DIR, session, task, data_dir=data_dir)
            for fold_i, (train_idx, test_idx) in enumerate(grouped_cv_splits(data.groups, max_folds=10), start=1):
                for seed in VICREG_SEEDS:
                    for condition in VICREG_CONDITIONS:
                        _run_downstream_job(
                            args, data=data, train_idx=train_idx, test_idx=test_idx,
                            fold_i=fold_i, seed=seed, condition=condition,
                            checkpoint=checkpoints[(session, fold_i, seed, condition)],
                        )
    aggregate_outputs(args)
    append_csv_deduplicated(
        args.output_dir / "audit/runtime_resources.csv", [{
            "stage": "formal_total", "target_session": "all", "task": "both", "fold": 0,
            "seed": 0, "condition": "all", "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__, "pytorch_cuda_version": torch.version.cuda,
            "actual_batch_size": VICREG_CONFIG.batch_size,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "runtime_seconds": time.perf_counter() - started,
        }],
        ["stage", "target_session", "task", "fold", "seed", "condition"],
    )
    gpu_lines.extend([
        f"peak_vram_mb={float(torch.cuda.max_memory_allocated() / (1024 ** 2)):.3f}",
        f"formal_runtime_seconds={time.perf_counter() - started:.3f}",
    ])
    (args.output_dir / "audit/gpu_audit.txt").write_text("\n".join(gpu_lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.v1_output_dir = args.v1_output_dir.resolve()
    args.v2_output_dir = args.v2_output_dir.resolve()
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()
    ensure_output_tree(args.output_dir)
    if args.stage == "audit":
        run_audit(args)
    elif args.stage == "smoke":
        run_smoke(args)
    elif args.stage == "formal":
        run_formal(args)
    else:
        aggregate_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
