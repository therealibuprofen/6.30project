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


PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
for search_path in (PROJECT_DIR, SRC_DIR):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_SESSIONS,
    default_block_data_dir,
    load_block_sequence_session,
)
from ultrasound_decoding.multiframe.within_session_cycle_drift import (
    ANALYSIS_VERSION,
    FLIP_TOLERANCE,
    INTERPRETATION_RULE,
    PRIMARY_BLOCK_TYPES,
    SECONDARY_STIMULUS_BLOCK_TYPES,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    association_analysis,
    build_primary_templates,
    flip_invariance_audit,
    fold_train_test_drift,
    mechanism_interpretation,
    reconstruct_historical_decoder,
    summarize_session_drift,
)


OUTPUT_VERSION = "within_session_cycle_drift_diagnostic_v1"
EXPECTED_CYCLES = 114
EXPECTED_FOLDS = 82
SANITY_SESSIONS = ("708", "807")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Within-session cycle drift diagnostic v1")
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


def parse_cycles(value: Any) -> list[int]:
    text = str(value).strip()
    return [] if not text else [int(item) for item in text.split(",")]


def formal_protocol() -> dict[str, Any]:
    return {
        "output_version": OUTPUT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_type": "mechanism diagnostic exploratory read-only CPU analysis",
        "sessions": list(EXPECTED_SESSIONS),
        "strong_sessions": list(STRONG_SESSIONS),
        "weak_sessions": list(WEAK_SESSIONS),
        "input": "frozen clean4 via existing formal dataset loader",
        "primary_block_types": list(PRIMARY_BLOCK_TYPES),
        "secondary_stimulus_block_types": list(SECONDARY_STIMULUS_BLOCK_TYPES),
        "secondary_stimulus_blocks_enter_primary_metric": False,
        "template": "mean over four arcsinh(clean4) frames",
        "spatial_standardization": "within-image z-score before every spatial correlation",
        "session_block_stability": "median of all different-cycle pair correlations",
        "primary_metric": "background_spatial_drift = 1 - equal mean stop/static spatial stability",
        "primary_target": "historical FCNN late-fusion three-seed mean session OOF BA",
        "fold_reference": "pixelwise median of outer-training-cycle templates only",
        "fold_metric": "1 - equal mean test-cycle x stop/static reference correlation",
        "global_intensity": "secondary only",
        "registration": False,
        "vertical_flip_correction_primary": False,
        "session_807_flip_audit": {
            "operation": "uniform in-memory vertical flip of every nonstimulus template/frame",
            "tolerance": FLIP_TOLERANCE,
            "changes_primary_data": False,
        },
        "statistics": {
            "session": "Pearson, Spearman, exact 9! two-sided Spearman permutation",
            "strong_weak": "exact choose(9,3)=84 two-sided group-label permutation",
            "fold": "pooled descriptive plus within-session Pearson/Spearman",
            "confirmatory": False,
        },
        "mechanism_interpretation_rule": INTERPRETATION_RULE,
        "model_training": False,
        "optimizer": False,
        "checkpoint_creation": False,
        "parameter_search": False,
        "primary_metric_adaptive_switching": False,
        "primary_target_adaptive_switching": False,
    }


def runtime_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    import scipy

    sources = [
        Path(__file__).resolve(),
        args.project_root
        / "src/ultrasound_decoding/multiframe/within_session_cycle_drift.py",
        args.project_root / "src/ultrasound_decoding/multiframe/dataset.py",
        args.project_root / "src/ultrasound_decoding/evaluate.py",
    ]
    return {
        "created_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "git_head": git_value(args.project_root, "rev-parse", "HEAD"),
        "git_branch": git_value(args.project_root, "branch", "--show-current"),
        "source_hashes": {str(path): file_sha256(path) for path in sources},
    }


def load_historical_assets(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    predictions_path = (
        args.historical_reference_dir / "late_fusion_reconstructed_predictions.csv"
    )
    summary_path = args.historical_reference_dir / "session_seed_summary.csv"
    plan_path = args.historical_reference_dir / "task_plan.csv"
    required = [predictions_path, summary_path, plan_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"historical assets missing: {missing}")
    predictions = pd.read_csv(predictions_path, dtype={"session": str})
    saved_summary = pd.read_csv(summary_path, dtype={"session": str})
    plan = pd.read_csv(plan_path, dtype={"session": str})
    formal_session, fold_performance, audit = reconstruct_historical_decoder(
        predictions, saved_summary
    )
    if len(fold_performance) != EXPECTED_FOLDS:
        raise AssertionError("historical fold performance does not contain 82 outer folds")
    audit.update(
        {
            "predictions_path": str(predictions_path),
            "predictions_sha256": file_sha256(predictions_path),
            "saved_summary_path": str(summary_path),
            "saved_summary_sha256": file_sha256(summary_path),
            "task_plan_path": str(plan_path),
            "task_plan_sha256": file_sha256(plan_path),
        }
    )
    return plan, predictions, formal_session, fold_performance, audit


def dataset_assets(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    rows = []
    hashes: dict[str, str] = {}
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(
            args.project_root, session, "binary", data_dir=args.data_dir
        )
        rows.append(
            {
                "session": session,
                "n_cycles": data.n_cycles,
                "n_blocks": data.n_blocks,
                "primary_templates": data.n_cycles * 2,
            }
        )
        hashes[str(data.source_h5_path)] = file_sha256(data.source_h5_path)
        hashes[str(data.source_metadata_path)] = file_sha256(data.source_metadata_path)
    total_cycles = sum(row["n_cycles"] for row in rows)
    if total_cycles != EXPECTED_CYCLES:
        raise AssertionError(f"frozen cycle count {total_cycles} != {EXPECTED_CYCLES}")
    return {"sessions": rows, "total_cycles": total_cycles}, hashes


def build_fold_plan(reference_plan: pd.DataFrame) -> pd.DataFrame:
    plan = reference_plan.copy()
    plan["session"] = plan["session"].astype(str)
    plan["seed"] = plan["seed"].astype(int)
    plan = plan[plan["seed"].eq(0)].copy()
    plan = plan[
        ["session", "fold", "train_cycles", "test_cycles", "n_train_samples", "n_test_samples"]
    ].sort_values(["session", "fold"]).reset_index(drop=True)
    if len(plan) != EXPECTED_FOLDS:
        raise AssertionError("diagnostic fold plan must exactly reuse 82 historical folds")
    return plan


def validate_fold_plan_against_data(args: argparse.Namespace, fold_plan: pd.DataFrame) -> None:
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(
            args.project_root, session, "binary", data_dir=args.data_dir
        )
        all_cycles = set(np.unique(data.groups).astype(int))
        for row in fold_plan[fold_plan["session"].eq(session)].itertuples(index=False):
            train = set(parse_cycles(row.train_cycles))
            test = set(parse_cycles(row.test_cycles))
            if not train or not test or train & test or train | test != all_cycles:
                raise AssertionError("historical fold cycle separation/coverage is invalid")
            if int(row.n_train_samples) != int(np.isin(data.groups, sorted(train)).sum()):
                raise AssertionError("historical fold training block count differs from data")
            if int(row.n_test_samples) != int(np.isin(data.groups, sorted(test)).sum()):
                raise AssertionError("historical fold test block count differs from data")


def write_plan(args: argparse.Namespace) -> pd.DataFrame:
    reference_plan, _predictions, _session, _fold, historical_audit = (
        load_historical_assets(args)
    )
    fold_plan = build_fold_plan(reference_plan)
    validate_fold_plan_against_data(args, fold_plan)
    cycle_audit, source_hashes = dataset_assets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "config.json", formal_protocol())
    atomic_json(args.output_dir / "runtime_fingerprint.json", runtime_fingerprint(args))
    atomic_csv(args.output_dir / "fold_plan.csv", fold_plan)
    atomic_json(
        args.output_dir / "historical_decoder_reconstruction_audit.json",
        historical_audit,
    )
    atomic_json(
        args.output_dir / "provenance_audit.json",
        {
            "status": "PASS",
            "source_file_hashes_before_analysis": source_hashes,
            "cycle_count_audit": cycle_audit,
            "formal_dataset_loader_used": True,
            "raw_data_modified": False,
            "registration_performed": False,
            "model_training_performed": False,
            "optimizer_created": False,
            "checkpoint_created": False,
            "formal_full_diagnostic_started": False,
        },
    )
    atomic_json(
        args.output_dir / "PLAN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "sessions": len(EXPECTED_SESSIONS),
            "cycles": EXPECTED_CYCLES,
            "primary_templates_planned": EXPECTED_CYCLES * 2,
            "outer_folds": EXPECTED_FOLDS,
            "model_trainings": 0,
            "formal_full_diagnostic_started": False,
        },
    )
    print(
        "PLAN PASS: sessions=9 cycles=114 primary_templates=228 folds=82 "
        "historical_reconstruction=PASS model_training=0 full=False",
        flush=True,
    )
    return fold_plan


def load_strict_plan(args: argparse.Namespace) -> pd.DataFrame:
    required = [
        args.output_dir / "config.json",
        args.output_dir / "runtime_fingerprint.json",
        args.output_dir / "fold_plan.csv",
        args.output_dir / "historical_decoder_reconstruction_audit.json",
        args.output_dir / "provenance_audit.json",
        args.output_dir / "PLAN_COMPLETE.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"plan outputs missing; run --stage plan: {missing}")
    reference_plan, _predictions, _session, _fold, historical_audit = (
        load_historical_assets(args)
    )
    current = build_fold_plan(reference_plan)
    saved = pd.read_csv(args.output_dir / "fold_plan.csv", dtype={"session": str})
    pd.testing.assert_frame_equal(saved, current, check_dtype=False)
    saved_historical = json.loads(
        (args.output_dir / "historical_decoder_reconstruction_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if saved_historical != historical_audit:
        raise AssertionError("saved historical reconstruction audit differs")
    if json.loads((args.output_dir / "config.json").read_text()) != formal_protocol():
        raise AssertionError("saved config differs from frozen diagnostic protocol")
    return saved


def analyze_sessions(
    args: argparse.Namespace, sessions: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    template_tables = []
    pair_tables = []
    summaries = []
    bundles = {}
    for session in sessions:
        data = load_block_sequence_session(
            args.project_root, session, "binary", data_dir=args.data_dir
        )
        bundle = build_primary_templates(data)
        pairs, summary = summarize_session_drift(bundle)
        bundles[session] = bundle
        template_tables.append(bundle.metrics)
        pair_tables.append(pairs)
        summaries.append(summary)
    flip = flip_invariance_audit(bundles["807"]) if "807" in bundles else {}
    return (
        pd.concat(template_tables, ignore_index=True),
        pd.concat(pair_tables, ignore_index=True),
        pd.DataFrame(summaries),
        flip,
    )


def verify_source_hashes(args: argparse.Namespace) -> bool:
    provenance = json.loads(
        (args.output_dir / "provenance_audit.json").read_text(encoding="utf-8")
    )
    before = provenance["source_file_hashes_before_analysis"]
    after = {path: file_sha256(Path(path)) for path in before}
    if after != before:
        raise AssertionError("one or more frozen source data files changed")
    return True


def run_sanity(args: argparse.Namespace) -> None:
    load_strict_plan(args)
    templates, pairs, summaries, flip = analyze_sessions(args, SANITY_SESSIONS)
    sanity_dir = args.output_dir / "sanity"
    atomic_csv(sanity_dir / "cycle_template_metrics.csv", templates)
    atomic_csv(sanity_dir / "pairwise_cycle_spatial_correlations.csv", pairs)
    atomic_csv(sanity_dir / "session_cycle_drift_summary.csv", summaries)
    atomic_json(sanity_dir / "807_flip_invariance_audit.json", flip)
    unchanged = verify_source_hashes(args)
    atomic_json(
        args.output_dir / "SANITY_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "sessions": list(SANITY_SESSIONS),
            "cycle_counts": {
                row.session: int(row.n_cycles) for row in summaries.itertuples(index=False)
            },
            "template_rows": int(len(templates)),
            "pairwise_rows": int(len(pairs)),
            "session_metrics": summaries.to_dict("records"),
            "flip_invariance": flip,
            "source_files_unchanged": unchanged,
            "model_trainings": 0,
            "formal_full_diagnostic_started": False,
        },
    )
    print(
        "SANITY PASS sessions=708,807 cycles=18 templates=36 "
        f"flip_max_diff={flip['maximum_absolute_difference']:.3g} "
        "source_unchanged=True model_training=0 full=False",
        flush=True,
    )


def _write_figures(
    output_dir: Path, session_summary: pd.DataFrame, fold_table: pd.DataFrame
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/within_session_cycle_drift_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(
        session_summary["background_spatial_drift"],
        session_summary["formal_session_FCNN_latefusion_BA"],
    )
    for row in session_summary.itertuples(index=False):
        axis.annotate(row.session, (row.background_spatial_drift, row.formal_session_FCNN_latefusion_BA))
    axis.set(xlabel="Background spatial drift", ylabel="Formal FCNN late-fusion BA")
    fig.tight_layout()
    fig.savefig(figure_dir / "session_drift_vs_decoder_BA.png", dpi=180)
    plt.close(fig)

    strong = session_summary[session_summary["session"].isin(STRONG_SESSIONS)][
        "background_spatial_drift"
    ]
    weak = session_summary[session_summary["session"].isin(WEAK_SESSIONS)][
        "background_spatial_drift"
    ]
    fig, axis = plt.subplots(figsize=(5, 5))
    axis.boxplot([strong, weak], tick_labels=["Strong3", "Weak6"])
    axis.set_ylabel("Background spatial drift")
    fig.tight_layout()
    fig.savefig(figure_dir / "strong_vs_weak_cycle_drift.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    for session, group in fold_table.groupby("session"):
        axis.scatter(
            group["fold_train_test_drift"],
            group["fold_FCNN_latefusion_BA_seedavg"],
            label=session,
            s=20,
        )
    axis.set(xlabel="Fold train-test drift", ylabel="Fold FCNN BA (3-seed average)")
    axis.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "fold_drift_vs_fold_BA.png", dpi=180)
    plt.close(fig)


def run_full(args: argparse.Namespace) -> None:
    if not args.review_approved:
        raise RuntimeError("formal full diagnostic requires --review-approved")
    fold_plan = load_strict_plan(args)
    _reference, _predictions, formal_session, fold_performance, historical_audit = (
        load_historical_assets(args)
    )
    templates, pairs, session_summary, flip = analyze_sessions(
        args, tuple(EXPECTED_SESSIONS)
    )
    fold_rows = []
    bundle_cache = {}
    for session in EXPECTED_SESSIONS:
        data = load_block_sequence_session(
            args.project_root, session, "binary", data_dir=args.data_dir
        )
        bundle_cache[session] = build_primary_templates(data)
    for row in fold_plan.itertuples(index=False):
        fold_rows.append(
            fold_train_test_drift(
                bundle_cache[str(row.session)],
                fold=int(row.fold),
                training_cycles=parse_cycles(row.train_cycles),
                test_cycles=parse_cycles(row.test_cycles),
            )
        )
    fold_drift = pd.DataFrame(fold_rows)
    fold_table = fold_drift.merge(
        fold_performance, on=["session", "fold"], validate="one_to_one"
    )
    session_summary = session_summary.merge(
        formal_session, on="session", validate="one_to_one"
    )
    within, association = association_analysis(session_summary, fold_table)
    interpretation = mechanism_interpretation(association)
    source_unchanged = verify_source_hashes(args)
    atomic_csv(args.output_dir / "cycle_template_metrics.csv", templates)
    atomic_csv(args.output_dir / "pairwise_cycle_spatial_correlations.csv", pairs)
    atomic_csv(args.output_dir / "session_cycle_drift_summary.csv", session_summary)
    atomic_csv(args.output_dir / "fold_train_test_drift.csv", fold_drift)
    atomic_csv(args.output_dir / "fold_decoder_performance.csv", fold_table)
    atomic_csv(args.output_dir / "within_session_fold_relationship.csv", within)
    atomic_json(args.output_dir / "807_flip_invariance_audit.json", flip)
    atomic_json(
        args.output_dir / "historical_decoder_reconstruction_audit.json",
        historical_audit,
    )
    atomic_json(args.output_dir / "association_summary.json", association)
    atomic_json(args.output_dir / "mechanism_interpretation.json", interpretation)
    _write_figures(args.output_dir, session_summary, fold_table)
    provenance = json.loads(
        (args.output_dir / "provenance_audit.json").read_text(encoding="utf-8")
    )
    provenance.update(
        {
            "source_files_unchanged_after_analysis": source_unchanged,
            "raw_data_modified": False,
            "registration_performed": False,
            "model_training_performed": False,
            "optimizer_created": False,
            "checkpoint_created": False,
            "formal_full_diagnostic_started": True,
            "formal_full_diagnostic_complete": True,
        }
    )
    atomic_json(args.output_dir / "provenance_audit.json", provenance)
    atomic_json(
        args.output_dir / "RUN_COMPLETE.json",
        {
            "status": "complete",
            "created_utc": utc_now(),
            "sessions": 9,
            "cycles": EXPECTED_CYCLES,
            "templates": EXPECTED_CYCLES * 2,
            "folds": EXPECTED_FOLDS,
            "model_trainings": 0,
            "interpretation": interpretation["interpretation"],
        },
    )
    print(
        f"RUN COMPLETE model_training=0 interpretation={interpretation['interpretation']}",
        flush=True,
    )


def run_status(args: argparse.Namespace) -> None:
    for name in (
        "PLAN_COMPLETE.json",
        "SANITY_COMPLETE.json",
        "RUN_COMPLETE.json",
        "association_summary.json",
    ):
        print(f"{name}: {'present' if (args.output_dir / name).is_file() else 'absent'}")


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
