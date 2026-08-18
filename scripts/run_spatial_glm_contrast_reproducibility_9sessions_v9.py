#!/usr/bin/env python3
"""Run the preregistered nine-session blockwise spatial GLM analysis v9."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.spatial_glm_reproducibility_reporting_v9 import make_all_figures, make_report
from ultrasound_decoding.spatial_glm_reproducibility_v9 import (
    CONDITION_ORDER,
    CONTRAST_WEIGHTS,
    EXPECTED_SESSIONS,
    FDR_ALPHA,
    FIXED_ORDER_WARNING,
    FIXED_ORIENTATIONS,
    FORMAL_DEVICE,
    GD_WARNING,
    N_BOOTSTRAP,
    N_SPLITS,
    RUN_NAME,
    V9_RANDOM_SEED,
    SessionAnalysis,
    analyze_session,
    build_diagnostic_table,
    clean4_identity_row,
    load_session_block_images,
    load_v8_metrics,
    load_within_session_ba,
    missing_outputs,
    planned_ba_associations,
    v8_v9_stability_association,
)


DEFAULT_CONFIG = PROJECT_DIR / "configs/spatial_glm_contrast_reproducibility_9sessions_v9.json"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / RUN_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "processed_data/block_sequences_v1")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--v7-output-dir", type=Path,
        default=PROJECT_DIR / "outputs/cross_session_feature_factor_analysis_9sessions_v7",
    )
    parser.add_argument(
        "--v8-output-dir", type=Path,
        default=PROJECT_DIR / "outputs/session_centered_stimulus_vector_alignment_9sessions_v8",
    )
    parser.add_argument("--smoke-cycles", type=int, default=3)
    return parser.parse_args()


def write_csv(path: Path, value: pd.DataFrame | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    frame.to_csv(path, index=False)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def ensure_tree(output_dir: Path) -> None:
    relatives = (
        "audit", "glm/contrast_maps", "glm/standard_error_maps", "glm/standardized_maps", "glm/t_maps", "glm/p_maps",
        "glm/q_maps", "glm/fdr_masks", "reproducibility", "summaries",
        "figures/primary_binary_maps", "figures/GS_maps", "figures/DS_maps",
        "figures/GS_DS_comparison", "figures/exploratory_GD_maps", "report", "smoke",
    )
    for relative in relatives:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def log(message: str, output_dir: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with (output_dir / "run_log_server.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def server_command() -> str:
    return (
        "python scripts/run_spatial_glm_contrast_reproducibility_9sessions_v9.py "
        "--stage formal --device cpu "
        "--config configs/spatial_glm_contrast_reproducibility_9sessions_v9.json "
        "--data-dir processed_data/block_sequences_v1 "
        "--output-dir outputs/spatial_glm_contrast_reproducibility_9sessions_v9 "
        "--v7-output-dir outputs/cross_session_feature_factor_analysis_9sessions_v7 "
        "--v8-output-dir outputs/session_centered_stimulus_vector_alignment_9sessions_v8"
    )


def load_and_validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "run_name": RUN_NAME,
        "sessions": list(EXPECTED_SESSIONS),
        "condition_order": list(CONDITION_ORDER),
        "contrast_weights": {name: weights.tolist() for name, weights in CONTRAST_WEIGHTS.items()},
        "formal_device": FORMAL_DEVICE,
        "n_splits": N_SPLITS,
        "n_bootstrap": N_BOOTSTRAP,
        "fdr_alpha": FDR_ALPHA,
        "fixed_orientations": FIXED_ORIENTATIONS,
        "block_frames": 4,
        "registration": False,
        "hrf_search": False,
        "decoder_training": False,
        "roi_selection": False,
        "searchlight": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise AssertionError(f"config freeze mismatch for {key}: {config.get(key)!r} != {value!r}")
    return config


def config_freeze_text(config_path: Path, v7_root: Path, v8_root: Path) -> str:
    weights = ", ".join(f"{name}={value.tolist()}" for name, value in CONTRAST_WEIGHTS.items())
    return f"""# v9 configuration freeze

- Config: {config_path}
- Sessions: {', '.join(EXPECTED_SESSIONS)}; none excluded
- Conditions/order: {' -> '.join(CONDITION_ORDER)}
- Contrasts: {weights}
- Unit: complete cycle; four block observations per cycle; four clean4 frames averaged within each block
- Preprocessing: finite float32 clean4 -> arcsinh per frame -> pixelwise mean within block
- Orientation: {FIXED_ORIENTATIONS}; session 807 uses the fixed confirmed flip_vertical
- Model: intercept + cycle fixed effects + categorical condition effect, separately per session and pixel
- Formal split-half iterations: {N_SPLITS}; complete-cycle split only
- Formal GS/DS bootstrap iterations: {N_BOOTSTRAP}; paired complete-cycle bootstrap only
- Pixel multiplicity: Benjamini-Hochberg independently within each session x contrast, q={FDR_ALPHA}
- Figure scaling: global 99th percentile of absolute map values, shared across sessions for each displayed map family
- Association: exact 9! Spearman label permutation; Holm family contains exactly the three planned BA associations
- v7 root: {v7_root}
- v8 root: {v8_root}
- No HRF parameter search was performed in v9.
- Registration=False; decoder_training=False; ROI_selection=False; searchlight=False; post_hoc_threshold_search=False
"""


def fixed_order_text() -> str:
    return f"""# Fixed within-cycle order confound

Observed order: **{' -> '.join(CONDITION_ORDER)}**.

{FIXED_ORDER_WARNING}

The primary and secondary maps are described as condition-associated or stimulus-presence-associated spatial contrasts, never as pure activation.

The grating-minus-dot contrast carries this label in every output: **{GD_WARNING}**
"""


def save_maps(analyses: Mapping[str, SessionAnalysis], output_dir: Path) -> None:
    destinations = {
        "contrast_maps": "effect",
        "standard_error_maps": "standard_error",
        "standardized_maps": "standardized",
        "t_maps": "t_map",
        "p_maps": "p_map",
        "q_maps": "q_map",
        "fdr_masks": "fdr_mask",
    }
    for session, analysis in analyses.items():
        for contrast_name, maps in analysis.contrasts.items():
            for directory, attribute in destinations.items():
                value = getattr(maps, attribute)
                dtype = np.uint8 if attribute == "fdr_mask" else np.float32
                np.save(
                    output_dir / f"glm/{directory}/session_{session}_{contrast_name}.npy",
                    np.asarray(value, dtype=dtype),
                    allow_pickle=False,
                )


def run_core(
    *,
    sessions: Sequence[str],
    project_dir: Path,
    data_dir: Path,
    v7_output_dir: Path,
    v8_output_dir: Path,
    output_dir: Path,
    max_cycles: int | None,
    n_splits: int,
    n_bootstrap: int,
    formal: bool,
    config_path: Path,
) -> None:
    ensure_tree(output_dir)
    session_data = {}
    clean_audit = []
    for session in sessions:
        log(f"Loading frozen clean4 block images for session {session}", output_dir)
        data = load_session_block_images(project_dir, data_dir, session, max_cycles=max_cycles)
        session_data[str(session)] = data
        clean_audit.append(clean4_identity_row(data))
    clean_frame = pd.DataFrame(clean_audit)
    if (clean_frame["status"] != "PASS").any():
        raise AssertionError("clean4 identity audit failed")
    write_csv(output_dir / "audit/clean4_identity_check.csv", clean_frame)

    within_ba_all, ba_audit_all, v7_root = load_within_session_ba(v7_output_dir)
    v8_all, v8_audit_all, v8_root = load_v8_metrics(v8_output_dir)
    selected = {str(value) for value in sessions}
    within_ba = within_ba_all[within_ba_all["session"].isin(selected)].reset_index(drop=True)
    v8_metrics = v8_all[v8_all["session"].isin(selected)].reset_index(drop=True)
    write_csv(output_dir / "audit/within_session_ba_reuse.csv", ba_audit_all)
    write_csv(output_dir / "audit/v8_metric_reuse.csv", v8_audit_all)
    write_text(output_dir / "audit/fixed_order_confounds.md", fixed_order_text())
    write_text(output_dir / "audit/config_freeze.md", config_freeze_text(config_path, v7_root, v8_root))

    analyses: dict[str, SessionAnalysis] = {}
    for session in sessions:
        log(f"Fitting session {session} pixelwise cycle-fixed GLM and complete-cycle reproducibility", output_dir)
        analyses[str(session)] = analyze_session(
            session_data[str(session)],
            n_splits=n_splits,
            n_bootstrap=n_bootstrap,
            seed=V9_RANDOM_SEED,
        )
    save_maps(analyses, output_dir)
    glm_summary = pd.DataFrame([row for analysis in analyses.values() for row in analysis.glm_rows])
    split_metrics = pd.DataFrame([row for analysis in analyses.values() for row in analysis.split_rows])
    concordance = pd.DataFrame([analysis.concordance_row for analysis in analyses.values()])
    bootstrap = pd.DataFrame([analysis.bootstrap_row for analysis in analyses.values()])
    write_csv(output_dir / "glm/pixelwise_glm_summary.csv", glm_summary)
    write_csv(output_dir / "reproducibility/split_half_metrics.csv", split_metrics)
    write_csv(output_dir / "reproducibility/gs_ds_concordance.csv", concordance)
    write_csv(output_dir / "reproducibility/bootstrap_metrics.csv", bootstrap)

    fdr_summary = glm_summary[[
        "session", "contrast", "n_valid_pixels", "n_fdr_pixels", "fdr_fraction",
        "mean_abs_effect_FDR_pixels", "fdr_method", "fdr_q",
    ]].copy()
    write_csv(output_dir / "summaries/fdr_summary.csv", fdr_summary)
    diagnostic = build_diagnostic_table(
        glm_summary, split_metrics, concordance, within_ba, v8_metrics, sessions=sessions
    )
    associations = planned_ba_associations(diagnostic)
    stability_association = v8_v9_stability_association(diagnostic)
    write_csv(output_dir / "summaries/session_spatial_diagnostic_table.csv", diagnostic)
    write_csv(output_dir / "summaries/spatial_vs_withinBA_associations.csv", associations)
    write_csv(output_dir / "summaries/v8_vs_v9_stability_association.csv", stability_association)

    log("Generating globally scaled spatial panels and scalar diagnostic figures", output_dir)
    make_all_figures(analyses, diagnostic, associations, output_dir)
    report = make_report(diagnostic, glm_summary, associations, stability_association)
    write_text(output_dir / "report/spatial_glm_reproducibility_report.md", report)

    if not formal:
        schema = pd.DataFrame([
            {"check": "blockwise_glm_solve", "status": "PASS"},
            {"check": "unthresholded_effect_and_standard_error_maps", "status": "PASS"},
            {"check": "BH_FDR", "status": "PASS"},
            {"check": "complete_cycle_split_half", "status": "PASS"},
            {"check": "within_session_GS_DS", "status": "PASS"},
            {"check": "v7_v8_artifact_reuse", "status": "PASS"},
            {"check": "figures_and_report", "status": "PASS"},
            {"check": "scientific_result", "status": "NOT_FORMAL"},
        ])
        write_csv(output_dir / "smoke_schema_check.csv", schema)


def run_smoke(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise RuntimeError("smoke must run on CPU")
    if not 2 <= int(args.smoke_cycles) <= 3:
        raise RuntimeError("smoke is restricted to 2-3 cycles per session")
    load_and_validate_config(args.config)
    ensure_tree(args.output_dir)
    write_text(args.output_dir / "run_command_server.txt", server_command() + "\n")
    smoke_dir = args.output_dir / "smoke"
    started = time.perf_counter()
    run_core(
        sessions=("626", "708"),
        project_dir=PROJECT_DIR,
        data_dir=args.data_dir,
        v7_output_dir=args.v7_output_dir,
        v8_output_dir=args.v8_output_dir,
        output_dir=smoke_dir,
        max_cycles=int(args.smoke_cycles),
        n_splits=7,
        n_bootstrap=7,
        formal=False,
        config_path=args.config,
    )
    write_text(
        smoke_dir / "SMOKE_NOT_SCIENTIFIC.txt",
        "SMOKE PASS: two sessions, three or fewer cycles, seven splits/bootstrap draws. NOT A FORMAL RESULT.\n",
    )
    print(f"SMOKE PASS in {time.perf_counter() - started:.2f}s", flush=True)


def run_formal(args: argparse.Namespace) -> None:
    if args.device != FORMAL_DEVICE:
        raise RuntimeError("v9 formal analysis is frozen to --device cpu")
    load_and_validate_config(args.config)
    ensure_tree(args.output_dir)
    write_text(args.output_dir / "run_command_server.txt", server_command() + "\n")
    started = time.perf_counter()
    log("FORMAL v9 START; server CPU vectorized statistics", args.output_dir)
    run_core(
        sessions=EXPECTED_SESSIONS,
        project_dir=PROJECT_DIR,
        data_dir=args.data_dir,
        v7_output_dir=args.v7_output_dir,
        v8_output_dir=args.v8_output_dir,
        output_dir=args.output_dir,
        max_cycles=None,
        n_splits=N_SPLITS,
        n_bootstrap=N_BOOTSTRAP,
        formal=True,
        config_path=args.config,
    )
    missing = missing_outputs(args.output_dir)
    if missing:
        raise AssertionError(f"formal output completeness failed: {missing}")
    log(f"FORMAL v9 PASS in {time.perf_counter() - started:.2f}s", args.output_dir)


def main() -> None:
    args = parse_args()
    if args.stage == "smoke":
        run_smoke(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
