#!/usr/bin/env python3
"""Run session-centered stimulus-vector alignment mechanism analysis v8."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cross_session_feature_factor_v7 import (
    EXPECTED_SESSIONS,
    GLOBAL_ENCODER_SEEDS,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    exact_spearman_permutation,
)
from ultrasound_decoding.session_centered_vector_reporting_v8 import (
    make_report,
    plot_alignment_transfer,
    plot_cosine_heatmap,
    plot_diagnostic_overview,
    plot_magnitude,
    plot_session_id_before_after,
    plot_stability,
    plot_transductive_probe,
)
from ultrasound_decoding.session_centered_vector_v8 import (
    FORMAL_MODE,
    GD_WARNING,
    N_BOOTSTRAP,
    N_SESSION_PERMUTATIONS,
    N_SPLIT_HALF,
    RUN_NAME,
    TRANSDUCTIVE_WARNING,
    V8_RANDOM_SEED,
    ReusedFeature,
    aggregate_masked_rows,
    centered_feature_frame,
    centroids_long_table,
    descriptive_pattern_scores,
    exact_sign_flip_test,
    load_and_audit_v7_features,
    loso_probe,
    missing_outputs,
    pairwise_transfer_audit,
    pairwise_vector_cosines,
    session_contrast_vectors,
    session_id_before_after,
    vector_alignment_transfer_association,
    vector_stability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--stats-only", action="store_true")
    parser.add_argument("--reuse-v7-features", action="store_true")
    parser.add_argument("--v7-output-dir", type=Path, default=PROJECT_DIR / "outputs/cross_session_feature_factor_analysis_9sessions_v7")
    parser.add_argument("--v5-output-dir", type=Path, default=PROJECT_DIR / "outputs/multisource_loso_smallcnn_9sessions_v5")
    parser.add_argument("--processed-data-dir", type=Path, default=PROJECT_DIR / "processed_data/block_sequences_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--smoke-cycles", type=int, default=3)
    return parser.parse_args()


def ensure_tree(output_dir: Path) -> None:
    for relative in ("audit", "features/session_centered_features", "summaries", "figures", "report", "smoke"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def log(message: str, output_dir: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with (output_dir / "run_log_server.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_csv(path: Path, value: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)).to_csv(path, index=False)


def _representation_filename(artifact: ReusedFeature) -> str:
    if artifact.representation == "RAW_SPATIAL_PCA":
        return "raw_spatial_pca.csv"
    return f"masked_seed_{artifact.seed}.csv"


def _subset_artifact(artifact: ReusedFeature, sessions: list[str], n_cycles: int) -> ReusedFeature:
    metadata = artifact.metadata
    keep = metadata["session"].astype(str).isin(sessions).to_numpy() & (metadata["cycle"].astype(int).to_numpy() < n_cycles)
    return ReusedFeature(
        artifact.representation, artifact.seed, artifact.path,
        metadata.loc[keep].reset_index(drop=True), artifact.feature_columns, artifact.values[keep],
    )


def _aggregate_session_id(seed_rows: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(seed_rows, ignore_index=True)
    numeric = [
        "uncentered_session_BA", "centered_session_BA", "delta_centered_minus_uncentered",
        "session_information_reduction", "uncentered_permutation_p", "centered_permutation_p", "permutation_p",
    ]
    row = {column: float(combined[column].mean()) for column in numeric}
    row.update({f"{column}_seed_std": float(combined[column].std(ddof=1)) for column in numeric[:4]})
    row.update({
        "representation": "GLOBAL_MASKED_SMALLCNN", "seed": "MEAN_3_SEEDS",
        "n_permutations_each": int(combined["n_permutations_each"].iloc[0]),
        "cv_grouping": "cycle_grouped_within_each_session", "centroid_scope": "all_unlabeled_blocks_of_session",
        "analysis_type": "DESCRIPTIVE_TRANSDUCTIVE_MECHANISM",
    })
    return pd.DataFrame([row])


def _run_core(
    artifacts: dict[tuple[str, int | None], ReusedFeature],
    *,
    v7_root: Path,
    v5_root: Path,
    output_dir: Path,
    n_bootstrap: int,
    n_split_half: int,
    n_session_permutations: int,
    session_folds: int,
    probe_max_iter: int,
    formal: bool,
) -> None:
    centroid_table, centered_by_key = centroids_long_table(artifacts)
    write_csv(output_dir / "features/session_centroids.csv", centroid_table)
    for key, artifact in artifacts.items():
        write_csv(output_dir / "features/session_centered_features" / _representation_filename(artifact),
                  centered_feature_frame(artifact, centered_by_key[key]))

    session_id_rows = []
    masked_session_id = []
    for key, artifact in artifacts.items():
        log(f"Session-ID before/after: {key}", output_dir)
        table = session_id_before_after(
            artifact, centered_by_key[key], n_permutations=n_session_permutations,
            n_folds=session_folds, max_iter=probe_max_iter,
        )
        session_id_rows.append(table)
        if artifact.representation == "GLOBAL_MASKED_SMALLCNN":
            masked_session_id.append(table)
    if len(masked_session_id) == 3:
        session_id_rows.append(_aggregate_session_id(masked_session_id))
    session_id_table = pd.concat(session_id_rows, ignore_index=True, sort=False)
    write_csv(output_dir / "summaries/session_id_before_after_centering.csv", session_id_table)

    v7_strict = pd.read_csv(v7_root / "summaries/source_only_stimulus_probe.csv")
    v7_strict = v7_strict[v7_strict["target"].astype(str).isin(EXPECTED_SESSIONS)][["target", "BA"]].rename(
        columns={"BA": "uncentered_BA"}
    )
    transductive_rows = []
    masked_transductive = []
    for key, artifact in artifacts.items():
        centered_probe = loso_probe(artifact.values, artifact.metadata, center_by_session=True).rename(columns={"BA": "centered_BA"})
        if artifact.representation == "RAW_SPATIAL_PCA":
            table = v7_strict.merge(centered_probe[["target", "centered_BA", "accuracy", "macro_F1"]], on="target")
            table["uncentered_protocol"] = "V7_STRICT_SOURCE_ONLY_PCA_LOSO"
        else:
            uncentered = loso_probe(artifact.values, artifact.metadata, center_by_session=False).rename(columns={"BA": "uncentered_BA"})
            table = uncentered[["target", "uncentered_BA"]].merge(
                centered_probe[["target", "centered_BA", "accuracy", "macro_F1"]], on="target"
            )
            table["uncentered_protocol"] = "GLOBAL_LABEL_FREE_COMMON_SPACE_UNCENTERED_CONTROL"
        table["delta"] = table["centered_BA"] - table["uncentered_BA"]
        table["representation"] = artifact.representation
        table["seed"] = "" if artifact.seed is None else artifact.seed
        table["centered_protocol"] = "TRANSDUCTIVE_UNSUPERVISED_CENTERING"
        table["target_centroid_uses_labels"] = False
        table["warning"] = TRANSDUCTIVE_WARNING
        sign_flip = exact_sign_flip_test(table["delta"])
        for name, value in sign_flip.items():
            table[name] = value
        transductive_rows.append(table)
        if artifact.representation == "GLOBAL_MASKED_SMALLCNN":
            masked_transductive.append(table)
    if len(masked_transductive) == 3:
        combined = pd.concat(masked_transductive)
        aggregate = combined.groupby("target", as_index=False).agg(
            uncentered_BA=("uncentered_BA", "mean"), centered_BA=("centered_BA", "mean"),
            delta=("delta", "mean"), uncentered_BA_seed_std=("uncentered_BA", "std"),
            centered_BA_seed_std=("centered_BA", "std"), delta_seed_std=("delta", "std"),
        )
        aggregate["representation"] = "GLOBAL_MASKED_SMALLCNN"
        aggregate["seed"] = "MEAN_3_SEEDS"
        aggregate["uncentered_protocol"] = "GLOBAL_LABEL_FREE_COMMON_SPACE_UNCENTERED_CONTROL"
        aggregate["centered_protocol"] = "TRANSDUCTIVE_UNSUPERVISED_CENTERING"
        aggregate["target_centroid_uses_labels"] = False
        aggregate["warning"] = TRANSDUCTIVE_WARNING
        sign_flip = exact_sign_flip_test(aggregate["delta"])
        for name, value in sign_flip.items():
            aggregate[name] = value
        transductive_rows.append(aggregate)
    transductive_table = pd.concat(transductive_rows, ignore_index=True, sort=False)
    write_csv(output_dir / "summaries/transductive_centering_probe.csv", transductive_table)

    magnitude_tables = []
    stability_tables = []
    cosine_tables = []
    masked_magnitudes = []
    masked_stabilities = []
    masked_cosines = []
    vectors_by_key = {}
    for artifact_i, (key, artifact) in enumerate(artifacts.items()):
        vectors, magnitude = session_contrast_vectors(artifact.values, artifact.metadata)
        magnitude["representation"] = artifact.representation
        magnitude["seed"] = "" if artifact.seed is None else artifact.seed
        magnitude_tables.append(magnitude)
        vectors_by_key[key] = vectors
        stability = pd.DataFrame([
            vector_stability(
                artifact.values, artifact.metadata, session=session,
                n_bootstrap=n_bootstrap, n_split_half=n_split_half,
                seed=V8_RANDOM_SEED + artifact_i * 100 + int(session),
            )
            for session in sorted(artifact.metadata["session"].astype(str).unique(), key=int)
        ])
        stability["representation"] = artifact.representation
        stability["seed"] = "" if artifact.seed is None else artifact.seed
        stability_tables.append(stability)
        cosine = pairwise_vector_cosines(
            vectors, representation=artifact.representation,
            seed="" if artifact.seed is None else artifact.seed,
        )
        cosine_tables.append(cosine)
        if artifact.representation == "GLOBAL_MASKED_SMALLCNN":
            masked_magnitudes.append(magnitude)
            masked_stabilities.append(stability)
            masked_cosines.append(cosine)
    if len(masked_magnitudes) == 3:
        magnitude_tables.append(aggregate_masked_rows(
            masked_magnitudes,
            ["stimulus_vector_norm", "within_condition_dispersion", "normalized_vector_norm"],
            ["session", "task"],
        ))
        stability_tables.append(aggregate_masked_rows(
            masked_stabilities,
            ["bootstrap_vector_stability", "bootstrap_stability_2_5pct", "bootstrap_stability_97_5pct",
             "split_half_vector_stability", "split_half_stability_2_5pct", "split_half_stability_97_5pct"],
            ["session", "task", "n_cycles"],
        ))
        aggregate_cosine = aggregate_masked_rows(masked_cosines, ["cosine_similarity"], ["session_a", "session_b", "task", "strong_pair"])
        aggregate_cosine["exploratory_warning"] = ""
        cosine_tables.append(aggregate_cosine)
    magnitude_table = pd.concat(magnitude_tables, ignore_index=True, sort=False)
    stability_table = pd.concat(stability_tables, ignore_index=True, sort=False)
    cosine_table = pd.concat(cosine_tables, ignore_index=True, sort=False)
    write_csv(output_dir / "summaries/stimulus_vector_magnitude.csv", magnitude_table)
    write_csv(output_dir / "summaries/stimulus_vector_stability.csv", stability_table)
    write_csv(output_dir / "summaries/pairwise_stimulus_vector_cosine.csv", cosine_table)

    pair_audit, pair_transfer = pairwise_transfer_audit(v5_root)
    write_csv(output_dir / "audit/pairwise_crosssession_reuse.csv", pair_audit)
    associations = []
    raw_cosine = cosine_table[cosine_table["representation"] == "RAW_SPATIAL_PCA"]
    masked_cosine = cosine_table[
        (cosine_table["representation"] == "GLOBAL_MASKED_SMALLCNN")
        & (cosine_table["seed"].astype(str) == "MEAN_3_SEEDS")
    ]
    if formal:
        associations.append(vector_alignment_transfer_association(raw_cosine, pair_transfer, representation="RAW_SPATIAL_PCA"))
        associations.append(vector_alignment_transfer_association(masked_cosine, pair_transfer, representation="GLOBAL_MASKED_SMALLCNN"))
    else:
        for representation, cosine in (("RAW_SPATIAL_PCA", raw_cosine), ("GLOBAL_MASKED_SMALLCNN", masked_cosine)):
            available = cosine.merge(pair_transfer, on=["session_a", "session_b"])
            rho = float(pd.Series(available["cosine_similarity"]).corr(pd.Series(available["symmetric_cross_BA"]), method="spearman"))
            associations.append({
                "representation": representation, "analysis": "SMOKE_SCHEMA_ONLY",
                "n_pairs": len(available), "rho": rho, "permutation_p_two_sided": np.nan,
                "permutation_method": "NOT_RUN_IN_SMOKE", "n_permutations": 0,
                "permutation_reason": "three-session smoke only", "interpretation_scope": "NOT_FORMAL",
            })
    association_table = pd.DataFrame(associations)
    write_csv(output_dir / "summaries/vector_alignment_vs_transfer.csv", association_table)

    v7_diagnostic = pd.read_csv(v7_root / "summaries/session_diagnostic_table.csv")
    v7_diagnostic["session"] = v7_diagnostic["session"].astype(str)
    raw_magnitude = magnitude_table[magnitude_table["representation"] == "RAW_SPATIAL_PCA"]
    raw_stability = stability_table[stability_table["representation"] == "RAW_SPATIAL_PCA"]
    raw_mean_cosine = pd.concat([
        raw_cosine[["session_a", "cosine_similarity"]].rename(columns={"session_a": "session"}),
        raw_cosine[["session_b", "cosine_similarity"]].rename(columns={"session_b": "session"}),
    ]).groupby("session", as_index=False)["cosine_similarity"].mean().rename(
        columns={"cosine_similarity": "mean_vector_cosine_to_other_sessions"}
    )
    diagnostic = v7_diagnostic[[
        "session", "within_session_BA", "separability_ratio", "cycle_consistency_mean", "mean_distance_RAW_SPATIAL_PCA"
    ]].rename(columns={
        "separability_ratio": "v7_separability_ratio", "cycle_consistency_mean": "v7_cycle_consistency",
        "mean_distance_RAW_SPATIAL_PCA": "mean_feature_distance_to_other_sessions",
    }).merge(raw_magnitude[["session", "stimulus_vector_norm", "normalized_vector_norm"]], on="session").merge(
        raw_stability[["session", "bootstrap_vector_stability", "split_half_vector_stability"]], on="session"
    ).merge(raw_mean_cosine, on="session")
    diagnostic["historical_group"] = np.where(diagnostic["session"].isin(WEAK_SESSIONS), "historically_weak", "historically_strong")
    diagnostic = descriptive_pattern_scores(diagnostic)
    write_csv(output_dir / "summaries/session_vector_diagnostic_table.csv", diagnostic)

    within_rows = []
    metric_sources = {
        "stimulus_vector_norm": diagnostic["stimulus_vector_norm"],
        "bootstrap_vector_stability": diagnostic["bootstrap_vector_stability"],
        "split_half_vector_stability": diagnostic["split_half_vector_stability"],
    }
    for metric, values in metric_sources.items():
        result = exact_spearman_permutation(values, diagnostic["within_session_BA"])
        within_rows.append({
            "representation": "RAW_SPATIAL_PCA", "metric": metric, "outcome": "within_session_BA",
            "n_sessions": len(diagnostic), "model": "none_spearman_only", **result,
        })
    masked_mag = magnitude_table[
        (magnitude_table["representation"] == "GLOBAL_MASKED_SMALLCNN")
        & (magnitude_table["seed"].astype(str) == "MEAN_3_SEEDS")
    ]
    masked_stab = stability_table[
        (stability_table["representation"] == "GLOBAL_MASKED_SMALLCNN")
        & (stability_table["seed"].astype(str) == "MEAN_3_SEEDS")
    ]
    for metric, values in (
        ("stimulus_vector_norm", masked_mag["stimulus_vector_norm"]),
        ("bootstrap_vector_stability", masked_stab["bootstrap_vector_stability"]),
        ("split_half_vector_stability", masked_stab["split_half_vector_stability"]),
    ):
        result = exact_spearman_permutation(values, diagnostic["within_session_BA"])
        within_rows.append({
            "representation": "GLOBAL_MASKED_SMALLCNN", "metric": metric, "outcome": "within_session_BA",
            "n_sessions": len(diagnostic), "model": "none_spearman_only", **result,
        })
    within_associations = pd.DataFrame(within_rows)
    write_csv(output_dir / "summaries/vector_metrics_vs_within_BA.csv", within_associations)

    gd_rows = []
    for artifact_i, (key, artifact) in enumerate(artifacts.items()):
        vectors, magnitude = session_contrast_vectors(artifact.values, artifact.metadata, task="grating_vs_dot")
        magnitude["row_type"] = "magnitude"
        magnitude["representation"] = artifact.representation
        magnitude["seed"] = "" if artifact.seed is None else artifact.seed
        gd_rows.append(magnitude)
        stability = pd.DataFrame([
            vector_stability(
                artifact.values, artifact.metadata, session=session, task="grating_vs_dot",
                n_bootstrap=n_bootstrap, n_split_half=n_split_half,
                seed=V8_RANDOM_SEED + 5000 + artifact_i * 100 + int(session),
            )
            for session in sorted(artifact.metadata["session"].astype(str).unique(), key=int)
        ])
        stability["row_type"] = "stability"
        stability["representation"] = artifact.representation
        stability["seed"] = "" if artifact.seed is None else artifact.seed
        gd_rows.append(stability)
        pairs = pairwise_vector_cosines(
            vectors, representation=artifact.representation,
            seed="" if artifact.seed is None else artifact.seed, task="grating_vs_dot",
        )
        pairs["row_type"] = "pairwise_cosine"
        gd_rows.append(pairs)
    exploratory = pd.concat(gd_rows, ignore_index=True, sort=False)
    exploratory["exploratory_warning"] = GD_WARNING
    write_csv(output_dir / "summaries/exploratory_GD_vector_alignment.csv", exploratory)

    if formal:
        plot_session_id_before_after(session_id_table, output_dir / "figures/session_id_before_after_centering.png")
        plot_transductive_probe(transductive_table, output_dir / "figures/transductive_centering_crosssession_BA.png")
        plot_cosine_heatmap(raw_cosine, output_dir / "figures/stimulus_vector_cosine_heatmap_raw.png",
                            "RAW_SPATIAL_PCA stimulus-vector alignment")
        plot_cosine_heatmap(masked_cosine, output_dir / "figures/stimulus_vector_cosine_heatmap_masked.png",
                            "GLOBAL_MASKED_SMALLCNN mean stimulus-vector alignment")
        plot_stability(stability_table, output_dir / "figures/stimulus_vector_stability_by_session.png")
        plot_magnitude(magnitude_table, output_dir / "figures/stimulus_vector_magnitude_by_session.png")
        plot_alignment_transfer(cosine_table, pair_transfer, association_table,
                                output_dir / "figures/vector_alignment_vs_crosssession_BA.png")
        plot_diagnostic_overview(diagnostic, output_dir / "figures/session_vector_diagnostic_overview.png")
        report = make_report(
            session_id_table, transductive_table, magnitude_table, stability_table, cosine_table,
            association_table, within_associations, diagnostic,
        )
        (output_dir / "report/session_centered_vector_alignment_report.md").write_text(report, encoding="utf-8")


def config_freeze(v7_root: Path) -> str:
    return f"""# v8 configuration freeze

- FORMAL_MODE = {FORMAL_MODE}
- v7 root resolved to: {v7_root}
- Sessions: {', '.join(EXPECTED_SESSIONS)}; none excluded
- Primary task: stimulus_presence (grating+dot vs stop+static), reused from v7
- Representations: exact v7 RAW_SPATIAL_PCA plus all three GLOBAL_MASKED_SMALLCNN seed feature CSVs
- No PCA refit, encoder training, checkpoint loading, or feature extraction
- Centering: subtract all-unlabeled-block session centroid; descriptive/transductive only
- Session-ID probe: same v7 cycle-grouped C=1 balanced L2 logistic and {N_SESSION_PERMUTATIONS} permutations
- Vector bootstrap: {N_BOOTSTRAP} complete-cycle resamples; split-half: {N_SPLIT_HALF} complete-cycle splits
- Transfer association: exact 9! Mantel-style session-label permutation
- Secondary grating-vs-dot: {GD_WARNING}
- Forbidden: new model training, CSU, DANN, CORAL training, MixStyle, Transformer, Mamba, SSL, registration, ROI, GLM, searchlight
"""


def run_smoke(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise RuntimeError("smoke must run on CPU")
    ensure_tree(args.output_dir)
    v7_root, all_artifacts, audit = load_and_audit_v7_features(
        args.v7_output_dir, processed_data_dir=args.processed_data_dir
    )
    sessions = ["626", "708", "709"]
    artifacts = {key: _subset_artifact(value, sessions, args.smoke_cycles) for key, value in all_artifacts.items()}
    smoke_dir = args.output_dir / "smoke"
    ensure_tree(smoke_dir)
    write_csv(smoke_dir / "audit/v7_feature_reuse.csv", audit)
    _run_core(
        artifacts, v7_root=v7_root, v5_root=args.v5_output_dir, output_dir=smoke_dir,
        n_bootstrap=5, n_split_half=5, n_session_permutations=2, session_folds=3,
        probe_max_iter=80, formal=False,
    )
    schema = pd.DataFrame([
        {"check": "v7_feature_reuse", "status": "PASS"},
        {"check": "centering", "status": "PASS"},
        {"check": "vector_construction", "status": "PASS"},
        {"check": "cosine_matrix", "status": "PASS"},
        {"check": "cycle_split_half", "status": "PASS"},
        {"check": "transductive_probe", "status": "PASS"},
        {"check": "scientific_result", "status": "NOT_FORMAL"},
    ])
    write_csv(smoke_dir / "smoke_schema_check.csv", schema)
    (smoke_dir / "SMOKE_NOT_SCIENTIFIC.txt").write_text(
        "SMOKE PASS: 3 sessions, few cycles, 5 bootstrap/split/permutation controls. NOT A FORMAL RESULT.\n",
        encoding="utf-8",
    )
    print("SMOKE PASS", flush=True)


def run_formal(args: argparse.Namespace) -> None:
    if args.device != "cpu" or not args.stats_only or not args.reuse_v7_features:
        raise RuntimeError("v8 formal reuse mode requires --device cpu --stats-only --reuse-v7-features")
    ensure_tree(args.output_dir)
    started = time.perf_counter()
    log(f"FORMAL v8 START; FORMAL_MODE={FORMAL_MODE}", args.output_dir)
    v7_root, artifacts, audit = load_and_audit_v7_features(
        args.v7_output_dir, processed_data_dir=args.processed_data_dir
    )
    write_csv(args.output_dir / "audit/v7_feature_reuse.csv", audit)
    (args.output_dir / "audit/config_freeze.md").write_text(config_freeze(v7_root), encoding="utf-8")
    (args.output_dir / "audit/gpu_audit.txt").write_text(
        "FORMAL_MODE=CPU_STATS_ONLY\nGPU used=False\nFeature extraction=False\nEncoder training=False\nReason=all v7 features passed exact reuse audit\n",
        encoding="utf-8",
    )
    _run_core(
        artifacts, v7_root=v7_root, v5_root=args.v5_output_dir, output_dir=args.output_dir,
        n_bootstrap=N_BOOTSTRAP, n_split_half=N_SPLIT_HALF,
        n_session_permutations=N_SESSION_PERMUTATIONS, session_folds=5,
        probe_max_iter=400, formal=True,
    )
    elapsed = time.perf_counter() - started
    missing = missing_outputs(args.output_dir)
    allowed_local = {"pytest_output_local.txt", "smoke_test_local.txt", "run_command_server.txt"}
    scientific_missing = [value for value in missing if value not in allowed_local]
    if scientific_missing:
        raise AssertionError(f"formal output completeness failed: {scientific_missing}")
    log(f"FORMAL v8 PASS in {elapsed:.2f}s", args.output_dir)


def main() -> None:
    args = parse_args()
    if args.stage == "smoke":
        run_smoke(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
