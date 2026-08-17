#!/usr/bin/env python3
"""Run cross-session feature distribution and factor attribution analysis v7."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".cache/matplotlib"))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cross_session_feature_factor_v7 import (
    CONDITION_TIME_WARNING,
    EXPECTED_SESSIONS,
    GLOBAL_ENCODER_SEEDS,
    N_BOOTSTRAP,
    N_SESSION_PERMUTATIONS,
    RUN_NAME,
    STATISTIC_SEED,
    WEAK_SESSIONS,
    STRONG_SESSIONS,
    add_v7_labels,
    aggregate_masked_distance_rows,
    aggregate_seed_factor_rows,
    assert_formal_cuda,
    audit_pairwise_cross_session,
    audit_v5_cross_session_metrics,
    audit_within_session_metrics,
    bootstrap_factor_decomposition,
    build_global_unlabeled_pool,
    clean4_identity_rows,
    cycle_consistency_from_pool,
    exact_spearman_permutation,
    extract_masked_block_features,
    fit_common_raw_pca,
    find_artifact,
    load_or_train_global_encoder,
    mantel_session_label_permutation,
    metadata_factor_audit,
    missing_formal_outputs,
    pairwise_session_distances,
    sample_metadata_table,
    save_pca_model,
    session_id_probe,
    session_separability,
    source_only_stimulus_probe,
    target_mean_energy_distance,
)
from ultrasound_decoding.cross_session_feature_factor_reporting_v7 import (
    categorical_pca_scatter,
    confusion_heatmap,
    diagnostic_overview,
    distance_heatmap,
    distance_performance_plot,
    factor_variance_plot,
    masked_seed_scatter,
    report_markdown,
    stimulus_probe_plot,
    weak_strong_plot,
)
from ultrasound_decoding.multiframe.dataset import (
    BlockSequenceData,
    default_block_data_dir,
    load_block_sequence_session,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--data-dir", type=Path, default=default_block_data_dir(PROJECT_DIR))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "outputs" / RUN_NAME)
    parser.add_argument("--v5-output-dir", type=Path, default=PROJECT_DIR / "outputs/multisource_loso_smallcnn_9sessions_v5")
    parser.add_argument("--v1-output-dir", type=Path, default=PROJECT_DIR / "outputs/ssl_masked_smallcnn_clean4_9sessions_v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-ssl-steps", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_output_tree(output_dir: Path) -> None:
    for relative in ("audit", "features/masked_smallcnn_features/checkpoints", "summaries", "figures", "report"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def log(message: str, output_dir: Path) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_log_server.txt").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    value.to_csv(path, index=False)


def load_sessions(sessions: list[str], data_dir: Path) -> dict[str, BlockSequenceData]:
    return {
        session: load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=data_dir)
        for session in sessions
    }


def truncate_cycles(data: BlockSequenceData, n_cycles: int) -> BlockSequenceData:
    keep_cycles = np.sort(np.unique(data.groups))[: int(n_cycles)]
    keep = np.isin(data.groups, keep_cycles)
    return replace(
        data,
        X=data.X[keep],
        y=data.y[keep],
        groups=data.groups[keep],
        metadata=data.metadata.loc[keep].reset_index(drop=True),
        clean4_relative_time_s=data.clean4_relative_time_s[keep],
        clean4_original_frame_indices=data.clean4_original_frame_indices[keep],
    )


def _feature_frame(metadata: pd.DataFrame, features: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = {
        f"{prefix}{i + 1}": features[:, i] for i in range(features.shape[1])
    }
    identifiers = metadata[
        ["block_id", "session", "cycle", "cycle_key", "block_name", "condition4", "stimulus_presence"]
    ].reset_index(drop=True)
    return pd.concat([identifiers, pd.DataFrame(columns)], axis=1)


def run_smoke(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise RuntimeError("smoke test must use --device cpu")
    sessions = ["626", "628"]
    smoke_dir = args.output_dir / "smoke"
    ensure_output_tree(smoke_dir)
    started = time.perf_counter()
    data = {session: truncate_cycles(value, 2) for session, value in load_sessions(sessions, args.data_dir).items()}
    pca, raw_features, raw_mean, raw_std, metadata = fit_common_raw_pca(data)
    raw_distances = pairwise_session_distances(
        raw_features, metadata["session"].astype(str).to_numpy(), representation="RAW_SPATIAL_PCA"
    )
    probe, matrix, predictions = session_id_probe(
        raw_features,
        metadata["session"].astype(str).to_numpy(),
        metadata["cycle"].astype(int).to_numpy(),
        n_permutations=2,
        n_folds=2,
        max_iter=80,
    )
    stimulus, leakage = source_only_stimulus_probe(data, max_iter=100)
    raw_binary, _ = bootstrap_factor_decomposition(
        raw_features, metadata, factor_column="stimulus_presence", representation="RAW_SPATIAL_PCA",
        n_bootstrap=3,
    )
    raw_condition, _ = bootstrap_factor_decomposition(
        raw_features, metadata, factor_column="condition4", representation="RAW_SPATIAL_PCA",
        n_bootstrap=3,
    )
    pool = build_global_unlabeled_pool(
        PROJECT_DIR, sessions, data_dir=args.data_dir, max_cycles_per_session=2
    )
    checkpoint = smoke_dir / f"features/masked_smallcnn_features/checkpoints/global_seed_{GLOBAL_ENCODER_SEEDS[0]}.pt"
    if checkpoint.exists() and args.overwrite:
        checkpoint.unlink()
    encoder, payload, reuse = load_or_train_global_encoder(
        checkpoint,
        pool,
        seed=GLOBAL_ENCODER_SEEDS[0],
        updates=int(args.smoke_ssl_steps),
        batch_size=min(2, int(args.batch_size)),
        device="cpu",
    )
    masked = np.concatenate([
        extract_masked_block_features(
            encoder, data[session].X,
            normalizer_mean=payload["normalization_mean"],
            normalizer_std=payload["normalization_std"],
            device="cpu", batch_size=2,
        )
        for session in sessions
    ])
    masked_distances = pairwise_session_distances(
        masked, metadata["session"].astype(str).to_numpy(), representation="GLOBAL_MASKED_SMALLCNN",
        seed=GLOBAL_ENCODER_SEEDS[0],
    )
    masked_binary, _ = bootstrap_factor_decomposition(
        masked, metadata, factor_column="stimulus_presence", representation="GLOBAL_MASKED_SMALLCNN",
        seed_label=GLOBAL_ENCODER_SEEDS[0], n_bootstrap=3,
    )
    cycle_table = cycle_consistency_from_pool(pool, normalizer_mean=raw_mean, normalizer_std=raw_std)
    write_csv(smoke_dir / "features/raw_pca_common_features.csv", _feature_frame(metadata, raw_features, "raw_PC"))
    write_csv(smoke_dir / "features/masked_smallcnn_features/seed_20260812.csv", _feature_frame(metadata, masked, "masked_F"))
    write_csv(smoke_dir / "summaries/pairwise_session_distances.csv", pd.concat([raw_distances, masked_distances]))
    write_csv(smoke_dir / "summaries/session_id_probe.csv", probe)
    write_csv(smoke_dir / "summaries/source_only_stimulus_probe.csv", stimulus)
    write_csv(smoke_dir / "summaries/factor_variance_binary.csv", pd.concat([raw_binary, masked_binary]))
    write_csv(smoke_dir / "summaries/factor_variance_condition4.csv", raw_condition)
    write_csv(smoke_dir / "summaries/cycle_consistency.csv", cycle_table)
    write_csv(smoke_dir / "audit/predictive_pca_leakage.csv", leakage)
    write_csv(smoke_dir / "audit/global_encoder_reuse.csv", [reuse])
    write_csv(smoke_dir / "audit/session_probe_predictions.csv", predictions)
    np.savetxt(smoke_dir / "audit/session_probe_confusion.csv", matrix, delimiter=",", fmt="%d")
    schema = pd.DataFrame([
        {"check": "feature_extraction", "status": "PASS", "detail": f"raw={raw_features.shape}; masked={masked.shape}"},
        {"check": "PCA", "status": "PASS", "detail": f"components={pca.n_components_}"},
        {"check": "energy_distance", "status": "PASS", "detail": f"rows={len(raw_distances)}"},
        {"check": "session_probe", "status": "PASS", "detail": "cycle-grouped 2-fold"},
        {"check": "source_only_probe", "status": "PASS", "detail": "target excluded from normalization/PCA/scaling"},
        {"check": "factor_decomposition", "status": "PASS", "detail": "binary and condition4 schemas"},
        {"check": "global_masked_encoder", "status": "PASS", "detail": f"one seed, {args.smoke_ssl_steps} step(s)"},
        {"check": "scientific_result", "status": "NOT_FORMAL", "detail": "2 sessions x 2 cycles; never report scientifically"},
    ])
    write_csv(smoke_dir / "smoke_schema_check.csv", schema)
    elapsed = time.perf_counter() - started
    (smoke_dir / "SMOKE_NOT_SCIENTIFIC.txt").write_text(
        f"SMOKE PASS in {elapsed:.2f}s. Two sessions, two cycles/session, one encoder seed, "
        f"{args.smoke_ssl_steps} masked-SSL step(s). NOT A FORMAL SCIENTIFIC RESULT.\n",
        encoding="utf-8",
    )
    print(f"SMOKE PASS: {elapsed:.2f}s; outputs={smoke_dir}", flush=True)


def _config_freeze(args: argparse.Namespace) -> str:
    return f"""# v7 configuration freeze

- Sessions: {', '.join(EXPECTED_SESSIONS)} (all retained)
- Sample: frozen clean4 complete-cycle block, shape 4 x 128 x 501
- RAW descriptive PCA: all-session frozen preprocessing, block frame mean, min(50, n-1) components
- Strict stimulus probe: source-only normalization, source-only PCA, source-only scaling, C=1 L2 logistic, balanced weights
- Session probe: cycle-grouped folds, C=1 L2 logistic, balanced weights, {N_SESSION_PERMUTATIONS} cycle-label permutations
- Global encoder: frozen SmallCNN masked reconstruction, 16x16 blocks, ratio 0.5, masked MSE, AdamW, lr=1e-3, weight_decay=1e-4, 50 epochs-equivalent updates
- Global encoder seeds: {', '.join(map(str, GLOBAL_ENCODER_SEEDS))}
- Global encoder sampling: source-session uniform, then frame uniform
- Bootstrap: {N_BOOTSTRAP}, equal cycles per session, complete four-condition cycles retained
- Exact nine-session association tests: complete 9! label enumeration
- condition4 warning: {CONDITION_TIME_WARNING}
- No CSU, DANN, CORAL training, MixStyle, transformer, Mamba, SimCLR, BYOL, new SSL, registration, ROI, GLM, or searchlight
- Formal device: {args.device}; CPU fallback forbidden
"""


def run_formal(args: argparse.Namespace) -> None:
    device = assert_formal_cuda(args.device)
    ensure_output_tree(args.output_dir)
    started = time.perf_counter()
    log("FORMAL v7 START", args.output_dir)
    data = load_sessions(list(EXPECTED_SESSIONS), args.data_dir)
    metadata = sample_metadata_table(data)
    if tuple(sorted(metadata["session"].astype(str).unique(), key=int)) != tuple(EXPECTED_SESSIONS):
        raise AssertionError("not all nine sessions entered formal metadata")
    write_csv(args.output_dir / "audit/clean4_identity_check.csv", clean4_identity_rows(data))
    metadata_audit = metadata_factor_audit(
        metadata, data_root=PROJECT_DIR / "data", sessions=EXPECTED_SESSIONS
    )
    write_csv(args.output_dir / "audit/metadata_factor_audit.csv", metadata_audit)
    (args.output_dir / "audit/config_freeze.md").write_text(_config_freeze(args), encoding="utf-8")

    v5_audit, v5_values = audit_v5_cross_session_metrics(args.v5_output_dir)
    within_audit, within_values = audit_within_session_metrics(args.v1_output_dir)
    pair_audit, pair_performance = audit_pairwise_cross_session(args.v5_output_dir)
    write_csv(args.output_dir / "audit/v5_cross_session_metric_reuse.csv", v5_audit)
    write_csv(args.output_dir / "audit/within_session_metric_reuse.csv", within_audit)
    write_csv(args.output_dir / "audit/pairwise_crosssession_availability.csv", pair_audit)

    log("Fitting all-session descriptive RAW_SPATIAL_PCA", args.output_dir)
    pca, raw_features, raw_mean, raw_std, metadata = fit_common_raw_pca(data)
    save_pca_model(args.output_dir / "features/raw_pca_model.pkl", pca,
                   normalizer_mean=raw_mean, normalizer_std=raw_std)
    write_csv(args.output_dir / "features/raw_pca_common_features.csv", _feature_frame(metadata, raw_features, "raw_PC"))
    explained = pd.DataFrame({
        "component": np.arange(1, pca.n_components_ + 1),
        "explained_variance": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    write_csv(args.output_dir / "features/raw_pca_explained_variance.csv", explained)

    for column, filename, title in (
        ("session", "raw_pca_colored_by_session.png", "RAW_SPATIAL_PCA colored by session"),
        ("stimulus_presence", "raw_pca_colored_by_stimulus_presence.png", "RAW_SPATIAL_PCA colored by stimulus presence"),
        ("condition4", "raw_pca_colored_by_condition4.png", "RAW_SPATIAL_PCA colored by condition/time-position"),
    ):
        categorical_pca_scatter(raw_features[:, :2], metadata, column=column,
                                path=args.output_dir / "figures" / filename, title=title,
                                explained_ratio=pca.explained_variance_ratio_[:2])

    log("Running cycle-grouped session-ID probe and strict source-only LOSO probe", args.output_dir)
    session_probe, confusion, session_predictions = session_id_probe(
        raw_features,
        metadata["session"].astype(str).to_numpy(),
        metadata["cycle"].astype(int).to_numpy(),
        n_permutations=N_SESSION_PERMUTATIONS,
    )
    session_probe["confusion_matrix_json"] = json.dumps(confusion.tolist())
    write_csv(args.output_dir / "summaries/session_id_probe.csv", session_probe)
    write_csv(args.output_dir / "summaries/session_id_probe_predictions.csv", session_predictions)
    confusion_heatmap(confusion, path=args.output_dir / "figures/session_id_confusion_matrix.png")
    stimulus_probe, predictive_audit = source_only_stimulus_probe(data)
    write_csv(args.output_dir / "summaries/source_only_stimulus_probe.csv", stimulus_probe)
    write_csv(args.output_dir / "audit/source_only_predictive_pca_leakage.csv", predictive_audit)
    stimulus_probe_plot(stimulus_probe, path=args.output_dir / "figures/stimulus_probe_by_target.png")

    raw_distances = pairwise_session_distances(
        raw_features, metadata["session"].astype(str).to_numpy(), representation="RAW_SPATIAL_PCA"
    )
    raw_binary, _ = bootstrap_factor_decomposition(
        raw_features, metadata, factor_column="stimulus_presence", representation="RAW_SPATIAL_PCA"
    )
    raw_condition, _ = bootstrap_factor_decomposition(
        raw_features, metadata, factor_column="condition4", representation="RAW_SPATIAL_PCA"
    )
    raw_separability = session_separability(raw_features, metadata)

    # The full block tensors are no longer needed for raw analysis. Reload one
    # session at a time for neural feature extraction to cap host memory.
    del data
    gc.collect()
    log("Loading all complete-cycle unlabeled frames for the global label-free encoder", args.output_dir)
    pool = build_global_unlabeled_pool(PROJECT_DIR, list(EXPECTED_SESSIONS), data_dir=args.data_dir)
    cycle_table = cycle_consistency_from_pool(pool, normalizer_mean=raw_mean, normalizer_std=raw_std)
    updates = int(math.ceil(pool.n_frames / int(args.batch_size)) * 50)
    features_by_seed: dict[int, np.ndarray] = {}
    masked_distance_seed_tables = []
    masked_binary_tables = []
    masked_condition_tables = []
    reuse_rows = []
    peak_vram_values = []
    encoder_runtime_values = []
    for seed in GLOBAL_ENCODER_SEEDS:
        log(f"Global label-free masked encoder seed {seed}: {updates} updates", args.output_dir)
        checkpoint = args.output_dir / f"features/masked_smallcnn_features/checkpoints/global_seed_{seed}.pt"
        encoder, payload, reuse = load_or_train_global_encoder(
            checkpoint, pool, seed=seed, updates=updates, batch_size=int(args.batch_size), device="cuda"
        )
        reuse_rows.append(reuse)
        peak_vram_values.append(float(payload.get("peak_gpu_memory_mb", 0.0)))
        encoder_runtime_values.append(float(payload.get("runtime_seconds", 0.0)))
        session_features = []
        for session in EXPECTED_SESSIONS:
            block_data = load_block_sequence_session(PROJECT_DIR, session, "binary", data_dir=args.data_dir)
            session_features.append(extract_masked_block_features(
                encoder, block_data.X,
                normalizer_mean=payload["normalization_mean"],
                normalizer_std=payload["normalization_std"],
                device="cuda", batch_size=int(args.batch_size),
            ))
            del block_data
        features = np.concatenate(session_features)
        features_by_seed[seed] = features
        write_csv(args.output_dir / f"features/masked_smallcnn_features/seed_{seed}.csv",
                  _feature_frame(metadata, features, "masked_F"))
        masked_distance_seed_tables.append(pairwise_session_distances(
            features, metadata["session"].astype(str).to_numpy(),
            representation="GLOBAL_MASKED_SMALLCNN_SEED", seed=seed,
        ))
        binary_table, _ = bootstrap_factor_decomposition(
            features, metadata, factor_column="stimulus_presence", representation="GLOBAL_MASKED_SMALLCNN_SEED",
            seed_label=seed, random_seed=STATISTIC_SEED + seed,
        )
        condition_table, _ = bootstrap_factor_decomposition(
            features, metadata, factor_column="condition4", representation="GLOBAL_MASKED_SMALLCNN_SEED",
            seed_label=seed, random_seed=STATISTIC_SEED + seed + 100,
        )
        masked_binary_tables.append(binary_table)
        masked_condition_tables.append(condition_table)
        encoder.cpu()
        torch.cuda.empty_cache()
    write_csv(args.output_dir / "audit/global_encoder_reuse.csv", reuse_rows)

    masked_seed_distances = pd.concat(masked_distance_seed_tables, ignore_index=True)
    masked_distances = aggregate_masked_distance_rows(masked_seed_distances)
    all_distances = pd.concat([raw_distances, masked_seed_distances, masked_distances], ignore_index=True, sort=False)
    write_csv(args.output_dir / "summaries/pairwise_session_distances.csv", all_distances)
    distance_heatmap(raw_distances, path=args.output_dir / "figures/session_energy_distance_heatmap_raw.png",
                     title="RAW_SPATIAL_PCA session energy distance")
    distance_heatmap(masked_distances, path=args.output_dir / "figures/session_energy_distance_heatmap_masked.png",
                     title="GLOBAL_MASKED_SMALLCNN mean session energy distance")

    for column, filename, title in (
        ("session", "masked_feature_colored_by_session.png", "Masked SmallCNN feature PCA colored by session"),
        ("stimulus_presence", "masked_feature_colored_by_stimulus_presence.png", "Masked SmallCNN feature PCA colored by stimulus presence"),
        ("condition4", "masked_feature_colored_by_condition4.png", "Masked SmallCNN feature PCA colored by condition/time-position"),
    ):
        masked_seed_scatter(features_by_seed, metadata, column=column,
                            path=args.output_dir / "figures" / filename, title=title)

    masked_binary_aggregate = aggregate_seed_factor_rows(masked_binary_tables)
    masked_condition_aggregate = aggregate_seed_factor_rows(masked_condition_tables)
    factor_binary = pd.concat([raw_binary, *masked_binary_tables, masked_binary_aggregate], ignore_index=True, sort=False)
    factor_condition = pd.concat([raw_condition, *masked_condition_tables, masked_condition_aggregate], ignore_index=True, sort=False)
    write_csv(args.output_dir / "summaries/factor_variance_binary.csv", factor_binary)
    write_csv(args.output_dir / "summaries/factor_variance_condition4.csv", factor_condition)
    factor_variance_plot(factor_binary, path=args.output_dir / "figures/factor_variance_binary.png",
                         title="Session vs stimulus-presence multivariate variance")
    factor_variance_plot(factor_condition, path=args.output_dir / "figures/factor_variance_condition4.png",
                         title="Session vs condition/time-position multivariate variance")

    raw_target_distance = target_mean_energy_distance(raw_distances).rename(
        columns={"mean_distance_to_other_sessions": "mean_distance_RAW_SPATIAL_PCA"}
    )
    masked_target_distance = target_mean_energy_distance(masked_distances).rename(
        columns={"mean_distance_to_other_sessions": "mean_distance_GLOBAL_MASKED_SMALLCNN"}
    )
    diagnostic = within_values.merge(v5_values, on="session").merge(cycle_table, on="session").merge(
        raw_separability, on="session"
    ).merge(raw_target_distance.rename(columns={"target": "session"}), on="session").merge(
        masked_target_distance.rename(columns={"target": "session"}), on="session"
    )
    diagnostic["historical_group"] = np.where(
        diagnostic["session"].astype(str).isin(WEAK_SESSIONS), "historically_weak", "historically_strong"
    )
    write_csv(args.output_dir / "summaries/session_diagnostic_table.csv", diagnostic)

    association_rows = []
    for representation, column in (
        ("RAW_SPATIAL_PCA", "mean_distance_RAW_SPATIAL_PCA"),
        ("GLOBAL_MASKED_SMALLCNN", "mean_distance_GLOBAL_MASKED_SMALLCNN"),
    ):
        result = exact_spearman_permutation(diagnostic[column], diagnostic["v5_cross_session_BA"])
        association_rows.append({
            "analysis": "target_outlier_distance_vs_v5_BA", "representation": representation,
            "x": column, "y": "v5 MULTI_SOURCE_ERM_BA", "n": 9, **result,
        })
    if pair_performance is not None:
        for representation, distances in (("RAW_SPATIAL_PCA", raw_distances), ("GLOBAL_MASKED_SMALLCNN", masked_distances)):
            result = mantel_session_label_permutation(distances, pair_performance)
            association_rows.append({
                "analysis": "pairwise_energy_distance_vs_symmetric_cross_BA",
                "representation": representation, "x": "energy_distance", "y": "symmetric_cross_BA",
                "n": 36, **result,
            })
    else:
        association_rows.append({
            "analysis": "pairwise_energy_distance_vs_symmetric_cross_BA", "representation": "NOT_RUN",
            "x": "energy_distance", "y": "symmetric_cross_BA", "n": 0, "rho": np.nan,
            "permutation_p_two_sided": np.nan, "permutation_method": "PAIRWISE_ANALYSIS_NOT_RUN",
            "n_permutations": 0, "permutation_reason": "no fully comparable 72-directed-pair artifact",
        })
    associations = pd.DataFrame(association_rows)
    write_csv(args.output_dir / "summaries/distance_performance_association.csv", associations)
    distance_performance_plot(diagnostic, associations, path=args.output_dir / "figures/distance_vs_crosssession_BA.png")

    within_assoc_rows = []
    for column in ("separability_ratio", "cycle_consistency_mean", "n_cycles"):
        result = exact_spearman_permutation(diagnostic[column], diagnostic["within_session_BA"])
        within_assoc_rows.append({
            "analysis": f"within_session_BA_vs_{column}", "x": column, "y": "within_session_BA",
            "n": 9, "model": "none_spearman_only", **result,
        })
    within_associations = pd.DataFrame(within_assoc_rows)
    write_csv(args.output_dir / "summaries/within_session_factor_associations.csv", within_associations)
    weak_strong_plot(diagnostic, path=args.output_dir / "figures/weak_vs_strong_diagnostics.png")
    diagnostic_overview(diagnostic, path=args.output_dir / "figures/session_diagnostic_overview.png")

    report = report_markdown(
        metadata_audit=metadata_audit,
        session_probe=session_probe,
        stimulus_probe=stimulus_probe,
        factor_binary=factor_binary,
        factor_condition=factor_condition,
        associations=associations,
        diagnostic=diagnostic,
        pairwise_distance=raw_distances,
        pairwise_available=pair_performance is not None,
        pca_explained_ratio=pca.explained_variance_ratio_,
    )
    (args.output_dir / "report/cross_session_feature_factor_report.md").write_text(report, encoding="utf-8")

    elapsed = time.perf_counter() - started
    gpu_audit = (
        f"GPU name: {torch.cuda.get_device_name(device)}\n"
        f"CUDA runtime: {torch.version.cuda}\n"
        f"PyTorch: {torch.__version__}\n"
        f"PyTorch CUDA available: {torch.cuda.is_available()}\n"
        f"Peak encoder VRAM MB: {max(peak_vram_values):.3f}\n"
        f"Global encoder runtime seconds (sum): {sum(encoder_runtime_values):.3f}\n"
        f"Formal total runtime seconds: {elapsed:.3f}\n"
    )
    (args.output_dir / "audit/gpu_audit.txt").write_text(gpu_audit, encoding="utf-8")
    missing = missing_formal_outputs(args.output_dir)
    # pytest/smoke/command are local handoff artifacts and are copied before
    # the server command.  The formal server run validates every scientific output.
    allowed_local = {"pytest_output_local.txt", "smoke_test_local.txt", "run_command_server.txt"}
    scientific_missing = [value for value in missing if value not in allowed_local]
    if scientific_missing:
        raise AssertionError(f"formal output completeness failed: {scientific_missing}")
    log(f"FORMAL v7 PASS in {elapsed:.2f}s", args.output_dir)


def main() -> None:
    args = parse_args()
    if args.stage == "smoke":
        run_smoke(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
