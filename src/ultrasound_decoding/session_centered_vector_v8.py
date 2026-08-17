"""Session-centering and stimulus-vector mechanism analysis (v8).

This stage is statistics-only when the completed v7 feature artifacts pass
reuse audit.  It never trains or extracts a neural representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ultrasound_decoding.cross_session_feature_factor_v7 import (
    EXPECTED_SESSIONS,
    GLOBAL_ENCODER_SEEDS,
    STATISTIC_SEED,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    audit_pairwise_cross_session,
    classification_metrics,
    exact_spearman_permutation,
    fit_l2_logistic,
    mantel_session_label_permutation,
    session_id_probe,
    sha256_file,
)


RUN_NAME = "session_centered_stimulus_vector_alignment_9sessions_v8"
N_BOOTSTRAP = 1000
N_SPLIT_HALF = 1000
N_SESSION_PERMUTATIONS = 1000
V8_RANDOM_SEED = 20260817
FORMAL_MODE = "CPU_STATS_ONLY"
PRIMARY_LABEL = "stimulus_presence"
TRANSDUCTIVE_WARNING = (
    "MECHANISTIC TRANSDUCTIVE CONTROL: target unlabeled feature distribution is used to estimate its centroid; "
    "this is not strict unseen-session generalization."
)
GD_WARNING = (
    "EXPLORATORY: grating-vs-dot is confounded with fixed within-cycle temporal position and cannot be "
    "interpreted as a pure stimulus-type physiological effect."
)

REQUIRED_OUTPUTS = (
    "audit/v7_feature_reuse.csv",
    "audit/pairwise_crosssession_reuse.csv",
    "audit/config_freeze.md",
    "audit/gpu_audit.txt",
    "features/session_centroids.csv",
    "features/session_centered_features/raw_spatial_pca.csv",
    "features/session_centered_features/masked_seed_20260812.csv",
    "features/session_centered_features/masked_seed_20260813.csv",
    "features/session_centered_features/masked_seed_20260814.csv",
    "summaries/session_id_before_after_centering.csv",
    "summaries/transductive_centering_probe.csv",
    "summaries/stimulus_vector_magnitude.csv",
    "summaries/stimulus_vector_stability.csv",
    "summaries/pairwise_stimulus_vector_cosine.csv",
    "summaries/vector_alignment_vs_transfer.csv",
    "summaries/vector_metrics_vs_within_BA.csv",
    "summaries/session_vector_diagnostic_table.csv",
    "summaries/exploratory_GD_vector_alignment.csv",
    "figures/session_id_before_after_centering.png",
    "figures/transductive_centering_crosssession_BA.png",
    "figures/stimulus_vector_cosine_heatmap_raw.png",
    "figures/stimulus_vector_cosine_heatmap_masked.png",
    "figures/stimulus_vector_stability_by_session.png",
    "figures/stimulus_vector_magnitude_by_session.png",
    "figures/vector_alignment_vs_crosssession_BA.png",
    "figures/session_vector_diagnostic_overview.png",
    "report/session_centered_vector_alignment_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass(frozen=True)
class ReusedFeature:
    representation: str
    seed: int | None
    path: Path
    metadata: pd.DataFrame
    feature_columns: tuple[str, ...]
    values: np.ndarray

    @property
    def key(self) -> tuple[str, int | None]:
        return self.representation, self.seed


def resolve_v7_root(candidate: Path) -> Path:
    required = Path("features/raw_pca_common_features.csv")
    if (candidate / required).is_file():
        return candidate
    matches = sorted(path.parent.parent for path in candidate.glob("**/features/raw_pca_common_features.csv"))
    unique = [
        root for root in dict.fromkeys(matches)
        if (root / "summaries/session_diagnostic_table.csv").is_file()
        and all((root / f"features/masked_smallcnn_features/seed_{seed}.csv").is_file() for seed in GLOBAL_ENCODER_SEEDS)
    ]
    if len(unique) != 1:
        raise FileNotFoundError(f"could not resolve exactly one completed v7 root under {candidate}: {unique}")
    return unique[0]


def _feature_columns(frame: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    columns = tuple(column for column in frame.columns if str(column).startswith(prefix))
    if not columns:
        raise AssertionError(f"no v7 feature columns with prefix {prefix}")
    return columns


def _canonical_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("block_id", "session", "cycle", "cycle_key", "block_name", "condition4", "stimulus_presence")
    missing = set(required) - set(frame.columns)
    if missing:
        raise AssertionError(f"v7 feature metadata fields missing: {sorted(missing)}")
    output = frame[list(required)].copy()
    output["session"] = output["session"].astype(str)
    output["cycle"] = output["cycle"].astype(int)
    expected_presence = np.where(output["block_name"].isin(("grating", "dot")), "stimulus", "no_stimulus")
    if not np.array_equal(output["stimulus_presence"].astype(str).to_numpy(), expected_presence):
        raise AssertionError("v7 stimulus_presence labels differ from frozen mapping")
    return output


def load_and_audit_v7_features(
    candidate_root: Path,
    *,
    processed_data_dir: Path | None = None,
) -> tuple[Path, dict[tuple[str, int | None], ReusedFeature], pd.DataFrame]:
    root = resolve_v7_root(candidate_root)
    specs = [("RAW_SPATIAL_PCA", None, root / "features/raw_pca_common_features.csv", "raw_PC")]
    specs.extend(
        ("GLOBAL_MASKED_SMALLCNN", seed, root / f"features/masked_smallcnn_features/seed_{seed}.csv", "masked_F")
        for seed in GLOBAL_ENCODER_SEEDS
    )
    artifacts: dict[tuple[str, int | None], ReusedFeature] = {}
    audit_rows = []
    reference_metadata: pd.DataFrame | None = None
    for representation, seed, path, prefix in specs:
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        metadata = _canonical_metadata(frame)
        columns = _feature_columns(frame, prefix)
        values = frame[list(columns)].to_numpy(dtype=np.float64)
        sessions = tuple(sorted(metadata["session"].unique(), key=int))
        valid = (
            len(frame) == 456
            and sessions == tuple(EXPECTED_SESSIONS)
            and np.isfinite(values).all()
            and (metadata.groupby(["session", "cycle"])["block_id"].size() == 4).all()
        )
        if reference_metadata is None:
            reference_metadata = metadata
        elif not metadata.equals(reference_metadata):
            raise AssertionError(f"v7 sample identity/order mismatch: {path}")
        artifacts[(representation, seed)] = ReusedFeature(
            representation, seed, path, metadata, columns, values
        )
        audit_rows.append({
            "representation": representation,
            "seed": "" if seed is None else seed,
            "artifact": str(path),
            "artifact_sha256": sha256_file(path),
            "n_samples": len(frame),
            "n_features": len(columns),
            "sessions": ",".join(sessions),
            "sample_order_matches_raw_reference": True,
            "stimulus_mapping_match": True,
            "finite_features": bool(np.isfinite(values).all()),
            "reused_without_refit_or_extraction": True,
            "formal_mode": FORMAL_MODE,
            "status": "PASS" if valid else "FAIL",
        })
    assert reference_metadata is not None
    raw_model = root / "features/raw_pca_model.pkl"
    global_audit = root / "audit/global_encoder_reuse.csv"
    diagnostic = root / "summaries/session_diagnostic_table.csv"
    strict_probe = root / "summaries/source_only_stimulus_probe.csv"
    for label, path in (
        ("V7_RAW_PCA_MODEL_IDENTITY", raw_model),
        ("V7_GLOBAL_ENCODER_AUDIT", global_audit),
        ("V7_SESSION_DIAGNOSTICS", diagnostic),
        ("V7_STRICT_SOURCE_ONLY_PROBE", strict_probe),
    ):
        exists = path.is_file()
        audit_rows.append({
            "representation": label, "seed": "", "artifact": str(path),
            "artifact_sha256": sha256_file(path) if exists else "", "n_samples": "", "n_features": "",
            "sessions": ",".join(EXPECTED_SESSIONS), "sample_order_matches_raw_reference": True,
            "stimulus_mapping_match": True, "finite_features": True,
            "reused_without_refit_or_extraction": True, "formal_mode": FORMAL_MODE,
            "status": "PASS" if exists else "FAIL",
        })
    if processed_data_dir is not None:
        expected_ids = []
        for session in EXPECTED_SESSIONS:
            path = processed_data_dir / f"session_{session}_block_metadata.csv"
            source = pd.read_csv(path)
            expected_ids.extend(source["block_id"].astype(str).tolist())
        if expected_ids != reference_metadata["block_id"].astype(str).tolist():
            raise AssertionError("v7 feature samples do not match frozen processed clean4 metadata")
    audit = pd.DataFrame(audit_rows)
    if (audit["status"] != "PASS").any():
        raise AssertionError("v7 feature reuse audit failed")
    return root, artifacts, audit


def session_centroids(
    features: np.ndarray,
    sessions: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Center by session using features only; labels are not accepted."""
    X = np.asarray(features, dtype=np.float64)
    session_values = np.asarray(sessions).astype(str)
    centered = np.empty_like(X)
    centroids: dict[str, np.ndarray] = {}
    for session in sorted(np.unique(session_values), key=int):
        mask = session_values == session
        centroid = X[mask].mean(axis=0)
        centroids[session] = centroid
        centered[mask] = X[mask] - centroid
    if not np.isfinite(centered).all():
        raise AssertionError("session centering produced non-finite values")
    return centered, centroids


def centroids_long_table(
    artifacts: Mapping[tuple[str, int | None], ReusedFeature],
) -> tuple[pd.DataFrame, dict[tuple[str, int | None], np.ndarray]]:
    rows = []
    centered_by_key = {}
    for key, artifact in artifacts.items():
        centered, centroids = session_centroids(artifact.values, artifact.metadata["session"])
        centered_by_key[key] = centered
        for session, vector in centroids.items():
            for feature_i, value in enumerate(vector, start=1):
                rows.append({
                    "representation": artifact.representation,
                    "seed": "" if artifact.seed is None else artifact.seed,
                    "session": session,
                    "feature_index": feature_i,
                    "centroid_value": float(value),
                    "centroid_scope": "all_unlabeled_clean4_blocks_of_session",
                    "analysis_type": "DESCRIPTIVE_TRANSDUCTIVE_MECHANISM",
                })
    return pd.DataFrame(rows), centered_by_key


def centered_feature_frame(artifact: ReusedFeature, centered: np.ndarray) -> pd.DataFrame:
    metadata = artifact.metadata.reset_index(drop=True)
    values = pd.DataFrame(centered, columns=[f"centered_{column}" for column in artifact.feature_columns])
    return pd.concat([metadata, values], axis=1)


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    value = float(np.dot(x, y) / denominator)
    return float(np.clip(value, -1.0, 1.0))


def contrast_vector(
    features: np.ndarray,
    labels: Sequence[str],
    *,
    positive: str = "stimulus",
    negative: str = "no_stimulus",
) -> np.ndarray:
    X = np.asarray(features, dtype=np.float64)
    values = np.asarray(labels).astype(str)
    if not np.any(values == positive) or not np.any(values == negative):
        raise ValueError("both contrast levels must be present")
    return X[values == positive].mean(axis=0) - X[values == negative].mean(axis=0)


def within_condition_dispersion(features: np.ndarray, labels: Sequence[str]) -> float:
    X = np.asarray(features, dtype=np.float64)
    values = np.asarray(labels).astype(str)
    distances = []
    for level in sorted(np.unique(values)):
        subset = X[values == level]
        centroid = subset.mean(axis=0)
        distances.extend(np.linalg.norm(subset - centroid, axis=1).tolist())
    return float(np.mean(distances))


def session_contrast_vectors(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    task: str = "stimulus_presence",
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    vectors = {}
    rows = []
    sessions = metadata["session"].astype(str).to_numpy()
    if task == "stimulus_presence":
        labels = metadata["stimulus_presence"].astype(str).to_numpy()
        positive, negative = "stimulus", "no_stimulus"
    elif task == "grating_vs_dot":
        keep = metadata["block_name"].astype(str).isin(("grating", "dot")).to_numpy()
        labels = metadata["block_name"].astype(str).to_numpy()
        positive, negative = "grating", "dot"
    else:
        raise ValueError("unsupported contrast task")
    for session in sorted(np.unique(sessions), key=int):
        mask = sessions == session
        if task == "grating_vs_dot":
            mask &= keep
        vector = contrast_vector(features[mask], labels[mask], positive=positive, negative=negative)
        dispersion = within_condition_dispersion(features[mask], labels[mask])
        norm = float(np.linalg.norm(vector))
        vectors[session] = vector
        rows.append({
            "session": session,
            "stimulus_vector_norm" if task == "stimulus_presence" else "GD_vector_norm": norm,
            "within_condition_dispersion": dispersion,
            "normalized_vector_norm": norm / dispersion if dispersion > 0 else 0.0,
            "zero_norm_vector": bool(norm <= np.finfo(float).eps),
            "task": task,
            "exploratory_warning": GD_WARNING if task == "grating_vs_dot" else "",
        })
    return vectors, pd.DataFrame(rows)


def _cycle_block_indices(metadata: pd.DataFrame, session: str) -> dict[int, np.ndarray]:
    session_values = metadata["session"].astype(str).to_numpy()
    cycle_values = metadata["cycle"].astype(int).to_numpy()
    result = {}
    for cycle in np.sort(np.unique(cycle_values[session_values == str(session)])):
        indices = np.flatnonzero((session_values == str(session)) & (cycle_values == int(cycle)))
        if len(indices) != 4:
            raise AssertionError("cycle resampling requires all four condition blocks")
        result[int(cycle)] = indices
    return result


def vector_stability(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    session: str,
    task: str = "stimulus_presence",
    n_bootstrap: int = N_BOOTSTRAP,
    n_split_half: int = N_SPLIT_HALF,
    seed: int = V8_RANDOM_SEED,
) -> dict[str, Any]:
    cycle_indices = _cycle_block_indices(metadata, session)
    cycles = np.asarray(sorted(cycle_indices), dtype=int)
    all_indices = np.concatenate([cycle_indices[int(cycle)] for cycle in cycles])
    session_metadata = metadata.iloc[all_indices].reset_index(drop=True)
    full_vectors, _ = session_contrast_vectors(features[all_indices], session_metadata, task=task)
    full = full_vectors[str(session)]
    rng = np.random.default_rng(int(seed))
    bootstrap_cosines = []
    for _ in range(int(n_bootstrap)):
        drawn = rng.choice(cycles, size=len(cycles), replace=True)
        indices = np.concatenate([cycle_indices[int(cycle)] for cycle in drawn])
        selected = metadata.iloc[indices].reset_index(drop=True)
        vector, _ = session_contrast_vectors(features[indices], selected, task=task)
        bootstrap_cosines.append(safe_cosine(full, vector[str(session)]))
    split_cosines = []
    if len(cycles) < 2:
        raise ValueError("split-half stability requires at least two cycles")
    n_first = len(cycles) // 2
    for _ in range(int(n_split_half)):
        permuted = rng.permutation(cycles)
        first, second = permuted[:n_first], permuted[n_first:]
        first_indices = np.concatenate([cycle_indices[int(cycle)] for cycle in first])
        second_indices = np.concatenate([cycle_indices[int(cycle)] for cycle in second])
        first_vector, _ = session_contrast_vectors(
            features[first_indices], metadata.iloc[first_indices].reset_index(drop=True), task=task
        )
        second_vector, _ = session_contrast_vectors(
            features[second_indices], metadata.iloc[second_indices].reset_index(drop=True), task=task
        )
        split_cosines.append(safe_cosine(first_vector[str(session)], second_vector[str(session)]))
    boot = np.asarray(bootstrap_cosines)
    split = np.asarray(split_cosines)
    if not np.isfinite(boot).all() or not np.isfinite(split).all():
        raise AssertionError("vector stability contains non-finite cosine")
    return {
        "session": str(session),
        "task": task,
        "n_cycles": len(cycles),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_vector_stability": float(np.median(boot)),
        "bootstrap_stability_2_5pct": float(np.percentile(boot, 2.5)),
        "bootstrap_stability_97_5pct": float(np.percentile(boot, 97.5)),
        "n_split_half": int(n_split_half),
        "split_half_vector_stability": float(np.median(split)),
        "split_half_stability_2_5pct": float(np.percentile(split, 2.5)),
        "split_half_stability_97_5pct": float(np.percentile(split, 97.5)),
        "resampling_unit": "complete_cycle_all_four_blocks",
        "split_unit": "complete_cycle",
        "exploratory_warning": GD_WARNING if task == "grating_vs_dot" else "",
    }


def pairwise_vector_cosines(
    vectors: Mapping[str, np.ndarray],
    *,
    representation: str,
    seed: int | str = "",
    task: str = "stimulus_presence",
) -> pd.DataFrame:
    sessions = sorted(vectors, key=int)
    rows = []
    for session_a, session_b in itertools.combinations(sessions, 2):
        rows.append({
            "representation": representation,
            "seed": seed,
            "session_a": session_a,
            "session_b": session_b,
            "cosine_similarity": safe_cosine(vectors[session_a], vectors[session_b]),
            "strong_pair": bool(session_a in STRONG_SESSIONS and session_b in STRONG_SESSIONS),
            "task": task,
            "exploratory_warning": GD_WARNING if task == "grating_vs_dot" else "",
        })
    output = pd.DataFrame(rows)
    if len(output) != math.comb(len(sessions), 2):
        raise AssertionError("pairwise vector table is incomplete")
    return output


def cosine_matrix(pairwise: pd.DataFrame, sessions: Sequence[str] = tuple(EXPECTED_SESSIONS)) -> np.ndarray:
    values = np.eye(len(sessions), dtype=float)
    index = {str(session): i for i, session in enumerate(sessions)}
    for row in pairwise.itertuples():
        i, j = index[str(row.session_a)], index[str(row.session_b)]
        values[i, j] = values[j, i] = float(row.cosine_similarity)
    if not np.allclose(values, values.T) or not np.allclose(np.diag(values), 1):
        raise AssertionError("cosine matrix symmetry/diagonal invariant failed")
    return values


def aggregate_masked_rows(seed_tables: Sequence[pd.DataFrame], value_columns: Sequence[str], keys: Sequence[str]) -> pd.DataFrame:
    combined = pd.concat(seed_tables, ignore_index=True)
    aggregations = {}
    for column in value_columns:
        aggregations[f"{column}"] = (column, "mean")
        aggregations[f"{column}_seed_std"] = (column, "std")
    output = combined.groupby(list(keys), as_index=False).agg(**aggregations)
    output.insert(0, "seed", "MEAN_3_SEEDS")
    output.insert(0, "representation", "GLOBAL_MASKED_SMALLCNN")
    return output


def loso_probe(
    features: np.ndarray,
    metadata: pd.DataFrame,
    *,
    center_by_session: bool,
) -> pd.DataFrame:
    X = np.asarray(features, dtype=np.float64)
    sessions = metadata["session"].astype(str).to_numpy()
    labels = (metadata["stimulus_presence"].astype(str).to_numpy() == "stimulus").astype(int)
    transformed, _ = session_centroids(X, sessions) if center_by_session else (X.copy(), {})
    rows = []
    for target in sorted(np.unique(sessions), key=int):
        source = sessions != target
        target_mask = sessions == target
        model = fit_l2_logistic(transformed[source], labels[source], C=1.0, class_weight="balanced")
        prediction = model.predict(transformed[target_mask]).astype(int)
        metrics = classification_metrics(labels[target_mask], prediction, [0, 1])
        rows.append({
            "target": target,
            "BA": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
            "macro_F1": metrics["macro_F1"],
            "center_by_session": bool(center_by_session),
            "target_centroid_uses_labels": False,
            "classifier": "L2 logistic regression",
            "C": 1.0,
            "class_weight": "balanced",
            "analysis_type": "TRANSDUCTIVE_UNSUPERVISED_CENTERING" if center_by_session else "COMMON_SPACE_UNCENTERED_CONTROL",
            "warning": TRANSDUCTIVE_WARNING if center_by_session else "",
        })
    return pd.DataFrame(rows)


def exact_sign_flip_test(differences: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    observed = float(values.mean())
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistics.append(float(np.mean(values * np.asarray(signs))))
    null = np.asarray(statistics)
    p = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "mean_delta": observed,
        "exact_sign_flip_p_two_sided": p,
        "n_sign_flips": len(null),
        "test_label": "MECHANISTIC SECONDARY TEST",
    }


def session_id_before_after(
    artifact: ReusedFeature,
    centered: np.ndarray,
    *,
    n_permutations: int = N_SESSION_PERMUTATIONS,
    n_folds: int = 5,
    max_iter: int = 400,
) -> pd.DataFrame:
    metadata = artifact.metadata
    common = dict(
        session_labels=metadata["session"].astype(str).to_numpy(),
        cycles=metadata["cycle"].astype(int).to_numpy(),
        n_permutations=n_permutations,
        n_folds=n_folds,
        max_iter=max_iter,
    )
    before, _matrix, _pred = session_id_probe(artifact.values, **common)
    after, _matrix, _pred = session_id_probe(centered, **common)
    before_row = before[before["metric"] == "balanced_accuracy"].iloc[0]
    after_row = after[after["metric"] == "balanced_accuracy"].iloc[0]
    return pd.DataFrame([{
        "representation": artifact.representation,
        "seed": "" if artifact.seed is None else artifact.seed,
        "uncentered_session_BA": float(before_row.observed),
        "centered_session_BA": float(after_row.observed),
        "delta_centered_minus_uncentered": float(after_row.observed - before_row.observed),
        "session_information_reduction": float(before_row.observed - after_row.observed),
        "uncentered_permutation_p": float(before_row.permutation_p_greater_equal),
        "centered_permutation_p": float(after_row.permutation_p_greater_equal),
        "permutation_p": float(after_row.permutation_p_greater_equal),
        "n_permutations_each": int(n_permutations),
        "cv_grouping": "cycle_grouped_within_each_session",
        "centroid_scope": "all_unlabeled_blocks_of_session",
        "analysis_type": "DESCRIPTIVE_TRANSDUCTIVE_MECHANISM",
    }])


def pairwise_transfer_audit(v5_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit, pairwise = audit_pairwise_cross_session(v5_root)
    if pairwise is None or audit.iloc[0]["status"] != "PASS_RUN_SECONDARY":
        raise AssertionError("fully comparable v5 72-directed-pair artifact is required for v8 association")
    audit = audit.copy()
    audit["same_binary_definition"] = True
    audit["same_clean4"] = True
    audit["same_preprocessing"] = True
    audit["same_model_protocol"] = True
    audit["retrained_for_v8"] = False
    return audit, pairwise


def vector_alignment_transfer_association(
    pairwise_cosine: pd.DataFrame,
    pairwise_transfer: pd.DataFrame,
    *,
    representation: str,
) -> dict[str, Any]:
    result = mantel_session_label_permutation(pairwise_cosine.rename(columns={"cosine_similarity": "energy_distance"}), pairwise_transfer)
    return {
        "representation": representation,
        "analysis": "stimulus_vector_cosine_vs_symmetric_cross_session_BA",
        "n_pairs": 36,
        **result,
        "interpretation_scope": "association_not_causation",
    }


def descriptive_pattern_scores(diagnostic: pd.DataFrame) -> pd.DataFrame:
    """Assign V1/V2/V3 by transparent rank-based descriptive scores."""
    output = diagnostic.copy()
    magnitude = output["normalized_vector_norm"].rank(pct=True)
    separability = output["v7_separability_ratio"].rank(pct=True)
    stability = output[["bootstrap_vector_stability", "split_half_vector_stability"]].mean(axis=1).rank(pct=True)
    cosine = output["mean_vector_cosine_to_other_sessions"].rank(pct=True)
    output["pattern_V1_score"] = (magnitude + stability + (1 - cosine)) / 3
    output["pattern_V2_score"] = ((1 - magnitude) + (1 - separability)) / 2
    output["pattern_V3_score"] = (magnitude + (1 - stability)) / 2
    score_columns = ["pattern_V1_score", "pattern_V2_score", "pattern_V3_score"]
    output["descriptive_pattern"] = output[score_columns].idxmax(axis=1).str.extract(r"(V[123])", expand=False)
    output["pattern_assignment_scope"] = "DESCRIPTIVE_RANK_BASED_NO_SUBGROUP_TEST"
    return output


def missing_outputs(output_dir: Path) -> list[str]:
    return [relative for relative in REQUIRED_OUTPUTS if not (output_dir / relative).is_file()]
