from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.multiframe.dataset import BlockSequenceData, EXPECTED_SESSIONS


ANALYSIS_VERSION = "within_session_cycle_drift_diagnostic_v1.0.0"
PRIMARY_BLOCK_TYPES = ("stop_after_grating", "static")
SECONDARY_STIMULUS_BLOCK_TYPES = ("grating", "dot")
STRONG_SESSIONS = ("708", "709", "710")
WEAK_SESSIONS = tuple(
    session for session in EXPECTED_SESSIONS if session not in STRONG_SESSIONS
)
SPATIAL_EPSILON = 1e-8
INTENSITY_EPSILON = 1e-12
FLIP_TOLERANCE = 1e-12
INTERPRETATION_RULE = {
    "candidate": {
        "session_pearson_maximum": -0.5,
        "session_spearman_maximum": -0.5,
        "weak_minus_strong_mean_drift_exclusive_minimum": 0.0,
        "pooled_fold_spearman_exclusive_maximum": 0.0,
        "negative_within_session_fold_spearman_minimum": 5,
    },
    "mixed_minimum_directional_indicators": 2,
}


@dataclass(frozen=True)
class TemplateBundle:
    session: str
    templates: dict[tuple[int, str], np.ndarray]
    arcsinh_frames: dict[tuple[int, str], np.ndarray]
    metrics: pd.DataFrame


def spatial_zscore(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("spatial image must be a finite [H,W] array")
    return (values - values.mean()) / (values.std() + SPATIAL_EPSILON)


def spatial_correlation(first: np.ndarray, second: np.ndarray) -> float:
    a = spatial_zscore(first).reshape(-1)
    b = spatial_zscore(second).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        raise ValueError("spatial correlation is undefined for a constant image")
    return float(np.dot(a, b) / denominator)


def cycle_template(clean4_block: np.ndarray) -> np.ndarray:
    values = np.asarray(clean4_block, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("clean4 block must have shape [4,H,W]")
    if not np.isfinite(values).all():
        raise ValueError("clean4 block contains NaN or Inf")
    return np.arcsinh(values).mean(axis=0, dtype=np.float64)


def median_pairwise_spatial_correlation(images: np.ndarray) -> float:
    values = np.asarray(images)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError("at least two [H,W] images are required")
    correlations = [
        spatial_correlation(values[first], values[second])
        for first, second in itertools.combinations(range(len(values)), 2)
    ]
    return float(np.median(correlations))


def within_block_frame_stability(arcsinh_frames: np.ndarray) -> float:
    values = np.asarray(arcsinh_frames)
    if values.ndim != 3 or values.shape[0] != 4:
        raise ValueError("arcsinh clean4 frames must have shape [4,H,W]")
    correlations = [
        spatial_correlation(values[first], values[second])
        for first, second in itertools.combinations(range(4), 2)
    ]
    if len(correlations) != 6:
        raise AssertionError("clean4 must yield exactly six frame pairs")
    return float(np.median(correlations))


def build_primary_templates(data: BlockSequenceData) -> TemplateBundle:
    names = data.metadata["block_name"].astype(str).to_numpy()
    if set(names) != {"grating", "stop_after_grating", "dot", "static"}:
        raise AssertionError("binary dataset does not contain the frozen four block types")
    templates: dict[tuple[int, str], np.ndarray] = {}
    frames_by_key: dict[tuple[int, str], np.ndarray] = {}
    rows = []
    for index, (cycle, name) in enumerate(zip(data.groups.astype(int), names)):
        if name not in PRIMARY_BLOCK_TYPES:
            continue
        key = (int(cycle), name)
        if key in templates:
            raise AssertionError("cycle contains duplicate primary block type")
        frames = np.arcsinh(data.X[index].astype(np.float32, copy=False)).astype(
            np.float64, copy=False
        )
        template = frames.mean(axis=0)
        templates[key] = template
        frames_by_key[key] = frames
        rows.append(
            {
                "session": str(data.session),
                "cycle": int(cycle),
                "block_type": name,
                "global_mean": float(template.mean()),
                "global_std": float(template.std()),
                "minimum": float(template.min()),
                "maximum": float(template.max()),
                "finite": bool(np.isfinite(template).all()),
                "within_block_frame_stability": within_block_frame_stability(frames),
            }
        )
    expected = data.n_cycles * len(PRIMARY_BLOCK_TYPES)
    if len(templates) != expected:
        raise AssertionError(f"primary template coverage {len(templates)} != {expected}")
    if {name for _, name in templates} != set(PRIMARY_BLOCK_TYPES):
        raise AssertionError("primary templates must use only stop and static")
    return TemplateBundle(
        session=str(data.session),
        templates=templates,
        arcsinh_frames=frames_by_key,
        metrics=pd.DataFrame(rows).sort_values(["cycle", "block_type"]).reset_index(drop=True),
    )


def summarize_session_drift(
    bundle: TemplateBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pair_rows = []
    block_stability: dict[str, float] = {}
    for block_type in PRIMARY_BLOCK_TYPES:
        cycles = sorted(cycle for cycle, name in bundle.templates if name == block_type)
        correlations = []
        for first, second in itertools.combinations(cycles, 2):
            correlation = spatial_correlation(
                bundle.templates[(first, block_type)],
                bundle.templates[(second, block_type)],
            )
            correlations.append(correlation)
            pair_rows.append(
                {
                    "session": bundle.session,
                    "block_type": block_type,
                    "cycle_i": int(first),
                    "cycle_j": int(second),
                    "spatial_correlation": correlation,
                }
            )
        block_stability[block_type] = float(np.median(correlations))
    background_stability = float(
        np.mean([block_stability[name] for name in PRIMARY_BLOCK_TYPES])
    )
    within_values = bundle.metrics["within_block_frame_stability"].to_numpy(float)
    global_values = bundle.metrics["global_mean"].to_numpy(float)
    global_median = float(np.median(global_values))
    global_mad = float(np.median(np.abs(global_values - global_median)))
    summary = {
        "session": bundle.session,
        "n_cycles": int(len({cycle for cycle, _ in bundle.templates})),
        "stop_spatial_stability": block_stability["stop_after_grating"],
        "stop_spatial_drift": 1.0 - block_stability["stop_after_grating"],
        "static_spatial_stability": block_stability["static"],
        "static_spatial_drift": 1.0 - block_stability["static"],
        "background_spatial_stability": background_stability,
        "background_spatial_drift": 1.0 - background_stability,
        "session_within_block_stability": float(np.mean(within_values)),
        "session_within_block_stability_median": float(np.median(within_values)),
        "cycle_drift_gap": float(np.mean(within_values) - background_stability),
        "global_intensity_median": global_median,
        "global_intensity_mad": global_mad,
        "global_intensity_robust_dispersion": 1.4826 * global_mad,
        "primary_metric": "background_spatial_drift",
    }
    return pd.DataFrame(pair_rows), summary


def pixelwise_training_reference(
    templates: Mapping[tuple[int, str], np.ndarray],
    training_cycles: Iterable[int],
    block_type: str,
) -> np.ndarray:
    if block_type not in PRIMARY_BLOCK_TYPES:
        raise ValueError("fold reference is defined only for stop and static")
    cycles = sorted({int(cycle) for cycle in training_cycles})
    if not cycles:
        raise ValueError("training cycles are empty")
    values = np.stack([templates[(cycle, block_type)] for cycle in cycles], axis=0)
    return np.median(values, axis=0)


def _robust_intensity_scale(training_values: np.ndarray) -> tuple[float, str]:
    values = np.asarray(training_values, dtype=np.float64)
    center = float(np.median(values))
    mad_scale = float(1.4826 * np.median(np.abs(values - center)))
    if mad_scale > INTENSITY_EPSILON:
        return mad_scale, "1.4826_times_MAD"
    return float(values.std() + INTENSITY_EPSILON), "training_std_plus_epsilon"


def fold_train_test_drift(
    bundle: TemplateBundle,
    *,
    fold: int,
    training_cycles: Iterable[int],
    test_cycles: Iterable[int],
) -> dict[str, Any]:
    train = sorted({int(value) for value in training_cycles})
    test = sorted({int(value) for value in test_cycles})
    if set(train) & set(test):
        raise AssertionError("fold training and test cycles overlap")
    similarities: dict[str, list[float]] = {name: [] for name in PRIMARY_BLOCK_TYPES}
    intensity_shifts: dict[str, list[float]] = {name: [] for name in PRIMARY_BLOCK_TYPES}
    scale_methods: dict[str, str] = {}
    for block_type in PRIMARY_BLOCK_TYPES:
        reference = pixelwise_training_reference(bundle.templates, train, block_type)
        training_global = np.asarray(
            [bundle.templates[(cycle, block_type)].mean() for cycle in train], dtype=np.float64
        )
        center = float(np.median(training_global))
        scale, scale_method = _robust_intensity_scale(training_global)
        scale_methods[block_type] = scale_method
        for cycle in test:
            template = bundle.templates[(cycle, block_type)]
            similarities[block_type].append(spatial_correlation(reference, template))
            intensity_shifts[block_type].append(
                abs(float(template.mean()) - center) / scale
            )
    all_similarities = [
        value for name in PRIMARY_BLOCK_TYPES for value in similarities[name]
    ]
    all_intensity = [
        value for name in PRIMARY_BLOCK_TYPES for value in intensity_shifts[name]
    ]
    if len(all_similarities) != len(test) * 2:
        raise AssertionError("fold drift must equally cover test cycles x stop/static")
    return {
        "session": bundle.session,
        "fold": int(fold),
        "train_cycles": ",".join(map(str, train)),
        "test_cycles": ",".join(map(str, test)),
        "n_train_cycles": len(train),
        "n_test_cycles": len(test),
        "stop_background_similarity": float(np.mean(similarities["stop_after_grating"])),
        "stop_train_test_drift": 1.0
        - float(np.mean(similarities["stop_after_grating"])),
        "static_background_similarity": float(np.mean(similarities["static"])),
        "static_train_test_drift": 1.0 - float(np.mean(similarities["static"])),
        "fold_background_similarity": float(np.mean(all_similarities)),
        "fold_train_test_drift": 1.0 - float(np.mean(all_similarities)),
        "fold_global_intensity_shift": float(np.mean(all_intensity)),
        "stop_intensity_scale_method": scale_methods["stop_after_grating"],
        "static_intensity_scale_method": scale_methods["static"],
        "training_reference_uses_test_cycles": False,
    }


def reconstruct_historical_decoder(
    predictions: pd.DataFrame, saved_session_seed_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"session", "seed", "fold", "block_id", "truth", "pred"}
    if not required.issubset(predictions.columns):
        raise AssertionError("historical predictions lack required columns")
    table = predictions.copy()
    table["session"] = table["session"].astype(str)
    table["seed"] = table["seed"].astype(int)
    if table.duplicated(["session", "seed", "block_id"]).any():
        raise AssertionError("historical OOF predictions contain duplicate blocks")
    session_seed_rows = []
    fold_seed_rows = []
    for (session, seed), group in table.groupby(["session", "seed"], sort=True):
        ba = classification_metrics(
            group["truth"].to_numpy(np.int64), group["pred"].to_numpy(np.int64)
        )["balanced_accuracy"]
        session_seed_rows.append(
            {"session": str(session), "seed": int(seed), "late_fusion_BA": ba}
        )
    for (session, fold, seed), group in table.groupby(
        ["session", "fold", "seed"], sort=True
    ):
        ba = classification_metrics(
            group["truth"].to_numpy(np.int64), group["pred"].to_numpy(np.int64)
        )["balanced_accuracy"]
        fold_seed_rows.append(
            {"session": str(session), "fold": int(fold), "seed": int(seed), "fold_BA": ba}
        )
    session_seed = pd.DataFrame(session_seed_rows)
    saved = saved_session_seed_summary.copy()
    saved["session"] = saved["session"].astype(str)
    saved = saved[["session", "seed", "late_fusion_BA"]].rename(
        columns={"late_fusion_BA": "saved_late_fusion_BA"}
    )
    compared = session_seed.merge(saved, on=["session", "seed"], validate="one_to_one")
    maximum_difference = float(
        np.max(np.abs(compared["late_fusion_BA"] - compared["saved_late_fusion_BA"]))
    )
    if maximum_difference > 1e-12:
        raise AssertionError("reconstructed historical BA differs from formal summary")
    formal_session = (
        session_seed.groupby("session", as_index=False)["late_fusion_BA"]
        .mean()
        .rename(columns={"late_fusion_BA": "formal_session_FCNN_latefusion_BA"})
    )
    fold_seed = pd.DataFrame(fold_seed_rows)
    fold_performance = (
        fold_seed.groupby(["session", "fold"], as_index=False)["fold_BA"]
        .mean()
        .rename(columns={"fold_BA": "fold_FCNN_latefusion_BA_seedavg"})
    )
    audit = {
        "status": "PASS",
        "decoder_retrained": False,
        "prediction_rows": int(len(table)),
        "session_seed_rows": int(len(session_seed)),
        "fold_seed_rows": int(len(fold_seed)),
        "fold_rows_after_seed_average": int(len(fold_performance)),
        "maximum_absolute_session_seed_BA_difference": maximum_difference,
        "overall_formal_session_mean_BA": float(
            formal_session["formal_session_FCNN_latefusion_BA"].mean()
        ),
        "session_metric": "concatenate all OOF blocks then BA, then mean three seeds",
        "fold_metric": "held-out fold BA then mean three seeds",
    }
    return formal_session, fold_performance, audit


def flip_bundle_vertically(bundle: TemplateBundle) -> TemplateBundle:
    flipped_frames = {
        key: np.flip(value, axis=1).copy() for key, value in bundle.arcsinh_frames.items()
    }
    flipped_metrics = bundle.metrics.copy()
    for row_index, row in flipped_metrics.iterrows():
        key = (int(row["cycle"]), str(row["block_type"]))
        flipped_metrics.loc[row_index, "within_block_frame_stability"] = (
            within_block_frame_stability(flipped_frames[key])
        )
    return TemplateBundle(
        session=bundle.session,
        templates={key: np.flip(value, axis=0).copy() for key, value in bundle.templates.items()},
        arcsinh_frames=flipped_frames,
        metrics=flipped_metrics,
    )


def flip_invariance_audit(bundle: TemplateBundle) -> dict[str, Any]:
    if bundle.session != "807":
        raise ValueError("vertical-flip audit is defined only for session 807")
    flipped = flip_bundle_vertically(bundle)
    _, original = summarize_session_drift(bundle)
    _, transformed = summarize_session_drift(flipped)
    keys = (
        "stop_spatial_stability",
        "static_spatial_stability",
        "background_spatial_drift",
        "session_within_block_stability",
    )
    differences = {
        key: abs(float(original[key]) - float(transformed[key])) for key in keys
    }
    maximum = float(max(differences.values()))
    if maximum > FLIP_TOLERANCE:
        raise AssertionError("uniform 807 vertical flip changed within-session drift")
    return {
        "status": "PASS",
        "session": "807",
        "operation": "uniform_vertical_flip_all_cycles_nonstimulus_templates_in_memory_only",
        "original_background_spatial_drift": original["background_spatial_drift"],
        "flipped_background_spatial_drift": transformed["background_spatial_drift"],
        "absolute_difference": differences["background_spatial_drift"],
        "metric_absolute_differences": differences,
        "maximum_absolute_difference": maximum,
        "tolerance": FLIP_TOLERANCE,
        "raw_data_modified": False,
        "interpretation": (
            "Uniform session-level vertical orientation does not change the "
            "within-session cycle-drift metric."
        ),
    }


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if len(first) < 2 or first.std() == 0 or second.std() == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def exact_spearman_permutation_test(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    first = np.asarray(x, dtype=np.float64)
    second = np.asarray(y, dtype=np.float64)
    if len(first) != 9 or len(second) != 9:
        raise ValueError("exact primary Spearman test is frozen to nine sessions")
    x_rank = rankdata(first).astype(np.float64)
    y_rank = rankdata(second).astype(np.float64)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = float(np.linalg.norm(x_rank) * np.linalg.norm(y_rank))
    observed = float(np.dot(x_rank, y_rank) / denominator)
    extreme = 0
    total = math.factorial(9)
    threshold = abs(observed) - 1e-15
    for permutation in itertools.permutations(y_rank.tolist()):
        statistic = float(np.dot(x_rank, np.asarray(permutation)) / denominator)
        extreme += int(abs(statistic) >= threshold)
    return {
        "spearman_rho": observed,
        "exact_two_sided_permutation_p": float(extreme / total),
        "permutations": total,
        "extreme_permutations": extreme,
    }


def exact_strong_group_permutation(
    sessions: Iterable[str], drift: np.ndarray
) -> dict[str, Any]:
    session_list = [str(value) for value in sessions]
    values = np.asarray(drift, dtype=np.float64)
    if len(session_list) != 9 or len(values) != 9:
        raise ValueError("strong/weak permutation is frozen to nine sessions")
    observed_indices = {session_list.index(session) for session in STRONG_SESSIONS}

    def difference(strong_indices: set[int]) -> float:
        strong = np.asarray([values[index] for index in sorted(strong_indices)])
        weak = np.asarray(
            [values[index] for index in range(9) if index not in strong_indices]
        )
        return float(weak.mean() - strong.mean())

    observed = difference(observed_indices)
    assignments = list(itertools.combinations(range(9), 3))
    extreme = sum(
        abs(difference(set(indices))) >= abs(observed) - 1e-15 for indices in assignments
    )
    strong_values = values[list(sorted(observed_indices))]
    weak_values = values[
        [index for index in range(9) if index not in observed_indices]
    ]
    return {
        "strong_mean_drift": float(strong_values.mean()),
        "strong_median_drift": float(np.median(strong_values)),
        "weak_mean_drift": float(weak_values.mean()),
        "weak_median_drift": float(np.median(weak_values)),
        "weak_minus_strong_mean_difference": observed,
        "exact_two_sided_group_label_permutation_p": float(extreme / len(assignments)),
        "assignments": len(assignments),
    }


def association_analysis(
    session_summary: pd.DataFrame,
    fold_table: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    drift = session_summary["background_spatial_drift"].to_numpy(float)
    ba = session_summary["formal_session_FCNN_latefusion_BA"].to_numpy(float)
    primary_spearman = exact_spearman_permutation_test(drift, ba)
    group = exact_strong_group_permutation(session_summary["session"], drift)
    pooled_pearson = pearson_correlation(
        fold_table["fold_train_test_drift"],
        fold_table["fold_FCNN_latefusion_BA_seedavg"],
    )
    pooled_spearman = float(
        spearmanr(
            fold_table["fold_train_test_drift"],
            fold_table["fold_FCNN_latefusion_BA_seedavg"],
        ).statistic
    )
    within_rows = []
    for session, group_rows in fold_table.groupby("session", sort=True):
        within_rows.append(
            {
                "session": str(session),
                "n_folds": int(len(group_rows)),
                "pearson_r": pearson_correlation(
                    group_rows["fold_train_test_drift"],
                    group_rows["fold_FCNN_latefusion_BA_seedavg"],
                ),
                "spearman_rho": float(
                    spearmanr(
                        group_rows["fold_train_test_drift"],
                        group_rows["fold_FCNN_latefusion_BA_seedavg"],
                    ).statistic
                ),
            }
        )
    within = pd.DataFrame(within_rows)
    negative_count = int((within["spearman_rho"] < 0).sum())
    intensity_pearson = pearson_correlation(
        session_summary["global_intensity_robust_dispersion"], ba
    )
    intensity_spearman = float(
        spearmanr(session_summary["global_intensity_robust_dispersion"], ba).statistic
    )
    summary = {
        "session_level_primary": {
            "metric": "background_spatial_drift",
            "target": "formal_session_FCNN_latefusion_BA",
            "pearson_r": pearson_correlation(drift, ba),
            **primary_spearman,
            "direction_hypothesis": "higher drift predicts lower BA",
        },
        "strong_vs_weak": group,
        "fold_level": {
            "label": "POOLED_DESCRIPTIVE_ONLY",
            "n_folds": int(len(fold_table)),
            "pooled_pearson_r": pooled_pearson,
            "pooled_spearman_rho": pooled_spearman,
            "negative_within_session_spearman_count": negative_count,
            "median_within_session_fold_spearman": float(
                np.nanmedian(within["spearman_rho"])
            ),
            "folds_are_independent_subjects": False,
        },
        "secondary_global_intensity": {
            "label": "SECONDARY_ONLY",
            "pearson_r": intensity_pearson,
            "spearman_rho": intensity_spearman,
            "changes_primary_conclusion": False,
        },
        "primary_metric_switched": False,
        "primary_target_switched": False,
        "confirmatory_test": False,
    }
    return within, summary


def mechanism_interpretation(association: Mapping[str, Any]) -> dict[str, Any]:
    session = association["session_level_primary"]
    group = association["strong_vs_weak"]
    fold = association["fold_level"]
    candidate_checks = {
        "session_pearson_at_most_minus_0_5": session["pearson_r"]
        <= INTERPRETATION_RULE["candidate"]["session_pearson_maximum"],
        "session_spearman_at_most_minus_0_5": session["spearman_rho"]
        <= INTERPRETATION_RULE["candidate"]["session_spearman_maximum"],
        "weak_mean_drift_greater_than_strong": group[
            "weak_minus_strong_mean_difference"
        ]
        > 0,
        "pooled_fold_spearman_negative": fold["pooled_spearman_rho"] < 0,
        "at_least_five_negative_within_session_fold_spearman": fold[
            "negative_within_session_spearman_count"
        ]
        >= 5,
    }
    directional = {
        "session_spearman_negative": session["spearman_rho"] < 0,
        "weak_mean_drift_greater_than_strong": group[
            "weak_minus_strong_mean_difference"
        ]
        > 0,
        "pooled_fold_spearman_negative": fold["pooled_spearman_rho"] < 0,
        "at_least_five_negative_within_session_fold_spearman": fold[
            "negative_within_session_spearman_count"
        ]
        >= 5,
    }
    if all(candidate_checks.values()):
        interpretation = (
            "cycle_spatial_drift_is_consistent_with_a_candidate_generalization_bottleneck"
        )
    elif sum(directional.values()) >= INTERPRETATION_RULE[
        "mixed_minimum_directional_indicators"
    ]:
        interpretation = "cycle_spatial_drift_shows_mixed_evidence"
    else:
        interpretation = "cycle_spatial_drift_is_not_supported_as_a_major_bottleneck"
    return {
        "interpretation": interpretation,
        "rule": INTERPRETATION_RULE,
        "candidate_checks": candidate_checks,
        "directional_checks": directional,
        "rule_changed_after_results": False,
        "exploratory_not_confirmatory": True,
    }
