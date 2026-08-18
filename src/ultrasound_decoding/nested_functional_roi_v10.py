"""Leakage-safe nested functional ROI decoding for nine-session fUS clean4 data."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.linear import LDAModel, fit_predict_linear
from ultrasound_decoding.multiframe.dataset import (
    BLOCK_NAMES,
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    BlockSequenceData,
    load_block_sequence_session,
)


RUN_NAME = "nested_functional_roi_decoding_9sessions_v10"
SESSIONS = tuple(EXPECTED_SESSIONS)
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")
FIXED_ORIENTATIONS = {session: "identity" for session in SESSIONS}
FIXED_ORIENTATIONS["807"] = "flip_vertical"
EPS = 1.0e-8
PCA_VARIANCE = 0.95
LDA_REG = 1.0e-3
MAX_FOLDS = 10
ROI_RULES = {"functional_roi_top10": 0.10, "functional_roi_top20": 0.20}
MODELS = (
    "whole_brain_clean4_flat4_pca_lda",
    "roi_mean4_top10_rlda",
    "roi_flat4_top10_pca_lda",
    "roi_mean4_top20_rlda",
    "roi_flat4_top20_pca_lda",
)
PRIMARY_MODELS = MODELS[:3]

REQUIRED_OUTPUTS = (
    "audit/config_freeze.md",
    "audit/class_balance_audit.csv",
    "audit/fold_roi_audit.csv",
    "audit/baseline_reuse_audit.csv",
    "audit/v9_metric_reuse_audit.csv",
    "summaries/within_session_roi_decoding_summary.csv",
    "summaries/fold_level_roi_results.csv",
    "summaries/oof_predictions.csv",
    "summaries/roi_stability_summary.csv",
    "summaries/roi_gain_vs_v9_metrics.csv",
    "summaries/roi_gain_vs_v9_associations.csv",
    "figures/functional_roi_overview_top10.png",
    "figures/functional_roi_overview_top20.png",
    "figures/within_session_binary_roi_vs_wholebrain.png",
    "figures/roi_gain_top10_by_session.png",
    "figures/roi_size_stability_by_session.png",
    "figures/roi_gain_vs_v9_metrics.png",
    "report/nested_functional_roi_decoding_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass
class SessionResult:
    summary_rows: list[dict[str, Any]]
    fold_rows: list[dict[str, Any]]
    oof_rows: list[dict[str, Any]]
    roi_audit_rows: list[dict[str, Any]]
    class_balance_row: dict[str, Any]
    stability_rows: list[dict[str, Any]]
    baseline_audit_row: dict[str, Any]
    representative: dict[str, tuple[np.ndarray, np.ndarray]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cycle_text(values: Sequence[int] | np.ndarray) -> str:
    return ",".join(str(int(value)) for value in sorted(np.unique(values).tolist()))


def orient_clean4(X: np.ndarray, orientation: str) -> np.ndarray:
    values = np.asarray(X)
    if values.ndim != 4 or tuple(values.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise ValueError(f"clean4 must be [N,{EXPECTED_BLOCK_SHAPE}], got {values.shape}")
    if orientation == "identity":
        return values
    if orientation == "flip_vertical":
        return np.flip(values, axis=-2).copy()
    raise ValueError(f"unsupported frozen orientation: {orientation}")


def arcsinh_clean4(X: np.ndarray) -> np.ndarray:
    values = np.arcsinh(np.asarray(X, dtype=np.float64))
    if not np.isfinite(values).all():
        raise ValueError("arcsinh clean4 contains NaN/Inf")
    return values


def cycle_response_maps(
    X_arcsinh: np.ndarray,
    metadata: pd.DataFrame,
    groups: np.ndarray,
    cycles: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct one fixed stimulus-presence contrast map per requested cycle."""
    maps: list[np.ndarray] = []
    used: list[int] = []
    groups = np.asarray(groups, dtype=np.int64)
    for cycle in np.asarray(cycles, dtype=np.int64):
        indices = np.flatnonzero(groups == cycle)
        names = metadata.iloc[indices]["block_name"].astype(str).tolist()
        if sorted(names) != sorted(BLOCK_NAMES) or len(indices) != 4:
            raise AssertionError(f"cycle {cycle} does not contain exactly the frozen four blocks")
        block_means = {
            str(metadata.iloc[index]["block_name"]): X_arcsinh[index].mean(axis=0)
            for index in indices
        }
        contrast = 0.5 * (block_means["grating"] + block_means["dot"]) - 0.5 * (
            block_means["stop_after_grating"] + block_means["static"]
        )
        maps.append(contrast)
        used.append(int(cycle))
    if not maps:
        raise ValueError("at least one training cycle is required")
    output = np.stack(maps).astype(np.float64, copy=False)
    if output.shape[1:] != EXPECTED_BLOCK_SHAPE[1:] or not np.isfinite(output).all():
        raise AssertionError("cycle response map shape/finite audit failed")
    return output, np.asarray(used, dtype=np.int64)


def training_z_map(cycle_maps: np.ndarray, eps: float = EPS) -> np.ndarray:
    maps = np.asarray(cycle_maps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[1:] != EXPECTED_BLOCK_SHAPE[1:]:
        raise ValueError(f"cycle maps must be [cycles,128,501], got {maps.shape}")
    if eps <= 0:
        raise ValueError("eps must be positive")
    z_map = maps.mean(axis=0) / (maps.std(axis=0, ddof=0) + float(eps))
    if not np.isfinite(z_map).all():
        raise ValueError("Z_map contains NaN/Inf")
    return z_map


def top_fraction_mask(z_map: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(z_map, dtype=np.float64)
    if values.shape != EXPECTED_BLOCK_SHAPE[1:] or not np.isfinite(values).all():
        raise ValueError("Z_map must be finite with shape 128x501")
    if not 0.0 < fraction < 1.0:
        raise ValueError("ROI fraction must be between 0 and 1")
    count = int(math.ceil(values.size * float(fraction)))
    order = np.argsort(-values.ravel(), kind="mergesort")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(values.shape)


def roi_mean4_features(X_arcsinh: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(X_arcsinh, dtype=np.float64)
    checked = np.asarray(mask, dtype=bool)
    if values.ndim != 4 or checked.shape != values.shape[2:] or not checked.any():
        raise ValueError("clean4/mask shape mismatch or empty ROI")
    return values[:, :, checked].mean(axis=2)


def roi_flat4_features(X_arcsinh: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(X_arcsinh, dtype=np.float64)
    checked = np.asarray(mask, dtype=bool)
    if values.ndim != 4 or checked.shape != values.shape[2:] or not checked.any():
        raise ValueError("clean4/mask shape mismatch or empty ROI")
    selected = values[:, :, checked]
    return selected.reshape(len(values), -1)


def whole_brain_flat4_features(X_arcsinh: np.ndarray) -> np.ndarray:
    values = np.asarray(X_arcsinh, dtype=np.float64)
    if values.ndim != 4 or tuple(values.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise ValueError("whole-brain input must be clean4")
    return values.reshape(len(values), -1)


def fit_predict_rlda(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    train = (X_train - mean) / scale
    test = (X_test - mean) / scale
    model = LDAModel(reg=LDA_REG).fit(train, y_train)
    prediction = model.predict(test).astype(np.int64, copy=False)
    if model.means_ is None or model.inv_cov_ is None or model.priors_ is None:
        raise AssertionError("regularized LDA fit did not produce model parameters")
    linear = test @ model.inv_cov_ @ model.means_.T
    quadratic = 0.5 * np.sum((model.means_ @ model.inv_cov_) * model.means_, axis=1)
    scores = linear - quadratic + np.log(model.priors_)
    decision = scores[:, 1] - scores[:, 0]
    return prediction, decision.astype(np.float64, copy=False)


def mask_overlap(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[float, float]:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes differ")
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    jaccard = intersection / union if union else 1.0
    denominator = int(a.sum() + b.sum())
    dice = 2.0 * intersection / denominator if denominator else 1.0
    return float(jaccard), float(dice)


def summarize_stability(session: str, rule: str, masks: Sequence[np.ndarray]) -> dict[str, Any]:
    sizes = np.asarray([int(mask.sum()) for mask in masks], dtype=float)
    pairwise = [mask_overlap(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
    jaccard = np.asarray([value[0] for value in pairwise], dtype=float)
    dice = np.asarray([value[1] for value in pairwise], dtype=float)
    return {
        "session": str(session),
        "roi_rule": rule,
        "n_folds": len(masks),
        "mean_roi_size_pixels": float(sizes.mean()),
        "std_roi_size_pixels": float(sizes.std(ddof=1)) if len(sizes) > 1 else 0.0,
        "mean_roi_fraction": float(sizes.mean() / np.prod(EXPECTED_BLOCK_SHAPE[1:])),
        "n_fold_pairs": len(pairwise),
        "mean_pairwise_jaccard": float(jaccard.mean()) if len(jaccard) else 1.0,
        "std_pairwise_jaccard": float(jaccard.std(ddof=1)) if len(jaccard) > 1 else 0.0,
        "mean_pairwise_dice": float(dice.mean()) if len(dice) else 1.0,
        "std_pairwise_dice": float(dice.std(ddof=1)) if len(dice) > 1 else 0.0,
    }


def _baseline_predictions_for_session(
    table: pd.DataFrame | None, data: BlockSequenceData, splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    base = {
        "session": data.session,
        "artifact": "",
        "artifact_sha256": "",
        "compatible": False,
        "reused": False,
        "fallback_rerun": True,
        "reason": "compatible artifact not supplied",
    }
    if table is None or table.empty:
        return None, base
    subset = table[
        (table["session"].astype(str) == str(data.session))
        & (table["task"].astype(str) == "binary")
        & (table["method"].astype(str) == "pca_lda_flat4")
    ].copy()
    if "seed" in subset:
        subset = subset[pd.to_numeric(subset["seed"], errors="coerce").fillna(0).astype(int) == 0]
    expected_ids = data.metadata["block_id"].astype(str).tolist()
    if len(subset) != len(expected_ids) or set(subset["block_id"].astype(str)) != set(expected_ids):
        base["reason"] = "sample IDs/count are incompatible"
        return None, base
    lookup = subset.set_index(subset["block_id"].astype(str))
    ordered = lookup.loc[expected_ids]
    truth = pd.to_numeric(ordered["truth"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(truth, data.y):
        base["reason"] = "truth labels are incompatible"
        return None, base
    predictions = pd.to_numeric(ordered["pred"], errors="raise").to_numpy(dtype=np.int64)
    for fold, (_, test_idx) in enumerate(splits, start=1):
        artifact_folds = pd.to_numeric(ordered.iloc[test_idx]["fold"], errors="raise").to_numpy(dtype=int)
        if not np.all(artifact_folds == fold):
            base["reason"] = "fold assignment is incompatible"
            return None, base
    base.update({"compatible": True, "reused": True, "fallback_rerun": False, "reason": "exact sample/truth/fold match"})
    return predictions, base


def _append_predictions(
    rows: list[dict[str, Any]], *, session: str, model: str, fold: int,
    indices: np.ndarray, data: BlockSequenceData, predictions: np.ndarray,
) -> None:
    for local, sample_idx in enumerate(indices):
        meta = data.metadata.iloc[int(sample_idx)]
        rows.append({
            "session": session,
            "model": model,
            "fold": fold,
            "sample_index": int(sample_idx),
            "block_id": str(meta["block_id"]),
            "cycle": int(data.groups[sample_idx]),
            "block_name": str(meta["block_name"]),
            "truth": int(data.y[sample_idx]),
            "prediction": int(predictions[local]),
        })


def run_session(
    data: BlockSequenceData,
    *,
    models: Sequence[str] = MODELS,
    roi_rules: Mapping[str, float] = ROI_RULES,
    max_folds: int = MAX_FOLDS,
    baseline_predictions: pd.DataFrame | None = None,
) -> SessionResult:
    if data.task != "binary" or str(data.session) not in SESSIONS:
        raise ValueError("v10 requires one frozen binary session")
    orientation = FIXED_ORIENTATIONS[str(data.session)]
    X = orient_clean4(data.X, orientation)
    X_asinh = arcsinh_clean4(X)
    splits = grouped_cv_splits(data.groups, max_folds=max_folds)
    reused_baseline, baseline_audit = _baseline_predictions_for_session(baseline_predictions, data, splits)
    baseline_oof = np.full(len(data.y), -1, dtype=np.int64)
    model_oof = {model: np.full(len(data.y), -1, dtype=np.int64) for model in models}
    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []
    masks_by_rule: dict[str, list[np.ndarray]] = {rule: [] for rule in roi_rules}
    representative: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    full_flat = whole_brain_flat4_features(X_asinh)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_cycles = np.unique(data.groups[train_idx]).astype(np.int64)
        test_cycles = np.unique(data.groups[test_idx]).astype(np.int64)
        if set(train_cycles.tolist()) & set(test_cycles.tolist()):
            raise AssertionError("cycle leakage")
        # ROI selection is defined on raw frozen clean4. The established
        # arcsinh transform belongs only to the decoding representations.
        maps, roi_fit_cycles = cycle_response_maps(X, data.metadata, data.groups, train_cycles)
        if not np.array_equal(np.sort(roi_fit_cycles), np.sort(train_cycles)):
            raise AssertionError("ROI fit cycles differ from training cycles")
        if set(roi_fit_cycles.tolist()) & set(test_cycles.tolist()):
            raise AssertionError("test cycles participated in ROI selection")
        z_map = training_z_map(maps)
        background = X[train_idx].mean(axis=(0, 1))

        fold_predictions: dict[str, np.ndarray] = {}
        if "whole_brain_clean4_flat4_pca_lda" in models:
            if reused_baseline is not None:
                prediction = reused_baseline[test_idx]
                n_components: int | str = "REUSED_ARTIFACT"
            else:
                prediction, n_components = fit_predict_linear(
                    "pca_lda", full_flat[train_idx], data.y[train_idx], full_flat[test_idx],
                    pca_variance=PCA_VARIANCE, standardize=True,
                )
            fold_predictions["whole_brain_clean4_flat4_pca_lda"] = prediction
            baseline_oof[test_idx] = prediction
            fold_rows.append({
                "session": data.session, "fold": fold, "model": "whole_brain_clean4_flat4_pca_lda",
                "roi_rule": "NONE", "n_train": len(train_idx), "n_test": len(test_idx),
                "train_cycles": cycle_text(train_cycles), "test_cycles": cycle_text(test_cycles),
                "n_components": n_components, **classification_metrics(data.y[test_idx], prediction),
            })

        for rule, fraction in roi_rules.items():
            mask = top_fraction_mask(z_map, fraction)
            masks_by_rule[rule].append(mask)
            if fold == 1:
                representative[rule] = (background, mask.copy())
            roi_values = z_map[mask]
            outside = z_map[~mask]
            roi_rows.append({
                "session": data.session,
                "fold": fold,
                "roi_rule": rule,
                "roi_fraction_rule": fraction,
                "roi_size_pixels": int(mask.sum()),
                "roi_fraction": float(mask.mean()),
                "mean_Z_within_roi": float(roi_values.mean()),
                "mean_Z_outside_roi": float(outside.mean()),
                "train_cycles": cycle_text(train_cycles),
                "test_cycles": cycle_text(test_cycles),
                "roi_fit_cycles": cycle_text(roi_fit_cycles),
                "test_cycles_used_for_roi": False,
                "full_session_roi_used": False,
                "user_drawn_roi_used": False,
                "v9_map_used_for_roi": False,
                "cross_session_transfer": False,
                "registration_used": False,
                "morphology_used": False,
                "orientation": orientation,
                "z_map_eps": EPS,
            })
            suffix = "top10" if np.isclose(fraction, 0.10) else "top20"
            mean_model = f"roi_mean4_{suffix}_rlda"
            flat_model = f"roi_flat4_{suffix}_pca_lda"
            if mean_model in models:
                features = roi_mean4_features(X_asinh, mask)
                prediction, _ = fit_predict_rlda(features[train_idx], data.y[train_idx], features[test_idx])
                fold_predictions[mean_model] = prediction
                fold_rows.append({
                    "session": data.session, "fold": fold, "model": mean_model, "roi_rule": rule,
                    "n_train": len(train_idx), "n_test": len(test_idx), "train_cycles": cycle_text(train_cycles),
                    "test_cycles": cycle_text(test_cycles), "n_components": 4,
                    **classification_metrics(data.y[test_idx], prediction),
                })
            if flat_model in models:
                features = roi_flat4_features(X_asinh, mask)
                prediction, n_components = fit_predict_linear(
                    "pca_lda", features[train_idx], data.y[train_idx], features[test_idx],
                    pca_variance=PCA_VARIANCE, standardize=True,
                )
                fold_predictions[flat_model] = prediction
                fold_rows.append({
                    "session": data.session, "fold": fold, "model": flat_model, "roi_rule": rule,
                    "n_train": len(train_idx), "n_test": len(test_idx), "train_cycles": cycle_text(train_cycles),
                    "test_cycles": cycle_text(test_cycles), "n_components": n_components,
                    **classification_metrics(data.y[test_idx], prediction),
                })

        for model, prediction in fold_predictions.items():
            if np.any(model_oof[model][test_idx] != -1):
                raise AssertionError("OOF sample predicted twice")
            model_oof[model][test_idx] = prediction
            _append_predictions(
                oof_rows, session=data.session, model=model, fold=fold,
                indices=test_idx, data=data, predictions=prediction,
            )

    summary_rows: list[dict[str, Any]] = []
    for model in models:
        if np.any(model_oof[model] < 0):
            raise AssertionError(f"incomplete OOF predictions for {model}")
        metric = classification_metrics(data.y, model_oof[model])
        summary_rows.append({
            "session": data.session,
            "model": model,
            "n_samples": len(data.y),
            "n_cycles": data.n_cycles,
            "n_folds": len(splits),
            **metric,
        })
        if not np.isclose(metric["accuracy"], metric["balanced_accuracy"], atol=1.0e-12):
            raise AssertionError("balanced binary design must have accuracy == balanced accuracy")
    stability = [summarize_stability(data.session, rule, masks) for rule, masks in masks_by_rule.items()]
    counts = dict(zip(*np.unique(data.y, return_counts=True)))
    class_row = {
        "session": data.session,
        "n_samples": len(data.y),
        "n_cycles": data.n_cycles,
        "n_no_stimulus": int(counts.get(0, 0)),
        "n_stimulus": int(counts.get(1, 0)),
        "class_ratio_stimulus": float(counts.get(1, 0) / len(data.y)),
        "accuracy_equals_balanced_accuracy_expected": True,
        "class_balance_exact_1_to_1": int(counts.get(0, 0)) == int(counts.get(1, 0)),
        "all_folds_cycle_grouped": True,
        "orientation": orientation,
    }
    return SessionResult(
        summary_rows, fold_rows, oof_rows, roi_rows, class_row, stability,
        baseline_audit, representative,
    )


def resolve_baseline_predictions(root: Path | None) -> tuple[pd.DataFrame | None, Path | None]:
    if root is None:
        return None, None
    candidates = (
        Path(root) / "aggregate/multiframe_all_models_predictions.csv",
        Path(root) / "multiframe_all_models_predictions.csv",
    )
    for path in candidates:
        if path.is_file():
            return pd.read_csv(path, dtype={"session": str}), path
    return None, None


def load_v9_metrics(root: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_columns = ["requested_root", "resolved_path", "exists", "sha256", "status", "reason"]
    if root is None:
        return pd.DataFrame(), pd.DataFrame([{
            "requested_root": "", "resolved_path": "", "exists": False, "sha256": "",
            "status": "MISSING_SAFE_DEGRADE", "reason": "v9 root not supplied",
        }], columns=audit_columns)
    candidates = (
        Path(root) / "summaries/session_spatial_diagnostic_table.csv",
        Path(root) / "spatial_glm_contrast_reproducibility_9sessions_v9/summaries/session_spatial_diagnostic_table.csv",
    )
    for path in candidates:
        if path.is_file():
            table = pd.read_csv(path, dtype={"session": str})
            required = {"session", "binary_RMS_standardized_effect", "binary_split_half_corr_median", "GS_DS_spatial_corr"}
            if not required.issubset(table.columns) or set(table["session"]) != set(SESSIONS):
                continue
            audit = pd.DataFrame([{
                "requested_root": str(root), "resolved_path": str(path.resolve()), "exists": True,
                "sha256": sha256_file(path), "status": "PASS", "reason": "frozen v9 session diagnostic table reused",
            }], columns=audit_columns)
            return table, audit
    return pd.DataFrame(), pd.DataFrame([{
        "requested_root": str(root), "resolved_path": "", "exists": False, "sha256": "",
        "status": "MISSING_SAFE_DEGRADE", "reason": "compatible v9 summary not found",
    }], columns=audit_columns)


def exact_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) != 9 or len(y_values) != 9 or not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        return {"rho": np.nan, "permutation_p_two_sided": np.nan, "n_permutations": 0, "status": "UNAVAILABLE"}
    rx = rankdata(x_values)
    ry = rankdata(y_values)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.linalg.norm(rx) * np.linalg.norm(ry)
    observed = float(np.dot(rx, ry) / denominator) if denominator else np.nan
    extreme = 0
    total = 0
    for permutation in itertools.permutations(range(9)):
        rho = float(np.dot(rx, ry[np.asarray(permutation)]) / denominator) if denominator else np.nan
        extreme += int(abs(rho) >= abs(observed) - 1.0e-12)
        total += 1
    return {
        "rho": observed,
        "permutation_p_two_sided": float(extreme / total),
        "n_permutations": total,
        "status": "PASS_EXACT_9_FACTORIAL",
    }


def link_v9(summary: pd.DataFrame, v9: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = summary.pivot(index="session", columns="model", values="balanced_accuracy").reset_index()
    baseline = "whole_brain_clean4_flat4_pca_lda"
    wide["roi_mean4_top10_gain"] = wide["roi_mean4_top10_rlda"] - wide[baseline]
    wide["roi_flat4_top10_gain"] = wide["roi_flat4_top10_pca_lda"] - wide[baseline]
    if v9.empty:
        linked = wide.copy()
        for column in ("binary_RMS_standardized_effect", "binary_split_half_corr_median", "GS_DS_spatial_corr"):
            linked[column] = np.nan
    else:
        linked = wide.merge(v9[[
            "session", "binary_RMS_standardized_effect", "binary_split_half_corr_median", "GS_DS_spatial_corr"
        ]], on="session", how="left", validate="one_to_one")
    rows = []
    for outcome in ("roi_mean4_top10_gain", "roi_flat4_top10_gain"):
        for predictor in ("binary_RMS_standardized_effect", "binary_split_half_corr_median"):
            result = exact_spearman(linked[predictor], linked[outcome])
            rows.append({"outcome": outcome, "predictor": predictor, "n_sessions": 9, **result})
    return linked, pd.DataFrame(rows)


def config_freeze_text(config_path: Path) -> str:
    return f"""# v10 configuration freeze

- Config: {config_path}
- Sessions: {', '.join(SESSIONS)}; none may be excluded
- Task: within-session stimulus-presence binary only
- Input: frozen block_sequences_v1 clean4, one sample = 4 x 128 x 501
- Orientation: {FIXED_ORIENTATIONS}; 807 uses fixed flip_vertical
- CV: cycle-grouped, at most {MAX_FOLDS} folds
- Functional ROI fit scope: training cycles only, newly fit for every session x fold
- Cycle map: raw frozen clean4 block-frame means, 0.5*(grating + dot) - 0.5*(stop_after_grating + static)
- Ranking map: training-cycle mean / (training-cycle std ddof=0 + {EPS})
- Primary ROI: exact highest 10% Z pixels; sensitivity: exact highest 20%; ties resolved stably by flat index
- Morphology/smoothing/manual editing: none
- Models: {', '.join(MODELS)}
- PCA: train-fold only, variance={PCA_VARIANCE}; LDA ridge={LDA_REG}
- User-drawn ROI: not used; v9 maps: not used for ROI; test cycles: not used for ROI
- Cross-session transfer/registration: none
- v9 linkage: frozen session-level metrics only after decoding, mechanistic secondary analysis
- Device: CPU; results are device-independent
"""


def _scaled(image: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(image[np.isfinite(image)], [1, 99])
    return np.clip((image - lo) / max(hi - lo, 1e-12), 0, 1)


def plot_roi_overview(
    representatives: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]], rule: str, path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(16, 8), constrained_layout=True)
    for ax, session in zip(axes.flat, SESSIONS):
        background, mask = representatives[session][rule]
        ax.imshow(_scaled(background), cmap="gray", vmin=0, vmax=1, aspect="auto", origin="upper")
        ax.contour(mask.astype(float), levels=[0.5], colors=["#ff3b30"], linewidths=1.0)
        ax.set_title(f"{session} | representative fold 1")
        ax.set_axis_off()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(
    summary: pd.DataFrame, stability: pd.DataFrame, linked: pd.DataFrame,
    representatives: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]], output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_roi_overview(representatives, "functional_roi_top10", figures / "functional_roi_overview_top10.png")
    plot_roi_overview(representatives, "functional_roi_top20", figures / "functional_roi_overview_top20.png")
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b"]
    pivot = summary.pivot(index="session", columns="model", values="balanced_accuracy").reindex(SESSIONS)
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(SESSIONS)); width = 0.16
    for i, model in enumerate(MODELS):
        ax.bar(x + (i - 2) * width, pivot[model], width, label=model, color=colors[i])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, SESSIONS); ax.set_ylim(0, 1); ax.set_ylabel("OOF balanced accuracy"); ax.legend(fontsize=7)
    fig.savefig(figures / "within_session_binary_roi_vs_wholebrain.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    width = 0.35
    ax.bar(x - width / 2, linked.set_index("session").reindex(SESSIONS)["roi_mean4_top10_gain"], width, label="ROI mean4 top10")
    ax.bar(x + width / 2, linked.set_index("session").reindex(SESSIONS)["roi_flat4_top10_gain"], width, label="ROI flat4 top10")
    ax.axhline(0, color="black", linewidth=1); ax.set_xticks(x, SESSIONS); ax.set_ylabel("BA gain vs whole brain"); ax.legend()
    fig.savefig(figures / "roi_gain_top10_by_session.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for rule, color in zip(ROI_RULES, ("#e45756", "#72b7b2")):
        part = stability[stability["roi_rule"] == rule].set_index("session").reindex(SESSIONS)
        axes[0].plot(SESSIONS, part["mean_roi_fraction"], marker="o", label=rule, color=color)
        axes[1].plot(SESSIONS, part["mean_pairwise_jaccard"], marker="o", label=f"{rule} Jaccard", color=color)
        axes[1].plot(SESSIONS, part["mean_pairwise_dice"], marker="s", linestyle="--", label=f"{rule} Dice", color=color)
    axes[0].set_ylabel("ROI fraction"); axes[1].set_ylabel("Fold overlap"); axes[1].set_ylim(0, 1)
    for ax in axes: ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=45)
    fig.savefig(figures / "roi_size_stability_by_session.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (gain, metric) in zip(axes.flat, itertools.product(
        ("roi_mean4_top10_gain", "roi_flat4_top10_gain"),
        ("binary_RMS_standardized_effect", "binary_split_half_corr_median"),
    )):
        if linked[metric].notna().all():
            ax.scatter(linked[metric], linked[gain])
            for _, row in linked.iterrows(): ax.annotate(str(row["session"]), (row[metric], row[gain]), fontsize=7)
        else:
            ax.text(0.5, 0.5, "v9 metric unavailable", ha="center", va="center", transform=ax.transAxes)
        ax.axhline(0, color="black", linewidth=0.8); ax.set_xlabel(metric); ax.set_ylabel(gain)
    fig.savefig(figures / "roi_gain_vs_v9_metrics.png", dpi=160); plt.close(fig)


def choose_scenario(summary: pd.DataFrame) -> str:
    pivot = summary.pivot(index="session", columns="model", values="balanced_accuracy")
    mean_gain = pivot["roi_mean4_top10_rlda"] - pivot["whole_brain_clean4_flat4_pca_lda"]
    flat_gain = pivot["roi_flat4_top10_pca_lda"] - pivot["whole_brain_clean4_flat4_pca_lda"]
    if mean_gain.mean() > 0 and flat_gain.mean() <= 0:
        return "R3"
    if flat_gain.mean() > 0 and mean_gain.mean() <= 0:
        return "R4"
    weak_best = pd.concat([mean_gain.loc[list(WEAK_SESSIONS)], flat_gain.loc[list(WEAK_SESSIONS)]], axis=1).max(axis=1)
    strong_best = pd.concat([mean_gain.loc[list(STRONG_SESSIONS)], flat_gain.loc[list(STRONG_SESSIONS)]], axis=1).max(axis=1)
    if (weak_best > 0).sum() >= 4:
        return "R1"
    if strong_best.mean() > 0 and weak_best.mean() <= 0:
        return "R2"
    return "R5"


def make_report(summary: pd.DataFrame, stability: pd.DataFrame, associations: pd.DataFrame, v9_audit: pd.DataFrame) -> str:
    scenario = choose_scenario(summary)
    lines = [
        "# Nested functional ROI decoding (9 sessions) v10", "",
        "This experiment uses training-fold data-driven functional ROIs. It is not an expert or anatomical ROI analysis.", "",
        "## Leakage controls", "",
        "- A new ROI was fit for every session x CV fold from training cycles only.",
        "- Test cycles, user-drawn ROIs, full-session v9 maps, registration, and cross-session transfer were not used for ROI selection.", "",
        "## OOF balanced accuracy", "",
        "```text", summary.pivot(index="session", columns="model", values="balanced_accuracy").reindex(SESSIONS).to_string(), "```", "",
        "## ROI stability", "", "```text", stability.to_string(index=False), "```", "",
        "## v9 mechanistic linkage", "",
        f"v9 reuse status: {v9_audit.iloc[0]['status']}.", "",
        "```text", associations.to_string(index=False), "```", "",
        "## Preregistered interpretation", "", f"Closest scenario: **{scenario}**.", "",
        "No claim of expert ROI, anatomical correspondence, or cross-session alignment is supported.",
    ]
    return "\n".join(lines) + "\n"


def expected_outputs(output_dir: Path) -> list[Path]:
    return [Path(output_dir) / relative for relative in REQUIRED_OUTPUTS]


def run_formal(
    *, project_root: Path, data_dir: Path, output_dir: Path, config_path: Path,
    baseline_root: Path | None, v9_root: Path | None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    for name in ("audit", "summaries", "figures", "report"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    baseline_table, baseline_path = resolve_baseline_predictions(baseline_root)
    v9, v9_audit = load_v9_metrics(v9_root)
    results = []
    representatives: dict[str, Mapping[str, tuple[np.ndarray, np.ndarray]]] = {}
    for session in SESSIONS:
        data = load_block_sequence_session(project_root, session, "binary", data_dir=data_dir)
        result = run_session(data, baseline_predictions=baseline_table)
        result.baseline_audit_row["artifact"] = str(baseline_path.resolve()) if baseline_path else ""
        result.baseline_audit_row["artifact_sha256"] = sha256_file(baseline_path) if baseline_path else ""
        results.append(result); representatives[session] = result.representative
    summary = pd.DataFrame([row for result in results for row in result.summary_rows])
    folds = pd.DataFrame([row for result in results for row in result.fold_rows])
    oof = pd.DataFrame([row for result in results for row in result.oof_rows])
    roi_audit = pd.DataFrame([row for result in results for row in result.roi_audit_rows])
    balance = pd.DataFrame([result.class_balance_row for result in results])
    stability = pd.DataFrame([row for result in results for row in result.stability_rows])
    baseline_audit = pd.DataFrame([result.baseline_audit_row for result in results])
    linked, associations = link_v9(summary, v9)
    balance.to_csv(output_dir / "audit/class_balance_audit.csv", index=False)
    roi_audit.to_csv(output_dir / "audit/fold_roi_audit.csv", index=False)
    baseline_audit.to_csv(output_dir / "audit/baseline_reuse_audit.csv", index=False)
    v9_audit.to_csv(output_dir / "audit/v9_metric_reuse_audit.csv", index=False)
    (output_dir / "audit/config_freeze.md").write_text(config_freeze_text(config_path), encoding="utf-8")
    summary.to_csv(output_dir / "summaries/within_session_roi_decoding_summary.csv", index=False)
    folds.to_csv(output_dir / "summaries/fold_level_roi_results.csv", index=False)
    oof.to_csv(output_dir / "summaries/oof_predictions.csv", index=False)
    stability.to_csv(output_dir / "summaries/roi_stability_summary.csv", index=False)
    linked.to_csv(output_dir / "summaries/roi_gain_vs_v9_metrics.csv", index=False)
    associations.to_csv(output_dir / "summaries/roi_gain_vs_v9_associations.csv", index=False)
    make_figures(summary, stability, linked, representatives, output_dir)
    (output_dir / "report/nested_functional_roi_decoding_report.md").write_text(
        make_report(summary, stability, associations, v9_audit), encoding="utf-8"
    )
    scientific_missing = [str(path) for path in expected_outputs(output_dir) if not path.exists() and path.name not in {
        "pytest_output_local.txt", "smoke_test_local.txt", "run_command_server.txt", "run_log_server.txt"
    }]
    if scientific_missing:
        raise AssertionError(f"formal output completeness failed: {scientific_missing}")
    return {"sessions": 9, "summary_rows": len(summary), "scenario": choose_scenario(summary)}
