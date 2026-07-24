from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.linear import fit_predict_linear, preprocess_frames
from ultrasound_decoding.interpretability.common import (
    PatchSpec,
    aggregate_patch_replicates,
    patch_dataframe,
)


def run_pca_lda_searchlight(
    *,
    session: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    patches: list[PatchSpec],
    output_dir: Path,
    pca_variance: float = 0.95,
    max_patches: int | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_patches = patches[:max_patches] if max_patches is not None else patches
    patch_dataframe(patches).to_csv(output_dir / "patch_definitions.csv", index=False)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_cycles = set(np.unique(groups[train_idx]).astype(int).tolist())
        test_cycles = set(np.unique(groups[test_idx]).astype(int).tolist())
        if train_cycles & test_cycles:
            raise AssertionError(f"cycle leakage in searchlight fold {fold}")
        y_train = y[train_idx]
        y_test = y[test_idx]
        for patch in selected_patches:
            X_patch = X[:, patch.row_start : patch.row_end, patch.col_start : patch.col_end]
            X_flat = preprocess_frames(X_patch)
            pred, n_components = fit_predict_linear(
                "pca_lda",
                X_flat[train_idx],
                y_train,
                X_flat[test_idx],
                pca_variance=pca_variance,
                standardize=False,
            )
            metrics = classification_metrics(y_test, pred)
            rows.append(
                {
                    "session": session,
                    "fold": fold,
                    "test_cycles": ",".join(str(x) for x in sorted(test_cycles)),
                    "patch_id": patch.patch_id,
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "n_train_samples": int(len(train_idx)),
                    "n_test_samples": int(len(test_idx)),
                    "pca_n_components": int(n_components),
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "fold_patch_metrics.csv", index=False)

    mean_map, std_map, coverage_count = aggregate_patch_replicates(
        selected_patches,
        metrics_df,
        "balanced_accuracy",
    )
    above = metrics_df.copy()
    above["above_chance"] = above["balanced_accuracy"].astype(float) > 0.5
    fraction_map, _, _ = aggregate_patch_replicates(selected_patches, above, "above_chance")
    np.save(output_dir / "searchlight_ba_mean.npy", mean_map)
    np.save(output_dir / "searchlight_ba_std.npy", std_map)
    np.save(output_dir / "searchlight_above_chance_fraction.npy", fraction_map)
    np.save(output_dir / "searchlight_coverage_count.npy", coverage_count)
    if max_patches is None and int((coverage_count == 0).sum()) != 0:
        raise AssertionError("searchlight aggregation left uncovered pixels")
    if np.isinf(mean_map[np.isfinite(mean_map)]).any():
        raise AssertionError("searchlight map contains Inf")

    patch_summary = (
        metrics_df.groupby("patch_id", as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            above_chance_fraction=("balanced_accuracy", lambda x: float(np.mean(np.asarray(x) > 0.5))),
            mean_pca_n_components=("pca_n_components", "mean"),
        )
        .merge(patch_dataframe(patches), on="patch_id", how="left")
        .sort_values("balanced_accuracy_mean", ascending=False)
        .head(10)
    )
    patch_summary.insert(
        0,
        "ranking_note",
        "descriptive ranking only; not an independently validated best-region performance estimate",
    )
    patch_summary.to_csv(output_dir / "searchlight_summary.csv", index=False)
    return {
        "metrics": metrics_df,
        "mean_map": mean_map,
        "std_map": std_map,
        "above_chance_fraction_map": fraction_map,
        "coverage_count": coverage_count,
        "summary": patch_summary,
    }

