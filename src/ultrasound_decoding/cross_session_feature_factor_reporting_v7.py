"""Figures and report for cross-session feature/factor analysis v7."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ultrasound_decoding.cross_session_feature_factor_v7 import (
    CONDITION_TIME_WARNING,
    EXPECTED_SESSIONS,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    fit_pca,
)


SESSION_COLORS = dict(zip(EXPECTED_SESSIONS, plt.cm.tab10(np.linspace(0, 0.9, len(EXPECTED_SESSIONS)))))


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def categorical_pca_scatter(
    coordinates: np.ndarray,
    metadata: pd.DataFrame,
    *,
    column: str,
    path: Path,
    title: str,
    explained_ratio: Sequence[float] | None = None,
) -> None:
    values = metadata[column].astype(str).to_numpy()
    levels = sorted(np.unique(values).tolist(), key=(lambda x: int(x)) if column == "session" else str)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    palette = SESSION_COLORS if column == "session" else dict(zip(levels, plt.cm.Set2(np.linspace(0, 1, len(levels)))))
    for level in levels:
        selected = values == level
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=25, alpha=0.72,
                   color=palette[level], label=level, edgecolors="none")
    x_label, y_label = "PC1", "PC2"
    if explained_ratio is not None and len(explained_ratio) >= 2:
        x_label += f" ({100 * explained_ratio[0]:.1f}%)"
        y_label += f" ({100 * explained_ratio[1]:.1f}%)"
    ax.set(xlabel=x_label, ylabel=y_label, title=title)
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.legend(title=column, frameon=False, fontsize=8, ncol=2)
    _save(fig, path)


def masked_seed_scatter(
    features_by_seed: Mapping[int, np.ndarray],
    metadata: pd.DataFrame,
    *,
    column: str,
    path: Path,
    title: str,
) -> None:
    levels = sorted(metadata[column].astype(str).unique(), key=(lambda x: int(x)) if column == "session" else str)
    palette = SESSION_COLORS if column == "session" else dict(zip(levels, plt.cm.Set2(np.linspace(0, 1, len(levels)))))
    fig, axes = plt.subplots(1, len(features_by_seed), figsize=(6 * len(features_by_seed), 5.2), squeeze=False)
    for ax, (seed, features) in zip(axes[0], sorted(features_by_seed.items())):
        pca = fit_pca(features, n_components=2, random_seed=seed)
        coordinates = pca.transform(features)
        values = metadata[column].astype(str).to_numpy()
        for level in levels:
            selected = values == level
            ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=20, alpha=0.7,
                       color=palette[level], label=level, edgecolors="none")
        ax.set_title(f"seed {seed}")
        ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(9, len(levels)), frameon=False)
    fig.suptitle(title + "\nGLOBAL_MASKED_SMALLCNN: descriptive label-free common representation")
    fig.subplots_adjust(bottom=0.18, top=0.84, wspace=0.25)
    _save(fig, path)


def distance_heatmap(distance: pd.DataFrame, *, path: Path, title: str) -> None:
    sessions = list(EXPECTED_SESSIONS)
    matrix = np.zeros((len(sessions), len(sessions)), dtype=float)
    index = {session: i for i, session in enumerate(sessions)}
    for row in distance.itertuples():
        i, j = index[str(row.session_a)], index[str(row.session_b)]
        matrix[i, j] = matrix[j, i] = float(row.energy_distance)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(matrix, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(sessions)), sessions, rotation=45, ha="right")
    ax.set_yticks(range(len(sessions)), sessions)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="multivariate energy distance")
    _save(fig, path)


def confusion_heatmap(matrix: np.ndarray, *, path: Path) -> None:
    sessions = list(EXPECTED_SESSIONS)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(sessions)):
        for j in range(len(sessions)):
            ax.text(j, i, f"{matrix[i, j]}\n{normalized[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if normalized[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(sessions)), sessions, rotation=45, ha="right")
    ax.set_yticks(range(len(sessions)), sessions)
    ax.set(xlabel="Predicted session", ylabel="True session", title="Cycle-grouped session-ID linear probe")
    fig.colorbar(image, ax=ax, label="row-normalized fraction")
    _save(fig, path)


def stimulus_probe_plot(table: pd.DataFrame, *, path: Path) -> None:
    selected = table[table["target"].astype(str).isin(EXPECTED_SESSIONS)].copy()
    selected["target"] = pd.Categorical(selected["target"].astype(str), EXPECTED_SESSIONS, ordered=True)
    selected = selected.sort_values("target")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(selected["target"].astype(str), selected["BA"], color=[SESSION_COLORS[str(s)] for s in selected["target"]])
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="binary chance reference")
    ax.set_ylim(0, 1)
    ax.set(ylabel="Balanced accuracy", xlabel="Unseen target session",
           title="Strict source-only RAW_PCA stimulus-presence probe")
    ax.legend(frameon=False)
    _save(fig, path)


def factor_variance_plot(table: pd.DataFrame, *, path: Path, title: str) -> None:
    selected = table[
        ((table["representation"] == "RAW_SPATIAL_PCA") & (table["seed"].astype(str).isin(("", "nan"))))
        | ((table["representation"] == "GLOBAL_MASKED_SMALLCNN") & (table["seed"].astype(str) == "MEAN_3_SEEDS"))
    ].copy()
    if selected.empty:
        selected = table.groupby(["representation", "factor"], as_index=False).first()
    factor_order = [value for value in ("session", "stimulus_presence", "condition4", "session_x_stimulus_presence", "session_x_condition4", "residual")
                    if value in set(selected["factor"])]
    representations = [value for value in ("RAW_SPATIAL_PCA", "GLOBAL_MASKED_SMALLCNN") if value in set(selected["representation"])]
    x = np.arange(len(factor_order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for i, representation in enumerate(representations):
        subset = selected[selected["representation"] == representation].set_index("factor").reindex(factor_order)
        values = subset["median_R2"].to_numpy(float)
        lower = values - subset["ci_2_5"].to_numpy(float)
        upper = subset["ci_97_5"].to_numpy(float) - values
        ax.bar(x + (i - 0.5) * width, values, width, yerr=np.vstack((lower, upper)), capsize=3,
               label=representation)
    ax.set_xticks(x, [value.replace("session_x_", "session ×\n") for value in factor_order])
    ax.set(ylabel="R²-like multivariate variance proportion", title=title)
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    _save(fig, path)


def distance_performance_plot(diagnostic: pd.DataFrame, associations: pd.DataFrame, *, path: Path) -> None:
    reps = [value for value in ("RAW_SPATIAL_PCA", "GLOBAL_MASKED_SMALLCNN") if f"mean_distance_{value}" in diagnostic.columns]
    fig, axes = plt.subplots(1, len(reps), figsize=(6.3 * max(len(reps), 1), 5.2), squeeze=False)
    for ax, representation in zip(axes[0], reps):
        x = diagnostic[f"mean_distance_{representation}"].to_numpy(float)
        y = diagnostic["v5_cross_session_BA"].to_numpy(float)
        ax.scatter(x, y, s=60, color="#3569a8")
        for row, xv, yv in zip(diagnostic.itertuples(), x, y):
            ax.annotate(str(row.session), (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=8)
        assoc = associations[
            (associations["analysis"] == "target_outlier_distance_vs_v5_BA")
            & (associations["representation"] == representation)
        ].iloc[0]
        ax.set(xlabel="Mean energy distance to other sessions", ylabel="v5 unseen-session BA",
               title=f"{representation}\nSpearman ρ={assoc.rho:.2f}, perm. p={assoc.permutation_p_two_sided:.3g}")
    _save(fig, path)


def weak_strong_plot(table: pd.DataFrame, *, path: Path) -> None:
    ordered = table.copy()
    ordered["session"] = pd.Categorical(ordered["session"].astype(str), EXPECTED_SESSIONS, ordered=True)
    ordered = ordered.sort_values("session")
    metrics = [
        ("within_session_BA", "Within-session BA"),
        ("cycle_consistency_mean", "Cycle consistency (mean r)"),
        ("separability_ratio", "Binary separability ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = ["#d36b59" if str(value) in WEAK_SESSIONS else "#3a78b4" for value in ordered["session"]]
    for ax, (column, label) in zip(axes, metrics):
        ax.bar(ordered["session"].astype(str), ordered[column], color=colors)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(label)
    fig.suptitle("Historical weak/strong labels are descriptive annotations only")
    _save(fig, path)


def diagnostic_overview(table: pd.DataFrame, *, path: Path) -> None:
    columns = [
        "within_session_BA", "v5_cross_session_BA", "cycle_consistency_mean",
        "separability_ratio", "mean_distance_RAW_SPATIAL_PCA",
    ]
    labels = ["Within BA", "v5 cross BA", "Cycle r", "Separability", "Mean raw distance"]
    ordered = table.set_index(table["session"].astype(str)).reindex(EXPECTED_SESSIONS)
    values = ordered[columns].to_numpy(float)
    scale = np.nanstd(values, axis=0)
    z = (values - np.nanmean(values, axis=0)) / np.where(scale > 0, scale, 1)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    image = ax.imshow(z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(z[i, j]) > 1.3 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(EXPECTED_SESSIONS)), EXPECTED_SESSIONS)
    ax.set_title("Session diagnostic overview (color = column-wise z-score; text = raw value)")
    fig.colorbar(image, ax=ax, label="within-column z-score")
    _save(fig, path)


def report_markdown(
    *,
    metadata_audit: pd.DataFrame,
    session_probe: pd.DataFrame,
    stimulus_probe: pd.DataFrame,
    factor_binary: pd.DataFrame,
    factor_condition: pd.DataFrame,
    associations: pd.DataFrame,
    diagnostic: pd.DataFrame,
    pairwise_distance: pd.DataFrame,
    pairwise_available: bool,
    pca_explained_ratio: Sequence[float],
) -> str:
    def metric(table: pd.DataFrame, factor: str, rep: str) -> float:
        selected = table[(table["factor"] == factor) & (table["representation"] == rep)]
        if rep == "GLOBAL_MASKED_SMALLCNN":
            aggregate = selected[selected["seed"].astype(str) == "MEAN_3_SEEDS"]
            if len(aggregate):
                selected = aggregate
        return float(selected.iloc[0]["median_R2"])

    session_ba = float(session_probe.loc[session_probe["metric"] == "balanced_accuracy", "observed"].iloc[0])
    stimulus_ba = float(stimulus_probe.loc[stimulus_probe["target"] == "MEAN_9_TARGETS", "BA"].iloc[0])
    raw_session = metric(factor_binary, "session", "RAW_SPATIAL_PCA")
    raw_stim = metric(factor_binary, "stimulus_presence", "RAW_SPATIAL_PCA")
    raw_inter = metric(factor_binary, "session_x_stimulus_presence", "RAW_SPATIAL_PCA")
    masked_session = metric(factor_binary, "session", "GLOBAL_MASKED_SMALLCNN")
    masked_stim = metric(factor_binary, "stimulus_presence", "GLOBAL_MASKED_SMALLCNN")
    masked_inter = metric(factor_binary, "session_x_stimulus_presence", "GLOBAL_MASKED_SMALLCNN")
    dominant = raw_session > raw_stim and masked_session > masked_stim
    interaction = raw_inter > raw_stim or masked_inter > masked_stim
    available = metadata_audit[metadata_audit["available"] == True]["factor"].tolist()  # noqa: E712
    unavailable = metadata_audit[metadata_audit["available"] == False]["factor"].tolist()  # noqa: E712
    largest = pairwise_distance.sort_values("energy_distance").iloc[-1]
    smallest = pairwise_distance.sort_values("energy_distance").iloc[0]
    assoc_raw = associations[
        (associations["analysis"] == "target_outlier_distance_vs_v5_BA")
        & (associations["representation"] == "RAW_SPATIAL_PCA")
    ].iloc[0]
    weak_mean = diagnostic[diagnostic["session"].astype(str).isin(WEAK_SESSIONS)][
        ["separability_ratio", "cycle_consistency_mean", "within_session_BA"]
    ].mean()
    strong_mean = diagnostic[diagnostic["session"].astype(str).isin(STRONG_SESSIONS)][
        ["separability_ratio", "cycle_consistency_mean", "within_session_BA"]
    ].mean()
    lines = [
        "# Cross-session feature distribution and factor attribution analysis v7",
        "",
        "## Scope and safeguards",
        "",
        "This stage explains the already observed cross-session generalization failure; it is not a new model benchmark. "
        "All nine preregistered sessions and every legal clean4 complete-cycle block are retained. No CSU tuning, "
        "domain-adaptation training, registration, ROI, GLM, searchlight, transformer, Mamba, or new SSL method was run.",
        "",
        "`GLOBAL_MASKED_SMALLCNN = descriptive label-free common representation`. It saw unlabeled images from all sessions "
        "and is never presented as strict unseen-session predictive evidence. The stimulus-presence LOSO result uses "
        "source-only normalization, PCA, scaling, and classifier fitting for each target.",
        "",
        "## Metadata factor audit",
        "",
        f"Available: {', '.join(available)}. NOT_AVAILABLE: {', '.join(unavailable)}. Cycle is nested within session; "
        "n_complete_cycles is a session-level attribute. No absent biological or acquisition metadata was invented.",
        "",
        "## Common representation results",
        "",
        f"RAW PCA PC1/PC2 explain {100*pca_explained_ratio[0]:.2f}% and {100*pca_explained_ratio[1]:.2f}% of variance. "
        f"Cycle-grouped session-ID BA is {session_ba:.3f}; strict source-only unseen-session stimulus BA is {stimulus_ba:.3f}. "
        "These are not expressed as a ratio because one task is 9-class and the other binary.",
        "",
        f"RAW binary variance proportions: session={raw_session:.3f}, stimulus={raw_stim:.3f}, interaction={raw_inter:.3f}. "
        f"GLOBAL_MASKED binary proportions: session={masked_session:.3f}, stimulus={masked_stim:.3f}, interaction={masked_inter:.3f}.",
        "",
        f"The smallest/largest RAW energy-distance pairs are {smallest.session_a}-{smallest.session_b} "
        f"({smallest.energy_distance:.3f}) and {largest.session_a}-{largest.session_b} ({largest.energy_distance:.3f}). "
        f"Target outlier distance versus v5 BA: Spearman rho={assoc_raw.rho:.3f}, "
        f"exact permutation p={assoc_raw.permutation_p_two_sided:.4g}.",
        "",
        "## Condition/time-position decomposition",
        "",
        CONDITION_TIME_WARNING,
        "",
        "## Weak-session diagnostics",
        "",
        f"Historical weak-session means: separability={weak_mean.separability_ratio:.3f}, cycle consistency="
        f"{weak_mean.cycle_consistency_mean:.3f}, within BA={weak_mean.within_session_BA:.3f}. Historical strong-session "
        f"means: separability={strong_mean.separability_ratio:.3f}, cycle consistency="
        f"{strong_mean.cycle_consistency_mean:.3f}, within BA={strong_mean.within_session_BA:.3f}. "
        "This annotation is descriptive only and no subgroup significance test is performed.",
        "",
        "## Preregistered interpretation",
        "",
        ("The two common spaces support a prominent session effect that may obscure transferable stimulus information."
         if dominant else "The two common spaces do not consistently show session variance dominating stimulus variance."),
        ("Session × stimulus structure is prominent relative to the stimulus main effect, supporting session-dependent stimulus representation."
         if interaction else "The session × stimulus component is not prominent relative to the stimulus main effect."),
        ("Larger distribution discrepancy is associated with lower unseen-session decoding; this is association, not causation."
         if assoc_raw.rho < 0 and assoc_raw.permutation_p_two_sided < 0.05
         else "Simple global distribution distance does not significantly explain cross-session decoding failure."),
        f"Comparable 72-directed-pair secondary analysis: {'run with an exact Mantel-style permutation' if pairwise_available else 'PAIRWISE_ANALYSIS_NOT_RUN'}.",
        "",
        "## Relation to the three proposed research directions",
        "",
        "### Direction 1",
        "",
        "This v7 stage directly serves cross-session discrepancy analysis and decoder generalization after registration, multi-source, and CSU work.",
        "",
        "### Direction 2",
        "",
        "If whole-brain session variance dominates task variance, a future preregistered analysis may test whether expert-defined ROI increases the task/session effect ratio. No ROI analysis was added here.",
        "",
        "### Direction 3",
        "",
        "The stimulus-presence and condition/time-position decompositions motivate later spatial activation work using GLM, searchlight, or deep interpretability. None was added here.",
        "",
        "Priority: return first to Direction 1 to address the measured cross-session representation discrepancy; use Direction 2 or 3 only as a separately preregistered follow-up.",
    ]
    return "\n".join(lines) + "\n"
