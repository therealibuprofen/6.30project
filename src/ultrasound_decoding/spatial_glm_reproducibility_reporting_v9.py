"""Figures and report for the preregistered v9 spatial analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from ultrasound_decoding.spatial_glm_reproducibility_v9 import (
    CONDITION_ORDER,
    EXPECTED_SESSIONS,
    FIXED_ORDER_WARNING,
    GD_WARNING,
    STRONG_SESSIONS,
    SessionAnalysis,
)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _symmetric_scale(values: Sequence[np.ndarray], percentile: float = 99.0) -> float:
    concatenated = np.concatenate([np.abs(np.asarray(value, dtype=float)).ravel() for value in values])
    scale = float(np.percentile(concatenated[np.isfinite(concatenated)], percentile))
    return max(scale, np.finfo(float).eps)


def _effect_image(ax: plt.Axes, value: np.ndarray, scale: float, title: str) -> None:
    image = ax.imshow(value, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-scale, vmax=scale), aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)


def plot_session_map_panels(
    analyses: Mapping[str, SessionAnalysis],
    output_dir: Path,
) -> None:
    primary_effect_scale = _symmetric_scale([
        analysis.contrasts["STIM_PRESENCE"].effect for analysis in analyses.values()
    ])
    primary_standardized_scale = _symmetric_scale([
        analysis.contrasts["STIM_PRESENCE"].standardized for analysis in analyses.values()
    ])
    contrast_scales = {
        name: _symmetric_scale([analysis.contrasts[name].effect for analysis in analyses.values()])
        for name in ("GS", "DS", "GD")
    }
    gs_ds_scale = _symmetric_scale([
        analysis.contrasts[name].effect for analysis in analyses.values() for name in ("GS", "DS")
    ])
    for session, analysis in analyses.items():
        primary = analysis.contrasts["STIM_PRESENCE"]
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.6), constrained_layout=True)
        low, high = np.percentile(analysis.background, [1, 99])
        axes[0].imshow(analysis.background, cmap="gray", vmin=low, vmax=high, aspect="auto")
        axes[0].set_title("block-mean background", fontsize=9)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        _effect_image(axes[1], primary.effect, primary_effect_scale, "binary effect (global scale)")
        _effect_image(axes[2], primary.standardized, primary_standardized_scale, "standardized effect (global scale)")
        axes[3].imshow(primary.fdr_mask, cmap="binary", vmin=0, vmax=1, aspect="auto")
        axes[3].set_title(f"BH-FDR q≤.05 (n={int(primary.fdr_mask.sum())})", fontsize=9)
        axes[3].set_xticks([])
        axes[3].set_yticks([])
        fig.suptitle(
            f"Session {session}: stimulus-presence-associated spatial contrast\n"
            "Four clean4 frames averaged within each block; no anatomical registration",
            fontsize=11,
        )
        _save(fig, output_dir / f"figures/primary_binary_maps/session_{session}_primary_binary_maps.png")

        for name in ("GS", "DS"):
            fig, ax = plt.subplots(figsize=(8.5, 3.3))
            _effect_image(ax, analysis.contrasts[name].effect, contrast_scales[name], f"Session {session}: {name} effect map")
            fig.suptitle("Secondary confirmatory condition-associated contrast; no anatomical registration", fontsize=9)
            _save(fig, output_dir / f"figures/{name}_maps/session_{session}_{name}_map.png")

        corr = float(analysis.concordance_row["GS_DS_spatial_corr"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), constrained_layout=True)
        _effect_image(axes[0], analysis.contrasts["GS"].effect, gs_ds_scale, "GS: grating - stop")
        _effect_image(axes[1], analysis.contrasts["DS"].effect, gs_ds_scale, "DS: dot - static")
        fig.suptitle(f"Session {session}: within-session GS–DS spatial corr = {corr:.3f}", fontsize=11)
        _save(fig, output_dir / f"figures/GS_DS_comparison/session_{session}_GS_DS_comparison.png")

        fig, ax = plt.subplots(figsize=(8.5, 3.3))
        _effect_image(ax, analysis.contrasts["GD"].effect, contrast_scales["GD"], f"Session {session}: GD = grating - dot")
        fig.suptitle("EXPLORATORY — fixed temporal-position confound", fontsize=11, color="#9b3d30")
        _save(fig, output_dir / f"figures/exploratory_GD_maps/session_{session}_exploratory_GD_map.png")


def plot_binary_maps_overview(analyses: Mapping[str, SessionAnalysis], path: Path) -> None:
    sessions = list(analyses)
    scale = _symmetric_scale([analyses[session].contrasts["STIM_PRESENCE"].effect for session in sessions])
    n_cols = 3
    n_rows = int(np.ceil(len(sessions) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.5 * n_rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, session in zip(axes_flat, sessions):
        _effect_image(ax, analyses[session].contrasts["STIM_PRESENCE"].effect, scale, f"Session {session}")
    for ax in axes_flat[len(sessions):]:
        ax.axis("off")
    fig.suptitle(
        "Primary binary condition-associated spatial maps — global 99th-percentile scale\n"
        "Side-by-side visualization only; sessions are not anatomically registered",
        fontsize=13,
    )
    _save(fig, path)


def plot_reproducibility(diagnostic: pd.DataFrame, path: Path) -> None:
    frame = diagnostic.copy()
    x = np.arange(len(frame))
    y = frame["binary_split_half_corr_median"].to_numpy(float)
    lower = y - frame["binary_split_half_corr_2.5pct"].to_numpy(float)
    upper = frame["binary_split_half_corr_97.5pct"].to_numpy(float) - y
    colors = ["#356a9a" if session in STRONG_SESSIONS else "#c37b62" for session in frame["session"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#555555", capsize=3)
    ax.scatter(x, y, c=colors, s=55, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, frame["session"])
    ax.set_ylabel("Median cycle split-half spatial Pearson r")
    ax.set_title("Binary spatial reproducibility by session (95% split interval)")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def _association_plot(
    diagnostic: pd.DataFrame,
    x_column: str,
    y_column: str,
    association: pd.Series,
    path: Path,
    x_label: str,
) -> None:
    colors = ["#356a9a" if session in STRONG_SESSIONS else "#c37b62" for session in diagnostic["session"]]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(diagnostic[x_column], diagnostic[y_column], c=colors, s=60)
    for row in diagnostic.itertuples():
        ax.annotate(str(row.session), (getattr(row, x_column), getattr(row, y_column)), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("SmallCNN feature-mean within-session BA")
    ax.set_title(
        f"Spearman ρ={association['rho']:.3f}; exact p={association['permutation_p_two_sided']:.4g}; "
        f"Holm p={association['holm_adjusted_p']:.4g}"
    )
    ax.grid(alpha=0.2)
    _save(fig, path)


def plot_gs_ds_concordance(diagnostic: pd.DataFrame, path: Path) -> None:
    colors = ["#356a9a" if session in STRONG_SESSIONS else "#c37b62" for session in diagnostic["session"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(diagnostic))
    ax.bar(x, diagnostic["GS_DS_spatial_corr"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, diagnostic["session"])
    ax.set_ylabel("Within-session spatial Pearson r")
    ax.set_title("GS–DS spatial concordance (maps are not compared across sessions)")
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)


def plot_diagnostic_overview(diagnostic: pd.DataFrame, path: Path) -> None:
    columns = [
        "within_session_BA",
        "v8_stimulus_vector_magnitude",
        "v8_split_half_vector_stability",
        "binary_RMS_standardized_effect",
        "binary_split_half_corr_median",
        "GS_DS_spatial_corr",
        "binary_fdr_fraction",
    ]
    labels = [
        "within BA", "v8 vector\nmagnitude", "v8 vector\nstability",
        "v9 spatial\neffect", "v9 map\nreproducibility", "GS–DS\nconcordance", "binary FDR\nfraction",
    ]
    raw = diagnostic[columns].to_numpy(dtype=float)
    scale = raw.std(axis=0, ddof=0)
    z = (raw - raw.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    image = ax.imshow(z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(len(diagnostic)):
        for j in range(len(columns)):
            value = raw[i, j]
            label = f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")
        ax.text(len(columns) - 0.05, i, f"  {diagnostic.iloc[i]['spatial_diagnostic_pattern']}", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(columns)), labels)
    ax.set_yticks(np.arange(len(diagnostic)), diagnostic["session"])
    ax.set_ylabel("Session")
    ax.set_title(
        f"{len(diagnostic)}-session spatial diagnostic overview "
        "(color: column z-score; text: raw value)"
    )
    plt.colorbar(image, ax=ax, label="Across-session column z-score", fraction=0.03, pad=0.08)
    _save(fig, path)


def make_all_figures(
    analyses: Mapping[str, SessionAnalysis],
    diagnostic: pd.DataFrame,
    associations: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_session_map_panels(analyses, output_dir)
    plot_binary_maps_overview(analyses, output_dir / "figures/binary_spatial_maps_9sessions.png")
    plot_reproducibility(diagnostic, output_dir / "figures/spatial_reproducibility_by_session.png")
    effect_assoc = associations[associations["planned_family_label"] == "effect_magnitude"].iloc[0]
    repro_assoc = associations[associations["planned_family_label"] == "reproducibility"].iloc[0]
    _association_plot(
        diagnostic, "binary_RMS_standardized_effect", "within_session_BA", effect_assoc,
        output_dir / "figures/spatial_effect_vs_within_BA.png", "Binary RMS standardized spatial effect",
    )
    _association_plot(
        diagnostic, "binary_split_half_corr_median", "within_session_BA", repro_assoc,
        output_dir / "figures/spatial_reproducibility_vs_within_BA.png", "Binary split-half spatial Pearson r",
    )
    plot_gs_ds_concordance(diagnostic, output_dir / "figures/GS_DS_spatial_concordance_by_session.png")
    plot_diagnostic_overview(diagnostic, output_dir / "figures/spatial_diagnostic_overview.png")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    subset = frame[list(columns)].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in subset.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append("NA" if not np.isfinite(value) else f"{float(value):.{digits}g}")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def make_report(
    diagnostic: pd.DataFrame,
    glm_summary: pd.DataFrame,
    associations: pd.DataFrame,
    stability_association: pd.DataFrame,
) -> str:
    primary = glm_summary[glm_summary["contrast"] == "STIM_PRESENCE"]
    n_any_fdr = int((primary["n_fdr_pixels"] > 0).sum())
    pattern_counts = diagnostic["spatial_diagnostic_pattern"].value_counts().sort_index().to_dict()
    sessions = diagnostic["session"].astype(str).tolist()
    formal = sessions == list(EXPECTED_SESSIONS)
    scope = (
        f"All nine preregistered sessions ({', '.join(sessions)}) are retained."
        if formal else
        f"SMOKE SCHEMA ONLY: {len(sessions)} sessions ({', '.join(sessions)}); this is not a formal scientific result."
    )
    return f"""# 9-session blockwise spatial GLM and contrast reproducibility v9

## Scope and frozen design

{scope} Each observation is one 128×501 block image obtained by applying arcsinh to the four frozen clean-middle frames and then taking their pixelwise mean. Every complete cycle contributes exactly four observations in the fixed order {CONDITION_ORDER}. Cycle fixed effects and categorical condition effects are fit separately at every pixel within each session. No frame is treated as an independent statistical sample.

No HRF parameter search was performed in v9.

No registration, ROI selection, searchlight, decoder training, attribution, or post-hoc threshold search was performed. Session maps are displayed side-by-side only and are not anatomically registered.

## Fixed-order limitation

{FIXED_ORDER_WARNING}

The primary stimulus-presence contrast and the GS/DS secondary contrasts are therefore described as condition-associated spatial contrasts. GD is always interpreted under this warning: **{GD_WARNING}**

## Primary spatial results

{_markdown_table(diagnostic, ['session', 'historical_group', 'within_session_BA', 'binary_RMS_effect', 'binary_RMS_standardized_effect', 'binary_split_half_corr_median', 'GS_DS_spatial_corr', 'binary_n_fdr_pixels', 'binary_fdr_fraction', 'spatial_diagnostic_pattern'])}

Effect maps are retained regardless of significance. BH-FDR was applied independently across valid pixels for each session × contrast at q=0.05. {n_any_fdr} of {len(sessions)} analyzed sessions had at least one primary pixel survive the predefined threshold. When a row has zero surviving pixels, the supported statement is: **no pixel survived the predefined FDR threshold**; this is not evidence that no spatial response exists.

Descriptive S1–S4 labels mean only relative-high/relative-low effect and reproducibility within the analyzed {len(sessions)}-session sample, using sample medians. They are sample-relative visualization aids, not subgroup significance tests. Counts: {pattern_counts}.

## Planned associations with within-session BA

{_markdown_table(associations, ['metric', 'rho', 'permutation_p_two_sided', 'holm_adjusted_p', 'permutation_method', 'n_permutations'])}

Holm correction is restricted to these three planned BA associations. No nine-point multivariable regression or result-dependent session exclusion was performed.

## v8–v9 stability link

{_markdown_table(stability_association, ['predictor', 'outcome', 'rho', 'permutation_p_two_sided', 'permutation_method', 'n_permutations'])}

This is a secondary mechanistic association and is not part of the three-test Holm family.

## Spatial interpretation limits

GS–DS concordance compares two maps only within the same session. No pixelwise map correlation, Dice overlap, same-region claim, or anatomical ROI identity is computed across sessions. FDR masks are not used to select ROIs or to test decoding on the same data.

## Relation to the three proposed research directions

### Direction 1

v7/v8 already characterize cross-session feature discrepancy and session-centered latent stimulus vectors. v9 does not continue model tuning.

### Direction 2

If whole-brain effect magnitude or spatial reproducibility is weak, a future independently specified expert-ROI analysis could examine ROI signal contrast and ROI decoding. v9 does not perform that analysis and does not derive ROIs from its significance maps.

### Direction 3

v9 is the unified formal nine-session analysis for this direction: pixelwise block-level GLM, unthresholded spatial contrast maps, complete-cycle split-half reproducibility, and within-session GS/DS concordance.
"""
