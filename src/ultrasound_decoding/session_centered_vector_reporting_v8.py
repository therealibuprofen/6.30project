"""Reporting utilities for session-centered stimulus-vector analysis v8."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ultrasound_decoding.cross_session_feature_factor_v7 import EXPECTED_SESSIONS, STRONG_SESSIONS, WEAK_SESSIONS
from ultrasound_decoding.session_centered_vector_v8 import GD_WARNING, TRANSDUCTIVE_WARNING


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _aggregate_masked(table: pd.DataFrame, keys: Sequence[str], values: Sequence[str]) -> pd.DataFrame:
    masked = table[table["representation"].astype(str).str.startswith("GLOBAL_MASKED")].copy()
    return masked.groupby(list(keys), as_index=False)[list(values)].mean()


def plot_session_id_before_after(table: pd.DataFrame, path: Path) -> None:
    raw = table[table["representation"] == "RAW_SPATIAL_PCA"].iloc[0]
    masked_rows = table[
        table["representation"].astype(str).str.startswith("GLOBAL_MASKED")
        & (table["seed"].astype(str) != "MEAN_3_SEEDS")
    ]
    masked = masked_rows[["uncentered_session_BA", "centered_session_BA"]].mean()
    values = np.asarray([
        [raw.uncentered_session_BA, raw.centered_session_BA],
        [masked.uncentered_session_BA, masked.centered_session_BA],
    ])
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.18, values[:, 0], 0.36, label="Uncentered", color="#547aa5")
    ax.bar(x + 0.18, values[:, 1], 0.36, label="Session-centered", color="#d77a61")
    ax.axhline(1 / 9, color="black", linestyle="--", linewidth=1, label="9-class chance reference")
    ax.set_xticks(x, ["RAW_SPATIAL_PCA", "GLOBAL_MASKED\n(mean of seeds)"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cycle-grouped session-ID balanced accuracy")
    ax.set_title("Session identity before and after descriptive session centering")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_transductive_probe(table: pd.DataFrame, path: Path) -> None:
    reps = ["RAW_SPATIAL_PCA", "GLOBAL_MASKED_SMALLCNN"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, rep in zip(axes, reps):
        subset = table[table["representation"] == rep]
        if rep == "GLOBAL_MASKED_SMALLCNN":
            subset = subset[subset["seed"].astype(str) == "MEAN_3_SEEDS"]
        else:
            subset = subset[subset["seed"].astype(str).isin(("", "nan"))]
        subset = subset[subset["target"].astype(str).isin(EXPECTED_SESSIONS)].copy()
        subset["target"] = pd.Categorical(subset["target"].astype(str), EXPECTED_SESSIONS, ordered=True)
        subset = subset.sort_values("target")
        x = np.arange(len(subset))
        ax.plot(x, subset["uncentered_BA"], "o-", label="Uncentered control")
        ax.plot(x, subset["centered_BA"], "o-", label="Transductive centered")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x, subset["target"].astype(str), rotation=45)
        ax.set_title(rep)
        ax.set_xlabel("Target session")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].legend(frameon=False)
    fig.suptitle("Mechanistic transductive centering control (not strict unseen-session generalization)")
    _save(fig, path)


def plot_cosine_heatmap(pairwise: pd.DataFrame, path: Path, title: str) -> None:
    matrix = np.eye(len(EXPECTED_SESSIONS))
    index = {session: i for i, session in enumerate(EXPECTED_SESSIONS)}
    for row in pairwise.itertuples():
        i, j = index[str(row.session_a)], index[str(row.session_b)]
        matrix[i, j] = matrix[j, i] = float(row.cosine_similarity)
    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    labels = list(EXPECTED_SESSIONS)
    ax.set_xticks(range(9), labels, rotation=45, ha="right")
    ax.set_yticks(range(9), labels)
    for tick, label in zip(ax.get_xticklabels(), labels):
        if label in STRONG_SESSIONS:
            tick.set_fontweight("bold"); tick.set_color("#7b2d26")
    for tick, label in zip(ax.get_yticklabels(), labels):
        if label in STRONG_SESSIONS:
            tick.set_fontweight("bold"); tick.set_color("#7b2d26")
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(matrix[i, j]) > 0.65 else "black")
    ax.set_title(title + "\nBold labels: historically strong sessions")
    fig.colorbar(image, ax=ax, label="Stimulus contrast vector cosine")
    _save(fig, path)


def plot_stability(stability: pd.DataFrame, path: Path) -> None:
    raw = stability[stability["representation"] == "RAW_SPATIAL_PCA"].copy()
    raw["session"] = pd.Categorical(raw["session"].astype(str), EXPECTED_SESSIONS, ordered=True)
    raw = raw.sort_values("session")
    x = np.arange(9)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, raw["bootstrap_vector_stability"], 0.36, label="Cycle-bootstrap vs full vector")
    ax.bar(x + 0.18, raw["split_half_vector_stability"], 0.36, label="Cycle split-half")
    ax.set_xticks(x, raw["session"].astype(str))
    ax.set_ylim(-1, 1)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("RAW_SPATIAL_PCA within-session stimulus-vector stability")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_magnitude(magnitude: pd.DataFrame, path: Path) -> None:
    raw = magnitude[magnitude["representation"] == "RAW_SPATIAL_PCA"].copy()
    raw["session"] = pd.Categorical(raw["session"].astype(str), EXPECTED_SESSIONS, ordered=True)
    raw = raw.sort_values("session")
    colors = ["#d47a6a" if str(value) in WEAK_SESSIONS else "#4472a3" for value in raw["session"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(raw["session"].astype(str), raw["stimulus_vector_norm"], color=colors)
    axes[0].set_title("Raw stimulus-vector L2 norm")
    axes[1].bar(raw["session"].astype(str), raw["normalized_vector_norm"], color=colors)
    axes[1].set_title("Norm / within-condition dispersion")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Session")
    fig.suptitle("RAW_SPATIAL_PCA stimulus contrast magnitude; group colors are descriptive only")
    _save(fig, path)


def plot_alignment_transfer(pairwise: pd.DataFrame, transfer: pd.DataFrame, associations: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, representation in zip(axes, ("RAW_SPATIAL_PCA", "GLOBAL_MASKED_SMALLCNN")):
        cosine = pairwise[pairwise["representation"] == representation]
        if representation == "GLOBAL_MASKED_SMALLCNN":
            cosine = cosine[cosine["seed"].astype(str) == "MEAN_3_SEEDS"]
        merged = cosine.merge(transfer, on=["session_a", "session_b"])
        colors = np.where(merged["strong_pair"].astype(bool), "#a43f3f", "#537ba6")
        ax.scatter(merged["cosine_similarity"], merged["symmetric_cross_BA"], c=colors, alpha=0.8)
        for row in merged[merged["strong_pair"].astype(bool)].itertuples():
            ax.annotate(f"{row.session_a}-{row.session_b}", (row.cosine_similarity, row.symmetric_cross_BA), fontsize=8)
        assoc = associations[associations["representation"] == representation].iloc[0]
        ax.set(xlabel="Stimulus-vector cosine", ylabel="Symmetric cross-session BA",
               title=f"{representation}\nMantel ρ={assoc.rho:.2f}, p={assoc.permutation_p_two_sided:.3g}")
    fig.suptitle("Vector alignment and transfer (association, not causation)")
    _save(fig, path)


def plot_diagnostic_overview(table: pd.DataFrame, path: Path) -> None:
    columns = [
        "within_session_BA", "v7_separability_ratio", "v7_cycle_consistency",
        "normalized_vector_norm", "bootstrap_vector_stability", "mean_vector_cosine_to_other_sessions",
    ]
    labels = ["Within BA", "Separability", "Cycle r", "Norm/disp.", "Vector stability", "Mean cross-session cosine"]
    ordered = table.set_index(table["session"].astype(str)).reindex(EXPECTED_SESSIONS)
    values = ordered[columns].to_numpy(float)
    std = np.nanstd(values, axis=0)
    z = (values - np.nanmean(values, axis=0)) / np.where(std > 0, std, 1)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    image = ax.imshow(z, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(z[i, j]) > 1.25 else "black")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(9), EXPECTED_SESSIONS)
    ax.set_title("RAW_SPATIAL_PCA session-vector diagnostic overview\ncolor = column z-score; text = raw value")
    fig.colorbar(image, ax=ax, label="Within-column z-score")
    _save(fig, path)


def make_report(
    session_id: pd.DataFrame,
    transductive: pd.DataFrame,
    magnitude: pd.DataFrame,
    stability: pd.DataFrame,
    pairwise: pd.DataFrame,
    associations: pd.DataFrame,
    within_associations: pd.DataFrame,
    diagnostic: pd.DataFrame,
) -> str:
    raw_sid = session_id[session_id["representation"] == "RAW_SPATIAL_PCA"].iloc[0]
    raw_trans = transductive[
        (transductive["representation"] == "RAW_SPATIAL_PCA")
        & (transductive["target"].astype(str).isin(EXPECTED_SESSIONS))
    ]
    raw_pairs = pairwise[pairwise["representation"] == "RAW_SPATIAL_PCA"]
    strong = raw_pairs[raw_pairs["strong_pair"].astype(bool)]
    mean_center_gain = float(raw_trans["delta"].mean())
    mean_cosine = float(raw_pairs["cosine_similarity"].mean())
    baseline_shift = raw_sid.session_information_reduction > 0.25
    directions_dependent = mean_cosine < 0.5
    if baseline_shift and mean_center_gain > 0.05:
        scenario = "V8-A"
        priority = "Direction 1"
    elif baseline_shift and directions_dependent:
        scenario = "V8-B"
        priority = "Direction 3"
    else:
        scenario = "V8-D/E mixed"
        priority = "Direction 2 or within-session variability, according to session pattern"
    lines = [
        "# Session-centered stimulus-vector alignment analysis v8", "",
        "## Scope and artifact reuse", "",
        "All nine sessions and the frozen v7 stimulus_presence mapping are retained. RAW_SPATIAL_PCA coordinates and all "
        "three GLOBAL_MASKED_SMALLCNN seed features are reused exactly; no PCA refit, feature extraction, encoder training, "
        "CSU, registration, ROI, GLM, or searchlight analysis is performed. FORMAL_MODE = CPU_STATS_ONLY.", "",
        "## Session-centering mechanism", "",
        f"RAW session-ID BA changed from {raw_sid.uncentered_session_BA:.3f} to {raw_sid.centered_session_BA:.3f} "
        f"(reduction {raw_sid.session_information_reduction:.3f}). Session centroids use all unlabeled blocks in each session, "
        "so this is descriptive/transductive mechanism evidence, not a strict deployment method.", "",
        TRANSDUCTIVE_WARNING, "",
        f"Across targets, RAW transductive centering changed BA by {mean_center_gain:+.3f} on average. Its paired sign-flip "
        "test is explicitly secondary; the uncentered strict v7 LOSO baseline is never overwritten.", "",
        "## Stimulus contrast vectors", "",
        "A stimulus contrast vector is mean(feature | stimulus) minus mean(feature | no_stimulus). It is not called an "
        "activation direction. Session centering leaves this contrast algebraically unchanged, so baseline-shift analysis "
        "and task-direction consistency answer different questions.", "",
        f"The mean RAW cosine across all 36 session pairs is {mean_cosine:.3f}. Historically strong pairs: " +
        ", ".join(f"{row.session_a}-{row.session_b}={row.cosine_similarity:.3f}" for row in strong.itertuples()) + ".", "",
        "## Strong/weak descriptive diagnosis", "",
        "```text", diagnostic[["session", "descriptive_pattern", "normalized_vector_norm", "bootstrap_vector_stability", "split_half_vector_stability", "mean_vector_cosine_to_other_sessions"]].to_string(index=False), "```", "",
        "Pattern labels V1/V2/V3 are rank-based descriptive summaries only; no weak-subgroup significance test is used.", "",
        "## Exploratory grating-versus-dot control", "",
        GD_WARNING, "",
        "## Implications for the three proposed directions", "",
        "### Direction 1", "",
        "If transductive baseline centering restores transfer, separately preregister feature-space normalization/alignment.", "",
        "### Direction 2", "",
        "For weak sessions with low whole-brain contrast magnitude and low separability, test whether an expert ROI increases stimulus contrast; no ROI analysis is run here.", "",
        "### Direction 3", "",
        "If within-session vectors are stable but cross-session directions disagree, spatial GLM/searchlight work is the priority for locating session-dependent stimulus-response patterns; none is run here.", "",
        "## Preregistered decision", "",
        f"Best-matching scenario: {scenario}. Priority: {priority}. Baseline-shift evidence: {'supported' if baseline_shift else 'not strong'}. "
        f"Session-dependent task-direction evidence: {'supported' if directions_dependent else 'not strong'}.", "",
        "Association language is used throughout; no causal claim is made.",
    ]
    return "\n".join(lines) + "\n"
