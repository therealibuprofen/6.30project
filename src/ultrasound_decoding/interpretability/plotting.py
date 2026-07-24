from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def _setup_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("lateral/pixel column", fontsize=8)
    ax.set_ylabel("depth/pixel row", fontsize=8)
    ax.tick_params(labelsize=7, length=2)


def save_map_figure(
    arr: np.ndarray,
    path_base: Path,
    *,
    title: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    chance_line: float | None = None,
) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor="white", constrained_layout=True)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    _setup_axis(ax, title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    if chance_line is not None:
        cbar.ax.axhline(chance_line, color="black", linewidth=0.8)
    fig.savefig(path_base.with_suffix(".png"), dpi=300)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def save_overlay_figure(
    background: np.ndarray,
    arr: np.ndarray,
    path_base: Path,
    *,
    title: str,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.55,
) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.8), facecolor="white", constrained_layout=True)
    ax.imshow(background, cmap="gray", aspect="auto", interpolation="nearest")
    masked = np.ma.masked_invalid(arr)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, aspect="auto", interpolation="nearest")
    _setup_axis(ax, title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.savefig(path_base.with_suffix(".png"), dpi=300)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_session_searchlight(session: str, background: np.ndarray, searchlight_dir: Path, figure_dir: Path) -> None:
    ba = np.load(searchlight_dir / "searchlight_ba_mean.npy")
    std = np.load(searchlight_dir / "searchlight_ba_std.npy")
    frac = np.load(searchlight_dir / "searchlight_above_chance_fraction.npy")
    save_map_figure(background, figure_dir / "mean_fus_background", title=f"session {session} mean fUS background", cmap="gray")
    save_map_figure(ba, figure_dir / "searchlight_ba_heatmap", title=f"session {session} binary PCA+LDA searchlight BA", cmap="viridis", vmin=0.0, vmax=1.0, chance_line=0.5)
    save_overlay_figure(background, ba, figure_dir / "searchlight_ba_overlay", title=f"session {session} binary PCA+LDA searchlight BA overlay", cmap="viridis", vmin=0.0, vmax=1.0)
    save_map_figure(std, figure_dir / "searchlight_fold_std", title=f"session {session} searchlight fold/window SD", cmap="magma")
    save_map_figure(frac, figure_dir / "searchlight_above_chance_fraction", title=f"session {session} BA > 0.5 fold/window fraction", cmap="viridis", vmin=0.0, vmax=1.0)


def plot_nn_maps(session: str, model: str, background: np.ndarray, aggregate_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("integrated_gradients_absolute_mean_mean.npy", "ig_absolute", "Integrated Gradients absolute", "magma", None, None),
        ("integrated_gradients_stimulus_signed_mean_mean.npy", "ig_stimulus_signed", "Integrated Gradients stimulus signed", "coolwarm", None, None),
        ("integrated_gradients_no_stimulus_signed_mean_mean.npy", "ig_no_stimulus_signed", "Integrated Gradients no_stimulus signed", "coolwarm", None, None),
        ("integrated_gradients_class_difference_signed_mean.npy", "ig_class_difference", "Integrated Gradients class difference", "coolwarm", None, None),
        ("occlusion_probability_drop_mean.npy", "occlusion_probability_drop", "occlusion true-class probability drop", "coolwarm", None, None),
        ("occlusion_ba_drop_mean.npy", "occlusion_ba_drop", "occlusion BA drop", "coolwarm", None, None),
        ("occlusion_flip_rate_mean.npy", "occlusion_flip_rate", "occlusion prediction flip rate", "viridis", 0.0, None),
    ]
    for filename, stem, label, cmap, vmin, vmax in specs:
        path = aggregate_dir / filename
        if not path.exists():
            continue
        arr = np.load(path)
        if "signed" in stem or "drop" in stem:
            finite = arr[np.isfinite(arr)]
            vmax_abs = float(np.max(np.abs(finite))) if len(finite) else 1.0
            vmin_plot, vmax_plot = -vmax_abs, vmax_abs
        else:
            vmin_plot, vmax_plot = vmin, vmax
        save_map_figure(arr, figure_dir / f"{model}_{stem}", title=f"session {session} {model} {label}", cmap=cmap, vmin=vmin_plot, vmax=vmax_plot)
        save_overlay_figure(background, arr, figure_dir / f"{model}_{stem}_overlay", title=f"session {session} {model} {label} overlay", cmap=cmap, vmin=vmin_plot, vmax=vmax_plot)


def save_main_figure(root: Path, sessions: list[str], output_base: Path) -> None:
    fig, axes = plt.subplots(len(sessions), 3, figsize=(10.5, 7.5), facecolor="white", constrained_layout=True)
    if len(sessions) == 1:
        axes = axes[None, :]
    for row_i, session in enumerate(sessions):
        sdir = root / f"session_{session}"
        panels = [
            (sdir / "pca_lda" / "searchlight" / "searchlight_ba_mean.npy", f"{session} PCA+LDA searchlight", "viridis", 0.0, 1.0),
            (sdir / "cnn" / "aggregate" / "integrated_gradients_absolute_mean_mean.npy", f"{session} CNN IG absolute", "magma", None, None),
            (sdir / "cnn" / "aggregate" / "occlusion_ba_drop_mean.npy", f"{session} CNN occlusion BA drop", "coolwarm", None, None),
        ]
        for col_i, (path, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_i, col_i]
            if path.exists():
                arr = np.load(path)
                if cmap == "coolwarm":
                    finite = arr[np.isfinite(arr)]
                    lim = float(np.max(np.abs(finite))) if len(finite) else 1.0
                    vmin, vmax = -lim, lim
                im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
                fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("lateral/pixel column", fontsize=8)
            ax.set_ylabel("depth/pixel row", fontsize=8)
            ax.tick_params(labelsize=7, length=2)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)

