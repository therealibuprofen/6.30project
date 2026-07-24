#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


IMAGE_SHAPE = (128, 501)
CAPTION = (
    "For each local searchlight region, the mean cross-validation decoding performance was assigned to the region center.\n"
    "The top 10% highest-performance locations were overlaid on the session-specific vascular background image."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Most informative voxels-style PCA+LDA searchlight overlays.")
    parser.add_argument("--sessions", nargs="+", default=["708", "709", "710"])
    parser.add_argument(
        "--run-root",
        default=str(PROJECT_DIR / "results" / "runs" / "interpretability" / "spatial_interpretability_binary_v1"),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=0.60)
    return parser.parse_args()


def resize_to_image(arr: np.ndarray, image_shape: tuple[int, int] = IMAGE_SHAPE) -> np.ndarray:
    if tuple(arr.shape) == image_shape:
        return arr
    import torch

    tensor = torch.from_numpy(arr.astype(np.float32, copy=False))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor,
        size=image_shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    return resized.astype(np.float64, copy=False)


def top_mask(values: np.ndarray, top_fraction: float) -> tuple[np.ndarray, float]:
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be between 0 and 1")
    valid = np.isfinite(values)
    if not valid.any():
        raise ValueError("searchlight map has no finite pixels")
    threshold = float(np.nanpercentile(values[valid], 100.0 * (1.0 - top_fraction)))
    mask = valid & (values >= threshold)
    return mask, threshold


def robust_limits(background: np.ndarray) -> tuple[float, float]:
    valid = background[np.isfinite(background)]
    if len(valid) == 0:
        raise ValueError("background has no finite pixels")
    return float(np.percentile(valid, 1.0)), float(np.percentile(valid, 99.0))


def draw_panel(
    ax,
    *,
    session: str,
    background: np.ndarray,
    ba_map: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    alpha: float,
) -> None:
    bg_vmin, bg_vmax = robust_limits(background)
    ax.imshow(background, cmap="gray", vmin=bg_vmin, vmax=bg_vmax, aspect="auto", interpolation="nearest")
    masked_values = np.ma.masked_where(~mask, ba_map)
    ax.imshow(masked_values, cmap="autumn", alpha=alpha, aspect="auto", interpolation="nearest")
    ax.set_title(f"Session {session}", fontsize=11)
    ax.set_xlabel("lateral/pixel column", fontsize=8)
    ax.set_ylabel("depth/pixel row", fontsize=8)
    ax.tick_params(labelsize=7, length=2)
    ax.text(
        0.015,
        0.965,
        f"Top 10% PCA+LDA BA\nthreshold >= {threshold:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 3},
    )


def save_single(
    *,
    session: str,
    background: np.ndarray,
    ba_map: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    output_dir: Path,
    alpha: float,
) -> None:
    fig = plt.figure(figsize=(7.2, 3.6), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.18])
    ax = fig.add_subplot(gs[0, 0])
    draw_panel(
        ax,
        session=session,
        background=background,
        ba_map=ba_map,
        mask=mask,
        threshold=threshold,
        alpha=alpha,
    )
    caption_ax = fig.add_subplot(gs[1, 0])
    caption_ax.axis("off")
    caption_ax.text(0.0, 0.9, CAPTION, ha="left", va="top", fontsize=7.5, color="black")
    base = output_dir / f"{session}_most_informative_regions"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def save_triptych(rows: list[dict[str, object]], output_dir: Path, alpha: float) -> None:
    fig = plt.figure(figsize=(12.2, 4.0), facecolor="white", constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.18])
    for i, row in enumerate(rows):
        ax = fig.add_subplot(gs[0, i])
        draw_panel(
            ax,
            session=str(row["session"]),
            background=row["background"],
            ba_map=row["ba_map"],
            mask=row["mask"],
            threshold=float(row["threshold"]),
            alpha=alpha,
        )
    caption_ax = fig.add_subplot(gs[1, :])
    caption_ax.axis("off")
    caption_ax.text(0.0, 0.9, CAPTION, ha="left", va="top", fontsize=8, color="black")
    base = output_dir / "most_informative_regions_triptych"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir) if args.output_dir else run_root / "aggregate" / "most_informative_regions"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for session in args.sessions:
        session_dir = run_root / f"session_{session}"
        ba_path = session_dir / "pca_lda" / "searchlight" / "searchlight_ba_mean.npy"
        bg_path = session_dir / "background" / "mean_fus_background.npy"
        if not ba_path.exists() or not bg_path.exists():
            raise FileNotFoundError(f"Missing searchlight/background inputs for session {session}")
        ba_map = resize_to_image(np.load(ba_path), IMAGE_SHAPE)
        background = resize_to_image(np.load(bg_path), IMAGE_SHAPE)
        if not np.isfinite(background).all():
            raise ValueError(f"Background contains NaN/Inf for session {session}")
        mask, threshold = top_mask(ba_map, args.top_fraction)
        save_single(
            session=session,
            background=background,
            ba_map=ba_map,
            mask=mask,
            threshold=threshold,
            output_dir=output_dir,
            alpha=args.alpha,
        )
        rows.append(
            {
                "session": session,
                "ba_path": str(ba_path),
                "background_path": str(bg_path),
                "threshold": threshold,
                "top_fraction_requested": args.top_fraction,
                "valid_pixel_count": int(np.isfinite(ba_map).sum()),
                "informative_pixel_count": int(mask.sum()),
                "background": background,
                "ba_map": ba_map,
                "mask": mask,
            }
        )
    save_triptych(rows, output_dir, args.alpha)
    summary = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key not in {"background", "ba_map", "mask"}}
            for row in rows
        ]
    )
    summary.to_csv(output_dir / "most_informative_regions_summary.csv", index=False)
    print(f"Saved Most informative regions figures under {output_dir}")


if __name__ == "__main__":
    main()

