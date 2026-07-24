#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example loader for block-level fUS sequence exports.")
    parser.add_argument("--output-dir", type=Path, default=Path("processed_data/block_sequences_v1"))
    parser.add_argument("--session", default="708")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--preview", type=Path, default=Path("example_session_708_block0_clean4.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = (PROJECT_DIR / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    h5_path = output_dir / f"session_{args.session}_blocks.h5"
    metadata_path = output_dir / f"session_{args.session}_block_metadata.csv"
    metadata = pd.read_csv(metadata_path)

    with h5py.File(h5_path, "r") as handle:
        X_clean4 = handle["/clean4/X"]
        y_binary = handle["/labels/binary"][:]
        y_stimulus_type = handle["/labels/stimulus_type"][:]
        cycles = handle["/metadata/cycle"][:]
        X_full_padded = handle["/full/X_padded"]
        valid_mask = handle["/full/valid_mask"][:]

        print(f"session: {args.session}")
        print(f"X_clean4 shape: {X_clean4.shape}")
        print(f"X_full_padded shape: {X_full_padded.shape}")
        print(f"metadata rows: {len(metadata)}")

        stimulus_blocks = np.where(y_binary == 1)[0]
        no_stimulus_blocks = np.where(y_binary == 0)[0]
        print(f"binary stimulus blocks: {len(stimulus_blocks)}")
        print(f"binary no_stimulus blocks: {len(no_stimulus_blocks)}")

        dot_blocks = np.where(y_stimulus_type == 0)[0]
        grating_blocks = np.where(y_stimulus_type == 1)[0]
        print(f"stimulus_type dot blocks: {len(dot_blocks)}")
        print(f"stimulus_type grating blocks: {len(grating_blocks)}")

        unique_cycles = np.unique(cycles)
        test_cycles = unique_cycles[:1]
        train_blocks = np.where(~np.isin(cycles, test_cycles))[0]
        test_blocks = np.where(np.isin(cycles, test_cycles))[0]
        print(f"group split by cycle: train_blocks={len(train_blocks)}, test_blocks={len(test_blocks)}")

        block_i = min(max(int(args.block_index), 0), X_clean4.shape[0] - 1)
        clean4_sequence = X_clean4[block_i]
        full_count = int(valid_mask[block_i].sum())
        full_sequence = X_full_padded[block_i, :full_count]
        print(f"example block_id: {metadata.iloc[block_i]['block_id']}")
        print(f"clean4 block shape: {clean4_sequence.shape}")
        print(f"full valid block shape: {full_sequence.shape}")

    try:
        mpl_cache = Path("/private/tmp/codex_mpl_cache")
        mpl_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped preview figure")
        return

    preview_path = args.preview
    if not preview_path.is_absolute():
        preview_path = output_dir / preview_path
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    finite = clean4_sequence[np.isfinite(clean4_sequence)]
    vmin, vmax = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    if float(vmin) == float(vmax):
        vmin, vmax = None, None
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for frame_i, ax in enumerate(axes):
        ax.imshow(clean4_sequence[frame_i], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"frame {frame_i}")
        ax.axis("off")
    fig.suptitle(str(metadata.iloc[block_i]["block_id"]))
    fig.savefig(preview_path, dpi=150)
    plt.close(fig)
    print(f"preview: {preview_path}")


if __name__ == "__main__":
    main()
