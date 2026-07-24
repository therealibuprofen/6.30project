#!/usr/bin/env python3
"""Inspect monkey ultrasound visual-stimulus MATLAB v7.3 files.

The project data are MATLAB v7.3 .mat files, which are HDF5 containers.
This script inventories files, records HDF5 dataset metadata, computes
lightweight summary statistics, and derives first-pass stimulus labels from
the readme.pptx timing description.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import unescape

try:
    import h5py
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    h5py = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    np = None


STIMULUS_BLOCKS = (
    ("grating", "stimulus"),
    ("stop_after_grating", "no_stimulus"),
    ("dot", "stimulus"),
    ("static", "no_stimulus"),
)


@dataclass
class FileRecord:
    dataset: str
    file: str
    index: int
    size_bytes: int
    mat_version: str
    hdf5_path: str | None = None
    shape: str | None = None
    dtype: str | None = None
    compression: str | None = None
    chunks: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    std_value: float | None = None
    inferred_cycle: int | None = None
    inferred_cycle_time_start_s: float | None = None
    inferred_cycle_time_center_s: float | None = None
    inferred_stimulus_name: str | None = None
    inferred_binary_label: str | None = None


def read_pptx_text(path: Path) -> list[dict[str, str]]:
    """Extract plain slide text from a pptx without external dependencies."""
    if not path.exists():
        return []
    slides = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(
            name
            for name in zf.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for name in names:
            xml = zf.read(name).decode("utf-8", errors="ignore")
            text = "\n".join(unescape(match) for match in re.findall(r"<a:t>(.*?)</a:t>", xml))
            if text.strip():
                slides.append({"slide": name, "text": text})
    return slides


def mat_version(path: Path) -> str:
    header = path.read_bytes()[:128]
    if header.startswith(b"MATLAB 7.3 MAT-file"):
        return "MATLAB 7.3/HDF5"
    if header.startswith(b"MATLAB"):
        return header[:116].decode("latin1", errors="replace").strip()
    return "unknown"


def parse_index(path: Path) -> int:
    match = re.search(r"(\d+)\.mat$", path.name)
    if not match:
        raise ValueError(f"Cannot parse numeric index from {path}")
    return int(match.group(1))


def infer_labels(index: int, group_seconds: float, cycle_seconds: float) -> dict[str, float | int | str]:
    """Infer labels using the center of each 4 s group within the 120 s cycle."""
    zero_based = index - 1
    start_s = zero_based * group_seconds
    center_s = start_s + group_seconds / 2.0
    cycle = math.floor(start_s / cycle_seconds)
    center_in_cycle = center_s % cycle_seconds
    block_i = min(int(center_in_cycle // 30.0), len(STIMULUS_BLOCKS) - 1)
    stim_name, binary = STIMULUS_BLOCKS[block_i]
    return {
        "inferred_cycle": cycle,
        "inferred_cycle_time_start_s": start_s % cycle_seconds,
        "inferred_cycle_time_center_s": center_in_cycle,
        "inferred_stimulus_name": stim_name,
        "inferred_binary_label": binary,
    }


def iter_hdf5_datasets(path: Path) -> Iterable[tuple[str, object]]:
    if h5py is None:
        return []
    datasets = []
    with h5py.File(path, "r") as handle:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append((name, obj))

        handle.visititems(visitor)
        out = []
        for name, dataset in datasets:
            data = dataset[()]
            stats = {}
            if np is not None and np.issubdtype(data.dtype, np.number):
                stats = {
                    "min_value": float(np.nanmin(data)),
                    "max_value": float(np.nanmax(data)),
                    "mean_value": float(np.nanmean(data)),
                    "std_value": float(np.nanstd(data)),
                }
            out.append(
                (
                    name,
                    {
                        "shape": tuple(int(x) for x in dataset.shape),
                        "dtype": str(dataset.dtype),
                        "compression": dataset.compression,
                        "chunks": dataset.chunks,
                        **stats,
                    },
                )
            )
    return out


def inspect_file(path: Path, root: Path, group_seconds: float, cycle_seconds: float) -> FileRecord:
    path = path.resolve()
    index = parse_index(path)
    record = FileRecord(
        dataset=path.parent.name,
        file=str(path.relative_to(root.resolve())),
        index=index,
        size_bytes=path.stat().st_size,
        mat_version=mat_version(path),
        **infer_labels(index, group_seconds, cycle_seconds),
    )
    datasets = list(iter_hdf5_datasets(path))
    if datasets:
        name, meta = datasets[0]
        record.hdf5_path = name
        record.shape = "x".join(str(x) for x in meta["shape"])
        record.dtype = meta["dtype"]
        record.compression = meta["compression"]
        record.chunks = "x".join(str(x) for x in meta["chunks"]) if meta["chunks"] else None
        record.min_value = meta.get("min_value")
        record.max_value = meta.get("max_value")
        record.mean_value = meta.get("mean_value")
        record.std_value = meta.get("std_value")
    return record


def summarize(records: list[FileRecord]) -> dict:
    by_dataset = {}
    for dataset in sorted({r.dataset for r in records}):
        rows = [r for r in records if r.dataset == dataset]
        indices = sorted(r.index for r in rows)
        missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices)) if indices else []
        labels = {}
        for row in rows:
            key = row.inferred_stimulus_name or "unknown"
            labels[key] = labels.get(key, 0) + 1
        by_dataset[dataset] = {
            "file_count": len(rows),
            "index_min": min(indices),
            "index_max": max(indices),
            "missing_indices": missing,
            "size_mb": round(sum(r.size_bytes for r in rows) / 1024 / 1024, 2),
            "hdf5_paths": sorted({r.hdf5_path for r in rows if r.hdf5_path}),
            "shapes": sorted({r.shape for r in rows if r.shape}),
            "dtypes": sorted({r.dtype for r in rows if r.dtype}),
            "label_counts": labels,
            "mean_of_file_means": statistics.fmean(
                r.mean_value for r in rows if r.mean_value is not None
            )
            if any(r.mean_value is not None for r in rows)
            else None,
        }
    return {"datasets": by_dataset}


def write_csv(path: Path, records: list[FileRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in records]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, slides: list[dict[str, str]], summary: dict) -> None:
    lines = [
        "# Monkey ultrasound visual-stimulus data inventory",
        "",
        "## README/PPT extracted notes",
        "",
    ]
    for slide in slides:
        lines.extend([f"### {slide['slide']}", "", "```text", slide["text"], "```", ""])
    lines.extend(
        [
            "## Inferred data format",
            "",
            "- MATLAB v7.3 `.mat` files are HDF5 containers.",
            "- Each inspected file contains `Data_SVD` with shape `128x501`, dtype `float64`, gzip compression.",
            "- First-pass timing labels assume 4 s per file and a 120 s cycle: grating, stop, dot, static, each 30 s.",
            "- Boundary labels use the center time of each 4 s file. Replace these with experiment-log labels if available.",
            "",
            "## Dataset summary",
            "",
            "| Dataset | Files | Index range | Missing | Size MB | Shapes | Label counts |",
            "| --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
    )
    for dataset, meta in summary["datasets"].items():
        label_counts = ", ".join(f"{k}: {v}" for k, v in sorted(meta["label_counts"].items()))
        lines.append(
            f"| {dataset} | {meta['file_count']} | {meta['index_min']}-{meta['index_max']} | "
            f"{len(meta['missing_indices'])} | {meta['size_mb']} | {', '.join(meta['shapes'])} | {label_counts} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--readme-pptx", type=Path, default=Path("readme.pptx"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--group-seconds", type=float, default=4.0)
    parser.add_argument("--cycle-seconds", type=float, default=120.0)
    args = parser.parse_args()

    root = Path.cwd()
    files = sorted(args.data_root.glob("*/*.mat"), key=lambda p: (p.parent.name, parse_index(p)))
    records = [inspect_file(path, root, args.group_seconds, args.cycle_seconds) for path in files]
    summary = summarize(records)
    slides = read_pptx_text(args.readme_pptx)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "data_inventory.csv", records)
    (args.out_dir / "data_summary.json").write_text(
        json.dumps({"readme_slides": slides, **summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "data_inventory.md", slides, summary)

    print(f"Inspected {len(records)} files.")
    for dataset, meta in summary["datasets"].items():
        print(
            f"{dataset}: files={meta['file_count']} range={meta['index_min']}-{meta['index_max']} "
            f"missing={len(meta['missing_indices'])} shapes={meta['shapes']}"
        )
    print(f"Wrote {args.out_dir / 'data_inventory.csv'}")
    print(f"Wrote {args.out_dir / 'data_summary.json'}")
    print(f"Wrote {args.out_dir / 'data_inventory.md'}")


if __name__ == "__main__":
    main()
