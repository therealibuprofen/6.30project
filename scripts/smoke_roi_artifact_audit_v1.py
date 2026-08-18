#!/usr/bin/env python3
"""Run roi_artifact_audit_v1 against temporary mock artifacts only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.roi_artifact_audit_v1 import EXPECTED_SHAPE, run_audit


def _write_roi(root: Path, session: str, mask: np.ndarray, metadata: dict) -> None:
    directory = root / "outputs/candidate_roi_mock" / f"session_{session}"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "candidate_roi_mask.npy", mask)
    rows, cols = np.indices(EXPECTED_SHAPE)
    np.save(directory / "mean_image.npy", (rows + cols / 1000).astype(np.float32))
    (directory / "candidate_roi_metadata.json").write_text(
        json.dumps({"session": session, "session_specific": True, **metadata}, indent=2), encoding="utf-8"
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="roi_artifact_audit_v1_smoke_") as temporary:
        root = Path(temporary)
        mask = np.zeros(EXPECTED_SHAPE, dtype=bool)
        mask[25:50, 120:200] = True
        _write_roi(root, "708", mask, {
            "roi_provenance": "analyst_reconstructed_from_expert_indicated_approximate_region",
            "annotation_reason": "redrawn from expert guidance on legacy searchlight display",
            "label_information_used": False,
        })
        _write_roi(root, "709", mask.copy(), {
            "roi_provenance": "expert_session_specific",
            "label_information_used": False,
        })
        _write_roi(root, "710", np.roll(mask, 30, axis=1), {
            "roi_provenance": "threshold_searchlight_accuracy_top_10_percent",
            "label_information_used": True,
        })
        output = root / "outputs/roi_artifact_audit_v1"
        summary = run_audit(root, output, device="cpu")
        if summary["n_valid_masks"] != 3:
            raise AssertionError(summary)
        if summary["n_duplicate_groups"] != 1:
            raise AssertionError(summary)
        if not (output / "figures/roi_overlay_9sessions.png").is_file():
            raise AssertionError("overview missing")
        report = (output / "roi_provenance_report.md").read_text(encoding="utf-8")
        if "Decision R-B" not in report:
            raise AssertionError("unexpected decision")
        print("SMOKE PASS")
        print("mock artifacts only: discovery, duplicate detection, classification, overlays, and report passed")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
