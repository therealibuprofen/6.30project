from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.image as mpimg
import numpy as np


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-roi-artifact-audit-tests")
PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.roi_artifact_audit_v1 import (
    CLASS_CIRCULAR,
    CLASS_EXPERT,
    CLASS_MISSING,
    CLASS_RECONSTRUCTED,
    CLASS_RECONSTRUCTED_SEARCHLIGHT,
    CLASS_TRANSFER,
    CLASS_UNKNOWN,
    EXPECTED_SESSIONS,
    EXPECTED_SHAPE,
    DiscoveredArtifact,
    audit_orientation,
    build_session_status,
    classify_provenance,
    detect_duplicate_masks,
    discover_roi_artifacts,
    expected_outputs,
    inspect_artifact,
    make_overlay,
    make_overview,
    mask_array_sha256,
    recursive_search_evidence,
    run_audit,
)


class RoiArtifactAuditV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "outputs/roi_artifact_audit_v1"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def mask() -> np.ndarray:
        mask = np.zeros(EXPECTED_SHAPE, dtype=bool)
        mask[20:45, 100:180] = True
        return mask

    def add_roi(self, session: str, metadata: dict | None = None, mask: np.ndarray | None = None) -> Path:
        directory = self.root / "outputs/candidate_roi_mock" / f"session_{session}"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "candidate_roi_mask.npy"
        np.save(path, self.mask() if mask is None else mask)
        np.save(directory / "mean_image.npy", np.indices(EXPECTED_SHAPE)[0].astype(np.float32))
        if metadata is not None:
            (directory / "candidate_roi_metadata.json").write_text(
                json.dumps({"session": session, **metadata}), encoding="utf-8"
            )
        return path

    def inspect(self, path: Path):
        return inspect_artifact(DiscoveredArtifact(path, "MASK"), self.root, "ROI-00001")

    def test_01_fixed_nine_session_list(self) -> None:
        self.assertEqual(EXPECTED_SESSIONS, ("626", "628", "708", "709", "710", "807", "813", "817", "822"))

    def test_02_recursive_roi_search_runs_and_searches_content(self) -> None:
        path = self.add_roi("708", {"roi_provenance": "expert_session_specific"})
        script = self.root / "scripts/example.py"
        script.parent.mkdir()
        script.write_text("roi_mask = load(mask_path)\n", encoding="utf-8")
        found = discover_roi_artifacts(self.root, self.output)
        evidence = recursive_search_evidence(self.root, self.output)
        self.assertIn(path.resolve(), [item.path for item in found])
        self.assertTrue(any(row["path"] == "scripts/example.py" and "roi_mask" in row["content_keyword_hits"] for row in evidence))

    def test_03_missing_roi_is_safe_and_explicit(self) -> None:
        status = build_session_status([])
        self.assertEqual(len(status), 9)
        self.assertTrue(all(row["best_available_classification"] == CLASS_MISSING for row in status))

    def test_04_mask_sha256_uses_canonical_array_bytes(self) -> None:
        mask = self.mask()
        expected = hashlib.sha256(f"{mask.shape}|bool|C".encode("ascii") + mask.tobytes(order="C")).hexdigest()
        self.assertEqual(mask_array_sha256(mask.copy(order="F")), expected)

    def test_05_duplicate_mask_detection_across_sessions(self) -> None:
        mask = self.mask()
        digest = mask_array_sha256(mask)
        rows = [
            {"artifact_id": "a", "session": "708", "mask_sha256": digest, "file_sha256": "one", "artifact_path": "a.npy"},
            {"artifact_id": "b", "session": "709", "mask_sha256": digest, "file_sha256": "two", "artifact_path": "b.npy"},
        ]
        result = detect_duplicate_masks(rows, {"a": mask, "b": mask.copy()})
        self.assertEqual(result[0]["finding"], "IDENTICAL_MASK_ACROSS_SESSIONS")
        self.assertEqual(result[0]["same_array"], "YES")
        self.assertEqual(result[0]["same_hash"], "NO")

    def test_06_shape_audit_rejects_mismatch(self) -> None:
        path = self.add_roi("708", {"roi_provenance": "expert_session_specific"}, np.ones((8, 9), dtype=bool))
        row, loaded = self.inspect(path)
        self.assertEqual(loaded.shape, (8, 9))
        self.assertEqual(row["shape_valid"], "NO")
        self.assertEqual(row["exclusion_reason"], "SHAPE_MISMATCH")

    def test_07_mask_fraction_is_exact(self) -> None:
        path = self.add_roi("708", {"roi_provenance": "expert_session_specific"})
        row, _ = self.inspect(path)
        self.assertAlmostEqual(row["mask_fraction"], self.mask().sum() / self.mask().size)

    def test_08_reconstructed_provenance_rules(self) -> None:
        basic = classify_provenance(Path("session_708/roi.npy"), {
            "roi_provenance": "analyst_reconstructed_from_expert_indicated_approximate_region",
            "label_information_used": False,
        })
        searchlight = classify_provenance(Path("session_708/roi.npy"), {
            "roi_provenance": "analyst_reconstructed_from_expert_indicated_approximate_region",
            "annotation_reason": "redraw on audited legacy searchlight display",
            "label_information_used": False,
        })
        self.assertEqual(basic["classification"], CLASS_RECONSTRUCTED)
        self.assertEqual(searchlight["classification"], CLASS_RECONSTRUCTED_SEARCHLIGHT)
        self.assertEqual(searchlight["usable_for_primary_roi_decoding"], "REVIEW_REQUIRED")

    def test_09_label_derived_mask_is_circular(self) -> None:
        result = classify_provenance(Path("session_708/searchlight_top_10_mask.npy"), {"label_information_used": True})
        self.assertEqual(result["classification"], CLASS_CIRCULAR)
        self.assertEqual(result["exclusion_reason"], "CIRCULAR_ANALYSIS_RISK")

    def test_10_cross_session_transfer_cannot_be_primary(self) -> None:
        result = classify_provenance(Path("session_709/roi_mask.npy"), {
            "roi_provenance": "expert_session_specific",
            "transferred_from_session": "708",
            "registration_used": True,
        })
        self.assertEqual(result["classification"], CLASS_TRANSFER)
        self.assertEqual(result["usable_for_primary_roi_decoding"], "NO")

    def test_11_unknown_provenance_cannot_be_primary(self) -> None:
        result = classify_provenance(Path("session_708/roi_mask.npy"), {})
        self.assertEqual(result["classification"], CLASS_UNKNOWN)
        self.assertEqual(result["usable_for_primary_roi_decoding"], "NO")

    def test_12_mismatched_mask_is_not_resized(self) -> None:
        original = np.ones((7, 11), dtype=bool)
        path = self.add_roi("708", {}, original)
        _, loaded = self.inspect(path)
        np.testing.assert_array_equal(loaded, original)

    def test_13_audit_does_not_run_registration(self) -> None:
        path = self.add_roi("708", {"roi_provenance": "expert_session_specific"})
        before = path.read_bytes()
        run_audit(self.root, self.output)
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(any("transform" in item.name.lower() for item in self.output.rglob("*")))

    def test_14_missing_sessions_do_not_get_generated_masks(self) -> None:
        self.add_roi("708", {"roi_provenance": "expert_session_specific"})
        run_audit(self.root, self.output)
        generated = [path for path in self.output.rglob("*.npy")]
        self.assertEqual(generated, [])

    def test_15_session_807_orientation_audit(self) -> None:
        self.assertEqual(audit_orientation("807", {}), "ORIENTATION_MISMATCH_OR_UNCERTAIN")
        self.assertEqual(audit_orientation("807", {"fixed_orientation_normalization": "flip_vertical"}), "NORMALIZED_FLIP_VERTICAL")

    def test_16_overlay_missing_background_is_safe(self) -> None:
        output = self.root / "overlay.png"
        make_overlay(self.mask(), None, "test", output)
        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 0)

    def test_17_overview_is_fixed_nine_panel_canvas(self) -> None:
        output = self.root / "overview.png"
        make_overview([], {}, self.root, output)
        image = mpimg.imread(output)
        self.assertTrue(output.is_file())
        self.assertGreater(image.shape[1], image.shape[0])

    def test_18_read_only_guard_preserves_all_input_files(self) -> None:
        path = self.add_roi("708", {"roi_provenance": "expert_session_specific"})
        metadata = path.parent / "candidate_roi_metadata.json"
        before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, metadata)}
        run_audit(self.root, self.output)
        after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path, metadata)}
        self.assertEqual(before, after)

    def test_19_output_completeness(self) -> None:
        self.add_roi("708", {"roi_provenance": "expert_session_specific"})
        summary = run_audit(self.root, self.output)
        self.assertTrue(all(path.is_file() for path in expected_outputs(self.output)))
        self.assertEqual(len(summary["outputs"]), len(expected_outputs(self.output)))
        with (self.output / "session_roi_status.csv").open(encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 9)

    def test_20_direct_expert_roi_is_primary_candidate(self) -> None:
        result = classify_provenance(Path("session_708/roi_mask.npy"), {
            "roi_provenance": "expert_session_specific",
            "label_information_used": False,
            "session_specific": True,
        })
        self.assertEqual(result["classification"], CLASS_EXPERT)
        self.assertEqual(result["usable_for_primary_roi_decoding"], "YES")

    def test_21_cpu_only_and_output_must_be_inside_project(self) -> None:
        with self.assertRaises(ValueError):
            run_audit(self.root, self.output, device="cuda")
        with self.assertRaises(ValueError):
            run_audit(self.root, self.root.parent / "outside-audit", device="cpu")

    def test_22_negative_provenance_flags_are_not_positive_evidence(self) -> None:
        result = classify_provenance(Path("session_708/candidate_roi_mask.npy"), {
            "roi_provenance": "analyst_reconstructed_from_expert_indicated_approximate_region",
            "label_information_used": False,
            "used_searchlight_or_activation_map": False,
            "used_searchlight_accuracy_or_hotspot": False,
            "registration_used": False,
        })
        self.assertEqual(result["classification"], CLASS_RECONSTRUCTED)
        self.assertEqual(result["registration_used"], "NO")


if __name__ == "__main__":
    unittest.main()
