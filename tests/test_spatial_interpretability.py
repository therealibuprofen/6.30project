from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.interpretability.aggregation import spearman_r, top10_overlap
from ultrasound_decoding.interpretability.common import (
    IMAGE_SHAPE,
    aggregate_patch_values,
    make_patch_specs,
    split_manifest_from_splits,
)


class SpatialInterpretabilityTests(unittest.TestCase):
    def test_patch_grid_covers_edges_without_overflow(self) -> None:
        patches = make_patch_specs(
            patch_height=32,
            patch_width=64,
            stride_height=16,
            stride_width=32,
        )
        self.assertEqual(len(patches), 105)
        self.assertEqual(max(p.row_end for p in patches), IMAGE_SHAPE[0])
        self.assertEqual(max(p.col_end for p in patches), IMAGE_SHAPE[1])
        coverage = np.zeros(IMAGE_SHAPE, dtype=np.int32)
        for patch in patches:
            self.assertLessEqual(patch.row_end, IMAGE_SHAPE[0])
            self.assertLessEqual(patch.col_end, IMAGE_SHAPE[1])
            coverage[patch.row_start : patch.row_end, patch.col_start : patch.col_end] += 1
        self.assertGreaterEqual(int(coverage.min()), 1)

    def test_split_manifest_rejects_cycle_leakage(self) -> None:
        groups = np.asarray([0, 0, 1, 1])
        with self.assertRaises(AssertionError):
            split_manifest_from_splits(
                "synthetic",
                [(np.asarray([0, 2]), np.asarray([1, 3]))],
                groups,
            )

    def test_negative_patch_values_are_not_clipped(self) -> None:
        patches = make_patch_specs(image_shape=(4, 4), patch_height=2, patch_width=2, stride_height=2, stride_width=2)
        values = np.asarray([-0.5, 0.25, -0.1, 0.9])
        arr, coverage = aggregate_patch_values(patches, values, image_shape=(4, 4))
        self.assertLess(float(np.nanmin(arr)), 0.0)
        self.assertEqual(int(coverage.min()), 1)

    def test_spearman_and_top_overlap_are_well_defined(self) -> None:
        a = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        b = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        rho, n = spearman_r(a, b)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(rho, 1.0)
        self.assertAlmostEqual(top10_overlap(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()

