from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODULE_PATH = PROJECT_DIR / "scripts" / "temporal" / "run_fixed_window_analysis.py"
SPEC = importlib.util.spec_from_file_location("run_fixed_window_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fixed_window_analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixed_window_analysis
SPEC.loader.exec_module(fixed_window_analysis)

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.datasets import make_fixed_temporal_windows


def synthetic_stimulus_type_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    index = 1
    for cycle in [0, 1]:
        for block_name, block_start_s in [("grating", 0.0), ("dot", 60.0)]:
            for position, offset_s in enumerate([10.0, 14.0, 18.0, 22.0]):
                rows.append(
                    {
                        "index": index,
                        "cycle": cycle,
                        "center_s": cycle * 120.0 + block_start_s + offset_s,
                        "center_in_cycle_s": block_start_s + offset_s,
                        "block_start_s": block_start_s,
                        "block_name": block_name,
                        "binary_label": "stimulus",
                        "after_analysis_limit": True,
                        "complete_cycle": True,
                        "clean_middle": True,
                        "selected_before_task": True,
                        "block_offset_s": offset_s,
                        "position": position,
                    }
                )
                labels.append(block_name)
                index += 1
    X = np.arange(len(rows) * 4, dtype=np.float32).reshape(len(rows), 2, 2)
    return X, np.asarray(labels), pd.DataFrame(rows)


def synthetic_binary_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    labels = []
    index = 1
    blocks = [
        ("grating", "stimulus", 0.0, [10.0, 14.0, 18.0, 22.0]),
        ("stop_after_grating", "no_stimulus", 30.0, [8.0, 12.0, 16.0, 20.0]),
        ("dot", "stimulus", 60.0, [10.0, 14.0, 18.0, 22.0]),
        ("static", "no_stimulus", 90.0, [8.0, 12.0, 16.0, 20.0]),
    ]
    for cycle in [0, 1]:
        for block_name, binary_label, block_start_s, offsets in blocks:
            for position, offset_s in enumerate(offsets):
                rows.append(
                    {
                        "index": index,
                        "cycle": cycle,
                        "center_s": cycle * 120.0 + block_start_s + offset_s,
                        "center_in_cycle_s": block_start_s + offset_s,
                        "block_start_s": block_start_s,
                        "block_name": block_name,
                        "binary_label": binary_label,
                        "after_analysis_limit": True,
                        "complete_cycle": True,
                        "clean_middle": True,
                        "selected_before_task": True,
                        "block_offset_s": offset_s,
                        "position": position,
                    }
                )
                labels.append(binary_label)
                index += 1
    X = np.arange(len(rows) * 4, dtype=np.float32).reshape(len(rows), 2, 2)
    return X, np.asarray(labels), pd.DataFrame(rows)


class FixedWindowAnalysisTests(unittest.TestCase):
    def test_fixed_window_validation_accepts_one_window_per_cycle_block(self) -> None:
        X, y, meta = synthetic_stimulus_type_data()
        Xw, yw, groups, window_meta = make_fixed_temporal_windows(
            X, y, meta, window_size=2, window_start_position=1
        )
        exp = fixed_window_analysis.Experiment(
            session="synthetic",
            task="stimulus_type",
            window_id="k2_p1-2",
            positions=(1, 2),
        )
        fixed_window_analysis.validate_fixed_window_samples(
            Xw, yw, groups, window_meta, meta, exp
        )
        self.assertEqual(Xw.shape, (4, 2, 2, 2))
        self.assertEqual(window_meta.groupby(["cycle", "block_name"]).size().max(), 1)
        self.assertTrue(all(indices in {"2,3", "6,7", "10,11", "14,15"} for indices in window_meta["window_indices"]))

    def test_grouped_cv_has_no_cycle_overlap(self) -> None:
        groups = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
        splits = grouped_cv_splits(groups, max_folds=3)
        for train_idx, test_idx in splits:
            self.assertFalse(set(groups[train_idx]) & set(groups[test_idx]))

    def test_window_specs_are_global_consecutive_positions(self) -> None:
        for window_id, positions in fixed_window_analysis.WINDOW_SPECS:
            expected = tuple(range(positions[0], positions[0] + len(positions)))
            self.assertEqual(positions, expected, window_id)
        self.assertEqual(len(fixed_window_analysis.WINDOW_SPECS), 10)

    def test_binary_block_dependent_times_are_not_silently_averaged(self) -> None:
        X, y, meta = synthetic_binary_data()
        _, _, _, window_meta = make_fixed_temporal_windows(
            X, y, meta, window_size=1, window_start_position=0
        )
        mapping = pd.DataFrame(
            fixed_window_analysis.block_time_rows_for_samples(
                session="synthetic",
                task="binary",
                window_id="k1_p0",
                samples=window_meta,
            )
        )
        centers = {
            row.block_name: row.nominal_center_s
            for row in mapping.itertuples(index=False)
        }
        self.assertEqual(centers["grating"], 10.0)
        self.assertEqual(centers["dot"], 10.0)
        self.assertEqual(centers["stop_after_grating"], 8.0)
        self.assertEqual(centers["static"], 8.0)

        info = fixed_window_analysis.time_info_for_window(
            mapping,
            session="synthetic",
            task="binary",
            window_id="k1_p0",
        )
        self.assertEqual(info["time_mapping_status"], "block_dependent_nominal")
        self.assertTrue(pd.isna(info["nominal_time_center_s"]))
        self.assertEqual(info["nominal_time_center_s_values"], "[8.0, 10.0]")


if __name__ == "__main__":
    unittest.main()
