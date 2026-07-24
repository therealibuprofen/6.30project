from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_PATH = PROJECT_DIR / "scripts" / "multiframe" / "run_multiframe_benchmark.py"
SPEC = importlib.util.spec_from_file_location("run_multiframe_benchmark", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
multiframe_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = multiframe_script
SPEC.loader.exec_module(multiframe_script)

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.linear import ClassContrastivePCATransformer
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    TASK_CLASS_NAMES,
    load_block_sequence_session,
    task_run_dir_name,
)
from ultrasound_decoding.multiframe.evaluation import CHANCE_LEVEL, metrics_with_flags
from ultrasound_decoding.multiframe.models import (
    CNN2DLSTM,
    CNN2DMeanPool,
    CNN2DTemporal1D,
    MODEL_DESCRIPTIONS,
    SmallCNNFrameEncoder,
    build_multiframe_model,
    count_trainable_parameters,
    encoder_architecture_signature,
)
from ultrasound_decoding.multiframe.training import (
    normalize_blocks_train_fold_only,
    order_sensitivity_for_trained_sequence_model,
)


class MultiframeBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        data_path = PROJECT_DIR / "processed_data" / "block_sequences_v1" / "session_708_blocks.h5"
        if not data_path.exists():
            raise unittest.SkipTest("block_sequences_v1 data is not available")
        cls.binary = load_block_sequence_session(PROJECT_DIR, "708", "binary")
        cls.stimulus = load_block_sequence_session(PROJECT_DIR, "708", "stimulus_type")

    def test_clean4_samples_have_required_block_shape_and_integrity(self) -> None:
        data = self.binary
        self.assertEqual(tuple(data.X.shape[1:]), EXPECTED_BLOCK_SHAPE)
        self.assertEqual(tuple(data.X[0].shape), (4, 128, 501))
        self.assertTrue((data.metadata["session"].astype(str) == "708").all())
        self.assertTrue((data.metadata["n_frames_clean4"].astype(int) == 4).all())
        self.assertTrue(np.all(np.diff(data.clean4_relative_time_s, axis=1) > 0))
        self.assertTrue(np.all(np.diff(data.clean4_original_frame_indices, axis=1) > 0))

    def test_binary_and_stimulus_type_counts_per_complete_cycle(self) -> None:
        binary_counts = self.binary.metadata.groupby("cycle")["block_id"].size()
        self.assertTrue(binary_counts.eq(4).all())
        for _, rows in self.binary.metadata.groupby("cycle"):
            self.assertEqual(set(rows["binary_label_int"].astype(int)), {0, 1})
            self.assertEqual(int((rows["binary_label_int"].astype(int) == 1).sum()), 2)

        stimulus_counts = self.stimulus.metadata.groupby("cycle")["block_id"].size()
        self.assertTrue(stimulus_counts.eq(2).all())
        self.assertEqual(set(self.stimulus.metadata["block_name"]), {"grating", "dot"})
        self.assertEqual(TASK_CLASS_NAMES["stimulus_type"], {0: "dot", 1: "grating"})

    def test_grouped_cv_never_splits_a_cycle_or_predicts_a_block_twice(self) -> None:
        data = self.binary
        splits = grouped_cv_splits(data.groups, max_folds=10)
        self.assertEqual(len(splits), min(10, data.n_cycles))
        all_test_indices: list[int] = []
        for train_idx, test_idx in splits:
            self.assertFalse(set(data.groups[train_idx]) & set(data.groups[test_idx]))
            all_test_indices.extend(int(value) for value in test_idx)
        self.assertEqual(sorted(all_test_indices), list(range(data.n_blocks)))
        self.assertEqual(len(all_test_indices), len(set(all_test_indices)))

    def test_flat4_preprocessing_keeps_time_order_then_pixels(self) -> None:
        X = np.arange(2 * 4 * 2 * 3, dtype=np.float32).reshape(2, 4, 2, 3)
        flat = multiframe_script.preprocess_blocks_flat4(X)
        expected = np.arcsinh(X.astype(np.float64)).reshape(2, -1)
        np.testing.assert_allclose(flat, expected)
        np.testing.assert_allclose(flat[0, :6], np.arcsinh(X[0, 0]).reshape(-1))
        np.testing.assert_allclose(flat[0, 6:12], np.arcsinh(X[0, 1]).reshape(-1))

    def test_deep_normalization_uses_train_blocks_only(self) -> None:
        X_train = np.ones((2, 4, 2, 3), dtype=np.float32)
        X_test = np.full((1, 4, 2, 3), 100.0, dtype=np.float32)
        Xn_train, Xn_test, audit = normalize_blocks_train_fold_only(
            X_train,
            X_test,
            session="synthetic",
            task="binary",
            method="cnn2d_meanpool",
            seed=0,
            fold=1,
            train_cycles="0",
            test_cycles="1",
        )
        self.assertTrue(np.allclose(Xn_train, 0.0))
        self.assertFalse(np.allclose(Xn_test, 0.0))
        self.assertFalse(audit["target_used_for_stats"])
        self.assertEqual(audit["n_train_frames_for_stats"], 8)
        self.assertEqual(audit["statistics_scope"], "train_blocks_all_four_frames_only")

    def test_sequence_models_share_the_same_smallcnn_encoder(self) -> None:
        models = [CNN2DMeanPool(2), CNN2DLSTM(2), CNN2DTemporal1D(2)]
        signatures = [encoder_architecture_signature() for _ in models]
        self.assertEqual(len(set(signatures)), 1)
        for model in models:
            self.assertIsInstance(model.encoder, SmallCNNFrameEncoder)
            self.assertEqual(model.encoder_feature_dim, 512)
            self.assertEqual(model.temporal_length, 4)

    def test_lstm_time_dimension_is_four_not_image_height(self) -> None:
        model = CNN2DLSTM(2)
        self.assertEqual(model.lstm.input_size, 512)
        self.assertEqual(model.lstm.hidden_size, 32)
        x = torch.zeros(2, 4, 1, 128, 501)
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_temporal_conv1d_runs_along_four_frame_axis(self) -> None:
        model = CNN2DTemporal1D(2)
        captured: dict[str, tuple[int, ...]] = {}

        def hook(_module, inputs):
            captured["shape"] = tuple(inputs[0].shape)

        handle = model.temporal_conv[0].register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(torch.zeros(2, 4, 1, 128, 501))
        finally:
            handle.remove()
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(captured["shape"], (2, 512, 4))
        self.assertEqual(model.temporal_conv[0].kernel_size, (3,))
        self.assertEqual(model.temporal_axis, "time")

    def test_late_fusion_model_is_current_single_frame_smallcnn(self) -> None:
        model = build_multiframe_model("single_frame_late_fusion", n_classes=2)
        self.assertGreater(count_trainable_parameters(model), 0)
        with torch.no_grad():
            logits = model(torch.zeros(2, 1, 128, 501))
        self.assertEqual(tuple(logits.shape), (2, 2))

    def test_order_sensitivity_keeps_labels_and_reports_required_columns(self) -> None:
        model = CNN2DLSTM(2)
        X = np.zeros((2, 4, 128, 501), dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        result = order_sensitivity_for_trained_sequence_model(
            model,
            X,
            y,
            np.asarray([0, 1], dtype=np.int64),
            device="cpu",
            batch_size=2,
        )
        self.assertFalse(result["labels_modified"])
        for column in ["original_order_ba", "reverse_order_ba", "shuffled_order_ba", "reverse_drop", "shuffle_drop"]:
            self.assertIn(column, result)
            self.assertTrue(np.isfinite(result[column]))

    def test_results_and_protocol_constants_are_explicit(self) -> None:
        self.assertEqual(CHANCE_LEVEL, 0.5)
        self.assertEqual(multiframe_script.DEFAULT_SEEDS, [0, 1, 2])
        self.assertEqual(task_run_dir_name("binary"), "block_clean4_binary_v1")
        self.assertEqual(task_run_dir_name("stimulus_type"), "block_clean4_stimulus_type_v1")
        self.assertIn("adapted", MODEL_DESCRIPTIONS["cnn2d_temporal1d"])
        self.assertNotIn("exact reproduction", MODEL_DESCRIPTIONS["cnn2d_temporal1d"].lower())
        self.assertEqual(ClassContrastivePCATransformer.__module__, "ultrasound_decoding.linear")

    def test_metrics_are_finite_and_block_level_for_late_fusion_reference(self) -> None:
        y_true = np.asarray([0, 1, 0, 1])
        y_pred = np.asarray([0, 1, 1, 1])
        metrics = metrics_with_flags(y_true, y_pred)
        self.assertTrue(np.isfinite(pd.Series(metrics[metric] for metric in ["accuracy", "balanced_accuracy", "macro_f1"])).all())
        self.assertFalse(metrics["prediction_is_single_class"])
        rows = multiframe_script.session_completeness_rows(
            self.binary,
            ["single_frame_late_fusion"],
            [0, 1, 2],
            pd.DataFrame(
                [
                    {
                        "method": "single_frame_late_fusion",
                        "seed": 0,
                        "accuracy": 1.0,
                        "balanced_accuracy": 1.0,
                        "macro_f1": 1.0,
                    }
                ]
            ),
            pd.DataFrame([{"method": "single_frame_late_fusion", "seed": 0}]),
            pd.DataFrame(
                {
                    "method": ["single_frame_late_fusion"] * self.binary.n_blocks,
                    "seed": [0] * self.binary.n_blocks,
                    "block_id": self.binary.metadata["block_id"].tolist(),
                }
            ),
        )
        self.assertTrue(rows[0]["all_test_blocks_predicted_once"])


if __name__ == "__main__":
    unittest.main()
