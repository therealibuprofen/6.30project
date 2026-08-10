from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib.axes
import numpy as np
import pandas as pd
import torch
from torch import nn


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

MERGE_SCRIPT_PATH = PROJECT_DIR / "scripts" / "multiframe" / "merge_multiframe_runs.py"
MERGE_SPEC = importlib.util.spec_from_file_location("merge_multiframe_runs", MERGE_SCRIPT_PATH)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
merge_script = importlib.util.module_from_spec(MERGE_SPEC)
sys.modules[MERGE_SPEC.name] = merge_script
MERGE_SPEC.loader.exec_module(merge_script)

EPOCH_SCRIPT_PATH = PROJECT_DIR / "scripts" / "multiframe" / "run_epoch_sensitivity.py"
EPOCH_SPEC = importlib.util.spec_from_file_location("run_epoch_sensitivity", EPOCH_SCRIPT_PATH)
assert EPOCH_SPEC is not None and EPOCH_SPEC.loader is not None
epoch_script = importlib.util.module_from_spec(EPOCH_SPEC)
sys.modules[EPOCH_SPEC.name] = epoch_script
EPOCH_SPEC.loader.exec_module(epoch_script)

EXPORT_SCRIPT_PATH = PROJECT_DIR / "scripts" / "data" / "export_block_sequences.py"
EXPORT_SPEC = importlib.util.spec_from_file_location("export_block_sequences", EXPORT_SCRIPT_PATH)
assert EXPORT_SPEC is not None and EXPORT_SPEC.loader is not None
export_block_sequences = importlib.util.module_from_spec(EXPORT_SPEC)
sys.modules[EXPORT_SPEC.name] = export_block_sequences
EXPORT_SPEC.loader.exec_module(export_block_sequences)

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.linear import ClassContrastivePCATransformer
from ultrasound_decoding.deep import FCNN
from ultrasound_decoding.multiframe.dataset import (
    EXPECTED_BLOCK_SHAPE,
    EXPECTED_SESSIONS,
    TASK_CLASS_NAMES,
    load_block_sequence_session,
    task_run_dir_name,
)
from ultrasound_decoding.multiframe.evaluation import CHANCE_LEVEL, metrics_with_flags
from ultrasound_decoding.multiframe.plotting import plot_parameter_count_vs_test_ba
from ultrasound_decoding.multiframe.models import (
    CNN2DLSTM,
    CNN2DMeanPool,
    CNN2DTemporal1D,
    FCNNFrameEncoder,
    FCNNLSTM,
    FCNNMeanPool,
    METHOD_USES_TEMPORAL_ORDER,
    MODEL_DESCRIPTIONS,
    SmallCNNFrameEncoder,
    build_multiframe_model,
    count_trainable_parameters,
    encoder_architecture_signature,
    fcnn_frame_encoder_architecture_signature,
)
from ultrasound_decoding.multiframe.training import (
    DeepTrainingConfig,
    blocks_to_frame_tensor,
    blocks_to_sequence_tensor,
    load_multiframe_checkpoint,
    normalize_blocks_train_fold_only,
    order_sensitivity_for_trained_sequence_model,
    predict_probabilities,
    save_fold_checkpoint,
    train_sequence_fold,
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

    def test_fcnn_frame_encoder_matches_official_fcnn_bottleneck(self) -> None:
        official = FCNN(input_shape=(128, 501), n_classes=2)
        encoder = FCNNFrameEncoder(input_shape=(128, 501))
        self.assertEqual(fcnn_frame_encoder_architecture_signature(), (
            ("MaxPool2d", (2, 2)),
            ("Flatten", None),
            ("Linear", (64 * 250, 3)),
            ("ReLU", None),
        ))
        self.assertEqual([type(layer) for layer in encoder.layers], [type(layer) for layer in official[:4]])
        self.assertEqual(encoder.layers[2].out_features, 3)
        with torch.no_grad():
            z = encoder(torch.zeros(2, 1, 128, 501))
        self.assertEqual(tuple(z.shape), (2, 3))

    def test_fcnn_meanpool_uses_time_dim_one_and_shared_encoder(self) -> None:
        model = FCNNMeanPool(n_classes=3)
        self.assertIsInstance(model.encoder, FCNNFrameEncoder)
        self.assertEqual(len([module for module in model.modules() if isinstance(module, FCNNFrameEncoder)]), 1)

        class DummyEncoder(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x[:, 0, 0, :3]

        model.encoder = DummyEncoder()
        model.classifier = nn.Identity()
        x = torch.zeros(2, 4, 1, 128, 501)
        values = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        x[:, :, 0, 0, :3] = values
        with torch.no_grad():
            pooled = model(x)
        np.testing.assert_allclose(pooled.numpy(), values.mean(dim=1).numpy())

    def test_fcnn_lstm_uses_three_dimensional_input_and_hidden_size_eight(self) -> None:
        model = FCNNLSTM(n_classes=2)
        captured: dict[str, tuple[int, ...]] = {}

        def hook(_module, inputs):
            captured["shape"] = tuple(inputs[0].shape)

        handle = model.lstm.register_forward_pre_hook(hook)
        try:
            with torch.no_grad():
                logits = model(torch.zeros(2, 4, 1, 128, 501))
        finally:
            handle.remove()
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(captured["shape"], (2, 4, 3))
        self.assertEqual(model.lstm.input_size, 3)
        self.assertEqual(model.lstm.hidden_size, 8)

    def test_fcnn_late_fusion_reshape_and_label_repeat_order_are_consistent(self) -> None:
        X = np.zeros((2, 4, 2, 3), dtype=np.float32)
        for block_i in range(2):
            for frame_i in range(4):
                X[block_i, frame_i, 0, 0] = block_i * 10 + frame_i
        frames = blocks_to_frame_tensor(X)
        np.testing.assert_allclose(frames[:, 0, 0, 0].numpy(), np.asarray([0, 1, 2, 3, 10, 11, 12, 13]))
        labels = np.asarray([5, 9])
        np.testing.assert_array_equal(np.repeat(labels, 4), np.asarray([5, 5, 5, 5, 9, 9, 9, 9]))
        frame_probs = np.arange(2 * 4 * 2, dtype=np.float32).reshape(8, 2)
        block_probs = frame_probs.reshape(2, 4, 2).mean(axis=1)
        np.testing.assert_allclose(block_probs[0], frame_probs[:4].mean(axis=0))
        np.testing.assert_allclose(block_probs[1], frame_probs[4:].mean(axis=0))

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

    def test_order_sensitivity_prediction_rows_keep_block_identity(self) -> None:
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
            session="708",
            task="binary",
            method="cnn2d_lstm",
            seed=0,
            fold=1,
            test_idx=np.asarray([0, 1]),
            metadata=self.binary.metadata,
            class_names=self.binary.class_names,
            include_prediction_rows=True,
        )
        rows = pd.DataFrame(result["prediction_rows"])
        self.assertEqual(set(rows["order_condition"]), {"original", "reverse", "fixed_shuffle"})
        self.assertTrue(rows.groupby(["block_id", "order_condition"]).size().eq(1).all())
        self.assertEqual(rows.groupby("block_id")["truth"].nunique().max(), 1)
        self.assertIn("prob_no_stimulus", rows.columns)
        self.assertIn("prob_stimulus", rows.columns)

    def test_fcnn_order_plot_handles_integer_session_index(self) -> None:
        order_oof = pd.DataFrame(
            {
                "session": [708, 708, 708, 709, 709, 709],
                "task": ["binary"] * 6,
                "method": ["fcnn_lstm"] * 6,
                "seed": [0] * 6,
                "order_condition": ["original", "reverse", "fixed_shuffle"] * 2,
                "balanced_accuracy": [0.8, 0.6, 0.7, 0.75, 0.55, 0.65],
            }
        )
        captured_heights: list[float] = []
        original_bar = matplotlib.axes.Axes.bar

        def capture_bar(ax, x, height, *args, **kwargs):
            captured_heights.extend(np.asarray(height, dtype=float).reshape(-1).tolist())
            return original_bar(ax, x, height, *args, **kwargs)

        matplotlib.axes.Axes.bar = capture_bar
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = merge_script.plot_fcnn_order_sensitivity(order_oof, "binary", Path(tmpdir))
        finally:
            matplotlib.axes.Axes.bar = original_bar
        self.assertEqual(len(paths), 2)
        self.assertTrue(captured_heights)
        self.assertTrue(np.isfinite(np.asarray(captured_heights, dtype=float)).all())
        self.assertIn(0.8, captured_heights)

    def test_parameter_count_plot_uses_positive_log_axis_for_zero_parameter_models(self) -> None:
        master = pd.DataFrame(
            {
                "session": [708, 708],
                "task": ["binary", "binary"],
                "method": ["pca_lda_flat4", "fcnn_lstm"],
                "method_display": ["PCA+LDA flat4", "FCNN-LSTM"],
                "seed": [0, 0],
                "model_parameters": [0, 48437],
                "accuracy": [0.7, 0.8],
                "balanced_accuracy": [0.7, 0.8],
                "macro_f1": [0.7, 0.8],
                "prediction_is_single_class": [False, False],
            }
        )
        captured_scales: list[str] = []
        captured_ticks: list[float] = []
        original_set_xscale = matplotlib.axes.Axes.set_xscale
        original_set_xticks = matplotlib.axes.Axes.set_xticks

        def capture_set_xscale(ax, value, *args, **kwargs):
            captured_scales.append(str(value))
            return original_set_xscale(ax, value, *args, **kwargs)

        def capture_set_xticks(ax, ticks, *args, **kwargs):
            captured_ticks.extend(np.asarray(ticks, dtype=float).reshape(-1).tolist())
            return original_set_xticks(ax, ticks, *args, **kwargs)

        matplotlib.axes.Axes.set_xscale = capture_set_xscale
        matplotlib.axes.Axes.set_xticks = capture_set_xticks
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = plot_parameter_count_vs_test_ba(master, "binary", Path(tmpdir), "parameter_count_vs_test_ba")
        finally:
            matplotlib.axes.Axes.set_xscale = original_set_xscale
            matplotlib.axes.Axes.set_xticks = original_set_xticks
        self.assertEqual(len(paths), 2)
        self.assertIn("log", captured_scales)
        self.assertTrue(captured_ticks)
        self.assertTrue((np.asarray(captured_ticks, dtype=float) > 0).all())

    def test_merge_accepts_complementary_session_method_coverage(self) -> None:
        def write_session(run_dir: Path, session: str, methods: list[str]) -> None:
            session_dir = run_dir / f"session_{session}"
            session_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "session": session,
                "task": "binary",
                "input_shape": [4, 128, 501],
                "data_version": "block_sequences_v1_clean4",
                "cv_group": "cycle",
                "max_folds": 10,
                "methods": methods,
                "seeds": [0, 1, 2],
                "class_mapping": {"0": "no_stimulus", "1": "stimulus"},
                "deep_config": {
                    "optimizer": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.001,
                    "batch_size": 16,
                    "max_epochs": 40,
                    "loss": "cross_entropy",
                },
            }
            (session_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "session": session,
                        "task": "binary",
                        "fold": 1,
                        "train_cycles": "1,2",
                        "test_cycles": "0",
                        "n_train_blocks": 8,
                        "n_test_blocks": 4,
                    }
                ]
            ).to_csv(session_dir / "split_manifest.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "session": session,
                        "task": "binary",
                        "method": method,
                        "seed": 0,
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "macro_f1": 0.5,
                        "prediction_is_single_class": False,
                        "model_parameters": 0,
                    }
                    for method in methods
                ]
            ).to_csv(session_dir / "master_summary.csv", index=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            additional = root / "additional"
            write_session(base, "626", ["pca_lda_flat4", "fcnn_lstm"])
            write_session(base, "708", ["pca_lda_flat4"])
            write_session(additional, "708", ["fcnn_lstm"])
            methods, task, sessions, seeds = merge_script.check_compatibility(base, additional)
            self.assertEqual(task, "binary")
            self.assertEqual(sessions, ["626", "708"])
            self.assertEqual(seeds, [0, 1, 2])
            self.assertEqual(set(methods), {"pca_lda_flat4", "fcnn_lstm"})

            write_session(additional, "626", ["fcnn_lstm"])
            with self.assertRaisesRegex(ValueError, "same session/method"):
                merge_script.check_compatibility(base, additional)

    def test_epoch_sensitivity_gap_keeps_epoch_specific_final_history(self) -> None:
        fold_rows = []
        history_rows = []
        for epochs, train_accuracy, train_loss in [(10, 0.61, 0.9), (20, 0.72, 0.6), (40, 0.93, 0.2)]:
            fold_rows.append(
                {
                    "session": "710",
                    "task": "binary",
                    "method": "fcnn_lstm",
                    "seed": 0,
                    "fold": 1,
                    "epochs": epochs,
                    "n_train_blocks": 64,
                    "n_test_blocks": 8,
                    "accuracy": 0.5 + epochs / 200.0,
                    "balanced_accuracy": 0.4 + epochs / 200.0,
                }
            )
            history_rows.append(
                {
                    "session": "710",
                    "task": "binary",
                    "method": "fcnn_lstm",
                    "seed": 0,
                    "fold": 1,
                    "epochs": epochs,
                    "epoch": 1,
                    "train_accuracy": 0.1,
                    "train_loss": 2.0,
                }
            )
            history_rows.append(
                {
                    "session": "710",
                    "task": "binary",
                    "method": "fcnn_lstm",
                    "seed": 0,
                    "fold": 1,
                    "epochs": epochs,
                    "epoch": epochs,
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                }
            )
        gap = epoch_script.fixed_train_test_gap_table(pd.DataFrame(fold_rows), pd.DataFrame(history_rows))
        self.assertEqual(len(gap), 3)
        self.assertFalse(gap.duplicated(["session", "method", "epochs", "seed", "fold"]).any())
        by_epoch = gap.set_index("epochs")
        self.assertEqual(int(by_epoch.loc[10, "final_epoch"]), 10)
        self.assertEqual(int(by_epoch.loc[20, "final_epoch"]), 20)
        self.assertEqual(int(by_epoch.loc[40, "final_epoch"]), 40)
        self.assertAlmostEqual(float(by_epoch.loc[10, "final_train_accuracy"]), 0.61)
        self.assertAlmostEqual(float(by_epoch.loc[20, "final_train_loss"]), 0.6)
        self.assertAlmostEqual(
            float(by_epoch.loc[40, "generalization_gap"]),
            float(by_epoch.loc[40, "final_train_accuracy"] - by_epoch.loc[40, "test_balanced_accuracy"]),
        )

    def test_results_and_protocol_constants_are_explicit(self) -> None:
        self.assertEqual(CHANCE_LEVEL, 0.5)
        self.assertEqual(multiframe_script.DEFAULT_SEEDS, [0, 1, 2])
        for session in ["626", "628"]:
            self.assertIn(session, EXPECTED_SESSIONS)
            self.assertIn(session, export_block_sequences.TARGET_SESSIONS)
        self.assertEqual(export_block_sequences.EXPECTED_COMPLETE_CYCLES["626"], 8)
        self.assertEqual(export_block_sequences.EXPECTED_COMPLETE_CYCLES["628"], 8)
        self.assertTrue(METHOD_USES_TEMPORAL_ORDER["cnn2d_lstm"])
        self.assertTrue(METHOD_USES_TEMPORAL_ORDER["cnn2d_temporal1d"])
        self.assertTrue(METHOD_USES_TEMPORAL_ORDER["fcnn_lstm"])
        self.assertFalse(METHOD_USES_TEMPORAL_ORDER["fcnn_meanpool"])
        parameter_rows = pd.DataFrame(multiframe_script.parameter_audit_rows(["cnn2d_lstm", "fcnn_lstm", "fcnn_meanpool"]))
        self.assertTrue(parameter_rows.set_index("method").loc["cnn2d_lstm", "uses_temporal_order"])
        self.assertTrue(parameter_rows.set_index("method").loc["fcnn_lstm", "temporal_adaptation"])
        self.assertFalse(parameter_rows.set_index("method").loc["fcnn_meanpool", "uses_temporal_order"])
        self.assertFalse(parameter_rows["uses_fcnn_paper_32"].astype(bool).any())
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

    def test_checkpoint_reload_preserves_fcnn_predictions_and_pixel_stats(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(4, 4, 128, 501)).astype(np.float32)
        X_test = rng.normal(size=(2, 4, 128, 501)).astype(np.float32)
        y_train = np.asarray([0, 1, 0, 1], dtype=np.int64)
        classes = np.asarray([0, 1], dtype=np.int64)
        result = train_sequence_fold(
            "fcnn_meanpool",
            X_train,
            y_train,
            X_test,
            classes,
            session="synthetic",
            task="binary",
            fold=1,
            seed=0,
            train_cycles="0,1",
            test_cycles="2",
            config=DeepTrainingConfig(max_epochs=1, batch_size=2),
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            manifest = save_fold_checkpoint(
                checkpoint_path,
                result,
                classes=classes,
                session="synthetic",
                task="binary",
                seed=0,
                fold=1,
                train_cycles="0,1",
                test_cycles="2",
                config=DeepTrainingConfig(max_epochs=1, batch_size=2),
                code_version="test",
            )
            reloaded, payload = load_multiframe_checkpoint(checkpoint_path)
            probs = predict_probabilities(
                reloaded,
                blocks_to_sequence_tensor(result.X_test_normalized),
                device="cpu",
                batch_size=2,
            )
        np.testing.assert_allclose(probs, result.probabilities, atol=1e-7)
        self.assertEqual(payload["normalization_mean"].shape, (1, 128, 501))
        self.assertEqual(payload["normalization_std"].shape, (1, 128, 501))
        self.assertGreater(payload["normalization_mean"].size, 1)
        self.assertEqual(manifest["status"], "available")


if __name__ == "__main__":
    unittest.main()
