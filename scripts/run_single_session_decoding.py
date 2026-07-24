#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.datasets import DEFAULT_ANALYSIS_LIMITS, load_monkey_session
from ultrasound_decoding.deep import (
    TORCH_MODEL_NAMES,
    fit_predict_torch,
    get_torch_model_defaults,
    torch_available,
)
from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix
from ultrasound_decoding.linear import fit_predict_linear, preprocess_frames
from ultrasound_decoding.preprocessing import SpatialFilterConfig


LINEAR_METHODS = {"pca_lda", "cpca_lda"}
TORCH_METHODS = set(TORCH_MODEL_NAMES)
ALL_METHODS = ["pca_lda", "cpca_lda", *TORCH_MODEL_NAMES]
DEFAULT_METHODS = ["pca_lda", "cpca_lda", "cnn", "cnn_lstm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grouped single-session decoding on monkey ultrasound data."
    )
    parser.add_argument("--session", default="708", help="Session folder under data/, e.g. 708")
    parser.add_argument(
        "--task",
        default="binary",
        choices=["binary", "stimulus_type"],
        help="binary = stimulus/no_stimulus; stimulus_type = grating/dot only",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=ALL_METHODS,
        help="Decoders to run",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=ALL_METHODS,
        default=None,
        help="Alias for --methods; accepts one or more decoder/model names.",
    )
    parser.add_argument("--no-clean-middle", action="store_true", help="Keep block-edge frames")
    parser.add_argument("--clean-margin-s", type=float, default=8.0)
    parser.add_argument(
        "--no-trim-incomplete-cycles",
        action="store_true",
        help="Keep cycles with fewer than 30 frames after analysis-limit filtering.",
    )
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Number of consecutive clean frames to concatenate for temporal-window decoding.",
    )
    parser.add_argument(
        "--window-mode",
        default="sliding",
        choices=["sliding", "fixed"],
        help="Temporal-window mode. fixed uses the same within-block start position in every cycle.",
    )
    parser.add_argument(
        "--fixed-window-start-position",
        type=int,
        default=None,
        help="Within-block start position for --window-mode fixed.",
    )
    parser.add_argument("--max-folds", type=int, default=10)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Backward-compatible alias for --max-epochs for Torch methods.",
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="Torch max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Torch batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Torch learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Torch weight decay")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience in epochs; model defaults are used when omitted.",
    )
    parser.add_argument(
        "--activation",
        default=None,
        choices=["elu", "relu", "gelu"],
        help="Activation override for configurable CNN models.",
    )
    parser.add_argument(
        "--normalization",
        default=None,
        choices=["none", "batchnorm", "groupnorm"],
        help="Normalization override for configurable CNN models.",
    )
    parser.add_argument("--dropout", type=float, default=None, help="Dropout override")
    parser.add_argument(
        "--optimizer",
        default=None,
        choices=["adam", "adamw"],
        help="Torch optimizer override.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seed set for Torch methods, e.g. --seeds 0 1 2.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Run label permutation test with this many shuffled-label repetitions.",
    )
    parser.add_argument(
        "--permutation-metric",
        default="macro_f1",
        choices=["accuracy", "balanced_accuracy", "macro_f1"],
        help="Metric used for permutation-test p-values.",
    )
    parser.add_argument(
        "--analysis-limit",
        default=None,
        help="Optional inclusive frame range like 1:180. Use 'default' for PPT-suggested limits.",
    )
    parser.add_argument(
        "--spatial-filter-method",
        default="none",
        choices=["none", "pillbox"],
        help="Spatial preprocessing applied to each fUS/Power Doppler frame before decoding.",
    )
    parser.add_argument(
        "--spatial-filter-radius",
        type=int,
        default=0,
        help="Pillbox spatial filter radius in voxels; use 0 with method=none.",
    )
    parser.add_argument(
        "--spatial-filter-mode",
        default="reflect",
        choices=["reflect", "edge", "symmetric"],
        help="Padding mode for spatial filtering at image boundaries.",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="Directory for summary/metrics/predictions outputs. Defaults to reports/decoding.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional suffix appended to output file stems to keep runs separate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output files for the resolved output stem.",
    )
    args = parser.parse_args()
    if args.model is not None:
        args.methods = args.model
    return args


def parse_analysis_limit(session: str, value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if value == "default":
        return DEFAULT_ANALYSIS_LIMITS.get(session)
    lo, hi = value.split(":")
    return int(lo), int(hi)


def valid_fold(y_train: np.ndarray, y_test: np.ndarray) -> bool:
    return len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2


def output_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "base": base,
        "summary": base / "summary",
        "metrics": base / "metrics",
        "predictions": base / "predictions",
        "samples": base / "samples",
        "audit": base / "audit",
        "permutation": base / "permutation",
        "models": base / "models",
        "history": base / "history",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "run"


def timestamp_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def stem_output_paths(out_dirs: dict[str, Path], stem: str) -> list[Path]:
    return [
        out_dirs["summary"] / f"{stem}_summary.json",
        out_dirs["metrics"] / f"{stem}_fold_metrics.csv",
        out_dirs["metrics"] / f"{stem}_overall_metrics.csv",
        out_dirs["metrics"] / f"{stem}_confusion_matrix.csv",
        out_dirs["audit"] / f"{stem}_normalization_stats.csv",
        out_dirs["history"] / f"{stem}_training_history.csv",
        out_dirs["predictions"] / f"{stem}_predictions.csv",
        out_dirs["samples"] / f"{stem}_samples.csv",
        out_dirs["audit"] / f"{stem}_cycle_selection_report.csv",
    ]


def resolve_output_stem(base_stem: str, args: argparse.Namespace, out_dirs: dict[str, Path]) -> tuple[str, str | None]:
    if args.run_id is not None:
        run_id = safe_run_id(args.run_id)
        stem = f"{base_stem}_{run_id}"
        existing = [path for path in stem_output_paths(out_dirs, stem) if path.exists()]
        if existing and not args.overwrite:
            paths = ", ".join(str(path) for path in existing[:3])
            raise FileExistsError(
                f"Output files already exist for run id '{run_id}': {paths}. "
                "Use a different --run-id or pass --overwrite."
            )
        return stem, run_id

    existing = [path for path in stem_output_paths(out_dirs, base_stem) if path.exists()]
    if existing and not args.overwrite:
        run_id = timestamp_run_id()
        return f"{base_stem}_{run_id}", run_id
    return base_stem, None


def fit_predict_method(
    method: str,
    X: np.ndarray,
    X_flat: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    classes: np.ndarray,
    args: argparse.Namespace,
    fold_i: int,
    seed: int,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, int | None, dict[str, object]]:
    if method in LINEAR_METHODS:
        pred, n_components = fit_predict_linear(
            method,
            X_flat[train_idx],
            y_train,
            X_flat[test_idx],
            pca_variance=args.pca_variance,
            standardize=args.window_size > 1,
        )
        metadata: dict[str, object] = {}
    else:
        result = fit_predict_torch(
            method,
            X[train_idx],
            y_train,
            X[test_idx],
            classes=classes,
            epochs=args.epochs,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=seed,
            train_groups=groups[train_idx],
            patience=args.patience,
            activation=args.activation,
            normalization=args.normalization,
            dropout=args.dropout,
            optimizer=args.optimizer,
            device=args.device,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
            return_metadata=True,
        )
        pred = result.predictions
        metadata = result.metadata
        n_components = None
    return pred, n_components, metadata


def run_method_cv(
    method: str,
    X: np.ndarray,
    X_flat: np.ndarray,
    y_for_training_and_scoring: np.ndarray,
    groups: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    classes: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    model_dir: Path | None = None,
    stem: str | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    method_true = []
    method_pred = []
    fold_rows = []
    skipped = []
    normalization_rows = []
    training_history_rows = []
    for fold_i, (train_idx, test_idx) in enumerate(splits, start=1):
        y_train = y_for_training_and_scoring[train_idx]
        y_test = y_for_training_and_scoring[test_idx]
        if not valid_fold(y_train, y_test):
            skipped.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "fold": fold_i,
                    "reason": "train or test fold has fewer than two classes",
                }
            )
            continue

        checkpoint_path = None
        if method in TORCH_METHODS and model_dir is not None and stem is not None:
            checkpoint_path = model_dir / f"{stem}_{method}_seed{seed}_fold{fold_i}_best.pt"
        pred, n_components, metadata = fit_predict_method(
            method,
            X,
            X_flat,
            y_train,
            groups,
            train_idx,
            test_idx,
            classes,
            args,
            fold_i,
            seed,
            checkpoint_path,
        )
        metrics = classification_metrics(y_test, pred)
        inner_validation = metadata.get("inner_validation", {}) if metadata else {}
        final_training = metadata.get("final_training", {}) if metadata else {}
        fold_rows.append(
            {
                "method": method,
                "seed": int(seed),
                "fold": fold_i,
                "test_cycles": ",".join(str(x) for x in np.unique(groups[test_idx])),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_components": n_components,
                "best_epoch": metadata.get("best_epoch") if metadata else None,
                "trained_epochs": metadata.get("trained_epochs") if metadata else None,
                "selection_trained_epochs": metadata.get("selection_trained_epochs")
                if metadata
                else None,
                "final_trained_epochs": metadata.get("final_trained_epochs") if metadata else None,
                "best_val_balanced_accuracy": metadata.get("best_val_balanced_accuracy")
                if metadata
                else None,
                "inner_validation_enabled": inner_validation.get("enabled")
                if inner_validation
                else None,
                "n_inner_train": inner_validation.get("n_inner_train")
                if inner_validation
                else None,
                "n_inner_val": inner_validation.get("n_val") if inner_validation else None,
                "inner_val_cycles": ",".join(str(x) for x in inner_validation.get("val_cycles", []))
                if inner_validation
                else None,
                "final_training_strategy": final_training.get("strategy")
                if final_training
                else None,
                "n_final_train": final_training.get("n_final_train") if final_training else None,
                "retrained_on_full_outer_train": final_training.get("retrained_on_full_outer_train")
                if final_training
                else None,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
                **metrics,
            }
        )
        for history_row in metadata.get("training_history", []) if metadata else []:
            training_history_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "fold": fold_i,
                    "test_cycles": ",".join(str(x) for x in np.unique(groups[test_idx])),
                    **history_row,
                }
            )
        if metadata and ("normalization_by_phase" in metadata or "normalization" in metadata):
            normalization_records = metadata.get("normalization_by_phase") or [metadata["normalization"]]
            for norm in normalization_records:
                normalization_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "fold": fold_i,
                        "test_cycles": ",".join(str(x) for x in np.unique(groups[test_idx])),
                        "phase": norm.get("phase"),
                        "config": norm.get("transform"),
                        "statistics_scope": norm.get("statistics_scope"),
                        "epsilon": norm.get("epsilon"),
                        "mean_mean": norm.get("mean_mean"),
                        "mean_std": norm.get("mean_std"),
                        "mean_min": norm.get("mean_min"),
                        "mean_max": norm.get("mean_max"),
                        "std_mean": norm.get("std_mean"),
                        "std_std": norm.get("std_std"),
                        "std_min": norm.get("std_min"),
                        "std_max": norm.get("std_max"),
                        "train_nan_count": norm.get("train_input_quality", {}).get("nan_count"),
                        "train_inf_count": norm.get("train_input_quality", {}).get("inf_count"),
                        "train_all_zero_images": norm.get("train_input_quality", {}).get("all_zero_images"),
                        "test_nan_count": norm.get("test_input_quality", {}).get("nan_count"),
                        "test_inf_count": norm.get("test_input_quality", {}).get("inf_count"),
                        "test_all_zero_images": norm.get("test_input_quality", {}).get("all_zero_images"),
                    }
                )
        method_true.extend(y_test.tolist())
        method_pred.extend(pred.tolist())
    return (
        np.asarray(method_true),
        np.asarray(method_pred),
        fold_rows,
        skipped,
        normalization_rows,
        training_history_rows,
    )


def main() -> None:
    args = parse_args()
    if args.window_mode == "fixed" and args.methods != ["pca_lda"]:
        raise ValueError("fixed temporal windows currently support only --methods pca_lda")
    out_base = Path(args.output_base) if args.output_base else PROJECT_DIR / "reports" / "decoding"
    out_dirs = output_dirs(out_base)
    spatial_filter = SpatialFilterConfig(
        method=args.spatial_filter_method,
        radius=args.spatial_filter_radius,
        mode=args.spatial_filter_mode,
    )

    X, y, groups, meta = load_monkey_session(
        PROJECT_DIR,
        session=args.session,
        task=args.task,
        clean_middle=not args.no_clean_middle,
        clean_margin_s=args.clean_margin_s,
        analysis_limit=parse_analysis_limit(args.session, args.analysis_limit),
        trim_incomplete_cycles=not args.no_trim_incomplete_cycles,
        window_size=args.window_size,
        window_mode=args.window_mode,
        fixed_window_start_position=args.fixed_window_start_position,
        spatial_filter=spatial_filter,
    )
    classes = np.unique(y)
    splits = grouped_cv_splits(groups, max_folds=args.max_folds)
    X_flat = preprocess_frames(X)
    if args.window_mode == "fixed":
        base_stem = (
            f"{args.session}_{args.task}_fixed_window"
            f"{args.window_size}_position{args.fixed_window_start_position}"
        )
    else:
        base_stem = (
            f"{args.session}_{args.task}"
            if args.window_size == 1
            else f"{args.session}_{args.task}_window{args.window_size}"
        )
    stem, resolved_run_id = resolve_output_stem(base_stem, args, out_dirs)
    if resolved_run_id is not None and args.run_id is None:
        print(f"Existing outputs found for {base_stem}; writing this run as {stem}")
    torch_seeds = [int(seed) for seed in (args.seeds if args.seeds is not None else [args.seed])]

    all_predictions = []
    fold_rows = []
    overall_rows = []
    confusion_rows = []
    confusion_by_method = {}
    skipped = []
    normalization_rows = []
    training_history_rows = []
    observed_metrics_by_method = {}
    for method in args.methods:
        if method in TORCH_METHODS and not torch_available():
            skipped.append({"method": method, "reason": "PyTorch is not installed"})
            continue

        seeds_for_method = torch_seeds if method in TORCH_METHODS else [int(args.seed)]
        for seed in seeds_for_method:
            (
                method_true,
                method_pred,
                method_fold_rows,
                method_skipped,
                method_normalization_rows,
                method_training_history_rows,
            ) = run_method_cv(
                method,
                X,
                X_flat,
                y,
                groups,
                splits,
                classes,
                args,
                seed=seed,
                model_dir=out_dirs["models"],
                stem=stem,
            )
            fold_rows.extend(method_fold_rows)
            skipped.extend(method_skipped)
            normalization_rows.extend(method_normalization_rows)
            training_history_rows.extend(method_training_history_rows)
            prediction_cursor = 0
            for row in method_fold_rows:
                fold_i = int(row["fold"])
                _, test_idx = splits[fold_i - 1]
                n_test = int(row["n_test"])
                fold_true = y[test_idx]
                fold_pred = method_pred[prediction_cursor : prediction_cursor + n_test]
                prediction_cursor += n_test
                for sample_i, truth, predicted in zip(test_idx, fold_true, fold_pred):
                    all_predictions.append(
                        {
                            "method": method,
                            "seed": int(seed),
                            "fold": fold_i,
                            "sample_i": int(sample_i),
                            "index": int(meta.loc[sample_i, "index"]),
                            "cycle": int(meta.loc[sample_i, "cycle"]),
                            "block_name": meta.loc[sample_i, "block_name"],
                            "truth": truth,
                            "pred": predicted,
                        }
                    )

            if len(method_true):
                overall = classification_metrics(method_true, method_pred)
                observed_metrics_by_method[(method, seed)] = overall
                cm = confusion_matrix(method_true, method_pred, classes)
                confusion_by_method[f"{method}/seed{seed}"] = {
                    "classes": classes.tolist(),
                    "matrix": cm.tolist(),
                }
                overall_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "n_test_predictions": int(len(method_true)),
                        **overall,
                    }
                )
                for i, truth_class in enumerate(classes):
                    for j, pred_class in enumerate(classes):
                        confusion_rows.append(
                            {
                                "method": method,
                                "seed": int(seed),
                                "truth": truth_class,
                                "pred": pred_class,
                                "count": int(cm[i, j]),
                            }
                        )
                print(f"{method} seed={seed}: {overall}, confusion={cm.tolist()}")

    permutation_rows = []
    permutation_pvalue_rows = []
    if args.n_permutations > 0:
        rng = np.random.default_rng(args.seed)
        runnable_method_seeds = [(row["method"], int(row["seed"])) for row in overall_rows]
        for method, seed in runnable_method_seeds:
            observed = observed_metrics_by_method[(method, seed)][args.permutation_metric]
            null_values = []
            for perm_i in range(1, args.n_permutations + 1):
                y_perm = rng.permutation(y)
                perm_true, perm_pred, _, perm_skipped, _, _ = run_method_cv(
                    method,
                    X,
                    X_flat,
                    y_perm,
                    groups,
                    splits,
                    classes,
                    args,
                    seed=seed,
                )
                skipped.extend(
                    {
                        **row,
                        "permutation": perm_i,
                        "reason": f"permutation: {row['reason']}",
                    }
                    for row in perm_skipped
                )
                if not len(perm_true):
                    continue
                perm_metrics = classification_metrics(perm_true, perm_pred)
                score = perm_metrics[args.permutation_metric]
                null_values.append(score)
                permutation_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "permutation": perm_i,
                        **perm_metrics,
                    }
                )
            if null_values:
                null_arr = np.asarray(null_values)
                p_value = (1.0 + float(np.sum(null_arr >= observed))) / (len(null_arr) + 1.0)
                permutation_pvalue_rows.append(
                    {
                        "method": method,
                        "seed": int(seed),
                        "metric": args.permutation_metric,
                        "observed": observed,
                        "n_permutations": int(len(null_arr)),
                        "null_mean": float(null_arr.mean()),
                        "null_std": float(null_arr.std(ddof=1)) if len(null_arr) > 1 else 0.0,
                        "null_min": float(null_arr.min()),
                        "null_max": float(null_arr.max()),
                        "p_value_greater_equal": p_value,
                    }
                )
                print(
                    f"{method} seed={seed} permutation {args.permutation_metric}: "
                    f"observed={observed:.4f}, null_mean={null_arr.mean():.4f}, p={p_value:.4f}"
                )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(all_predictions)
    overall_df = pd.DataFrame(overall_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    normalization_df = pd.DataFrame(normalization_rows)
    training_history_df = pd.DataFrame(training_history_rows)
    permutation_df = pd.DataFrame(permutation_rows)
    permutation_pvalue_df = pd.DataFrame(permutation_pvalue_rows)
    fold_df.to_csv(out_dirs["metrics"] / f"{stem}_fold_metrics.csv", index=False)
    overall_df.to_csv(out_dirs["metrics"] / f"{stem}_overall_metrics.csv", index=False)
    confusion_df.to_csv(out_dirs["metrics"] / f"{stem}_confusion_matrix.csv", index=False)
    normalization_df.to_csv(out_dirs["audit"] / f"{stem}_normalization_stats.csv", index=False)
    training_history_df.to_csv(out_dirs["history"] / f"{stem}_training_history.csv", index=False)
    pred_df.to_csv(out_dirs["predictions"] / f"{stem}_predictions.csv", index=False)
    meta.to_csv(out_dirs["samples"] / f"{stem}_samples.csv", index=False)
    selection_info = meta.attrs.get("selection_info", {})
    cycle_report = meta.attrs.get("cycle_report", [])
    pd.DataFrame(cycle_report).to_csv(
        out_dirs["audit"] / f"{stem}_cycle_selection_report.csv", index=False
    )
    if not permutation_df.empty:
        for (method, seed), method_perm_df in permutation_df.groupby(["method", "seed"], sort=True):
            method_perm_df.to_csv(
                out_dirs["permutation"] / f"{stem}_{method}_seed{seed}_permutation_distribution.csv",
                index=False,
            )
    if not permutation_pvalue_df.empty:
        for (method, seed), method_pvalue_df in permutation_pvalue_df.groupby(["method", "seed"], sort=True):
            method_pvalue_df.to_csv(
                out_dirs["permutation"] / f"{stem}_{method}_seed{seed}_permutation_pvalues.csv",
                index=False,
            )

    class_counts = pd.Series(y).value_counts().sort_index().to_dict()
    block_counts = meta["block_name"].value_counts().sort_index().to_dict()
    cycle_counts = meta["cycle"].value_counts().sort_index().to_dict()
    torch_methods_requested = [method for method in args.methods if method in TORCH_METHODS]
    torch_defaults = {
        method: get_torch_model_defaults(method).__dict__
        for method in torch_methods_requested
    }
    torch_overrides = {
        "epochs": args.epochs,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "activation": args.activation,
        "normalization": args.normalization,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "device": args.device,
    }
    summary = {
        "session": args.session,
        "task": args.task,
        "n_samples": int(len(X)),
        "classes": classes.tolist(),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "block_counts": {str(key): int(value) for key, value in block_counts.items()},
        "cycle_counts": {str(key): int(value) for key, value in cycle_counts.items()},
        "n_cycles": int(len(np.unique(groups))),
        "window_size": int(args.window_size),
        "window_mode": args.window_mode,
        "fixed_window_start_position": args.fixed_window_start_position,
        "clean_middle": not args.no_clean_middle,
        "clean_margin_s": args.clean_margin_s,
        "trim_incomplete_cycles": not args.no_trim_incomplete_cycles,
        "analysis_limit": parse_analysis_limit(args.session, args.analysis_limit),
        "base_output_stem": base_stem,
        "output_stem": stem,
        "run_id": resolved_run_id,
        "overwrite": bool(args.overwrite),
        "random_seed": int(args.seed),
        "random_seeds": torch_seeds,
        "torch_training": {
            "model_defaults": torch_defaults,
            "cli_overrides": torch_overrides,
        },
        "spatial_filter": spatial_filter.to_dict(),
        "data_shape_after_spatial_filter_and_windowing": [int(value) for value in X.shape],
        "selection_info": selection_info,
        "cv": {
            "group": "cycle",
            "n_splits": len(splits),
            "max_folds": args.max_folds,
        },
        "methods_requested": args.methods,
        "skipped": skipped,
        "permutation_test": {
            "n_permutations_requested": args.n_permutations,
            "metric": args.permutation_metric,
            "p_values": (
                permutation_pvalue_df.to_dict("records")
                if not permutation_pvalue_df.empty
                else []
            ),
        },
        "overall_metrics": (
            overall_df.to_dict("records") if not overall_df.empty else []
        ),
        "confusion_matrices": confusion_by_method,
        "mean_fold_metrics": (
            fold_df.groupby(["method", "seed"])[["accuracy", "balanced_accuracy", "macro_f1"]]
            .mean()
            .reset_index()
            .to_dict("records")
            if not fold_df.empty
            else []
        ),
    }
    summary["output_files"] = {
        "summary": str(out_dirs["summary"] / f"{stem}_summary.json"),
        "fold_metrics": str(out_dirs["metrics"] / f"{stem}_fold_metrics.csv"),
        "overall_metrics": str(out_dirs["metrics"] / f"{stem}_overall_metrics.csv"),
        "confusion_matrix": str(out_dirs["metrics"] / f"{stem}_confusion_matrix.csv"),
        "normalization_stats": str(out_dirs["audit"] / f"{stem}_normalization_stats.csv"),
        "training_history": str(out_dirs["history"] / f"{stem}_training_history.csv"),
        "predictions": str(out_dirs["predictions"] / f"{stem}_predictions.csv"),
        "samples": str(out_dirs["samples"] / f"{stem}_samples.csv"),
        "cycle_selection_report": str(out_dirs["audit"] / f"{stem}_cycle_selection_report.csv"),
        "permutation_by_model": {
            f"{row['method']}/seed{row['seed']}": {
                "distribution": str(
                    out_dirs["permutation"]
                    / f"{stem}_{row['method']}_seed{row['seed']}_permutation_distribution.csv"
                ),
                "pvalues": str(
                    out_dirs["permutation"]
                    / f"{stem}_{row['method']}_seed{row['seed']}_permutation_pvalues.csv"
                ),
            }
            for row in permutation_pvalue_rows
        },
    }
    (out_dirs["summary"] / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved reports under {out_dirs['base']}")


if __name__ == "__main__":
    main()
