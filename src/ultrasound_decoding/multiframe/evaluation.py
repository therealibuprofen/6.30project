from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ultrasound_decoding.evaluate import classification_metrics, confusion_matrix

from .dataset import TASK_CLASS_NAMES, csv_json
from .models import MODEL_DISPLAY_NAMES


CHANCE_LEVEL = 0.5


def metrics_with_flags(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = classification_metrics(np.asarray(y_true), np.asarray(y_pred))
    if not np.isfinite(np.asarray(list(metrics.values()), dtype=float)).all():
        raise AssertionError(f"non-finite metrics: {metrics}")
    return {
        **metrics,
        "prediction_is_single_class": bool(len(np.unique(y_pred)) == 1),
    }


def confusion_rows(
    *,
    session: str,
    task: str,
    method: str,
    seed: int | None,
    fold: int | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    classes = np.asarray(sorted(TASK_CLASS_NAMES[task]))
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred), classes)
    rows = []
    for i, truth in enumerate(classes):
        for j, pred in enumerate(classes):
            rows.append(
                {
                    "session": str(session),
                    "task": task,
                    "method": method,
                    "method_display": MODEL_DISPLAY_NAMES.get(method, method),
                    "seed": seed,
                    "fold": fold,
                    "truth": TASK_CLASS_NAMES[task][int(truth)],
                    "pred": TASK_CLASS_NAMES[task][int(pred)],
                    "count": int(cm[i, j]),
                }
            )
    return rows


def prediction_rows(
    *,
    session: str,
    task: str,
    method: str,
    seed: int | None,
    fold: int,
    test_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    metadata: pd.DataFrame,
) -> list[dict[str, Any]]:
    class_names = TASK_CLASS_NAMES[task]
    rows = []
    for local_i, (sample_i, truth, pred) in enumerate(zip(test_idx, y_true, y_pred)):
        row = metadata.iloc[int(sample_i)]
        payload = {
            "session": str(session),
            "task": task,
            "method": method,
            "method_display": MODEL_DISPLAY_NAMES.get(method, method),
            "seed": seed,
            "fold": int(fold),
            "sample_i": int(sample_i),
            "block_id": str(row["block_id"]),
            "cycle": int(row["cycle"]),
            "block_name": str(row["block_name"]),
            "truth": int(truth),
            "truth_name": class_names[int(truth)],
            "pred": int(pred),
            "pred_name": class_names[int(pred)],
            "clean4_original_frame_indices": str(row["clean4_original_frame_indices"]),
            "clean4_relative_time_s": str(row["clean4_relative_time_s"]),
        }
        if probabilities is not None and len(probabilities):
            for class_i, class_value in enumerate(sorted(class_names)):
                payload[f"prob_{class_names[class_value]}"] = float(probabilities[local_i, class_i])
        rows.append(payload)
    return rows


def mean_std_text(values: pd.Series | np.ndarray, digits: int = 3) -> str:
    series = pd.Series(values).dropna().astype(float)
    if series.empty:
        return ""
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return f"{mean:.{digits}f}+/-{std:.{digits}f}"


def method_summary_table(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame(columns=["Session", "Task", *MODEL_DISPLAY_NAMES.values()])
    rows: list[dict[str, Any]] = []
    for (session, task), group in master.groupby(["session", "task"], sort=True):
        row: dict[str, Any] = {"Session": str(session), "Task": str(task)}
        for method, display in MODEL_DISPLAY_NAMES.items():
            subset = group[group["method"] == method]
            if subset.empty:
                row[display] = ""
            elif subset["seed"].nunique(dropna=True) > 1:
                row[display] = mean_std_text(subset["balanced_accuracy"])
            else:
                row[display] = f"{float(subset['balanced_accuracy'].iloc[0]):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def seed_mean_summary(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in master.groupby(["session", "task", "method"], sort=True):
        session, task, method = keys
        values = group["balanced_accuracy"].astype(float)
        rows.append(
            {
                "session": str(session),
                "task": task,
                "method": method,
                "method_display": MODEL_DISPLAY_NAMES.get(method, method),
                "n_seeds": int(group["seed"].nunique(dropna=True)),
                "balanced_accuracy_mean": float(values.mean()),
                "balanced_accuracy_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "accuracy_mean": float(group["accuracy"].astype(float).mean()),
                "accuracy_std": float(group["accuracy"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
                "macro_f1_mean": float(group["macro_f1"].astype(float).mean()),
                "macro_f1_std": float(group["macro_f1"].astype(float).std(ddof=1)) if len(group) > 1 else 0.0,
                "prediction_is_single_class_any": bool(group["prediction_is_single_class"].astype(bool).any()),
                "chance_level": CHANCE_LEVEL,
            }
        )
    return pd.DataFrame(rows)


def vs_singleframe_reference(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    summary = seed_mean_summary(master)
    if summary.empty:
        return summary
    late = summary[summary["method"] == "single_frame_late_fusion"][
        ["session", "task", "balanced_accuracy_mean"]
    ].rename(columns={"balanced_accuracy_mean": "single_frame_late_fusion_ba"})
    out = summary.merge(late, on=["session", "task"], how="left")
    out["delta_vs_single_frame_late_fusion"] = (
        out["balanced_accuracy_mean"] - out["single_frame_late_fusion_ba"]
    )
    return out


def overfitting_audit_tables(
    fold_summary: pd.DataFrame,
    training_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_columns = [
        "session",
        "task",
        "method",
        "seed",
        "fold",
        "final_train_accuracy",
        "final_train_loss",
        "test_balanced_accuracy",
        "test_accuracy",
        "generalization_gap",
        "n_train_blocks",
        "n_test_blocks",
        "final_epoch",
    ]
    summary_columns = [
        "session",
        "task",
        "method",
        "method_display",
        "mean_train_accuracy",
        "mean_test_ba",
        "mean_generalization_gap",
        "std_generalization_gap",
        "fraction_train_accuracy_above_0_95",
        "fraction_test_ba_below_0_5",
    ]
    if fold_summary.empty or training_history.empty:
        return pd.DataFrame(columns=audit_columns), pd.DataFrame(columns=summary_columns)

    history = training_history.copy()
    fold_df = fold_summary.copy()
    key_cols = ["session", "task", "method", "seed", "fold"]
    for col in ["seed", "fold"]:
        history[col] = pd.to_numeric(history[col], errors="coerce").astype("Int64")
        fold_df[col] = pd.to_numeric(fold_df[col], errors="coerce").astype("Int64")
    history["epoch"] = pd.to_numeric(history["epoch"], errors="coerce")
    history = history.dropna(subset=["epoch"]).sort_values(key_cols + ["epoch"])
    final_history = history.groupby(key_cols, sort=True, dropna=False).tail(1).copy()
    merged = final_history.merge(
        fold_df[
            key_cols
            + [
                "n_train_blocks",
                "n_test_blocks",
                "accuracy",
                "balanced_accuracy",
            ]
        ],
        on=key_cols,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=audit_columns), pd.DataFrame(columns=summary_columns)

    audit = pd.DataFrame(
        {
            "session": merged["session"].astype(str),
            "task": merged["task"],
            "method": merged["method"],
            "seed": merged["seed"].astype(int),
            "fold": merged["fold"].astype(int),
            "final_train_accuracy": pd.to_numeric(merged["train_accuracy"], errors="coerce"),
            "final_train_loss": pd.to_numeric(merged["train_loss"], errors="coerce"),
            "test_balanced_accuracy": pd.to_numeric(merged["balanced_accuracy"], errors="coerce"),
            "test_accuracy": pd.to_numeric(merged["accuracy"], errors="coerce"),
            "n_train_blocks": pd.to_numeric(merged["n_train_blocks"], errors="coerce").astype(int),
            "n_test_blocks": pd.to_numeric(merged["n_test_blocks"], errors="coerce").astype(int),
            "final_epoch": pd.to_numeric(merged["epoch"], errors="coerce").astype(int),
        }
    )
    audit["generalization_gap"] = audit["final_train_accuracy"] - audit["test_balanced_accuracy"]
    audit = audit[audit_columns]

    rows: list[dict[str, Any]] = []
    for (session, task, method), group in audit.groupby(["session", "task", "method"], sort=True):
        gaps = group["generalization_gap"].astype(float)
        rows.append(
            {
                "session": str(session),
                "task": task,
                "method": method,
                "method_display": MODEL_DISPLAY_NAMES.get(method, method),
                "mean_train_accuracy": float(group["final_train_accuracy"].astype(float).mean()),
                "mean_test_ba": float(group["test_balanced_accuracy"].astype(float).mean()),
                "mean_generalization_gap": float(gaps.mean()),
                "std_generalization_gap": float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0,
                "fraction_train_accuracy_above_0_95": float((group["final_train_accuracy"].astype(float) > 0.95).mean()),
                "fraction_test_ba_below_0_5": float((group["test_balanced_accuracy"].astype(float) < 0.5).mean()),
            }
        )
    return audit, pd.DataFrame(rows, columns=summary_columns)


def order_sensitivity_oof_summary(order_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session",
        "task",
        "method",
        "seed",
        "order_condition",
        "balanced_accuracy",
        "accuracy",
        "macro_f1",
        "prediction_is_single_class",
        "n_blocks",
        "original_minus_reverse",
        "original_minus_fixed_shuffle",
    ]
    if order_predictions.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for keys, group in order_predictions.groupby(["session", "task", "method", "seed", "order_condition"], sort=True):
        session, task, method, seed, order_condition = keys
        y_true = group["truth"].to_numpy(dtype=np.int64)
        y_pred = group["prediction"].to_numpy(dtype=np.int64)
        metrics = metrics_with_flags(y_true, y_pred)
        rows.append(
            {
                "session": str(session),
                "task": task,
                "method": method,
                "seed": int(seed),
                "order_condition": order_condition,
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "prediction_is_single_class": bool(metrics["prediction_is_single_class"]),
                "n_blocks": int(len(group)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)
    deltas = []
    for keys, group in out.groupby(["session", "task", "method", "seed"], sort=True):
        by_condition = group.set_index("order_condition")["balanced_accuracy"].to_dict()
        deltas.append(
            {
                "session": keys[0],
                "task": keys[1],
                "method": keys[2],
                "seed": int(keys[3]),
                "original_minus_reverse": (
                    float(by_condition["original"] - by_condition["reverse"])
                    if {"original", "reverse"}.issubset(by_condition)
                    else np.nan
                ),
                "original_minus_fixed_shuffle": (
                    float(by_condition["original"] - by_condition["fixed_shuffle"])
                    if {"original", "fixed_shuffle"}.issubset(by_condition)
                    else np.nan
                ),
            }
        )
    out = out.merge(pd.DataFrame(deltas), on=["session", "task", "method", "seed"], how="left")
    return out[columns]


def block_type_accuracy(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "session",
        "task",
        "method",
        "method_display",
        "dot_accuracy",
        "grating_accuracy",
        "static_accuracy",
        "stop_after_grating_accuracy",
        "n_predictions",
    ]
    if predictions.empty or "block_name" not in predictions:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (session, task, method), group in predictions.groupby(["session", "task", "method"], sort=True):
        row: dict[str, Any] = {
            "session": str(session),
            "task": task,
            "method": method,
            "method_display": MODEL_DISPLAY_NAMES.get(method, method),
            "n_predictions": int(len(group)),
        }
        for block_name in ["dot", "grating", "static", "stop_after_grating"]:
            subset = group[group["block_name"].astype(str) == block_name]
            row[f"{block_name}_accuracy"] = (
                float((subset["truth"].astype(int) == subset["pred"].astype(int)).mean())
                if not subset.empty
                else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def completeness_report(
    *,
    task: str,
    sessions: list[str],
    methods: list[str],
    seeds: list[int],
    master: pd.DataFrame,
    fold_summary: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for session in sessions:
        for method in methods:
            expected_seeds = seeds if method not in {"pca_lda_flat4", "cpca_lda_flat4"} else [0]
            for seed in expected_seeds:
                master_subset = master[
                    (master["session"].astype(str) == str(session))
                    & (master["task"] == task)
                    & (master["method"] == method)
                    & (master["seed"].astype(int) == int(seed))
                ]
                fold_subset = fold_summary[
                    (fold_summary["session"].astype(str) == str(session))
                    & (fold_summary["task"] == task)
                    & (fold_summary["method"] == method)
                    & (fold_summary["seed"].astype(int) == int(seed))
                ]
                pred_subset = predictions[
                    (predictions["session"].astype(str) == str(session))
                    & (predictions["task"] == task)
                    & (predictions["method"] == method)
                    & (predictions["seed"].fillna(0).astype(int) == int(seed))
                ] if not predictions.empty else pd.DataFrame()
                rows.append(
                    {
                        "session": str(session),
                        "task": task,
                        "method": method,
                        "seed": int(seed),
                        "master_row_present": bool(len(master_subset) == 1),
                        "n_fold_rows": int(len(fold_subset)),
                        "n_prediction_rows": int(len(pred_subset)),
                        "has_nan_or_inf_metric": bool(
                            master_subset[["accuracy", "balanced_accuracy", "macro_f1"]]
                            .apply(pd.to_numeric, errors="coerce")
                            .isna()
                            .any()
                            .any()
                        )
                        if not master_subset.empty
                        else True,
                        "prediction_rows_unique_blocks": int(pred_subset["block_id"].nunique())
                        if not pred_subset.empty and "block_id" in pred_subset
                        else 0,
                        "all_test_blocks_predicted_once": bool(
                            not pred_subset.empty
                            and pred_subset.groupby("block_id").size().eq(1).all()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def csv_payload_dict(row: pd.Series, columns: list[str]) -> str:
    return csv_json({column: row[column] for column in columns if column in row})
