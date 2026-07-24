#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_DIR / "reports" / "decoding" / "metrics"
OUT_DIR = PROJECT_DIR / "reports" / "summary" / "figures"
SUMMARY_DIR = PROJECT_DIR / "reports" / "summary"
MPLCONFIG_DIR = PROJECT_DIR / ".cache" / "matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHOD_ORDER = [
    "pca_lda",
    "cpca_lda",
    "cnn",
    "cnn_lstm",
    "fcnn",
    "fcnn_paper_32",
    "fus_lite_cnn",
]
METHOD_LABELS = {
    "pca_lda": "PCA+LDA",
    "cpca_lda": "cPCA+LDA",
    "cnn": "CNN",
    "cnn_lstm": "CNN+LSTM",
    "fcnn": "fCNN",
    "fcnn_paper_32": "fCNN paper 32",
    "fus_lite_cnn": "fUS Lite CNN",
}
COLORS = {
    "pca_lda": "#4C78A8",
    "cpca_lda": "#F58518",
    "cnn": "#54A24B",
    "cnn_lstm": "#B279A2",
    "fcnn": "#7F7F7F",
    "fcnn_paper_32": "#E45756",
    "fus_lite_cnn": "#72B7B2",
}


def load_task_metrics(task: str) -> pd.DataFrame:
    rows = []
    for path in sorted(METRICS_DIR.glob(f"*_{task}_overall_metrics.csv")):
        session = path.name.split("_", 1)[0]
        df = pd.read_csv(path)
        df.insert(0, "session", session)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No {task} overall metrics found in {METRICS_DIR}")
    out = pd.concat(rows, ignore_index=True)
    out["session"] = out["session"].astype(str)
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values(["session", "method"]).reset_index(drop=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, output_name: str, chance_line: bool) -> None:
    sessions = sorted(df["session"].unique(), key=lambda value: int(value) if value.isdigit() else value)
    methods = [method for method in METHOD_ORDER if method in set(df["method"].astype(str))]

    fig_width = max(8.5, 1.1 * len(sessions) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2), dpi=160)

    x = np.arange(len(sessions), dtype=float)
    width = min(0.18, 0.78 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for method, offset in zip(methods, offsets):
        pivot = (
            df[df["method"].astype(str) == method]
            .set_index("session")
            .reindex(sessions)
        )
        values = pivot[metric].to_numpy(dtype=float)
        ax.bar(
            x + offset,
            values,
            width=width * 0.92,
            label=METHOD_LABELS.get(method, method),
            color=COLORS.get(method),
            edgecolor="white",
            linewidth=0.7,
        )

    if chance_line:
        ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="Chance = 0.5")

    ax.set_xticks(x)
    ax.set_xticklabels(sessions)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=min(len(methods) + int(chance_line), 5), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(OUT_DIR / output_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for task in ["stimulus_type", "binary"]:
        df = load_task_metrics(task)
        df.to_csv(SUMMARY_DIR / f"{task}_metrics_by_session.csv", index=False)
        plot_metric(
            df,
            metric="balanced_accuracy",
            ylabel="Balanced Accuracy",
            output_name=f"{task}_balanced_accuracy_by_session.png",
            chance_line=True,
        )
        plot_metric(
            df,
            metric="macro_f1",
            ylabel="Macro-F1",
            output_name=f"{task}_macro_f1_by_session.png",
            chance_line=False,
        )
    print(f"Saved figures to {OUT_DIR}")
    print(f"Saved summary CSVs to {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
