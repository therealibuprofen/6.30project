from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ultrasound_decoding.cv import grouped_cv_splits
from ultrasound_decoding.datasets import load_monkey_session
from ultrasound_decoding.deep import build_torch_model, resolve_torch_config


ALLOWED_SESSIONS = {"708", "709", "710"}
IMAGE_SHAPE = (128, 501)
CLASS_ORDER = np.asarray(["no_stimulus", "stimulus"])


@dataclass(frozen=True)
class PatchSpec:
    patch_id: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    @property
    def center_row(self) -> float:
        return (self.row_start + self.row_end - 1) / 2.0

    @property
    def center_col(self) -> float:
        return (self.col_start + self.col_end - 1) / 2.0

    @property
    def patch_height(self) -> int:
        return self.row_end - self.row_start

    @property
    def patch_width(self) -> int:
        return self.col_end - self.col_start

    def to_row(self) -> dict[str, object]:
        return {
            "patch_id": self.patch_id,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "center_row": self.center_row,
            "center_col": self.center_col,
            "patch_height": self.patch_height,
            "patch_width": self.patch_width,
        }


@dataclass
class SessionData:
    session: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    meta: pd.DataFrame
    splits: list[tuple[np.ndarray, np.ndarray]]
    split_manifest: pd.DataFrame


def validate_request(task: str, sessions: Iterable[str]) -> list[str]:
    sessions = [str(session) for session in sessions]
    if task != "binary":
        raise ValueError("Spatial interpretability is restricted to task='binary'")
    illegal = sorted(set(sessions) - ALLOWED_SESSIONS)
    if illegal:
        raise ValueError(f"Spatial interpretability is restricted to sessions 708/709/710, got {illegal}")
    return sessions


def edge_covering_starts(length: int, patch: int, stride: int) -> list[int]:
    if patch < 1 or stride < 1:
        raise ValueError("patch and stride must be positive")
    if patch > length:
        raise ValueError(f"patch size {patch} exceeds image dimension {length}")
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_patch_specs(
    image_shape: tuple[int, int] = IMAGE_SHAPE,
    patch_height: int = 32,
    patch_width: int = 64,
    stride_height: int = 16,
    stride_width: int = 32,
) -> list[PatchSpec]:
    h, w = image_shape
    row_starts = edge_covering_starts(h, patch_height, stride_height)
    col_starts = edge_covering_starts(w, patch_width, stride_width)
    patches: list[PatchSpec] = []
    patch_id = 0
    for row_start in row_starts:
        for col_start in col_starts:
            patches.append(
                PatchSpec(
                    patch_id=patch_id,
                    row_start=row_start,
                    row_end=row_start + patch_height,
                    col_start=col_start,
                    col_end=col_start + patch_width,
                )
            )
            patch_id += 1
    coverage = np.zeros(image_shape, dtype=np.int32)
    for patch in patches:
        if patch.row_end > h or patch.col_end > w:
            raise AssertionError("searchlight patch exceeds image bounds")
        coverage[patch.row_start : patch.row_end, patch.col_start : patch.col_end] += 1
    if int((coverage == 0).sum()) != 0:
        raise AssertionError("searchlight patches do not cover all image pixels")
    return patches


def patch_dataframe(patches: list[PatchSpec]) -> pd.DataFrame:
    return pd.DataFrame([patch.to_row() for patch in patches])


def load_session_for_interpretability(project_dir: Path, session: str, task: str = "binary") -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    X, y, groups, meta = load_monkey_session(
        project_dir,
        session=session,
        task=task,
        clean_middle=True,
        clean_margin_s=8.0,
        analysis_limit=None,
        trim_incomplete_cycles=True,
        window_size=1,
        window_mode="sliding",
        spatial_filter={"method": "none", "radius": 0, "mode": "reflect"},
    )
    if tuple(X.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"Expected fUS image shape {IMAGE_SHAPE}, got {X.shape[1:]}")
    if set(np.unique(y).tolist()) != set(CLASS_ORDER.tolist()):
        raise ValueError(f"Expected binary class labels {CLASS_ORDER.tolist()}, got {np.unique(y).tolist()}")
    return X, y, groups, meta


def split_manifest_from_splits(
    session: str,
    splits: list[tuple[np.ndarray, np.ndarray]],
    groups: np.ndarray,
) -> pd.DataFrame:
    rows = []
    seen_test_samples: set[int] = set()
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        train_cycles = set(np.unique(groups[train_idx]).astype(int).tolist())
        test_cycles = set(np.unique(groups[test_idx]).astype(int).tolist())
        if train_cycles & test_cycles:
            raise AssertionError(f"cycle leakage in fold {fold}: {sorted(train_cycles & test_cycles)}")
        overlap = seen_test_samples & set(test_idx.astype(int).tolist())
        if overlap:
            raise AssertionError(f"test sample appears in multiple folds: {sorted(overlap)[:5]}")
        seen_test_samples.update(test_idx.astype(int).tolist())
        rows.append(
            {
                "session": session,
                "fold": fold,
                "train_cycles": ",".join(str(x) for x in sorted(train_cycles)),
                "test_cycles": ",".join(str(x) for x in sorted(test_cycles)),
                "n_train_samples": int(len(train_idx)),
                "n_test_samples": int(len(test_idx)),
            }
        )
    if len(seen_test_samples) != len(groups):
        raise AssertionError("not every sample appears in exactly one test fold")
    return pd.DataFrame(rows)


def splits_from_saved_fold_metrics(
    benchmark_root: Path,
    session: str,
    task: str,
    groups: np.ndarray,
    max_folds: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_path = benchmark_root / "benchmark" / "fold_metrics_all.csv"
    if not fold_path.exists():
        return grouped_cv_splits(groups, max_folds=max_folds)
    fold_df = pd.read_csv(fold_path)
    rows = fold_df[
        (fold_df["session"].astype(str) == str(session))
        & (fold_df["task"].astype(str) == task)
        & (fold_df["method"].astype(str) == "pca_lda")
    ].sort_values("fold")
    if rows.empty:
        return grouped_cv_splits(groups, max_folds=max_folds)
    splits = []
    for row in rows.itertuples(index=False):
        cycles = [int(value) for value in str(row.test_cycles).split(",") if str(value).strip()]
        test_mask = np.isin(groups, cycles)
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)
        if len(test_idx) != int(row.n_test) or len(train_idx) != int(row.n_train):
            raise AssertionError(
                f"Saved fold manifest mismatch for session {session} fold {row.fold}: "
                f"got train/test {len(train_idx)}/{len(test_idx)}, expected {row.n_train}/{row.n_test}"
            )
        splits.append((train_idx, test_idx))
    regenerated = grouped_cv_splits(groups, max_folds=max_folds)
    if len(regenerated) != len(splits):
        raise AssertionError("saved fold count differs from regenerated grouped CV")
    return splits


def load_session_data(
    project_dir: Path,
    benchmark_root: Path,
    session: str,
    task: str = "binary",
    max_folds: int = 10,
) -> SessionData:
    X, y, groups, meta = load_session_for_interpretability(project_dir, session, task)
    splits = splits_from_saved_fold_metrics(benchmark_root, session, task, groups, max_folds=max_folds)
    manifest = split_manifest_from_splits(session, splits, groups)
    return SessionData(session=session, X=X, y=y, groups=groups, meta=meta, splits=splits, split_manifest=manifest)


def mean_arcsinh_background(X: np.ndarray) -> np.ndarray:
    background = np.arcsinh(X.astype(np.float32, copy=False)).mean(axis=0)
    if background.shape != IMAGE_SHAPE:
        raise ValueError(f"Expected background shape {IMAGE_SHAPE}, got {background.shape}")
    if not np.isfinite(background).all():
        raise ValueError("background contains NaN or Inf")
    return background


def aggregate_patch_values(
    patches: list[PatchSpec],
    values_by_patch: np.ndarray,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    values_by_patch = np.asarray(values_by_patch, dtype=np.float64)
    if len(values_by_patch) != len(patches):
        raise ValueError("patch values length does not match patch definitions")
    total = np.zeros(image_shape, dtype=np.float64)
    count = np.zeros(image_shape, dtype=np.int32)
    for patch, value in zip(patches, values_by_patch):
        if not np.isfinite(value):
            continue
        total[patch.row_start : patch.row_end, patch.col_start : patch.col_end] += float(value)
        count[patch.row_start : patch.row_end, patch.col_start : patch.col_end] += 1
    out = np.full(image_shape, np.nan, dtype=np.float64)
    np.divide(total, count, out=out, where=count > 0)
    return out, count


def aggregate_patch_replicates(
    patches: list[PatchSpec],
    rows: pd.DataFrame,
    metric_column: str,
    image_shape: tuple[int, int] = IMAGE_SHAPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacks = [[] for _ in range(image_shape[0] * image_shape[1])]
    for patch in patches:
        vals = rows.loc[rows["patch_id"] == patch.patch_id, metric_column].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        flat_indices = np.ravel_multi_index(
            np.mgrid[patch.row_start : patch.row_end, patch.col_start : patch.col_end].reshape(2, -1),
            image_shape,
        )
        for value in vals:
            for flat_i in flat_indices:
                stacks[int(flat_i)].append(float(value))
    mean = np.full(image_shape[0] * image_shape[1], np.nan, dtype=np.float64)
    std = np.full_like(mean, np.nan)
    count = np.zeros_like(mean, dtype=np.int32)
    for i, values in enumerate(stacks):
        if values:
            arr = np.asarray(values, dtype=np.float64)
            mean[i] = float(arr.mean())
            std[i] = float(arr.std(ddof=0))
            count[i] = len(arr)
    return mean.reshape(image_shape), std.reshape(image_shape), count.reshape(image_shape)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_model_name(name: str) -> str:
    if name == "fcnn_berthon2023":
        return "fcnn"
    return name


def display_model_name(name: str) -> str:
    if name == "fcnn":
        return "fcnn_berthon2023"
    return name


def load_torch_checkpoint_model(
    checkpoint_path: Path,
    model_name: str,
    n_classes: int,
    input_shape: tuple[int, int] = IMAGE_SHAPE,
    device: str = "cpu",
):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actual_model = resolve_model_name(model_name)
    if checkpoint.get("method") != actual_model:
        raise ValueError(f"checkpoint {checkpoint_path} is for {checkpoint.get('method')}, not {actual_model}")
    config_dict = checkpoint.get("config", {})
    if int(config_dict.get("max_epochs", -1)) != 40:
        raise ValueError(f"checkpoint {checkpoint_path} is not from the fixed max_epochs=40 benchmark")
    if config_dict.get("optimizer") != "adamw":
        raise ValueError(f"checkpoint {checkpoint_path} optimizer is not the formal benchmark optimizer")
    if float(config_dict.get("lr", np.nan)) != 1e-3:
        raise ValueError(f"checkpoint {checkpoint_path} learning rate is not the formal benchmark value")
    if float(config_dict.get("weight_decay", np.nan)) != 1e-3:
        raise ValueError(f"checkpoint {checkpoint_path} weight decay is not the formal benchmark value")
    config = resolve_torch_config(
        actual_model,
        max_epochs=int(config_dict.get("max_epochs", 40)),
        batch_size=int(config_dict.get("batch_size", 16)),
        lr=float(config_dict.get("lr", 1e-3)),
        weight_decay=float(config_dict.get("weight_decay", 1e-3)),
        patience=config_dict.get("patience"),
        activation=config_dict.get("activation"),
        normalization=config_dict.get("normalization"),
        dropout=float(config_dict.get("dropout", 0.0)),
        optimizer=config_dict.get("optimizer"),
    )
    model = build_torch_model(actual_model, n_classes, input_shape, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint, config


def checkpoint_path_for(
    benchmark_root: Path,
    session: str,
    task: str,
    model_name: str,
    seed: int,
    fold: int,
) -> Path:
    actual_model = resolve_model_name(model_name)
    path = benchmark_root / "models" / f"{session}_{task}_{actual_model}_seed{seed}_fold{fold}_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Compatible checkpoint not found: {path}")
    return path
