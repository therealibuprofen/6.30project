from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ultrasound_decoding.evaluate import classification_metrics
from ultrasound_decoding.interpretability.common import PatchSpec, aggregate_patch_values, display_model_name, write_json
from ultrasound_decoding.interpretability.nn_utils import (
    CLASS_ORDER,
    load_fold_model_and_inputs,
    original_metrics_payload,
    predict_logits_probabilities,
    tensor_from_normalized_frames,
)


def run_occlusion_for_fold(
    *,
    project_dir: Path,
    benchmark_root: Path,
    session: str,
    task: str,
    model_name: str,
    seed: int,
    fold: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    patches: list[PatchSpec],
    output_dir: Path,
    device: str = "auto",
    max_patches: int | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_payload = load_fold_model_and_inputs(
        project_dir=project_dir,
        benchmark_root=benchmark_root,
        session=session,
        task=task,
        model_name=model_name,
        seed=seed,
        fold=fold,
        X=X,
        y=y,
        train_idx=train_idx,
        test_idx=test_idx,
        device=device,
    )
    selected_patches = patches[:max_patches] if max_patches is not None else patches
    model = fold_payload["model"]
    test_tensor = fold_payload["test_tensor"]
    y_test = fold_payload["y_test"]
    y_test_i = fold_payload["y_test_i"]
    X_test_norm = fold_payload["X_test_norm"]
    config = fold_payload["config"]
    _, base_probs, base_pred_i = predict_logits_probabilities(model, test_tensor, batch_size=config.batch_size)
    base_pred = CLASS_ORDER[base_pred_i]
    base_metrics = classification_metrics(y_test, base_pred)
    base_true_probs = base_probs[np.arange(len(y_test_i)), y_test_i]
    write_json(
        output_dir / "original_metrics.json",
        original_metrics_payload(
            model_name=model_name,
            seed=seed,
            fold=fold,
            checkpoint_path=fold_payload["checkpoint_path"],
            y_test=y_test,
            pred_i=base_pred_i,
            probabilities=base_probs,
        ),
    )

    rows = []
    for patch in selected_patches:
        X_occ = X_test_norm.copy()
        X_occ[:, patch.row_start : patch.row_end, patch.col_start : patch.col_end] = 0.0
        occ_tensor = tensor_from_normalized_frames(X_occ, fold_payload["device"])
        _, occ_probs, occ_pred_i = predict_logits_probabilities(model, occ_tensor, batch_size=config.batch_size)
        occ_pred = CLASS_ORDER[occ_pred_i]
        occ_metrics = classification_metrics(y_test, occ_pred)
        occ_true_probs = occ_probs[np.arange(len(y_test_i)), y_test_i]
        rows.append(
            {
                "session": session,
                "model": display_model_name(model_name),
                "seed": int(seed),
                "fold": int(fold),
                "patch_id": patch.patch_id,
                "true_class_probability_drop": float(np.mean(base_true_probs - occ_true_probs)),
                "balanced_accuracy_drop": float(base_metrics["balanced_accuracy"] - occ_metrics["balanced_accuracy"]),
                "prediction_flip_rate": float(np.mean(base_pred_i != occ_pred_i)),
                "occluded_balanced_accuracy": occ_metrics["balanced_accuracy"],
                "original_balanced_accuracy": base_metrics["balanced_accuracy"],
                "n_test_samples": int(len(y_test)),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "occlusion_patch_metrics.csv", index=False)
    for metric, filename in [
        ("true_class_probability_drop", "occlusion_probability_drop.npy"),
        ("balanced_accuracy_drop", "occlusion_ba_drop.npy"),
        ("prediction_flip_rate", "occlusion_flip_rate.npy"),
    ]:
        patch_values = metrics.set_index("patch_id").reindex([p.patch_id for p in selected_patches])[metric].to_numpy()
        map_arr, _ = aggregate_patch_values(selected_patches, patch_values)
        if np.isinf(map_arr[np.isfinite(map_arr)]).any():
            raise AssertionError(f"{filename} contains Inf")
        np.save(output_dir / filename, map_arr)
    return metrics


def aggregate_occlusion(seed_dirs: list[Path], output_dir: Path) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for stem, source_name in [
        ("occlusion_probability_drop", "occlusion_probability_drop.npy"),
        ("occlusion_ba_drop", "occlusion_ba_drop.npy"),
        ("occlusion_flip_rate", "occlusion_flip_rate.npy"),
    ]:
        maps = []
        for directory in seed_dirs:
            for path in sorted(directory.glob(f"fold*/{source_name}")):
                arr = np.load(path)
                maps.append(arr)
        if not maps:
            continue
        stack = np.stack(maps, axis=0)
        payload[f"{stem}_mean"] = np.nanmean(stack, axis=0)
        payload[f"{stem}_std"] = np.nanstd(stack, axis=0)
        np.save(output_dir / f"{stem}_mean.npy", payload[f"{stem}_mean"])
        np.save(output_dir / f"{stem}_std.npy", payload[f"{stem}_std"])
    return payload

