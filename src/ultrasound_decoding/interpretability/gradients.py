from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ultrasound_decoding.interpretability.common import CLASS_ORDER, display_model_name, write_json
from ultrasound_decoding.interpretability.nn_utils import (
    load_fold_model_and_inputs,
    original_metrics_payload,
    predict_logits_probabilities,
)


def _target_logit_sum(model, inputs, target_i):
    import torch

    logits = model(inputs)
    return logits.gather(1, target_i[:, None]).sum()


def input_gradient(model, inputs, target_i) -> np.ndarray:
    x = inputs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    score = _target_logit_sum(model, x, target_i)
    score.backward()
    grad = x.grad.detach().cpu().numpy()[:, 0]
    return grad


def gradient_x_input(model, inputs, target_i) -> np.ndarray:
    grad = input_gradient(model, inputs, target_i)
    x_np = inputs.detach().cpu().numpy()[:, 0]
    return grad * x_np


def integrated_gradients(model, inputs, target_i, steps: int = 32) -> np.ndarray:
    import torch

    if steps < 1:
        raise ValueError("Integrated Gradients steps must be >= 1")
    baseline = torch.zeros_like(inputs)
    total_grad = torch.zeros_like(inputs)
    for alpha in torch.linspace(1.0 / steps, 1.0, steps, device=inputs.device):
        x = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = _target_logit_sum(model, x, target_i)
        score.backward()
        total_grad += x.grad.detach()
    avg_grad = total_grad / float(steps)
    attr = (inputs - baseline) * avg_grad
    return attr.detach().cpu().numpy()[:, 0]


def aggregate_attributions(attributions: np.ndarray, y_test: np.ndarray) -> dict[str, np.ndarray]:
    if attributions.ndim != 3:
        raise ValueError(f"single-frame attributions must be [N,H,W], got {attributions.shape}")
    if not np.isfinite(attributions).all():
        raise AssertionError("attribution contains NaN or Inf")
    out: dict[str, np.ndarray] = {}
    abs_attr = np.abs(attributions)
    out["absolute_mean"] = abs_attr.mean(axis=0)
    for cls in CLASS_ORDER:
        mask = y_test == cls
        if not np.any(mask):
            out[f"{cls}_signed_mean"] = np.full(attributions.shape[1:], np.nan, dtype=np.float64)
            out[f"{cls}_absolute_mean"] = np.full(attributions.shape[1:], np.nan, dtype=np.float64)
        else:
            out[f"{cls}_signed_mean"] = attributions[mask].mean(axis=0)
            out[f"{cls}_absolute_mean"] = abs_attr[mask].mean(axis=0)
    out["class_difference_signed"] = out["stimulus_signed_mean"] - out["no_stimulus_signed_mean"]
    return out


def save_attribution_maps(output_dir: Path, method: str, maps: dict[str, np.ndarray]) -> None:
    for key, arr in maps.items():
        if np.isinf(arr[np.isfinite(arr)]).any():
            raise AssertionError(f"{method} {key} map contains Inf")
        np.save(output_dir / f"{method}_{key}.npy", arr)


def run_gradients_for_fold(
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
    output_dir: Path,
    interpretation_methods: set[str],
    ig_steps: int = 32,
    device: str = "auto",
    max_test_samples: int | None = None,
) -> pd.DataFrame:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_fold_model_and_inputs(
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
    model = payload["model"]
    tensor = payload["test_tensor"]
    y_test = payload["y_test"]
    y_test_i = payload["y_test_i"]
    if max_test_samples is not None:
        tensor = tensor[:max_test_samples]
        y_test = y_test[:max_test_samples]
        y_test_i = y_test_i[:max_test_samples]
    target_i = torch.from_numpy(y_test_i).to(payload["device"])
    _, probabilities, pred_i = predict_logits_probabilities(model, tensor, batch_size=payload["config"].batch_size)
    write_json(
        output_dir / "gradient_original_metrics.json",
        original_metrics_payload(
            model_name=model_name,
            seed=seed,
            fold=fold,
            checkpoint_path=payload["checkpoint_path"],
            y_test=y_test,
            pred_i=pred_i,
            probabilities=probabilities,
        ),
    )
    rows = []
    calculators = {
        "input_gradient": lambda: input_gradient(model, tensor, target_i),
        "gradient_x_input": lambda: gradient_x_input(model, tensor, target_i),
        "integrated_gradients": lambda: integrated_gradients(model, tensor, target_i, steps=ig_steps),
    }
    for method in ["input_gradient", "gradient_x_input", "integrated_gradients"]:
        if method not in interpretation_methods:
            continue
        attr = calculators[method]()
        if not np.isfinite(attr).all():
            raise AssertionError(f"{method} attribution contains NaN or Inf")
        if method == "integrated_gradients" and not np.allclose(torch.zeros_like(tensor).detach().cpu().numpy(), 0.0):
            raise AssertionError("Integrated Gradients baseline is not zero")
        maps = aggregate_attributions(attr, y_test)
        save_attribution_maps(output_dir, method, maps)
        rows.append(
            {
                "session": session,
                "model": display_model_name(model_name),
                "seed": int(seed),
                "fold": int(fold),
                "method": method,
                "n_test_samples": int(len(y_test)),
                "mean_absolute_attribution": float(np.abs(attr).mean()),
                "signed_positive_fraction": float(np.mean(attr > 0)),
                "signed_negative_fraction": float(np.mean(attr < 0)),
                "all_zero_attribution": bool(np.allclose(attr, 0.0)),
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "gradient_attribution_audit.csv", index=False)
    if "integrated_gradients" in interpretation_methods and audit.loc[audit["method"] == "integrated_gradients", "all_zero_attribution"].any():
        raise AssertionError("Integrated Gradients map is all zero")
    return audit


def aggregate_gradient_maps(seed_dirs: list[Path], output_dir: Path) -> dict[str, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    suffixes = [
        "absolute_mean",
        "stimulus_signed_mean",
        "no_stimulus_signed_mean",
        "stimulus_absolute_mean",
        "no_stimulus_absolute_mean",
        "class_difference_signed",
    ]
    for method in ["input_gradient", "gradient_x_input", "integrated_gradients"]:
        for suffix in suffixes:
            maps = []
            for directory in seed_dirs:
                for path in sorted(directory.glob(f"fold*/{method}_{suffix}.npy")):
                    maps.append(np.load(path))
            if not maps:
                continue
            stack = np.stack(maps, axis=0)
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            payload[f"{method}_{suffix}_mean"] = mean
            payload[f"{method}_{suffix}_std"] = std
            np.save(output_dir / f"{method}_{suffix}_mean.npy", mean)
            np.save(output_dir / f"{method}_{suffix}_std.npy", std)
    return payload

