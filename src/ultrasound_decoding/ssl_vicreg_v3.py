"""VICReg-style invariance SSL with the frozen fUS SmallCNN backbone.

Only the SSL objective and its conservative intensity/noise augmentation are
new. Data loading, normalization, folds, session-balanced sampling, and the
downstream architecture are inherited from the audited v1/v2 benchmarks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ultrasound_decoding.multiframe.models import SmallCNNFrameEncoder
from ultrasound_decoding.multiframe.training import set_reproducible_seed
from ultrasound_decoding.ssl_masked import SSLFrameData, apply_ssl_frame_normalizer
from ultrasound_decoding.ssl_multisession_v2 import (
    SessionBalancedSampler,
    SessionFramePool,
    architecture_fingerprint,
    build_ssl_pool,
    fit_ssl_pool_normalizer,
)


V3_CONDITIONS = (
    "RANDOM_INIT",
    "WITHIN_MASKED_SSL_FT",
    "MULTI_MASKED_SSL_FT",
    "WITHIN_VICREG_SSL_FT",
    "MULTI_VICREG_SSL_FT",
)
VICREG_CONDITIONS = V3_CONDITIONS[-2:]
VICREG_SEEDS = (20260812, 20260813, 20260814)
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")

REQUIRED_FORMAL_OUTPUTS = (
    "audit/prior_artifact_reuse.csv",
    "audit/fold_identity_check.csv",
    "audit/vicreg_target_test_leakage.csv",
    "audit/augmentation_config.md",
    "audit/vicreg_sampling_distribution.csv",
    "audit/vicreg_compute_match.csv",
    "audit/config_freeze.md",
    "audit/gpu_audit.txt",
    "pretraining/vicreg_losses.csv",
    "downstream/fold_metrics.csv",
    "downstream/predictions.csv",
    "summaries/session_level_comparison.csv",
    "summaries/planned_statistical_tests.csv",
    "summaries/generalization_gap_summary.csv",
    "summaries/seed_stability.csv",
    "figures/augmentation_qc/vicreg_augmentation_contact_sheet.png",
    "figures/binary_ssl_objective_comparison.png",
    "figures/stimulus_type_ssl_objective_comparison.png",
    "figures/binary_vicreg_delta.png",
    "figures/stimulus_type_vicreg_delta.png",
    "figures/weak_sessions_vicreg.png",
    "figures/train_test_gap_binary.png",
    "figures/train_test_gap_stimulus_type.png",
    "report/vicreg_ssl_report.md",
    "pytest_output_local.txt",
    "smoke_test_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass(frozen=True)
class VICRegConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    projector_hidden_dim: int = 256
    projection_dim: int = 256
    invariance_weight: float = 25.0
    variance_weight: float = 25.0
    covariance_weight: float = 1.0
    variance_epsilon: float = 1e-4


@dataclass(frozen=True)
class VICRegAugmentationConfig:
    gain_probability: float = 0.8
    gain_min: float = 0.90
    gain_max: float = 1.10
    offset_probability: float = 0.8
    offset_min: float = -0.05
    offset_max: float = 0.05
    noise_probability: float = 0.8
    noise_sigma_min: float = 0.0
    noise_sigma_max: float = 0.03
    blur_probability: float = 0.3
    blur_sigma_min: float = 0.1
    blur_sigma_max: float = 0.6


@dataclass
class VICRegPretrainingResult:
    encoder: SmallCNNFrameEncoder
    history: list[dict[str, Any]]
    normalization_mean: np.ndarray
    normalization_std: np.ndarray
    actual_batch_size: int
    reference_updates: int
    actual_updates: int
    frame_exposure_count: int
    unique_frame_coverage: int
    sampling_counts: dict[str, int]
    qc: dict[str, np.ndarray | int | str]
    runtime_seconds: float
    peak_gpu_memory_mb: float
    device: str


class VICRegProjector(nn.Module):
    """Fixed SSL-only 512 -> 256 -> 256 projector."""

    def __init__(self, feature_dim: int = SmallCNNFrameEncoder.feature_dim) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(int(feature_dim), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class VICRegSmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = SmallCNNFrameEncoder()
        self.projector = VICRegProjector()

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(frames))


def _uniform(
    shape: tuple[int, ...],
    low: float,
    high: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def _apply_variable_gaussian_blur(
    frames: torch.Tensor,
    active: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Separable per-sample blur; inactive samples receive an identity kernel."""
    batch = int(frames.shape[0])
    coordinates = torch.arange(-2, 3, dtype=frames.dtype, device=frames.device)[None]
    safe_sigma = torch.clamp(sigma.reshape(batch, 1), min=0.1)
    kernels = torch.exp(-0.5 * torch.square(coordinates / safe_sigma))
    kernels = kernels / kernels.sum(dim=1, keepdim=True)
    identity = torch.zeros_like(kernels)
    identity[:, 2] = 1.0
    kernels = torch.where(active.reshape(batch, 1), kernels, identity)
    grouped = frames.transpose(0, 1)
    horizontal = kernels[:, None, None, :]
    grouped = F.conv2d(F.pad(grouped, (2, 2, 0, 0), mode="reflect"), horizontal, groups=batch)
    vertical = kernels[:, None, :, None]
    grouped = F.conv2d(F.pad(grouped, (0, 0, 2, 2), mode="reflect"), vertical, groups=batch)
    return grouped.transpose(0, 1)


def conservative_vicreg_augmentation(
    frames: torch.Tensor,
    *,
    seed: int,
    config: VICRegAugmentationConfig = VICRegAugmentationConfig(),
    return_audit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Apply only geometry-preserving intensity, noise, and mild blur changes."""
    if frames.ndim != 4 or int(frames.shape[1]) != 1:
        raise ValueError(f"expected [B,1,H,W], got {tuple(frames.shape)}")
    device = frames.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    batch = int(frames.shape[0])
    parameter_shape = (batch, 1, 1, 1)
    output = frames.clone()

    gain_active = torch.rand(parameter_shape, generator=generator, device=device) < config.gain_probability
    gain = _uniform(parameter_shape, config.gain_min, config.gain_max, generator=generator, device=device)
    output = output * torch.where(gain_active, gain, torch.ones_like(gain))

    offset_active = torch.rand(parameter_shape, generator=generator, device=device) < config.offset_probability
    offset = _uniform(parameter_shape, config.offset_min, config.offset_max, generator=generator, device=device)
    output = output + torch.where(offset_active, offset, torch.zeros_like(offset))

    noise_active = torch.rand(parameter_shape, generator=generator, device=device) < config.noise_probability
    noise_sigma = _uniform(
        parameter_shape, config.noise_sigma_min, config.noise_sigma_max,
        generator=generator, device=device,
    )
    noise = torch.randn(output.shape, generator=generator, device=device, dtype=output.dtype)
    output = output + noise * torch.where(noise_active, noise_sigma, torch.zeros_like(noise_sigma))

    blur_active = (
        torch.rand((batch,), generator=generator, device=device) < config.blur_probability
    )
    blur_sigma = _uniform(
        (batch,), config.blur_sigma_min, config.blur_sigma_max,
        generator=generator, device=device,
    )
    output = _apply_variable_gaussian_blur(output, blur_active, blur_sigma)
    if not torch.isfinite(output).all():
        raise ValueError("augmentation produced non-finite values")
    if not return_audit:
        return output
    audit = {
        "seed": int(seed),
        "gain_applied": int(gain_active.sum().detach().cpu()),
        "gain_sample_min": float(gain.min().detach().cpu()),
        "gain_sample_max": float(gain.max().detach().cpu()),
        "offset_applied": int(offset_active.sum().detach().cpu()),
        "offset_sample_min": float(offset.min().detach().cpu()),
        "offset_sample_max": float(offset.max().detach().cpu()),
        "noise_applied": int(noise_active.sum().detach().cpu()),
        "noise_sigma_sample_min": float(noise_sigma.min().detach().cpu()),
        "noise_sigma_sample_max": float(noise_sigma.max().detach().cpu()),
        "blur_applied": int(blur_active.sum().detach().cpu()),
        "blur_sigma_sample_min": float(blur_sigma.min().detach().cpu()),
        "blur_sigma_sample_max": float(blur_sigma.max().detach().cpu()),
        "spatial_transform_applied": False,
    }
    return output, audit


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("square matrix required")
    n = int(matrix.shape[0])
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss_components(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    variance_epsilon: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("two equal [batch,dimension] projections required")
    if int(z1.shape[0]) < 2:
        raise ValueError("VICReg covariance requires batch size >= 2")
    invariance = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0, unbiased=True) + variance_epsilon)
    std2 = torch.sqrt(z2.var(dim=0, unbiased=True) + variance_epsilon)
    variance = 0.5 * (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean())
    centered1 = z1 - z1.mean(dim=0)
    centered2 = z2 - z2.mean(dim=0)
    denominator = int(z1.shape[0]) - 1
    covariance1 = centered1.T @ centered1 / denominator
    covariance2 = centered2.T @ centered2 / denominator
    dimension = int(z1.shape[1])
    covariance = (
        off_diagonal(covariance1).square().sum() / dimension
        + off_diagonal(covariance2).square().sum() / dimension
    )
    return invariance, variance, covariance


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    config: VICRegConfig = VICRegConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    invariance, variance, covariance = vicreg_loss_components(
        z1, z2, variance_epsilon=config.variance_epsilon
    )
    total = (
        config.invariance_weight * invariance
        + config.variance_weight * variance
        + config.covariance_weight * covariance
    )
    return total, {"invariance": invariance, "variance": variance, "covariance": covariance}


def build_vicreg_pool(
    all_session_frames: Mapping[str, SSLFrameData],
    *,
    target_session: str,
    target_train_cycles: Iterable[int],
    target_test_cycles: Iterable[int],
    condition: str,
) -> SessionFramePool:
    if condition not in VICREG_CONDITIONS:
        raise ValueError(f"unknown VICReg condition: {condition}")
    mapped = "WITHIN_SSL_FT" if condition == "WITHIN_VICREG_SSL_FT" else "MULTI_SSL_FT"
    return build_ssl_pool(
        all_session_frames,
        target_session=target_session,
        target_ssl_train_cycles=target_train_cycles,
        target_test_cycles=target_test_cycles,
        condition=mapped,
    )


def _pretrain_vicreg_once(
    pool: SessionFramePool,
    *,
    seed: int,
    reference_updates: int,
    batch_size: int,
    config: VICRegConfig,
    augmentation_config: VICRegAugmentationConfig,
    device: torch.device,
    normalization_stats: tuple[np.ndarray, np.ndarray],
) -> VICRegPretrainingResult:
    if batch_size < 8:
        raise RuntimeError("FORMAL STOP: VICReg actual batch size must be >= 8")
    set_reproducible_seed(seed)
    mean, std = normalization_stats
    model = VICRegSmallCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sampler = SessionBalancedSampler(pool, seed=seed)
    sampling_counts = {session: 0 for session in pool.source_sessions}
    covered: set[tuple[str, int]] = set()
    history: list[dict[str, Any]] = []
    exposure_count = 0
    qc: dict[str, np.ndarray | int | str] = {}
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for update in range(1, int(reference_updates) + 1):
        sampled = sampler.sample(batch_size)
        raw = np.stack([pool.frames_by_session[session][frame_i] for session, frame_i in sampled])
        normalized = apply_ssl_frame_normalizer(raw, mean, std)
        original = torch.from_numpy(normalized[:, None]).to(device)
        view_seed = int(seed) * 1_000_003 + update * 2
        view1 = conservative_vicreg_augmentation(original, seed=view_seed, config=augmentation_config)
        view2 = conservative_vicreg_augmentation(original, seed=view_seed + 1, config=augmentation_config)
        optimizer.zero_grad(set_to_none=True)
        z1 = model(view1)
        z2 = model(view2)
        loss, components = vicreg_loss(z1, z2, config=config)
        loss.backward()
        optimizer.step()
        for session, frame_i in sampled:
            sampling_counts[session] += 1
            covered.add((session, frame_i))
        exposure_count += len(sampled)
        history.append({
            "update": int(update),
            "total_loss": float(loss.detach().cpu()),
            "invariance_loss": float(components["invariance"].detach().cpu()),
            "variance_loss": float(components["variance"].detach().cpu()),
            "covariance_loss": float(components["covariance"].detach().cpu()),
            "actual_batch_size": int(batch_size),
            "frame_exposure_count": int(exposure_count),
        })
        if update == reference_updates:
            qc = {
                "source_session": sampled[0][0],
                "source_frame_index": int(sampled[0][1]),
                "original": original[0, 0].detach().cpu().numpy(),
                "view1": view1[0, 0].detach().cpu().numpy(),
                "view2": view2[0, 0].detach().cpu().numpy(),
            }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    else:
        peak_mb = 0.0
    return VICRegPretrainingResult(
        encoder=model.encoder.cpu(),
        history=history,
        normalization_mean=mean,
        normalization_std=std,
        actual_batch_size=int(batch_size),
        reference_updates=int(reference_updates),
        actual_updates=len(history),
        frame_exposure_count=int(exposure_count),
        unique_frame_coverage=len(covered),
        sampling_counts=sampling_counts,
        qc=qc,
        runtime_seconds=float(time.perf_counter() - started),
        peak_gpu_memory_mb=peak_mb,
        device=str(device),
    )


def pretrain_vicreg_smallcnn(
    pool: SessionFramePool,
    *,
    seed: int,
    reference_updates: int,
    config: VICRegConfig = VICRegConfig(),
    augmentation_config: VICRegAugmentationConfig = VICRegAugmentationConfig(),
    device: str = "cuda",
    normalization_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> VICRegPretrainingResult:
    """Train with fixed updates; only 32 -> 16 -> 8 OOM fallback is allowed."""
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("FORMAL STOP: CUDA requested but unavailable; CPU fallback is forbidden")
    if reference_updates < 1:
        raise ValueError("reference_updates must be positive")
    stats = normalization_stats or fit_ssl_pool_normalizer(pool)
    candidates = [value for value in (32, 16, 8) if value <= int(config.batch_size)]
    if int(config.batch_size) not in candidates and int(config.batch_size) >= 8:
        candidates.insert(0, int(config.batch_size))
    if not candidates:
        raise RuntimeError("FORMAL STOP: VICReg batch size below 8 is forbidden")
    last_oom: RuntimeError | None = None
    for batch_size in candidates:
        try:
            return _pretrain_vicreg_once(
                pool,
                seed=seed,
                reference_updates=reference_updates,
                batch_size=batch_size,
                config=config,
                augmentation_config=augmentation_config,
                device=torch_device,
                normalization_stats=stats,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            last_oom = exc
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
    if last_oom is not None:
        raise RuntimeError("FORMAL STOP: VICReg OOM even at minimum batch size 8") from last_oom
    raise AssertionError("unreachable VICReg batch fallback state")


def save_vicreg_encoder_checkpoint(
    path: Path,
    result: VICRegPretrainingResult,
    *,
    target_session: str,
    fold: int,
    seed: int,
    condition: str,
    pool: SessionFramePool,
    target_train_cycles: Iterable[int],
    target_test_cycles: Iterable[int],
    config: VICRegConfig,
    augmentation_config: VICRegAugmentationConfig,
    implementation_fingerprint: str,
) -> dict[str, Any]:
    if condition not in VICREG_CONDITIONS:
        raise ValueError("invalid VICReg checkpoint condition")
    if result.actual_batch_size < 8 or result.actual_updates != result.reference_updates:
        raise AssertionError("invalid VICReg compute result")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state_dict": {key: value.detach().cpu() for key, value in result.encoder.state_dict().items()},
        "encoder_class": "SmallCNNFrameEncoder",
        "architecture_fingerprint": architecture_fingerprint(),
        "implementation_fingerprint": implementation_fingerprint,
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "source_sessions": list(pool.source_sessions),
        "source_cycles_by_session": {
            session: sorted(np.unique(pool.cycles_by_session[session]).astype(int).tolist())
            for session in pool.source_sessions
        },
        "target_train_cycles": sorted(map(int, target_train_cycles)),
        "target_test_cycles": sorted(map(int, target_test_cycles)),
        "normalization_mean": result.normalization_mean,
        "normalization_std": result.normalization_std,
        "vicreg_config": asdict(config),
        "augmentation_config": asdict(augmentation_config),
        "ssl_pool_frames": pool.n_frames,
        "actual_batch_size": result.actual_batch_size,
        "reference_updates": result.reference_updates,
        "actual_updates": result.actual_updates,
        "frame_exposure_count": result.frame_exposure_count,
        "unique_frame_coverage": result.unique_frame_coverage,
        "sampling_counts": dict(result.sampling_counts),
        "training_history": list(result.history),
        "projector_discarded": True,
        "contains_projector_state": False,
        "contains_labels": False,
        "runtime_seconds": result.runtime_seconds,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
    }
    forbidden = [key for key in payload if "label" in key.lower() and key != "contains_labels"]
    if forbidden:
        raise AssertionError(f"label fields entered VICReg checkpoint: {forbidden}")
    torch.save(payload, path)
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
    }


def validate_vicreg_checkpoint(
    path: Path,
    *,
    target_session: str,
    fold: int,
    seed: int,
    condition: str,
    reference_updates: int,
    implementation_fingerprint: str,
    source_sessions: Iterable[str] | None = None,
    target_train_cycles: Iterable[int] | None = None,
    target_test_cycles: Iterable[int] | None = None,
    config: VICRegConfig | None = None,
    augmentation_config: VICRegAugmentationConfig | None = None,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected = {
        "encoder_class": "SmallCNNFrameEncoder",
        "architecture_fingerprint": architecture_fingerprint(),
        "implementation_fingerprint": implementation_fingerprint,
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "reference_updates": int(reference_updates),
        "actual_updates": int(reference_updates),
        "projector_discarded": True,
        "contains_projector_state": False,
        "contains_labels": False,
    }
    if source_sessions is not None:
        expected["source_sessions"] = sorted(map(str, source_sessions), key=int)
    if target_train_cycles is not None:
        expected["target_train_cycles"] = sorted(map(int, target_train_cycles))
    if target_test_cycles is not None:
        expected["target_test_cycles"] = sorted(map(int, target_test_cycles))
    if config is not None:
        expected["vicreg_config"] = asdict(config)
    if augmentation_config is not None:
        expected["augmentation_config"] = asdict(augmentation_config)
    mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise AssertionError(f"incompatible cached VICReg checkpoint {path}: {mismatches}")
    if int(payload["actual_batch_size"]) < 8:
        raise AssertionError("cached VICReg checkpoint used forbidden batch size")
    encoder = SmallCNNFrameEncoder()
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    if any("projector" in key for key in payload["encoder_state_dict"]):
        raise AssertionError("projector leaked into downstream encoder state")
    return payload


def implementation_fingerprint(project_dir: Path) -> str:
    files = (
        "src/ultrasound_decoding/multiframe/models.py",
        "src/ultrasound_decoding/ssl_masked.py",
        "src/ultrasound_decoding/ssl_multisession_v2.py",
        "src/ultrasound_decoding/ssl_vicreg_v3.py",
        "scripts/run_ssl_vicreg_smallcnn_9sessions_v3.py",
    )
    digest = hashlib.sha256()
    for relative in files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((project_dir / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def checkpoint_contains_no_labels_or_projector(path: Path) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    keys = json.dumps(sorted(payload)).lower()
    encoder_keys = list(payload.get("encoder_state_dict", {}))
    return bool(
        payload.get("contains_labels") is False
        and payload.get("projector_discarded") is True
        and payload.get("contains_projector_state") is False
        and '"y"' not in keys
        and "target_label" not in keys
        and not any("projector" in key for key in encoder_keys)
    )


def missing_formal_outputs(output_dir: Path) -> list[str]:
    missing = [relative for relative in REQUIRED_FORMAL_OUTPUTS if not (output_dir / relative).exists()]
    if not any((output_dir / "pretraining/checkpoints").glob("**/*.pt")):
        missing.append("pretraining/checkpoints/**/*.pt")
    if not any((output_dir / "downstream/training_curves").glob("*.csv")):
        missing.append("downstream/training_curves/*.csv")
    return missing
