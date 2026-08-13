"""Multi-session extension of the frozen SmallCNN masked-SSL benchmark.

The only experimental variable in this module is the set of sessions exposed
to the label-free reconstruction objective.  Architecture, masking,
preprocessing, and downstream training are imported from the audited v1 code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch

from ultrasound_decoding.multiframe.dataset import EXPECTED_SESSIONS
from ultrasound_decoding.multiframe.models import SmallCNNFrameEncoder, encoder_architecture_signature
from ultrasound_decoding.multiframe.training import set_reproducible_seed
from ultrasound_decoding.ssl_masked import (
    MASK_BLOCK_SIZE,
    MASK_RATIO,
    SSL_SEEDS,
    MaskedReconstructionSmallCNN,
    SSLFrameData,
    SSLPretrainingConfig,
    apply_ssl_frame_normalizer,
    deterministic_block_mask,
    load_ssl_encoder_checkpoint,
    masked_pixel_mse,
)


V2_CONDITIONS = (
    "RANDOM_INIT",
    "WITHIN_SSL_FT",
    "OTHER_ONLY_SSL_FT",
    "MULTI_SSL_FT",
)
PRETRAINED_CONDITIONS = V2_CONDITIONS[1:]
NEW_PRETRAINING_CONDITIONS = V2_CONDITIONS[2:]
WEAK_SESSIONS = ("626", "628", "807", "813", "817", "822")
STRONG_SESSIONS = ("708", "709", "710")

# This is the byte-level fingerprint of the three clean v1 implementation files
# at the parent v1 benchmark commit.  v2 refuses v1 reuse if any is changed.
V1_FROZEN_SOURCE_FINGERPRINT = "a462bcfed0d285f46ad9baa371e32ee3333a256b875ccb022f811e708afbadcb"
V1_FROZEN_FILES = (
    "src/ultrasound_decoding/multiframe/models.py",
    "src/ultrasound_decoding/ssl_masked.py",
    "scripts/run_ssl_masked_smallcnn_clean4_9sessions_v1.py",
)

REQUIRED_FORMAL_OUTPUTS = (
    "audit/fold_identity_check.csv",
    "audit/target_test_leakage_audit.csv",
    "audit/pretraining_compute_match.csv",
    "audit/session_sampling_distribution.csv",
    "audit/ssl_pool_audit.csv",
    "audit/config_freeze.md",
    "downstream/fold_metrics.csv",
    "downstream/predictions.csv",
    "summaries/session_level_comparison.csv",
    "summaries/planned_statistical_tests.csv",
    "summaries/generalization_gap_summary.csv",
    "summaries/seed_stability.csv",
    "figures/binary_multisession_ssl_BA.png",
    "figures/stimulus_type_multisession_ssl_BA.png",
    "figures/binary_delta_multi_vs_within.png",
    "figures/stimulus_type_delta_multi_vs_within.png",
    "figures/weak_sessions_multisession_ssl.png",
    "figures/train_test_gap_binary.png",
    "figures/train_test_gap_stimulus_type.png",
    "figures/session_sampling_distribution.png",
    "report/multisession_ssl_report.md",
    "pytest_output_local.txt",
    "run_command_server.txt",
    "run_log_server.txt",
)


@dataclass
class SessionFramePool:
    """Label-free frames partitioned by source session."""

    frames_by_session: dict[str, np.ndarray]
    cycles_by_session: dict[str, np.ndarray]
    original_indices_by_session: dict[str, np.ndarray]
    source_paths: dict[str, Path]

    @property
    def source_sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self.frames_by_session, key=int))

    @property
    def n_frames(self) -> int:
        return int(sum(len(value) for value in self.frames_by_session.values()))


@dataclass
class BalancedPretrainingResult:
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


def frozen_v1_source_fingerprint(project_dir: Path) -> str:
    digest = hashlib.sha256()
    for relative in V1_FROZEN_FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((project_dir / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def architecture_fingerprint() -> str:
    value = json.dumps(encoder_architecture_signature(), separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assert_formal_cuda(device: str) -> torch.device:
    """Formal v2 is GPU-only and can never silently fall back to CPU."""
    if str(device).lower() != "cuda":
        raise RuntimeError("FORMAL STOP: --device must be exactly 'cuda'")
    if not torch.cuda.is_available():
        raise RuntimeError("FORMAL STOP: torch.cuda.is_available() is False; CPU fallback is forbidden")
    return torch.device("cuda")


def reference_optimizer_updates(n_target_train_ssl_frames: int, actual_batch_size: int) -> int:
    if n_target_train_ssl_frames < 1 or actual_batch_size < 1:
        raise ValueError("frame count and batch size must be positive")
    return int(math.ceil(n_target_train_ssl_frames / actual_batch_size) * 50)


def _read_unlabeled_frame_index(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read only validity, original index, and cycle; never open label datasets."""
    with h5py.File(path, "r") as handle:
        valid = handle["/full/valid_mask"][:].astype(bool)
        indices = handle["/full/original_frame_indices"][:].astype(np.int64)
        cycles = handle["/metadata/cycle"][:].astype(np.int64)
    output_cycles: list[int] = []
    output_indices: list[int] = []
    output_rows: list[int] = []
    seen: set[tuple[int, int]] = set()
    for row_i, cycle in enumerate(cycles.tolist()):
        for time_i in np.flatnonzero(valid[row_i]):
            key = (int(cycle), int(indices[row_i, time_i]))
            if key in seen:
                continue
            seen.add(key)
            output_cycles.append(key[0])
            output_indices.append(key[1])
            output_rows.append(int(row_i * valid.shape[1] + time_i))
    order = np.lexsort((np.asarray(output_indices), np.asarray(output_cycles)))
    return (
        np.asarray(output_cycles, dtype=np.int64)[order],
        np.asarray(output_indices, dtype=np.int64)[order],
        np.asarray(output_rows, dtype=np.int64)[order],
    )


def complete_cycles_from_unlabeled_h5(path: Path) -> np.ndarray:
    cycles, _indices, _rows = _read_unlabeled_frame_index(path)
    values, counts = np.unique(cycles, return_counts=True)
    complete = values[counts == 30]
    if len(complete) != len(values):
        bad = dict(zip(values[counts != 30].tolist(), counts[counts != 30].tolist()))
        raise AssertionError(f"incomplete cycle(s) in formal unlabeled source {path}: {bad}")
    return complete.astype(np.int64)


def load_unlabeled_cycles(path: Path, allowed_cycles: Iterable[int]) -> SSLFrameData:
    """Load every unique frame in allowed cycles without touching any labels."""
    allowed = {int(value) for value in allowed_cycles}
    if not allowed:
        raise ValueError("allowed_cycles must be nonempty")
    with h5py.File(path, "r") as handle:
        frames = handle["/full/X_padded"][:]
        valid = handle["/full/valid_mask"][:].astype(bool)
        indices = handle["/full/original_frame_indices"][:].astype(np.int64)
        cycles = handle["/metadata/cycle"][:].astype(np.int64)
    values: list[np.ndarray] = []
    value_cycles: list[int] = []
    value_indices: list[int] = []
    seen: set[tuple[int, int]] = set()
    for row_i, cycle in enumerate(cycles.tolist()):
        if int(cycle) not in allowed:
            continue
        for time_i in np.flatnonzero(valid[row_i]):
            key = (int(cycle), int(indices[row_i, time_i]))
            if key in seen:
                continue
            seen.add(key)
            values.append(frames[row_i, time_i])
            value_cycles.append(key[0])
            value_indices.append(key[1])
    if set(value_cycles) != allowed:
        raise AssertionError("not every requested cycle was present")
    counts = dict(zip(*np.unique(value_cycles, return_counts=True)))
    if any(int(count) != 30 for count in counts.values()):
        raise AssertionError(f"non-complete cycle entered SSL: {counts}")
    order = np.lexsort((np.asarray(value_indices), np.asarray(value_cycles)))
    return SSLFrameData(
        frames=np.stack(values)[order].astype(np.float32, copy=False),
        cycles=np.asarray(value_cycles, dtype=np.int64)[order],
        original_frame_indices=np.asarray(value_indices, dtype=np.int64)[order],
        source_h5_path=path,
    )


def build_ssl_pool(
    all_session_frames: Mapping[str, SSLFrameData],
    *,
    target_session: str,
    target_ssl_train_cycles: Iterable[int],
    target_test_cycles: Iterable[int],
    condition: str,
) -> SessionFramePool:
    """Construct the legal pool for one target/fold and assert zero leakage."""
    target_session = str(target_session)
    if condition not in PRETRAINED_CONDITIONS:
        raise ValueError(f"condition has no SSL pool: {condition}")
    if set(map(str, all_session_frames)) != set(EXPECTED_SESSIONS):
        raise ValueError("formal pool requires exactly the nine frozen sessions")
    target_train = {int(value) for value in target_ssl_train_cycles}
    target_test = {int(value) for value in target_test_cycles}
    if target_train & target_test:
        raise AssertionError("target train and test cycles overlap")
    sources = [target_session] if condition == "WITHIN_SSL_FT" else [
        value for value in EXPECTED_SESSIONS if value != target_session
    ]
    if condition == "MULTI_SSL_FT":
        sources.append(target_session)
    frames_by_session: dict[str, np.ndarray] = {}
    cycles_by_session: dict[str, np.ndarray] = {}
    indices_by_session: dict[str, np.ndarray] = {}
    paths: dict[str, Path] = {}
    for source in sorted(sources, key=int):
        data = all_session_frames[source]
        if source == target_session:
            mask = np.isin(data.cycles, sorted(target_train))
            selected_frames = data.frames[mask]
            selected_cycles = data.cycles[mask]
            selected_indices = data.original_frame_indices[mask]
        else:
            selected_frames = data.frames
            selected_cycles = data.cycles
            selected_indices = data.original_frame_indices
        if source == target_session and np.any(np.isin(selected_cycles, sorted(target_test))):
            raise AssertionError("target test frame entered SSL pool")
        frames_by_session[source] = selected_frames
        cycles_by_session[source] = selected_cycles
        indices_by_session[source] = selected_indices
        paths[source] = data.source_h5_path
    if condition == "OTHER_ONLY_SSL_FT" and target_session in frames_by_session:
        raise AssertionError("OTHER_ONLY contains target frames")
    return SessionFramePool(frames_by_session, cycles_by_session, indices_by_session, paths)


def fit_ssl_pool_normalizer(pool: SessionFramePool) -> tuple[np.ndarray, np.ndarray]:
    """Fit the frozen v1 frame-pooled arcsinh pixel normalizer, streaming."""
    total_sum: np.ndarray | None = None
    total_square_sum: np.ndarray | None = None
    total_frames = 0
    for session in pool.source_sessions:
        frames = pool.frames_by_session[session]
        if not len(frames):
            raise ValueError(f"empty SSL source session {session}")
        transformed = np.arcsinh(frames.astype(np.float32, copy=False)).astype(np.float64, copy=False)
        session_sum = transformed.sum(axis=0)
        session_square_sum = np.square(transformed).sum(axis=0)
        total_sum = session_sum if total_sum is None else total_sum + session_sum
        total_square_sum = session_square_sum if total_square_sum is None else total_square_sum + session_square_sum
        total_frames += len(transformed)
    assert total_sum is not None and total_square_sum is not None and total_frames > 0
    mean = (total_sum / total_frames)[None]
    second = (total_square_sum / total_frames)[None]
    variance = np.maximum(second - np.square(mean), 0.0)
    std = np.sqrt(variance) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


class SessionBalancedSampler:
    """IID source-session-uniform, then source-frame-uniform sampling."""

    def __init__(self, pool: SessionFramePool, *, seed: int) -> None:
        self.pool = pool
        self.rng = np.random.default_rng(int(seed))
        self.sessions = np.asarray(pool.source_sessions, dtype=object)
        if not len(self.sessions):
            raise ValueError("empty session-balanced pool")

    def sample(self, n: int) -> list[tuple[str, int]]:
        source_i = self.rng.integers(0, len(self.sessions), size=int(n))
        output: list[tuple[str, int]] = []
        for value in source_i:
            session = str(self.sessions[int(value)])
            frame_i = int(self.rng.integers(0, len(self.pool.frames_by_session[session])))
            output.append((session, frame_i))
        return output


def sampling_distribution_rows(
    sampling_counts: Mapping[str, int],
    *,
    target_session: str,
    fold: int,
    seed: int,
    condition: str,
) -> list[dict[str, Any]]:
    total = int(sum(sampling_counts.values()))
    n_sources = len(sampling_counts)
    expected = 1.0 / n_sources
    # A preregistered descriptive tolerance, not a rerun/selection criterion.
    tolerance = max(0.02, 5.0 / math.sqrt(max(total, 1)))
    rows = []
    for source, count in sorted(sampling_counts.items(), key=lambda item: int(item[0])):
        actual = count / max(total, 1)
        rows.append({
            "target_session": str(target_session),
            "fold": int(fold),
            "seed": int(seed),
            "condition": condition,
            "source_session": str(source),
            "sample_count": int(count),
            "total_frame_exposures": total,
            "expected_proportion": expected,
            "actual_proportion": actual,
            "absolute_deviation": abs(actual - expected),
            "descriptive_tolerance": tolerance,
            "sampling_status": "PASS" if abs(actual - expected) <= tolerance else "WARN",
        })
    return rows


def pretrain_session_balanced_smallcnn(
    pool: SessionFramePool,
    *,
    seed: int,
    reference_updates: int,
    actual_batch_size: int,
    config: SSLPretrainingConfig = SSLPretrainingConfig(),
    device: str = "cuda",
    normalization_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> BalancedPretrainingResult:
    """Run the frozen masked objective for an exact number of optimizer updates."""
    if reference_updates < 1 or actual_batch_size < 1:
        raise ValueError("positive update count and batch size required")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    set_reproducible_seed(seed)
    mean, std = normalization_stats if normalization_stats is not None else fit_ssl_pool_normalizer(pool)
    model = MaskedReconstructionSmallCNN().to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    sampler = SessionBalancedSampler(pool, seed=seed)
    sampling_counts = {session: 0 for session in pool.source_sessions}
    covered: set[tuple[str, int]] = set()
    history: list[dict[str, Any]] = []
    exposure_i = 0
    started = time.perf_counter()
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    last_qc: dict[str, np.ndarray | int | str] = {}
    model.train()
    for update in range(1, int(reference_updates) + 1):
        sampled = sampler.sample(actual_batch_size)
        raw = np.stack([pool.frames_by_session[session][frame_i] for session, frame_i in sampled])
        target_np = apply_ssl_frame_normalizer(raw, mean, std)
        masks = np.stack([
            deterministic_block_mask(
                target_np.shape[1], target_np.shape[2], seed=seed,
                epoch=update, sample_index=exposure_i + local_i,
                block_size=MASK_BLOCK_SIZE, mask_ratio=MASK_RATIO,
            )
            for local_i in range(len(sampled))
        ])
        masked_np = target_np.copy()
        masked_np[masks] = 0.0
        masked = torch.from_numpy(masked_np[:, None]).to(torch_device)
        target = torch.from_numpy(target_np[:, None]).to(torch_device)
        mask = torch.from_numpy(masks[:, None]).to(torch_device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(masked)
        loss = masked_pixel_mse(prediction, target, mask)
        loss.backward()
        optimizer.step()
        for session, frame_i in sampled:
            sampling_counts[session] += 1
            covered.add((session, frame_i))
        exposure_i += len(sampled)
        history.append({
            "update": update,
            "train_reconstruction_loss": float(loss.detach().cpu()),
            "actual_batch_size": int(actual_batch_size),
            "frame_exposure_count": int(exposure_i),
        })
        if update == reference_updates:
            last_qc = {
                "source_session": sampled[0][0],
                "source_frame_index": int(sampled[0][1]),
                "original": target_np[0],
                "masked": masked_np[0],
                "mask": masks[0],
                "reconstruction": prediction[0, 0].detach().cpu().numpy(),
            }
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
        peak_mb = float(torch.cuda.max_memory_allocated(torch_device) / (1024 ** 2))
    else:
        peak_mb = 0.0
    runtime = time.perf_counter() - started
    return BalancedPretrainingResult(
        encoder=model.encoder.cpu(),
        history=history,
        normalization_mean=mean,
        normalization_std=std,
        actual_batch_size=int(actual_batch_size),
        reference_updates=int(reference_updates),
        actual_updates=len(history),
        frame_exposure_count=int(exposure_i),
        unique_frame_coverage=len(covered),
        sampling_counts=sampling_counts,
        qc=last_qc,
        runtime_seconds=float(runtime),
        peak_gpu_memory_mb=peak_mb,
        device=str(torch_device),
    )


def save_multisession_checkpoint(
    path: Path,
    result: BalancedPretrainingResult,
    *,
    target_session: str,
    fold: int,
    seed: int,
    condition: str,
    pool: SessionFramePool,
    target_ssl_train_cycles: Iterable[int],
    target_test_cycles: Iterable[int],
    config: SSLPretrainingConfig,
    source_fingerprint: str,
) -> dict[str, Any]:
    if condition not in NEW_PRETRAINING_CONDITIONS:
        raise ValueError("only new v2 SSL conditions are saved here")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "encoder_state_dict": {key: value.detach().cpu() for key, value in result.encoder.state_dict().items()},
        "encoder_class": "SmallCNNFrameEncoder",
        "architecture_fingerprint": architecture_fingerprint(),
        "v1_source_fingerprint": source_fingerprint,
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "source_sessions": list(pool.source_sessions),
        "source_cycles_by_session": {
            session: sorted(np.unique(pool.cycles_by_session[session]).astype(int).tolist())
            for session in pool.source_sessions
        },
        "target_ssl_train_cycles": sorted(map(int, target_ssl_train_cycles)),
        "target_test_cycles": sorted(map(int, target_test_cycles)),
        "normalization_mean": result.normalization_mean,
        "normalization_std": result.normalization_std,
        "pretraining_config": asdict(config),
        "ssl_pool_frames": pool.n_frames,
        "actual_batch_size": result.actual_batch_size,
        "reference_updates": result.reference_updates,
        "actual_updates": result.actual_updates,
        "frame_exposure_count": result.frame_exposure_count,
        "unique_frame_coverage": result.unique_frame_coverage,
        "sampling_counts": dict(result.sampling_counts),
        "training_history": list(result.history),
        "runtime_seconds": result.runtime_seconds,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        "decoder_discarded": True,
        "contains_labels": False,
    }
    forbidden = [key for key in payload if "label" in key.lower() and key != "contains_labels"]
    if forbidden:
        raise AssertionError(f"label fields entered SSL checkpoint: {forbidden}")
    torch.save(payload, path)
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "actual_updates": result.actual_updates,
        "contains_labels": False,
    }


def validate_multisession_checkpoint(
    path: Path,
    *,
    target_session: str,
    fold: int,
    seed: int,
    condition: str,
    reference_updates: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    _encoder, payload = load_ssl_encoder_checkpoint(path)
    expected = {
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "reference_updates": int(reference_updates),
        "actual_updates": int(reference_updates),
        "v1_source_fingerprint": source_fingerprint,
        "architecture_fingerprint": architecture_fingerprint(),
        "contains_labels": False,
    }
    mismatches = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if mismatches:
        raise AssertionError(f"incompatible cached v2 checkpoint {path}: {mismatches}")
    return payload


def validate_v1_checkpoint_payload(
    path: Path,
    *,
    target_session: str,
    fold: int,
    seed: int,
    ssl_train_cycles: Iterable[int],
    target_test_cycles: Iterable[int],
    config: SSLPretrainingConfig = SSLPretrainingConfig(),
) -> dict[str, Any]:
    """Strict semantic compatibility gate for reusing a v1 WITHIN checkpoint."""
    _encoder, payload = load_ssl_encoder_checkpoint(path)
    expected = {
        "encoder_class": "SmallCNNFrameEncoder",
        "session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "ssl_train_cycles": sorted(map(int, ssl_train_cycles)),
        "outer_test_cycles": sorted(map(int, target_test_cycles)),
        "pretraining_config": asdict(config),
        "final_epoch": 50,
        "decoder_discarded": True,
        "contains_labels": False,
    }
    comparable = dict(payload)
    comparable["ssl_train_cycles"] = sorted(map(int, payload.get("ssl_train_cycles", [])))
    comparable["outer_test_cycles"] = sorted(map(int, payload.get("outer_test_cycles", [])))
    mismatches = {key: (comparable.get(key), value) for key, value in expected.items() if comparable.get(key) != value}
    if mismatches:
        raise AssertionError(f"v1 WITHIN checkpoint is incompatible: {path}: {mismatches}")
    if tuple(payload["normalization_mean"].shape) != (1, 128, 501):
        raise AssertionError("v1 normalization mean has wrong shape")
    if tuple(payload["normalization_std"].shape) != (1, 128, 501):
        raise AssertionError("v1 normalization std has wrong shape")
    return payload


def compute_match_row(
    *,
    target_session: str,
    fold: int,
    seed: int,
    condition: str,
    ssl_pool_frames: int,
    actual_batch_size: int,
    reference_updates: int,
    actual_updates: int,
    frame_exposure_count: int,
    unique_frame_coverage: int,
    reused_artifact: bool,
) -> dict[str, Any]:
    if actual_updates != reference_updates:
        raise AssertionError("equal-update invariant failed")
    return {
        "target_session": str(target_session),
        "fold": int(fold),
        "seed": int(seed),
        "condition": condition,
        "ssl_pool_frames": int(ssl_pool_frames),
        "actual_batch_size": int(actual_batch_size),
        "reference_updates": int(reference_updates),
        "actual_updates": int(actual_updates),
        "frame_exposure_count": int(frame_exposure_count),
        "unique_frame_coverage": int(unique_frame_coverage),
        "reused_artifact": bool(reused_artifact),
        "compute_match": True,
    }


def checkpoint_contains_no_label_information(path: Path) -> bool:
    _encoder, payload = load_ssl_encoder_checkpoint(path)
    keys = json.dumps(sorted(payload)).lower()
    return payload.get("contains_labels") is False and '"y"' not in keys and "target_label" not in keys


def missing_formal_outputs(output_dir: Path) -> list[str]:
    missing = [relative for relative in REQUIRED_FORMAL_OUTPUTS if not (output_dir / relative).exists()]
    if not any((output_dir / "pretraining/checkpoints").glob("**/*.pt")):
        missing.append("pretraining/checkpoints/**/*.pt")
    if not any((output_dir / "pretraining/losses").glob("*.csv")):
        missing.append("pretraining/losses/*.csv")
    if not any((output_dir / "downstream/training_curves").glob("*.csv")):
        missing.append("downstream/training_curves/*.csv")
    return missing
