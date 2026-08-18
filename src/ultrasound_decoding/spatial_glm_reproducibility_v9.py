"""Preregistered blockwise spatial GLM and reproducibility analysis (v9).

The inferential unit is one complete cycle.  Each of its four observations is
the pixelwise mean of the four frozen clean-middle frames belonging to one
condition block.  This module deliberately contains no decoder, registration,
ROI, searchlight, HRF, or model-selection code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from ultrasound_decoding.cross_session_feature_factor_v7 import (
    EXPECTED_SESSIONS,
    STRONG_SESSIONS,
    WEAK_SESSIONS,
    exact_spearman_permutation,
)
from ultrasound_decoding.multiframe.dataset import (
    BLOCK_NAMES,
    EXPECTED_BLOCK_SHAPE,
    BlockSequenceData,
    load_block_sequence_session,
)


RUN_NAME = "spatial_glm_contrast_reproducibility_9sessions_v9"
FORMAL_DEVICE = "cpu"
N_SPLITS = 1000
N_BOOTSTRAP = 1000
FDR_ALPHA = 0.05
V9_RANDOM_SEED = 20260818
IMAGE_SHAPE = EXPECTED_BLOCK_SHAPE[1:]
FIXED_ORIENTATIONS = {session: "identity" for session in EXPECTED_SESSIONS}
FIXED_ORIENTATIONS["807"] = "flip_vertical"
CONDITION_ORDER = tuple(BLOCK_NAMES)
FIXED_ORDER_WARNING = (
    "Condition identity is confounded with fixed within-cycle temporal position: "
    "grating -> stop_after_grating -> dot -> static. Effects are condition-associated, "
    "not pure stimulus-induced activation."
)
GD_WARNING = "EXPLORATORY: TEMPORAL-POSITION CONFOUNDED (fixed within-cycle order)."

CONTRAST_WEIGHTS: dict[str, np.ndarray] = {
    "STIM_PRESENCE": np.asarray([0.5, -0.5, 0.5, -0.5], dtype=np.float64),
    "GS": np.asarray([1.0, -1.0, 0.0, 0.0], dtype=np.float64),
    "DS": np.asarray([0.0, 0.0, 1.0, -1.0], dtype=np.float64),
    "GD": np.asarray([1.0, 0.0, -1.0, 0.0], dtype=np.float64),
}


@dataclass(frozen=True)
class SessionBlockImages:
    session: str
    cycle_ids: np.ndarray
    images: np.ndarray  # cycle x condition x height x width
    background: np.ndarray
    source: BlockSequenceData
    orientation: str

    @property
    def n_cycles(self) -> int:
        return int(len(self.cycle_ids))


@dataclass(frozen=True)
class ContrastMaps:
    effect: np.ndarray
    standard_error: np.ndarray
    t_map: np.ndarray
    p_map: np.ndarray
    q_map: np.ndarray
    fdr_mask: np.ndarray
    standardized: np.ndarray
    cycle_maps: np.ndarray


@dataclass(frozen=True)
class SessionAnalysis:
    session: str
    n_cycles: int
    design_rank: int
    residual_df: int
    background: np.ndarray
    contrasts: Mapping[str, ContrastMaps]
    glm_rows: tuple[dict[str, Any], ...]
    split_rows: tuple[dict[str, Any], ...]
    concordance_row: dict[str, Any]
    bootstrap_row: dict[str, Any]


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def apply_fixed_orientation(array: np.ndarray, orientation: str) -> np.ndarray:
    """Apply a frozen in-plane orientation without creating registration DOF."""
    values = np.asarray(array)
    if values.shape[-2:] != IMAGE_SHAPE:
        raise ValueError(f"expected trailing image shape {IMAGE_SHAPE}, got {values.shape}")
    if orientation == "identity":
        output = values
    elif orientation == "flip_vertical":
        output = np.flip(values, axis=-2)
    else:
        raise ValueError(f"unsupported frozen orientation: {orientation}")
    if output.shape[-2:] != IMAGE_SHAPE:
        raise AssertionError("fixed orientation changed the spatial grid")
    return output


def block_mean_clean4(data: BlockSequenceData, *, orientation: str) -> np.ndarray:
    """Transform each clean4 frame, then reduce it to one block observation."""
    frames = np.asarray(data.X, dtype=np.float32)
    if frames.ndim != 4 or tuple(frames.shape[1:]) != EXPECTED_BLOCK_SHAPE:
        raise AssertionError(f"expected clean4 block arrays, got {frames.shape}")
    # This is the frozen visual-decoding intensity transform.  Averaging is
    # strictly within a block, so the four frames never become four samples.
    transformed = np.arcsinh(frames).mean(axis=1, dtype=np.float64)
    transformed = apply_fixed_orientation(transformed, orientation)
    if not np.isfinite(transformed).all():
        raise AssertionError("block-level images contain non-finite values")
    return transformed


def load_session_block_images(
    project_dir: Path,
    data_dir: Path,
    session: str,
    *,
    max_cycles: int | None = None,
) -> SessionBlockImages:
    session = str(session)
    if session not in EXPECTED_SESSIONS:
        raise ValueError(f"unexpected session: {session}")
    data = load_block_sequence_session(project_dir, session, "binary", data_dir)
    metadata = data.metadata.copy()
    if max_cycles is not None:
        selected_cycles = sorted(metadata["cycle"].astype(int).unique())[: int(max_cycles)]
        keep = metadata["cycle"].astype(int).isin(selected_cycles).to_numpy()
        metadata = metadata.loc[keep].reset_index(drop=True)
        block_images = block_mean_clean4(data, orientation=FIXED_ORIENTATIONS[session])[keep]
    else:
        block_images = block_mean_clean4(data, orientation=FIXED_ORIENTATIONS[session])
    cycle_ids = np.asarray(sorted(metadata["cycle"].astype(int).unique()), dtype=np.int64)
    images = []
    for cycle in cycle_ids:
        rows = metadata[metadata["cycle"].astype(int) == int(cycle)].sort_values("block_order_in_cycle")
        if rows["block_name"].astype(str).tolist() != list(CONDITION_ORDER):
            raise AssertionError(f"{session} cycle {cycle} does not contain the fixed four-condition order")
        indices = rows.index.to_numpy(dtype=int)
        images.append(block_images[indices])
    stacked = np.stack(images).astype(np.float64, copy=False)
    if stacked.shape != (len(cycle_ids), 4, *IMAGE_SHAPE):
        raise AssertionError(f"cycle/condition block image shape is invalid: {stacked.shape}")
    return SessionBlockImages(
        session=session,
        cycle_ids=cycle_ids,
        images=stacked,
        background=stacked.mean(axis=(0, 1)),
        source=data,
        orientation=FIXED_ORIENTATIONS[session],
    )


def clean4_identity_row(data: SessionBlockImages) -> dict[str, Any]:
    source = data.source
    metadata = source.metadata
    ordered = metadata.groupby("cycle", sort=True)["block_name"].apply(list)
    passed = bool(
        tuple(source.X.shape[1:]) == EXPECTED_BLOCK_SHAPE
        and source.n_blocks == 4 * source.n_cycles
        and metadata["complete_cycle"].astype(bool).all()
        and ordered.apply(lambda value: value == list(CONDITION_ORDER)).all()
        and np.all(metadata["n_frames_clean4"].astype(int).to_numpy() == 4)
    )
    return {
        "session": data.session,
        "source_h5": str(source.source_h5_path),
        "source_metadata_csv": str(source.source_metadata_path),
        "n_blocks": source.n_blocks,
        "n_complete_cycles": source.n_cycles,
        "frames_per_block": 4,
        "block_observations_per_cycle": 4,
        "block_image_shape": "128x501",
        "preprocessing": "arcsinh_each_clean4_frame_then_pixelwise_mean_within_block",
        "statistical_unit": "complete_cycle_with_four_block_observations",
        "fixed_orientation": data.orientation,
        "orientation_normalized": True,
        "registration_applied": False,
        "status": "PASS" if passed else "FAIL",
    }


def repeated_measures_design(n_cycles: int) -> tuple[np.ndarray, tuple[str, ...]]:
    """Intercept + n-1 cycle dummies + three condition dummies (static ref)."""
    n_cycles = int(n_cycles)
    if n_cycles < 2:
        raise ValueError("repeated-measures GLM requires at least two complete cycles")
    cycle_index = np.repeat(np.arange(n_cycles), 4)
    condition_index = np.tile(np.arange(4), n_cycles)
    columns = [np.ones(4 * n_cycles, dtype=np.float64)]
    names = ["intercept"]
    for cycle in range(1, n_cycles):
        columns.append((cycle_index == cycle).astype(np.float64))
        names.append(f"cycle_{cycle}")
    for condition_i, name in enumerate(CONDITION_ORDER[:3]):
        columns.append((condition_index == condition_i).astype(np.float64))
        names.append(f"condition_{name}")
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise AssertionError("cycle-fixed condition GLM design is not full rank")
    return design, tuple(names)


def design_contrast(weights: Sequence[float], n_cycles: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (4,) or not np.isclose(weights.sum(), 0.0):
        raise ValueError("condition contrast must contain four zero-sum weights")
    # Static is the treatment-coded reference.  Zero-sum weights remove the
    # intercept and every cycle fixed effect.
    output = np.zeros(int(n_cycles) + 3, dtype=np.float64)
    output[-3:] = weights[:3]
    return output


def cycle_contrast_maps(images: np.ndarray, weights: Sequence[float]) -> np.ndarray:
    values = np.asarray(images, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 4 or values.shape[1:] != (4, *IMAGE_SHAPE):
        raise ValueError(f"expected cycle x four conditions x 128 x 501, got {values.shape}")
    if weights.shape != (4,):
        raise ValueError("contrast weights must have length four")
    output = np.einsum("c,schw->shw", weights, values, optimize=True)
    if not np.isfinite(output).all():
        raise AssertionError("cycle contrast maps contain non-finite values")
    return output


def benjamini_hochberg(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return finite BH-adjusted q-values and the predefined q<=.05 mask."""
    values = np.asarray(p_values, dtype=np.float64)
    valid = np.isfinite(values)
    q_values = np.ones_like(values, dtype=np.float64)
    flat = values[valid]
    if np.any((flat < 0) | (flat > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    if len(flat):
        order = np.argsort(flat, kind="mergesort")
        ranked = flat[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        q_values[valid] = restored
    return q_values, valid & (q_values <= FDR_ALPHA)


def fit_pixelwise_glm(images: np.ndarray) -> tuple[dict[str, ContrastMaps], int, int]:
    values = np.asarray(images, dtype=np.float64)
    n_cycles = int(values.shape[0])
    design, _ = repeated_measures_design(n_cycles)
    response = values.reshape(4 * n_cycles, -1)
    # NumPy 2/OpenBLAS on some macOS builds emits spurious floating warnings
    # inside otherwise finite GEMM calls; the explicit finiteness checks below
    # remain authoritative.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pinv = np.linalg.pinv(design)
        coefficients = pinv @ response
        residuals = response - design @ coefficients
    rank = int(np.linalg.matrix_rank(design))
    residual_df = int(len(design) - rank)
    if residual_df <= 0:
        raise AssertionError("GLM residual degrees of freedom must be positive")
    residual_variance = np.sum(np.square(residuals), axis=0) / residual_df
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        xtx_inverse = np.linalg.pinv(design.T @ design)
    output: dict[str, ContrastMaps] = {}
    eps = np.finfo(np.float64).eps
    for name, weights in CONTRAST_WEIGHTS.items():
        contrast = design_contrast(weights, n_cycles)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            effect = (contrast @ coefficients).reshape(IMAGE_SHAPE)
        variance_multiplier = float(contrast @ xtx_inverse @ contrast)
        standard_error = np.sqrt(np.maximum(residual_variance * variance_multiplier, 0.0)).reshape(IMAGE_SHAPE)
        t_map = effect / np.maximum(standard_error, eps)
        p_map = (2.0 * student_t.sf(np.abs(t_map), df=residual_df)).astype(np.float64)
        p_map = np.nan_to_num(p_map, nan=1.0, posinf=0.0, neginf=0.0)
        q_map, fdr_mask = benjamini_hochberg(p_map)
        per_cycle = cycle_contrast_maps(values, weights)
        if not np.allclose(effect, per_cycle.mean(axis=0), rtol=1e-8, atol=1e-10):
            raise AssertionError("balanced GLM estimate differs from mean complete-cycle contrast")
        cycle_std = per_cycle.std(axis=0, ddof=1)
        standardized = effect / (cycle_std + 1e-12)
        maps = (effect, standard_error, t_map, p_map, q_map, standardized)
        if not all(np.isfinite(value).all() for value in maps):
            raise AssertionError(f"{name} GLM produced a non-finite map")
        output[name] = ContrastMaps(
            effect=effect,
            standard_error=standard_error,
            t_map=t_map,
            p_map=p_map,
            q_map=q_map,
            fdr_mask=fdr_mask,
            standardized=standardized,
            cycle_maps=per_cycle,
        )
    return output, rank, residual_df


def safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def random_cycle_halves(cycles: Sequence[int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(cycles, dtype=np.int64)
    if len(values) < 2:
        raise ValueError("split-half requires at least two cycles")
    permuted = rng.permutation(values)
    low = len(values) // 2
    n_a = low
    if len(values) % 2 and bool(rng.integers(0, 2)):
        n_a += 1
    return permuted[:n_a], permuted[n_a:]


def split_half_reproducibility(
    cycle_maps: np.ndarray,
    *,
    n_splits: int = N_SPLITS,
    seed: int = V9_RANDOM_SEED,
) -> dict[str, Any]:
    values = np.asarray(cycle_maps, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    correlations = np.empty(int(n_splits), dtype=np.float64)
    cosines = np.empty(int(n_splits), dtype=np.float64)
    cycles = np.arange(len(values), dtype=np.int64)
    sizes_a = []
    for split_i in range(int(n_splits)):
        half_a, half_b = random_cycle_halves(cycles, rng)
        map_a = values[half_a].mean(axis=0)
        map_b = values[half_b].mean(axis=0)
        correlations[split_i] = safe_pearson(map_a, map_b)
        cosines[split_i] = safe_cosine(map_a, map_b)
        sizes_a.append(len(half_a))
    if not np.isfinite(correlations).all() or not np.isfinite(cosines).all():
        raise AssertionError("split-half metrics must be finite")
    return {
        "n_splits": int(n_splits),
        "split_half_corr_median": float(np.median(correlations)),
        "split_half_corr_2.5pct": float(np.percentile(correlations, 2.5)),
        "split_half_corr_97.5pct": float(np.percentile(correlations, 97.5)),
        "split_half_cosine_median": float(np.median(cosines)),
        "split_half_cosine_2.5pct": float(np.percentile(cosines, 2.5)),
        "split_half_cosine_97.5pct": float(np.percentile(cosines, 97.5)),
        "half_a_min_n": int(min(sizes_a)),
        "half_a_max_n": int(max(sizes_a)),
        "split_unit": "complete_cycle",
    }


def bootstrap_gs_ds_concordance(
    gs_cycle_maps: np.ndarray,
    ds_cycle_maps: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = V9_RANDOM_SEED,
) -> dict[str, Any]:
    gs = np.asarray(gs_cycle_maps, dtype=np.float64)
    ds = np.asarray(ds_cycle_maps, dtype=np.float64)
    if gs.shape != ds.shape or len(gs) < 2:
        raise ValueError("GS/DS bootstrap requires matching maps from at least two cycles")
    rng = np.random.default_rng(int(seed))
    correlations = np.empty(int(n_bootstrap), dtype=np.float64)
    cosines = np.empty(int(n_bootstrap), dtype=np.float64)
    for bootstrap_i in range(int(n_bootstrap)):
        indices = rng.integers(0, len(gs), size=len(gs))
        gs_mean = gs[indices].mean(axis=0)
        ds_mean = ds[indices].mean(axis=0)
        correlations[bootstrap_i] = safe_pearson(gs_mean, ds_mean)
        cosines[bootstrap_i] = safe_cosine(gs_mean, ds_mean)
    return {
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_corr_median": float(np.median(correlations)),
        "bootstrap_corr_2.5pct": float(np.percentile(correlations, 2.5)),
        "bootstrap_corr_97.5pct": float(np.percentile(correlations, 97.5)),
        "bootstrap_cosine_median": float(np.median(cosines)),
        "bootstrap_cosine_2.5pct": float(np.percentile(cosines, 2.5)),
        "bootstrap_cosine_97.5pct": float(np.percentile(cosines, 97.5)),
        "resampling_unit": "complete_cycle_paired_GS_DS",
    }


def analyze_session(
    data: SessionBlockImages,
    *,
    n_splits: int = N_SPLITS,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = V9_RANDOM_SEED,
) -> SessionAnalysis:
    contrasts, rank, residual_df = fit_pixelwise_glm(data.images)
    glm_rows = []
    split_rows = []
    n_valid = int(np.prod(IMAGE_SHAPE))
    for contrast_i, (name, maps) in enumerate(contrasts.items()):
        n_fdr = int(maps.fdr_mask.sum())
        glm_rows.append({
            "session": data.session,
            "contrast": name,
            "contrast_role": "primary" if name == "STIM_PRESENCE" else (
                "secondary_confirmatory" if name in ("GS", "DS") else "exploratory_temporal_position_confounded"
            ),
            "n_cycles": data.n_cycles,
            "n_observations": 4 * data.n_cycles,
            "design_rank": rank,
            "residual_df": residual_df,
            "RMS_effect": float(np.sqrt(np.mean(np.square(maps.effect)))),
            "mean_abs_effect": float(np.mean(np.abs(maps.effect))),
            "RMS_standardized_effect": float(np.sqrt(np.mean(np.square(maps.standardized)))),
            "n_valid_pixels": n_valid,
            "n_fdr_pixels": n_fdr,
            "fdr_fraction": float(n_fdr / n_valid),
            "mean_abs_effect_FDR_pixels": (
                float(np.mean(np.abs(maps.effect[maps.fdr_mask]))) if n_fdr else np.nan
            ),
            "fdr_method": "Benjamini-Hochberg within session x contrast",
            "fdr_q": FDR_ALPHA,
            "fixed_order_warning": GD_WARNING if name == "GD" else FIXED_ORDER_WARNING,
        })
        split_rows.append({
            "session": data.session,
            "contrast": name,
            "n_cycles": data.n_cycles,
            **split_half_reproducibility(
                maps.cycle_maps,
                n_splits=n_splits,
                seed=int(seed) + int(data.session) * 10 + contrast_i,
            ),
        })
    gs = contrasts["GS"].cycle_maps
    ds = contrasts["DS"].cycle_maps
    concordance = {
        "session": data.session,
        "n_cycles": data.n_cycles,
        "GS_DS_spatial_corr": safe_pearson(gs.mean(axis=0), ds.mean(axis=0)),
        "GS_DS_spatial_cosine": safe_cosine(gs.mean(axis=0), ds.mean(axis=0)),
        "comparison_scope": "within_session_only_not_anatomically_registered",
    }
    bootstrap = {
        "session": data.session,
        "contrast_pair": "GS_vs_DS",
        **bootstrap_gs_ds_concordance(
            gs, ds, n_bootstrap=n_bootstrap, seed=int(seed) + int(data.session) * 100
        ),
    }
    return SessionAnalysis(
        session=data.session,
        n_cycles=data.n_cycles,
        design_rank=rank,
        residual_df=residual_df,
        background=data.background,
        contrasts=contrasts,
        glm_rows=tuple(glm_rows),
        split_rows=tuple(split_rows),
        concordance_row=concordance,
        bootstrap_row=bootstrap,
    )


def _resolve_completed_root(candidate: Path, required: Sequence[str], label: str) -> Path:
    candidate = Path(candidate)
    if all((candidate / value).is_file() for value in required):
        return candidate
    first = required[0]
    matches = sorted(
        path.parents[len(Path(first).parts) - 1]
        for path in candidate.glob(f"**/{first}")
        if all((path.parents[len(Path(first).parts) - 1] / value).is_file() for value in required)
    )
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise FileNotFoundError(f"could not resolve exactly one completed {label} root under {candidate}: {unique}")
    return unique[0]


def _exact_session_table(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    output = frame.copy()
    output["session"] = output["session"].astype(str)
    sessions = output["session"].tolist()
    if sessions != list(EXPECTED_SESSIONS) or output["session"].duplicated().any():
        raise AssertionError(f"{label} does not contain exactly the frozen nine sessions in order")
    return output


def load_within_session_ba(v7_candidate: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    root = _resolve_completed_root(
        v7_candidate,
        (
            "summaries/session_diagnostic_table.csv",
            "audit/clean4_identity_check.csv",
            "report/cross_session_feature_factor_report.md",
        ),
        "v7",
    )
    path = root / "summaries/session_diagnostic_table.csv"
    source = pd.read_csv(path)
    required = {"session", "within_session_BA", "within_session_BA_seed_std"}
    if not required.issubset(source.columns):
        raise AssertionError(f"v7 within-session BA artifact missing {sorted(required - set(source.columns))}")
    values = _exact_session_table(
        source[["session", "within_session_BA", "within_session_BA_seed_std"]],
        label="v7 within-session BA",
    )
    if not np.isfinite(values["within_session_BA"]).all():
        raise AssertionError("v7 within-session BA contains non-finite values")
    audit = values.rename(columns={"within_session_BA": "reused_within_session_BA"}).copy()
    audit["source_artifact"] = str(path)
    audit["source_sha256"] = sha256_file(path)
    audit["source_model"] = "SmallCNN feature-mean"
    audit["recomputed_or_reselected"] = False
    audit["status"] = "PASS"
    return values[["session", "within_session_BA", "within_session_BA_seed_std"]], audit, root


def load_v8_metrics(v8_candidate: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    root = _resolve_completed_root(
        v8_candidate,
        (
            "summaries/stimulus_vector_magnitude.csv",
            "summaries/stimulus_vector_stability.csv",
            "report/session_centered_vector_alignment_report.md",
        ),
        "v8",
    )
    magnitude_path = root / "summaries/stimulus_vector_magnitude.csv"
    stability_path = root / "summaries/stimulus_vector_stability.csv"
    magnitude = pd.read_csv(magnitude_path)
    stability = pd.read_csv(stability_path)
    selector_m = (magnitude["representation"] == "RAW_SPATIAL_PCA") & (magnitude["task"] == "stimulus_presence")
    selector_s = (stability["representation"] == "RAW_SPATIAL_PCA") & (stability["task"] == "stimulus_presence")
    mag = _exact_session_table(
        magnitude.loc[selector_m, ["session", "stimulus_vector_norm"]].reset_index(drop=True),
        label="v8 RAW_SPATIAL_PCA magnitude",
    )
    stab = _exact_session_table(
        stability.loc[selector_s, ["session", "split_half_vector_stability"]].reset_index(drop=True),
        label="v8 RAW_SPATIAL_PCA stability",
    )
    values = mag.merge(stab, on="session", validate="one_to_one").rename(columns={
        "stimulus_vector_norm": "v8_stimulus_vector_magnitude",
        "split_half_vector_stability": "v8_split_half_vector_stability",
    })
    if not np.isfinite(values.iloc[:, 1:].to_numpy(dtype=float)).all():
        raise AssertionError("v8 reused metrics contain non-finite values")
    audit = values.copy()
    audit["magnitude_artifact"] = str(magnitude_path)
    audit["magnitude_sha256"] = sha256_file(magnitude_path)
    audit["stability_artifact"] = str(stability_path)
    audit["stability_sha256"] = sha256_file(stability_path)
    audit["representation"] = "RAW_SPATIAL_PCA"
    audit["task"] = "stimulus_presence"
    audit["recomputed_or_reselected"] = False
    audit["status"] = "PASS"
    return values, audit, root


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm correction requires a finite vector of probabilities")
    order = np.argsort(values, kind="mergesort")
    sorted_p = values[order]
    adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * sorted_p)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def planned_ba_associations(diagnostic: pd.DataFrame) -> pd.DataFrame:
    specifications = (
        ("binary_RMS_standardized_effect", "effect_magnitude"),
        ("binary_split_half_corr_median", "reproducibility"),
        ("GS_DS_spatial_corr", "GS_DS_concordance"),
    )
    rows = []
    for metric, family_label in specifications:
        result = exact_spearman_permutation(diagnostic[metric], diagnostic["within_session_BA"])
        rows.append({
            "metric": metric,
            "planned_family_label": family_label,
            "outcome": "within_session_BA",
            "n_sessions": len(diagnostic),
            **result,
        })
    output = pd.DataFrame(rows)
    output["holm_adjusted_p"] = holm_adjust(output["permutation_p_two_sided"])
    output["multiple_testing_family"] = "three_planned_BA_associations_only"
    return output


def v8_v9_stability_association(diagnostic: pd.DataFrame) -> pd.DataFrame:
    result = exact_spearman_permutation(
        diagnostic["v8_split_half_vector_stability"],
        diagnostic["binary_split_half_corr_median"],
    )
    return pd.DataFrame([{
        "predictor": "v8_RAW_SPATIAL_PCA_split_half_vector_stability",
        "outcome": "v9_binary_spatial_map_split_half_corr_median",
        "analysis_role": "secondary_mechanistic_association",
        "n_sessions": len(diagnostic),
        "included_in_primary_Holm_family": False,
        **result,
    }])


def build_diagnostic_table(
    glm_summary: pd.DataFrame,
    split_metrics: pd.DataFrame,
    concordance: pd.DataFrame,
    within_ba: pd.DataFrame,
    v8_metrics: pd.DataFrame,
    *,
    sessions: Sequence[str] = tuple(EXPECTED_SESSIONS),
) -> pd.DataFrame:
    primary_glm = glm_summary[glm_summary["contrast"] == "STIM_PRESENCE"].copy().rename(columns={
        "RMS_effect": "binary_RMS_effect",
        "RMS_standardized_effect": "binary_RMS_standardized_effect",
        "n_fdr_pixels": "binary_n_fdr_pixels",
        "fdr_fraction": "binary_fdr_fraction",
    })
    primary_split = split_metrics[split_metrics["contrast"] == "STIM_PRESENCE"].copy().rename(columns={
        "split_half_corr_median": "binary_split_half_corr_median",
        "split_half_corr_2.5pct": "binary_split_half_corr_2.5pct",
        "split_half_corr_97.5pct": "binary_split_half_corr_97.5pct",
    })
    keep_glm = [
        "session", "n_cycles", "binary_RMS_effect", "binary_RMS_standardized_effect",
        "binary_n_fdr_pixels", "binary_fdr_fraction",
    ]
    keep_split = [
        "session", "binary_split_half_corr_median", "binary_split_half_corr_2.5pct",
        "binary_split_half_corr_97.5pct",
    ]
    output = (
        primary_glm[keep_glm]
        .merge(primary_split[keep_split], on="session", validate="one_to_one")
        .merge(concordance[["session", "GS_DS_spatial_corr", "GS_DS_spatial_cosine"]], on="session", validate="one_to_one")
        .merge(within_ba, on="session", validate="one_to_one")
        .merge(v8_metrics, on="session", validate="one_to_one")
    )
    sessions = [str(value) for value in sessions]
    output = output.set_index("session").reindex(sessions).reset_index()
    output["historical_group"] = np.where(
        output["session"].isin(STRONG_SESSIONS), "historically_strong", "historically_weak"
    )
    effect_threshold = float(output["binary_RMS_standardized_effect"].median())
    reproducibility_threshold = float(output["binary_split_half_corr_median"].median())
    high_effect = output["binary_RMS_standardized_effect"] >= effect_threshold
    high_repro = output["binary_split_half_corr_median"] >= reproducibility_threshold
    output["spatial_diagnostic_pattern"] = np.select(
        [high_effect & high_repro, ~high_effect & high_repro, high_effect & ~high_repro],
        ["S1", "S2", "S3"],
        default="S4",
    )
    descriptions = {
        "S1": "strong and repeatable condition-associated spatial contrast",
        "S2": "stable images but weak spatial task contrast",
        "S3": "spatial response varies substantially across cycles",
        "S4": "weak and unstable spatial task contrast",
    }
    output["spatial_diagnostic_description"] = output["spatial_diagnostic_pattern"].map(descriptions)
    output["descriptive_effect_threshold"] = effect_threshold
    output["descriptive_reproducibility_threshold"] = reproducibility_threshold
    output["threshold_rule"] = f"{len(sessions)}-session median split; descriptive only; no subgroup test"
    if output["session"].tolist() != sessions or output.isna().any().any():
        raise AssertionError("diagnostic table lost a preregistered session")
    return output


def required_output_paths() -> tuple[str, ...]:
    paths = [
        "audit/clean4_identity_check.csv",
        "audit/within_session_ba_reuse.csv",
        "audit/v8_metric_reuse.csv",
        "audit/fixed_order_confounds.md",
        "audit/config_freeze.md",
        "glm/pixelwise_glm_summary.csv",
        "reproducibility/split_half_metrics.csv",
        "reproducibility/gs_ds_concordance.csv",
        "reproducibility/bootstrap_metrics.csv",
        "summaries/session_spatial_diagnostic_table.csv",
        "summaries/spatial_vs_withinBA_associations.csv",
        "summaries/v8_vs_v9_stability_association.csv",
        "summaries/fdr_summary.csv",
        "figures/binary_spatial_maps_9sessions.png",
        "figures/spatial_reproducibility_by_session.png",
        "figures/spatial_effect_vs_within_BA.png",
        "figures/spatial_reproducibility_vs_within_BA.png",
        "figures/GS_DS_spatial_concordance_by_session.png",
        "figures/spatial_diagnostic_overview.png",
        "report/spatial_glm_reproducibility_report.md",
        "run_command_server.txt",
        "run_log_server.txt",
    ]
    for session in EXPECTED_SESSIONS:
        paths.extend([
            f"figures/primary_binary_maps/session_{session}_primary_binary_maps.png",
            f"figures/GS_maps/session_{session}_GS_map.png",
            f"figures/DS_maps/session_{session}_DS_map.png",
            f"figures/GS_DS_comparison/session_{session}_GS_DS_comparison.png",
            f"figures/exploratory_GD_maps/session_{session}_exploratory_GD_map.png",
        ])
        for contrast in CONTRAST_WEIGHTS:
            paths.extend([
                f"glm/contrast_maps/session_{session}_{contrast}.npy",
                f"glm/standard_error_maps/session_{session}_{contrast}.npy",
                f"glm/standardized_maps/session_{session}_{contrast}.npy",
                f"glm/t_maps/session_{session}_{contrast}.npy",
                f"glm/p_maps/session_{session}_{contrast}.npy",
                f"glm/q_maps/session_{session}_{contrast}.npy",
                f"glm/fdr_masks/session_{session}_{contrast}.npy",
            ])
    return tuple(paths)


def missing_outputs(output_dir: Path) -> list[str]:
    return [relative for relative in required_output_paths() if not (Path(output_dir) / relative).is_file()]
