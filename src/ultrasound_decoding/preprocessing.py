from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpatialFilterConfig:
    method: str = "none"
    radius: int = 0
    mode: str = "reflect"

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "radius": int(self.radius), "mode": self.mode}


def pillbox_kernel(radius: int) -> np.ndarray:
    """Return a normalized circular mean-filter kernel."""
    if radius < 0:
        raise ValueError("pillbox radius must be >= 0")
    if radius == 0:
        return np.ones((1, 1), dtype=np.float64)

    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    mask = (xx**2 + yy**2) <= radius**2
    kernel = mask.astype(np.float64)
    kernel /= kernel.sum()
    return kernel


def apply_pillbox_spatial_filter(
    X: np.ndarray,
    radius: int,
    mode: str = "reflect",
) -> np.ndarray:
    """Apply a 2D pillbox filter independently over the last two image axes.

    Supported shapes include [H, W], [T, H, W], and [n_trials, T, H, W].
    Any leading dimensions are treated as frame/trial/session-like axes and are
    never mixed by the filter.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if X.ndim < 2:
        raise ValueError(f"spatial filtering requires at least 2 dimensions, got {X.shape}")
    if radius == 0:
        return X.copy()

    kernel = pillbox_kernel(radius)
    padded = np.pad(
        X.astype(np.float64, copy=False),
        [(0, 0)] * (X.ndim - 2) + [(radius, radius), (radius, radius)],
        mode=mode,
    )
    out = np.zeros(X.shape, dtype=np.float64)
    for dy in range(kernel.shape[0]):
        for dx in range(kernel.shape[1]):
            weight = kernel[dy, dx]
            if weight == 0:
                continue
            out += weight * padded[
                ...,
                dy : dy + X.shape[-2],
                dx : dx + X.shape[-1],
            ]
    return out.astype(X.dtype, copy=False) if np.issubdtype(X.dtype, np.floating) else out


def apply_spatial_filter(
    X: np.ndarray,
    config: SpatialFilterConfig | dict[str, object] | None = None,
) -> np.ndarray:
    """Apply the configured spatial filter to fUS/Power Doppler image frames."""
    if config is None:
        cfg = SpatialFilterConfig()
    elif isinstance(config, SpatialFilterConfig):
        cfg = config
    else:
        cfg = SpatialFilterConfig(
            method=str(config.get("method", "none")),
            radius=int(config.get("radius", 0)),
            mode=str(config.get("mode", "reflect")),
        )

    method = cfg.method.lower()
    if method == "none":
        return X
    if method == "pillbox":
        return apply_pillbox_spatial_filter(X, radius=cfg.radius, mode=cfg.mode)
    raise ValueError(f"Unknown spatial filter method: {cfg.method}")
