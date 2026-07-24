from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def _rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman_r(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(valid.sum()) < 3:
        return np.nan, int(valid.sum())
    ra = _rankdata_average(aa[valid])
    rb = _rankdata_average(bb[valid])
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt(np.sum(ra**2) * np.sum(rb**2)))
    if denom <= 0:
        return np.nan, int(valid.sum())
    return float(np.sum(ra * rb) / denom), int(valid.sum())


def top_fraction_mask(arr: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(values)
    mask = np.zeros(values.shape, dtype=bool)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return mask
    n_top = max(1, int(np.ceil(n_valid * fraction)))
    valid_values = values[valid]
    threshold = np.partition(valid_values, max(0, n_valid - n_top))[n_valid - n_top]
    mask[valid] = values[valid] >= threshold
    return mask


def top10_overlap(a: np.ndarray, b: np.ndarray) -> float:
    ma = top_fraction_mask(a, 0.10)
    mb = top_fraction_mask(b, 0.10)
    denom = int(ma.sum() + mb.sum())
    if denom == 0:
        return np.nan
    return float(2 * np.logical_and(ma, mb).sum() / denom)


def peak_location(arr: np.ndarray) -> tuple[int, int]:
    values = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(values).any():
        return -1, -1
    idx = int(np.nanargmax(values))
    return tuple(int(x) for x in np.unravel_index(idx, values.shape))


def compare_maps(
    *,
    session: str,
    model: str,
    method: str,
    comparison_type: str,
    item_a: str,
    item_b: str,
    map_a: np.ndarray,
    map_b: np.ndarray,
) -> dict[str, object]:
    rho, _ = spearman_r(map_a, map_b)
    peak_a = peak_location(map_a)
    peak_b = peak_location(map_b)
    return {
        "session": session,
        "model": model,
        "method": method,
        "comparison_type": comparison_type,
        "item_a": item_a,
        "item_b": item_b,
        "spearman_r": rho,
        "top10_overlap": top10_overlap(map_a, map_b),
        "peak_row_distance": abs(peak_a[0] - peak_b[0]) if min(*peak_a, *peak_b) >= 0 else np.nan,
        "peak_col_distance": abs(peak_a[1] - peak_b[1]) if min(*peak_a, *peak_b) >= 0 else np.nan,
    }


def pairwise_stability_rows(
    *,
    session: str,
    model: str,
    method: str,
    maps: dict[str, np.ndarray],
    comparison_type: str,
) -> list[dict[str, object]]:
    rows = []
    for item_a, item_b in combinations(sorted(maps), 2):
        rows.append(
            compare_maps(
                session=session,
                model=model,
                method=method,
                comparison_type=comparison_type,
                item_a=item_a,
                item_b=item_b,
                map_a=maps[item_a],
                map_b=maps[item_b],
            )
        )
    return rows


def cross_method_agreement_row(
    *,
    session: str,
    method_a: str,
    method_b: str,
    map_a: np.ndarray,
    map_b: np.ndarray,
) -> dict[str, object]:
    rho, valid_n = spearman_r(map_a, map_b)
    return {
        "session": session,
        "method_a": method_a,
        "method_b": method_b,
        "spearman_r": rho,
        "top10_overlap": top10_overlap(map_a, map_b),
        "valid_pixel_count": valid_n,
    }


def signed_region_summary(arr: np.ndarray) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) == 0:
        return {"positive_fraction": np.nan, "negative_fraction": np.nan}
    return {
        "positive_fraction": float(np.mean(values[valid] > 0)),
        "negative_fraction": float(np.mean(values[valid] < 0)),
    }


def load_existing_map(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path)
    if np.isinf(arr[np.isfinite(arr)]).any():
        raise AssertionError(f"{path} contains Inf")
    return arr


def write_completeness_report(path: Path, rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

