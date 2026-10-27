"""Percent reference levels (§5.3.2) and interpolated instants (§5.3.3)."""
from __future__ import annotations

import numpy as np

__all__ = ["interpolate_instant", "reference_levels"]


def reference_levels(
    levels: tuple[float, float],
    reference: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Absolute (lower, mid, upper) reference levels: low + (high - low) * p / 100  (§5.3.2).

    No argument checks are performed here; validation happens in the public API.
    """
    low = float(levels[0])
    high = float(levels[1])
    percents = np.asarray(reference, dtype=float)
    lower, mid, upper = (low + (high - low) * percents / 100.0).tolist()
    return lower, mid, upper


def interpolate_instant(
    x0: np.ndarray | float,
    y0: np.ndarray | float,
    x1: np.ndarray | float,
    y1: np.ndarray | float,
    level: np.ndarray | float,
) -> np.ndarray | float:
    """x0 + (x1 - x0) * (level - y0) / (y1 - y0); x0 where y1 == y0  (§5.3.3).

    Vectorised over numpy arrays; returns an ndarray for array input and a float for scalar input.
    """
    x0_arr = np.asarray(x0, dtype=float)
    y0_arr = np.asarray(y0, dtype=float)
    x1_arr = np.asarray(x1, dtype=float)
    y1_arr = np.asarray(y1, dtype=float)
    level_arr = np.asarray(level, dtype=float)

    dy = y1_arr - y0_arr
    flat = dy == 0.0
    safe_dy = np.where(flat, 1.0, dy)
    result = np.where(flat, x0_arr, x0_arr + (x1_arr - x0_arr) * (level_arr - y0_arr) / safe_dy)

    if result.ndim == 0:
        return float(result)
    return result
