"""Flatness of a window."""

from __future__ import annotations

import math

import numpy as np

__all__ = ["flatness_of"]

_MIN_WINDOW_SAMPLES = 5
_FINAL_LEVEL_TOL = 1e-12
_FINAL_LEVEL_SAMPLES = 10


def flatness_of(
    x: np.ndarray, y: np.ndarray, bounds: tuple[float, float], normalize: bool = True
) -> float:
    """Population std-dev of y within bounds, optionally normalised by the record's final level.

    x and y are the whole record. The window is the samples with bounds[0] <= x <= bounds[1].
    The std-dev uses ddof=0. When normalize is true, the result is divided by the absolute mean
    of the last 10 samples of the whole y record (not of the window).

    Returns NaN when the window holds fewer than 5 samples, or when normalizing by a final
    level below 1e-12.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    lower, upper = bounds
    window = y_arr[(x_arr >= lower) & (x_arr <= upper)]
    if window.size < _MIN_WINDOW_SAMPLES:
        return math.nan
    std = float(np.std(window, ddof=0))
    if not normalize:
        return std
    final = abs(float(np.mean(y_arr[-_FINAL_LEVEL_SAMPLES:])))
    if final < _FINAL_LEVEL_TOL:
        return math.nan
    return std / final
