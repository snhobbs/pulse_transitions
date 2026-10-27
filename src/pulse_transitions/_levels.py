"""State levels (§5.2)."""
from __future__ import annotations

import numpy as np

from ._types import LevelError, StateLevels

__all__ = ["detect_levels"]


def detect_levels(
    y: np.ndarray,
    method: str = "mode",
    bins: int = 100,
    split: tuple[float, float] = (0.5, 0.5),
    endpoint_samples: int = 100,
) -> StateLevels:
    """Return the low and high state levels of a 1-D float64 record.

    Parameters are assumed validated. ``mode`` and ``mean`` build a histogram
    over ``[y.min(), y.max()]`` and take the levels from the lower and upper
    sub-histograms selected by ``split``; ``peak`` uses the extremes; and
    ``endpoints`` uses the means of the first and last ``endpoint_samples``.

    Raises:
        LevelError: If the record is constant, or a sub-histogram has no bins.
    """
    y_min = float(y.min())
    y_max = float(y.max())
    if y_min == y_max:
        raise LevelError("signal is constant; two state levels cannot be determined")

    if method == "peak":
        return StateLevels(low=y_min, high=y_max, method=method)

    if method == "endpoints":
        head = float(y[:endpoint_samples].mean())
        tail = float(y[-endpoint_samples:].mean())
        low, high = sorted((head, tail))
        return StateLevels(low=low, high=high, method=method)

    # One pass for the bin index (equal bins, last bin closed), then counts and sums from it.
    edges = np.linspace(y_min, y_max, bins + 1)
    scaled = np.subtract(y, y_min)
    scaled *= bins / (y_max - y_min)
    indices = scaled.astype(np.int32)
    np.minimum(indices, bins - 1, out=indices)
    sums = np.bincount(indices, weights=y, minlength=bins)
    counts = np.bincount(indices, minlength=bins)
    bin_counts = counts

    lower_stop = int(np.floor(split[0] * bins))
    upper_start = int(np.ceil(split[1] * bins))
    low = _sub_level(method, lower_stop, 0, bins, counts, sums, bin_counts, "lower")
    high = _sub_level(method, bins, upper_start, bins, counts, sums, bin_counts, "upper")
    return StateLevels(low=low, high=high, method=method, histogram=counts, bin_edges=edges)


def _sub_level(
    method: str,
    stop: int,
    start: int,
    bins: int,
    counts: np.ndarray,
    sums: np.ndarray,
    bin_counts: np.ndarray,
    label: str,
) -> float:
    """Return the level of one sub-histogram spanning bins ``start`` to ``stop``."""
    if stop <= start:
        raise LevelError(f"split leaves no bins for the {label} state level; use a smaller split or more bins")
    if method == "mode":
        best = start + int(np.argmax(counts[start:stop]))
        return float(sums[best] / bin_counts[best])
    return float(sums[start:stop].sum() / bin_counts[start:stop].sum())
