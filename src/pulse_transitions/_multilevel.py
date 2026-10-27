"""Scale-adaptive edge location and flat-level measurement for multi-level (TDR) traces."""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from ._reference import reference_levels
from ._transitions import find_transitions
from ._types import EdgeMeasurement, EdgeNotFoundError, Polarity

__all__ = ["locate_and_measure"]

_MIN_SAMPLES = 16
_NOISE_SIGMAS = 8.0  # a step stands out at this many noise sigmas of a difference
_MIN_FRACTION = (
    0.1  # ... and at this fraction of the largest difference at the same scale
)
_MAD_TO_SIGMA = 1.4826
_MIN_LEVEL_SAMPLES = 3
_LEVEL_SPAN_RUNS = 3  # flat samples used for a level, in run lengths
_LEVEL_SPAN_MIN = 20
_RESOLVED_GROWTH = (
    1.5  # peak growth from s to 2s below which the edge counts as resolved
)
_TRUSTED_PEAK = (
    2.0  # a run peaking below this many noise thresholds is noise-limited fragments
)


Run = tuple[
    int, int, int, float
]  # (first sample, last sample, sign, peak |difference|)


def _steep_runs(y: np.ndarray, scale: int, floor: float) -> list[Run]:
    """Runs of steep same-sign change at `scale`; runs of the same sign closer than `scale`
    samples are merged."""
    change = y[scale:] - y[:-scale]
    magnitude = np.abs(change)
    largest = float(magnitude.max())
    if largest <= 0.0:
        return []
    steep = (magnitude > 0) & (magnitude >= max(floor, _MIN_FRACTION * largest))
    signed = np.where(steep, np.sign(change), 0).astype(np.int8)
    padded = np.r_[np.int8(0), signed, np.int8(0)]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[:-1], edges[1:] - 1
    keep = signed[starts] != 0
    runs: list[Run] = []
    for first, last in zip(starts[keep], ends[keep]):
        sign = int(signed[first])
        span_end = int(last) + scale
        peak = float(magnitude[first : last + 1].max())
        if runs and runs[-1][2] == sign and first - runs[-1][1] < scale:
            runs[-1] = (runs[-1][0], span_end, sign, max(runs[-1][3], peak))
        else:
            runs.append((int(first), span_end, sign, peak))
    return runs


def _distance(x: np.ndarray, run: Run, around: float) -> float:
    lo, hi = float(x[run[0]]), float(x[min(run[1], len(x) - 1)])
    if lo <= around <= hi:
        return 0.0
    return min(abs(around - lo), abs(around - hi))


def _nearest(
    x: np.ndarray, runs: list[Run], around: float, max_distance: float
) -> int | None:
    """Index of the run nearest `around` if it lies within `max_distance`."""
    if not runs:
        return None
    distances = [_distance(x, run, around) for run in runs]
    nearest = int(np.argmin(distances))
    return nearest if distances[nearest] <= max_distance else None


def _merges(runs: list[Run], keep: int, wider: Run) -> bool:
    """Whether `wider` contains the centre of a run other than runs[keep]."""
    return any(
        wider[0] <= 0.5 * (r[0] + r[1]) <= wider[1]
        for n, r in enumerate(runs)
        if n != keep
    )


def _locate(
    x: np.ndarray, y: np.ndarray, around: float, max_distance: float
) -> tuple[list[Run], int]:
    """The runs at the scale that resolves the edge near `around`, and that edge's index.

    Walks the scales up from 1 and stops at the first scale s with a run near `around` once,
    at 2s, that run's peak grows by less than 1.5x (resolved), it would swallow a neighbour
    (merged), or it disappears or 2s exceeds n // 8 (lost). Resolved and merged only count
    once the run's peak is at least twice the noise threshold.
    """
    n = len(y)
    diffs = np.diff(y)
    sigma = (
        _MAD_TO_SIGMA
        * float(np.median(np.abs(diffs - np.median(diffs))))
        / math.sqrt(2.0)
    )
    floor = _NOISE_SIGMAS * math.sqrt(2.0) * sigma
    chosen: tuple[list[Run], int] | None = None
    scale = 1
    while scale <= n // 8:
        runs = _steep_runs(y, scale, floor)
        near = _nearest(x, runs, around, max_distance)
        if chosen is not None:
            previous, keep = chosen
            if near is None:
                return chosen
            wider = runs[near]
            trusted = previous[keep][3] >= _TRUSTED_PEAK * floor
            if trusted and (
                wider[3] < _RESOLVED_GROWTH * previous[keep][3]
                or _merges(previous, keep, wider)
            ):
                return chosen
        chosen = (runs, near) if near is not None else None
        scale *= 2
    if chosen is not None:
        return chosen
    msg = f"No edge within {max_distance} of {around}: point at an edge."
    raise EdgeNotFoundError(msg)


def _flat_levels(
    y: np.ndarray, first: int, last: int, lower: int, upper: int, sign: int
) -> tuple[float, float]:
    """(bottom, top) from the medians of the flat samples either side of the run first..last."""
    span = max(_LEVEL_SPAN_RUNS * (last - first + 1), _LEVEL_SPAN_MIN)
    before = y[max(lower, first - span) : first]
    after = y[last + 1 : min(upper, last + span) + 1]
    if len(before) < _MIN_LEVEL_SAMPLES or len(after) < _MIN_LEVEL_SAMPLES:
        msg = (
            "This edge has fewer than 3 flat samples on one side to take a level from."
        )
        raise EdgeNotFoundError(msg)
    start_level, end_level = float(np.median(before)), float(np.median(after))
    if (end_level - start_level) * sign <= 0:
        msg = "This edge does not settle to a new level in its direction."
        raise EdgeNotFoundError(msg)
    return min(start_level, end_level), max(start_level, end_level)


def locate_and_measure(
    x: np.ndarray,
    y: np.ndarray,
    around: float,
    max_distance: float,
    reference: tuple[float, float, float],
    levels: tuple[float, float] | None,
    mid_method: str = "median",
) -> EdgeMeasurement:
    """The edge nearest `around`, measured between the flat levels either side of it (or `levels`).

    Args:
        x: Strictly increasing sample instants.
        y: Samples.
        around: An x position at or near the edge.
        max_distance: How far from `around` the edge may be.
        reference: Percent (lower, mid, upper) reference levels.
        levels: (bottom, top) to measure between; None to take them from the flat parts.
        mid_method: "median" or "first" mid-level crossing.

    Returns:
        The edge's transition and the (bottom, top) levels used. Indices refer to `x`/`y`.

    Raises:
        EdgeNotFoundError: No edge near `around` (§E11), too few flat samples beside it (§E12), or
            it doesn't settle to a new level / cross the reference levels (§E13).
    """
    n = len(y)
    if n < _MIN_SAMPLES:
        msg = (
            f"There is no edge to find in {n} samples (at least {_MIN_SAMPLES} needed)."
        )
        raise EdgeNotFoundError(msg)
    runs, nearest = _locate(x, y, around, max_distance)
    first, last, sign, _ = runs[nearest]
    last = min(last, n - 1)
    lower = runs[nearest - 1][1] + 1 if nearest > 0 else 0
    upper = runs[nearest + 1][0] - 1 if nearest + 1 < len(runs) else n - 1

    if levels is None:
        bottom, top = _flat_levels(y, first, last, lower, upper, sign)
    else:
        bottom, top = float(levels[0]), float(levels[1])

    window = slice(lower, upper + 1)
    found = find_transitions(
        x[window],
        y[window],
        reference_levels((bottom, top), reference),
        mid_method,
        (bottom, top),
    )
    polarity = Polarity.RISING if sign > 0 else Polarity.FALLING
    wanted = [t for t in found if t.polarity == polarity]
    if not wanted:
        msg = f"The edge does not cross the reference levels of {(bottom, top)}: check the levels."
        raise EdgeNotFoundError(msg)
    centre = 0.5 * (float(x[first]) + float(x[last]))
    best = min(wanted, key=lambda t: abs(t.mid_instant - centre))
    best = dataclasses.replace(
        best, start_index=best.start_index + lower, end_index=best.end_index + lower
    )
    return EdgeMeasurement(transition=best, levels=(bottom, top))
