"""Vectorised hysteresis transition finder, shared by every feature."""
from __future__ import annotations

import numpy as np

from ._reference import interpolate_instant
from ._types import Polarity, Transition

__all__ = ["find_transitions"]


def _crossings(above: np.ndarray) -> np.ndarray:
    """Segments k (k -> k+1) where `above` changes."""
    return np.flatnonzero(above[1:] != above[:-1])


def _median_instant(
    x: np.ndarray, y: np.ndarray, crossing: np.ndarray, lo: np.ndarray, hi: np.ndarray, level: float, pick_median: bool
) -> np.ndarray:
    """Interpolated instant of the median (or first) crossing in segments lo .. hi-1 of each window."""
    first = np.searchsorted(crossing, lo, side="left")
    count = np.searchsorted(crossing, hi, side="left") - first
    k = crossing[first + (count // 2 if pick_median else 0)]
    return interpolate_instant(x[k], y[k], x[k + 1], y[k + 1], level)


def _fitted_instants(
    x: np.ndarray, y: np.ndarray, i: np.ndarray, j: np.ndarray, rising: np.ndarray, level: float,
    start: np.ndarray, end: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mid-level instant of a least-squares line through samples i .. j of each transition, clamped
    to [start, end]; NaN where the slope isn't finite or has the wrong sign. Also returns the mask of
    usable fits."""
    instant = np.full(len(i), np.nan)
    for n in range(len(i)):
        xs, ys = x[i[n]:j[n] + 1], y[i[n]:j[n] + 1]
        x_mean, y_mean = xs.mean(), ys.mean()
        xc = xs - x_mean
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = float(xc @ (ys - y_mean)) / float(xc @ xc)
        if np.isfinite(slope) and (slope > 0 if rising[n] else slope < 0):
            instant[n] = min(max(x_mean + (level - y_mean) / slope, start[n]), end[n])
    return instant, np.isfinite(instant)


def _last_at_or_before(mask: np.ndarray) -> np.ndarray:
    """For each index, the last index at or before it where `mask` holds; -1 if none."""
    return np.maximum.accumulate(np.where(mask, np.arange(len(mask)), -1))


def _first_at_or_after(mask: np.ndarray) -> np.ndarray:
    """For each index, the first index at or after it where `mask` holds; len(mask) if none."""
    n = len(mask)
    return np.minimum.accumulate(np.where(mask, np.arange(n), n)[::-1])[::-1]


def find_transitions(
    x: np.ndarray,
    y: np.ndarray,
    ref_levels: tuple[float, float, float],
    mid_method: str = "median",
    state_levels: tuple[float, float] | None = None,
) -> tuple[Transition, ...]:
    """Every transition between the low and high states, in time order.

    A sample is in the low state at or below the lower reference level and in the high
    state at or above the upper one. A transition runs from the last sample of one state
    (`start_index`) to the first sample of the other (`end_index`).

    The lower and upper instants are the median crossings of their levels within windows
    that reach back to the last sample at the pre-transition state level and forward to the
    first at the post-transition state level (§C7), so noise near a reference level doesn't
    bias them. Without `state_levels` the windows are just `start_index .. end_index`.

    Args:
        x: Strictly increasing sample instants.
        y: Samples, same length as `x`.
        ref_levels: Absolute (lower, mid, upper) reference levels.
        mid_method: "median" or "first" mid-level crossing within each transition, or "fit" for the
            mid-level instant of a least-squares line through it (median where the fit is unusable).
        state_levels: (low, high) state levels guarding the lower/upper windows.

    Returns:
        The transitions, alternating in polarity; () when there are none.
    """
    lower, mid, upper = ref_levels
    state = np.zeros(len(y), dtype=np.int8)
    state[y <= lower] = -1
    state[y >= upper] = 1

    known = np.flatnonzero(state)
    if known.size < 2:
        return ()
    known_state = state[known]
    change = np.flatnonzero(known_state[1:] != known_state[:-1])
    if change.size == 0:
        return ()

    i = known[change]
    j = known[change + 1]
    polarity = known_state[change + 1]
    rising = polarity > 0

    # Windows for the start-side level (a .. j) and the end-side level (i .. b): out to the
    # nearest sample at the pre/post state level, never past the neighbouring transitions.
    a, b = i, j
    if state_levels is not None:
        low, high = state_levels
        at_low, at_high = y <= low, y >= high
        reach_back = np.where(rising, _last_at_or_before(at_low)[i], _last_at_or_before(at_high)[i])
        previous_end = np.r_[0, j[:-1]]
        a = np.where(reach_back >= previous_end, reach_back, i)
        reach_on = np.where(rising, _first_at_or_after(at_high)[j], _first_at_or_after(at_low)[j])
        next_start = np.r_[i[1:], len(y) - 1]
        b = np.where(reach_on <= next_start, reach_on, j)

    # A segment crosses the lower level where `y > lower` changes and the upper level where
    # `y >= upper` changes, matching the state definitions; each window holds an odd count.
    lower_instant = _median_instant(
        x, y, _crossings(y > lower), np.where(rising, a, i), np.where(rising, j, b), lower, True
    )
    upper_instant = _median_instant(
        x, y, _crossings(y >= upper), np.where(rising, i, a), np.where(rising, b, j), upper, True
    )
    mid_instant = _median_instant(x, y, _crossings(y >= mid), i, j, mid, mid_method != "first")
    methods = np.full(len(i), mid_method, dtype=object)
    if mid_method == "fit":
        # Least-squares line through i .. j (rev 5); the median crossing stands in where the fit can't be used.
        start = np.minimum(lower_instant, upper_instant)
        end = np.maximum(lower_instant, upper_instant)
        fitted, usable = _fitted_instants(x, y, i, j, rising, mid, start, end)
        mid_instant = np.where(usable, fitted, mid_instant)
        methods[~usable] = "median"

    return tuple(
        Transition(
            polarity=Polarity(int(polarity[n])),
            lower_instant=float(lower_instant[n]),
            mid_instant=float(mid_instant[n]),
            upper_instant=float(upper_instant[n]),
            lower_level=float(lower),
            mid_level=float(mid),
            upper_level=float(upper),
            start_index=int(i[n]),
            end_index=int(j[n]),
            mid_method=str(methods[n]),
        )
        for n in range(len(i))
    )
