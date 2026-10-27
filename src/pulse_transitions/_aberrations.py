"""Overshoot/undershoot in the aberration regions (§5.3.5) and settling time (§5.3.7)."""
from __future__ import annotations

import math

import numpy as np

from ._reference import interpolate_instant
from ._types import Aberration, Polarity, Transition, TransitionMetrics

__all__ = ["metrics_for"]

_EMPTY = Aberration(math.nan, math.nan, math.nan)


def _region(x: np.ndarray, start: float, end: float) -> slice:
    """Indices of the samples with start <= x <= end."""
    return slice(int(np.searchsorted(x, start, side="left")), int(np.searchsorted(x, end, side="right")))


def _excursion(
    x: np.ndarray, y: np.ndarray, region: slice, level: float, direction: int, amplitude: float, tol: float
) -> Aberration:
    """The extreme of `direction * y` in `region` relative to `level`: percent of |amplitude| when it
    passes `level` by more than `tol` in that direction, else 0."""
    if region.stop <= region.start:
        return _EMPTY
    k = region.start + int(np.argmax(direction * y[region]))
    excess = direction * (y[k] - level)
    percent = 100.0 * abs(y[k] - level) / abs(amplitude) if excess > tol else 0.0
    return Aberration(percent=percent, instant=float(x[k]), value=float(y[k]))


def _settling_time(
    x: np.ndarray, y: np.ndarray, transition: Transition, stop: int, final: float, tol: float
) -> float:
    """From the mid instant to the last entry into `final ± tol` before `stop`; NaN if not settled."""
    start = transition.start_index
    outside = np.flatnonzero(np.abs(y[start:stop] - final) > tol)
    if outside.size == 0:
        return math.nan
    k = start + int(outside[-1])
    if k >= stop - 1:
        return math.nan
    edge = final + tol if y[k] > final else final - tol
    return float(interpolate_instant(x[k], y[k], x[k + 1], y[k + 1], edge)) - transition.mid_instant


def metrics_for(
    x: np.ndarray,
    y: np.ndarray,
    transitions: tuple[Transition, ...],
    levels: tuple[float, float],
    aberration_factor: float = 3.0,
    state_tolerance: float = 2.0,
) -> tuple[TransitionMetrics, ...]:
    """Overshoot, undershoot and settling time of each transition.

    Args:
        x: Strictly increasing sample instants.
        y: Samples.
        transitions: The waveform's transitions, in time order.
        levels: (low, high) state levels.
        aberration_factor: Region length in transition durations.
        state_tolerance: State boundary and settling band, percent of the amplitude.

    Returns:
        One TransitionMetrics per transition; () when there are none.
    """
    low, high = levels
    amplitude = high - low
    tol = state_tolerance / 100.0 * abs(amplitude)
    results = []
    for n, tr in enumerate(transitions):
        previous = transitions[n - 1] if n > 0 else None
        following = transitions[n + 1] if n + 1 < len(transitions) else None
        direction = int(tr.polarity)
        pre_level, post_level = (low, high) if tr.polarity == Polarity.RISING else (high, low)
        span = aberration_factor * tr.duration

        pre = _region(x, max(tr.start - span, previous.end if previous else x[0]), tr.start)
        post = _region(x, tr.end, min(tr.end + span, following.start if following else x[-1]))
        stop = following.start_index + 1 if following else len(y)

        results.append(
            TransitionMetrics(
                transition=tr,
                levels=(float(low), float(high)),
                pre_overshoot=_excursion(x, y, pre, pre_level, direction, amplitude, tol),
                pre_undershoot=_excursion(x, y, pre, pre_level, -direction, amplitude, tol),
                post_overshoot=_excursion(x, y, post, post_level, direction, amplitude, tol),
                post_undershoot=_excursion(x, y, post, post_level, -direction, amplitude, tol),
                settling_time=_settling_time(x, y, tr, stop, post_level, tol),
            )
        )
    return tuple(results)
