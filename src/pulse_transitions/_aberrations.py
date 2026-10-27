"""Overshoot/undershoot in the aberration regions (§5.3.5) and settling time (§5.3.7)."""
from __future__ import annotations

import math

import numpy as np

from ._reference import interpolate_instant
from ._types import Aberration, Polarity, Transition, TransitionMetrics

__all__ = ["metrics_for"]

_EMPTY = Aberration(math.nan, math.nan, math.nan)
_EMPTY_REGION = slice(0, 0)


def _region(x: np.ndarray, start: float, end: float) -> slice:
    """Indices of the samples with start <= x <= end."""
    return slice(int(np.searchsorted(x, start, side="left")), int(np.searchsorted(x, end, side="right")))


def _excursion(
    x: np.ndarray, y: np.ndarray, region: slice, level: float, direction: int, amplitude: float, tol: float
) -> Aberration:
    """The extreme of `direction * y` in `region` relative to `level`: how far it passes `level` in that
    direction, as a percent of |amplitude|, or 0 if it doesn't pass it (rev 7: every excursion counts)."""
    if region.stop <= region.start:
        return _EMPTY
    k = region.start + int(np.argmax(direction * y[region]))
    excess = direction * (y[k] - level)
    percent = 100.0 * excess / abs(amplitude) if excess > 0 else 0.0
    return Aberration(percent=percent, instant=float(x[k]), value=float(y[k]))


def _post_anchor(y: np.ndarray, tr: Transition, last: int, level: float, tol: float) -> int | None:
    """First index in end_index .. last that has reached the post-state level (overshoot counts), else
    the first that has reached its band; None if neither."""
    distance = int(tr.polarity) * (y[tr.end_index:last + 1] - level)
    for bound in (0.0, -tol):
        reached = np.flatnonzero(distance >= bound)
        if reached.size:
            return tr.end_index + int(reached[0])
    return None


def _pre_anchor(y: np.ndarray, tr: Transition, first: int, level: float, tol: float) -> int | None:
    """Last index in first .. start_index still at the pre-state level, else the last still in its band;
    None if neither."""
    distance = int(tr.polarity) * (y[first:tr.start_index + 1] - level)
    for bound in (0.0, tol):
        held = np.flatnonzero(distance <= bound)
        if held.size:
            return first + int(held[-1])
    return None


def _settling_time(
    x: np.ndarray, y: np.ndarray, transition: Transition, stop: int | None, final: float, tol: float
) -> float:
    """From the mid instant to the last entry into `final ± tol` in start_index .. stop (inclusive);
    NaN if not settled or `stop` is None."""
    if stop is None:
        return math.nan
    start = transition.start_index
    outside = np.flatnonzero(np.abs(y[start:stop + 1] - final) > tol)
    if outside.size == 0:
        return math.nan
    k = start + int(outside[-1])
    if k >= stop:
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

    Each aberration region is anchored in its state (§C14, rev 7): the post region starts at the first
    sample that has reached the post-state level (else its band), the pre region ends at the last sample
    still at the pre-state level (else in its band), and neither reaches past the neighbouring transitions' anchors. Settling is
    searched up to the next transition's pre anchor (§C16).

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
    count = len(transitions)
    states = [(low, high) if tr.polarity == Polarity.RISING else (high, low) for tr in transitions]
    pre_anchor = [
        _pre_anchor(y, tr, transitions[n - 1].end_index if n > 0 else 0, states[n][0], tol)
        for n, tr in enumerate(transitions)
    ]
    post_anchor = [
        _post_anchor(y, tr, transitions[n + 1].start_index if n + 1 < count else len(y) - 1, states[n][1], tol)
        for n, tr in enumerate(transitions)
    ]

    results = []
    for n, tr in enumerate(transitions):
        direction = int(tr.polarity)
        pre_level, post_level = states[n]
        span = aberration_factor * tr.duration
        q, p = pre_anchor[n], post_anchor[n]
        # The next transition's pre anchor bounds this one's post region and settling search.
        stop = pre_anchor[n + 1] if n + 1 < count else len(y) - 1
        previous_p = post_anchor[n - 1] if n > 0 else 0

        pre = _EMPTY_REGION if q is None or previous_p is None else _region(x, max(x[q] - span, x[previous_p]), x[q])
        post = _EMPTY_REGION if p is None or stop is None else _region(x, x[p], min(x[p] + span, x[stop]))

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
