"""MATLAB-named facade over the public API."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import _api
from ._types import Aberration, Polarity, Pulse, Transition
from ._validate import check_choice, check_reference

__all__ = [
    "dutycycle", "falltime", "midcross", "overshoot", "pulseperiod", "pulsesep", "pulsewidth",
    "risetime", "settlingtime", "slewrate", "statelevels", "undershoot",
]


def _array(values: ArrayLike) -> np.ndarray:
    """Float array of `values`; empty when there is nothing to return."""
    return np.asarray(values, dtype=float)


def _triple(reference: Sequence[float]) -> tuple[float, float, float]:
    """MATLAB (lower, upper) percentages, checked, as the (lower, 50, upper) triple of the API."""
    lower, upper = check_reference(reference, pair=True)
    return (lower, 50.0, upper)


def _levels(
    y: ArrayLike, t: ArrayLike | None, fs: float | None, levels: Sequence[float] | None
) -> Sequence[float]:
    """`levels` as given, or the low and high states detected by `state_levels`."""
    if levels is not None:
        return levels
    return _api.state_levels(y, x=t, fs=fs).levels


def _pulses(
    y: ArrayLike,
    t: ArrayLike | None,
    fs: float | None,
    levels: Sequence[float] | None,
    polarity: str,
    mid_method: str,
) -> tuple[Pulse, ...]:
    """Pulses of MATLAB `polarity` ("positive" is rising, "negative" falling) from the API."""
    choice = check_choice("polarity", polarity, ("positive", "negative"))
    direction = Polarity.RISING if choice == "positive" else Polarity.FALLING
    return _api.pulses(y, x=t, fs=fs, levels=levels, polarity=direction, mid_method=mid_method)


def _edge_times(
    y: ArrayLike,
    t: ArrayLike | None,
    fs: float | None,
    levels: Sequence[float] | None,
    reference: Sequence[float],
    mid_method: str,
    direction: Polarity,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Durations and reference instants of the `direction` transitions, with the reference levels."""
    triple = _triple(reference)
    state = _levels(y, t, fs, levels)
    found = _api.transitions(
        y, x=t, fs=fs, levels=state, reference=triple, polarity=direction, mid_method=mid_method,
    )
    lower, _, upper = _api.reference_levels(state, triple)
    return (
        _array([edge.duration for edge in found]),
        _array([edge.lower_instant for edge in found]),
        _array([edge.upper_instant for edge in found]),
        float(lower),
        float(upper),
    )


def _aberration_arrays(aberrations: Sequence[Aberration]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(percent, value, instant) arrays of a sequence of aberrations."""
    return (
        _array([item.percent for item in aberrations]),
        _array([item.value for item in aberrations]),
        _array([item.instant for item in aberrations]),
    )


def _cycle_timing(complete: Sequence[Pulse]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(initcross, finalcross, nextcross) of pulses with a period, nextcross = initcross + period."""
    initcross = _array([item.first.mid_instant for item in complete])
    finalcross = _array([item.second.mid_instant for item in complete])
    periods = _array([item.period for item in complete])
    return initcross, finalcross, initcross + periods


def statelevels(y: ArrayLike, nbins: int = 100, method: str = "mode") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [levels, histogram, bin centres] = statelevels(y, nbins, method)."""
    state = _api.state_levels(y, bins=nbins, method=method)
    levels = _array([state.low, state.high])
    if state.histogram is None or state.bin_edges is None:
        return levels, _array([]), _array([])
    edges = state.bin_edges
    return levels, _array(state.histogram), _array((edges[:-1] + edges[1:]) / 2)


def midcross(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    mid_method: str = "median",
) -> np.ndarray:
    """MATLAB midcross(y, t): mid-level crossing instants of every transition."""
    found = _api.transitions(y, x=t, fs=fs, levels=levels, mid_method=mid_method)
    return _array([edge.mid_instant for edge in found])


def risetime(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = (10, 90),
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """MATLAB [dur, tlower, tupper, ll, ul] = risetime(y, t): rising transitions, with reference levels."""
    return _edge_times(y, t, fs, levels, reference, mid_method, Polarity.RISING)


def falltime(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = (10, 90),
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """MATLAB [dur, tlower, tupper, ll, ul] = falltime(y, t): falling transitions, with reference levels."""
    return _edge_times(y, t, fs, levels, reference, mid_method, Polarity.FALLING)


def slewrate(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = (10, 90),
    mid_method: str = "median",
) -> np.ndarray:
    """MATLAB slewrate(y, t): signed slew rate of every transition."""
    found = _api.transitions(
        y, x=t, fs=fs, levels=levels, reference=_triple(reference), mid_method=mid_method,
    )
    return _array([edge.slew_rate for edge in found])


def overshoot(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [pct, val, t] = overshoot(y, t): post-transition overshoot of every transition."""
    metrics = _api.transition_metrics(y, x=t, fs=fs, levels=levels, mid_method=mid_method)
    return _aberration_arrays([item.post_overshoot for item in metrics])


def undershoot(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [pct, val, t] = undershoot(y, t): post-transition undershoot of every transition."""
    metrics = _api.transition_metrics(y, x=t, fs=fs, levels=levels, mid_method=mid_method)
    return _aberration_arrays([item.post_undershoot for item in metrics])


def settlingtime(
    y: ArrayLike,
    d: float,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    mid_method: str = "median",
) -> np.ndarray:
    """MATLAB settlingtime(y, d, t): settling time of every transition to within d percent."""
    metrics = _api.transition_metrics(
        y, x=t, fs=fs, levels=levels, state_tolerance=d, mid_method=mid_method,
    )
    return _array([item.settling_time for item in metrics])


def pulsewidth(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    polarity: str = "positive",
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [width, t1, t2] = pulsewidth(y, t, polarity): width of every pulse and its edge mid instants."""
    found = _pulses(y, t, fs, levels, polarity, mid_method)
    return (
        _array([item.width for item in found]),
        _array([item.first.mid_instant for item in found]),
        _array([item.second.mid_instant for item in found]),
    )


def pulseperiod(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    polarity: str = "positive",
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [period, initcross, finalcross, nextcross] = pulseperiod(y, t, polarity), complete periods only."""
    found = _pulses(y, t, fs, levels, polarity, mid_method)
    complete = [item for item in found if math.isfinite(item.period)]
    initcross, finalcross, nextcross = _cycle_timing(complete)
    return _array([item.period for item in complete]), initcross, finalcross, nextcross


def pulsesep(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    polarity: str = "positive",
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [sep, finalcross, nextcross] = pulsesep(y, t, polarity), complete periods only."""
    found = _pulses(y, t, fs, levels, polarity, mid_method)
    complete = [item for item in found if math.isfinite(item.separation)]
    separations = _array([item.separation for item in complete])
    finalcross = _array([item.second.mid_instant for item in complete])
    return separations, finalcross, finalcross + separations


def dutycycle(
    y: ArrayLike,
    t: ArrayLike | None = None,
    *,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    polarity: str = "positive",
    mid_method: str = "median",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MATLAB [duty, initcross, finalcross, nextcross] = dutycycle(y, t, polarity), complete periods only."""
    found = _pulses(y, t, fs, levels, polarity, mid_method)
    complete = [item for item in found if math.isfinite(item.period)]
    initcross, finalcross, nextcross = _cycle_timing(complete)
    return _array([item.duty_cycle for item in complete]), initcross, finalcross, nextcross
