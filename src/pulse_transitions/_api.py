"""Public functions: validate once, compose the internal modules."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from . import _reference
from ._aberrations import metrics_for
from ._levels import detect_levels
from ._multilevel import locate_and_measure
from ._plateau import flatness_of
from ._pulses import make_pulses
from ._transitions import find_transitions
from ._types import (
    EdgeMeasurement,
    EdgeNotFoundError,
    InputError,
    Polarity,
    Pulse,
    StateLevels,
    Transition,
    TransitionMetrics,
)
from ._validate import as_waveform, check_choice, check_levels, check_reference

__all__ = [
    "flatness",
    "measure_edge",
    "pulses",
    "reference_levels",
    "state_levels",
    "step_metrics",
    "transition_metrics",
    "transitions",
]

_DEFAULT_REFERENCE = (10.0, 50.0, 90.0)
_LEVEL_METHODS = ("mode", "mean", "peak", "endpoints")
_MID_METHODS = ("median", "first", "fit")


def _levels_for(y: np.ndarray, levels: Sequence[float] | None) -> tuple[float, float]:
    """`levels` checked, or detected from `y` with the histogram mode."""
    if levels is None:
        return detect_levels(y).levels
    return check_levels(levels)


def _find(
    y: ArrayLike,
    x: ArrayLike | None,
    fs: float | None,
    levels: Sequence[float] | None,
    reference: Sequence[float],
    bounds: Sequence[float] | None,
    mid_method: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], tuple[Transition, ...]]:
    """Validated (x, y), the levels, and all transitions."""
    ref = check_reference(reference)
    check_choice("mid_method", mid_method, _MID_METHODS)
    xs, ys = as_waveform(y, x, fs, bounds)
    lv = _levels_for(ys, levels)
    found = find_transitions(
        xs, ys, _reference.reference_levels(lv, ref), mid_method, lv
    )
    return xs, ys, lv, found


def _check_metric_options(aberration_factor: float, state_tolerance: float) -> None:
    if not (math.isfinite(aberration_factor) and aberration_factor > 0):
        msg = f"aberration_factor must be > 0, not {aberration_factor!r}"
        raise InputError(msg)
    if not (math.isfinite(state_tolerance) and 0 < state_tolerance < 50):
        msg = (
            f"state_tolerance must be between 0 and 50 percent, not {state_tolerance!r}"
        )
        raise InputError(msg)


def reference_levels(
    levels: Sequence[float], reference: Sequence[float] = _DEFAULT_REFERENCE
) -> tuple[float, float, float]:
    """Absolute (lower, mid, upper) reference levels: low + (high - low) * p / 100 (§5.3.2).

    Raises:
        InputError: Bad `levels` (§E4) or `reference` (§E5).
    """
    return _reference.reference_levels(check_levels(levels), check_reference(reference))


def state_levels(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    fs: float | None = None,
    bounds: Sequence[float] | None = None,
    method: str = "mode",
    bins: int = 100,
    split: Sequence[float] = (0.5, 0.5),
    endpoint_samples: int = 100,
) -> StateLevels:
    """Low and high state levels by histogram mode or mean (§5.2.1), peak or endpoints (§5.2.2).

    Raises:
        InputError: Bad waveform or options (§E1–§E3, §E6, §E8, §E9).
        LevelError: The data is constant (§E7).
    """
    check_choice("method", method, _LEVEL_METHODS)
    if not (isinstance(bins, int) and bins >= 2):
        msg = f"bins must be an integer >= 2, not {bins!r}"
        raise InputError(msg)
    if not (len(split) == 2 and 0 < split[0] <= split[1] < 1):
        msg = f"split must be (f1, f2) with 0 < f1 <= f2 < 1, not {split!r}"
        raise InputError(msg)
    _, ys = as_waveform(y, x, fs, bounds)
    if method == "endpoints" and not (
        isinstance(endpoint_samples, int)
        and 1 <= endpoint_samples
        and 2 * endpoint_samples <= len(ys)
    ):
        msg = f"endpoint_samples must be between 1 and {len(ys) // 2} for {len(ys)} samples, not {endpoint_samples!r}"
        raise InputError(msg)
    return detect_levels(
        ys, method, bins, (float(split[0]), float(split[1])), endpoint_samples
    )


def transitions(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = _DEFAULT_REFERENCE,
    bounds: Sequence[float] | None = None,
    polarity: Polarity | None = None,
    mid_method: str = "median",
) -> tuple[Transition, ...]:
    """Every transition of the waveform, in time order (`polarity` keeps one direction).

    Raises:
        InputError: Bad arguments (§E1–§E6, §E15).
        LevelError: `levels` is None and the data is constant (§E7).
    """
    if polarity is not None and not isinstance(polarity, Polarity):
        msg = f"polarity must be a Polarity or None, not {polarity!r}"
        raise InputError(msg)
    found = _find(y, x, fs, levels, reference, bounds, mid_method)[3]
    if polarity is None:
        return found
    return tuple(t for t in found if t.polarity == polarity)


def transition_metrics(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = _DEFAULT_REFERENCE,
    bounds: Sequence[float] | None = None,
    aberration_factor: float = 3.0,
    state_tolerance: float = 2.0,
    mid_method: str = "median",
) -> tuple[TransitionMetrics, ...]:
    """Overshoot, undershoot (§5.3.5) and settling time (§5.3.7) of every transition.

    Raises:
        InputError: Bad arguments (§E1–§E6, §E15).
        LevelError: `levels` is None and the data is constant (§E7).
    """
    _check_metric_options(aberration_factor, state_tolerance)
    xs, ys, lv, found = _find(y, x, fs, levels, reference, bounds, mid_method)
    return metrics_for(xs, ys, found, lv, aberration_factor, state_tolerance)


def step_metrics(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = _DEFAULT_REFERENCE,
    bounds: Sequence[float] | None = None,
    aberration_factor: float = 3.0,
    state_tolerance: float = 2.0,
    mid_method: str = "median",
) -> TransitionMetrics:
    """Metrics of the single transition of a step response.

    Raises:
        EdgeNotFoundError: Not exactly one transition in the analysed data (§E10).
        InputError: Bad arguments (§E1–§E6, §E15).
        LevelError: `levels` is None and the data is constant (§E7).
    """
    found = transition_metrics(
        y,
        x=x,
        fs=fs,
        levels=levels,
        reference=reference,
        bounds=bounds,
        aberration_factor=aberration_factor,
        state_tolerance=state_tolerance,
        mid_method=mid_method,
    )
    if len(found) != 1:
        msg = (
            f"A step has exactly one transition; found {len(found)}. "
            "Use bounds= to select the step, or transition_metrics() for all of them."
        )
        raise EdgeNotFoundError(msg)
    return found[0]


def pulses(
    y: ArrayLike,
    *,
    x: ArrayLike | None = None,
    fs: float | None = None,
    levels: Sequence[float] | None = None,
    reference: Sequence[float] = _DEFAULT_REFERENCE,
    bounds: Sequence[float] | None = None,
    polarity: Polarity = Polarity.RISING,
    mid_method: str = "median",
) -> tuple[Pulse, ...]:
    """Width, period, separation and duty cycle of every pulse of `polarity` (§5.4).

    Raises:
        InputError: Bad arguments (§E1–§E6, §E15).
        LevelError: `levels` is None and the data is constant (§E7).
    """
    if not isinstance(polarity, Polarity):
        msg = f"polarity must be a Polarity, not {polarity!r}"
        raise InputError(msg)
    return make_pulses(
        _find(y, x, fs, levels, reference, bounds, mid_method)[3], polarity
    )


def measure_edge(
    y: ArrayLike,
    *,
    around: float,
    x: ArrayLike | None = None,
    fs: float | None = None,
    max_distance: float = math.inf,
    reference: Sequence[float] = _DEFAULT_REFERENCE,
    levels: Sequence[float] | None = None,
    mid_method: str = "median",
) -> EdgeMeasurement:
    """The edge nearest `around` in a multi-level (TDR) trace, between the flat levels either side.

    Raises:
        EdgeNotFoundError: No edge near `around`, no flat level beside it, or the reference
            levels aren't crossed (§E11–§E13).
        InputError: Bad arguments (§E1–§E5, §E15).
    """
    ref = check_reference(reference)
    check_choice("mid_method", mid_method, _MID_METHODS)
    lv = None if levels is None else check_levels(levels)
    xs, ys = as_waveform(y, x, fs)
    return locate_and_measure(
        xs, ys, float(around), float(max_distance), ref, lv, mid_method
    )


def flatness(
    y: ArrayLike,
    *,
    bounds: Sequence[float],
    x: ArrayLike | None = None,
    fs: float | None = None,
    normalize: bool = True,
) -> float:
    """Std-dev of `y` within `bounds`, normalised by the record's final level; NaN when undefined.

    Raises:
        InputError: Bad waveform or bounds (§E1–§E3, §E6).
    """
    xs, ys = as_waveform(y, x, fs)
    start, end = (float(b) for b in bounds)
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        msg = f"bounds must be two finite numbers with start < end, not {bounds!r}"
        raise InputError(msg)
    return flatness_of(xs, ys, (start, end), normalize)
