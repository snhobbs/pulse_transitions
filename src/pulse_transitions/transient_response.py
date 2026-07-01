"""
Transient response edge detection module.
"""

import logging
from collections.abc import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from . import impl
from .common import CrossingDetectionSettings
from .common import Edge
from .common import EdgeSign, EdgeMetrics

Number = Union[float, int]
NumberIterable = Iterable[Number]
log = logging.getLogger("pulse_transitions")

calculate_thresholds = impl._calculate_thresholds
detect_signal_levels = impl._detect_signal_levels
detect_thresholds = impl._detect_thresholds
detect_first_edge = impl._detect_first_edge


def detect_edges(
    x: NumberIterable,
    y: NumberIterable,
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9),
    levels: Optional[Tuple[float, float]] = None,
    *,
    bounds=Optional[None],
    settings: Optional[CrossingDetectionSettings] = None,
    **kwargs,
) -> list[Edge]:
    """
    Takes a 2 level signal.
    Either receives or calculates the levels.
    Use a fractional threshold (10/90%, 20/80% etc) to find the crossings
    Find the midpoint crossings and split at 50% between them.
    If no crossing before or after then include all the rest of the signal.

    Args:
        x (array-like): Time or index array.
        y (array-like): Signal data.
        fractional_thresholds (tuple): Threshold values.
        bounds (tuple, optional): Time bounds to restrict analysis.
        settings (CrossingDetectionSettings): Detection configuration.

    Returns:
        list[Edge]: List of detected edges.
    """
    if levels is None:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(
            None, y=y, **kwargs
        )
        levels = (low_level, high_level)

    absolute_thresholds = impl._calculate_thresholds(
        x, y, levels, fractional_thresholds=fractional_thresholds
    )
    if not settings:
        settings = CrossingDetectionSettings()
    return impl._detect_edges(
        x=x, y=y, thresholds=absolute_thresholds, settings=settings
    )


def get_rising_edge(
    x: NumberIterable,
    y: NumberIterable,
    levels: Optional[Tuple[float, float]] = None,
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9),
    settings: Optional[CrossingDetectionSettings] = None,
    **kwargs,
) -> Optional[Edge]:
    """
    Detect rising edge timing with interpolation.

    Returns:
        Edge or None
    """
    return impl._detect_edge_wrapper(
        sign=EdgeSign.rising,
        x=x,
        y=y,
        levels=levels,
        fractional_thresholds=fractional_thresholds,
        settings=settings,
        **kwargs,
    )


def get_falling_edge(
    x: NumberIterable,
    y: NumberIterable,
    levels: Optional[Tuple[float, float]] = None,
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9),
    settings: Optional[CrossingDetectionSettings] = None,
    **kwargs,
) -> Optional[Edge]:
    """
    Detect falling edge timing with interpolation.

    Returns:
        Edge or None
    """
    return impl._detect_edge_wrapper(
        sign=EdgeSign.falling,
        x=x,
        y=y,
        levels=levels,
        fractional_thresholds=fractional_thresholds,
        settings=settings,
        **kwargs,
    )


def calculate_risetime(
    x: NumberIterable,
    y: NumberIterable,
    levels: Optional[Tuple[float, float]] = None,
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9),
    settings: Optional[CrossingDetectionSettings] = None,
    **kwargs,
) -> Optional[Tuple[float, Edge]]:
    """
    Detect rising edge timing with interpolation.

    Returns:
        Edge or None
    """
    edge = impl._detect_edge_wrapper(
        sign=EdgeSign.rising,
        x=x,
        y=y,
        levels=levels,
        fractional_thresholds=fractional_thresholds,
        settings=settings,
        **kwargs,
    )
    if edge:
        return edge.end - edge.start, edge
    return None


def calculate_falltime(
    x: NumberIterable,
    y: NumberIterable,
    levels: Optional[Tuple[float, float]] = None,
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9),
    settings: Optional[CrossingDetectionSettings] = None,
    **kwargs,
) -> Optional[Tuple[float, Edge]]:
    """
    Detect falling edge timing with interpolation.

    Returns:
        Edge or None
    """
    edge = impl._detect_edge_wrapper(
        sign=EdgeSign.falling,
        x=x,
        y=y,
        levels=levels,
        fractional_thresholds=fractional_thresholds,
        settings=settings,
        **kwargs,
    )
    if edge:
        return edge.end - edge.start, edge
    return None


def calculate_midcross(
    x: NumberIterable,
    y: NumberIterable,
    levels: Optional[Tuple[float, float]] = None,
    **kwargs,
) -> float:
    """
    Find mid-level crossing time of a bilevel signal.

    Args:
        x (array-like): Time or index vector.
        y (array-like): Signal data.
        levels (tuple, optional): Reference levels.

    Returns:
        float: Time of mid-reference crossing.
    """

    if not levels:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(
            None, y=y, **kwargs
        )
        levels = (low_level, high_level)

    return impl._calculate_midcross(x=x, y=y, levels=levels)


def calculate_overshoot(
    y: NumberIterable, levels: Optional[Tuple[float, float]] = None, **kwargs
) -> Tuple[int, float]:
    """
    Compute normalized overshoot fraction of a step response.

    Args:

        y (array-like): Signal data.
        levels (tuple, optional): Low/high state levels.

    Returns:
        tuple[int, float]: Index, Overshoot fraction.
    """

    if not levels:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(
            None, y=y, **kwargs
        )
        levels = (low_level, high_level)

    return impl._calculate_overshoot(y=y, levels=levels)


def calculate_undershoot(
    y: NumberIterable, levels: Optional[Tuple[float, float]] = None, **kwargs
) -> Tuple[int, float]:
    """
    Compute normalized undershoot fraction of a step response.

    Args:
        y (array-like): Signal data.
        levels (tuple, optional): Low/high state levels.

    Returns:
        tuple[int, float]: Index, Undershoot fraction.
    """

    if not levels:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(
            None, y=y, **kwargs
        )
        levels = (low_level, high_level)

    return impl._calculate_undershoot(y=y, levels=levels)


def calculate_slew_rate(x: NumberIterable, y: NumberIterable, **kwargs):
    """
    Calculate the slew rate of a signal y with respect to x.

    Args:
        x (array-like): Signal data.
        fs (float): Sampling rate.
        t (array-like, optional): Time array.

    Returns:
        float: Slew rate (units of y per unit of x)
    """
    return impl._calculate_slew_rate(x=x, y=y)


def calculate_settling_time(
    x: NumberIterable,
    y: NumberIterable,
    settling_time_fraction: float = 0.02,
    settling_time_margin: float = 0,
    levels: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    """
    Calculate the settling time of a step response signal.

    Args:
        x (array-like): Time array.
        y (array-like): Signal data.
        d (float): Fractional tolerance band for settling (default 0.02 = 2%).
        levels (tuple, optional): Low/high reference levels.
        settling_time_margin (float, optional): Additional time margin added after last deviation.

    Returns:
        float: Settling time in units of t (or samples if t is None).
    """

    if levels is None:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(
            None, y=y, **kwargs
        )
        levels = (low_level, high_level)

    return impl._calculate_settling_time(
        y=y,
        x=x,
        settling_time_margin=settling_time_margin,
        settling_time_fraction=settling_time_fraction,
        levels=levels,
    )


def calculate_flatness(
    x: NumberIterable,
    y: NumberIterable,
    bounds: Tuple[float, float],
    normalize: bool = True,
) -> Optional[float]:
    """
    Measure signal flatness (normalised std-dev) within a time window.

    Useful for quantifying ripple on a TDR plateau — a perfectly flat plateau
    returns 0; impedance discontinuities that haven't fully settled return a
    larger value.

    Args:
        x: Time array (any consistent units).
        y: Signal array.
        bounds: (t_start, t_end) window to measure.
        normalize: If True, divide std-dev by |mean of the last 10 samples|
            so the result is dimensionless (fraction of final level).
            If False, return raw std-dev in signal units.

    Returns:
        Normalised std-dev within the window, or None if the window contains
        fewer than 5 samples or the final level is too small to normalise.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x >= bounds[0]) & (x <= bounds[1])
    if mask.sum() < 5:
        return None
    std = float(np.std(y[mask]))
    if not normalize:
        return std
    v_final = float(np.abs(np.mean(y[-10:])))
    if v_final < 1e-12:
        return None
    return std / v_final


def get_edge_metrics(
    x: NumberIterable,
    y: NumberIterable,
    settling_time_fraction: float = 0.02,
    levels: Optional[Tuple[float, float]] = None,
    fractional_thresholds: Tuple[float, float] = (0.1, 0.9),
    **kwargs,
) -> EdgeMetrics:
    if levels is None:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(
            None, y=y, **kwargs
        )
        levels = (low_level, high_level)

    thresholds = impl._calculate_thresholds(
        x, y, levels, fractional_thresholds=fractional_thresholds
    )

    return EdgeMetrics(
        fractional_thresholds=fractional_thresholds,
        settling_time_fraction=settling_time_fraction,
        thresholds=thresholds,
        levels=levels,
        midcross=calculate_midcross(x, y, levels=levels),
        risetime=calculate_risetime(
            x, y, fractional_thresholds=fractional_thresholds, levels=levels
        ),
        falltime=calculate_falltime(
            x, y, fractional_thresholds=fractional_thresholds, levels=levels
        ),
        slewrate=calculate_slew_rate(x, y),
        overshoot=calculate_overshoot(y, levels=levels),
        undershoot=calculate_undershoot(y, levels=levels),
        settling_time=calculate_settling_time(
            x, y, levels=levels, settling_time_fraction=settling_time_fraction
        ),
    )
