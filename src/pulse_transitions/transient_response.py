"""
Transient response edge detection module.

"""
import logging
from collections.abc import Iterable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from . import impl
from .common import CrossingDetectionSettings
from .common import Edge
from .common import EdgeSign

NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log = logging.getLogger("pulse_transitions")

calculate_thresholds = impl._calculate_thresholds
detect_signal_levels = impl._detect_signal_levels
detect_thresholds = impl._detect_thresholds
detect_first_edge = impl._detect_first_edge


def detect_edges(x: NumberIterable, y: NumberIterable,
                 thresholds: Tuple[float,float]=(0.1, 0.9),
                 levels: Optional[Tuple[float, float]] = None,
                 *, bounds=None,
                 settings: Optional[CrossingDetectionSettings] = None, **kwargs) -> list[Edge]:
    """
    Takes a 2 level signal.
    Either receives or calculates the levels.
    Use a fractional threshold (10/90%, 20/80% etc) to find the crossings
    Find the midpoint crossings and split at 50% between them. If no crossing before or after then include all the rest of the signal.

    Args:
        x (array-like): Time or index array.
        y (array-like): Signal data.
        thresholds (tuple): Threshold values.
        bounds (tuple, optional): Time bounds to restrict analysis.
        settings (CrossingDetectionSettings): Detection configuration.

    Returns:
        list[Edge]: List of detected edges.
    """
    if levels is None:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = (low_level, high_level)

    absolute_thresholds = impl._calculate_thresholds(x, y, levels, thresholds)
    if not settings:
        settings = CrossingDetectionSettings()
    return impl._detect_edges(x=x, y=y, thresholds=absolute_thresholds, settings=settings)


def get_rising_edge(x: NumberIterable, y: NumberIterable,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:
    """
    Detect rising edge timing with interpolation.

    Returns:
        Edge or None
    """
    return impl._detect_edge_wrapper(sign=EdgeSign.rising,
                        x=x, y=y,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)

def get_falling_edge(x: NumberIterable, y: NumberIterable,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:
    """
    Detect falling edge timing with interpolation.

    Returns:
        Edge or None
    """
    return impl._detect_edge_wrapper(sign=EdgeSign.falling,
                        x=x, y=y,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)


def calculate_risetime(x: NumberIterable, y: NumberIterable,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[float]:
    """
    Detect rising edge timing with interpolation.

    Returns:
        Edge or None
    """
    edge = impl._detect_edge_wrapper(sign=EdgeSign.rising,
                        x=x, y=y,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)
    if edge:
        return edge.end - edge.start
    return None

def calculate_falltime(x: NumberIterable, y: NumberIterable,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[float]:
    """
    Detect falling edge timing with interpolation.

    Returns:
        Edge or None
    """
    edge = impl._detect_edge_wrapper(sign=EdgeSign.falling,
                        x=x, y=y,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)
    if edge:
        return edge.end - edge.start
    return None

def calculate_midcross(x: NumberIterable, y: NumberIterable,
             levels: Optional[Tuple[float,float]]=None,
             **kwargs) -> float:
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
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = (low_level, high_level)

    return impl._calculate_midcross(x=x, y=y, levels=levels)

def calculate_overshoot(y: NumberIterable,
              levels: Optional[Tuple[float, float]]=None,
              **kwargs) -> float:
    """
    Compute normalized overshoot fraction of a step response.

    Args:

        y (array-like): Signal data.
        levels (tuple, optional): Low/high state levels.

    Returns:
        float: Overshoot fraction.
    """

    if not levels:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = (low_level, high_level)

    return impl._calculate_overshoot(y=y, levels=levels)


def calculate_undershoot(
               y: NumberIterable,
               levels: Optional[Tuple[float, float]]=None,
               **kwargs) -> float:
    """
    Compute normalized undershoot fraction of a step response.

    Args:
        y (array-like): Signal data.
        levels (tuple, optional): Low/high state levels.

    Returns:
        float: Undershoot fraction.
    """

    if not levels:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = (low_level, high_level)

    return impl._calculate_undershoot(y=y, levels=levels)

def calculate_slew_rate(x: NumberIterable, y: NumberIterable,
             **kwargs):
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

def calculate_settling_time(x: NumberIterable,
                 y: NumberIterable,
                 settling_time_fraction: float = 0.02,
                 settling_time_margin: Optional[float] = None,
                 levels: Optional[Tuple[float, float]] = None,
                 **kwargs):
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
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = (low_level, high_level)

    return impl._calculate_settling_time(y=y, x=x,
                        settling_time_margin=settling_time_margin,
                        settling_time_fraction=settling_time_fraction,
                        levels=levels)


def get_edge_metrics(x: NumberIterable,
                     y: NumberIterable,
                     settling_time_fraction: float = 0.02,
                     levels: Optional[Tuple[float, float]] = None,
                     thresholds: Tuple[float,float]=(0.1, 0.9), **kwargs) -> Dict[str, float]:
    if levels is None:
        low_level, high_level, *_ = impl.detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = (low_level, high_level)

    absolute_thresholds = impl._calculate_thresholds(x, y, levels, thresholds=thresholds)

    return {
        "fractional_thresholds": thresholds,
        "absolute_thresholds": absolute_thresholds,
        "levels": levels,
        "midcross": calculate_midcross(x, y, levels=levels),
        "risetime": calculate_risetime(x, y, thresholds=thresholds, levels=levels),
        "falltime": calculate_falltime(x, y, thresholds=thresholds, levels=levels),
        "slewrate": calculate_slew_rate(x, y),
        "overshoot": calculate_overshoot(y, levels=levels),
        "undershoot": calculate_undershoot(y, levels=levels),
        "settling_time": calculate_settling_time(x, y, levels=levels, settling_time_fraction=settling_time_fraction),
    }
