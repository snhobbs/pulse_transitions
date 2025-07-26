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
from .common import EdgeSign

NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log = logging.getLogger("pulse_transitions")


#=========================
# Matlab naming starts
#=========================


def statelevels(A: NumberIterable, nbins: int = 100, method="mode",
                bounds: Tuple[float, float] = None, **kwargs):
    """
    Estimate low/high levels for a bilevel waveform using histogram analysis.

    Args:
        A (array-like): Signal data.
        nbins (int): Number of histogram bins.
        method (str): Level extraction method.
        bounds (tuple, optional): Range for histogram.

    Returns:
        tuple: ((low_level, high_level), bin_centers, histogram)
    """
    y = A
    low_level, high_level, bin_centers, smoothed_hist, _ = impl.detect_signal_levels_with_histogram(None, y, nbins=nbins, smooth_sigma=0)
    return (low_level, high_level), bin_centers, smoothed_hist


def risetime(x: NumberIterable, fs: Optional[float]=1,
             t: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:
    """
    Detect rising edge timing with interpolation.

    Returns:
        Edge or None
    """
    x_uniform, t_uniform = impl._get_xtime_from_t_fs(x=x, fs=fs, t=t)
    return impl._detect_edge_wrapper(sign=EdgeSign.rising,
                        x=t_uniform,
                        y=x_uniform,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)

def falltime(x: NumberIterable, fs: Optional[float]=1,
             t: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:
    """
    Detect falling edge timing with interpolation.

    Returns:
        Edge or None
    """
    x_uniform, t_uniform = impl._get_xtime_from_t_fs(x=x, fs=fs, t=t)
    return impl._detect_edge_wrapper(sign=EdgeSign.falling,
                        x=t_uniform,
                        y=x_uniform,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)

def midcross(x, fs: Optional[float] = 1,
             t: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             **kwargs) -> float:
    """
    Find mid-level crossing time of a bilevel signal.

    Args:
        x (array-like): Signal data.
        fs (float): Sampling rate.
        t (array-like, optional): Time array.
        levels (tuple, optional): Reference levels.

    Returns:
        float: Time of mid-reference crossing.
    """

    x_uniform, t_uniform = impl._get_xtime_from_t_fs(x=x, fs=fs, t=t)
    if not levels:
        levels, *_ = statelevels(A=x_uniform, **kwargs)

    return impl._calculate_midcross(x=t_uniform, y=x_uniform, levels=levels)

def overshoot(x: NumberIterable,
              levels: Optional[Tuple[float, float]]=None,
              **kwargs) -> float:
    """
    Compute normalized overshoot fraction of a step response.

    Args:
        x (array-like): Signal data.
        levels (tuple, optional): Low/high state levels.

    Returns:
        float: Overshoot fraction.
    """

    if not levels:
        levels, *_ = statelevels(A=x, **kwargs)

    return impl._calculate_overshoot(y=x, levels=levels)


def undershoot(x: NumberIterable,
               levels: Optional[Tuple[float, float]]=None,
               **kwargs) -> float:
    """
    Compute normalized undershoot fraction of a step response.

    Args:
        x (array-like): Signal data.
        levels (tuple, optional): Low/high state levels.

    Returns:
        float: Undershoot fraction.
    """

    if not levels:
        levels, *_ = statelevels(A=x, **kwargs)

    return impl._calculate_undershoot(y=x, levels=levels)

def slew_rate(x: NumberIterable, fs: Optional[float] = 1,
             t: Optional[NumberIterable]=None,
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
    x_uniform, t_uniform = impl._get_xtime_from_t_fs(x=x, fs=fs, t=t)

    return impl._calculate_slew_rate(x=t_uniform, y=x_uniform)

def settling_time(x: NumberIterable, d: float = 0.02,
                 fs: Optional[float] = 1,
                 t: Optional[NumberIterable] = None,
                 levels: Optional[Tuple[float, float]] = None,
                 settling_time_margin: Optional[float] = None,
                 **kwargs):
    """
    Calculate the settling time of a step response signal.

    Args:
        x (array-like): Signal data.
        d (float): Fractional tolerance band for settling (default 0.02 = 2%).
        fs (float): Sampling rate.
        t (array-like, optional): Time array.
        levels (tuple, optional): Low/high reference levels.
        settling_time_margin (float, optional): Additional time margin added after last deviation.

    Returns:
        float: Settling time in units of t (or samples if t is None).
    """

    if levels is None:
        levels, *_ = statelevels(A=x, **kwargs)

    x_uniform, t_uniform = impl._get_xtime_from_t_fs(x=x, fs=fs, t=t)

    return impl._calculate_settling_time(y=x_uniform, x=t_uniform, settling_time_margin=settling_time_margin, settling_time_fraction=d, levels=levels)
