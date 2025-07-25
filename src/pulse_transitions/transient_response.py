'''
Transient response edge detection module.

'''
from dataclasses import dataclass
from typing import Iterable, Union, Optional, Tuple
import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from . common import PairedEdge, CrossingDetectionSettings, Edge, Peak, EdgeSign
from . import impl

NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log = logging.getLogger("pulse_transitions")


def detect_signal_levels(x: NumberIterable, y: NumberIterable, method="histogram", **kwargs):
    """
    Estimate low and high signal levels in a two-mode system using a selected method.

    Args:
        x (array-like): Time or index vector.
        y (array-like): Signal data.
        method (str): Detection method ('histogram', 'derivative', 'endpoint').

    Returns:
        tuple: (low_level, high_level)
    """

    methods = {
        "histogram": impl.detect_signal_levels_with_histogram,
        "derivative": impl.detect_signal_levels_with_derivative,
        "endpoint": impl.detect_signal_levels_with_endpoints,
    }

    if method not in methods:
        raise ValueError(f"Method '{method}' not one of: {methods.keys()}")

    low_level, high_level, *_ = methods[method](x, y, **kwargs)
    return low_level, high_level


def detect_thresholds(x: NumberIterable, y: NumberIterable, method='histogram',
                      thresholds: Tuple[float,float]=(0.1, 0.9), **kwargs):
    """
    Estimate threshold levels based on flat signal regions.

    Args:
        x (array-like): Time or index array.
        y (array-like): Signal data.
        method (str): Method for level detection ('histogram', etc.).
        thresholds (tuple): Normalized threshold fractions.

    Returns:
        list: Absolute threshold values.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    low, high, *_ = detect_signal_levels(x, y, method=method, **kwargs)

    return impl._calculate_thresholds(x, y, [low, high], low_fraction, high_fraction)


def detect_first_edge(x: NumberIterable, y: NumberIterable,
                      sign: Union[EdgeSign, int],
                      thresholds: Tuple[float,float]=(0.1, 0.9),
                      *, settings: Optional[CrossingDetectionSettings] = None) -> Optional[Edge]:
    """
    Detect the first threshold crossing edge of specified polarity.

    Args:
        x (array-like): Time or index array.
        y (array-like): Signal data.
        thresholds (tuple): Threshold values.
        sign (EdgeSign or int): Desired edge polarity.
        settings (CrossingDetectionSettings): Optional crossing settings.

    Returns:
        Edge or None: Detected edge or None if not found.
    """

    assert len(x)
    assert len(y)
    return impl._detect_first_edge(x=x, y=y,
                      sign=sign,
                      thresholds=thresholds,
                      settings=settings)


def detect_edges(x: NumberIterable, y: NumberIterable,
                 thresholds: Tuple[float,float]=(0.1, 0.9),
                 *, bounds=None,
                 settings: Optional[CrossingDetectionSettings] = None) -> list[Edge]:
    """
    Detect rising and falling edges using derivative peaks and interpolation.

    Args:
        x (array-like): Time or index array.
        y (array-like): Signal data.
        thresholds (tuple): Threshold values.
        bounds (tuple, optional): Time bounds to restrict analysis.
        settings (CrossingDetectionSettings): Detection configuration.

    Returns:
        list[Edge]: List of detected edges.
    """

     # FIXME this should split the signals into each par of signal

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Apply our bounds and redefine our inputs to in bound
    if bounds is not None:
        mask = (x >= bounds[0]) & (x <= bounds[1])
        x, y = x[mask], y[mask]

    # Find peaks by derivative only
    peaks : Tuple[Peak] = impl._find_peaks_and_types(x, y, thresholds=thresholds, settings=settings)
    edges = []
    for peak in peaks:
        try:
            edge = detect_first_edge(x, y, thresholds, sign=peak.sign,
                                     settings=settings)
            edges.append(edge)
        except (IndexError, ValueError) as e:
            msg = f"Skipping peak {peak.start}-{peak.end}, interpolation not possible: {e}"
            log.debug(msg)
            continue
    return edges

def detect_overshoot(x, y: NumberIterable, method="histogram"):
    """
    Estimate signal levels, then compute overshoot and undershoot.

    Args:
        x (array-like): Time or index array.
        y (array-like): Signal data.
        method (str): Level detection method.

    Returns:
        tuple: (undershoot_fraction, overshoot_fraction)
    """

    levels = detect_signal_levels(x, y, method=method)
    return impl._calculate_overshoot(y, levels=levels)

def pair_edges(edges: list[Edge], *, max_gap: float = None) -> list[PairedEdge]:
    """
    Pair rising and falling edges into pulses based on order and timing.

    Args:
        edges (list[Edge]): List of edges to pair.
        max_gap (float, optional): Maximum allowed gap between rising and falling edge.

    Returns:
        list[PairedEdge]: List of valid rising/falling edge pairs.
    """

    edges = sorted(edges, key=lambda e: e.start)
    pairs = []
    used_indices = set()

    for i, first_edge in enumerate(edges):
        if first_edge.sign != EdgeSign.rising or i in used_indices:
            continue
        for j in range(i + 1, len(edges)):
            if j in used_indices:
                continue
            second_edge = edges[j]
            if second_edge.sign == EdgeSign.falling:
                if max_gap is not None and (second_edge.start - first_edge.end) > max_gap:
                    break
                pair = PairedEdge(rise=first_edge, fall=second_edge)
                if pair.is_valid:
                    pairs.append(pair)
                    used_indices.update({i, j})
                    break
                log.error("Edge is invalid")
    return pairs

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
    return impl._detect_edge_wrapper(sign=EdgeSign.rising,
                        x=x, fs=fs,
                        t=t,
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
    return impl._detect_edge_wrapper(sign=EdgeSign.falling,
                        x=x, fs=fs,
                        t=t,
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

    # Midpoint level
    mid = 0.5 * (levels[0] + levels[1])

    # Find the first crossing
    above = x_uniform > mid
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    if crossings.size == 0:
        raise ValueError("No midpoint crossing found")

    idx = crossings[0]
    # Linear interpolation for more precise crossing time
    x0, x1 = x_uniform[idx], x_uniform[idx + 1]
    t0, t1 = t_uniform[idx], t_uniform[idx + 1]

    frac = (mid - x0) / (x1 - x0)
    t_cross = t0 + frac * (t1 - t0)

    return t_cross

def overshoot(x: NumberIterable,
              levels: Optional[Tuple[float, float]]=None,
              **kwargs):
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

    low, high = levels
    step_height = high - low
    if step_height == 0:
        raise ValueError("State levels are equal — cannot compute overshoot")

    # Detect step direction
    rising = np.abs(x[-1] - high) < np.abs(x[-1] - low)

    if rising:
        max_val = np.max(x)
        overshoot_val = max_val - high
    else:
        min_val = np.min(x)
        overshoot_val = low - min_val

    # Normalize
    overshoot_frac = max(0.0, overshoot_val / step_height)
    return overshoot_frac


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

    # Estimate state levels if not provided
    if levels is None:
        levels, *_ = statelevels(x, **kwargs)

    low, high = levels
    step_height = high - low
    if step_height == 0:
        raise ValueError("State levels are equal — cannot compute undershoot")

    x = np.asarray(x)
    # Detect step direction (rising or falling)
    rising = np.abs(x[-1] - high) < np.abs(x[-1] - low)

    # Find step edge index by locating midpoint crossing
    mid = 0.5 * (low + high)
    above = x > mid
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    if crossings.size == 0:
        raise ValueError("No step edge crossing found")

    edge_idx = crossings[0]

    # Define post-edge analysis window (make sure not to exceed signal length)
    start_idx = edge_idx + 1
    end_idx = len(x)
    post_edge = x[start_idx:end_idx]

    if rising:
        # Undershoot is how far the minimum after the edge falls below low level
        min_val = np.min(post_edge)
        undershoot_val = high - min_val
    else:
        # Undershoot is how far the maximum after the edge rises above high level
        max_val = np.max(post_edge)
        undershoot_val = max_val - low

    undershoot_frac = undershoot_val / abs(step_height)
    return undershoot_frac


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

    # Calculate the discrete derivative dy/dx
    slew = np.gradient(x_uniform, t_uniform)

    # Remove any NaN or infinite values due to zero dx
    slew = slew[np.isfinite(slew)]

    if len(slew) == 0:
        raise ValueError("No valid slew rate data points found")

    return np.max(np.abs(slew))
