'''
Transient response edge detection module.
Provides functions to detect rising and falling edges in signals using derivative-based peak detection and interpolation.
'''
from dataclasses import dataclass
from typing import Iterable, Union, Optional, Tuple
import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from .common import PairedEdge, CrossingDetectionSettings, Edge, Peak, EdgeSign
from . import impl
#from .impl import normalize, denormalize, closest_index, _interpolate_crossing, \
#                  _find_peaks_and_types, detect_signal_levels_with_derivative, detect_signal_levels_with_endpoints, \
#                  detect_signal_levels_with_endpoints, filter_overlapping_edges, \
#                  detect_signal_levels_with_histogram, detect_signal_levels_with_derivative, detect_signal_levels_with_endpoints


NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log = logging.getLogger("pulse_transitions")


def detect_signal_levels(x: NumberIterable, y: NumberIterable, method="histogram", **kwargs):
    """
    Detect the levels of a 2-mode system using one of several methods.
    Returns:
        (low_level, high_level)
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


def calculate_thresholds(x: NumberIterable, y: NumberIterable,
                         levels: Tuple[float, float],
                         thresholds: Tuple[float,float]=(0.1, 0.9)):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    low, high = levels
    diff = high-low
    assert diff >= 0

    return list(sorted((
        low + min(thresholds) * diff,
        low + max(thresholds) * diff
    )))

def detect_thresholds(x: NumberIterable, y: NumberIterable, method='histogram',
                      thresholds: Tuple[float,float]=(0.1, 0.9),
                      **kwargs):
    '''
    Estimate low and high reference levels using flat portions of the signal where the derivative is minimal.

    Args:
        x (array-like): Time or sample index array.
        y (array-like): Signal amplitude array.
        low_fraction (float): Fractional level for the low reference (e.g., 0.1).
        high_fraction (float): Fractional level for the high reference (e.g., 0.9).

    Returns:
        tuple: (low_level, high_level)
    '''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    low, high, *_ = detect_signal_levels(x, y, method=method, **kwargs)

    return calculate_thresholds(x, y, [low, high], low_fraction, high_fraction)


def detect_first_edge(x: NumberIterable, y: NumberIterable,
                      thresholds: Tuple[float,float]=(0.1, 0.9),
                      sign: Union[EdgeSign, int],
                      *, settings: Optional[CrossingDetectionSettings] = None) -> Optional[Edge]:
    '''
    Detect the first edge in a data series.
    return Edge
    '''
    assert len(x)
    assert len(y)
    xbound = x
    ybound = y
    if settings is None:
        settings = CrossingDetectionSettings()

    if settings.window > 0:
        # FIXME instead of just a window use the multiple edges to find an appropriate separation between points
        mask = (x>(peak.start-settings.window/2)) & (x<(peak.end+settings.window/2))
        assert peak.end > peak.start
        xbound = xbound[mask]
        ybound = ybound[mask]
    # Interpolate the crossing point to find a more accurate crossing

    if min(y) > min(thresholds) or max(y) < max(thresholds):
        return None

    x1, x2 = impl._interpolate_crossing(
        x=xbound, y=ybound,
        thresholds=thresholds,
        sign=EdgeSign(sign)
    )

    # Confirm that the segment around the peak actually crosses both levels
    idxs = [impl.closest_index(xbound, x1), impl.closest_index(xbound, x2)]

    # Handle discontinuous edge
    if idxs[0] == idxs[1]:
        idxs = [max([0, idxs[0]-1]), min([len(xbound), idxs[0]+1])]

    # Make an edge object
    return Edge(
            start=x1, end=x2, sign=sign,
            thresholds=thresholds,
            ymin=min(ybound[idxs]), ymax=max(ybound[idxs]))


def detect_edges(x: NumberIterable, y: NumberIterable,
                 thresholds: Tuple[float,float]=(0.1, 0.9),
                 *, bounds=None,
                 settings: Optional[CrossingDetectionSettings] = None) -> list[Edge]:
    '''
    Detects rising and falling edges in a signal using derivative peak widths and interpolated threshold crossings.

    Args:
        x (array-like): Time or sample index array.
        y (array-like): Signal amplitude array.
        thresholds (tuple): Low and high thresholds for signal.
        bounds (tuple, optional): Time bounds (min, max) to restrict analysis.
        hysteresis_window (int, optional): Limit threshold search around peaks to N samples.

    Returns:
        List[Edge]: Detected edges with precise timings.
    '''
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


def calculate_overshoot(y: NumberIterable, levels: Tuple[float, float]):
    '''
    Take the data and an Edge object, returns the overshoot / undershoot of the edge.
    thresholds (tuple): Low and high thresholds for signal.
    returns undershoot, overshoot as a fraction
    '''
    low, high = levels
    height = abs(high-low)
    return (min(y)-low)/height, (max(y)-high)/height

def detect_overshoot(x, y: NumberIterable, method="histogram"):
    '''
    Take the data and an Edge object, returns the overshoot / undershoot of the edge.
    thresholds (tuple): Low and high thresholds for signal.
    returns undershoot, overshoot as a fraction
    '''
    levels = detect_signal_levels(x, y, method=method)
    return calculate_overshoot(y, levels=levels)

def pair_edges(edges: list[Edge], *, max_gap: float = None) -> list[PairedEdge]:
    """
    Pairs rising and falling edges in order of occurrence. Assumes positive pulses.
    Invert y if using negative pulses.

    Args:
        edges (list[Edge]): List of rising and falling edges.
        max_gap (float, optional): Maximum allowed time between rising and falling edges.

    Returns:
        list[PairedEdge]: List of paired edges.
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


def _get_xtime_from_t_fs(x: NumberIterable,
                         fs: Optional[float]=1,
                         t: Optional[NumberIterable]=None):
    x = np.asarray(x)
    # Time vector handling
    if t is not None:
        t = np.asarray(t)
        if t.shape != x.shape:
            raise ValueError("t must be the same shape as x")
        # Use a spline to resample signal uniformly
        f = interp1d(t, x, kind='cubic', fill_value='extrapolate')
        t_uniform = np.linspace(t[0], t[-1], len(x))
        x_uniform = f(t_uniform)
    else:
        # Uniform time base
        t_uniform = np.arange(len(x)) / fs
        x_uniform = x
    return x_uniform, t_uniform


def _detect_edge_wrapper(sign: EdgeSign, x: NumberIterable, fs: Optional[float]=1,
             t: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:
    x_uniform, t_uniform = _get_xtime_from_t_fs(x=x, fs=fs, t=t)
    if not levels:
        levels, *_ = statelevels(A=x_uniform, **kwargs)

    level_diff = max(levels) - min(levels)
    edge_thresholds = (
        min(levels) + level_diff*min(thresholds),
        min(levels) + level_diff*max(thresholds))

    return detect_first_edge(x=t_uniform, y=x_uniform,
                             thresholds=edge_thresholds,
                             sign=sign, settings=settings)

#=========================
# Matlab naming starts
#=========================


def statelevels(A: NumberIterable, nbins: int = 100, method="mode",
                bounds: Tuple[float, float] = None, **kwargs):
    '''
    Estimate state-level for bilevel waveform A using histogram method.
    '''
    y = A
    low_level, high_level, bin_centers, smoothed_hist, _ = impl.detect_signal_levels_with_histogram(None, y, nbins=nbins, smooth_sigma=0)
    return (low_level, high_level), bin_centers, smoothed_hist


def risetime(x: NumberIterable, fs: Optional[float]=1,
             t: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:
    return _detect_edge_wrapper(sign=EdgeSign.rising,
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
    return _detect_edge_wrapper(sign=EdgeSign.falling,
                        x=x, fs=fs,
                        t=t,
                        levels=levels,
                        thresholds=thresholds,
                        settings=settings, **kwargs)

def midcross(x, fs:Optional[float] =1,
             t: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             **kwargs):
    '''
    Mid-reference level crossing for bilevel waveform.
    Calculate statelevels and return the average.

    Args:
        x (array-like): Bilevel signal waveform.
        Fs (float): Sampling frequency (ignored if `t` is provided).
        t (array-like, optional): Time vector (same length as x).
        levels (tuple, optional): Reference levels used, otherwise they are calculated
        **kwargs: Passed to `statele()` for level detection.

    Returns:
        float: Time or index of mid-reference crossing.
    '''
    x_uniform, t_uniform = _get_xtime_from_t_fs(x=x, fs=fs, t=t)
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
    '''
    Compute normalized undershoot immediately after the step edge.

    Args:
        x (array-like): Step response signal.
        levels (tuple, optional): (low_level, high_level). If None, estimated via `statele()`.
        window (int): Number of samples after edge to consider for undershoot.
        **kwargs: Passed to `statele()` if levels is None.

    Returns:
        float: Undershoot fraction of step height (0 if no undershoot).
    '''
    x = np.asarray(x)

    # Estimate state levels if not provided
    if levels is None:
        levels, *_ = statelevels(x, **kwargs)

    low, high = levels
    step_height = high - low
    if step_height == 0:
        raise ValueError("State levels are equal — cannot compute undershoot")

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
        undershoot_val = max(0, low - min_val)
    else:
        # Undershoot is how far the maximum after the edge rises above high level
        max_val = np.max(post_edge)
        undershoot_val = max(0, max_val - high)

    undershoot_frac = undershoot_val / abs(step_height)
    return undershoot_frac
