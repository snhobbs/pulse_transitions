'''
Transient response edge detection module.
Provides functions to detect rising and falling edges in signals using derivative-based peak detection and interpolation.
'''
from dataclasses import dataclass
from typing import Iterable, Union
import logging

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log = logging.getLogger("photoreceiver_analysis")


def normalize(y: NumberIterable) -> np.ndarray:
    # Normalize to minimize suprises. Large excursions can still screw us up.
    y = np.asarray(y, dtype=float)
    y_min, y_max = y.min(), y.max()
    denom = y_max - y_min
    y_norm = (y - y_min) / denom if denom != 0 else np.zeros_like(y)
    return y_norm


def denormalize(y: NumberIterable, y_norm: NumberIterable) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y_min, y_max = y.min(), y.max()
    denom = y_max - y_min
    if denom == 0:
        return [y[0]]*len(y_norm)

    return np.asarray(y_norm, dtype=float)*denom+y_min

def closest_index(arr: np.ndarray, value: float) -> int:
    return np.abs(arr - value).argmin()


@dataclass
class Edge:
    '''
    Represents an edge transition in a signal.

    Attributes:
        start (float): Time of the low threshold crossing.
        end (float): Time of the high threshold crossing.
        dx (float): Duration between low and high threshold crossings.
        type (str): 'rise' or 'fall'.
    '''
    start: float
    end: float
    ymin: float
    ymax: float
    reference_high: float
    reference_low: float
    type: str  # 'rise' or 'fall'

    @property
    def dx(self):
        return abs(self.end - self.start)

    @property
    def bound_low(self):
        return [min([self.end, self.start]), self.reference_low]

    @property
    def bound_high(self):
        return [max([self.end, self.start]), self.reference_high]


@dataclass
class PairedEdge:
    rise: Edge
    fall: Edge

    @property
    def pulse_width(self) -> float:
        return self.fall.end - self.rise.start

    @property
    def amplitude(self) -> float:
        return self.rise.ymax - self.fall.ymin

    @property
    def is_valid(self) -> bool:
        return (self.rise.type == 'rise') and (self.fall.type == 'fall') and (self.fall.start >= self.rise.end)


def _find_peaks_and_types(x: NumberIterable, y: NumberIterable):
    '''
    Identify rising and falling peak indices and their types using peak widths and filtered transitions.

    Args:
        x (np.ndarray): Time array.
        y (np.ndarray): signal.

    Returns:
        List of tuples (index, type, sign).
    '''
    dydx = np.gradient(y, x)
    peaks = []
    for edge_type, sign in [('rise', 1), ('fall', -1)]:
        idxs, _ = find_peaks(sign * dydx)
        peaks.extend([(idx, edge_type, sign) for idx in idxs])

    return sorted(peaks, key=lambda t: t[0])


def _interpolate_crossing(x: np.ndarray, y: np.ndarray, idx: int, lo_val: float, hi_val: float, sign: int, *, window: int = None):
    '''
    Interpolate precise crossing times for low and high thresholds around a peak with optional hysteresis window.

    Args:
        x (np.ndarray): Time array.
        y (np.ndarray): Signal array.
        idx (int): Peak index in the arrays.
        lo_val (float): Low threshold value (closer to baseline).
        hi_val (float): High threshold value (closer to peak).
        sign (int): +1 for rising pulse, -1 for falling pulse.
        window (int, optional): Number of samples before/after peak to limit search.

    Returns:
        tuple: (start_time, end_time) for lo_val and hi_val crossings, in time order.
    '''
    assert hi_val >= lo_val, "Thresholds must satisfy hi_val >= lo_val"

    n = len(y)
    i1_range = slice(max(0, idx - window) if window else 0, idx)
    i2_range = slice(idx, min(n, idx + window) if window else n)

    if sign == 1:
        # rising edge
        pre_mask = (y <= lo_val)
        post_mask = (y >= hi_val)
    else:
        # falling edge
        pre_mask = (y >= hi_val)
        post_mask = (y <= lo_val)

    i1_candidates = np.where(pre_mask[i1_range])[0]
    i2_candidates = np.where(post_mask[i2_range])[0]

    if len(i1_candidates) == 0 or len(i2_candidates) == 0:
        raise IndexError(f"Edge {idx} doesn't cross thresholds")

    i1 = i1_candidates[-1] + i1_range.start
    i2 = i2_candidates[0] + i2_range.start

    if i1 + 1 >= n or i2 >= n or i2 < 1:
        raise IndexError(f"Edge {idx} interpolation range out of bounds")

    x_cross_lo = np.interp(lo_val, [y[i1], y[i1 + 1]], [x[i1], x[i1 + 1]])
    x_cross_hi = np.interp(hi_val, [y[i2 - 1], y[i2]], [x[i2 - 1], x[i2]])

    return (x_cross_lo, x_cross_hi)



def detect_signal_levels_with_histogram(_, y: NumberIterable, *, nbins: int =1000, smooth_sigma: float =1):
    """
    Estimate low and high voltage levels using a histogram of the signal.

    Parameters:
        signal (np.ndarray): The input voltage waveform.
        bins (int): Number of histogram bins.
        smooth_sigma (float): Gaussian smoothing for the histogram.

    Returns:
        (low_level, high_level, bin_centers, smoothed_hist, peak_voltages):
        Levels and data for optional plotting.
    """
    # Build histogram
    # data is normalized, include extra bin on each side for peak finding

    y_norm = normalize(y)

    assert min(y_norm) == 0
    assert max(y_norm) <= 1

    bin_size = 1/nbins
    bin_edges = np.linspace(-5*bin_size, 1+5*bin_size, nbins+1)
    hist, bin_edges_ = np.histogram(y_norm, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    assert all(bin_edges == bin_edges_)
    assert len(bin_centers) == len(hist)
    assert len(hist) == nbins

    # Smooth histogram
    smoothed_hist = gaussian_filter1d(hist, sigma=smooth_sigma)

    # Find peaks
    peak_indices, _ = find_peaks(smoothed_hist)#, prominence=np.max(smoothed_hist) * 0.05)

    # Take two largest peaks
    peaks = sorted(
        [(idx, bin_centers[idx], smoothed_hist[idx]) for idx in peak_indices],
        key=lambda x: x[2])[-2:]
    peak_voltages = bin_centers[[pt[0] for pt in peaks]]

    if len(peak_voltages) < 2:
        raise ValueError("Could not find two distinct voltage levels.")

    # Sort and assign low/high levels. Remove the normalization transformation
    low_level_norm, high_level_norm = (sorted(peak_voltages[:2]))
    low_level, high_level = denormalize(y, [low_level_norm, high_level_norm])

    return low_level, high_level, bin_centers, smoothed_hist, peak_voltages

def detect_signal_levels_with_endpoints(_, y: NumberIterable, *, n: int = 100):
    """
    Estimate low and high levels from the average of the endpoints of the trace.

    Parameters:
        y (np.ndarray): Signal array.
        n (int): Number of points to average from each end.

    Returns:
        tuple: (low_level, high_level)
    """
    if n * 2 > len(y):
        raise ValueError(f"n ({n}) is too large for the input length ({len(y)}).")

    return sorted([np.mean(y[:n]), np.mean(y[-n:])])

def detect_signal_levels_with_derivative(_, y: NumberIterable, *, smooth_sigma: float = 1.0, fraction: float = 0.1):
    """
    Estimate low and high signal levels by detecting flat regions based on the signal derivative.

    Parameters:
        x (np.ndarray): Time or index array.
        y (np.ndarray): Signal array.
        smooth_sigma (float): Sigma for Gaussian smoothing before derivative computation.
        fraction (float): Fraction of lowest derivative points to consider for level averaging.

    Returns:
        tuple: (low_level, high_level)
    """
    if smooth_sigma > 0:
        y_smooth = gaussian_filter1d(y, sigma=smooth_sigma)
    else:
        y_smooth = y

    dy = np.gradient(y_smooth)
    flatness = np.abs(dy)
    n_flat = max(1, int(len(y) * fraction))

    # Find indices of flattest points
    flat_indices = np.argsort(flatness)[:n_flat]
    flat_values = y[flat_indices]
    low_level = np.min(flat_values)
    high_level = np.max(flat_values)

    return low_level, high_level


def detect_signal_levels(x: NumberIterable, y: NumberIterable, method="histogram", **kwargs):
    """
    Returns:
        (low_level, high_level)
    """
    methods = {
        "histogram": detect_signal_levels_with_histogram,
        "derivative": detect_signal_levels_with_derivative,
        "endpoint": detect_signal_levels_with_endpoints,
    }

    if method not in methods:
        raise ValueError(f"Method '{method}' not one of: {methods.keys()}")

    low_level, high_level, *_ = methods[method](x, y, **kwargs)
    return low_level, high_level

def calculate_thresholds(x: NumberIterable, y: NumberIterable, levels: tuple[float, float], low_fraction=0.1, high_fraction=0.9):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    low, high = levels
    diff = high-low
    assert diff >= 0

    return list(sorted((
        low + low_fraction * diff,
        low + high_fraction * diff
    )))

def detect_thresholds(x: NumberIterable, y: NumberIterable, method='histogram', low_fraction=0.1, high_fraction=0.9, **kwargs):
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


def filter_overlapping_edges(edges: list[Edge], *, min_separation: float = 0.0) -> list[Edge]:
    """
    Filters overlapping Edge objects. Keeps only the edge with the shortest transition time (dx)
    from each overlapping group.

    Args:
        edges (List[Edge]): List of Edge objects.
        min_separation (float): Optional buffer between edge groups in time.

    Returns:
        List[Edge]: Filtered list of non-overlapping edges.
    """
    if not edges:
        return []

    # Sort edges by start time
    edges = sorted(edges, key=lambda x: x.start)

    filtered = []
    current_group = []

    for edge in edges:
        if not current_group:
            current_group.append(edge)
            continue

        last_group_end = max(e.end for e in current_group)

        # Overlap if current edge starts before end of last group (+ optional buffer)
        if edge.start < last_group_end + min_separation:
            current_group.append(edge)
        else:
            # Select the shortest edge in the current group
            best_edge = min(current_group, key=lambda e: e.dx)
            filtered.append(best_edge)
            current_group = [edge]

    if current_group:
        best_edge = min(current_group, key=lambda e: e.dx)
        filtered.append(best_edge)

    return filtered


def detect_edges(x: NumberIterable, y: NumberIterable, thresholds, *, bounds=None, hysteresis_window=None) -> list[Edge]:
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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if bounds is not None:
        mask = (x >= bounds[0]) & (x <= bounds[1])
        x, y = x[mask], y[mask]

    peaks = _find_peaks_and_types(x, y)

    edges = []
    for idx, edge_type, sign in peaks:
        try:
            x1, x2 = _interpolate_crossing(
                x, y, idx,
                lo_val=min(thresholds),
                hi_val=max(thresholds),
                sign=sign,
                window=hysteresis_window
            )
            # Confirm that the segment around the peak actually crosses both levels
            idxs = [closest_index(x, x1), closest_index(x, x2)]

            # Handle discontinuous edge
            if idxs[0] == idxs[1]:
                idxs = [max([0, idxs[0]-1]), min([len(x), idxs[0]+1])]

            edge = Edge(
                    start=x1, end=x2, type=edge_type,
                    reference_high=max(thresholds),
                    reference_low=min(thresholds),
                    ymin=min(y[idxs]), ymax=max(y[idxs]))

            ref_range = max(thresholds) - min(thresholds)
            if (edge.ymax < (max(thresholds) - ref_range/5) or
                edge.ymin > (min(thresholds) + ref_range/5)):
                log.warning(
                    "Discarding edge at index %d: \
                    does not fully cross reference levels. %s", idx, str(edge))
                continue
            edges.append(edge)
        except (IndexError, ValueError) as e:
            msg = f"Skipping peak {idx}, interpolation not possible: {e}"
            log.debug(msg)
            continue

    return filter_overlapping_edges(edges)


def calculate_overshoot(y: NumberIterable, levels: tuple[float, float]):
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
        if first_edge.type != 'rise' or i in used_indices:
            continue
        for j in range(i + 1, len(edges)):
            if j in used_indices:
                continue
            second_edge = edges[j]
            if second_edge.type == 'fall':
                if max_gap is not None and (second_edge.start - first_edge.end) > max_gap:
                    break
                pair = PairedEdge(rise=first_edge, fall=second_edge)
                if pair.is_valid:
                    pairs.append(pair)
                    used_indices.update({i, j})
                    break
                log.error("Edge is invalid")
    return pairs
