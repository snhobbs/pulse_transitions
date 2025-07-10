from dataclasses import dataclass
from typing import Iterable, Union, Tuple, List, Optional
import logging
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
from .common import CrossingDetectionSettings, GroupStrategy, Peak, closest_index, Edge


NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log = logging.getLogger("pulse_transitions")


def normalize(y: NumberIterable) -> np.ndarray:
    '''
    Normalize to minimize suprises. Large excursions can still screw us up.
    '''
    y = np.asarray(y, dtype=float)
    y_min, y_max = y.min(), y.max()
    denom = y_max - y_min
    y_norm = (y - y_min) / denom if denom != 0 else np.zeros_like(y)
    return y_norm


def denormalize(y: NumberIterable, y_norm: NumberIterable) -> np.ndarray:
    '''
    Remove the normalization of an array
    For a subsection of points y_norm undo the normalization transformation
    applied to them by passing in the original array or the min and max values of it.
    The normalization values are calculated and undone from y_norm.
    '''
    y = np.asarray(y, dtype=float)
    y_min, y_max = y.min(), y.max()
    denom = y_max - y_min
    if denom == 0:
        return [y[0]]*len(y_norm)

    return np.asarray(y_norm, dtype=float)*denom+y_min


def smooth_zero_phase(y: np.ndarray, normal_cutoff: int, fs: float, order: int = 3):
    """
    Apply zero-phase low-pass Butterworth filter.

    Args:
        y (np.ndarray): Signal array.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Smoothed signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, y)


def _interpolate_crossing(x: np.ndarray, y: np.ndarray, thresholds: Tuple[float, float], sign: int):
    '''
    Interpolate precise crossing times for low and high thresholds around a peak with optional hysteresis window.

    Args:
        x (np.ndarray): Time array.
        y (np.ndarray): Signal array.
        thresholds (float,float): Threshold crossing values.
        sign (int): +1 for rising pulse, -1 for falling pulse.

    Returns:
        tuple: (start_time, end_time) for lo_val and hi_val crossings, in time order.
    '''
    lo_val = min(thresholds)
    hi_val = max(thresholds)

    n = len(y)
    if sign == 1:
        # rising edge
        pre_mask = (y <= lo_val)
        post_mask = (y >= hi_val)
    else:
        # falling edge
        pre_mask = (y >= hi_val)
        post_mask = (y <= lo_val)

    # The start candidates (i1) are those that are above the max thresholds for falling or below min for rising
    # The end candidates (i2) are those that are above the max thresholds for rising or below min for falling
    # Where will select the parts based on the boolean mask
    i1_candidates = np.where(pre_mask)[0]
    i2_candidates = np.where(post_mask)[0]

    if len(i1_candidates) == 0 or len(i2_candidates) == 0:
        raise IndexError(f"Edge doesn't cross thresholds")

    i1 = i1_candidates[-1] # + i1_range.start
    i2 = i2_candidates[0]  #+ i2_range.start

    if i1 + 1 >= n or i2 >= n or i2 < 1:
        raise IndexError(f"Edge interpolation range out of bounds")

    # Do a linear interpolation for both thresholds to find the closest crossing in x
    x_cross_lo = np.interp(
        lo_val,
        [y[i1 - 1], y[i1 + 1]],
        [x[i1 - 1], x[i1 + 1]])

    x_cross_hi = np.interp(
        hi_val,
        [y[i2 - 1], y[i2 + 1]],
        [x[i2 - 1], x[i2 + 1]])

    return (x_cross_lo, x_cross_hi)

def _split_pulses(
    x: NumberIterable,
    y: NumberIterable,
    thresholds: Tuple[float, float],
    *, settings: CrossingDetectionSettings) -> List[Tuple[int, str, int]]:
    '''
    Takes a 2 level signal.
    Either receives or calculates the levels.
    Use a fractional threshold (10/90%, 20/80% etc) to find the crossings
    Find the midpoint crossings and split at 50% between them. If no crossing before or after then include all the rest of the signal.
    '''
    raise NotImplementedError("Pulse splitting logic not implemented yet.")

def _find_peaks_and_types_histogram():
    '''
    Look for midpoint crossings as center of edge.
    Histrogram the midpoint crossings and use peak finding on the histrograms.
    Take the center of the histogram as the midpoint.
    Histogram the low and high threshold crossings and run peak finding on those also.
    Take the ceter of each of the closest threshold crossings as the crossing points.
    Determine if the slope for the sign of the edge.
    Return a list of edges sorted by time.
    '''
    pass


def choose_peak_from_group(group: List[Peak], strategy: GroupStrategy = GroupStrategy.median):
    '''
    Filter a single peak from a group in a crossing window
    '''
    if strategy == GroupStrategy.first:
        return group[0]
    if strategy == GroupStrategy.last:
        return group[-1]
    if strategy == GroupStrategy.mode:
        peaks = {p.index: p for p in group}
        idx = int(np.median(list(peaks.keys())))
        peak = peaks[idx]
        return peak
    if strategy == GroupStrategy.median:
        peaks = {p.index: p for p in group}
        idx = int(np.median(list(peaks.keys())))
        peak = peaks[idx]
        return peak
    raise ValueError(f"Unknown GroupStrategy {strategy}")


def _find_peaks_and_types(
    x: NumberIterable,
    y: NumberIterable,
    thresholds: Tuple[float, float],
    *,
    settings: CrossingDetectionSettings
) -> List[Tuple[Peak]]:
    """
    Identify rising/falling edges by selecting the widest derivative peaks that span thresholds.

    Returns:
        List[Tuple[int, str, int]]: Filtered (index, type, sign) peaks.
        Note this algorithm is optimistic when noise is present
    """
    x = np.asarray(x)
    y = np.asarray(y)
    fs = 1 / np.mean(np.diff(x))
    yfilt = smooth_zero_phase(y, normal_cutoff=settings.filter_cutoff, fs=fs, order = settings.filter_order)

    raw_peaks = []

    def get_state(pt, thresholds):
        if pt >= max(thresholds):
            return 1
        if pt <= min(thresholds):
            return -1
        return 0

    # Iterate through all points in a state machine, ends up taking the first than last crossing.
    # Will only find one edge per full crossing.

    prev_state = get_state(yfilt[0], thresholds=thresholds)
    edge_in_progress = False
    i1 = None
    direction = 0

    for i, val in enumerate(yfilt):
        state = get_state(val, thresholds=thresholds)

        if state == prev_state:
            continue

        if not edge_in_progress:
            if prev_state != 0 and state == 0:
                # Entering transition region from a known level
                i1 = i
                direction = -prev_state
                edge_in_progress = True

        else:
            if state == direction:
                # Completed transition to opposite level
                i2 = i
                raw_peaks.append(Peak(start=x[i1], end=x[i2], sign=direction))
                edge_in_progress = False
                i1 = None
                direction = 0

        prev_state = state

    return raw_peaks


def group_close_peaks(peaks: List[Peak], min_separation: float):
    '''
    Sort peaks into groups. Each group is defined by a dead time
    of min_separation in time.
    '''
    # Sort peaks so they appear in order
    peaks.sort(key=lambda p: p.position)

    # Group close peaks
    groups = []
    current_group = []
    for peak in peaks:
        if not current_group or peak.position - current_group[-1].position <= min_separation:
            current_group.append(peak)
        else:
            groups.append(current_group)
            current_group = [peak]

    if current_group:
        groups.append(current_group)
    return groups


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


def detect_signal_levels_with_histogram(_, y: NumberIterable, *, nbins: int =100, smooth_sigma: float =1):
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
    if smooth_sigma > 0:
        smoothed_hist = gaussian_filter1d(hist, sigma=smooth_sigma)
    else:
        smoothed_hist = hist

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
