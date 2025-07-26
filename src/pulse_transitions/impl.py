import logging
from collections.abc import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from .common import CrossingDetectionSettings
from .common import Edge
from .common import EdgeSign
from .common import PairedEdge
from .common import Peak

NumberIterable = Union[np.ndarray, Iterable[Union[int, float]]]
log_ = logging.getLogger("pulse_transitions")

def closest_index(arr: np.ndarray, value: float) -> int:
    """
    Return the index of the point closest to the given value
    """
    return np.abs(arr - value).argmin()

def normalize(y: NumberIterable) -> np.ndarray:
    """
    Normalize to minimize suprises. Large excursions can still screw us up.
    """
    y = np.asarray(y, dtype=float)
    y_min, y_max = y.min(), y.max()
    denom = y_max - y_min
    return (y - y_min) / denom if denom != 0 else np.zeros_like(y)


def denormalize(y: NumberIterable, y_norm: NumberIterable) -> np.ndarray:
    """
    Remove the normalization of an array
    For a subsection of points y_norm undo the normalization transformation
    applied to them by passing in the original array or the min and max values of it.
    The normalization values are calculated and undone from y_norm.
    """
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
    b, a = scipy.signal.butter(order, normal_cutoff, btype="low", analog=False)
    return scipy.signal.filtfilt(b, a, y)


def _interpolate_crossing(x: np.ndarray, y: np.ndarray, thresholds: Tuple[float, float], sign: EdgeSign):
    """
    Interpolate precise crossing times for low and high thresholds around a peak with optional hysteresis window.

    Args:
        x (np.ndarray): Time array.
        y (np.ndarray): Signal array.
        thresholds (float,float): Threshold crossing values.
        sign (EdgeSign): EdgeSign.rising or EdgeSign.falling

    Returns:
        tuple: (start_time, end_time) for lo_val and hi_val crossings, in time order.
    """
    lo_val = min(thresholds)
    hi_val = max(thresholds)

    n = len(y)
    if sign == EdgeSign.rising:
        # rising edge
        pre_mask = (y <= lo_val)
        post_mask = (y >= hi_val)
    else:
        # falling edge
        assert sign == EdgeSign.falling
        pre_mask = (y >= hi_val)
        post_mask = (y <= lo_val)

    # The start candidates (i1) are those that are above the max thresholds for falling or below min for rising
    # The end candidates (i2) are those that are above the max thresholds for rising or below min for falling
    # Where will select the parts based on the boolean mask
    i1_candidates = np.where(pre_mask)[0]
    i2_candidates = np.where(post_mask)[0]

    if len(i1_candidates) == 0 or len(i2_candidates) == 0:
        msg = "Edge doesn't cross thresholds"
        raise IndexError(msg)

    i1 = i1_candidates[-1] # + i1_range.start
    i2 = i2_candidates[0]  #+ i2_range.start

    if i1 + 1 >= n or i2 >= n or i2 < 1:
        msg = "Edge interpolation range out of bounds"
        raise IndexError(msg)

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


def _find_peaks_and_types_histogram():
    """
    Look for midpoint crossings as center of edge.
    Histrogram the midpoint crossings and use peak finding on the histrograms.
    Take the center of the histogram as the midpoint.
    Histogram the low and high threshold crossings and run peak finding on those also.
    Take the ceter of each of the closest threshold crossings as the crossing points.
    Determine if the slope for the sign of the edge.
    Return a list of edges sorted by time.
    """


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

    def get_state(pt, thresholds) -> int:
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

        elif state == direction:
            # Completed transition to opposite level
            i2 = i
            raw_peaks.append(
                Peak(start=x[i1], end=x[i2], sign=EdgeSign(direction)))
            edge_in_progress = False
            i1 = None
            direction = 0

        prev_state = state

    return raw_peaks


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
    peak_indices, _ = scipy.signal.find_peaks(smoothed_hist)#, prominence=np.max(smoothed_hist) * 0.05)

    # Take two largest peaks
    peaks = sorted(
        [(idx, bin_centers[idx], smoothed_hist[idx]) for idx in peak_indices],
        key=lambda x: x[2])[-2:]
    peak_voltages = bin_centers[[pt[0] for pt in peaks]]

    if len(peak_voltages) < 2:
        msg = "Could not find two distinct voltage levels."
        raise ValueError(msg)

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
        msg = f"n ({n}) is too large for the input length ({len(y)})."
        raise ValueError(msg)

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
    y_smooth = gaussian_filter1d(y, sigma=smooth_sigma) if smooth_sigma > 0 else y

    dy = np.gradient(y_smooth)
    flatness = np.abs(dy)
    n_flat = max(1, int(len(y) * fraction))

    # Find indices of flattest points
    flat_indices = np.argsort(flatness)[:n_flat]
    flat_values = y[flat_indices]
    low_level = np.min(flat_values)
    high_level = np.max(flat_values)

    return low_level, high_level


def _get_xtime_from_t_fs(x: NumberIterable,
                         fs: Optional[float]=1,
                         t: Optional[NumberIterable]=None):
    """
    Returns normalized spacing in the data. If a time vector is passed
    then both x and t will be interpolated to generate evenly spaced data.
    """
    if(len(x) == 0):
        msg = "x cannot have a length of 0"
        raise ValueError(msg)

    x = np.asarray(x)
    # Time vector handling
    if t is not None:
        t = np.asarray(t)
        if t.shape != x.shape:
            msg = "t must be the same shape as x"
            raise ValueError(msg)
        # Use a spline to resample signal uniformly
        f = interp1d(t, x, kind="cubic", fill_value="extrapolate")
        t_uniform = np.linspace(t[0], t[-1], len(x))
        x_uniform = f(t_uniform)
    else:
        # Uniform time base
        t_uniform = np.arange(len(x)) / fs
        x_uniform = x
    return x_uniform, t_uniform


def _detect_edge_wrapper(sign: EdgeSign, y: NumberIterable,
             x: Optional[NumberIterable]=None,
             levels: Optional[Tuple[float,float]]=None,
             thresholds: Tuple[float,float]=(0.1, 0.9),
             settings: Optional[CrossingDetectionSettings]=None, **kwargs) -> Optional[Edge]:

    if not levels:
        low_level, high_level, *_ = detect_signal_levels_with_histogram(None, y=y, **kwargs)
        levels = low_level, high_level


    level_diff = max(levels) - min(levels)
    edge_thresholds = (
        min(levels) + level_diff*min(thresholds),
        min(levels) + level_diff*max(thresholds))

    return _detect_first_edge(x=x, y=y,
                             thresholds=edge_thresholds,
                             sign=sign)


def _detect_edges(x: NumberIterable, y: NumberIterable,
                 thresholds: Tuple[float,float],
                 *, bounds=None,
                 settings: Optional[CrossingDetectionSettings] = None) -> list[Edge]:
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

    # FIXME this should split the signals into each par of signal

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Apply our bounds and redefine our inputs to in bound
    if bounds is not None:
        mask = (x >= bounds[0]) & (x <= bounds[1])
        x, y = x[mask], y[mask]

    # Find peaks by derivative only
    peaks : Tuple[Peak] = _find_peaks_and_types(x, y,
                            thresholds=thresholds, settings=settings)
    edges = []
    for peak in peaks:
        try:
            edge = _detect_first_edge(x=x, y=y, thresholds=thresholds,
                                     sign=peak.sign)
            edges.append(edge)
        except (IndexError, ValueError) as e:
            msg = f"Skipping peak {peak.start}-{peak.end}, interpolation not possible: {e}"
            log_.debug(msg)
            continue
    return edges


def _calculate_thresholds(x: NumberIterable, y: NumberIterable,
                         levels: Tuple[float, float],
                         thresholds: Tuple[float,float]=(0.1, 0.9)):
    """
    Calculate absolute threshold values from normalized levels.

    Args:
        x (array-like): Time or index vector (not used directly).
        y (array-like): Signal data (not used directly).
        levels (tuple): (low_level, high_level).
        thresholds (tuple): Normalized threshold fractions (e.g. 0.1, 0.9).

    Returns:
        list: Absolute threshold values [threshold_low, threshold_high].
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    low, high = levels
    diff = high-low
    assert diff >= 0

    return sorted((
        low + min(thresholds) * diff,
        low + max(thresholds) * diff
    ))

def _calculate_overshoot(y: NumberIterable, levels: Tuple[float, float]) -> Tuple[int, float]:
    """
    Calculate overshoot and undershoot as fractions of the step height.

    Args:
        y (array-like): Signal data.
        levels (tuple): (low_level, high_level)

    Returns:
        tuple(int, float): peak position, overshoot_fraction
    """

    low, high = levels
    step_height = high - low
    if step_height == 0:
        return 0

    # Detect step direction
    rising = np.abs(y[-1] - high) < np.abs(y[-1] - low)
    min_val = np.min(y)
    max_val = np.max(y)

    if rising:
        overshoot_val = max_val - high
        peak_value = max_val

    else:
        overshoot_val = low - min_val
        peak_value = min_val

    loc = np.where(np.isclose(y, peak_value))[0][0]
    return loc, max(0.0, overshoot_val / step_height)


def _calculate_undershoot(y: NumberIterable, levels: Tuple[float, float]) -> Tuple[int, float]:
    """
    Calculate overshoot and undershoot as fractions of the step height.

    Args:
        y (array-like): Signal data.
        levels (tuple): (low_level, high_level)

    Returns:
        tuple(int1, float): peak position, overshoot_fraction
    """

    low, high = levels
    step_height = high - low
    if step_height == 0:
        return 0

    y = np.asarray(y)
    # Detect step direction (rising or falling)
    rising = np.abs(y[-1] - high) < np.abs(y[-1] - low)

    # Find step edge index by locating midpoint crossing
    mid = 0.5 * (low + high)
    above = y > mid
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    if crossings.size == 0:
        log_.warning("No crossing found in undershoot")
        return 0

    edge_idx = crossings[0]

    if rising:
        thresh_idx = edge_idx + np.where(y[edge_idx:] == max(y[edge_idx:]))[0][0]
        peak_value = min(y[thresh_idx:])
        undershoot_val = high - peak_value

    else:
        thresh_idx = edge_idx + np.where(y[edge_idx:] == min(y[edge_idx:]))[0][0]
        peak_value = max(y[thresh_idx:])
        undershoot_val = peak_value - low

    loc = thresh_idx + np.where(np.isclose(y[thresh_idx:], peak_value))[0][0]
    return loc, undershoot_val / abs(step_height)


def _detect_first_edge(x: NumberIterable, y: NumberIterable,
                      sign: Union[EdgeSign, int],
                      thresholds: Tuple[float,float]=(0.1, 0.9)) -> Optional[Edge]:
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
    xbound = x
    ybound = y
    # Interpolate the crossing point to find a more accurate crossing

    if min(y) > min(thresholds) or max(y) < max(thresholds):
        return None

    try:
        x1, x2 = _interpolate_crossing(
            x=xbound, y=ybound,
            thresholds=thresholds,
            sign=EdgeSign(sign)
        )
    except IndexError:
        return None

    # Confirm that the segment around the peak actually crosses both levels
    idxs = [closest_index(xbound, x1), closest_index(xbound, x2)]

    # Handle discontinuous edge
    if idxs[0] == idxs[1]:
        idxs = [max([0, idxs[0]-1]), min([len(xbound), idxs[0]+1])]

    # Make an edge object
    return Edge(
            start=x1, end=x2, sign=EdgeSign(sign),
            thresholds=thresholds,
            ymin=min(ybound[idxs]), ymax=max(ybound[idxs]))


def _calculate_settling_time(x, y, levels, settling_time_fraction=0.02, settling_time_margin=0):
    low, high = levels

    step_height = high - low
    if step_height == 0:
        return 0

    # Define settling band around final value
    final_val = y[-1]
    tol = settling_time_fraction * abs(step_height)
    lower_bound = final_val - tol
    upper_bound = final_val + tol

    # Find indices where signal is outside settling band
    out_of_bounds = np.where((y < lower_bound) | (y > upper_bound))[0]

    if len(out_of_bounds) == 0:
        # Signal is always within settling band
        return x[0]

    # Last time index outside settling band
    last_out_idx = out_of_bounds[-1]

    settling_t = x[last_out_idx]

    if settling_time_margin is not None:
        settling_t += settling_time_margin

    return settling_t


def _calculate_slew_rate(x, y):

    # Calculate the discrete derivative dy/dx
    slew = np.gradient(y, x)

    # Remove any NaN or infinite values due to zero dx
    slew = slew[np.isfinite(slew)]

    if len(slew) == 0:
        return 0

    return np.max(np.abs(slew))


def _calculate_midcross(x, y, levels):
    # Midpoint level
    mid = 0.5 * (levels[0] + levels[1])

    # Find the first crossing
    above = y > mid
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    if crossings.size == 0:
        return 0

    idx = crossings[0]
    # Linear interpolation for more precise crossing time
    y0, y1 = y[idx], y[idx + 1]
    x0, x1 = x[idx], x[idx + 1]

    frac = (mid - y0) / (y1 - y0)
    return x0 + frac * (x1 - x0)  # tcross


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
                log_.error("Edge is invalid")
    return pairs


def _detect_signal_levels(x: NumberIterable, y: NumberIterable, method="histogram", **kwargs):
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
        "histogram": detect_signal_levels_with_histogram,
        "derivative": detect_signal_levels_with_derivative,
        "endpoint": detect_signal_levels_with_endpoints,
    }

    if method not in methods:
        msg = f"Method '{method}' not one of: {methods.keys()}"
        raise ValueError(msg)

    low_level, high_level, *_ = methods[method](x, y, **kwargs)
    return low_level, high_level


def _detect_thresholds(x: NumberIterable, y: NumberIterable, method="histogram",
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

    low, high, *_ = _detect_signal_levels(x, y, method=method, **kwargs)
    levels = (low, high)

    return _calculate_thresholds(x, y, levels=levels, thresholds=thresholds)
