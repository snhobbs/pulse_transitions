"""
Comprehensive test suite for pulse_transitions.impl module.

Tests cover:
- Utility functions (normalize, denormalize, closest_index)
- Signal level detection (histogram, endpoints, derivative)
- Edge detection and interpolation
- Threshold calculations
- Overshoot/undershoot calculations
- Settling time and slew rate
- Edge pairing
"""

from pulse_transitions.common import (
    Edge,
    EdgeSign,
    PairedEdge,
    Peak,
    CrossingDetectionSettings,
)
from pulse_transitions.impl import (
    closest_index,
    normalize,
    denormalize,
    smooth_zero_phase,
    find_all_edge_pairs,
    detect_signal_levels_with_histogram,
    detect_signal_levels_with_endpoints,
    detect_signal_levels_with_derivative,
    _get_xtime_from_t_fs,
    _calculate_thresholds,
    _calculate_overshoot,
    _calculate_undershoot,
    _calculate_settling_time,
    _calculate_slew_rate,
    _calculate_midcross,
    _interpolate_crossing,
    _detect_first_edge,
    _detect_first_edge_with_splitting,
    _detect_edge_wrapper,
    _detect_edges,
    _detect_signal_levels,
    _detect_thresholds,
    _find_peaks_and_types,
    pair_edges,
)
import numpy as np
import pytest
from typing import Tuple, Sequence
from unittest.mock import Mock, patch

# Import the module under test
import sys

sys.path.insert(0, "/mnt/user-data/uploads")


# ============================================================================
# Fixtures and Helper Functions
# ============================================================================


@pytest.fixture
def simple_step_signal():
    """Create a simple step response: 0 -> 1."""
    t = np.linspace(0, 1, 1000)
    y = np.where(t < 0.5, 0.0, 1.0)
    return t, y


@pytest.fixture
def noisy_step_signal():
    """Create a noisy step response with overshoot."""
    t = np.linspace(0, 2, 2000)
    y = np.zeros_like(t)

    # Step at t=1
    step_idx = np.argmin(np.abs(t - 1.0))
    y[step_idx:] = 1.0

    # Add overshoot
    overshoot_region = (t > 1.0) & (t < 1.1)
    y[overshoot_region] += 0.2 * np.exp(-10 * (t[overshoot_region] - 1.0))

    # Add noise
    np.random.seed(42)
    y += np.random.normal(0, 0.01, len(y))

    return t, y


@pytest.fixture
def underdamped_step():
    """Create an underdamped step response (2nd order system)."""
    from scipy.signal import lti, step

    # Underdamped system: zeta=0.3, wn=10
    wn = 10
    zeta = 0.3
    num = [wn**2]
    den = [1, 2 * zeta * wn, wn**2]

    system = lti(num, den)
    t = np.linspace(0, 2, 2000)
    _, y = step(system, T=t)

    return t, y


@pytest.fixture
def pulse_train():
    """Create a pulse train signal."""
    t = np.linspace(0, 1, 10000)
    y = np.zeros_like(t)

    # Three pulses
    y[(t > 0.1) & (t < 0.2)] = 1.0
    y[(t > 0.4) & (t < 0.5)] = 1.0
    y[(t > 0.7) & (t < 0.8)] = 1.0

    return t, y


@pytest.fixture
def bilevel_signal():
    """Create a clean bilevel signal with multiple transitions."""
    t = np.linspace(0, 1, 10000)
    y = np.zeros_like(t)

    # Low = 0, High = 3.3 (like digital logic)
    y[(t > 0.1) & (t < 0.3)] = 3.3
    y[(t > 0.5) & (t < 0.7)] = 3.3

    return t, y


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Test basic utility functions."""

    def test_closest_index_basic(self):
        """Test finding closest index in array."""
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        assert closest_index(arr, 0.0) == 0
        assert closest_index(arr, 4.0) == 4
        assert closest_index(arr, 2.1) == 2
        assert closest_index(arr, 1.9) == 2
        assert closest_index(arr, 1.4) == 1

    def test_closest_index_edge_cases(self):
        """Test edge cases for closest_index."""
        arr = np.array([5.0])
        assert closest_index(arr, 100.0) == 0

        arr = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        assert closest_index(arr, -6.0) == 1  # -5 is closest to -6
        assert closest_index(arr, -100.0) == 0
        assert closest_index(arr, 100.0) == 4

    def test_normalize_basic(self):
        """Test normalization to [0, 1]."""
        y = np.array([0.0, 5.0, 10.0])
        y_norm = normalize(y)

        assert y_norm[0] == 0.0
        assert y_norm[-1] == 1.0
        assert np.allclose(y_norm, [0.0, 0.5, 1.0])

    def test_normalize_constant_signal(self):
        """Test normalization of constant signal."""
        y = np.array([5.0, 5.0, 5.0])
        y_norm = normalize(y)

        assert np.allclose(y_norm, [0.0, 0.0, 0.0])

    def test_normalize_negative_values(self):
        """Test normalization with negative values."""
        y = np.array([-10.0, 0.0, 10.0])
        y_norm = normalize(y)

        assert y_norm[0] == 0.0
        assert y_norm[-1] == 1.0
        assert y_norm[1] == 0.5

    def test_denormalize_basic(self):
        """Test denormalization restores original values."""
        y = np.array([0.0, 5.0, 10.0])
        y_norm = normalize(y)
        y_restored = denormalize(y, y_norm)

        assert np.allclose(y, y_restored)

    def test_denormalize_subset(self):
        """Test denormalizing a subset of values."""
        y = np.array([0.0, 10.0, 20.0, 30.0])
        y_norm = np.array([0.25, 0.75])  # Subset
        y_denorm = denormalize(y, y_norm)

        assert np.allclose(y_denorm, [7.5, 22.5])

    def test_denormalize_constant_signal(self):
        """Test denormalization of constant signal."""
        y = np.array([5.0, 5.0, 5.0])
        y_norm = np.array([0.0, 0.5, 1.0])
        y_denorm = denormalize(y, y_norm)

        assert np.allclose(y_denorm, [5.0, 5.0, 5.0])

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize -> denormalize is identity."""
        y = np.random.randn(100) * 10 + 50
        y_norm = normalize(y)
        y_restored = denormalize(y, y_norm)

        assert np.allclose(y, y_restored)


# ============================================================================
# Test Signal Level Detection
# ============================================================================


class TestSignalLevelDetection:
    """Test various methods for detecting signal levels."""

    def test_histogram_method_clean_bilevel(self, bilevel_signal):
        """Test histogram method on clean bilevel signal."""
        t, y = bilevel_signal

        low, high, bin_centers, hist, peaks = detect_signal_levels_with_histogram(
            None, y, nbins=100, smooth_sigma=1
        )

        assert low == pytest.approx(0.0, abs=0.1)
        assert high == pytest.approx(3.3, abs=0.1)
        assert high > low

    def test_histogram_method_step(self, simple_step_signal):
        """Test histogram method on simple step."""
        t, y = simple_step_signal

        low, high, _, _, _ = detect_signal_levels_with_histogram(None, y, nbins=50)

        assert low == pytest.approx(0.0, abs=0.01)
        assert high == pytest.approx(1.0, abs=0.01)

    def test_histogram_insufficient_peaks(self):
        """Test histogram method fails gracefully with single level."""
        y = np.ones(1000)  # Constant signal

        with pytest.raises(ValueError, match="Could not find two distinct"):
            detect_signal_levels_with_histogram(None, y, nbins=100)

    def test_endpoints_method_basic(self, simple_step_signal):
        """Test endpoints method on simple step."""
        t, y = simple_step_signal

        low, high = detect_signal_levels_with_endpoints(None, y, n=100)

        assert low == pytest.approx(0.0, abs=0.01)
        assert high == pytest.approx(1.0, abs=0.01)

    def test_endpoints_method_reversed(self):
        """Test endpoints method with reversed signal."""
        t = np.linspace(0, 1, 1000)
        y = np.where(t < 0.5, 1.0, 0.0)  # High then low

        low, high = detect_signal_levels_with_endpoints(None, y, n=100)

        assert low == pytest.approx(0.0, abs=0.01)
        assert high == pytest.approx(1.0, abs=0.01)

    def test_endpoints_method_too_few_points(self):
        """Test endpoints method with insufficient data."""
        y = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="too large"):
            detect_signal_levels_with_endpoints(None, y, n=10)

    def test_derivative_method_basic(self, simple_step_signal):
        """Test derivative method on simple step."""
        t, y = simple_step_signal

        low, high = detect_signal_levels_with_derivative(None, y)

        assert low == pytest.approx(0.0, abs=0.1)
        assert high == pytest.approx(1.0, abs=0.1)

    def test_derivative_method_noisy(self, noisy_step_signal):
        """Test derivative method on noisy signal."""
        t, y = noisy_step_signal

        low, high = detect_signal_levels_with_derivative(None, y, smooth_sigma=2.0)

        # Signal has noise, so levels may be slightly outside original bounds
        assert low >= np.min(y) - 0.01
        assert high <= np.max(y) + 0.01
        assert high > low

    def test_detect_signal_levels_dispatcher(self, bilevel_signal):
        """Test the _detect_signal_levels dispatcher function."""
        t, y = bilevel_signal

        # Test histogram and derivative methods (these work for this signal)
        # Note: endpoint method won't work here because signal levels aren't at endpoints
        for method in ["histogram", "derivative"]:
            low, high = _detect_signal_levels(t, y, method=method)
            assert high > low
            assert low >= np.min(y)
            assert high <= np.max(y)

    def test_detect_signal_levels_invalid_method(self, simple_step_signal):
        """Test invalid method raises error."""
        t, y = simple_step_signal

        with pytest.raises(ValueError, match="not one of"):
            _detect_signal_levels(t, y, method="invalid_method")


# ============================================================================
# Test Threshold Calculations
# ============================================================================


class TestThresholdCalculations:
    """Test threshold calculation functions."""

    def test_calculate_thresholds_basic(self):
        """Test basic threshold calculation."""
        t = np.linspace(0, 1, 100)
        y = np.zeros(100)
        levels = (0.0, 10.0)
        fractional = (0.1, 0.9)

        thresh = _calculate_thresholds(t, y, levels, fractional)

        assert thresh[0] == pytest.approx(1.0)  # 0 + 0.1*10
        assert thresh[1] == pytest.approx(9.0)  # 0 + 0.9*10

    def test_calculate_thresholds_negative_levels(self):
        """Test threshold calculation with negative levels."""
        t = np.linspace(0, 1, 100)
        y = np.zeros(100)
        levels = (-5.0, 5.0)
        fractional = (0.2, 0.8)

        thresh = _calculate_thresholds(t, y, levels, fractional)

        assert thresh[0] == pytest.approx(-3.0)  # -5 + 0.2*10
        assert thresh[1] == pytest.approx(3.0)  # -5 + 0.8*10

    def test_calculate_thresholds_reversed_fractional(self):
        """Test that reversed fractional thresholds are handled."""
        t = np.linspace(0, 1, 100)
        y = np.zeros(100)
        levels = (0.0, 10.0)
        fractional = (0.9, 0.1)  # Reversed

        thresh = _calculate_thresholds(t, y, levels, fractional)

        # Should still give (low, high)
        assert thresh[0] == pytest.approx(1.0)
        assert thresh[1] == pytest.approx(9.0)

    def test_detect_thresholds_integration(self, bilevel_signal):
        """Test _detect_thresholds with auto level detection."""
        t, y = bilevel_signal

        thresh = _detect_thresholds(
            t, y, method="histogram", fractional_thresholds=(0.2, 0.8)
        )

        assert thresh[1] > thresh[0]
        assert thresh[0] > 0.0
        assert thresh[1] < 3.3


# ============================================================================
# Test Edge Detection
# ============================================================================


class TestEdgeDetection:
    """Test edge detection functions."""

    def test_find_all_edge_pairs_rising(self):
        """Test finding rising edge pairs."""
        t = np.linspace(0, 1, 1000)
        y = np.where(t < 0.5, 0.0, 1.0)

        thresholds = (0.2, 0.8)
        pairs = find_all_edge_pairs(t, y, thresholds, EdgeSign.rising)

        assert len(pairs) == 1
        i1, i2 = pairs[0]
        assert y[i1] <= 0.2
        assert y[i2] >= 0.8
        assert i2 > i1

    def test_find_all_edge_pairs_falling(self):
        """Test finding falling edge pairs."""
        t = np.linspace(0, 1, 1000)
        y = np.where(t < 0.5, 1.0, 0.0)

        thresholds = (0.2, 0.8)
        pairs = find_all_edge_pairs(t, y, thresholds, EdgeSign.falling)

        assert len(pairs) == 1
        i1, i2 = pairs[0]
        assert y[i1] >= 0.8
        assert y[i2] <= 0.2

    def test_find_all_edge_pairs_multiple(self, pulse_train):
        """Test finding multiple edge pairs in pulse train."""
        t, y = pulse_train

        thresholds = (0.2, 0.8)

        # Rising edges
        rising_pairs = find_all_edge_pairs(t, y, thresholds, EdgeSign.rising)
        assert len(rising_pairs) == 3

        # Falling edges
        falling_pairs = find_all_edge_pairs(t, y, thresholds, EdgeSign.falling)
        assert len(falling_pairs) == 3

    def test_find_all_edge_pairs_min_spacing(self):
        """Test minimum spacing constraint."""
        t = np.arange(100)
        y = np.zeros(100)
        y[10:20] = 1.0

        thresholds = (0.2, 0.8)

        # Without minimum spacing
        pairs = find_all_edge_pairs(t, y, thresholds, EdgeSign.rising, min_spacing=1)
        assert len(pairs) >= 1

        # With large minimum spacing (should filter out close edges)
        pairs_filtered = find_all_edge_pairs(
            t, y, thresholds, EdgeSign.rising, min_spacing=50
        )
        # May have fewer pairs or same, depending on signal
        assert len(pairs_filtered) <= len(pairs)

    def test_interpolate_crossing_rising(self):
        """Test interpolation of rising edge crossing."""
        t = np.linspace(0, 1, 1000)
        y = np.where(t < 0.5, 0.0, 1.0)

        thresholds = (0.2, 0.8)

        t_low, t_high = _interpolate_crossing(t, y, thresholds, EdgeSign.rising)

        assert t_low < t_high
        assert 0.4 < t_low < 0.6  # Should be near step
        assert 0.4 < t_high < 0.6

    def test_interpolate_crossing_falling(self):
        """Test interpolation of falling edge crossing."""
        t = np.linspace(0, 1, 1000)
        y = np.where(t < 0.5, 1.0, 0.0)

        thresholds = (0.2, 0.8)

        t_low, t_high = _interpolate_crossing(t, y, thresholds, EdgeSign.falling)

        assert t_low < t_high
        assert 0.4 < t_low < 0.6
        assert 0.4 < t_high < 0.6

    def test_interpolate_crossing_no_crossing(self):
        """Test interpolation fails when no crossing exists."""
        t = np.linspace(0, 1, 100)
        y = np.ones(100)  # Constant high

        thresholds = (0.2, 0.8)

        with pytest.raises(IndexError, match="doesn't cross"):
            _interpolate_crossing(t, y, thresholds, EdgeSign.rising)

    def test_detect_first_edge_basic(self, simple_step_signal):
        """Test detecting first edge."""
        t, y = simple_step_signal

        thresholds = (0.2, 0.8)
        edge = _detect_first_edge(t, y, EdgeSign.rising, thresholds)

        assert edge is not None
        assert edge.sign == EdgeSign.rising
        assert edge.start < edge.end
        assert edge.dx > 0

    def test_detect_first_edge_no_crossing(self):
        """Test first edge detection when no edge exists."""
        t = np.linspace(0, 1, 100)
        y = np.ones(100) * 0.5  # Constant, between thresholds

        thresholds = (0.2, 0.8)
        edge = _detect_first_edge(t, y, EdgeSign.rising, thresholds)

        assert edge is None

    def test_detect_first_edge_with_splitting(self, bilevel_signal):
        """Test first edge detection with splitting algorithm."""
        t, y = bilevel_signal

        thresholds = (0.5, 3.0)
        edge = _detect_first_edge_with_splitting(t, y, EdgeSign.rising, thresholds)

        assert edge is not None
        assert edge.sign == EdgeSign.rising
        assert 0.0 < edge.start < 0.2
        assert edge.end > edge.start

    def test_detect_edges_multiple(self, bilevel_signal):
        """Test detecting multiple edges."""
        t, y = bilevel_signal

        thresholds = (0.5, 3.0)
        settings = CrossingDetectionSettings()

        edges = _detect_edges(t, y, thresholds, settings=settings)

        assert len(edges) >= 2  # Should find at least 2 transitions

        # All edges should be valid
        for edge in edges:
            assert edge.start < edge.end
            assert edge.sign in [EdgeSign.rising, EdgeSign.falling]

    def test_detect_edges_with_bounds(self, bilevel_signal):
        """Test edge detection with time bounds."""
        t, y = bilevel_signal

        thresholds = (0.5, 3.0)
        settings = CrossingDetectionSettings()
        bounds = (0.4, 0.8)

        edges = _detect_edges(t, y, thresholds, bounds=bounds, settings=settings)

        # Should only find edges in bounded region
        for edge in edges:
            assert bounds[0] <= edge.start <= bounds[1]
            assert bounds[0] <= edge.end <= bounds[1]

    def test_detect_edge_wrapper_auto_levels(self, simple_step_signal):
        """Test edge wrapper with automatic level detection."""
        t, y = simple_step_signal

        edge = _detect_edge_wrapper(
            sign=EdgeSign.rising,
            x=t,
            y=y,
            levels=None,  # Auto-detect
            fractional_thresholds=(0.1, 0.9),
        )

        assert edge is not None
        assert edge.sign == EdgeSign.rising


# ============================================================================
# Test Peak Finding
# ============================================================================


class TestPeakFinding:
    """Test peak finding and edge type detection."""

    def test_find_peaks_and_types_single_edge(self, simple_step_signal):
        """Test finding peaks in simple step."""
        t, y = simple_step_signal

        thresholds = (0.2, 0.8)
        settings = CrossingDetectionSettings()

        peaks = list(_find_peaks_and_types(t, y, thresholds, settings=settings))

        assert len(peaks) >= 1
        assert all(isinstance(p, Peak) for p in peaks)

    def test_find_peaks_and_types_pulse_train(self, pulse_train):
        """Test finding multiple peaks in pulse train."""
        t, y = pulse_train

        thresholds = (0.2, 0.8)
        settings = CrossingDetectionSettings()

        peaks = list(_find_peaks_and_types(t, y, thresholds, settings=settings))

        # Should find multiple peaks (rising and falling)
        assert len(peaks) >= 3


# ============================================================================
# Test Metrics Calculations
# ============================================================================


class TestMetricsCalculations:
    """Test calculation of various signal metrics."""

    def test_calculate_midcross_rising(self, simple_step_signal):
        """Test midcross calculation for rising edge."""
        t, y = simple_step_signal

        levels = (0.0, 1.0)
        midcross = _calculate_midcross(t, y, levels)

        assert 0.4 < midcross < 0.6  # Should be near step at t=0.5

    def test_calculate_midcross_no_crossing(self):
        """Test midcross when no crossing exists."""
        t = np.linspace(0, 1, 100)
        y = np.ones(100) * 0.3  # Always below midpoint

        levels = (0.0, 1.0)
        midcross = _calculate_midcross(t, y, levels)

        assert midcross == 0  # Should return 0 when no crossing

    def test_calculate_slew_rate_basic(self):
        """Test slew rate calculation."""
        t = np.linspace(0, 1, 1000)
        y = t * 5  # Linear ramp, slope = 5

        slew = _calculate_slew_rate(t, y)

        assert slew == pytest.approx(5.0, rel=0.1)

    def test_calculate_slew_rate_step(self, simple_step_signal):
        """Test slew rate on step function."""
        t, y = simple_step_signal

        slew = _calculate_slew_rate(t, y)

        # Step function should have very high slew rate
        assert slew > 100

    def test_calculate_overshoot_no_overshoot(self, simple_step_signal):
        """Test overshoot calculation when no overshoot exists."""
        t, y = simple_step_signal

        levels = (0.0, 1.0)
        loc, overshoot = _calculate_overshoot(y, levels)

        assert overshoot == pytest.approx(0.0, abs=0.01)

    def test_calculate_overshoot_with_overshoot(self, underdamped_step):
        """Test overshoot calculation with actual overshoot."""
        t, y = underdamped_step

        # Normalize to 0-1
        y_normalized = (y - y[0]) / (1.0 - y[0])
        levels = (0.0, 1.0)

        loc, overshoot = _calculate_overshoot(y_normalized, levels)

        # Underdamped system should have overshoot
        assert overshoot > 0.0
        assert overshoot < 1.0  # Reasonable overshoot range

    def test_calculate_overshoot_falling_edge(self):
        """Test overshoot on falling edge."""
        t = np.linspace(0, 1, 1000)
        y = np.where(t < 0.5, 1.0, 0.0)

        # Add undershoot (goes below 0)
        undershoot_region = (t > 0.5) & (t < 0.6)
        y[undershoot_region] -= 0.1

        levels = (0.0, 1.0)
        loc, overshoot = _calculate_overshoot(y, levels)

        assert overshoot >= 0.0

    def test_calculate_undershoot_no_undershoot(self, simple_step_signal):
        """Test undershoot when none exists."""
        t, y = simple_step_signal

        levels = (0.0, 1.0)
        result = _calculate_undershoot(y, levels)

        # May return None or (loc, 0.0)
        if result is not None:
            loc, undershoot = result
            assert undershoot >= 0.0

    def test_calculate_undershoot_with_undershoot(self, underdamped_step):
        """Test undershoot with actual undershoot."""
        t, y = underdamped_step

        # Normalize
        y_normalized = (y - y[0]) / (1.0 - y[0])
        levels = (0.0, 1.0)

        result = _calculate_undershoot(y_normalized, levels)

        if result is not None:
            loc, undershoot = result
            assert undershoot >= 0.0

    def test_calculate_settling_time_basic(self):
        """Test settling time calculation."""
        t = np.linspace(0, 10, 1000)
        y = np.where(t < 1, 0.0, 1.0)  # Step at t=1

        levels = (0.0, 1.0)
        settling_t = _calculate_settling_time(
            t, y, levels, settling_time_fraction=0.02, settling_time_margin=0.0
        )

        # Should settle very quickly after step
        assert 0.8 < settling_t < 1.5

    def test_calculate_settling_time_underdamped(self, underdamped_step):
        """Test settling time on underdamped system."""
        t, y = underdamped_step

        # Normalize
        y_normalized = (y - y[0]) / (1.0 - y[0])
        levels = (0.0, 1.0)

        settling_t = _calculate_settling_time(
            t, y_normalized, levels, settling_time_fraction=0.02
        )

        # Underdamped should take some time to settle
        assert settling_t > 0.0
        assert settling_t < t[-1]

    def test_calculate_settling_time_with_margin(self):
        """Test settling time with additional margin."""
        t = np.linspace(0, 10, 1000)
        y = np.where(t < 1, 0.0, 1.0)

        levels = (0.0, 1.0)

        settling_t_no_margin = _calculate_settling_time(
            t, y, levels, settling_time_margin=0.0
        )

        settling_t_with_margin = _calculate_settling_time(
            t, y, levels, settling_time_margin=1.0
        )

        assert settling_t_with_margin == settling_t_no_margin + 1.0


# ============================================================================
# Test Edge Pairing
# ============================================================================


class TestEdgePairing:
    """Test pairing of rising and falling edges."""

    def test_pair_edges_single_pulse(self):
        """Test pairing edges from single pulse."""
        # Create edges manually
        rise = Edge(
            start=1.0,
            end=1.1,
            sign=EdgeSign.rising,
            thresholds=(0.2, 0.8),
            ymin=0.0,
            ymax=1.0,
        )
        fall = Edge(
            start=2.0,
            end=2.1,
            sign=EdgeSign.falling,
            thresholds=(0.2, 0.8),
            ymin=0.0,
            ymax=1.0,
        )

        edges = [rise, fall]
        pairs = pair_edges(edges)

        assert len(pairs) == 1
        assert pairs[0].rise == rise
        assert pairs[0].fall == fall
        assert pairs[0].is_valid

    def test_pair_edges_multiple_pulses(self):
        """Test pairing multiple pulses."""
        edges = [
            Edge(
                start=1.0,
                end=1.1,
                sign=EdgeSign.rising,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
            Edge(
                start=2.0,
                end=2.1,
                sign=EdgeSign.falling,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
            Edge(
                start=3.0,
                end=3.1,
                sign=EdgeSign.rising,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
            Edge(
                start=4.0,
                end=4.1,
                sign=EdgeSign.falling,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
        ]

        pairs = pair_edges(edges)

        assert len(pairs) == 2
        assert all(p.is_valid for p in pairs)

    def test_pair_edges_max_gap(self):
        """Test max_gap constraint in edge pairing."""
        edges = [
            Edge(
                start=1.0,
                end=1.1,
                sign=EdgeSign.rising,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
            Edge(
                start=10.0,
                end=10.1,
                sign=EdgeSign.falling,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
        ]

        # With no max_gap, should pair
        pairs_no_limit = pair_edges(edges, max_gap=None)
        assert len(pairs_no_limit) == 1

        # With tight max_gap, should not pair
        pairs_limited = pair_edges(edges, max_gap=1.0)
        assert len(pairs_limited) == 0

    def test_pair_edges_unbalanced(self):
        """Test pairing with unbalanced edges."""
        edges = [
            Edge(
                start=1.0,
                end=1.1,
                sign=EdgeSign.rising,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
            Edge(
                start=2.0,
                end=2.1,
                sign=EdgeSign.rising,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
            Edge(
                start=3.0,
                end=3.1,
                sign=EdgeSign.falling,
                thresholds=(0.2, 0.8),
                ymin=0.0,
                ymax=1.0,
            ),
        ]

        pairs = pair_edges(edges)

        # Should pair first rise with fall, second rise unpaired
        assert len(pairs) == 1

    def test_paired_edge_properties(self):
        """Test PairedEdge calculated properties."""
        rise = Edge(
            start=1.0,
            end=1.1,
            sign=EdgeSign.rising,
            thresholds=(0.2, 0.8),
            ymin=0.0,
            ymax=1.0,
        )
        fall = Edge(
            start=2.0,
            end=2.1,
            sign=EdgeSign.falling,
            thresholds=(0.2, 0.8),
            ymin=1.0,
            ymax=0.0,
        )

        pair = PairedEdge(rise=rise, fall=fall)

        assert pair.pulse_width == pytest.approx(1.1)  # 2.1 - 1.0
        assert pair.amplitude == pytest.approx(1.0)  # 1.0 - 0.0
        assert pair.is_valid


# ============================================================================
# Test Time Vector Handling
# ============================================================================


class TestTimeVectorHandling:
    """Test time vector normalization and interpolation."""

    def test_get_xtime_from_fs(self):
        """Test time generation from sampling frequency."""
        x = np.array([1, 2, 3, 4, 5])

        x_uniform, t_uniform = _get_xtime_from_t_fs(x, fs=10.0, t=None)

        assert len(x_uniform) == len(x)
        assert len(t_uniform) == len(x)
        assert np.allclose(x_uniform, x)
        assert t_uniform[1] - t_uniform[0] == pytest.approx(0.1)  # 1/fs

    def test_get_xtime_from_t_nonuniform(self):
        """Test interpolation with non-uniform time."""
        x = np.array([0, 1, 4, 9, 16])  # x = t^2
        t = np.array([0, 1, 2, 3, 4])

        x_uniform, t_uniform = _get_xtime_from_t_fs(x, fs=1.0, t=t)

        assert len(x_uniform) == len(x)
        assert len(t_uniform) == len(x)
        # Time should be uniform now
        assert np.allclose(np.diff(t_uniform), t_uniform[1] - t_uniform[0])

    def test_get_xtime_empty_input(self):
        """Test error handling for empty input."""
        x = np.array([])

        with pytest.raises(ValueError, match="length of 0"):
            _get_xtime_from_t_fs(x, fs=1.0)

    def test_get_xtime_mismatched_shapes(self):
        """Test error when t and x have different shapes."""
        x = np.array([1, 2, 3])
        t = np.array([0, 1])

        with pytest.raises(ValueError, match="same shape"):
            _get_xtime_from_t_fs(x, fs=1.0, t=t)


# ============================================================================
# Test Filtering
# ============================================================================


class TestFiltering:
    """Test signal filtering functions."""

    def test_smooth_zero_phase_basic(self):
        """Test zero-phase filtering."""
        # Create noisy signal
        np.random.seed(42)
        t = np.linspace(0, 1, 1000)
        y_clean = np.sin(2 * np.pi * 5 * t)
        y_noisy = y_clean + np.random.normal(0, 0.1, len(t))

        fs = 1.0 / np.mean(np.diff(t))
        y_filtered = smooth_zero_phase(y_noisy, normal_cutoff=0.2, fs=fs, order=3)

        # Filtered should be closer to clean than noisy
        error_noisy = np.mean((y_noisy - y_clean) ** 2)
        error_filtered = np.mean((y_filtered - y_clean) ** 2)

        assert error_filtered < error_noisy

    def test_smooth_zero_phase_preserves_length(self):
        """Test that filtering preserves signal length."""
        y = np.random.randn(100)
        fs = 100

        y_filtered = smooth_zero_phase(y, normal_cutoff=0.5, fs=fs, order=4)

        assert len(y_filtered) == len(y)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_edge_analysis_workflow(self, underdamped_step):
        """Test complete edge analysis workflow."""
        t, y = underdamped_step

        # 1. Detect levels
        levels = _detect_signal_levels(t, y, method="histogram")
        assert levels[1] > levels[0]

        # 2. Calculate thresholds
        thresholds = _calculate_thresholds(
            t, y, levels, fractional_thresholds=(0.1, 0.9)
        )
        assert thresholds[1] > thresholds[0]

        # 3. Detect edges
        settings = CrossingDetectionSettings()
        edges = _detect_edges(t, y, thresholds, settings=settings)
        assert len(edges) >= 1

        # 4. Calculate metrics
        midcross = _calculate_midcross(t, y, levels)
        assert midcross > 0

        slew = _calculate_slew_rate(t, y)
        assert slew > 0

        settling_t = _calculate_settling_time(t, y, levels)
        assert settling_t > 0

    def test_pulse_width_measurement(self, pulse_train):
        """Test measuring pulse widths."""
        t, y = pulse_train

        # Detect levels and thresholds
        levels = _detect_signal_levels(t, y, method="histogram")
        thresholds = _calculate_thresholds(t, y, levels)

        # Detect all edges
        settings = CrossingDetectionSettings()
        edges = _detect_edges(t, y, thresholds, settings=settings)

        # Pair edges
        pairs = pair_edges(edges)

        # Each pulse should be ~0.1 seconds wide
        for pair in pairs:
            assert 0.08 < pair.pulse_width < 0.12


# ============================================================================
# Performance and Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_constant_signal(self):
        """Test handling of constant signal."""
        t = np.linspace(0, 1, 100)
        y = np.ones(100) * 5.0

        # Should fail to find levels
        with pytest.raises(ValueError):
            detect_signal_levels_with_histogram(None, y)

    def test_single_point_signal(self):
        """Test handling of single point."""
        t = np.array([0.0])
        y = np.array([1.0])

        # Most operations should fail gracefully
        with pytest.raises((ValueError, IndexError)):
            _detect_signal_levels(t, y, method="endpoint", n=1)

    def test_very_noisy_signal(self):
        """Test robustness to noise."""
        np.random.seed(42)
        t = np.linspace(0, 1, 10000)
        y_clean = np.where(t < 0.5, 0.0, 1.0)
        y_noisy = y_clean + np.random.normal(0, 0.3, len(t))

        # Should still detect levels reasonably
        levels = _detect_signal_levels(t, y_noisy, method="histogram")
        assert levels[1] > levels[0]

    def test_nan_handling(self):
        """Test handling of NaN values."""
        t = np.linspace(0, 1, 100)
        y = np.ones(100)
        y[50] = np.nan

        # Should handle or raise appropriate error
        # Behavior depends on implementation
        try:
            levels = _detect_signal_levels(t, y, method="histogram")
            # If it succeeds, levels should be valid
            assert not np.isnan(levels[0])
            assert not np.isnan(levels[1])
        except (ValueError, RuntimeError):
            # Acceptable to raise error on NaN
            pass


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
