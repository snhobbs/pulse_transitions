import logging
import unittest
from scipy.signal import find_peaks
from scipy.signal import lti, step, lsim
import numpy as np
from pulse_transitions.transient_response import detect_edges, detect_thresholds, slew_rate, settling_time
from pulse_transitions.impl import _interpolate_crossing, _find_peaks_and_types
from pulse_transitions.common import CrossingDetectionSettings, Edge, EdgeSign

import numpy as np
from pulse_transitions import (
    statelevels, risetime, falltime,
    midcross, overshoot, undershoot
)


log = logging.getLogger("testing")


def simulate_second_order_step(
    t: np.ndarray,
    zeta: float = 0.2,
    omega_n: float = 25.0,
) -> np.ndarray:
    """Simulate underdamped second-order step response using scipy.signal."""
    # System: H(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)
    num = [omega_n**2]
    den = [1, 2 * zeta * omega_n, omega_n**2]
    system = lti(num, den)

    # Step input (u(t))
    u = np.ones_like(t)

    # Simulate response
    tout, y, _ = lsim(system, U=u, T=t)
    return y

@unittest.skip("No fitering in the impl")
class TestFindPeaksAndTypes(unittest.TestCase):
    def setUp(self):
        self.config = CrossingDetectionSettings(window=10)

    def test_single_rising_peak_crossing_thresholds(self):
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        y[40:60] = np.linspace(0.0, 1.0, 20)  # Rising slope
        y[60:] = 1.0

        peaks = _find_peaks_and_types(x, y, thresholds=[0.1, 0.9], settings=self.config)

        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][1], 'rise')

    def test_single_falling_peak_crossing_thresholds(self):
        x = np.linspace(0, 1, 100)
        y = np.ones_like(x)
        y[40:60] = np.linspace(1.0, 0.0, 20)  # Falling slope
        y[60:] = 0.0

        peaks = _find_peaks_and_types(x, y, thresholds=[0.1, 0.9], settings=self.config)

        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][1], 'fall')

    def test_rise_and_fall_peaks(self):
        x = np.linspace(0, 2, 200)
        y = np.concatenate([
            np.linspace(0.0, 1.0, 50),    # Rise
            np.linspace(1.0, 0.0, 50),    # Fall
            np.linspace(0.0, 1.0, 50),    # Rise again
            np.linspace(1.0, 0.0, 50)     # Fall again
        ])

        peaks = _find_peaks_and_types(x, y, thresholds=[0.2, 0.8], settings=self.config)

        self.assertEqual(len(peaks), 4)
        rise_count = sum(1 for _, typ, _ in peaks if typ == 'rise')
        fall_count = sum(1 for _, typ, _ in peaks if typ == 'fall')

        self.assertEqual(rise_count, 2)
        self.assertEqual(fall_count, 2)

    def test_peak_without_crossing_thresholds_is_ignored(self):
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        y[40:60] = np.linspace(0.4, 0.5, 20)  # Peak does not cross high threshold
        y[60:] = 0.5

        peaks = _find_peaks_and_types(x, y, thresholds=[0.1, 0.8], settings=self.config)
        self.assertEqual(len(peaks), 0)

    def test_small_window_misses_crossing(self):
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        y[45:55] = np.linspace(0.0, 1.0, 10)  # Very narrow rise
        y[55:] = 1.0

        # Window is too small to see full crossing

        config = CrossingDetectionSettings(window=2)
        peaks = _find_peaks_and_types(x, y, thresholds=[0.1, 0.9], settings=config)
        self.assertEqual(len(peaks), 0)

        # Window large enough to capture crossing
        peaks = _find_peaks_and_types(x, y, thresholds=[0.1, 0.9], settings=self.config)
        self.assertEqual(len(peaks), 1)

    def test_min_separation_filters_close_peaks(self):
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        y[20:30] = np.linspace(0.0, 1.0, 10)
        y[30:40] = np.linspace(1.0, 0.0, 10)
        y[40:50] = np.linspace(0.0, 1.0, 10)
        y[50:] = 1.0

        config = CrossingDetectionSettings(window=10, min_separation=0.2)
        peaks = _find_peaks_and_types(x, y, thresholds=[0.1, 0.9], settings=config)

        # Expect only 1 rising edge due to min_separation filtering
        self.assertEqual(len(peaks), 1)
        self.assertEqual(peaks[0][1], 'rise')


class TestWaveformMetrics(unittest.TestCase):

    def setUp(self):
        self.t = np.linspace(0, 1, 1000)
        self.rising = np.where(self.t >= 0.5, 1.0, 0.0)
        self.falling = np.where(self.t >= 0.5, 0.0, 1.0)

    def test_statelevels_rising(self):
        levels, bins, hist = statelevels(self.rising)
        low, high = levels
        self.assertLess(low, high)
        self.assertAlmostEqual(0, low, 2)
        self.assertAlmostEqual(1, high, 2)
        self.assertEqual(len(bins), len(hist))

    def test_risetime(self):
        edge = risetime(x=self.rising, fs=len(self.t), t=self.t)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.sign, EdgeSign.rising)
        self.assertTrue(0.4 < edge.start < edge.end < 0.6)

    def test_falltime(self):
        edge = falltime(x=self.falling, fs=len(self.t), t=self.t)
        self.assertIsNotNone(edge)
        self.assertEqual(edge.sign, EdgeSign.falling)
        self.assertTrue(0.4 < edge.start < edge.end < 0.6)

    def test_midcross_rising(self):
        mid = midcross(x=self.rising, fs=len(self.t), t=self.t)
        self.assertTrue(0.4 < mid < 0.6)

    def test_midcross_falling(self):
        mid = midcross(x=self.falling, fs=len(self.t), t=self.t)
        self.assertTrue(0.4 < mid < 0.6)

    def test_overshoot_clean_step(self):
        self.assertAlmostEqual(overshoot(self.rising), 0.0, places=6)

    def test_undershoot_clean_step(self):
        self.assertAlmostEqual(undershoot(self.rising), 0.0, places=2)

    def test_overshoot_detected(self):
        t = np.linspace(0, 1, 1000)
        rising_overshoot = np.where(t >= 0.5, 1.0, 0.0) + np.where((t >= 0.5) & (t <= 0.6), 0.2, 0.0)
        rising_overshoot_undershoot = rising_overshoot + np.where((t >= 0.55) & (t <= 0.65), -0.2, 0)
        un = overshoot(rising_overshoot_undershoot)
        self.assertAlmostEqual(un, 0.2, 2)

    def test_undershoot_detected(self):
        t = np.linspace(0, 1, 1000)
        rising_overshoot = np.where(t >= 0.5, 1.0, 0.0) + np.where((t >= 0.5) & (t <= 0.6), 0.2, 0.0)
        rising_overshoot_undershoot = rising_overshoot + np.where((t >= 0.55) & (t <= 0.65), -0.2, 0)
        un = undershoot(rising_overshoot_undershoot)
        self.assertAlmostEqual(un, 0.2, 2)


class TestSecondOrderSystem(unittest.TestCase):

    def setUp(self):
        self.t = np.linspace(-1, 1.0, 2000)
        self.y = [0]*len(np.where(self.t < 0)[0]) + list(simulate_second_order_step(self.t[np.where(self.t >= 0)[0]], zeta=0.2, omega_n=25.0))

    def test_statelevels(self):
        levels, bins, hist = statelevels(self.y)
        low, high = levels
        self.assertLess(low, high)
        self.assertAlmostEqual(low, 0.0, delta=0.1)
        self.assertAlmostEqual(high, 1.0, delta=0.1)

    def test_risetime(self):
        edge = risetime(x=self.y, fs=len(self.t), t=self.t)
        self.assertEqual(edge.sign, EdgeSign.rising)
        self.assertTrue(0.01 < edge.start < edge.end < 0.3)

    def test_midcross(self):
        mc = midcross(x=self.y, fs=len(self.t), t=self.t)
        self.assertTrue(0.01 < mc < 0.3)

    def test_overshoot(self):
        ov = overshoot(self.y)
        self.assertGreater(ov, 0.5)
        self.assertLess(ov, 0.6)  # should overshoot but not too much

    def test_undershoot(self):
        un = undershoot(self.y)
        self.assertGreater(un, 0.4)
        self.assertLess(un, 0.6)  # should but not too much


class TestSlewRateSimple(unittest.TestCase):
    def test_zero_slew_rate_constant_signal(self):
        x = np.full(100, 5.0)
        sr = slew_rate(x)
        self.assertEqual(sr, 0)

    def test_positive_slew_rate_linear_rise(self):
        x = np.linspace(0, 1000, 100)
        sr = slew_rate(x)
        self.assertAlmostEqual(sr, 10, delta=1)

    def test_error_on_empty_input(self):
        with self.assertRaises(ValueError):
            slew_rate(np.array([]))

    def test_nonzero_slew_rate_sine_wave(self):
        t = np.linspace(0, 1, 1000)
        x = np.sin(2 * np.pi * 5 * t)
        sr = slew_rate(x)
        self.assertGreater(sr, 0)


class TestSettlingTime(unittest.TestCase):
    def test_settling_time_simple_step(self):
        # Signal: step from 0 to 1 at t=5, settles immediately after t=6
        t = np.linspace(0, 10, 1000)
        x = np.zeros_like(t)
        t_rise = 5
        t_overshoot_margin = 0.1
        x[t >= t_rise] = 1.0
        # Add a brief overshoot before settling
        x[(t > t_rise) & (t < t_rise+t_overshoot_margin)] = 1.1
        settling = settling_time(x, d=0.05, fs=100, t=t)
        self.assertAlmostEqual(settling, 5.1, 2)
        self.assertLessEqual(settling, 6.0)

    def test_settling_time_no_settle(self):
        # Signal oscillates outside bounds forever
        t = np.linspace(0, 1, 100)
        x = 1 + 0.1 * np.sin(50 * np.pi * t)
        settling = settling_time(x, d=0.01, fs=100, t=t)
        self.assertAlmostEqual(settling, max(t), 1)

    def test_settling_time_with_margin(self):
        # Simple step with settling margin
        t = np.linspace(0, 10, 1000)
        x = np.zeros_like(t)
        x[t >= 3] = 1
        settling = settling_time(x, d=0.05, fs=100, t=t, settling_time_margin=0.5)
        self.assertGreaterEqual(settling, 3)
        self.assertLessEqual(settling, 10.5)

class TestInterpolateCrossing(unittest.TestCase):
    def setUp(self):
        self.x = np.array(list(range(10000)))

    def test_rising_edge_no_window(self):
        thresholds = (1, 2)

        y = np.linspace(0, 3, len(self.x))
        start, end = _interpolate_crossing(self.x, y, thresholds, sign=EdgeSign.rising)
        self.assertLess(start, end)
        self.assertAlmostEqual(np.interp(start, self.x, y), min(thresholds), delta=1e-3)
        self.assertAlmostEqual(np.interp(end, self.x, y), max(thresholds), delta=1e-3)


    def test_falling_edge_no_window(self):
        thresholds = (1, 2)

        y = np.linspace(3, 0, len(self.x))
        start, end = _interpolate_crossing(self.x, y, thresholds, sign=EdgeSign.falling)
        self.assertLessEqual(start, end)
        self.assertAlmostEqual(np.interp(start, self.x, y), max(thresholds), delta=1e-3)
        self.assertAlmostEqual(np.interp(end, self.x, y), min(thresholds), delta=1e-3)


if __name__ == '__main__':
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("pulse_transitions").setLevel(logging.DEBUG)

    unittest.main()
