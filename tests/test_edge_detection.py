import logging
import unittest
from scipy.signal import find_peaks
from scipy.signal import lti, step
import numpy as np
from pulse_transitions.transient_response import detect_edges, detect_thresholds
from pulse_transitions.impl import _interpolate_crossing, _find_peaks_and_types
from pulse_transitions.common import CrossingDetectionSettings, Edge, EdgeSign

log = logging.getLogger("testing")

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


class TestDetectEdges(unittest.TestCase):
    def setUp(self):
        self.config = CrossingDetectionSettings(window=0)

    def test_detect_single_rising_edge(self):
        x = np.linspace(0, 1, 100)
        y = np.zeros_like(x)
        y[40:60] = np.linspace(0.0, 1.0, 20)
        y[60:] = 1.0

        edges = detect_edges(x, y, thresholds=(0.1, 0.9), settings=self.config)
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.type, 'rise')
        self.assertTrue(0.1 <= edge.ymin < 0.5)
        self.assertTrue(0.5 < edge.ymax <= 1.0)

    def test_detect_multiple_edges(self):
        x = np.linspace(0, np.pi*20, 100000)
        y = np.sin(x)

        edges = detect_edges(x, y, thresholds=(0.2, 0.8), settings=self.config)
        self.assertEqual(len(edges), 4)
        rise_count = sum(1 for e in edges if e.type == 'rise')
        fall_count = sum(1 for e in edges if e.type == 'fall')

        self.assertEqual(rise_count, 2)
        self.assertEqual(fall_count, 2)

    def test_no_edges_detected(self):
        x = np.linspace(0, 1, 100)
        y = np.full_like(x, 0.5)
        edges = detect_edges(x, y, thresholds=(0.1, 0.9), settings=self.config)
        self.assertEqual(len(edges), 0)

    def test_bounds_limit_detection(self):
        y = np.concatenate([
            np.linspace(0.0, 1.0, 50),
            np.linspace(1.0, 0.0, 50),
            np.linspace(0.0, 1.0, 50),
            np.linspace(1.0, 0.0, 50)
        ])
        x = np.linspace(0, 2, len(y))

        edges = detect_edges(x, y, thresholds=(0.2, 0.8), bounds=(0.0, 2.0), settings=self.config)
        self.assertEqual(len(edges), 2)  # Only the first rise and fall should be in bounds



'''
def _interpolate_crossing(x: np.ndarray, y: np.ndarray, idx: int, thresholds: Tuple[float, float], sign: int, *, window: int = None):

    Interpolate precise crossing times for low and high thresholds around a peak with optional hysteresis window.

    Args:
        x (np.ndarray): Time array.
        y (np.ndarray): Signal array.
        idx (int): Peak index in the arrays.
        thresholds (float,float): Threshold crossing values.
        sign (int): +1 for rising pulse, -1 for falling pulse.
        window (int, optional): Number of samples before/after peak to limit search.

    Returns:
        tuple: (start_time, end_time) for lo_val and hi_val crossings, in time order.
'''


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
