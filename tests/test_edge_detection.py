import unittest
from photoreceiver_analysis.transient_response import get_edges, Edge, find_reference_levels, _interpolate_crossing, _find_peaks_and_types
from photoreceiver_analysis import transient_response
from scipy.signal import find_peaks, peak_widths
from scipy.signal import lti, step
import logging
import numpy as np

from matplotlib import pyplot as plt

log = logging.getLogger("testing")

class TestEdgeDetection(unittest.TestCase):
    def test_single_rising_edge(self):
        x = np.linspace(0, 10, 1000)
        y = np.where(x > 5, 1.0, 0.0)
        levels = find_reference_levels(x, y)
        edges = get_edges(x, y, levels)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].type, 'rise')
        self.assertAlmostEqual(edges[0].start, 5.0, places=1)

    def test_single_falling_edge(self):
        x = np.linspace(0, 10, 1000)
        y = np.where(x < 5, 1.0, 0.0)
        levels = find_reference_levels(x, y)
        edges = get_edges(x, y, levels)
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].type, 'fall')
        self.assertAlmostEqual(edges[0].end, 5.0, places=1)

    def test_multiple_edges(self):
        x = np.linspace(0, 10, 1000)
        y = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 0.5 * x)))
        levels = find_reference_levels(x, y)
        edges = get_edges(x, y, levels)
        self.assertGreaterEqual(len(edges), 8)

    def test_rising_falling_symmetry(self):
        damping = 0.5  # Damping ratio < 1 for overshoot
        omega = 2 * np.pi * 1  # Natural frequency (1 Hz)
        # Define the LTI system
        system = lti([omega**2], [1, 2*damping*omega, omega**2])
        x = np.linspace(-1, 5, 20000)
        _, yp = step(system, T=x[x>=0])
        y = [0]*len(x[x<0])
        y.extend(yp)
        y = np.array(y)
        levels = transient_response.find_reference_levels(x, y)
        edges = get_edges(x, y, levels)

        y_m = -y
        levels_m = transient_response.find_reference_levels(x, y_m)
        edges_m = get_edges(x, y_m, levels_m)

        self.assertAlmostEqual(max(levels), -1*min(levels_m), 3)
        self.assertAlmostEqual(min(levels), -1*max(levels_m), 3)
        for p, m in zip(edges, edges_m):
            self.assertAlmostEqual(p.start, m.start, 3)

class TestDetectLevels(unittest.TestCase):

    def test_two_level_signal(self):
        # Known low and high levels
        low_val = 0.0
        high_val = 1.0
        num_samples = 10000

        signal = np.concatenate([
            np.full(num_samples, low_val),
            np.full(num_samples, high_val)
        ])

        y_norm = transient_response.normalize(signal)
        low, high, *_ = transient_response.detect_levels(y_norm, bins=100)

        self.assertAlmostEqual(low, low_val, places=2)
        self.assertAlmostEqual(high, high_val, places=2)

    def test_two_level_signal_with_noise(self):
        low_val = 0.1
        high_val = 0.9
        noise_std = 0.02
        num_samples = 100000

        signal = np.concatenate([
            np.random.normal(low_val, noise_std, num_samples),
            np.random.normal(high_val, noise_std, num_samples)
        ])
        y_norm = transient_response.normalize(signal)
        low, high, *_ = transient_response.detect_levels(y_norm, bins=100)
        self.assertAlmostEqual(low, low_val, delta=0.05)
        self.assertAlmostEqual(high, high_val, delta=0.05)

    def test_single_level_signal_should_fail(self):
        signal = np.full(2000, 0.5)

        with self.assertRaises(ValueError):
            y_norm = transient_response.normalize(signal)
            low, high, *_ = transient_response.detect_levels(y_norm)

    def test_three_level_signal_should_return_two(self):
        # Should return the two most prominent levels
        levels = [0.0, 0.5, 1.0]
        counts = [10000, 300, 10000]  # Middle level is less prominent
        signal = np.concatenate([
            np.full(counts[0], levels[0]),
            np.full(counts[1], levels[1]),
            np.full(counts[2], levels[2])
        ])
        y_norm = transient_response.normalize(signal)
        low, high, *_ = transient_response.detect_levels(y_norm, bins=100)
        log.error(f"{low} {high}")
        self.assertAlmostEqual(low, min(levels), places=2)
        self.assertAlmostEqual(high, max(levels), places=2)

if __name__ == '__main__':
    logging.basicConfig()
    log.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("photoreceiver_analysis").setLevel(logging.DEBUG)

    unittest.main()
