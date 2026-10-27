"""Tests for pulse_transitions._levels — derived from: pulse_transitions Expanded Spec (v0.3), state_levels (§5.2)."""
import unittest

import numpy as np

from pulse_transitions._levels import detect_levels
from pulse_transitions._types import LevelError, StateLevels

from tests.signals import two_level_noisy

Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5])


class TestDetectLevels(unittest.TestCase):
    def assertLevels(self, result, low, high, method, clause, delta=1e-12):
        self.assertIsInstance(result, StateLevels, msg=f"{clause}: expected StateLevels, got {result!r}")
        self.assertAlmostEqual(result.low, low, delta=delta, msg=f"{clause}: low is {result.low!r}, spec says {low}")
        self.assertAlmostEqual(result.high, high, delta=delta, msg=f"{clause}: high is {result.high!r}, spec says {high}")
        self.assertEqual(result.method, method, msg=f"{clause}: method is {result.method!r}")

    def test_detect_levels_mode(self):
        self.assertLevels(detect_levels(Y, bins=4), 0.0, 1.0, "mode",
                          "spec state_levels / D2: mode = mean of the samples in each half's modal bin")

    def test_detect_levels_mean(self):
        self.assertLevels(detect_levels(Y, method="mean", bins=4), 0.0, 0.9166666666666666, "mean",
                          "spec state_levels: mean of all samples in each sub-histogram (upper half holds the 0.5 and five 1s)")

    def test_detect_levels_peak(self):
        self.assertLevels(detect_levels(np.array([0.0, 3, -1, 2]), method="peak"), -1.0, 3.0, "peak",
                          "spec §5.2.2: peak = (y_min, y_max)")

    def test_detect_levels_endpoints(self):
        self.assertLevels(detect_levels(np.array([1.0, 1, 5, 0, 0]), method="endpoints", endpoint_samples=2), 0.0, 1.0,
                          "endpoints", "spec §5.2.2: means of the first and last n samples, sorted")

    def test_detect_levels_mode_ties_take_first_bin(self):
        y = np.array([0, 0, 0.3, 0.3, 1, 1], dtype=float)
        self.assertLevels(detect_levels(y, bins=4), 0.0, 1.0, "mode",
                          "spec state_levels: first bin on ties (0s and 0.3s tie in the lower half)")

    def test_detect_levels_histogram_shapes(self):
        result = detect_levels(Y, bins=4)
        self.assertEqual(len(result.histogram), 4, msg="spec Types: histogram holds the counts of `bins` bins")
        np.testing.assert_allclose(result.bin_edges, [0, 0.25, 0.5, 0.75, 1.0], err_msg="spec state_levels: bins over [y_min, y_max]")
        np.testing.assert_array_equal(result.histogram, [4, 0, 1, 5], err_msg="spec state_levels: last bin closed")

    def test_detect_levels_peak_has_no_histogram(self):
        self.assertIsNone(detect_levels(np.array([0.0, 1.0]), method="peak").histogram,
                          msg="spec Types: histogram only for the histogram methods")

    def test_detect_levels_constant_raises(self):
        with self.assertRaises(LevelError, msg="spec §E7: a constant record has no two levels"):
            detect_levels(np.array([2.0, 2.0, 2.0]))

    def test_detect_levels_noisy_two_level_accuracy(self):
        result = detect_levels(two_level_noisy())
        self.assertAlmostEqual(result.low, 0.0, delta=0.01, msg=f"spec §C13: mode within 1 % of the amplitude, got {result!r}")
        self.assertAlmostEqual(result.high, 1.0, delta=0.01, msg=f"spec §C13: mode within 1 % of the amplitude, got {result!r}")


if __name__ == "__main__":
    unittest.main()
