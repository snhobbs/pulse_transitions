"""Tests for pulse_transitions._plateau — derived from: pulse_transitions Expanded Spec (v0.3), flatness / §C21."""
import math
import unittest

import numpy as np

from pulse_transitions._plateau import flatness_of


def plateau():
    y = np.full(20, 2.0)
    y[5:10] = [2.0, 2.2, 1.8, 2.0, 2.0]
    return np.arange(20, dtype=float), y


class TestFlatnessOf(unittest.TestCase):
    def test_flatness_of_normalized(self):
        x, y = plateau()
        self.assertAlmostEqual(flatness_of(x, y, (5, 9)), 0.0632455532, delta=1e-9,
                               msg="spec flatness / §C21: population std 0.1264911064 over |final level| 2")

    def test_flatness_of_raw(self):
        x, y = plateau()
        self.assertAlmostEqual(flatness_of(x, y, (5, 9), normalize=False), 0.1264911064, delta=1e-9,
                               msg="spec flatness / §C21: population std (ddof=0)")

    def test_flatness_of_too_few_samples(self):
        x, y = plateau()
        self.assertTrue(math.isnan(flatness_of(x, y, (5, 8))), msg="spec flatness / D9: fewer than 5 samples → NaN")

    def test_flatness_of_zero_final_level(self):
        x = np.arange(20, dtype=float)
        self.assertTrue(math.isnan(flatness_of(x, np.zeros(20), (0, 19))), msg="spec flatness / D9: final level < 1e-12 → NaN")

    def test_flatness_of_uses_whole_record_final_level(self):
        x, y = plateau()
        y[-10:] = 4.0  # final level 4 lies outside the window
        self.assertAlmostEqual(flatness_of(x, y, (5, 9)), 0.1264911064 / 4.0, delta=1e-9,
                               msg="spec §C21: normalised by the last 10 samples of the whole record, not of the window")


if __name__ == "__main__":
    unittest.main()
