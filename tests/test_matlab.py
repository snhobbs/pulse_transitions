"""Tests for pulse_transitions.matlab — derived from: pulse_transitions Expanded Spec (v0.3), matlab facade / §C27."""
import unittest

import numpy as np

import pulse_transitions as pt
from pulse_transitions import InputError, Polarity, matlab

from tests.signals import OVERSHOOT

Y = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0], dtype=float)


def close(actual, expected, clause):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=0, atol=1e-9, err_msg=clause)


class TestMatlabExamples(unittest.TestCase):
    def test_statelevels(self):
        levels, histogram, centers = matlab.statelevels(Y)
        close(levels, [0.0, 1.0], "spec matlab example: statelevels(y)[0] → [0, 1]")
        self.assertEqual((len(histogram), len(centers)), (100, 100), msg="spec matlab: histogram and bin centres of nbins=100")

    def test_midcross(self):
        close(matlab.midcross(Y), [1.5, 4.5, 6.5, 9.5], "spec matlab example: midcross")
        close(matlab.midcross(Y, fs=2.0), [0.75, 2.25, 3.25, 4.75], "spec matlab: fs scales the instants")
        close(matlab.midcross(Y, np.arange(11) * 2.0), [3.0, 9.0, 13.0, 19.0], "spec matlab: t is positional, as in MATLAB")

    def test_risetime(self):
        r, lt, ut, ll, ul = matlab.risetime(Y)
        close(r, [0.8, 0.8], "spec matlab example: risetime durations")
        close(lt, [1.1, 6.1], "spec matlab example: risetime lower instants")
        close(ut, [1.9, 6.9], "spec matlab example: risetime upper instants")
        close([ll, ul], [0.1, 0.9], "spec matlab example: risetime reference levels")

    def test_falltime(self):
        f, lt, ut, ll, ul = matlab.falltime(Y)
        close(f, [0.8, 0.8], "spec matlab: falltime durations")
        close(lt, [4.9, 9.9], "spec matlab: falltime lower-level instants")
        close(ut, [4.1, 9.1], "spec matlab: falltime upper-level instants")

    def test_slewrate(self):
        close(matlab.slewrate(Y), [1.0, -1.0, 1.0, -1.0], "spec matlab example: signed slew rates")

    def test_settlingtime(self):
        close(matlab.settlingtime(Y, 2), [0.48] * 4, "spec matlab example: settlingtime(y, 2)")

    def test_overshoot_undershoot(self):
        close(np.concatenate(matlab.overshoot(OVERSHOOT)), [20.0, 1.2, 3.0], "spec matlab / D12: post overshoot (percent, value, instant)")
        close(np.concatenate(matlab.undershoot(OVERSHOOT)), [0.0, 1.0, 4.0], "spec matlab / D12: post undershoot (percent, value, instant)")

    def test_pulse_functions(self):
        close(np.concatenate(matlab.pulsewidth(Y)), [3, 3, 1.5, 6.5, 4.5, 9.5], "spec matlab example: pulsewidth")
        close(np.concatenate(matlab.pulseperiod(Y)), [5.0, 1.5, 4.5, 6.5], "spec matlab / D12: pulseperiod, complete periods only")
        close(np.concatenate(matlab.pulsesep(Y)), [2.0, 4.5, 6.5], "spec matlab example: pulsesep")
        close(np.concatenate(matlab.dutycycle(Y)), [0.6, 1.5, 4.5, 6.5], "spec matlab example: dutycycle")
        close(np.concatenate(matlab.pulsewidth(Y, polarity="negative")), [2.0, 4.5, 6.5], "spec matlab: negative pulse width")

    def test_returns_numpy_arrays(self):
        for name, result in (("midcross", matlab.midcross(Y)), ("slewrate", matlab.slewrate(Y)), ("risetime[0]", matlab.risetime(Y)[0])):
            with self.subTest(case=name):
                self.assertIsInstance(result, np.ndarray, msg=f"spec matlab: {name} returns a numpy array")

    def test_errors(self):
        with self.assertRaises(TypeError, msg="spec §E14: risetime has no polarity keyword"):
            matlab.risetime(Y, polarity="positive")
        with self.assertRaises(InputError, msg="spec §E15: polarity must be 'positive' or 'negative'"):
            matlab.pulsewidth(Y, polarity="up")
        with self.assertRaises(InputError, msg="spec §E5: matlab reference is an increasing pair"):
            matlab.risetime(Y, reference=(90, 10))


class TestFacadeAddsNoLogic(unittest.TestCase):
    """§C27: the facade's values equal the Pythonic API's for the same levels and reference."""

    def setUp(self):
        x = np.linspace(0, 3, 3000)
        ph = x % 1.0
        self.y = np.clip((ph - 0.2) / 0.05, 0, 1) - np.clip((ph - 0.6) / 0.05, 0, 1)
        self.x = x

    def test_matches_pythonic(self):
        trs = pt.transitions(self.y, x=self.x, levels=(0, 1))
        close(matlab.midcross(self.y, self.x, levels=(0, 1)), [t.mid_instant for t in trs], "spec §C27: midcross")
        close(matlab.slewrate(self.y, self.x, levels=(0, 1)), [t.slew_rate for t in trs], "spec §C27: slewrate")
        rises = [t for t in trs if t.polarity == Polarity.RISING]
        close(matlab.risetime(self.y, self.x, levels=(0, 1))[0], [t.duration for t in rises], "spec §C27: risetime")
        widths = [p.width for p in pt.pulses(self.y, x=self.x, levels=(0, 1))]
        close(matlab.pulsewidth(self.y, self.x, levels=(0, 1))[0], widths, "spec §C27: pulsewidth")

    def test_matches_pythonic_fit(self):
        trs = pt.transitions(self.y, x=self.x, levels=(0, 1), mid_method="fit")
        close(matlab.midcross(self.y, self.x, levels=(0, 1), mid_method="fit"), [t.mid_instant for t in trs],
              "spec §C27 / §C7 (rev 5): midcross with mid_method='fit'")
        widths = [p.width for p in pt.pulses(self.y, x=self.x, levels=(0, 1), mid_method="fit")]
        close(matlab.pulsewidth(self.y, self.x, levels=(0, 1), mid_method="fit")[0], widths,
              "spec §C27 / §C7 (rev 5): pulsewidth with mid_method='fit'")


if __name__ == "__main__":
    unittest.main()
