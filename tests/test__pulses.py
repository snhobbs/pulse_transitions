"""Tests for pulse_transitions._pulses — derived from: pulse_transitions Expanded Spec (v0.3), pulses (§5.4)."""
import math
import unittest

from pulse_transitions._pulses import make_pulses
from pulse_transitions._types import Polarity, Pulse

from tests.signals import SQUARE_T1, SQUARE_T2, SQUARE_T3, SQUARE_T4, SQUARE_TRANSITIONS


class TestMakePulses(unittest.TestCase):
    def assertPulse(self, p, polarity, first, second, width, period, separation, duty, clause):
        self.assertIsInstance(p, Pulse, msg=f"{clause}: expected a Pulse, got {p!r}")
        self.assertEqual((p.polarity, p.first, p.second), (polarity, first, second), msg=f"{clause}: polarity/first/second of {p!r}")
        for name, want in (("width", width), ("period", period), ("separation", separation), ("duty_cycle", duty)):
            got = getattr(p, name)
            if math.isnan(want):
                self.assertTrue(math.isnan(got), msg=f"{clause}: {name} should be NaN (no next pulse), got {got!r}")
            else:
                self.assertAlmostEqual(got, want, delta=1e-9, msg=f"{clause}: {name} is {got!r}, spec says {want}")

    def test_make_pulses_positive(self):
        result = make_pulses(SQUARE_TRANSITIONS)
        self.assertIsInstance(result, tuple, msg="spec pulses: returns a tuple")
        self.assertEqual(len(result), 2, msg=f"spec pulses example: two positive pulses, got {result!r}")
        self.assertPulse(result[0], Polarity.RISING, SQUARE_T1, SQUARE_T2, 3.0, 5.0, 2.0, 0.6, "spec §C17 example, pulse 1")
        nan = math.nan
        self.assertPulse(result[1], Polarity.RISING, SQUARE_T3, SQUARE_T4, 3.0, nan, nan, nan, "spec §C17 example, last pulse")

    def test_make_pulses_negative(self):
        result = make_pulses(SQUARE_TRANSITIONS, Polarity.FALLING)
        self.assertEqual(len(result), 1, msg=f"spec pulses example: one negative pulse (T4 has no follower), got {result!r}")
        self.assertPulse(result[0], Polarity.FALLING, SQUARE_T2, SQUARE_T3, 2.0, 5.0, 3.0, 0.4, "spec §C17 example, negative pulse")

    def test_make_pulses_rise_without_fall(self):
        self.assertEqual(make_pulses((SQUARE_T1,)), (), msg="spec pulses: a rise with no fall is not a pulse")

    def test_make_pulses_empty(self):
        self.assertEqual(make_pulses(()), (), msg="spec §C29: no transitions → ()")


if __name__ == "__main__":
    unittest.main()
