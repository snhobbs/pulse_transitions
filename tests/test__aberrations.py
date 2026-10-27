"""Tests for pulse_transitions._aberrations — derived from: pulse_transitions Expanded Spec (v0.3), transition_metrics (§5.3.5, §5.3.7)."""
import math
import unittest

import numpy as np

from pulse_transitions._aberrations import metrics_for
from pulse_transitions._types import Aberration, Polarity, Transition, TransitionMetrics

from tests.signals import OVERSHOOT, OVERSHOOT_FALLING_T, OVERSHOOT_T, SQUARE, SQUARE_TRANSITIONS


def run(y, transitions, levels=(0.0, 1.0), **kwargs):
    y = np.asarray(y, dtype=float)
    return metrics_for(np.arange(len(y), dtype=float), y, tuple(transitions), levels, **kwargs)


class TestMetricsFor(unittest.TestCase):
    def assertAberration(self, actual, percent, instant, value, clause):
        self.assertIsInstance(actual, Aberration, msg=f"{clause}: expected an Aberration, got {actual!r}")
        for name, want in (("percent", percent), ("instant", instant), ("value", value)):
            got = getattr(actual, name)
            if math.isnan(want):
                self.assertTrue(math.isnan(got), msg=f"{clause}: {name} should be NaN, got {got!r}")
            else:
                self.assertAlmostEqual(got, want, delta=1e-9, msg=f"{clause}: {name} is {got!r}, spec says {want}")

    def test_metrics_for_overshoot_example(self):
        (m,) = run(OVERSHOOT, [OVERSHOOT_T])
        self.assertIsInstance(m, TransitionMetrics, msg="spec transition_metrics: returns TransitionMetrics")
        self.assertEqual(m.transition, OVERSHOOT_T, msg="spec Types: the metrics carry their transition")
        self.assertEqual(m.levels, (0.0, 1.0), msg="spec Types: the metrics carry the levels")
        self.assertAberration(m.post_overshoot, 20.0, 3.0, 1.2, "spec §C15 example: post overshoot 20 % at x=3")
        self.assertAberration(m.post_undershoot, 0.0, 4.0, 1.0, "spec §C15 example: post undershoot 0, extreme still reported")
        self.assertAberration(m.pre_overshoot, 0.0, 1.0, 0.0, "spec §C14/§C15 example: pre region [0.083, 2.083] holds x=1, 2")
        self.assertAberration(m.pre_undershoot, 0.0, 1.0, 0.0, "spec §C14/§C15 example: first sample on ties")
        self.assertAlmostEqual(m.settling_time, 4.383333333333333, delta=1e-9,
                               msg=f"spec §C16 example: band 0.98 crossed at 6.8, minus mid 2.4167; got {m.settling_time!r}")

    def test_metrics_for_falling_overshoot_is_below_low(self):
        (m,) = run([1 - v for v in OVERSHOOT], [OVERSHOOT_FALLING_T])
        self.assertAberration(m.post_overshoot, 20.0, 3.0, -0.2, "spec §C15 / D3: falling overshoot is the excursion below low")
        self.assertAlmostEqual(m.settling_time, 4.383333333333333, delta=1e-9, msg="spec §C16: mirrored settling time")

    def test_metrics_for_square_settling(self):
        result = run(SQUARE, SQUARE_TRANSITIONS)
        self.assertEqual(len(result), 4, msg="spec transition_metrics: one entry per transition")
        for i, m in enumerate(result):
            with self.subTest(case=i):
                self.assertAlmostEqual(m.settling_time, 0.48, delta=1e-9, msg=f"spec §C16 example: settling 0.48 for each; got {m.settling_time!r}")

    def test_metrics_for_not_settled_is_nan(self):
        t = Transition(Polarity.RISING, 2.1, 2.5, 2.9, 0.1, 0.5, 0.9, 2, 3)
        (m,) = run([0, 0, 0, 1, 1, 1.5], [t])
        self.assertTrue(math.isnan(m.settling_time), msg=f"spec §C16: last searched sample out of band → NaN; got {m.settling_time!r}")

    def test_metrics_for_regions_clipped_at_neighbours(self):
        first = run(SQUARE, SQUARE_TRANSITIONS, aberration_factor=10.0)[0]
        self.assertAberration(first.post_undershoot, 0.0, 2.0, 1.0,
                              "spec §C14: the post region stops at the next transition's start (4.1), so the 0 at x=5 is excluded")

    def test_metrics_for_empty_region_is_nan(self):
        first = run(SQUARE, SQUARE_TRANSITIONS, aberration_factor=0.01)[0]
        for name in ("pre_overshoot", "pre_undershoot", "post_overshoot", "post_undershoot"):
            with self.subTest(case=name):
                self.assertAberration(getattr(first, name), math.nan, math.nan, math.nan,
                                      f"spec §C14: a region with no samples gives Aberration(nan, nan, nan) ({name})")

    def test_metrics_for_excursion_within_tolerance_reads_zero(self):
        t = Transition(Polarity.RISING, 2 + 0.1 / 1.01, 2 + 0.5 / 1.01, 2 + 0.9 / 1.01, 0.1, 0.5, 0.9, 2, 3)
        (m,) = run([0, 0, 0, 1.01, 1, 1, 1, 1, 1, 1], [t])
        self.assertAberration(m.post_overshoot, 0.0, 3.0, 1.01,
                              "spec §C15: a 1 % overshoot is inside the 2 % state boundary, so percent is 0 (extreme still reported)")

    def test_metrics_for_no_transitions(self):
        self.assertEqual(run([0, 0, 0], ()), (), msg="spec §C29: nothing to report → ()")


if __name__ == "__main__":
    unittest.main()
