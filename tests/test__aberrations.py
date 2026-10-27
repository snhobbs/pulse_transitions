"""Tests for pulse_transitions._aberrations — derived from: pulse_transitions Expanded Spec (v0.3), transition_metrics (§5.3.5, §5.3.7)."""

import math
import unittest

import numpy as np

from pulse_transitions._aberrations import metrics_for
from pulse_transitions._types import Aberration, Polarity, Transition, TransitionMetrics

from tests.signals import (
    OVERSHOOT,
    OVERSHOOT_FALLING_T,
    OVERSHOOT_T,
    SQUARE,
    SQUARE_TRANSITIONS,
)


def run(y, transitions, levels=(0.0, 1.0), **kwargs):
    y = np.asarray(y, dtype=float)
    return metrics_for(
        np.arange(len(y), dtype=float), y, tuple(transitions), levels, **kwargs
    )


class TestMetricsFor(unittest.TestCase):
    def assertAberration(self, actual, percent, instant, value, clause):
        self.assertIsInstance(
            actual, Aberration, msg=f"{clause}: expected an Aberration, got {actual!r}"
        )
        for name, want in (
            ("percent", percent),
            ("instant", instant),
            ("value", value),
        ):
            got = getattr(actual, name)
            if math.isnan(want):
                self.assertTrue(
                    math.isnan(got), msg=f"{clause}: {name} should be NaN, got {got!r}"
                )
            else:
                self.assertAlmostEqual(
                    got,
                    want,
                    delta=1e-9,
                    msg=f"{clause}: {name} is {got!r}, spec says {want}",
                )

    def test_metrics_for_overshoot_example(self):
        (m,) = run(OVERSHOOT, [OVERSHOOT_T])
        self.assertIsInstance(
            m,
            TransitionMetrics,
            msg="spec transition_metrics: returns TransitionMetrics",
        )
        self.assertEqual(
            m.transition,
            OVERSHOOT_T,
            msg="spec Types: the metrics carry their transition",
        )
        self.assertEqual(
            m.levels, (0.0, 1.0), msg="spec Types: the metrics carry the levels"
        )
        self.assertAberration(
            m.post_overshoot,
            20.0,
            3.0,
            1.2,
            "spec §C15 example: post overshoot 20 % at x=3",
        )
        self.assertAberration(
            m.post_undershoot,
            0.0,
            4.0,
            1.0,
            "spec §C15 example: post undershoot 0, extreme still reported",
        )
        self.assertAberration(
            m.pre_overshoot,
            0.0,
            1.0,
            0.0,
            "spec §C14 (rev 6)/§C15 example: pre region ends at the pre anchor x=2; its first sample in double precision is x=1",
        )
        self.assertAberration(
            m.pre_undershoot,
            0.0,
            1.0,
            0.0,
            "spec §C14/§C15 example: first sample on ties",
        )
        self.assertAlmostEqual(
            m.settling_time,
            4.383333333333333,
            delta=1e-9,
            msg=f"spec §C16 example: band 0.98 crossed at 6.8, minus mid 2.4167; got {m.settling_time!r}",
        )

    def test_metrics_for_falling_overshoot_is_below_low(self):
        (m,) = run([1 - v for v in OVERSHOOT], [OVERSHOOT_FALLING_T])
        self.assertAberration(
            m.post_overshoot,
            20.0,
            3.0,
            -0.2,
            "spec §C15 / D3: falling overshoot is the excursion below low",
        )
        self.assertAlmostEqual(
            m.settling_time,
            4.383333333333333,
            delta=1e-9,
            msg="spec §C16: mirrored settling time",
        )

    def test_metrics_for_square_settling(self):
        result = run(SQUARE, SQUARE_TRANSITIONS)
        self.assertEqual(
            len(result), 4, msg="spec transition_metrics: one entry per transition"
        )
        for i, m in enumerate(result):
            with self.subTest(case=i):
                self.assertAlmostEqual(
                    m.settling_time,
                    0.48,
                    delta=1e-9,
                    msg=f"spec §C16 example: settling 0.48 for each; got {m.settling_time!r}",
                )

    def test_metrics_for_not_settled_is_nan(self):
        t = Transition(Polarity.RISING, 2.1, 2.5, 2.9, 0.1, 0.5, 0.9, 2, 3)
        (m,) = run([0, 0, 0, 1, 1, 1.5], [t])
        self.assertTrue(
            math.isnan(m.settling_time),
            msg=f"spec §C16: last searched sample out of band → NaN; got {m.settling_time!r}",
        )

    def test_metrics_for_regions_clipped_at_neighbours(self):
        first = run(SQUARE, SQUARE_TRANSITIONS, aberration_factor=10.0)[0]
        self.assertAberration(
            first.post_undershoot,
            0.0,
            2.0,
            1.0,
            "spec §C14 (rev 6): the post region stops at the next transition's pre anchor (x=4), so the 0 at x=5 is excluded",
        )

    def test_metrics_for_empty_region_is_nan(self):
        # Never reaches the 0.98 boundary: no post anchor, so the post region is empty.
        t = Transition(
            Polarity.RISING,
            2 + 0.1 / 0.95,
            2 + 0.5 / 0.95,
            2 + 0.9 / 0.95,
            0.1,
            0.5,
            0.9,
            2,
            3,
        )
        (m,) = run([0, 0, 0, 0.95, 0.95, 0.95], [t])
        for name in ("post_overshoot", "post_undershoot"):
            with self.subTest(case=name):
                self.assertAberration(
                    getattr(m, name),
                    math.nan,
                    math.nan,
                    math.nan,
                    f"spec §C14 (rev 6) example: no sample reaches 0.98, so no post anchor and {name} is NaN",
                )
        self.assertAberration(
            m.pre_overshoot,
            0.0,
            0.0,
            0.0,
            "spec §C14 (rev 6) example: the pre region still exists",
        )
        # Starts above the 0.02 boundary: no pre anchor, so the pre region is empty.
        t = Transition(
            Polarity.RISING,
            1 + 0.05 / 0.95,
            1 + 0.45 / 0.95,
            1 + 0.85 / 0.95,
            0.1,
            0.5,
            0.9,
            1,
            2,
        )
        (m,) = run([0.05, 0.05, 1, 1, 1], [t])
        for name in ("pre_overshoot", "pre_undershoot"):
            with self.subTest(case=name):
                self.assertAberration(
                    getattr(m, name),
                    math.nan,
                    math.nan,
                    math.nan,
                    f"spec §C14 (rev 6): no sample is within 0.02 of low, so no pre anchor and {name} is NaN",
                )

    def test_metrics_for_monotonic_edge_has_no_aberrations(self):
        t = Transition(Polarity.RISING, 2.2, 3.0, 4.0, 0.1, 0.5, 0.9, 2, 4)
        (m,) = run([0, 0, 0, 0.5, 0.9, 0.95, 0.99, 1, 1, 1], [t])
        self.assertAberration(
            m.post_undershoot,
            0.0,
            7.0,
            1.0,
            "spec §C14 (rev 7) example: post region starts at the anchor x=7 (first sample at the level), "
            "so neither the 0.9 at the upper crossing nor the 0.99 approach is an undershoot",
        )
        self.assertAberration(
            m.post_overshoot, 0.0, 7.0, 1.0, "spec rev 6 example: post overshoot"
        )
        self.assertAberration(
            m.pre_overshoot, 0.0, 0.0, 0.0, "spec rev 6 example: pre overshoot"
        )
        self.assertAlmostEqual(
            m.settling_time,
            2.75,
            delta=1e-9,
            msg=f"spec §C16 rev 6 example: 0.98 crossed at 5.75, minus mid 3.0; got {m.settling_time!r}",
        )

    def test_metrics_for_sloped_edges_settle(self):
        rising = Transition(Polarity.RISING, 1.2, 2.0, 2.8, 0.1, 0.5, 0.9, 1, 3)
        falling = Transition(Polarity.FALLING, 7.8, 7.0, 6 + 1 / 9, 0.1, 0.5, 0.9, 6, 8)
        r, f = run([0, 0, 0.5, 1, 1, 1, 0.95, 0.5, 0, 0], [rising, falling])
        self.assertAlmostEqual(
            r.settling_time,
            0.96,
            delta=1e-9,
            msg=f"spec §C16 (rev 6) example: the search stops at the falling edge's pre anchor (x=5), "
            f"so the 0.95 at x=6 doesn't count; got {r.settling_time!r}",
        )
        self.assertAberration(
            r.post_undershoot,
            0.0,
            3.0,
            1.0,
            "spec §C14 (rev 6) example: the post region ends at the next pre anchor x=5, before the 0.95",
        )
        self.assertAlmostEqual(
            f.settling_time,
            0.96,
            delta=1e-9,
            msg=f"spec rev 6 example: falling settles; got {f.settling_time!r}",
        )
        self.assertAberration(
            f.pre_overshoot,
            0.0,
            3.0,
            1.0,
            "spec §C14 (rev 6) example: the pre region starts at the rising edge's post anchor x=3",
        )

    def test_metrics_for_next_edge_without_pre_anchor_is_not_settled(self):
        rising = Transition(
            Polarity.RISING,
            1 + 0.1 / 0.95,
            1 + 0.5 / 0.95,
            1 + 0.9 / 0.95,
            0.1,
            0.5,
            0.9,
            1,
            2,
        )
        falling = Transition(
            Polarity.FALLING,
            3 + 0.85 / 0.95,
            3 + 0.45 / 0.95,
            3 + 0.05 / 0.95,
            0.1,
            0.5,
            0.9,
            3,
            4,
        )
        r, _ = run([0, 0, 0.95, 0.95, 0, 0], [rising, falling])
        self.assertTrue(
            math.isnan(r.settling_time),
            msg=f"spec §C16 (rev 6): the falling edge has no pre anchor (nothing >= 0.98), so NaN; got {r.settling_time!r}",
        )

    def test_metrics_for_excursion_within_band_is_reported(self):
        t = Transition(
            Polarity.RISING,
            2 + 0.1 / 1.01,
            2 + 0.5 / 1.01,
            2 + 0.9 / 1.01,
            0.1,
            0.5,
            0.9,
            2,
            3,
        )
        (m,) = run([0, 0, 0, 1.01, 1, 1, 1, 1, 1, 1], [t])
        self.assertAberration(
            m.post_overshoot,
            1.0,
            3.0,
            1.01,
            "spec §C15 (rev 7) example: a 1 % overshoot inside the 2 % band is still reported as 1 %",
        )

    def test_metrics_for_anchor_falls_back_to_band(self):
        t = Transition(
            Polarity.RISING,
            2 + 0.1 / 0.99,
            2 + 0.5 / 0.99,
            2 + 0.9 / 0.99,
            0.1,
            0.5,
            0.9,
            2,
            3,
        )
        (m,) = run([0, 0, 0, 0.99, 0.99, 0.99], [t])
        self.assertAberration(
            m.post_undershoot,
            1.0,
            3.0,
            0.99,
            "spec §C14/§C15 (rev 7) example: never reaches 1.0, so the post anchor falls back to the "
            "band (sample 3) and the 1 % shortfall is the undershoot",
        )

    def test_metrics_for_no_transitions(self):
        self.assertEqual(
            run([0, 0, 0], ()), (), msg="spec §C29: nothing to report → ()"
        )


if __name__ == "__main__":
    unittest.main()
