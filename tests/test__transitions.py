"""Tests for pulse_transitions._transitions — derived from: pulse_transitions Expanded Spec (v0.3), transitions / §C5–§C12."""
import unittest

import numpy as np

from pulse_transitions._transitions import find_transitions
from pulse_transitions._types import Polarity, Transition

from tests.signals import (
    DITHER, NOISY_MID, REF_01, SQUARE, SQUARE_TRANSITIONS, TAU_LN9, TransitionAsserts, pulse_train, ramp, rc_step,
)


def run(y, x=None, ref=REF_01, **kwargs):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, dtype=float)
    return find_transitions(x, y, ref, **kwargs)


class TestFindTransitionsExamples(TransitionAsserts, unittest.TestCase):
    def test_find_transitions_square(self):
        result = run(SQUARE)
        self.assertIsInstance(result, tuple, msg="spec transitions: returns a tuple")
        self.assertEqual(len(result), 4, msg=f"spec transitions example: SQUARE has 4 transitions, got {result!r}")
        for i, (got, want) in enumerate(zip(result, SQUARE_TRANSITIONS)):
            with self.subTest(case=i):
                self.assertTransition(got, want, "spec transitions example (SQUARE)")
                self.assertEqual(got.mid_method, "median", msg="spec §C7: default mid_method is 'median'")

    def test_find_transitions_runt_ignored(self):
        result = run([0, 0, 0.6, 0, 0, 1, 1])
        self.assertEqual(len(result), 1, msg=f"spec §C6: the 0.6 runt never reaches 0.9, so one transition; got {result!r}")
        self.assertTransition(result[0], Transition(Polarity.RISING, 4.1, 4.5, 4.9, 0.1, 0.5, 0.9, 4, 5), "spec §C6 runt example")

    def test_find_transitions_none(self):
        self.assertEqual(run([1, 1, 1]), (), msg="spec §C29: no transition → ()")

    def test_find_transitions_equal_to_reference_levels(self):
        self.assertEqual(len(run([0, 0, 1, 0.9, 1])), 1, msg="spec §C6: a sample equal to the upper level is in the high state")
        self.assertEqual(len(run([1, 1, 0, 0.1, 0])), 1, msg="spec §C6: a sample equal to the lower level is in the low state")

    def test_find_transitions_sample_at_lower_level_is_low(self):
        result = run([1, 1, 0.1, 0.1, 1])
        self.assertEqual([t.polarity for t in result], [Polarity.FALLING, Polarity.RISING],
                         msg=f"spec §C6: samples equal to the lower level are in the low state, so this dips and returns; got {result!r}")

    def test_find_transitions_no_partial_at_record_start(self):
        result = run([0.5, 1, 1, 0, 0])
        self.assertEqual([t.polarity for t in result], [Polarity.FALLING],
                         msg=f"spec §C6: a record starting mid-transition yields no partial transition; got {result!r}")

    def test_find_transitions_nonuniform_x(self):
        result = run([0, 0, 1, 1], x=[0, 1, 3, 4])
        self.assertTransition(result[0], Transition(Polarity.RISING, 1.2, 2.0, 2.8, 0.1, 0.5, 0.9, 1, 2),
                              "spec §C3: instants interpolate between the actual (non-uniform) sample times")


class TestMidMethod(TransitionAsserts, unittest.TestCase):
    def test_mid_method_median(self):
        result = run(NOISY_MID)
        self.assertTransition(result[0], Transition(Polarity.RISING, 0.2222222222222222, 2.5, 4.777777777777778,
                                                    0.1, 0.5, 0.9, 0, 5), "spec §C7: median of the mid crossings 1.5, 2.5, 3.5")
        self.assertEqual(result[0].mid_method, "median", msg="spec §C7: Transition records mid_method")

    def test_mid_method_first(self):
        result = run(NOISY_MID, mid_method="first")
        self.assertAlmostEqual(result[0].mid_instant, 1.5, delta=1e-9, msg="spec §C7: mid_method='first' takes the first crossing")
        self.assertEqual(result[0].mid_method, "first", msg="spec §C7: Transition records mid_method")


class TestMedianReferenceInstants(TransitionAsserts, unittest.TestCase):
    """§C7 (rev 3): lower/upper instants are median crossings within windows guarded by the state levels."""

    def test_dither_median_lower_and_upper(self):
        (t,) = run(DITHER, state_levels=(0.0, 1.0))
        self.assertTransition(t, Transition(Polarity.RISING, 1.5, 4.0, 6.5, 0.1, 0.5, 0.9, 2, 6),
                              "spec §C7 (rev 3) example: 10 % crossed at 0.833, 1.5, 2.5 → 1.5; 90 % at 5.5, 6.5, 7.167 → 6.5")

    def test_dither_without_state_levels_uses_bracketing_crossings(self):
        (t,) = run(DITHER)
        self.assertAlmostEqual(t.lower_instant, 2.5, delta=1e-9,
                               msg="spec transitions step 3 (rev 3): state_levels=None narrows the lower window to i .. j")
        self.assertAlmostEqual(t.upper_instant, 5.5, delta=1e-9,
                               msg="spec transitions step 3 (rev 3): state_levels=None narrows the upper window to i .. j")

    def test_clean_edges_unchanged_by_state_levels(self):
        for got, want in zip(run(SQUARE, state_levels=(0.0, 1.0)), SQUARE_TRANSITIONS):
            self.assertTransition(got, want, "spec §C7 (rev 3): one crossing per level gives the same instants")


class TestInvariants(unittest.TestCase):
    def setUp(self):
        self.x, self.y = pulse_train(20_000, periods=4)
        self.result = run(self.y, x=self.x)

    def test_transitions_alternate_and_are_time_ordered(self):
        self.assertEqual(len(self.result), 8, msg=f"spec §C5: 4 pulses → 8 transitions, got {len(self.result)}")
        polarities = [t.polarity for t in self.result]
        self.assertTrue(all(a != b for a, b in zip(polarities, polarities[1:])), msg=f"spec §C5: polarities alternate: {polarities}")
        mids = [t.mid_instant for t in self.result]
        self.assertEqual(mids, sorted(mids), msg="spec §C5: transitions in time order")

    def test_transition_bracketing(self):
        for i, t in enumerate(self.result):
            with self.subTest(case=i):
                self.assertLess(t.start_index, t.end_index, msg=f"spec §C9: start_index < end_index in {t!r}")
                self.assertTrue(t.start <= t.mid_instant <= t.end,
                                msg=f"spec §C9 (rev 3): start <= mid_instant <= end in {t!r}")
                self.assertTrue(t.start < self.x[t.start_index + 1] and t.end > self.x[t.end_index - 1],
                                msg=f"spec §C9 (rev 3): start < x[start_index + 1] and end > x[end_index - 1] in {t!r}")
                self.assertGreater(t.duration, 0, msg=f"spec §C8: duration > 0 in {t!r}")

    def test_scale_and_offset_invariance(self):
        a, b = 2.5, -3.0
        scaled = run(a * self.y + b, x=self.x, ref=tuple(a * r + b for r in REF_01))
        self.assertEqual(len(scaled), len(self.result), msg="spec §C10: a*y + b (a > 0) finds the same transitions")
        for got, want in zip(scaled, self.result):
            self.assertEqual(got.polarity, want.polarity, msg="spec §C10: polarity unchanged for a > 0")
            for name in ("lower_instant", "mid_instant", "upper_instant"):
                self.assertAlmostEqual(getattr(got, name), getattr(want, name), delta=1e-9, msg=f"spec §C10: {name} unchanged")

    def test_negation_flips_polarity(self):
        flipped = run(-self.y, x=self.x, ref=tuple(-r for r in reversed(REF_01)))
        self.assertEqual(len(flipped), len(self.result), msg="spec §C10: a < 0 finds the same transitions")
        for got, want in zip(flipped, self.result):
            self.assertEqual(int(got.polarity), -int(want.polarity), msg="spec §C10: a < 0 flips every polarity")
            self.assertAlmostEqual(got.mid_instant, want.mid_instant, delta=1e-9, msg="spec §C10: instants unchanged for a < 0")

    def test_time_shift(self):
        shifted = run(self.y, x=self.x + 7.25)
        for got, want in zip(shifted, self.result):
            self.assertAlmostEqual(got.mid_instant, want.mid_instant + 7.25, delta=1e-9, msg="spec §C11: instants shift by c")
            self.assertAlmostEqual(got.duration, want.duration, delta=1e-9, msg="spec §C11: durations unchanged")


class TestAccuracy(unittest.TestCase):
    def test_linear_ramp_exact_at_any_subsample_position(self):
        for offset in (0.0, 0.25, 0.5, 0.77):
            with self.subTest(case=offset):
                x, y, s = ramp(offset)
                (t,) = run(y, x=x)
                for name, want in (("lower_instant", s + 0.2), ("mid_instant", s + 1.0), ("upper_instant", s + 1.8)):
                    self.assertAlmostEqual(getattr(t, name), want, delta=1e-9 * want,
                                           msg=f"spec §C12: linear ramp {name} exact at offset {offset}; got {getattr(t, name)!r}")

    def test_rc_step_rise_time(self):
        for samples_per_tau, rel in ((50, 1e-2), (200, 2e-3)):
            with self.subTest(case=samples_per_tau):
                x, y = rc_step(samples_per_tau)
                (t,) = run(y, x=x)
                self.assertAlmostEqual(t.duration, TAU_LN9, delta=rel * TAU_LN9,
                                       msg=f"spec §C12: RC 10-90 % = tau ln 9 within {rel:.1%} at {samples_per_tau}/tau; got {t.duration!r}")


if __name__ == "__main__":
    unittest.main()
