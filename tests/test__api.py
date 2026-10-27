"""Tests for pulse_transitions._api (public functions) — derived from: pulse_transitions Expanded Spec (v0.3), Public API / Error contract."""
import logging
import math
import unittest
import warnings

import numpy as np

import pulse_transitions as pt
from pulse_transitions import (
    EdgeNotFoundError, InputError, LevelError, Polarity, StateLevels, Transition,
)

from tests.signals import DITHER, NOISY_MID, OVERSHOOT, SQUARE, SQUARE_TRANSITIONS, TransitionAsserts, staircase


class TestReferenceLevels(unittest.TestCase):
    def test_reference_levels_default(self):
        for got, want in zip(pt.reference_levels((0.0, 2.0)), (0.2, 1.0, 1.8)):
            self.assertAlmostEqual(got, want, delta=1e-12, msg="spec reference_levels example: (0, 2) → (0.2, 1.0, 1.8)")

    def test_reference_levels_errors(self):
        with self.assertRaises(InputError, msg="spec §E4: equal levels"):
            pt.reference_levels((1.0, 1.0))
        with self.assertRaises(InputError, msg="spec §E5: reference not increasing"):
            pt.reference_levels((0.0, 1.0), (50.0, 10.0, 90.0))


class TestStateLevels(unittest.TestCase):
    Y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5]

    def test_state_levels_examples(self):
        cases = {
            "mode": (dict(y=self.Y, bins=4), (0.0, 1.0)),
            "mean": (dict(y=self.Y, bins=4, method="mean"), (0.0, 0.9166666666666666)),
            "peak": (dict(y=[0, 3, -1, 2], method="peak"), (-1.0, 3.0)),
            "endpoints": (dict(y=[1, 1, 5, 0, 0], method="endpoints", endpoint_samples=2), (0.0, 1.0)),
        }
        for case, (kwargs, (low, high)) in cases.items():
            with self.subTest(case=case):
                r = pt.state_levels(**kwargs)
                self.assertIsInstance(r, StateLevels, msg=f"spec state_levels: returns StateLevels ({case})")
                self.assertAlmostEqual(r.low, low, delta=1e-12, msg=f"spec state_levels example ({case}): low")
                self.assertAlmostEqual(r.high, high, delta=1e-12, msg=f"spec state_levels example ({case}): high")

    def test_state_levels_bounds(self):
        r = pt.state_levels([0, 0, 1, 1, 7, 7], method="peak", bounds=(0, 3))
        self.assertEqual(r.levels, (0.0, 1.0), msg=f"spec §C4: levels come from the bounded data only; got {r!r}")

    def test_state_levels_errors(self):
        cases = {
            "§E7 constant": (LevelError, dict(y=[2.0, 2.0, 2.0])),
            "§E8 unknown method": (InputError, dict(y=self.Y, method="median")),
            "§E8 bins < 2": (InputError, dict(y=self.Y, bins=1)),
            "§E8 split reversed": (InputError, dict(y=self.Y, split=(0.6, 0.4))),
            "§E9 too many endpoint samples": (InputError, dict(y=[1, 2, 3, 4, 5], method="endpoints", endpoint_samples=3)),
            "§E9 zero endpoint samples": (InputError, dict(y=[1, 2, 3, 4, 5], method="endpoints", endpoint_samples=0)),
        }
        for case, (exc, kwargs) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(exc, msg=f"spec {case}"):
                    pt.state_levels(**kwargs)


class TestTransitions(TransitionAsserts, unittest.TestCase):
    def test_transitions_detected_levels(self):
        result = pt.transitions(SQUARE)
        self.assertEqual(len(result), 4, msg="spec transitions example: detected levels are exactly (0, 1) (D2)")
        for got, want in zip(result, SQUARE_TRANSITIONS):
            self.assertTransition(got, want, "spec transitions example with detected levels")

    def test_transitions_fs(self):
        self.assertAlmostEqual(pt.transitions(SQUARE, fs=2.0, levels=(0, 1))[0].mid_instant, 0.75, delta=1e-12,
                               msg="spec transitions example: fs=2 halves the instants")

    def test_transitions_polarity_filter(self):
        result = pt.transitions(SQUARE, levels=(0, 1), polarity=Polarity.FALLING)
        self.assertEqual(len(result), 2, msg=f"spec transitions example: polarity=FALLING keeps the 2nd and 4th; got {result!r}")
        for got, want in zip(result, (SQUARE_TRANSITIONS[1], SQUARE_TRANSITIONS[3])):
            self.assertTransition(got, want, "spec transitions example: polarity=FALLING")

    def test_transitions_nan_dropped_and_indices_refer_to_cleaned_data(self):
        (t,) = pt.transitions([0, math.nan, 0, 1, 1], levels=(0, 1))
        self.assertTransition(t, Transition(Polarity.RISING, 2.1, 2.5, 2.9, 0.1, 0.5, 0.9, 1, 2),
                              "spec §C1: NaN sample dropped with its x; indices refer to the cleaned array")

    def test_transitions_mid_method(self):
        self.assertAlmostEqual(pt.transitions(NOISY_MID, levels=(0, 1))[0].mid_instant, 2.5, delta=1e-9,
                               msg="spec §C7: median mid crossing by default")
        self.assertAlmostEqual(pt.transitions(NOISY_MID, levels=(0, 1), mid_method="first")[0].mid_instant, 1.5, delta=1e-9,
                               msg="spec §C7: mid_method='first'")

    def test_transitions_median_reference_instants(self):
        (t,) = pt.transitions(DITHER, levels=(0, 1))
        self.assertAlmostEqual(t.lower_instant, 1.5, delta=1e-9, msg="spec §C7 (rev 3): public transitions pass the state levels; median 10 % crossing")
        self.assertAlmostEqual(t.upper_instant, 6.5, delta=1e-9, msg="spec §C7 (rev 3): median 90 % crossing")

    def test_transitions_errors(self):
        cases = {
            "§E4 reversed levels": (InputError, dict(levels=(1, 0))),
            "§E5 bad reference": (InputError, dict(levels=(0, 1), reference=(10, 90))),
            "§E15 mid_method fit (deferred)": (InputError, dict(levels=(0, 1), mid_method="fit")),
            "§E15 polarity not a Polarity": (InputError, dict(levels=(0, 1), polarity="rising")),
            "§E14 unknown keyword": (TypeError, dict(levels=(0, 1), thresholds=(0.1, 0.9))),
        }
        for case, (exc, kwargs) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(exc, msg=f"spec {case}"):
                    pt.transitions(SQUARE, **kwargs)
        with self.assertRaises(LevelError, msg="spec §E7: detecting levels on a constant record"):
            pt.transitions([2.0, 2.0, 2.0])

    def test_x_is_keyword_only(self):
        # declaration check (D1): passes against the stub signatures on purpose, and guards the real ones
        x = np.arange(len(SQUARE), dtype=float)
        for name in ("state_levels", "transitions", "transition_metrics", "step_metrics", "pulses"):
            with self.subTest(case=name):
                with self.assertRaises(TypeError, msg=f"spec D1 / §E14: {name}(y, x) positional must raise TypeError"):
                    getattr(pt, name)(SQUARE, x)


class TestStepAndTransitionMetrics(unittest.TestCase):
    def test_step_metrics_example(self):
        m = pt.step_metrics(OVERSHOOT, levels=(0, 1))
        self.assertAlmostEqual(m.post_overshoot.percent, 20.0, delta=1e-9, msg="spec step_metrics example: post overshoot 20 %")
        self.assertAlmostEqual(m.settling_time, 4.383333333333333, delta=1e-9, msg="spec step_metrics example: settling time")
        self.assertAlmostEqual(m.transition.mid_instant, 2.4166666666666665, delta=1e-9, msg="spec step_metrics example: mid instant")

    def test_step_metrics_falling(self):
        m = pt.step_metrics([1 - v for v in OVERSHOOT], levels=(0, 1))
        self.assertAlmostEqual(m.post_overshoot.value, -0.2, delta=1e-9, msg="spec step_metrics example / D3: falling overshoot below low")

    def test_step_metrics_bounds_select_the_step(self):
        m = pt.step_metrics([0, 0, 1, 1, 1, 0, 0], levels=(0, 1), bounds=(0, 3))
        self.assertAlmostEqual(m.transition.mid_instant, 1.5, delta=1e-9, msg="spec step_metrics example / D4: bounds select the step")

    def test_step_metrics_errors(self):
        cases = {
            "§E10 no transition": (EdgeNotFoundError, dict(y=[0, 0, 0], levels=(0, 1))),
            "§E10 two transitions": (EdgeNotFoundError, dict(y=[0, 0, 1, 1, 1, 0, 0], levels=(0, 1))),
            "§E15 aberration_factor 0": (InputError, dict(y=[0, 0, 1, 1], levels=(0, 1), aberration_factor=0)),
            "§E15 state_tolerance 50": (InputError, dict(y=[0, 0, 1, 1], levels=(0, 1), state_tolerance=50)),
            "§E15 state_tolerance 0": (InputError, dict(y=[0, 0, 1, 1], levels=(0, 1), state_tolerance=0)),
        }
        for case, (exc, kwargs) in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(exc, msg=f"spec {case}"):
                    pt.step_metrics(**kwargs)

    def test_step_metrics_two_transitions_message(self):
        with self.assertRaises(EdgeNotFoundError) as caught:
            pt.step_metrics([0, 0, 1, 1, 1, 0, 0], levels=(0, 1))
        self.assertIn("2", str(caught.exception), msg="spec step_metrics: the message says how many transitions were found")

    def test_transition_metrics_square(self):
        result = pt.transition_metrics(SQUARE, levels=(0, 1))
        self.assertEqual(len(result), 4, msg="spec transition_metrics example: four transitions")
        for m in result:
            self.assertAlmostEqual(m.settling_time, 0.48, delta=1e-9, msg="spec transition_metrics example: settling 0.48 each")

    def test_transition_metrics_not_settled(self):
        self.assertTrue(math.isnan(pt.transition_metrics([0, 0, 0, 1, 1, 1.5], levels=(0, 1))[0].settling_time),
                        msg="spec §C16 example: not settled → NaN")


class TestPulses(unittest.TestCase):
    def test_pulses_examples(self):
        positive = pt.pulses(SQUARE, levels=(0, 1))
        self.assertEqual([(p.width, p.period, p.separation, p.duty_cycle) for p in positive[:1]], [(3.0, 5.0, 2.0, 0.6)],
                         msg="spec pulses example: first positive pulse")
        self.assertTrue(math.isnan(positive[1].period), msg="spec §C17: last pulse has NaN period")
        (negative,) = pt.pulses(SQUARE, levels=(0, 1), polarity=Polarity.FALLING)
        self.assertAlmostEqual(negative.duty_cycle, 0.4, delta=1e-9, msg="spec pulses example: negative pulse duty 0.4")

    def test_pulses_none(self):
        self.assertEqual(pt.pulses([0, 0, 1, 1], levels=(0, 1)), (), msg="spec pulses example: a rise with no fall → ()")


class TestMeasureEdgeAndFlatness(unittest.TestCase):
    def test_measure_edge_public(self):
        x, y = staircase()
        r = pt.measure_edge(y, x=x, around=7.0)
        self.assertAlmostEqual(r.duration, 1.6, delta=1e-6, msg="spec measure_edge example via the public API")
        self.assertAlmostEqual(r.levels[0], 1.0, delta=1e-6, msg="spec measure_edge example: bottom level")

    def test_measure_edge_errors(self):
        x, y = staircase()
        with self.assertRaises(EdgeNotFoundError, msg="spec §E11 via the public API"):
            pt.measure_edge(y, x=x, around=11.0, max_distance=1.0)
        with self.assertRaises(InputError, msg="spec §E15: mid_method checked by measure_edge"):
            pt.measure_edge(y, x=x, around=7.0, mid_method="fit")
        with self.assertRaises(TypeError, msg="spec Public API: around is required and keyword-only"):
            pt.measure_edge(y, 7.0)

    def test_flatness_public(self):
        y = np.full(20, 2.0)
        y[5:10] = [2.0, 2.2, 1.8, 2.0, 2.0]
        self.assertAlmostEqual(pt.flatness(y, bounds=(5, 9)), 0.0632455532, delta=1e-9, msg="spec flatness example")
        self.assertTrue(math.isnan(pt.flatness(y, bounds=(5, 8))), msg="spec flatness example: 4 samples → NaN")


class TestQuietAndEmpty(unittest.TestCase):
    def test_empty_results_are_empty_tuples(self):
        for name in ("transitions", "transition_metrics", "pulses"):
            with self.subTest(case=name):
                self.assertEqual(getattr(pt, name)([0, 0, 0], levels=(0, 1)), (), msg=f"spec §C29: {name} → () when nothing found")

    def test_no_warnings_or_logs(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertNoLogs(level=logging.INFO):
                pt.transitions([0, 0, 0], levels=(0, 1))
                pt.pulses([0, 0, 1, 1], levels=(0, 1))
                pt.transition_metrics([0, 0, 0, 1, 1, 1.5], levels=(0, 1))
        self.assertEqual(caught, [], msg="spec §C28: no warnings for 'nothing found' outcomes")


class TestCallingInterface(unittest.TestCase):
    """The spec's calling-interface example, end to end."""

    def setUp(self):
        self.t = np.linspace(0, 10e-6, 100_000)
        t = self.t
        self.y = np.clip((t % 2e-6 - 0.4e-6) / 40e-9, 0, 1) - np.clip((t % 2e-6 - 1.2e-6) / 40e-9, 0, 1)

    def test_calling_interface(self):
        lv = pt.state_levels(self.y, x=self.t)
        self.assertAlmostEqual(lv.low, 0.0, delta=0.01, msg="spec Calling interface: low ≈ 0")
        self.assertAlmostEqual(lv.high, 1.0, delta=0.01, msg="spec Calling interface: high ≈ 1")
        trs = pt.transitions(self.y, x=self.t, levels=lv.levels)
        self.assertEqual(len(trs), 10, msg=f"spec Calling interface: 5 periods → 10 transitions, got {len(trs)}")
        step = pt.step_metrics(self.y, x=self.t, bounds=(0, 1e-6))
        self.assertEqual(step.transition.polarity, Polarity.RISING, msg="spec Calling interface: the first rise only")
        ps = pt.pulses(self.y, x=self.t)
        self.assertEqual(len(ps), 5, msg="spec Calling interface: 5 positive pulses")
        self.assertAlmostEqual(ps[0].width, 0.8e-6, delta=1e-9, msg="spec Calling interface: width 0.8 µs")
        self.assertAlmostEqual(ps[0].period, 2e-6, delta=1e-9, msg="spec Calling interface: period 2 µs")
        self.assertAlmostEqual(ps[0].duty_cycle, 0.4, delta=1e-3, msg="spec Calling interface: duty 0.4")
        edge = pt.measure_edge(self.y, x=self.t, around=0.42e-6, reference=(20, 50, 80))
        self.assertAlmostEqual(edge.duration, 0.6 * 40e-9, delta=1e-9, msg="spec Calling interface: 20-80 % of a 40 ns edge")


if __name__ == "__main__":
    unittest.main()
