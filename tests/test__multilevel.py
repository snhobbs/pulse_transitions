"""Tests for pulse_transitions._multilevel — derived from: pulse_transitions Expanded Spec (v0.3), measure_edge / §C18–§C20."""

import math
import unittest

import numpy as np

from pulse_transitions._multilevel import locate_and_measure
from pulse_transitions._types import EdgeMeasurement, EdgeNotFoundError, Polarity
from tests.signals import dense_noisy_edge, pulse_train, staircase

REF = (10.0, 50.0, 90.0)


def run(y, around, x=None, max_distance=math.inf, reference=REF, levels=None, **kwargs):
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y), dtype=float) if x is None else x
    return locate_and_measure(x, y, around, max_distance, reference, levels, **kwargs)


class TestLocateAndMeasure(unittest.TestCase):
    def assertEdge(
        self, r, polarity, levels, lower, upper, duration, clause, delta=1e-6
    ):
        self.assertIsInstance(
            r, EdgeMeasurement, msg=f"{clause}: expected EdgeMeasurement, got {r!r}"
        )
        self.assertEqual(
            r.transition.polarity,
            polarity,
            msg=f"{clause}: polarity of {r.transition!r}",
        )
        for got, want, name in (
            (r.levels[0], levels[0], "bottom level"),
            (r.levels[1], levels[1], "top level"),
            (r.transition.lower_instant, lower, "lower_instant"),
            (r.transition.upper_instant, upper, "upper_instant"),
            (r.duration, duration, "duration"),
        ):
            self.assertAlmostEqual(
                got,
                want,
                delta=delta,
                msg=f"{clause}: {name} is {got!r}, spec says {want}",
            )

    def test_staircase_second_step(self):
        x, y = staircase()
        self.assertEdge(
            run(y, 7.0, x=x),
            Polarity.RISING,
            (1.0, 3.0),
            6.2,
            7.8,
            1.6,
            "spec measure_edge example / §C19: second step measured between its own levels 1 and 3",
        )

    def test_staircase_first_step(self):
        x, y = staircase()
        self.assertEdge(
            run(y, 3.0, x=x),
            Polarity.RISING,
            (0.0, 1.0),
            2.2,
            3.8,
            1.6,
            "spec §C19: first step measured between 0 and 1",
        )

    def test_staircase_reference_20_80(self):
        x, y = staircase()
        self.assertAlmostEqual(
            run(y, 7.0, x=x, reference=(20.0, 50.0, 80.0)).duration,
            1.2,
            delta=1e-6,
            msg="spec measure_edge example: 20-80 % duration 1.2",
        )

    def test_staircase_falling(self):
        x, y = staircase()
        self.assertEdge(
            run(4 - y, 7.0, x=x),
            Polarity.FALLING,
            (1.0, 3.0),
            7.8,
            6.2,
            1.6,
            "spec §C20: a falling edge reports levels as (bottom, top)",
        )

    def test_given_levels(self):
        x, y = staircase()
        self.assertEdge(
            run(y, 7.0, x=x, levels=(1.0, 3.0)),
            Polarity.RISING,
            (1.0, 3.0),
            6.2,
            7.8,
            1.6,
            "spec measure_edge: given levels are used as (bottom, top)",
        )

    def test_dense_slow_noisy_edge_is_found(self):
        x, y, centre = dense_noisy_edge()
        r = run(y, centre, x=x)
        self.assertAlmostEqual(
            r.duration,
            1600.0,
            delta=16.0,
            msg=f"spec §C18: a 2000-sample edge in 1e6 noisy samples is found, 10-90 % within 1 % of 1600; got {r.duration!r}",
        )

    def test_fast_noisy_pulse_train_edge(self):
        x, y = pulse_train(10_000)
        r = run(y, 0.2, x=x)
        self.assertEqual(
            r.transition.polarity,
            Polarity.RISING,
            msg="spec measure_edge step 3 (rev 3): the rise at 0.2 is found",
        )
        self.assertAlmostEqual(
            r.duration,
            0.016,
            delta=0.0016,
            msg=f"spec measure_edge step 3 (rev 3): 10-90 % of a 0.02 edge ≈ 0.016 within 10 %; got {r.duration!r}",
        )
        for got, want in zip(r.levels, (0.0, 1.0), strict=True):
            self.assertAlmostEqual(
                got,
                want,
                delta=0.01,
                msg=f"spec §C19: flat levels ≈ (0, 1); got {r.levels!r}",
            )

    def test_errors(self):
        x, y = staircase()
        short = np.r_[np.zeros(7), np.ones(8)]
        cases = {
            "§E11 no edge within max_distance": dict(
                y=y, around=11.0, x=x, max_distance=1.0
            ),
            "§E11 flat trace": dict(y=np.ones(100), around=50.0),
            "§E11 fewer than 16 samples": dict(y=short, around=7.0),
            "§E12 under 3 flat samples before the edge": dict(
                y=np.r_[0.0, 0.0, np.ones(98)], around=1.5
            ),
            "§E13 reference levels not crossed": dict(
                y=y, around=7.0, x=x, levels=(5.0, 6.0)
            ),
        }
        for case, kwargs in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(EdgeNotFoundError, msg=f"spec {case}"):
                    run(**kwargs)

    def test_error_message_says_what_is_missing(self):
        x, y = staircase()
        with self.assertRaises(EdgeNotFoundError) as caught:
            run(y, 11.0, x=x, max_distance=1.0)
        self.assertTrue(
            str(caught.exception),
            msg="spec Error contract: EdgeNotFoundError says what is missing (empty message)",
        )


if __name__ == "__main__":
    unittest.main()
