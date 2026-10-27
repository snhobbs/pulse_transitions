"""Tests for pulse_transitions._types — derived from: pulse_transitions Expanded Spec (v0.3), Types."""

import dataclasses
import unittest

import numpy as np

from pulse_transitions._types import (
    Aberration,
    EdgeMeasurement,
    EdgeNotFoundError,
    InputError,
    LevelError,
    Polarity,
    PulseTransitionsError,
    StateLevels,
    Transition,
)


class TestTransitionProperties(unittest.TestCase):
    RISING = Transition(Polarity.RISING, 1.1, 1.5, 1.9, 0.1, 0.5, 0.9, 1, 2)
    FALLING = Transition(Polarity.FALLING, 4.9, 4.5, 4.1, 0.1, 0.5, 0.9, 4, 5)

    def test_transition_duration_rising(self):
        self.assertAlmostEqual(
            self.RISING.duration,
            0.8,
            delta=1e-12,
            msg="spec Types: duration = end - start (§5.3.4)",
        )

    def test_transition_duration_falling_is_positive(self):
        self.assertAlmostEqual(
            self.FALLING.duration,
            0.8,
            delta=1e-12,
            msg="spec §C8: duration > 0 for falling edges too",
        )

    def test_transition_slew_rate_sign(self):
        self.assertAlmostEqual(
            self.RISING.slew_rate,
            1.0,
            delta=1e-12,
            msg="spec §C8: slew_rate positive when rising",
        )
        self.assertAlmostEqual(
            self.FALLING.slew_rate,
            -1.0,
            delta=1e-12,
            msg="spec §C8: slew_rate negative when falling",
        )

    def test_transition_start_end_are_time_ordered(self):
        self.assertEqual(
            self.FALLING.start,
            4.1,
            msg="spec Types: start = min(lower_instant, upper_instant)",
        )
        self.assertEqual(
            self.FALLING.end,
            4.9,
            msg="spec Types: end = max(lower_instant, upper_instant)",
        )
        self.assertEqual(
            (self.RISING.start, self.RISING.end),
            (1.1, 1.9),
            msg="spec Types: start/end of a rising edge",
        )

    def test_transition_mid_method_defaults_to_median(self):
        # declaration check: passes against the stub on purpose
        self.assertEqual(
            self.RISING.mid_method,
            "median",
            msg="spec §C7: Transition records mid_method, default 'median'",
        )


class TestStateLevels(unittest.TestCase):
    def test_state_levels_amplitude(self):
        self.assertEqual(
            StateLevels(0.0, 2.0, "peak").amplitude,
            2.0,
            msg="spec Types: amplitude = high - low",
        )

    def test_state_levels_levels_tuple(self):
        self.assertEqual(
            StateLevels(-1.0, 3.0, "peak").levels,
            (-1.0, 3.0),
            msg="spec Types: levels = (low, high)",
        )

    def test_state_levels_arrays_do_not_affect_equality(self):
        # declaration check (§C26): array fields are compare=False
        a = StateLevels(0.0, 1.0, "mode", np.array([1, 2]), np.array([0.0, 0.5, 1.0]))
        b = StateLevels(0.0, 1.0, "mode", np.array([9, 9]), np.array([0.0, 0.5, 1.0]))
        self.assertEqual(
            a, b, msg="spec §C26: array fields don't take part in equality"
        )


class TestEdgeMeasurement(unittest.TestCase):
    def test_edge_measurement_duration_is_transition_duration(self):
        t = Transition(Polarity.RISING, 6.2, 7.0, 7.8, 1.2, 2.0, 2.8, 620, 780)
        self.assertAlmostEqual(
            EdgeMeasurement(t, (1.0, 3.0)).duration,
            1.6,
            delta=1e-12,
            msg="spec Types: EdgeMeasurement.duration = transition.duration",
        )


class TestDeclarations(unittest.TestCase):
    """Declaration checks: these pass against the stubs on purpose (the spec states them verbatim)."""

    def test_polarity_values(self):
        self.assertEqual(
            (int(Polarity.FALLING), int(Polarity.RISING)),
            (-1, 1),
            msg="spec Types: Polarity values",
        )

    def test_exception_hierarchy(self):
        for cls in (InputError, LevelError, EdgeNotFoundError):
            with self.subTest(case=cls.__name__):
                self.assertTrue(
                    issubclass(cls, PulseTransitionsError),
                    msg=f"spec Error contract: {cls.__name__} under PulseTransitionsError",
                )
                self.assertTrue(
                    issubclass(cls, ValueError),
                    msg=f"spec Error contract / D6: {cls.__name__} is a ValueError",
                )
        self.assertFalse(
            issubclass(EdgeNotFoundError, IndexError),
            msg="spec D6: EdgeNotFoundError is no longer an IndexError",
        )

    def test_results_are_frozen(self):
        records = [
            StateLevels(0.0, 1.0, "peak"),
            Transition(Polarity.RISING, 1.1, 1.5, 1.9, 0.1, 0.5, 0.9, 1, 2),
            Aberration(0.0, 1.0, 0.0),
        ]
        for record in records:
            with self.subTest(case=type(record).__name__):
                with self.assertRaises(
                    dataclasses.FrozenInstanceError,
                    msg=f"spec §C26: {type(record).__name__} must be frozen",
                ):
                    record.__setattr__(dataclasses.fields(record)[0].name, None)


if __name__ == "__main__":
    unittest.main()
