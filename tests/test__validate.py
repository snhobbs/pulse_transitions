"""Tests for pulse_transitions._validate — derived from: pulse_transitions Expanded Spec (v0.3), Internal interfaces / Error contract."""

import math
import unittest

import numpy as np

from pulse_transitions._types import InputError
from pulse_transitions._validate import (
    as_waveform,
    check_choice,
    check_levels,
    check_reference,
)


class TestAsWaveform(unittest.TestCase):
    def assertArray(self, actual, expected, clause):
        self.assertIsInstance(
            actual,
            np.ndarray,
            msg=f"{clause}: expected an ndarray, got {type(actual)!r}",
        )
        self.assertEqual(
            actual.dtype,
            np.float64,
            msg=f"{clause}: dtype is {actual.dtype}, spec says float64",
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, err_msg=clause)

    def test_as_waveform_default_x_is_index(self):
        x, y = as_waveform([1, 2, 3])
        self.assertArray(
            x, [0.0, 1.0, 2.0], "spec §C2: x = arange(n) when neither x nor fs"
        )
        self.assertArray(y, [1.0, 2.0, 3.0], "spec §C2: y as float64")

    def test_as_waveform_fs(self):
        x, _ = as_waveform([1, 2, 3], fs=2.0)
        self.assertArray(x, [0.0, 0.5, 1.0], "spec Public API: x = arange(n) / fs")

    def test_as_waveform_explicit_nonuniform_x_kept(self):
        x, _ = as_waveform([0, 0, 1, 1], x=[0, 1, 3, 4])
        self.assertArray(
            x, [0.0, 1.0, 3.0, 4.0], "spec §C3: non-uniform x is used as given"
        )

    def test_as_waveform_drops_nan_pairs(self):
        for case, (y, x) in {
            "nan in y": ([1, math.nan, 3], [0, 1, 2]),
            "nan in x": ([1, 2, 3], [0, math.nan, 2]),
        }.items():
            with self.subTest(case=case):
                xo, yo = as_waveform(y, x=x)
                self.assertArray(
                    xo, [0.0, 2.0], f"spec §C1: NaN pairs dropped ({case})"
                )
                self.assertArray(
                    yo,
                    [1.0, 3.0] if case == "nan in y" else [1.0, 3.0],
                    f"spec §C1: NaN pairs dropped ({case})",
                )

    def test_as_waveform_bounds_inclusive(self):
        x, y = as_waveform([1, 2, 3, 4], bounds=(1, 2))
        self.assertArray(x, [1.0, 2.0], "spec §C4: bounds are inclusive")
        self.assertArray(y, [2.0, 3.0], "spec §C4: bounds are inclusive")

    def test_as_waveform_errors(self):
        cases = {
            "§E1 2-D y": dict(y=[[1, 2], [3, 4]]),
            "§E1 one sample": dict(y=[1.0]),
            "§E1 one sample after NaN removal": dict(y=[1.0, math.nan, math.nan]),
            "§E1 non-numeric": dict(y=["a", "b"]),
            "§E1 bounds leave one sample": dict(y=[1, 2, 3, 4], bounds=(1, 1.5)),
            "§E2 x and fs": dict(y=[1, 2], x=[0, 1], fs=1.0),
            "§E2 fs zero": dict(y=[1, 2], fs=0.0),
            "§E2 fs negative": dict(y=[1, 2], fs=-1.0),
            "§E2 fs nan": dict(y=[1, 2], fs=math.nan),
            "§E2 fs inf": dict(y=[1, 2], fs=math.inf),
            "§E3 length mismatch": dict(y=[1, 2, 3], x=[0, 1]),
            "§E3 x decreasing": dict(y=[1, 2, 3], x=[0, 2, 1]),
            "§E3 x repeated": dict(y=[1, 2, 3], x=[0, 1, 1]),
            "§E6 bounds reversed": dict(y=[1, 2, 3], bounds=(2, 1)),
            "§E6 bounds equal": dict(y=[1, 2, 3], bounds=(1, 1)),
            "§E6 bounds nan": dict(y=[1, 2, 3], bounds=(0, math.nan)),
        }
        for case, kwargs in cases.items():
            with self.subTest(case=case):
                with self.assertRaises(
                    InputError,
                    msg=f"spec {case}: as_waveform({kwargs!r}) must raise InputError",
                ):
                    as_waveform(**kwargs)

    def test_as_waveform_error_names_argument(self):
        with self.assertRaises(InputError) as caught:
            as_waveform([1, 2, 3], x=[0, 1])
        self.assertIn(
            "x",
            str(caught.exception),
            msg="spec Error contract: the InputError message names the argument",
        )
        with self.assertRaises(InputError) as caught:
            as_waveform([1, 2], fs=0.0)
        self.assertIn(
            "fs",
            str(caught.exception),
            msg="spec Error contract: the InputError message names the argument",
        )


class TestCheckLevels(unittest.TestCase):
    def test_check_levels_ok(self):
        self.assertEqual(
            check_levels((0, 1)),
            (0.0, 1.0),
            msg="spec §E4: valid levels come back as floats",
        )

    def test_check_levels_errors(self):
        for case in [
            (1.0, 1.0),
            (1.0, 0.0),
            (0.0, math.nan),
            (0.0, math.inf),
            (0.0, 1.0, 2.0),
            (0.0,),
            "ab",
        ]:
            with self.subTest(case=case):
                with self.assertRaises(
                    InputError, msg=f"spec §E4: levels {case!r} must raise InputError"
                ):
                    check_levels(case)


class TestCheckReference(unittest.TestCase):
    def test_check_reference_ok(self):
        self.assertEqual(
            check_reference((10, 50, 90)),
            (10.0, 50.0, 90.0),
            msg="spec §E5: a valid triple",
        )
        self.assertEqual(
            check_reference((0, 50, 100)),
            (0.0, 50.0, 100.0),
            msg="spec §E5: 0 and 100 are allowed",
        )
        self.assertEqual(
            check_reference((10, 90), pair=True),
            (10.0, 90.0),
            msg="spec §E5: matlab pair form",
        )

    def test_check_reference_errors(self):
        cases = [
            ((50, 10, 90), False),
            ((10, 10, 90), False),
            ((-1, 50, 90), False),
            ((10, 50, 101), False),
            ((10, 90), False),
            ((10, math.nan, 90), False),
            ((90, 10), True),
            ((10, 50, 90), True),
            ((10, 10), True),
        ]
        for reference, pair in cases:
            with self.subTest(case=(reference, pair)):
                with self.assertRaises(
                    InputError,
                    msg=f"spec §E5: reference {reference!r} (pair={pair}) must raise InputError",
                ):
                    check_reference(reference, pair=pair)


class TestCheckChoice(unittest.TestCase):
    def test_check_choice_returns_value(self):
        self.assertEqual(
            check_choice("mid_method", "first", ("median", "first")),
            "first",
            msg="spec §E15: an allowed value is returned",
        )

    def test_check_choice_raises_naming_argument(self):
        with self.assertRaises(
            InputError,
            msg="spec §E15 (rev 5): mid_method='slope' is not an allowed value",
        ) as caught:
            check_choice("mid_method", "slope", ("median", "first", "fit"))
        self.assertIn(
            "mid_method",
            str(caught.exception),
            msg="spec Error contract: the message names the argument",
        )


if __name__ == "__main__":
    unittest.main()
