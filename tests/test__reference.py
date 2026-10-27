"""Tests for pulse_transitions._reference — derived from: pulse_transitions Expanded Spec (v0.3), reference_levels / interpolate_instant."""

import unittest

import numpy as np

from pulse_transitions._reference import interpolate_instant, reference_levels


class TestReferenceLevels(unittest.TestCase):
    def test_reference_levels_examples(self):
        cases = {
            ((0.0, 2.0), (10.0, 50.0, 90.0)): (0.2, 1.0, 1.8),
            ((-1.0, 1.0), (20.0, 50.0, 80.0)): (-0.6, 0.0, 0.6),
        }
        for (levels, reference), expected in cases.items():
            with self.subTest(case=(levels, reference)):
                result = reference_levels(levels, reference)
                self.assertEqual(
                    len(result), 3, msg=f"spec §5.3.2: three levels, got {result!r}"
                )
                for got, want in zip(result, expected, strict=True):
                    self.assertAlmostEqual(
                        got,
                        want,
                        delta=1e-12,
                        msg=f"spec §5.3.2: {levels}, {reference} → {expected}, got {result!r}",
                    )


class TestInterpolateInstant(unittest.TestCase):
    def test_interpolate_instant_scalar_examples(self):
        cases = [
            ((0.49, 0.4, 0.51, 0.6, 0.5), 0.5),
            ((2.0, 0.0, 3.0, 1.2, 0.9), 2.75),
            ((1.0, 0.3, 2.0, 0.3, 0.3), 1.0),
        ]
        for args, expected in cases:
            with self.subTest(case=args):
                self.assertAlmostEqual(
                    float(interpolate_instant(*args)),
                    expected,
                    delta=1e-12,
                    msg=f"spec §5.3.3: interpolate_instant{args} → {expected}",
                )

    def test_interpolate_instant_vectorised(self):
        result = interpolate_instant(
            np.array([0.49, 2.0, 1.0]),
            np.array([0.4, 0.0, 0.3]),
            np.array([0.51, 3.0, 2.0]),
            np.array([0.6, 1.2, 0.3]),
            np.array([0.5, 0.9, 0.3]),
        )
        self.assertIsInstance(
            result,
            np.ndarray,
            msg="spec Internal interfaces: interpolate_instant is vectorised over arrays",
        )
        np.testing.assert_allclose(
            result,
            [0.5, 2.75, 1.0],
            rtol=0,
            atol=1e-12,
            err_msg="spec §5.3.3: vectorised interpolation, x0 where y1 == y0",
        )


if __name__ == "__main__":
    unittest.main()
