"""Deterministic signals and shared assertions for the tests — values from docs/spec/pulse_transitions-expanded.md."""

import math

import numpy as np

from pulse_transitions._types import Polarity, Transition

# Worked-example waveforms (x = 0, 1, 2, ... unless stated)
SQUARE = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
OVERSHOOT = [0, 0, 0, 1.2, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0]
NOISY_MID = [0, 0.45, 0.55, 0.45, 0.55, 1]
DITHER = [
    0,
    0.12,
    0.08,
    0.12,
    0.5,
    0.88,
    0.92,
    0.88,
    1,
]  # 10 % crossed 3 times, 90 % 3 times (rev 3)
REF_01 = (0.1, 0.5, 0.9)  # reference levels of (0, 1) at 10/50/90 %

# The four transitions of SQUARE with levels (0, 1), as listed in the spec
SQUARE_T1 = Transition(Polarity.RISING, 1.1, 1.5, 1.9, 0.1, 0.5, 0.9, 1, 2)
SQUARE_T2 = Transition(Polarity.FALLING, 4.9, 4.5, 4.1, 0.1, 0.5, 0.9, 4, 5)
SQUARE_T3 = Transition(Polarity.RISING, 6.1, 6.5, 6.9, 0.1, 0.5, 0.9, 6, 7)
SQUARE_T4 = Transition(Polarity.FALLING, 9.9, 9.5, 9.1, 0.1, 0.5, 0.9, 9, 10)
SQUARE_TRANSITIONS = (SQUARE_T1, SQUARE_T2, SQUARE_T3, SQUARE_T4)

# The single transition of OVERSHOOT with levels (0, 1)
OVERSHOOT_T = Transition(
    Polarity.RISING, 2 + 0.1 / 1.2, 2 + 0.5 / 1.2, 2.75, 0.1, 0.5, 0.9, 2, 3
)
# ... and of 1 - OVERSHOOT (falling: the upper level is crossed first)
OVERSHOOT_FALLING_T = Transition(
    Polarity.FALLING, 2.75, 2 + 0.5 / 1.2, 2 + 0.1 / 1.2, 0.1, 0.5, 0.9, 2, 3
)

TAU_LN9 = math.log(9.0)  # 10-90 % duration of an RC step in units of tau


def staircase():
    """0 -> 1 over x 2..4, 1 -> 3 over x 6..8, sampled every 0.01 up to 14 (spec measure_edge example)."""
    x = np.arange(0, 14, 0.01)
    y = np.clip((x - 2) / 2, 0, 1) + 2 * np.clip((x - 6) / 2, 0, 1)
    return x, y


def ramp(offset_samples=0.0, step=0.01, start=4.0, length=2.0, end=10.0):
    """One linear 0 -> 1 edge starting at start + offset_samples * step; its 10/50/90 % instants are
    start' + 0.2, start' + 1.0, start' + 1.8 exactly (§C12)."""
    x = np.arange(0.0, end, step)
    s = start + offset_samples * step
    return x, np.clip((x - s) / length, 0.0, 1.0), s


def rc_step(samples_per_tau, tau=1.0):
    """1 - exp(-(t - 2 tau) / tau) after t = 2 tau, 0 before; 10-90 % duration tau ln 9 (§C12)."""
    step = tau / samples_per_tau
    x = np.arange(0.0, 12 * tau, step)
    y = np.where(x < 2 * tau, 0.0, 1.0 - np.exp(-(x - 2 * tau) / tau))
    return x, y


def pulse_train(n, periods=10, noise=0.01, seed=0):
    """The brief spec's benchmark signal: `periods` pulses (20 edges for 10), 2 %-of-period edges, noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, periods, n)
    ph = x % 1.0
    y = np.clip((ph - 0.2) / 0.02, 0, 1) - np.clip((ph - 0.6) / 0.02, 0, 1)
    return x, y + noise * rng.standard_normal(n)


def two_level_noisy(n=10_000, sigma=0.02, seed=0):
    """Half the record at 0, half at 1, plus Gaussian noise (§C13)."""
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) < n // 2, 0.0, 1.0)
    return y + sigma * rng.standard_normal(n)


def dense_noisy_edge(n=1_000_000, edge_samples=2000, sigma=0.01, seed=0):
    """One 0 -> 1 linear edge of `edge_samples` samples at n // 2 - edge_samples // 2, noise sigma (§C18).
    x = arange(n); the noiseless 10-90 % duration is 0.8 * edge_samples."""
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    start = n // 2 - edge_samples // 2
    y = np.clip((x - start) / edge_samples, 0, 1) + sigma * rng.standard_normal(n)
    return x, y, start + edge_samples / 2


class TransitionAsserts:
    """Mixin for unittest.TestCase: compare a Transition field by field with a clause in the message."""

    def assertTransition(self, actual, expected, clause, delta=1e-9):
        self.assertIsInstance(
            actual, Transition, msg=f"{clause}: expected a Transition, got {actual!r}"
        )
        self.assertEqual(
            actual.polarity, expected.polarity, msg=f"{clause}: polarity of {actual!r}"
        )
        for name in (
            "lower_instant",
            "mid_instant",
            "upper_instant",
            "lower_level",
            "mid_level",
            "upper_level",
        ):
            self.assertAlmostEqual(
                getattr(actual, name),
                getattr(expected, name),
                delta=delta,
                msg=f"{clause}: {name} is {getattr(actual, name)!r}, spec says {getattr(expected, name)!r}",
            )
        self.assertEqual(
            (actual.start_index, actual.end_index),
            (expected.start_index, expected.end_index),
            msg=f"{clause}: (start_index, end_index) of {actual!r}",
        )
