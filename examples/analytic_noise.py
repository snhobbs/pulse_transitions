"""Analytic against measured: how noise biases the step metrics, with and without filtering.

    PYTHONPATH=src python examples/analytic_noise.py

A second-order step (damping ratio 0.5) with increasing RMS noise, measured raw and after a
zero-phase low-pass, over many seeds. Each panel shows the measurement error against the
exact value from the closed form. Overshoot and undershoot are the largest excursion past the
state level, so noise peaks add to them: about twice the RMS noise, unfiltered. Once the
largest noise peaks (about four times the RMS noise in this 5000-sample record) reach the
±2 % band, the settling time follows the last noise peak instead of the ringing. Rise time is barely affected. Filter before measuring.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import (
    ACCENT,
    FALLING,
    RISING,
    lowpass,
    save,
    second_order_exact,
    second_order_step,
)

WN = 1.0
ZETA = 0.5
NOISES = np.array(
    [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
)  # RMS, percent of the step
TOLERANCE = 2.0  # settling band, percent of the step (the library's default)
CUTOFF = 15.0  # low-pass -3 dB point, in wn
SEEDS = 50
ABERRATION_FACTOR = 5.0
DT = 0.02
T = np.arange(-20.0, 80.0, DT)
VARIANTS = {"Unfiltered": FALLING, f"Low-pass at {CUTOFF:g} ωn": RISING}
PANELS = (
    ("overshoot", "Overshoot error (points of %)"),
    ("undershoot", "Undershoot error (points of %)"),
    ("rise_time", "Rise time error (%)"),
    ("settling_time", "Settling time error (1/ωn)"),
)


def errors(y: np.ndarray, exact) -> dict[str, float]:
    """Measured minus exact: in percentage points for the aberrations, relative for the rise time."""
    m = pt.step_metrics(y, x=T, aberration_factor=ABERRATION_FACTOR)
    return {
        "overshoot": m.post_overshoot.percent - exact.overshoot,
        "undershoot": m.post_undershoot.percent - exact.undershoot,
        "rise_time": 100 * (m.transition.duration / exact.rise_time - 1),
        "settling_time": m.settling_time - exact.settling_time,
    }


def main() -> None:
    exact = second_order_exact(WN, ZETA)
    clean = second_order_step(T, WN, ZETA)
    result = {
        (v, name): np.empty((NOISES.size, SEEDS))
        for v in VARIANTS
        for name, _ in PANELS
    }
    for i, sigma in enumerate(NOISES):
        for seed in range(SEEDS):
            noisy = clean + np.random.default_rng(seed).normal(0.0, sigma / 100, T.size)
            filtered = lowpass(noisy, 1 / DT, CUTOFF * WN / (2 * np.pi))
            for variant, y in zip(VARIANTS, (noisy, filtered), strict=True):
                for name, value in errors(y, exact).items():
                    result[variant, name][i, seed] = value

    print(
        f"Exact: overshoot {exact.overshoot:.2f} %, undershoot {exact.undershoot:.2f} %, "
        f"rise time {exact.rise_time:.3f}, settling {exact.settling_time:.2f} (1/wn)"
    )
    for variant in VARIANTS:
        print(variant)
        for i, sigma in enumerate(NOISES):
            cells = "  ".join(
                f"{name} {np.nanmean(result[variant, name][i]):+7.2f}"
                for name, _ in PANELS
            )
            print(f"  noise {sigma:4.2f} %: {cells}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (name, label) in zip(axes.flat, PANELS, strict=True):
        for variant, color in VARIANTS.items():
            values = result[variant, name]
            low, high = np.nanpercentile(values, [5, 95], axis=1)
            ax.fill_between(NOISES, low, high, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                NOISES,
                np.nanmean(values, axis=1),
                "o-",
                color=color,
                markersize=3.5,
                label=variant,
            )
        ax.axhline(0.0, color="#718096", linewidth=0.8)
        ax.set_ylabel(label)
    axes[0, 0].plot(NOISES, 2 * NOISES, ":", color=ACCENT, label="2 × RMS noise")
    axes[0, 0].legend(
        loc="upper left", title=f"Mean, 5-95 % of {SEEDS} seeds", title_fontsize=9
    )
    axes[1, 1].axvline(TOLERANCE / 4, color=ACCENT, linestyle=":", linewidth=1)
    axes[1, 1].text(
        TOLERANCE / 4 + 0.25,
        58,
        "4 × RMS noise\n= ±2 % band:\nsettling follows\nthe last noise peak",
        fontsize=8,
        color=ACCENT,
        va="top",
    )
    for ax in axes[1]:
        ax.set_xlabel("RMS noise (% of the step)")
    fig.suptitle(
        f"Second-order step (ζ = {ZETA:g}): measured minus exact as noise grows",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
