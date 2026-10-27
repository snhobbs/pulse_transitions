"""Analytic against measured: the step response of a first-order (RC) system with noise.

    PYTHONPATH=src python examples/analytic_first_order.py

y = 1 - exp(-t / tau) has no overshoot, and every crossing has a closed form: level L is
crossed at tau ln(1 / (1 - L)), so the 10-90 % rise time is tau ln 9, the 20-80 % one tau ln 4,
and the ±2 % settling time, from the 50 % instant, tau ln 25. What noise does to each crossing
also has a closed form: the instant moves by the noise over the slope there, and an RC edge
gets slower as it goes. At 10 % the slope is 0.9 / tau, at 90 % 0.1 / tau and at the edge of
the ±2 % band 0.02 / tau, so the upper crossing jitters nine times as much as the lower one
and the settling time 45 times as much, and reads late, since it is the last entry into the band. Once the noise exceeds the slope times
the sample spacing a level is crossed several times, and the library takes the median crossing
(§C7), so the jitter falls below noise / slope from there on.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, FALLING, REFERENCE, RISING, SIGNAL, save

TAU = 1.0
DT = 0.002
NOISES = np.geomspace(1e-5, 3e-3, 12)  # RMS, fraction of the step
SEEDS = 100
TOLERANCE = 2.0  # settling band, percent of the step (the library's default)
EXACT = {
    "rise_time": TAU * np.log(9),
    "rise_time_2080": TAU * np.log(4),
    "settling_time": TAU * np.log(25),
}
CROSSINGS = (
    ("lower_instant", 0.1, RISING),
    ("upper_instant", 0.9, FALLING),
)  # name, level, colour


def first_order_step(t: np.ndarray) -> np.ndarray:
    return np.where(t >= 0.0, 1.0 - np.exp(-np.clip(t, 0.0, None) / TAU), 0.0)


def slope(level: float) -> float:
    """The slope where the response crosses `level`: (1 - level) / tau."""
    return (1.0 - level) / TAU


def main() -> None:
    rng = np.random.default_rng(0)
    instants = {name: np.empty((NOISES.size, SEEDS)) for name, _, _ in CROSSINGS}
    settling = np.empty((NOISES.size, SEEDS))
    for i, noise in enumerate(NOISES):
        for j in range(SEEDS):
            offset = rng.uniform(
                0.0, DT
            )  # a random sample phase, as on a real digitiser
            t = np.arange(-10 * TAU, 40 * TAU, DT) + offset
            m = pt.step_metrics(
                first_order_step(t) + rng.normal(0.0, noise, t.size), x=t
            )
            for name, level, _ in CROSSINGS:
                instants[name][i, j] = getattr(m.transition, name) - TAU * np.log(
                    1 / (1 - level)
                )
            settling[i, j] = m.settling_time - EXACT["settling_time"]

    long = np.arange(-10 * TAU, 40 * TAU, DT)
    clean = pt.step_metrics(first_order_step(long), x=long)
    clean_2080 = pt.step_metrics(
        first_order_step(long), x=long, reference=(20.0, 50.0, 80.0)
    )
    print(
        f"Noiseless: rise time {clean.transition.duration:.4f} (exact tau ln 9 = {EXACT['rise_time']:.4f}), "
        f"20-80 % {clean_2080.transition.duration:.4f} (tau ln 4 = {EXACT['rise_time_2080']:.4f}), "
        f"settling {clean.settling_time:.4f} (tau ln 25 = {EXACT['settling_time']:.4f})"
    )
    print(" noise   std 10 % / 90 % crossing   settling error mean (std)")
    for i, noise in enumerate(NOISES):
        print(
            f" {noise:7.1e}   {instants['lower_instant'][i].std():.2e} / {instants['upper_instant'][i].std():.2e}"
            f"       {np.nanmean(settling[i]):+.3f} ({np.nanstd(settling[i]):.3f})"
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    ax = axes[0]
    t = np.arange(-1.0, 6.0, DT)
    ax.plot(t, first_order_step(t), color=SIGNAL)
    for level, colour, label in (
        (0.1, RISING, "0.9 / τ"),
        (0.9, FALLING, "0.1 / τ"),
        (1 - TOLERANCE / 100, ACCENT, "0.02 / τ"),
    ):
        at = TAU * np.log(1 / (1 - level))
        ax.plot(at, level, "o", color=colour, markersize=5)
        span = np.array([-1.0, 1.0]) * min(
            0.6, 0.15 / slope(level)
        )  # short tangents, about ±0.15 high
        ax.plot(at + span, level + slope(level) * span, color=colour, linewidth=1)
        ax.annotate(
            f"slope {label}",
            (at, level),
            xytext=(8, -14),
            textcoords="offset points",
            color=colour,
            fontsize=9,
        )
    ax.axhspan(
        1 - TOLERANCE / 100,
        1 + TOLERANCE / 100,
        color=REFERENCE,
        alpha=0.25,
        linewidth=0,
    )
    ax.set(
        xlabel="Time (τ)",
        ylabel="Response",
        title="Crossings and their slopes",
        xlim=(-1, 6),
    )

    ax = axes[1]
    for name, level, colour in CROSSINGS:
        ax.loglog(
            NOISES,
            instants[name].std(axis=1),
            "o",
            color=colour,
            markersize=4,
            label=f"{level:.0%} crossing, measured",
        )
        ax.loglog(
            NOISES,
            NOISES / slope(level),
            color=colour,
            linewidth=1,
            label=f"noise / slope ({slope(level):g} / τ)",
        )
        ax.axvline(
            slope(level) * DT,
            color=colour,
            linestyle=":",
            linewidth=1,
            label="noise = slope × sample spacing" if level > 0.5 else None,
        )
    ax.set(
        xlabel="RMS noise (fraction of step)",
        ylabel="Std of crossing instant (τ)",
        title="Timing jitter",
    )
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    mean = np.nanmean(settling, axis=1)
    low, high = np.nanpercentile(settling, [5, 95], axis=1)
    ax.fill_between(
        NOISES,
        low,
        high,
        color=ACCENT,
        alpha=0.15,
        linewidth=0,
        label=f"5-95 % of {SEEDS} seeds",
    )
    ax.plot(NOISES, mean, "o-", color=ACCENT, markersize=4, label="Mean")
    ax.plot(
        NOISES,
        NOISES / slope(1 - TOLERANCE / 100),
        ":",
        color=SIGNAL,
        label="noise / slope (0.02 / τ)",
    )
    ax.axhline(0.0, color="#718096", linewidth=0.8)
    ax.set_xscale("log")
    ax.set(
        xlabel="RMS noise (fraction of step)",
        ylabel="Measured - exact (τ)",
        title="Settling time error",
    )
    ax.legend(loc="upper left")

    fig.suptitle(
        "First-order step: exact crossings τ ln(1 / (1 - L)) against measured, as noise grows",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
