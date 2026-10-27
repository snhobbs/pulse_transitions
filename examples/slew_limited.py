"""Analytic against measured: a slew-limited step, from linear settling to a pure ramp.

    PYTHONPATH=src python examples/slew_limited.py

An amplifier with time constant tau and slew rate S, given a step of height V, ramps at S
until the remaining error is S tau, then settles exponentially from there (the slope is
continuous at the join). The one parameter is a = V / (S tau): below 1 the amplifier never
slews and the step is a first-order exponential, well above 1 it is nearly a ramp. Each
level L then has a closed-form crossing: L a tau on the ramp, else
max(a - 1, 0) tau + tau ln(min(1, 1/a) / (1 - L)) on the tail.

The ratio of the 20-80 % to the 10-90 % rise time tells the two apart without knowing tau or
S: ln 4 / ln 9 = 0.631 for an exponential, 0.6 / 0.8 = 0.75 for a ramp. A ratio near 0.75
on a measured edge says the edge is slew-limited, and a smaller step will be faster.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, FALLING, REFERENCE, RISING, SIGNAL, save

TAU = 1.0
A = np.geomspace(
    0.1, 100.0, 19
)  # V / (S tau): step height over the slew rate times tau
SHOWN = (0.5, 2.0, 5.0, 30.0)  # waveforms drawn in the first panel
NOISE = 0.0005  # RMS, fraction of the step
SEEDS = 20
SAMPLES_PER_RISE = 200
TOLERANCE = 2.0  # settling band, percent of the step (the library's default)


def slewed_step(t: np.ndarray, a: float) -> np.ndarray:
    """Unit step of an amplifier with time constant TAU slewing at 1 / (a TAU) per unit time."""
    tail = min(
        1.0, 1.0 / a
    )  # the error left when the ramp hands over to the exponential
    join = max(a - 1.0, 0.0) * TAU
    tp = np.clip(t, 0.0, None)
    y = np.where(tp < join, tp / (a * TAU), 1.0 - tail * np.exp(-(tp - join) / TAU))
    return np.where(t >= 0.0, y, 0.0)


def crossing(level: float, a: float) -> float:
    """Exact instant at which `slewed_step` reaches `level`."""
    tail = min(1.0, 1.0 / a)
    if level <= 1.0 - tail:
        return level * a * TAU
    return max(a - 1.0, 0.0) * TAU + TAU * np.log(tail / (1.0 - level))


def exact(a: float) -> dict[str, float]:
    rise = crossing(0.9, a) - crossing(0.1, a)
    return {
        "rise_time": rise,
        "ratio": (crossing(0.8, a) - crossing(0.2, a)) / rise,
        "settling_time": crossing(1.0 - TOLERANCE / 100, a) - crossing(0.5, a),
    }


def record(a: float, offset: float) -> np.ndarray:
    dt = exact(a)["rise_time"] / SAMPLES_PER_RISE
    end = (
        crossing(1.0 - TOLERANCE / 100, a) + 15 * TAU
    )  # long enough for the tail to reach the high level
    return np.arange(-0.5 * end, end, dt) + offset * dt


def main() -> None:
    rng = np.random.default_rng(0)
    names = ("rise_time", "ratio", "settling_time")
    measured = {name: np.empty((A.size, SEEDS)) for name in names}
    for i, a in enumerate(A):
        for j in range(SEEDS):
            t = record(a, rng.uniform())
            y = slewed_step(t, a) + rng.normal(0.0, NOISE, t.size)
            m = pt.step_metrics(y, x=t)
            m2080 = pt.step_metrics(
                y, x=t, levels=m.levels, reference=(20.0, 50.0, 80.0)
            )
            measured["rise_time"][i, j] = m.transition.duration
            measured["ratio"][i, j] = m2080.transition.duration / m.transition.duration
            measured["settling_time"][i, j] = m.settling_time

    print(
        "     a   rise time exact/measured   20-80/10-90 exact/measured   settling exact/measured (tau)"
    )
    for i, a in enumerate(A):
        e = exact(a)
        print(
            f" {a:6.2f}   {e['rise_time']:8.3f} / {measured['rise_time'][i].mean():8.3f}"
            f"         {e['ratio']:.4f} / {measured['ratio'][i].mean():.4f}"
            f"            {e['settling_time']:7.3f} / {np.nanmean(measured['settling_time'][i]):7.3f}"
        )

    dense = np.geomspace(A[0], A[-1], 300)
    exact_dense = [exact(a) for a in dense]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    ax = axes[0, 0]
    for a, colour in zip(SHOWN, (SIGNAL, RISING, ACCENT, FALLING), strict=True):
        rise, mid = exact(a)["rise_time"], crossing(0.5, a)
        u = np.linspace(-2.0, 3.0, 500)
        ax.plot(u, slewed_step(mid + u * rise, a), color=colour, label=f"a = {a:g}")
    ax.axhline(0.1, color=REFERENCE, linewidth=0.8)
    ax.axhline(0.9, color=REFERENCE, linewidth=0.8)
    ax.set(
        xlabel="(t - 50 % instant) / 10-90 % rise time",
        ylabel="Response",
        title="Shape: exponential (a < 1) to ramp (a ≫ 1)",
    )
    ax.legend(loc="lower right")

    panels = (
        (axes[0, 1], "rise_time", "10-90 % rise time (τ)", "loglog"),
        (axes[1, 0], "ratio", "20-80 % / 10-90 % rise time", "semilogx"),
        (axes[1, 1], "settling_time", "Settling time, ±2 %, from 50 % (τ)", "semilogx"),
    )
    for ax, name, label, scale in panels:
        getattr(ax, scale)(
            dense,
            [e[name] for e in exact_dense],
            color=SIGNAL,
            label="Exact (closed form)",
        )
        mean = np.nanmean(measured[name], axis=1)
        low, high = np.nanpercentile(measured[name], [5, 95], axis=1)
        ax.errorbar(
            A,
            mean,
            yerr=(mean - low, high - mean),
            fmt="o",
            color=RISING,
            markersize=4,
            capsize=2,
            linewidth=1,
            label=f"Measured, {NOISE:.2%} RMS noise, {SEEDS} seeds",
        )
        ax.set(xlabel="a = V / (S τ)", ylabel=label)
    ax = axes[0, 1]
    ax.loglog(
        dense,
        np.full_like(dense, TAU * np.log(9)),
        ":",
        color=ACCENT,
        label="τ ln 9 (linear)",
    )
    ax.loglog(dense, 0.8 * dense * TAU, ":", color=FALLING, label="0.8 V / S (slewing)")
    ax.set_ylim(1.0, None)
    ax.legend(loc="upper left")
    ax = axes[1, 0]
    ax.axhline(
        np.log(4) / np.log(9),
        linestyle=":",
        color=ACCENT,
        label="ln 4 / ln 9 = 0.631 (exponential)",
    )
    ax.axhline(0.75, linestyle=":", color=FALLING, label="0.6 / 0.8 = 0.75 (ramp)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.94), fontsize=8)
    axes[1, 1].legend(loc="upper left")

    fig.suptitle(
        "Slew-limited step: exact against measured, from linear settling to slewing",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
