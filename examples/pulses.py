"""pulses: width, period, separation and duty cycle of a PWM signal.

    PYTHONPATH=src python examples/pulses.py

A positive pulse runs from a rising mid instant to the next falling one. Its period runs
to the next rising mid instant and its separation from its falling edge to that instant,
so the last pulse in a record has NaN period, separation and duty cycle.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import ACCENT, FALLING, RISING, SIGNAL, ramp_edges, save

PERIOD = 100e-6
DUTY = (0.2, 0.35, 0.5, 0.65, 0.8, 0.3)


def main() -> None:
    t = np.linspace(0.0, 620e-6, 31_000)
    edges = []
    for k, duty in enumerate(DUTY):
        start = 10e-6 + k * PERIOD
        edges += [(start, 1e-6, 5.0), (start + duty * PERIOD, 1e-6, -5.0)]
    y = ramp_edges(t, edges) + np.random.default_rng(4).normal(0.0, 0.05, t.size)

    found = pt.pulses(y, x=t)
    for pulse in found:
        print(
            f"width {pulse.width * 1e6:5.1f} us, period {pulse.period * 1e6:5.1f} us, "
            f"separation {pulse.separation * 1e6:5.1f} us, duty cycle {pulse.duty_cycle:.3f}"
        )

    fig, (ax, duty_ax) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 5.5), gridspec_kw={"height_ratios": [3, 1.3]}
    )
    ax.plot(t * 1e6, y, color=SIGNAL, linewidth=0.8)

    shown = found[2]
    first, second = shown.first.mid_instant * 1e6, shown.second.mid_instant * 1e6
    for start, end, height, label, color in (
        (first, second, 5.9, f"width\n{shown.width * 1e6:.1f} µs", RISING),
        (
            second,
            first + shown.period * 1e6,
            5.9,
            f"separation\n{shown.separation * 1e6:.1f} µs",
            FALLING,
        ),
        (
            first,
            first + shown.period * 1e6,
            7.1,
            f"period {shown.period * 1e6:.1f} µs",
            ACCENT,
        ),
    ):
        ax.annotate(
            "",
            xy=(end, height),
            xytext=(start, height),
            arrowprops={
                "arrowstyle": "<->",
                "color": color,
                "linewidth": 1.1,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.text(
            (start + end) / 2,
            height + 0.1,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
        )
    for instant in (first, second, first + shown.period * 1e6):
        ax.axvline(instant, color="#a0aec0", linestyle=":", linewidth=0.9)
    for n, pulse in enumerate(found, start=1):
        ax.plot(
            [pulse.first.mid_instant * 1e6, pulse.second.mid_instant * 1e6],
            [pulse.first.mid_level, pulse.second.mid_level],
            "o",
            color=RISING,
            markersize=3.5,
        )
        ax.text(
            (pulse.first.mid_instant + pulse.second.mid_instant) / 2 * 1e6,
            4.4,
            f"#{n}\n{pulse.width * 1e6:.0f} µs",
            ha="center",
            va="top",
            fontsize=8,
            color=RISING,
        )
    ax.set_ylim(-0.5, 7.9)
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("pulses: width, separation, period and duty cycle")

    complete = [p for p in found if math.isfinite(p.duty_cycle)]
    # Each duty cycle covers its period: from the pulse's rising mid instant to the next one.
    duty_ax.bar(
        [p.first.mid_instant * 1e6 for p in complete],
        [p.duty_cycle * 100 for p in complete],
        width=[p.period * 1e6 for p in complete],
        align="edge",
        color=RISING,
        alpha=0.35,
        edgecolor="white",
        linewidth=1.5,
    )
    for p in complete:
        duty_ax.text(
            (p.first.mid_instant + p.period / 2) * 1e6,
            p.duty_cycle * 100 + 3,
            f"{p.duty_cycle * 100:.0f} %",
            ha="center",
            fontsize=8,
        )
    for p in found:
        if not math.isfinite(p.duty_cycle):
            duty_ax.text(
                p.first.mid_instant * 1e6 + 6,
                8,
                "last pulse: no next rising edge,\nso duty cycle is NaN",
                ha="left",
                va="bottom",
                fontsize=8,
                color="#718096",
            )
    duty_ax.set_ylim(0, 100)
    duty_ax.set_ylabel("Duty cycle (%)")
    duty_ax.set_xlabel("Time (µs)")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
