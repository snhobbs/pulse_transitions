"""transitions: every edge of a noisy pulse train, with its instants, duration and slew rate.

    PYTHONPATH=src python examples/transitions.py

Each Transition records its polarity, the instants where it crosses the lower, mid and
upper reference levels, its duration (the rise or fall time) and its slew rate. Pass
polarity=Polarity.RISING or FALLING to keep one direction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import FALLING, REFERENCE, RISING, SIGNAL, ramp_edges, save


def main() -> None:
    t = np.linspace(0.0, 10e-6, 20_000)
    # Rising edges get slower and falling edges faster through the record.
    edges = [(0.5e-6, 0.2e-6, 1.0), (2.0e-6, 0.4e-6, -1.0), (3.5e-6, 0.4e-6, 1.0),
             (5.0e-6, 0.2e-6, -1.0), (6.5e-6, 0.8e-6, 1.0), (8.5e-6, 0.1e-6, -1.0)]
    y = ramp_edges(t, edges) + np.random.default_rng(2).normal(0.0, 0.02, t.size)

    levels = pt.state_levels(y, x=t)
    found = pt.transitions(y, x=t, levels=levels.levels)
    for edge in found:
        print(f"{edge.polarity.name:>7} at {edge.mid_instant * 1e6:5.2f} us: "
              f"duration {edge.duration * 1e9:6.1f} ns, slew rate {edge.slew_rate / 1e6:+6.2f} V/us")
    print(f"{len(pt.transitions(y, x=t, polarity=pt.Polarity.RISING))} rising edges")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t * 1e6, y, color=SIGNAL, linewidth=0.8)
    for value in pt.reference_levels(levels.levels):
        ax.axhline(value, color=REFERENCE, linestyle="--", linewidth=0.8)
    for edge in found:
        color = RISING if edge.polarity is pt.Polarity.RISING else FALLING
        ax.axvspan(edge.start * 1e6, edge.end * 1e6, color=color, alpha=0.15, linewidth=0)
        ax.plot([edge.lower_instant * 1e6, edge.mid_instant * 1e6, edge.upper_instant * 1e6],
                [edge.lower_level, edge.mid_level, edge.upper_level], "o", color=color, markersize=4)
        ax.annotate(f"{edge.duration * 1e9:.0f} ns\n{edge.slew_rate / 1e6:+.1f} V/µs",
                    (edge.end * 1e6, edge.mid_level), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8, color=color)
    ax.plot([], [], "o", color=RISING, label="Rising: lower, mid and upper instants")
    ax.plot([], [], "o", color=FALLING, label="Falling: lower, mid and upper instants")
    ax.plot([], [], "--", color=REFERENCE, label="10 / 50 / 90 % reference levels")

    ax.set_title("transitions: duration and slew rate of every edge")
    ax.set_xlabel("Time (µs)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_ylim(-0.15, 1.35)
    ax.legend(loc="upper left", ncols=3)
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
