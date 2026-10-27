"""Measured photoreceiver pulse response: two photodiodes, small and large signal.

    PYTHONPATH=src python examples/photoreceiver_pulse.py

A photoreceiver (10 MΩ transimpedance) with two photodiodes, A and B,
driven by 10 µs light pulses at 0.5 V and at 5 V / 10 V output swing (examples/data).
Each capture starts with a one-sample trigger glitch that reads as a 2.5 ns falling edge,
so the analysis uses bounds= to start after it. Traces are normalised to their own state
levels so the two photodiodes overlay; the panels zoom on the rising and falling edges.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import FALLING, RISING, load_csv, save

DIODES = {"A": RISING, "B": FALLING}
SIGNALS = {"small": "0.5 V swing", "large": "large swing (5 V A, 10 V B)"}
SKIP_GLITCH = 200.0  # ns: start the analysis after the trigger glitch in the first sample
ZOOM = 1500.0  # ns either side of each edge's mid instant


def main() -> None:
    data = load_csv("photoreceiver_pulses.csv")
    t = data["time_ns"]
    glitch = pt.transitions(data["A_small_V"], x=t)[0]
    print(f"Without bounds the glitch is a {glitch.polarity.name.lower()} edge of {glitch.duration:.1f} ns at t=0")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, (signal, title) in enumerate(SIGNALS.items()):
        for diode, color in DIODES.items():
            y = data[f"{diode}_{signal}_V"]
            rise, fall = pt.transition_metrics(y, x=t, bounds=(SKIP_GLITCH, t[-1]))
            low, high = rise.levels
            normalised = (y - low) / (high - low)
            for col, m in enumerate((rise, fall)):
                edge = m.transition
                print(f"{diode:>8} {signal}: {edge.polarity.name.lower():>7} {edge.duration:5.0f} ns, "
                      f"overshoot {m.post_overshoot.percent:3.1f} %, settling {m.settling_time:4.0f} ns")
                ax = axes[row, col]
                window = (t > edge.mid_instant - ZOOM) & (t < edge.mid_instant + ZOOM)
                ax.plot((t[window] - edge.mid_instant) / 1e3, normalised[window], color=color, linewidth=1.2,
                        label=f"Photodiode {diode}: {edge.duration:.0f} ns, {m.post_overshoot.percent:.1f} %, {m.settling_time:.0f} ns")
        for col, name in enumerate(("Rising edge", "Falling edge")):
            ax = axes[row, col]
            for level in (0.1, 0.5, 0.9):
                ax.axhline(level, color="#a0aec0", linestyle="--", linewidth=0.8)
            ax.set_title(f"{name}, {title}", fontsize=10)
            ax.legend(loc="center left", fontsize=8, title="10–90 %, overshoot, settling",
                      title_fontsize=8)
            ax.set_ylim(-0.08, 1.12)
    for ax in axes[1]:
        ax.set_xlabel("Time from the mid instant (µs)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalised output")
    fig.suptitle("Photoreceiver pulse response: 10–90 % duration, overshoot and settling (10 MΩ photoreceiver)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
