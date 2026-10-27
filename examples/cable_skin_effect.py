"""Measured skin-effect step: why a cable's rise time needs its levels stated.

    PYTHONPATH=src python examples/cable_skin_effect.py

The open end of 50 m of RG58 seen through the cable (examples/data). Skin effect gives the
step a long tail that is still creeping up when the record ends, so the "high" state is a
matter of choice: each state_levels method picks a different one, and the 10-90 % rise
time follows it. Narrower reference levels (20-80 %) cut the tail's weight but still
depend on the levels. Report the method and reference with the number.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import pulse_transitions as pt
from _plot import SIGNAL, load_csv, save

METHODS = {"mode": "#2b6cb0", "mean": "#2f855a", "peak": "#c05621", "endpoints": "#805ad5"}
REFERENCES = {"10–90 %": (10, 50, 90), "20–80 %": (20, 50, 80)}


def main() -> None:
    data = load_csv("rg58_50m_open.csv")
    t, y = data["time_ps"] / 1e3, data["open_V"]  # ns

    fig, (ax, bars) = plt.subplots(1, 2, figsize=(10.5, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    ax.plot(t, y, color=SIGNAL, linewidth=0.8)
    rise_times = {}
    for method, color in METHODS.items():
        levels = pt.state_levels(y, x=t, method=method)
        rise_times[method] = [pt.transitions(y, x=t, levels=levels.levels, reference=ref)[0].duration
                              for ref in REFERENCES.values()]
        print(f"{method:>9}: high {levels.high:.3f} V, " +
              ", ".join(f"{name} rise {d:.1f} ns" for name, d in zip(REFERENCES, rise_times[method], strict=True)))
        ax.axhline(levels.high, color=color, linestyle="--", linewidth=1.0, label=f"{method}: high {levels.high:.3f} V")
        ax.axhline(levels.low, color=color, linestyle="--", linewidth=1.0)
    ax.set_title("50 m RG58, open end: where is the top?")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Sampler output (V)")
    ax.legend(loc="center right", fontsize=8)

    width = 0.38
    positions = np.arange(len(METHODS))
    for k, (name, shade) in enumerate(zip(REFERENCES, ("#4a5568", "#a0aec0"), strict=True)):
        values = [rise_times[m][k] for m in METHODS]
        bars.bar(positions + (k - 0.5) * width, values, width, color=shade, label=name)
        for position, value in zip(positions + (k - 0.5) * width, values, strict=True):
            bars.text(position, value, f"{value:.0f}", ha="center", va="bottom", fontsize=8)
    bars.set_xticks(positions, list(METHODS))
    bars.set_xlabel("state_levels method")
    bars.set_ylabel("Rise time (ns)")
    bars.set_title("Rise time by levels and reference")
    bars.legend(loc="upper left")
    bars.grid(axis="x", visible=False)
    bars.set_ylim(0, 1.2 * max(max(v) for v in rise_times.values()))
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
