"""Measured TDR traces: one probe and line into five terminations.

    PYTHONPATH=src python examples/tdr_terminations.py

Each trace has an incident edge near 1 ns and the termination's reflection near 4.4 ns
(the TDR01 sampler's output is inverted, so the incident step falls and an open steps
down). measure_edge finds each edge near where it's expected and measures it between the
flat levels either side; on the waveguide trace it also picks out the second edge, where
the waveguide's open far end is reached. The matched 50 Ω load reflects nothing, so it has no edge to find
and measure_edge raises EdgeNotFoundError.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import pulse_transitions as pt
from _plot import load_csv, save

TERMINATIONS = {
    "open": "#c05621",
    "short": "#2b6cb0",
    "100_ohm": "#805ad5",
    "waveguide": "#2f855a",
    "50_ohm": "#4a5568",
}
INCIDENT, REFLECTION, SEARCH = (
    1050.0,
    4450.0,
    600.0,
)  # ps: expected edge instants and how far to look
WAVEGUIDE_END = (
    5500.0  # ps: the waveguide's far end (open), a second edge on that trace
)


def main() -> None:
    data = load_csv("tdr_terminations.csv")
    t = data["time_ps"]

    incident = pt.measure_edge(
        data["open_V"], x=t, around=INCIDENT, max_distance=SEARCH
    )
    print(
        f"Incident edge: {incident.duration:.0f} ps (10-90 %), {incident.levels[1] - incident.levels[0]:.3f} V"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    window = t <= 10000
    rows = []  # (text, color) for the results table
    for name, color in TERMINATIONS.items():
        y = data[f"{name}_V"]
        label = name.replace("_ohm", " Ω")
        ax.plot(t[window] / 1e3, y[window], color=color, linewidth=1.1)
        try:
            edge = pt.measure_edge(y, x=t, around=REFLECTION, max_distance=SEARCH)
        except pt.EdgeNotFoundError as error:
            print(f"{label:>9}: no reflection ({error})")
            rows.append((f"{label:<9}  no edge: matched, nothing reflected", color))
            continue
        edges = [(label, edge)]
        if name == "waveguide":
            edges.append(
                (
                    "  far end",
                    pt.measure_edge(y, x=t, around=WAVEGUIDE_END, max_distance=SEARCH),
                )
            )
        for text, found in edges:
            tr = found.transition
            step = (found.levels[1] - found.levels[0]) * int(tr.polarity)
            print(
                f"{text:>9}: {tr.polarity.name.lower():>7} {step * 1e3:+6.0f} mV, {found.duration:4.0f} ps at {tr.mid_instant:.0f} ps"
            )
            ax.plot(
                [tr.lower_instant / 1e3, tr.upper_instant / 1e3],
                [tr.lower_level, tr.upper_level],
                "o",
                color=color,
                markersize=4,
            )
            rows.append(
                (
                    f"{text:<9}  {step * 1e3:+5.0f} mV  {found.duration:4.0f} ps  at {tr.mid_instant / 1e3:.2f} ns",
                    color,
                )
            )

    ax.axvspan(
        (INCIDENT - SEARCH) / 1e3,
        (INCIDENT + SEARCH) / 1e3,
        color="#a0aec0",
        alpha=0.12,
        linewidth=0,
    )
    ax.axvspan(
        (REFLECTION - SEARCH) / 1e3,
        (REFLECTION + SEARCH) / 1e3,
        color="#a0aec0",
        alpha=0.12,
        linewidth=0,
    )
    ax.text(
        INCIDENT / 1e3,
        2.33,
        f"incident edge\n{incident.duration:.0f} ps",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.text(
        REFLECTION / 1e3,
        2.33,
        "reflection\n(search window)",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    ax.text(
        5.6,
        2.42,
        "Reflection (10-90 % duration)",
        fontsize=8,
        fontweight="bold",
        va="top",
    )
    for k, (text, color) in enumerate(rows):
        ax.text(
            5.6,
            2.385 - k * 0.032,
            text,
            fontsize=8,
            va="top",
            color=color,
            family="monospace",
        )
    ax.set_xlim(0, 10.2)
    ax.set_ylim(1.7, 2.45)
    ax.set_title(
        "measure_edge on measured TDR traces: the reflection from each termination"
    )
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Sampler output (V)")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
