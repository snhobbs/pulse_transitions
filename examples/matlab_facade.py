"""matlab: the same measurements with MATLAB's function names, arguments and array outputs.

    PYTHONPATH=src python examples/matlab_facade.py

pulse_transitions.matlab takes (y, t) positionally, returns numpy arrays like MATLAB's
[a, b, c] outputs, and adds no logic of its own: every value matches the main API.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _plot import ACCENT, FALLING, RISING, SIGNAL, ramp_edges, save
from pulse_transitions import matlab


def main() -> None:
    t = np.linspace(0.0, 50e-3, 25_000)
    edges = []
    for k in range(5):
        start = 2e-3 + k * 10e-3
        edges += [(start, 0.4e-3, 1.0), (start + (3 + k) * 1e-3, 0.2e-3, -1.0)]
    y = ramp_edges(t, edges) + np.random.default_rng(8).normal(0.0, 0.01, t.size)

    levels, _, _ = matlab.statelevels(y)
    rise, rise_lower, rise_upper, ll, ul = matlab.risetime(y, t)
    fall, fall_lower, fall_upper, _, _ = matlab.falltime(y, t)
    mid = matlab.midcross(y, t)
    width, _, _ = matlab.pulsewidth(y, t)
    duty, initcross, _, nextcross = matlab.dutycycle(y, t)
    print(f"statelevels   {np.round(levels, 3)}")
    print(
        f"risetime (ms) {np.round(rise * 1e3, 3)}  reference levels {ll:.3f}, {ul:.3f}"
    )
    print(f"falltime (ms) {np.round(fall * 1e3, 3)}")
    print(f"pulsewidth    {np.round(width * 1e3, 2)} ms")
    print(f"dutycycle     {np.round(duty, 3)}")

    fig, (ax, table) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 5.5), gridspec_kw={"height_ratios": [2.4, 1]}
    )
    ms = 1e3
    ax.plot(t * ms, y, color=SIGNAL, linewidth=0.8)
    ax.axhline(
        ll,
        color="#a0aec0",
        linestyle="--",
        linewidth=0.8,
        label="risetime levels ll, ul",
    )
    ax.axhline(ul, color="#a0aec0", linestyle="--", linewidth=0.8)
    ax.plot(
        rise_lower * ms, np.full_like(rise_lower, ll), ">", color=RISING, markersize=4
    )
    ax.plot(
        rise_upper * ms,
        np.full_like(rise_upper, ul),
        ">",
        color=RISING,
        markersize=4,
        label="risetime: tlower, tupper",
    )
    ax.plot(
        fall_upper * ms, np.full_like(fall_upper, ul), "<", color=FALLING, markersize=4
    )
    ax.plot(
        fall_lower * ms,
        np.full_like(fall_lower, ll),
        "<",
        color=FALLING,
        markersize=4,
        label="falltime: tupper, tlower",
    )
    ax.plot(
        mid * ms,
        np.full_like(mid, np.mean(levels)),
        "o",
        color=ACCENT,
        markersize=4,
        mfc="none",
        label="midcross",
    )
    ax.set_ylim(-0.15, 1.45)
    ax.set_ylabel("Amplitude (V)")
    ax.set_title(
        "matlab: statelevels, risetime, falltime, midcross, pulsewidth, dutycycle"
    )
    ax.legend(loc="upper left", ncols=4)

    # Each duty cycle covers its period, initcross to nextcross; the last pulse has no complete period.
    table.bar(
        initcross * ms,
        duty * 100,
        width=(nextcross - initcross) * ms,
        align="edge",
        color=RISING,
        alpha=0.35,
        edgecolor="white",
        linewidth=1.5,
    )
    for x0, x1, w, d in zip(
        initcross * ms, nextcross * ms, width * ms, duty, strict=False
    ):
        table.text(
            (x0 + x1) / 2,
            d * 100 + 4,
            f"width {w:.1f} ms\nduty {d * 100:.0f} %",
            ha="center",
            fontsize=8,
        )
    table.set_ylim(0, 100)
    table.set_ylabel("dutycycle (%)")
    table.set_xlabel("Time (ms)")
    fig.tight_layout()
    save(fig, __file__)


if __name__ == "__main__":
    main()
