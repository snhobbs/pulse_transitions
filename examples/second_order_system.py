import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lsim
from scipy.signal import lti

from pulse_transitions import detect_edges
from pulse_transitions import detect_signal_levels
from pulse_transitions import get_edge_metrics


# Example: synthetic underdamped step response
def generate_step_response(t, damping=0.2, freq=10):
    # Second-order system parameters
    wn = freq          # Natural frequency (rad/s)
    zeta = damping        # Damping ratio

    # Transfer function: H(s) = wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
    num = [wn**2]
    den = [1, 2*zeta*wn, wn**2]

    # Create the system
    system = lti(num, den)

    u = [0]*len(t[t<0]) + [1]*len(t[t>=0])
    t_shift = t-min(t)
    t_out, y_out, _ = lsim(system, U=u, T=t_shift)

    return t, y_out


def make_annotations_plot(t, y, edge, metrics):
    t_low, t_high = edge.start, edge.end
    trise = t_high - t_low
    mid_level = np.mean(metrics["levels"])

    fractional_thresholds = metrics["fractional_thresholds"]
    absolute_thresholds = metrics["absolute_thresholds"]
    threshold_colors = ["C1", "C2"]
    edge_color = "black"
    signal_color = "C0"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, y, label="Signal", color=signal_color)

    # Draw absolute thresholds and level lines
    for frac, abs_val, color in zip(fractional_thresholds, absolute_thresholds, threshold_colors, strict=False):
        ax.axhline(abs_val, linestyle="--", color=color, label=f"{int(frac * 100)}% Threshold")
        ax.annotate(f"{int(frac * 100)}%",
                    xy=(t[0], abs_val),
                    xytext=(20, 2.5), textcoords="offset points",
                    color=color)

    # Edge start/end lines
    edge_label = metrics.get("edge_label", "Edge")
    ax.axvline(t_low, linestyle=":", color=edge_color, label=f"{edge_label} start")
    ax.axvline(t_high, linestyle=":", color=edge_color, label=f"{edge_label} end")
    ax.plot([metrics["midcross"]], [mid_level], "kx", label="Mid Cross")

    # Annotate rise time & slew rate
    slewrate = metrics.get("slewrate", np.nan)
    ax.annotate(f"Rise: {trise * 1e3:.0f} ps\nSlew: {slewrate:.1f} V/ns",
                xy=(t_high, mid_level),
                xytext=(20, 0), textcoords="offset points",
                ha="left", va="center")

    idx, overshoot = metrics["overshoot"]
    ax.plot(t[idx], y[idx], "ro", label="Overshoot")
    ax.annotate(f"Overshoot: {overshoot * 100:.0f}%",
                xy=(t[idx], y[idx]),
                xytext=(20, 0), textcoords="offset points",
                ha="left", va="center")

    idx, undershoot = metrics["undershoot"]
    ax.plot(t[idx], y[idx], "go", label="Undershoot")
    ax.annotate(f"Undershoot: {undershoot * 100:.0f}%",
                xy=(t[idx], y[idx]),
                xytext=(20, 0), textcoords="offset points",
                ha="left", va="center")

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("Step Response with Edge Detection")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, color="lightgray")  #  noqa: FBT003
    plt.tight_layout()
    return fig, ax



# Generate data
t = np.linspace(-0.25, 1, 1000)
t_out, y = generate_step_response(t, freq=25/(max(t)))


#plt.plot(t, y)
#plt.show()

threshold_fractions=(0.1, 0.9)

# Estimate levels (e.g. using histogram or endpoints)
levels = detect_signal_levels(x=t, y=y, method="histogram")
low, high = levels

# Detect edges
edge = detect_edges(t, y, levels=levels, thresholds=threshold_fractions)[0]

# Compute overshoot / undershoot
metrics = get_edge_metrics(x=t, y=y, thresholds=threshold_fractions, levels=levels)

fig, ax = make_annotations_plot(t, y, edge, metrics)
fig.savefig("second-order-annotations.png", dpi=300)
plt.show()
