"""
Underdamped square-wave response — predicted vs measured pulse metrics.

Simulates a 2nd-order LTI system with known (ζ, fn) driven by a square wave,
then exercises every metric in pulse_transitions and compares against
closed-form / numerically-exact predictions from the step-response formula.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lsim, lti, square

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pulse_transitions import get_edge_metrics
from pulse_transitions.transient_response import calculate_settling_time

# ── Configuration ─────────────────────────────────────────────────────────────
ZETA = 0.30
FN_HZ = 1_000.0
FS = 500_000
F_SQ = 100.0
N_CYCLES = 3
AMPL_LOW = 0.0
AMPL_HIGH = 2.5

THRESHOLD_LOW = 0.10
THRESHOLD_HIGH = 0.90
SETTLING_BAND = 0.02
N_FINE = 500_000


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class SystemParams:
    omega_n: float
    omega_d: float
    sigma: float
    fd_hz: float
    step_ampl: float
    t_half_s: float
    T_d: float  # ringing period = 1/fd

    @classmethod
    def from_config(
        cls,
        zeta: float,
        fn_hz: float,
        f_sq: float,
        ampl_low: float,
        ampl_high: float,
    ) -> "SystemParams":
        omega_n = 2.0 * np.pi * fn_hz
        omega_d = omega_n * np.sqrt(1.0 - zeta**2)
        sigma = zeta * omega_n
        fd_hz = omega_d / (2.0 * np.pi)
        return cls(
            omega_n=omega_n,
            omega_d=omega_d,
            sigma=sigma,
            fd_hz=fd_hz,
            step_ampl=ampl_high - ampl_low,
            t_half_s=0.5 / f_sq,
            T_d=1.0 / fd_hz,
        )


@dataclass
class MeasurementConfig:
    threshold_low: float = THRESHOLD_LOW
    threshold_high: float = THRESHOLD_HIGH
    settling_band: float = SETTLING_BAND
    n_fine: int = N_FINE

    @property
    def thresholds(self) -> tuple[float, float]:
        return (self.threshold_low, self.threshold_high)


@dataclass
class WaveformSegments:
    t_rise: np.ndarray
    y_rise: np.ndarray
    t_fall: np.ndarray
    y_fall: np.ndarray


@dataclass
class TiltResult:
    value_pct: float
    t_pts: np.ndarray | None  # midpoints of start/end plateau windows
    y_pts: np.ndarray | None  # mean y in each window


@dataclass
class Predictions:
    overshoot_pct: float
    rise_time_us: float
    settling_time_us: float
    undershoot_bounce_pct: float
    true_undershoot_pct: float
    mid_crossing_us: float
    slew_rate_vms: float
    frequency_hz: float
    duty_cycle_pct: float
    tilt_pct: float
    damped_freq_hz: float  # reference only — no measured counterpart
    peak_time_us: float  # reference only — no measured counterpart


@dataclass
class Measurements:
    overshoot_pct: float
    rise_time_us: float
    fall_time_us: float
    settling_rise_us: float
    settling_fall_us: float
    undershoot_bounce_pct: float
    true_undershoot_pct: float
    mid_crossing_us: float
    slew_rate_vms: float
    frequency_hz: float
    duty_cycle_pct: float
    tilt: TiltResult
    # Extras for plot annotations (not in the comparison table)
    trough_abs_v: float
    peak_idx: int
    rise_edge_start_us: float
    rise_edge_end_us: float
    fall_edge_start_us: float
    fall_edge_end_us: float
    bounce_idx: int
    peak_time_us: float
    rising_mid_times: np.ndarray
    falling_mid_times: np.ndarray


@dataclass
class MetricRow:
    name: str
    predicted: float
    measured: float

    @property
    def error_str(self) -> str:
        if not (np.isfinite(self.predicted) and np.isfinite(self.measured)):
            return "—"
        if self.predicted != 0:
            return f"{abs(self.predicted - self.measured) / abs(self.predicted) * 100:.2f}%"
        return f"{abs(self.measured - self.predicted):.4f} abs"

    def to_table_row(self) -> list[str]:
        pred_s = f"{self.predicted:.3f}" if np.isfinite(self.predicted) else "—"
        meas_s = f"{self.measured:.3f}" if np.isfinite(self.measured) else "—"
        return [self.name, pred_s, meas_s, self.error_str]

    def print_row(self) -> None:
        if np.isfinite(self.predicted) and np.isfinite(self.measured):
            print(
                f"{self.name:<28}  {self.predicted:>12.3f}  {self.measured:>12.3f}  {self.error_str}"
            )
        else:
            meas_s = (
                f"{self.measured:>12.3f}" if np.isfinite(self.measured) else "      —"
            )
            print(f"{self.name:<28}  {self.predicted:>12.3f}  {meas_s}        —")


# ── Step-response formula ─────────────────────────────────────────────────────


def step_response(t_arr: np.ndarray, omega_n: float, zeta: float) -> np.ndarray:
    """Normalised (0→1) step response of 2nd-order underdamped system."""
    sigma = zeta * omega_n
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    phi = np.arccos(zeta)
    return 1.0 - np.exp(-sigma * t_arr) / np.sqrt(1.0 - zeta**2) * np.sin(
        omega_d * t_arr + phi
    )


def step_response_deriv(t_arr: np.ndarray, omega_n: float, zeta: float) -> np.ndarray:
    """Time-derivative of the normalised step response."""
    sigma = zeta * omega_n
    omega_d = omega_n * np.sqrt(1.0 - zeta**2)
    phi = np.arccos(zeta)
    env = np.exp(-sigma * t_arr) / np.sqrt(1.0 - zeta**2)
    return env * (
        sigma * np.sin(omega_d * t_arr + phi) - omega_d * np.cos(omega_d * t_arr + phi)
    )


# ── Helper functions ──────────────────────────────────────────────────────────


def midcross_times(
    t: np.ndarray,
    y: np.ndarray,
    level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolated rising and falling midpoint crossing times."""
    above = y >= level
    deltas = np.diff(above.astype(int))

    def _cross(indices):
        return [
            float(t[i] + (level - y[i]) / (y[i + 1] - y[i]) * (t[i + 1] - t[i]))
            for i in indices
        ]

    return (
        np.array(_cross(np.where(deltas == 1)[0])),
        np.array(_cross(np.where(deltas == -1)[0])),
    )


def compute_tilt(
    t_seg: np.ndarray,
    y_seg: np.ndarray,
    ts_rise_s: float,
    params: SystemParams,
) -> TiltResult:
    """Tilt = (mean end plateau − mean start plateau) / amplitude, in %."""
    mask_start = (t_seg >= ts_rise_s) & (t_seg < ts_rise_s + params.T_d)
    mask_end = (t_seg >= params.t_half_s - params.T_d) & (t_seg < params.t_half_s)
    if mask_start.sum() < 3 or mask_end.sum() < 3:
        return TiltResult(np.nan, None, None)
    mean_start = float(np.mean(y_seg[mask_start]))
    mean_end = float(np.mean(y_seg[mask_end]))
    return TiltResult(
        value_pct=(mean_end - mean_start) / params.step_ampl * 100.0,
        t_pts=np.array([ts_rise_s + params.T_d / 2, params.t_half_s - params.T_d / 2]),
        y_pts=np.array([mean_start, mean_end]),
    )


# ── Compute predictions and measurements ──────────────────────────────────────


def compute_predictions(
    params: SystemParams, cfg: MeasurementConfig, zeta: float, f_sq: float
) -> Predictions:
    t_fine = np.linspace(0.0, params.t_half_s, cfg.n_fine)
    yr = step_response(t_fine, params.omega_n, zeta)

    os_frac = float(np.exp(-np.pi * zeta / np.sqrt(1.0 - zeta**2)))

    idx10 = int(np.searchsorted(yr, cfg.threshold_low))
    idx90 = int(np.searchsorted(yr, cfg.threshold_high))
    idx50 = int(np.searchsorted(yr, 0.5))

    outside = np.where(np.abs(yr - 1.0) > cfg.settling_band)[0]
    ts_s = float(t_fine[outside[-1]]) if len(outside) else 0.0

    slew_vms = (
        float(np.max(np.abs(step_response_deriv(t_fine, params.omega_n, zeta))))
        * params.step_ampl
        * 1e-3
    )

    return Predictions(
        overshoot_pct=os_frac * 100.0,
        rise_time_us=(t_fine[idx90] - t_fine[idx10]) * 1e6,
        settling_time_us=ts_s * 1e6,
        undershoot_bounce_pct=os_frac**2 * 100.0,
        true_undershoot_pct=os_frac * 100.0,
        mid_crossing_us=t_fine[idx50] * 1e6,
        slew_rate_vms=slew_vms,
        frequency_hz=f_sq,
        duty_cycle_pct=50.0,
        tilt_pct=0.0,
        damped_freq_hz=params.fd_hz,
        peak_time_us=np.pi / params.omega_d * 1e6,
    )


def compute_measurements(
    t: np.ndarray,
    y_out: np.ndarray,
    segs: WaveformSegments,
    levels: tuple[float, float],
    cfg: MeasurementConfig,
    params: SystemParams,
) -> Measurements:
    kwargs = dict(
        levels=levels,
        fractional_thresholds=cfg.thresholds,
        settling_time_fraction=cfg.settling_band,
    )

    rise_m = get_edge_metrics(segs.t_rise, segs.y_rise, **kwargs)
    fall_m = get_edge_metrics(segs.t_fall, segs.y_fall, **kwargs)
    ts_rise = calculate_settling_time(
        segs.t_rise,
        segs.y_rise,
        levels=levels,
        settling_time_fraction=cfg.settling_band,
    )
    ts_fall = calculate_settling_time(
        segs.t_fall,
        segs.y_fall,
        levels=levels,
        settling_time_fraction=cfg.settling_band,
    )

    trough_abs = float(np.min(segs.y_fall))
    rising_mid, falling_mid = midcross_times(t, y_out, 0.5 * (levels[0] + levels[1]))

    periods = np.diff(rising_mid[1:])
    period = float(np.mean(periods)) if len(periods) > 0 else np.nan
    freq = 1.0 / period if np.isfinite(period) else np.nan

    duty_vals = [
        (falling_mid[ri] - rising_mid[ri])
        / (rising_mid[ri + 1] - rising_mid[ri])
        * 100.0
        for ri in range(1, min(len(rising_mid) - 1, len(falling_mid)))
        if (rising_mid[ri + 1] - rising_mid[ri]) > 0
    ]
    duty = float(np.mean(duty_vals)) if duty_vals else np.nan

    rt_edge = rise_m.risetime[1] if rise_m.risetime else None
    ft_edge = fall_m.falltime[1] if fall_m.falltime else None
    peak_idx = rise_m.overshoot[0]

    return Measurements(
        overshoot_pct=rise_m.overshoot[1] * 100.0,
        rise_time_us=(rise_m.risetime[0] if rise_m.risetime else np.nan) * 1e6,
        fall_time_us=(fall_m.falltime[0] if fall_m.falltime else np.nan) * 1e6,
        settling_rise_us=ts_rise * 1e6,
        settling_fall_us=ts_fall * 1e6,
        undershoot_bounce_pct=abs(fall_m.undershoot[1]) * 100.0,
        true_undershoot_pct=(levels[0] - trough_abs) / params.step_ampl * 100.0,
        mid_crossing_us=rise_m.midcross * 1e6,
        slew_rate_vms=rise_m.slewrate * 1e-3,
        frequency_hz=freq,
        duty_cycle_pct=duty,
        tilt=compute_tilt(segs.t_rise, segs.y_rise, ts_rise, params),
        trough_abs_v=trough_abs,
        peak_idx=peak_idx,
        peak_time_us=float(segs.t_rise[peak_idx]) * 1e6,
        rise_edge_start_us=rt_edge.start * 1e6 if rt_edge else np.nan,
        rise_edge_end_us=rt_edge.end * 1e6 if rt_edge else np.nan,
        fall_edge_start_us=ft_edge.start * 1e6 if ft_edge else np.nan,
        fall_edge_end_us=ft_edge.end * 1e6 if ft_edge else np.nan,
        bounce_idx=fall_m.undershoot[0],
        rising_mid_times=rising_mid,
        falling_mid_times=falling_mid,
    )


def build_rows(pred: Predictions, meas: Measurements) -> list[MetricRow]:
    nan = float("nan")
    return [
        MetricRow("Overshoot (%)", pred.overshoot_pct, meas.overshoot_pct),
        MetricRow("Rise time 10-90% (µs)", pred.rise_time_us, meas.rise_time_us),
        MetricRow("Fall time 10-90% (µs)", pred.rise_time_us, meas.fall_time_us),
        MetricRow(
            "Settling time 2% (µs)", pred.settling_time_us, meas.settling_rise_us
        ),
        MetricRow(
            "Undershoot bounce (%)",
            pred.undershoot_bounce_pct,
            meas.undershoot_bounce_pct,
        ),
        MetricRow(
            "True undershoot (%)", pred.true_undershoot_pct, meas.true_undershoot_pct
        ),
        MetricRow("Mid-crossing (µs)", pred.mid_crossing_us, meas.mid_crossing_us),
        MetricRow("Peak time (µs)", pred.peak_time_us, meas.peak_time_us),
        MetricRow("Slew rate (V/ms)", pred.slew_rate_vms, meas.slew_rate_vms),
        MetricRow("Frequency (Hz)", pred.frequency_hz, meas.frequency_hz),
        MetricRow("Duty cycle (%)", pred.duty_cycle_pct, meas.duty_cycle_pct),
        MetricRow("Pulse tilt (%/pulse)", pred.tilt_pct, meas.tilt.value_pct),
        MetricRow("Damped freq fd (Hz)", pred.damped_freq_hz, nan),
    ]


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot(
    pred: Predictions,
    meas: Measurements,
    rows: list[MetricRow],
    params: SystemParams,
    cfg: MeasurementConfig,
    t: np.ndarray,
    y_out: np.ndarray,
    u_norm: np.ndarray,
    segs: WaveformSegments,
) -> plt.Figure:
    fig = plt.figure(figsize=(14, 13))
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.48, wspace=0.38, height_ratios=[1.4, 1.3, 1.3]
    )
    ax_full = fig.add_subplot(gs[0, :])
    ax_rise = fig.add_subplot(gs[1, 0])
    ax_fall = fig.add_subplot(gs[1, 1])
    ax_tbl = fig.add_subplot(gs[2, :])

    thr_low = AMPL_LOW + cfg.threshold_low * params.step_ampl
    thr_high = AMPL_LOW + cfg.threshold_high * params.step_ampl
    t_us = t * 1e6
    zoom_dur_us = min(params.t_half_s, 6.0 / params.sigma) * 1e6

    # ── Full waveform ─────────────────────────────────────────────────────────
    ax_full.plot(
        t_us,
        AMPL_LOW + params.step_ampl * u_norm,
        "k--",
        lw=0.9,
        alpha=0.45,
        label="Input (ideal)",
    )
    ax_full.plot(t_us, y_out, color="C0", lw=1.5, label="System response")
    ax_full.axhline(AMPL_HIGH, color="C2", lw=0.7, ls=":", alpha=0.7)
    ax_full.axhline(
        AMPL_HIGH * (1 + pred.overshoot_pct / 100),
        color="C3",
        lw=0.7,
        ls="--",
        alpha=0.6,
        label=f"Peak = {AMPL_HIGH * (1 + pred.overshoot_pct / 100):.3f} V",
    )
    ax_full.axhline(
        AMPL_LOW - AMPL_HIGH * pred.overshoot_pct / 100,
        color="C5",
        lw=0.7,
        ls="--",
        alpha=0.6,
        label=f"Trough = {-AMPL_HIGH * pred.overshoot_pct / 100:.3f} V",
    )
    ax_full.set(xlabel="Time (µs)", ylabel="Amplitude (V)")
    ax_full.set_title(
        f"Underdamped 2nd-order Square-Wave Response   ζ={ZETA}  fn={FN_HZ:.0f} Hz  "
        f"fd={params.fd_hz:.1f} Hz   drive={F_SQ:.0f} Hz",
        fontsize=10,
    )
    ax_full.legend(fontsize=8, loc="upper right")
    ax_full.grid(True, alpha=0.25)

    # Period / duty cycle brackets
    if len(meas.rising_mid_times) >= 3:
        t_r1, t_r2 = meas.rising_mid_times[1] * 1e6, meas.rising_mid_times[2] * 1e6
        y_bot, y_hi = (
            AMPL_LOW - 0.55 * params.step_ampl,
            AMPL_LOW - 0.30 * params.step_ampl,
        )
        ax_full.annotate(
            "",
            xy=(t_r2, y_bot),
            xytext=(t_r1, y_bot),
            arrowprops=dict(arrowstyle="<->", color="C7", lw=1.2),
            annotation_clip=False,
        )
        ax_full.text(
            (t_r1 + t_r2) / 2,
            y_bot - 0.05 * params.step_ampl,
            f"T = {meas.frequency_hz and 1 / meas.frequency_hz * 1e3:.2f} ms  ({meas.frequency_hz:.1f} Hz)",
            ha="center",
            va="top",
            fontsize=8,
            color="C7",
        )
        if len(meas.falling_mid_times) >= 2:
            t_f1 = meas.falling_mid_times[1] * 1e6
            ax_full.annotate(
                "",
                xy=(t_f1, y_hi),
                xytext=(t_r1, y_hi),
                arrowprops=dict(arrowstyle="<->", color="C8", lw=1.2),
                annotation_clip=False,
            )
            ax_full.text(
                (t_r1 + t_f1) / 2,
                y_hi + 0.05 * params.step_ampl,
                f"duty={meas.duty_cycle_pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="C8",
            )

    # ── Rising edge zoom ──────────────────────────────────────────────────────
    t_rise_us = segs.t_rise * 1e6
    mask_r = t_rise_us <= zoom_dur_us
    ax_rise.plot(t_rise_us[mask_r], segs.y_rise[mask_r], color="C0", lw=1.5)
    ax_rise.axhline(AMPL_HIGH, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_rise.axhline(
        AMPL_HIGH * (1 + cfg.settling_band),
        color="C2",
        lw=0.8,
        ls=":",
        label=f"±{cfg.settling_band * 100:.0f}% band",
    )
    ax_rise.axhline(AMPL_HIGH * (1 - cfg.settling_band), color="C2", lw=0.8, ls=":")
    ax_rise.axhline(
        thr_high,
        color="C1",
        lw=0.7,
        ls="--",
        alpha=0.7,
        label=f"{cfg.threshold_low * 100:.0f}/{cfg.threshold_high * 100:.0f}%",
    )
    ax_rise.axhline(thr_low, color="C1", lw=0.7, ls="--", alpha=0.7)
    ax_rise.plot(
        t_rise_us[meas.peak_idx],
        segs.y_rise[meas.peak_idx],
        "v",
        color="C3",
        ms=8,
        zorder=5,
        label=f"Peak {segs.y_rise[meas.peak_idx]:.3f} V ({meas.overshoot_pct:.1f}%)",
    )
    if np.isfinite(meas.rise_edge_start_us):
        ax_rise.annotate(
            "",
            xy=(meas.rise_edge_end_us, thr_high),
            xytext=(meas.rise_edge_start_us, thr_low),
            arrowprops=dict(arrowstyle="<->", color="C4", lw=1.5),
        )
        ax_rise.text(
            (meas.rise_edge_start_us + meas.rise_edge_end_us) / 2 + 5,
            AMPL_HIGH * 0.50,
            f"tr={meas.rise_time_us:.1f} µs",
            fontsize=8,
            color="C4",
        )
    ax_rise.axvline(
        meas.settling_rise_us,
        color="C6",
        lw=1.2,
        ls="-.",
        label=f"ts(2%)={meas.settling_rise_us:.1f} µs",
    )
    if meas.tilt.t_pts is not None:
        ax_rise.plot(
            meas.tilt.t_pts * 1e6,
            meas.tilt.y_pts,
            "o--",
            color="C9",
            lw=1.5,
            ms=5,
            label=f"tilt={meas.tilt.value_pct:.3f}%/pulse",
        )
    ax_rise.set(xlabel="Time from edge (µs)", ylabel="Amplitude (V)")
    ax_rise.set_title("Rising edge detail", fontsize=9)
    ax_rise.legend(fontsize=7, loc="lower right")
    ax_rise.grid(True, alpha=0.25)

    # ── Falling edge zoom ─────────────────────────────────────────────────────
    t_fall_us = segs.t_fall * 1e6
    mask_f = t_fall_us <= zoom_dur_us
    ax_fall.plot(t_fall_us[mask_f], segs.y_fall[mask_f], color="C0", lw=1.5)
    ax_fall.axhline(AMPL_LOW, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_fall.axhline(
        AMPL_LOW + cfg.settling_band * params.step_ampl,
        color="C2",
        lw=0.8,
        ls=":",
        label=f"±{cfg.settling_band * 100:.0f}% band",
    )
    ax_fall.axhline(
        AMPL_LOW - cfg.settling_band * params.step_ampl, color="C2", lw=0.8, ls=":"
    )
    ax_fall.axhline(
        thr_low,
        color="C1",
        lw=0.7,
        ls="--",
        alpha=0.7,
        label=f"{cfg.threshold_low * 100:.0f}/{cfg.threshold_high * 100:.0f}%",
    )
    ax_fall.axhline(thr_high, color="C1", lw=0.7, ls="--", alpha=0.7)
    trough_idx = int(np.argmin(segs.y_fall))
    ax_fall.plot(
        t_fall_us[trough_idx],
        segs.y_fall[trough_idx],
        "v",
        color="C3",
        ms=8,
        zorder=5,
        label=f"Trough {meas.trough_abs_v:.3f} V  ({meas.true_undershoot_pct:.1f}% below LOW)",
    )
    ax_fall.axhline(meas.trough_abs_v, color="C3", lw=0.7, ls="--", alpha=0.6)
    ax_fall.plot(
        t_fall_us[meas.bounce_idx],
        segs.y_fall[meas.bounce_idx],
        "^",
        color="C5",
        ms=8,
        zorder=5,
        label=f"Bounce {segs.y_fall[meas.bounce_idx]:.3f} V  ({meas.undershoot_bounce_pct:.1f}% = OS²)",
    )
    if np.isfinite(meas.fall_edge_start_us):
        ax_fall.annotate(
            "",
            xy=(meas.fall_edge_end_us, thr_low),
            xytext=(meas.fall_edge_start_us, thr_high),
            arrowprops=dict(arrowstyle="<->", color="C4", lw=1.5),
        )
        ax_fall.text(
            (meas.fall_edge_start_us + meas.fall_edge_end_us) / 2 + 5,
            AMPL_HIGH * 0.50,
            f"tf={meas.fall_time_us:.1f} µs",
            fontsize=8,
            color="C4",
        )
    ax_fall.axvline(
        meas.settling_fall_us,
        color="C6",
        lw=1.2,
        ls="-.",
        label=f"ts(2%)={meas.settling_fall_us:.1f} µs",
    )
    ax_fall.set(xlabel="Time from edge (µs)", ylabel="Amplitude (V)")
    ax_fall.set_title("Falling edge detail", fontsize=9)
    ax_fall.legend(fontsize=7, loc="lower right")
    ax_fall.grid(True, alpha=0.25)

    # ── Comparison table ──────────────────────────────────────────────────────
    ax_tbl.axis("off")
    tbl = ax_tbl.table(
        cellText=[row.to_table_row() for row in rows],
        colLabels=["Metric", "Predicted", "Measured", "Error"],
        loc="center",
        cellLoc="right",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.0, 1.4)
    for col in range(4):
        tbl[0, col].set_facecolor("#2C5F8A")
        tbl[0, col].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(rows) + 1):
        tbl[row_idx, 0].set_text_props(ha="left")
        if row_idx % 2 == 0:
            for col in range(4):
                tbl[row_idx, col].set_facecolor("#F0F4F8")

    fig.suptitle(
        "Underdamped 2nd-Order Pulse Metric Verification\n"
        f"ζ = {ZETA}   fn = {FN_HZ:.0f} Hz   Amplitude = {AMPL_HIGH} V   Drive = {F_SQ:.0f} Hz",
        fontsize=11,
        fontweight="bold",
    )
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    params = SystemParams.from_config(ZETA, FN_HZ, F_SQ, AMPL_LOW, AMPL_HIGH)
    cfg = MeasurementConfig()
    levels = (AMPL_LOW, AMPL_HIGH)

    # Simulate
    t = np.arange(0.0, N_CYCLES / F_SQ, 1.0 / FS)
    u_norm = 0.5 * (1.0 + square(2.0 * np.pi * F_SQ * t + np.pi))
    system = lti(
        [params.omega_n**2], [1.0, 2.0 * ZETA * params.omega_n, params.omega_n**2]
    )
    _, y_norm, _ = lsim(system, U=u_norm, T=t)
    y_out = AMPL_LOW + params.step_ampl * y_norm

    # Extract edge segments (time-zeroed at each edge start)
    t_half_idx = round(params.t_half_s * FS)
    segs = WaveformSegments(
        t_rise=t[t_half_idx : 2 * t_half_idx] - t[t_half_idx],
        y_rise=y_out[t_half_idx : 2 * t_half_idx],
        t_fall=t[2 * t_half_idx : 3 * t_half_idx] - t[2 * t_half_idx],
        y_fall=y_out[2 * t_half_idx : 3 * t_half_idx],
    )

    pred = compute_predictions(params, cfg, ZETA, F_SQ)
    meas = compute_measurements(t, y_out, segs, levels, cfg, params)
    rows = build_rows(pred, meas)

    # Print
    print(f"\n{'Metric':<28}  {'Predicted':>12}  {'Measured':>12}  {'Error %':>8}")
    print("─" * 68)
    for row in rows:
        row.print_row()
    print(
        f"\n  ζ = {ZETA}   fn = {FN_HZ:.0f} Hz   fd = {params.fd_hz:.1f} Hz   σ = {params.sigma:.1f} rad/s"
    )
    print(
        f"  True undershoot: dips to {meas.trough_abs_v:.3f} V = {meas.true_undershoot_pct:.1f}% below LOW"
    )
    print(f"  Undershoot bounce (= OS²): {meas.undershoot_bounce_pct:.1f}%")
    print(
        f"  Frequency: {meas.frequency_hz:.4f} Hz   Duty cycle: {meas.duty_cycle_pct:.4f}%"
    )

    fig = plot(pred, meas, rows, params, cfg, t, y_out, u_norm, segs)
    out_path = Path(__file__).parent / "predicted_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
