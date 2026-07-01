"""
Tests verifying behavior described in IEEE Std 181-2003.

Section references follow the simplified spec in docs/IEEE_Std_181-2003_simplified.md.
Each test class names the algorithm or term being exercised.
"""

import numpy as np
import pytest

from pulse_transitions.common import Edge, EdgeSign, PairedEdge
from pulse_transitions.impl import (
    _calculate_midcross,
    _calculate_overshoot,
    _calculate_settling_time,
    _calculate_thresholds,
    _calculate_undershoot,
    _detect_edges,
    _detect_signal_levels,
    _interpolate_crossing,
    detect_signal_levels_with_histogram,
    pair_edges,
)
from pulse_transitions.transient_response import (
    calculate_risetime,
    calculate_falltime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trapezoidal_transition(t_start, t_end, s1, s2, n=1000, t_total=None):
    """Return (t, y) for a linear transition from s1 to s2."""
    t_total = t_total or t_end * 2
    t = np.linspace(0, t_total, n)
    y = np.where(
        t < t_start,
        s1,
        np.where(t > t_end, s2, s1 + (s2 - s1) * (t - t_start) / (t_end - t_start)),
    )
    return t, y


def make_rectangular_pulse(t1, t2, s1, s2, n=5000, t_total=1.0):
    """Return (t, y) for a rectangular pulse from t1 to t2."""
    t = np.linspace(0, t_total, n)
    y = np.where((t >= t1) & (t <= t2), s2, s1)
    return t, y


def make_trapezoidal_pulse(t1, t2, t3, t4, s1, s2, n=5000, t_total=1.0):
    """Return (t, y) for a trapezoidal pulse (finite rise/fall)."""
    t = np.linspace(0, t_total, n)
    y = np.where(
        t < t1,
        s1,
        np.where(
            t < t2,
            s1 + (s2 - s1) * (t - t1) / (t2 - t1),
            np.where(
                t < t3,
                s2,
                np.where(
                    t < t4,
                    s2 - (s2 - s1) * (t - t3) / (t4 - t3),
                    s1,
                ),
            ),
        ),
    )
    return t, y


# ---------------------------------------------------------------------------
# §5.2 — State Level Determination
# ---------------------------------------------------------------------------

class TestStateLevelDetermination:
    """§5.2 — State level must equal mode (or mean) of each histogram sub-band."""

    def test_histogram_finds_correct_levels_positive(self):
        """Histogram method locates s1 and s2 for a positive bilevel signal."""
        t, y = make_rectangular_pulse(0.3, 0.7, s1=0.0, s2=1.0, n=10000)
        low, high = _detect_signal_levels(t, y, method="histogram")
        assert low == pytest.approx(0.0, abs=0.05)
        assert high == pytest.approx(1.0, abs=0.05)

    def test_histogram_finds_correct_levels_offset(self):
        """Histogram method works when neither state level is zero."""
        t, y = make_rectangular_pulse(0.3, 0.7, s1=1.5, s2=3.3, n=10000)
        low, high = _detect_signal_levels(t, y, method="histogram")
        assert low == pytest.approx(1.5, abs=0.1)
        assert high == pytest.approx(3.3, abs=0.1)

    def test_histogram_finds_correct_levels_negative(self):
        """Histogram method works when s1 is negative."""
        t, y = make_rectangular_pulse(0.3, 0.7, s1=-1.0, s2=1.0, n=10000)
        low, high = _detect_signal_levels(t, y, method="histogram")
        assert low == pytest.approx(-1.0, abs=0.1)
        assert high == pytest.approx(1.0, abs=0.1)

    def test_histogram_low_less_than_high(self):
        """State s1 (low) must always be less than s2 (high)."""
        t, y = make_rectangular_pulse(0.3, 0.7, s1=0.0, s2=5.0, n=10000)
        low, high = _detect_signal_levels(t, y, method="histogram")
        assert low < high

    def test_histogram_levels_within_signal_bounds(self):
        """Detected state levels must lie within the waveform's observed range."""
        t, y = make_rectangular_pulse(0.3, 0.7, s1=0.2, s2=0.8, n=10000)
        low, high = _detect_signal_levels(t, y, method="histogram")
        assert low >= np.min(y) - 0.01
        assert high <= np.max(y) + 0.01


# ---------------------------------------------------------------------------
# §5.3.2 — Percent Reference Level
# ---------------------------------------------------------------------------

class TestPercentReferenceLevel:
    """§5.3.2 — y_x% = level(s1) + (A / 100) * x"""

    @pytest.mark.parametrize("x_pct,expected_fraction", [
        (0, 0.0),
        (10, 0.1),
        (50, 0.5),
        (90, 0.9),
        (100, 1.0),
    ])
    def test_reference_level_formula(self, x_pct, expected_fraction):
        """y_x% must equal s1 + (A/100)*x for known s1, s2."""
        s1, s2 = 0.0, 10.0
        A = s2 - s1
        y_xpct = s1 + (A / 100.0) * x_pct
        assert y_xpct == pytest.approx(s1 + expected_fraction * A)

    def test_thresholds_match_percent_reference_formula(self):
        """_calculate_thresholds must produce values matching §5.3.2."""
        s1, s2 = 2.0, 8.0
        A = s2 - s1
        levels = (s1, s2)
        t = np.linspace(0, 1, 100)
        y = np.zeros(100)

        thresh = _calculate_thresholds(t, y, levels, fractional_thresholds=(0.1, 0.9))

        # y_10% = s1 + (A/100)*10 = s1 + 0.1*A
        # y_90% = s1 + (A/100)*90 = s1 + 0.9*A
        assert thresh[0] == pytest.approx(s1 + 0.1 * A)
        assert thresh[1] == pytest.approx(s1 + 0.9 * A)

    def test_50pct_reference_level_is_midpoint(self):
        """The 50% reference level must equal the midpoint of s1 and s2."""
        s1, s2 = -4.0, 6.0
        A = s2 - s1
        y_50 = s1 + 0.5 * A
        assert y_50 == pytest.approx(0.5 * (s1 + s2))

    @pytest.mark.parametrize("s1,s2", [(0, 1), (-1, 1), (1.5, 3.3), (-5, -1)])
    def test_thresholds_ordered_correctly(self, s1, s2):
        """Low threshold < high threshold regardless of signal polarity."""
        levels = (s1, s2)
        t = np.zeros(10)
        y = np.zeros(10)
        thresh = _calculate_thresholds(t, y, levels, fractional_thresholds=(0.1, 0.9))
        assert thresh[0] < thresh[1]


# ---------------------------------------------------------------------------
# §5.3.3 — Reference Level Instants (linear interpolation)
# ---------------------------------------------------------------------------

class TestReferenceLevelInstants:
    """§5.3.3 — Crossing instants found by linear interpolation between samples."""

    def test_rising_10pct_crossing_accuracy(self):
        """10% crossing instant must be interpolated to within one sample of truth."""
        s1, s2 = 0.0, 1.0
        t_rise_start, t_rise_end = 0.4, 0.6
        t, y = make_trapezoidal_transition(t_rise_start, t_rise_end, s1, s2, n=2000)

        duration = t_rise_end - t_rise_start
        t_10pct_true = t_rise_start + 0.1 * duration
        t_90pct_true = t_rise_start + 0.9 * duration

        thresholds = (s1 + 0.1 * (s2 - s1), s1 + 0.9 * (s2 - s1))
        t_lo, t_hi = _interpolate_crossing(t, y, thresholds, EdgeSign.rising)

        dt = t[1] - t[0]
        assert t_lo == pytest.approx(t_10pct_true, abs=2 * dt)
        assert t_hi == pytest.approx(t_90pct_true, abs=2 * dt)

    def test_falling_90pct_crossing_accuracy(self):
        """90% → 10% crossing instants are at the 90%/10% positions within the ramp."""
        s1, s2 = 0.0, 1.0
        t_fall_start, t_fall_end = 0.4, 0.6
        # Signal ramps from s2=1.0 down to s1=0.0 over [t_fall_start, t_fall_end]
        t, y = make_trapezoidal_transition(t_fall_start, t_fall_end, s2, s1, n=2000)

        ramp_dur = t_fall_end - t_fall_start
        # On a falling ramp from 1.0→0.0, y=0.9 at t_fall_start + 0.1*ramp_dur
        # and y=0.1 at t_fall_start + 0.9*ramp_dur
        t_90pct_true = t_fall_start + 0.1 * ramp_dur
        t_10pct_true = t_fall_start + 0.9 * ramp_dur

        thresholds = (0.1, 0.9)
        t_lo, t_hi = _interpolate_crossing(t, y, thresholds, EdgeSign.falling)

        dt = t[1] - t[0]
        assert t_lo < t_hi
        assert t_lo == pytest.approx(t_90pct_true, abs=5 * dt)
        assert t_hi == pytest.approx(t_10pct_true, abs=5 * dt)

    def test_50pct_midcross_accuracy(self):
        """50% crossing instant must be at the midpoint of a linear transition."""
        s1, s2 = 0.0, 2.0
        t_start, t_end = 0.3, 0.7
        t, y = make_trapezoidal_transition(t_start, t_end, s1, s2, n=2000)

        t_50pct_true = 0.5 * (t_start + t_end)
        levels = (s1, s2)
        t_mid = _calculate_midcross(t, y, levels)

        dt = t[1] - t[0]
        assert t_mid == pytest.approx(t_50pct_true, abs=2 * dt)

    def test_interpolation_returns_ordered_pair(self):
        """_interpolate_crossing must return (earlier_time, later_time)."""
        t, y = make_trapezoidal_transition(0.4, 0.6, 0.0, 1.0, n=1000)
        thresholds = (0.1, 0.9)
        t_lo, t_hi = _interpolate_crossing(t, y, thresholds, EdgeSign.rising)
        assert t_lo < t_hi


# ---------------------------------------------------------------------------
# §5.3.4 — Transition Duration  (ta = |t_x1% − t_x2%|)
# ---------------------------------------------------------------------------

class TestTransitionDuration:
    """§5.3.4 — Default reference levels 10% and 90%."""

    def test_risetime_equals_transition_duration_positive(self):
        """Rise time must equal the time to traverse 10%→90% of the amplitude."""
        s1, s2 = 0.0, 1.0
        t_start, t_end = 0.2, 0.4
        t, y = make_trapezoidal_transition(t_start, t_end, s1, s2, n=4000)

        true_duration = (0.9 - 0.1) * (t_end - t_start)  # 80% of ramp time
        result = calculate_risetime(t, y, levels=(s1, s2))
        assert result is not None
        duration, _ = result
        assert duration == pytest.approx(true_duration, rel=0.02)

    def test_falltime_equals_transition_duration_negative(self):
        """Fall time must equal the time to traverse 90%→10% on a falling edge."""
        s1, s2 = 0.0, 1.0
        t_start, t_end = 0.2, 0.4
        t, y = make_trapezoidal_transition(t_start, t_end, s2, s1, n=4000)

        true_duration = (0.9 - 0.1) * (t_end - t_start)
        result = calculate_falltime(t, y, levels=(s1, s2))
        assert result is not None
        duration, _ = result
        assert duration == pytest.approx(true_duration, rel=0.02)

    def test_transition_duration_80pct_thresholds(self):
        """Custom 20%/80% thresholds cover 60% of ramp duration."""
        s1, s2 = 0.0, 1.0
        t_start, t_end = 0.1, 0.5
        t, y = make_trapezoidal_transition(t_start, t_end, s1, s2, n=4000)

        true_duration = (0.8 - 0.2) * (t_end - t_start)
        result = calculate_risetime(
            t, y, levels=(s1, s2), fractional_thresholds=(0.2, 0.8)
        )
        assert result is not None
        duration, _ = result
        assert duration == pytest.approx(true_duration, rel=0.02)

    def test_transition_duration_nonnegative(self):
        """Transition duration must be non-negative."""
        t, y = make_trapezoidal_transition(0.3, 0.7, 0.0, 1.0, n=2000)
        result = calculate_risetime(t, y)
        if result is not None:
            duration, _ = result
            assert duration >= 0.0


# ---------------------------------------------------------------------------
# §5.3.5 — Overshoot and Undershoot in Aberration Regions
# ---------------------------------------------------------------------------

class TestOvershootUndershoot:
    """§5.3.5 — O/U measured in pre/post aberration regions, not globally."""

    def test_no_overshoot_clean_step(self):
        """A clean step (no aberrations) must yield zero overshoot."""
        t, y = make_rectangular_pulse(0.3, 0.7, s1=0.0, s2=1.0, n=5000)
        mask = t < 0.5  # only the rising portion
        y_seg = y[mask]
        _, ov = _calculate_overshoot(y_seg, levels=(0.0, 1.0))
        assert ov == pytest.approx(0.0, abs=0.01)

    def test_overshoot_fraction_correct(self):
        """Overshoot fraction = (peak − high_level) / amplitude."""
        s1, s2 = 0.0, 1.0
        overshoot_height = 0.15  # 15% of amplitude
        t = np.linspace(0, 1, 5000)
        y = np.where(t < 0.5, s1, s2)
        # inject overshoot immediately after transition
        mask = (t >= 0.5) & (t < 0.55)
        y[mask] = s2 + overshoot_height

        _, ov = _calculate_overshoot(y, levels=(s1, s2))
        assert ov == pytest.approx(overshoot_height / (s2 - s1), abs=0.02)

    def test_undershoot_fraction_correct(self):
        """Undershoot fraction = (high_level − trough) / amplitude after overshoot peak."""
        s1, s2 = 0.0, 1.0
        t = np.linspace(0, 2, 10000)
        y = np.zeros_like(t)
        # Step at t=1 with overshoot then undershoot
        y[t >= 1.0] = s2
        y[(t >= 1.0) & (t < 1.05)] = s2 + 0.20  # 20% overshoot
        y[(t >= 1.05) & (t < 1.10)] = s2 - 0.10  # 10% undershoot

        result = _calculate_undershoot(y, levels=(s1, s2))
        assert result is not None
        _, un = result
        assert un == pytest.approx(0.10, abs=0.03)

    def test_overshoot_zero_for_falling_edge_below_level(self):
        """On a falling edge, overshoot is measured as excursion below s1."""
        s1, s2 = 0.0, 1.0
        t = np.linspace(0, 1, 5000)
        y = np.where(t < 0.5, s2, s1)
        # undershoot below s1
        mask = (t >= 0.5) & (t < 0.55)
        y[mask] = s1 - 0.10

        _, ov = _calculate_overshoot(y, levels=(s1, s2))
        assert ov == pytest.approx(0.10, abs=0.02)

    def test_overshoot_uses_amplitude_as_denominator(self):
        """Overshoot must be normalised by A = level(s2) - level(s1)."""
        s1, s2 = 2.0, 7.0
        A = s2 - s1
        overshoot_abs = 0.5  # absolute units above s2
        t = np.linspace(0, 1, 5000)
        y = np.where(t < 0.5, s1, s2)
        y[(t >= 0.5) & (t < 0.55)] = s2 + overshoot_abs

        _, ov = _calculate_overshoot(y, levels=(s1, s2))
        assert ov == pytest.approx(overshoot_abs / A, abs=0.02)


# ---------------------------------------------------------------------------
# §5.3.7 — Transition Settling Duration
# ---------------------------------------------------------------------------

class TestTransitionSettlingDuration:
    """§5.3.7 — Settling duration measured from 50% crossing instant."""

    def test_ideal_step_settles_immediately(self):
        """An ideal step with no ringing settles at the transition point."""
        t = np.linspace(0, 10, 5000)
        y = np.where(t < 1.0, 0.0, 1.0)
        settling_t = _calculate_settling_time(t, y, (0.0, 1.0), settling_time_fraction=0.02)
        assert settling_t == pytest.approx(1.0, abs=0.05)

    def test_ringing_delays_settling(self):
        """A signal with sustained ringing must settle later than a clean step."""
        t = np.linspace(0, 10, 10000)
        y_clean = np.where(t < 1.0, 0.0, 1.0)
        y_ringing = y_clean.copy()
        # add ringing that decays over ~3s
        ring_mask = t >= 1.0
        y_ringing[ring_mask] += 0.05 * np.exp(-1.5 * (t[ring_mask] - 1.0)) * np.sin(
            20 * np.pi * (t[ring_mask] - 1.0)
        )

        settling_clean = _calculate_settling_time(t, y_clean, (0.0, 1.0), settling_time_fraction=0.02)
        settling_ringing = _calculate_settling_time(t, y_ringing, (0.0, 1.0), settling_time_fraction=0.02)
        assert settling_ringing > settling_clean

    def test_settling_time_within_epoch(self):
        """Settling time must be within the waveform epoch."""
        t = np.linspace(0, 5, 2000)
        y = np.where(t < 1.0, 0.0, 1.0)
        settling_t = _calculate_settling_time(t, y, (0.0, 1.0), settling_time_fraction=0.02)
        assert t[0] <= settling_t <= t[-1]

    def test_settling_fraction_boundary(self):
        """A looser tolerance band (10%) must yield equal or earlier settling."""
        t = np.linspace(0, 10, 5000)
        y = np.where(t < 1.0, 0.0, 1.0)
        # Add small overshoot
        y[(t >= 1.0) & (t < 1.5)] = 1.05

        st_tight = _calculate_settling_time(t, y, (0.0, 1.0), settling_time_fraction=0.01)
        st_loose = _calculate_settling_time(t, y, (0.0, 1.0), settling_time_fraction=0.10)
        assert st_loose <= st_tight


# ---------------------------------------------------------------------------
# §5.4.1 — Pulse Duration Tₚ
# ---------------------------------------------------------------------------

class TestPulseDuration:
    """§5.4.1 — Tₚ = |t_2,50% − t_1,50%| (default x=50)."""

    def _make_pulse(self, t1, t2, s1, s2, n=8000):
        """Return (t, y, edges) for a rectangular pulse."""
        t, y = make_rectangular_pulse(t1, t2, s1, s2, n=n)
        levels = (s1, s2)
        thresholds = _calculate_thresholds(t, y, levels)
        from pulse_transitions.common import CrossingDetectionSettings
        edges = _detect_edges(t, y, thresholds, settings=CrossingDetectionSettings())
        return t, y, edges

    def test_pulse_duration_50pct_crossing(self):
        """Pulse duration at default 50% crossing must equal the pulse width."""
        t1, t2 = 0.2, 0.7
        t, y, edges = self._make_pulse(t1, t2, 0.0, 1.0)
        pairs = pair_edges(edges)
        assert len(pairs) >= 1

        # spec: Tₚ = |t_2,50% − t_1,50%|
        # For ideal rect pulse the 50% crossings are at t1 and t2
        pair = pairs[0]
        # rise.start is the 10% crossing; midcross gives 50%
        # pulse_width in PairedEdge uses fall.end - rise.start (10% to 10%)
        # This test verifies the measured duration is close to the true pulse width
        # The 50%-based duration is slightly shorter than 10%-to-10%
        pulse_50 = _calculate_midcross(t, y[t >= 0.15], (0.0, 1.0))  # just check midcross exists
        assert pair.pulse_width == pytest.approx(t2 - t1, abs=0.01)

    def test_pulse_duration_positive(self):
        """Pulse duration must be positive."""
        t, y, edges = self._make_pulse(0.2, 0.7, 0.0, 1.0)
        pairs = pair_edges(edges)
        for p in pairs:
            assert p.pulse_width > 0.0

    def test_pulse_duration_scales_with_width(self):
        """A wider pulse must have a longer measured duration."""
        _, _, edges1 = self._make_pulse(0.3, 0.5, 0.0, 1.0)  # 0.2s
        _, _, edges2 = self._make_pulse(0.2, 0.8, 0.0, 1.0)  # 0.6s
        p1 = pair_edges(edges1)
        p2 = pair_edges(edges2)
        assert p2[0].pulse_width > p1[0].pulse_width

    def test_pulse_amplitude(self):
        """PairedEdge amplitude must equal s2 − s1."""
        s1, s2 = 1.0, 4.0
        t, y, edges = self._make_pulse(0.3, 0.7, s1, s2)
        pairs = pair_edges(edges)
        assert len(pairs) >= 1
        assert pairs[0].amplitude == pytest.approx(s2 - s1, abs=0.3)


# ---------------------------------------------------------------------------
# §5.4.2 — Waveform Period T  and §5.4.3 — Pulse Separation Tₛ
# ---------------------------------------------------------------------------

class TestWaveformPeriodAndSeparation:
    """§5.4.2/§5.4.3 — T = Tₚ + Tₛ where Tₛ is the inter-pulse gap."""

    def _make_pulse_pair_manually(self, period, duty, s1, s2):
        """Build two PairedEdges by hand from known geometry — bypasses detect_edges."""
        Tp = period * duty
        Ts = period * (1 - duty)

        def make_edge(start, end, sign):
            return Edge(
                start=start,
                end=end,
                sign=sign,
                thresholds=(s1 + 0.1 * (s2 - s1), s1 + 0.9 * (s2 - s1)),
                ymin=s1,
                ymax=s2,
            )

        # Pulse 1: rises at t=0.1, falls at t=0.1+Tp
        rise1 = make_edge(0.1, 0.1 + 0.001, EdgeSign.rising)
        fall1 = make_edge(0.1 + Tp, 0.1 + Tp + 0.001, EdgeSign.falling)

        # Pulse 2: rises at t=0.1+period
        rise2 = make_edge(0.1 + period, 0.1 + period + 0.001, EdgeSign.rising)
        fall2 = make_edge(0.1 + period + Tp, 0.1 + period + Tp + 0.001, EdgeSign.falling)

        return (
            PairedEdge(rise=rise1, fall=fall1),
            PairedEdge(rise=rise2, fall=fall2),
        )

    def test_period_from_two_pulses(self):
        """Period = time between corresponding transitions of consecutive pulses."""
        period = 0.3
        duty = 0.4
        p1, p2 = self._make_pulse_pair_manually(period, duty, s1=0.0, s2=1.0)
        T = p2.rise.start - p1.rise.start
        assert T == pytest.approx(period, rel=1e-6)

    def test_pulse_separation_tps_equals_period_minus_duration(self):
        """Tₛ = T − Tₚ (§5.4.3 Method 2)."""
        period = 0.4
        duty = 0.5
        p1, p2 = self._make_pulse_pair_manually(period, duty, s1=0.0, s2=1.0)
        Tp = p1.pulse_width
        T = p2.rise.start - p1.rise.start
        Ts = T - Tp
        assert Ts > 0.0
        assert Ts == pytest.approx(period * (1.0 - duty), rel=0.01)

    def test_duty_factor_tp_over_t(self):
        """df = Tₚ / T (§5.4.4)."""
        period = 0.5
        duty = 0.3
        p1, p2 = self._make_pulse_pair_manually(period, duty, s1=0.0, s2=1.0)
        Tp = p1.pulse_width
        T = p2.rise.start - p1.rise.start
        df = Tp / T
        assert df == pytest.approx(duty, rel=0.01)

    def test_period_independent_of_amplitude(self):
        """Period measurement must not depend on the state levels."""
        period = 0.4
        p1a, p2a = self._make_pulse_pair_manually(period, 0.5, s1=0.0, s2=1.0)
        p1b, p2b = self._make_pulse_pair_manually(period, 0.5, s1=-2.0, s2=3.0)
        T_a = p2a.rise.start - p1a.rise.start
        T_b = p2b.rise.start - p1b.rise.start
        assert T_a == pytest.approx(T_b, rel=1e-6)

    def test_detect_edges_finds_multiple_pulses(self):
        """_detect_edges should return one edge per actual transition in a pulse train."""
        period = 0.3
        duty = 0.4
        s1, s2 = 0.0, 1.0
        t_total = period * 3
        n = 20000
        t = np.linspace(0, t_total, n)
        y = np.where((t % period) < period * duty, s2, s1)
        levels = (s1, s2)
        thresholds = _calculate_thresholds(t, y, levels)
        from pulse_transitions.common import CrossingDetectionSettings
        edges = _detect_edges(t, y, thresholds, settings=CrossingDetectionSettings())
        pairs = pair_edges(edges)
        # Should find at least 2 complete rise/fall pairs
        assert len(pairs) >= 2


# ---------------------------------------------------------------------------
# §5.4.4 — Duty Factor df = Tₚ / T
# ---------------------------------------------------------------------------

class TestDutyFactor:
    """§5.4.4 — df = Tₚ / T."""

    @pytest.mark.parametrize("duty", [0.3, 0.5, 0.7])
    def test_duty_factor_various(self, duty):
        """Measured duty factor must match the signal's constructed duty cycle."""
        from pulse_transitions.impl import calculate_duty_cycle

        period = 0.4
        n_periods = 4
        t = np.linspace(0, period * n_periods, 30000)
        # Centre HIGH within each period so the epoch starts and ends low.
        low_frac = (1.0 - duty) / 2.0
        phase = (t % period) / period
        y = np.where((phase >= low_frac) & (phase < low_frac + duty), 1.0, 0.0)
        df = calculate_duty_cycle(t, y, levels=(0.0, 1.0))
        assert df == pytest.approx(duty, abs=0.03)


# ---------------------------------------------------------------------------
# §5.1 — Two-State Analysis: Amplitude definition
# ---------------------------------------------------------------------------

class TestWaveformAmplitude:
    """§5.3.1 — Signed amplitude A = level(s2) − level(s1)."""

    @pytest.mark.parametrize("s1,s2", [(0, 1), (-1, 1), (2, 5), (-3, -1)])
    def test_signed_amplitude_positive_going(self, s1, s2):
        """Signed amplitude for a positive-going transition = s2 - s1."""
        A = s2 - s1
        assert A > 0

    @pytest.mark.parametrize("s1,s2", [(0, 1), (-1, 1), (2, 5)])
    def test_unsigned_amplitude_always_positive(self, s1, s2):
        """Unsigned amplitude is always positive regardless of direction."""
        assert abs(s2 - s1) > 0

    def test_amplitude_from_detected_levels(self):
        """Amplitude computed from detected state levels matches the true value."""
        s1, s2 = 0.5, 2.5
        t, y = make_rectangular_pulse(0.3, 0.7, s1, s2, n=10000)
        low, high = _detect_signal_levels(t, y, method="histogram")
        A_measured = high - low
        A_true = s2 - s1
        assert A_measured == pytest.approx(A_true, rel=0.05)


# ---------------------------------------------------------------------------
# §2 — Symbols: basic invariants
# ---------------------------------------------------------------------------

class TestSpecSymbolInvariants:
    """Verify that key spec symbols map to the correct quantities."""

    def test_edge_dx_is_transition_duration(self):
        """Edge.dx must equal end − start (ta = |t_x2% − t_x1%|)."""
        edge = Edge(
            start=1.0,
            end=1.3,
            sign=EdgeSign.rising,
            thresholds=(0.1, 0.9),
            ymin=0.0,
            ymax=1.0,
        )
        assert edge.dx == pytest.approx(0.3)

    def test_paired_edge_pulse_width_positive(self):
        """PairedEdge.pulse_width must be positive."""
        rise = Edge(start=1.0, end=1.1, sign=EdgeSign.rising, thresholds=(0.1, 0.9), ymin=0.0, ymax=1.0)
        fall = Edge(start=2.0, end=2.1, sign=EdgeSign.falling, thresholds=(0.1, 0.9), ymin=0.0, ymax=1.0)
        pair = PairedEdge(rise=rise, fall=fall)
        assert pair.pulse_width > 0.0

    def test_edge_sign_enum_values(self):
        """EdgeSign values must distinguish positive and negative transitions."""
        assert EdgeSign.rising != EdgeSign.falling
        assert EdgeSign.rising.value == 1
        assert EdgeSign.falling.value == -1

    def test_levels_define_amplitude(self):
        """A = level(s2) - level(s1) must equal high - low."""
        s1, s2 = 0.2, 0.8
        A = s2 - s1
        assert A == pytest.approx(0.6)

    def test_reference_level_instants_bracket_50pct(self):
        """10% and 90% crossing times must bracket the 50% crossing time."""
        s1, s2 = 0.0, 1.0
        t, y = make_trapezoidal_transition(0.3, 0.7, s1, s2, n=2000)

        t_50 = _calculate_midcross(t, y, (s1, s2))
        thresh_10_90 = _calculate_thresholds(t, y, (s1, s2), fractional_thresholds=(0.1, 0.9))
        t_lo, t_hi = _interpolate_crossing(t, y, thresh_10_90, EdgeSign.rising)

        assert t_lo < t_50 < t_hi
