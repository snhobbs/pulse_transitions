"""The public functions (`detect_edges`, `get_edge_metrics`, ...) and the result records, rather than `impl`."""

import numpy as np
import pytest

import pulse_transitions as pt
from pulse_transitions.common import CrossingDetectionSettings, Edge, EdgeSign

SAMPLE_STEP = 0.01
RAMP_START = 4.0
RAMP_LENGTH = 2.0
TEN_NINETY = (0.1, 0.9)
TWENTY_EIGHTY = (0.2, 0.8)


def ramp_signal(levels=(0.0, 1.0), start=RAMP_START, length=RAMP_LENGTH, end=10.0):
    """One linear edge from levels[0] to levels[1], flat either side. 10-90 % takes 0.8 * length."""
    x = np.arange(0.0, end, SAMPLE_STEP)
    y = levels[0] + (levels[1] - levels[0]) * np.clip((x - start) / length, 0.0, 1.0)
    return x, y


def staircase(steps, first=2.0, spacing=4.0, length=RAMP_LENGTH, end=14.0):
    """Linear edges to each of `steps` (the levels after each), starting at 0."""
    x = np.arange(0.0, end, SAMPLE_STEP)
    y = np.zeros_like(x)
    level = 0.0
    for index, target in enumerate(steps):
        start = first + index * spacing
        y = y + (target - level) * np.clip((x - start) / length, 0.0, 1.0)
        level = target
    return x, y


class TestDetectEdgesArguments:
    def test_fractional_thresholds_keyword_changes_the_measurement(self):
        x, y = ramp_signal()
        wide = pt.detect_edges(x, y, levels=(0, 1))[0]
        narrow = pt.detect_edges(
            x, y, fractional_thresholds=TWENTY_EIGHTY, levels=(0, 1)
        )[0]
        assert wide.dx == pytest.approx(0.8 * RAMP_LENGTH, abs=SAMPLE_STEP)
        assert narrow.dx == pytest.approx(0.6 * RAMP_LENGTH, abs=SAMPLE_STEP)

    def test_thresholds_positionally_equal_the_keyword(self):
        x, y = ramp_signal()
        assert pt.detect_edges(x, y, TWENTY_EIGHTY, (0, 1)) == pt.detect_edges(
            x, y, fractional_thresholds=TWENTY_EIGHTY, levels=(0, 1)
        )

    def test_a_misspelt_keyword_with_levels_is_an_error_not_ignored(self):
        x, y = ramp_signal()
        with pytest.raises(TypeError, match="thresholds"):
            pt.detect_edges(x, y, levels=(0, 1), thresholds=TWENTY_EIGHTY)

    def test_a_misspelt_keyword_without_levels_is_an_error(self):
        x, y = ramp_signal()
        with pytest.raises(TypeError):
            pt.detect_edges(x, y, thresholds=TWENTY_EIGHTY)

    def test_edge_metrics_rejects_the_old_keyword_too(self):
        x, y = ramp_signal()
        with pytest.raises(TypeError, match="thresholds"):
            pt.get_edge_metrics(x=x, y=y, thresholds=TWENTY_EIGHTY, levels=(0, 1))

    def test_level_options_are_accepted_when_levels_are_detected(self):
        x, y = ramp_signal()
        assert pt.detect_edges(x, y, nbins=50)

    def test_bounds_restrict_which_edges_are_found(self):
        x = np.arange(0.0, 20.0, SAMPLE_STEP)
        y = np.clip((x - 4) / 2, 0, 1) - np.clip((x - 14) / 2, 0, 1)
        assert [edge.sign for edge in pt.detect_edges(x, y, levels=(0, 1))] == [
            EdgeSign.rising,
            EdgeSign.falling,
        ]
        first = pt.detect_edges(x, y, levels=(0, 1), bounds=(0, 10))
        assert [edge.sign for edge in first] == [EdgeSign.rising]
        assert pt.detect_edges(x, y, levels=(0, 1), bounds=(7, 11)) == []
        last = pt.detect_edges(x, y, levels=(0, 1), bounds=(10, 20))
        assert [edge.sign for edge in last] == [EdgeSign.falling]

    def test_equal_levels_are_a_value_error(self):
        x, y = ramp_signal()
        with pytest.raises(ValueError, match="do not span"):
            pt.calculate_risetime(x, y, levels=(1, 1))

    def test_reversed_levels_are_a_value_error(self):
        with pytest.raises(ValueError, match="low, high"):
            pt.calculate_thresholds([0, 1], [0, 1], levels=(1, 0))


class TestStaircase:
    """A TDR trace steps through several levels; the levels given pick the step."""

    def test_the_step_between_the_given_levels_is_the_one_found(self):
        x, y = staircase([1.0, 3.0])
        edges = pt.detect_edges(x, y, levels=(1.0, 3.0))
        assert len(edges) == 1
        assert edges[0].sign == EdgeSign.rising
        second_start = 2.0 + 4.0
        assert edges[0].start == pytest.approx(
            second_start + 0.1 * RAMP_LENGTH, abs=SAMPLE_STEP
        )
        assert edges[0].dx == pytest.approx(0.8 * RAMP_LENGTH, abs=SAMPLE_STEP)

    def test_rising_then_falling_step_between_the_given_levels(self):
        x, y = staircase([1.0, 3.0, 1.0], end=20.0)
        edges = pt.detect_edges(x, y, levels=(1.0, 3.0))
        assert [edge.sign for edge in edges] == [EdgeSign.rising, EdgeSign.falling]
        assert all(
            edge.dx == pytest.approx(0.8 * RAMP_LENGTH, abs=SAMPLE_STEP)
            for edge in edges
        )


class TestAccuracy:
    def test_rc_step_rise_time_is_tau_ln_nine(self):
        tau = 1.0
        for step_per_tau in (50, 200, 1000):
            step = tau / step_per_tau
            x = np.arange(0.0, 12 * tau, step)
            y = np.where(x < 2 * tau, 0.0, 1.0 - np.exp(-(x - 2 * tau) / tau))
            rise, edge = pt.calculate_risetime(x, y, levels=(0, 1))
            assert rise == pytest.approx(
                tau * np.log(9), rel=1e-2 if step_per_tau == 50 else 2e-3
            )
            assert edge.end - edge.start == rise

    @pytest.mark.parametrize("offset_samples", [0.0, 0.25, 0.5, 0.77])
    def test_the_position_of_the_edge_between_samples_does_not_matter(
        self, offset_samples
    ):
        x, y = ramp_signal(start=RAMP_START + offset_samples * SAMPLE_STEP)
        edge = pt.detect_edges(x, y, levels=(0, 1))[0]
        assert edge.dx == pytest.approx(0.8 * RAMP_LENGTH, abs=1e-9)


class TestInvariance:
    def test_scale_and_offset_leave_the_duration_unchanged(self):
        x, y = ramp_signal()
        reference = pt.detect_edges(x, y, levels=(0, 1))[0]
        scaled = pt.detect_edges(x, 5.0 * y - 3.0, levels=(-3.0, 2.0))[0]
        assert scaled.start == pytest.approx(reference.start)
        assert scaled.dx == pytest.approx(reference.dx)

    def test_a_falling_edge_lasts_as_long_as_the_mirrored_rising_edge(self):
        x, y = ramp_signal()
        rising = pt.detect_edges(x, y, levels=(0, 1))[0]
        falling = pt.detect_edges(x, 1.0 - y, levels=(0, 1))[0]
        assert falling.sign == EdgeSign.falling
        assert falling.dx == pytest.approx(rising.dx)

    def test_shifting_the_time_axis_shifts_the_edge(self):
        x, y = ramp_signal()
        shift = 123.4
        reference = pt.detect_edges(x, y, levels=(0, 1))[0]
        shifted = pt.detect_edges(x + shift, y, levels=(0, 1))[0]
        assert shifted.start == pytest.approx(reference.start + shift)
        assert shifted.dx == pytest.approx(reference.dx)

    def test_unevenly_spaced_samples(self):
        rng = np.random.default_rng(1)
        x = np.sort(rng.uniform(0.0, 10.0, 2000))
        y = np.clip((x - RAMP_START) / RAMP_LENGTH, 0.0, 1.0)
        edge = pt.detect_edges(x, y, levels=(0, 1))[0]
        assert edge.dx == pytest.approx(0.8 * RAMP_LENGTH, abs=0.02)


class TestRecords:
    def make_edge(self, **overrides):
        fields = dict(
            start=1.0,
            end=3.0,
            ymin=0.1,
            ymax=0.9,
            thresholds=[0.1, 0.9],
            sign=EdgeSign.rising,
        )
        fields.update(overrides)
        return Edge(**fields)

    def test_edge_duration_and_bounds(self):
        edge = self.make_edge()
        assert edge.dx == 2.0
        assert edge.bound_low == (1.0, 0.1)
        assert edge.bound_high == (3.0, 0.9)

    def test_thresholds_are_tuples_whatever_sequence_was_given(self):
        edge = self.make_edge(thresholds=[0.1, 0.9], fractional_thresholds=[0.2, 0.8])
        assert edge.thresholds == (0.1, 0.9)
        assert edge.fractional_thresholds == (0.2, 0.8)

    @pytest.mark.parametrize("start,end", [(3.0, 1.0), (2.0, 2.0)])
    def test_an_edge_must_start_before_it_ends(self, start, end):
        with pytest.raises(ValueError, match="start must be before end"):
            self.make_edge(start=start, end=end)

    def test_settings_defaults_and_unknown_fields(self):
        settings = CrossingDetectionSettings()
        assert (settings.window, settings.filter_order, settings.filter_cutoff) == (
            0,
            3,
            0.9,
        )
        with pytest.raises(TypeError):
            CrossingDetectionSettings(bogus=1)

    def test_detected_edges_are_edge_records(self):
        x, y = ramp_signal()
        edge = pt.detect_edges(x, y, levels=(0, 1))[0]
        assert isinstance(edge, Edge)
        assert edge.fractional_thresholds == TEN_NINETY
        assert edge.thresholds == pytest.approx((0.1, 0.9))
