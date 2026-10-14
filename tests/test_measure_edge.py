"""`measure_edge`: one edge of a signal with several, near a given place, between its own levels."""

import numpy as np
import pytest

import pulse_transitions as pt
from pulse_transitions import EdgeNotFoundError, EdgeSign

SAMPLE_STEP = 0.01
RAMP_LENGTH = 2.0
TEN_NINETY_FRACTION = 0.8
FLAT_LENGTH = 4.0
NEAR = 0.5  # a pointer this far from an edge counts as on it
FAR = 100.0
NOISE_SEED = 7
NOISE_SIGMA = 0.002


def staircase(steps, first=FLAT_LENGTH / 2, spacing=FLAT_LENGTH, length=RAMP_LENGTH):
    """Linear edges to each of `steps`, from 0, each `spacing` after the last; returns x, y and the edges' centres."""
    x = np.arange(0.0, first + spacing * (len(steps) + 1), SAMPLE_STEP)
    y = np.zeros_like(x)
    level, centres = 0.0, []
    for index, target in enumerate(steps):
        start = first + index * spacing
        y = y + (target - level) * np.clip((x - start) / length, 0.0, 1.0)
        centres.append(start + length / 2)
        level = target
    return x, y, centres


def test_exports():
    for name in (
        "measure_edge",
        "EdgeMeasurement",
        "EdgeNotFoundError",
        "get_rising_edge",
        "get_falling_edge",
        "pair_edges",
    ):
        assert name in pt.__all__


def test_edge_not_found_is_an_index_error_and_a_value_error():
    assert issubclass(EdgeNotFoundError, IndexError) and issubclass(
        EdgeNotFoundError, ValueError
    )


def test_the_edge_nearest_the_pointer_is_measured_between_its_own_levels():
    x, y, centres = staircase([1.0, 3.0, 2.0])
    for centre, levels in zip(centres, [(0.0, 1.0), (1.0, 3.0), (2.0, 3.0)]):
        measured = pt.measure_edge(x, y, around=centre, max_distance=NEAR)
        assert measured.levels == pytest.approx(levels, abs=1e-6)
        assert measured.duration == pytest.approx(
            TEN_NINETY_FRACTION * RAMP_LENGTH, abs=SAMPLE_STEP
        )
        assert 0.5 * (measured.edge.start + measured.edge.end) == pytest.approx(
            centre, abs=SAMPLE_STEP
        )


def test_a_falling_edge_is_marked_falling():
    x, y, centres = staircase([3.0, 1.0])
    measured = pt.measure_edge(x, y, around=centres[1], max_distance=NEAR)
    assert measured.edge.sign == EdgeSign.falling
    assert measured.levels == pytest.approx((1.0, 3.0), abs=1e-6)
    assert measured.duration == pytest.approx(
        TEN_NINETY_FRACTION * RAMP_LENGTH, abs=SAMPLE_STEP
    )


def test_other_reference_levels():
    x, y, centres = staircase([1.0])
    measured = pt.measure_edge(x, y, centres[0], NEAR, fractional_thresholds=(0.2, 0.8))
    assert measured.duration == pytest.approx(0.6 * RAMP_LENGTH, abs=SAMPLE_STEP)


def test_given_levels_replace_the_flat_parts():
    x, y, centres = staircase([1.0])
    measured = pt.measure_edge(
        x, y, centres[0], NEAR, levels=(0.0, 0.5)
    )  # a lower top than the signal reaches
    assert measured.levels == (0.0, 0.5)
    assert measured.duration == pytest.approx(0.4 * RAMP_LENGTH, abs=SAMPLE_STEP)


def test_the_pointer_between_two_edges_takes_the_nearer():
    x, y, centres = staircase([1.0, 3.0])
    nearer_second = 0.5 * (centres[0] + centres[1]) + 1.0
    assert pt.measure_edge(x, y, around=nearer_second).levels == pytest.approx(
        (1.0, 3.0), abs=1e-6
    )


def test_a_pointer_far_from_any_edge_is_not_found():
    x, y, centres = staircase([1.0])
    with pytest.raises(EdgeNotFoundError, match="No edge near the pointer"):
        pt.measure_edge(x, y, around=centres[0] + FAR / 10, max_distance=NEAR)


def test_a_flat_trace_has_no_edge():
    x = np.arange(0.0, 10.0, SAMPLE_STEP)
    with pytest.raises(EdgeNotFoundError, match="no edge"):
        pt.measure_edge(x, np.ones_like(x), around=5.0)


def test_too_short_a_trace():
    with pytest.raises(EdgeNotFoundError, match="no trace"):
        pt.measure_edge([0, 1, 2], [0, 1, 2], around=1.0)


def test_an_edge_at_the_start_has_no_level_before_it():
    x = np.arange(0.0, 10.0, SAMPLE_STEP)
    y = np.clip(x / RAMP_LENGTH, 0.0, 1.0)
    with pytest.raises(EdgeNotFoundError, match="flat level"):
        pt.measure_edge(x, y, around=RAMP_LENGTH / 2)


def test_levels_the_edge_does_not_reach():
    x, y, centres = staircase([1.0])
    with pytest.raises(EdgeNotFoundError):
        pt.measure_edge(x, y, centres[0], NEAR, levels=(5.0, 6.0))


def test_not_a_number_samples_are_skipped():
    x, y, centres = staircase([1.0])
    y[10] = np.nan
    assert pt.measure_edge(x, y, centres[0], NEAR).duration == pytest.approx(
        TEN_NINETY_FRACTION * RAMP_LENGTH, abs=SAMPLE_STEP
    )


def test_noise_does_not_hide_the_edge():
    x, y, centres = staircase([1.0, 3.0])
    y = y + np.random.default_rng(NOISE_SEED).normal(0.0, NOISE_SIGMA, len(y))
    measured = pt.measure_edge(x, y, around=centres[1], max_distance=NEAR)
    assert measured.levels == pytest.approx((1.0, 3.0), abs=10 * NOISE_SIGMA)
    assert measured.duration == pytest.approx(
        TEN_NINETY_FRACTION * RAMP_LENGTH, abs=0.05
    )
