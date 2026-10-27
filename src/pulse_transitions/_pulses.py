"""Pulses from alternating transitions (§5.4)."""
from __future__ import annotations

import math

from ._types import Polarity, Pulse, Transition

__all__ = ["make_pulses"]


def make_pulses(transitions: tuple[Transition, ...], polarity: Polarity = Polarity.RISING) -> tuple[Pulse, ...]:
    """Build the pulses that start at each transition of the given polarity.

    A pulse starts at transition ``i`` of ``polarity`` when transition ``i + 1`` exists.
    Its width is measured between the mid instants of the two transitions. When
    transition ``i + 2`` exists, the period is measured from ``t[i]`` to ``t[i + 2]``,
    the separation from ``t[i + 1]`` to ``t[i + 2]``, and the duty cycle is width over
    period; otherwise all three are NaN. Transitions are expected in time order and
    strictly alternating in polarity (§C5).

    Args:
        transitions: Transitions in time order.
        polarity: Polarity of the pulse-starting transitions.

    Returns:
        The pulses, in time order; an empty tuple when there are none.
    """
    pulses: list[Pulse] = []
    count = len(transitions)
    for i in range(count - 1):
        first = transitions[i]
        if first.polarity != polarity:
            continue
        second = transitions[i + 1]
        width = second.mid_instant - first.mid_instant
        if i + 2 < count:
            third = transitions[i + 2]
            period = third.mid_instant - first.mid_instant
            separation = third.mid_instant - second.mid_instant
            duty_cycle = width / period
        else:
            period = math.nan
            separation = math.nan
            duty_cycle = math.nan
        pulses.append(Pulse(
            polarity=polarity,
            first=first,
            second=second,
            width=width,
            period=period,
            separation=separation,
            duty_cycle=duty_cycle,
        ))
    return tuple(pulses)
