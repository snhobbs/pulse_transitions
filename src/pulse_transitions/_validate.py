"""Input coercion and checks at the public boundary."""

from __future__ import annotations

import math
import numbers
from collections.abc import Collection

import numpy as np

from ._types import InputError

__all__ = ["as_waveform", "check_choice", "check_levels", "check_reference"]

_NUMERIC_KINDS = "biuf"


def _as_numeric_1d(value: object, name: str) -> np.ndarray:
    """Return value as a 1-D float64 array, raising InputError naming `name` if it is not numeric 1-D."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise InputError(
            f"{name} must be a 1-D numeric array, got {value!r}"
        ) from error
    if array.ndim != 1 or array.dtype.kind not in _NUMERIC_KINDS:
        raise InputError(
            f"{name} must be a 1-D numeric array, got shape {array.shape} and dtype {array.dtype}"
        )
    return array.astype(np.float64)


def _finite_real(value: object) -> float | None:
    """Return value as a float if it is a finite real number (not bool), otherwise None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _real_sequence(value: object, count: int, name: str) -> tuple[float, ...]:
    """Return value as a tuple of `count` finite floats, raising InputError naming `name` otherwise."""
    if isinstance(value, (str, bytes)):
        raise InputError(f"{name} must be {count} finite numbers, got {value!r}")
    try:
        items = tuple(value)  # type: ignore[call-overload]
    except TypeError as error:
        raise InputError(
            f"{name} must be {count} finite numbers, got {value!r}"
        ) from error
    parsed = tuple(_finite_real(item) for item in items)
    numbers_only = tuple(number for number in parsed if number is not None)
    if len(parsed) != count or len(numbers_only) != count:
        raise InputError(f"{name} must be {count} finite numbers, got {value!r}")
    return numbers_only


def _check_bounds(bounds: object) -> tuple[float, float]:
    """Return bounds as a (start, end) pair of finite floats with start < end, else raise InputError."""
    start, end = _real_sequence(bounds, 2, "bounds")
    if not start < end:
        raise InputError(f"bounds must satisfy start < end, got {bounds!r}")
    return start, end


def as_waveform(
    y: object,
    x: object = None,
    fs: float | None = None,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) as 1-D float64 arrays, cleaned and bounded.

    Samples where y or x is NaN are dropped. When x is None the time axis is arange(n) / fs
    (fs defaulting to 1). Bounds are applied inclusively. Raises InputError naming the
    offending argument for invalid input, including fewer than two samples remaining.
    """
    y_array = _as_numeric_1d(y, "y")
    if x is not None and fs is not None:
        raise InputError("give either x or fs, not both")
    if fs is not None:
        fs_value = _finite_real(fs)
        if fs_value is None or fs_value <= 0:
            raise InputError(f"fs must be finite and > 0, got {fs!r}")
        x_array = np.arange(y_array.size, dtype=np.float64) / fs_value
    elif x is not None:
        x_array = _as_numeric_1d(x, "x")
        if x_array.size != y_array.size:
            raise InputError(
                f"x must have the same length as y, got {x_array.size} and {y_array.size}"
            )
    else:
        x_array = np.arange(y_array.size, dtype=np.float64)

    given_x = x is not None
    missing = np.isnan(y_array)
    if given_x:
        missing |= np.isnan(x_array)
    if missing.any():
        keep = ~missing
        x_array = x_array[keep]
        y_array = y_array[keep]

    # a generated time axis is increasing by construction
    if given_x and x_array.size >= 2 and not np.all(np.diff(x_array) > 0):
        raise InputError("x must be strictly increasing after NaN removal")

    if bounds is not None:
        start, end = _check_bounds(bounds)
        window = (x_array >= start) & (x_array <= end)
        x_array = x_array[window]
        y_array = y_array[window]

    if x_array.size < 2:
        raise InputError(
            f"y must have at least 2 samples after NaN removal and bounds, got {x_array.size}"
        )
    return x_array, y_array


def check_levels(levels: object) -> tuple[float, float]:
    """Return levels as a (low, high) pair of finite floats with low < high, else raise InputError."""
    low, high = _real_sequence(levels, 2, "levels")
    if not low < high:
        raise InputError(f"levels must satisfy low < high, got {levels!r}")
    return low, high


def check_reference(reference: object, *, pair: bool = False) -> tuple[float, ...]:
    """Return reference as finite percentages, else raise InputError.

    Three values (lower, mid, upper) must satisfy 0 <= lower < mid < upper <= 100. With pair=True
    two values (lower, upper) must satisfy 0 <= lower < upper <= 100.
    """
    if pair:
        values = _real_sequence(reference, 2, "reference")
        lower, upper = values
        if not 0 <= lower < upper <= 100:
            raise InputError(
                f"reference must satisfy 0 <= lower < upper <= 100 for pairs, got {reference!r}"
            )
        return values
    values = _real_sequence(reference, 3, "reference")
    lower, mid, upper = values
    if not 0 <= lower < mid < upper <= 100:
        raise InputError(
            f"reference must satisfy 0 <= lower < mid < upper <= 100, got {reference!r}"
        )
    return values


def check_choice(name: str, value: object, allowed: Collection[object]) -> object:
    """Return value if it is one of allowed, else raise InputError naming `name`."""
    if value not in allowed:
        raise InputError(f"{name} must be one of {tuple(allowed)!r}, got {value!r}")
    return value
