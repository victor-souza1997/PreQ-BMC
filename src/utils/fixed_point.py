from __future__ import annotations

from typing import TypeVar

import numpy as np

ArrayOrScalar = TypeVar("ArrayOrScalar", bound=np.ndarray | np.generic | float | int)


def real_round(value: float) -> float:
    """Round half away from zero, matching the original Quadapter helper."""

    if value < 0:
        return float(np.ceil(value - 0.5))
    if value > 0:
        return float(np.floor(value + 0.5))
    return 0.0


def round_half_away_from_zero(values: np.ndarray | float) -> np.ndarray | float:
    """Vectorized half-away-from-zero rounding."""

    if isinstance(values, np.ndarray):
        return np.where(values < 0, np.ceil(values - 0.5), np.floor(values + 0.5))
    return real_round(float(values))


def int_get_min_max(num_bits: int, frac_bits: int) -> tuple[float, float]:
    """Return the representable floating-point range for a signed fixed-point format."""

    num_value_bits = num_bits - 1
    min_value = -(2**num_value_bits) / (2**frac_bits)
    max_value = ((2**num_value_bits) - 1) / (2**frac_bits)
    return (float(min_value), float(max_value))


def signed_int_bounds(total_bits: int) -> tuple[int, int]:
    """Return the integer min/max for a signed two's-complement width."""

    return (-(1 << (total_bits - 1)), (1 << (total_bits - 1)) - 1)


def clamp_to_signed_range(value: int, total_bits: int) -> int:
    """Saturate an integer into a signed fixed-point container."""

    lower, upper = signed_int_bounds(total_bits)
    return max(lower, min(upper, int(value)))


def quantize_int(float_value: np.ndarray | float, num_bits: int, frac_bits: int) -> np.ndarray | np.int64:
    """Quantize floats into signed fixed-point integers with saturation."""

    min_value, max_value = int_get_min_max(num_bits, frac_bits)
    clipped = np.clip(float_value, min_value, max_value)
    scaled = clipped * (2**frac_bits)
    rounded = round_half_away_from_zero(np.asarray(scaled) if isinstance(scaled, np.ndarray) else float(scaled))
    if isinstance(rounded, np.ndarray):
        return rounded.astype(np.int64)
    return np.int64(rounded)


def dequantize_int(int_value: np.ndarray | int, frac_bits: int) -> np.ndarray | float:
    """Convert signed fixed-point integers back to floating-point."""

    return np.asarray(int_value, dtype=np.float64) / float(2**frac_bits)


def round_divide_half_away_from_zero(numerator: int, denominator: int) -> int:
    """Integer division rounded to nearest, ties away from zero."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -(((-numerator) + denominator // 2) // denominator)
