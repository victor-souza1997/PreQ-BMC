from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixedPointProfile:
    """Explicit fixed-point execution semantics shared by Python and generated C."""

    rounding_mode: str = "half_away_from_zero"
    overflow_mode: str = "saturate"
    accumulator_type: str = "__int128"


DEFAULT_FIXED_POINT_PROFILE = FixedPointProfile()
