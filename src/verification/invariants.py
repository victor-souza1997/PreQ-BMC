from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from utils.fixed_point import (
    clamp_to_signed_range,
    round_divide_half_away_from_zero,
)


@dataclass(frozen=True)
class ExactLayerIntervals:
    """Exact box transformer result for one deployed affine layer."""

    output_low: np.ndarray
    output_high: np.ndarray
    accumulator_low: np.ndarray
    accumulator_high: np.ndarray
    accumulator_c_type: str


def _signed_64_safe(low: int, high: int) -> bool:
    return -(1 << 63) <= int(low) and int(high) <= (1 << 63) - 1


def _layer_interval(
    layer: Any,
    *,
    input_low: np.ndarray,
    input_high: np.ndarray,
    input_fractional_bits: int,
    total_bits: int,
    apply_relu: bool,
) -> ExactLayerIntervals:
    weights = np.asarray(layer.weights_int, dtype=object)
    biases = np.asarray(layer.biases_int, dtype=object).reshape(-1)
    low = np.asarray(input_low, dtype=object).reshape(-1)
    high = np.asarray(input_high, dtype=object).reshape(-1)
    if weights.ndim != 2 or weights.shape[1] != low.size or high.size != low.size:
        raise ValueError("Layer interval dimensions do not match.")
    if weights.shape[0] != biases.size:
        raise ValueError("Layer bias dimension does not match.")

    denominator = 1 << int(input_fractional_bits)
    accumulator_low: list[int] = []
    accumulator_high: list[int] = []
    output_low: list[int] = []
    output_high: list[int] = []
    all_prefixes_fit_i64 = True

    for row, bias in zip(weights, biases):
        acc_low = 0
        acc_high = 0
        for weight, lo, hi in zip(row, low, high):
            w = int(weight)
            lo_value = int(lo)
            hi_value = int(hi)
            if lo_value > hi_value:
                raise ValueError("Invalid input interval.")
            product_low = w * (lo_value if w >= 0 else hi_value)
            product_high = w * (hi_value if w >= 0 else lo_value)
            acc_low += product_low
            acc_high += product_high
            all_prefixes_fit_i64 = (
                all_prefixes_fit_i64
                and _signed_64_safe(product_low, product_high)
                and _signed_64_safe(acc_low, acc_high)
            )

        value_low = round_divide_half_away_from_zero(
            acc_low, denominator
        ) + int(bias)
        value_high = round_divide_half_away_from_zero(
            acc_high, denominator
        ) + int(bias)
        value_low = clamp_to_signed_range(value_low, total_bits)
        value_high = clamp_to_signed_range(value_high, total_bits)
        if apply_relu:
            value_low = max(value_low, 0)
            value_high = max(value_high, 0)
        value_low = clamp_to_signed_range(value_low, total_bits)
        value_high = clamp_to_signed_range(value_high, total_bits)

        accumulator_low.append(acc_low)
        accumulator_high.append(acc_high)
        output_low.append(value_low)
        output_high.append(value_high)

    return ExactLayerIntervals(
        output_low=np.asarray(output_low, dtype=np.int64),
        output_high=np.asarray(output_high, dtype=np.int64),
        accumulator_low=np.asarray(accumulator_low, dtype=object),
        accumulator_high=np.asarray(accumulator_high, dtype=object),
        accumulator_c_type="int64_t" if all_prefixes_fit_i64 else "__int128",
    )


def propagate_exact_interval_details(
    layers: Sequence[Any],
    fmts: Sequence[Any],
    x_lo_int: np.ndarray,
    x_hi_int: np.ndarray,
) -> list[ExactLayerIntervals]:
    """Propagate exact deployed-kernel interval endpoints through a network."""

    if len(layers) != len(fmts):
        raise ValueError("Expected one fixed-point format per layer.")
    if not layers:
        return []

    current_low = np.asarray(x_lo_int, dtype=np.int64).reshape(-1)
    current_high = np.asarray(x_hi_int, dtype=np.int64).reshape(-1)
    input_fractional_bits = int(fmts[0].fractional_bits)
    results: list[ExactLayerIntervals] = []

    for index, (layer, fmt) in enumerate(zip(layers, fmts)):
        result = _layer_interval(
            layer,
            input_low=current_low,
            input_high=current_high,
            input_fractional_bits=input_fractional_bits,
            total_bits=int(fmt.total_bits),
            apply_relu=index < len(layers) - 1,
        )
        results.append(result)
        current_low = result.output_low
        current_high = result.output_high
        input_fractional_bits = int(fmt.fractional_bits)

    return results


def propagate_exact_intervals(
    layers: Sequence[Any],
    fmts: Sequence[Any],
    x_lo_int: np.ndarray,
    x_hi_int: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return exact per-layer integer output boxes for the deployed kernel."""

    return [
        (result.output_low, result.output_high)
        for result in propagate_exact_interval_details(
            layers,
            fmts,
            x_lo_int,
            x_hi_int,
        )
    ]
