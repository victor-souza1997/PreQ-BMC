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
    """Exact affine-box transformer result for one deployed layer."""

    output_low: np.ndarray
    output_high: np.ndarray
    accumulator_low: np.ndarray
    accumulator_high: np.ndarray
    accumulator_c_type: str
    arithmetic_safety: dict[str, Any]


class FixedPointArithmeticRangeError(ValueError):
    """The generated fixed-point kernel cannot represent an intermediate."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_I128_MIN = -(1 << 127)
_I128_MAX = (1 << 127) - 1


def _signed_64_safe(low: int, high: int) -> bool:
    return _I64_MIN <= int(low) and int(high) <= _I64_MAX


def _raise_range_error(
    reason: str,
    *,
    layer_output: int | None = None,
    input_index: int | None = None,
    value: int | None = None,
) -> None:
    details: dict[str, Any] = {
        "status": "UNSAFE",
        "accumulator_type": "__int128",
        "reason": reason,
    }
    if layer_output is not None:
        details["layer_output"] = int(layer_output)
    if input_index is not None:
        details["input_index"] = int(input_index)
    if value is not None:
        details["value"] = str(int(value))
    raise FixedPointArithmeticRangeError(
        f"Fixed-point arithmetic exceeds the generated C kernel range: {reason}",
        details,
    )


def _check_i128(value: int, reason: str, *, output: int, input_index: int | None = None) -> None:
    if not (_I128_MIN <= int(value) <= _I128_MAX):
        _raise_range_error(
            reason,
            layer_output=output,
            input_index=input_index,
            value=value,
        )


def _check_rounding_i128(value: int, denominator: int, *, output: int) -> None:
    # The shared C helper evaluates num + den/2 for non-negative values and
    # (-num) + den/2 for negative values. Both intermediates must fit.
    magnitude = int(value) if value >= 0 else -int(value)
    if magnitude > _I128_MAX - denominator // 2:
        _raise_range_error(
            "round-half-away intermediate exceeds signed __int128",
            layer_output=output,
            value=value,
        )


def exact_layer_interval(
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

    if not 2 <= int(total_bits) <= 64:
        _raise_range_error(
            "TOTAL_BITS must be in [2, 64] because generated values use int64_t",
            value=total_bits,
        )
    if not 0 <= int(input_fractional_bits) <= 62:
        _raise_range_error(
            "input fractional bits must be in [0, 62]",
            value=input_fractional_bits,
        )
    denominator = 1 << int(input_fractional_bits)
    accumulator_low: list[int] = []
    accumulator_high: list[int] = []
    output_low: list[int] = []
    output_high: list[int] = []
    all_prefixes_fit_i64 = True
    max_abs_product = 0
    max_abs_accumulator = 0
    max_abs_preclamp = 0

    for output_index, (row, bias) in enumerate(zip(weights, biases)):
        bias_value = int(bias)
        if not _signed_64_safe(bias_value, bias_value):
            _raise_range_error(
                "bias cannot be represented by int64_t",
                layer_output=output_index,
                value=bias_value,
            )
        acc_low = 0
        acc_high = 0
        for input_index, (weight, lo, hi) in enumerate(zip(row, low, high)):
            w = int(weight)
            lo_value = int(lo)
            hi_value = int(hi)
            if not _signed_64_safe(w, w):
                _raise_range_error(
                    "weight cannot be represented by int64_t",
                    layer_output=output_index,
                    input_index=input_index,
                    value=w,
                )
            if not _signed_64_safe(lo_value, hi_value):
                _raise_range_error(
                    "input endpoint cannot be represented by int64_t",
                    layer_output=output_index,
                    input_index=input_index,
                    value=max(abs(lo_value), abs(hi_value)),
                )
            if lo_value > hi_value:
                raise ValueError("Invalid input interval.")
            product_low = w * (lo_value if w >= 0 else hi_value)
            product_high = w * (hi_value if w >= 0 else lo_value)
            _check_i128(
                product_low,
                "MAC product exceeds signed __int128",
                output=output_index,
                input_index=input_index,
            )
            _check_i128(
                product_high,
                "MAC product exceeds signed __int128",
                output=output_index,
                input_index=input_index,
            )
            acc_low += product_low
            acc_high += product_high
            _check_i128(
                acc_low,
                "MAC prefix sum exceeds signed __int128",
                output=output_index,
                input_index=input_index,
            )
            _check_i128(
                acc_high,
                "MAC prefix sum exceeds signed __int128",
                output=output_index,
                input_index=input_index,
            )
            max_abs_product = max(max_abs_product, abs(product_low), abs(product_high))
            max_abs_accumulator = max(max_abs_accumulator, abs(acc_low), abs(acc_high))
            all_prefixes_fit_i64 = (
                all_prefixes_fit_i64
                and _signed_64_safe(product_low, product_high)
                and _signed_64_safe(acc_low, acc_high)
            )

        _check_rounding_i128(acc_low, denominator, output=output_index)
        _check_rounding_i128(acc_high, denominator, output=output_index)
        value_low = round_divide_half_away_from_zero(acc_low, denominator) + bias_value
        value_high = round_divide_half_away_from_zero(acc_high, denominator) + bias_value
        _check_i128(value_low, "rescaled accumulator plus bias exceeds signed __int128", output=output_index)
        _check_i128(value_high, "rescaled accumulator plus bias exceeds signed __int128", output=output_index)
        max_abs_preclamp = max(max_abs_preclamp, abs(value_low), abs(value_high))
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
        arithmetic_safety={
            "status": "VERIFIED",
            "accumulator_type": "__int128",
            "value_storage_type": "int64_t",
            "total_bits": int(total_bits),
            "input_fractional_bits": int(input_fractional_bits),
            "max_abs_mac_product": str(max_abs_product),
            "max_abs_accumulator": str(max_abs_accumulator),
            "max_abs_preclamp": str(max_abs_preclamp),
            "i128_limit": str(_I128_MAX),
        },
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
        result = exact_layer_interval(
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
