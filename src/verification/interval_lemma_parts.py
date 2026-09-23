"""Small ESBMC obligations composing the fixed-point interval theorem."""
from __future__ import annotations

import numpy as np

from backends.conv_fixed_point import QuantizedConv2D, QuantizedOperator
from verification.arith_kernel import render_arith_kernel


def _limits(layer: QuantizedOperator, input_total_bits: int):
    if not 2 <= input_total_bits <= 16 or not 2 <= layer.spec.total_bits <= 16:
        raise ValueError("Interval lemmas support signed formats up to 16 bits")
    input_low = -(1 << (input_total_bits - 1))
    input_high = (1 << (input_total_bits - 1)) - 1
    weight_low, weight_high = layer.spec.signed_range
    terms = (int(np.prod(layer.geometry.kernel_shape[:3]))
             if isinstance(layer, QuantizedConv2D) else layer.input_size)
    accumulator = terms * max(abs(input_low), abs(input_high)) \
        * max(abs(weight_low), abs(weight_high))
    if accumulator >= 1 << 62:
        raise ValueError("Accumulator envelope does not fit int64 lemma witnesses")
    return input_low, input_high, weight_low, weight_high, accumulator


def render_rounding_monotonicity_lemma(layer: QuantizedOperator, input_total_bits: int):
    _, _, _, _, accumulator = _limits(layer, input_total_bits)
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
int nondet_int(void);
{render_arith_kernel()}
int main(void) {{
    int64_t low = nondet_long_long();
    int64_t value = nondet_long_long();
    int64_t high = nondet_long_long();
    __ESBMC_assume(low >= -{accumulator} && high <= {accumulator});
    __ESBMC_assume(low <= value && value <= high);
    __int128 rounded_low = div_round_half_away_from_zero_i128(low, ((__int128)1 << {layer.input_fractional_bits}));
    __int128 rounded = div_round_half_away_from_zero_i128(value, ((__int128)1 << {layer.input_fractional_bits}));
    __int128 rounded_high = div_round_half_away_from_zero_i128(high, ((__int128)1 << {layer.input_fractional_bits}));
    __ESBMC_assert(rounded_low <= rounded && rounded <= rounded_high,
                   "round-half-away-from-zero is monotone");
    return 0;
}}
"""


def render_clamp_relu_monotonicity_lemma(layer: QuantizedOperator):
    low, high = layer.spec.signed_range
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
int nondet_int(void);
{render_arith_kernel()}
int main(void) {{
    int64_t low = nondet_long_long();
    int64_t value = nondet_long_long();
    int64_t high = nondet_long_long();
    int64_t bias = nondet_long_long();
    __ESBMC_assume(low <= value && value <= high);
    __ESBMC_assume(bias >= {low} && bias <= {high});
    __int128 result_low = clamp_to_signed_range_i128((__int128)low + bias, {layer.spec.total_bits});
    __int128 result = clamp_to_signed_range_i128((__int128)value + bias, {layer.spec.total_bits});
    __int128 result_high = clamp_to_signed_range_i128((__int128)high + bias, {layer.spec.total_bits});
    if ({1 if layer.apply_relu else 0}) {{
        result_low = result_low < 0 ? 0 : result_low;
        result = result < 0 ? 0 : result;
        result_high = result_high < 0 ? 0 : result_high;
    }}
    __ESBMC_assert(result_low <= result && result <= result_high,
                   "bias, clamp, and ReLU are monotone");
    return 0;
}}
"""
