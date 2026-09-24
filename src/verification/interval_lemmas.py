"""ESBMC lemmas and concrete certificates for integer interval propagation."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from backends.conv_fixed_point import QuantizedConv2D, QuantizedDense
from verification.arith_kernel import render_arith_kernel
from verification.conv_contracts import (
    LayerIntervalCertificate,
    conv_output_terms,
    dense_output_terms,
)


def _array(values: Iterable[int]) -> str:
    return "{" + ", ".join(str(int(value)) for value in values) + "}"


def _relu_states(certificate: LayerIntervalCertificate, outputs: Iterable[int]):
    active = set(certificate.stable_active)
    inactive = set(certificate.stable_inactive)
    return tuple(1 if index in active else -1 if index in inactive else 0
                 for index in outputs)


def render_conv_interval_certificate(
    layer: QuantizedConv2D,
    certificate: LayerIntervalCertificate,
    output_indices: Iterable[int],
) -> str:
    """Recompute proposed convolution endpoints using concrete C arithmetic."""

    outputs = tuple(int(index) for index in output_indices)
    if not outputs:
        raise ValueError("Certificate block cannot be empty")
    terms_by_output = [conv_output_terms(layer, index) for index in outputs]
    global_inputs = sorted({index for terms in terms_by_output for index, _ in terms}) or [0]
    local = {global_index: local_index for local_index, global_index in enumerate(global_inputs)}
    max_terms = max(1, *(len(terms) for terms in terms_by_output))
    term_count = []
    term_input = []
    term_weight = []
    biases = []
    for output_index, terms in zip(outputs, terms_by_output, strict=True):
        term_count.append(len(terms))
        term_input.extend([local[index] for index, _ in terms] + [0] * (max_terms - len(terms)))
        term_weight.extend([weight for _, weight in terms] + [0] * (max_terms - len(terms)))
        channel = np.unravel_index(output_index, layer.geometry.output_shape)[-1]
        biases.append(int(layer.bias_int[channel]))
    states = _relu_states(certificate, outputs)
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
{render_arith_kernel()}
#define BLOCK_SIZE {len(outputs)}
#define LOCAL_INPUT_SIZE {len(global_inputs)}
#define MAX_TERMS {max_terms}
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}
static const int64_t INPUT_LOW[LOCAL_INPUT_SIZE] = {_array(certificate.input_invariant.low[i] for i in global_inputs)};
static const int64_t INPUT_HIGH[LOCAL_INPUT_SIZE] = {_array(certificate.input_invariant.high[i] for i in global_inputs)};
static const int TERM_COUNT[BLOCK_SIZE] = {_array(term_count)};
static const int TERM_INPUT[BLOCK_SIZE * MAX_TERMS] = {_array(term_input)};
static const int64_t TERM_WEIGHT[BLOCK_SIZE * MAX_TERMS] = {_array(term_weight)};
static const int64_t BIAS[BLOCK_SIZE] = {_array(biases)};
static const int64_t EXPECTED_LOW[BLOCK_SIZE] = {_array(certificate.output_invariant.low[i] for i in outputs)};
static const int64_t EXPECTED_HIGH[BLOCK_SIZE] = {_array(certificate.output_invariant.high[i] for i in outputs)};
static const int STABLE_RELU[BLOCK_SIZE] = {_array(states)};
int main(void) {{
    for (int out = 0; out < BLOCK_SIZE; ++out) {{
        __int128 low = 0;
        __int128 high = 0;
        for (int term = 0; term < TERM_COUNT[out]; ++term) {{
            const int offset = out * MAX_TERMS + term;
            const int input_index = TERM_INPUT[offset];
            __int128 first = (__int128)TERM_WEIGHT[offset] * INPUT_LOW[input_index];
            __int128 second = (__int128)TERM_WEIGHT[offset] * INPUT_HIGH[input_index];
            low += first < second ? first : second;
            high += first < second ? second : first;
        }}
        low = div_round_half_away_from_zero_i128(
            low, ((__int128)1 << INPUT_FRACTIONAL_BITS)) + BIAS[out];
        high = div_round_half_away_from_zero_i128(
            high, ((__int128)1 << INPUT_FRACTIONAL_BITS)) + BIAS[out];
        low = clamp_to_signed_range_i128(low, TOTAL_BITS);
        high = clamp_to_signed_range_i128(high, TOTAL_BITS);
        if (STABLE_RELU[out] > 0)
            __ESBMC_assert(low >= 0, "stable active ReLU classification");
        if (STABLE_RELU[out] < 0)
            __ESBMC_assert(high <= 0, "stable inactive ReLU classification");
        if ({1 if layer.apply_relu else 0}) {{
            if (low < 0) low = 0;
            if (high < 0) high = 0;
        }}
        low = clamp_to_signed_range_i128(low, TOTAL_BITS);
        high = clamp_to_signed_range_i128(high, TOTAL_BITS);
        __ESBMC_assert(low == EXPECTED_LOW[out], "certificate lower endpoint");
        __ESBMC_assert(high == EXPECTED_HIGH[out], "certificate upper endpoint");
    }}
    return 0;
}}
"""


def render_dense_interval_certificate(
    layer: QuantizedDense,
    certificate: LayerIntervalCertificate,
    output_indices: Iterable[int],
) -> str:
    """Recompute proposed dense endpoints using concrete C arithmetic."""

    outputs = tuple(int(index) for index in output_indices)
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("Certificate block requires unique output indices")
    if any(index < 0 or index >= layer.output_size for index in outputs):
        raise IndexError("Dense certificate output index out of range")
    terms_by_output = [dense_output_terms(layer, index) for index in outputs]
    global_inputs = sorted({index for terms in terms_by_output for index, _ in terms}) or [0]
    local = {global_index: local_index for local_index, global_index in enumerate(global_inputs)}
    max_terms = max(1, *(len(terms) for terms in terms_by_output))
    term_count, term_input, term_weight = [], [], []
    for terms in terms_by_output:
        padding = max_terms - len(terms)
        term_count.append(len(terms))
        term_input.extend([local[index] for index, _ in terms] + [0] * padding)
        term_weight.extend([weight for _, weight in terms] + [0] * padding)
    states = _relu_states(certificate, outputs)
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
{render_arith_kernel()}
#define BLOCK_SIZE {len(outputs)}
#define LOCAL_INPUT_SIZE {len(global_inputs)}
#define MAX_TERMS {max_terms}
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}
static const int64_t INPUT_LOW[LOCAL_INPUT_SIZE] = {_array(certificate.input_invariant.low[i] for i in global_inputs)};
static const int64_t INPUT_HIGH[LOCAL_INPUT_SIZE] = {_array(certificate.input_invariant.high[i] for i in global_inputs)};
static const int TERM_COUNT[BLOCK_SIZE] = {_array(term_count)};
static const int TERM_INPUT[BLOCK_SIZE * MAX_TERMS] = {_array(term_input)};
static const int64_t TERM_WEIGHT[BLOCK_SIZE * MAX_TERMS] = {_array(term_weight)};
static const int64_t BIAS[BLOCK_SIZE] = {_array(layer.bias_int[index] for index in outputs)};
static const int64_t EXPECTED_LOW[BLOCK_SIZE] = {_array(certificate.output_invariant.low[index] for index in outputs)};
static const int64_t EXPECTED_HIGH[BLOCK_SIZE] = {_array(certificate.output_invariant.high[index] for index in outputs)};
static const int STABLE_RELU[BLOCK_SIZE] = {_array(states)};
int main(void) {{
    for (int out = 0; out < BLOCK_SIZE; ++out) {{
        __int128 low = 0;
        __int128 high = 0;
        for (int term = 0; term < TERM_COUNT[out]; ++term) {{
            const int offset = out * MAX_TERMS + term;
            const int input = TERM_INPUT[offset];
            __int128 first = (__int128)TERM_WEIGHT[offset] * INPUT_LOW[input];
            __int128 second = (__int128)TERM_WEIGHT[offset] * INPUT_HIGH[input];
            low += first < second ? first : second;
            high += first < second ? second : first;
        }}
        low = div_round_half_away_from_zero_i128(
            low, ((__int128)1 << INPUT_FRACTIONAL_BITS)) + BIAS[out];
        high = div_round_half_away_from_zero_i128(
            high, ((__int128)1 << INPUT_FRACTIONAL_BITS)) + BIAS[out];
        low = clamp_to_signed_range_i128(low, TOTAL_BITS);
        high = clamp_to_signed_range_i128(high, TOTAL_BITS);
        if (STABLE_RELU[out] > 0)
            __ESBMC_assert(low >= 0, "stable active ReLU classification");
        if (STABLE_RELU[out] < 0)
            __ESBMC_assert(high <= 0, "stable inactive ReLU classification");
        if ({1 if layer.apply_relu else 0}) {{
            if (low < 0) low = 0;
            if (high < 0) high = 0;
        }}
        low = clamp_to_signed_range_i128(low, TOTAL_BITS);
        high = clamp_to_signed_range_i128(high, TOTAL_BITS);
        __ESBMC_assert(low == EXPECTED_LOW[out], "dense certificate lower endpoint");
        __ESBMC_assert(high == EXPECTED_HIGH[out], "dense certificate upper endpoint");
    }}
    return 0;
}}
"""
