"""Sound interval contracts for convolution-native fixed-point operators.

Invariant synthesis is deliberately separate from proof checking.  Functions
in this module propose integer bounds and render ESBMC obligations that check
the exact deployed arithmetic over those bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np

from backends.conv_fixed_point import QuantizedConv2D, QuantizedDense, QuantizedOperator
from utils.fixed_point import clamp_to_signed_range, round_divide_half_away_from_zero
from verification.arith_kernel import render_arith_kernel


@dataclass(frozen=True)
class IntegerInvariant:
    low: tuple[int, ...]
    high: tuple[int, ...]
    shape: tuple[int, ...]
    provenance: str = "integer_interval"

    def __post_init__(self) -> None:
        size = math.prod(self.shape)
        if len(self.low) != size or len(self.high) != size:
            raise ValueError("Invariant shape does not match its bounds")
        if any(int(low) > int(high) for low, high in zip(self.low, self.high, strict=True)):
            raise ValueError("Invariant contains an empty coordinate")

    @classmethod
    def from_arrays(cls, low, high, shape, provenance="integer_interval"):
        return cls(
            tuple(int(value) for value in np.asarray(low).reshape(-1)),
            tuple(int(value) for value in np.asarray(high).reshape(-1)),
            tuple(int(value) for value in shape),
            provenance,
        )

    def contains(self, values) -> bool:
        flat = np.asarray(values).reshape(-1)
        return len(flat) == len(self.low) and all(
            low <= int(value) <= high
            for low, value, high in zip(self.low, flat, self.high, strict=True)
        )

    def subset_of(self, other: "IntegerInvariant") -> bool:
        return self.shape == other.shape and all(
            outer_low <= inner_low and inner_high <= outer_high
            for inner_low, inner_high, outer_low, outer_high in zip(
                self.low, self.high, other.low, other.high, strict=True
            )
        )


@dataclass(frozen=True)
class LayerIntervalCertificate:
    layer_index: int
    input_invariant: IntegerInvariant
    output_invariant: IntegerInvariant
    pre_relu_low: tuple[int, ...]
    pre_relu_high: tuple[int, ...]
    stable_active: tuple[int, ...]
    stable_inactive: tuple[int, ...]

    @property
    def unstable_count(self) -> int:
        return len(self.pre_relu_low) - len(self.stable_active) - len(self.stable_inactive)


def _weighted_interval(weights: Iterable[int], lows: Iterable[int], highs: Iterable[int]):
    lower = 0
    upper = 0
    for weight, low, high in zip(weights, lows, highs, strict=True):
        products = (int(weight) * int(low), int(weight) * int(high))
        lower += min(products)
        upper += max(products)
    return lower, upper


def _finish_interval(acc_low: int, acc_high: int, bias: int, layer: QuantizedOperator):
    denominator = 1 << layer.input_fractional_bits
    low = round_divide_half_away_from_zero(acc_low, denominator) + int(bias)
    high = round_divide_half_away_from_zero(acc_high, denominator) + int(bias)
    low = clamp_to_signed_range(low, layer.spec.total_bits)
    high = clamp_to_signed_range(high, layer.spec.total_bits)
    pre_low, pre_high = low, high
    if layer.apply_relu:
        low, high = max(0, low), max(0, high)
    return (
        clamp_to_signed_range(low, layer.spec.total_bits),
        clamp_to_signed_range(high, layer.spec.total_bits),
        pre_low,
        pre_high,
    )


def conv_output_terms(layer: QuantizedConv2D, output_index: int):
    """Return ``(flat_input_index, integer_weight)`` for one receptive field."""

    if not 0 <= output_index < layer.output_size:
        raise IndexError("Convolution output index out of range")
    oy, ox, oc = np.unravel_index(output_index, layer.geometry.output_shape)
    ih, iw, channels = layer.geometry.input_shape
    kh, kw, _, _ = layer.geometry.kernel_shape
    ph, pw = layer.geometry.pad_before
    terms = []
    for ky in range(kh):
        iy = oy * layer.geometry.strides[0] + ky - ph
        if not 0 <= iy < ih:
            continue
        for kx in range(kw):
            ix = ox * layer.geometry.strides[1] + kx - pw
            if not 0 <= ix < iw:
                continue
            for ic in range(channels):
                terms.append(((iy * iw + ix) * channels + ic,
                              int(layer.kernel_int[ky, kx, ic, oc])))
    return terms


def propagate_interval(
    layer: QuantizedOperator,
    invariant: IntegerInvariant,
    *,
    layer_index: int,
) -> LayerIntervalCertificate:
    """Compute a sound box transformer using monotonic integer operations."""

    if len(invariant.low) != layer.input_size:
        raise ValueError("Invariant does not match operator input")
    output_low: list[int] = []
    output_high: list[int] = []
    pre_low: list[int] = []
    pre_high: list[int] = []
    for output_index in range(layer.output_size):
        if isinstance(layer, QuantizedConv2D):
            terms = conv_output_terms(layer, output_index)
            indices = [index for index, _ in terms]
            weights = [weight for _, weight in terms]
            acc_low, acc_high = _weighted_interval(
                weights,
                (invariant.low[index] for index in indices),
                (invariant.high[index] for index in indices),
            )
            bias = int(layer.bias_int[np.unravel_index(
                output_index, layer.geometry.output_shape
            )[-1]])
        else:
            acc_low, acc_high = _weighted_interval(
                layer.weights_int[output_index], invariant.low, invariant.high
            )
            bias = int(layer.bias_int[output_index])
        low, high, before_low, before_high = _finish_interval(
            acc_low, acc_high, bias, layer
        )
        output_low.append(low)
        output_high.append(high)
        pre_low.append(before_low)
        pre_high.append(before_high)
    shape = (layer.output_size,) if isinstance(layer, QuantizedDense) else layer.geometry.output_shape
    stable_active = tuple(index for index, low in enumerate(pre_low) if low >= 0)
    stable_inactive = tuple(index for index, high in enumerate(pre_high) if high <= 0)
    return LayerIntervalCertificate(
        layer_index=layer_index,
        input_invariant=invariant,
        output_invariant=IntegerInvariant(
            tuple(output_low), tuple(output_high), shape, "sound_integer_interval_transform"
        ),
        pre_relu_low=tuple(pre_low),
        pre_relu_high=tuple(pre_high),
        stable_active=stable_active,
        stable_inactive=stable_inactive,
    )


def propagate_network(layers, input_invariant: IntegerInvariant):
    certificates = []
    current = input_invariant
    for index, layer in enumerate(layers):
        certificate = propagate_interval(layer, current, layer_index=index)
        certificates.append(certificate)
        current = certificate.output_invariant
    return tuple(certificates)


def _c_array(values) -> str:
    return "{" + ", ".join(str(int(value)) for value in values) + "}"


def render_conv_block_contract(
    layer: QuantizedConv2D,
    certificate: LayerIntervalCertificate,
    output_indices: Iterable[int],
) -> str:
    """Render one cone-of-influence ESBMC obligation for a convolution tile."""

    outputs = tuple(int(index) for index in output_indices)
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("A block requires unique output indices")
    terms_by_output = [conv_output_terms(layer, index) for index in outputs]
    global_inputs = sorted({index for terms in terms_by_output for index, _ in terms})
    local = {global_index: local_index for local_index, global_index in enumerate(global_inputs)}
    max_terms = max(len(terms) for terms in terms_by_output)
    term_counts = [len(terms) for terms in terms_by_output]
    term_indices = []
    term_weights = []
    biases = []
    for output_index, terms in zip(outputs, terms_by_output, strict=True):
        padded_indices = [local[index] for index, _ in terms] + [0] * (max_terms - len(terms))
        padded_weights = [weight for _, weight in terms] + [0] * (max_terms - len(terms))
        term_indices.extend(padded_indices)
        term_weights.extend(padded_weights)
        channel = np.unravel_index(output_index, layer.geometry.output_shape)[-1]
        biases.append(int(layer.bias_int[channel]))
    expected_states = []
    for output_index in outputs:
        if output_index in certificate.stable_active:
            expected_states.append(1)
        elif output_index in certificate.stable_inactive:
            expected_states.append(-1)
        else:
            expected_states.append(0)
    return f"""#include <stdint.h>

void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);

{render_arith_kernel()}

#define BLOCK_SIZE {len(outputs)}
#define LOCAL_INPUT_SIZE {len(global_inputs)}
#define MAX_TERMS {max_terms}
#define TOTAL_BITS {layer.spec.total_bits}
#define INPUT_FRACTIONAL_BITS {layer.input_fractional_bits}

static const int64_t INPUT_LOW[LOCAL_INPUT_SIZE] = {_c_array(certificate.input_invariant.low[i] for i in global_inputs)};
static const int64_t INPUT_HIGH[LOCAL_INPUT_SIZE] = {_c_array(certificate.input_invariant.high[i] for i in global_inputs)};
static const int TERM_COUNT[BLOCK_SIZE] = {_c_array(term_counts)};
static const int TERM_INPUT[BLOCK_SIZE * MAX_TERMS] = {_c_array(term_indices)};
static const int64_t TERM_WEIGHT[BLOCK_SIZE * MAX_TERMS] = {_c_array(term_weights)};
static const int64_t BIAS[BLOCK_SIZE] = {_c_array(biases)};
static const int64_t OUTPUT_LOW[BLOCK_SIZE] = {_c_array(certificate.output_invariant.low[i] for i in outputs)};
static const int64_t OUTPUT_HIGH[BLOCK_SIZE] = {_c_array(certificate.output_invariant.high[i] for i in outputs)};
static const int STABLE_RELU[BLOCK_SIZE] = {_c_array(expected_states)};

int main(void) {{
    int64_t input[LOCAL_INPUT_SIZE];
    for (int i = 0; i < LOCAL_INPUT_SIZE; ++i) {{
        input[i] = nondet_long_long();
        __ESBMC_assume(input[i] >= INPUT_LOW[i] && input[i] <= INPUT_HIGH[i]);
    }}
    for (int out = 0; out < BLOCK_SIZE; ++out) {{
        __int128 acc = 0;
        for (int term = 0; term < TERM_COUNT[out]; ++term) {{
            const int offset = out * MAX_TERMS + term;
            acc = mac_i128(acc, TERM_WEIGHT[offset], input[TERM_INPUT[offset]]);
        }}
        __int128 pre_relu = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << INPUT_FRACTIONAL_BITS)
        ) + (__int128)BIAS[out];
        pre_relu = clamp_to_signed_range_i128(pre_relu, TOTAL_BITS);
        if (STABLE_RELU[out] > 0)
            __ESBMC_assert(pre_relu >= 0, "stable active ReLU classification");
        if (STABLE_RELU[out] < 0)
            __ESBMC_assert(pre_relu <= 0, "stable inactive ReLU classification");
        __int128 value = pre_relu < 0 ? 0 : pre_relu;
        value = clamp_to_signed_range_i128(value, TOTAL_BITS);
        __ESBMC_assert(
            value >= OUTPUT_LOW[out] && value <= OUTPUT_HIGH[out],
            "convolution output outside proposed integer invariant"
        );
    }}
    return 0;
}}
"""


def render_invariant_bridge(
    guarantee: IntegerInvariant,
    assumption: IntegerInvariant,
) -> str:
    """Render the explicit set-inclusion obligation used for contract chaining."""

    if guarantee.shape != assumption.shape:
        raise ValueError("Cannot chain differently shaped invariants")
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
#define INVARIANT_SIZE {len(guarantee.low)}
static const int64_t GUARANTEE_LOW[INVARIANT_SIZE] = {_c_array(guarantee.low)};
static const int64_t GUARANTEE_HIGH[INVARIANT_SIZE] = {_c_array(guarantee.high)};
static const int64_t ASSUMPTION_LOW[INVARIANT_SIZE] = {_c_array(assumption.low)};
static const int64_t ASSUMPTION_HIGH[INVARIANT_SIZE] = {_c_array(assumption.high)};
int main(void) {{
    for (int i = 0; i < INVARIANT_SIZE; ++i) {{
        __ESBMC_assert(ASSUMPTION_LOW[i] <= GUARANTEE_LOW[i], "lower-bound chaining");
        __ESBMC_assert(GUARANTEE_HIGH[i] <= ASSUMPTION_HIGH[i], "upper-bound chaining");
    }}
    return 0;
}}
"""


def render_margin_contract(invariant: IntegerInvariant, target: int) -> str:
    if len(invariant.shape) != 1 or not 0 <= target < len(invariant.low):
        raise ValueError("Final margin requires a flat logit invariant and valid target")
    return f"""#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
#define OUTPUT_SIZE {len(invariant.low)}
#define TARGET_CLASS {target}
static const int64_t OUTPUT_LOW[OUTPUT_SIZE] = {_c_array(invariant.low)};
static const int64_t OUTPUT_HIGH[OUTPUT_SIZE] = {_c_array(invariant.high)};
int main(void) {{
    for (int competitor = 0; competitor < OUTPUT_SIZE; ++competitor) {{
        if (competitor != TARGET_CLASS)
            __ESBMC_assert(
                OUTPUT_LOW[TARGET_CLASS] > OUTPUT_HIGH[competitor],
                "strict lowest-index classification margin"
            );
    }}
    return 0;
}}
"""


def obligation_sha256(source: str, metadata: dict) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((source + "\n" + payload).encode("utf-8")).hexdigest()
