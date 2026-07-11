from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from .fixed_point import FixedPointNetwork, forward_fixed_point_single, quantize_network_input
from verification.arith_kernel import render_arith_kernel


def _c_array_1d(values: np.ndarray) -> str:
    return "{" + ", ".join(str(int(value)) for value in values.tolist()) + "}"


def _c_array_2d(values: np.ndarray) -> str:
    rows = ["{" + ", ".join(str(int(value)) for value in row.tolist()) + "}" for row in values]
    return "{" + ", ".join(rows) + "}"


def generate_c_qnn_source(network: FixedPointNetwork) -> str:
    """Generate a full-network C implementation for the fixed-point forward pass.

    The generated reference backend mirrors Python exactly:
    acc = sum(input_int * weight_int)
    value = round_half_away_from_zero(acc / 2**F_in) + bias_int

    Inputs have F_in fractional bits. Weights, biases, and outputs for a layer use
    that layer's F_w fractional bits, so the accumulator has F_in + F_w fractional
    bits and division by 2**F_in leaves values in F_w before adding bias.
    """

    max_width = max(max(layer.weights_int.shape) for layer in network.layers)
    input_dim = int(network.layers[0].weights_int.shape[1])
    output_dim = int(network.layers[-1].weights_int.shape[0])

    layer_blocks: list[str] = []
    for index, layer in enumerate(network.layers):
        layer_blocks.append(
            f"""
static const int LAYER_{index}_IN = {layer.weights_int.shape[1]};
static const int LAYER_{index}_OUT = {layer.weights_int.shape[0]};
static const int LAYER_{index}_Q = {layer.spec.total_bits};
static const int LAYER_{index}_I = {layer.spec.integer_bits};
static const int LAYER_{index}_F = {layer.spec.fractional_bits};
static const int64_t LAYER_{index}_WEIGHTS[{layer.weights_int.shape[0]}][{layer.weights_int.shape[1]}] = {_c_array_2d(layer.weights_int)};
static const int64_t LAYER_{index}_BIASES[{layer.biases_int.shape[0]}] = {_c_array_1d(layer.biases_int)};
"""
        )

    layer_steps: list[str] = []
    current_input_frac = network.input_fractional_bits
    for index, layer in enumerate(network.layers):
        in_buffer = "buffer_a" if index % 2 == 0 else "buffer_b"
        out_buffer = "buffer_b" if index % 2 == 0 else "buffer_a"
        layer_input_frac = current_input_frac
        relu_step = (
            "        if (value < 0) {\n"
            "            value = 0;\n"
            "        }\n"
            if not layer.is_output_layer
            else ""
        )
        layer_steps.append(
            f"""
    for (int out_idx = 0; out_idx < LAYER_{index}_OUT; ++out_idx) {{
        __int128 acc = 0;
        for (int in_idx = 0; in_idx < LAYER_{index}_IN; ++in_idx) {{
            acc = mac_i128(acc, LAYER_{index}_WEIGHTS[out_idx][in_idx], {in_buffer}[in_idx]);
        }}
        /* acc has F_in({layer_input_frac}) + F_w(LAYER_{index}_F) fractional bits.
           Divide by 2**F_in to keep LAYER_{index}_F, matching Python exactly. */
        __int128 value = div_round_half_away_from_zero_i128(
            acc,
            ((__int128)1 << {current_input_frac})
        ) + (__int128)LAYER_{index}_BIASES[out_idx];
        value = clamp_to_signed_range_i128(value, LAYER_{index}_Q);
{relu_step}        value = clamp_to_signed_range_i128(value, LAYER_{index}_Q);
        {out_buffer}[out_idx] = (int64_t)value;
    }}
"""
        )
        current_input_frac = layer.spec.fractional_bits

    final_buffer = "buffer_b" if (len(network.layers) - 1) % 2 == 0 else "buffer_a"

    return f"""\
#include <stdint.h>
#include <limits.h>

#ifdef QNN_VERIFY_WITH_ESBMC
void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))
#else
#define QNN_ASSERT(cond, msg) ((void)0)
#endif

{render_arith_kernel()}

{''.join(layer_blocks)}

int qnn_input_dim(void) {{
    return {input_dim};
}}

int qnn_output_dim(void) {{
    return {output_dim};
}}

int qnn_input_fractional_bits(void) {{
    return {network.input_fractional_bits};
}}

int qnn_output_fractional_bits(void) {{
    return {network.output_fractional_bits};
}}

void qnn_forward_fixed(const int64_t* input, int64_t* output) {{
    int64_t buffer_a[{max(max_width, input_dim)}] = {{0}};
    int64_t buffer_b[{max_width}] = {{0}};

    for (int i = 0; i < {input_dim}; ++i) {{
        buffer_a[i] = input[i];
    }}

{''.join(layer_steps)}

    for (int i = 0; i < {output_dim}; ++i) {{
        output[i] = {final_buffer}[i];
    }}
}}
"""


def generate_fixed_point_semantics_test_source() -> str:
    """Generate a tiny C library exposing fixed-point semantic primitives for tests."""

    return """\
#include <stdint.h>

""" + render_arith_kernel() + """\

int64_t qnn_semantic_round_div_i64(int64_t numerator, int64_t denominator) {
    return (int64_t)div_round_half_away_from_zero_i128((__int128)numerator, (__int128)denominator);
}

int64_t qnn_semantic_clamp_i64(int64_t value, int total_bits) {
    return (int64_t)clamp_to_signed_range_i128((__int128)value, total_bits);
}

int64_t qnn_semantic_relu_i64(int64_t value) {
    return value < 0 ? 0 : value;
}

int64_t qnn_semantic_step_i64(
    int64_t accumulator,
    int input_fractional_bits,
    int64_t bias,
    int total_bits,
    int apply_relu
) {
    __int128 value = div_round_half_away_from_zero_i128(
        (__int128)accumulator,
        ((__int128)1 << input_fractional_bits)
    ) + (__int128)bias;
    value = clamp_to_signed_range_i128(value, total_bits);
    if (apply_relu && value < 0) {
        value = 0;
    }
    return (int64_t)clamp_to_signed_range_i128(value, total_bits);
}

int64_t qnn_semantic_affine2_i64(
    int64_t x0,
    int64_t x1,
    int64_t w0,
    int64_t w1,
    int64_t bias,
    int input_fractional_bits,
    int total_bits,
    int apply_relu
) {
    __int128 accumulator = 0;
    accumulator = mac_i128(accumulator, w0, x0);
    accumulator = mac_i128(accumulator, w1, x1);
    return qnn_semantic_step_i64(
        (int64_t)accumulator,
        input_fractional_bits,
        bias,
        total_bits,
        apply_relu
    );
}
"""


def write_fixed_point_semantics_test_source(destination: Path) -> Path:
    """Write the fixed-point semantic primitive C source to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(generate_fixed_point_semantics_test_source(), encoding="utf-8")
    return destination


def write_c_qnn_source(network: FixedPointNetwork, destination: Path) -> Path:
    """Write the generated C source to disk."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(generate_c_qnn_source(network), encoding="utf-8")
    return destination


def compile_c_qnn_shared_library(source_path: Path, output_path: Path, compiler: str = "gcc") -> Path:
    """Compile the generated C source into a shared library."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [compiler, "-shared", "-fPIC", "-O2", str(source_path), "-o", str(output_path)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return output_path


@dataclass
class CompiledCQNN:
    """ctypes wrapper around the compiled fixed-point C network."""

    network: FixedPointNetwork
    shared_library_path: Path

    def __post_init__(self) -> None:
        self.library = ctypes.CDLL(str(self.shared_library_path))
        self.library.qnn_forward_fixed.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self.library.qnn_forward_fixed.restype = None

    def forward(self, sample: np.ndarray) -> np.ndarray:
        quantized_input = np.asarray(quantize_network_input(self.network, sample), dtype=np.int64)
        output = np.zeros(self.network.layers[-1].biases_int.shape[0], dtype=np.int64)
        input_ptr = quantized_input.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
        self.library.qnn_forward_fixed(input_ptr, output_ptr)
        return output


def compare_python_c_fixed_point_outputs(
    network: FixedPointNetwork,
    samples: np.ndarray,
    compiled_qnn: CompiledCQNN,
) -> dict[str, Any]:
    """Compare final integer outputs from Python and compiled C fixed-point backends."""

    total = 0
    mismatch_count = 0
    max_integer_difference = 0
    first_mismatch: dict[str, Any] | None = None

    for index, sample in enumerate(np.asarray(samples, dtype=np.float64)):
        expected = np.asarray(forward_fixed_point_single(network, sample), dtype=np.int64)
        actual = np.asarray(compiled_qnn.forward(sample), dtype=np.int64)
        diff = np.abs(expected - actual)
        sample_max_diff = int(np.max(diff)) if diff.size else 0
        max_integer_difference = max(max_integer_difference, sample_max_diff)
        total += 1
        if not np.array_equal(expected, actual):
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "index": int(index),
                    "python_output": expected.tolist(),
                    "c_output": actual.tolist(),
                    "max_integer_difference": sample_max_diff,
                }

    return {
        "samples": int(total),
        "exact_match": bool(mismatch_count == 0),
        "mismatch_count": int(mismatch_count),
        "max_integer_difference": int(max_integer_difference),
        "first_mismatch": first_mismatch,
    }
