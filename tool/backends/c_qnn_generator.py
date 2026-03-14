from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np

from .fixed_point import FixedPointNetwork, quantize_network_input


def _c_array_1d(values: np.ndarray) -> str:
    return "{" + ", ".join(str(int(value)) for value in values.tolist()) + "}"


def _c_array_2d(values: np.ndarray) -> str:
    rows = ["{" + ", ".join(str(int(value)) for value in row.tolist()) + "}" for row in values]
    return "{" + ", ".join(rows) + "}"


def generate_c_qnn_source(network: FixedPointNetwork) -> str:
    """Generate a full-network C implementation for the fixed-point forward pass."""

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
        relu_clause = "if (value < 0) value = 0;" if not layer.is_output_layer else ""
        layer_steps.append(
            f"""
    for (int out_idx = 0; out_idx < LAYER_{index}_OUT; ++out_idx) {{
        __int128 acc = 0;
        for (int in_idx = 0; in_idx < LAYER_{index}_IN; ++in_idx) {{
            acc += (__int128)LAYER_{index}_WEIGHTS[out_idx][in_idx] * (__int128){in_buffer}[in_idx];
        }}
        __int128 value = div_round_half_away_from_zero_i128(acc, 1LL << {current_input_frac}) + (__int128)LAYER_{index}_BIASES[out_idx];
        value = clamp_to_signed_range(value, LAYER_{index}_Q);
        {relu_clause}
        {out_buffer}[out_idx] = (int64_t)clamp_to_signed_range(value, LAYER_{index}_Q);
    }}
"""
        )
        current_input_frac = layer.spec.fractional_bits

    final_buffer = "buffer_b" if (len(network.layers) - 1) % 2 == 0 else "buffer_a"

    return f"""\
#include <stdint.h>

static inline __int128 clamp_to_signed_range(__int128 value, int total_bits) {{
    const __int128 lower = -((__int128)1 << (total_bits - 1));
    const __int128 upper = (((__int128)1 << (total_bits - 1)) - 1);
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}}

static inline __int128 div_round_half_away_from_zero_i128(__int128 numerator, int64_t denominator) {{
    if (numerator >= 0) {{
        return (numerator + denominator / 2) / denominator;
    }}
    return -(((-numerator) + denominator / 2) / denominator);
}}

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
