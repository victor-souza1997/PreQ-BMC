"""Convolution-native fixed-point network and deployment C generator.

The existing affine backend remains the reference for fully connected models.
This module preserves convolution geometry so verification and deployment do
not materialize mostly-zero Toeplitz matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TypeAlias

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from models.restricted_conv import ConvGeometry, RestrictedSequentialCNN, direct_conv
from utils.fixed_point import clamp_to_signed_range, quantize_int, round_divide_half_away_from_zero
from verification.arith_kernel import render_arith_kernel


@dataclass(frozen=True)
class QuantizedConv2D:
    geometry: ConvGeometry
    kernel_int: np.ndarray
    bias_int: np.ndarray
    spec: LayerQuantizationSpec
    input_fractional_bits: int
    apply_relu: bool = True

    def __post_init__(self) -> None:
        kernel = np.asarray(self.kernel_int, dtype=np.int64)
        bias = np.asarray(self.bias_int, dtype=np.int64)
        if kernel.shape != self.geometry.kernel_shape:
            raise ValueError("Quantized convolution kernel shape mismatch")
        if bias.shape != (self.geometry.kernel_shape[-1],):
            raise ValueError("Quantized convolution bias shape mismatch")
        if not 0 <= self.input_fractional_bits < 63:
            raise ValueError("Invalid convolution input fractional width")
        object.__setattr__(self, "kernel_int", kernel)
        object.__setattr__(self, "bias_int", bias)

    @property
    def input_size(self) -> int:
        return math.prod(self.geometry.input_shape)

    @property
    def output_size(self) -> int:
        return math.prod(self.geometry.output_shape)


@dataclass(frozen=True)
class QuantizedDense:
    weights_int: np.ndarray  # output-by-input
    bias_int: np.ndarray
    spec: LayerQuantizationSpec
    input_fractional_bits: int
    apply_relu: bool

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights_int, dtype=np.int64)
        bias = np.asarray(self.bias_int, dtype=np.int64)
        if weights.ndim != 2 or bias.ndim != 1 or weights.shape[0] != len(bias):
            raise ValueError("Quantized dense parameter shape mismatch")
        if not 0 <= self.input_fractional_bits < 63:
            raise ValueError("Invalid dense input fractional width")
        object.__setattr__(self, "weights_int", weights)
        object.__setattr__(self, "bias_int", bias)

    @property
    def input_size(self) -> int:
        return int(self.weights_int.shape[1])

    @property
    def output_size(self) -> int:
        return int(self.weights_int.shape[0])


QuantizedOperator: TypeAlias = QuantizedConv2D | QuantizedDense


@dataclass(frozen=True)
class ConvFixedPointNetwork:
    input_shape: tuple[int, int, int]
    input_fractional_bits: int
    input_total_bits: int
    layers: tuple[QuantizedOperator, ...]

    def __post_init__(self) -> None:
        if not self.layers or math.prod(self.input_shape) != self.layers[0].input_size:
            raise ValueError("Network input does not match its first operator")
        if not 2 <= self.input_total_bits <= 63:
            raise ValueError("Input storage must fit a signed int64")
        if not 0 <= self.input_fractional_bits < self.input_total_bits:
            raise ValueError("Invalid network input format")
        for previous, current in zip(self.layers, self.layers[1:]):
            if previous.output_size != current.input_size:
                raise ValueError("Quantized operators do not compose")
            if previous.spec.fractional_bits != current.input_fractional_bits:
                raise ValueError("Layer fractional formats do not compose")
        input_bits = self.input_total_bits
        for index, layer in enumerate(self.layers):
            max_input = 1 << (input_bits - 1)
            parameters = (layer.kernel_int if isinstance(layer, QuantizedConv2D)
                          else layer.weights_int)
            max_weight = max((abs(int(value)) for value in parameters.reshape(-1)), default=0)
            terms = (math.prod(layer.geometry.kernel_shape[:3])
                     if isinstance(layer, QuantizedConv2D) else layer.input_size)
            envelope = terms * max_input * max_weight
            if envelope >= 1 << 127:
                raise ValueError(
                    f"Layer {index} accumulator envelope does not fit signed __int128"
                )
            input_bits = layer.spec.total_bits

    @property
    def accumulator_envelopes(self) -> tuple[int, ...]:
        """Conservative absolute MAC bounds used to exclude __int128 overflow."""

        bounds = []
        input_bits = self.input_total_bits
        for layer in self.layers:
            max_input = 1 << (input_bits - 1)
            parameters = (layer.kernel_int if isinstance(layer, QuantizedConv2D)
                          else layer.weights_int)
            max_weight = max((abs(int(value)) for value in parameters.reshape(-1)), default=0)
            terms = (math.prod(layer.geometry.kernel_shape[:3])
                     if isinstance(layer, QuantizedConv2D) else layer.input_size)
            bounds.append(terms * max_input * max_weight)
            input_bits = layer.spec.total_bits
        return tuple(bounds)


    @property
    def output_size(self) -> int:
        return self.layers[-1].output_size

    @property
    def output_fractional_bits(self) -> int:
        return self.layers[-1].spec.fractional_bits


def quantize_restricted_sequential(
    model: RestrictedSequentialCNN,
    specs: list[LayerQuantizationSpec] | tuple[LayerQuantizationSpec, ...],
    *,
    input_fractional_bits: int = 8,
    input_total_bits: int = 16,
) -> ConvFixedPointNetwork:
    """Quantize compact kernels without invoking affine convolution lowering."""

    expected = len(model.conv_kernels) + len(model.dense_kernels)
    if len(specs) != expected:
        raise ValueError(f"Expected one shared Q/I/F for each of {expected} affine stages")
    layers: list[QuantizedOperator] = []
    input_fractional = input_fractional_bits
    for geometry, kernel, bias, spec in zip(
        model.geometries, model.conv_kernels, model.conv_biases, specs[: len(model.conv_kernels)]
    ):
        layers.append(QuantizedConv2D(
            geometry=geometry,
            kernel_int=np.asarray(quantize_int(kernel, spec.total_bits, spec.fractional_bits)),
            bias_int=np.asarray(quantize_int(bias, spec.total_bits, spec.fractional_bits)),
            spec=spec,
            input_fractional_bits=input_fractional,
            apply_relu=True,
        ))
        input_fractional = spec.fractional_bits
    offset = len(model.conv_kernels)
    for index, (kernel, bias, spec) in enumerate(zip(
        model.dense_kernels, model.dense_biases, specs[offset:]
    )):
        layers.append(QuantizedDense(
            weights_int=np.asarray(
                quantize_int(kernel.T, spec.total_bits, spec.fractional_bits), dtype=np.int64
            ),
            bias_int=np.asarray(quantize_int(bias, spec.total_bits, spec.fractional_bits)),
            spec=spec,
            input_fractional_bits=input_fractional,
            apply_relu=index + 1 < len(model.dense_kernels),
        ))
        input_fractional = spec.fractional_bits
    return ConvFixedPointNetwork(
        input_shape=model.input_shape,
        input_fractional_bits=input_fractional_bits,
        input_total_bits=input_total_bits,
        layers=tuple(layers),
    )


def _finish_value(accumulator: int, bias: int, layer: QuantizedOperator) -> int:
    value = round_divide_half_away_from_zero(
        int(accumulator), 1 << layer.input_fractional_bits
    ) + int(bias)
    value = clamp_to_signed_range(value, layer.spec.total_bits)
    if layer.apply_relu and value < 0:
        value = 0
    return clamp_to_signed_range(value, layer.spec.total_bits)


def forward_conv_fixed_point_single(
    network: ConvFixedPointNetwork,
    sample_int: np.ndarray,
    *,
    return_trace: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Execute the exact compact integer semantics using Python integers."""

    value = np.asarray(sample_int, dtype=np.int64)
    if value.size != math.prod(network.input_shape):
        raise ValueError("Input size mismatch")
    value = value.reshape(network.input_shape)
    trace: list[np.ndarray] = []
    for layer in network.layers:
        if isinstance(layer, QuantizedConv2D):
            if value.shape != layer.geometry.input_shape:
                raise ValueError("Convolution input shape mismatch")
            accumulator = direct_conv(value, layer.kernel_int, layer.geometry)
            output = np.empty(layer.geometry.output_shape, dtype=np.int64)
            for index in np.ndindex(output.shape):
                output[index] = _finish_value(
                    int(accumulator[index]), int(layer.bias_int[index[-1]]), layer
                )
            value = output
        else:
            flat = value.reshape(-1)
            output = np.empty(layer.output_size, dtype=np.int64)
            for out_index in range(layer.output_size):
                accumulator = sum(
                    int(x) * int(w)
                    for x, w in zip(flat, layer.weights_int[out_index], strict=True)
                )
                output[out_index] = _finish_value(
                    accumulator, int(layer.bias_int[out_index]), layer
                )
            value = output
        trace.append(np.asarray(value, dtype=np.int64).copy())
    final = np.asarray(value, dtype=np.int64).reshape(-1)
    return (final, tuple(trace)) if return_trace else final


def _c_values(values: np.ndarray) -> str:
    return "{" + ", ".join(str(int(value)) for value in np.asarray(values).reshape(-1)) + "}"


def generate_conv_qnn_source(
    network: ConvFixedPointNetwork, *, encoder_source: str = ""
) -> str:
    """Render compact deployment C with the shared arithmetic kernel verbatim."""

    declarations: list[str] = []
    steps: list[str] = []
    for index, layer in enumerate(network.layers):
        if isinstance(layer, QuantizedConv2D):
            ih, iw, ic = layer.geometry.input_shape
            oh, ow, oc = layer.geometry.output_shape
            kh, kw, _, _ = layer.geometry.kernel_shape
            sh, sw = layer.geometry.strides
            ph, pw = layer.geometry.pad_before
            declarations.append(f"""
static const int64_t LAYER_{index}_KERNEL[{layer.kernel_int.size}] = {_c_values(layer.kernel_int)};
static const int64_t LAYER_{index}_BIAS[{len(layer.bias_int)}] = {_c_values(layer.bias_int)};
""")
            relu = "        if (value < 0) value = 0;\n" if layer.apply_relu else ""
            steps.append(f"""
    for (int oy = 0; oy < {oh}; ++oy) {{
        for (int ox = 0; ox < {ow}; ++ox) {{
            for (int oc = 0; oc < {oc}; ++oc) {{
                __int128 acc = 0;
                for (int ky = 0; ky < {kh}; ++ky) {{
                    const int iy = oy * {sh} + ky - {ph};
                    if (iy < 0 || iy >= {ih}) continue;
                    for (int kx = 0; kx < {kw}; ++kx) {{
                        const int ix = ox * {sw} + kx - {pw};
                        if (ix < 0 || ix >= {iw}) continue;
                        for (int ic = 0; ic < {ic}; ++ic) {{
                            const int input_index = (iy * {iw} + ix) * {ic} + ic;
                            const int kernel_index = ((ky * {kw} + kx) * {ic} + ic) * {oc} + oc;
                            acc = mac_i128(acc, LAYER_{index}_KERNEL[kernel_index], buffer_{index % 2}[input_index]);
                        }}
                    }}
                }}
                __int128 value = div_round_half_away_from_zero_i128(
                    acc, ((__int128)1 << {layer.input_fractional_bits})
                ) + (__int128)LAYER_{index}_BIAS[oc];
                value = clamp_to_signed_range_i128(value, {layer.spec.total_bits});
{relu}                value = clamp_to_signed_range_i128(value, {layer.spec.total_bits});
                buffer_{(index + 1) % 2}[(oy * {ow} + ox) * {oc} + oc] = (int64_t)value;
            }}
        }}
    }}
""")
        else:
            declarations.append(f"""
static const int64_t LAYER_{index}_WEIGHTS[{layer.weights_int.size}] = {_c_values(layer.weights_int)};
static const int64_t LAYER_{index}_BIAS[{len(layer.bias_int)}] = {_c_values(layer.bias_int)};
""")
            relu = "        if (value < 0) value = 0;\n" if layer.apply_relu else ""
            steps.append(f"""
    for (int out_index = 0; out_index < {layer.output_size}; ++out_index) {{
        __int128 acc = 0;
        for (int in_index = 0; in_index < {layer.input_size}; ++in_index) {{
            acc = mac_i128(
                acc,
                LAYER_{index}_WEIGHTS[out_index * {layer.input_size} + in_index],
                buffer_{index % 2}[in_index]
            );
        }}
        __int128 value = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << {layer.input_fractional_bits})
        ) + (__int128)LAYER_{index}_BIAS[out_index];
        value = clamp_to_signed_range_i128(value, {layer.spec.total_bits});
{relu}        value = clamp_to_signed_range_i128(value, {layer.spec.total_bits});
        buffer_{(index + 1) % 2}[out_index] = (int64_t)value;
    }}
""")
    maximum = max(math.prod(network.input_shape), *(layer.output_size for layer in network.layers))
    final_buffer = len(network.layers) % 2
    return f"""#include <stdint.h>

{render_arith_kernel()}

{encoder_source}

{''.join(declarations)}

int qnn_input_dim(void) {{ return {math.prod(network.input_shape)}; }}
int qnn_output_dim(void) {{ return {network.output_size}; }}
int qnn_input_fractional_bits(void) {{ return {network.input_fractional_bits}; }}
int qnn_output_fractional_bits(void) {{ return {network.output_fractional_bits}; }}

void qnn_forward_fixed(const int64_t *input, int64_t *output) {{
    int64_t buffer_0[{maximum}] = {{0}};
    int64_t buffer_1[{maximum}] = {{0}};
    for (int i = 0; i < {math.prod(network.input_shape)}; ++i) buffer_0[i] = input[i];
{''.join(steps)}
    for (int i = 0; i < {network.output_size}; ++i) output[i] = buffer_{final_buffer}[i];
}}
"""
