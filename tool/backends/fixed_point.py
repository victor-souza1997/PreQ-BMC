from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from models.deep_model import DeepModel
from utils.fixed_point import clamp_to_signed_range, dequantize_int, quantize_int, round_divide_half_away_from_zero


@dataclass(frozen=True)
class LayerQuantizationSpec:
    """Per-layer fixed-point configuration."""

    total_bits: int
    integer_bits: int
    fractional_bits: int


@dataclass(frozen=True)
class QuantizedLayer:
    """Fixed-point parameters for one affine layer."""

    weights_int: np.ndarray
    biases_int: np.ndarray
    spec: LayerQuantizationSpec
    is_output_layer: bool


@dataclass(frozen=True)
class FixedPointNetwork:
    """Full-network fixed-point representation with explicit layer chaining."""

    input_fractional_bits: int
    input_total_bits: int
    layers: tuple[QuantizedLayer, ...]

    @property
    def output_fractional_bits(self) -> int:
        return self.layers[-1].spec.fractional_bits


def _extract_layer_units(model: DeepModel) -> list[int]:
    return [int(layer.units) for layer in model.dense_layers]


def clone_quantized_keras_model(model: DeepModel, layer_specs: list[LayerQuantizationSpec]) -> DeepModel:
    """Create a float Keras clone with quantized weights and biases applied."""

    quantized_model = DeepModel(_extract_layer_units(model), input_scale=model.input_scale)
    input_dim = int(model.dense_layers[0].kernel.shape[0])
    quantized_model.build((None, input_dim))
    _ = quantized_model(tf.zeros((1, input_dim), dtype=tf.float32))

    for layer_index, (source_layer, target_layer, spec) in enumerate(
        zip(model.dense_layers, quantized_model.dense_layers, layer_specs, strict=True)
    ):
        del layer_index
        kernel, bias = source_layer.get_weights()
        kernel_q = dequantize_int(quantize_int(kernel, spec.total_bits, spec.fractional_bits), spec.fractional_bits)
        bias_q = dequantize_int(quantize_int(bias, spec.total_bits, spec.fractional_bits), spec.fractional_bits)
        target_layer.set_weights([np.asarray(kernel_q, dtype=np.float32), np.asarray(bias_q, dtype=np.float32)])

    return quantized_model


def build_fixed_point_network(
    model: DeepModel,
    layer_specs: list[LayerQuantizationSpec],
    input_fractional_bits: int | None = None,
    input_total_bits: int | None = None,
) -> FixedPointNetwork:
    """Quantize a `DeepModel` into a chained integer fixed-point network."""

    if len(layer_specs) != len(model.dense_layers):
        raise ValueError("Expected one quantization spec per dense layer.")

    network_layers: list[QuantizedLayer] = []
    for index, (dense_layer, spec) in enumerate(zip(model.dense_layers, layer_specs, strict=True)):
        kernel, bias = dense_layer.get_weights()
        kernel_int = np.asarray(quantize_int(kernel.T, spec.total_bits, spec.fractional_bits), dtype=np.int64)
        bias_int = np.asarray(quantize_int(bias, spec.total_bits, spec.fractional_bits), dtype=np.int64)
        network_layers.append(
            QuantizedLayer(
                weights_int=kernel_int,
                biases_int=bias_int,
                spec=spec,
                is_output_layer=index == len(layer_specs) - 1,
            )
        )

    first_spec = layer_specs[0]
    return FixedPointNetwork(
        input_fractional_bits=first_spec.fractional_bits if input_fractional_bits is None else input_fractional_bits,
        input_total_bits=first_spec.total_bits if input_total_bits is None else input_total_bits,
        layers=tuple(network_layers),
    )


def quantize_network_input(network: FixedPointNetwork, sample: np.ndarray) -> np.ndarray:
    """Quantize a floating-point input sample for the first network layer."""

    quantized = quantize_int(np.asarray(sample, dtype=np.float64), network.input_total_bits, network.input_fractional_bits)
    return np.asarray(quantized, dtype=np.int64)


def forward_fixed_point_single(network: FixedPointNetwork, sample: np.ndarray) -> np.ndarray:
    """Run one sample through the integer fixed-point network."""

    activations = quantize_network_input(network, sample)
    input_frac_bits = network.input_fractional_bits

    for layer in network.layers:
        next_activations = np.zeros(layer.biases_int.shape[0], dtype=np.int64)
        for out_index in range(layer.biases_int.shape[0]):
            acc = 0
            for in_index in range(activations.shape[0]):
                acc += int(activations[in_index]) * int(layer.weights_int[out_index, in_index])
            value = round_divide_half_away_from_zero(acc, 1 << input_frac_bits) + int(layer.biases_int[out_index])
            value = clamp_to_signed_range(value, layer.spec.total_bits)
            if not layer.is_output_layer and value < 0:
                value = 0
            next_activations[out_index] = clamp_to_signed_range(value, layer.spec.total_bits)
        activations = next_activations
        input_frac_bits = layer.spec.fractional_bits

    return activations


def forward_fixed_point_batch(network: FixedPointNetwork, features: np.ndarray) -> np.ndarray:
    """Run a batch through the Python fixed-point interpreter and return dequantized logits."""

    outputs = [forward_fixed_point_single(network, sample) for sample in np.asarray(features, dtype=np.float64)]
    outputs_int = np.asarray(outputs, dtype=np.int64)
    return np.asarray(dequantize_int(outputs_int, network.output_fractional_bits), dtype=np.float64)
