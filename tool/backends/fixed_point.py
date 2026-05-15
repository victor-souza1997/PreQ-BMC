from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from utils.fixed_point import (
    clamp_to_signed_range,
    dequantize_int,
    quantize_int,
    round_divide_half_away_from_zero,
    signed_int_bounds,
)

if TYPE_CHECKING:
    from models.deep_model import DeepModel


@dataclass(frozen=True)
class LayerQuantizationSpec:
    """Per-layer fixed-point configuration.

    `integer_bits` excludes the sign bit. For a signed Q format the invariant is:
    total_bits == integer_bits + fractional_bits + 1.
    """

    total_bits: int
    integer_bits: int
    fractional_bits: int

    def __post_init__(self) -> None:
        if self.total_bits <= 1:
            raise ValueError(f"total_bits must include at least one value bit and one sign bit: Q={self.total_bits}")
        if self.integer_bits < 0:
            raise ValueError(f"integer_bits must be non-negative: I={self.integer_bits}")
        if self.fractional_bits < 0:
            raise ValueError(f"fractional_bits must be non-negative: F={self.fractional_bits}")
        expected_total_bits = self.integer_bits + self.fractional_bits + 1
        if self.total_bits != expected_total_bits:
            raise ValueError(
                "Invalid fixed-point format: total_bits must equal integer_bits + "
                f"fractional_bits + 1 sign bit (Q={self.total_bits}, I={self.integer_bits}, "
                f"F={self.fractional_bits}; expected Q={expected_total_bits})."
            )

    @property
    def signed_range(self) -> tuple[int, int]:
        """Return the signed integer container range for this layer."""

        return signed_int_bounds(self.total_bits)

    @property
    def real_range(self) -> tuple[float, float]:
        """Return the representable real range for this layer."""

        q_min, q_max = self.signed_range
        scale = float(2**self.fractional_bits)
        return (q_min / scale, q_max / scale)


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

    import tensorflow as tf
    from models.deep_model import DeepModel

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
    """Quantize a `DeepModel` into a chained integer fixed-point network.

    Arithmetic convention:
    - layer input activations carry `F_in` fractional bits;
    - layer weights and biases are quantized with the layer spec's `F_w`;
    - each product has `F_in + F_w` fractional bits;
    - dividing the accumulator by `2**F_in` leaves `F_w` fractional bits;
    - `bias_int` is also stored with `F_w` fractional bits;
    - the layer output therefore has `F_w` fractional bits and becomes the next layer input.
    """

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
            # Input has F_in fractional bits and weights/biases/output use F_w.
            # acc is scaled by 2**(F_in + F_w); division by 2**F_in leaves F_w.
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


def forward_fixed_point_batch_with_diagnostics(
    network: FixedPointNetwork,
    features: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Run a batch and report per-layer fixed-point saturation diagnostics."""

    layer_diagnostics = [
        {
            "layer_index": int(index),
            "total_values": 0,
            "saturation_count": 0,
            "saturation_rate": 0.0,
            "min_value_seen": None,
            "max_value_seen": None,
            "q_min": int(layer.spec.signed_range[0]),
            "q_max": int(layer.spec.signed_range[1]),
        }
        for index, layer in enumerate(network.layers)
    ]

    outputs: list[np.ndarray] = []
    normalized_features = np.asarray(features, dtype=np.float64)
    for sample in normalized_features:
        activations = quantize_network_input(network, sample)
        input_frac_bits = network.input_fractional_bits

        for layer_index, layer in enumerate(network.layers):
            next_activations = np.zeros(layer.biases_int.shape[0], dtype=np.int64)
            diag = layer_diagnostics[layer_index]
            q_min, q_max = layer.spec.signed_range

            for out_index in range(layer.biases_int.shape[0]):
                acc = 0
                for in_index in range(activations.shape[0]):
                    acc += int(activations[in_index]) * int(layer.weights_int[out_index, in_index])
                value_before_clamp = round_divide_half_away_from_zero(acc, 1 << input_frac_bits) + int(
                    layer.biases_int[out_index]
                )

                diag["total_values"] += 1
                diag["min_value_seen"] = (
                    int(value_before_clamp)
                    if diag["min_value_seen"] is None
                    else min(int(diag["min_value_seen"]), int(value_before_clamp))
                )
                diag["max_value_seen"] = (
                    int(value_before_clamp)
                    if diag["max_value_seen"] is None
                    else max(int(diag["max_value_seen"]), int(value_before_clamp))
                )
                if value_before_clamp < q_min or value_before_clamp > q_max:
                    diag["saturation_count"] += 1

                value = clamp_to_signed_range(value_before_clamp, layer.spec.total_bits)
                if not layer.is_output_layer and value < 0:
                    value = 0
                next_activations[out_index] = clamp_to_signed_range(value, layer.spec.total_bits)

            activations = next_activations
            input_frac_bits = layer.spec.fractional_bits

        outputs.append(activations)

    for diag in layer_diagnostics:
        total_values = int(diag["total_values"])
        diag["saturation_count"] = int(diag["saturation_count"])
        diag["saturation_rate"] = float(diag["saturation_count"] / total_values) if total_values else 0.0
        diag["min_value_seen"] = int(diag["min_value_seen"]) if diag["min_value_seen"] is not None else 0
        diag["max_value_seen"] = int(diag["max_value_seen"]) if diag["max_value_seen"] is not None else 0

    outputs_int = np.asarray(outputs, dtype=np.int64)
    logits = np.asarray(dequantize_int(outputs_int, network.output_fractional_bits), dtype=np.float64)
    return logits, {"samples": int(normalized_features.shape[0]), "layers": layer_diagnostics}


def forward_fixed_point_single_trace(network: FixedPointNetwork, sample: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run one sample and return final integer output plus dequantized layer outputs."""

    activations = quantize_network_input(network, sample)
    input_frac_bits = network.input_fractional_bits
    layer_outputs: list[np.ndarray] = []

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
        layer_outputs.append(np.asarray(dequantize_int(activations, layer.spec.fractional_bits), dtype=np.float64))

    return activations, layer_outputs
