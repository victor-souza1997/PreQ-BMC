from __future__ import annotations

from pathlib import Path
from typing import Any

from backends.fixed_point import FixedPointNetwork


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def compute_fixed_point_resource_metrics(
    network: FixedPointNetwork,
    c_source_path: str | Path | None = None,
    c_shared_library_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compute compact resource metrics for a fixed-point QNN backend."""

    parameters_per_layer = [
        int(layer.weights_int.size + layer.biases_int.size)
        for layer in network.layers
    ]
    num_parameters = int(sum(parameters_per_layer))
    fixed_parameter_bits = int(
        sum(count * int(layer.spec.total_bits) for count, layer in zip(parameters_per_layer, network.layers, strict=True))
    )
    float_parameter_memory_bytes = int(num_parameters * 4)
    fixed_parameter_memory_bytes = int(_ceil_div(fixed_parameter_bits, 8)) if fixed_parameter_bits else 0
    weighted_avg_bits = float(fixed_parameter_bits / num_parameters) if num_parameters else 0.0

    input_dim = int(network.layers[0].weights_int.shape[1]) if network.layers else 0
    activation_widths = [input_dim] + [int(layer.biases_int.shape[0]) for layer in network.layers]
    activation_bits = [input_dim * int(network.input_total_bits)]
    activation_bits.extend(int(layer.biases_int.shape[0]) * int(layer.spec.total_bits) for layer in network.layers)
    peak_activation_values = int(max(activation_widths)) if activation_widths else 0
    activation_memory_bytes_estimate = int(_ceil_div(max(activation_bits), 8)) if activation_bits else 0

    source_lines: int | None = None
    if c_source_path is not None:
        source_path = Path(c_source_path)
        if source_path.exists():
            source_lines = len(source_path.read_text(encoding="utf-8", errors="ignore").splitlines())

    shared_size: int | None = None
    if c_shared_library_path is not None:
        shared_path = Path(c_shared_library_path)
        if shared_path.exists():
            shared_size = int(shared_path.stat().st_size)

    return {
        "num_layers": int(len(network.layers)),
        "num_parameters": num_parameters,
        "parameters_per_layer": parameters_per_layer,
        "float_parameter_memory_bytes": float_parameter_memory_bytes,
        "fixed_parameter_memory_bytes": fixed_parameter_memory_bytes,
        "compression_ratio_vs_float32": (
            float(float_parameter_memory_bytes / fixed_parameter_memory_bytes)
            if fixed_parameter_memory_bytes
            else None
        ),
        "weighted_avg_bits_per_parameter": weighted_avg_bits,
        "activation_memory_bytes_estimate": activation_memory_bytes_estimate,
        "peak_activation_values": peak_activation_values,
        "c_source_lines": source_lines,
        "c_shared_library_size_bytes": shared_size,
    }
