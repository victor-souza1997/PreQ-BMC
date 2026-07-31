from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from utils.fixed_point import (
    clamp_to_signed_range,
    round_divide_half_away_from_zero,
)


@dataclass(frozen=True)
class LayerReplayFormat:
    """Arithmetic metadata needed to replay one generated affine harness."""

    input_fractional_bits: int
    total_bits: int
    apply_relu: bool = False


def replay_on_python(
    inputs_int: np.ndarray | list[int],
    layer: Any,
    fmt: LayerReplayFormat,
) -> np.ndarray:
    """Replay one quantized affine layer with the deployed integer semantics."""

    inputs = np.asarray(inputs_int, dtype=object).reshape(-1)
    weights = np.asarray(layer.weights_int, dtype=object)
    biases = np.asarray(layer.biases_int, dtype=object).reshape(-1)
    if weights.ndim != 2 or weights.shape[1] != inputs.size:
        raise ValueError("Replay input dimension does not match layer weights.")
    if weights.shape[0] != biases.size:
        raise ValueError("Replay bias dimension does not match layer outputs.")
    if fmt.input_fractional_bits < 0:
        raise ValueError("input_fractional_bits must be non-negative.")

    denominator = 1 << int(fmt.input_fractional_bits)
    outputs: list[int] = []
    for row, bias in zip(weights, biases):
        accumulator = sum(
            int(weight) * int(value)
            for weight, value in zip(row, inputs)
        )
        value = round_divide_half_away_from_zero(
            accumulator,
            denominator,
        ) + int(bias)
        value = clamp_to_signed_range(value, int(fmt.total_bits))
        if fmt.apply_relu and value < 0:
            value = 0
        outputs.append(
            clamp_to_signed_range(value, int(fmt.total_bits))
        )
    return np.asarray(outputs, dtype=np.int64)


def replay_on_so(
    inputs_int: np.ndarray | list[int],
    so_path: str | Path,
) -> np.ndarray:
    """Replay a whole fixed-point network through the generated shared object."""

    library = ctypes.CDLL(str(Path(so_path).resolve()))
    library.qnn_input_dim.argtypes = []
    library.qnn_input_dim.restype = ctypes.c_int
    library.qnn_output_dim.argtypes = []
    library.qnn_output_dim.restype = ctypes.c_int
    library.qnn_forward_fixed.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.qnn_forward_fixed.restype = None

    input_dim = int(library.qnn_input_dim())
    output_dim = int(library.qnn_output_dim())
    values = np.asarray(inputs_int, dtype=np.int64).reshape(-1)
    if values.size != input_dim:
        raise ValueError(
            f"Shared-object replay expected {input_dim} inputs, got {values.size}."
        )

    input_array = (ctypes.c_int64 * input_dim)(
        *(int(value) for value in values)
    )
    output_array = (ctypes.c_int64 * output_dim)()
    library.qnn_forward_fixed(input_array, output_array)
    return np.asarray(list(output_array), dtype=np.int64)
