"""Vectorized integer box propagation for candidate measurement.

This mirrors ``conv_contracts.propagate_network`` bit for bit (checked by
``test_conv_interval_fast``) but uses int64 im2col matrix products instead of
per-neuron Python loops, so thousands of regions can be screened per minute.

Everything here is a MEASUREMENT used to choose which candidate to send to the
prover.  Certificates still come only from ESBMC obligations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backends.conv_fixed_point import ConvFixedPointNetwork, QuantizedConv2D

# Largest accumulator any supported layer can produce must stay far below this.
_INT64_HEADROOM = 1 << 62


def _round_half_away(numerator: np.ndarray, denominator: int) -> np.ndarray:
    magnitude = (np.abs(numerator) + denominator // 2) // denominator
    return np.where(numerator >= 0, magnitude, -magnitude)


def _im2col(value: np.ndarray, layer: QuantizedConv2D) -> np.ndarray:
    """Rows are output pixels; columns follow the (ky, kx, ic) kernel order."""

    geometry = layer.geometry
    ph, pw = geometry.pad_before
    kh, kw, channels, _ = geometry.kernel_shape
    oh, ow, _ = geometry.output_shape
    sh, sw = geometry.strides
    padded = np.zeros(
        (max(value.shape[0] + ph, (oh - 1) * sh + kh),
         max(value.shape[1] + pw, (ow - 1) * sw + kw), channels),
        dtype=np.int64,
    )
    padded[ph:ph + value.shape[0], pw:pw + value.shape[1], :] = value
    columns = np.empty((oh * ow, kh * kw * channels), dtype=np.int64)
    for row, (oy, ox) in enumerate(np.ndindex((oh, ow))):
        columns[row] = padded[oy * sh:oy * sh + kh, ox * sw:ox * sw + kw, :].reshape(-1)
    return columns


def _check_headroom(weights: np.ndarray, low: np.ndarray, high: np.ndarray) -> None:
    magnitude = np.maximum(np.abs(low), np.abs(high)).max(initial=0)
    fan_in = weights.shape[0]
    if int(np.abs(weights).max(initial=0)) * int(magnitude) * fan_in >= _INT64_HEADROOM:
        raise OverflowError("int64 accumulator headroom exceeded; use propagate_network")


def _finish(acc_low, acc_high, bias, layer):
    denominator = 1 << layer.input_fractional_bits
    limit_low = -(1 << (layer.spec.total_bits - 1))
    limit_high = (1 << (layer.spec.total_bits - 1)) - 1
    pre_low = np.clip(_round_half_away(acc_low, denominator) + bias, limit_low, limit_high)
    pre_high = np.clip(_round_half_away(acc_high, denominator) + bias, limit_low, limit_high)
    if layer.apply_relu:
        return np.maximum(pre_low, 0), np.maximum(pre_high, 0), pre_low, pre_high
    return pre_low, pre_high, pre_low, pre_high


@dataclass(frozen=True)
class BoxTrace:
    """Per-layer flat integer boxes; ``pre_*`` are before ReLU, after clamp."""

    low: tuple[np.ndarray, ...]
    high: tuple[np.ndarray, ...]
    pre_low: tuple[np.ndarray, ...]
    pre_high: tuple[np.ndarray, ...]
    # Unclamped accumulator images for the final layer, used by the
    # difference-row margin, which must see through the output clamp.
    final_input_low: np.ndarray
    final_input_high: np.ndarray


def propagate_box(network: ConvFixedPointNetwork, low, high) -> BoxTrace:
    low = np.asarray(low, dtype=np.int64).reshape(network.input_shape)
    high = np.asarray(high, dtype=np.int64).reshape(network.input_shape)
    lows, highs, pre_lows, pre_highs = [], [], [], []
    final_input_low = final_input_high = None
    for index, layer in enumerate(network.layers):
        if index == len(network.layers) - 1:
            final_input_low, final_input_high = low.reshape(-1), high.reshape(-1)
        if isinstance(layer, QuantizedConv2D):
            weights = np.asarray(layer.kernel_int, dtype=np.int64).reshape(
                -1, layer.geometry.kernel_shape[-1])
            _check_headroom(weights, low, high)
            cols_low, cols_high = _im2col(low, layer), _im2col(high, layer)
            bias = np.asarray(layer.bias_int, dtype=np.int64)
            shape = layer.geometry.output_shape
        else:
            weights = np.asarray(layer.weights_int, dtype=np.int64).T
            _check_headroom(weights, low, high)
            cols_low, cols_high = low.reshape(1, -1), high.reshape(1, -1)
            bias = np.asarray(layer.bias_int, dtype=np.int64)
            shape = (layer.output_size,)
        positive, negative = np.maximum(weights, 0), np.minimum(weights, 0)
        acc_low = cols_low @ positive + cols_high @ negative
        acc_high = cols_high @ positive + cols_low @ negative
        out_low, out_high, pre_low, pre_high = _finish(acc_low, acc_high, bias, layer)
        low, high = out_low.reshape(shape), out_high.reshape(shape)
        lows.append(low.reshape(-1))
        highs.append(high.reshape(-1))
        pre_lows.append(pre_low.reshape(-1))
        pre_highs.append(pre_high.reshape(-1))
    return BoxTrace(tuple(lows), tuple(highs), tuple(pre_lows), tuple(pre_highs),
                    final_input_low, final_input_high)


def box_margin(trace: BoxTrace, target: int) -> int:
    """``low(target) - max high(competitor)``, the check the margin harness makes."""

    high = trace.high[-1].copy()
    high[target] = np.iinfo(np.int64).min
    return int(trace.low[-1][target] - high.max())


def difference_row_margin(network: ConvFixedPointNetwork, trace: BoxTrace, target: int):
    """Lower bound on ``v_t - v_c`` over the final-layer input box, worst competitor.

    Uses ``|round(a/s) - a/s| <= 1/2`` on both logits, so
    ``v_t - v_c >= min_h (w_t - w_c) . h / s - 1 + (b_t - b_c)`` before the clamp.
    The clamp keeps strictness when ``v_c < upper`` and ``v_t > lower``; those
    side conditions are checked on the pre-clamp logit boxes and reported.
    This is a candidate for a future proof obligation, not a proof.
    """

    layer = network.layers[-1]
    if isinstance(layer, QuantizedConv2D) or layer.apply_relu:
        raise ValueError("Difference-row margin needs a final dense layer without ReLU")
    weights = np.asarray(layer.weights_int, dtype=np.int64)
    bias = np.asarray(layer.bias_int, dtype=np.int64)
    low, high = trace.final_input_low, trace.final_input_high
    difference = weights[target][None, :] - weights
    minimum = (np.where(difference >= 0, difference * low, difference * high)).sum(axis=1)
    scale = 1 << layer.input_fractional_bits
    lower_bound = minimum / scale - 1 + (bias[target] - bias)
    lower_bound[target] = np.inf
    # Unclamped logit boxes for the clamp side conditions.
    positive, negative = np.maximum(weights, 0), np.minimum(weights, 0)
    logit_low = _round_half_away(positive @ low + negative @ high, scale) + bias
    logit_high = _round_half_away(positive @ high + negative @ low, scale) + bias
    upper = (1 << (layer.spec.total_bits - 1)) - 1
    competitors = np.arange(len(bias)) != target
    clamp_safe = bool(logit_low[target] > -upper - 1
                      and np.all(logit_high[competitors] < upper))
    return float(lower_bound.min()), clamp_safe
