"""Explicit, size-limited NHWC Conv/ReLU/Flatten/Dense affine lowering.

This adapter preserves real affine functions, not IEEE summation order.
Integer convolution and lowering are equal when accumulator range checks pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from backends.fixed_point import FixedPointNetwork, LayerQuantizationSpec, QuantizedLayer
from utils.fixed_point import quantize_int, round_divide_half_away_from_zero


@dataclass(frozen=True)
class ConvGeometry:
    input_shape: tuple[int, int, int]
    kernel_shape: tuple[int, int, int, int]
    strides: tuple[int, int] = (1, 1)
    padding: str = "VALID"

    def __post_init__(self):
        if len(self.input_shape) != 3 or len(self.kernel_shape) != 4 or len(self.strides) != 2:
            raise ValueError("Expected HWC input, HWIO kernel and two strides")
        if any(type(v) is not int or v <= 0 for v in (*self.input_shape, *self.kernel_shape, *self.strides)):
            raise ValueError("Dimensions and strides must be positive integers")
        if self.input_shape[2] != self.kernel_shape[2] or self.padding not in {"SAME", "VALID"}:
            raise ValueError("Channel mismatch or unsupported padding")
        if min(self.output_shape) <= 0:
            raise ValueError("Kernel does not fit the VALID input")

    @property
    def output_shape(self):
        h, w, _ = self.input_shape
        kh, kw, _, channels = self.kernel_shape
        sh, sw = self.strides
        if self.padding == "SAME":
            return ((h + sh - 1) // sh, (w + sw - 1) // sw, channels)
        return ((h - kh) // sh + 1, (w - kw) // sw + 1, channels)

    @property
    def pad_before(self):
        if self.padding == "VALID":
            return (0, 0)
        return tuple(max(0, (self.output_shape[i] - 1) * self.strides[i]
                         + self.kernel_shape[i] - self.input_shape[i]) // 2 for i in range(2))


def lower_conv(kernel, bias, geometry: ConvGeometry, *, max_affine_entries=1_000_000):
    """Return output-by-input weights; padding is zero, never edge replication."""
    kernel, bias = np.asarray(kernel), np.asarray(bias)
    if kernel.shape != geometry.kernel_shape or bias.shape != (geometry.kernel_shape[-1],):
        raise ValueError("Conv parameter shape mismatch")
    ni, no = math.prod(geometry.input_shape), math.prod(geometry.output_shape)
    if ni * no > max_affine_entries:
        raise ValueError(f"Affine expansion needs {ni * no} entries; limit={max_affine_entries}")
    weights = np.zeros((no, ni), dtype=kernel.dtype)
    h, w, ci = geometry.input_shape
    kh, kw, _, co = geometry.kernel_shape
    ph, pw = geometry.pad_before
    for oy, ox, oc in np.ndindex(geometry.output_shape):
        row = np.ravel_multi_index((oy, ox, oc), geometry.output_shape)
        for ky, kx, ic in np.ndindex((kh, kw, ci)):
            iy, ix = oy * geometry.strides[0] + ky - ph, ox * geometry.strides[1] + kx - pw
            if 0 <= iy < h and 0 <= ix < w:
                weights[row, (iy * w + ix) * ci + ic] = kernel[ky, kx, ic, oc]
    return weights, np.tile(bias, no // co)


def direct_conv(image, kernel, geometry: ConvGeometry):
    """Independent sliding-window reference, with Python integer accumulation."""
    image, kernel = np.asarray(image), np.asarray(kernel)
    if image.shape != geometry.input_shape or kernel.shape != geometry.kernel_shape:
        raise ValueError("Direct convolution shape mismatch")
    integer = image.dtype.kind in "iuO" and kernel.dtype.kind in "iuO"
    dtype = object if integer else np.float64
    ph, pw = geometry.pad_before
    kh, kw, _, co = geometry.kernel_shape
    oh, ow, _ = geometry.output_shape
    padded = np.zeros((max(image.shape[0] + ph, (oh - 1) * geometry.strides[0] + kh),
                       max(image.shape[1] + pw, (ow - 1) * geometry.strides[1] + kw),
                       image.shape[2]), dtype=dtype)
    padded[ph:ph + image.shape[0], pw:pw + image.shape[1], :] = image
    result = np.zeros(geometry.output_shape, dtype=dtype)
    for oy, ox in np.ndindex((oh, ow)):
        y, x = oy * geometry.strides[0], ox * geometry.strides[1]
        patch = padded[y:y + kh, x:x + kw, :].reshape(-1)
        for oc in range(co):
            result[oy, ox, oc] = sum(a * b for a, b in zip(patch, kernel[..., oc].astype(dtype).reshape(-1)))
    return result


@dataclass(frozen=True)
class RestrictedCNN:
    geometry: ConvGeometry
    conv_kernel: np.ndarray
    conv_bias: np.ndarray
    dense_kernel: np.ndarray  # flattened NHWC by classes
    dense_bias: np.ndarray
    max_affine_entries: int = 1_000_000

    def __post_init__(self):
        for name in ("conv_kernel", "conv_bias", "dense_kernel", "dense_bias"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        if (np.shape(self.conv_kernel) != self.geometry.kernel_shape
                or np.shape(self.conv_bias) != (self.geometry.kernel_shape[-1],)
                or np.ndim(self.dense_bias) != 1 or len(self.dense_bias) < 2
                or np.shape(self.dense_kernel) != (math.prod(self.geometry.output_shape), len(self.dense_bias))):
            raise ValueError("Expected exactly Conv/ReLU/Flatten/Dense logits")
        if not all(np.all(np.isfinite(a)) for a in
                   (self.conv_kernel, self.conv_bias, self.dense_kernel, self.dense_bias)):
            raise ValueError("Nonfinite parameters")

    def affine_parameters(self):
        w, b = lower_conv(self.conv_kernel, self.conv_bias, self.geometry,
                          max_affine_entries=self.max_affine_entries)
        return [(w.T, b), (np.asarray(self.dense_kernel), np.asarray(self.dense_bias))]

    def as_deep_model(self):
        """Reuse the existing DeepPoly/MILP/search API; no new solver formulation."""
        from models.deep_model import DeepModel
        import tensorflow as tf
        model = DeepModel([math.prod(self.geometry.output_shape), len(self.dense_bias)], input_scale=1.0)
        model.build((None, math.prod(self.geometry.input_shape)))
        model(tf.zeros((1, math.prod(self.geometry.input_shape))))
        for layer, params in zip(model.dense_layers, self.affine_parameters()):
            layer.set_weights(params)
        return model

    def float_reference(self, normalized_image):
        hidden = np.maximum(direct_conv(normalized_image, self.conv_kernel, self.geometry) + self.conv_bias, 0)
        return hidden.reshape(-1) @ self.dense_kernel + self.dense_bias

    def quantized(self, specs, *, input_fractional_bits=8, input_total_bits=16):
        if len(specs) != 2 or not 2 <= input_total_bits <= 63 or not 0 <= input_fractional_bits < input_total_bits:
            raise ValueError("Two shared layer formats and a signed int64 input format are required")
        layers = []
        for index, ((kernel, bias), spec) in enumerate(zip(self.affine_parameters(), specs)):
            if spec.total_bits > 63:
                raise ValueError("Prototype stores values in signed int64")
            layers.append(QuantizedLayer(
                np.asarray(quantize_int(kernel.T, spec.total_bits, spec.fractional_bits), dtype=np.int64),
                np.asarray(quantize_int(bias, spec.total_bits, spec.fractional_bits), dtype=np.int64),
                spec, index == 1))
        return FixedPointNetwork(input_fractional_bits, input_total_bits, tuple(layers))

    def integer_reference(self, image_int, specs, input_fractional_bits=8):
        """Return direct-convolution hidden values and logits; bias after rescale."""
        hidden_spec, output_spec = specs
        kernel = quantize_int(self.conv_kernel, hidden_spec.total_bits, hidden_spec.fractional_bits)
        bias = quantize_int(self.conv_bias, hidden_spec.total_bits, hidden_spec.fractional_bits)
        acc = direct_conv(image_int, kernel, self.geometry)
        hidden = np.zeros(self.geometry.output_shape, dtype=np.int64)
        for idx in np.ndindex(hidden.shape):
            value = round_divide_half_away_from_zero(int(acc[idx]), 1 << input_fractional_bits) + int(bias[idx[-1]])
            hidden[idx] = max(0, min(hidden_spec.signed_range[1], max(hidden_spec.signed_range[0], value)))
        w = quantize_int(self.dense_kernel, output_spec.total_bits, output_spec.fractional_bits)
        b = quantize_int(self.dense_bias, output_spec.total_bits, output_spec.fractional_bits)
        outputs = []
        for oc in range(len(b)):
            acc = sum(int(x) * int(y) for x, y in zip(hidden.reshape(-1), w[:, oc]))
            value = round_divide_half_away_from_zero(acc, 1 << hidden_spec.fractional_bits) + int(b[oc])
            outputs.append(min(output_spec.signed_range[1], max(output_spec.signed_range[0], value)))
        return hidden.reshape(-1), np.asarray(outputs, dtype=np.int64)
