"""Exact byte-crop encoder shared with generated C, independent of float32 boxes."""
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np


@dataclass(frozen=True)
class ByteImageEncoder:
    output_shape: tuple[int, int, int]
    fractional_bits: int = 8
    total_bits: int = 16

    def __post_init__(self):
        if (len(self.output_shape) != 3 or any(type(n) is not int or not 0 < n <= 4096 for n in self.output_shape)
                or self.output_shape[-1] > 4 or math.prod(self.output_shape) > 1_000_000
                or not 2 <= self.total_bits <= 63 or not 0 <= self.fractional_bits < self.total_bits):
            raise ValueError("Invalid image or signed fixed-point dimensions")

    def resize(self, image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != self.output_shape[2] or min(image.shape) <= 0 or max(image.shape[:2]) > 4096:
            raise ValueError("Input must be a nonempty HWC uint8 crop with matching channels")
        h, w, _ = self.output_shape
        rows = np.arange(h) * image.shape[0] // h
        cols = np.arange(w) * image.shape[1] // w
        return image[rows[:, None], cols[None, :], :]

    def encode(self, image):
        resized = self.resize(image)
        high = (1 << (self.total_bits - 1)) - 1
        return np.array([min(high, (int(v) * (1 << self.fractional_bits) + 128) // 256)
                         for v in resized.flat], dtype=np.int64)

    def box(self, image, epsilon):
        """Exact image of the byte-domain box, including all resize repetitions."""
        eps = Fraction(str(epsilon))
        if eps < 0:
            raise ValueError("epsilon must be nonnegative and finite")
        resized = self.resize(image)
        low = np.array([max(0, math.ceil(Fraction(int(v)) - eps)) for v in resized.flat], dtype=np.uint8).reshape(resized.shape)
        high = np.array([min(255, math.floor(Fraction(int(v)) + eps)) for v in resized.flat], dtype=np.uint8).reshape(resized.shape)
        return self.encode(low), self.encode(high)

    def render_c(self):
        h, w, c = self.output_shape
        return f"""
/* uint8 crop, top-left nearest-neighbor, normalization /256, half-away rounding.
   Decoder and RGB conversion are outside this function's certificate. */
int qnn_encoder_channels(void) {{ return {c}; }}
int qnn_encoder_size(void) {{ return {h * w * c}; }}
int qnn_encode_bytes(const uint8_t *image, int height, int width, int64_t *out) {{
    if (!image || !out || height <= 0 || width <= 0 || height > 4096 || width > 4096) return -1;
    for (int y = 0; y < {h}; ++y)
        for (int x = 0; x < {w}; ++x)
            for (int c = 0; c < {c}; ++c) {{
                int src = ((y * height / {h}) * width + x * width / {w}) * {c} + c;
                __int128 v = div_round_half_away_from_zero_i128(
                    (__int128)image[src] * (((__int128)1) << {self.fractional_bits}), 256);
                out[(y * {w} + x) * {c} + c] = (int64_t)clamp_to_signed_range_i128(v, {self.total_bits});
            }}
    return 0;
}}
"""
