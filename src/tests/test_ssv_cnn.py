import ctypes
import itertools
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backends.c_qnn_generator import generate_c_qnn_source, compile_c_qnn_shared_library
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from models.restricted_conv import ConvGeometry, RestrictedCNN, direct_conv, lower_conv
from scripts.run_ssv_cnn_gate import tiny_model


class RestrictedConvTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow is optional")
    def test_keras_conv_reference_for_padding_and_strides(self):
        import tensorflow as tf
        rng = np.random.default_rng(23)
        for padding, stride in itertools.product(("SAME", "VALID"), (1, 2)):
            g = ConvGeometry((4, 5, 2), (2, 2, 2, 3), (stride, stride), padding)
            x = rng.normal(size=g.input_shape).astype(np.float32)
            kernel = rng.normal(size=g.kernel_shape).astype(np.float32)
            with tf.device("/CPU:0"):
                expected = tf.nn.conv2d(
                    x[None], kernel, strides=[1, stride, stride, 1], padding=padding,
                ).numpy()[0]
            np.testing.assert_allclose(direct_conv(x, kernel, g), expected, atol=2e-6, rtol=2e-6)

    def test_layout_padding_strides_float_and_integer(self):
        rng = np.random.default_rng(8)
        for padding, stride, kernel_size in itertools.product(("VALID", "SAME"), (1, 2), (2, 3)):
            g = ConvGeometry((4, 5, 2), (kernel_size, kernel_size, 2, 3), (stride, stride), padding)
            for integer in (True, False):
                image = rng.integers(-3, 4, g.input_shape)
                kernel = rng.integers(-3, 4, g.kernel_shape)
                if not integer:
                    image, kernel = image / 7, kernel / 5
                weights, bias = lower_conv(kernel, np.array([2, -1, 3]), g)
                expected = (direct_conv(image, kernel, g) + [2, -1, 3]).reshape(-1)
                actual = weights @ image.reshape(-1) + bias
                np.testing.assert_allclose(np.asarray(expected, dtype=float), actual, atol=1e-12)

    def test_reject_unsupported_shape_and_large_expansion(self):
        with self.assertRaises(ValueError):
            ConvGeometry((2, 2, 1), (1, 1, 2, 1))
        with self.assertRaises(ValueError):
            ConvGeometry((2, 2, 1), (1, 1, 1, 1), padding="reflect")
        with self.assertRaises(ValueError):
            cnn = tiny_model()
            lower_conv(cnn.conv_kernel, cnn.conv_bias, cnn.geometry, max_affine_entries=1)

    def test_bias_rounding_clamping_and_host_layer_parity(self):
        g = ConvGeometry((2, 2, 1), (1, 1, 1, 2))
        cnn = RestrictedCNN(g, np.array([[[[1.5, -1.5]]]]), np.array([0.5, -0.5]),
                            np.array([[1., -1.]] * 8), np.array([-0.5, 0.5]))
        specs = [LayerQuantizationSpec(4, 2, 1)] * 2
        network = cnn.quantized(specs, input_fractional_bits=1, input_total_bits=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qnn.c"
            path.write_text(generate_c_qnn_source(network))
            lib = ctypes.CDLL(str(compile_c_qnn_shared_library(path, path.with_suffix(".so"))))
            ptr = ctypes.POINTER(ctypes.c_int64)
            lib.qnn_forward_fixed.argtypes = [ptr, ptr]
            for vector in itertools.product((-8, -1, 0, 1, 7), repeat=4):
                x = np.array(vector, dtype=np.int64)
                hidden, expected = cnn.integer_reference(x.reshape(g.input_shape), specs, 1)
                self.assertTrue(np.all((0 <= hidden) & (hidden <= 7)))
                actual = np.zeros(2, dtype=np.int64)
                lib.qnn_forward_fixed(x.ctypes.data_as(ptr), actual.ctypes.data_as(ptr))
                np.testing.assert_array_equal(actual, expected)
        # Half-away ties and bias-after-rescale, checked without using the C result.
        hidden, _ = cnn.integer_reference(np.ones(g.input_shape, dtype=np.int64), specs, 1)
        np.testing.assert_array_equal(hidden, [3, 0] * 4)

    def test_float_reference_and_lowered_reference(self):
        cnn = tiny_model()
        x = np.array([0.0, 0.25, 0.5, 1.0]).reshape(cnn.geometry.input_shape)
        (w, b), (wo, bo) = cnn.affine_parameters()
        np.testing.assert_allclose(cnn.float_reference(x), np.maximum(x.reshape(-1) @ w + b, 0) @ wo + bo)


class ByteEncoderTest(unittest.TestCase):
    def test_byte_box_coverage_and_resize(self):
        center = np.array([0, 127, 128, 255], dtype=np.uint8).reshape(2, 2, 1)
        for frac in (0, 1, 7, 8, 12):
            encoder = ByteImageEncoder((3, 3, 1), frac, 16)
            for epsilon in ("0", "0.499999999999", "1", "1.5"):
                low, high = encoder.box(center, epsilon)
                for differences in itertools.product((-1, 0, 1), repeat=4):
                    if max(abs(v) for v in differences) > float(epsilon):
                        continue
                    image = np.clip(center.astype(int).reshape(-1) + differences, 0, 255).astype(np.uint8).reshape(center.shape)
                    encoded = encoder.encode(image)
                    self.assertTrue(np.all(low <= encoded) and np.all(encoded <= high))

    def test_half_ulp_and_exact_cpp_encoder_parity(self):
        from verification.arith_kernel import render_arith_kernel
        with tempfile.TemporaryDirectory() as directory:
            for frac in (0, 1, 7, 8, 12):
                enc = ByteImageEncoder((2, 3, 1), frac, 10 if frac < 8 else 16)
                path = Path(directory) / f"encoder{frac}.c"
                path.write_text("#include <stdint.h>\n" + render_arith_kernel() + enc.render_c())
                lib = ctypes.CDLL(str(compile_c_qnn_shared_library(path, path.with_suffix(".so"))))
                lib.qnn_encode_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
                for byte in range(256):
                    raw = np.full((3, 4, 1), byte, dtype=np.uint8)
                    raw[0, 0, 0] = 255 - byte
                    actual = np.zeros(6, dtype=np.int64)
                    self.assertEqual(lib.qnn_encode_bytes(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 3, 4,
                                     actual.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))), 0)
                    np.testing.assert_array_equal(actual, enc.encode(raw))
        enc = ByteImageEncoder((1, 1, 1), 0, 8)
        self.assertEqual(enc.encode(np.array([128], dtype=np.uint8).reshape(1, 1, 1))[0], 1)
        self.assertEqual(enc.encode(np.array([127], dtype=np.uint8).reshape(1, 1, 1))[0], 0)

    def test_bad_input_rejected(self):
        encoder = ByteImageEncoder((2, 2, 1))
        with self.assertRaises(ValueError):
            encoder.encode(np.zeros((2, 2, 1)))
        with self.assertRaises(ValueError):
            encoder.box(np.zeros((2, 2, 1), dtype=np.uint8), -1)


if __name__ == "__main__":
    unittest.main()
