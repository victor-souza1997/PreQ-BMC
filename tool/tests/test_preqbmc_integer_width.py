from __future__ import annotations

import unittest

import numpy as np

from synthesis.preqbmc import GPEncoding


class PreqBMCIntegerWidthTest(unittest.TestCase):
    def test_required_internal_integer_bits_cover_signed_real_range(self) -> None:
        self.assertEqual(GPEncoding._required_internal_integer_bits_for_interval(-2.0, 1.875), 2)
        self.assertEqual(GPEncoding._required_internal_integer_bits_for_interval(-2.0, 2.0), 3)
        self.assertEqual(GPEncoding._required_internal_integer_bits_for_interval(4.13, 6.56), 4)
        self.assertEqual(GPEncoding._required_internal_integer_bits_for_interval(-8.0, 7.99), 4)

    def test_layer_contract_width_uses_relaxed_preimage_bounds(self) -> None:
        encoder = object.__new__(GPEncoding)
        layer = type("Layer", (), {})()
        layer.lb = np.asarray([-0.5, 0.25], dtype=np.float32)
        layer.ub = np.asarray([0.5, 1.0], dtype=np.float32)
        layer.relaxed_lb = np.asarray([4.13, -0.25], dtype=np.float32)
        layer.relaxed_ub = np.asarray([6.56, 0.75], dtype=np.float32)

        self.assertEqual(encoder._required_internal_integer_bits_for_layer_contract(layer), 4)


if __name__ == "__main__":
    unittest.main()
