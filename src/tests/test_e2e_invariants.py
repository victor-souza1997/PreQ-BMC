from __future__ import annotations

from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from verification.invariants import propagate_exact_intervals
from verification.replay import LayerReplayFormat, replay_on_python


class EndToEndInvariantTest(unittest.TestCase):
    def test_exact_interval_contains_bruteforce_layer_outputs(self) -> None:
        layer = SimpleNamespace(
            weights_int=np.asarray([[3, -2], [-4, 1]], dtype=np.int64),
            biases_int=np.asarray([1, -1], dtype=np.int64),
        )
        fmt = LayerQuantizationSpec(
            total_bits=6,
            integer_bits=2,
            fractional_bits=3,
        )
        low = np.asarray([-2, 0], dtype=np.int64)
        high = np.asarray([1, 2], dtype=np.int64)

        [(bound_low, bound_high)] = propagate_exact_intervals(
            [layer],
            [fmt],
            low,
            high,
        )

        observed = []
        for values in product(
            range(int(low[0]), int(high[0]) + 1),
            range(int(low[1]), int(high[1]) + 1),
        ):
            observed.append(
                replay_on_python(
                    values,
                    layer,
                    LayerReplayFormat(
                        input_fractional_bits=fmt.fractional_bits,
                        total_bits=fmt.total_bits,
                        apply_relu=False,
                    ),
                )
            )
        outputs = np.asarray(observed, dtype=np.int64)
        self.assertTrue(np.all(outputs >= bound_low))
        self.assertTrue(np.all(outputs <= bound_high))
        self.assertEqual(bound_low.tolist(), outputs.min(axis=0).tolist())
        self.assertEqual(bound_high.tolist(), outputs.max(axis=0).tolist())


if __name__ == "__main__":
    unittest.main()
