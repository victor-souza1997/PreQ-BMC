from __future__ import annotations

from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from verification.c_templates import render_network_end_to_end_program
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


class NarrowAccumulatorEncodingTest(unittest.TestCase):
    """Pin the accumulator encoding used when invariants.py proves int64 suffices.

    When ``accumulator_c_type`` is ``int64_t`` the emitted code declares a 64-bit
    accumulator but still evaluates each MAC through ``mac_i128`` and truncates.
    That is semantically identical to native int64 arithmetic under the
    ``all_prefixes_fit_i64`` obligation (invariants.py:52-68), because every
    prefix sum is proved to fit, so the truncation is a no-op.

    It looks like a missed optimisation and it is not. Replacing the body with
    native ``int64_t`` operations was measured on this harness (bitwuzla, --bv,
    5-5-3 network, six seeds) at a median 0.22x -- roughly 4.5x SLOWER -- and
    turned three of six instances that verified in ~5s into 10G memouts. The
    128-bit form bit-blasts to a query bitwuzla handles far better. Do not
    "fix" this without re-running that comparison.
    """

    @staticmethod
    def _layer(accumulator_c_type: str) -> dict[str, object]:
        return {
            "input_size": 2,
            "output_size": 2,
            "total_bits": 16,
            "fractional_bits": 4,
            "input_fractional_bits": 4,
            "weights_c_int": "{{1,2},{3,4}}",
            "biases_c_int": "{0,0}",
            "invariant_low_c_int": "{-1000,-1000}",
            "invariant_high_c_int": "{1000,1000}",
            "accumulator_c_type": accumulator_c_type,
        }

    def _forward_body(self, accumulator_c_type: str) -> str:
        program = render_network_end_to_end_program(
            input_size=2,
            input_bounds_low_c_int="{0,0}",
            input_bounds_high_c_int="{32,32}",
            layers=[self._layer(accumulator_c_type)],
            target_label=0,
        )
        body = program[program.index(f"{accumulator_c_type} acc = 0;"):]
        return body[: body.index("buffer_b[out_idx]")]

    def test_int64_accumulator_still_evaluates_through_int128(self) -> None:
        body = self._forward_body("int64_t")
        self.assertIn("mac_i128(", body)
        self.assertIn("(__int128)acc", body)

    def test_default_accumulator_is_int128(self) -> None:
        self.assertIn("mac_i128(", self._forward_body("__int128"))


if __name__ == "__main__":
    unittest.main()
