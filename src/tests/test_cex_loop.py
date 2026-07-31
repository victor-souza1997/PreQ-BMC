from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from synthesis.preqbmc import GPEncoding
from verification.esbmc import parse_counterexample_trace
from verification.replay import LayerReplayFormat, replay_on_python


class CounterexampleLoopTest(unittest.TestCase):
    def test_parses_real_esbmc_trace_fixture(self) -> None:
        fixture = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "esbmc_counterexample_trace.txt"
        )
        inputs, neuron, preclamp = parse_counterexample_trace(
            fixture.read_text(encoding="utf-8")
        )

        self.assertEqual(inputs, [4, 288, 1171, 0])
        self.assertIsNone(neuron)
        self.assertEqual(preclamp, -1498)

    def test_python_layer_replay_matches_kernel_arithmetic(self) -> None:
        layer = SimpleNamespace(
            weights_int=np.asarray([[3, -2], [-1, 4]], dtype=np.int64),
            biases_int=np.asarray([1, -3], dtype=np.int64),
        )

        outputs = replay_on_python(
            [5, -2],
            layer,
            LayerReplayFormat(
                input_fractional_bits=2,
                total_bits=6,
                apply_relu=True,
            ),
        )

        # RHAZ((3*5 + -2*-2)/4)+1 = 6; second output is negative then ReLU.
        self.assertEqual(outputs.tolist(), [6, 0])

    def test_confirmed_counterexample_filters_candidate_without_esbmc(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.cex_feedback = "filter"
        encoder.cex_pool = {
            0: [
                {
                    "inputs_int": [1, -2],
                    "input_fractional_bits": 3,
                    "replay_confirmed": True,
                }
            ]
        }
        encoder.cex_filtered_counts = {}
        encoder.esbmc_call_records = []
        encoder.dense_layers = []
        current = SimpleNamespace(layer_index=1)

        with patch.object(
            encoder,
            "_candidate_replay_violates",
            return_value=True,
        ) as replay:
            record = encoder._candidate_filtered_by_counterexample(
                cur_layer=current,
                in_layer=SimpleNamespace(),
                qu_w_int=np.asarray([[1, 1]], dtype=np.int64),
                qu_b_int=np.asarray([0], dtype=np.int64),
                frac_bit=4,
                all_bit=8,
                layer_index=0,
            )

        self.assertIsNotNone(record)
        self.assertEqual(record["status"], "CEX_FILTERED")
        self.assertEqual(encoder.cex_filtered_counts[0], 1)
        self.assertEqual(len(encoder.esbmc_call_records), 1)
        replay.assert_called_once()

    def test_counterexample_is_reencoded_for_candidate_fractional_width(self) -> None:
        upscaled = GPEncoding._rescale_integer_vector(
            [1, -2, 3],
            from_fractional_bits=2,
            to_fractional_bits=4,
        )
        downscaled = GPEncoding._rescale_integer_vector(
            [6, -6, 2, -2],
            from_fractional_bits=3,
            to_fractional_bits=1,
        )

        self.assertEqual(upscaled.tolist(), [4, -8, 12])
        self.assertEqual(downscaled.tolist(), [2, -2, 1, -1])


if __name__ == "__main__":
    unittest.main()
