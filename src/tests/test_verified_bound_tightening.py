from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from scripts import run_article_experiments
from scripts.aggregate_article_results import _verified_bound_tightening_rows
from synthesis.preqbmc import GPEncoding
from verification.c_templates import render_hidden_affine_bounds_block_program
from verification.esbmc import ESBMCConfig
from verification.esbmc import ESBMCResult
from verification.invariants import (
    FixedPointArithmeticRangeError,
    exact_layer_interval,
)


class VerifiedBoundTighteningTest(unittest.TestCase):
    def test_iris_seeds_pilot_is_matched_and_excludes_mnist(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = (
            repo_root / "experiments" / "iris_seeds_verified_bounds_pilot.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        args = run_article_experiments.build_parser().parse_args(
            ["--config", str(config_path), "--dry-run"]
        )
        runs = run_article_experiments._expand_runs(config, args)

        self.assertEqual(len(runs), 12)
        self.assertEqual({run["dataset"] for run in runs}, {"iris", "seeds"})
        self.assertEqual({run["bit_lb"] for run in runs}, {8})
        self.assertEqual({run["bit_ub"] for run in runs}, {8})
        self.assertEqual(
            {run["tighten_verified_bounds"] for run in runs},
            {False, True},
        )
        self.assertTrue(all(run["harness_scope"] == "layer" for run in runs))
        self.assertTrue(all(run["e2e_fallback"] == "off" for run in runs))

    def test_affine_box_reports_kernel_arithmetic_safety(self) -> None:
        result = exact_layer_interval(
            SimpleNamespace(
                weights_int=np.asarray([[3, -2]], dtype=np.int64),
                biases_int=np.asarray([1], dtype=np.int64),
            ),
            input_low=np.asarray([-2, 0], dtype=np.int64),
            input_high=np.asarray([1, 2], dtype=np.int64),
            input_fractional_bits=1,
            total_bits=8,
            apply_relu=False,
        )

        self.assertEqual(result.output_low.tolist(), [-4])
        self.assertEqual(result.output_high.tolist(), [3])
        self.assertEqual(result.arithmetic_safety["status"], "VERIFIED")
        self.assertEqual(result.arithmetic_safety["accumulator_type"], "__int128")

    def test_i128_prefix_overflow_is_rejected(self) -> None:
        maximum = (1 << 63) - 1
        with self.assertRaises(FixedPointArithmeticRangeError) as raised:
            exact_layer_interval(
                SimpleNamespace(
                    weights_int=np.asarray(
                        [[maximum, maximum, maximum]], dtype=object
                    ),
                    biases_int=np.asarray([0], dtype=object),
                ),
                input_low=np.asarray([maximum] * 3, dtype=object),
                input_high=np.asarray([maximum] * 3, dtype=object),
                input_fractional_bits=0,
                total_bits=63,
                apply_relu=False,
            )

        self.assertEqual(raised.exception.details["status"], "UNSAFE")
        self.assertIn("prefix sum", raised.exception.details["reason"])

    def test_unsafe_candidate_stops_before_esbmc(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.tighten_verified_bounds = True
        encoder.arithmetic_safety_records = []
        encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
        maximum = (1 << 63) - 1
        bounds = np.asarray([maximum] * 3, dtype=object)

        with patch.object(
            encoder,
            "_assumption_box_int",
            return_value=(bounds, bounds, 0),
        ):
            result = encoder.verify_layer_with_esbmc(
                cur_layer=SimpleNamespace(layer_index=1),
                in_layer=SimpleNamespace(layer_size=3),
                qu_w_int=np.asarray([[maximum] * 3], dtype=object),
                qu_b_int=np.asarray([0], dtype=object),
                frac_bit=0,
                all_bit=63,
                layer_index=0,
            )

        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(
            result.resource_control["mode"],
            "pre_esbmc_arithmetic_safety_gate",
        )
        self.assertEqual(encoder.arithmetic_safety_records[-1]["status"], "UNSAFE")

    def test_verified_intersection_is_propagated_with_shared_qif(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.tighten_verified_bounds = True
        encoder.error_budget_mode = "derived"
        encoder.unsound_contract_tolerance = False
        encoder.propagate_contract_tolerance = False
        encoder.enforce_contract_chaining = True
        encoder.chaining_records = []
        current = SimpleNamespace(
            layer_index=1,
            clipped_lb=np.asarray([0.0]),
            clipped_ub=np.asarray([5.0]),
        )
        previous = SimpleNamespace()

        with (
            patch.object(
                encoder,
                "_candidate_contract_target_bounds_int",
                return_value=(
                    np.asarray([-5]),
                    np.asarray([5]),
                    np.asarray([0]),
                    True,
                ),
            ),
            patch.object(
                encoder,
                "_assumption_box_int",
                return_value=(np.asarray([-1]), np.asarray([1]), 0),
            ),
        ):
            record = encoder._record_hidden_chaining_check(
                cur_layer=current,
                layer_index=0,
                all_bit=4,
                frac_bit=0,
                in_layer=previous,
                weights_int=np.asarray([[2]], dtype=np.int64),
                biases_int=np.asarray([0], dtype=np.int64),
            )

        tightening = record["verified_bound_tightening"]
        self.assertEqual(tightening["propagated_low_int"], [0])
        self.assertEqual(tightening["propagated_high_int"], [2])
        self.assertEqual(tightening["arithmetic_safety"]["status"], "VERIFIED")
        self.assertEqual(current.verified_activation_lb_int.tolist(), [0])
        self.assertEqual(current.verified_activation_ub_int.tolist(), [2])

    def test_block_harness_keeps_contract_and_tightening_assertions(self) -> None:
        source = render_hidden_affine_bounds_block_program(
            block_size=1,
            input_size=1,
            weights_c_int="{{2}}",
            biases_c_int="{0}",
            preimage_low_c_int="{-5}",
            preimage_high_c_int="{5}",
            input_bounds_low_c_int="{-1}",
            input_bounds_high_c_int="{1}",
            scale_factor=1,
            total_bits=4,
            reachable_low_c_int="{-2}",
            reachable_high_c_int="{2}",
        )

        self.assertIn("affine bounds not within tolerated preimage", source)
        self.assertIn("affine output outside proposed reachable bounds", source)
        self.assertIn("long long reachable_low[LAYER_SIZE] = {-2};", source)

    def test_fixed_qif_reverification_uses_same_tightened_esbmc_path(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.dense_layers = [
            SimpleNamespace(
                layer_index=1,
                layer_paras=[
                    np.asarray([[0.5]], dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                ],
            )
        ]
        encoder.output_layer = SimpleNamespace(
            layer_index=2,
            layer_paras=[
                np.asarray([[0.5]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
            ],
        )
        encoder.input_layer = SimpleNamespace(layer_index=0)
        encoder.deepPolyNets_DNN = SimpleNamespace(load_dnn=Mock())
        encoder.deep_model = object()
        encoder.error_budget_mode = "derived"
        encoder.tighten_verified_bounds = True
        encoder.enforce_contract_chaining = True
        verified = ESBMCResult(
            status="VERIFIED",
            command=(),
            stdout="",
            stderr="",
            return_code=0,
        )
        encoder.verify_layer_with_esbmc = Mock(return_value=verified)
        encoder._record_output_margin_check = Mock(
            return_value={"analytic_margin_ok": True, "status": "VERIFIED"}
        )
        encoder._resolve_output_margin_result = Mock(return_value=verified)
        encoder._record_hidden_chaining_check = Mock(
            return_value={"chaining_ok": True}
        )
        encoder._run_vacuity_sentinel = Mock(
            return_value={"status": "NONVACUOUS"}
        )
        encoder.update_quantized_weights_affine = Mock()

        accepted, records = encoder.verify_exported_quantization_with_esbmc(
            total_bits=[5, 5],
            fractional_bits=[2, 2],
            integer_bits=[2, 2],
            formal_saturation_check=False,
        )

        self.assertTrue(accepted)
        self.assertEqual(len(records), 2)
        # Tightening disables the analytic output bypass, so both hidden and
        # output candidates follow the same formal harness path.
        self.assertEqual(encoder.verify_layer_with_esbmc.call_count, 2)
        chaining_call = encoder._record_hidden_chaining_check.call_args.kwargs
        self.assertIn("biases_int", chaining_call)
        self.assertEqual(chaining_call["all_bit"], 5)
        self.assertEqual(chaining_call["frac_bit"], 2)

    def test_per_layer_aggregate_rows_keep_proof_endpoints(self) -> None:
        rows = _verified_bound_tightening_rows(
            [
                {
                    "run_name": "iris_treatment",
                    "dataset": "iris",
                    "arch": "2blk_10_10",
                    "sample_id": 1,
                    "input_epsilon": 0.01,
                    "method": "formal_only",
                    "mode": "derived_layer_contract_verified_bounds",
                    "composition_path": "layer_exact_output",
                    "verified_bound_tightening_layers": [
                        {
                            "layer_index": 0,
                            "Q": 13,
                            "I": 4,
                            "F": 8,
                            "status": "VERIFIED",
                            "contract_low_int": [0],
                            "contract_high_int": [100],
                            "reachable_affine_low_int": [-2],
                            "reachable_affine_high_int": [12],
                            "propagated_low_int": [0],
                            "propagated_high_int": [12],
                            "contract_total_width_int": 100,
                            "propagated_total_width_int": 12,
                            "arithmetic_safety": {
                                "status": "VERIFIED",
                                "max_abs_accumulator": "24",
                            },
                        }
                    ],
                }
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["width_reduction_int"], 88)
        self.assertEqual(rows[0]["width_reduction_fraction"], 0.88)
        self.assertEqual(rows[0]["arithmetic_safety_status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
