from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backends.fixed_point import LayerQuantizationSpec, QuantizedLayer
from synthesis.preqbmc import GPEncoding
from utils.fixed_point import quantize_int
from verification.c_templates import render_output_target_program
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.esbmc import ESBMCResult
from verification.esbmc_install import resolve_esbmc_executable
from verification.replay import LayerReplayFormat, replay_on_python


class OutputMarginHarnessTest(unittest.TestCase):
    def test_common_hidden_vector_can_verify_when_independent_extrema_fail(self) -> None:
        # For h in [0, 2]^3, independent bounds compare target low=1 with
        # competitor high=6 and fail. At every common h, however, the target is
        # exactly one deployed integer unit above class 1 and two above class 2.
        analytic_target_low = 1
        analytic_competitor_high = 6
        self.assertLessEqual(analytic_target_low, analytic_competitor_high)

        source = render_output_target_program(
            output_size=3,
            input_size=3,
            weights_c_int="{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}",
            biases_c_int="{1, 0, -1}",
            input_bounds_low_c_int="{0, 0, 0}",
            input_bounds_high_c_int="{2, 2, 2}",
            target_label=0,
            scale_factor=1,
            total_bits=8,
            input_scale_factor=1,
        )
        self.assertIn("input[k] = nondet_longlong()", source)
        self.assertIn("max_other < target", source)

        executable = resolve_esbmc_executable()
        if executable is None:
            self.skipTest("ESBMC is not installed")
        with tempfile.TemporaryDirectory() as temp_dir:
            harness = Path(temp_dir) / "output_margin_common_h.c"
            harness.write_text(source, encoding="utf-8")
            result = ESBMCRunner(
                ESBMCConfig(
                    executable=executable,
                    timeout_seconds=60,
                    memlimit="1g",
                    default_profile="paper-z3",
                )
            ).run_file(harness)
        self.assertEqual(result.status, "VERIFIED", result.stderr)

    def test_mnist_pilot_enables_exact_output_query_path(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "experiments" / "mnist_sound_v2_pilot.json").read_text(
                encoding="utf-8"
            )
        )
        [derived] = [
            run
            for run in config["runs"]
            if run["name"] == "mnist_1blk10_derived_blocks_z3_sample3_eps1"
        ]
        self.assertEqual(derived["harness_scope"], "layer")
        self.assertEqual(derived["esbmc_profile"], "paper-z3")
        self.assertTrue(derived["enabled"])

        source = (root / "src" / "synthesis" / "preqbmc.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("analytic_fail_pending_exact_query", source)
        self.assertIn("_record_exact_output_margin_result", source)

    def test_margin_cut_contains_every_bruteforce_deployed_hidden_vector(self) -> None:
        frac = 2
        scale = 1 << frac
        hidden_weights = np.asarray([[1.0, -0.5], [0.5, 1.0]], dtype=np.float64)
        hidden_biases = np.asarray([0.25, 0.0], dtype=np.float64)
        hidden = SimpleNamespace(
            layer_index=1,
            layer_size=2,
            layer_paras=[hidden_weights, hidden_biases],
            lb=np.asarray([-0.5, -0.75], dtype=np.float64),
            ub=np.asarray([1.0, 0.75], dtype=np.float64),
            realVal=np.asarray([0.25, 0.0], dtype=np.float64),
            frac_bit=frac,
            error_budget_int=np.asarray([1, 1], dtype=np.int64),
        )
        output = SimpleNamespace(
            layer_index=2,
            layer_size=3,
            layer_paras=[
                np.asarray(
                    [[0.75, -0.25], [0.10, 0.80], [-0.30, 0.40]],
                    dtype=np.float64,
                ),
                np.asarray([0.5, -0.25, -0.5], dtype=np.float64),
            ],
        )
        quantized_hidden = QuantizedLayer(
            weights_int=np.asarray(
                quantize_int(hidden_weights, 8, frac),
                dtype=np.int64,
            ),
            biases_int=np.asarray(
                quantize_int(hidden_biases, 8, frac),
                dtype=np.int64,
            ),
            spec=LayerQuantizationSpec(8, 5, frac),
            is_output_layer=False,
        )
        integer_inputs = list(product(range(-2, 3), range(-2, 3)))
        deployed_hidden = np.asarray(
            [
                replay_on_python(
                    values,
                    quantized_hidden,
                    LayerReplayFormat(
                        input_fractional_bits=frac,
                        total_bits=8,
                        apply_relu=True,
                    ),
                )
                for values in integer_inputs
            ],
            dtype=np.int64,
        )
        hidden.verified_activation_lb = deployed_hidden.min(axis=0) / scale
        hidden.verified_activation_ub = deployed_hidden.max(axis=0) / scale

        encoder = GPEncoding.__new__(GPEncoding)
        encoder.dense_layers = [hidden]
        encoder.output_layer = output
        encoder.x_low_real = np.asarray([-0.5, -0.5], dtype=np.float64)
        encoder.x_high_real = np.asarray([0.5, 0.5], dtype=np.float64)
        encoder.solver = "cbc"
        encoder.config = SimpleNamespace(gurobi_threads=1)
        encoder.margin_cuts = True
        encoder.error_budget_mode = "derived"
        encoder.property_spec = SimpleNamespace(target_label=0, valid_labels=None)
        encoder.targetCls = 0
        encoder.margin_cut_records = []

        with patch.object(
            encoder,
            "_formally_validate_relational_cut",
            side_effect=lambda record, **_: (
                record.update(
                    {
                        "formal_validation_status": "VERIFIED",
                        "soundness": "esbmc_exact_deployed_prefix_validated",
                    }
                )
                is None
            ),
        ):
            cuts = encoder._margin_cut_bounds(output, hidden, frac, 8)
        self.assertEqual(len(cuts), 2)
        for cut in cuts:
            direction = np.asarray(cut["direction_int"], dtype=np.int64)
            observed = deployed_hidden @ direction
            self.assertGreaterEqual(int(observed.min()), int(cut["cut_low_int"]))
            self.assertLessEqual(int(observed.max()), int(cut["cut_high_int"]))
            self.assertGreaterEqual(cut["total_widening_product_units"], 0)

    def test_inconclusive_output_box_can_conclude_via_single_e2e_fallback(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.e2e_fallback = True
        encoder.e2e_fallback_attempted = False
        encoder.composition_path = "layer_contracts"
        encoder.synthesis_final_status = "UNKNOWN"
        encoder.end_to_end_record = {}
        record = {
            "analytic_margin_ok": False,
            "margin_ok": False,
            "status": "PENDING_EXACT_QUERY",
        }
        box_result = ESBMCResult(
            status="FAILED",
            command=("esbmc",),
            stdout="",
            stderr="",
            return_code=1,
        )

        def verify_e2e(*_: object) -> bool:
            encoder.end_to_end_record = {
                "enabled": True,
                "status": "VERIFIED",
                "harness": "network_e2e.c",
                "resource_control": {"status": "VERIFIED"},
            }
            return True

        with patch.object(encoder, "_verify_network_end_to_end", side_effect=verify_e2e) as fallback:
            result = encoder._resolve_output_margin_result(
                record,
                box_result,
                total_bits=[8, 8],
                fractional_bits=[3, 3],
                integer_bits=[4, 4],
            )

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(record["output_margin"], "e2e_fallback_pass")
        self.assertEqual(record["composition_path"], "e2e_fallback")
        self.assertEqual(encoder.composition_path, "e2e_fallback")
        self.assertTrue(encoder.e2e_fallback_attempted)
        fallback.assert_called_once()

    def test_recovery_diagnostics_do_not_invent_values_after_timeout(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.harness_scope = "layer"
        encoder.error_budget_mode = "derived"
        encoder.margin_cuts = True
        encoder.output_margin_records = [
            {
                "margin_ok": True,
                "status": "VERIFIED",
                "solver_status": "TIMEOUT",
                "class_margins": [{"residual_margin_int": -1905}],
                "margin_cuts": [{"status": "OPTIMAL"}],
                "e2e_fallback": {"status": "VERIFIED"},
            }
        ]

        diagnostics = encoder.output_margin_summary()["recovery_diagnostics"]

        self.assertEqual(diagnostics["analytic_worst_residual_margin_int"], -1905)
        self.assertEqual(diagnostics["step_a_alone_status"], "NOT_RUN_WITHOUT_CUTS")
        self.assertEqual(diagnostics["step_b_status"], "TIMEOUT")
        self.assertIsNone(diagnostics["step_a_residual_recovery_int"])
        self.assertIsNone(diagnostics["step_b_additional_recovery_int"])
        self.assertEqual(diagnostics["reachable_recovery_lower_bound_int"], 1906)


if __name__ == "__main__":
    unittest.main()
