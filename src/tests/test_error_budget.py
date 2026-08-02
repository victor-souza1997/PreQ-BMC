from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from synthesis.preqbmc import GPEncoding
from verification.c_templates import (
    render_hidden_affine_bounds_program,
    render_prefix_direction_cut_validation_program,
)
from verification.esbmc import ESBMCConfig, ESBMCResult


def _round_half_away(value: Fraction) -> int:
    if value >= 0:
        return (value.numerator * 2 + value.denominator) // (2 * value.denominator)
    positive = -value
    return -(
        (positive.numerator * 2 + positive.denominator)
        // (2 * positive.denominator)
    )


class DerivedErrorBudgetTest(unittest.TestCase):
    def test_derived_budget_bounds_exhaustive_integer_box(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        frac_in = 3
        frac_out = 3
        scale_in = 1 << frac_in
        scale_out = 1 << frac_out

        rng = np.random.default_rng(17)
        weight_numerators = rng.integers(-11, 12, size=(3, 3))
        bias_numerators = rng.integers(-7, 8, size=3)
        real_weights = [
            [Fraction(int(value), 16) for value in row]
            for row in weight_numerators
        ]
        real_biases = [
            Fraction(int(value), 32)
            for value in bias_numerators
        ]
        weights_int = np.asarray(
            [
                [_round_half_away(weight * scale_out) for weight in row]
                for row in real_weights
            ],
            dtype=np.int64,
        )
        biases_int = [
            _round_half_away(bias * scale_out)
            for bias in real_biases
        ]
        assumed_low = np.asarray([-2, 0, 1], dtype=np.int64)
        assumed_high = np.asarray([1, 2, 3], dtype=np.int64)

        budget = encoder._derived_error_budget_int(
            cur_layer=SimpleNamespace(),
            weights_int=weights_int,
            assumed_lo_int=assumed_low,
            assumed_hi_int=assumed_high,
            frac_in=frac_in,
            delta_in_int=0,
        )

        maximum_error = [Fraction(0) for _ in range(3)]
        ranges = [
            range(int(low), int(high) + 1)
            for low, high in zip(assumed_low, assumed_high)
        ]
        for integer_input in product(*ranges):
            for neuron, (weight_row, bias) in enumerate(
                zip(real_weights, real_biases)
            ):
                accumulator = sum(
                    int(weight) * int(value)
                    for weight, value in zip(weights_int[neuron], integer_input)
                )
                implemented = _round_half_away(
                    Fraction(accumulator, scale_in)
                ) + biases_int[neuron]
                real_output_ulps = sum(
                    weight * Fraction(value, scale_in) * scale_out
                    for weight, value in zip(weight_row, integer_input)
                ) + bias * scale_out
                error = abs(Fraction(implemented) - real_output_ulps)
                maximum_error[neuron] = max(maximum_error[neuron], error)

        for neuron, error in enumerate(maximum_error):
            self.assertLessEqual(
                error,
                int(budget[neuron]),
                msg=f"neuron={neuron}, error={error}, budget={budget.tolist()}",
            )
        self.assertTrue(np.all(budget >= 1))

    def test_inherited_budget_is_amplified_with_integer_l1_gain(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        budget = encoder._derived_error_budget_int(
            cur_layer=SimpleNamespace(),
            weights_int=np.asarray([[8, -4]], dtype=np.int64),
            assumed_lo_int=np.asarray([0, 0], dtype=np.int64),
            assumed_hi_int=np.asarray([0, 0], dtype=np.int64),
            frac_in=2,
            delta_in_int=np.asarray([2, 1], dtype=np.int64),
        )

        # The no-real-weight fallback uses |W_real*S_out| <= |W_int| + 1/2.
        self.assertEqual(budget.tolist(), [7])

    def test_invalid_integer_assumption_box_is_vacuous_without_esbmc(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.error_budget_mode = "derived"
        encoder.x_low_real = np.asarray([2.0], dtype=np.float64)
        encoder.x_high_real = np.asarray([1.0], dtype=np.float64)
        encoder.vacuity_records = []
        encoder.esbmc_call_records = []
        encoder.synthesis_final_status = "UNKNOWN"
        encoder.config = SimpleNamespace(esbmc=ESBMCConfig())

        result = encoder.verify_layer_with_esbmc(
            cur_layer=SimpleNamespace(layer_index=1),
            in_layer=SimpleNamespace(layer_size=1),
            qu_w_int=np.asarray([[1]], dtype=np.int64),
            qu_b_int=np.asarray([0], dtype=np.int64),
            frac_bit=0,
            all_bit=4,
            layer_index=0,
        )

        self.assertEqual(result.status, "VACUOUS")
        self.assertEqual(encoder.synthesis_final_status, "VACUOUS")
        self.assertEqual(encoder.vacuity_records[-1]["assumption_box_cardinality"], "0")

    def test_derived_tolerance_is_emitted_per_neuron(self) -> None:
        source = render_hidden_affine_bounds_program(
            output_size=2,
            input_size=1,
            weights_c_int="{{4}, {-4}}",
            biases_c_int="{0, 0}",
            preimage_low_c_int="{-2, -3}",
            preimage_high_c_int="{2, 3}",
            input_bounds_low_c_int="{-1}",
            input_bounds_high_c_int="{1}",
            scale_factor=8,
            total_bits=8,
            input_scale_factor=4,
            contract_tolerance_c_int="{2, 5}",
        )

        self.assertIn("#define INPUT_SCALE_FACTOR 4LL", source)
        self.assertIn("long long contract_tolerance[LAYER_SIZE] = {2, 5};", source)
        self.assertIn(
            "const __int128 preimage_tolerance = (__int128)contract_tolerance[i];",
            source,
        )
        self.assertIn(
            "div_round_half_away_from_zero_i128(s_lb, (__int128)INPUT_SCALE_FACTOR)",
            source,
        )

    def test_output_margin_failure_requires_exact_query(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.property_spec = SimpleNamespace(target_label=0)
        encoder.targetCls = 0
        encoder.output_margin_records = []
        encoder.synthesis_final_status = "UNKNOWN"
        current = SimpleNamespace(
            layer_index=2,
            layer_size=2,
            lb=np.asarray([1.25, 0.0], dtype=np.float64),
            ub=np.asarray([1.5, 1.0], dtype=np.float64),
            layer_paras=[
                np.asarray([[1.0], [1.0]], dtype=np.float64),
                np.asarray([0.0, 0.0], dtype=np.float64),
            ],
        )
        previous = SimpleNamespace()

        with patch.object(
            encoder,
            "_candidate_error_budget_int",
            return_value=(
                np.asarray([2, 1], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                3,
            ),
        ):
            record = encoder._record_output_margin_check(
                cur_layer=current,
                in_layer=previous,
                weights_int=np.asarray([[1], [1]], dtype=np.int64),
                layer_index=1,
                all_bit=8,
                frac_bit=3,
            )

        self.assertFalse(record["margin_ok"])
        self.assertEqual(record["status"], "PENDING_EXACT_QUERY")
        self.assertEqual(record["output_margin"], "analytic_fail_pending_exact_query")
        self.assertEqual(encoder.synthesis_final_status, "UNKNOWN")

    def test_sentinel_failure_proves_box_is_nonvacuous(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.vacuity_check = True
        encoder.vacuity_records = []
        encoder.esbmc_call_records = []
        encoder.synthesis_final_status = "UNKNOWN"
        encoder._stats = {"esbmc_calls": 0.0}
        encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
        encoder.esbmc_runner = SimpleNamespace(
            run_file=lambda _: ESBMCResult(
                status="FAILED",
                command=("esbmc",),
                stdout="",
                stderr="",
                return_code=1,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            encoder.output_dir = Path(temp_dir)
            with patch.object(
                encoder,
                "_assumption_box_int",
                return_value=(
                    np.asarray([0, -1], dtype=np.int64),
                    np.asarray([1, 1], dtype=np.int64),
                    0,
                ),
            ):
                record = encoder._run_vacuity_sentinel(
                    cur_layer=SimpleNamespace(),
                    in_layer=SimpleNamespace(layer_size=2),
                    layer_index=0,
                    all_bit=8,
                    frac_bit=3,
                )

        self.assertEqual(record["status"], "NONVACUOUS")
        self.assertEqual(record["sentinel_status"], "FAILED")
        self.assertEqual(record["assumption_box_cardinality"], "6")

    def test_last_hidden_preimage_deflation_expands_back_inside_preimage(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        hidden = SimpleNamespace(
            layer_index=1,
            int_bit=4,
            preimage_source="milp_preimage",
        )
        encoder.dense_layers = [hidden]
        encoder.error_budget_mode = "derived"
        encoder.preimage_deflation_records = []
        pre_low = np.asarray([-20, 5], dtype=np.int64)
        pre_high = np.asarray([30, 25], dtype=np.int64)
        budget = np.asarray([3, 4], dtype=np.int64)
        with (
            patch.object(
                encoder,
                "_layer_preimage_bounds_int",
                return_value=(pre_low, pre_high),
            ),
            patch.object(
                encoder,
                "_candidate_error_budget_int",
                return_value=(budget, np.asarray([0]), np.asarray([0]), 3),
            ),
        ):
            low, high, emitted_budget, valid = (
                encoder._candidate_contract_target_bounds_int(
                    hidden,
                    SimpleNamespace(),
                    np.asarray([[1]], dtype=np.int64),
                    3,
                    record=True,
                )
            )

        self.assertTrue(valid)
        np.testing.assert_array_equal(low, pre_low + budget)
        np.testing.assert_array_equal(high, pre_high - budget)
        np.testing.assert_array_equal(low - emitted_budget, pre_low)
        np.testing.assert_array_equal(high + emitted_budget, pre_high)
        self.assertEqual(encoder.preimage_deflation_records[-1]["status"], "DEFLATED")

    def test_forward_fallback_is_not_deflated_as_property_preimage(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        hidden = SimpleNamespace(
            layer_index=1,
            int_bit=4,
            preimage_source="deeppoly_forward_FALLBACK",
        )
        encoder.dense_layers = [hidden]
        encoder.error_budget_mode = "derived"
        encoder.preimage_deflation_records = []
        pre_low = np.asarray([-2, 4], dtype=np.int64)
        pre_high = np.asarray([3, 7], dtype=np.int64)
        budget = np.asarray([2, 2], dtype=np.int64)
        with (
            patch.object(
                encoder,
                "_layer_preimage_bounds_int",
                return_value=(pre_low, pre_high),
            ),
            patch.object(
                encoder,
                "_candidate_error_budget_int",
                return_value=(budget, np.asarray([0]), np.asarray([0]), 3),
            ),
        ):
            low, high, emitted_budget, valid = (
                encoder._candidate_contract_target_bounds_int(
                    hidden,
                    SimpleNamespace(),
                    np.asarray([[1]], dtype=np.int64),
                    3,
                    record=True,
                )
            )

        self.assertFalse(valid)
        np.testing.assert_array_equal(low, pre_low)
        np.testing.assert_array_equal(high, pre_high)
        np.testing.assert_array_equal(emitted_budget, budget)
        record = encoder.preimage_deflation_records[-1]
        self.assertEqual(record["status"], "PREIMAGE_UNAVAILABLE")
        self.assertFalse(record["property_preimage_available"])
        self.assertEqual(record["collapsed_neurons"], [])

    def test_forward_fallback_stops_before_esbmc(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        hidden = SimpleNamespace(
            layer_index=1,
            layer_size=1,
            int_bit=4,
            preimage_source="deeppoly_forward_FALLBACK",
        )
        encoder.dense_layers = [hidden]
        encoder.error_budget_mode = "derived"
        encoder.preimage_deflation_records = []
        encoder.esbmc_layer_block_size = 0
        encoder.synthesis_final_status = "UNKNOWN"
        encoder.x_low_real = np.asarray([0.0], dtype=np.float64)
        encoder.x_high_real = np.asarray([1.0], dtype=np.float64)
        encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
        encoder.esbmc_runner = SimpleNamespace(
            run_file=lambda _: self.fail("ESBMC must not run without a property preimage")
        )
        with (
            patch.object(
                encoder,
                "_assumption_box_int",
                return_value=(
                    np.asarray([0], dtype=np.int64),
                    np.asarray([1], dtype=np.int64),
                    0,
                ),
            ),
            patch.object(
                encoder,
                "_candidate_error_budget_int",
                return_value=(
                    np.asarray([1], dtype=np.int64),
                    np.asarray([0], dtype=np.int64),
                    np.asarray([1], dtype=np.int64),
                    0,
                ),
            ),
            patch.object(
                encoder,
                "_layer_preimage_bounds_int",
                return_value=(
                    np.asarray([0], dtype=np.int64),
                    np.asarray([1], dtype=np.int64),
                ),
            ),
        ):
            result = encoder.verify_layer_with_esbmc(
                cur_layer=hidden,
                in_layer=SimpleNamespace(layer_size=1),
                qu_w_int=np.asarray([[1]], dtype=np.int64),
                qu_b_int=np.asarray([0], dtype=np.int64),
                frac_bit=0,
                all_bit=5,
                layer_index=0,
            )

        self.assertEqual(result.status, "PREIMAGE_UNAVAILABLE")
        self.assertEqual(encoder.synthesis_final_status, "PREIMAGE_UNAVAILABLE")
        self.assertIn("cannot be deflated", result.resource_control["reason"])

    def test_hidden_relational_cut_is_widened_by_inherited_budget(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.margin_cuts = True
        encoder.error_budget_mode = "derived"
        encoder.solver = "cbc"
        encoder.config = SimpleNamespace(gurobi_threads=1)
        encoder.hidden_contract_cut_records = []
        encoder._hidden_contract_cut_cache = {}
        current = SimpleNamespace(layer_index=2)
        previous = SimpleNamespace(
            layer_index=1,
            layer_size=2,
            frac_bit=2,
            int_bit=6,
            error_budget_int=np.asarray([1, 2], dtype=np.int64),
        )
        encoder.dense_layers = [previous]
        with (
            patch.object(
                encoder,
                "_assumption_box_int",
                return_value=(
                    np.asarray([0, 0], dtype=np.int64),
                    np.asarray([4, 4], dtype=np.int64),
                    2,
                ),
            ),
            patch.object(
                encoder,
                "_solve_margin_direction_milp",
                return_value=(0.5, 1.5, 0.01),
            ),
            patch.object(
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
            ),
        ):
            cuts = encoder._hidden_contract_cut_bounds(
                cur_layer=current,
                in_layer=previous,
                weights_int=np.asarray([[2, -1]], dtype=np.int64),
                frac_bit=2,
                all_bit=8,
                start_neuron=0,
                end_neuron=1,
                maximum_cuts=1,
            )

        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["inherited_widening_int"], 4)
        self.assertEqual(cuts[0]["cut_low_int"], -2)
        self.assertEqual(cuts[0]["cut_high_int"], 10)
        self.assertEqual(
            cuts[0]["soundness"],
            "esbmc_exact_deployed_prefix_validated",
        )

    def test_hidden_cut_harness_tightens_accumulator_endpoints(self) -> None:
        source = render_hidden_affine_bounds_program(
            output_size=1,
            input_size=2,
            weights_c_int="{{2, -1}}",
            biases_c_int="{0}",
            preimage_low_c_int="{-2}",
            preimage_high_c_int="{2}",
            input_bounds_low_c_int="{0, 0}",
            input_bounds_high_c_int="{4, 4}",
            scale_factor=4,
            total_bits=8,
            input_scale_factor=4,
            contract_tolerance_c_int="{1}",
            contract_cut_directions_c_int="{{2, -1}}",
            contract_cut_low_c_int="{-2}",
            contract_cut_high_c_int="{10}",
            contract_cut_output_indices_c_int="{0}",
            contract_cut_count=1,
        )

        self.assertIn("contract_cut_output_indices[cut] == i", source)
        self.assertIn("lower_acc = (__int128)contract_cut_low[cut]", source)
        self.assertNotIn("nondet_longlong", source)

    def test_relational_cut_validator_uses_exact_deployed_prefix(self) -> None:
        source = render_prefix_direction_cut_validation_program(
            input_size=1,
            input_bounds_low_c_int="{-2}",
            input_bounds_high_c_int="{2}",
            layers=[
                {
                    "input_size": 1,
                    "output_size": 2,
                    "total_bits": 8,
                    "fractional_bits": 3,
                    "input_fractional_bits": 3,
                    "weights_c_int": "{{4}, {-4}}",
                    "biases_c_int": "{0, 0}",
                }
            ],
            direction_c_int="{2, -1}",
            cut_low_int=-4,
            cut_high_int=8,
        )

        self.assertIn("nondet_longlong", source)
        self.assertIn("div_round_half_away_from_zero_i128", source)
        self.assertIn("clamp_to_signed_range_i128", source)
        self.assertIn("direction_value >= (__int128)CUT_LOW", source)
        self.assertNotIn("LAYER_0_LOW", source)

    def test_failed_exact_prefix_validation_marks_cut_untrusted(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.dense_layers = [
            SimpleNamespace(
                frac_bit=2,
                int_bit=6,
                layer_paras=[
                    np.asarray([[1.0]], dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                ],
            )
        ]
        encoder.x_low_real = np.asarray([-0.5], dtype=np.float64)
        encoder.x_high_real = np.asarray([0.5], dtype=np.float64)
        encoder._relational_cut_validation_cache = {}
        encoder._stats = {"esbmc_calls": 0.0}
        encoder.esbmc_call_records = []
        encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
        encoder.esbmc_runner = SimpleNamespace(
            run_file=lambda _: ESBMCResult(
                status="FAILED",
                command=("esbmc",),
                stdout="",
                stderr="",
                return_code=1,
            )
        )
        record = {
            "direction_int": [1],
            "cut_low_int": 0,
            "cut_high_int": 0,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder.output_dir = Path(temp_dir)
            validated = encoder._formally_validate_relational_cut(
                record,
                hidden_layer_count=1,
                layer_index=1,
                all_bit=8,
                frac_bit=2,
                cut_kind="hidden_contract",
                identifier="neuron_0",
            )

        self.assertFalse(validated)
        self.assertEqual(record["formal_validation_status"], "FAILED")
        self.assertEqual(record["soundness"], "untrusted_milp_proposal_not_injected")
        self.assertEqual(
            encoder.esbmc_call_records[0]["property_type"],
            "relational_cut_validation",
        )

    def test_hidden_harness_rejects_unvalidated_or_miswired_cuts(self) -> None:
        weights = np.asarray([[2, -1]], dtype=np.int64)
        unvalidated = {
            "neuron_index": 0,
            "direction_int": [2, -1],
            "formal_validation_status": "TIMEOUT",
        }
        with self.assertRaisesRegex(ValueError, "formally validated"):
            GPEncoding._require_validated_hidden_row_cuts(weights, [unvalidated])

        miswired = {
            "neuron_index": 0,
            "direction_int": [1, -1],
            "formal_validation_status": "VERIFIED",
        }
        with self.assertRaisesRegex(ValueError, "quantized affine row"):
            GPEncoding._require_validated_hidden_row_cuts(weights, [miswired])


if __name__ == "__main__":
    unittest.main()
