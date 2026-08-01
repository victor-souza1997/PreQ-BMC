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
from verification.c_templates import render_hidden_affine_bounds_program
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


if __name__ == "__main__":
    unittest.main()
