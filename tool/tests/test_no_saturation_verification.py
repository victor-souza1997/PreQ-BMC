from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import unittest

from verification.c_templates import (
    render_clamp_correctness_program,
    render_hidden_affine_bounds_block_program,
    render_no_saturation_program,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner


def _single_neuron_program(total_bits: int) -> str:
    return render_no_saturation_program(
        output_size=1,
        input_size=1,
        weights_c_int="{{1}}",
        biases_c_int="{0}",
        input_bounds_low_c_int="{0}",
        input_bounds_high_c_int="{4}",
        scale_factor=1,
        total_bits=total_bits,
    )


class NoSaturationVerificationTest(unittest.TestCase):
    def test_no_saturation_template_contains_esbmc_assertions(self) -> None:
        source = _single_neuron_program(total_bits=4)

        self.assertIn("__ESBMC_assert(lower_pre_clamp >= q_min", source)
        self.assertIn("__ESBMC_assert(upper_pre_clamp <= q_max", source)
        self.assertIn("fixed-point saturation possible", source)

    def test_clamp_correctness_template_contains_branch_properties(self) -> None:
        source = render_clamp_correctness_program(total_bits=8)

        self.assertIn("clamp output below q_min", source)
        self.assertIn("clamp changed in-range input", source)
        self.assertIn("clamp did not saturate low input to q_min", source)
        self.assertIn("clamp did not saturate high input to q_max", source)

    def test_hidden_block_template_uses_block_size_with_full_input_bounds(self) -> None:
        source = render_hidden_affine_bounds_block_program(
            block_size=1,
            input_size=3,
            weights_c_int="{{1, 2, 3}}",
            biases_c_int="{0}",
            preimage_low_c_int="{-1}",
            preimage_high_c_int="{1}",
            input_bounds_low_c_int="{0, 0, 0}",
            input_bounds_high_c_int="{4, 4, 4}",
            scale_factor=1,
        )

        self.assertIn("#define INPUT_SIZE 3", source)
        self.assertIn("#define LAYER_SIZE 1", source)
        self.assertIn("long long weights[LAYER_SIZE][INPUT_SIZE] = {{1, 2, 3}};", source)
        self.assertIn("long long input_bounds_low[INPUT_SIZE] = {0, 0, 0};", source)

    def test_cli_exposes_formal_and_empirical_saturation_flags(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness_pipeline.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn("--formal-saturation-check", source)
        self.assertIn("--no-formal-saturation-check", source)
        self.assertIn("--empirical-saturation-check", source)
        self.assertIn("--no-empirical-saturation-check", source)
        self.assertIn("--esbmc-layer-block-size", source)

    @unittest.skipUnless(shutil.which("esbmc"), "esbmc binary is not installed")
    def test_esbmc_no_saturation_fails_for_too_small_q_and_passes_for_larger_q(self) -> None:
        runner = ESBMCRunner(ESBMCConfig(timeout_seconds=20, verbosity=10))
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            small_q = temp_path / "small_q.c"
            large_q = temp_path / "large_q.c"
            small_q.write_text(_single_neuron_program(total_bits=2), encoding="utf-8")
            large_q.write_text(_single_neuron_program(total_bits=4), encoding="utf-8")

            small_result = runner.run_file(small_q)
            large_result = runner.run_file(large_q)

        self.assertEqual(small_result.status, "FAILED")
        self.assertEqual(large_result.status, "VERIFIED")


if __name__ == "__main__":
    unittest.main()
