from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
import unittest

from verification.c_templates import render_no_saturation_program
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

    def test_cli_exposes_formal_and_empirical_saturation_flags(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness_pipeline.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn("--formal-saturation-check", source)
        self.assertIn("--no-formal-saturation-check", source)
        self.assertIn("--empirical-saturation-check", source)
        self.assertIn("--no-empirical-saturation-check", source)

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
