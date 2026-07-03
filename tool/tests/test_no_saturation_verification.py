from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from synthesis.quadapter import GPEncoding
from verification.c_templates import (
    render_clamp_correctness_program,
    render_hidden_affine_bounds_block_program,
    render_no_saturation_block_program,
    render_no_saturation_program,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner, ESBMCResult


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

    def test_no_saturation_block_template_uses_block_size_with_full_input_bounds(self) -> None:
        source = render_no_saturation_block_program(
            block_size=2,
            input_size=3,
            weights_c_int="{{1, 2, 3}, {4, 5, 6}}",
            biases_c_int="{0, 1}",
            input_bounds_low_c_int="{0, 0, 0}",
            input_bounds_high_c_int="{4, 4, 4}",
            scale_factor=4,
            total_bits=8,
            integer_bits=4,
            fractional_bits=3,
        )

        self.assertIn("#define INPUT_SIZE 3", source)
        self.assertIn("#define LAYER_SIZE 2", source)
        self.assertIn("#define TOTAL_BITS 8", source)
        self.assertIn("#define INTEGER_BITS 4", source)
        self.assertIn("#define FRACTIONAL_BITS 3", source)
        self.assertIn("long long weights[LAYER_SIZE][INPUT_SIZE] = {{1, 2, 3}, {4, 5, 6}};", source)
        self.assertIn("long long input_bounds_low[INPUT_SIZE] = {0, 0, 0};", source)

    def test_cli_exposes_formal_and_empirical_saturation_flags(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness_pipeline.py"
        source = script.read_text(encoding="utf-8")

        self.assertIn("--formal-saturation-check", source)
        self.assertIn("--no-formal-saturation-check", source)
        self.assertIn("--empirical-saturation-check", source)
        self.assertIn("--no-empirical-saturation-check", source)
        self.assertIn("--esbmc-layer-block-size", source)
        self.assertIn("--blockwise-fail-fast", source)
        self.assertIn("--blockwise-run-all-blocks-on-failure", source)
        self.assertIn("--esbmc-memlimit", source)
        self.assertIn("--esbmc-profile", source)
        self.assertIn("--gurobi-threads", source)

    def test_paper_fast_profile_uses_low_noise_resource_limited_flags(self) -> None:
        runner = ESBMCRunner(ESBMCConfig())
        command = runner.build_command(Path("harness.c"), unwind=4, profile="paper-fast")

        self.assertIn("--bitwuzla", command)
        self.assertIn("--bv", command)
        self.assertIn("--memlimit", command)
        self.assertIn("6g", command)
        self.assertIn("--result-only", command)
        self.assertIn("--interval-analysis", command)
        self.assertIn("--interval-analysis-simplify", command)
        self.assertNotIn("--verbosity", command)
        self.assertNotIn("--print-stack-traces", command)
        self.assertNotIn("--loop-invariant", command)

    def test_debug_profile_keeps_verbose_diagnostics(self) -> None:
        runner = ESBMCRunner(ESBMCConfig())
        command = runner.build_command(Path("harness.c"), unwind=4, profile="debug")

        self.assertIn("--verbosity", command)
        self.assertIn("--print-stack-traces", command)
        self.assertIn("--memstats", command)
        self.assertIn("--show-claims", command)

    def test_blockwise_verification_fail_fast_skips_remaining_blocks(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.calls: list[Path] = []

            def run_file(self, c_file: Path) -> ESBMCResult:
                self.calls.append(c_file)
                return ESBMCResult(
                    status="FAILED",
                    command=("esbmc", str(c_file), "--memlimit", "6g"),
                    stdout="VERIFICATION FAILED",
                    stderr="",
                    return_code=10,
                    elapsed_seconds=0.1,
                    timeout_seconds=900,
                    memlimit="6g",
                    stdout_log_path=f"{c_file}.stdout.log",
                    stderr_log_path=f"{c_file}.stderr.log",
                    resource_control={
                        "timeout": "900s",
                        "memlimit": "6g",
                        "stdout_log_path": f"{c_file}.stdout.log",
                        "stderr_log_path": f"{c_file}.stderr.log",
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = object.__new__(GPEncoding)
            encoder.output_dir = Path(temp_dir)
            encoder.esbmc_layer_block_size = 1
            encoder.blockwise_fail_fast = True
            encoder.blockwise_run_all_blocks_on_failure = False
            encoder.esbmc_jobs = 1
            encoder.esbmc_block_records = []
            encoder.blockwise_skipped_blocks_due_to_fail_fast = 0
            encoder.blockwise_first_failed_block = None
            encoder._stats = {"esbmc_calls": 0.0, "esbmc_block_calls": 0.0}
            encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
            encoder.esbmc_runner = FakeRunner()
            encoder.generate_esbmc_hidden_block_verification_code = lambda **_: "int main(void) { return 0; }"

            result = encoder.verify_hidden_layer_blocks_with_esbmc(
                cur_layer=SimpleNamespace(layer_size=3, layer_index=1),
                in_layer=SimpleNamespace(layer_size=2),
                qu_w_int=np.zeros((3, 2), dtype=np.int64),
                qu_b_int=np.zeros(3, dtype=np.int64),
                frac_bit=2,
                all_bit=4,
                layer_index=0,
            )

        self.assertEqual(result.status, "FAILED")
        self.assertEqual(len(encoder.esbmc_runner.calls), 1)
        self.assertEqual(encoder.blockwise_skipped_blocks_due_to_fail_fast, 2)
        self.assertEqual(len(result.blocks), 3)
        self.assertEqual(result.blocks[0]["status"], "FAILED")
        self.assertEqual(result.blocks[1]["status"], "SKIPPED")
        self.assertEqual(result.blocks[2]["status"], "SKIPPED")

    def test_no_saturation_block_verification_stops_on_timeout_by_default(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.calls: list[Path] = []

            def run_file(self, c_file: Path) -> ESBMCResult:
                self.calls.append(c_file)
                return ESBMCResult(
                    status="TIMEOUT",
                    command=("esbmc", str(c_file), "--memlimit", "6g"),
                    stdout="Timed out",
                    stderr="",
                    return_code=124,
                    elapsed_seconds=0.1,
                    timeout_seconds=900,
                    memlimit="6g",
                    stdout_log_path=f"{c_file}.stdout.log",
                    stderr_log_path=f"{c_file}.stderr.log",
                    resource_control={
                        "timeout": "900s",
                        "memlimit": "6g",
                        "stdout_log_path": f"{c_file}.stdout.log",
                        "stderr_log_path": f"{c_file}.stderr.log",
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            encoder = object.__new__(GPEncoding)
            encoder.output_dir = Path(temp_dir)
            encoder.esbmc_layer_block_size = 1
            encoder.no_saturation_continue_on_unknown = False
            encoder.esbmc_no_saturation_block_records = []
            encoder._stats = {"esbmc_calls": 0.0, "esbmc_block_calls": 0.0}
            encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
            encoder.esbmc_runner = FakeRunner()
            encoder.generate_esbmc_no_saturation_block_code = lambda **_: "int main(void) { return 0; }"

            result = encoder.verify_layer_no_saturation_blocks_with_esbmc(
                cur_layer=SimpleNamespace(layer_size=3, layer_index=1),
                in_layer=SimpleNamespace(layer_size=2),
                qu_w_int=np.zeros((3, 2), dtype=np.int64),
                qu_b_int=np.zeros(3, dtype=np.int64),
                frac_bit=2,
                all_bit=4,
                layer_index=0,
            )

        self.assertEqual(result.status, "TIMEOUT")
        self.assertEqual(len(encoder.esbmc_runner.calls), 1)
        self.assertIn("layer_0_no_sat_block_0_n0_1_Q4_F2.c", str(encoder.esbmc_runner.calls[0]))
        self.assertEqual(len(result.blocks), 3)
        self.assertEqual(result.blocks[0]["status"], "TIMEOUT")
        self.assertEqual(result.blocks[1]["status"], "SKIPPED")
        self.assertEqual(result.blocks[2]["status"], "SKIPPED")

    def test_hidden_no_saturation_uses_block_harnesses_when_enabled(self) -> None:
        class FakeRunner:
            def __init__(self) -> None:
                self.calls: list[Path] = []

            def run_file(self, c_file: Path) -> ESBMCResult:
                self.calls.append(c_file)
                return ESBMCResult(
                    status="VERIFIED",
                    command=("esbmc", str(c_file), "--memlimit", "6g"),
                    stdout="VERIFICATION SUCCESSFUL",
                    stderr="",
                    return_code=0,
                    elapsed_seconds=0.1,
                    timeout_seconds=900,
                    memlimit="6g",
                    stdout_log_path=f"{c_file}.stdout.log",
                    stderr_log_path=f"{c_file}.stderr.log",
                    resource_control={
                        "timeout": "900s",
                        "memlimit": "6g",
                        "stdout_log_path": f"{c_file}.stdout.log",
                        "stderr_log_path": f"{c_file}.stderr.log",
                    },
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            encoder = object.__new__(GPEncoding)
            encoder.output_dir = output_dir
            encoder.esbmc_layer_block_size = 1
            encoder.no_saturation_continue_on_unknown = False
            encoder.esbmc_no_saturation_block_records = []
            encoder.dense_layers = [SimpleNamespace(layer_index=1)]
            encoder._stats = {"esbmc_calls": 0.0, "esbmc_block_calls": 0.0}
            encoder.config = SimpleNamespace(esbmc=ESBMCConfig())
            encoder.esbmc_runner = FakeRunner()
            encoder.generate_esbmc_no_saturation_block_code = lambda **_: "int main(void) { return 0; }"

            result = encoder.verify_layer_no_saturation_with_esbmc(
                cur_layer=SimpleNamespace(layer_size=2, layer_index=1),
                in_layer=SimpleNamespace(layer_size=2),
                qu_w_int=np.zeros((2, 2), dtype=np.int64),
                qu_b_int=np.zeros(2, dtype=np.int64),
                frac_bit=2,
                all_bit=4,
                layer_index=0,
            )

            full_layer_harness = output_dir / "layers" / "layer_0_Q4_F2_no_saturation.c"

        self.assertEqual(result.status, "VERIFIED")
        self.assertEqual(len(encoder.esbmc_runner.calls), 2)
        self.assertFalse(full_layer_harness.exists())
        self.assertTrue(all("no_sat_block" in call.name for call in encoder.esbmc_runner.calls))

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
