from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from synthesis.preqbmc import GPEncoding
from verification.esbmc import ESBMCConfig, ESBMCRunner


@unittest.skipUnless(Path("/proc").is_dir(), "Linux /proc is required for RSS monitoring")
class ESBMCResourceMonitoringTest(unittest.TestCase):
    def test_call_summary_reports_verification_rate_and_cpu_utilization(self) -> None:
        encoder = GPEncoding.__new__(GPEncoding)
        encoder.esbmc_call_records = [
            {
                "status": "VERIFIED",
                "elapsed_seconds": 2.0,
                "cpu_time_seconds": 3.0,
                "average_cpu_utilization_percent": 150.0,
                "peak_memory_bytes": 10_000_000,
            },
            {
                "status": "TIMEOUT",
                "elapsed_seconds": 2.0,
                "cpu_time_seconds": 1.0,
                "average_cpu_utilization_percent": 50.0,
                "peak_memory_bytes": 20_000_000,
            },
            {
                "status": "SENTINEL_EXPECTED_FAILURE",
                "elapsed_seconds": 0.1,
                "cpu_time_seconds": 0.05,
                "average_cpu_utilization_percent": 50.0,
                "peak_memory_bytes": 1_000_000,
            },
        ]

        summary = encoder.esbmc_call_summary()

        self.assertEqual(summary["verification_denominator"], 2)
        self.assertEqual(summary["verification_rate_percent"], 50.0)
        self.assertAlmostEqual(
            summary["average_cpu_utilization_percent"],
            100.0 * 4.05 / 4.1,
        )
        self.assertEqual(summary["max_query_cpu_utilization_percent"], 150.0)

    def test_records_peak_process_tree_rss_without_capturing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_esbmc = root / "fake_esbmc"
            fake_esbmc.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "payload = bytearray(12 * 1024 * 1024)\n"
                "deadline = time.monotonic() + 0.15\n"
                "value = 0\n"
                "while time.monotonic() < deadline:\n"
                "    value += 1\n"
                "time.sleep(0.20)\n"
                "print('VERIFICATION SUCCESSFUL')\n",
                encoding="utf-8",
            )
            fake_esbmc.chmod(fake_esbmc.stat().st_mode | 0o111)
            harness = root / "query.c"
            harness.write_text(
                "#define INPUT_SIZE 1\n#define LAYER_SIZE 1\nint main(void) { return 0; }\n",
                encoding="utf-8",
            )
            runner = ESBMCRunner(
                ESBMCConfig(
                    executable=str(fake_esbmc),
                    timeout_seconds=2,
                    memory_poll_interval_seconds=0.01,
                )
            )

            result = runner.run_file(harness)

        self.assertEqual(result.status, "VERIFIED")
        self.assertIsNotNone(result.peak_memory_bytes)
        assert result.peak_memory_bytes is not None
        self.assertGreater(result.peak_memory_bytes, 8 * 1024 * 1024)
        self.assertEqual(result.memory_measurement, "linux_procfs_process_tree_rss")
        self.assertEqual(
            result.resource_control["peak_memory_bytes"],
            result.peak_memory_bytes,
        )
        self.assertGreater(result.resource_control["peak_memory_mib"], 8.0)
        self.assertIsNotNone(result.cpu_time_seconds)
        self.assertIsNotNone(result.average_cpu_utilization_percent)
        assert result.cpu_time_seconds is not None
        assert result.average_cpu_utilization_percent is not None
        self.assertGreater(result.cpu_time_seconds, 0.05)
        self.assertGreater(result.average_cpu_utilization_percent, 10.0)
        self.assertEqual(
            result.cpu_measurement,
            "linux_procfs_process_tree_cpu_time",
        )
        self.assertEqual(
            result.resource_control["cpu_time_seconds"],
            result.cpu_time_seconds,
        )
        self.assertTrue(result.stdout_log_path.endswith(".stdout.log"))
        self.assertIn("VERIFICATION SUCCESSFUL", result.stdout)


if __name__ == "__main__":
    unittest.main()
