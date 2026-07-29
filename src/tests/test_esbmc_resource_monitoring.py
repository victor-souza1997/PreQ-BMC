from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from verification.esbmc import ESBMCConfig, ESBMCRunner


@unittest.skipUnless(Path("/proc").is_dir(), "Linux /proc is required for RSS monitoring")
class ESBMCResourceMonitoringTest(unittest.TestCase):
    def test_records_peak_process_tree_rss_without_capturing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_esbmc = root / "fake_esbmc"
            fake_esbmc.write_text(
                "#!/usr/bin/env python3\n"
                "import time\n"
                "payload = bytearray(12 * 1024 * 1024)\n"
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
        self.assertTrue(result.stdout_log_path.endswith(".stdout.log"))
        self.assertIn("VERIFICATION SUCCESSFUL", result.stdout)


if __name__ == "__main__":
    unittest.main()
