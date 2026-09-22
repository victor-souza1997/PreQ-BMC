import json
from pathlib import Path
import struct
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from datasets.gtsrb_study import sha256
from scripts.prepare_ssv_android_bundle import MAGIC, prepare
from scripts.run_ssv_android_device import run


class AndroidBundleTest(unittest.TestCase):
    def fixture(self, root):
        method = root / "preqbmc_selected"
        method.mkdir()
        (method / "qnn.c").write_text("void qnn(void) {}\n")
        rows = [
            {"image_id": "a.ppm", "image_sha256": "a" * 64,
             "encoded_input": [1, -2], "hidden": [3], "logits": [4, -5], "label": 0},
            {"image_id": "b.ppm", "image_sha256": "b" * 64,
             "encoded_input": [6, 7], "hidden": [8], "logits": [-9, 10], "label": 1},
        ]
        with (method / "device_vectors.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        report = {"quality_status": "MEASURED", "all_host_parity_passed": True,
                  "android_parity": "NOT_MEASURED", "model_sha256": "model",
                  "methods": [{"method": "preqbmc_selected", "source_sha256": sha256(method / "qnn.c"),
                               "n_test_images": 2, "host_c_accuracy": 1.0,
                               "python_c_exact_layer_rate": 1.0,
                               "qif": [{"total_bits": 12, "integer_bits": 3, "fractional_bits": 8}]}]}
        (root / "host_quality.json").write_text(json.dumps(report))

    def test_bundle_is_identity_bound_and_little_endian(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            result = prepare(root, root / "bundle")
            self.assertEqual(result["formal_claim"], "NONE_ADDED_BY_THIS_BUNDLE")
            self.assertEqual(result["formal_artifact_status"], "NO_CERTIFICATE_BOUND")
            self.assertEqual(result["android_status"], "NOT_MEASURED")
            self.assertEqual(result["n_test_images"], 2)
            payload = (root / "bundle" / "vectors.bin").read_bytes()
            self.assertEqual(payload[:8], MAGIC)
            self.assertEqual(struct.unpack("<III", payload[8:20]), (2, 2, 2))
            self.assertEqual(struct.unpack("<i2q2q", payload[20:56]), (0, 1, -2, 4, -5))
            with self.assertRaises(FileExistsError):
                prepare(root, root / "bundle")

    def test_only_exact_verified_region_certificates_are_bound(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            source_hash = sha256(root / "preqbmc_selected" / "qnn.c")
            report = json.loads((root / "host_quality.json").read_text())
            certificate = {"run_id": "r", "final_status": "VERIFIED",
                           "byte_crop_property_verified": True, "generated_source_sha256": source_hash,
                           "model_sha256": report["model_sha256"], "sample": {"id": "a"},
                           "epsilon": 1, "guarantee_level": "byte-crop-integer-C"}
            path = root / "certificate.json"
            path.write_text(json.dumps(certificate))
            result = prepare(root, root / "bound", [path])
            self.assertEqual(result["formal_artifact_status"], "REGION_CERTIFICATES_BOUND")
            self.assertEqual(result["bound_region_certificates"][0]["run_id"], "r")
            certificate["generated_source_sha256"] = "wrong"
            path.write_text(json.dumps(certificate))
            with self.assertRaisesRegex(ValueError, "exact model and C"):
                prepare(root, root / "unbound", [path])

    def test_tampered_source_and_failed_host_parity_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            (root / "preqbmc_selected" / "qnn.c").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "identity"):
                prepare(root, root / "bad")
            report = json.loads((root / "host_quality.json").read_text())
            report["all_host_parity_passed"] = False
            (root / "host_quality.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "host-parity"):
                prepare(root, root / "bad2")

    def test_device_report_preserves_empirical_scope_and_detects_aaos(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            prepare(root, root / "bundle")
            binary = root / "benchmark"
            binary.write_bytes(b"arm64 fixture")
            trial = {"schema": "preqbmc_android_benchmark_v1", "parity_status": "EXACT_MATCH",
                     "vectors": 2, "latency_median_ns": 10, "latency_p95_ns": 20,
                     "throughput_per_second": 1000, "peak_process_rss_kib": 64,
                     "cpu_time_over_wall_percent": 99, "compiler_id": "Clang",
                     "compiler_version": "fixture", "ndk_version": "fixture",
                     "compiled_abi": "arm64-v8a", "android_platform": "android-26"}

            def adb(_serial, *args, **_kwargs):
                if args[:3] == ("shell", "pm", "list"):
                    return SimpleNamespace(stdout="feature:android.hardware.type.automotive\n", stderr="")
                if args[:3] == ("shell", "getprop", "ro.build.fingerprint"):
                    return SimpleNamespace(stdout="test/fingerprint\n", stderr="")
                if args[:2] == ("shell", "getprop"):
                    return SimpleNamespace(stdout="fixture\n", stderr="")
                if args[:3] == ("shell", "cat", "/proc/cpuinfo"):
                    return SimpleNamespace(stdout="processor: fixture\n", stderr="")
                if args and args[0] == "shell" and ("benchmark" in str(args[1:])):
                    return SimpleNamespace(stdout=json.dumps(trial) + "\n", stderr="")
                return SimpleNamespace(stdout="", stderr="")

            with patch("scripts.run_ssv_android_device._adb", side_effect=adb), \
                 patch("scripts.run_ssv_android_device.shutil.which", return_value="/usr/bin/adb"):
                result = run(root / "bundle", binary, root / "device", serial="fixture",
                             core_mask="0x3", trials=3, warmup=1,
                             latency_samples=1, throughput_iterations=1)
            self.assertEqual(result["device"]["platform_classification"], "AAOS")
            self.assertEqual(result["native_c_parity"], "EXACT_MATCH")
            self.assertEqual(result["build_metadata"]["compiled_abi"], "arm64-v8a")
            self.assertEqual(result["jni_parity"], "NOT_MEASURED")
            self.assertFalse(result["formal_status_modified"])


if __name__ == "__main__":
    unittest.main()
