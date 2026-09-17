import json
from pathlib import Path
import unittest
import tempfile
from unittest.mock import patch
import numpy as np

from datasets.gtsrb_study import split_tracks, select_regions
from reports.ssv_measurements import integrate_power, missing_device_report
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.report_ssv_regions import summarize
from scripts.evaluate_ssv_host import evaluate
from datasets.gtsrb_study import sha256
from backends.fixed_point import LayerQuantizationSpec
from backends.image_encoder import ByteImageEncoder
from backends.c_qnn_generator import generate_c_qnn_source
from models.restricted_conv import RestrictedCNN
from dataclasses import asdict


class SsvStudyTest(unittest.TestCase):
    def test_host_evaluation_on_synthetic_rgb_fixture(self):
        config = json.loads((Path(__file__).resolve().parents[2] / "experiments/ssv2026_gtsrb_pilot.json").read_text())
        g = validate_config(config)
        cnn = RestrictedCNN(g, np.ones(g.kernel_shape, dtype=np.float32), np.zeros(2),
                            np.zeros((18, 43)), np.r_[1., np.zeros(42)])
        specs = [LayerQuantizationSpec(12, 3, 8)] * 2
        encoder = ByteImageEncoder(g.input_shape, 8, 12)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = root / "model.npz"
            np.savez(weights, conv_kernel=cnn.conv_kernel, conv_bias=cnn.conv_bias,
                     dense_kernel=cnn.dense_kernel, dense_bias=cnn.dense_bias)
            source = root / "qnn.c"
            source.write_text(generate_c_qnn_source(cnn.quantized(specs, input_total_bits=12)) + encoder.render_c())
            study = {"config": config, "dataset_root": "synthetic", "model_path": str(weights),
                     "model_sha256": sha256(weights), "source_float32_test_accuracy": 1.,
                     "records": [{"id": "synthetic", "sha256": "fixture", "class_id": 0, "split": "test"}]}
            region = {"run_id": "synthetic_fixture_not_a_certificate", "byte_crop_property_verified": True,
                      "model_sha256": sha256(weights), "generated_source_sha256": sha256(source),
                      "qif": [asdict(s) for s in specs]}
            with patch("scripts.evaluate_ssv_host.load_crop", return_value=np.full((10, 9, 3), 128, dtype=np.uint8)):
                result = evaluate(study, region, root, root / "quality")
            self.assertTrue(result["all_host_parity_passed"])
            self.assertEqual(result["methods"][0]["host_c_accuracy"], 1.)
            self.assertEqual(result["android_parity"], "NOT_MEASURED")

    def test_tracks_never_cross_validation_boundary(self):
        rows = [{"id": f"{c}/{t}/{i}", "class_id": c, "track_id": f"{c}:{t}"}
                for c in range(3) for t in range(5) for i in range(3)]
        result = split_tracks(rows)
        for track in {r["track_id"] for r in result}:
            self.assertEqual(len({r["split"] for r in result if r["track_id"] == track}), 1)
        self.assertEqual({r["id"]: r["split"] for r in result},
                         {r["id"]: r["split"] for r in split_tracks(list(reversed(rows)))})
        self.assertEqual(sum(r["split"] == "validation" for r in result), 9)

    def test_selection_is_fixed_before_proofs_and_distinct(self):
        records = [{"id": str(i), "class_id": i % 3, "split": "test"} for i in range(30)]
        logits = np.zeros((30, 3))
        for i in range(30):
            logits[i, i % 3] = i + 1
        selected = select_regions(records, logits)
        self.assertEqual(len({r["id"] for r in selected}), 9)
        self.assertEqual({r["class_id"] for r in selected}, {0, 1, 2})
        self.assertEqual([r["stratum"] for r in selected], ["low"] * 3 + ["median"] * 3 + ["high"] * 3)
        self.assertEqual(selected, select_regions(records, logits))
        for epsilon in (1, 2, 4):
            self.assertEqual({(r["id"], epsilon)[0] for r in selected}, {r["id"] for r in selected})

    def test_energy_uses_synchronized_power_not_cpu(self):
        result = integrate_power([0, 1, 2], [2, 2, 2], .5, 1.5, 10, idle_watts=1)
        self.assertAlmostEqual(result["gross_joules_per_inference"], .2)
        self.assertAlmostEqual(result["idle_subtracted_joules_per_inference"], .1)
        with self.assertRaises(ValueError):
            integrate_power([0, 1], [2, 2], 0, 2, 10)
        with self.assertRaises(ValueError):
            integrate_power([0, 0], [2, 2], 0, 1, 10)
        self.assertIsNone(missing_device_report()["power"]["gross_joules_per_inference"])

    def test_study_config_is_explicit_and_bounded(self):
        path = Path(__file__).resolve().parents[2] / "experiments/ssv2026_gtsrb_pilot.json"
        config = json.loads(path.read_text())
        g = validate_config(config)
        self.assertEqual(g.output_shape, (3, 3, 2))
        self.assertEqual(config["block_sizes"], [0, 1, 2])
        config["verification"]["unsound_contract_tolerance"] = True
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_region_denominators_and_diagnostic_timeouts(self):
        common = {"epsilon": 1, "block_size": 2, "margin_cuts": True,
                  "android_transfer_verified": False, "total_runtime_seconds": 1.0}
        good = {**common, "sample": {"id": "a"}, "source_region": {"status": "VERIFIED"},
                "final_status": "VERIFIED", "byte_crop_property_verified": True,
                "encoded_contract_verified": True,
                "calls": [{"status": "TIMEOUT", "peak_memory_bytes": 1024}]}
        inconclusive = {**common, "sample": {"id": "b"}, "source_region": {"status": "INCONCLUSIVE"},
                        "final_status": "SOURCE_PROPERTY_INCONCLUSIVE", "byte_crop_property_verified": False,
                        "encoded_contract_verified": False, "calls": []}
        row = summarize([good, inconclusive])[0]
        self.assertEqual(row["certified_fraction_completed_regions"], .5)
        self.assertEqual(row["certified_fraction_given_source_verified"], 1.0)
        self.assertEqual(row["timeout_regions"], 0)
        self.assertEqual(row["query_timeout_count_including_diagnostics"], 1)
        self.assertIsNone(row["android_accuracy"])
        with self.assertRaises(ValueError):
            summarize([good, good])


if __name__ == "__main__":
    unittest.main()
