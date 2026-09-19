import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from datasets.gtsrb_study import sha256
from scripts.export_ssv_tflite import calibration_rows, export, quantize_input
from scripts.prepare_ssv_gtsrb import validate_config


class TfliteBaselineTest(unittest.TestCase):
    def test_calibration_cannot_use_test_or_validation(self):
        records = [{"id": str(i), "split": "train" if i < 5 else "test" if i < 8 else "validation"}
                   for i in range(10)]
        rows = calibration_rows(records, 3)
        self.assertEqual(rows, calibration_rows(list(reversed(records)), 3))
        self.assertTrue(all(r["split"] == "train" for r in rows))
        with self.assertRaises(ValueError):
            calibration_rows(records, 6)
        with self.assertRaises(ValueError):
            calibration_rows(records + [records[0]], 3)

    def test_baseline_input_adapter_declares_rounding_and_clipping(self):
        detail = {"dtype": np.int8, "quantization": (.5, 0)}
        np.testing.assert_array_equal(quantize_input([.25, .75, -.25, -.75, 200], detail), [0, 2, 0, -2, 127])
        with self.assertRaises(ValueError):
            quantize_input([0], {"dtype": np.int8, "quantization": (0., 0)})

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow optional baseline dependency")
    def test_real_converter_on_synthetic_model_is_not_a_certificate(self):
        config = json.loads((Path(__file__).resolve().parents[2] / "experiments/ssv2026_gtsrb_pilot.json").read_text())
        geom = validate_config(config)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "model.npz"
            np.savez(path, conv_kernel=np.full(geom.kernel_shape, .01, np.float32), conv_bias=np.zeros(2),
                     dense_kernel=np.ones((18, 43), np.float32) * .01, dense_bias=np.r_[1., np.zeros(42)])
            study = {"config": config, "model_path": str(path), "model_sha256": sha256(path),
                     "dataset_root": "synthetic", "records": [
                         {"id": "a", "sha256": "a", "split": "train", "class_id": 0},
                         {"id": "b", "sha256": "b", "split": "test", "class_id": 0}]}
            with patch("scripts.export_ssv_tflite.load_crop", return_value=np.full((9, 9, 3), 128, np.uint8)):
                report = export(study, root / "baseline", calibration_count=1, evaluate_test=True)
            self.assertEqual(report["android_status"], "NOT_MEASURED")
            self.assertEqual(report["methods"][1]["input_dtype"], "int8")
            for row in report["methods"]:
                self.assertEqual(row["certification_status"], "UNCERTIFIED_BASELINE")
                self.assertEqual(row["host_test_accuracy"], 1.)
                self.assertGreater(row["artifact_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
