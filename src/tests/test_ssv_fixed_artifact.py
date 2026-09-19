import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from datasets.gtsrb_study import sha256
from models.ssv_artifact import freeze_artifact, load_artifact, parse_qif
from scripts.freeze_ssv_campaign import campaign_runs
from scripts.prepare_ssv_gtsrb import validate_config
from scripts.report_ssv_regions import summarize
from synthesis.preqbmc import GPEncoding
from tests.test_source_region_gate import _encoder


class FixedArtifactTest(unittest.TestCase):
    def fixture(self, root):
        config = json.loads((Path(__file__).resolve().parents[2] / "experiments/ssv2026_gtsrb_pilot.json").read_text())
        g = validate_config(config)
        path = root / "model.npz"
        np.savez(path, conv_kernel=np.ones(g.kernel_shape, dtype=np.float32), conv_bias=np.zeros(2),
                 dense_kernel=np.zeros((18, 43)), dense_bias=np.r_[1., np.zeros(42)])
        return {"config": config, "model_path": str(path), "model_sha256": sha256(path)}

    def test_identity_is_independent_of_region_and_detects_drift(self):
        rows = [asdict(LayerQuantizationSpec(11, 2, 8)), asdict(LayerQuantizationSpec(13, 4, 8))]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            study = self.fixture(root)
            artifact = freeze_artifact(study, rows, root / "artifact")
            loaded, specs = load_artifact(root / "artifact/artifact.json", study)
            self.assertEqual(loaded, artifact)
            self.assertEqual([asdict(s) for s in specs], rows)
            changed_region = {**study, "selected_images": [{"id": "different-region"}]}
            self.assertEqual(load_artifact(root / "artifact/artifact.json", changed_region)[0], artifact)
            with self.assertRaises(FileExistsError):
                freeze_artifact(study, rows, root / "artifact")
            manifest_path = root / "artifact/artifact.json"
            mutated = copy.deepcopy(artifact)
            mutated["qif"][0] = asdict(LayerQuantizationSpec(12, 3, 8))
            manifest_path.write_text(json.dumps(mutated))
            with self.assertRaisesRegex(ValueError, "drift"):
                load_artifact(manifest_path, study)
            manifest_path.write_text(json.dumps(artifact))
            with (root / "artifact/qnn.c").open("a") as handle:
                handle.write("\n/* changed */\n")
            with self.assertRaisesRegex(ValueError, "drift"):
                load_artifact(manifest_path, study)

    def test_formats_are_integers_not_just_fixed_f(self):
        rows = [asdict(LayerQuantizationSpec(11, 2, 8))] * 2
        self.assertEqual(len(parse_qif(rows)), 2)
        for bad in (11.0, True, "11"):
            with self.assertRaises(ValueError):
                parse_qif([{**rows[0], "total_bits": bad}, rows[1]])
        with self.assertRaises(ValueError):
            parse_qif([{**rows[0], "total_bits": 12}, rows[1]])

    def test_fixed_run_never_enters_search_or_widens_formats(self):
        specs = [LayerQuantizationSpec(11, 2, 8), LayerQuantizationSpec(13, 4, 8)]
        for source_ok, verifier_status in ((True, "VERIFIED"), (True, "TIMEOUT"), (False, "VERIFIED")):
            enc = _encoder([.8, 0] if source_ok else [.2, 0], [1, .5])
            enc.harness_scope = "layer"
            enc.error_budget_mode = "derived"
            enc.e2e_fallback = False
            enc.dense_layers = [SimpleNamespace(preimage_source="milp_preimage", layer_index=1)]
            enc.backward_preimage_computation = Mock()
            enc._widen_internal_integer_bits_for_fixed_point_contracts = Mock(side_effect=AssertionError("widen"))
            enc.forward_quantization_with_esbmc = Mock(side_effect=AssertionError("search"))
            enc.verify_exported_quantization_with_esbmc = Mock(return_value=(
                verifier_status == "VERIFIED", [{"final_status": verifier_status}]))
            result = enc.run(np.array([0.]), np.array([1.]), fixed_qif=specs)
            self.assertEqual(result.final_status, verifier_status if source_ok else "SOURCE_PROPERTY_INCONCLUSIVE")
            self.assertEqual(result.success, source_ok and verifier_status == "VERIFIED")
            if source_ok:
                enc.verify_exported_quantization_with_esbmc.assert_called_once_with([11, 13], [8, 8], [2, 4])
                enc.backward_preimage_computation.assert_called_once()
            else:
                enc.verify_exported_quantization_with_esbmc.assert_not_called()
                enc.backward_preimage_computation.assert_not_called()

    def test_explicit_verifier_binds_first_layer_input_clamp(self):
        enc = object.__new__(GPEncoding)
        enc.dense_layers = []
        enc.output_layer = SimpleNamespace(layer_index=1, int_bit=20, frac_bit=None,
                                           layer_paras=(np.array([[1.]]), np.array([0.])))
        enc.input_layer = SimpleNamespace()
        enc.deep_model = None
        enc.deepPolyNets_DNN = SimpleNamespace(load_dnn=lambda _: None)
        enc.error_budget_mode = "heuristic"
        def check(**_):
            self.assertEqual(enc.output_layer.int_bit, 3)
            self.assertEqual(enc.output_layer.frac_bit, 8)
            return SimpleNamespace(status="TIMEOUT", blocks=[], resource_control={})
        enc.verify_layer_with_esbmc = check
        ok, records = enc.verify_exported_quantization_with_esbmc([11], [8], [2])
        self.assertFalse(ok)
        self.assertEqual(records[-1]["final_status"], "TIMEOUT")

    def test_campaign_is_36_runs_and_preserves_frozen_ids(self):
        selected = [{"id": str(i), "stratum": ["low", "median", "high"][i // 3]} for i in range(9)]
        runs = [{"run_id": f"{s['id']}_{e}_{b}_{c}", "sample": s, "epsilon": e,
                 "block_size": b, "margin_cuts": c} for s in selected
                for e in (1, 2, 4) for b in (0, 1, 2) for c in (False, True)]
        actual = campaign_runs({"selected_images": selected, "runs": runs})
        self.assertEqual(len(actual), 36)
        self.assertEqual(sum(r["block_size"] == 1 and not r["margin_cuts"] for r in actual), 27)
        self.assertEqual({r["sample"]["id"] for r in actual}, {s["id"] for s in selected})
        self.assertEqual({r["sample"]["id"] for r in actual if r["block_size"] == 0}, {"0", "3", "6"})

    def test_fixed_reports_cannot_mix_programs(self):
        common = {"epsilon": 1, "block_size": 1, "margin_cuts": False,
                  "verification_mode": "fixed_qif_check", "sample": {"id": "x"},
                  "source_region": {"status": "VERIFIED"}, "calls": [], "total_runtime_seconds": 1.,
                  "final_status": "VERIFIED", "byte_crop_property_verified": True,
                  "encoded_contract_verified": True, "android_transfer_verified": False}
        a = {**common, "fixed_artifact_source_sha256": "a", "generated_source_sha256": "a"}
        b = {**common, "fixed_artifact_source_sha256": "b", "generated_source_sha256": "b"}
        self.assertEqual(len(summarize([a, b])), 2)
        with self.assertRaises(ValueError):
            summarize([{**a, "generated_source_sha256": "changed"}])


if __name__ == "__main__":
    unittest.main()
