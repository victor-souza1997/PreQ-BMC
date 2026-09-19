import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from backends.fixed_point import LayerQuantizationSpec
from datasets.gtsrb_study import sha256
from models.ssv_artifact import load_artifact
from scripts.search_ssv_qif import check_candidate, enumerate_candidates, execute, integer_bit_floors, prepare, search, validate
from tests.test_ssv_fixed_artifact import FixedArtifactTest


class SharedQifSearchTest(unittest.TestCase):
    def settings(self):
        return {"fractional_bits_min": 8, "fractional_bits_max": 10,
                "integer_bit_padding": [0, 1], "max_candidates": 36,
                "objective": "min_affine_parameter_bits"}

    def fixture(self, root):
        study = FixedArtifactTest().fixture(root)
        study.update({"dataset_root": "synthetic", "source_float32_test_accuracy": 1.,
                      "selected_images": [{"id": str(i), "stratum": ["low", "median", "high"][i // 3]}
                                          for i in range(9)]})
        study["runs"] = [{"run_id": f"image{i}_eps{e}_beta{b}_cuts{int(c)}", "sample": s,
                          "epsilon": e, "block_size": b, "margin_cuts": c,
                          **study["config"]["verification"]}
                         for i, s in enumerate(study["selected_images"])
                         for e in (1, 2, 4) for b in (0, 1, 2) for c in (False, True)]
        return study

    def test_domain_searches_fraction_and_range_independently_in_cost_order(self):
        candidates = enumerate_candidates([2, 4], [100, 20], self.settings())
        self.assertEqual(len(candidates), 36)
        costs = [c["affine_parameter_bits"] for c in candidates]
        self.assertEqual(costs, sorted(costs))
        self.assertTrue(any(c["qif"][0]["fractional_bits"] != c["qif"][1]["fractional_bits"] for c in candidates))
        for c in candidates:
            for row in c["qif"]:
                self.assertEqual(row["total_bits"], row["integer_bits"] + row["fractional_bits"] + 1)
        for bad in (7, 17, 8.0, True):
            with self.assertRaises(ValueError):
                validate({"schema": "ssv_qif_search_v1", "search": {**self.settings(), "fractional_bits_min": bad}})

    def test_range_floor_includes_preimage_and_unrounded_parameters(self):
        layer = SimpleNamespace(layer_paras=(np.array([[1.99]]), np.zeros(1)),
                                lb=np.array([0.]), ub=np.array([1.]),
                                relaxed_lb=np.array([-8.]), relaxed_ub=np.array([9.]))
        floors, counts = integer_bit_floors(SimpleNamespace(dense_layers=[], output_layer=layer))
        self.assertEqual(floors, [4])
        self.assertEqual(counts, [2])
        layer.ub[0] = np.inf
        with self.assertRaises(ValueError):
            integer_bit_floors(SimpleNamespace(dense_layers=[], output_layer=layer))

    def test_prepare_keeps_every_eligible_region_and_takes_maximum_floor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calibration = self.fixture(root)
            evaluation = copy.deepcopy(calibration)
            evaluation["selected_images"] = [{**s, "id": f"other-{s['id']}"} for s in evaluation["selected_images"]]
            (root / "calibration.json").write_text(json.dumps(calibration))
            (root / "evaluation.json").write_text(json.dumps(evaluation))
            config = {"schema": "ssv_qif_search_v1", "search": self.settings(),
                      "calibration_study": str(root / "calibration.json"),
                      "evaluation_study": str(root / "evaluation.json"),
                      "verification": calibration["config"]["verification"]}
            contexts = []
            def context(_, run, __):
                eligible = run["run_id"] in ("image0_eps1_beta1_cuts0", "image1_eps1_beta1_cuts0")
                synth = SimpleNamespace(check_source_region=lambda *_: eligible,
                    source_region_summary=lambda: {"status": "VERIFIED" if eligible else "INCONCLUSIVE", "certified_margin_lower_bound": .5},
                    backward_preimage_computation=lambda: None,
                    preimage_provenance_summary=lambda: {"status": "AVAILABLE", "all_property_preimages_available": True})
                contexts.append(synth)
                return SimpleNamespace(synthesizer=synth, low=np.zeros(1), high=np.ones(1))
            with patch("scripts.search_ssv_qif.build_region_context", side_effect=context), patch(
                    "scripts.search_ssv_qif.integer_bit_floors", side_effect=[([2, 4], [100, 20]), ([3, 5], [100, 20])]):
                manifest = prepare(root / "config.json", config, root / "search", {})
            self.assertEqual(manifest["shared_integer_bit_floors"], [3, 5])
            self.assertEqual(len(manifest["eligible_run_ids"]), 2)
            self.assertEqual(len(manifest["sources"]), 27)
            self.assertTrue(all(c["qif"][0]["integer_bits"] >= 3 for c in manifest["candidates"]))

    def test_candidate_fail_fast_and_resume_keep_one_shared_program(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            study = self.fixture(root)
            (root / "calibration_base.json").write_text(json.dumps(study))
            qif = [asdict(LayerQuantizationSpec(11, 2, 8)), asdict(LayerQuantizationSpec(13, 4, 8))]
            candidate = {"id": "candidate", "qif": qif, "affine_parameter_bits": 1}
            manifest = {"model_sha256": study["model_sha256"], "eligible_run_ids": [
                "image0_eps1_beta1_cuts0", "image1_eps1_beta1_cuts0", "image2_eps1_beta1_cuts0"]}
            seen = []
            def run(frozen, region, output, *, fixed_artifact):
                artifact, specs = load_artifact(fixed_artifact, frozen)
                seen.append(region["run_id"])
                passed = len(seen) == 1
                report = {"run_id": region["run_id"], "model_sha256": frozen["model_sha256"],
                          "verification_mode": "fixed_qif_check", "fixed_artifact_sha256": sha256(fixed_artifact),
                          "qif": [asdict(s) for s in specs], "source_region": {"status": "VERIFIED"},
                          "final_status": "VERIFIED" if passed else "TIMEOUT", "encoded_contract_verified": passed,
                          "input_bridge_checked": passed, "byte_crop_property_verified": passed,
                          "chaining": {"all_ok": passed}, "vacuity": {"status": "PASSED"},
                          "generated_source_sha256": artifact["generated_source_sha256"] if passed else None}
                output.mkdir(parents=True)
                (output / "region_summary.json").write_text(json.dumps(report))
                return report
            with patch("scripts.search_ssv_qif.run_region", side_effect=run) as verifier:
                first = check_candidate(candidate, manifest, root)
                second = check_candidate(candidate, manifest, root)
                self.assertEqual(verifier.call_count, 2)
            self.assertEqual(first, second)
            self.assertFalse(first["certified"])
            self.assertEqual(first["status"], "TIMEOUT")
            self.assertEqual(first["skipped_run_ids"], ["image2_eps1_beta1_cuts0"])
            self.assertEqual(first["first_failed_region"]["run_id"], "image1_eps1_beta1_cuts0")

    def test_source_inconclusive_never_enters_candidate_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = {"sources": [{"source_region": {"status": "INCONCLUSIVE"}}], "eligible_run_ids": [],
                        "preimage_unavailable_run_ids": [], "candidates": [], "search": self.settings(),
                        "shared_integer_bit_floors": None, "identity": {}}
            with patch("scripts.search_ssv_qif.check_candidate") as verifier:
                result = execute(manifest, Path(temp))
                verifier.assert_not_called()
            self.assertEqual(result["status"], "SOURCE_PROPERTY_INCONCLUSIVE")
            self.assertIsNone(result["selected"])


if __name__ == "__main__":
    unittest.main()
