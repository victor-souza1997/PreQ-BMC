import json
import tempfile
import unittest
from pathlib import Path

from scripts.select_ssv_fixed_qif_sweep import select


class FixedQifSweepSelectionTest(unittest.TestCase):
    def _write_report(self, root, candidate, run_id, artifact, source_status, verified):
        path = root / candidate / run_id
        path.mkdir(parents=True)
        (path / "region_summary.json").write_text(json.dumps({
            "run_id": run_id,
            "model_sha256": "source-model",
            "verification_mode": "fixed_qif_check",
            "fixed_artifact_sha256": artifact,
            "source_region": {"status": source_status},
            "byte_crop_property_verified": verified,
        }))

    def test_selection_only_requires_source_eligible_calibration_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            studies = {}
            for candidate, artifact in (("small", "artifact-small"), ("large", "artifact-large")):
                study = root / f"{candidate}.json"
                study.write_text(json.dumps({"runs": [{"run_id": run_id} for run_id in ("eligible-a", "eligible-b", "inconclusive")]}))
                studies[candidate] = study
            sweep = {
                "schema": "ssv_fixed_qif_sweep_materialized_v1",
                "model_sha256": "source-model",
                "selection": {"rule": "first_ordered_candidate_all_source_eligible_calibration_regions_verified"},
                "evaluation_study": str(studies["small"]),
                "candidates": [
                    {"id": "small", "qif": [], "study": str(studies["small"]), "artifact_sha256": "artifact-small"},
                    {"id": "large", "qif": [], "study": str(studies["large"]), "artifact_sha256": "artifact-large"},
                ],
            }
            sweep_path = root / "sweep.json"
            sweep_path.write_text(json.dumps(sweep))
            self._write_report(runs, "small", "eligible-a", "artifact-small", "VERIFIED", True)
            self._write_report(runs, "small", "eligible-b", "artifact-small", "VERIFIED", False)
            self._write_report(runs, "small", "inconclusive", "artifact-small", "INCONCLUSIVE", False)
            self._write_report(runs, "large", "eligible-a", "artifact-large", "VERIFIED", True)

            result = select(sweep_path, runs, root / "selection")

            self.assertEqual(result["source_eligibility_anchor"]["source_eligible_run_ids"], ["eligible-a", "eligible-b"])
            self.assertEqual(result["candidates"][0]["verified_regions"], 1)
            self.assertEqual(result["candidates"][1]["missing_run_ids"], ["eligible-b"])
            self.assertEqual(result["evaluation_status"], "NO_CANDIDATE_PASSED_CALIBRATION")


if __name__ == "__main__":
    unittest.main()
