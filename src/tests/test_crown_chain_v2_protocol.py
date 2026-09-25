import argparse
import json
from pathlib import Path
import tempfile
import unittest

from scripts import run_crown_chain_v2_protocol as protocol_driver

REPO = protocol_driver.REPO
DRYRUN = REPO / "experiments" / "ssv2026_validation_dryrun_protocol.json"


@unittest.skipUnless(DRYRUN.exists(), "needs the drafted validation dry-run protocol")
class ProtocolGateTests(unittest.TestCase):
    def test_current_sources_match_the_drafted_protocol(self):
        self.assertEqual(protocol_driver.validate_protocol(DRYRUN)["split"], "validation")

    def test_changed_source_hash_is_refused(self):
        protocol = json.loads(DRYRUN.read_text(encoding="utf-8"))
        key = next(iter(protocol["sources"]))
        protocol["sources"][key] = "0" * 64
        with tempfile.TemporaryDirectory(dir=REPO / "experiments") as directory:
            path = Path(directory) / "p.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "sources"):
                protocol_driver.validate_protocol(path)

    def test_uncommitted_test_protocol_is_refused(self):
        protocol = json.loads(DRYRUN.read_text(encoding="utf-8"))
        protocol["split"] = "test"
        with tempfile.TemporaryDirectory(dir=REPO / "experiments") as directory:
            path = Path(directory) / "p.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "committed"):
                protocol_driver.validate_protocol(path)

    def test_test_protocol_cannot_name_its_images(self):
        args = argparse.Namespace(output=Path("/nonexistent/p.json"), split="test", ids=["x"])
        with self.assertRaisesRegex(SystemExit, "draws from --seed"):
            protocol_driver.draft(args)


class TallyTests(unittest.TestCase):
    def _tally(self, outcomes, cutoff):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"split": "test", "epsilon_raw_bytes": 1, "launch_cutoff": cutoff,
                        "images_in_draw_order": [f"img{i}" for i in range(len(outcomes))]}
            path = root / "p.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            out = root / "out"
            out.mkdir()
            (out / "protocol_binding.json").write_text(json.dumps(
                {"protocol": str(path), "protocol_sha256": protocol_driver._sha256(path)}), encoding="utf-8")
            for ordinal, outcome in enumerate(outcomes):
                work = out / f"{ordinal:03d}"
                work.mkdir()
                if outcome is not None:
                    (work / "outcome.json").write_text(json.dumps(
                        {"image": f"img{ordinal}", "outcome": outcome}), encoding="utf-8")
            protocol_driver.tally(argparse.Namespace(protocol=path, output=out))
            return json.loads((out / "results.json").read_text(encoding="utf-8"))

    def test_unstarted_images_count_against_the_rate(self):
        results = self._tally(["VERIFIED", "MISCLASSIFIED", "FINDER_INCONCLUSIVE", None],
                              "2000-01-01T00:00:00+00:00")
        self.assertTrue(results["complete"])
        self.assertEqual(results["counts"]["NOT_RUN_BUDGET"], 1)
        self.assertEqual(results["certified_rate"], 0.25)
        self.assertAlmostEqual(results["certified_rate_among_correct"], 1 / 3)

    def test_before_cutoff_unstarted_images_are_pending(self):
        results = self._tally(["VERIFIED", None], "2999-01-01T00:00:00+00:00")
        self.assertFalse(results["complete"])
        self.assertEqual(results["counts"]["PENDING"], 1)


if __name__ == "__main__":
    unittest.main()
