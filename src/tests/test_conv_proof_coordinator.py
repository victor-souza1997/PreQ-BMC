import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from backends.conv_fixed_point import quantize_restricted_sequential
from backends.fixed_point import LayerQuantizationSpec
from tests.test_conv_native_verification import tiny_sequential_model
from verification.conv_contracts import IntegerInvariant
from verification.conv_proof import ConvProofConfig, ConvProofCoordinator, ESBMCProofStore
from verification.esbmc import ESBMCConfig


def verified_record(_self, _source, name, metadata):
    return {
        "digest": name,
        "status": "VERIFIED",
        "harness": name + ".c",
        "cache_reused": False,
        "identity": metadata,
        "command": ["esbmc"],
        "elapsed_seconds": 0.0,
        "return_code": 0,
        "resource_control": {},
    }


class ConvProofCoordinatorTest(unittest.TestCase):
    def setUp(self):
        specs = [LayerQuantizationSpec(8, 3, 4)] * 3
        self.network = quantize_restricted_sequential(
            tiny_sequential_model(), specs, input_fractional_bits=4, input_total_bits=8
        )
        self.initial = IntegerInvariant((0,) * 9, (0,) * 9, (3, 3, 1))

    def test_verified_cache_identity_reuses_only_identical_obligation(self):
        class FakeRunner:
            def __init__(self):
                self.config = SimpleNamespace(
                    executable="esbmc", timeout_seconds=7, memlimit="1g"
                )
                self.calls = 0

            def run_file(self, _harness, profile):
                self.calls += 1
                return SimpleNamespace(
                    status="VERIFIED", command=("esbmc", profile),
                    elapsed_seconds=0.1, return_code=0, resource_control={},
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = FakeRunner()
            with patch("verification.conv_proof.resolve_esbmc_executable", return_value="/bin/true"):
                first = ESBMCProofStore(
                    root / "first", runner, "paper-z3", cache_directory=root / "shared"
                )
                first.check("int main(void){return 0;}", "same", {"property_type": "p"})
                second = ESBMCProofStore(
                    root / "second", runner, "paper-z3", cache_directory=root / "shared"
                )
                reused = second.check(
                    "int main(void){return 0;}", "same", {"property_type": "p"}
                )
                changed = second.check(
                    "int main(void){return 1;}", "changed", {"property_type": "p"}
                )
        self.assertEqual(runner.calls, 2)
        self.assertTrue(reused["cache_reused"])
        self.assertFalse(changed["cache_reused"])

    @patch.object(ESBMCProofStore, "check", verified_record)
    def test_bounded_pilot_can_never_be_certified(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = ConvProofCoordinator(
                self.network, Path(directory) / "proof",
                esbmc=ESBMCConfig(timeout_seconds=1),
                config=ConvProofConfig(block_size=1, max_blocks_per_layer=1),
            )
            summary = coordinator.verify(
                self.initial, input_witness=np.zeros(9, dtype=int), target_class=0
            )
        self.assertEqual(summary["final_status"], "PARTIAL_NOT_CERTIFIED")
        self.assertFalse(summary["certified"])
        self.assertFalse(summary["all_required_blocks_executed"])
        self.assertFalse(any(
            record["identity"].get("property_type") == "strict_output_margin"
            for record in summary["obligations"]
        ))

    @patch.object(ESBMCProofStore, "check", verified_record)
    def test_complete_verified_ledger_requires_margin_and_bridges(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = ConvProofCoordinator(
                self.network, Path(directory) / "proof",
                esbmc=ESBMCConfig(timeout_seconds=1),
                config=ConvProofConfig(block_size=2),
            )
            summary = coordinator.verify(
                self.initial, input_witness=np.zeros(9, dtype=int), target_class=0
            )
        self.assertEqual(summary["final_status"], "VERIFIED")
        kinds = [record["identity"]["property_type"] for record in summary["obligations"]]
        self.assertIn("input_nonvacuity", kinds)
        self.assertIn("integer_layer_contract", kinds)
        self.assertIn("contract_chaining", kinds)
        self.assertEqual(kinds[-1], "strict_output_margin")

    def test_parallel_mode_batches_only_independent_blocks(self):
        batches = []

        def verified_many(_store, obligations, *, jobs, min_available_gib):
            batches.append((jobs, min_available_gib, [row[2] for row in obligations]))
            return [
                verified_record(_store, source, name, metadata)
                for source, name, metadata in obligations
            ]

        with patch.object(ESBMCProofStore, "check", verified_record), \
                patch.object(ESBMCProofStore, "check_many", verified_many):
            with tempfile.TemporaryDirectory() as directory:
                summary = ConvProofCoordinator(
                    self.network, Path(directory) / "proof",
                    config=ConvProofConfig(
                        block_size=2,
                        jobs=4,
                        min_available_gib=6.0,
                        fail_fast=False,
                    ),
                ).verify(
                    self.initial,
                    input_witness=np.zeros(9, dtype=int),
                    target_class=0,
                )
        self.assertEqual(summary["final_status"], "VERIFIED")
        self.assertTrue(batches)
        self.assertTrue(all(jobs == 4 and guard == 6.0 for jobs, guard, _ in batches))
        self.assertTrue(all(
            metadata["property_type"] == "integer_layer_contract"
            for _, _, batch in batches for metadata in batch
        ))
        self.assertEqual(summary["resource_scheduling"]["jobs"], 4)
        self.assertTrue(summary["resource_scheduling"]["dependency_barriers_preserved"])

    @patch.object(ESBMCProofStore, "check", verified_record)
    def test_proof_carrying_mode_requires_lemmas_certificates_and_margins(self):
        with tempfile.TemporaryDirectory() as directory:
            coordinator = ConvProofCoordinator(
                self.network, Path(directory) / "proof",
                esbmc=ESBMCConfig(timeout_seconds=1),
                config=ConvProofConfig(block_size=2, proof_mode="proof_carrying_interval"),
            )
            summary = coordinator.verify(
                self.initial, input_witness=np.zeros(9, dtype=int), target_class=0
            )
        self.assertEqual(summary["final_status"], "VERIFIED")
        self.assertEqual(summary["proof_mode"], "proof_carrying_interval")
        kinds = [record["identity"]["property_type"] for record in summary["obligations"]]
        self.assertIn("fixed_point_arithmetic_lemma", kinds)
        self.assertIn("integer_interval_certificate", kinds)
        self.assertIn("contract_chaining", kinds)
        self.assertEqual(kinds[-1], "strict_output_margin_interval")

    def test_failed_interval_margin_can_only_be_replaced_by_verified_exact_refinement(self):
        def interval_fails_exact_verifies(_store, _source, name, metadata):
            status = ("FAILED" if metadata["property_type"]
                      == "strict_output_margin_interval" else "VERIFIED")
            return {
                "digest": name, "status": status, "harness": name + ".c",
                "cache_reused": False, "identity": metadata, "command": ["esbmc"],
                "elapsed_seconds": 0.0, "return_code": 1 if status == "FAILED" else 0,
                "resource_control": {},
            }

        with patch.object(ESBMCProofStore, "check", interval_fails_exact_verifies):
            with tempfile.TemporaryDirectory() as directory:
                coordinator = ConvProofCoordinator(
                    self.network, Path(directory) / "proof",
                    config=ConvProofConfig(
                        block_size=2, proof_mode="proof_carrying_interval",
                        exact_margin_refinement=True,
                    ),
                )
                summary = coordinator.verify(
                    self.initial, input_witness=np.zeros(9, dtype=int), target_class=0
                )
        self.assertEqual(summary["final_status"], "VERIFIED")
        self.assertEqual(summary["refinement"]["triggered_competitors"], [1])
        diagnostics = [record for record in summary["obligations"]
                       if record["identity"]["property_type"]
                       == "strict_output_margin_interval"]
        self.assertEqual(diagnostics[0]["status"], "FAILED")
        self.assertFalse(diagnostics[0]["required_for_certificate"])

    def test_failed_diagnostic_margins_cannot_be_promoted_when_fail_fast_is_off(self):
        def failed_abstract_witness(_store, _source, name, metadata):
            kind = metadata["property_type"]
            status = "FAILED" if kind in {
                "strict_output_margin_interval",
                "strict_output_margin_abstraction_witness",
            } else "VERIFIED"
            record = verified_record(_store, _source, name, metadata)
            record["status"] = status
            record["return_code"] = 1 if status == "FAILED" else 0
            return record

        with patch.object(ESBMCProofStore, "check", failed_abstract_witness):
            with tempfile.TemporaryDirectory() as directory:
                summary = ConvProofCoordinator(
                    self.network, Path(directory) / "proof",
                    config=ConvProofConfig(
                        block_size=2,
                        proof_mode="proof_carrying_interval",
                        exact_margin_refinement=True,
                        fail_fast=False,
                    ),
                ).verify(
                    self.initial,
                    input_witness=np.zeros(9, dtype=int),
                    target_class=0,
                )
        self.assertEqual(summary["final_status"], "ABSTRACTION_INCONCLUSIVE")
        self.assertFalse(summary["certified"])
        self.assertFalse(summary["all_required_margins_verified"])

    def test_exact_refinement_timeout_remains_timeout(self):
        def timeout_exact(_store, _source, name, metadata):
            kind = metadata["property_type"]
            status = ("FAILED" if kind == "strict_output_margin_interval"
                      else "TIMEOUT" if kind == "strict_output_margin_exact_refinement"
                      else "VERIFIED")
            return {
                "digest": name, "status": status, "harness": name + ".c",
                "cache_reused": False, "identity": metadata, "command": ["esbmc"],
                "elapsed_seconds": 0.0, "return_code": 1, "resource_control": {},
            }

        with patch.object(ESBMCProofStore, "check", timeout_exact):
            with tempfile.TemporaryDirectory() as directory:
                summary = ConvProofCoordinator(
                    self.network, Path(directory) / "proof",
                    config=ConvProofConfig(
                        block_size=2, proof_mode="proof_carrying_interval",
                        exact_margin_refinement=True,
                    ),
                ).verify(self.initial, input_witness=np.zeros(9, dtype=int), target_class=0)
        self.assertEqual(summary["final_status"], "TIMEOUT")
        self.assertFalse(summary["certified"])

    def test_failure_stops_and_cannot_be_promoted(self):
        calls = []

        def fail_first_block(_store, _source, name, metadata):
            calls.append(metadata["property_type"])
            status = "FAILED" if metadata["property_type"] == "integer_layer_contract" else "VERIFIED"
            return {
                "digest": name, "status": status, "harness": name + ".c",
                "cache_reused": False, "identity": metadata, "command": ["esbmc"],
                "elapsed_seconds": 0.0, "return_code": 1 if status == "FAILED" else 0,
                "resource_control": {},
            }

        with patch.object(ESBMCProofStore, "check", fail_first_block):
            with tempfile.TemporaryDirectory() as directory:
                coordinator = ConvProofCoordinator(
                    self.network, Path(directory) / "proof",
                    config=ConvProofConfig(block_size=1, fail_fast=True),
                )
                summary = coordinator.verify(
                    self.initial, input_witness=np.zeros(9, dtype=int), target_class=0
                )
        self.assertEqual(summary["final_status"], "FAILED")
        self.assertFalse(summary["certified"])
        self.assertEqual(calls, ["input_nonvacuity", "integer_layer_contract"])


if __name__ == "__main__":
    unittest.main()
