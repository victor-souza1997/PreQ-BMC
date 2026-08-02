from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from reports.experiment_summary import build_experiment_summary
from synthesis.preqbmc import GPEncoding


def _encoder(output_low: list[float], output_high: list[float]) -> GPEncoding:
    encoder = object.__new__(GPEncoding)
    encoder.targetCls = 0
    encoder.eps = 0.01
    encoder.output_layer = SimpleNamespace(
        lb=np.asarray(output_low, dtype=np.float64),
        ub=np.asarray(output_high, dtype=np.float64),
    )
    encoder._stats = {
        "encoding_time": 0.0,
        "solving_time": 0.0,
        "backward_time": 0.0,
        "forward_time": 0.0,
        "total_time": 0.0,
        "esbmc_calls": 0.0,
        "esbmc_block_calls": 0.0,
    }
    encoder.source_region_record = {}
    encoder.synthesis_final_status = "UNKNOWN"
    encoder.esbmc_call_records = []
    encoder.verify_mode = "esbmc"
    encoder.harness_scope = "network"
    encoder.config = SimpleNamespace(no_gurobi=False, save_preimage_cache=False)
    encoder.end_to_end_record = {"enabled": True, "status": "NOT_RUN"}
    encoder.assert_input_box = lambda _lb, _ub: None
    encoder.symbolic_propagate = lambda: None
    encoder._widen_internal_integer_bits_for_fixed_point_contracts = lambda: None
    encoder._network_lower_bound_configuration = lambda: ([8], [4], [3])
    return encoder


class SourceRegionGateTest(unittest.TestCase):
    def test_inconclusive_source_region_stops_before_esbmc(self) -> None:
        encoder = _encoder([0.2, 0.0], [0.4, 0.5])
        esbmc_called = False

        def verify(*_args: object) -> bool:
            nonlocal esbmc_called
            esbmc_called = True
            return True

        encoder._verify_network_end_to_end = verify

        result = encoder.run(np.asarray([0.0]), np.asarray([1.0]))

        self.assertFalse(result.success)
        self.assertEqual(result.final_status, "SOURCE_PROPERTY_INCONCLUSIVE")
        self.assertFalse(esbmc_called)
        source = encoder.source_region_summary()
        self.assertEqual(source["method"], "deeppoly")
        self.assertEqual(source["status"], "INCONCLUSIVE")
        self.assertFalse(source["eligible_for_transfer"])
        self.assertFalse(source["quantized_pipeline_started"])
        self.assertFalse(source["esbmc_attempted"])
        self.assertAlmostEqual(source["certified_margin_lower_bound"], -0.3)
        self.assertIn("not a counterexample", source["interpretation"])

    def test_verified_source_region_continues_to_esbmc(self) -> None:
        encoder = _encoder([0.8, 0.0], [1.0, 0.5])

        def verify(*_args: object) -> bool:
            encoder.esbmc_call_records.append({"status": "VERIFIED"})
            encoder.end_to_end_record["status"] = "VERIFIED"
            return True

        encoder._verify_network_end_to_end = verify

        result = encoder.run(np.asarray([0.0]), np.asarray([1.0]))

        self.assertTrue(result.success)
        self.assertEqual(result.final_status, "VERIFIED")
        source = encoder.source_region_summary()
        self.assertEqual(source["status"], "VERIFIED")
        self.assertTrue(source["eligible_for_transfer"])
        self.assertTrue(source["quantized_pipeline_started"])
        self.assertTrue(source["esbmc_attempted"])

    def test_inconclusive_source_summary_uses_skipped_contract_status(self) -> None:
        pipeline = {
            "dataset": "iris",
            "arch": "1blk_10",
            "sample_id": 1,
            "eps": 0.01,
            "synthesis": {
                "success": False,
                "total_bits": [],
                "fractional_bits": [],
                "integer_bits": [],
                "stats": {},
                "final_status": "SOURCE_PROPERTY_INCONCLUSIVE",
            },
            "final_status": "SOURCE_PROPERTY_INCONCLUSIVE",
            "source_region": {
                "method": "deeppoly",
                "status": "INCONCLUSIVE",
                "eligible_for_transfer": False,
                "quantized_pipeline_started": False,
                "esbmc_attempted": False,
            },
            "end_to_end_verification": {"enabled": False, "status": "NOT_RUN"},
            "baseline": {"reference_accuracy": 0.9},
        }

        summary = build_experiment_summary(
            pipeline_summary=pipeline,
            formal_metrics=None,
            refined_metrics=None,
            formal_resource_metrics=None,
            refined_resource_metrics=None,
            external_baselines=[],
            artifacts={},
        )

        for method in ("formal_only", "quality_refined"):
            section = summary[method]
            self.assertEqual(section["final_status"], "SOURCE_PROPERTY_INCONCLUSIVE")
            self.assertEqual(section["contract_status"], "SKIPPED")
            self.assertEqual(section["no_saturation_status"], "SKIPPED")
            self.assertEqual(section["guarantee_level"], "unknown")
        self.assertEqual(summary["source_region"]["status"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
