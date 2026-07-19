from __future__ import annotations

import unittest
from typing import Any

from reports.experiment_summary import build_experiment_summary


def _layer(
    *,
    contract_status: str = "VERIFIED",
    no_saturation_status: str = "SKIPPED",
    no_saturation_verified: bool = False,
) -> dict[str, Any]:
    return {
        "layer_index": 0,
        "total_bits": 8,
        "integer_bits": 3,
        "fractional_bits": 4,
        "status": contract_status,
        "contract_status": contract_status,
        "contract_verified": contract_status == "VERIFIED",
        "no_saturation_formally_checked": no_saturation_status != "SKIPPED",
        "no_saturation_status": no_saturation_status,
        "no_saturation_verified": no_saturation_verified,
        "deployment_quality_accepted": True,
        "final_status": "VERIFIED" if contract_status == "VERIFIED" else contract_status,
    }


def _summary(
    *,
    layers: list[dict[str, Any]],
    quality_accepted: bool = True,
    chaining_ok: bool = True,
    chaining_enforced: bool = True,
    soundness: str = "strict",
    clamp_in_contract_harnesses: bool = True,
    no_saturation_required: bool = False,
    no_saturation_continue_on_unknown: bool = False,
    include_contract_harness_semantics: bool = True,
) -> dict[str, Any]:
    pipeline_summary = {
        "dataset": "iris",
        "base_dataset": "iris",
        "arch": "1blk_10",
        "sample_id": 0,
        "eps": 0.01,
        "compare_split": "test",
        "synthesis": {
            "success": quality_accepted,
            "total_bits": [8],
            "integer_bits": [3],
            "fractional_bits": [4],
            "stats": {},
        },
        "quality_refinement": {
            "enabled": True,
            "accepted": quality_accepted,
            "steps": [{"esbmc": {"layers": layers}}],
        },
        "comparison": {
            "python_c_integer_comparison": {"exact_match": True},
        },
        "formal_saturation_verification": {
            "enabled": True,
            "required_for_acceptance": no_saturation_required,
            "layers": layers,
        },
        "resource_controls": {
            "no_saturation_continue_on_unknown": no_saturation_continue_on_unknown,
        },
        "soundness": soundness,
        "chaining_ok": {
            "all_ok": chaining_ok,
            "enforced": chaining_enforced,
        },
    }
    if include_contract_harness_semantics:
        pipeline_summary["contract_harness_semantics"] = {
            "uses_shared_deployed_arithmetic_kernel": True,
            "clamp_in_contract_harnesses": clamp_in_contract_harnesses,
            "no_saturation_required_for_deployed_transfer": no_saturation_required,
        }
    return build_experiment_summary(
        pipeline_summary=pipeline_summary,
        formal_metrics=None,
        refined_metrics=None,
        formal_resource_metrics=None,
        refined_resource_metrics=None,
        external_baselines=[],
        artifacts={},
    )


class ExperimentSummaryGuaranteeLevelTest(unittest.TestCase):
    def test_clamped_contracts_can_claim_deployed_transfer_without_no_saturation(self) -> None:
        summary = _summary(layers=[_layer()])

        self.assertEqual(summary["guarantee_level"], "deployed-transfer")
        self.assertEqual(summary["quality_refined"]["guarantee_level"], "deployed-transfer")
        self.assertFalse(summary["transfer_preconditions"]["no_saturation_required"])

    def test_verified_harness_with_broken_chaining_is_not_deployed_transfer(self) -> None:
        summary = _summary(
            layers=[_layer()],
            chaining_ok=False,
            chaining_enforced=False,
            soundness="degraded",
        )

        self.assertEqual(summary["guarantee_level"], "harness-verified")
        self.assertFalse(summary["transfer_preconditions"]["chaining_ok"])

    def test_clamp_free_harness_requires_verified_no_saturation_for_transfer(self) -> None:
        missing_no_sat = _summary(
            layers=[_layer(no_saturation_status="UNKNOWN", no_saturation_verified=False)],
            clamp_in_contract_harnesses=False,
            no_saturation_required=True,
        )
        verified_no_sat = _summary(
            layers=[_layer(no_saturation_status="VERIFIED", no_saturation_verified=True)],
            clamp_in_contract_harnesses=False,
            no_saturation_required=True,
        )

        self.assertEqual(missing_no_sat["guarantee_level"], "harness-verified")
        self.assertEqual(verified_no_sat["guarantee_level"], "deployed-transfer")

    def test_continue_on_unknown_never_yields_deployed_transfer(self) -> None:
        summary = _summary(
            layers=[_layer(no_saturation_status="VERIFIED", no_saturation_verified=True)],
            no_saturation_continue_on_unknown=True,
        )

        self.assertEqual(summary["guarantee_level"], "harness-verified")
        self.assertTrue(summary["transfer_preconditions"]["no_saturation_continue_on_unknown"])

    def test_missing_contract_semantics_never_yields_deployed_transfer(self) -> None:
        summary = _summary(
            layers=[_layer()],
            include_contract_harness_semantics=False,
        )

        self.assertEqual(summary["guarantee_level"], "harness-verified")
        self.assertFalse(summary["transfer_preconditions"]["fidelity_by_construction"])

    def test_contract_failure_sets_failed_guarantee_level(self) -> None:
        summary = _summary(layers=[_layer(contract_status="FAILED")])

        self.assertEqual(summary["guarantee_level"], "failed")

    def test_missing_contract_verification_sets_unknown_guarantee_level(self) -> None:
        summary = _summary(layers=[_layer(contract_status="UNKNOWN")])

        self.assertEqual(summary["guarantee_level"], "unknown")


if __name__ == "__main__":
    unittest.main()
