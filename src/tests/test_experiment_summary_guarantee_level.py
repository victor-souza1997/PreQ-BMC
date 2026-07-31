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
    error_budget_mode: str | None = None,
    derived_margin_ok: bool = True,
    vacuity_status: str = "SKIPPED",
    pipeline_final_status: str | None = None,
    end_to_end_status: str | None = None,
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
        "esbmc_memory_metrics": {
            "measurement": "linux_procfs_process_tree_rss",
            "queries_measured": 2,
            "max_query_peak_memory_bytes": 1048576,
            "max_query_peak_memory_mib": 1.0,
        },
        "soundness": soundness,
        "chaining_ok": {
            "all_ok": chaining_ok,
            "enforced": chaining_enforced,
        },
    }
    if pipeline_final_status is not None:
        pipeline_summary["final_status"] = pipeline_final_status
        pipeline_summary["synthesis"]["final_status"] = pipeline_final_status
    if end_to_end_status is not None:
        pipeline_summary["end_to_end_verification"] = {
            "enabled": True,
            "status": end_to_end_status,
            "invariants_injected": True,
        }
    if error_budget_mode is not None:
        pipeline_summary["contract_tolerance"] = {
            "error_budget_mode": error_budget_mode,
        }
        pipeline_summary["output_margin_check"] = {
            "all_ok": derived_margin_ok,
            "status": "VERIFIED" if derived_margin_ok else "MARGIN_TOO_SMALL",
        }
    if vacuity_status != "SKIPPED":
        pipeline_summary["vacuity_check"] = {
            "enabled": True,
            "status": vacuity_status,
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
    def test_copies_esbmc_memory_metrics(self) -> None:
        summary = _summary(layers=[_layer()])

        self.assertEqual(summary["esbmc_memory_metrics"]["queries_measured"], 2)
        self.assertEqual(summary["esbmc_memory_metrics"]["max_query_peak_memory_mib"], 1.0)

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

    def test_derived_budget_transfer_requires_margin_and_nonvacuity(self) -> None:
        verified = _summary(
            layers=[_layer()],
            soundness="derived_budget",
            error_budget_mode="derived",
            derived_margin_ok=True,
            vacuity_status="PASSED",
        )
        small_margin = _summary(
            layers=[_layer()],
            soundness="derived_budget_incomplete",
            error_budget_mode="derived",
            derived_margin_ok=False,
            vacuity_status="PASSED",
        )
        vacuous = _summary(
            layers=[_layer()],
            soundness="derived_budget",
            error_budget_mode="derived",
            derived_margin_ok=True,
            vacuity_status="VACUOUS",
        )

        self.assertEqual(verified["guarantee_level"], "deployed-transfer")
        self.assertEqual(small_margin["guarantee_level"], "harness-verified")
        self.assertEqual(vacuous["guarantee_level"], "harness-verified")

    def test_margin_too_small_has_explicit_terminal_reason(self) -> None:
        summary = _summary(
            layers=[],
            quality_accepted=False,
            soundness="derived_budget_incomplete",
            error_budget_mode="derived",
            derived_margin_ok=False,
            vacuity_status="PASSED",
            pipeline_final_status="MARGIN_TOO_SMALL",
        )

        self.assertEqual(summary["formal_only"]["final_status"], "MARGIN_TOO_SMALL")
        self.assertEqual(
            summary["formal_only"]["final_reason"],
            "derived_output_margin_too_small",
        )
        self.assertEqual(
            summary["quality_refined"]["final_reason"],
            "derived_output_margin_too_small",
        )

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

    def test_end_to_end_verdict_does_not_depend_on_layer_contract_records(self) -> None:
        summary = _summary(
            layers=[],
            end_to_end_status="VERIFIED",
        )

        self.assertEqual(summary["final_status"], "VERIFIED")
        self.assertEqual(summary["contract_status"], "SKIPPED")
        self.assertEqual(summary["end_to_end_status"], "VERIFIED")
        self.assertTrue(summary["python_c_exact_match"])
        self.assertEqual(summary["guarantee_level"], "deployed-transfer")

    def test_end_to_end_timeout_remains_explicit(self) -> None:
        summary = _summary(
            layers=[],
            end_to_end_status="TIMEOUT",
        )

        self.assertEqual(summary["final_status"], "TIMEOUT")
        self.assertEqual(summary["guarantee_level"], "unknown")


if __name__ == "__main__":
    unittest.main()
