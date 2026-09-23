"""Demand-driven relational refinement control for convolution proofs.

This module controls soundness, not cut synthesis.  A proposer may use MILP or
abstract interpretation, but a proposed relation is never available to a retry
until a caller-supplied ESBMC validator reports VERIFIED.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class SparseRelationalCut:
    layer_index: int
    indices: tuple[int, ...]
    coefficients: tuple[int, ...]
    scale: int
    lower: int
    upper: int
    provenance: str = "candidate_only_until_esbmc_verified"

    def __post_init__(self) -> None:
        if (not self.indices or len(self.indices) != len(self.coefficients)
                or len(set(self.indices)) != len(self.indices)
                or any(index < 0 for index in self.indices)
                or self.scale <= 0 or self.lower > self.upper):
            raise ValueError("Invalid sparse relational cut")


@dataclass(frozen=True)
class CEGARConfig:
    max_rounds: int = 3
    max_cuts_per_competitor: int = 4

    def __post_init__(self) -> None:
        if self.max_rounds < 0 or self.max_cuts_per_competitor <= 0:
            raise ValueError("Invalid CEGAR budget")


class DemandDrivenMarginCEGAR:
    """Refine only failed output competitors, never all classes eagerly."""

    def __init__(
        self,
        *,
        propose: Callable[[int, int, list[SparseRelationalCut]], Iterable[SparseRelationalCut]],
        validate_with_esbmc: Callable[[SparseRelationalCut], dict[str, Any]],
        retry_margin_with_esbmc: Callable[[int, list[SparseRelationalCut]], dict[str, Any]],
        config: CEGARConfig | None = None,
    ):
        self.propose = propose
        self.validate_with_esbmc = validate_with_esbmc
        self.retry_margin_with_esbmc = retry_margin_with_esbmc
        self.config = config or CEGARConfig()

    def refine(self, initial_margin_results: dict[int, dict[str, Any]]):
        records = []
        final = dict(initial_margin_results)
        unresolved = [
            competitor for competitor, result in initial_margin_results.items()
            if result.get("status") != "VERIFIED"
        ]
        for competitor in unresolved:
            accepted: list[SparseRelationalCut] = []
            for round_index in range(self.config.max_rounds):
                candidates = list(self.propose(competitor, round_index, list(accepted)))
                candidates = candidates[: self.config.max_cuts_per_competitor]
                if not candidates:
                    break
                newly_verified = []
                for cut in candidates:
                    proof = self.validate_with_esbmc(cut)
                    record = {
                        "competitor": competitor,
                        "round": round_index,
                        "candidate": asdict(cut),
                        "validation": proof,
                        "assumed_by_retry": proof.get("status") == "VERIFIED",
                    }
                    records.append(record)
                    if proof.get("status") == "VERIFIED":
                        newly_verified.append(cut)
                accepted.extend(newly_verified)
                if not newly_verified:
                    break
                retry = self.retry_margin_with_esbmc(competitor, list(accepted))
                records.append({
                    "competitor": competitor,
                    "round": round_index,
                    "validated_cut_count": len(accepted),
                    "margin_retry": retry,
                })
                final[competitor] = retry
                if retry.get("status") == "VERIFIED":
                    break
        all_verified = bool(final) and all(
            result.get("status") == "VERIFIED" for result in final.values()
        )
        return {
            "strategy": "demand_driven_esbmc_validated_sparse_relations",
            "status": "VERIFIED" if all_verified else "ABSTRACTION_INCONCLUSIVE",
            "all_competitors_verified": all_verified,
            "initial_unresolved_competitors": unresolved,
            "final_margin_results": final,
            "records": records,
            "soundness_rule": "only ESBMC-VERIFIED cuts may be assumed by a margin retry",
        }
