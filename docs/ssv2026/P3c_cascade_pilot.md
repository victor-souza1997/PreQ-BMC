# P3c: slope search in the layer-2 and layer-3 bound chains, cascaded (worktree `feat/crown-v2-qformat`, 2026-10-31)

Uncommitted. Development images 013, 016 and 017 (control). No ESBMC runs; every number is a finder screen (SCREEN_BOUND_UNCHECKED). It follows from `P3b_bound_slope_pilot.md`.

## Specification (written before the run)

- **Script.** `src/scripts/p3c_cascade_pilot.py`, standalone. It uses the import guard and changes no production file. Its inputs are the same hash-checked diagnostic run as P3b (`output/crown_diagnostics_20261031T161801Z/`).
- **Per layer L ∈ {2, 3}.** For every side of every neuron that is unstable in the current selected bounds (`l < 0 < u`, not saturated, not clipped), the unit chain is computed under three policies:
  - the baseline ADAPTIVE heuristic;
  - arm A, the production `search_policies()`;
  - arm B, the I-S single-coordinate family on the top 8 ranked coordinates.
  The best chain value per side becomes that side's chain candidate.
- **Bound selection.** The new bounds of layer L use exactly the `bounds_v2` selection rule (box lower for stable-active neurons; box or chain upper by that rule). Each side keeps one source.
  - Audit: re-applying the rule to the cached chain values must reproduce the cached layer-2 bounds exactly.
- **Cascade.** Layer 2 is tightened first. Then every chain-sourced side of layer 3 is recomputed under the new layer-2 bounds, ADAPTIVE for all sides, plus arms A and B for the unstable ones. Cached layer-3 chain values are **not** reused, because they cite layer-2 facts that no longer exist. A side may come out looser; it is taken as it comes.
- **Configurations measured.** All 42 margins with ADAPTIVE margin chains, for each of:
  - baseline;
  - P3b (layer 3 only), which must reproduce the P3b primary margins (audit);
  - layer 2 only (layer 3 recomputed ADAPTIVE, no arms);
  - cascade.
  A secondary arm adds P3 margin search to the cascade; it is reported, but it is not the gate.
- **Soundness checks (stop on failure).**
  - Every chain value and every selected bound of layers 2 and 3 must hold at the nominal input and at every retained witness.
  - Exact slack accounting (`account_chain`) of every focus margin chain in the cascade configuration must pass at those witnesses.
- **Gate (vs the P3b result).**
  - Cascade primary margin 013 c40 ≥ −947 (P3b −997, +50), or 016 c5 ≥ −648 (P3b −683, +35). Each is 5 % of the remaining gap.
  - Or any image with all 42 competitors closed.
  - The control 017 must keep 42/42 closed; a loss is flagged.
- **If the gate passes.** Cost measurement for the cascade, as in P3b.
- **If it fails.** Stop the finder line and record the negative.
- **Budget.** 2 h wall time and 8 GiB of memory. An unfinished side is NOT_RUN_BUDGET, never NO_GAIN.

## Result (2026-10-31, MEASURED screen): gate passed by 1 unit on 016; no image recovered

Run: `output/p3c_cascade_pilot_20261031T192435Z/` (3,290 s; no audit or budget stop). Checks that held:
- the P3b reproduction is exact on all three images;
- every chain value and every selected layer-2/3 bound held at the nominal input and every retained witness;
- exact slack accounting of every focus margin chain in the cascade passed.

The smoke test `…smoke017_*` carries no gate.

| Image | Primary: baseline / P3b / L2 only / cascade / + margin search | Closed (cascade + search) | Layer-2 width (targets) | Layer-3 width (targets) |
|---|---|---|---|---|
| 013 c40 | −997 / −997 / −976 / −991 / −991 | 37 | 240,540 → 239,087 (−0.6 %) | 141,052 → 121,404 |
| 016 c5 | −876 / −683 / −843 / **−647** / −647 | 38 | 332,619 → 329,727 (−0.9 %) | 181,887 → 153,858 |
| 017 c37 | 352 / 386 / 354 / 388 / 388 | 42 | 39,129 → 38,914 | 21,109 → 19,586 |

- **Gate.** 016 reached −647 against a threshold of −648, so it passes by **one unit**. 013 (−991 vs −947) does not. 017 kept 42/42.
- **Layer 2 is not worth it.** Its bounds tighten by under 1 %. The search changed 643–1,068 sides per image, but each by very little, so the cascade adds only 6 (013) and 36 (016) units over P3b.
- **What remains open on 016 with margin search:** c31 −713, c1 −705, c5 −647, c21 −252. On 013: c40 −991, c36 −976, c34 −363, c37 −290, c35 −64.

**Reading (ASSUMED).** The rule says PASS, but only by its smallest possible margin, and the effect is clearly diminishing: P3b gave +193 on 016, the cascade a further +36. Slope search in the bound chains is exhausted as a route to recovering 013/016. The remaining gaps (650–1,000 units) need a different kind of precision (for example input splitting or relational constraints, spec rows I-P / I-R), not more of the same. Recommendation: stop the finder line here and record P3b/P3c as a partial, sound, cost-neutral improvement.
