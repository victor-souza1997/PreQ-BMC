# P3: bounded finder pilot (worktree `feat/crown-v2-qformat`, 2026-10-31)

Uncommitted. Development images only: the 19 correctly classified October validation images (000 is MISCLASSIFIED). No ESBMC runs in P3.

## Specification (written before the run)

- **Finder change** (`src/scripts/crown_finder/crown_chain_export_v2.py`):
  - **Policies.** `relu_choice()` takes a per-row policy: ADAPTIVE (the existing coordinatewise heuristic) or `p ∈ {0..4}`.
    - A scalar policy sets `a' = (p·a + 2) // 4` only on unstable (`l < 0 < u`), unsaturated, unclipped coordinates with `a > 0`.
    - Negative coefficients, stable, saturated and clipped coordinates keep the heuristic.
  - **One policy table.** `run_chain()`, `needs()` and `emit()` all receive the same immutable table.
  - **Search.** `search_policies()` only touches margin roots with a baseline margin < 1.
    - Every layer starts ADAPTIVE. The search visits layers 3, 2, 1, 0 once each and tries the 5 scalars.
    - It switches only on a strict improvement of the exact scaled bound. Ties go to fewer terms, then the lower p.
  - **What stays baseline.** Intermediate bounds, logit roots and supporting chains.
  - **Export (`--policy`).** It replays the screened table, refuses it unless it reproduces every screened root bound exactly, and checks that every exported margin chain equals its screened margin. The policies are written to `finder_policy.json` as provenance; the checker never reads it.
- **Cost (`--cost`).** Counts the whole planned dependency graph: roots with their policies, every supporting chain, multiply terms and two-sided ReLU coordinates.
- **Budget.** 2 h wall time for the whole pilot. A margin not finished by then is NOT_RUN_PILOT_BUDGET.
- **Reserved checking budget (MEASURED basis).**
  - The October positives cost 298–330 µs/term.
  - The largest, 006, took 22.63 CPU-h.
  - The estimated cost of a candidate is terms × 330 µs. A recovered image is rejected if this exceeds 1.5 × 22.63 = 33.9 CPU-h.
- **Gate.** At least 1 of the 13 baseline FINDER_INCONCLUSIVE images recovered within budget, and 0 of the 6 baseline positives (002, 006, 008, 011, 012, 017) lost.

## Controls run before the pilot

- 006, no search: margin_min 1657, 42/42 closed, 5,639 chains. This is identical to the October screen.
- 013, search + cost: runs end to end. On all 5 open margins every scalar policy was worse than ADAPTIVE at every layer, so nothing changed (margin_min −997).

## Result (2026-10-31, MEASURED): gate NOT met, P3 stops

Run: `output/p3_finder_pilot_20261031/` (driver `run_pilot.sh` sha256 68dfe243…, finder sha256 1541896e…). 19 images in 29 min of the 2 h cap; no NOT_RUN_PILOT_BUDGET.

- **Recovered: 0 of 13.** No inconclusive image gained a single closed competitor.
- **Lost: 0 of 6.** Positives are untouched (nothing searched; margins identical).
- **Where the search moved anything:** only on far-negative images, whose margin_min improved by 2–10 %: 003 (−63,487 → −56,860), 005, 007, 014, 019. These stay thousands of units short of 1.
- **Near-miss images (013, 016, 018, 009, 004, 010):** ADAPTIVE beat every scalar policy at every layer on every open margin. Nothing changed.
- **Cost counter cross-check:** for 006, 008, 011, 012 and 017 the `--cost` term count equals the October export's `multiply_terms` exactly.

**Interpretation (ASSUMED).** With intermediate bounds frozen at the baseline, a single per-layer slope cannot beat the coordinatewise heuristic where it matters. The gap of the near misses (−876 to −5,114) comes from the width of the intermediate bounds, not from the margin's final back-substitution. Tightening those would be a different variant (intermediate-bound optimization), with a much larger cost; it was explicitly out of scope for this pilot.

**Consequence.** Per the agreed plan, the improvement branch stops here and the negative is reported. P4/P5 do not proceed on this finder variant. P1/P2 stand independently.

## Provenance limitation (added 2026-10-31, after the run)

The P3 driver launched the finder with working directory `/home/joao/code/preqbmc` (the main checkout). Two mechanisms put that checkout's `src/` on the import path:

- `affine_export.py` runs `sys.path.insert(0, "src")`, which resolves against the working directory;
- the `quad` conda env has an editable install (`__editable__.preqbmc-0.1.0.pth`) pointing to `/home/joao/code/preqbmc/src`.

As a result, only the four `crown_finder` modules (including the modified `crown_chain_export_v2.py`, sha256 `1541896e…`) were loaded from this worktree. `backends.*`, `verification.*`, `datasets.*`, `models.*` and `utils.*` were loaded from the main checkout.

**Check (MEASURED, 2026-10-31).** Every leaked module that the finder imports was compared with its worktree copy, and all were byte-identical. The one differing file in the main checkout, `verification/crown_chain_v2.py`, is not imported by the finder.

The numerical results and conclusions above are therefore unaffected. The limitation is that the P3 run recorded only the finder's hash, not the resolved path and hash of every imported module. The diagnostic driver (`scripts/diagnose_crown_finder.py`) now pins imports to the worktree and records them in its manifest.
