# P3b: ReLU-slope search inside layer-3 bound chains (worktree `feat/crown-v2-qformat`, 2026-10-31)

Uncommitted. Development images only: 013 and 016 (open near misses) and 017 (control), as in the diagnostics. No ESBMC runs; every number here is a finder screen (SCREEN_BOUND_UNCHECKED).

## Why

- P3 searched ReLU slopes only in the margin chains, with the intermediate bounds frozen. It recovered nothing.
- Diagnostics (`output/crown_diagnostics_20261031T161801Z/`) showed that the loss sits in the eps=1 intermediate bounds. D-0 measured the singleton gap with inherited bounds at 2,600–4,500, against 184–333 with fresh ones.
- I-B on 017 showed that a single slope change inside a layer-3 **bound** chain can tighten that bound a lot (L3:323 upper 142 → 36). No search has ever looked inside the bound chains.

## Specification (written before the run)

- **Script.** `src/scripts/p3b_bound_slope_pilot.py`, a standalone diagnostic. It uses the import guard and does not change any production file.
- **Inputs.** The eps=1 baseline caches, the retained witnesses and the oracle of `output/crown_diagnostics_20261031T161801Z/`, all checked against their recorded sha256.
- **Targets.** Every layer-3 neuron that is unstable in the baseline (`l < 0 < u`, not saturated, not clipped), on both sides. Layers 0–2 keep their baseline bounds.
- **For each side, the unit chain is computed under three policies.**
  - **Baseline.** The ADAPTIVE heuristic. It must reproduce the cached `bounds_v2` chain value exactly, or the run stops.
  - **Arm A.** The production `search_policies()` (the P3 per-layer scalar family) applied to the unit root.
  - **Arm B.** The I-S single-coordinate family on the 8 top-ranked coordinates of the unit chain, using the D-T ranking rule over nominal plus retained witnesses. This is the same code as the I-B attained arm.
- **New endpoint.** The tightest of the three, but only if it is strictly tighter than the baseline selected endpoint. Each side keeps its own chain and policy.
- **Margin.** All 42 margins are recomputed with the new layer-3 bounds and the ADAPTIVE margin policy. A secondary arm adds P3 margin search on top; it is reported, but it is not the gate.
- **Soundness checks (stop on failure).**
  - Every new endpoint must hold at the nominal input and at every retained witness.
  - Exact slack accounting (`account_chain`) of every primary margin chain at those witnesses must give non-negative slacks and the exact identity.
- **Gate.** At least one of the primary margins 013 c40 and 016 c5 must gain at least the material threshold (50 and 44 integer units, i.e. 5 % of the gap).
  - A newly closed competitor is reported, but it is not required.
  - The control 017 must not lose any closed competitor; a loss there is flagged.
- **If the gate passes.** The next step is cost measurement (the extra chains and terms each new bound needs), not export.
- **If it fails.** The finder-improvement line stops and the negative is recorded here.
- **Budget.** 1 h wall time for the whole pilot and 8 GiB of memory. A side not finished in time is NOT_RUN_BUDGET, never NO_GAIN.
- **Certificate form.** Each new endpoint is an ordinary chain-backed bound, so it fits the current checker rules. Using it needs exporter support for chain-backed layer-3 bounds with their own policy tables; that is not built here.

## Result (2026-10-31, MEASURED screen): gate PASSED on 016, no image recovered

Run: `output/p3b_bound_slope_pilot_20261031T173914Z/` (673 s, every side finished, no audit stop). Every new endpoint held at the nominal input and every retained witness. Exact slack accounting of every focus margin chain passed.

Two earlier runs are void:
- `…smoke017_*` was a smoke test with no gate.
- `…172800Z` hit a deadline bug: a monotonic-clock deadline was passed to `search_policies()`, which uses `time.time()`, so arm A never ran.

| Image | Primary baseline → new | Gain (threshold) | Arm A only | Arm B only | Closed (baseline / new / + margin search) | Layer-3 width sum |
|---|---|---|---|---|---|---|
| 013 c40 | −997 → −997 | 0 (50) | −984 | −984 | 37 / 37 / 37 | 141,052 → 121,749 |
| 016 c5 | −876 → −683 | **193 (44)** | −689 | −869 | 37 / 37 / **38** | 181,887 → 154,782 |
| 017 c37 (control) | 352 → 386 | 34 (1) | 385 | 371 | 42 / 42 / 42 | 21,109 → 19,607 |

- **Where the gain comes from.** Arm A (the P3 per-layer scalar search, applied to the bound chains) does most of the work. It tightened 189 / 268 / 33 sides; arm B tightened 149 / 207 / 19.
- **016 with margin search.** Competitor 0 closes (−117 → 142), and margin_min goes from −876 to −752. Four competitors stay open (−747, −298, −752, −683), so the image stays FINDER_INCONCLUSIVE.
- **Tighter bounds do not always help.** The ADAPTIVE margin heuristic is not monotone in the bounds:
  - 013 c40 gains 13 with either arm alone, but 0 with both combined;
  - 013 c36 drops from −876 to −986;
  - 016 c31 drops from −825 to −925 before margin search.
  Every one of these numbers is still a valid lower bound under its own facts. The heuristic simply picks worse slopes for some roots when the intervals change.

**Reading (ASSUMED).** By the pre-registered rule the gate passes (016). Still, no image is recovered, and 013 does not move. Searching slopes inside the bound chains is the first finder change that moves a near miss materially. What remains is roughly 700 units on 016 and 1,000 on 013, so this alone will not close them.

**Next step per the spec: cost measurement.** That means the extra chains, the extra terms and the separate policy tables each tightened bound would need, with no export yet. A cheap, sound refinement to measure alongside it: for each margin, keep the better of the baseline and tightened bound sets, since both are valid under their own facts.

## Cost measurement (2026-10-31, MEASURED screen; CPU hours are an ESTIMATE)

Run: `output/p3b_cost_20261031T183000Z/` (script `src/scripts/p3b_cost.py`). It mirrors the exporter's dependency closure and `--cost` tally, adding a separate policy for each tightened layer-3 side.
- **Mirror check.** The baseline configuration reproduces the production screen exactly: chains_planned and multiply_terms for all three images.
- **Endpoint check.** Every tightened side's own policy chain reproduces its pilot endpoint exactly.

| Image | Configuration | Closed | margin_min | Chains | Multiply terms | Layer-3 policy chains | Est. ESBMC CPU-h (330 µs/term) |
|---|---|---|---|---|---|---|---|
| 013 | baseline | 37 | −997 | 6,770 | 331,893,368 | 0 | 30.4 |
| 013 | tightened | 37 | −997 | 6,708 | 316,935,794 | 334 | 29.1 |
| 013 | + margin search | 37 | −997 | 6,708 | 316,843,033 | 334 | 29.0 |
| 016 | baseline | 37 | −876 | 7,851 | 365,356,538 | 0 | 33.5 |
| 016 | tightened | 37 | −925 | 7,781 | 348,809,064 | 471 | 32.0 |
| 016 | + margin search | 38 | −752 | 7,781 | 348,410,813 | 471 | 31.9 |
| 017 | baseline | 42 | 352 | 3,240 | 157,246,375 | 0 | 14.4 |
| 017 | tightened | 42 | 386 | 3,238 | 156,055,337 | 52 | 14.3 |

**Reading.**
- **Measured.** Tightening costs nothing: the graphs get 4–5 % *smaller*, because tighter layer-3 intervals make more neurons stable and drop supporting chains.
- **Not measured.** The extra artefact (one policy row or one override per tightened side) is small. Its checker cost was not measured.

**Rejected variant.** "Best of the baseline and tightened bound sets per margin" is not representable. The checker (`crown_chain_v2.py`, ReLU step) admits one fact per neuron side and recomputes every ReLU constant from that fact's value. Allowing two facts per side would need a checker change, which is not proposed.
