# GPT-6 Sol — Round 1

## 1. Re-shape, do not weaken, affine ESBMC obligations (Q1)

**What.** After the freeze, run a validation-only renderer tournament over affine chunk size, flat versus balanced partial sums, scalar temporaries versus fixed loops, and z3/bitwuzla/boolector. Keep one ESBMC verdict per generated obligation and choose by summed affine CPU subject to memory. The present renderer emits a flat multiplication expression per column and chunks by a 25,000-term default (`src/verification/crown_chain_v2.py:728`, `src/verification/crown_chain_v2.py:766`, `src/verification/crown_chain_v2.py:1094`).

**Soundness.** **ASSUMED:** the proposed renderers preserve the contract and arithmetic. **PROVED:** the existing aggregator requires every record VERIFIED and rejects unsafe flags (`src/scripts/aggregate_crown_chain_v2.py:104`, `src/scripts/aggregate_crown_chain_v2.py:130`). Each alternative must pass the 14 corruption classes.

**Expected gain.** **GUESS:** 1.5–2.0x on affine CPU would imply 1.3–1.6x end-to-end by the measured 72% affine share; batching alone is not the experiment.

**Cost.** **GUESS:** 10–16 h.

**Validation pilot.** Three fixed validation certificates, stratified by affine term count; accept only a configuration with at least 1.5x affine speed, zero non-VERIFIED obligations, and peak/job at most 1.1 GB.

## 2. Optimize integer ReLU slopes, then split only unresolved inputs (Q2)

**What.** First replace the single closed-form slope choice with competitor-specific coordinate/gradient search over integer slopes. The current finder fixes `slope = round(a*u/(u-l))` (`src/scripts/crown_finder/crown_chain_export.py:50`). If that still leaves an open competitor, branch the integer input box on one selected byte and export a partition manifest whose leaves cover the original box.

**Soundness.** **PROVED:** slopes remain proposals: the checker independently recomputes affine coefficients and validates every ReLU class/constant (`src/verification/crown_chain_v2.py:404`, `src/verification/crown_chain_v2.py:468`). For splitting, VERIFIED requires every leaf VERIFIED plus an ESBMC-VERIFIED disjoint-cover obligation; no timeout counts. **PROVED (literature):** Beta-CROWN establishes completeness of exhaustive branching in its setting (Wang et al., 2021, §§2.3, 3.4, NotebookLM Turn R0), but our integer/clamped composition must be proved independently.

**Expected gain.** **GUESS:** optimized slopes offer the best coverage/hour; splitting trades approximately linearly more checking for residual cases. The literature's 70.5% versus 43.4% example is context, not a forecast (Wang et al., 2021, Table 2, NotebookLM Turn R0).

**Cost.** **GUESS:** 18–28 h for slope search; 16–24 h for sound partition certificates.

**Validation pilot.** Lock 30 validation failures; report additional finder-positive images and certificate/check cost. Proceed to splitting only if slopes recover at least 20% relative positives; cap leaves at four and require complete union verification.

## 3. Content-addressed reuse of identical ESBMC obligations (Q1)

**What.** Canonicalize assertion labels, key each harness by source bytes plus ESBMC binary/version, profile, flags, arithmetic-kernel hash and deployment hash, and reuse only a stored VERIFICATION SUCCESSFUL record. The aggregator already re-hashes every harness (`src/scripts/aggregate_crown_chain_v2.py:131`). This targets exact repeats across shards/images, not “similar” chains.

**Soundness.** **ASSUMED:** replay of an identical closed ESBMC query preserves its verdict; SHA-256 collision resistance and provenance storage enter the trust base. Any key mismatch reruns ESBMC. Incremental solving is not the same mechanism and did not help Sena et al. (2021, §4.3.2, NotebookLM Turn R0).

**Expected gain.** **GUESS:** unknown until weighted duplicate analysis; reject if reusable hits represent under 15% of affine CPU.

**Cost.** **GUESS:** 12–18 h.

**Validation pilot.** Canonicalize five validation certificates without solving, compute exact hit rate weighted by recorded obligation time, then replay cache versus fresh results and inject wrong-key/wrong-log mutations.

## 4. Minimum publishable evaluation matrix (Q3)

**What.** Report all 80 primary outcomes with the **MEASURED** 1/80 certified rate and a binomial interval; show the disclosed rerun in a separate column, never replacing primary data. Include float/deployed accuracy, monolithic ESBMC and IBP failures, screen/box/check ablations, obligation-time/RSS distributions, and VIM3 model bytes, latency, energy/inference and EDP under matched workloads. Quantized-resource metrics follow Hashemi et al. (2016) and Al-Shaikh et al. (2023) (NotebookLM Turn R0).

**Soundness.** **PROVED:** the registered rule counts every non-VERIFIED outcome as not certified (`experiments/ssv2026_test_protocol.json:98`, `experiments/ssv2026_test_protocol.json:108`). **ASSUMED:** board measurements do not affect proof claims.

**Expected gain.** **GUESS:** reviewer credibility, not certification speed.

**Cost.** **GUESS:** 12–20 h excluding already-running checks.

**Validation pilot.** Freeze table schemas now; reproduce every table from manifests and disclose sampling, class imbalance, weak worst-class accuracy, ε=1, and workstation instability.

## 5. Publish the trust contract and reject a compiled generic checker for the headline claim (Q4)

**What.** Keep a generic compiled checker only as an exploratory secondary result. For the headline, ESBMC must still judge certificate-specific obligations. A once-proved checker followed by native execution adds the C compiler/runtime and proof-to-binary correspondence as trusted components and weakens the plain meaning of “ESBMC is the final judge.”

**Soundness.** **ASSUMED:** this is the conservative interpretation of constraint 1. The paper must enumerate Python parsing/composition already in the TCB; the current summary acknowledges parsing, hashing and composition (`src/scripts/aggregate_crown_chain_v2.py:185`). Marabou proof checking supports the general architecture, not this trust transfer (Wu et al., 2024, §2.3; Cordeiro et al., 2025, §3.5, NotebookLM Turn R0).

**Expected gain.** **GUESS:** avoids the strongest soundness objection; zero runtime gain.

**Cost.** **GUESS:** 6–10 h for a trust diagram, threat table, and added omission/reordering/cache/partition corruption cases.

**Validation pilot.** Every mutation must fail closed; independently trace each final margin through all roots, facts, hashes and ESBMC successes.
