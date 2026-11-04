# SBESC SSV 2026 #37337 — evidence for the camera-ready (deadline 2026-11-08)

Purpose: give GPT the facts behind each reviewer comment so it can propose text changes. Every item is tagged:
- **PROVED**: ESBMC VERIFICATION SUCCESSFUL.
- **MEASURED**: code run on data, with its source.
- **ASSUMED**: interpretation, estimate or proposal.
- **NOT MEASURED**: needs new work before the paper can claim it.

Submitted text: `docs/papersANDdissertation/SBESC___SSV_2026___AAOS_with_PreQ_BMC (2).pdf` (6 pages). The `output/` paths are on the external disk, which was **not mounted** when this file was written. Figures below come from the repo docs that recorded those runs: `docs/ssv2026/RESULTS_SO_FAR.md`, `docs/gpt-context_papers.md`, and `experiments/ssv2026_test_protocol.json` (committed, with amendments).

---

## 0. Corrections to the submitted text (found while preparing this file)

| Where in the paper | Issue | Evidence |
|---|---|---|
| Abstract, §V-B, RQ2 box: "12–20 CPU-hours" | These are **summed per-call ESBMC elapsed hours**, not CPU time. Table IV's caption says "ESBMC time is summed over calls", but the abstract says "CPU-hours". | MEASURED note from 2026-09-29: "CPU-h figures are summed per-call elapsed seconds, not CPU time". Values: 11.82 h (06267), 20.48 h (04083). |
| §V-A: "Resource abort (16 pairs)" vs Table III primary "Checker 17" | The text uses the rerun count (16); the primary count is 17. State both. | Table III; `RESULTS_SO_FAR.md` Table 2. |
| Table IV: 04083 peak RSS 8.4 GiB vs "every individual harness stays under the 4 GB limit" | The 8.4 GiB is the **aggregate** over concurrent jobs, partly at 12 jobs; the per-call limit was 4 GB. The caption says so, but the text could be read as contradicting it. | `RESULTS_SO_FAR.md` §4: "015 cost inflated by 12-job contention, L4 at 4 jobs". |
| §V-A: median screen margin −6,038.5 | Correct for the 54 finder-inconclusive pairs. `RESULTS_SO_FAR.md` still says −1,975, which is wrong; the median over all 72 correctly classified pairs is −1,959.5. Do not use the −1,975 figure. | `gpt-context_papers.md` §12. |

---

## 1. Reviewer 2 — "Miscl." column (8 pairs) not explained

**MEASURED.** For 8 of the 80 drawn test images, the deployed C already misclassifies the **unperturbed centre image**: argmax ≠ ground-truth label. Property (2) is stated for the true class c, so it is already false at x′ = x and there is nothing to certify. These pairs are recorded as MISCLASSIFIED, not REFUTED (no solver involved), and they stay in the denominator.

**Rates.**
- Unconditional: 1/80 = 1.25 % [0.2, 6.7] primary; 2/80 = 2.5 % [0.7, 8.7] with the rerun.
- Among the 72 correctly classified pairs: 1.39 % [0.2, 7.5] and 2.78 % [0.8, 9.6], Wilson 95 %.

Source: `RESULTS_SO_FAR.md` Table 2.

**ASSUMED, wording.** One sentence where Table III is introduced, plus the conditional rate.

## 2. Reviewer 1 — rerun ambiguity (1/80 primary vs 2/80 rerun)

**MEASURED, from the committed protocol amendments.**
- **What stayed fixed.** The rerun changed **only resource parameters**. Every obligation, the per-call timeout (60 s) and memory limit (4 GB), the solver profile, the pinned sources and certificates, the image list, ε, the launch cutoff and the outcome rules were unchanged. Per-obligation verdicts do not depend on the number of parallel jobs.
- **Why jobs changed.** The workstation froze and rebooted three times, because WSL's page cache exhausted host RAM.
- **Amendment 1** (`7aa4a929`, 2026-09-26): ESBMC jobs 4 → 12, to recover lost time.
- **Amendment 2** (`b390496e`, 2026-09-27): jobs 12 → 6. Under memory contention, the checker's host free-memory guard (6 GiB) aborted shards: 011, 015, 016, 017, 026, 027, 032, 037, 042, 044 and 058 became CHECK_INCONCLUSIVE (resource abort).
- **Amendment 3** (`23e4fcf3`, 2026-09-27): jobs 6 → 4, after a third reboot. At jobs = 6, five more images (059, 062, 064, 069, 070) had been aborted by the same guard, because editor/IDE processes held several GB. jobs = 4 was the value that had run image 005 cleanly.
- **Primary result.** 005 (`Final_Test/Images/06267.ppm`) VERIFIED, 48,871 obligations.
- **Rerun (disclosed, user-approved before it ran).** The aborted images were re-checked under the same certificate and limits.
  - 015 (`04083.ppm`) became **VERIFIED** (50,851 obligations).
  - 011, 016, 017, 026 and 027 aborted again, at L4, L2, L4, L3 and L2.
  - The remaining 11 were still "rerun pending" at the last recorded checkpoint, and stayed CHECK_INCONCLUSIVE in the paper.
  - 04083 ran partly at 12 jobs and finished L4 at 4 jobs, which is why its cost and aggregate RSS are higher.

Sources: `experiments/ssv2026_test_protocol.json` (`amendments`), `RESULTS_SO_FAR.md` Table 3.

**MEASURED, verified on the external disk on 2026-11-04** (`output/crown_chain_v2_test_20260925/`).
- **Two ledgers.**
  - `results_preregistered_primary.json` was written 2026-09-27 14:31: 1 VERIFIED, 17 CHECK_INCONCLUSIVE.
  - `results.json`, the rerun ledger, was written 2026-09-27 16:18: 2 VERIFIED, 16 CHECK_INCONCLUSIVE.
  - Only the records of the 17 resource-aborted images differ between the two.
- **How the rerun works.** It **resumes at the aborted layer shard**, with the same certificate and the same harness files. Shards that had already finished with every harness VERIFIED are kept. Each aborted attempt is preserved as `outcome.resource_abort.<timestamp>.json` and `check_lN.resource_abort.<timestamp>/`.
- **Image 015 (`04083.ppm`).**
  - L0–L3 shards had finished, all harnesses VERIFIED, at 12 jobs during the primary attempt.
  - L4 was aborted twice by the low-memory guard (at 12 and 6 jobs; `aborted_low_memory: true`).
  - L4 was then re-run at 4 jobs: 2,720 harnesses, all VERIFIED.
  - The final aggregation re-checked the hashes of **all 50,851 harnesses** (`harness_hashes_rechecked: 50851`), along with the deployment binding (`dd535f54…`) and the margin closure. Summed ESBMC time 73,714 s; peak aggregate RSS 8,635 MiB; minimum host free memory 6.38 GiB.
- **The other 16.** Their re-attempts aborted again by the same low-memory guard, with in-flight obligations UNKNOWN or MEMOUT, never FAILED. They stay CHECK_INCONCLUSIVE.
- **Correction.** The `RESULTS_SO_FAR.md` wording "rerun pending" is out of date.

**ASSUMED, wording for the paper.** "The disclosed rerun used the same certificates, harnesses, per-call limits and outcome rules, and changed only the number of parallel ESBMC jobs (protocol amendments). It resumed each resource-aborted image at the aborted layer, reusing completed shards whose harness hashes were re-checked at aggregation. One image (04083) completed; the other 16 aborted again under the host free-memory guard."

**PROVED.** Both certificates' obligations were all VERIFIED: 48,871 and 50,851.

**MEASURED, post-submission re-audit (P2).** Under the later, closed checker, the only coverage missing from the saved exports was the zero-product affine columns. These were checked in 1,510 supplementary harnesses, **all VERIFIED** (`docs/ssv2026/P2_reaudit.md` in the worktree branch). This is the only new PROVED item since submission.

## 3. Reviewers 1 and 3 — fixed-point slower than float32 on the Cortex-A73

**MEASURED** (`output/vim3_runs_20260928/results_vim3.jsonl`, summarized in `RESULTS_SO_FAR.md` §6).

| Build | A73 median (ms) | x86-64 host median (ms) | Peak RSS (KiB) | Static text+data |
|---|---|---|---|---|
| int64 certified | 31.67 | 2.55 | 3,132 | 3.01 MB |
| int16 compact | 27.75 | 1.99 | 1,260 | 1.17 MB |
| float32 | 21.99 | 2.38 | 1,856 | 1.79 MB |

**MEASURED, key fact for the "intrinsic vs implementation-specific" question.** On the x86-64 host, the compact int16 build is **faster** than float32 (1.99 vs 2.38 ms). On the A73 it is slower. The ordering depends on the platform, so the slowdown is not intrinsic to Q7.8 quantization.

**MEASURED, code facts relevant to the cause.**
- Every multiply-accumulate in the generated C is `mac_i128`: `acc + (__int128)w * (__int128)x`, then `div_round_half_away_from_zero_i128` and `clamp_to_signed_range_i128`, all on `__int128`. This applies to **both** QNN builds, which differ only in storage type.
- The depthwise 2×2 layer is stored **dense** in all three builds: 147,456 entries, of which 767 are nonzero. All three therefore do roughly 190× more work than needed in that layer.
- Weight storage: int64, 2.46 MB of weights; int16, 0.32 MB; float32 model, about 0.64 MB.
- Build flags: `-std=c11 -O2 -Wall -Wextra -Werror -static`, GCC 13.3 aarch64. The float build adds `-ffp-contract=off`.

**MEASURED: disassembly of the exact binaries that ran on the board** (2026-11-04).
- **Binaries.** `output/vim3_bundle_20260928/bench_*_aarch64`. All 18 bundle files match `bundle_manifest.json` (SHA-256).
- **Tools.** `aarch64-linux-gnu-objdump -d` on `fnn_forward_bytes` (float32) and `qnn_forward_fixed` (int16 and int64), plus the x86-64 host binaries. Listings are saved in `output/ssv_camera_ready_20261104/` (`float.s`, `int16.s`, `int64.s`, `host_*.s`, `loops.txt`, `binaries.sha256`).

| Build | Innermost multiply-accumulate loop (per weight term) | NEON in the MAC loop |
|---|---|---|
| float32 | `ldr s0; ldr s2; fmul s0,s0,s2; fadd s1,s1,s0; cmp; b.ne` (6 instructions, scalar) | **none** |
| int16 compact | `ldrsh; ldrsh; mul; smulh; adds; adc; mov; add; cmp; b.ne` (10 instructions, scalar) | **none** |
| int64 certified | same 10-instruction pattern with `ldr` | **none** (0 vector-register instructions in the whole function) |

- **Reviewer 1's NEON hypothesis is not what happened.** GCC 13.3 at `-O2` did **not** vectorize the float32 loops either: scalar `fmul`/`fadd`, and the only vector-register instructions are 4 `movi` zeroing instructions. That is expected: a float sum reduction is not vectorized without `-ffast-math` or `-fassociative-math`, and `-ffp-contract=off` also prevents a fused `fmadd`.
- **The measured difference is the `__int128` accumulator.** Each QNN term needs two multiplies (`mul` for the low half and `smulh` for the high half of the 64×64→128 product) and a 128-bit carry chain (`adds`, `adc`): 10 instructions against 6, all scalar.
- **The x86-64 host fits this.** There, the int16 loop uses a single `imul %rdx`, a 64×64→128 multiply in one instruction, and the int16 build is faster than float32 on the host (1.99 vs 2.38 ms).
- **int16 vs int64.** Their loops are identical apart from the load width. The int16 build's advantage on the A73 (27.75 vs 31.67 ms) is consistent with lower memory traffic; this is ASSUMED, since cache counters were not measured.

**ASSUMED.**
- The slowdown is specific to the generated implementation on AArch64: a 128-bit accumulator in the C, with no SIMD in any build. It is not intrinsic to Q7.8 quantization.
- A 32-bit accumulator would allow single-instruction `smaddl`/`madd`, and NEON `smlal`, if vectorized. It is justified when a static bound shows the sums fit; the measured layer accumulators stay below 2^35.
- Such a build is a **different artifact** and would need its own certificate.
- None of this was benchmarked; state it as a likely cause supported by the disassembly, not as a measured speed-up.

**Optional, larger: NOT MEASURED.** An int32-accumulator or NEON-optimized build, or an established runtime (e.g. TFLite int8), as the comparison Reviewer 3 asks for. Any such build is a new artifact and is **not** covered by the certificate.

**Memory trade-off.**
- **MEASURED.** The certified int64 build uses more memory than float32 (3,132 vs 1,856 KiB RSS). Only the compact int16 build is smaller (1,260 KiB).
- **MEASURED.** The compact build is tied to the certified C only by identical logits on 7,830 validation images (0 mismatches on host and board).
- **ASSUMED.** Certifying the int16 file directly would need a new certificate, because the file has a new hash. The arithmetic is identical, so the proof structure would be the same.

## 4. Reviewers 1 and 3 — low certification rate and the relaxation gap: what limits it, and future work

**MEASURED at submission.**
- **54 of 72 correctly classified pairs** failed at the finder: the CROWN lower bound stayed < 1 for at least one competitor (median −6,038.5).
- **The other 18 were finder-positive.** Of these, 16–17 were stopped by resource aborts and 2 were certified. If every finder-positive pair had been verified, the rate would be 18/80 = 22.5 % [14.7, 32.8]; this is the finder's ceiling.
- **No obligation returned FAILED.** No counterexample was found.

**MEASURED, post-submission (validation development images, not the test cohort).**
- **The finder, not the checker, is the binding stage.**
  - Validation campaign of 2026-10-14 (n = 20, seed 20261014, 4 jobs, stable machine): **5/20 VERIFIED** (Wilson 11.2–46.9 %); 13 finder-inconclusive, 1 misclassified, 1 low-memory abort.
  - ESBMC closed 5 of 6 finder-positive images; each certificate took 45,836–76,383 obligations and 13–23 h of summed per-call time.
  - **ASSUMED.** This suggests the test aborts were largely environmental, but it does not show they would have certified, since the samples differ and n is small. Report it separately; never pool it with the test cohort.
- **Diagnostics on the two closest misses** (013 and 016):
  - a fixed concrete search found no violation;
  - in a singleton (ε = 0) control, the bound loses only 184–333 units, but 2,600–4,500 units when the ε = 1 intermediate bounds are inherited;
  - so **most of the loss comes from loose intermediate (hidden-layer) bounds at ε = 1**.
- **Finder refinements tried, none of which recovered an image:**
  - ReLU-slope search in the margin chains (P3);
  - in the bound chains (P3b/P3c): 016 margin −876 → −647, and dependency graphs 4–5 % *smaller*;
  - a one-byte input split (I-P): +1;
  - a two-neuron upstream relation (I-R): ESBMC undecided within 60 s, both as deployed and with an exact int32 re-encoding.
- Docs: worktree branch `feat/crown-v2-qformat`, `docs/ssv2026/` (`finder_diagnostic_spec_013_016.md`, `P3_*`, `P3b_*`, `P3c_*`, `IP_partition_016.md`, `IR_upstream_016.md`, `IR_encoding_016.md`).

**ASSUMED, future-work directions supported by these measurements.**
- Tighter intermediate bounds:
  - optimized relaxations inside bound chains, which gave partial gains;
  - ESBMC- or MILP-tightened box facts as a new fact source (MILP stays in the tool as a bound finder).
- Input-domain splitting with a proved two-child composition rule.
- Relational facts across neurons that share ancestry.

Each needs its own checker extension. Present these as directions, not as demonstrated fixes.

## 5. Reviewer 3 — scalability: cost drivers, and dependence on size, depth, neurons and radius

**MEASURED cost structure** (`RESULTS_SO_FAR.md` §4, `output/ssv_scalability_20260928/TABLES.md`).
- **Checking cost is linear in exported multiply terms.** Above about 4k terms per harness it is flat at 177–188 µs per term.
- **Layer 2 dominates**, carrying 57–69 % of the cost.
- **Per-layer, on image 3652 (validation pilot):**

  | Layer | Harnesses | Summed hours |
  |---|---|---|
  | L0 | 1,257 | 0.06 |
  | L1 | 6,143 | 0.54 |
  | L2 | 32,445 | 7.97 |
  | L3 | 16,115 | 3.81 |
  | L4 | 2,735 | 1.54 |
  | Total | 58,690 | 13.92 |

- **Wall time per image.** Screening about 35–45 s; box stage about 650–800 s; check 3–5 h per finder-positive image at 4 jobs.
- **Caps don't help.** Splitting harnesses more finely (an 8k-term cap) adds harnesses and CPU and only lowers per-job memory. The only lever is fewer exported terms, which is finder work.

**MEASURED, monolithic wall vs problem size** (Table IV of the paper, plus `RESULTS_SO_FAR.md` Table 5).
- Direct ESBMC on the deployed C handles 24 layer-0 outputs (Bitwuzla) with up to 12 free input bytes.
- 18–24 free bytes, or 48 outputs, run out of memory at about 5 GB.
- The whole network times out in symbolic execution, with about 230 of 6,144 layer-0 outputs unrolled.

**MEASURED, depth (other architectures explored during model selection).**
- On a deeper, more accurate model (`conv24_48_96_128_dense384`, 91.54 % test), sound interval boxes at ε = 1 LSB saturate to the full Q16 range by block 4. The solver also failed to finish block 5, even given an artificially tight box: no verdict after about 22 min at 20.3 GB.
- **ASSUMED.** Depth compounds relaxation looseness, and this, not width or solver speed, is the binding constraint (`docs/ssv2026/source_model_search.md`).

**NOT MEASURED.**
- A radius sweep: only ε = 1 was attempted. A finder-only radius screen is cheap (about 35 s per image per radius), but on the test cohort it would be a second use of the test set. On validation it is clean.
- A controlled sweep over network size: only one deployed architecture was fully verified.

**ASSUMED.** State these as limitations and future work, unless one cheap screen is added on validation.

## 6. Reviewer 3 — novelty and comparison with related tools

**MEASURED, closest comparison already in the paper.** The monolithic baseline in Table IV is ESBMC run directly on the generated C. That is the approach of SMT-based C-level QNN verification (QNNVerifier-style): it stops at 24–48 layer-0 neurons, while the decomposed pipeline covers all 14,123.

**ASSUMED.** The paper may frame this as a comparison with the monolithic SMT-based approach on the same artifact, without claiming to have run QNNVerifier itself, which was not run. No CEG4N/Quadapter comparison exists. Those tools solve different problems (quantization synthesis or equivalence), not implementation-level robustness of a given C file.

**Candidate contribution statements (ASSUMED, to be checked against the text).**
- The integer CROWN finder as an **untrusted** proposer, with every step re-checked by ESBMC on the deployed arithmetic (rounding, clamp, `__int128`).
- The certification criterion with an explicit trust base.
- Exact host↔board parity of the certified C on AAOS.

## 7. Reviewer 3 — role of Android Automotive OS

**MEASURED.**
- The benchmark is a statically linked AArch64 executable, pushed with `adb` and run from `adb shell` under `taskset` (A73 cores). See `tools/ssv2026/VIM3_README.md` and `run_on_vim3.sh`.
- It uses **no AAOS-specific service, API, HAL/VHAL or app framework**.
- The board image is `aosp_yukawa_car`, Android 16, userdebug.

**ASSUMED.**
- On the same SoC, embedded Linux would likely give similar latency, but this was not measured.
- **The formal results do not depend on AAOS at all.** The verification boundary ends at the C source.
- The honest framing: AAOS is the **target deployment environment**. The study shows that the certified C runs unmodified, with bit-exact outputs, in the native layer of an AAOS build. Integration through NDK/JNI, a vehicle HAL or an AAOS service is future work.
- A title or abstract adjustment may be advisable.

## 8. Reviewer 3 — reproducibility and artifacts

**MEASURED.**
- Every run pins source hashes, seeds and limits.
- The test protocol is committed (`92f17ea2`, amendments `7aa4a929`, `b390496e`, `23e4fcf3`).
- The board bundle carries SHA-256 manifests.

**Decision for João (not technical).** The repository is private. Releasing the generated C, the harnesses, the protocol and the scripts (for example as a Zenodo or GitHub artifact, or "available on request") is an authors' decision. Nothing is published without that decision.

## 9. Reviewer 3 — LaTeX template

**NOT MEASURED here.** The submitted PDF must be checked against the official IEEE conference template required by SBESC. This is a formatting task, independent of the results. The local drafts (`docs/ssv2026/paper_text/main.tex`) predate the final PDF.

---

## Summary of what can change in the text without new experiments

1. Explain "Miscl." and add the conditional rate.
2. Describe the rerun precisely, as resource-only amendments.
3. Fix the "CPU-hours" wording.
4. Reconcile 16 vs 17 resource aborts in the text.
5. Discuss cost drivers: linear in terms, layer 2 dominant, the monolithic wall.
6. Frame the relaxation gap with future-work directions.
7. Clarify the AAOS role.
8. Frame the monolithic baseline as the comparison with C-level SMT verification.
9. Add the x86-vs-A73 latency ordering.

**Done on 2026-11-04:** the objdump analysis (§3) and the rerun ledger reconciliation (§2).

**Optional and outside the original scope:** the validation-only campaign (5/20) and the post-submission diagnostics. Report them separately, if the page budget allows.
