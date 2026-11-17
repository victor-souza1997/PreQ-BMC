# PreQ-BMC Paper Work — Transfer Summary

Prepared 2026-10-29 for continuing work in a fresh chat. This document records the repository state, the paper direction, the latest formal-methods findings, and the immediate work still needed. Treat measured results mentioned below as reported evidence unless the cited artifacts are available and independently rechecked in the new session.

**Read Section 12 first.** It adds the completed October validation campaign (5/20 VERIFIED) and corrects Sections 2, 7, and 11. In particular, `output/` looked empty only because the external disk was not mounted.

## 1. Project Goal

PreQ-BMC is being developed as a formal-methods approach for producing fixed-point neural-network C implementations and checking local robustness with ESBMC. The current research application is GTSRB traffic-sign recognition on an Android Automotive-class platform. The intended paper combines two essential claims:

1. **Formal:** for a particular correctly classified image and a specified raw-byte (L_\infty) perturbation box, the exact integer C program preserves its prediction.
2. **Deployment:** the identity-bound program can execute on the intended Android Automotive hardware, with parity, latency, memory, and power measured there.

The current formal route is the convolution-native fixed-point pipeline with an untrusted CROWN-style certificate finder and ESBMC checking arithmetic obligations. The source model and certificate cover local robustness only. They do not establish robustness for other images, norms, transformations, physical signs, or an NNAPI/NPU implementation.

The work is for traffic-sign recognition, not license-plate recognition. The target is Android Automotive OS (AAOS), not Android Auto phone projection. The Khadas VIM3 should only be called an AAOS device if its installed image declares the automotive system feature; otherwise describe it as Android on an automotive-class board.

## 2. Repository and Worktree State

- Repository: `/home/joao/code/preqbmc`.
- Current branch at inspection: `branch/ssv-files`, HEAD `fb69215e` (`add ssv code to run on vim3`).
- `output/` is a symlink to `/mnt/wsl/PHYSICALDRIVE1p1/preqbmc_ext/output`. In this session, listing that target showed no files, so the large run artifacts could not be inspected from the current mount.
- No active verification process was started during preparation of this summary.

Uncommitted work observed with `git status --short`:

- Deleted tracked file: `output/sign_deep_source_model_test_20260922/test_evaluation.json`. Preserve this user change; do not restore it without being asked.
- Untracked methodology diagrams: `docs/ssv2026/methodology.drawio`, `methodology_v2.drawio`, and `methodology_v3.drawio`.
- Untracked protocol: `experiments/ssv2026_validation_protocol_20261014.json`.
- Untracked monitoring tool: `tools/ssv2026/run_protocol_monitor.py`.
- An untracked `output` symlink is reported by Git; it points to the external output mount.
- The latest inspection did not show uncommitted edits under `src/`; the protocol driver and v2 checker are tracked source files.

Do not broadly stage or clean the worktree. In particular, preserve the deletion, diagrams, protocol, and monitor until their owner says otherwise.

## 3. Paper and Model Facts, with Documentation Conflicts

The campaign files point to a compact GTSRB CNN called `conv24_64_192_dwpool43`, with five affine layers and 43 logits. Its deployed arithmetic is Q7.8/int16 range, round-half-away-from-zero, clamp before ReLU, and `__int128` accumulation. The C parameter arrays use `int64_t` storage. Calling this particular artifact “int8” or saying it uses int32 accumulators is inaccurate.

Some repository documents describe different model generations or different experimental branches:

- `docs/ssv2026/RESULTS_SO_FAR.md` reports 91.32% float32 and 91.35% deployed-C accuracy on 12,630 test images for the convolution-native model.
- `docs/ssv2026/paper_strategy_android_automotive.md` contains a later/other low-accuracy `<12,3,8>` / `<14,5,8>` result around 41.7% and says Android results are not yet measured.
- `docs/ssv2026/README.md` also describes an opt-in small 8x8 research prototype.

These figures must not be combined. Before writing the paper or creating tables, identify the exact frozen model, deployment C, data split, experiment protocol, and matching result files by their pinned SHA-256 values. The October validation protocol pins these identities, but its associated run outputs were unavailable in this checkout.

## 4. Formal Method and Prior Certificate Result

The v2 certificate pipeline has these stages:

1. A box proof emits per-neuron intervals, with every required interval block checked by ESBMC.
2. An untrusted exact-integer CROWN-style finder emits affine and ReLU relaxation chains and margin bounds. Finder output alone is never a proof.
3. The checker emits C harnesses for affine arithmetic, ReLU inequalities, concretization, chain closure, and shared lemmas. Each reported obligation must return VERIFIED.
4. An aggregate checks completeness, source references, output-margin/clamp conditions, and the deployment binding.

Prior chat records report a complete validation-split pilot for image 3652, target class 12, epsilon 1 raw byte: 4,054 chains, 42 competitor margins, and a minimum reported raw margin of +2,622. A previous aggregate reported all 58,690 ESBMC obligations VERIFIED, about 3.35 hours of shard wall time with four workers, and about 4,063 MiB peak aggregate RSS. A corruption run reportedly made 15 deliberately false harnesses return FAILED, and 22 focused tests passed.

These are **prior-session reported results**, not remeasured in this session. There are inconsistencies across notes: some files list 58,690 obligations and per-layer counts 1,256/6,142/32,444/16,114/2,734, while `docs/ssv2026/RESULTS_SO_FAR.md` lists 58,695 and counts five larger by one each. Reconcile the raw manifests, records, files, and summaries before reusing the total in a paper. The detailed prior record is [claude_codex_chat.md](ssv2026/claude_codex_chat.md), Turn 12.

## 5. Current Soundness Findings — Blocking

The previous session reviewed Fable 5.1’s criticism against the v2 checker. Two concrete validator gaps remain in the tracked implementation:

1. **No enforced layer traversal.** `validate_chain()` checks the initial property and individual step arithmetic but does not require the sequence to be `affine(L), relu(L-1), affine(L-1), ... affine(0), concretize`. A read-only synthetic probe showed that a one-step “margin” chain using input byte indices as if they were logits passes `validate_chain()` and receives a positive bound. This is a possible false-VERIFIED path for a malformed certificate. Relevant code: [crown_chain_v2.py](../src/verification/crown_chain_v2.py:380).
2. **Chain-backed fact identity is not tied to the neuron identity.** `_validate_facts()` verifies that a cited chain has the same numeric bound as a fact, but for chain facts it does not require the chain ID to match that fact’s layer, neuron, and side. A synthetic probe showed it accepts a lower fact for neuron 7 backed by a chain for neuron 8. Relevant code: [crown_chain_v2.py](../src/verification/crown_chain_v2.py:312).

An audit of the **actual image-3652 export** found all 4,054 chain step sequences had the expected layer order and all chain-backed fact IDs matched their expected neuron/side. This reduces concern that this particular export exploited the bugs; it does not make the validator safe for other exports. Do not make an unqualified soundness claim from the old aggregate until these validator checks are fixed and the result is revalidated under a frozen checker.

Other proof-boundary qualifications:

- The arithmetic computations in v2 are now largely emitted as C obligations, addressing Fable’s earlier Python-literal critique.
- The deployment CSR-to-C binding is performed by Python parsing and reconstructing arrays/loop geometry, not by an ESBMC proof over the complete deployed source. Roundtable notes report ESBMC corroboration for only sampled rows (7 of 14,123 in one audit); full-file ESBMC binding exceeded memory. Describe this as a Python-checked binding with sampled ESBMC corroboration, unless a stronger binding proof is completed.
- Margin clamp closure and proof-DAG composition include Python checks. Be precise about ESBMC-discharge versus Python composition in the trust table.
- The `paper-z3` command includes interval-analysis flags. A roundtable ablation reports sampled obligations also verified without them, but the full run used them, so interval analysis remains in the primary run’s trust base unless every required result is rerun without those flags.
- Overflow checking was reportedly run on 300 large affine chunks and selected ReLU chunks/lemmas, not every obligation. Report the exact scope; do not state a global overflow proof unless it exists.
- The GPT-6 Sol Round 3 omission/deletion regression was **not run** because the attempt to create its scratch directory was refused. Counts were manually inspected; this was not a passing mutation test. See [GPT-6 Sol R3](ssv2026/roundtable/rounds/R3/gpt6-sol.md) and [roundtable synthesis](ssv2026/roundtable/SYNTHESIS.md).

## 6. Decisions and Constraints Already Established

- ESBMC remains the final formal judge. MILP, CROWN, Python, and attacks may find/refine certificates or candidate counterexamples, but their outputs alone do not produce VERIFIED.
- Preserve the deployed fixed-point semantics and the complete input region. No relaxed contracts, heuristic slack, disabled chaining, per-block formats, or changed arithmetic to improve the rate.
- TIMEOUT, MEMOUT, UNKNOWN, finder-inconclusive, and resource aborts are not proofs and are not evidence of non-robustness.
- Report a pre-registered primary test cohort separately from any rerun. Include every selected image in the denominator and report both unconditional and correctly-classified conditional rates.
- Tune on validation only. The test protocol must be fixed and committed before its test outcomes are run.
- The current method is expensive: prior reports show roughly 3–5 hours wall per finder-positive image with four jobs. Most cost is in the deep affine chain checks; more solver variants do not help ground affine obligations.
- The certified program’s parameter storage is reported as larger than the corresponding float32 weights, so do not claim compression or resource savings from nominal bit width. Device measurements are required for latency/energy claims.
- Adversarial training, a shallower architecture, stronger slopes/splits, and brightness-only properties are proposals, not established gains. A brightness-band theorem would be a different and weaker property and must be reported separately.

## 7. Protocol and Experiment State

The untracked [October 14 validation protocol](../experiments/ssv2026_validation_protocol_20261014.json) specifies:

- validation split, epsilon 1 raw byte;
- a fixed draw of 20 out of 7,830 validation records, uniform without replacement, seed 20261014;
- Q7.8 across the five layers;
- box proof: blocks of 24, four ESBMC jobs, 120 s/query, 6 GB/query, 6 GiB available-memory guard;
- chain checking: four jobs, 60 s/query, 4 GB/query, `paper-z3`, all five layers;
- memory scopes: 8 GB for screen/box/export, 12 GB for checking;
- hashes for model/data/deployment and Python sources.

Its launch cutoff is **2026-10-21 23:59 -03:00**, already expired as of 2026-10-29. The protocol was untracked at inspection and no corresponding output files were present in the mounted `output/` path. Do not start this file as a current experiment: its budget rule would turn positive images not started before cutoff into NOT_RUN_BUDGET. To run a new campaign, make a newly dated protocol, re-pin hashes after checker fixes, freeze/commit the protocol before test use, and choose a future cutoff. Keep this old file as historical evidence unless asked to replace it.

The untracked [protocol monitor](../tools/ssv2026/run_protocol_monitor.py) wraps the tracked driver and displays estimated per-image progress based on prior measured stage weights. Percent completion is an estimate, not a measured solver progress guarantee. The protocol driver is resumable and moves interrupted stage output aside rather than deleting it.

The repo also has `experiments/ssv2026_test_protocol.json` and `experiments/ssv2026_validation_dryrun_protocol.json`; inspect their exact pins/status and any amendment logs before treating them as runnable. `docs/ssv2026/RESULTS_SO_FAR.md` reports an 80-image, epsilon-1 test cohort: primary 1/80 VERIFIED, 8 MISCLASSIFIED, 54 FINDER_INCONCLUSIVE, and 17 resource CHECK_INCONCLUSIVE; a disclosed rerun reportedly had 2 VERIFIED, with other work still unresolved at that document’s checkpoint. These counts are reported by the document and need raw-output reconciliation, not replacement by validation-protocol results.

## 8. Paper Direction and Evidence Gaps

The most defensible paper is a feasibility/evaluation of per-image bit-precise fixed-point C certification for traffic-sign recognition, with carefully bounded claims. The contribution is not “the model is globally adversarially robust.” A VERIFIED result means only that the named integer program retains the prediction for the named image and encoded (L_\infty) region, conditional on the stated checker/trusted base.

For the AAOS framing, the formal proof and actual device execution are both required. Existing documentation says Android/AAOS, JNI/APK, NPU/NNAPI parity, and physical energy have not been measured for the identified certificate. The newest test results and VIM3/deployment artifacts must be checked before asserting this remains true. At minimum, the paper needs:

- a frozen model/C/protocol identity and clear accuracy denominator;
- complete outcome tables with raw statuses and resource usage;
- a trust-base table separating ESBMC obligations, Python checks, and assumptions;
- formal local robustness results on a pre-registered test sample, with non-VERIFIED rows retained;
- exact C/Python/device parity on a declared corpus;
- actual AAOS identification and repeated device latency/memory/energy measurements against named float32 and int8 baselines;
- a precise account of the network format and storage (current main artifact is not int8).

The roundtable consensus before the paper deadline was to prioritize claim hygiene, artifact auditing, overflow-scope reporting, and sampled binding corroboration; avoid risky finder/checker rewrites during a pinned campaign. Since the paper deadline (2026-10-04) has passed, treat that schedule as historical. Current work should first resolve the v2 certificate validator gaps and the October protocol status, then decide whether to prepare a revised submission or continue the AAOS study.

## 9. Recommended First Actions in a Fresh Chat

1. Inspect the actual output mount and identify authoritative protocol run directories; reconcile the 58,690/58,695 count discrepancy and primary/rerun result ledgers.
2. Fix chain layer-sequence validation and chain-fact identity validation. Add focused negative tests that inject a skipped-layer chain and a cross-neuron fact reference; both must be rejected before any VERIFIED aggregate can be emitted.
3. Make the aggregate require independent completeness evidence: records count, unique paths, on-disk harness count, expected chain count from certificate shards, required root chains, every competitor, and pinned checker source hashes. Retain the deletion regression as an explicit gate.
4. Freeze the corrected checker. Existing results cannot silently inherit the new checker version; establish whether unchanged harness hashes and recorded ESBMC results can be re-audited/reused, or rerun the affected checks. Keep old and new ledgers separate.
5. Confirm the user’s current target: finish/revise the SSV paper, or continue the broader AAOS deployment evaluation. Do not start the expired 20261014 validation protocol as if its cutoff were current.

## 10. Key Files

- [Paper strategy: Android Automotive](ssv2026/paper_strategy_android_automotive.md)
- [Results so far](ssv2026/RESULTS_SO_FAR.md)
- [Formal methodology](ssv2026/conv_native_verification.md)
- [Fable 5.1 soundness review](ssv2026/fable5.1%20thought/fable_review_1.md)
- [Roundtable synthesis](ssv2026/roundtable/SYNTHESIS.md)
- [Protocol driver](../src/scripts/run_crown_chain_v2_protocol.py)
- [CROWN v2 validator](../src/verification/crown_chain_v2.py)
- [Shard checker](../src/scripts/check_crown_chain_v2.py)
- [Strict aggregate](../src/scripts/aggregate_crown_chain_v2.py)
- [Corruption regression](../src/scripts/regress_crown_chain_v2_data_corruption.py)

## 11. Latest SSV Paper Review Update — 2026-10-29

This update reflects the later, artifact-backed review of `docs/ssv2026/paper_text/main.tex` and supersedes earlier statements below where they conflict. Reviewed checkout: branch `feat/cnn-feature`, commit `ff482fc4`; relevant review: [FINAL_PASS_REVIEW.md](ssv2026/paper_text/FINAL_PASS_REVIEW.md). The revised text is [main.reviewed.tex](ssv2026/paper_text/main.reviewed.tex); the original `main.tex` was not updated during that review. Apply and recheck the reviewed changes before treating the submission draft as corrected.

### Acceptance assessment

- The defensible framing is an artifact-backed feasibility/evaluation case study of local robustness certificates for one fixed-point C model, plus native execution measurements on a VIM3. It does not establish a general scalable verifier, quantization synthesis, globally robust classifier, or automotive safety case.
- Rejection risk is substantial but cannot responsibly be expressed as a numeric probability. Main risks are inconsistent methods/results in the current draft, low primary certification yield and high cost, and dependence on Python-side binding/composition and completeness logic. SSV scope includes verification, automotive testbeds, and safe AI, so the subject is relevant; relevance alone does not address those evidence gaps.
- The official SSV 2026 submission deadline listed by the workshop is 2026-10-04 23:59 BRT, already past on the date of this update. Verify whether the paper is under an extension or another submission route before using the old final-pass deadline as current.

### Verified results and corrections to older notes

- Authoritative primary cohort: 1/80 VERIFIED, 8 MISCLASSIFIED, 54 FINDER_INCONCLUSIVE, 17 CHECK_INCONCLUSIVE. Disclosed rerun: 2/80 VERIFIED, 8 misclassified, 54 finder-inconclusive, 16 checker-inconclusive. Retain the primary and rerun as separate results; non-VERIFIED is not evidence of non-robustness. Sources: [primary ledger](../output/crown_chain_v2_test_20260925/results_preregistered_primary.json) and [rerun ledger](../output/crown_chain_v2_test_20260925/results.json).
- Completed certificates: test image 06267 has 48,871 verified harnesses and minimum raw margin 584; disclosed rerun image 04083 has 50,851 and margin 2,552. Validation image 3652 must be cited from the hardened run as one unit: 58,695 harnesses, 50,123.145 summed per-check elapsed seconds, 4,064.391 MiB peak aggregate RSS. Do not combine the older 58,690 count with hardened timings. Sources: each `complete_check_summary.json` under `output/crown_chain_v2_test_20260925/{005,015}/complete/` and `output/crown_chain_v2_20260924/image3652/claude_hardened_complete/`.
- Report CROWN checking as approximately 11.82, 20.48, and 13.92 hours of **summed per-check elapsed time**, respectively; these are neither end-to-end pipeline wall time nor CPU-hours. The preceding box stage includes 84 FAILED abstract-margin obligations; these are failures of those proposed abstract bounds, not counterexamples to network robustness. Avoid the draft's statement that no obligation returned FAILED.
- Board runs are standalone static AArch64 native executables built with GCC 13.3, not an NDK/JNI app. They ran on the VIM3 A73 cluster, and the taskset mask permits migration within that cluster. There is no recorded NPU, energy, paired-cluster, or app/JNI measurement. The board fingerprint is consistent with a custom automotive AOSP image, but available evidence does not verify its automotive feature set/API/CarService; retain an author check before calling it confirmed AAOS. Source: `output/vim3_runs_20260928/` and `tools/ssv2026/run_on_vim3.sh`.
- Across 7,830 validation images, both integer builds match the host integer logits exactly; the float baseline check compares predicted classes. Three-pass median/p95 latency is compact int16 27.74854/27.88417 ms, reference int64 31.67358/31.83683 ms, and float32 21.98708/22.12721 ms. One-image process peak RSS is 1,260, 3,132, and 1,856 KiB, respectively. This is finite parity, not universal equivalence. Source: `output/vim3_runs_20260928/results_vim3.jsonl`.
- Correct the current methodology's int8/int32, NDK/JNI, and paired-core claims to match the selected Q7.8/int16 numerical format with `__int128` accumulators and the native benchmark, or mark unverified details as author checks. Preserve the explicit distinction between `int64_t` reference storage and compact `int16_t` storage.

### Necessary pre-submission work

1. Reconcile the actual submission body with the implementation, then use the frozen primary results prominently and label the rerun separately. Correct the proof scope and trust assumptions throughout, not only in the conclusion.
2. Explain the Python-side certificate-to-C binding, dependency composition, arithmetic-safety argument, and completeness checks. The review found sampled ESBMC binding corroboration and no independent obligation enumeration/count enforcement in the aggregator; do not imply those are proved unless closed and evidenced.
3. Resolve author checks in `main.reviewed.tex`: AAOS image metadata; comparison-table classifications and literature details; the deployment figure's source/path and three-pass values; test 04083 missing export time; the supporting sample for cost-per-term claims; recorded host metadata; and obligation-completeness provenance.
4. Keep measured deployment claims limited to the native CPU benchmark, finite input corpus, and recorded latency/RSS. Do not claim JNI integration, NPU placement, energy, thermal/frequency telemetry, or universal binary equivalence.
5. Build the complete submission, verify citations and figure inclusion, and meet the applicable paper-length/template constraints. `main.tex` is a body fragment in the reviewed checkout, and the original referenced `BKP/figs/vim3_deployment_results_v2.pdf` path was not available there; a final complete-paper build was not verified.

Do not treat this update as a rerun of ESBMC or the deployment experiments. It records a review of saved artifacts and the exact scope those artifacts support.

## 12. Results Added After the Final-Pass Review (2026-09-29 to 2026-10-15)

This section supersedes Sections 2, 7, and 11 where they conflict. The numbers below were audited from the raw JSON on the dates given. Earlier on 2026-10-29 the external output disk was **not attached** (`lsblk` showed no `sde`), which is why Section 2 found `output/` empty. It was remounted later that day, and the results below were re-confirmed from the raw files.

### 12.1 October validation campaign: completed, 5/20 VERIFIED

Section 7 is wrong that this protocol has no outputs and must not be run. It ran to completion on 2026-10-14/15, **before** its 2026-10-21 cutoff, from `output/crown_chain_v2_validation_20261014/`. It was launched through `tools/ssv2026/run_protocol_monitor.py`.

- **Pre-registration evidence:** the protocol's `drafted_at` is 2026-10-14T12:52:50-03:00. The first screen log line is 12:53:05.
- **Sample:** validation split, n=20, seed 20261014, epsilon 1 raw byte.

| Outcome | Count | Images (ordinal) |
|---|---|---|
| VERIFIED | **5** | 006, 008, 011, 012, 017 |
| CHECK_INCONCLUSIVE | 1 | 002: the low-memory guard aborted L2 after 11,066 of 45,049 checks (11,065 VERIFIED, 1 UNKNOWN); L3/L4 never ran |
| FINDER_INCONCLUSIVE | 13 | 001, 003, 004, 005, 007, 009, 010, 013, 014, 015, 016, 018, 019 |
| MISCLASSIFIED | 1 | 000 |

- Certified rate 5/20 = 25% (Wilson 95% CI ≈ 11–47%), or 5/19 among correctly classified.
- **Provenance wording.** The protocol file is still untracked in Git, so call it a *timestamped, pre-specified* validation protocol, not "pre-registered". The test protocol is committed (`92f17ea2`, 2026-09-25) and keeps the stronger claim.
- **Same-track frames.** The 20 images include 3 tracks with two frames each (`00013/00012`, `00023/00004`, `00010/00066`), so they cover 17 physical tracks. Inference is over images, not over independent signs.
- ESBMC closed 5 of the 6 finder-positive images.
- No FAILED checker obligation in any CROWN shard.

Per-certificate audit, from each `complete/complete_check_summary.json` (2026-10-15). Every row passed all of these checks:
- `status_counts` is all VERIFIED, and `harness_hashes_rechecked` equals `harnesses_verified`;
- `deployment_sha256` is `dd535f54…45097`, with binding `EXACT_ORDERED_MATCH`;
- `encoder_status` is VERIFIED, and 42/42 competitors are closed;
- `checker_sources` is identical to those of test certificate 005;
- `split` is `validation`.

| Ordinal | Image | Target | Harnesses | Min raw margin | Summed per-check elapsed | Peak aggregate RSS |
|---|---|---|---|---|---|---|
| 006 | 00013/00012_00001 | 13 | 76,383 | 1,657 | 22.63 h | 3.9 GiB |
| 008 | 00033/00004_00017 | 33 | 64,799 | 1,129 | 20.38 h | 3.9 GiB |
| 011 | 00038/00049_00026 | 38 | 53,299 | 2,731 | 15.19 h | 4.0 GiB |
| 012 | 00025/00014_00003 | 25 | 67,850 | 321 | 19.57 h | 3.9 GiB |
| 017 | 00039/00007_00017 | 39 | 45,836 | 352 | 13.01 h | 3.9 GiB |

For image 006 alone, the shard costs were:

| Layer | Obligations | Hours |
|---|---|---|
| L0 | 2,057 | 0.11 |
| L1 | 10,655 | 1.08 |
| L2 | 41,025 | 13.05 |
| L3 | 19,713 | 5.90 |

Its remaining obligations were in L4 and the aggregate.

**Reporting rules for this campaign**
- These are **validation** results. Report them separately. Never pool them with the 80-image test cohort, and never recompute the test rate. The primary test result remains 1/80, and the disclosed rerun remains 2/80.
- The campaign ran 4 jobs on a stable machine, and 5 of 6 finder-positive images closed. In the test rerun, 16 of 18 finder-positive images aborted, under 12-job contention and WSL crashes. This *suggests* the test aborts were mostly environmental. It does **not** show that the 16 test aborts would have certified; the samples differ, and n is small.
- **Section 5 still applies to the validator, but the exports themselves are clean.** These five certificates were checked by the same validator, which has the layer-traversal gap and the chain-fact identity gap. On 2026-10-29, with the disk mounted, the outcomes and the five certificate audits above were re-confirmed from raw JSON. A read-only audit of every chain in all 7 completed exports (06267, 04083, val 006/008/011/012/017) found:
  - **0 chains out of layer order.** Every chain is `affine(L), relu(L-1), …, affine(0), concretize`.
  - **0 chain-backed facts with a mismatched identity.** Every `chain_id` equals `chain/L{layer}/n{neuron}/{side}` of its fact and exists in the export.
  - **Margin chains:** exactly 42 per export, all `chain/margin/{target}-{j}`, covering every j ≠ target and matching `margin_closure`.
  - **Two extra chains per export**, `chain/logit/{target}/lower|upper`. These are the target-logit bounds; they equal `export.json` `raw_t` and are by design.

  So none of these exports exploits the two validator gaps. The gaps remain open in the code, and the trust table must still describe them until they are fixed.
- **Image 002 is not one query away from certification (corrected 2026-10-29, found by GPT in `docs/claude_gpt_chat.md` Turn 2).** `002/check_l2/check_summary.json` has `harness_count=45049`, 11,066 records and `aborted_low_memory=true`. The single UNKNOWN (`chain/L2/n2052/lower/s4`, 3.15 s, return code 15) is an artifact of the abort, not a hard query. Completing 002 needs 33,984 L2 checks without a VERIFIED result (33,983 never run plus the 1 UNKNOWN), all of L3 and L4, and aggregation, in a new amendment directory. Do not edit the saved summary. Any eventual certificate is a separately disclosed follow-up, never part of the 5/20.
- `tally` had not been run for this protocol as of 2026-10-15, so `results.json` may not exist. Run `run_crown_chain_v2_protocol.py tally` once the disk is mounted.

### 12.2 Corrections to Section 11 and older notes

- **AAOS identity is now verified.** A read-only `adb` readout on 2026-09-29 is saved at `output/vim3_runs_20260928/board_props_20260929.txt`. It shows:
  - `ro.hardware.type=automotive`, `feature:android.hardware.type.automotive`, and the `com.android.car` (CarService) and `carlauncher` packages;
  - `getconf PAGESIZE=4096`, kernel `7.1.0-mainline-maybe-dirty-4k`.

  The Android version needs care. The board reports `codename=Baklava`, `preview_sdk=1`, `sdk=35`, `release=15`, with build `BP1A.250505.005.B1` userdebug. Describe it as an **Android 16 pre-release (preview) build, SDK_INT 35**, not as stable Android 16 / API 36.
- **Median screen margin of the 54 FINDER_INCONCLUSIVE test pairs is −6,038.5**, not −1,975. That figure in `RESULTS_SO_FAR.md` is wrong. The median over all 72 correctly classified pairs is −1,959.5.
- **Cost per term.** The "177–188 µs per term" figure was never saved as an artifact. Recomputed from saved records as seconds divided by export `multiply_terms`:

  | Image | `affine_chunk` only | Whole summed check |
  |---|---|---|
  | 3652 | 202 µs | 279 µs |
  | 06267 | 193 µs | 260 µs |
  | 04083 (12-job contention) | 333 µs | 446 µs |

  Cost is still linear in exported terms, and L2 dominates.
- **Units of the cost fields.** `summed_esbmc_seconds` is the sum of per-call elapsed time (`check_crown_chain_v2.py:123`). `maximum_shard_peak_aggregate_rss_mib` is the process-tree aggregate over concurrent jobs. The per-call limit is 4 GB.
- **Box stage for 06267 and 04083.** Each had 689 ESBMC calls: 605 VERIFIED and 84 FAILED abstract margins. The summed elapsed time was 2,002 s and 2,210 s respectively. The 616 s figure for 06267 is stage wall time.
- **Box blocks used by the 06267 certificate.** It lists 588 of the 589 verified blocks. This is benign: block L2/46's bounds came from CROWN chains, and `export.json` records `box_blocks_used: 588`. It does illustrate that completeness is checked against the certificate's own declaration, not against an independent enumeration.
- **Export time for 04083.** 75.8 s export plus 33.8 s screen, about 110 s.
- **ESBMC assertions never contain SHA-256.** Hashes are used only by the Python layer, as identity checks binding obligations to one C file, one certificate, and one checker version. They establish identity, not semantics. In `chain_close` harnesses, the step-link assertions are trivially true (`__ESBMC_assert(1, …)`, `c == c`), so chain linkage is guaranteed by the Python generator, not by ESBMC. State this in the trust table.
- **Monolithic baseline** (`output/ssv_scalability_20260928/`). These runs used the compact int16 C:
  - whole network, concrete input: Z3 TIMEOUT at 600 s, ~2.8 GiB;
  - L0, 24 outputs, 12 free bytes: Bitwuzla VERIFIED, interval containment only, 186 s, 4.4 GiB;
  - L0 with 18 free bytes: OUT_OF_MEMORY;
  - L0, 48 outputs, concrete input: OUT_OF_MEMORY.

  The VERIFIED rows prove interval containment, not robustness. Label them as such.

### 12.3 Paper artefacts produced in this period (untracked)

- `docs/ssv2026/paper_text/table_cost.tex`: corrected cost table. It has a Property column (Interval vs Robustness), uses GiB consistently, and uses daggers for the development image and the 84 box-margin FAILs. It does not yet include the five validation certificates.
- `docs/ssv2026/methodology_v3.drawio`: IEEE-sized methodology figure. Use v3; v1 and v2 are superseded. It deliberately shows:
  - ε = 1 byte, not 2/255;
  - GCC static binaries, no NDK/JNI, with SHA-256 matching the source rather than the binary;
  - FAILED as "a bound refuted, not a counterexample".
- `docs/ssv2026/plot_images.py`: the deployment figure still uses single-pass latencies (31.68 / 27.94 / 21.99 ms). Switch to the three-pass values in Section 11.
- Paper sentence on test sampling, with the essentials kept: *"The 80 test images were drawn uniformly without replacement (seed 20260925), fixed in a protocol committed before any test run."*

### 12.4 Destructive changes already made (cannot be undone)

- **2026-10-03:** at the user's request, all `check_l*` directories, including the `.resource_abort.*` ones, were deleted for the **16 CHECK_INCONCLUSIVE images** of `output/crown_chain_v2_test_20260925`. That was 90 directories, and the folder went from 192 GB to 29 GB. Kept for each image: `outcome*.json`, `screen.json`, `export.json`, `certificate/`, `box/`, and logs. The two certified images (005, 015) and the 3652 run are untouched. Resuming those 16 images now requires regenerating harnesses and re-checking (12–20 h each). The empty, invalid `011/check_l4.resource_abort.*/check_summary.json` that the final-pass review flagged was among the deleted files.
- **2026-10-03:** 50 GB of WSL ESBMC crash dumps were found in `C:\Users\victo\AppData\Local\Temp\wsl-crashes`. They were dated 2026-09-28, the scalability-curve OOMs. By 2026-10-14 the directory was gone; they were not deleted by the assistant.
- **2026-10-14:** `output/` (69.4 GB, 955,968 files, rsync verified with zero differences on a second pass) was moved to the external HDD. The SSD copy was deleted. `output` is now a symlink to `/mnt/wsl/PHYSICALDRIVE1p1/preqbmc_ext/output`. As a result, Git shows the tracked `output/sign_deep_source_model_test_20260922/test_evaluation.json` as deleted, and `output` as untracked. The file is on the HDD, not lost. **Do not commit either change.**

### 12.5 Machine constraints observed

- **Host and WSL memory.** The host is a Core Ultra 7 155H with 31.5 GB RAM. `.wslconfig` sets `memory=20GB`, `swap=60GB`, and `autoMemoryReclaim=dropCache`.
- **What causes the freezes.** WSL freezes come from **Windows** starving: 0.4 GB available, with 14.75 GB held by vmmemWSL, about 10 GB of it Linux page cache. Running natively on Windows would not help. The driver needs `systemd-run`, ESBMC needs the same memory, and Windows pages instead of OOM-killing.
- **Recommended, not yet applied:**
  - `memory=16GB`, `swap=8GB`;
  - at most `jobs × memlimit ≤ 12 GB`;
  - exclude `output/**` from VS Code indexing. Pylance reached 3.3 GB, and `~/.cache/vscode-cpptools` reached 40 GB.
- **Google Colab does not help.** ESBMC/Z3 are single-core CPU per call with no GPU use. There is no systemd, sessions drop, and the protocol pins `jobs=4`, one image at a time.

### 12.6 Before using any of this in a new session

1. Mount the disk from an **elevated** Windows PowerShell: `wsl --mount \\.\PHYSICALDRIVE1 --partition 1`. Check that `ls ~/code/preqbmc/output/` lists run folders. The same partition has also appeared at `/mnt/aosp`, which is not configured in fstab.
2. ~~Reread the outcomes and certificate summaries~~ — done 2026-10-29; they match 12.1.
3. ~~Run the chain-order and fact-identity audit on the seven exports~~ — done 2026-10-29; clean (see 12.1).
4. Run `run_crown_chain_v2_protocol.py tally` for the validation protocol; `results.json` does not exist yet.
5. Add the validation campaign to the paper as a separate paragraph or table row, with its own denominator.


## 13. Agreed plan and P1 status (2026-10-30)

The plan was agreed with GPT in `docs/claude_gpt_chat.md` Turns 1–7. Each phase needs João's separate approval:
- **P1:** checker closure.
- **P2:** read-only re-audit of the 7 exports.
- **P3:** 2 h bounded finder pilot.
- **P4:** freeze, without committing.
- **P5:** two-stage validation, N=400 screened and m=10 checked, finite-population interval.

**P1 is done in the worktree** `/home/joao/code/preqbmc-qformat` (branch `feat/crown-v2-qformat`). It is uncommitted, and the main checkout is untouched. The full record is in that worktree at `docs/ssv2026/P1_checker_closure.md`. In summary:
- **Both demonstrated gaps are closed.**
  - Layer traversal is now enforced. A mocked check shows the old code *accepted* a skipped-layer chain.
  - Chain-fact identity is now enforced.
- **Clamp references** are checked by identity, and the target must equal the recorded nominal class.
- **Independent coverage module** (`crown_chain_v2_coverage.py`): it derives required assertions from the CSR rows and the validated steps, never from the renderer.
- **New gap found by P1.** The renderer emitted no assertion for affine columns whose products cancel to 0, so ESBMC never checked those products. In 06267 this is 448 columns across 241 of 9,832 affine steps. The renderer is fixed. The historical exports will therefore fail independent coverage in P2 until a supplementary check is run under a new ledger. This is a trust gap (the values were computed exactly in Python), not evidence of a wrong certificate.
- **Tests:** 25 new mutation and control tests pass without ESBMC. The existing suites pass, except the protocol-pin test, which fails by design.
- **Real-data check:** the corrected loader accepts 06267 and validation 017 as `STRUCTURALLY_CLOSED`.

### P2 result (2026-10-30, MEASURED)

`tools/ssv2026/p2_reaudit.py` in the worktree re-audited all 7 exports under the P1 checker. It wrote only under `output/p2_reaudit_20261030/`; no old summary was modified.
- **Structure:** all 7 PASS (layer traversal, fact identity, clamp references, binding). All are `STRUCTURALLY_CLOSED`.
- **Coverage before the supplement:** every missing assertion was exactly an affine column whose product cancels to 0, skipped by the old renderer. Per export: 360–535 columns. There were 0 unexplained gaps, 0 unexpected assertions and 0 duplicates.
- **Supplement:** 1,510 small harnesses for exactly those columns, same ESBMC settings (4 jobs, 60 s, 4 GB, paper-z3). All 1,510 VERIFIED, in 241 s of summed ESBMC time.
- **Verdict:** all 7 are **ACCEPTED_WITH_SUPPLEMENT**, with complete independent coverage over old plus supplementary records. The certificates 06267, 04083 and validation 006/008/011/012/017 now hold under the corrected checker. The paper should state that the zero-product columns were checked in a later supplementary pass.

### P3 result (2026-10-31, MEASURED): gate not met

Bounded discrete search over integer ReLU coefficients (ADAPTIVE + p/4 scalars per layer, margin roots only, baseline intermediate bounds) on the 19 October development images, 29 min of the 2 h cap. **0 of 13 inconclusive images recovered, 0 of 6 positives lost.** Improvements only on far-negative images (e.g. 003: −63,487 → −56,860). On the near misses (013, 016, 018, …) ADAPTIVE beat every scalar at every layer. The cost counter reproduces the October export term counts exactly. Per the plan, the improvement branch stops; the negative is reportable. Details: worktree `docs/ssv2026/P3_finder_pilot.md`, outputs `output/p3_finder_pilot_20261031/`.

### Status 2026-10-31 (handoff)

P1 and P2 committed on the worktree branch (`08b682be`, `d4d8a752`). P3 failed its gate (above). Diagnostics phase 1 (spec `finder_diagnostic_spec_013_016.md`) implemented; D-P preflight passed on 013/016/017 (`output/crown_diagnostics_20261031T131653Z/`); D-0/D-V/D-T not yet run. Full handoff and next steps: `docs/HANDOFF_20261031.md`.

### Transfer note: diagnostic import isolation (2026-10-31)

This context was updated for continuation on another computer. Use `docs/HANDOFF_20261031.md` and `docs/RESTORE_20261031.md` to restore the main checkout, `feat/crown-v2-qformat` worktree, external output disk, environment, and Claude history. The final user request here was to preserve this state and resume diagnostics on the other machine; it did not authorize launching the remaining stages on this machine.

**Import-isolation issue identified and fixed in the diagnostic driver only.** The P3 run used cwd `/home/joao/code/preqbmc`; `affine_export.py` later executes `sys.path.insert(0, "src")`, and the `quad` environment has `__editable__.preqbmc-0.1.0.pth` pointing at the main checkout. Thus P3 loaded some helper modules from the main checkout. The P3 imported runtime files were compared against the selected worktree and were byte-identical; the differing `verification/crown_chain_v2.py` was not imported. This is a provenance limitation, not evidence that P3's numerical results were wrong. Do not change P3's numerical conclusions.

The new `src/scripts/diagnostic_import_guard.py` is loaded by absolute file path before project imports. It pins project-local top-level packages/modules dynamically from every importable root under the selected worktree `src/`, including namespace packages, and adds fixed finder search paths so a later relative `sys.path.insert(0, "src")` cannot redirect imports. The audit also rejects any loaded project module outside the worktree and records resolved module paths, source SHA-256 values, effective `sys.path`, editable `.pth` files, and worktree/main-checkout paths. A subprocess regression test puts the main checkout first on `sys.path`; it verifies all project imports resolve to the worktree. Other tests cover pre-guard imports and an intentionally injected outside-worktree module.

**D-P preflight is measured as passed** in `output/crown_diagnostics_20261031T131653Z/preflight.json` and `manifest.json`: all imports resolve within `/home/joao/code/preqbmc-qformat/src`, audit has zero violations, the editable install is detected but cannot escape the guard, and the late `"src"` entry remains harmless. The exact P3 roots match for all 42 competitors on each image 013/016/017; baseline margins, closed counts, chain counts and multiply-term counts match; C/Python, O0/O2 and observer parity report zero mismatches; frozen production hashes match their recorded expected values. Phase-1 tests and source/config hashes are summarized in the handoff; the full audit/module hashes are in the saved manifest.

Diagnostic implementation currently supports D-P (`preflight`), D-0 (`epsilon0`), D-V (`violations`), D-T (`trace`) and `report`. D-0/D-V/D-T are **not yet run**. Interventions I-S/I-B/I-R/I-P/I-Q/I-C/I-E and cost accounting are specified, not implemented. The next scoped work is to continue D-0 → D-V → D-T → report in the same provenance-pinned run on the restored machine, then decide whether any intervention warrants a separate phase. Preserve the four-hour global diagnostic cap and per-stage caps in `experiments/crown_diagnostics_013_016_017.json`; resource availability and the mounted output disk must be checked on the destination machine before resuming.
