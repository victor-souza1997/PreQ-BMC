### What a VERIFIED rests on today

| Component | Decided by | Label |
|---|---|---|
| Box bounds on the deployed loop (`qnn_conv_native.c:87`), symbolic over the ε-box | ESBMC | PROVED |
| ReLU instances, class lemmas, rounding lemma | ESBMC + solver | PROVED |
| Affine, close, concretize steps | ESBMC constant folding, no solver | PROVED (ground evaluation) |
| Chain weights = deployed C (`crown_chain_v2.py:144-251`: regex plus a Python conv-index model) | Python | ASSUMED |
| Coverage, citations, margin closure (`:520-636`); column support (`:92-99`) | Python | ASSUMED |
| No `__int128` wrap (harnesses run without `--overflow-check`) | unstated | ASSUMED |

The 14-case regression mutates certificate *data*. It cannot catch a renderer or aggregator that
*omits* an obligation.

### Measured cost structure

MEASURED from the ESBMC logs of all 58,695 harnesses of image 3652:

- **44,251 harnesses never reach a solver.** They include every affine_chunk, affine_close,
  relu_close, concretize and chain_close, and 1,015 of the 15,430 relu_chunks; each prints
  `0 remaining after simplification`.
- **Of the 13.9 h:** 8.5 h (61%) is the front end (parse and GOTO creation), 2.4 h is symex, and
  0.76 h (5.5%) is SMT solving, all of it in relu_chunk.
- **affine_chunk:** 6.1 h front end, 2.3 h symex, 0 s solver. Peak memory: median 313 MiB,
  maximum 1,090 MiB. The 1.1 GB per-job sizing is set by the largest chunks at
  `max_affine_terms = 25000`.
- **Consequence:** a solver sweep cannot touch 72% of the time.
- **Test run, 2026-09-27:** the checker process itself peaks at about 2.7 GB while it renders a
  shard. That, with 4 ESBMC jobs and the editor, tripped the 6 GiB guard again.

### Proposals, ranked

**1. Cheaper ground harnesses; no change to the trust base.**
- *What:*
  - (a) Lower `max_affine_terms` (e.g. 8,000), so that the peak per job drops and more jobs fit.
  - (b) Render each chunk as `static const` arrays with a read-only loop and a scalar
    accumulator, instead of one 25k-term expression per column. Reads only: we measured that
    array writes make symex quadratic.
- *Soundness:* the same assertions.
- *Gate:* identical verdicts on all 15,792 affine harnesses of image 3652, and the regression
  passes.
- *Gain:* GUESS; (a) up to 2× from more jobs, (b) 2–4× on the 6.1 h front end.
- *Hours:* (a) 2, (b) 6.
- *Measure:* re-render layer 2 of 3652; wall time, peak RSS, verdicts.

**2. The trust-base question: a "verified generic checker, run natively" is REJECTED as a
replacement.**
- *Why:*
  - For ground obligations ESBMC is already a C interpreter.
  - "Prove the checker once" is vacuous for `g == Σ w·c`: the specification *is* the code, so
    ESBMC could prove only the absence of undefined behaviour and overflow.
  - Native execution swaps ESBMC for gcc and the CPU. That is a change of trust, not a proof,
    and it violates constraint 1.
- *Allowed:* as an untrusted pre-filter.
- *If the owner accepts it anyway:* a named trust tier, an ESBMC proof of no undefined
  behaviour or overflow over the envelope, and 15,792/15,792 differential agreement.
- *Gain:* 10.1 h down to minutes (GUESS). *Hours:* 8.

**3. Bind the chain weights to the deployed loop in ESBMC.**
- *What:* a harness per model (not per image), asserting
  `acc_deployed(row, x) == Σ csr_row·x` for nondet Q16 x over each row's receptive field.
- *Gain:* removes an ASSUMED row; no speed.
- *Hours:* 6.
- *Measure:* whether Z3 decides rows of up to 768 terms (layer 4) within 60 s (GUESS: uncertain).
- *Fallback:* disclose Python binding as trusted.

**4. Check the overflow argument.**
- *Argument:* |a|, |a_out| ≤ 2³¹ is asserted (`crown_chain_v2.py:878`). MEASURED from the
  deployed C: |w| ≤ 1,349 and fan-out ≤ 1,728, so |g| < 2⁵³. It is argued only on paper.
- *What:* one ground assert per affine_close, `max|w|·max|c|·fan_in < 2¹²⁶`, or
  `--overflow-check` on the ground harnesses.
- *Hours:* 1. *Gain:* GUESS: about 0 s, since it is constant-folded.

**5. An omission regression.**
- *What:* drop a column, a harness or a citation; the aggregator must refuse each.
- *Hours:* 2.

### Q2: gate for coverage proposals

Input splitting, or α/β-CROWN splits (Wang et al. 2021, R0 §3), need three things:
- an ESBMC coverage lemma: any x in the ε-box lies in some branch;
- branch-tagged facts, with no citations across branches;
- neuron splits on raw `r`, assumed only inside their own branch, with both branches present.

Every branch pays close to the full check cost, so Q1 first.

### Q3: the paper

- **Primary result: 1/80, Wilson 95% CI [0.2%, 6.8%].** Lin et al. 2021 (R0 §5) use binomial
  CIs; pre-registration is "not in sources", so it is a rigour point we can claim.
- **The disclosed rerun:** a separate table, with `aborted_low_memory` as the recorded cause.
- **Expected reviewer attacks:**
  - *"ESBMC is a calculator":* the decision-mode split above.
  - *"Aborts hide failures":* SIGTERM is recorded as UNKNOWN, and the primary result stays as
    recorded.
  - *"Python is trusted":* proposals 3 and 5.
- **Resources:** QNN versus float, memory, latency and energy on the VIM3 (Hashemi et al. 2016;
  Al-Shaikh et al. 2023, R0 §6).

### Q4: what is missing

- "58,695 ESBMC obligations" overstates the symbolic work: report 44,251 ground and 14,444
  solver-backed.
- The Python rows of the trust table are absent from the paper framing.
