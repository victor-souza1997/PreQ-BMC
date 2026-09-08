# Related work: gaps to close (T6, T7)

---

## 1. QNNVerifier — the missing nearest neighbour (R1.11)

This is the most damaging omission in the paper. QNNVerifier verifies fixed-point
implementations of neural networks using ESBMC — the same model checker, the same
arithmetic target. An expert reviewer who knows it will read our "bit-precise
verification approaches are generally applied only after quantization" (§3) as
evasive rather than accurate, because that sentence describes QNNVerifier without
naming it.

The differentiation is real. Make it on **what is quantified over**, not on tooling:

| Dimension | QNNVerifier | PreQ-BMC |
| --- | --- | --- |
| Problem direction | Analysis: precision is an *input* | Synthesis: precision is an *output* of a search |
| Query structure | The property over the whole network | Per-layer contracts + explicit composition (Eq. 9) + output obligation (Eq. 18) |
| Where the contracts come from | n/a | MILP backward preimages, deflated by a derived error budget |
| Decomposition | — | Block-wise partition of output neurons (Eq. 14), proved equivalent |
| Artifact | Verified model | Verified model **and** the exported C backend, sharing one arithmetic kernel |

Two sentences in §3 plus a row in Table 1. State it flatly; the comparison is
favourable and defensiveness would be worse than the omission.

**Before writing:** pull the paper and check the current claims against it rather
than against memory — in particular how it handles the input region (discretization
vs. symbolic), and whether it targets fixed-point or floating-point semantics in the
configuration we would be compared against. Getting a competitor's capabilities wrong
in a rebuttal is worse than not citing them.

## 2. QA-IBP and certified training (R1.12)

QA-IBP is quantization-aware interval bound propagation for *training* certifiably
robust quantized networks. It belongs in the paper as a **complementary** line, not a
competitor: it makes the network easier to certify; we certify the deployed
implementation of a given network. One or two sentences, positioned as an orthogonal
axis, plus the observation that the two compose — a QA-IBP-trained network would
plausibly widen our preimages and raise our certification frontier, which is a
genuinely interesting future direction and costs nothing to say.

## 3. LADC prior editions (R1.6, T7) — **do not invent these citations**

The review form asks specifically for prior LADC work. This requires a real
literature pass; fabricated or mis-attributed venue citations are detected
immediately by that venue's own reviewers, and the cost is unrecoverable.

**Search protocol:**

1. LADC proceedings, recent editions (IEEE Xplore / ACM DL; LADC is the Latin-American
   Symposium on Dependable and Secure Computing). Search within-venue for:
   `neural network`, `machine learning`, `robustness`, `formal verification`,
   `model checking`, `fault injection`, `embedded`, `quantization`, `safety-critical`.
2. Cross-check the LADC steering committee and recent programme committees for authors
   working on ML dependability; their LADC papers are the natural citation set.
3. Check whether ESBMC-related work has appeared at LADC — the ESBMC group has a
   strong Brazilian presence and this is the most likely productive hit.
4. Record each candidate in the table below with a verified DOI before it enters the
   bibliography.

| Candidate | Venue/year | Why cited | DOI verified? |
| --- | --- | --- | --- |
| | | | ☐ |
| | | | ☐ |
| | | | ☐ |

**Framing to use once the citations exist.** The dependability argument is the bridge,
and it is a strong one:

- The certificate and the executed artifact are different objects. A robustness proof
  about the real-valued model is, from a dependability standpoint, a proof about a
  specification rather than about the implementation — the gap between them is a
  latent fault in the safety argument.
- Quantization-induced deviation is a *systematic* fault mode, not noise: rounding,
  saturation and accumulation order are deterministic and reproducible, so they can be
  verified exhaustively rather than tested statistically. This is precisely the setting
  where formal methods beat sampling, and it is the argument LADC's audience responds to.
- The failure taxonomy from T3 (source-ineligible / precision-limited / cost-limited)
  is itself a dependability contribution: it says which regions a deployment can rely
  on and, for the rest, *why* the evidence is absent — an inconclusive result with a
  named cause is usable in a safety case; an unqualified one is not.

Put this in §1 (¶3 of the restructured introduction) and echo it in §3.

## 4. Optional strengthening (not requested)

Only if space permits after T12's theorem:

- **Exact bit-level QNN verification** (BDD/SAT-based approaches to binarized and
  low-bit networks) — situates our fixed-point encoding among exact methods.
- **Quantization-error analysis in DSP/control implementations** — the classical
  ancestry of Eq. (10), and it lands well with a dependability audience.

Neither is required by the reviews. Do not expand §3 at the expense of §4's missing
theorem; the theorem is worth more.
