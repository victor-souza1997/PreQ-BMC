# Critical Review of the PreQ-BMC Work

## 1. Overall assessment

The work is scientifically promising and suitable for a master's methodology because it combines three valuable pieces:

1. formal layer-wise contract preservation through ESBMC;
2. synthesis of per-layer fixed-point bit-widths rather than manual quantization;
3. deployment-quality validation through Python/C fixed-point semantics, accuracy, mismatch, and saturation metrics.

The strongest contribution is not simply “using ESBMC on a neural network.” The stronger contribution is:

> A layer-wise formal-method pipeline that synthesizes QNN bit-widths from backward preimage contracts and then closes the deployment gap with formal/empirical saturation and fixed-point semantic validation.

That is a defensible master’s contribution.

## 2. Main strengths

### 2.1 Good formal decomposition

Verifying a full neural network monolithically with a bit-precise bounded model checker is hard. Your decomposition into affine-layer contracts is the right engineering and methodological choice. It gives ESBMC smaller verification problems and makes the paper more explainable.

### 2.2 Stronger than plain post-training quantization

The method does not just train a model and quantize it. It searches for a bit-width configuration under a formal contract. This is stronger than reporting accuracy after quantization.

### 2.3 Saturation addition is important

Adding saturation is not a weakness. It is a good research insight. It shows that preserving a mathematical preimage contract is not the same as preserving deployable fixed-point behavior. This makes the paper more honest and more embedded-systems-oriented.

### 2.4 Python/C backend comparison improves credibility

The generated C backend and Python fixed-point comparison are very useful. They help prevent a common problem in quantization papers: reporting results for a high-level model that does not match deployment arithmetic.

### 2.5 Block-wise ESBMC is a practical formal-method contribution

The block-wise harness is valuable because it addresses solver scalability without changing the layer-level Q/I/F semantics. This should be emphasized as an engineering technique for making formal verification feasible.

## 3. Most important problems to fix

### Problem 1 — Paper config currently disables saturation checks

The current `experiments/paper_experiments.json` sets:

```json
"formal_saturation_check": false,
"empirical_saturation_check": false
```

Because defaults are merged into every run, the paper command currently disables both saturation mechanisms. This conflicts with the intended methodology if the article claims that saturation is part of the main pipeline.

Recommended fix:

- If saturation is part of the proposed method, set both to `true` in the default paper config.
- If you want an ablation, keep this config but rename it clearly, for example:

```text
paper_experiments_no_saturation_ablation.json
```

Then create:

```text
paper_experiments_main_saturation.json
```

with both saturation checks enabled.

### Problem 2 — README and config disagree about MNIST scope

The experiment README says MNIST is limited to `1blk_10`, `1blk_25`, and `2blk_25_25`, and excludes larger models. However, the current config includes `1blk_50`, `1blk_100`, and `2blk_50_50`.

Recommended fix:

- Update the README to reflect the real config; or
- remove the larger runs from the default config and keep them in a scalability/manual config.

This matters because inconsistent documentation reduces reviewer trust.

### Problem 3 — Be careful with the phrase “formal verification was not enough”

Do not write that formal verification is useless or insufficient in a generic way. That weakens your own formal-method paper.

Better framing:

> The original formal contract verifies layer-wise preimage preservation. However, deployment on fixed-point C code introduces additional semantic obligations, especially saturation and rounding behavior. Therefore, the method extends the verified property set with no-saturation and backend-equivalence evidence.

This turns the issue into a contribution.

### Problem 4 — Empirical saturation is not formal

The empirical saturation check is valuable, but it should never be described as proof. It is a dataset-level diagnostic and quality gate.

Correct wording:

> Empirical saturation diagnostics estimate whether the synthesized QNN behaves safely on the evaluation distribution. Formal no-saturation checks are used when an exhaustive interval-level guarantee is required.

### Problem 5 — Formal no-saturation may be conservative

The formal no-saturation harness uses interval bounds. If the interval over-approximates the true reachable set, a failed check may indicate possible saturation in the abstraction, not necessarily a real input that saturates.

Recommended wording:

> A formal no-saturation failure is treated as a refinement trigger. It is not automatically interpreted as a confirmed concrete adversarial/saturation counterexample unless the witness is inspected.

### Problem 6 — Clarify exactly what is verified

Your paper should not claim full end-to-end network equivalence unless you add an explicit full-network equivalence proof. The current method is best described as:

- formally verifying affine layer contracts;
- checking output robustness through the output-layer contract;
- composing the argument through backward preimage intervals;
- validating deployment behavior through fixed-point/C metrics;
- optionally verifying no-saturation per layer.

This is still strong. It just needs precise wording.

### Problem 7 — Activation scope must be explicit

The implementation and your description focus on affine layers. Hidden-layer ReLU behavior appears in the fixed-point backend, while the ESBMC contract harness is primarily affine/preimage-oriented. The paper should explicitly say whether:

- activations are included in the preimage abstraction;
- activations are verified as part of the ESBMC harness;
- activations are outside the formal scope and validated through execution.

A clean statement would be:

> In the current implementation, the formal ESBMC obligations are generated for affine transformations and output classification contracts. Activation effects are accounted for in the interval/preimage propagation and in the executable fixed-point backend, but independent activation-correctness verification is left as future work.

Only use this sentence if it matches your final code and advisor-approved scope.

## 4. Suggested scientific positioning

Your work should be positioned as an extension/adaptation of certified quantization, not as a claim that previous work is wrong.

Suggested contribution list:

1. A PreQ-BMC pipeline for synthesizing per-layer fixed-point QNN formats under local robustness contracts.
2. A layer-wise ESBMC verification backend that translates affine preimage contracts into C harnesses.
3. A block-wise verification strategy for scaling ESBMC to wider affine layers while preserving shared layer-level Q/I/F semantics.
4. A deployment-quality refinement loop using fixed-point accuracy, mismatch, Python/C equivalence, and saturation diagnostics.
5. A formal no-saturation verification property for fixed-point affine layers.
6. An experiment runner that exports reproducible CSV/JSON/plot artifacts for dissertation and article evaluation.

## 5. Recommended thesis narrative

A strong narrative is:

```text
Original problem:
  Quantization reduces memory and computation, but can break robustness and deployment semantics.

Formal-method insight:
  Robustness can be decomposed into layer-wise preimage contracts.

Engineering challenge:
  Bit-precise verification of whole QNNs does not scale well.

Proposed method:
  Use backward preimage synthesis to guide bit-width search, then verify each affine layer with ESBMC.

Deployment gap:
  A layer contract may hold while fixed-point saturation/rounding affects practical inference.

Extension:
  Add formal no-saturation verification and empirical fixed-point/C diagnostics.

Result:
  A QNN configuration with explicit formal-contract evidence and deployment-quality evidence.
```

This is coherent and does not attack formal methods.

## 6. Recommended article structure

### 6.1 Introduction

- Embedded neural networks need reduced precision.
- Naive quantization can break robustness.
- Formal methods can certify properties but face scalability issues.
- The paper proposes layer-wise ESBMC-guided bit-width synthesis with saturation-aware refinement.

### 6.2 Background

- Fixed-point quantization.
- Local robustness.
- Preimage analysis.
- Bounded model checking and ESBMC.

### 6.3 Method

- Define input region and target property.
- DeepPoly propagation.
- Backward preimage computation.
- Bit-width search.
- ESBMC harness generation.
- Saturation verification.
- Quality refinement.

### 6.4 Implementation

- Python pipeline.
- Generated C harnesses.
- ESBMC command profile.
- C backend generation.
- Experiment runner and artifacts.

### 6.5 Experiments

- Accuracy preservation.
- Formal verification status.
- Saturation ablation.
- Block-wise scalability.
- Resource metrics.

### 6.6 Limitations

- Local robustness, not global robustness.
- Affine-layer focus.
- Conservative intervals.
- ESBMC scalability.
- Empirical metrics are not proofs.

## 7. Strong claims you can make

These are safe if the reported experiments support them:

- The method synthesizes per-layer fixed-point formats rather than requiring manually selected bit-widths.
- ESBMC verifies generated C harnesses for layer-wise affine preimage contracts.
- Block-wise verification reduces individual ESBMC query size while preserving the layer-level shared bit-width semantics.
- Saturation diagnostics reveal deployment failures that are not visible from the original preimage contract alone.
- Quality refinement can recover deployment accuracy by increasing integer or fractional precision where needed.
- The pipeline reports formal status, timeouts, memory failures, and unknown results explicitly.

## 8. Claims to avoid

Avoid these unless you add more evidence:

- “The whole neural network is formally verified end-to-end.”
- “The QNN is robust for all possible inputs.”
- “Empirical saturation proves absence of saturation.”
- “A formal no-saturation failure means a real input definitely saturates.”
- “Formal methods are not enough.”
- “The C backend is verified equivalent to Python for all inputs.”

Better alternatives:

- “The method verifies layer-wise contracts that compose through preimage intervals.”
- “The robustness claim is local to the selected perturbation region.”
- “Empirical saturation diagnostics complement formal checks.”
- “The C backend is empirically compared to the Python fixed-point interpreter over the evaluation set.”

## 9. Recommended ablation text

Use this kind of paragraph:

> We evaluate two configurations: `formal_only`, which reports the first bit-width assignment satisfying the layer-wise preimage contracts, and `quality_refined`, which further enforces deployment-oriented criteria such as fixed-point accuracy, mismatch rate, and saturation. This separation allows us to isolate the effect of formal contract preservation from the additional constraints required by fixed-point deployment.

## 10. Recommended limitations text

> The current implementation focuses on local robustness around selected samples and verifies affine-layer contracts generated from backward preimage intervals. The no-saturation property is checked over interval abstractions and can therefore be conservative. Empirical accuracy and saturation diagnostics are evaluated over selected datasets and do not constitute exhaustive proofs. Finally, ESBMC scalability remains a limiting factor for large networks, motivating the use of block-wise harnesses and future work on tighter abstractions and incremental verification strategies.

## 11. Final recommendation

This work can be written as a strong master's methodology if you keep the claims precise. The best paper angle is not “formal verification alone failed.” The best angle is:

> Formal preimage-contract synthesis gives a principled way to find robust bit-widths, and saturation-aware verification/refinement is necessary to make those bit-widths deployable in fixed-point C implementations.

That is a good contribution.