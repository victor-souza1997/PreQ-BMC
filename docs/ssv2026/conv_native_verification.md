# Convolution-native proof-carrying verification

## Scope

This path verifies the compact fixed-point C implementation directly, without
lowering convolutions to dense Toeplitz matrices.  The frozen reference model is
the selected four-convolution GTSRB network.  Its held-out float test accuracy is
91.54%.  The current Q16/F8 quality control obtains 91.55% fixed-C accuracy and
exact Python/C logits on all 16 parity samples.

These measurements are not formal certificates.  Certification is local to one
frozen test image and one raw-byte L-infinity radius.

## Integer semantics

Each affine stage has one shared Q/I/F.  The compact Python evaluator, generated
C, and ESBMC harnesses use the same arithmetic kernel:

1. signed `__int128` multiply-accumulate;
2. round-half-away-from-zero rescaling;
3. quantized bias addition;
4. signed clamp;
5. ReLU where configured; and
6. a final signed clamp.

Construction rejects a network if a conservative MAC envelope can reach the
signed `__int128` limit.  This is a static integer bound, not an empirical test.

## Proof obligations

For input invariant A0 and exact integer transition Tl, the checker establishes
the following chain:

```text
Encode(X_epsilon) subset A0
Tl(Al) subset A(l+1)             for every affine stage
A(l+1) subset next assumption    for every bridge
logit_low[target] > logit_high[j] for every competitor j
```

Proof-carrying interval mode decomposes each transition proof into:

- ESBMC lemmas for monotonic round-half-away, clamp, and ReLU;
- the elementary ordered-integer signed-product endpoint lemma;
- ESBMC replay of every concrete convolution or dense endpoint certificate;
- ESBMC set-inclusion bridges; and
- ESBMC class-margin assertions.

The signed-product lemma is part of the explicit mathematical proof basis.  Its
universal nonlinear bit-vector encoding timed out with Z3 and exhausted the
configured memory with Bitwuzla.  Network-specific weights, endpoint products,
rescaling, clamps, and expected bounds are nevertheless replayed in generated C
and checked by ESBMC.  This distinction is recorded in `proof_summary.json` and
must not be described as an ESBMC proof of the generic multiplication theorem.

No bounded pilot can be certified.  A complete result is `VERIFIED` only when
the required and executed transition-block counts match, all bridges execute,
all competitor margins execute, and every proof-relevant result is `VERIFIED`.
`FAILED` interval margins are abstraction counterexamples, not deployed-network
counterexamples.

## Refinement and reuse

An interval-margin failure can trigger a stronger exact final-layer query over
the shared hidden variables.  Only the failed competitor is retried.  The failed
interval diagnostic is superseded only by a `VERIFIED` exact query; timeout,
memory exhaustion, and failure remain inconclusive.  The sparse-cut CEGAR
controller additionally enforces that no proposed relation may enter a retry
until its ESBMC validation is `VERIFIED`. The exact shared-input dense
cut-validation harness is implemented. Automatic cut candidate synthesis and
convolutional backward-cone validation remain future work.

Only successful ESBMC results are cached.  Cache identity includes generated C,
deployment source hash, weights and biases embedded in the source, Q/I/F,
invariants, property metadata, ESBMC executable and version, profile, timeout,
and memory limit.

## Commands

Quality evaluation:

```bash
preqbmc gtsrb evaluate-conv \
  --config experiments/sign_conv_native_pilot.json \
  --output output/sign_conv_native_quality
```

Bounded first-layer feasibility:

```bash
preqbmc gtsrb conv-pilot \
  --config experiments/sign_conv_native_proof_carrying_pilot.json \
  --output output/sign_conv_native_pilot
```

Complete attempt:

```bash
preqbmc gtsrb conv-verify \
  --config experiments/sign_conv_native_complete_proof.json \
  --output output/sign_conv_native_complete
```

Every output directory must be new.  Shared cache paths are optional and remain
sound because a changed obligation produces a different cryptographic identity.

## Measured status on 2026-09-23

The final Android-bound pilot verified raw-byte encoding in 55.52 seconds at
575.53 MiB and its 24-output first-layer certificate in 0.66 seconds at
87.19 MiB.  The prior exact 24-output transition reached 120 seconds and
about 1.45 GiB.  The complete proof-carrying attempt verified 488 transition
blocks, five bridges, input encoding, non-vacuity, and arithmetic obligations.
The first interval margin failed immediately; its demand-driven exact refinement
then timed out at 120 seconds.  The result is therefore `TIMEOUT`, not certified.

This path currently proves the integer QNN directly; it does not run a
convolution-native MILP source-model local-robustness gate, so its summaries set float-to-integer
transfer to false.

This result establishes that convolution representation and transition replay
are no longer the immediate bottleneck.  Correlation loss before the output
layer remains the next research task.  The present artifact must not claim a
complete GTSRB certificate or Android transfer guarantee until that obligation
is discharged and device parity is measured.
