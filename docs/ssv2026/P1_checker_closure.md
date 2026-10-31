# P1: v2 checker closure (worktree `feat/crown-v2-qformat`, 2026-10-30)

Uncommitted. The pinned main checkout is unchanged. Every historical ledger and protocol stays as it was.

## What changed

| Change | Where | Closes |
|---|---|---|
| Each chain must step `affine(L), relu(L-1), …, affine(0), concretize`, with step IDs `{chain}/s{n}` | `validate_chain`, `expected_traversal` in `src/verification/crown_chain_v2.py` | Demonstrated gap: a chain could skip layers |
| A chain-backed fact must cite `chain/L{layer}/n{neuron}/{side}` of that same fact | `_validate_facts` | Demonstrated gap: a fact could cite another neuron's chain with an equal value |
| A chain must sit in the shard of its own λ-layer | `load_certificate_v2` indexing | Misfiled chains |
| A single clamp-closure function for the loader and the aggregator. Roots come from the output dimension and the recorded nominal class. Each clamp edge must cite a bound on the logit and side its rule is about. Top and bottom rules cannot be swapped. | `validate_margin_closure`; `aggregate_crown_chain_v2.py` | Clamp references were checked by value only |
| Affine harnesses assert every touched column, including columns whose products cancel to 0 | `_render_affine` | **New gap found in P1.** For `g = 0` columns the renderer emitted no assertion, so ESBMC never checked those products. In export 06267 this was 448 columns in 241 of 9,832 affine steps. |
| Independent coverage: required assertions derived from CSR rows and validated steps (no renderer import) and compared with the labels in the verified, hash-matched harness files. Checks exact match, no duplicates, and that every ReLU instance lies inside the envelope of its shared lemma. | new `src/verification/crown_chain_v2_coverage.py`; called per shard by the aggregator | Completeness was counted against the renderer's own manifest |

The checker source pins now include the coverage module. Shard summaries made by the old checker are therefore refused by the new aggregator. Old results cannot inherit the new checker silently.

## Evidence (MEASURED)

- `src/tests/test_crown_chain_v2_closure.py`: 25 tests pass, with no ESBMC needed.
  - **Negative mutations, all rejected:**
    - M1 skipped layer. A mocked variant without the traversal check *accepts* the same chain, so this was a real false-acceptance path.
    - M2 duplicated layer; step ID out of position.
    - M3 fact citing another neuron's chain; M4 fact citing the other side's chain.
    - Clamp edge citing another logit; clamp edge citing the wrong side; bottom rule in a top slot.
    - M7 missing competitor; M8 target different from the nominal class; non-positive margin.
    - M5 missing affine chunk; M6 missing ReLU chunk; missing shared lemma.
    - Missing concretize, affine_close, relu_close or chain_close.
    - The old renderer's skipped zero-product column.
    - M11 duplicated record; M12 harness edited after verification; non-VERIFIED record.
    - ReLU instance outside the shared lemma envelope.
  - **Positive controls, all accepted:** valid chain; own-chain fact; closed margins; complete coverage; M10 reordered records.
- **M9, altered deployed weight:** already covered by `test_dense_binding_rejects_weight_permutation` in `test_crown_chain_v2.py`.
- **Existing suites** (`test_crown_chain_v2`, `test_crown_chain_certificate`, the rest of `test_crown_chain_v2_protocol`) pass. The exception is `test_current_sources_match_the_drafted_protocol`. It fails by design, because the sources no longer match the old dry-run pins. The finder files were already modified in this worktree before P1.
- **Real-data positive control (read-only, no ESBMC):** the corrected loader accepts both test export 06267 (3,348 chains, 17,672 facts, 207 s) and validation export 017 (3,240 chains, 17,849 facts, 37 s). Both are `STRUCTURALLY_CLOSED` with an `EXACT_ORDERED_MATCH` binding.

## What remains trusted

1. **Harness translation.** Coverage matches obligations by assertion label. That each C expression encodes the obligation its label names is trusted, and so is the renderer's arithmetic in those expressions.
2. **Step linkage.** `validate_chain` passes each step's output coefficients and constant to the next step in Python. The `chain_close` link assertions restate this and are not an independent check.
3. **Closure arithmetic** and the clamp rules in `validate_margin_closure`.
4. **Deployment binding.** The Python CSR and geometry parser; ESBMC corroborates sampled rows only.
5. **Box ledger and encoder correspondence.** Python selects the image, ε and preprocessing that the ESBMC encoder and non-vacuity harnesses check.
6. **Deployment MAC overflow.** Python static envelopes, not an ESBMC overflow proof.
7. **ESBMC 7.11 with Z3** under the `paper-z3` flags, including interval analysis.
8. **Compiler, libc and hardware.**

## Consequence for P2 (written before P2; result in `P2_reaudit.md`)

Historical shards lack assertions for the zero-product columns. So the 7 saved exports should pass the structural checks but **fail independent coverage**. For 06267 the gap is 448 column checks. P2 must record that precisely. The cheapest honest closure is a supplementary set of small harnesses for exactly those columns, under a new provenance ledger. Old shard summaries must not be edited.
