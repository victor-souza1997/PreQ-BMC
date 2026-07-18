# Novelty Delta From Quadapter To PreQ-BMC

## Delta Table

| Area | Base Quadapter | PreQ-BMC Addition |
| --- | --- | --- |
| Quantization objective | Certified quantization strategy synthesis. | Deployment-aware fixed-point verification and diagnostics around the synthesized strategy. |
| Preimage reasoning | MILP/preimage-based quantization and model-level reasoning. | Layer contracts exported into ESBMC C harnesses for fixed-point implementation checks. |
| Arithmetic semantics | Primarily model-level quantization reasoning. | Bit-precise integer fixed-point semantics: `__int128` accumulators, round-half-away rescale, saturation, hidden ReLU, final clamp. |
| Verification decomposition | Whole-layer style reasoning. | Block-wise dense-layer ESBMC decomposition with shared layer `<Q,I,F>`. |
| Deployment implementation | Not the central artifact claim. | Python fixed-point interpreter and generated C backend with exact-match comparison. |
| Implementation-gap diagnostics | Not the main focus. | Keras vs Python fixed-point, Python vs C, saturation rate, logit error, and mismatch rate reports. |
| Saturation evidence | Not a separate formal artifact. | Optional formal no-saturation harness as a stronger property. |
| Solver reproducibility | Gurobi-centered MILP workflow. | CBC/python-mip backend for open-source reproducibility, with Gurobi retained as a reference backend. |
| Artifact automation | Research prototype scripts. | Article experiment automation, summary tables, artifact CLI, and reproducibility metadata. |

## PreQ-BMC Additions

1. C-level ESBMC verification harnesses.
2. Bit-precise fixed-point C semantics shared by deployment and verification harnesses.
3. Block-wise decomposition of dense hidden layers.
4. Python fixed-point interpreter and generated C backend.
5. Implementation-gap diagnostics:
   - Keras vs Python fixed-point;
   - Python vs C;
   - saturation rate;
   - logit error;
   - mismatch rate.
6. Formal no-saturation harness.
7. CBC/python-mip backend for open-source reproducibility.
8. Article experiment automation and artifact CLI.

## What Is Not Claimed

- PreQ-BMC does not claim global robustness; claims are local to the specified input region and benchmark configuration.
- PreQ-BMC does not claim to verify arbitrary neural-network architectures; the artifact targets the supported affine/ReLU MLP-style benchmark pipeline.
- PreQ-BMC does not claim that CBC and Gurobi are bit-for-bit identical solvers; they are expected to agree within the documented MILP tolerance for supported experiments.
- PreQ-BMC does not claim that empirical deployment diagnostics replace formal ESBMC checks.
- PreQ-BMC does not claim that formal methods are insufficient; it uses formal checking for the C-level contracts and reports non-proof statuses separately.
- PreQ-BMC does not claim formal no-saturation unless the no-saturation harness is enabled and verified.
- PreQ-BMC does not claim empirical saturation measurements are formal no-saturation proofs.
- PreQ-BMC does not claim legacy backdoor scripts have been migrated to the CBC-backed robustness pipeline in this milestone.
- PreQ-BMC does not claim GPU/TensorFlow floating-point execution is the deployed semantics; the deployment claim is about the generated fixed-point C backend and matching Python fixed-point interpreter.

## License Note

This repository contains an Apache-2.0 `LICENSE` file and Apache-2.0 package metadata. Before public artifact release, verify the redistribution status of any original Quadapter-derived code and third-party copied assets. If the Quadapter provenance is not Apache-compatible or otherwise unclear, treat that as a blocking release issue until it is resolved.
