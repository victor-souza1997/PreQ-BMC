# Current Results Presentation

The review package is in `../LADC_2026/results_revision/`:

- [Compiled Results preview](../LADC_2026/results_revision/preview.pdf)
- [Replacement Results LaTeX](../LADC_2026/results_revision/results_section.tex)
- [Integration and reproducibility notes](../LADC_2026/results_revision/README.md)
- [Reviewer evidence map](../LADC_2026/results_revision/reviewer_coverage.md)
- [Frozen numerical validation](../LADC_2026/results_revision/generated/validation.json)

A second package preserves the old manuscript structure and is the recommended
integration route:

- [Compact compiled preview](../LADC_2026/results_minimal_integration/preview.pdf)
- [Exact integration guide](../LADC_2026/results_minimal_integration/INTEGRATION_GUIDE.md)
- [Compact drop-in Results section](../LADC_2026/results_minimal_integration/results_drop_in.tex)
- [Minimal abstract and conclusion edits](../LADC_2026/results_minimal_integration/abstract_conclusion_edits.tex)
- [Optional appendix material](../LADC_2026/results_minimal_integration/optional_material.tex)

The original manuscript and experiment files were not modified. Six LaTeX
tables and two vector figures were generated from canonical run reports,
with source hashes, a run ledger, per-query records, and a replayable snapshot.
The parent paper directory is already Git-ignored; the package exists locally.

The main cohort has recorded encoded-integer contract evidence for 26 regions
(13 MNIST), while retaining `PARTIAL_VERIFIED` as the raw final status.
Clamp-inclusive robustness does not require a separate no-saturation proof.
Real-input encoding coverage remains a distinct condition, as described in
the methodology audit and the revised Results limitations. No raw outcome was
upgraded, no missing run was assumed to time out, and no new solver run was started.
