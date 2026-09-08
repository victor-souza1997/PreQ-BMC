# Verified affine-box tightening

`--tighten-verified-bounds` is an opt-in scalability refinement for sound,
derived, layer-scope verification. It does not replace the MILP preimage
contract and it does not change quantization.

For hidden layer l, let A(l-1) be the already verified integer input box,
Wq(l) and bq(l) the candidate's quantized parameters, S(l-1) the input scale,
and C(Q) the signed clamp for total width Q. PreQ-BMC computes:

```text
R_low(l,j)  = C(Q)(round_away(min_{a in A(l-1)} Wq(l,j) a / S(l-1)) + bq(l,j))
R_high(l,j) = C(Q)(round_away(max_{a in A(l-1)} Wq(l,j) a / S(l-1)) + bq(l,j))
```

The extrema are exact for one affine row over a Cartesian input box. They are
not claimed to be the exact globally reachable set of the complete network.
Round-half-away-from-zero and the signed clamp are monotone non-decreasing, so
endpoint propagation is sound.

The existing hidden harness continues to prove its preimage obligation. The
same harness additionally asserts:

```text
forall j: R_low(l,j) <= z_low(l,j) <= z_high(l,j) <= R_high(l,j).
```

Only after every neuron block is `VERIFIED` does the next layer receive:

```text
A(l) = ReLU(G(l)) intersect ReLU(R(l)),
```

where G(l) is the original verified preimage guarantee. Consequently the new
box is a proved strengthening of the existing contract, not an assumed
invariant. All blocks use the same candidate Q/I/F.

## Arithmetic prerequisite

The Python endpoint calculation uses unbounded integers, whereas generated C
uses `int64_t` values and a signed `__int128` MAC accumulator. Before ESBMC
is called, PreQ-BMC checks that:

- weights, biases, and input endpoints fit `int64_t`;
- every product and every MAC prefix endpoint fits signed `__int128`;
- intermediates in round-half-away-from-zero fit signed `__int128`;
- the rescaled accumulator plus bias fits signed `__int128`; and
- `TOTAL_BITS` is representable by the generated `int64_t` storage.

An unsafe candidate is rejected before harness generation. A
`deployed-transfer` result under bound tightening additionally requires this
arithmetic-safety record and every selected tightening proof to be
`VERIFIED`.

## Running the matched pilot

```bash
preqbmc reproduce \
  --config experiments/iris_seeds_verified_bounds_pilot.json \
  --aggregate
```

The configuration contains fixed-F=8 control and treatment runs over the same
Iris and Seeds regions. Per-run details are stored in
`pipeline_summary.json`. Aggregation adds the treatment columns to
`all_experiments.csv` and writes one row per hidden layer to
`table_verified_bound_tightening.csv`.

This option is intentionally incompatible with heuristic/unsound contract
tolerance, disabled chaining, network-scope harnesses, and non-ESBMC
verification.
