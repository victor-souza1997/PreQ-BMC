# Restricted prototype: proof scope and limitations

The convolution-native extension and its proof basis are specified in
[conv_native_verification.md](conv_native_verification.md).  The affine-lowering
statements below describe the legacy/preimage adapter and remain applicable only
when that path is selected.

Let a crop x be an HWC array of unsigned bytes, D={0,...,255}^(HWC), and
R(x,e)={x' in D : max_j |x'_j-x_j| <= e}. Image decoding and sign detection
are outside this domain. E first selects nearest-neighbor coordinates using
floor(output_index * input_extent / output_extent), then applies

    E_j(x) = clamp_Q(round_half_away(x_selected(j) * 2^F / 256)).

For each selected coordinate let l=max(0,ceil(x_j-e)) and
u=min(255,floor(x_j+e)), computed with exact rational arithmetic. Monotonicity
of selection, scaling, rounding and clamping gives E(R(x,e)) subseteq [E(l),E(u)].
Repeated resize coordinates may add correlation; replacing their joint image
by a box only enlarges it. Normalized byte endpoints l/256 and u/256 are exact
float32 dyadics. The runner additionally checks containment in the **actual**
selected first-layer harness A0 before recording a byte-crop certificate.
This argument does not extend to bilinear resize, JPEG decoding, float-valued
sensor pixels, rotations, blur, occlusion, or physical perturbations by default.

For a convolution output coordinate (y,x,o), the real affine form is

    z[y,x,o] = b[o] + sum_{ky,kx,c} K[ky,kx,c,o] a[y*s_y+ky-p_y,x*s_x+kx-p_x,c],

with out-of-range inputs equal to zero. NHWC flattening maps each valid summand
to a column of a sparse affine matrix W. Hence W vec(a)+b_repeated is the same
real function. A shared quantizer maps duplicated coefficients to the same
integer, including exact zero padding. In the integer program, both direct
convolution and lowered C use

    t = sum_j w_int[j] * a_int[j]              (signed __int128)
    v = clamp_Q(round_half_away(t / 2^F_in) + b_int)
    a_next = max(v,0)                         (hidden layers only).

Equality relies on checked accumulator/rounding ranges, the same bias placement,
and one Q/I/F for the entire convolution layer. Tests compare independent
sliding-window and matrix implementations; they are not a general compiler
correctness proof. IEEE float32 convolution/matmul may reassociate sums, so
their comparison is reported as numerical testing, not bitwise equivalence.

The existing DeepPoly source gate and CBC MILP preimage synthesis receive the
lowered affine network. Numerical source/preimage checks are not new formal
IEEE proofs. Candidate obligations remain preimage-derived; accepted integer
layer bounds, cut validity and output properties are discharged through the
existing ESBMC pipeline. Input coverage is checked separately rather than
inferred from source eligibility or a status string.

For a layer whose output indices partition into blocks J_b, the proof rule is

    (for every b, {A_l} C_l restricted to J_b {G_l restricted to J_b})
       implies {A_l} C_l {G_l},

only with the **same quantized parameters and QIF** in every block. Composition
also requires activation(G_l) subseteq A_(l+1). No counterexample-derived cut
may enter an assumption until the existing exact-prefix validation accepts it.
A solver counterexample may be spurious relative to reachable hidden states;
replay and refinement do not imply that every failure is a real adversarial image.

The final classifier selects the lowest-index maximum logit. Output checks
must cover every competing class, including that tie rule. Clamp-inclusive
contracts describe saturating inference; proving absence of saturation is a
separate stronger property and is not claimed here.

A byte-crop integer-C certificate is conditional on arithmetic safety, complete
unwinding, validated assumptions, chaining and artifact identity. The Android
transfer additionally needs the exact encoder, generated source and tested
target toolchain. The current runner deliberately leaves Android transfer false.
Neither passing unit tests nor an ESBMC VERIFIED label proves all these trust
assumptions automatically. No vehicle-level safety/security theorem is claimed.

Experimental denominators distinguish images, image/radius regions, variants
and timing repetitions. Report source eligibility, encoded verification,
input-bridge completion and Android parity separately. Preserve resource-limit
outcomes and preimage inconclusiveness. Do not turn missing runs into solver
timeouts, or report completed-only certification fractions as the full fixed
study result. Dense lowering may be substantially less memory-efficient than
native convolution; this prototype establishes a feasibility boundary, not a
scalability claim for production traffic-sign recognition.
