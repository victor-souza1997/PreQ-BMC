# SSV feasibility gate

This is a new, opt-in cropped traffic-sign study, not a change to the LADC
experiments. GTSRB is not a registration-plate or full-scene detection dataset.

## Repository inspection before implementation

The dense assumptions are in `models/deep_model.py` (`tf.matmul`, hidden ReLU),
`models/loading.py` (rank-two kernels and architecture names),
`datasets/loaders.py` (flattening and float32 conversion),
`symbolic_pp/DeepPoly_preqbmc.py` (`load_dnn`, scalar affine neurons),
`synthesis/preqbmc.py` (`GPEncoding`, dense MILP preimages, shared-format search),
`synthesis/pipeline.py` (dataset/model loading and normalization), and
`backends/fixed_point.py` / `c_qnn_generator.py` (two-dimensional weight arrays).
`verification/c_templates.py` already handles general affine matrices and
neuron blocks; its integer arithmetic kernel is shared with deployment.
Existing report parameter counts describe dense storage, not convolutional
parameter sharing. Existing CNN-shaped ONNX files are not evidence of support.

The audit in `docs/ladc2026_review/methodology_audit_response.md`, section 3.5,
reproduces loss of input coverage through round-to-nearest float32 endpoints.
This study must not inherit an unrestricted transfer claim from that pipeline.

## Implementation stages and stop conditions

1. Restrict the prototype to one NHWC Conv2D, HWIO weights, VALID or SAME
   zero padding, positive integer strides, ReLU, NHWC Flatten, Dense logits.
   Reject pooling, groups, dilation, batch normalization and implicit layout
   conversion. Lower convolution to an affine matrix with explicit replicated
   biases. Cap the expanded matrix before allocation. Reuse, do not rewrite,
   the MILP preimage and shared-QIF verification pipeline.
2. Independently compare direct convolution with the lowering, including
   exact integer layer values and host C. Float equivalence is numerical;
   reassociated float32 sums are not claimed bit-identical.
3. Run actual ESBMC true/false tests on an enumerable model, plus the derived
   preimage path with beta=0,1,2. Stop the research campaign if the required
   proof, coverage, or parity checks fail. Preserve the negative result.
4. Prepare a track-disjoint GTSRB manifest, fixed sample-selection protocol,
   and tiny-model training/experiment specifications. Do not invent trained
   weights, sample IDs, accuracy, dataset license, or GTSRB certificates.
5. Generate an NDK/JNI integration target using the exact generated C.
   An absent NDK/device blocks Android validation, not host development.
   Deployment parity and externally measured energy remain NOT_MEASURED
   until a physical device and meter are supplied.

## Input bridge

Initial deployment domain is decoded RGB uint8 crops, not arbitrary JPEG
decoders or real sensor scenes. Resize uses top-left nearest-neighbor indices
`floor(out_index * input_size / output_size)`. Normalize by **256**, not 255;
this deliberate dyadic convention makes float32 source inputs exact.
Integer encoding uses half-away rounding of `pixel * 2**F / 256` and the
declared signed clamp. The integer box is computed from exact rational raw
pixel endpoints, clipped to [0,255], with ceil/floor for the byte domain.
Nearest-neighbor selection and the scalar quantizer are monotone, establishing
`E(B_inf(x,epsilon) intersect uint8-domain) subseteq A0` without float32
endpoint conversion. Certificates do not cover unmodeled image decoding,
interpolation, physical attacks, or a whole vehicle.

## Resources and deliverable boundary

Initial inspection found TensorFlow, CBC and Pillow in the `quad` environment,
ESBMC in `/home/joao/esbmc-linux/bin`, and host C compilers. No Android NDK
environment is configured. A dense lowering is a correctness prototype, not
an efficient native convolution implementation: report expanded storage and
MACs separately from the original shared parameter count. Stop at the small
model if expansion or solver budgets are exceeded. Do not claim general CNN,
ONNX, Android performance, or completed GTSRB study from this feasibility gate.
