#!/usr/bin/env python3
"""Layers-included curve for monolithic ESBMC on verbatim prefixes of the compact C (validation 3652 only).

Point (d, k): the first d+1 layer loop nests of qnn_forward_fixed, copied verbatim from qnn_compact.c, run
on k free raw bytes (epsilon-1 box of validation image 3652, fixed permutation, seed 0; other bytes at
their centre). The property is that every layer-d output lies in its integer interval box, computed by
src/verification/conv_contracts.propagate_network over the same k-byte box. That box is an untrusted
proposal: ESBMC checks it. A FAILED means the box or the slice is wrong, not that the network is.

Before ESBMC, each prefix is compiled with gcc and run on the centre bytes; its outputs must equal the
interval box at k = 0, where the box is a point. The box is seeded with the compiled encoder applied to the
low and high byte vectors; the encoder (b * 2^f / 256, half-away rounding, clamp) is monotone per byte.
MEASURED per point: status, time, peak process-tree RSS, last ESBMC phase. One ESBMC process at a time.
"""
import argparse
import ctypes
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantization_candidates as qc  # noqa: E402
from scalability_curve import SAMPLE, run  # noqa: E402

import numpy as np  # noqa: E402

from datasets.gtsrb_study import load_crop  # noqa: E402
from verification.conv_contracts import IntegerInvariant, propagate_network  # noqa: E402


def prefix_source(compact, depth, sizes):
    """qnn_compact.c up to qnn_forward_fixed, then a function running layer nests 0..depth verbatim."""
    head, _, body = compact.partition("void qnn_forward_fixed(const int64_t *input, int64_t *output) {\n")
    if not body:
        raise ValueError("qnn_forward_fixed not found")
    # Top-level statements of the body are separated by blank lines: buffers + input copy, then one nest per layer.
    blocks = [b for b in re.split(r"\n\s*\n", body.rsplit("}", 1)[0]) if b.strip()]
    setup, nests = blocks[0], blocks[1:-1]
    if len(nests) != len(sizes) or not blocks[-1].lstrip().startswith("for (int i = 0; i <"):
        raise ValueError("Unexpected qnn_forward_fixed layout")
    target = re.findall(r"(buffer_[01])\[[^\]]*\] = \(int\d+_t\)value;", nests[depth])
    if len(set(target)) != 1:
        raise ValueError("Cannot identify the output buffer of the last nest")
    kept = "\n\n".join(nests[:depth + 1])
    return (f"{head}void qnn_forward_prefix(const int64_t *input, int64_t *output) {{\n{setup}\n\n{kept}\n\n"
            f"    for (int i = 0; i < {sizes[depth]}; ++i) output[i] = {target[0]}[i];\n}}\n")


def harness(prefix_path, depth_size, low, high, box_low, box_high):
    arr = lambda values: "{" + ",".join(str(int(v)) for v in values) + "}"  # noqa: E731
    count = len(low)
    return f'''#include <stdint.h>
void __ESBMC_assert(_Bool, const char *);
void __ESBMC_assume(_Bool);
uint8_t nondet_uchar(void);
#define QNN_ASSERT(c,m) __ESBMC_assert((c),(m))
#define INPUT_SIZE {count}
#define LAYER_SIZE {max(count, depth_size)}
#include "{prefix_path}"
static const uint8_t low[{count}] = {arr(low)};
static const uint8_t high[{count}] = {arr(high)};
static const int64_t out_low[{depth_size}] = {arr(box_low)};
static const int64_t out_high[{depth_size}] = {arr(box_high)};
int main(void) {{
  uint8_t raw[{count}];
  int64_t input[{count}], output[{depth_size}];
  for (int i = 0; i < {count}; ++i) {{
    raw[i] = nondet_uchar();
    __ESBMC_assume(raw[i] >= low[i] && raw[i] <= high[i]);
  }}
  __ESBMC_assert(qnn_encode_bytes(raw, 32, 32, input) == 0, "encoder accepts resized crop");
  qnn_forward_prefix(input, output);
  for (int j = 0; j < {depth_size}; ++j)
    __ESBMC_assert(output[j] >= out_low[j] && output[j] <= out_high[j], "layer output inside its interval box");
  return 0;
}}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--depths", type=int, nargs="+", required=True)
    p.add_argument("--free-bytes", type=int, nargs="+", required=True)
    p.add_argument("--memlimit", default="5g")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--no-interval", action="store_true")
    p.add_argument("--min-available-gib", type=float, default=6)
    p.add_argument("--stop-after-failure", action="store_true",
                   help="per depth, stop increasing k at the first point that is neither VERIFIED nor FAILED")
    args = p.parse_args()
    folder, manifest, network, encoder, study, rows = qc.load_candidate(args.artifact)
    row = next(r for r in rows if r["id"] == SAMPLE)
    image = load_crop(study["dataset_root"], row)
    centre = np.asarray(encoder.resize(image), dtype=np.int64).reshape(-1)
    byte_low, byte_high = (np.asarray(a, dtype=np.int64).reshape(-1) for a in encoder.byte_box(image, 1))
    order = np.random.default_rng(0).permutation(centre.size)
    sizes = [layer.output_size for layer in network.layers]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    compact = (folder / "qnn_compact.c").read_text()
    for depth in args.depths:
        source = out / f"prefix_d{depth}.c"
        source.write_text(prefix_source(compact, depth, sizes))
        # Host check at k = 0: the prefix equals the interval box, which is a point there.
        library = out / f"prefix_d{depth}.so"
        subprocess.run(["gcc", "-std=c11", "-O1", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
        lib = ctypes.CDLL(str(library))

        def encode(values):
            encoded = (ctypes.c_int64 * values.size)()
            raw = np.ascontiguousarray(values, dtype=np.uint8)
            if lib.qnn_encode_bytes(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), 32, 32, encoded):
                raise RuntimeError("encoder rejected a byte vector")
            return tuple(encoded)

        encoded = encode(centre)
        host = (ctypes.c_int64 * sizes[depth])()
        lib.qnn_forward_prefix((ctypes.c_int64 * centre.size)(*encoded), host)
        point = propagate_network(network.layers[:depth + 1], IntegerInvariant(
            encoded, encoded, tuple(network.input_shape)))[-1].output_invariant
        if list(host) != list(point.low) or list(host) != list(point.high):
            raise RuntimeError(f"depth {depth}: prefix C disagrees with the interval transformer at k = 0")
        for k in args.free_bytes:
            low, high = centre.copy(), centre.copy()
            low[order[:k]], high[order[:k]] = byte_low[order[:k]], byte_high[order[:k]]
            encoded_box = IntegerInvariant(encode(low), encode(high), tuple(network.input_shape))
            box = propagate_network(network.layers[:depth + 1], encoded_box)[-1].output_invariant
            tag = f"d{depth}_k{k:04d}_{'nointerval' if args.no_interval else 'interval'}"
            path = out / f"harness_{tag}.c"
            path.write_text(harness(source, sizes[depth], low, high, box.low, box.high))
            result = run(path, args.memlimit, args.timeout, not args.no_interval, args.min_available_gib)
            record = {"artifact": str(folder), "bits": manifest["bits"],
                      "fractional_bits": manifest["fractional_bits"], "sample_id": SAMPLE, "split": "validation",
                      "depth": depth, "layers_included": depth + 1, "output_size": sizes[depth], "free_bytes": k,
                      "box_width_sum": int(np.sum(np.array(box.high) - np.array(box.low))),
                      "interval_analysis": not args.no_interval, "memlimit": args.memlimit,
                      "timeout": args.timeout, **result}
            with (out / "prefix_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps({x: record[x] for x in ("bits", "depth", "free_bytes", "status", "elapsed_seconds",
                                                    "peak_process_tree_rss_bytes", "last_phase")}), flush=True)
            if args.stop_after_failure and result["status"] not in ("VERIFIED", "FAILED"):
                break
        library.unlink()


if __name__ == "__main__":
    main()
