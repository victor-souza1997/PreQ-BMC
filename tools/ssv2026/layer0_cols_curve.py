#!/usr/bin/env python3
"""Problem-size curve for monolithic ESBMC: the first C output columns of layer-0 row 0 (validation 3652 only).

qnn_compact.c's layer-0 nest is copied verbatim (prefix_curve.prefix_source) except the output bounds,
oy < 16 and ox < 16, which become oy < 1 and ox < C; each column is 24 outputs. Two input modes per C:
"centre" fixes every byte (one concrete execution); "receptive" frees every byte feeding those outputs,
input rows 0..3 and columns 0..min(31, 2C+2), to the epsilon box (default 1); --free k frees only the
first k of those bytes in a fixed permutation (seed 0). The property is that every computed output lies in
its interval box (src/verification/conv_contracts.propagate_network, untrusted, checked by ESBMC).
A slice is not the deployed function: a VERIFIED covers only those rows. MEASURED per point: status,
time, peak process-tree RSS, last ESBMC phase. One ESBMC process at a time.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantization_candidates as qc  # noqa: E402
from prefix_curve import harness, prefix_source  # noqa: E402
from scalability_curve import SAMPLE, run  # noqa: E402

import numpy as np  # noqa: E402

from datasets.gtsrb_study import load_crop  # noqa: E402
from verification.conv_contracts import IntegerInvariant, propagate_network  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--columns", type=int, nargs="+", required=True)
    p.add_argument("--mode", choices=("centre", "receptive"), required=True)
    p.add_argument("--memlimit", default="5g")
    p.add_argument("--epsilon", type=int, default=1)
    p.add_argument("--free", type=int, help="receptive mode: free only this many receptive bytes")
    p.add_argument("--profile", choices=("paper-z3", "paper-fast"), default="paper-z3")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--min-available-gib", type=float, default=6)
    p.add_argument("--stop-after-failure", action="store_true",
                   help="stop increasing C at the first point that is neither VERIFIED nor FAILED")
    args = p.parse_args()
    folder, manifest, network, encoder, study, rows = qc.load_candidate(args.artifact)
    row = next(r for r in rows if r["id"] == SAMPLE)
    image = load_crop(study["dataset_root"], row)
    centre = np.asarray(encoder.resize(image), dtype=np.int64).reshape(-1)
    byte_low, byte_high = (np.asarray(a, dtype=np.int64).reshape(-1) for a in encoder.byte_box(image, args.epsilon))
    height, width, channels = network.input_shape
    out_h, out_w, out_c = network.layers[0].geometry.output_shape
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    compact = (folder / "qnn_compact.c").read_text()
    for r in args.columns:
        size = r * out_c
        sizes = [size] + [layer.output_size for layer in network.layers[1:]]
        text = prefix_source(compact, 0, sizes)
        for bound, new in ((f"for (int oy = 0; oy < {out_h}; ++oy)", "for (int oy = 0; oy < 1; ++oy)"),
                           (f"for (int ox = 0; ox < {out_w}; ++ox)", f"for (int ox = 0; ox < {r}; ++ox)")):
            if text.count(bound) != 1:
                raise ValueError(f"Layer-0 loop {bound!r} not found exactly once")
            text = text.replace(bound, new)
        source = out / f"layer0_cols{r:02d}.c"
        source.write_text(text)
        library = out / f"layer0_cols{r:02d}.so"
        subprocess.run(["gcc", "-std=c11", "-O1", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
        lib = ctypes.CDLL(str(library))

        def encode(values):
            encoded = (ctypes.c_int64 * values.size)()
            raw = np.ascontiguousarray(values, dtype=np.uint8)
            if lib.qnn_encode_bytes(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), height, width, encoded):
                raise RuntimeError("encoder rejected a byte vector")
            return tuple(encoded)

        # Host check: the slice equals the interval box at the centre, where the box is a point.
        encoded = encode(centre)
        host = (ctypes.c_int64 * size)()
        lib.qnn_forward_prefix((ctypes.c_int64 * centre.size)(*encoded), host)
        library.unlink()
        point = propagate_network(network.layers[:1], IntegerInvariant(
            encoded, encoded, tuple(network.input_shape)))[-1].output_invariant
        if list(host) != list(point.low[:size]) or list(host) != list(point.high[:size]):
            raise RuntimeError(f"C = {r}: slice disagrees with the interval transformer at the centre")
        low, high = centre.copy(), centre.copy()
        free = 0
        if args.mode == "receptive":
            mask = np.zeros((height, width, channels), dtype=bool)
            mask[:4, :min(width, 2 * r + 3)] = True
            mask = mask.reshape(-1)
            if args.free is not None:
                chosen = np.random.default_rng(0).permutation(np.flatnonzero(mask))[:args.free]
                mask = np.zeros_like(mask)
                mask[chosen] = True
            free = int(mask.sum())
            low[mask], high[mask] = byte_low[mask], byte_high[mask]
        # Encoder is monotone per byte, so enc(low) and enc(high) bound every encoded input.
        box = propagate_network(network.layers[:1], IntegerInvariant(
            encode(low), encode(high), tuple(network.input_shape)))[-1].output_invariant
        path = out / f"harness_cols{r:02d}_{args.mode}_eps{args.epsilon}_free{free}_{args.profile}.c"
        path.write_text(harness(source, size, low, high, box.low[:size], box.high[:size]))
        result = run(path, args.memlimit, args.timeout, True, args.min_available_gib, args.profile)
        record = {"artifact": str(folder), "bits": manifest["bits"], "fractional_bits": manifest["fractional_bits"],
                  "sample_id": SAMPLE, "split": "validation", "layer": 0, "output_rows": 1, "output_columns": r, "outputs": size,
                  "mode": args.mode, "profile": args.profile, "epsilon": args.epsilon, "free_bytes": free,
                  "box_width_sum": int(np.sum(np.array(box.high[:size]) - np.array(box.low[:size]))),
                  "interval_analysis": True, "memlimit": args.memlimit, "timeout": args.timeout, **result}
        with (out / "layer0_cols_curve.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps({x: record[x] for x in ("bits", "output_columns", "mode", "profile", "epsilon", "free_bytes", "status",
                                                "elapsed_seconds", "peak_process_tree_rss_bytes", "last_phase")}),
              flush=True)
        if args.stop_after_failure and result["status"] not in ("VERIFIED", "FAILED"):
            break


if __name__ == "__main__":
    main()
