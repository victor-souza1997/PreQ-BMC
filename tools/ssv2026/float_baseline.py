#!/usr/bin/env python3
"""Float32 C baseline (the unquantized network) for the VIM3 resource comparison. Not verified.

Same frozen folded model, same layers and the same dense storage of the depthwise layer as the QNN
export; same centre-aligned nearest resize, input byte / 256. Writes float32_baseline.c and
expected_float_{all,one}.bin into an existing VIM3 bundle, after checking the C predictions against a
float64 NumPy forward on every validation image. Validation split only.
"""
import argparse
import ctypes
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantization_candidates as qc  # noqa: E402

import numpy as np  # noqa: E402

from datasets.gtsrb_study import load_crop  # noqa: E402


def array(name, values):
    body = ",".join(f"{float(v)!r}f" for v in np.asarray(values, dtype=np.float32).flat)
    return f"static const float {name}[{np.asarray(values).size}] = {{{body}}};\n"


def render(model):
    h, w, c = model.input_shape
    sizes = [h * w * c] + [int(np.prod(g.output_shape)) for g in model.geometries]
    parts = ["#include <stdint.h>\n\n"]
    for i, (kernel, bias) in enumerate(zip(model.conv_kernels, model.conv_biases)):
        parts += [array(f"CONV_{i}_KERNEL", kernel), array(f"CONV_{i}_BIAS", bias)]
    parts += [array("DENSE_KERNEL", model.dense_kernels[0]), array("DENSE_BIAS", model.dense_biases[0])]
    size = max(sizes)
    parts.append(f"static float buffer_0[{size}], buffer_1[{size}];\n\n")
    parts.append(f"int fnn_input_dim(void) {{ return {sizes[0]}; }}\n"
                 f"int fnn_output_dim(void) {{ return {len(model.dense_biases[0])}; }}\n\n")
    parts.append(f"""int fnn_forward_bytes(const uint8_t *image, int height, int width, float *out) {{
    if (height <= 0 || width <= 0) return 1;
    for (int y = 0; y < {h}; ++y)
        for (int x = 0; x < {w}; ++x)
            for (int ch = 0; ch < {c}; ++ch) {{
                int sy = (2 * y + 1) * height / {2 * h}, sx = (2 * x + 1) * width / {2 * w};
                buffer_0[(y * {w} + x) * {c} + ch] = image[((long)sy * width + sx) * {c} + ch] / 256.0f;
            }}
""")
    src, dst = "buffer_0", "buffer_1"
    for i, g in enumerate(model.geometries):
        ih, iw, ci = g.input_shape
        kh, kw, _, co = g.kernel_shape
        oh, ow, _ = g.output_shape
        ph, pw = g.pad_before
        sh, sw = g.strides
        parts.append(f"""    for (int oy = 0; oy < {oh}; ++oy)
        for (int ox = 0; ox < {ow}; ++ox)
            for (int oc = 0; oc < {co}; ++oc) {{
                float acc = CONV_{i}_BIAS[oc];
                for (int ky = 0; ky < {kh}; ++ky)
                    for (int kx = 0; kx < {kw}; ++kx) {{
                        int iy = oy * {sh} + ky - {ph}, ix = ox * {sw} + kx - {pw};
                        if (iy < 0 || iy >= {ih} || ix < 0 || ix >= {iw}) continue;
                        for (int ic = 0; ic < {ci}; ++ic)
                            acc += {src}[(iy * {iw} + ix) * {ci} + ic] * CONV_{i}_KERNEL[((ky * {kw} + kx) * {ci} + ic) * {co} + oc];
                    }}
                {dst}[(oy * {ow} + ox) * {co} + oc] = acc > 0.0f ? acc : 0.0f;
            }}
""")
        src, dst = dst, src
    n_in, n_out = model.dense_kernels[0].shape
    parts.append(f"""    for (int k = 0; k < {n_out}; ++k) {{
        float acc = DENSE_BIAS[k];
        for (int i = 0; i < {n_in}; ++i) acc += {src}[i] * DENSE_KERNEL[i * {n_out} + k];
        out[k] = acc;
    }}
    return 0;
}}
""")
    return "".join(parts)


def build_float(artifact, bundle):
    folder, manifest, network, encoder, study, rows = qc.load_candidate(artifact)
    config, model, study, rows, provenance = qc.load_source(manifest["provenance"]["config_path"])
    bundle = Path(bundle).resolve()
    source = bundle / "float32_baseline.c"
    source.write_text(render(model))
    library = bundle / "float32_baseline_check.so"
    subprocess.run(["gcc", "-std=c11", "-O2", "-ffp-contract=off", "-shared", "-fPIC", str(source), "-o", str(library)],
                   check=True)
    lib = ctypes.CDLL(str(library))
    lib.fnn_forward_bytes.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_float)]
    affine = [(np.asarray(wt, np.float64), np.asarray(b, np.float64)) for wt, b in model.affine_parameters()]
    predictions, numpy_predictions, labels, max_gap = [], [], [], 0.0
    for row in rows:
        crop = np.ascontiguousarray(load_crop(study["dataset_root"], row), dtype=np.uint8)
        out = (ctypes.c_float * len(model.dense_biases[0]))()
        if lib.fnn_forward_bytes(crop.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), crop.shape[0], crop.shape[1], out):
            raise RuntimeError("float encoder rejected a crop")
        value = encoder.resize(crop).reshape(-1).astype(np.float64) / 256
        for index, (weights, bias) in enumerate(affine):
            value = value @ weights + bias
            if index + 1 < len(affine):
                value = np.maximum(value, 0)
        logits = np.array(out[:], dtype=np.float64)
        max_gap = max(max_gap, float(np.max(np.abs(logits - value))))
        predictions.append(int(np.argmax(logits)))
        numpy_predictions.append(int(np.argmax(value)))
        labels.append(int(row["class_id"]))
    library.unlink()
    predictions, numpy_predictions, labels = map(np.array, (predictions, numpy_predictions, labels))
    for name, count in (("all", len(rows)), ("one", 1)):
        blob = b"QNNF" + struct.pack("<I", count) + struct.pack(f"<{count}i", *predictions[:count]) + struct.pack(
            f"<{count}i", *labels[:count])
        (bundle / f"expected_float_{name}.bin").write_bytes(blob)
    report = {"schema": "ssv_float_baseline_check_v1", "claim": "MEASURED", "split": "validation",
              "images": len(rows), "c_accuracy": float(np.mean(predictions == labels)),
              "numpy_float64_accuracy": float(np.mean(numpy_predictions == labels)),
              "c_vs_numpy_prediction_mismatches": int(np.sum(predictions != numpy_predictions)),
              "max_abs_logit_gap_c_float32_vs_numpy_float64": max_gap,
              "folded_model_sha256": provenance["folded_model_sha256"]}
    (bundle / "float_baseline_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    args = p.parse_args()
    build_float(args.artifact, args.bundle)


if __name__ == "__main__":
    main()
