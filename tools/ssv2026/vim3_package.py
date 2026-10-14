#!/usr/bin/env python3
"""Build the VIM3 bundle: validation crops, host expected logits, and static aarch64 benchmarks.

Expected logits come from qnn_reference.so, whose source is the pinned certified deployment (the
candidate build refuses otherwise). Benchmarks: int16 compact storage, int64 certified storage and the
float32 baseline (float_baseline.py, not verified; built with -ffp-contract=off), all cross-compiled with aarch64-linux-gnu-gcc -static, so they also run from an Android (AAOS) adb shell.
Each is first built and run natively on the host against the same files. Validation split only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quantization_candidates as qc  # noqa: E402
from float_baseline import build_float  # noqa: E402

from datasets.gtsrb_study import load_crop  # noqa: E402

HERE = Path(__file__).resolve().parent
FLAGS = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(out, name, rows, study, reference, network):
    images, logits, labels = bytearray(b"QNNI" + struct.pack("<I", len(rows))), [], []
    for row in rows:
        crop = load_crop(study["dataset_root"], row)
        images += struct.pack("<HH", crop.shape[0], crop.shape[1]) + crop.tobytes()
        logits.append(reference.forward(reference.encode(crop)))
        labels.append(int(row["class_id"]))
    (out / f"images_{name}.bin").write_bytes(bytes(images))
    expected = bytearray(b"QNNE" + struct.pack("<II", len(rows), network.output_size))
    for row_logits in logits:
        expected += struct.pack(f"<{network.output_size}q", *map(int, row_logits))
    expected += struct.pack(f"<{len(labels)}i", *labels)
    (out / f"expected_{name}.bin").write_bytes(bytes(expected))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    folder, manifest, network, encoder, study, rows = qc.load_candidate(args.artifact)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reference = qc.CNetwork(folder / "qnn_reference.so", network)
    pack(out, "all", rows, study, reference, network)
    pack(out, "one", rows[:1], study, reference, network)
    float_check = build_float(folder, out)
    shutil.copy(HERE / "vim3_bench.c", out / "vim3_bench.c")
    shutil.copy(HERE / "run_on_vim3.sh", out / "run_on_vim3.sh")
    shutil.copy(HERE / "VIM3_README.md", out / "README.md")
    shutil.copy(folder / "qnn_compact.c", out / "int16_compact.c")
    shutil.copy(folder / "qnn_reference.c", out / "int64_certified.c")
    binaries, host_runs = {}, {}
    variants = (("int16_compact", "expected", []), ("int64_certified", "expected", []),
                ("float32_baseline", "expected_float", ["-DFLOAT_MODEL", "-ffp-contract=off"]))
    for variant, expected, variant_flags in variants:
        define = f'-DQNN_SOURCE="{variant}.c"'
        for target, compiler, extra in (("host", "gcc", []), ("aarch64", "aarch64-linux-gnu-gcc", ["-static"])):
            binary = out / f"bench_{variant}_{target}"
            command = [compiler, *FLAGS, *variant_flags, *extra, define, "vim3_bench.c", "-o", binary.name]
            subprocess.run(command, cwd=out, check=True)
            binaries[binary.name] = {"sha256": digest(binary), "command": command,
                                     "size": subprocess.run(["size", binary.name] if target == "host" else
                                                            ["aarch64-linux-gnu-size", binary.name], cwd=out,
                                                            check=True, capture_output=True, text=True).stdout}
        run = subprocess.run([f"./bench_{variant}_host", "images_all.bin", f"{expected}_all.bin", "1", variant],
                             cwd=out, capture_output=True, text=True)
        host_runs[variant] = {"return_code": run.returncode, "result": json.loads(run.stdout)}
        print(variant, run.stdout.strip(), flush=True)
    (out / "bundle_manifest.json").write_text(json.dumps({
        "schema": "ssv_vim3_bundle_v1", "split": "validation", "images": len(rows),
        "artifact": str(folder), "manifest_sha256": digest(folder / "manifest.json"),
        "expected_logits_from": "qnn_reference.so (pinned certified deployment source)",
        "files": {f.name: digest(f) for f in sorted(out.iterdir()) if f.is_file()},
        "binaries": binaries, "host_runs": host_runs, "float_baseline_check": float_check,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
