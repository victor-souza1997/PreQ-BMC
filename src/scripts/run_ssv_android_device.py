"""Run the generated integer C benchmark on one connected Android device."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np

from datasets.gtsrb_study import sha256
from scripts.run_ssv_cnn_gate import write_new_json


def _adb(serial, *arguments, check=True):
    command = ["adb"]
    if serial:
        command += ["-s", serial]
    command += [str(value) for value in arguments]
    return subprocess.run(command, check=check, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True)


def _property(serial, name):
    return _adb(serial, "shell", "getprop", name).stdout.strip()


def _quantile(values, q):
    return float(np.quantile(np.asarray(values, dtype=float), q))


def run(bundle_path: Path, binary: Path, output: Path, *, serial=None, core_mask=None,
        trials=10, warmup=10000, latency_samples=10000, throughput_iterations=100000):
    bundle_path, binary, output = Path(bundle_path), Path(binary), Path(output)
    manifest_path = bundle_path / "bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source, vectors = bundle_path / manifest["qnn_source"], bundle_path / manifest["vectors"]
    if (manifest.get("schema") != "preqbmc_android_replay_bundle_v1"
            or sha256(source) != manifest.get("qnn_source_sha256")
            or sha256(vectors) != manifest.get("vectors_sha256")):
        raise ValueError("Android replay bundle identity mismatch")
    controls = (trials, warmup, latency_samples, throughput_iterations)
    if (not binary.is_file() or any(type(value) is not int or value <= 0 for value in controls)
            or trials < 3):
        raise ValueError("Need an Android benchmark binary and at least three independent trials")
    if core_mask is not None and not re.fullmatch(r"(?:0x)?[0-9a-fA-F]+", core_mask):
        raise ValueError("Core mask must be a hexadecimal taskset mask")
    if shutil.which("adb") is None:
        raise FileNotFoundError("adb is required")
    output.mkdir(parents=True, exist_ok=False)
    selected_serial = serial or _adb(None, "get-serialno").stdout.strip()
    if not selected_serial or selected_serial in {"unknown", "no permissions"}:
        raise RuntimeError("No unambiguous Android device is available")
    token = manifest["vectors_sha256"][:12]
    remote_dir = f"/data/local/tmp/preqbmc_ssv_{token}"
    remote_binary, remote_vectors = f"{remote_dir}/benchmark", f"{remote_dir}/vectors.bin"
    _adb(selected_serial, "shell", "mkdir", "-p", remote_dir)
    _adb(selected_serial, "push", str(binary), remote_binary)
    _adb(selected_serial, "push", str(vectors), remote_vectors)
    _adb(selected_serial, "shell", "chmod", "700", remote_binary)
    features = _adb(selected_serial, "shell", "pm", "list", "features").stdout.splitlines()
    automotive = any(line.strip() == "feature:android.hardware.type.automotive" for line in features)
    device = {
        "serial": selected_serial,
        "build_fingerprint": _property(selected_serial, "ro.build.fingerprint"),
        "android_release": _property(selected_serial, "ro.build.version.release"),
        "sdk": _property(selected_serial, "ro.build.version.sdk"),
        "product": _property(selected_serial, "ro.product.device"),
        "hardware": _property(selected_serial, "ro.hardware"),
        "abi": _property(selected_serial, "ro.product.cpu.abi"),
        "kernel": _adb(selected_serial, "shell", "uname", "-a").stdout.strip(),
        "online_cpus": _adb(selected_serial, "shell", "cat", "/sys/devices/system/cpu/online").stdout.strip(),
        "cpu_scaling": _adb(selected_serial, "shell", "sh", "-c",
                            "for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor "
                            "/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do "
                            "[ -r \"$f\" ] && printf '%s=' \"$f\" && cat \"$f\"; done").stdout,
        "automotive_feature_declared": automotive,
        "platform_classification": "AAOS" if automotive else "ANDROID_NON_AUTOMOTIVE",
        "cpuinfo": _adb(selected_serial, "shell", "cat", "/proc/cpuinfo").stdout,
    }
    reports = []
    for index in range(trials):
        command = ["shell"]
        if core_mask:
            command += ["taskset", core_mask]
        command += [remote_binary, remote_vectors, warmup, latency_samples, throughput_iterations]
        started = time.time()
        completed = _adb(selected_serial, *command)
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid benchmark output in trial {index}: {completed.stdout}") from exc
        if (payload.get("schema") != "preqbmc_android_benchmark_v1"
                or payload.get("parity_status") != "EXACT_MATCH"
                or payload.get("vectors") != manifest["n_test_images"]):
            raise RuntimeError(f"Device parity failed in trial {index}: {payload}")
        reports.append({**payload, "trial_index": index, "host_started_unix": started,
                        "host_finished_unix": time.time()})
    latency = [row["latency_median_ns"] for row in reports]
    throughput = [row["throughput_per_second"] for row in reports]
    build_keys = ("compiler_id", "compiler_version", "ndk_version", "compiled_abi", "android_platform")
    if any(tuple(row.get(key) for key in build_keys) != tuple(reports[0].get(key) for key in build_keys)
           for row in reports[1:]):
        raise RuntimeError("Benchmark build metadata changed between trials")
    summary = {
        "schema": "preqbmc_android_device_report_v1",
        "measurement_status": "MEASURED",
        "formal_status_modified": False,
        "formal_artifact_status": manifest.get("formal_artifact_status"),
        "bound_region_certificates": manifest.get("bound_region_certificates", []),
        "device": device,
        "bundle_manifest_sha256": sha256(manifest_path),
        "qnn_source_sha256": manifest["qnn_source_sha256"],
        "vectors_sha256": manifest["vectors_sha256"],
        "benchmark_binary_sha256": sha256(binary),
        "build_metadata": {key: reports[0].get(key) for key in build_keys},
        "core_mask": core_mask,
        "native_c_parity": "EXACT_MATCH",
        "jni_parity": "NOT_MEASURED",
        "apk_integration": "NOT_MEASURED",
        "nnapi_or_npu": "NOT_MEASURED",
        "power": "NOT_MEASURED",
        "scope": "native qnn_forward_fixed only; excludes decode, resize, JNI, application scheduling and UI",
        "n_independent_process_trials": trials,
        "latency_median_of_trial_medians_ns": float(np.median(latency)),
        "latency_trial_medians_q1_ns": _quantile(latency, .25),
        "latency_trial_medians_q3_ns": _quantile(latency, .75),
        "throughput_median_per_second": float(np.median(throughput)),
        "peak_process_rss_kib_max": max(row["peak_process_rss_kib"] for row in reports),
        "cpu_time_over_wall_percent_median": float(np.median(
            [row["cpu_time_over_wall_percent"] for row in reports])),
        "trials": reports,
        "interpretation": ("AAOS native measurement" if automotive else
                           "Android measurement only; do not label this device image AAOS"),
    }
    write_new_json(output / "android_device_report.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True, help="arm64 benchmark built with the bundle qnn.c")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial")
    parser.add_argument("--core-mask", help="Hexadecimal taskset mask; identify cores before choosing it")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10000)
    parser.add_argument("--latency-samples", type=int, default=10000)
    parser.add_argument("--throughput-iterations", type=int, default=100000)
    args = parser.parse_args()
    result = run(args.bundle, args.binary, args.output, serial=args.serial,
                 core_mask=args.core_mask, trials=args.trials, warmup=args.warmup,
                 latency_samples=args.latency_samples,
                 throughput_iterations=args.throughput_iterations)
    print(args.output / "android_device_report.json")
    raise SystemExit(0 if result["native_c_parity"] == "EXACT_MATCH" else 1)


if __name__ == "__main__":
    main()
