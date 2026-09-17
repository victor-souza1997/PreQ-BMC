"""Explicit denominators and measured power, never imputed from CPU load."""
import numpy as np


def integrate_power(times, watts, start, end, inference_count, *, idle_watts=None):
    times, watts = np.asarray(times, dtype=float), np.asarray(watts, dtype=float)
    if (times.ndim != 1 or watts.shape != times.shape or len(times) < 2
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(watts))
            or np.any(np.diff(times) <= 0) or np.any(watts < 0)
            or not np.isfinite(start) or not np.isfinite(end) or not times[0] <= start < end <= times[-1]
            or type(inference_count) is not int or inference_count <= 0):
        raise ValueError("Need a synchronized, covering, finite power trace and completed inference count")
    if idle_watts is not None and (not np.isfinite(idle_watts) or idle_watts < 0):
        raise ValueError("Idle power must be measured and nonnegative")
    selected = (times > start) & (times < end)
    t = np.r_[start, times[selected], end]
    p = np.interp(t, times, watts)
    energy = float(np.sum(np.diff(t) * (p[:-1] + p[1:]) / 2))
    return {"status": "MEASURED", "duration_seconds": end - start,
            "average_power_watts": energy / (end - start), "gross_energy_joules": energy,
            "gross_joules_per_inference": energy / inference_count,
            "idle_watts": idle_watts,
            "idle_subtracted_joules_per_inference": None if idle_watts is None else
                (energy - idle_watts * (end - start)) / inference_count}


def missing_device_report():
    return {"status": "NOT_MEASURED", "device": None, "android_version": None,
            "ndk_version": None, "compiler_version": None, "artifact_sha256": None,
            "test_accuracy": None, "latency_median_ms": None, "latency_p95_ms": None,
            "throughput_per_second": None, "cpu_utilization_percent": None,
            "peak_process_rss_bytes": None, "binary_size_bytes": None,
            "layer_logit_parity": "NOT_MEASURED", "trials": [],
            "power": {"status": "NOT_MEASURED", "instrument": None,
                      "sampling_hz": None, "calibration": None, "uncertainty": None,
                      "gross_joules_per_inference": None, "idle_subtracted_joules_per_inference": None}}
