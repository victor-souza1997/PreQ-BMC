"""Load the ESBMC box ledger and turn it into one-sided facts on RAW pre-activations.

UNTRUSTED LOADER. It reads what each interval harness actually asserts; the
facts it derives are the only box facts a chain may cite.

What a layer-k interval harness asserts (see harnesses/layer_k_block_b_*.c):
  EXPECTED_LOW/HIGH = the post-clamp, post-ReLU interval of h_k (no ReLU on the
  logit layer), and, where STABLE_RELU = +1 / -1, that the post-clamp pre-ReLU
  interval is >= 0 / <= 0.
One-sided facts on the raw value r (z = clamp_q16(r), h = relu(z)):
  upper  r <= hh             when hh < 32767 (clamp(r) <= hh < 32767 forces r <= hh)
  lower  r >= hl             when hl > -32768 and (STABLE_RELU = +1 or hl > 0 or logit layer)
Nothing else. In particular the ledger does NOT prove the negative pre-ReLU lower
endpoint of an unstable or stable-inactive neuron.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

Q_LOW, Q_HIGH = -(1 << 15), (1 << 15) - 1
ARRAY = re.compile(r"static const (?:int64_t|int) (\w+)\[[^\]]*\] = \{([^}]*)\};")


def parse_harness(path):
    arrays = {m.group(1): np.array([int(v) for v in m.group(2).split(",")], dtype=np.int64)
              for m in ARRAY.finditer(Path(path).read_text())}
    return arrays


def load_ledger(summary_path, n_layers):
    summary = json.loads(Path(summary_path).read_text())
    cache = Path(summary["verified_cache_directory"])
    deployment = summary["deployment_source_sha256"]
    blocks = {}
    for o in summary["obligations"]:
        ident = o["identity"]
        if ident["property_type"] != "integer_interval_certificate":
            continue
        cached = json.loads((cache / f"{o['digest']}.json").read_text())
        if o["status"] != "VERIFIED" or cached["status"] != "VERIFIED" or cached["identity"] != ident:
            raise ValueError(f"block {o['digest']} not VERIFIED or identity mismatch")
        if ident["deployment"] != deployment:
            raise ValueError(f"block {o['digest']} bound to another deployment")
        harness = Path(o["harness"])
        arrays = parse_harness(harness)
        idx = np.array(ident["output_indices"], dtype=np.int64)
        if arrays["EXPECTED_LOW"].size != idx.size:
            raise ValueError(f"block {o['digest']}: output size mismatch")
        blocks[o["digest"]] = {"identity": ident, "harness": str(harness),
                               "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
                               "cache_file": str(cache / f"{o['digest']}.json"),
                               "elapsed_seconds": o["elapsed_seconds"], "arrays": arrays, "indices": idx}
    return summary, blocks


def facts_from_ledger(summary, blocks, sizes):
    """Per layer: hl, hh, stable, digest-of-block, slot; plus proved one-sided raw facts."""
    L = len(sizes)
    out = []
    for k, n in enumerate(sizes):
        hl = np.full(n, np.iinfo(np.int64).min)
        hh = np.full(n, np.iinfo(np.int64).max)
        stable = np.zeros(n, dtype=np.int64)
        digest = np.empty(n, dtype=object)
        slot = np.full(n, -1)
        for d, b in blocks.items():
            if b["identity"]["layer_index"] != k:
                continue
            i = b["indices"]
            if (slot[i] >= 0).any():
                raise ValueError(f"layer {k}: neuron covered twice")
            hl[i], hh[i] = b["arrays"]["EXPECTED_LOW"], b["arrays"]["EXPECTED_HIGH"]
            stable[i] = b["arrays"].get("STABLE_RELU", np.zeros(i.size, dtype=np.int64))
            digest[i] = d
            slot[i] = np.arange(i.size)
        if (slot < 0).any():
            raise ValueError(f"layer {k}: {(slot < 0).sum()} neurons without a VERIFIED block")
        logit = k == L - 1
        upper_ok = hh < Q_HIGH
        lower_ok = (hl > Q_LOW) & (logit | (stable > 0) | (hl > 0))
        out.append({"hl": hl, "hh": hh, "stable": stable, "digest": digest, "slot": slot,
                    "upper_ok": upper_ok, "lower_ok": lower_ok})
    return out


def facts_from_box(box, sizes):
    """Screening only: the facts a VERIFIED ledger for this propagated box would yield.

    Mirrors what the interval harnesses assert (conv_proof.py): EXPECTED_LOW/HIGH
    are the post-clamp interval, ReLU'd on hidden layers; STABLE_RELU is +1 when
    pre_low >= 0, else -1 when pre_high <= 0, else 0. No ESBMC result backs these
    facts, so nothing derived from them may be exported as a certificate.
    """
    L = len(sizes)
    out = []
    for k, n in enumerate(sizes):
        pl, ph = box.pre_low[k].astype(np.int64), box.pre_high[k].astype(np.int64)
        logit = k == L - 1
        hl, hh = (pl, ph) if logit else (np.maximum(pl, 0), np.maximum(ph, 0))
        stable = np.where(pl >= 0, 1, np.where(ph <= 0, -1, 0)).astype(np.int64)
        out.append({"hl": hl, "hh": hh, "stable": stable, "digest": np.full(n, "screen", dtype=object),
                    "slot": np.arange(n), "upper_ok": hh < Q_HIGH,
                    "lower_ok": (hl > Q_LOW) & (logit | (stable > 0) | (hl > 0))})
    return out
