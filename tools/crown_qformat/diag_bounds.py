"""MEASURED diagnostic: per hidden layer, chain vs box bounds on the screen facts (untrusted)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src/scripts/crown_finder")
sys.path.insert(0, "src")
import crown_chain_export_v2 as v2  # noqa: E402
from box_ledger import facts_from_box  # noqa: E402

eps, image, bits, frac = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
v2.configure_format(bits, frac)
sel, row, net, mats, target, correct, xl, xu, box = v2.setup(eps, image)
facts = facts_from_box(box, [m[0].shape[0] for m in mats])
bounds, info = v2.bounds_v2(mats, xl, facts)
for k, (lo, hi, sat, clip) in enumerate(bounds):
    st = facts[k]["stable"]
    dead = st < 0
    print(f"L{k} n={lo.size} active={int((st > 0).sum())} dead={int(dead.sum())} unstable={int((st == 0).sum())} "
          f"saturated={int(sat.sum())} clipped={int(clip.sum())} dead&lo>=0={int((dead & (lo >= 0)).sum())} "
          f"dead&lo>0={int((dead & (lo > 0)).sum())} dead&lo>hi={int((dead & (lo > hi)).sum())} "
          f"unstable->chain-active={int(((st == 0) & (lo >= 0)).sum())} "
          f"unstable->chain-dead={int(((st == 0) & (hi <= 0)).sum())}")

# sampled (exact deployed, clamped) ranges on random corners vs chain/box bounds
from verification.conv_interval_fast import propagate_box  # noqa: E402
rng = np.random.default_rng(0)
smin = smax = None
for _ in range(int(sys.argv[5]) if len(sys.argv) > 5 else 100):
    x = np.where(rng.random(xl.shape) < 0.5, xl, xu).astype(np.int64)
    b = propagate_box(net, x, x)
    v = [np.asarray(b.pre_low[k], dtype=np.int64) for k in range(len(mats))]
    smin = v if smin is None else [np.minimum(a, c) for a, c in zip(smin, v)]
    smax = v if smax is None else [np.maximum(a, c) for a, c in zip(smax, v)]
for k, (lo, hi, sat, clip) in enumerate(bounds):
    w_used = np.minimum(hi, v2.Q_HIGH) - np.maximum(lo, v2.Q_LOW)
    w_box = box.pre_high[k].astype(np.int64) - box.pre_low[k].astype(np.int64)
    w_s = smax[k] - smin[k]
    print(f"L{k} width median/max: used {np.median(w_used):.0f}/{w_used.max()} box {np.median(w_box):.0f}/{w_box.max()} "
          f"sampled {np.median(w_s):.0f}/{w_s.max()}; used/box mean {w_used.sum() / max(w_box.sum(), 1):.2f}")
