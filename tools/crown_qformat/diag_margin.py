"""MEASURED diagnostic: exact deployed raw logits at the centre and on random points of the eps byte box."""
import sys

import numpy as np

sys.path.insert(0, "src/scripts/crown_finder")
sys.path.insert(0, "src")
import crown_chain_export_v2 as v2  # noqa: E402
from crown_chain_export import ENC  # noqa: E402
from verification.conv_interval_fast import propagate_box  # noqa: E402

eps, image, bits, frac, n = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
v2.configure_format(bits, frac)
sel, row, net, mats, target, correct, xl, xu, box = v2.setup(eps, image)


def margin(x):
    b = propagate_box(net, x, x)
    r = np.asarray(b.pre_low[-1], dtype=np.int64)
    return int(r[target] - np.delete(r, target).max()), int(np.delete(np.arange(r.size), target)[np.argmax(np.delete(r, target))]), r


m, c, r = margin(xl.astype(np.int64) * 0 + ((xl + xu) // 2))
print("free inputs", int((xu > xl).sum()), "centre raw margin", m, "vs", c, "raw_t", int(r[target]), "max|raw|", int(np.abs(r).max()))
rng = np.random.default_rng(0)
worst = None
for i in range(n):
    x = np.where(rng.random(xl.shape) < 0.5, xl, xu).astype(np.int64)
    mm, cc, _ = margin(x)
    worst = min(worst, mm) if worst is not None else mm
print(f"{n} random corners: worst raw margin {worst}")
