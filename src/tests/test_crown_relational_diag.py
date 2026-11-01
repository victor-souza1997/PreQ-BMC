"""Tests for the upstream relational diagnostic harness (docs/ssv2026/IR_upstream_016.md).

  PYTHONPATH=/home/joao/code/preqbmc-qformat/src python -m unittest tests.test_crown_relational_diag
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from verification import crown_relational_diag as rd


def random_context(seed, delta=0):
    rng = np.random.default_rng(seed)
    coords = list(range(12))
    cones = []
    for ci in range(2):
        pool = coords[:8] if ci == 0 else coords[4:]          # coords 4..7 shared
        rows2, b2, bounds2 = {}, {}, {}
        for q in range(2):
            row = sorted({int(c): int(rng.integers(-900, 900)) for c in rng.choice(pool, size=4, replace=False)}.items())
            rows2[10 * ci + q] = row
            b2[10 * ci + q] = int(rng.integers(-3000, 3000))
            bounds2[10 * ci + q] = (None, None) if q == 0 else (-20000, 20000)
        w3 = [(10 * ci + q, int(rng.integers(-600, 600))) for q in range(2)]
        cones.append(rd.Cone(name="AB"[ci], neuron3=ci, w3=w3, b3=int(rng.integers(-500, 500)), rows2=rows2, b2=b2,
                             bounds2=bounds2, bounds3=(None, 30000), a=int(rng.integers(1, 5000)),
                             a_out=int(rng.integers(0, 3000)), d=int(rng.integers(-10**6, 0))))
    h1 = {c: (0, int(rng.integers(1, 400))) for c in coords}
    return rd.Context(cones=tuple(cones), h1=h1, delta=delta, label="diag/test/rel")


class HarnessParity(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gcc"), "gcc required")
    def test_concrete_harness_matches_python_replay(self):
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(6):
                ctx = random_context(seed)
                for decoupled in (False, True):
                    rng = np.random.default_rng(100 + seed)
                    for _ in range(4):
                        vals = {n: int(rng.integers(ctx.h1[c][0], ctx.h1[c][1] + 1))
                                for n, c in ctx.variables(decoupled).items()}
                        src = Path(tmp) / "h.c"
                        src.write_text(rd.harness(ctx, decoupled, concrete=vals))
                        exe = Path(tmp) / "h"
                        subprocess.run(["gcc", "-O0", str(src), "-o", str(exe)], check=True)
                        ok, ea, eb = (int(v) for v in subprocess.run([str(exe)], capture_output=True, text=True,
                                                                       check=True).stdout.split())
                        rep = rd.replay(ctx, decoupled, vals)
                        self.assertEqual([ea, eb], rep["e"])
                        self.assertEqual(bool(ok), rep["assumptions_hold"])
                        checked += 1
        self.assertEqual(checked, 48)

    def test_decoupled_differs_only_by_shared_copies(self):
        ctx = random_context(1)
        shared = ctx.variables(False)
        dec = ctx.variables(True)
        self.assertEqual(len(dec) - len(shared), len(ctx.shared()))
        self.assertEqual(sorted(set(dec.values())), sorted(set(shared.values())))
        a = rd.harness(ctx, False).splitlines()
        b = rd.harness(ctx, True).splitlines()
        self.assertEqual(len(b) - len(a), 2 * len(ctx.shared()))     # one extra decl + assume per copy
        import re
        norm = lambda lines: [re.sub(r"x_(\d+)_[ab]\b", r"x_\1", l) for l in lines]
        self.assertEqual(sorted(set(norm(b))), sorted(set(norm(a))))

    def test_helpers_are_verbatim_from_deployed_c(self):
        deployed = Path("/home/joao/code/preqbmc/output/crown_diagnostics_20261031T161801Z/oracle/qnn_conv_native.c")
        if not deployed.exists():
            self.skipTest("deployed C not mounted")
        text = deployed.read_text()
        for block in rd.DEPLOYED_HELPERS.strip().split("\n\n"):
            body = [l for l in block.splitlines() if "QNN_ASSERT" not in l]
            for line in body:
                self.assertIn(line, text)

    def test_parse_assignment(self):
        log = "State 3 file h.c line 9\n----\n  x_12 = 34 (00000000)\n  x_7_a = -5 (1111)\n  x_12 = 35 (0)\n"
        self.assertEqual(rd.parse_assignment(log), {"x_12": 35, "x_7_a": -5})


class ReducedEncoding(unittest.TestCase):
    """Constant substitution + int32: identical e values to the int128 harness and the exact replay."""

    def context_with_fixed(self, seed, big=False):
        ctx = random_context(seed)
        rng = np.random.default_rng(500 + seed)
        for c in list(ctx.h1)[::2]:
            v = 0 if rng.random() < 0.7 else int(rng.integers(1, 50))
            ctx.h1[c] = (v, v)
        if big:
            ctx.h1[0] = (0, 30000)
            ctx.cones[0].rows2[0] = [(0, 32000)] + [(c, w) for c, w in ctx.cones[0].rows2[0] if c != 0]
        return ctx

    @unittest.skipUnless(shutil.which("gcc"), "gcc required")
    def test_reduced_matches_original_and_replay(self):
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            for seed in range(6):
                ctx = self.context_with_fixed(seed)
                fixed = rd.fixed_inputs(ctx)
                self.assertTrue(fixed)
                for decoupled in (False, True):
                    proof = rd.width_proof(ctx, decoupled)
                    self.assertTrue(proof["int32_ok"])
                    rng = np.random.default_rng(900 + seed)
                    for _ in range(4):
                        vals = {n: int(rng.integers(ctx.h1[c][0], ctx.h1[c][1] + 1))
                                for n, c in ctx.variables(decoupled).items()}
                        res = []
                        for build in (lambda: rd.harness(ctx, decoupled, concrete=vals),
                                      lambda: rd.harness_reduced(ctx, decoupled, concrete=vals)):
                            src = Path(tmp) / "h.c"
                            src.write_text(build())
                            exe = Path(tmp) / "h"
                            subprocess.run(["gcc", "-O0", "-fsanitize=signed-integer-overflow",
                                            "-fno-sanitize-recover", str(src), "-o", str(exe)], check=True)
                            res.append([int(v) for v in subprocess.run([str(exe)], capture_output=True, text=True,
                                                                         check=True).stdout.split()])
                        self.assertEqual(res[0], res[1])
                        rep = rd.replay(ctx, decoupled, vals)
                        self.assertEqual(res[1][1:], rep["e"])
                        checked += 1
                    src_red = rd.harness_reduced(ctx, decoupled)
                    for c in fixed:      # fixed inputs no longer appear as variables
                        self.assertNotIn(f"x_{c} = nondet", src_red)
                        self.assertNotIn(f"x_{c}_a = nondet", src_red)
        self.assertEqual(checked, 48)

    def test_width_proof_refuses_overflow(self):
        ctx = random_context(2)
        ctx.h1[0] = (0, 32767)
        ctx.cones[0].rows2[0] = [(0, 2 ** 20)] + [(c, w) for c, w in ctx.cones[0].rows2[0] if c != 0]
        with self.assertRaises(rd.WidthProofFailed):
            rd.width_proof(ctx, False)
        with self.assertRaises(rd.WidthProofFailed):
            rd.harness_reduced(ctx, False)

    def test_complete_assignment_fills_only_fixed(self):
        ctx = self.context_with_fixed(3)
        fixed = rd.fixed_inputs(ctx)
        names = ctx.variables(True)
        free = {n: ctx.h1[c][0] for n, c in names.items() if c not in fixed}
        full = rd.complete_assignment(ctx, True, free)
        self.assertEqual(set(full), set(names))
        self.assertTrue(all(full[n] == fixed[c] for n, c in names.items() if c in fixed))


class ShallowSolver(unittest.TestCase):
    """A two-cone context where sharing makes the relation true and decoupling makes it false."""

    def context(self):
        cone_a = rd.Cone(name="A", neuron3=0, w3=[(0, 256)], b3=0, rows2={0: [(0, 256)]}, b2={0: 0},
                         bounds2={0: (None, None)}, bounds3=(None, None), a=1, a_out=0, d=0)
        cone_b = rd.Cone(name="B", neuron3=1, w3=[(1, 256)], b3=0, rows2={1: [(0, -256)]}, b2={1: 10},
                         bounds2={1: (None, None)}, bounds3=(None, None), a=1, a_out=0, d=0)
        return rd.Context(cones=(cone_a, cone_b), h1={0: (0, 10)}, delta=10, label="diag/test/shallow")

    def test_replay_semantics(self):
        ctx = self.context()
        self.assertEqual(rd.replay(ctx, False, {"x_0": 3})["e_sum"], 10)
        r = rd.replay(ctx, True, {"x_0_a": 0, "x_0_b": 10})
        self.assertTrue(r["violates"] and r["assumptions_hold"])
        self.assertFalse(rd.replay(ctx, True, {"x_0_a": 0})["complete"])

    def test_esbmc_shared_verifies_decoupled_fails_with_trace(self):
        from verification.esbmc import ESBMCConfig, ESBMCRunner
        from verification.esbmc_install import resolve_esbmc_executable
        from verification.crown_diagnostics import parse_violation
        if not resolve_esbmc_executable():
            self.skipTest("ESBMC not installed")

        class DiagnosticRunner(ESBMCRunner):
            def build_command(self, c_file, unwind, profile):
                return tuple(a for a in super().build_command(c_file, unwind, profile) if a != "--result-only")

        runner = DiagnosticRunner(ESBMCConfig(timeout_seconds=30, memlimit="4g", default_profile="paper-z3"))
        ctx = self.context()
        with tempfile.TemporaryDirectory() as tmp:
            out = {}
            for decoupled in (False, True):
                path = Path(tmp) / f"h_{int(decoupled)}.c"
                path.write_text(rd.harness(ctx, decoupled))
                res = runner.run_file(path, "paper-z3", extract_counterexample=False)
                log = "\n".join(Path(p).read_text(errors="replace") for p in (res.stdout_log_path, res.stderr_log_path)
                                if p and Path(p).exists())
                out[decoupled] = (res.status, log)
        self.assertEqual(out[False][0], "VERIFIED")
        self.assertEqual(out[True][0], "FAILED")
        self.assertEqual(parse_violation(out[True][1], ctx.label)["failed_claim_kind"], "TARGET")
        values = rd.parse_assignment(out[True][1])
        rep = rd.replay(ctx, True, values)
        self.assertTrue(rep["complete"] and rep["assumptions_hold"] and rep["violates"], (values, rep))


if __name__ == "__main__":
    unittest.main()
