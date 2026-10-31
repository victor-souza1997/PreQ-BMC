#!/usr/bin/env python
"""Bounded diagnostics for unresolved CROWN margins (D-P, D-0, D-V, D-T, I-B, report).

Specification: docs/ssv2026/finder_diagnostic_spec_013_016.md. DIAGNOSTIC ONLY: writes
measurements and screening records under a fresh output directory; never a certificate,
never a VERIFIED label without an actual ESBMC run.

Run from the main checkout (the loaders read relative output/ paths):

  cd /home/joao/code/preqbmc
  systemd-run --user --scope -q -p MemoryMax=8G \
    /home/joao/miniconda3/envs/quad/bin/python \
    /home/joao/code/preqbmc-qformat/src/scripts/diagnose_crown_finder.py \
    --config /home/joao/code/preqbmc-qformat/experiments/crown_diagnostics_013_016_017.json \
    --stage preflight --output /home/joao/code/preqbmc/output/crown_diagnostics_<UTC>

Project imports are pinned to this worktree by scripts/diagnostic_import_guard.py, which is
loaded by file path and installed before any project import.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

WORKTREE = Path(__file__).resolve().parents[2]
SRC = WORKTREE / "src"
_guard_spec = importlib.util.spec_from_file_location("diagnostic_import_guard",
                                                     SRC / "scripts" / "diagnostic_import_guard.py")
guard = importlib.util.module_from_spec(_guard_spec)
_guard_spec.loader.exec_module(guard)

IMPLEMENTED = ("preflight", "epsilon0", "violations", "trace", "bounds", "report")
ALL_STAGES = ("preflight", "epsilon0", "violations", "trace", "slopes", "bounds", "relation", "split", "rounding",
              "coefficient_scale", "exact_fallback", "cost", "report")
FROZEN_SOURCES = {   # spec section 1, before diagnostic implementation
    "src/scripts/crown_finder/crown_chain_export_v2.py": "1541896e43a01ada3d688d311f8c3db676b83ba371a88920cc5d3f2db94a9146",
    "src/scripts/crown_finder/crown_chain_export.py": "84e61a573f180e39a6e537caf3232405d77e2f48fac15eae178ffd6df4342c36",
    "src/scripts/crown_finder/affine_export.py": "7657897bac0420885f6825f1e15179ce498a77134e4c13065abe8a4ef4fdbfc0",
    "src/verification/crown_chain_v2.py": "c6628c449d48f3968cbb7b0ce2a0495b101ef793903f02edffe9a2ca284f76ac",
    "src/verification/crown_chain_v2_coverage.py": "768168d8cbe10eb749e7ae6ccd55ccd3d3fbfcff0ecb2d6eb614feb51a562dbb",
}
SPEC_HEAD = "23e4fcf358c0502a001192b70a26ed9393a82d96"
DIAGNOSTIC_FILES = ("src/scripts/diagnose_crown_finder.py", "src/scripts/diagnostic_import_guard.py",
                    "src/verification/crown_diagnostics.py", "src/tests/test_crown_diagnostics.py",
                    "experiments/crown_diagnostics_013_016_017.json")
EXPECTED_BASELINE = {   # spec section 1 table (read from output/p3_finder_pilot_20261031)
    "013": {"margin_min": -997, "closed": 37, "chains_planned": 6770, "multiply_terms": 331893368},
    "016": {"margin_min": -876, "closed": 37, "chains_planned": 7851, "multiply_terms": 365356538},
    "017": {"margin_min": 352, "closed": 42, "chains_planned": 3240, "multiply_terms": 157246375},
}
P3_RUN = Path("output/p3_finder_pilot_20261031")
OCTOBER_RUN = Path("output/crown_chain_v2_validation_20261014")
PROTOCOL = Path("experiments/ssv2026_validation_protocol_20261014.json")
SCREEN_BOOT = r"""
import importlib.util, json, os, runpy, sys
from pathlib import Path
src, main_src, script = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
spec = importlib.util.spec_from_file_location("diagnostic_import_guard", src / "scripts" / "diagnostic_import_guard.py")
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)
finder = guard.install(src, main_src)
sys.argv = [script] + sys.argv[4:]
try:
    runpy.run_path(script, run_name="__main__")
finally:
    Path(os.environ["DIAG_IMPORT_AUDIT"]).write_text(json.dumps(guard.audit(finder, main_src)))
"""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="CROWN finder diagnostics (phase 1)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", choices=ALL_STAGES + ("all",))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--from-run", type=Path)
    ap.add_argument("--import-audit-only", action="store_true",
                    help="install the import guard, import the finder and diagnostic modules, print the audit")
    return ap.parse_args(argv)


def load_project():
    """Project imports, only after guard.install()."""

    from types import SimpleNamespace
    import affine_export
    import box_ledger
    import crown_chain_export
    import crown_chain_export_v2
    from backends import c_qnn_generator, conv_fixed_point, image_encoder
    from datasets import gtsrb_study
    from verification import conv_interval_fast, crown_diagnostics, esbmc, esbmc_install
    return SimpleNamespace(affine_export=affine_export, box_ledger=box_ledger, cce=crown_chain_export,
                           v2=crown_chain_export_v2, c_qnn_generator=c_qnn_generator,
                           conv_fixed_point=conv_fixed_point, image_encoder=image_encoder,
                           gtsrb_study=gtsrb_study, conv_interval_fast=conv_interval_fast,
                           cd=crown_diagnostics, esbmc=esbmc, esbmc_install=esbmc_install)


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(WORKTREE), *args], check=True, capture_output=True,
                          text=True).stdout


class StageFailure(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status, self.reason = status, reason


class AuditStop(RuntimeError):
    """A global stopping rule fired (spec section 9): stop all dependent work."""


MARGIN_COLUMNS = ("run_id,ordinal,domain,epsilon,target,competitor,scale_bits,scaled_raw_lb,raw_lb,nominal_raw_margin,"
                  "nominal_deployed_margin,best_seen_raw_margin,best_seen_deployed_margin,witness_id,top_rule,"
                  "top_reference,top_holds,bottom_rule,bottom_reference,bottom_holds,closed,baseline_raw_lb,"
                  "delta_raw_lb,material_gain,evidence_kind,dependency_status").split(",")
COST_COLUMNS = ("run_id,metric_scope,wall_seconds,cpu_seconds,peak_process_tree_rss_bytes,peak_cgroup_memory_bytes,"
                "solver_elapsed_sum_seconds,serialized_bytes,compressed_bytes,planned_chains,planned_multiply_terms,"
                "planned_assertions,planned_harnesses,additional_assertions,additional_harnesses,"
                "executed_solver_calls,cost_status,missing_reason").split(",")


class Run:
    def __init__(self, args, config, config_text, project, audit, finder):
        import numpy as np
        self.np = np
        self.args, self.config, self.P, self.finder = args, config, project, finder
        self.cd = project.cd
        self.config_sha = hashlib.sha256(config_text.encode()).hexdigest()
        self.import_audit = audit
        self.cwd = Path(config["working_directory"])
        self.main_src = self.cwd / "src"
        self.out = args.output.resolve() if args.output else None
        self.prior = args.from_run.resolve() if args.from_run else None
        self.start = time.monotonic()
        self.global_deadline = self.start + float(config["global_wall_cap_seconds"])
        self.images = config["images"]
        self.eps = int(config["epsilon_raw_bytes"])

    # ------------------------------------------------------------ setup and provenance

    def check_environment(self):
        if Path.cwd().resolve() != self.cwd.resolve():
            raise SystemExit(f"run from {self.cwd} (the loaders read relative output/ paths)")
        parent = Path(self.config["output_parent"]).resolve()
        if not self.out.is_relative_to(parent):
            raise SystemExit(f"--output must lie under {parent}")
        for protected in (OCTOBER_RUN, P3_RUN, Path("output/p2_reaudit_20261030")):
            if self.out.is_relative_to(protected.resolve()):
                raise SystemExit("--output must not be inside a historical run")

    def open_output(self, first_stage: str):
        if self.out.exists() and any(self.out.iterdir()):
            if self.prior != self.out:
                raise SystemExit(f"{self.out} is not empty; use a fresh directory, or --from-run {self.out} "
                                 "to append a not-yet-run stage")
        self.out.mkdir(parents=True, exist_ok=True)
        for sub in ("witnesses", "cache", "oracle", "obligations", "logs"):
            (self.out / sub).mkdir(exist_ok=True)
        manifest_path = self.out / "manifest.json"
        current = self.build_manifest()
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text())
            if old["source_manifest_sha256"] != current["source_manifest_sha256"] or old["config_sha256"] != self.config_sha:
                raise SystemExit("this run's manifest has different sources or config; use a fresh output")
            self.manifest = old
        else:
            if self.prior is not None and self.prior != self.out:
                old = json.loads((self.prior / "manifest.json").read_text())
                if old["source_manifest_sha256"] != current["source_manifest_sha256"] or old["config_sha256"] != self.config_sha:
                    raise SystemExit("--from-run has different sources or config")
                current["from_run"] = str(self.prior)
            self.manifest = current
            self.write_manifest()
        self.context = {"deployment_sha256": self.config["deployment_sha256"],
                        "source_manifest_sha256": self.manifest["source_manifest_sha256"],
                        "scale_bits": int(self.config["coefficient_scale_bits"])}
        self.runs = self.cd.RunLog(self.out / "runs.jsonl", self.context)

    def write_manifest(self):
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=1, sort_keys=True) + "\n")

    def build_manifest(self):
        np = self.np
        frozen = {path: {"expected": sha, "actual": sha256_file(WORKTREE / path)}
                  for path, sha in FROZEN_SOURCES.items()}
        modules = self.import_audit["modules"]
        sources = {m["module"]: {"file": m["file"], "sha256": m["sha256"], "search_locations": m["search_locations"]}
                   for m in modules}
        head = git("rev-parse", "HEAD").strip()
        production_diff = git("diff", "HEAD", "--", "src", ":(exclude)src/scripts/diagnose_crown_finder.py",
                              ":(exclude)src/scripts/diagnostic_import_guard.py",
                              ":(exclude)src/verification/crown_diagnostics.py",
                              ":(exclude)src/tests/test_crown_diagnostics.py")
        diagnostic = {p: sha256_file(WORKTREE / p) for p in DIAGNOSTIC_FILES if (WORKTREE / p).exists()}
        source_section = {"modules": sources, "frozen": frozen, "diagnostic_files": diagnostic,
                          "production_diff_sha256": hashlib.sha256(production_diff.encode()).hexdigest()}
        esbmc = self.P.esbmc_install.resolve_esbmc_executable() or "esbmc"
        esbmc_version = subprocess.run([esbmc, "--version"], capture_output=True, text=True).stdout.strip()
        gcc = shutil.which("gcc")
        meminfo = {l.split(":")[0]: l.split(":")[1].strip() for l in Path("/proc/meminfo").read_text().splitlines()
                   if l.startswith(("MemTotal", "SwapTotal"))}
        cg_max = self.cd.cgroup_memory_file("memory.max")
        return {
            "schema": "preqbmc_crown_diagnostic_manifest_v1",
            "created_utc": utc_now(),
            "phase": "1 + I-B (D-P, D-0, D-V, D-T, I-B, report)",
            "specification": {"path": "docs/ssv2026/finder_diagnostic_spec_013_016.md",
                              "sha256": sha256_file(WORKTREE / "docs/ssv2026/finder_diagnostic_spec_013_016.md")},
            "config_path": str(self.args.config.resolve()), "config_sha256": self.config_sha, "config": self.config,
            "worktree": str(WORKTREE), "worktree_src": str(SRC), "main_checkout": str(self.cwd),
            "main_checkout_src": str(self.main_src), "working_directory": str(Path.cwd()),
            "git": {"branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(), "head": head,
                    "spec_head": SPEC_HEAD,
                    "spec_head_is_ancestor": subprocess.run(["git", "-C", str(WORKTREE), "merge-base", "--is-ancestor",
                                                             SPEC_HEAD, head]).returncode == 0,
                    "commits_since_spec_head": git("log", "--oneline", f"{SPEC_HEAD}..{head}").strip().splitlines(),
                    "status_short": git("status", "--short").splitlines()},
            "import_isolation": {k: v for k, v in self.import_audit.items() if k != "modules"},
            "sources": source_section,
            "source_manifest_sha256": sha256_json(source_section),
            "versions": {"python": sys.version, "executable": sys.executable, "numpy": np.__version__,
                         "platform": platform.platform(), "compiler_path": gcc,
                         "compiler_version": subprocess.run([gcc, "--version"], capture_output=True,
                                                            text=True).stdout.splitlines()[0] if gcc else None,
                         "esbmc_path": esbmc, "esbmc_sha256": sha256_file(esbmc) if Path(esbmc).exists() else None,
                         "esbmc_version": esbmc_version},
            "machine": {"meminfo": meminfo, "cpu_count": os.cpu_count(),
                        "cgroup_memory_max": cg_max.read_text().strip() if cg_max else None},
            "seeds": {"seed_base": self.config["seed_base"],
                      "formula": "PCG64(seed_base + 1000*ordinal_int + 10*competitor + restart)"},
            "stages": {},
        }

    def note_unimplemented(self, names):
        self._unimplemented = names

    # ------------------------------------------------------------ records

    def envelope(self, image, **fields):
        base = {"ordinal": image["ordinal"], "image_id": image["record_id"], "target": image["target"]}
        base.update(fields)
        reasons = dict(base.get("missing_reason") or {})
        for key in ("competitor", "epsilon_raw_bytes", "domain", "region_sha256", "assumption_context_sha256",
                    "parent_run_id", "dependency_status"):
            if base.get(key) is None:
                reasons.setdefault(key, "not applicable to this record")
        metrics = dict(base.get("metrics") or {})
        for key in self.cd.METRIC_FIELDS:
            metrics.setdefault(key, None)
            if metrics[key] is None:
                reasons.setdefault(f"metrics.{key}", "not measured by this stage")
        base["metrics"], base["missing_reason"] = metrics, reasons
        return self.runs.write(**base)

    def margin_row(self, **row):
        path = self.out / "margins.csv"
        new = not path.exists()
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MARGIN_COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({k: ("" if row.get(k) is None else str(row[k]).lower() if isinstance(row.get(k), bool)
                                 else str(row[k])) for k in MARGIN_COLUMNS})

    def cost_row(self, record):
        path = self.out / "costs.csv"
        new = not path.exists()
        m = record["metrics"]
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=COST_COLUMNS)
            if new:
                writer.writeheader()
            missing = {k: v for k, v in record["missing_reason"].items() if k.startswith("metrics.")}
            writer.writerow({"run_id": record["run_id"], "metric_scope": record["metric_scope"],
                             **{k: "" if m.get(k) is None else m[k] for k in COST_COLUMNS[2:16]},
                             "cost_status": "MEASURED" if m.get("wall_seconds") is not None else "MISSING",
                             "missing_reason": json.dumps(missing, sort_keys=True)})

    def record(self, image, resources=None, **fields):
        if resources is not None:
            metrics, reasons = resources.metrics()
            fields["metrics"] = {**metrics, **(fields.get("metrics") or {})}
            fields["missing_reason"] = {**reasons, **(fields.get("missing_reason") or {})}
        rec = self.envelope(image, **fields)
        self.cost_row(rec)
        return rec

    def wait_for_memory(self, deadline):
        floor = int(self.config["minimum_available_memory_gib"]) << 30
        while self.cd.mem_available_bytes() < floor:
            if time.monotonic() > deadline:
                return False
            time.sleep(1.0)
        return True

    # ------------------------------------------------------------ driver

    def execute(self, stages):
        self.check_environment()
        self.open_output(stages[0])
        status = 0
        for stage in stages:
            cap = float(self.config["stage_wall_caps_seconds"].get(stage, 0)) if stage != "report" else 600.0
            deadline = min(time.monotonic() + cap, self.global_deadline)
            started = utc_now()
            try:
                result = getattr(self, f"stage_{stage}")(deadline)
                outcome = "COMPLETED" if result is None else result
            except AuditStop as stop:
                outcome = f"AUDIT_STOP: {stop}"
                status = 4
            except StageFailure as failure:
                outcome = f"{failure.status}: {failure.reason}"
                status = 5
            self.manifest["stages"][stage] = {"started_utc": started, "ended_utc": utc_now(), "outcome": outcome}
            self.write_manifest()
            print(json.dumps({"stage": stage, "outcome": outcome}), flush=True)
            if status:
                remaining = stages[stages.index(stage) + 1:]
                for later in remaining:
                    self.manifest["stages"][later] = {"outcome": "NOT_RUN: earlier stage stopped"}
                self.write_manifest()
                break
        if getattr(self, "_unimplemented", None):
            self.manifest["not_implemented_phase1"] = self._unimplemented
            self.write_manifest()
        return status

    def require(self, name):
        """Load a prerequisite artifact from this run or --from-run, checking its recorded hash."""

        for base in [self.out] + ([self.prior] if self.prior and self.prior != self.out else []):
            gate = base / "preflight.json"
            if gate.exists():
                break
        else:
            raise StageFailure("DEPENDENCY_FAILED", "no preflight.json; run --stage preflight first")
        preflight = json.loads(gate.read_text())
        if not preflight.get("passed"):
            raise StageFailure("DEPENDENCY_FAILED", "preflight did not pass; downstream stages refuse to run")
        if preflight["source_manifest_sha256"] != self.manifest["source_manifest_sha256"]:
            raise StageFailure("DEPENDENCY_FAILED", "preflight ran on different sources")
        if name == "preflight":
            return base, preflight
        path = base / name
        if not path.exists():
            raise StageFailure("DEPENDENCY_FAILED", f"{name} missing in {base}")
        return base, json.loads(path.read_text())

    # ------------------------------------------------------------ shared model state

    def model(self):
        if getattr(self, "_model", None) is None:
            P = self.P
            P.v2.configure_format(int(self.config["qif"]["total_bits"]), int(self.config["qif"]["fractional_bits"]))
            sel, base, net, enc, rows = P.affine_export.load()
            mats = [P.affine_export.lowered(layer) for layer in net.layers]
            source = P.conv_fixed_point.generate_conv_qnn_source(net, encoder_source=enc.render_c())
            self._model = {"sel": sel, "base": base, "net": net, "enc": enc, "rows": rows, "mats": mats,
                           "source": source, "relu": [bool(l.apply_relu) for l in net.layers],
                           "sizes": [m[0].shape[0] for m in mats], "biases": [m[1] for m in mats]}
        return self._model

    def region(self, image, eps):
        """Fresh region via the production setup(); returns state and the XU it must run under."""

        P, np = self.P, self.np
        sel, row, net, mats, target, correct, xl, xu, box = P.cce.setup(eps, image["record_id"])
        if row["id"] != image["record_id"] or target != image["target"]:
            raise AuditStop(f"{image['ordinal']}: record/target mismatch ({row['id']}, {target})")
        m = self.model()
        crop = P.gtsrb_study.load_crop(m["base"]["dataset_root"], row)
        blo, bhi = m["enc"].byte_box(crop, eps)
        return {"row": row, "target": target, "correct": bool(correct), "xl": xl.astype(np.int64),
                "xu": xu.astype(np.int64), "box": box, "crop": crop, "blo": blo, "bhi": bhi, "eps": eps}

    class XUState:
        """Pin crown_chain_export.XU[0] for one block of chain computations, then restore it."""

        def __init__(self, cce, xu):
            self.cce, self.xu = cce, xu

        def __enter__(self):
            self.saved = self.cce.XU[0]
            self.cce.XU[0] = self.xu
            return self

        def __exit__(self, *exc):
            if self.cce.XU[0] is not self.xu:
                raise RuntimeError("XU was replaced inside a pinned block")
            self.cce.XU[0] = self.saved

    def region_hash(self, reg, groups):
        np = self.np
        return sha256_json({"xl": self.cd.sha256_array(reg["xl"]), "xu": self.cd.sha256_array(reg["xu"]),
                            "blo": self.cd.sha256_array(reg["blo"]), "bhi": self.cd.sha256_array(reg["bhi"]),
                            "alias_source": self.cd.sha256_array(groups.source),
                            "alias_of": self.cd.sha256_array(groups.of_coordinate), "eps": reg["eps"]})

    def baseline(self, image, reg):
        """In-process eps-1 screen: bounds, facts, root bounds, closure, focus steps. Screening only."""

        P, np, cd = self.P, self.np, self.cd
        m = self.model()
        mats, sizes = m["mats"], m["sizes"]
        top = len(mats) - 1
        target = reg["target"]
        comps = [c for c in range(sizes[-1]) if c != target]
        with self.XUState(P.cce, reg["xu"]):
            facts = P.box_ledger.facts_from_box(reg["box"], sizes)
            bounds, info = P.v2.bounds_v2(mats, reg["xl"], facts)
            lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
            lams[:, target] = 1 << cd.SCALE_BITS
            lams[np.arange(len(comps)), comps] = -(1 << cd.SCALE_BITS)
            scaled, terms, _ = P.v2.run_chain(lams, top, mats, bounds, reg["xl"])
            e = np.zeros((2, sizes[-1]), dtype=np.int64)
            e[0, target], e[1, target] = 1 << cd.SCALE_BITS, -(1 << cd.SCALE_BITS)
            t_scaled, _, _ = P.v2.run_chain(e, top, mats, bounds, reg["xl"])
            focus_rows = [comps.index(c) for c in image["focus"]]
            _, _, steps = P.v2.run_chain(lams[focus_rows], top, mats, bounds, reg["xl"], record=True)
        raw_lb = {c: cd.ceil_div(int(s), 1 << cd.SCALE_BITS) for c, s in zip(comps, scaled.tolist())}
        raw_t_low = cd.ceil_div(int(t_scaled[0]), 1 << cd.SCALE_BITS)
        raw_t_high = (-int(t_scaled[1])) // (1 << cd.SCALE_BITS)
        closure = cd.closure_edges(raw_lb, facts[top]["hh"], facts[top]["hl"], facts[top]["lower_ok"],
                                   raw_t_low, raw_t_high, target)
        final = {c: steps[-1]["a"][i].copy() for i, c in enumerate(image["focus"])}
        return {"facts": facts, "bounds": bounds, "info": info, "comps": comps, "lams": lams,
                "scaled": [int(v) for v in scaled.tolist()], "terms": [int(v) for v in terms.tolist()],
                "raw_lb": raw_lb, "raw_t": [raw_t_low, raw_t_high], "closure": closure, "final_coefficients": final}

    def save_cache(self, image, reg, base, groups):
        np = self.np
        arrays = {"xl": reg["xl"], "xu": reg["xu"], "blo": reg["blo"], "bhi": reg["bhi"], "crop": reg["crop"],
                  "alias_source": groups.source, "alias_of": groups.of_coordinate}
        for k in range(len(reg["box"].pre_low)):
            arrays[f"p{k}_lo"] = np.asarray(reg["box"].pre_low[k], dtype=np.int64)    # post-clamp, pre-ReLU
            arrays[f"p{k}_hi"] = np.asarray(reg["box"].pre_high[k], dtype=np.int64)
        for k, (lo, hi, sat, clip) in enumerate(base["bounds"]):
            arrays.update({f"b{k}_lo": lo, f"b{k}_hi": hi, f"b{k}_sat": sat, f"b{k}_clip": clip})
            for key in ("lo_chain", "hi_chain", "lower_box", "upper_chain"):
                arrays[f"i{k}_{key}"] = base["info"][k][key]
        for k, f in enumerate(base["facts"]):
            for key in ("hl", "hh", "stable", "upper_ok", "lower_ok"):
                arrays[f"f{k}_{key}"] = np.asarray(f[key])
        for c, a in base["final_coefficients"].items():
            arrays[f"final_{c}"] = a
        path = self.out / "cache" / f"{image['ordinal']}_eps1.npz"
        np.savez(path, **arrays)
        meta = {"comps": base["comps"], "scaled": [str(v) for v in base["scaled"]], "terms": base["terms"],
                "raw_lb": {str(c): v for c, v in base["raw_lb"].items()}, "raw_t": base["raw_t"],
                "closure": {str(c): v for c, v in base["closure"].items()}, "npz_sha256": sha256_file(path)}
        meta_path = self.out / "cache" / f"{image['ordinal']}_eps1.json"
        meta_path.write_text(json.dumps(meta, sort_keys=True))
        return {"npz": str(path), "npz_sha256": meta["npz_sha256"], "json": str(meta_path),
                "json_sha256": sha256_file(meta_path)}

    def load_cache(self, base_dir, image, preflight):
        np = self.np
        entry = preflight["caches"][image["ordinal"]]
        npz, meta = Path(entry["npz"]), Path(entry["json"])
        if sha256_file(npz) != entry["npz_sha256"] or sha256_file(meta) != entry["json_sha256"]:
            raise StageFailure("DEPENDENCY_FAILED", f"{image['ordinal']}: cached baseline changed after preflight")
        data = dict(np.load(npz, allow_pickle=False))
        meta = json.loads(meta.read_text())
        layers = len(self.model()["mats"]) - 1
        bounds = [(data[f"b{k}_lo"], data[f"b{k}_hi"], data[f"b{k}_sat"], data[f"b{k}_clip"]) for k in range(layers)]
        info = [{key: data[f"i{k}_{key}"] for key in ("lo_chain", "hi_chain", "lower_box", "upper_chain")}
                for k in range(layers)]
        facts = [{key: data[f"f{k}_{key}"] for key in ("hl", "hh", "stable", "upper_ok", "lower_ok")}
                 for k in range(layers + 1)]
        return data, meta, bounds, info, facts

    # ------------------------------------------------------------ C oracle

    def build_oracle(self):
        P, cd = self.P, self.cd
        m = self.model()
        oracle = self.out / "oracle"
        source = m["source"]
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != self.config["deployment_sha256"]:
            raise AuditStop(f"regenerated deployment C {digest} != pinned {self.config['deployment_sha256']}")
        c_path = oracle / "qnn_conv_native.c"
        c_path.write_text(source)
        observer = cd.observer_source(source, m["sizes"])
        if cd.strip_observer(observer) != source:
            raise AuditStop("observer instrumentation changed unobserved source bytes")
        obs_path = oracle / "qnn_conv_native_observer.c"
        obs_path.write_text(observer)
        libs, commands = {}, {}
        for name, path, opt in (("O0", c_path, "-O0"), ("O2", c_path, "-O2"), ("observer_O2", obs_path, "-O2")):
            target = oracle / f"{name}.so"
            P.c_qnn_generator.compile_c_qnn_shared_library(path, target, "gcc", optimization=opt)
            libs[name] = target
            commands[name] = ["gcc", "-shared", "-fPIC", opt, str(path), "-o", str(target)]
        n_in, n_out = int(self.np.prod(m["net"].input_shape)), m["sizes"][-1]
        self.oracles = {"O0": cd.COracle(libs["O0"], n_in, n_out), "O2": cd.COracle(libs["O2"], n_in, n_out),
                        "observer": cd.COracle(libs["observer_O2"], n_in, n_out, m["sizes"])}
        return {"source": str(c_path), "source_sha256": digest, "observer": str(obs_path),
                "observer_sha256": sha256_file(obs_path), "compile_commands": commands,
                "libraries": {k: {"path": str(v), "sha256": sha256_file(v)} for k, v in libs.items()}}

    def load_oracle(self, preflight):
        cd = self.cd
        m = self.model()
        info = preflight["oracle"]
        for name, lib in info["libraries"].items():
            if sha256_file(lib["path"]) != lib["sha256"]:
                raise StageFailure("DEPENDENCY_FAILED", f"oracle library {name} changed after preflight")
        n_in, n_out = int(self.np.prod(m["net"].input_shape)), m["sizes"][-1]
        self.oracles = {"O0": cd.COracle(Path(info["libraries"]["O0"]["path"]), n_in, n_out),
                        "O2": cd.COracle(Path(info["libraries"]["O2"]["path"]), n_in, n_out)}

    def c_eval(self, image_bytes, lib="O2"):
        oracle = self.oracles[lib]
        x = oracle.encode(image_bytes)
        return x, oracle.forward(x)

    # ------------------------------------------------------------ D-P

    def stage_preflight(self, deadline):
        P, np, cd = self.P, self.np, self.cd
        failures = []
        audit = self.import_audit
        if not audit["passed"]:
            failures.append({"check": "import_isolation", "violations": audit["violations"]})
        for path, entry in self.manifest["sources"]["frozen"].items():
            if entry["expected"] != entry["actual"]:
                failures.append({"check": "frozen_source", "path": path, **entry})
        protocol = json.loads(PROTOCOL.read_text())
        search_path = Path("output/sign_conv_depthwise_pool_search_20260924/search_summary.json")
        search = json.loads(search_path.read_text())
        data_actual = {"search_summary_sha256": sha256_file(search_path),
                       "base_study_sha256": sha256_file(search["base_study_path"]),
                       "folded_model_sha256": sha256_file(search["selected_candidate_summary"]["folded_model_path"]),
                       "deployment_source_sha256": self.config["deployment_sha256"]}
        for key, value in data_actual.items():
            if protocol["data"][key] != value:
                failures.append({"check": "data_identity", "field": key, "protocol": protocol["data"][key],
                                 "actual": value})
        test_eval = Path("output/sign_conv_depthwise_pool_test_20260924/test_evaluation.json")
        data_actual["test_evaluation_sha256"] = sha256_file(test_eval) if test_eval.exists() else None
        if data_actual["test_evaluation_sha256"] not in (None, protocol["data"]["test_evaluation_sha256"]):
            failures.append({"check": "data_identity", "field": "test_evaluation_sha256"})
        t0 = time.monotonic()
        with cd.ResourceMonitor() as setup_res:
            m = self.model()
            oracle_info = self.build_oracle()
        setup_record = self.record({"ordinal": "all", "record_id": None, "target": None}, setup_res,
                                   run_id="all/D-P/setup", diagnostic="D-P", evidence_kind="MEASURED_EXACT_EXECUTION",
                                   dependency_status="NONE", metric_scope="setup_and_compile",
                                   parameters={"compile_commands": oracle_info["compile_commands"]},
                                   missing_reason={"image_id": "stage-wide record", "target": "stage-wide record"},
                                   interpretation={"hypothesis_id": "H-P", "assessment": "INCONCLUSIVE",
                                                   "reason_code": "SETUP_ONLY"})
        images_json, oracle_json, caches, baselines = [], {"build": oracle_info, "inputs": []}, {}, {}
        counts = {"O0_O2_mismatch_count": 0, "C_Python_mismatch_count": 0, "trace_mismatch_count": 0,
                  "encoder_mismatch_count": 0}
        for image in self.images:
            if time.monotonic() > deadline:
                failures.append({"check": "budget", "ordinal": image["ordinal"]})
                break
            if not self.wait_for_memory(deadline):
                failures.append({"check": "memory", "ordinal": image["ordinal"]})
                break
            ordinal = image["ordinal"]
            draw = protocol["images_in_draw_order"][int(ordinal)]
            if draw != image["record_id"]:
                failures.append({"check": "draw_order", "ordinal": ordinal, "protocol": draw})
            with cd.ResourceMonitor() as res:
                reg = self.region(image, self.eps)
                crop, enc = reg["crop"], m["enc"]
                idx = cd.resize_source_index(crop.shape, enc.output_shape, enc.resize_mode)
                if not np.array_equal(crop.reshape(-1)[idx], enc.resize(crop).reshape(-1)):
                    failures.append({"check": "alias_map", "ordinal": ordinal})
                groups = cd.alias_groups(idx)
                region_sha = self.region_hash(reg, groups)
                inputs = {"nominal": crop, "box_low": reg["blo"], "box_high": reg["bhi"]}
                logits = {}
                for name, image_bytes in inputs.items():
                    x0, y0 = self.c_eval(image_bytes, "O0")
                    x2, y2 = self.c_eval(image_bytes, "O2")
                    xo = self.oracles["observer"].encode(image_bytes)
                    yo = self.oracles["observer"].forward(xo)
                    observed = self.oracles["observer"].observed()
                    x_py = enc.encode(image_bytes)
                    exact = cd.exact_forward(m["mats"], x_py, m["relu"])
                    ref, ref_trace = P.conv_fixed_point.forward_conv_fixed_point_single(m["net"], x_py, return_trace=True)
                    enc_bad = int(not (np.array_equal(x0, x2) and np.array_equal(x2, x_py) and np.array_equal(xo, x_py)))
                    o0o2 = int(not np.array_equal(y0, y2))
                    cpy = int(not (np.array_equal(y2, ref) and np.array_equal(y2, exact[-1]["z"]) and np.array_equal(yo, y2)))
                    trace_bad = 0
                    for k, layer in enumerate(exact):
                        for key in "srzh":
                            trace_bad += int(np.count_nonzero(layer[key] != observed[k][key]))
                        trace_bad += int(np.count_nonzero(np.asarray(ref_trace[k]).reshape(-1) != layer["h"]))
                    counts["encoder_mismatch_count"] += enc_bad
                    counts["O0_O2_mismatch_count"] += o0o2
                    counts["C_Python_mismatch_count"] += cpy
                    counts["trace_mismatch_count"] += trace_bad
                    margins = cd.margins_from_logits(y2, exact[-1]["r"], reg["target"])
                    logits[name] = {"deployed": [int(v) for v in y2], "raw": [cd.dec(v) for v in exact[-1]["r"]],
                                    "byte_sha256": cd.sha256_array(np.asarray(image_bytes, dtype=np.uint8)),
                                    "encoded_sha256": cd.sha256_array(x2), "shape": list(np.asarray(image_bytes).shape),
                                    "min_deployed_margin": min(v["deployed"] for v in margins.values()),
                                    "violating_competitors": sorted(c for c, v in margins.items()
                                                                    if cd.deployed_violation(v["deployed"]))}
                    if name == "nominal":
                        nominal_margins, nominal_exact = margins, exact
                nominal_bad = logits["nominal"]["violating_competitors"]
                screen = self.production_screen(image)
                base = self.baseline(image, reg)
                caches[ordinal] = self.save_cache(image, reg, base, groups)
            comparison = self.compare_baseline(image, base, screen)
            if nominal_bad:
                failures.append({"check": "nominal_strict_dominance", "ordinal": ordinal, "violations": nominal_bad})
                if image.get("control"):
                    raise AuditStop(f"control {ordinal} violates strict dominance at the nominal input")
            if not comparison["passed"]:
                failures.append({"check": "baseline_reproduction", "ordinal": ordinal, "details": comparison})
            images_json.append({
                "ordinal": ordinal, "record_id": image["record_id"], "target": image["target"],
                "focus": image["focus"], "primary": image["primary"], "control": bool(image.get("control")),
                "aliases": {"draw_position": int(ordinal), "protocol": str(PROTOCOL),
                            "validation_index": next(i for i, r in enumerate(m["rows"]) if r["id"] == image["record_id"])},
                "dataset_sha256": reg["row"].get("sha256"), "crop_shape": list(crop.shape),
                "crop_sha256": cd.sha256_array(crop), "encoded_nominal_sha256": logits["nominal"]["encoded_sha256"],
                "box_low_sha256": logits["box_low"]["byte_sha256"], "box_high_sha256": logits["box_high"]["byte_sha256"],
                "xl_sha256": cd.sha256_array(reg["xl"]), "xu_sha256": cd.sha256_array(reg["xu"]),
                "region_sha256": region_sha,
                "alias_groups": {"groups": int(groups.source.size), "resized_coordinates": int(idx.size),
                                 "size_histogram": groups.sizes,
                                 "unused_crop_bytes": int(crop.size - groups.source.size)},
                "encoder_scalar_identity_F8": bool(all((v * 256 + 128) // 256 == v for v in range(256))),
                "expected_baseline": EXPECTED_BASELINE[ordinal], "baseline_comparison": comparison,
                "historical": self.historical(ordinal)})
            oracle_json["inputs"].append({"ordinal": ordinal, "logits": logits})
            parent = f"{ordinal}/D-P/baseline/eps1"
            rec = self.record(image, res, run_id=parent, diagnostic="D-P", epsilon_raw_bytes=self.eps, domain="D_box",
                              region_sha256=region_sha,
                              assumption_context_sha256=sha256_json({"region": region_sha,
                                                                     "facts": "facts_from_box (screening)"}),
                              evidence_kind="SCREEN_BOUND_UNCHECKED", dependency_status="SCREEN_ONLY",
                              metric_scope="image_preflight",
                              metrics={"planned_chains": screen["report"].get("chains_planned"),
                                       "planned_multiply_terms": (screen["report"].get("cost") or {}).get("multiply_terms")},
                              artifacts={"production_screen": screen["report_path"], **caches[ordinal]},
                              parameters={"production_screen_command": screen["command"]},
                              interpretation={"hypothesis_id": "H-P",
                                              "assessment": "SUPPORTS" if comparison["passed"] and not nominal_bad
                                              else "IMPLEMENTATION_ERROR",
                                              "reason_code": "BASELINE_REPRODUCED" if comparison["passed"]
                                              else "BASELINE_MISMATCH"})
            for c in base["comps"]:
                edge = base["closure"][c]
                self.margin_row(run_id=parent, ordinal=ordinal, domain="D_box", epsilon=self.eps, target=image["target"],
                                competitor=c, scale_bits=cd.SCALE_BITS, scaled_raw_lb=base["scaled"][base["comps"].index(c)],
                                raw_lb=base["raw_lb"][c], nominal_raw_margin=nominal_margins[c]["raw"],
                                nominal_deployed_margin=nominal_margins[c]["deployed"],
                                top_rule=(edge["top"] or {}).get("rule"), top_reference=(edge["top"] or {}).get("reference"),
                                top_holds=edge["top"] is not None, bottom_rule=(edge["bottom"] or {}).get("rule"),
                                bottom_reference=(edge["bottom"] or {}).get("reference"),
                                bottom_holds=edge["bottom"] is not None, closed=edge["closed"],
                                baseline_raw_lb=base["raw_lb"][c], delta_raw_lb=0, material_gain=False,
                                evidence_kind="SCREEN_BOUND_UNCHECKED", dependency_status="SCREEN_ONLY")
            del base, reg, nominal_exact
        if any(counts.values()):
            failures.append({"check": "parity", **counts})
        (self.out / "images.json").write_text(json.dumps(images_json, indent=1, sort_keys=True) + "\n")
        oracle_json.update(counts, alias_groups={i["ordinal"]: i["alias_groups"] for i in images_json})
        (self.out / "oracle.json").write_text(json.dumps(oracle_json, indent=1, sort_keys=True) + "\n")
        passed = not failures
        gate = {"passed": passed, "failures": failures, "source_manifest_sha256": self.manifest["source_manifest_sha256"],
                "config_sha256": self.config_sha, "oracle": oracle_info, "caches": caches, "data_identity": data_actual,
                "seconds": round(time.monotonic() - t0, 1)}
        (self.out / "preflight.json").write_text(json.dumps(gate, indent=1, sort_keys=True) + "\n")
        if not passed:
            raise StageFailure("PARITY_FAILURE", json.dumps(failures)[:2000])
        return None

    def production_screen(self, image):
        """The unmodified production screen (--cost, no --search) in a guarded subprocess."""

        out = self.out / "logs" / f"{image['ordinal']}_production_screen.json"
        audit_path = self.out / "logs" / f"{image['ordinal']}_production_screen_imports.json"
        script = SRC / "scripts" / "crown_finder" / "crown_chain_export_v2.py"
        command = [sys.executable, "-c", SCREEN_BOOT, str(SRC), str(self.main_src), str(script), "screen",
                   str(self.eps), image["record_id"], "--cost", "--report", str(out)]
        env = dict(os.environ, DIAG_IMPORT_AUDIT=str(audit_path))
        proc = subprocess.run(command, cwd=self.cwd, env=env, capture_output=True, text=True)
        (self.out / "logs" / f"{image['ordinal']}_production_screen.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0 or not out.exists():
            raise StageFailure("DEPENDENCY_FAILED", f"{image['ordinal']}: production screen exited {proc.returncode}")
        imports = json.loads(audit_path.read_text())
        if not imports["passed"]:
            raise StageFailure("PARITY_FAILURE", f"{image['ordinal']}: production screen imported outside the worktree")
        return {"report": json.loads(out.read_text()), "report_path": str(out),
                "command": command[:2] + ["<SCREEN_BOOT>"] + command[3:], "imports_passed": True}

    def compare_baseline(self, image, base, screen):
        report, expected = screen["report"], EXPECTED_BASELINE[image["ordinal"]]
        closed = sorted(c for c, e in base["closure"].items() if e["closed"])
        open_set = sorted(int(c) for c in (report.get("open_competitors") or {}))
        checks = {
            "screen_margin_min": report.get("margin_min") == expected["margin_min"],
            "screen_closed": report.get("closed") == expected["closed"],
            "screen_chains_planned": report.get("chains_planned") == expected["chains_planned"],
            "screen_multiply_terms": (report.get("cost") or {}).get("multiply_terms") == expected["multiply_terms"],
            "screen_no_search": "search" not in report,
            "inprocess_margin_min": min(base["raw_lb"].values()) == report.get("margin_min"),
            "inprocess_closed_count": len(closed) == report.get("closed"),
            "inprocess_open_set": sorted(set(base["comps"]) - set(closed)) == open_set,
            "focus_equals_open_set": (sorted(image["focus"]) == open_set) if not image.get("control") else
            (open_set == [] and image["primary"] in base["comps"]),
        }
        saved = P3_RUN / f"{image['ordinal']}.json"
        p3 = json.loads(saved.read_text()) if saved.exists() else None
        if p3 and "root_scaled" in p3:
            trace = (p3.get("search") or {}).get("trace") or {}
            unchanged = all(t is not None and all(x["chosen"] == -1 for x in t) for t in trace.values())
            mismatched = [c for i, c in enumerate(base["comps"])
                          if int(p3["root_scaled"][f"chain/margin/{image['target']}-{c}"]) != base["scaled"][i]]
            roots = {"source": str(saved), "sha256": sha256_file(saved), "p3_search_changed_nothing": unchanged,
                     "compared": len(base["comps"]), "mismatched": mismatched}
            if unchanged:
                checks["p3_root_scaled_all_42"] = not mismatched
        else:
            roots = {"source": None, "unavailable_reason": "no saved per-root scaled bounds"}
        return {"checks": checks, "passed": all(checks.values()), "saved_roots": roots,
                "screen_values": {k: report.get(k) for k in ("margin_min", "closed", "chains_planned", "tightest")},
                "screen_multiply_terms": (report.get("cost") or {}).get("multiply_terms")}

    def historical(self, ordinal):
        out = {}
        for name, path in (("october_screen", OCTOBER_RUN / ordinal / "screen.json"),
                           ("october_outcome", OCTOBER_RUN / ordinal / "outcome.json"),
                           ("p3_report", P3_RUN / f"{ordinal}.json"),
                           ("p2_reaudit", Path(f"output/p2_reaudit_20261030/validation_{ordinal}.json"))):
            if path.exists():
                out[name] = {"path": str(path), "sha256": sha256_file(path)}
                if name == "p2_reaudit":
                    out[name]["verdict"] = json.loads(path.read_text()).get("verdict")
        return out

    # ------------------------------------------------------------ D-0

    def stage_epsilon0(self, deadline):
        P, np, cd = self.P, self.np, self.cd
        base_dir, preflight = self.require("preflight")
        self.load_oracle(preflight)
        m = self.model()
        mats, sizes = m["mats"], m["sizes"]
        top = len(mats) - 1
        out = self.out / "singleton.jsonl"
        arms_cache = {}
        for image in self.images:
            ordinal, target = image["ordinal"], image["target"]
            if time.monotonic() > deadline or not self.wait_for_memory(deadline):
                for arm in ("Z-inherited", "Z-fresh", "Z-exact"):
                    self.record(image, run_id=f"{ordinal}/D-0/{arm}", diagnostic="D-0", status="NOT_RUN_BUDGET",
                                evidence_kind="SCREEN_BOUND_UNCHECKED", dependency_status="SCREEN_ONLY",
                                interpretation={"hypothesis_id": "H-0", "assessment": "INCONCLUSIVE",
                                                "reason_code": "NOT_RUN_BUDGET"})
                continue
            data, meta, bounds1, info1, facts1 = self.load_cache(base_dir, image, preflight)
            crop = data["crop"]
            x_c, y = self.c_eval(crop)
            x = m["enc"].encode(crop).astype(np.int64)
            if not np.array_equal(x, x_c):
                raise AuditStop(f"{ordinal}: encoder parity failed in D-0")
            exact = cd.exact_forward(mats, x, m["relu"])
            if not np.array_equal(exact[-1]["z"], y):
                raise AuditStop(f"{ordinal}: C/Python parity failed in D-0")
            comps = meta["comps"]
            nominal = cd.margins_from_logits(y, exact[-1]["r"], target)
            lams = np.zeros((len(comps), sizes[-1]), dtype=np.int64)
            lams[:, target] = 1 << cd.SCALE_BITS
            lams[np.arange(len(comps)), comps] = -(1 << cd.SCALE_BITS)
            box0 = P.conv_interval_fast.propagate_box(m["net"], x, x)
            facts0 = P.box_ledger.facts_from_box(box0, sizes)
            arms = {}
            for arm in ("Z-inherited", "Z-fresh", "Z-exact"):
                with cd.ResourceMonitor() as res:
                    status, evidence, substitutions, error = "COMPLETED", "SCREEN_BOUND_UNCHECKED", {}, None
                    with self.XUState(P.cce, x):
                        if arm == "Z-inherited":
                            bounds, facts = bounds1, facts1
                            substitutions = {"input": "nominal singleton", "intermediate": "eps-1 baseline bounds"}
                            dependency = "SCREEN_ONLY"
                        elif arm == "Z-fresh":
                            try:
                                bounds, _ = P.v2.bounds_v2(mats, x, facts0)
                            except ValueError as exc:
                                bounds, error = None, str(exc)
                            facts = facts0
                            substitutions = {"input": "nominal singleton",
                                             "intermediate": "fresh propagate_box/facts_from_box/bounds_v2 at eps 0"}
                            dependency = "SCREEN_ONLY"
                        else:
                            bounds = []
                            for k in range(top):
                                r = exact[k]["r"]
                                saturated = r >= cd.Q_HIGH
                                bounds.append((r.copy(), r.copy(), saturated, np.zeros_like(saturated)))
                            facts = facts0
                            substitutions = {"input": "nominal singleton",
                                             "intermediate": "exact raw r_j from the nominal trace (not BoxTrace.pre_*)"}
                            dependency = "MEASURED_SINGLETON_BOUNDS"
                        if bounds is None:
                            status = "UNSUPPORTED"
                            scaled = None
                        else:
                            scaled, _, steps = P.v2.run_chain(lams, top, mats, bounds, x, record=True)
                            e = np.zeros((2, sizes[-1]), dtype=np.int64)
                            e[0, target], e[1, target] = 1 << cd.SCALE_BITS, -(1 << cd.SCALE_BITS)
                            t_scaled, _, _ = P.v2.run_chain(e, top, mats, bounds, x)
                            representable, reasons = self.representable(steps, bounds)
                            if arm == "Z-exact" and not representable:
                                evidence = "UNSUPPORTED_CERTIFICATE_FORM"
                if bounds is None:
                    self.record(image, res, run_id=f"{ordinal}/D-0/{arm}", parent_run_id=f"{ordinal}/D-P/baseline/eps1",
                                diagnostic="D-0", epsilon_raw_bytes=0, domain="D_box", status=status,
                                evidence_kind=evidence, dependency_status=dependency,
                                parameters={"arm": arm, "error": error},
                                interpretation={"hypothesis_id": "H-0", "assessment": "INCONCLUSIVE",
                                                "reason_code": "BOUNDS_UNSUPPORTED"})
                    continue
                raw_t_low = cd.ceil_div(int(t_scaled[0]), 1 << cd.SCALE_BITS)
                raw_t_high = (-int(t_scaled[1])) // (1 << cd.SCALE_BITS)
                raw_lbs = {c: cd.ceil_div(int(s), 1 << cd.SCALE_BITS) for c, s in zip(comps, scaled.tolist())}
                closure = cd.closure_edges(raw_lbs, facts[top]["hh"], facts[top]["hl"], facts[top]["lower_ok"],
                                           raw_t_low, raw_t_high, target)
                widths = [{"layer": k, "max": cd.dec((b[1] - b[0]).max()), "sum": cd.dec(int((b[1] - b[0]).sum())),
                           "nonzero": int(np.count_nonzero(b[1] - b[0]))} for k, b in enumerate(bounds)]
                rows, significant_any, negative = [], False, []
                for i, c in enumerate(comps):
                    nominal_raw = nominal[c]["raw"]
                    gap_scaled = nominal_raw * (1 << cd.SCALE_BITS) - int(scaled[i])
                    gap = nominal_raw - raw_lbs[c]
                    if gap_scaled < 0:
                        negative.append(c)
                    sig = cd.significant_singleton_gap(gap, nominal_raw)
                    significant_any |= sig and c in image["focus"]
                    rows.append({"competitor": c, "scaled_raw_lb": cd.dec(scaled[i]), "raw_lb": raw_lbs[c],
                                 "nominal_raw_margin": nominal_raw, "nominal_deployed_margin": nominal[c]["deployed"],
                                 "raw_gap": gap, "raw_gap_scaled": cd.dec(gap_scaled),
                                 "gap_fraction": cd.gap_fraction(gap, nominal_raw), "significant_gap": sig,
                                 "closure": closure[c]})
                if negative:
                    raise AuditStop(f"{ordinal} {arm}: bound exceeds concrete execution for competitors {negative}")
                run_id = f"{ordinal}/D-0/{arm}"
                rec = self.record(image, res, run_id=run_id, parent_run_id=f"{ordinal}/D-P/baseline/eps1",
                                  diagnostic="D-0", epsilon_raw_bytes=0, domain="D_box",
                                  region_sha256=sha256_json({"singleton": cd.sha256_array(x)}),
                                  assumption_context_sha256=sha256_json({"arm": arm, "substitutions": substitutions}),
                                  evidence_kind=evidence, dependency_status=dependency,
                                  parameters={"arm": arm},
                                  interpretation={"hypothesis_id": "H-0",
                                                  "assessment": "SUPPORTS" if significant_any else "NO_SUPPORT_WITHIN_PROBE",
                                                  "reason_code": "SIGNIFICANT_SINGLETON_GAP_ON_FOCUS" if significant_any
                                                  else "NO_SIGNIFICANT_SINGLETON_GAP_ON_FOCUS"})
                with out.open("a") as fh:
                    fh.write(json.dumps({**{k: rec[k] for k in ("run_id", "ordinal", "image_id", "target",
                                                                   "evidence_kind", "dependency_status")},
                                         "arm": arm, "nominal_witness_id": f"{ordinal}/nominal",
                                         "bound_widths_by_layer": widths, "source_substitutions": substitutions,
                                         "production_representable": representable,
                                         "representability_reasons": reasons, "raw_t": [raw_t_low, raw_t_high],
                                         "margins": rows}, sort_keys=True) + "\n")
                for row in rows:
                    e = row["closure"]
                    baseline_lb = int(meta["raw_lb"][str(row["competitor"])])
                    self.margin_row(run_id=run_id, ordinal=ordinal, domain="singleton", epsilon=0, target=target,
                                    competitor=row["competitor"], scale_bits=cd.SCALE_BITS,
                                    scaled_raw_lb=row["scaled_raw_lb"], raw_lb=row["raw_lb"],
                                    nominal_raw_margin=row["nominal_raw_margin"],
                                    nominal_deployed_margin=row["nominal_deployed_margin"],
                                    witness_id=f"{ordinal}/nominal", top_rule=(e["top"] or {}).get("rule"),
                                    top_reference=(e["top"] or {}).get("reference"), top_holds=e["top"] is not None,
                                    bottom_rule=(e["bottom"] or {}).get("rule"),
                                    bottom_reference=(e["bottom"] or {}).get("reference"),
                                    bottom_holds=e["bottom"] is not None, closed=e["closed"],
                                    baseline_raw_lb=baseline_lb, delta_raw_lb=row["raw_lb"] - baseline_lb,
                                    material_gain=None, evidence_kind=evidence, dependency_status=dependency)
                arms[arm] = bounds
            arms_cache[ordinal] = self.save_arm_cache(ordinal, arms)
            del data, bounds1, facts1, exact
        (self.out / "epsilon0.json").write_text(json.dumps({"arm_caches": arms_cache,
                                                            "source_manifest_sha256": self.manifest["source_manifest_sha256"]},
                                                           indent=1, sort_keys=True) + "\n")
        return None

    def save_arm_cache(self, ordinal, arms):
        arrays = {}
        for arm, bounds in arms.items():
            tag = arm.replace("-", "_")
            for k, (lo, hi, sat, clip) in enumerate(bounds):
                arrays.update({f"{tag}_{k}_lo": lo, f"{tag}_{k}_hi": hi, f"{tag}_{k}_sat": sat,
                               f"{tag}_{k}_clip": clip})
        path = self.out / "cache" / f"{ordinal}_eps0_arms.npz"
        self.np.savez(path, **arrays)
        return {"npz": str(path), "npz_sha256": sha256_file(path), "arms": sorted(arms)}

    def representable(self, steps, bounds):
        """Would the production class validator accept these ReLU coordinates? (Five checker classes,
        two-sided facts inside Q16, nosat upper fact <= Q_HIGH.)"""

        np, cd = self.np, self.cd
        reasons = []
        for s in steps:
            if s["kind"] != "relu":
                continue
            lo, hi, sat, clip = bounds[s["j"]]
            cls = self.P.v2.classify(s["a"], s["a_out"], sat, clip)
            if ((cls == 5) | (cls == 6)).any():
                reasons.append(f"layer {s['j']}: saturated_top/clipped class not in the production checker")
            two = (cls == 4)
            if (two & ((lo < cd.Q_LOW)[None, :] | (hi > cd.Q_HIGH)[None, :])).any():
                reasons.append(f"layer {s['j']}: two-sided fact outside Q16")
            if ((cls == 1) & (hi > cd.Q_HIGH)[None, :]).any():
                reasons.append(f"layer {s['j']}: nosat_upper fact above Q_HIGH")
        return not reasons, reasons

    # ------------------------------------------------------------ D-V

    def stage_violations(self, deadline):
        P, np, cd = self.P, self.np, self.cd
        base_dir, preflight = self.require("preflight")
        self.load_oracle(preflight)
        m = self.model()
        enc = m["enc"]
        trace_log = cd.GzipJsonl(self.out / "dv_proposals.jsonl.gz")
        retained, esbmc_budget, summary = [], {"used": 0, "cap": 3}, {}
        try:
            for image in self.images:
                ordinal, target = image["ordinal"], image["target"]
                data, meta, *_ = self.load_cache(base_dir, image, preflight)
                crop, blo, bhi = data["crop"], data["blo"], data["bhi"]
                groups = cd.AliasGroups(source=data["alias_source"],
                                        members=tuple(tuple(np.nonzero(data["alias_of"] == g)[0].tolist())
                                                      for g in range(data["alias_source"].size)),
                                        of_coordinate=data["alias_of"])
                nominal_values = crop.reshape(-1)[groups.source].astype(int).tolist()
                allowed = cd.group_allowed_values(crop, groups, self.eps)
                cache, violations = {}, []
                image_summary = {"focus": {}, "initial": {}, "violations": []}

                def evaluate_bytes(image_bytes, domain, label):
                    key = f"{domain}:{cd.sha256_array(np.asarray(image_bytes, dtype=np.uint8))}"
                    if key in cache:
                        return cache[key], False
                    x, y = self.c_eval(image_bytes)
                    x_py = enc.encode(image_bytes)
                    if not np.array_equal(x, x_py):
                        raise AuditStop(f"{ordinal}: encoder parity failed on {label}")
                    if not (np.all(x >= data["xl"]) and np.all(x <= data["xu"])):
                        raise AuditStop(f"{ordinal}: {label} encodes outside D_box")
                    margins = {c: int(y[target]) - int(y[c]) for c in range(len(y)) if c != target}
                    result = {"key": key, "x": x, "y": y, "margins": margins, "bytes": np.asarray(image_bytes, np.uint8),
                              "domain": domain}
                    cache[key] = result
                    bad = sorted(c for c, v in margins.items() if cd.deployed_violation(v))
                    if bad:
                        if image.get("control"):
                            raise AuditStop(f"control {ordinal}: deployed violation {bad} on {label}")
                        violations.append((result, bad, label))
                    return result, True

                def raw_margin(result, c):
                    exact = cd.exact_forward(m["mats"], result["x"], m["relu"])
                    return int(exact[-1]["r"][target]) - int(exact[-1]["r"][c])

                # step 1: nominal, corners, per-focus affine-minimizing D_box corner
                resized_nominal = enc.resize(crop)
                initial = {"nominal": (crop, "D_crop"), "box_low": (blo, "D_box"), "box_high": (bhi, "D_box")}
                for c in image["focus"]:
                    a = data[f"final_{c}"].reshape(resized_nominal.shape)
                    corner = np.where(a > 0, blo, np.where(a < 0, bhi, resized_nominal)).astype(np.uint8)
                    initial[f"affine_corner_c{c}"] = (corner, "D_box")
                for name, (image_bytes, domain) in initial.items():
                    result, new = evaluate_bytes(image_bytes, domain, name)
                    lift = None
                    if domain == "D_crop":
                        result["crop"], lift = image_bytes, True
                    else:
                        lifted = cd.lift_resized(image_bytes, groups, crop)
                        lift = lifted is not None
                        if lifted is not None:
                            check = cd.in_crop_domain(lifted, crop, groups, self.eps)
                            lift = all(check.values()) and np.array_equal(enc.resize(lifted), image_bytes)
                            if lift:
                                result["crop"] = lifted
                    image_summary["initial"][name] = {"key": result["key"], "domain": domain, "crop_lift": lift,
                                                      "min_deployed_margin": min(result["margins"].values()),
                                                      "new_evaluation": new}
                    retained.append(self.retain(image, f"{ordinal}/{name}", result, lift, groups, crop,
                                                kind="initial"))

                # steps 2-7: fixed search per focus competitor, numeric order
                for c in sorted(image["focus"]):
                    a = data[f"final_{c}"]
                    group_coefficient = [sum(int(a[i]) for i in members) for members in groups.members]

                    def evaluate(values, c=c):
                        candidate = cd.crop_from_assignment(crop, groups, values)
                        result, new = evaluate_bytes(candidate, "D_crop", f"search c{c}")
                        result.setdefault("crop", candidate)
                        raw = result.setdefault("raw_margins", {}).get(c)
                        if raw is None:
                            raw = raw_margin(result, c)
                            result["raw_margins"][c] = raw
                        return cd.Evaluation(deployed=result["margins"][c], raw=raw, margins=result["margins"],
                                             key=result["key"], new=new)

                    def log(entry, c=c):
                        trace_log.write({"ordinal": ordinal, "competitor": c, **entry})

                    seed = lambda r, c=c: int(self.config["seed_base"]) + 1000 * int(ordinal) + 10 * c + r
                    with cd.ResourceMonitor() as res:
                        record = cd.search_focus(groups, allowed, nominal_values, group_coefficient, evaluate, seed,
                                                 deadline=deadline, restarts=int(self.config["attack_restarts"]),
                                                 per_restart=int(self.config["attack_proposals_per_restart"]), log=log)
                    image_summary["focus"][str(c)] = {k: v for k, v in record.items() if k != "starts"} | \
                        {"starts": record["starts"]}
                    best = record["best"]
                    witness_id = None
                    if best is not None:
                        result = cache[best["key"]]
                        witness_id = f"{ordinal}/best_c{c}"
                        retained.append(self.retain(image, witness_id, result, True, groups, crop, kind="focus_best",
                                                    search={"seed": seed(best["restart"]), "restart": best["restart"],
                                                            "objective": [best["deployed"], best["raw"]]}))
                    found = record["violation"] is not None
                    assessment = ("SUPPORTS" if found else "NO_SUPPORT_WITHIN_PROBE"
                                  if record["status"] == "COMPLETED" else "INCONCLUSIVE")
                    run_id = f"{ordinal}/D-V/c{c}"
                    self.record(image, res, run_id=run_id, parent_run_id=f"{ordinal}/D-P/baseline/eps1",
                                diagnostic="D-V", competitor=c, epsilon_raw_bytes=self.eps, domain="D_crop",
                                status=record["status"], evidence_kind="MEASURED_EXACT_EXECUTION",
                                dependency_status="NONE",
                                parameters={"seeds": record["seeds"], "proposals": record["proposals"],
                                            "evaluations": record["evaluations"],
                                            "incomplete_starts": record["incomplete_starts"],
                                            "unexecuted_proposals": record["unexecuted_proposals"],
                                            "numpy_version": np.__version__},
                                artifacts={"best_witness": witness_id},
                                interpretation={"hypothesis_id": "H-V", "assessment": assessment,
                                                "reason_code": "LIFTED_VIOLATION_FOUND" if found else
                                                "NO_WITNESS_FOUND" if record["status"] == "COMPLETED"
                                                else record["status"]})
                    raw_lb = int(meta["raw_lb"][str(c)])
                    self.margin_row(run_id=run_id, ordinal=ordinal, domain="D_crop", epsilon=self.eps, target=target,
                                    competitor=c, scale_bits=cd.SCALE_BITS, raw_lb=raw_lb,
                                    best_seen_raw_margin=None if best is None else best["raw"],
                                    best_seen_deployed_margin=None if best is None else best["deployed"],
                                    witness_id=witness_id, baseline_raw_lb=raw_lb,
                                    evidence_kind="MEASURED_EXACT_EXECUTION", dependency_status="NONE")
                    if best is not None and best["raw"] < raw_lb:
                        raise AuditStop(f"{ordinal} c{c}: concrete raw margin {best['raw']} below the screened "
                                        f"raw lower bound {raw_lb}")
                # every distinct violation: retain, replay, at most one ESBMC replay per image
                for n, (result, bad, label) in enumerate(violations):
                    domain = result["domain"]
                    lifted = result.get("crop") is not None
                    witness_id = f"{ordinal}/violation_{n}"
                    retained.append(self.retain(image, witness_id, result, lifted, groups, crop, kind="violation"))
                    replay = self.replay_violation(image, witness_id, result, bad,
                                                   run_esbmc=n == 0 and esbmc_budget["used"] < esbmc_budget["cap"],
                                                   deadline=deadline)
                    if replay.get("esbmc_ran"):
                        esbmc_budget["used"] += 1
                    image_summary["violations"].append({"witness_id": witness_id, "competitors": bad, "label": label,
                                                        "domain": domain, "crop_lift": lifted, **replay})
                summary[ordinal] = image_summary
        finally:
            trace_log.close()
        (self.out / "violations.json").write_text(json.dumps({"images": summary, "retained": retained,
                                                             "esbmc_replays": esbmc_budget,
                                                             "source_manifest_sha256": self.manifest["source_manifest_sha256"]},
                                                            indent=1, sort_keys=True, default=str) + "\n")
        return None

    def retain(self, image, witness_id, result, lift, groups, crop, kind, search=None):
        np, cd = self.np, self.cd
        directory = self.out / "witnesses" / witness_id.replace("/", "_")
        if directory.exists():
            return json.loads((directory / "witness.json").read_text())
        directory.mkdir(parents=True)
        resized = self.model()["enc"].resize(result["crop"]) if result.get("crop") is not None else result["bytes"]
        np.save(directory / "bytes.npy", np.asarray(resized, dtype=np.uint8), allow_pickle=False)
        np.save(directory / "encoded.npy", result["x"], allow_pickle=False)
        if result.get("crop") is not None:
            np.save(directory / "crop.npy", np.asarray(result["crop"], dtype=np.uint8), allow_pickle=False)
        exact = cd.exact_forward(self.model()["mats"], result["x"], self.model()["relu"])
        target = image["target"]
        margins = cd.margins_from_logits(result["y"], exact[-1]["r"], target)
        worst = min(margins, key=lambda c: (margins[c]["deployed"], margins[c]["raw"]))
        crop_check = (cd.in_crop_domain(result["crop"], crop, groups, self.eps)
                      if result.get("crop") is not None else None)
        x = result["x"]
        meta = {"witness_id": witness_id, "kind": kind, "ordinal": image["ordinal"], "image_id": image["record_id"],
                "target": target, "domain": "D_crop" if lift else "D_box",
                "search_seed": (search or {}).get("seed"), "restart": (search or {}).get("restart"),
                "proposal_index": None, "search_objective": (search or {}).get("objective"),
                "bytes_path": str(directory / "bytes.npy"), "bytes_sha256": sha256_file(directory / "bytes.npy"),
                "crop_path": str(directory / "crop.npy") if result.get("crop") is not None else None,
                "crop_sha256": sha256_file(directory / "crop.npy") if result.get("crop") is not None else None,
                "encoded_path": str(directory / "encoded.npy"), "encoded_sha256": sha256_file(directory / "encoded.npy"),
                "byte_bounds_satisfied": True, "source_bounds_satisfied": None if crop_check is None
                else crop_check["source_bounds_satisfied"],
                "alias_constraints_satisfied": bool(lift), "crop_lift_exists": bool(lift),
                "raw_logits": [cd.dec(v) for v in exact[-1]["r"]], "deployed_logits": [int(v) for v in result["y"]],
                "worst_competitor": worst, "raw_margin": margins[worst]["raw"],
                "deployed_margin": margins[worst]["deployed"], "is_tie": margins[worst]["deployed"] == 0,
                "changed_byte_count": int(np.count_nonzero(np.asarray(resized, np.int64) !=
                                                           self.model()["enc"].resize(crop).astype(np.int64))),
                "linf_raw": int(np.abs(np.asarray(resized, np.int64) -
                                       self.model()["enc"].resize(crop).astype(np.int64)).max()),
                "O0_replay": None, "O2_replay": None, "python_replay": None, "esbmc_replay": None,
                "violated_assertion_id": None, "status_label": "CROP_WITNESS" if lift else "BOX_ONLY_WITNESS",
                "evidence_kind": "MEASURED_EXACT_EXECUTION"}
        (directory / "witness.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        return meta

    def replay_violation(self, image, witness_id, result, competitors, run_esbmc, deadline):
        np, cd = self.np, self.cd
        directory = self.out / "witnesses" / witness_id.replace("/", "_")
        meta = json.loads((directory / "witness.json").read_text())
        image_bytes = result.get("crop") if result.get("crop") is not None else result["bytes"]
        x0, y0 = self.c_eval(image_bytes, "O0")
        x2, y2 = self.c_eval(image_bytes, "O2")
        exact = cd.exact_forward(self.model()["mats"], x2, self.model()["relu"])
        c = competitors[0]
        t = image["target"]
        meta["O0_replay"] = {"deployed_margin": int(y0[t]) - int(y0[c]), "equal_O2": bool(np.array_equal(y0, y2))}
        meta["O2_replay"] = {"deployed_margin": int(y2[t]) - int(y2[c])}
        meta["python_replay"] = {"deployed_margin": int(exact[-1]["z"][t]) - int(exact[-1]["z"][c]),
                                 "equal_C": bool(np.array_equal(exact[-1]["z"], y2))}
        if not (meta["O0_replay"]["equal_O2"] and meta["python_replay"]["equal_C"]):
            raise AuditStop(f"{witness_id}: replay parity failed")
        out = {"replayed": True, "esbmc_ran": False}
        if run_esbmc and time.monotonic() < deadline:
            meta["esbmc_replay"] = self.esbmc_witness_replay(image, witness_id, image_bytes, c)
            meta["violated_assertion_id"] = meta["esbmc_replay"].get("violated_assertion_id")
            out["esbmc_ran"] = True
            out["esbmc"] = meta["esbmc_replay"]
        (directory / "witness.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
        return out

    # ------------------------------------------------------------ ESBMC (diagnostic runner)

    def esbmc_runner(self, timeout, memlimit):
        E = self.P.esbmc

        class DiagnosticRunner(E.ESBMCRunner):
            """paper-z3 without --result-only, so the first run keeps the trace."""

            def build_command(self, c_file, unwind, profile):
                return tuple(a for a in super().build_command(c_file, unwind, profile) if a != "--result-only")

        return DiagnosticRunner(E.ESBMCConfig(timeout_seconds=int(timeout), memlimit=memlimit,
                                              default_profile="paper-z3"))

    def run_obligation(self, run_id, source, label, timeout, memlimit):
        directory = self.out / "obligations" / run_id.replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "harness.c"
        path.write_text(source)
        runner = self.esbmc_runner(timeout, memlimit)
        result = runner.run_file(path, "paper-z3", extract_counterexample=False)
        # ESBMC prints the counterexample trace on stderr; parse both complete logs
        log = "\n".join(Path(p).read_text(errors="replace") for p in (result.stdout_log_path, result.stderr_log_path)
                        if p and Path(p).exists()) or (result.stdout + result.stderr)
        parsed = self.cd.parse_violation(log, label) if result.status == "FAILED" else {"failed_claim_kind": None}
        record = {"harness": str(path), "harness_sha256": sha256_file(path), "command": list(result.command),
                  "effective_profile": "diagnostic-paper-z3-with-trace", "status": result.status,
                  "elapsed_seconds": result.elapsed_seconds, "peak_memory_bytes": result.peak_memory_bytes,
                  "cpu_time_seconds": result.cpu_time_seconds, "stdout_log": result.stdout_log_path,
                  "stderr_log": result.stderr_log_path,
                  "target_assertion": label, **parsed,
                  "violated_assertion_id": label if parsed.get("failed_claim_kind") == "TARGET" else None}
        (directory / "status.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
        return record

    def solver_controls(self):
        """Spec section 9 controls: parsing of status and assertion identity, not production soundness."""

        if getattr(self, "_controls", None) is not None:
            return self._controls
        cd = self.cd
        results, failed = [], []
        for control in cd.solver_control_harnesses():
            rec = self.run_obligation(f"controls/{control['name']}", control["source"], control["label"], 10, "4g")
            ok = rec["status"] == control["expected"] and (control["expected"] != "FAILED" or
                                                           rec["failed_claim_kind"] == "TARGET")
            results.append({"name": control["name"], "expected": control["expected"], "status": rec["status"],
                            "failed_claim_kind": rec["failed_claim_kind"], "passed": ok,
                            "elapsed_seconds": rec["elapsed_seconds"]})
            if not ok:
                failed.append(control["name"])
        self._controls = {"results": results, "passed": not failed}
        (self.out / "solver_controls.json").write_text(json.dumps(self._controls, indent=1, sort_keys=True) + "\n")
        if failed:
            raise AuditStop(f"solver controls failed: {failed}")
        return self._controls

    def esbmc_witness_replay(self, image, witness_id, image_bytes, competitor):
        cd = self.cd
        self.solver_controls()
        image_bytes = self.np.asarray(image_bytes, dtype=self.np.uint8)
        run_id = f"{witness_id}/esbmc_replay"
        label = f"diag/{run_id}/target_dominance"
        source, inventory = cd.witness_replay_harness(self.model()["source"], image_bytes, image["target"],
                                                      competitor, label)
        rec = self.run_obligation(run_id, source, label, int(self.config["full_margin_timeout_seconds"]),
                                  f"{int(self.config['solver_memory_gib'])}g")
        rec["loop_inventory"] = inventory
        if rec["status"] == "VERIFIED":
            raise AuditStop(f"{witness_id}: ESBMC VERIFIED dominance that exact execution violates")
        self.record(image, run_id=run_id, parent_run_id=witness_id, diagnostic="D-V", competitor=competitor,
                    epsilon_raw_bytes=self.eps, domain="fixed_input", evidence_kind="ESBMC_CHECKED_OBLIGATION",
                    solver_status=rec["status"] if rec["status"] in cd.SOLVER_STATUSES else "UNKNOWN",
                    dependency_status="NONE",
                    metrics={"solver_elapsed_sum_seconds": rec["elapsed_seconds"], "executed_solver_calls": 1,
                             "wall_seconds": rec["elapsed_seconds"]},
                    artifacts={"harness": rec["harness"]},
                    interpretation={"hypothesis_id": "H-V",
                                    "assessment": "SUPPORTS" if rec.get("failed_claim_kind") == "TARGET" else "INCONCLUSIVE",
                                    "reason_code": f"ESBMC_{rec['status']}_{rec.get('failed_claim_kind')}"})
        return rec

    # ------------------------------------------------------------ D-T

    def stage_trace(self, deadline):
        P, np, cd = self.P, self.np, self.cd
        base_dir, preflight = self.require("preflight")
        _, dv = self.require("violations.json")
        _, d0 = self.require("epsilon0.json")
        self.load_oracle(preflight)
        m = self.model()
        mats, sizes, biases = m["mats"], m["sizes"], m["biases"]
        top = len(mats) - 1
        steps_out = cd.GzipJsonl(self.out / "steps.jsonl.gz")
        coords_out = cd.GzipJsonl(self.out / "coordinates.jsonl.gz")
        bounds_out = cd.GzipJsonl(self.out / "bounds.jsonl.gz")
        rankings, failures = {}, []
        try:
            for image in self.images:
                ordinal, target = image["ordinal"], image["target"]
                if time.monotonic() > deadline:
                    self.record(image, run_id=f"{ordinal}/D-T", diagnostic="D-T", status="NOT_RUN_BUDGET",
                                evidence_kind="MEASURED_EXACT_EXECUTION", dependency_status="SCREEN_ONLY",
                                interpretation={"hypothesis_id": "H-T", "assessment": "INCONCLUSIVE",
                                                "reason_code": "NOT_RUN_BUDGET"})
                    continue
                data, meta, bounds, info, facts = self.load_cache(base_dir, image, preflight)
                comps = meta["comps"]
                witnesses = [w for w in dv["retained"] if w["ordinal"] == ordinal]
                with cd.ResourceMonitor() as res:
                    lams = np.zeros((len(image["focus"]), sizes[-1]), dtype=np.int64)
                    lams[:, target] = 1 << cd.SCALE_BITS
                    for i, c in enumerate(image["focus"]):
                        lams[i, c] = -(1 << cd.SCALE_BITS)
                    with self.XUState(P.cce, data["xu"]):
                        scaled, _, steps = P.v2.run_chain(lams, top, mats, bounds, data["xl"], record=True)
                    for i, c in enumerate(image["focus"]):
                        if int(scaled[i]) != int(meta["scaled"][comps.index(c)]):
                            raise AuditStop(f"{ordinal} c{c}: recomputed chain differs from the cached baseline")
                    traces = {}
                    for w in witnesses:
                        x = np.load(w["encoded_path"], allow_pickle=False)
                        if sha256_file(w["encoded_path"]) != w["encoded_sha256"]:
                            raise AuditStop(f"{w['witness_id']}: witness file changed")
                        if not (np.all(x >= data["xl"]) and np.all(x <= data["xu"])):
                            continue           # not a valid witness for the eps-1 chain
                        traces[w["witness_id"]] = (x, cd.exact_forward(mats, x, m["relu"]))
                    per_neuron = {}
                    for i, c in enumerate(image["focus"]):
                        chain_id = f"chain/margin/{target}-{c}"
                        for wid, (x, trace) in traces.items():
                            run_id = f"{ordinal}/D-T/c{c}/{wid.split('/')[-1]}"
                            sink = self.coordinate_sink(coords_out, run_id, wid, chain_id, steps, i, bounds, info, facts)
                            try:
                                acc = cd.account_chain(steps, i, lams[i], trace, x, data["xl"], data["xu"], biases,
                                                       int(scaled[i]), sink)
                            except cd.ImplementationError as exc:
                                failures.append({"run_id": run_id, "error": str(exc)})
                                self.record(image, run_id=run_id, parent_run_id=f"{ordinal}/D-P/baseline/eps1",
                                            diagnostic="D-T", competitor=c, epsilon_raw_bytes=self.eps,
                                            domain="D_box", status="COMPLETED",
                                            evidence_kind="MEASURED_EXACT_EXECUTION", dependency_status="SCREEN_ONLY",
                                            interpretation={"hypothesis_id": "H-T",
                                                            "assessment": "IMPLEMENTATION_ERROR",
                                                            "reason_code": "ACCOUNTING_IDENTITY_FAILED"})
                                raise AuditStop(f"{run_id}: {exc}")
                            self.write_steps(steps_out, run_id, wid, chain_id, acc, steps, i, mats)
                            for rec in acc["steps"]:
                                for neuron, slack in rec.pop("_per_coordinate", {}).items():
                                    key = (c, rec["layer"], neuron)
                                    per_neuron[key] = max(per_neuron.get(key, 0), slack)
                            self.record(image, run_id=run_id, parent_run_id=f"{ordinal}/D-P/baseline/eps1",
                                        diagnostic="D-T", competitor=c, epsilon_raw_bytes=self.eps, domain="D_box",
                                        evidence_kind="MEASURED_EXACT_EXECUTION", dependency_status="SCREEN_ONLY",
                                        parameters={"witness_id": wid, "chain_id": chain_id,
                                                    "total_slack": acc["total_slack"],
                                                    "slack_by_kind": self.slack_by_kind(acc),
                                                    "final_rounding_credit": acc["final_rounding_credit"]},
                                        interpretation={"hypothesis_id": "H-T", "assessment": "INCONCLUSIVE",
                                                        "reason_code": "CONCENTRATION_CRITERION_NOT_SPECIFIED"})
                    rankings[ordinal] = self.rank(image, per_neuron, steps, bounds, lams)
                    self.write_bounds(bounds_out, f"{ordinal}/D-T/bounds/eps1", data, bounds, info, facts, traces)
                    self.trace_singleton_arms(image, d0, data, steps_out, coords_out, failures)
                self.record(image, res, run_id=f"{ordinal}/D-T", diagnostic="D-T", epsilon_raw_bytes=self.eps,
                            domain="D_box", evidence_kind="MEASURED_EXACT_EXECUTION", dependency_status="SCREEN_ONLY",
                            parameters={"witnesses": sorted(traces), "focus": image["focus"],
                                        "truncation_executed": False,
                                        "truncation_note": "crown_chain_export_v2.run_chain performs no top-k "
                                                           "truncation; affine_export.truncate is not called"},
                            metric_scope="image_trace_stage",
                            interpretation={"hypothesis_id": "H-T", "assessment": "INCONCLUSIVE",
                                            "reason_code": "CONCENTRATION_CRITERION_NOT_SPECIFIED"})
        finally:
            for writer in (steps_out, coords_out, bounds_out):
                writer.close()
        (self.out / "rankings.json").write_text(json.dumps(rankings, indent=1, sort_keys=True) + "\n")
        (self.out / "trace.json").write_text(json.dumps({"failures": failures,
                                                        "source_manifest_sha256": self.manifest["source_manifest_sha256"]},
                                                       indent=1) + "\n")
        return None

    def slack_by_kind(self, acc):
        out = {"affine_residual_endpoint_num": 0, "affine_rounding_envelope_num": 0, "affine_constant_floor_num": 0,
               "affine": 0, "relu": 0, "input": 0}
        for rec in acc["steps"]:
            if rec["kind"] == "affine":
                out["affine"] += int(rec["slack"])
                out["affine_residual_endpoint_num"] += int(rec["residual_endpoint_slack_num"])
                out["affine_rounding_envelope_num"] += int(rec["rounding_envelope_slack_num"])
                out["affine_constant_floor_num"] += int(rec["constant_floor_slack_num"])
            elif rec["kind"] == "relu":
                out["relu"] += int(rec["slack"])
            else:
                out["input"] += int(rec["slack"])
        total = int(acc["total_slack"])
        return {k: str(v) for k, v in out.items()} | {
            "shares": {k: self.cd.rational(out[k], total) for k in ("affine", "relu", "input")} if total else None}

    def coordinate_sink(self, writer, run_id, wid, chain_id, steps, row, bounds, info, facts):
        np, cd = self.np, self.cd
        classify = self.P.v2.classify

        def sink(entry):
            layer, neuron = entry.get("layer"), entry["neuron"]
            record = {"run_id": run_id, "witness_id": wid, "chain_id": chain_id,
                      "step_id": f"{chain_id}/s{entry['step_index']}", **{k: v for k, v in entry.items()
                                                                          if k != "step_index"}}
            if layer is not None and 0 <= layer < len(bounds):
                lo, hi, sat, clip = bounds[layer]
                record.update(lower=cd.dec(lo[neuron]), upper=cd.dec(hi[neuron]),
                              width=cd.dec(int(hi[neuron]) - int(lo[neuron])),
                              lower_source="box" if bool(info[layer]["lower_box"][neuron]) else "chain",
                              upper_source="chain" if bool(info[layer]["upper_chain"][neuron]) else "box",
                              lower_evidence="SCREEN_BOUND_UNCHECKED", upper_evidence="SCREEN_BOUND_UNCHECKED",
                              clipped=bool(clip[neuron]), saturated=bool(sat[neuron]))
                if entry.get("d") is not None:
                    step = steps[entry["step_index"]]
                    cls = int(classify(step["a"][row][neuron:neuron + 1], step["a_out"][row][neuron:neuron + 1],
                                       sat[neuron:neuron + 1], clip[neuron:neuron + 1])[0])
                    record["relu_class"] = self.P.v2.CLASSES[cls] if cls >= 0 else None
            writer.write(record)

        return sink

    def write_steps(self, writer, run_id, wid, chain_id, acc, steps, row, mats):
        np, cd = self.np, self.cd
        for rec in acc["steps"]:
            s = steps[rec["step_index"]]
            extra = {}
            if s["kind"] == "affine":
                w = mats[s["j"]][0]
                a = np.asarray(s["a"][row], dtype=np.int64)
                touched = np.unique(w[np.nonzero(a)[0]].indices)
                g_nonzero = np.nonzero(np.asarray(s["a_out"][row]) * (1 << cd.F) + np.asarray(s["R"][row]))[0]
                extra = {"csr_touched_columns": int(touched.size),
                         "zero_cancelled_columns": int(np.setdiff1d(touched, g_nonzero).size),
                         "g_envelope": cd.dec(s["g_env"][row]) if "g_env" in s else None,
                         "max_product": cd.dec(s["max_prod"][row]) if "max_prod" in s else None}
            writer.write({"run_id": run_id, "witness_id": wid, "chain_id": chain_id,
                          "step_id": f"{chain_id}/s{rec['step_index']}",
                          **{k: v for k, v in rec.items() if not k.startswith("_")}, **extra,
                          "accounting_identity_passed": acc["accounting_identity_passed"]})

    def rank(self, image, per_neuron, steps, bounds, lams):
        """Spec section 7 ranking: max observed per-coordinate ReLU slack desc, then |a|*(u-l) desc,
        layer desc, neuron asc. Only finite, representable bounds qualify."""

        cd = self.cd
        out = {}
        for i, c in enumerate(image["focus"]):
            relu_a = {s["j"]: s["a"][i] for s in steps if s["kind"] == "relu"}
            entries = []
            for (cc, layer, neuron), slack in per_neuron.items():
                if cc != c:
                    continue
                lo, hi, sat, clip = bounds[layer]
                if bool(sat[neuron]) or bool(clip[neuron]) or int(lo[neuron]) < cd.Q_LOW or int(hi[neuron]) > cd.Q_HIGH:
                    continue
                width_term = abs(int(relu_a[layer][neuron])) * (int(hi[neuron]) - int(lo[neuron]))
                entries.append((slack, width_term, layer, neuron))
            entries.sort(key=lambda e: (-e[0], -e[1], -e[2], e[3]))
            fmt = lambda e: {"layer": e[2], "neuron": e[3], "max_relu_slack": cd.dec(e[0]),
                             "abs_a_times_width": cd.dec(e[1])}
            out[str(c)] = {"top16_global": [fmt(e) for e in entries[:16]],
                           "top16_per_layer": {str(L): [fmt(e) for e in entries if e[2] == L][:16]
                                               for L in sorted({e[2] for e in entries})},
                           "candidates": len(entries)}
        return out

    def write_bounds(self, writer, run_id, data, bounds, info, facts, traces):
        np, cd = self.np, self.cd
        nominal = next((t for wid, t in traces.items() if wid.endswith("/nominal")), None)
        box_lo = [np.asarray(f["hl"]) for f in facts]
        for k, (lo, hi, sat, clip) in enumerate(bounds):
            raws = [t[1][k]["r"] for t in traces.values()]
            acts = [t[1][k]["h"] for t in traces.values()]
            raw_min, raw_max = (np.min(raws, axis=0), np.max(raws, axis=0)) if raws else (None, None)
            act_min, act_max = (np.min(acts, axis=0), np.max(acts, axis=0)) if acts else (None, None)
            stable = np.asarray(facts[k]["stable"])
            for n in range(lo.size):
                writer.write({
                    "run_id": run_id, "layer": k, "neuron": n,
                    "box_postclamp_pre_relu_low": cd.dec(data[f"p{k}_lo"][n]),
                    "box_postclamp_pre_relu_high": cd.dec(data[f"p{k}_hi"][n]),
                    "box_activation_low": cd.dec(facts[k]["hl"][n]), "box_activation_high": cd.dec(facts[k]["hh"][n]),
                    "chain_raw_low": cd.dec(info[k]["lo_chain"][n]), "chain_raw_high": cd.dec(info[k]["hi_chain"][n]),
                    "selected_raw_low": cd.dec(lo[n]), "selected_raw_high": cd.dec(hi[n]),
                    "lower_box": bool(info[k]["lower_box"][n]), "upper_chain": bool(info[k]["upper_chain"][n]),
                    "raw_source_validity": "SCREEN_BOUND_UNCHECKED", "selected_width": cd.dec(int(hi[n]) - int(lo[n])),
                    "stable_class": int(stable[n]), "clipped": bool(clip[n]), "saturated": bool(sat[n]),
                    "tighter_chain_low_unselected": bool(int(info[k]["lo_chain"][n]) > int(lo[n])),
                    "tighter_chain_high_unselected": bool(int(info[k]["hi_chain"][n]) < int(hi[n])),
                    "stable_inactive_lower_unavailable": bool(stable[n] < 0),
                    "sampled_raw_min": None if raw_min is None else cd.dec(raw_min[n]),
                    "sampled_raw_max": None if raw_max is None else cd.dec(raw_max[n]),
                    "sampled_activation_min": None if act_min is None else cd.dec(act_min[n]),
                    "sampled_activation_max": None if act_max is None else cd.dec(act_max[n]),
                    "sample_count": len(raws),
                    "lower_slack_at_nominal": None if nominal is None else cd.dec(int(nominal[1][k]["r"][n]) - int(lo[n])),
                    "upper_slack_at_nominal": None if nominal is None else cd.dec(int(hi[n]) - int(nominal[1][k]["r"][n])),
                    "note": "box_postclamp_pre_relu_* are BoxTrace.pre_* (clamped, not raw r); box_activation_* "
                            "are the ReLU'd facts. Sample extrema are observations, never sound bounds."})

    def trace_singleton_arms(self, image, d0, data, steps_out, coords_out, failures):
        """Accounting for the primary margin of each D-0 arm at the nominal witness."""

        P, np, cd = self.P, self.np, self.cd
        entry = d0["arm_caches"].get(image["ordinal"])
        if entry is None:
            return
        if sha256_file(entry["npz"]) != entry["npz_sha256"]:
            raise AuditStop("eps-0 arm cache changed")
        arrays = dict(np.load(entry["npz"], allow_pickle=False))
        m = self.model()
        mats, sizes = m["mats"], m["sizes"]
        top = len(mats) - 1
        x = m["enc"].encode(data["crop"]).astype(np.int64)
        trace = cd.exact_forward(mats, x, m["relu"])
        c, target = image["primary"], image["target"]
        lam = np.zeros((1, sizes[-1]), dtype=np.int64)
        lam[0, target], lam[0, c] = 1 << cd.SCALE_BITS, -(1 << cd.SCALE_BITS)
        for arm in entry["arms"]:
            tag = arm.replace("-", "_")
            bounds = [(arrays[f"{tag}_{k}_lo"], arrays[f"{tag}_{k}_hi"], arrays[f"{tag}_{k}_sat"],
                       arrays[f"{tag}_{k}_clip"]) for k in range(top)]
            with self.XUState(P.cce, x):
                scaled, _, steps = P.v2.run_chain(lam, top, mats, bounds, x, record=True)
            run_id = f"{image['ordinal']}/D-T/{arm}/c{c}"
            chain_id = f"chain/margin/{target}-{c}"
            try:
                acc = cd.account_chain(steps, 0, lam[0], trace, x, x, x, m["biases"], int(scaled[0]))
            except cd.ImplementationError as exc:
                failures.append({"run_id": run_id, "error": str(exc)})
                raise AuditStop(f"{run_id}: {exc}")
            self.write_steps(steps_out, run_id, f"{image['ordinal']}/nominal", chain_id, acc, steps, 0, mats)
            self.record(image, run_id=run_id, parent_run_id=f"{image['ordinal']}/D-0/{arm}", diagnostic="D-T",
                        competitor=c, epsilon_raw_bytes=0, domain="singleton",
                        evidence_kind="MEASURED_EXACT_EXECUTION", dependency_status="SCREEN_ONLY",
                        parameters={"arm": arm, "total_slack": acc["total_slack"],
                                    "slack_by_kind": self.slack_by_kind(acc)},
                        interpretation={"hypothesis_id": "H-T", "assessment": "INCONCLUSIVE",
                                        "reason_code": "CONCENTRATION_CRITERION_NOT_SPECIFIED"})

    # ------------------------------------------------------------ I-B

    def stage_bounds(self, deadline):
        """Spec section 8 I-B on each primary margin: one intermediate endpoint side at a time.

        Hypothetical arm (COUNTERFACTUAL_ONLY): midpoint endpoint injected, margin rerun.
        Attained arm (SCREEN_BOUND_UNCHECKED): unit-root chain for that side under the baseline policy
        and I-S one-coordinate alternatives; a strictly tighter integer endpoint is applied alone.
        Exact prefix probe: one per image, gated by known witnesses and the structural size caps."""

        P, np, cd = self.P, self.np, self.cd
        v2 = P.v2
        base_dir, preflight = self.require("preflight")
        _, dv = self.require("violations.json")
        _, rankings = self.require("rankings.json")
        m = self.model()
        mats, sizes = m["mats"], m["sizes"]
        top = len(mats) - 1
        per_side = int(self.config["slope_coordinates_per_root"])
        per_primary = int(self.config["bound_coordinates_per_primary"])
        out_path = self.out / "interventions.jsonl"
        summary = {"images": {}, "source_manifest_sha256": self.manifest["source_manifest_sha256"],
                   "override_mechanism": "in-process wrapper of crown_chain_export_v2.relu_choice "
                                         "(crown_diagnostics.SlopeOverride); finder file unchanged"}

        def write_intervention(rec):
            with out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")

        for image in self.images:
            o, target, c = image["ordinal"], image["target"], image["primary"]
            control = bool(image.get("control"))
            img_summary = summary["images"].setdefault(o, {"primary": c, "targets": [], "prefix": None})
            if time.monotonic() > deadline or not self.wait_for_memory(deadline):
                self.record(image, run_id=f"{o}/I-B", diagnostic="I-B", competitor=c, status="NOT_RUN_BUDGET",
                            evidence_kind="SCREEN_BOUND_UNCHECKED", dependency_status="SCREEN_ONLY",
                            interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                            "reason_code": "NOT_RUN_BUDGET"})
                img_summary["status"] = "NOT_RUN_BUDGET"
                continue
            data, meta, bounds, info, facts = self.load_cache(base_dir, image, preflight)
            comps = meta["comps"]
            xl, xu = data["xl"], data["xu"]
            base_scaled = int(meta["scaled"][comps.index(c)])
            base_raw = cd.ceil_div(base_scaled, 1 << cd.SCALE_BITS)
            threshold = cd.material_gain_threshold(base_raw, control)
            closure = meta["closure"][str(c)]
            lam = np.zeros((1, sizes[-1]), dtype=np.int64)
            lam[0, target], lam[0, c] = 1 << cd.SCALE_BITS, -(1 << cd.SCALE_BITS)
            x_nom = m["enc"].encode(data["crop"]).astype(np.int64)
            nominal = cd.exact_forward(mats, x_nom, m["relu"])
            traces = {f"{o}/nominal": (x_nom, nominal)}
            for w in dv["retained"]:
                if w["ordinal"] != o:
                    continue
                if sha256_file(w["encoded_path"]) != w["encoded_sha256"]:
                    raise AuditStop(f"{w['witness_id']}: witness file changed")
                x = np.load(w["encoded_path"], allow_pickle=False)
                if np.all(x >= xl) and np.all(x <= xu):
                    traces[w["witness_id"]] = (x, cd.exact_forward(mats, x, m["relu"]))

            def margin(bnds, record=False):
                with self.XUState(P.cce, xu):
                    scaled, terms, steps = v2.run_chain(lam, top, mats, bnds, xl, record=record)
                return int(scaled[0]), int(terms[0]), steps

            def relu_classes(steps, bnds):
                out = {}
                for s in steps:
                    if s["kind"] == "relu":
                        lo, hi, sat, clip = bnds[s["j"]]
                        out[s["j"]] = v2.classify(s["a"][0:1], s["a_out"][0:1], sat, clip)[0]
                return out

            def classes_changed(before, after):
                diff = []
                for j in before:
                    idx = np.nonzero(before[j] != after[j])[0]
                    diff += [{"layer": j, "neuron": int(i), "before": self.class_name(before[j][i]),
                              "after": self.class_name(after[j][i])} for i in idx.tolist()]
                return {"count": len(diff), "first": diff[:20]}

            def root_witness_checks(scaled):
                return {wid: cd.idot(lam[0], t[1][top]["r"]) >= scaled for wid, t in traces.items()}

            with cd.ResourceMonitor() as image_res:
                scaled0, terms0, steps0 = margin(bounds, record=True)
                if scaled0 != base_scaled:
                    raise AuditStop(f"{o} c{c}: recomputed primary chain differs from the cached baseline")
                classes0 = relu_classes(steps0, bounds)
                ranking = rankings[o][str(c)]["top16_global"]
                picked = []
                for e in ranking:
                    L, n = int(e["layer"]), int(e["neuron"])
                    lo, hi, sat, clip = bounds[L]
                    if (int(hi[n]) > int(lo[n]) and not bool(sat[n]) and not bool(clip[n])
                            and cd.Q_LOW <= int(lo[n]) and int(hi[n]) <= cd.Q_HIGH):
                        picked.append((L, n, e))
                    if len(picked) == per_primary:
                        break
                img_summary["eligible_targets"] = [f"{L}:{n}" for L, n, _ in picked]
                for rank_index, (L, n, entry) in enumerate(picked):
                    r0 = int(nominal[L]["r"][n])
                    for side in ("lower", "upper"):
                        tag = f"{o}/I-B/L{L}n{n}/{side}"
                        if time.monotonic() > deadline:
                            for arm in ("hypothetical", "attained"):
                                self.record(image, run_id=f"{tag}/{arm}", diagnostic="I-B", competitor=c,
                                            status="NOT_RUN_BUDGET", evidence_kind="COUNTERFACTUAL_ONLY"
                                            if arm == "hypothetical" else "SCREEN_BOUND_UNCHECKED",
                                            dependency_status="SCREEN_ONLY",
                                            interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                                            "reason_code": "NOT_RUN_BUDGET"})
                            continue
                        result = self.bound_target(image, tag, L, n, side, r0, rank_index, entry, bounds, info,
                                                   traces, nominal, margin, relu_classes, classes_changed,
                                                   root_witness_checks, classes0, base_scaled, base_raw,
                                                   threshold, closure, per_side, xl, xu, write_intervention)
                        img_summary["targets"].append(result)
                img_summary["prefix"] = self.prefix_probe(image, img_summary["targets"], mats, write_intervention)
            self.record(image, image_res, run_id=f"{o}/I-B", diagnostic="I-B", competitor=c,
                        epsilon_raw_bytes=self.eps, domain="D_box", evidence_kind="SCREEN_BOUND_UNCHECKED",
                        dependency_status="SCREEN_ONLY", metric_scope="image_bounds_stage",
                        parameters={"targets": img_summary["eligible_targets"], "witnesses": sorted(traces)},
                        interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                        "reason_code": "STAGE_SUMMARY"})
        (self.out / "bounds.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
        return None

    def class_name(self, idx):
        idx = int(idx)
        return self.P.v2.CLASSES[idx] if idx >= 0 else None

    def bound_target(self, image, tag, L, n, side, r0, rank_index, entry, bounds, info, traces, nominal, margin,
                     relu_classes, classes_changed, root_witness_checks, classes0, base_scaled, base_raw,
                     threshold, closure, per_side, xl, xu, write_intervention):
        P, np, cd = self.P, self.np, self.cd
        v2 = P.v2
        o, c, target = image["ordinal"], image["primary"], image["target"]
        mats = self.model()["mats"]
        lo, hi, sat, clip = bounds[L]
        endpoint = int(lo[n]) if side == "lower" else int(hi[n])
        source = ("box" if bool(info[L]["lower_box"][n]) else "chain") if side == "lower" else \
            ("chain" if bool(info[L]["upper_chain"][n]) else "box")
        hypo = cd.midpoint_endpoint(side, endpoint, r0)
        frozen_hash = sha256_json({f"L{k}": [cd.sha256_array(a) for a in b] for k, b in enumerate(bounds)})
        common = {"target_layer": L, "target_neuron": n, "side": side, "baseline_endpoint": endpoint,
                  "baseline_endpoint_source": source, "nominal_raw_value": r0, "candidate_rank": rank_index,
                  "selection_score": {"max_relu_slack": entry["max_relu_slack"],
                                      "abs_a_times_width": entry["abs_a_times_width"]},
                  "frozen_fields_hash": frozen_hash, "baseline_raw_lb": base_raw,
                  "baseline_scaled_raw_lb": str(base_scaled), "material_threshold": threshold}
        result = {"tag": tag, "layer": L, "neuron": n, "side": side, "rank": rank_index}

        def with_endpoint(value):
            new_lo, new_hi = lo.copy(), hi.copy()
            (new_lo if side == "lower" else new_hi)[n] = value
            return [b if k != L else (new_lo, new_hi, sat, clip) for k, b in enumerate(bounds)]

        def margin_arm(value):
            bnds = with_endpoint(value)
            scaled, terms, steps = margin(bnds, record=True)
            raw = cd.ceil_div(scaled, 1 << cd.SCALE_BITS)
            return {"scaled": scaled, "raw": raw, "terms": terms, "gain": raw - base_raw,
                    "scaled_gain": scaled - base_scaled,
                    "classes": classes_changed(classes0, relu_classes(steps, bnds)),
                    "root_witness_checks": root_witness_checks(scaled)}

        def margin_csv(run_id, arm, evidence):
            self.margin_row(run_id=run_id, ordinal=o, domain="D_box", epsilon=self.eps, target=target, competitor=c,
                            scale_bits=cd.SCALE_BITS, scaled_raw_lb=arm["scaled"], raw_lb=arm["raw"],
                            top_rule=(closure["top"] or {}).get("rule"),
                            top_reference=(closure["top"] or {}).get("reference"), top_holds=closure["top"] is not None,
                            bottom_rule=(closure["bottom"] or {}).get("rule"),
                            bottom_reference=(closure["bottom"] or {}).get("reference"),
                            bottom_holds=closure["bottom"] is not None,
                            closed=closure["top"] is not None and closure["bottom"] is not None and arm["raw"] >= 1,
                            baseline_raw_lb=base_raw, delta_raw_lb=arm["gain"],
                            material_gain=arm["gain"] >= threshold or (arm["raw"] >= 1 > base_raw and closure["top"]
                                                                       is not None and closure["bottom"] is not None),
                            evidence_kind=evidence, dependency_status="SCREEN_ONLY")

        # ---- hypothetical (COUNTERFACTUAL_ONLY)
        hyp_id = f"{tag}/hypothetical"
        if hypo == endpoint:
            self.record(image, run_id=hyp_id, diagnostic="I-B", competitor=c, epsilon_raw_bytes=self.eps,
                        domain="D_box", status="SKIPPED_NO_CANDIDATE", evidence_kind="COUNTERFACTUAL_ONLY",
                        dependency_status="SCREEN_ONLY", parameters={**common, "hypothetical_endpoint": hypo},
                        interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                        "reason_code": "ENDPOINT_UNCHANGED"})
            result.update(hypothetical=None, counterfactual_gain=None)
        else:
            refuting = sorted(wid for wid, t in traces.items() if cd.endpoint_refuted(side, hypo, t[1][L]["r"][n]))
            with cd.ResourceMonitor() as res:
                arm = margin_arm(hypo)
            fields = {**common, "hypothetical_endpoint": hypo,
                      "hypothetical_endpoint_refuted_by_witness": bool(refuting),
                      "refuting_witness_id": refuting[0] if refuting else None, "refuting_witness_ids": refuting,
                      "counterfactual_gain": arm["gain"], "raw_scaled_delta": str(arm["scaled_gain"]),
                      "raw_integer_delta": arm["gain"], "material_gain": arm["gain"] >= threshold,
                      "root_classes_changed": arm["classes"], "root_terms": arm["terms"],
                      "root_inequality_witness_failures": sorted(w for w, ok in arm["root_witness_checks"].items()
                                                                 if not ok),
                      "changed_fields": [f"bounds[{L}].{'lo' if side == 'lower' else 'hi'}[{n}]"],
                      "before": {"endpoint": endpoint, "raw_lb": base_raw},
                      "after": {"endpoint": hypo, "raw_lb": arm["raw"]},
                      "certificate_compatibility": "NEVER_EXPORTABLE (hypothetical endpoint)"}
            self.record(image, res, run_id=hyp_id, parent_run_id=f"{o}/D-P/baseline/eps1", diagnostic="I-B",
                        competitor=c, epsilon_raw_bytes=self.eps, domain="D_box",
                        evidence_kind="COUNTERFACTUAL_ONLY", dependency_status="COUNTERFACTUAL", parameters=fields,
                        interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                        "reason_code": "HYPOTHETICAL_SENSITIVITY" if arm["gain"] > 0
                                        else "HYPOTHETICAL_NO_ROOT_GAIN"})
            write_intervention({"run_id": hyp_id, "arm": "hypothetical", **fields})
            margin_csv(hyp_id, arm, "COUNTERFACTUAL_ONLY")
            result.update(hypothetical=hypo, counterfactual_gain=arm["gain"], refuted=bool(refuting))

        # ---- attained (SCREEN_BOUND_UNCHECKED): unit root chain for this side
        att_id = f"{tag}/attained"
        sign = 1 if side == "lower" else -1
        A = np.zeros((1, lo.size), dtype=np.int64)
        A[0, n] = sign << cd.SCALE_BITS
        unit_traces = {wid: (t[0], t[1][:L + 1]) for wid, t in traces.items()}

        def unit(record=False):
            with self.XUState(P.cce, xu):
                scaled, terms, steps = v2.run_chain(A, L, mats, bounds, xl, record=record)
                need = v2.needs(A, L, mats, bounds)
            return int(scaled[0]), int(terms[0]), steps, need

        def to_endpoint(scaled):
            return cd.ceil_div(scaled, 1 << cd.SCALE_BITS) if side == "lower" else (-scaled) // (1 << cd.SCALE_BITS)

        def dependency_sides(need):
            return {str(j): {"lower": int(np.count_nonzero(v[0])), "upper_value": int(np.count_nonzero(v[1])),
                             "upper_nosat": int(np.count_nonzero(v[2]))} for j, v in need.items()}

        with cd.ResourceMonitor() as res:
            u_scaled, u_terms, u_steps, u_need = unit(record=True)
            chain_value = int(info[L]["lo_chain" if side == "lower" else "hi_chain"][n])
            if to_endpoint(u_scaled) != chain_value:
                raise AuditStop(f"{att_id}: unit chain {to_endpoint(u_scaled)} != cached bounds_v2 chain {chain_value}")
            per_neuron = {}
            for wid, (x, tr) in unit_traces.items():
                try:
                    acc = cd.account_chain(u_steps, 0, A[0], tr, x, xl, xu, self.model()["biases"], u_scaled)
                except cd.ImplementationError as exc:
                    raise AuditStop(f"{att_id} baseline unit chain at {wid}: {exc}")
                for rec in acc["steps"]:
                    for neuron, slack in rec.pop("_per_coordinate", {}).items():
                        key = (rec["layer"], neuron)
                        per_neuron[key] = max(per_neuron.get(key, 0), slack)
            relu_a = {s["j"]: s["a"][0] for s in u_steps if s["kind"] == "relu"}
            relu_out = {s["j"]: s["a_out"][0] for s in u_steps if s["kind"] == "relu"}
            ranked = cd.rank_coordinates(per_neuron, relu_a, bounds)[:per_side]
            trials = [{"order": -1, "policy": "baseline", "scaled": u_scaled, "terms": u_terms,
                       "endpoint": to_endpoint(u_scaled), "dependencies": dependency_sides(u_need)}]
            for rank, (slack, width_term, j, i) in enumerate(ranked):
                a, a_base = int(relu_a[j][i]), int(relu_out[j][i])
                for value in cd.slope_candidates(a, a_base):
                    with cd.SlopeOverride(v2, bounds[j], 0, i, a, value) as ov:
                        scaled, terms, _, need = unit()
                    if ov.applied != 2:     # once in run_chain, once in needs
                        raise AuditStop(f"{att_id}: override applied {ov.applied} times, expected 2")
                    trials.append({"order": len(trials) - 1, "policy": {"layer": j, "neuron": i, "a": a,
                                                                         "baseline_a_out": a_base,
                                                                         "trial_a_out": value, "trace_rank": rank,
                                                                         "max_relu_slack": str(slack)},
                                   "scaled": int(scaled), "terms": int(terms), "endpoint": to_endpoint(int(scaled)),
                                   "dependencies": dependency_sides(need)})
            best = max(trials, key=lambda t: (t["scaled"], -t["terms"], -t["order"]))
            attained = best["endpoint"] if cd.tighter(side, best["endpoint"], endpoint) else None
            for t in trials:   # every unit-chain bound must hold at every concrete execution
                bad = [wid for wid, tr in traces.items() if cd.endpoint_refuted(side, t["endpoint"], tr[1][L]["r"][n])]
                if bad:
                    raise AuditStop(f"{att_id}: unit-chain endpoint {t['endpoint']} ({t['policy']}) contradicted "
                                    f"by concrete execution {bad[0]}")
            arm = margin_arm(attained) if attained is not None else None
            if arm is not None and not all(arm["root_witness_checks"].values()):
                raise AuditStop(f"{att_id}: root bound with attained endpoint fails at a concrete execution")
        hyp_gain = result.get("counterfactual_gain")
        fields = {**common, "bound_chain_id": f"chain/{side}/L{L}/n{n}",
                  "bound_policy": best["policy"], "bound_policy_scaled": str(best["scaled"]),
                  "bound_policy_terms": best["terms"], "bound_dependencies": best["dependencies"],
                  "baseline_unit_chain_endpoint": chain_value, "trials": len(trials) - 1,
                  "trial_results": [{**{k: v for k, v in t.items() if k != "scaled"}, "scaled": str(t["scaled"])}
                                    for t in trials],
                  "ranked_coordinates": [{"layer": j, "neuron": i, "max_relu_slack": str(s),
                                          "abs_a_times_width": str(w)} for s, w, j, i in ranked],
                  "attained_endpoint": attained, "hypothetical_endpoint": result.get("hypothetical"),
                  "attained_gain": None if arm is None else arm["gain"],
                  "attained_fraction_of_counterfactual": None if arm is None or not hyp_gain or hyp_gain <= 0
                  else cd.rational(arm["gain"], hyp_gain),
                  "material_gain": arm is not None and arm["gain"] >= threshold,
                  "root_classes_changed": None if arm is None else arm["classes"],
                  "root_terms": None if arm is None else arm["terms"],
                  "changed_fields": [] if attained is None else
                  [f"bounds[{L}].{'lo' if side == 'lower' else 'hi'}[{n}]"],
                  "certificate_compatibility": "ordinary chain-backed bound; current checker form if exported with "
                                               "its supporting chains (not exported here)",
                  "conditional_dependencies": "all other intermediate bounds are the baseline screen bounds "
                                              "(SCREEN_BOUND_UNCHECKED)"}
        if attained is None:
            interp = ("INCONCLUSIVE", "NO_TIGHTER_UNIT_CHAIN")
        elif arm["gain"] >= threshold:
            interp = ("SUPPORTS", "ATTAINED_MATERIAL_ROOT_GAIN")
        elif arm["gain"] > 0:
            interp = ("INCONCLUSIVE", "ATTAINED_SUBMATERIAL_ROOT_GAIN")
        else:
            interp = ("NO_SUPPORT_WITHIN_PROBE", "ATTAINED_ENDPOINT_NO_ROOT_GAIN")
        self.record(image, res, run_id=att_id, parent_run_id=f"{o}/D-P/baseline/eps1", diagnostic="I-B",
                    competitor=c, epsilon_raw_bytes=self.eps, domain="D_box", evidence_kind="SCREEN_BOUND_UNCHECKED",
                    dependency_status="SCREEN_ONLY", parameters=fields,
                    interpretation={"hypothesis_id": "H-B", "assessment": interp[0], "reason_code": interp[1]})
        write_intervention({"run_id": att_id, "arm": "attained", **fields})
        if arm is not None:
            margin_csv(att_id, arm, "SCREEN_BOUND_UNCHECKED")
        result.update(attained=attained, attained_gain=None if arm is None else arm["gain"],
                      attained_assessment=interp[0], attained_reason=interp[1],
                      material=fields["material_gain"])
        return result

    def prefix_probe(self, image, targets, mats, write_intervention):
        """One exact full-prefix probe per image for the side with the greatest positive counterfactual gain."""

        cd = self.cd
        o, c = image["ordinal"], image["primary"]
        run_id = f"{o}/I-B/prefix"
        positive = [t for t in targets if (t.get("counterfactual_gain") or 0) > 0]
        if not positive:
            self.record(image, run_id=run_id, diagnostic="I-B", competitor=c, status="SKIPPED_NO_CANDIDATE",
                        evidence_kind="COUNTERFACTUAL_ONLY", dependency_status="SCREEN_ONLY",
                        interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                        "reason_code": "NO_POSITIVE_COUNTERFACTUAL_GAIN"})
            return {"status": "SKIPPED_NO_CANDIDATE"}
        # ties: ranking order, lower before upper (targets are already in that order)
        sel = max(positive, key=lambda t: (t["counterfactual_gain"], -targets.index(t)))
        cone = cd.cone_size(mats, sel["layer"], sel["neuron"])
        size_cap = int(self.config["prefix_multiply_term_cap"])
        if sel.get("refuted"):
            status, reason = "SKIPPED_NO_CANDIDATE", "REFUTED_BY_KNOWN_WITNESS"
        elif cone["multiply_terms"] > size_cap:
            status, reason = "SKIPPED_NO_CANDIDATE", "STRUCTURAL_SIZE_CAP"
        else:
            status, reason = "UNSUPPORTED", "PREFIX_HARNESS_NOT_IMPLEMENTED"
        fields = {"selected": sel["tag"], "target_layer": sel["layer"], "target_neuron": sel["neuron"],
                  "side": sel["side"], "hypothetical_endpoint": sel["hypothetical"],
                  "counterfactual_gain": sel["counterfactual_gain"], "prefix_terms": cone["multiply_terms"],
                  "prefix_neurons_by_layer": cone["neurons_by_layer"], "prefix_source_bytes": cone["source_inputs"],
                  "prefix_term_cap": size_cap, "prefix_esbmc_status": "NOT_RUN", "prefix_skip_reason": reason,
                  "prefix_counterexample_kind": None,
                  "note": "source inputs are encoded input coordinates; every alias group has size 1 for these crops"}
        self.record(image, run_id=run_id, diagnostic="I-B", competitor=c, epsilon_raw_bytes=self.eps, domain="D_box",
                    status=status, evidence_kind="COUNTERFACTUAL_ONLY", dependency_status="SCREEN_ONLY",
                    parameters=fields, interpretation={"hypothesis_id": "H-B", "assessment": "INCONCLUSIVE",
                                                       "reason_code": reason})
        write_intervention({"run_id": run_id, "arm": "exact_prefix", **fields})
        return {"status": status, **fields}

    # ------------------------------------------------------------ report

    def stage_report(self, deadline):
        runs = [json.loads(l) for l in (self.out / "runs.jsonl").read_text().splitlines() if l] \
            if (self.out / "runs.jsonl").exists() else []
        if self.prior and self.prior != self.out and (self.prior / "runs.jsonl").exists():
            runs = [json.loads(l) for l in (self.prior / "runs.jsonl").read_text().splitlines() if l] + runs
        base = self.prior if self.prior and not (self.out / "preflight.json").exists() else self.out
        self._runs_cache = runs

        def load(name):
            path = base / name
            return json.loads(path.read_text()) if path.exists() else None

        preflight, dv, singleton_path = load("preflight.json"), load("violations.json"), base / "singleton.jsonl"
        rankings = load("rankings.json")
        singleton = [json.loads(l) for l in singleton_path.read_text().splitlines()] if singleton_path.exists() else []
        ib = [json.loads(l) for l in (base / "interventions.jsonl").read_text().splitlines() if l] \
            if (base / "interventions.jsonl").exists() else []
        lines = ["# CROWN finder diagnostics: phase 1 + I-B report", "",
                 "Development images only (October validation ordinals 013, 016, 017). These results do not replace "
                 "the 5/20, 1/80, 2/80 or P3 results.", "",
                 f"Sources: `{self.manifest['source_manifest_sha256']}`; config `{self.config_sha}`.", "",
                 "This run covers D-P, D-0, D-V, D-T and I-B" + ("" if ib else " (I-B not run)") + ". "
                 "Interventions I-S, I-R, I-P, I-Q, I-C, I-E and the cost stage were **not implemented**; "
                 "items (e) and (f) below are `NOT_RUN`. Column (d) summarises I-B on the primary competitor "
                 "only: best hypothetical (COUNTERFACTUAL_ONLY) and best attained (SCREEN_BOUND_UNCHECKED) "
                 "integer root gains.", ""]
        decisions = []
        lines.append("## D-P preflight")
        lines.append("")
        if preflight is None:
            lines.append("Not run.")
        else:
            lines.append(f"- passed: **{preflight['passed']}**; failures: {json.dumps(preflight['failures'])}")
            if not preflight["passed"]:
                decisions.append(("all", None, "Source/parity failure or baseline mismatch", "P0", "all/D-P"))
        lines.append("")
        for image in self.images:
            o = image["ordinal"]
            lines.append(f"## {o} `{image['record_id']}` target {image['target']}"
                         f"{' (control)' if image.get('control') else ''}")
            lines.append("")
            lines.append("| competitor | (a) violation observed | (b) singleton gap Z-inh / Z-fresh / Z-exact | "
                         "(c) top ReLU slack (layer:neuron) | (d) interventions | (e) representability | (f) burden |")
            lines.append("|---|---|---|---|---|---|---|")
            for c in image["focus"]:
                a = "not run"
                if dv is not None:
                    img = dv["images"].get(o, {})
                    focus = img.get("focus", {}).get(str(c), {})
                    viols = [v for v in img.get("violations", []) if c in v["competitors"]]
                    if viols:
                        v = viols[0]
                        a = f"YES ({v['domain']}, crop lift {v['crop_lift']})"
                        decisions.append((o, c, "Lifted byte witness with deployed margin <=0" if v["crop_lift"]
                                          else "Box-only violation failing original-crop alias constraints",
                                          "P0" if v["crop_lift"] else "P1", f"{o}/D-V/c{c}"))
                    elif focus:
                        best = focus.get("best") or {}
                        a = (f"NO_WITNESS_FOUND ({focus.get('status')}, best deployed {best.get('deployed')}, "
                             f"raw {best.get('raw')})")
                gaps = []
                for arm in ("Z-inherited", "Z-fresh", "Z-exact"):
                    row = next((r for s in singleton if s["ordinal"] == o and s["arm"] == arm
                                for r in s["margins"] if r["competitor"] == c), None)
                    gaps.append("-" if row is None else f"{row['raw_gap']}{'*' if row['significant_gap'] else ''}")
                top = ((rankings or {}).get(o, {}).get(str(c), {}).get("top16_global") or [None])[0]
                ctext = "-" if top is None else f"{top['layer']}:{top['neuron']} ({top['max_relu_slack']})"
                dtext = "NOT_RUN"
                rows = [r for r in ib if r["run_id"].startswith(f"{o}/I-B/") and r["arm"] != "exact_prefix"]
                if c == image["primary"] and rows:
                    hyp = [r["counterfactual_gain"] for r in rows if r["arm"] == "hypothetical"]
                    att = [r["attained_gain"] for r in rows if r["arm"] == "attained" and r["attained_gain"] is not None]
                    dtext = (f"I-B: hyp {max(hyp) if hyp else '-'}, attained {max(att) if att else 'none'} "
                             f"(threshold {rows[0]['material_threshold']})")
                lines.append(f"| {c} | {a} | {' / '.join(gaps)} | {ctext} | {dtext} | NOT_RUN | NOT_RUN |")
            lines.append("")
        lines += ["`*` marks a significant singleton gap (raw_gap >= max(1, ceil(0.05*|nominal raw margin|))).", ""]
        if ib:
            lines += self.report_bounds(ib, decisions)
        lines += ["## (g) Decision-table rows matched", ""]
        if decisions:
            for o, c, row, priority, rid in decisions:
                lines.append(f"- {o} c{c}: **{row}** ({priority}); run `{rid}`")
        else:
            lines.append("- No decision row matched. **Bottleneck unresolved within this diagnostic budget.**")
        lines += ["", "## Runs", "", f"{len(runs)} records in runs.jsonl; statuses: "
                  f"{json.dumps(self._count(runs, 'status'))}; assessments: "
                  f"{json.dumps(self._count([r['interpretation'] for r in runs], 'assessment'))}.", ""]
        (self.out / "report.md").write_text("\n".join(lines) + "\n")
        with (self.out / "decisions.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ordinal", "competitor", "observed_evidence_row", "priority", "run_id"])
            writer.writerows(decisions)
        return None

    def report_bounds(self, ib, decisions):
        """I-B table and the three I-B rows of the spec section 11 decision table."""

        lines = ["## I-B: one intermediate endpoint side (primary competitor)", "",
                 "Hypothetical = midpoint endpoint injected (COUNTERFACTUAL_ONLY, never exportable). "
                 "Attained = best unit-root chain under the baseline policy or one I-S coordinate alternative "
                 "(SCREEN_BOUND_UNCHECKED: a finder bound, not ESBMC-checked). Gains are integer raw root-bound gains.",
                 "", "| target | baseline endpoint (source) | r0 | hypothetical | refuted by witness | hyp gain | "
                 "attained endpoint | attained gain | material | assessment |", "|---|---|---|---|---|---|---|---|---|---|"]
        by_image = {}
        for r in ib:
            o = r["run_id"].split("/")[0]
            by_image.setdefault(o, []).append(r)
        for image in self.images:
            o, c = image["ordinal"], image["primary"]
            rows = by_image.get(o, [])
            hyp = {r["run_id"].rsplit("/", 1)[0]: r for r in rows if r["arm"] == "hypothetical"}
            att = [r for r in rows if r["arm"] == "attained"]
            for a in att:
                key = a["run_id"].rsplit("/", 1)[0]
                h = hyp.get(key, {})
                assess = next((x["interpretation"] for x in self._runs_cache if x["run_id"] == a["run_id"]), {})
                lines.append(f"| {o} c{c} {a['target_layer']}:{a['target_neuron']} {a['side']} | "
                             f"{a['baseline_endpoint']} ({a['baseline_endpoint_source']}) | {a['nominal_raw_value']} | "
                             f"{h.get('hypothetical_endpoint', '-')} | {h.get('hypothetical_endpoint_refuted_by_witness', '-')} | "
                             f"{h.get('counterfactual_gain', '-')} | {a['attained_endpoint']} | {a['attained_gain']} | "
                             f"{a['material_gain']} | {assess.get('reason_code', '-')} |")
            prefix = next((r for r in rows if r["arm"] == "exact_prefix"), None)
            if prefix:
                lines.append(f"| {o} exact prefix {prefix['target_layer']}:{prefix['target_neuron']} {prefix['side']} | "
                             f"cone {prefix['prefix_terms']} terms (cap {prefix['prefix_term_cap']}) | | | | | | | | "
                             f"ESBMC {prefix['prefix_esbmc_status']}: {prefix['prefix_skip_reason']} |")
            material = [a for a in att if a["material_gain"]]
            attained_small = [a for a in att if a["attained_endpoint"] is not None and not a["material_gain"]]
            hyp_material = [h for h in hyp.values() if h["material_gain"]]
            for a in material:
                decisions.append((o, c, "I-B obtains a tighter ordinary bound chain and material root gain", "P1",
                                  a["run_id"]))
            if hyp_material and not material:
                decisions.append((o, c, "Only hypothetical I-B bound changes help; attempts cannot establish them",
                                  "P2", ";".join(h["run_id"] for h in hyp_material)))
            if attained_small:
                decisions.append((o, c, "I-B proves/refines an endpoint but root bound barely changes", "P2",
                                  ";".join(a["run_id"] for a in attained_small)))
        lines += ["", "A hypothetical endpoint refuted by a retained witness is an expected possible outcome of a "
                  "hypothetical arm, not an implementation failure. The exact-prefix probe is skipped when the "
                  "ancestor cone exceeds the spec's 250,000-term cap; no ESBMC conclusion is drawn from a skip.", ""]
        return lines

    @staticmethod
    def _count(items, key):
        out = {}
        for item in items:
            out[item[key]] = out.get(item[key], 0) + 1
        return out


def main(argv=None):
    args = parse_args(argv)
    config_text = args.config.read_text(encoding="utf-8")
    config = json.loads(config_text)
    main_src = Path(config["working_directory"]) / "src"
    finder = guard.install(SRC, main_src)
    project = load_project()
    audit = guard.audit(finder, main_src)
    if args.import_audit_only:
        print(json.dumps(audit))
        return 0 if audit["passed"] else 3
    if args.stage is None or args.output is None:
        raise SystemExit("--stage and --output are required")
    if args.stage != "all" and args.stage not in IMPLEMENTED:
        raise SystemExit(f"stage {args.stage} is not implemented in phase 1 (implemented: {IMPLEMENTED})")
    run = Run(args, config, config_text, project, audit, finder)
    stages = list(IMPLEMENTED) if args.stage == "all" else [args.stage]
    if args.stage == "all":
        run.note_unimplemented([s for s in ALL_STAGES if s not in IMPLEMENTED])
    return run.execute(stages)


if __name__ == "__main__":
    sys.exit(main())
