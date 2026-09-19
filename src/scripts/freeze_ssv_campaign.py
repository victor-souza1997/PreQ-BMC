"""Freeze an existing trained traffic-sign model, formats and 36-run campaign.

No training, sample reselection, synthesis or verification is performed.
"""
import argparse
import copy
import json
from pathlib import Path

from datasets.gtsrb_study import sha256
from models.ssv_artifact import freeze_artifact
from scripts.run_ssv_cnn_gate import write_new_json


def campaign_runs(study):
    selected = study["selected_images"]
    if len(selected) != 9 or len({s["id"] for s in selected}) != 9:
        raise ValueError("Expected nine distinct, already frozen images")
    controls = set()
    for stratum in ("low", "median", "high"):
        rows = [s for s in selected if s["stratum"] == stratum]
        if len(rows) != 3:
            raise ValueError("Expected three frozen images per margin stratum")
        controls.add(rows[0]["id"])
    runs = [copy.deepcopy(r) for r in study["runs"] if
            (r["epsilon"] in (1, 2, 4) and r["block_size"] == 1 and not r["margin_cuts"])
            or (r["sample"]["id"] in controls and r["epsilon"] == 1 and
                ((r["block_size"] in (0, 2) and not r["margin_cuts"])
                 or (r["block_size"] == 1 and r["margin_cuts"])))]
    identities = [(r["sample"]["id"], r["epsilon"], r["block_size"], r["margin_cuts"]) for r in runs]
    if (len(runs) != 36 or len(set(identities)) != 36
            or len({r["run_id"] for r in runs}) != 36
            or {r["sample"]["id"] for r in runs} != {s["id"] for s in selected}):
        raise ValueError("Frozen study does not contain the required 27 main + 9 control configurations")
    return runs


def main_runs(study):
    """The 27-region primary cohort, excluding decomposition controls."""
    runs = [copy.deepcopy(r) for r in study["runs"] if
            r["epsilon"] in (1, 2, 4) and r["block_size"] == 1 and not r["margin_cuts"]]
    identities = [(r["sample"]["id"], r["epsilon"]) for r in runs]
    if len(runs) != 27 or len(set(identities)) != 27:
        raise ValueError("Frozen study does not contain the required 27 primary configurations")
    return runs


def freeze(study_path, output, rows, *, campaign="full"):
    study_path, output = Path(study_path), Path(output)
    study = json.loads(study_path.read_text())
    if campaign not in {"full", "main"}:
        raise ValueError("Campaign must be 'full' or 'main'")
    runs = campaign_runs(study) if campaign == "full" else main_runs(study)
    output.mkdir(parents=True, exist_ok=False)
    artifact = freeze_artifact(study, rows, output / "artifact")
    updated = copy.deepcopy(study)
    updated.update({"runs": runs, "fixed_artifact": str((output / "artifact/artifact.json").resolve()),
                    "fixed_artifact_sha256": sha256(output / "artifact/artifact.json"),
                    "parent_study": str(study_path.resolve()), "parent_study_sha256": sha256(study_path),
                    "campaign": ("27_main_plus_9_predeclared_controls" if campaign == "full" else "27_main"),
                    "proof_status": "NOT_RUN"})
    for run in updated["runs"]:
        run["verification_mode"] = "fixed_qif_check"
        run["fixed_qif"] = artifact["qif"]
    write_new_json(output / "study.json", updated)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qif", nargs=2, required=True, metavar="Q,I,F",
                        help="Explicit convolution and output formats, e.g. 11,2,8 13,4,8")
    parser.add_argument("--campaign", choices=["full", "main"], default="full")
    args = parser.parse_args()
    rows = []
    for entry in args.qif:
        try:
            q, i, f = map(int, entry.split(","))
        except ValueError:
            parser.error("Each format must be Q,I,F")
        rows.append({"total_bits": q, "integer_bits": i, "fractional_bits": f})
    freeze(args.study, args.output, rows, campaign=args.campaign)
    print(args.output / "study.json")


if __name__ == "__main__":
    main()
