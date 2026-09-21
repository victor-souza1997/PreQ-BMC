"""Prepare matched output-refinement variants of an existing frozen artifact."""
import argparse
import copy
import json
from pathlib import Path

from datasets.gtsrb_study import sha256
from models.ssv_artifact import load_artifact
from scripts.run_ssv_cnn_gate import write_new_json
from scripts.prepare_ssv_gtsrb import validate_config


def prepare(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config.get('schema') != 'ssv_output_refinement_study_v1':
        raise ValueError('Expected ssv_output_refinement_study_v1')
    base_path = Path(config['base_study'])
    study = json.loads(base_path.read_text())
    artifact_path = study.get('fixed_artifact')
    if not artifact_path or sha256(artifact_path) != study.get('fixed_artifact_sha256'):
        raise ValueError('An unchanged frozen artifact is required')
    artifact, _ = load_artifact(artifact_path, study)
    variants = config['variants']
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in {'box', 'legacy_cuts', 'affine_residual'} for v in variants)):
        raise ValueError('Invalid refinement variants')
    selected = config.get('run_ids')
    runs = [r for r in study['runs'] if r['block_size'] == 1 and not r['margin_cuts']
            and (selected is None or r['run_id'] in selected)]
    if not runs or (selected is not None and {r['run_id'] for r in runs} != set(selected)):
        raise ValueError('Requested base regions are missing')
    verification = dict(config['verification'])
    allowed = {'source_verification', 'source_milp_timeout_seconds', 'timeout_seconds', 'memlimit', 'profile'}
    if set(verification)-allowed:
        raise ValueError('This preparer only changes proof resource limits and the source gate')
    updated = copy.deepcopy(study)
    updated['config']['verification'].update(verification)
    validate_config(updated['config'])
    updated['runs'] = []
    for run in runs:
        for variant in variants:
            row = copy.deepcopy(run)
            row.update(verification)
            row.update(run_id=run['run_id']+'_'+variant,
                       margin_cuts=variant == 'legacy_cuts',
                       output_refinement='affine_residual' if variant == 'affine_residual' else 'none',
                       verification_mode='fixed_qif_check', fixed_qif=artifact['qif'],
                       refinement_variant=variant, base_run_id=run['run_id'])
            updated['runs'].append(row)
    updated.update(parent_study=str(base_path.resolve()), parent_study_sha256=sha256(base_path),
                   refinement_config_sha256=sha256(config_path), proof_status='NOT_RUN',
                   campaign='matched_output_refinement', n_regions=len(runs),
                   n_images=len({r['sample']['id'] for r in runs}),
                   interpretation=config.get('purpose', 'Matched verification variants; no sample reselection'))
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output/'study.json', updated)
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)
    print(args.output/'study.json')


if __name__ == '__main__':
    main()
