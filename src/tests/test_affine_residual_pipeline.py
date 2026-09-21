import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from synthesis.preqbmc import GPEncoding
from verification.esbmc import ESBMCResult
from scripts.report_ssv_regions import summarize
from scripts.prepare_ssv_refinement import prepare


def outcome(status):
    return ESBMCResult(status=status, command=('esbmc',), stdout='', stderr='',
                       return_code=0 if status == 'VERIFIED' else 1, elapsed_seconds=.1)


def encoder(root):
    enc = object.__new__(GPEncoding)
    enc.output_dir = root
    enc.config = SimpleNamespace(output_refinement='affine_residual',
                                 esbmc=SimpleNamespace(timeout_seconds=30, memlimit='6g'))
    enc.property_spec = SimpleNamespace(valid_labels=(), target_label=0)
    enc.targetCls = 0
    enc.x_low_real, enc.x_high_real = np.array([0.]), np.array([4.])
    enc.dense_layers = [SimpleNamespace(frac_bit=0, int_bit=8, layer_size=2,
        layer_paras=(np.array([[1.], [1.]]), np.array([4., 4.])))]
    enc.affine_residual_records = []
    enc.esbmc_call_records = []
    enc._stats = {'esbmc_calls': 0.}
    enc._run_esbmc_file = Mock(return_value=outcome('VERIFIED'))
    return enc


ARGS = dict(qu_w_int=np.array([[2, -2], [0, 0], [0, 0]]), qu_b_int=np.array([1, 0, 0]),
            all_bit=8, frac_bit=0, layer_index=1, competitor=1)


class AffineResidualPipelineTest(unittest.TestCase):
    def test_kernel_failure_or_timeout_cannot_certify_or_run_composition(self):
        for status in ('FAILED', 'TIMEOUT', 'MEMOUT', 'UNKNOWN'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                enc = encoder(Path(temp))
                enc._run_esbmc_file.side_effect = [outcome(status)]
                result = enc._verify_affine_residual_margin(**ARGS)
                self.assertEqual(result.status, status)
                self.assertFalse(enc.affine_residual_records[-1]['all_prerequisites_verified'])
                self.assertEqual(enc._run_esbmc_file.call_count, 1)

    def test_nonnegative_bound_and_deeper_prefix_do_not_invoke_esbmc(self):
        with tempfile.TemporaryDirectory() as temp:
            enc = encoder(Path(temp))
            self.assertIsNone(enc._verify_affine_residual_margin(**{**ARGS, 'qu_b_int': np.array([-1, 0, 0])}))
            enc._run_esbmc_file.assert_not_called()
            enc.dense_layers *= 2
            self.assertIsNone(enc._verify_affine_residual_margin(**ARGS))
            enc._run_esbmc_file.assert_not_called()

    def test_cache_binds_kernel_parameters_and_all_prerequisites_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            enc = encoder(Path(temp))
            self.assertEqual(enc._verify_affine_residual_margin(**ARGS).status, 'VERIFIED')
            initial = enc._run_esbmc_file.call_count
            self.assertEqual(enc._verify_affine_residual_margin(**ARGS).status, 'VERIFIED')
            self.assertEqual(enc._run_esbmc_file.call_count, initial+1)  # composition always checked
            proofs = enc.affine_residual_records[-1]['proofs']
            self.assertEqual(len(proofs), 5)
            self.assertTrue(all(p['cache_hit'] for p in proofs[:-1]))
            enc.dense_layers[0].int_bit = 9
            enc._verify_affine_residual_margin(**ARGS)
            self.assertFalse(enc.affine_residual_records[-1]['proofs'][0]['cache_hit'])

    def test_output_recovery_preserves_failed_call_and_checks_remaining_classes(self):
        with tempfile.TemporaryDirectory() as temp:
            enc = encoder(Path(temp))
            enc.cex_feedback = 'off'
            enc.tighten_verified_bounds = True
            enc.output_margin_records = [{'class_margins': [{'other_class': j, 'ok': False} for j in (1, 2)]}]
            enc.generate_esbmc_verification_code = Mock(return_value='int main(void){return 0;}')
            def run(path, **kwargs):
                return outcome('FAILED' if 'competitor_1_' in path.name else 'VERIFIED')
            enc._run_esbmc_file.side_effect = run
            args = {k: v for k, v in ARGS.items() if k != 'competitor'}
            result = enc._verify_output_margin_competitors_with_esbmc(
                **args, cur_layer=SimpleNamespace(layer_index=2),
                in_layer=SimpleNamespace(layer_size=2), margin_cuts=[], assumption_box_cardinality='25')
            self.assertEqual(result.status, 'VERIFIED')
            self.assertEqual([r['competitor_class'] for r in result.blocks], [1, 2])
            self.assertEqual(enc.esbmc_call_records[0]['status'], 'FAILED')
            self.assertTrue(enc.esbmc_call_records[0]['resolved_by_affine_residual'])
            self.assertEqual(enc.affine_residual_summary()['verified_comparisons'], 1)
            self.assertEqual(result.resource_control['affine_residual_comparisons'][0]['initial_status'], 'FAILED')

    def test_refinement_variants_are_not_merged_in_tables(self):
        row = dict(epsilon=4, block_size=1, margin_cuts=False,
                   verification_mode='fixed_qif_check', fixed_artifact_source_sha256='same',
                   generated_source_sha256='same', sample={'id':'same'},
                   source_region={'status':'VERIFIED'}, calls=[], total_runtime_seconds=1.,
                   final_status='VERIFIED', byte_crop_property_verified=True,
                   encoded_contract_verified=True, android_transfer_verified=False)
        summaries = summarize([row, {**row, 'output_refinement':'affine_residual'}])
        self.assertEqual(len(summaries), 2)
        self.assertEqual({s['output_refinement'] for s in summaries}, {'none','affine_residual'})

    def test_preparer_preserves_formats_and_only_expands_requested_regions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = dict(fixed_artifact='artifact.json', fixed_artifact_sha256='hash', config={'verification':{}},
                        runs=[dict(run_id='r', sample={'id':'x'}, epsilon=4, block_size=1, margin_cuts=False,
                                   fixed_qif=['unchanged'])])
            (root/'base.json').write_text(json.dumps(base))
            config = dict(schema='ssv_output_refinement_study_v1', base_study=str(root/'base.json'),
                          run_ids=['r'], variants=['box','affine_residual'], verification={'timeout_seconds':300})
            (root/'config.json').write_text(json.dumps(config))
            with patch('scripts.prepare_ssv_refinement.load_artifact', return_value=({'qif':['unchanged']},None)), \
                 patch('scripts.prepare_ssv_refinement.sha256', return_value='hash'), \
                 patch('scripts.prepare_ssv_refinement.validate_config'):
                study = prepare(root/'config.json', root/'prepared')
                self.assertEqual(len(study['runs']), 2)
                self.assertEqual({r['sample']['id'] for r in study['runs']}, {'x'})
                self.assertTrue(all(r['fixed_qif']==['unchanged'] for r in study['runs']))
                with self.assertRaises(FileExistsError):
                    prepare(root/'config.json', root/'prepared')


if __name__ == '__main__':
    unittest.main()
