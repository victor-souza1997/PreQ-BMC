"""Independent enumeration and solver rejection tests for residual certificates."""
import copy
import itertools
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from verification.affine_residual import (
    ResidualCertificateUnsupported, build_certificate, render_certificate, render_kernel_lemma,
)
from verification.esbmc import ESBMCConfig, ESBMCRunner
from verification.replay import LayerReplayFormat, replay_on_python


def certificate(**changes):
    args = dict(input_low=[0], input_high=[4], hidden_weights=[[1], [1]],
                hidden_biases=[4, 4], hidden_bits=8, input_fractional_bits=0,
                hidden_fractional_bits=0, output_weights=[[2, -2], [0, 0]],
                output_biases=[1, 0], output_bits=8, target=0, competitor=1)
    args.update(changes)
    return build_certificate(**args)


class AffineResidualTest(unittest.TestCase):
    def test_shared_inputs_recover_margin_that_independent_hidden_box_loses(self):
        c = certificate()
        self.assertTrue(c['sufficient'])
        self.assertEqual(c['upper_numerator'], -1)
        # Independent hidden coordinates [4,8] permit target=-7, competitor=0.
        self.assertLess(2*4-2*8+1, 0)

    def test_exhaustive_soundness_with_rounding_unstable_relu_and_saturation(self):
        regimes = set()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            w = rng.integers(-6, 7, (3, 2)); b = rng.integers(-4, 5, 3)
            v = rng.integers(-6, 7, (2, 3)); bias = rng.integers(-4, 5, 2)
            f0, f1, q0, q1 = seed % 3, (seed//3) % 3, 3+seed%3, 3+(seed//3)%3
            c = certificate(input_low=[-2, -1], input_high=[2, 2], hidden_weights=w,
                            hidden_biases=b, output_weights=v, output_biases=bias,
                            input_fractional_bits=f0, hidden_fractional_bits=f1,
                            hidden_bits=q0, output_bits=q1)
            for r in c['hidden_lemmas']:
                if r['pre_low'] < 0 < r['pre_high']: regimes.add('unstable_relu')
                if r['pre_high'] > (1 << (q0-1))-1: regimes.add('hidden_saturation')
            for r in c['output_lemmas']:
                if r['correction_low'] or r['correction_high']: regimes.add('output_saturation')
            for point in itertools.product(range(-2, 3), range(-1, 3)):
                h = replay_on_python(point, SimpleNamespace(weights_int=w, biases_int=b),
                                     LayerReplayFormat(f0, q0, True))
                o = replay_on_python(h, SimpleNamespace(weights_int=v, biases_int=bias),
                                     LayerReplayFormat(f1, q1, False))
                diff = int(o[1])-int(o[0])
                self.assertLessEqual(diff*c['denominator'], c['upper_numerator'], (seed, point, c))
        self.assertEqual(regimes, {'unstable_relu', 'hidden_saturation', 'output_saturation'})

    def test_hidden_saturation_correction_prevents_false_certificate(self):
        c = certificate(input_low=[4], input_high=[4], hidden_weights=[[1]], hidden_biases=[0],
                        hidden_bits=3, output_weights=[[1], [0]], output_biases=[0, 3])
        self.assertLess(c['affine_high'], 0)
        self.assertEqual(c['upper_numerator'], 0)
        self.assertFalse(c['sufficient'])

    def test_output_saturation_correction_prevents_false_certificate(self):
        c = certificate(input_low=[1], input_high=[1], hidden_weights=[[1]], hidden_biases=[0],
                        output_weights=[[6], [4]], output_biases=[0, 0], output_bits=3)
        self.assertLess(c['affine_high'], 0)
        self.assertEqual(c['upper_numerator'], 0)
        self.assertFalse(c['sufficient'])

    def test_rounding_bound_prevents_false_certificate(self):
        # With accumulator 1 and S=4 the target rounds to zero: it ties class 1.
        c = certificate(input_low=[1], input_high=[1], hidden_weights=[[1]], hidden_biases=[0],
                        input_fractional_bits=2, output_weights=[[1], [0]], output_biases=[0, 0])
        self.assertLess(c['affine_high'], 0)
        self.assertFalse(c['sufficient'])

    def test_unsupported_numeric_domain_is_rejected(self):
        with self.assertRaises(ResidualCertificateUnsupported):
            certificate(input_fractional_bits=62)
        with self.assertRaises(ResidualCertificateUnsupported):
            certificate(hidden_weights=[[1 << 60], [1]])
        with self.assertRaises(ValueError):
            certificate(input_low=[5], input_high=[4])


ESBMC = os.environ.get('ESBMC_EXECUTABLE') or shutil.which('esbmc')


@unittest.skipUnless(ESBMC, 'ESBMC is not installed')
class AffineResidualESBMCTest(unittest.TestCase):
    def check(self, source):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'proof.c'
            path.write_text(source)
            runner = ESBMCRunner(ESBMCConfig(executable=ESBMC, default_profile='paper-z3', timeout_seconds=30))
            return runner.run_file(path).status

    def test_composition_and_corrupted_coefficient(self):
        c = certificate()
        self.assertEqual(self.check(render_certificate(c)), 'VERIFIED')
        c['composed_coefficients'][0] += 1
        self.assertEqual(self.check(render_certificate(c)), 'FAILED')

    def test_kernel_lemma_and_corrupted_bound(self):
        c = certificate(input_fractional_bits=2)
        r = copy.deepcopy(c['hidden_lemmas'][0])
        self.assertEqual(self.check(render_kernel_lemma(r)), 'VERIFIED')
        r['value_high'] = r['value_low']-1
        self.assertEqual(self.check(render_kernel_lemma(r)), 'FAILED')

    def test_activation_and_output_saturation_lemmas(self):
        c = certificate(input_low=[-8], input_high=[8], hidden_bits=3, hidden_biases=[0, 0], output_bits=3)
        for r in c['hidden_lemmas'][:1] + c['output_lemmas']:
            self.assertEqual(self.check(render_kernel_lemma(r)), 'VERIFIED')

    def test_real_violation_is_not_certified(self):
        c = certificate(output_biases=[-1, 0])
        self.assertFalse(c['sufficient'])
        self.assertEqual(self.check(render_certificate(c)), 'FAILED')


if __name__ == '__main__':
    unittest.main()
