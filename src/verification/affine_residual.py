"""Compositional integer certificates for an affine/ReLU/affine network.

The input-dependent affine terms are composed before interval maximization.
Kernel rounding and activation corrections are separate, ESBMC-checked lemmas.
All certificate construction uses integers; numerical MILP bounds are not used.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

import numpy as np

from utils.fixed_point import clamp_to_signed_range, round_divide_half_away_from_zero
from verification.arith_kernel import render_arith_kernel
from verification.invariants import exact_layer_interval


class ResidualCertificateUnsupported(ValueError):
    """The certificate checker cannot safely encode this instance."""


def _integers(values: Any) -> list:
    array = np.asarray(values, dtype=object)
    if any(int(v) != v for v in array.flat):
        raise ValueError("Residual certificates require integer parameters")
    if any(abs(int(v)) >= 1 << 60 for v in array.flat):
        raise ResidualCertificateUnsupported("Certificate data exceeds the signed-60-bit domain")
    return np.vectorize(int, otypes=[object])(array).tolist()


def _endpoints(row, low, high):
    return (
        sum(w * (a if w >= 0 else b) for w, a, b in zip(row, low, high)),
        sum(w * (b if w >= 0 else a) for w, a, b in zip(row, low, high)),
    )


def _lemma(acc_low, acc_high, bias, scale, bits, relu):
    p_low = round_divide_half_away_from_zero(acc_low, scale) + bias
    p_high = round_divide_half_away_from_zero(acc_high, scale) + bias
    lower = 0 if relu else -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    clip = lambda p: max(lower, min(upper, p))
    # clip(p)-p is monotone non-increasing, including unstable ReLU regions.
    result = dict(acc_low=acc_low, acc_high=acc_high, bias=bias, scale=scale,
                  bits=bits, relu=relu, pre_low=p_low, pre_high=p_high,
                  value_low=clip(p_low), value_high=clip(p_high),
                  correction_low=clip(p_high)-p_high,
                  correction_high=clip(p_low)-p_low)
    _integers(list(result.values()))
    return result


def build_certificate(*, input_low, input_high, hidden_weights, hidden_biases,
                      hidden_bits, input_fractional_bits, hidden_fractional_bits,
                      output_weights, output_biases, output_bits, target, competitor):
    """Build a sufficient bound on output[competitor]-output[target].

Only one hidden affine/ReLU layer is encoded. Negative/unstable activations and
both layers' saturation are retained as bounded corrections, never removed.
"""
    if not (0 <= input_fractional_bits <= 30 and 0 <= hidden_fractional_bits <= 30):
        raise ResidualCertificateUnsupported("Residual scales require fractional bits in [0,30]")
    if not (2 <= hidden_bits <= 60 and 2 <= output_bits <= 60):
        raise ResidualCertificateUnsupported("Residual formats require total bits in [2,60]")
    low, high = _integers(input_low), _integers(input_high)
    w, b = _integers(hidden_weights), _integers(hidden_biases)
    v, c = _integers(output_weights), _integers(output_biases)
    if not low or len(low) != len(high) or any(a > z for a, z in zip(low, high)):
        raise ValueError("Invalid residual input box")
    if not w or len(w) != len(b) or any(len(row) != len(low) for row in w):
        raise ValueError("Invalid hidden affine dimensions")
    if not v or len(v) != len(c) or any(len(row) != len(w) for row in v):
        raise ValueError("Invalid output affine dimensions")
    if not (0 <= target < len(v) and 0 <= competitor < len(v) and target != competitor):
        raise ValueError("Invalid residual target/competitor")
    s0, s1 = 1 << input_fractional_bits, 1 << hidden_fractional_bits
    # Check deployed MAC prefixes, rounding intermediates, and int64 storage.
    hidden_range = exact_layer_interval(
        SimpleNamespace(weights_int=w, biases_int=b), input_low=np.asarray(low),
        input_high=np.asarray(high), input_fractional_bits=input_fractional_bits,
        total_bits=hidden_bits, apply_relu=True)
    output_range = exact_layer_interval(
        SimpleNamespace(weights_int=[v[target], v[competitor]], biases_int=[c[target], c[competitor]]),
        input_low=hidden_range.output_low, input_high=hidden_range.output_high,
        input_fractional_bits=hidden_fractional_bits, total_bits=output_bits, apply_relu=False)
    hidden = [_lemma(int(a), int(z), bias, s0, hidden_bits, True)
              for a, z, bias in zip(hidden_range.accumulator_low, hidden_range.accumulator_high, b)]
    outputs = [_lemma(int(a), int(z), c[j], s1, output_bits, False)
               for a, z, j in zip(output_range.accumulator_low, output_range.accumulator_high,
                                  (target, competitor))]
    d = [a-z for a, z in zip(v[competitor], v[target])]
    coefficients = [sum(d[i]*w[i][k] for i in range(len(w))) for k in range(len(low))]
    _integers(d)
    _integers(coefficients)
    base = s0*sum(a*z for a, z in zip(d, b)) + s0*s1*(c[competitor]-c[target])
    affine_low, affine_high = (q+base for q in _endpoints(coefficients, low, high))
    round_bound = sum(abs(a) for a in d)*(s0//2) + s0*2*(s1//2)
    correction_low, correction_high = _endpoints(
        d, [r['correction_low'] for r in hidden], [r['correction_high'] for r in hidden])
    activation_low = s0*correction_low + s0*s1*(outputs[1]['correction_low']-outputs[0]['correction_high'])
    activation_high = s0*correction_high + s0*s1*(outputs[1]['correction_high']-outputs[0]['correction_low'])
    upper = affine_high + round_bound + activation_high
    # The checker uses int128 intermediates but nondeterministic int64 summaries.
    # A conservative absolute sum also rules out overflow before cancellations.
    absolute = sum(sum(abs(d[i]*w[i][k]) for i in range(len(w)))*max(abs(low[k]), abs(high[k]))
                   for k in range(len(low)))
    absolute += s0*sum(abs(a*z) for a, z in zip(d, b)) + s0*s1*(abs(c[target])+abs(c[competitor]))
    absolute += round_bound + s0*sum(abs(a)*max(abs(r['correction_low']), abs(r['correction_high']))
                                   for a, r in zip(d, hidden))
    absolute += s0*s1*sum(max(abs(r['correction_low']), abs(r['correction_high'])) for r in outputs)
    if absolute >= 1 << 60:
        raise ResidualCertificateUnsupported("Residual certificate intermediate range exceeds signed 60 bits")
    result = dict(schema='affine_residual_certificate_v1', input_low=low, input_high=high,
                  hidden_weights=w, hidden_biases=b, hidden_bits=hidden_bits,
                  input_scale=s0, hidden_scale=s1, output_bits=output_bits,
                  output_weights=[v[target], v[competitor]], output_biases=[c[target], c[competitor]],
                  target_class=target, competitor_class=competitor, hidden_lemmas=hidden,
                  output_lemmas=outputs, composed_coefficients=coefficients,
                  affine_low=affine_low, affine_high=affine_high, rounding_bound=round_bound,
                  activation_low=activation_low, activation_high=activation_high,
                  upper_numerator=upper, denominator=s0*s1,
                  integer_difference_upper=upper//(s0*s1), sufficient=upper < 0,
                  arithmetic_safety=[hidden_range.arithmetic_safety, output_range.arithmetic_safety])
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def _c(values):
    if isinstance(values, list):
        return '{' + ', '.join(_c(v) for v in values) + '}'
    return str(int(values)) + 'LL'


def _preamble():
    return '''#include <stdint.h>
extern long long nondet_longlong(void);
void __ESBMC_assume(_Bool);
void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(c,m) __ESBMC_assert((c),(m))
''' + render_arith_kernel() + '\n'


def render_kernel_lemma(lemma):
    """Prove a scalar contract of the exact shared deployed C kernel."""
    r = lemma
    relu = 'if (value < 0) value = 0;' if r['relu'] else ''
    return _preamble() + f'''
int main(void) {{
    __int128 acc = nondet_longlong();
    __ESBMC_assume(acc >= {_c(r['acc_low'])} && acc <= {_c(r['acc_high'])});
    __int128 rounded = div_round_half_away_from_zero_i128(acc, {_c(r['scale'])});
    __int128 pre = rounded + {_c(r['bias'])};
    __int128 value = clamp_to_signed_range_i128(pre, {r['bits']});
    {relu}
    value = clamp_to_signed_range_i128(value, {r['bits']});
    __int128 residual = {_c(r['scale'])} * rounded - acc;
    __int128 correction = value - pre;
    __ESBMC_assert(residual >= -{r['scale']//2}LL && residual <= {r['scale']//2}LL, "rounding residual");
    __ESBMC_assert(pre >= {_c(r['pre_low'])} && pre <= {_c(r['pre_high'])}, "preclamp interval");
    __ESBMC_assert(value >= {_c(r['value_low'])} && value <= {_c(r['value_high'])}, "kernel interval");
    __ESBMC_assert(correction >= {_c(r['correction_low'])} && correction <= {_c(r['correction_high'])}, "activation correction");
    return 0;
}}
'''


def render_certificate(cert):
    """Check the arithmetic certificate and its universal residual implication.

The assumptions on the three scalar summaries follow from affine box
maximization and the separately verified kernel lemmas. They are not assumed
properties of the network without those prerequisites.
"""
    n, h = len(cert['input_low']), len(cert['hidden_weights'])
    # All ranges are recomputed from the matrices, never accepted as an oracle.
    rows = cert['hidden_lemmas'] + cert['output_lemmas']
    fields = ('acc_low', 'acc_high', 'pre_low', 'pre_high', 'value_low', 'value_high', 'correction_low', 'correction_high')
    declarations = '\n'.join(f'static const int64_t expected_{k}[{h+2}] = {_c([r[k] for r in rows])};' for k in fields)
    return _preamble() + f'''
#define INPUT_SIZE {n}
#define LAYER_SIZE {h}
static const int64_t LOW[INPUT_SIZE] = {_c(cert['input_low'])};
static const int64_t HIGH[INPUT_SIZE] = {_c(cert['input_high'])};
static const int64_t W[LAYER_SIZE][INPUT_SIZE] = {_c(cert['hidden_weights'])};
static const int64_t B[LAYER_SIZE] = {_c(cert['hidden_biases'])};
static const int64_t V[2][LAYER_SIZE] = {_c(cert['output_weights'])};
static const int64_t C[2] = {_c(cert['output_biases'])};
static const int64_t COEFF[INPUT_SIZE] = {_c(cert['composed_coefficients'])};
{declarations}
static void check_range(int index, __int128 low, __int128 high, int64_t bias, int64_t scale, int bits, int relu) {{
    __int128 pl = div_round_half_away_from_zero_i128(low, scale) + bias;
    __int128 ph = div_round_half_away_from_zero_i128(high, scale) + bias;
    __int128 vl = clamp_to_signed_range_i128(pl, bits);
    __int128 vh = clamp_to_signed_range_i128(ph, bits);
    if (relu && vl < 0) vl = 0;
    if (relu && vh < 0) vh = 0;
    __ESBMC_assert(low == expected_acc_low[index] && high == expected_acc_high[index], "accumulator certificate");
    __ESBMC_assert(pl == expected_pre_low[index] && ph == expected_pre_high[index], "preclamp certificate");
    __ESBMC_assert(vl == expected_value_low[index] && vh == expected_value_high[index], "value certificate");
    __ESBMC_assert(vh-ph == expected_correction_low[index] && vl-pl == expected_correction_high[index], "correction certificate");
}}
int main(void) {{
    const __int128 S0 = {_c(cert['input_scale'])}, S1 = {_c(cert['hidden_scale'])};
    __int128 d[LAYER_SIZE];
    __int128 base = S0*S1*((__int128)C[1]-C[0]);
    __int128 rb = S0*2*(S1/2), al = 0, ah = 0;
    for (int i=0; i<LAYER_SIZE; ++i) {{
        __int128 low=0, high=0;
        for (int k=0; k<INPUT_SIZE; ++k) {{
            low += (__int128)W[i][k]*(W[i][k]>=0 ? LOW[k] : HIGH[k]);
            high += (__int128)W[i][k]*(W[i][k]>=0 ? HIGH[k] : LOW[k]);
        }}
        check_range(i,low,high,B[i],(int64_t)S0,{cert['hidden_bits']},1);
        d[i]=(__int128)V[1][i]-V[0][i];
        base += S0*d[i]*B[i];
        rb += (d[i]>=0 ? d[i] : -d[i])*(S0/2);
        al += S0*d[i]*(d[i]>=0 ? expected_correction_low[i] : expected_correction_high[i]);
        ah += S0*d[i]*(d[i]>=0 ? expected_correction_high[i] : expected_correction_low[i]);
    }}
    for (int j=0; j<2; ++j) {{
        __int128 low=0, high=0;
        for (int i=0; i<LAYER_SIZE; ++i) {{
            low += (__int128)V[j][i]*(V[j][i]>=0 ? expected_value_low[i] : expected_value_high[i]);
            high += (__int128)V[j][i]*(V[j][i]>=0 ? expected_value_high[i] : expected_value_low[i]);
        }}
        check_range(LAYER_SIZE+j,low,high,C[j],(int64_t)S1,{cert['output_bits']},0);
    }}
    al += S0*S1*((__int128)expected_correction_low[LAYER_SIZE+1]-expected_correction_high[LAYER_SIZE]);
    ah += S0*S1*((__int128)expected_correction_high[LAYER_SIZE+1]-expected_correction_low[LAYER_SIZE]);
    __int128 fl=base, fh=base;
    for (int k=0; k<INPUT_SIZE; ++k) {{
        __int128 coefficient=0;
        for (int i=0; i<LAYER_SIZE; ++i) coefficient += d[i]*W[i][k];
        __ESBMC_assert(coefficient == COEFF[k], "composed coefficient");
        fl += coefficient*(coefficient>=0 ? LOW[k] : HIGH[k]);
        fh += coefficient*(coefficient>=0 ? HIGH[k] : LOW[k]);
    }}
    __ESBMC_assert(fl == {_c(cert['affine_low'])} && fh == {_c(cert['affine_high'])}, "affine bound");
    __ESBMC_assert(rb == {_c(cert['rounding_bound'])}, "rounding bound");
    __ESBMC_assert(al == {_c(cert['activation_low'])} && ah == {_c(cert['activation_high'])}, "activation bound");
    __ESBMC_assert(fh+rb+ah == {_c(cert['upper_numerator'])}, "margin bound certificate");
    __int128 affine=nondet_longlong(), residual=nondet_longlong(), activation=nondet_longlong();
    __ESBMC_assume(affine>=fl && affine<=fh);
    __ESBMC_assume(residual>=-rb && residual<=rb);
    __ESBMC_assume(activation>=al && activation<=ah);
    __ESBMC_assert(affine+residual+activation < 0, "composed strict target margin");
    return 0;
}}
'''
