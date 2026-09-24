"""ESBMC harnesses for untrusted input-affine convolution certificates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from verification.arith_kernel import render_arith_kernel

SCHEMA = "input_affine_envelope_block_probe_v1"


def _ints(values: Iterable[Any], name: str) -> list[int]:
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain integers")
        result.append(int(value))
    return result


def _form(form: dict[str, Any], input_size: int, name: str) -> dict[str, Any]:
    indices = _ints(form.get("input_indices", []), f"{name}.input_indices")
    coefficients = _ints(form.get("coefficients", []), f"{name}.coefficients")
    constant, scale_bits = form.get("constant"), form.get("scale_bits")
    if (len(indices) != len(coefficients) or len(set(indices)) != len(indices)
            or any(index < 0 or index >= input_size for index in indices)):
        raise ValueError(f"{name} has invalid terms")
    if isinstance(constant, bool) or not isinstance(constant, int):
        raise ValueError(f"{name}.constant must be an integer")
    if isinstance(scale_bits, bool) or not isinstance(scale_bits, int) \
            or not 0 <= scale_bits <= 62:
        raise ValueError(f"{name}.scale_bits is unsupported")
    return {"input_indices": indices, "coefficients": coefficients,
            "constant": int(constant), "scale_bits": int(scale_bits)}


def load_affine_certificate(path: Path | str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported affine certificate schema: {data.get('schema')!r}")
    box = data.get("input_box", {})
    low = _ints(box.get("low", []), "input_box.low")
    high = _ints(box.get("high", []), "input_box.high")
    if not low or len(low) != len(high) or any(a > b for a, b in zip(low, high)):
        raise ValueError("Invalid encoded input box")
    qif = data.get("qif", {})
    total_bits, fractional_bits = qif.get("total_bits"), qif.get("fractional_bits")
    if not isinstance(total_bits, int) or not 2 <= total_bits <= 62:
        raise ValueError("Unsupported total_bits")
    if not isinstance(fractional_bits, int) or not 0 <= fractional_bits < total_bits:
        raise ValueError("Unsupported fractional_bits")
    previous, block = data.get("previous_layer_envelopes"), data.get("block")
    if not isinstance(previous, list) or not previous or not isinstance(block, list) or not block:
        raise ValueError("Certificate needs predecessor envelopes and a block")
    previous_indices, scales, normalized_previous = set(), set(), []
    for position, row in enumerate(previous):
        neuron = row.get("neuron_index")
        bounds = _ints(row.get("box", []), f"previous[{position}].box")
        if (not isinstance(neuron, int) or neuron < 0 or neuron in previous_indices
                or len(bounds) != 2 or bounds[0] > bounds[1]):
            raise ValueError(f"Invalid previous envelope {position}")
        previous_indices.add(neuron)
        lower = _form(row.get("h_lower", {}), len(low), f"previous[{position}].h_lower")
        upper = _form(row.get("h_upper", {}), len(low), f"previous[{position}].h_upper")
        scales.update((lower["scale_bits"], upper["scale_bits"]))
        normalized_previous.append({"neuron_index": neuron, "box": bounds,
                                    "h_lower": lower, "h_upper": upper})
    normalized_block, block_indices = [], set()
    for position, row in enumerate(block):
        neuron, bias = row.get("neuron_index"), row.get("bias")
        bounds = _ints(row.get("pre_clamp_bounds", []), f"block[{position}].pre_clamp_bounds")
        box_bounds = _ints(row.get("esbmc_box_pre_activation", []),
                           f"block[{position}].esbmc_box_pre_activation")
        weights = row.get("weights", {})
        inputs = _ints(weights.get("input_neurons", []), f"block[{position}].weight inputs")
        values = _ints(weights.get("values", []), f"block[{position}].weight values")
        regime, rule = row.get("relu_regime"), row.get("relu_lower_rule")
        if (not isinstance(neuron, int) or neuron < 0 or neuron in block_indices
                or not isinstance(bias, int) or len(bounds) != 2 or bounds[0] > bounds[1]
                or len(box_bounds) != 2 or box_bounds[0] > box_bounds[1]
                or regime not in {"active", "dead", "unstable"}
                or rule not in {"h>=z", "h>=0"} or not inputs
                or len(inputs) != len(values) or len(set(inputs)) != len(inputs)
                or any(index not in previous_indices for index in inputs)):
            raise ValueError(f"Invalid block neuron {position}")
        block_indices.add(neuron)
        forms = {key: _form(row.get(key, {}), len(low), f"block[{position}].{key}")
                 for key in ("z_lower", "z_upper", "h_lower", "h_upper")}
        scales.update(form["scale_bits"] for form in forms.values())
        normalized_block.append({"neuron_index": neuron, "bias": int(bias),
            "pre_clamp_bounds": bounds, "esbmc_box_pre_activation": box_bounds,
            "relu_regime": regime, "relu_lower_rule": rule,
            "weights": {"input_neurons": inputs, "values": values}, **forms})
    if len(scales) != 1:
        raise ValueError("Certificate block must use one scale")
    return {**data, "input_box": {"low": low, "high": high},
            "qif": {**qif, "total_bits": total_bits, "fractional_bits": fractional_bits},
            "previous_layer_envelopes": normalized_previous, "block": normalized_block,
            "scale_bits": scales.pop()}


def render_affine_rounding_error_lemma(certificate: dict[str, Any]) -> str:
    """Prove the exact half-away error bound over this block's MAC envelope."""
    previous = {row["neuron_index"]: row["box"]
                for row in certificate["previous_layer_envelopes"]}
    envelope = 0
    for row in certificate["block"]:
        bound = 0
        for index, weight in zip(row["weights"]["input_neurons"],
                                 row["weights"]["values"], strict=True):
            low, high = previous[index]
            bound += max(abs(int(weight) * low), abs(int(weight) * high))
        envelope = max(envelope, bound)
    if envelope >= 1 << 63:
        raise ValueError("Affine rounding lemma requires an int64 accumulator witness")
    fractional_bits = certificate["qif"]["fractional_bits"]
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
{render_arith_kernel()}
int main(void) {{
    int64_t acc = nondet_long_long();
    const __int128 denominator = ((__int128)1 << {fractional_bits});
    __ESBMC_assume(acc >= -{envelope} && acc <= {envelope});
    __int128 rounded = div_round_half_away_from_zero_i128(acc, denominator);
    __int128 error = denominator * rounded - acc;
    __ESBMC_assert(error >= -denominator / 2 && error <= denominator / 2,
                   "round-half-away error is at most one half unit");
    return 0;
}}
"""


def render_integer_relu_hull_lemma(total_bits: int) -> str:
    if not 2 <= total_bits <= 62:
        raise ValueError("Unsupported ReLU lemma width")
    low, high = -(1 << (total_bits - 1)), (1 << (total_bits - 1)) - 1
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
int main(void) {{
    int64_t lower = nondet_long_long();
    int64_t upper = nondet_long_long();
    int64_t value = nondet_long_long();
    __ESBMC_assume(lower >= {low} && upper <= {high});
    __ESBMC_assume(lower < 0 && upper > 0 && lower <= value && value <= upper);
    __int128 relu = value < 0 ? 0 : value;
    __ESBMC_assert(relu >= 0 && relu >= value, "integer ReLU lower hull");
    __ESBMC_assert(((__int128)upper - lower) * relu
                       <= (__int128)upper * (value - lower),
                   "integer ReLU upper hull");
    return 0;
}}
"""


def render_bounded_relu_hull_lemma(certificate: dict[str, Any]) -> str:
    """Prove the ReLU hull instances with each certificate's constant bounds."""
    unstable = [row for row in certificate["block"] if row["relu_regime"] == "unstable"]
    if not unstable:
        return "#include <stdint.h>\nint main(void) { return 0; }\n"
    return f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
#define AFFINE_BLOCK_SIZE {len(unstable)}
static const int64_t LOWER[AFFINE_BLOCK_SIZE] = {_c_array(row['pre_clamp_bounds'][0] for row in unstable)};
static const int64_t UPPER[AFFINE_BLOCK_SIZE] = {_c_array(row['pre_clamp_bounds'][1] for row in unstable)};
int main(void) {{
    for (int out = 0; out < AFFINE_BLOCK_SIZE; ++out) {{
        int64_t value = nondet_long_long();
        __ESBMC_assume(LOWER[out] < 0 && UPPER[out] > 0);
        __ESBMC_assume(value >= LOWER[out] && value <= UPPER[out]);
        __int128 relu = value < 0 ? 0 : value;
        __ESBMC_assert(relu >= 0 && relu >= value, "bounded integer ReLU lower hull");
        __ESBMC_assert(((__int128)UPPER[out] - LOWER[out]) * relu
                           <= (__int128)UPPER[out] * (value - LOWER[out]),
                       "bounded integer ReLU upper hull");
    }}
    return 0;
}}
"""


def slice_affine_certificate(certificate: dict[str, Any], positions: Iterable[int]):
    """Return a block subset and only the predecessor envelopes it references."""
    selected = [certificate["block"][int(position)] for position in positions]
    if not selected:
        raise ValueError("An affine certificate slice cannot be empty")
    required = {index for row in selected for index in row["weights"]["input_neurons"]}
    previous = [row for row in certificate["previous_layer_envelopes"]
                if row["neuron_index"] in required]
    if {row["neuron_index"] for row in previous} != required:
        raise ValueError("Certificate slice is missing predecessor envelopes")
    return {**certificate, "previous_layer_envelopes": previous, "block": selected}


def _c_array(values: Iterable[int]) -> str:
    values = list(values)
    return "{" + ", ".join(str(int(value)) for value in (values or [0])) + "}"


def _support(certificate):
    indices = set()
    for row in certificate["previous_layer_envelopes"]:
        for key in ("h_lower", "h_upper"):
            indices.update(row[key]["input_indices"])
    for row in certificate["block"]:
        for key in ("z_lower", "z_upper", "h_lower", "h_upper"):
            indices.update(row[key]["input_indices"])
    support = sorted(indices) or [0]
    return support, {index: position for position, index in enumerate(support)}


def _form_arrays(prefix, rows, key, positions):
    offsets, indices, coefficients, constants = [0], [], [], []
    for row in rows:
        form = row[key]
        constants.append(form["constant"])
        indices.extend(positions[index] for index in form["input_indices"])
        coefficients.extend(form["coefficients"])
        offsets.append(len(indices))
    return f"""static const int {prefix}_OFFSET[{len(offsets)}] = {_c_array(offsets)};
static const int {prefix}_INPUT[{max(1, len(indices))}] = {_c_array(indices)};
static const int64_t {prefix}_COEFF[{max(1, len(coefficients))}] = {_c_array(coefficients)};
static const int64_t {prefix}_CONST[{len(constants)}] = {_c_array(constants)};
"""


def _common(certificate):
    previous, block = certificate["previous_layer_envelopes"], certificate["block"]
    support, positions = _support(certificate)
    previous_position = {row["neuron_index"]: i for i, row in enumerate(previous)}
    weight_offsets, weight_previous, weight_values = [0], [], []
    for row in block:
        weight_previous.extend(previous_position[i] for i in row["weights"]["input_neurons"])
        weight_values.extend(row["weights"]["values"])
        weight_offsets.append(len(weight_values))
    regimes = {"dead": -1, "unstable": 0, "active": 1}
    qif = certificate["qif"]
    code = f"""#include <stdint.h>
void __ESBMC_assume(_Bool); void __ESBMC_assert(_Bool, const char *);
long long nondet_long_long(void);
{render_arith_kernel()}
#define AFFINE_SUPPORT_SIZE {len(support)}
#define AFFINE_PREVIOUS_COUNT {len(previous)}
#define AFFINE_BLOCK_SIZE {len(block)}
#define AFFINE_MAX_FORM_TERMS {max(len(row[key]['input_indices']) for row in [*previous, *block] for key in ('h_lower', 'h_upper') if key in row)}
#define AFFINE_MAX_WEIGHT_TERMS {max(len(row['weights']['values']) for row in block)}
#define AFFINE_SCALE_BITS {certificate['scale_bits']}
#define AFFINE_TOTAL_BITS {qif['total_bits']}
#define AFFINE_FRACTIONAL_BITS {qif['fractional_bits']}
static const int64_t INPUT_LOW[AFFINE_SUPPORT_SIZE] = {_c_array(certificate['input_box']['low'][i] for i in support)};
static const int64_t INPUT_HIGH[AFFINE_SUPPORT_SIZE] = {_c_array(certificate['input_box']['high'][i] for i in support)};
static const int64_t PREVIOUS_LOW[AFFINE_PREVIOUS_COUNT] = {_c_array(row['box'][0] for row in previous)};
static const int64_t PREVIOUS_HIGH[AFFINE_PREVIOUS_COUNT] = {_c_array(row['box'][1] for row in previous)};
static const int WEIGHT_OFFSET[AFFINE_BLOCK_SIZE + 1] = {_c_array(weight_offsets)};
static const int WEIGHT_PREVIOUS[{len(weight_previous)}] = {_c_array(weight_previous)};
static const int64_t WEIGHT_VALUE[{len(weight_values)}] = {_c_array(weight_values)};
static const int64_t BIAS[AFFINE_BLOCK_SIZE] = {_c_array(row['bias'] for row in block)};
static const int64_t PRE_CLAMP_LOW[AFFINE_BLOCK_SIZE] = {_c_array(row['pre_clamp_bounds'][0] for row in block)};
static const int64_t PRE_CLAMP_HIGH[AFFINE_BLOCK_SIZE] = {_c_array(row['pre_clamp_bounds'][1] for row in block)};
static const int64_t BOX_PRE_LOW[AFFINE_BLOCK_SIZE] = {_c_array(row['esbmc_box_pre_activation'][0] for row in block)};
static const int64_t BOX_PRE_HIGH[AFFINE_BLOCK_SIZE] = {_c_array(row['esbmc_box_pre_activation'][1] for row in block)};
static const int RELU_REGIME[AFFINE_BLOCK_SIZE] = {_c_array(regimes[row['relu_regime']] for row in block)};
static const int RELU_LOWER_USES_Z[AFFINE_BLOCK_SIZE] = {_c_array(row['relu_lower_rule'] == 'h>=z' for row in block)};
"""
    for prefix, rows, key in (("PREV_L", previous, "h_lower"),
                              ("PREV_U", previous, "h_upper"),
                              ("Z_L", block, "z_lower"), ("Z_U", block, "z_upper"),
                              ("H_L", block, "h_lower"), ("H_U", block, "h_upper")):
        code += _form_arrays(prefix, rows, key, positions)
    code += r"""
static __int128 eval_form(int form, const int *offset, const int *input,
        const int64_t *coefficient, const int64_t *constant, const int64_t *x) {
    __int128 value = constant[form];
    for (int term = offset[form]; term < offset[form + 1]; ++term)
        value += (__int128)coefficient[term] * x[input[term]];
    return value;
}
static __int128 floor_div(__int128 value, __int128 denominator) {
    __int128 quotient = value / denominator, remainder = value % denominator;
    return quotient - (remainder != 0 && value < 0);
}
static __int128 ceil_div(__int128 value, __int128 denominator) {
    return -floor_div(-value, denominator);
}
"""
    return code


def render_symbolic_affine_block(certificate: dict[str, Any]) -> str:
    """Option A: execute the exact transition under input-relative assumptions."""
    code = _common(certificate)
    return code + r"""
int main(void) {
    int64_t x[AFFINE_SUPPORT_SIZE], previous[AFFINE_PREVIOUS_COUNT];
    const __int128 scale = ((__int128)1 << AFFINE_SCALE_BITS);
    for (int i = 0; i < AFFINE_SUPPORT_SIZE; ++i) {
        x[i] = nondet_long_long();
        __ESBMC_assume(x[i] >= INPUT_LOW[i] && x[i] <= INPUT_HIGH[i]);
    }
    for (int i = 0; i < AFFINE_PREVIOUS_COUNT; ++i) {
        previous[i] = nondet_long_long();
        __ESBMC_assume(previous[i] >= PREVIOUS_LOW[i] && previous[i] <= PREVIOUS_HIGH[i]);
        __int128 lower = eval_form(i, PREV_L_OFFSET, PREV_L_INPUT, PREV_L_COEFF, PREV_L_CONST, x);
        __int128 upper = eval_form(i, PREV_U_OFFSET, PREV_U_INPUT, PREV_U_COEFF, PREV_U_CONST, x);
        __ESBMC_assume(lower <= scale * previous[i] && scale * previous[i] <= upper);
    }
    for (int out = 0; out < AFFINE_BLOCK_SIZE; ++out) {
        __int128 acc = 0;
        for (int term = WEIGHT_OFFSET[out]; term < WEIGHT_OFFSET[out + 1]; ++term)
            acc = mac_i128(acc, WEIGHT_VALUE[term], previous[WEIGHT_PREVIOUS[term]]);
        __int128 raw = div_round_half_away_from_zero_i128(
            acc, ((__int128)1 << AFFINE_FRACTIONAL_BITS)) + BIAS[out];
        __ESBMC_assert(raw >= -((__int128)1 << (AFFINE_TOTAL_BITS - 1)) &&
                       raw <= (((__int128)1 << (AFFINE_TOTAL_BITS - 1)) - 1), "clamp inactivity");
        __ESBMC_assert(raw >= PRE_CLAMP_LOW[out] && raw <= PRE_CLAMP_HIGH[out], "scalar pre-clamp bounds");
        __int128 zl = eval_form(out, Z_L_OFFSET, Z_L_INPUT, Z_L_COEFF, Z_L_CONST, x);
        __int128 zu = eval_form(out, Z_U_OFFSET, Z_U_INPUT, Z_U_COEFF, Z_U_CONST, x);
        __ESBMC_assert(zl <= scale * raw && scale * raw <= zu, "pre-activation envelope");
        __int128 value = clamp_to_signed_range_i128(raw, AFFINE_TOTAL_BITS);
        if (value < 0) value = 0;
        value = clamp_to_signed_range_i128(value, AFFINE_TOTAL_BITS);
        __int128 hl = eval_form(out, H_L_OFFSET, H_L_INPUT, H_L_COEFF, H_L_CONST, x);
        __int128 hu = eval_form(out, H_U_OFFSET, H_U_INPUT, H_U_COEFF, H_U_CONST, x);
        __ESBMC_assert(hl <= scale * value && scale * value <= hu, "activation envelope");
    }
    return 0;
}
"""


def render_recomputed_affine_block(certificate: dict[str, Any]) -> str:
    """Option B: validate the certificate algebra and scalar regimes."""
    code = _common(certificate)
    return code + r"""
static void clear_form(__int128 *coefficient, __int128 *constant) {
    *constant = 0;
    for (int i = 0; i < AFFINE_SUPPORT_SIZE; ++i) coefficient[i] = 0;
}
static void add_form(__int128 *coefficient, __int128 *constant, __int128 multiplier,
        int form, const int *offset, const int *input, const int64_t *value,
        const int64_t *form_constant) {
    *constant += multiplier * form_constant[form];
    for (int term = offset[form]; term < offset[form + 1]; ++term)
        coefficient[input[term]] += multiplier * value[term];
}
static __int128 extreme(const __int128 *coefficient, __int128 constant, _Bool minimum) {
    __int128 value = constant;
    for (int i = 0; i < AFFINE_SUPPORT_SIZE; ++i) {
        int64_t endpoint = minimum ? (coefficient[i] >= 0 ? INPUT_LOW[i] : INPUT_HIGH[i])
                                   : (coefficient[i] >= 0 ? INPUT_HIGH[i] : INPUT_LOW[i]);
        value += coefficient[i] * endpoint;
    }
    return value;
}
static void difference(__int128 *coefficient, __int128 *constant,
        __int128 lm, int left, const int *lo, const int *li, const int64_t *lv, const int64_t *lc,
        __int128 rm, int right, const int *ro, const int *ri, const int64_t *rv, const int64_t *rc) {
    clear_form(coefficient, constant);
    add_form(coefficient, constant, lm, left, lo, li, lv, lc);
    add_form(coefficient, constant, rm, right, ro, ri, rv, rc);
}
int main(void) {
    const __int128 scale = ((__int128)1 << AFFINE_SCALE_BITS);
    const __int128 denominator = ((__int128)1 << AFFINE_FRACTIONAL_BITS);
    const __int128 half_error = scale * denominator / 2;
    const __int128 q_low = -((__int128)1 << (AFFINE_TOTAL_BITS - 1));
    const __int128 q_high = ((__int128)1 << (AFFINE_TOTAL_BITS - 1)) - 1;
    __int128 coefficient[AFFINE_SUPPORT_SIZE], constant;
    for (int out = 0; out < AFFINE_BLOCK_SIZE; ++out) {
        clear_form(coefficient, &constant);
        for (int term = WEIGHT_OFFSET[out]; term < WEIGHT_OFFSET[out + 1]; ++term) {
            int p = WEIGHT_PREVIOUS[term]; int64_t w = WEIGHT_VALUE[term];
            if (w >= 0) add_form(coefficient, &constant, w, p, PREV_L_OFFSET, PREV_L_INPUT, PREV_L_COEFF, PREV_L_CONST);
            else add_form(coefficient, &constant, w, p, PREV_U_OFFSET, PREV_U_INPUT, PREV_U_COEFF, PREV_U_CONST);
        }
        add_form(coefficient, &constant, -denominator, out, Z_L_OFFSET, Z_L_INPUT, Z_L_COEFF, Z_L_CONST);
        constant += scale * denominator * BIAS[out] - half_error;
        __ESBMC_assert(extreme(coefficient, constant, 1) >= 0, "lower affine derivation");
        clear_form(coefficient, &constant);
        for (int term = WEIGHT_OFFSET[out]; term < WEIGHT_OFFSET[out + 1]; ++term) {
            int p = WEIGHT_PREVIOUS[term]; int64_t w = WEIGHT_VALUE[term];
            if (w >= 0) add_form(coefficient, &constant, -w, p, PREV_U_OFFSET, PREV_U_INPUT, PREV_U_COEFF, PREV_U_CONST);
            else add_form(coefficient, &constant, -w, p, PREV_L_OFFSET, PREV_L_INPUT, PREV_L_COEFF, PREV_L_CONST);
        }
        add_form(coefficient, &constant, denominator, out, Z_U_OFFSET, Z_U_INPUT, Z_U_COEFF, Z_U_CONST);
        constant += -scale * denominator * BIAS[out] - half_error;
        __ESBMC_assert(extreme(coefficient, constant, 1) >= 0, "upper affine derivation");
        clear_form(coefficient, &constant);
        add_form(coefficient, &constant, 1, out, Z_L_OFFSET, Z_L_INPUT, Z_L_COEFF, Z_L_CONST);
        __int128 relation_low = ceil_div(extreme(coefficient, constant, 1), scale);
        clear_form(coefficient, &constant);
        add_form(coefficient, &constant, 1, out, Z_U_OFFSET, Z_U_INPUT, Z_U_COEFF, Z_U_CONST);
        __int128 relation_high = floor_div(extreme(coefficient, constant, 0), scale);
        __ESBMC_assert(relation_low >= q_low && relation_high <= q_high, "clamp inactivity");
        __int128 low = relation_low > BOX_PRE_LOW[out] ? relation_low : BOX_PRE_LOW[out];
        __int128 high = relation_high < BOX_PRE_HIGH[out] ? relation_high : BOX_PRE_HIGH[out];
        __ESBMC_assert(low == PRE_CLAMP_LOW[out] && high == PRE_CLAMP_HIGH[out] && low <= high,
                       "checked scalar bounds");
        if (RELU_REGIME[out] > 0) __ESBMC_assert(low >= 0, "active regime");
        else if (RELU_REGIME[out] < 0) __ESBMC_assert(high <= 0, "dead regime");
        else __ESBMC_assert(low < 0 && high > 0, "unstable regime");
        if (RELU_REGIME[out] > 0 || (RELU_REGIME[out] == 0 && RELU_LOWER_USES_Z[out]))
            difference(coefficient, &constant, 1, out, Z_L_OFFSET, Z_L_INPUT, Z_L_COEFF, Z_L_CONST,
                       -1, out, H_L_OFFSET, H_L_INPUT, H_L_COEFF, H_L_CONST);
        else { clear_form(coefficient, &constant); add_form(coefficient, &constant, -1, out,
                    H_L_OFFSET, H_L_INPUT, H_L_COEFF, H_L_CONST); }
        __ESBMC_assert(extreme(coefficient, constant, 1) >= 0, "lower ReLU derivation");
        if (RELU_REGIME[out] > 0)
            difference(coefficient, &constant, 1, out, H_U_OFFSET, H_U_INPUT, H_U_COEFF, H_U_CONST,
                       -1, out, Z_U_OFFSET, Z_U_INPUT, Z_U_COEFF, Z_U_CONST);
        else if (RELU_REGIME[out] < 0) { clear_form(coefficient, &constant); add_form(coefficient, &constant, 1, out,
                    H_U_OFFSET, H_U_INPUT, H_U_COEFF, H_U_CONST); }
        else {
            difference(coefficient, &constant, high - low, out, H_U_OFFSET, H_U_INPUT, H_U_COEFF, H_U_CONST,
                       -high, out, Z_U_OFFSET, Z_U_INPUT, Z_U_COEFF, Z_U_CONST);
            constant += high * scale * low;
        }
        __ESBMC_assert(extreme(coefficient, constant, 1) >= 0, "upper ReLU derivation");
    }
    return 0;
}
"""
