from __future__ import annotations

from verification.arith_kernel import render_arith_kernel


def outerlayer_fixed_int(
    in_layer_layer_size: int,
    cur_layer_layer_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_int: str,
    input_bounds_high_int: str,
    targetCls: int,
    scale_factor: int,
    total_bits: int,
    input_scale_factor: int | None = None,
    margin_cut_directions_c_int: str | None = None,
    margin_cut_low_c_int: str | None = None,
    margin_cut_high_c_int: str | None = None,
    margin_cut_scale: int | None = None,
    margin_cut_count: int = 0,
) -> str:
    input_scale = scale_factor if input_scale_factor is None else int(input_scale_factor)
    cuts_enabled = int(margin_cut_count) > 0
    cut_declarations = ""
    cut_application = ""
    if cuts_enabled:
        if (
            margin_cut_directions_c_int is None
            or margin_cut_low_c_int is None
            or margin_cut_high_c_int is None
            or margin_cut_scale is None
        ):
            raise ValueError("Enabled output margin cuts require directions, bounds, and scale.")
        cut_declarations = f"""
#define MARGIN_CUT_COUNT {int(margin_cut_count)}
#define MARGIN_CUT_DIRECTION_SCALE {int(margin_cut_scale)}LL
long long margin_cut_directions[MARGIN_CUT_COUNT][INPUT_SIZE] = {margin_cut_directions_c_int};
long long margin_cut_low[MARGIN_CUT_COUNT] = {margin_cut_low_c_int};
long long margin_cut_high[MARGIN_CUT_COUNT] = {margin_cut_high_c_int};

static void assume_sound_margin_cuts(const long long input[INPUT_SIZE])
{{
    for (int cut = 0; cut < MARGIN_CUT_COUNT; ++cut)
    {{
        __int128 directional_value = 0;
        for (int j = 0; j < INPUT_SIZE; ++j)
        {{
            directional_value = mac_i128(
                directional_value,
                margin_cut_directions[cut][j],
                input[j]
            );
        }}
        __ESBMC_assume(
            directional_value >= (__int128)margin_cut_low[cut] &&
            directional_value <= (__int128)margin_cut_high[cut]
        );
    }}
}}
"""
        cut_application = "    assume_sound_margin_cuts(input);\n"
    return f"""\
#include <stdint.h>
#include <limits.h>


#define INPUT_SIZE   {in_layer_layer_size}
#define LAYER_SIZE   {cur_layer_layer_size}
#define TARGET_CLASS {targetCls}
#define SCALE_FACTOR {scale_factor}LL
#define INPUT_SCALE_FACTOR {input_scale}LL
#define TOTAL_BITS   {total_bits}

extern long long nondet_longlong(void);
void __ESBMC_assume(_Bool);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}
{cut_declarations}

/* Transformacao afim em ponto fixo com escala de entrada explicita. */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{{
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        __int128 acc = 0; /* acumulador em SCALE_FACTOR * INPUT_SCALE_FACTOR */

        for (int j = 0; j < INPUT_SIZE; ++j) {{
            /* w usa SCALE_FACTOR; x usa INPUT_SCALE_FACTOR. */
            acc = mac_i128(acc, weights[i][j], in_[j]);
        }}

        __int128 value = div_round_half_away_from_zero_i128(
            acc,
            (__int128)INPUT_SCALE_FACTOR
        ) + (__int128)biases[i];
        value = clamp_to_signed_range_i128(value, TOTAL_BITS);
        out_[i] = (long long)clamp_to_signed_range_i128(value, TOTAL_BITS);
    }}
}}

/* verifica se o output condiz a classe esperada */
static int verify_classification(const long long out_[LAYER_SIZE])
{{
    const int T = TARGET_CLASS;
    const long long target = out_[T];
    long long max_other = LLONG_MIN / 2;

    int i = 0;

    while (i < LAYER_SIZE)
    {{
        __ESBMC_loop_invariant(0 <= i && i <= LAYER_SIZE && max_other <= target);
        if (i != T) {{
            const long long cand = out_[i];
            if (cand > max_other) {{
                max_other = cand;
            }}
        }}
        ++i;
    }}

    return max_other < target;
}}

int main(void)
{{
    long long input[INPUT_SIZE];
    long long output[LAYER_SIZE];

    /* Entrada nao-deterministica */
    for (int k = 0; k < INPUT_SIZE; ++k) {{
        input[k] = nondet_longlong();
        __ESBMC_assume(input[k] >= input_bounds_low[k] &&
                       input[k] <= input_bounds_high[k]);
    }}

{cut_application}\
    affine_transform_fixed(input, output);

    __ESBMC_assert(verify_classification(output),
                   "Classification property violated (output layer, fixed-point)");

    return 0;
}}
"""

def innerlayer_fixed_int_bounds_only(
    cur_layer_layer_size: int,
    in_layer_layer_size: int,
    weights_c_int: str,
    biases_c_int: str,
    preimage_low_int: str,
    preimage_high_int: str,
    input_bounds_low_int: str,
    input_bounds_high_int: str,
    scale_factor: int,
    total_bits: int,
    activation: str = "none",
    unsound_contract_tolerance: bool = False,
    input_scale_factor: int | None = None,
    contract_tolerance_c_int: str | None = None,
) -> str:
    if activation not in {"none", "relu", "relu6"}:
        raise ValueError(
            "activation must be one of: 'none', 'relu', 'relu6'"
        )

    activation_id = {
        "none": 0,
        "relu": 1,
        "relu6": 2,
    }[activation]
    abs_tol_expr = "(__int128)(SCALE_FACTOR / 1000)" if unsound_contract_tolerance else "0"
    rel_tol_num = 1 if unsound_contract_tolerance else 0
    preimage_tolerance_expr = (
        "(__int128)contract_tolerance[i]"
        if contract_tolerance_c_int is not None
        else (
            "abs_tol + (rel_tol_num * range) / rel_tol_den"
            if unsound_contract_tolerance
            else "0"
        )
    )
    input_scale = scale_factor if input_scale_factor is None else int(input_scale_factor)
    contract_tolerance_declaration = (
        f"long long contract_tolerance[LAYER_SIZE] = {contract_tolerance_c_int};"
        if contract_tolerance_c_int is not None
        else ""
    )

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {in_layer_layer_size}
#define LAYER_SIZE {cur_layer_layer_size}
#define SCALE_FACTOR {scale_factor}LL
#define INPUT_SCALE_FACTOR {input_scale}LL
#define TOTAL_BITS {total_bits}
#define ACTIVATION_KIND {activation_id}

/*
 * ACTIVATION_KIND:
 *   0 = none
 *   1 = ReLU
 *   2 = ReLU6
 *
 * This harness verifies a local preimage contract:
 *
 *   if input_j ∈ [input_bounds_low[j], input_bounds_high[j]]
 *   then layer_output_i ∈ [preimage_low[i], preimage_high[i]]
 *
 * The affine computation is performed over intervals using __int128.
 * Rescaling uses the monotone round-half-away endpoint transform, then applies
 * the same deployed saturation/ReLU/saturation order. This is a conservative
 * over-approximation of the pointwise deployed kernel.
 */

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE] = {biases_c_int};

long long preimage_low[LAYER_SIZE] = {preimage_low_int};
long long preimage_high[LAYER_SIZE] = {preimage_high_int};

long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};
{contract_tolerance_declaration}

/* Failure-trace instrumentation. These values do not alter the property. */
long long counterexample_input[INPUT_SIZE] = {{0}};
int counterexample_neuron = -1;
__int128 counterexample_preclamp = 0;

void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}

static inline __int128 abs_i128(__int128 x)
{{
    return x < 0 ? -x : x;
}}

static inline void clamp_bounds_to_signed_range(__int128 *lb, __int128 *ub, int total_bits)
{{
    *lb = clamp_to_signed_range_i128(*lb, total_bits);
    *ub = clamp_to_signed_range_i128(*ub, total_bits);
    __ESBMC_assert(*lb <= *ub, "invalid interval after clamp");
}}

static inline void apply_activation_bounds(__int128 *lb, __int128 *ub)
{{
    if (ACTIVATION_KIND == 1)
    {{
        /* ReLU interval transformer */
        if (*ub <= 0)
        {{
            *lb = 0;
            *ub = 0;
        }}
        else if (*lb < 0)
        {{
            *lb = 0;
        }}
    }}
    else if (ACTIVATION_KIND == 2)
    {{
        const __int128 six = (__int128)6 * (__int128)SCALE_FACTOR;

        if (*ub <= 0)
        {{
            *lb = 0;
            *ub = 0;
        }}
        else
        {{
            if (*lb < 0)
            {{
                *lb = 0;
            }}
            if (*lb > six)
            {{
                *lb = six;
            }}
            if (*ub > six)
            {{
                *ub = six;
            }}
        }}
    }}

    __ESBMC_assert(*lb <= *ub, "invalid interval after activation");
}}

static void check_affine_bounds_fixed_bounds_only(void)
{{
    __ESBMC_assert(SCALE_FACTOR > 0, "SCALE_FACTOR must be positive");
    __ESBMC_assert(INPUT_SCALE_FACTOR > 0, "INPUT_SCALE_FACTOR must be positive");
    __ESBMC_assert(TOTAL_BITS > 1 && TOTAL_BITS < 127, "TOTAL_BITS must fit in __int128");

    /*
     * Derived mode emits a sound per-neuron integer error budget. Zero mode
     * emits no slack. Heuristic mode reproduces the legacy unproved slack and
     * is reported as degraded soundness.
     */
    const __int128 abs_tol = {abs_tol_expr};
    const __int128 rel_tol_num = {rel_tol_num};
    const __int128 rel_tol_den = 100;

    __ESBMC_assert(rel_tol_den > 0, "relative tolerance denominator must be positive");

    for (int i = 0; i < LAYER_SIZE; ++i)
    {{
        __int128 s_lb = 0;
        __int128 s_ub = 0;

        const __int128 pre_lo = (__int128)preimage_low[i];
        const __int128 pre_hi = (__int128)preimage_high[i];

        __ESBMC_assert(pre_lo <= pre_hi, "invalid preimage interval");

        const __int128 range = abs_i128(pre_hi - pre_lo);
        const __int128 preimage_tolerance = {preimage_tolerance_expr};

        for (int j = 0; j < INPUT_SIZE; ++j)
        {{
            const long long w = weights[i][j];
            const long long lo = input_bounds_low[j];
            const long long hi = input_bounds_high[j];

            __ESBMC_assert(lo <= hi, "invalid input interval");

            s_lb = mac_i128(s_lb, w, (w >= 0) ? lo : hi);
            s_ub = mac_i128(s_ub, w, (w >= 0) ? hi : lo);
        }}

        /*
         * round-half-away-from-zero division by a positive denominator is
         * monotone non-decreasing, so propagating interval endpoints is sound.
         */
        const __int128 raw_out_lb = div_round_half_away_from_zero_i128(s_lb, (__int128)INPUT_SCALE_FACTOR)
            + (__int128)biases[i];
        const __int128 raw_out_ub = div_round_half_away_from_zero_i128(s_ub, (__int128)INPUT_SCALE_FACTOR)
            + (__int128)biases[i];
        __int128 out_lb = raw_out_lb;
        __int128 out_ub = raw_out_ub;

        clamp_bounds_to_signed_range(&out_lb, &out_ub, TOTAL_BITS);
        apply_activation_bounds(&out_lb, &out_ub);
        clamp_bounds_to_signed_range(&out_lb, &out_ub, TOTAL_BITS);

        const __int128 accepted_low = pre_lo - preimage_tolerance;
        const __int128 accepted_high = pre_hi + preimage_tolerance;
        const int lower_violated = out_lb < accepted_low;
        const int upper_violated = out_ub > accepted_high;

        if (lower_violated || upper_violated)
        {{
            counterexample_neuron = i;
            counterexample_preclamp = lower_violated ? raw_out_lb : raw_out_ub;
            for (int j = 0; j < INPUT_SIZE; ++j)
            {{
                const long long w = weights[i][j];
                if (lower_violated)
                {{
                    counterexample_input[j] =
                        (w >= 0) ? input_bounds_low[j] : input_bounds_high[j];
                }}
                else
                {{
                    counterexample_input[j] =
                        (w >= 0) ? input_bounds_high[j] : input_bounds_low[j];
                }}
            }}

            /*
             * Recompute the attained endpoint from the recorded vector so the
             * trace retains a concrete replay witness. This is the same
             * deployed kernel and therefore does not strengthen or weaken the
             * interval assertion below.
             */
            __int128 witness_acc = 0;
            for (int j = 0; j < INPUT_SIZE; ++j)
            {{
                witness_acc = mac_i128(
                    witness_acc,
                    weights[i][j],
                    counterexample_input[j]
                );
            }}
            counterexample_preclamp =
                div_round_half_away_from_zero_i128(
                    witness_acc,
                    (__int128)INPUT_SCALE_FACTOR
                ) + (__int128)biases[i];
            __int128 witness_value = clamp_to_signed_range_i128(
                counterexample_preclamp,
                TOTAL_BITS
            );
            apply_activation_bounds(&witness_value, &witness_value);
            witness_value = clamp_to_signed_range_i128(
                witness_value,
                TOTAL_BITS
            );
            __ESBMC_assert(
                counterexample_neuron == i &&
                witness_value >= accepted_low &&
                witness_value <= accepted_high,
                "concrete affine endpoint outside tolerated preimage"
            );
        }}

        __ESBMC_assert(
            out_lb >= accepted_low && out_ub <= accepted_high,
            "affine bounds not within tolerated preimage"
        );
    }}
}}

int main(void)
{{
    check_affine_bounds_fixed_bounds_only();
    return 0;
}}
"""


def innerlayer_fixed_int(
    cur_layer_layer_size: int,
    in_layer_layer_size: int,
    weights_c_int: str,
    biases_c_int: str,
    preimage_low_int: str,
    preimage_high_int: str,
    input_bounds_low_int: str,
    input_bounds_high_int: str,
    scale_factor: int,
    total_bits: int,
    activation: str = "none",
    unsound_contract_tolerance: bool = False,
    input_scale_factor: int | None = None,
    contract_tolerance_c_int: str | None = None,
) -> str:
    return innerlayer_fixed_int_bounds_only(
        cur_layer_layer_size=cur_layer_layer_size,
        in_layer_layer_size=in_layer_layer_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        preimage_low_int=preimage_low_int,
        preimage_high_int=preimage_high_int,
        input_bounds_low_int=input_bounds_low_int,
        input_bounds_high_int=input_bounds_high_int,
        scale_factor=scale_factor,
        total_bits=total_bits,
        activation=activation,
        unsound_contract_tolerance=unsound_contract_tolerance,
        input_scale_factor=input_scale_factor,
        contract_tolerance_c_int=contract_tolerance_c_int,
    )


def render_no_saturation_program(
    output_size: int,
    input_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    scale_factor: int,
    total_bits: int,
    integer_bits: int | None = None,
    fractional_bits: int | None = None,
    input_scale_factor: int | None = None,
) -> str:
    integer_bits_value = max(int(total_bits) - 1, 0) if integer_bits is None else int(integer_bits)
    fractional_bits_value = 0 if fractional_bits is None else int(fractional_bits)
    input_scale = scale_factor if input_scale_factor is None else int(input_scale_factor)

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {input_size}
#define LAYER_SIZE {output_size}
#define SCALE_FACTOR {scale_factor}LL
#define INPUT_SCALE_FACTOR {input_scale}LL
#define TOTAL_BITS {total_bits}
#define INTEGER_BITS {integer_bits_value}
#define FRACTIONAL_BITS {fractional_bits_value}

/*
 * Formal no-saturation harness for one affine fixed-point layer.
 *
 * Backend arithmetic:
 *   acc = sum(input_int * weight_int)
 *   pre_clamp = rescale(acc, INPUT_SCALE_FACTOR) + bias_int
 *
 * This harness checks the interval image of the affine layer before clamp
 * and before activation/ReLU.
 */

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE] = {biases_c_int};

long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_c_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_c_int};

void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}

static void check_no_saturation_fixed_bounds(void)
{{
    __ESBMC_assert(SCALE_FACTOR > 0, "SCALE_FACTOR must be positive");
    __ESBMC_assert(INPUT_SCALE_FACTOR > 0, "INPUT_SCALE_FACTOR must be positive");
    __ESBMC_assert(TOTAL_BITS > 1 && TOTAL_BITS < 127, "TOTAL_BITS must fit in __int128");

    const __int128 q_min = -((__int128)1 << (TOTAL_BITS - 1));
    const __int128 q_max = ((__int128)1 << (TOTAL_BITS - 1)) - 1;

    for (int i = 0; i < LAYER_SIZE; ++i)
    {{
        __int128 lower = 0;
        __int128 upper = 0;

        for (int j = 0; j < INPUT_SIZE; ++j)
        {{
            const long long w = weights[i][j];
            const long long lo = input_bounds_low[j];
            const long long hi = input_bounds_high[j];

            __ESBMC_assert(lo <= hi, "invalid input interval");

            lower = mac_i128(lower, w, (w >= 0) ? lo : hi);
            upper = mac_i128(upper, w, (w >= 0) ? hi : lo);
        }}

        /*
         * round-half-away-from-zero division by a positive denominator is
         * monotone non-decreasing, so propagating interval endpoints is sound.
         */
        const __int128 lower_rescaled = div_round_half_away_from_zero_i128(lower, (__int128)INPUT_SCALE_FACTOR);
        const __int128 upper_rescaled = div_round_half_away_from_zero_i128(upper, (__int128)INPUT_SCALE_FACTOR);
        const __int128 lower_pre_clamp = lower_rescaled + (__int128)biases[i];
        const __int128 upper_pre_clamp = upper_rescaled + (__int128)biases[i];

        __ESBMC_assert(lower_pre_clamp >= q_min,
                      "fixed-point saturation possible: lower below q_min");
        __ESBMC_assert(upper_pre_clamp <= q_max,
                      "fixed-point saturation possible: upper above q_max");
    }}
}}

int main(void)
{{
    check_no_saturation_fixed_bounds();
    return 0;
}}
"""


def render_no_saturation_block_program(
    block_size: int,
    input_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    scale_factor: int,
    total_bits: int,
    integer_bits: int,
    fractional_bits: int,
    input_scale_factor: int | None = None,
) -> str:
    """Render a no-saturation harness for a contiguous output-neuron block."""

    return render_no_saturation_program(
        output_size=block_size,
        input_size=input_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        input_bounds_low_c_int=input_bounds_low_c_int,
        input_bounds_high_c_int=input_bounds_high_c_int,
        scale_factor=scale_factor,
        total_bits=total_bits,
        integer_bits=integer_bits,
        fractional_bits=fractional_bits,
        input_scale_factor=input_scale_factor,
    )


def render_clamp_correctness_program(total_bits: int) -> str:
    return f"""\
#include <stdint.h>
#include <limits.h>

#define TOTAL_BITS {total_bits}

extern long long nondet_longlong(void);

/*
 * Clamp correctness harness.
 *
 * The nondeterministic input is long long and then promoted to __int128.
 * This verifies clamp behavior over the long long input domain, which covers
 * the generated backend's int64_t storage interface.
 */

void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}

int main(void)
{{
    __ESBMC_assert(TOTAL_BITS > 1 && TOTAL_BITS < 127, "TOTAL_BITS must fit in __int128");

    const __int128 q_min = -((__int128)1 << (TOTAL_BITS - 1));
    const __int128 q_max = (((__int128)1 << (TOTAL_BITS - 1)) - 1);
    const __int128 input = (__int128)nondet_longlong();
    const __int128 output = clamp_to_signed_range_i128(input, TOTAL_BITS);

    __ESBMC_assert(output >= q_min, "clamp output below q_min");
    __ESBMC_assert(output <= q_max, "clamp output above q_max");

    if (input >= q_min && input <= q_max)
    {{
        __ESBMC_assert(output == input, "clamp changed in-range input");
    }}
    if (input < q_min)
    {{
        __ESBMC_assert(output == q_min, "clamp did not saturate low input to q_min");
    }}
    if (input > q_max)
    {{
        __ESBMC_assert(output == q_max, "clamp did not saturate high input to q_max");
    }}

    return 0;
}}
"""


def outerlayer_fixed_int_multiclass(
    in_layer_layer_size: int,
    cur_layer_layer_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_int: str,
    input_bounds_high_int: str,
    valid_classes: tuple[int, ...] | list[int],
    scale_factor: int,
    total_bits: int,
    input_scale_factor: int | None = None,
) -> str:
    valid_classes_array = "{" + ", ".join(map(str, valid_classes)) + "}"
    num_valid_classes = len(valid_classes)

    input_scale = scale_factor if input_scale_factor is None else int(input_scale_factor)
    return f"""\
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>


#define INPUT_SIZE       {in_layer_layer_size}
#define LAYER_SIZE       {cur_layer_layer_size}
#define NUM_VALID_CLASSES {num_valid_classes}
#define SCALE_FACTOR     {scale_factor}LL
#define INPUT_SCALE_FACTOR {input_scale}LL
#define TOTAL_BITS       {total_bits}

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

int valid_classes[NUM_VALID_CLASSES] = {valid_classes_array};

void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}

/* Verifica se uma classe está no conjunto de classes válidas */
static bool is_valid_class(int class_id) {{
    for (int i = 0; i < NUM_VALID_CLASSES; ++i) {{
        if (valid_classes[i] == class_id) {{
            return true;
        }}
    }}
    return false;
}}

/* Transformacao na funcao afim em ponto fixo */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{{
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        __int128 acc = 0;

        for (int j = 0; j < INPUT_SIZE; ++j) {{
            acc = mac_i128(acc, weights[i][j], in_[j]);
        }}

        __int128 value = div_round_half_away_from_zero_i128(
            acc,
            (__int128)INPUT_SCALE_FACTOR
        ) + (__int128)biases[i];
        value = clamp_to_signed_range_i128(value, TOTAL_BITS);
        out_[i] = (long long)clamp_to_signed_range_i128(value, TOTAL_BITS);
    }}
}}

/* Verifica se a classificacao esta entre as classes validas */
static int verify_classification_multiclass(const long long out_[LAYER_SIZE])
{{
    long long max_valid = LLONG_MIN;
    long long max_invalid = LLONG_MIN;

    /* Encontra os valores maximos nas classes validas e invalidas */
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        if (is_valid_class(i)) {{
            if (out_[i] > max_valid) {{
                max_valid = out_[i];
            }}
        }} else {{
            if (out_[i] > max_invalid) {{
                max_invalid = out_[i];
            }}
        }}
    }}

    /* (maior valor entre validas > maior valor entre invalidas) */
    return max_valid > max_invalid;
}}

int main(void)
{{
    long long input[INPUT_SIZE];
    long long output[LAYER_SIZE];

    /* Entrada nao-deterministica dentro dos bounds */
    for (int k = 0; k < INPUT_SIZE; ++k) {{
        input[k] = nondet_longlong();
        __ESBMC_assume(input[k] >= input_bounds_low[k] &&
                       input[k] <= input_bounds_high[k]);
    }}

    affine_transform_fixed(input, output);

    __ESBMC_assert(verify_classification_multiclass(output),
                   "Classification property violated - output not in valid classes");

    return 0;
}}
"""


def render_hidden_affine_bounds_program(
    output_size: int,
    input_size: int,
    weights_c_int: str,
    biases_c_int: str,
    preimage_low_c_int: str,
    preimage_high_c_int: str,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    scale_factor: int,
    total_bits: int,
    activation: str = "none",
    unsound_contract_tolerance: bool = False,
    input_scale_factor: int | None = None,
    contract_tolerance_c_int: str | None = None,
) -> str:
    return innerlayer_fixed_int_bounds_only(
        cur_layer_layer_size=output_size,
        in_layer_layer_size=input_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        preimage_low_int=preimage_low_c_int,
        preimage_high_int=preimage_high_c_int,
        input_bounds_low_int=input_bounds_low_c_int,
        input_bounds_high_int=input_bounds_high_c_int,
        scale_factor=scale_factor,
        total_bits=total_bits,
        activation=activation,
        unsound_contract_tolerance=unsound_contract_tolerance,
        input_scale_factor=input_scale_factor,
        contract_tolerance_c_int=contract_tolerance_c_int,
    )


def render_hidden_affine_bounds_block_program(
    block_size: int,
    input_size: int,
    weights_c_int: str,
    biases_c_int: str,
    preimage_low_c_int: str,
    preimage_high_c_int: str,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    scale_factor: int,
    total_bits: int,
    activation: str = "none",
    unsound_contract_tolerance: bool = False,
    input_scale_factor: int | None = None,
    contract_tolerance_c_int: str | None = None,
) -> str:
    """Render a hidden affine contract harness for a contiguous output-neuron block."""

    return render_hidden_affine_bounds_program(
        output_size=block_size,
        input_size=input_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        preimage_low_c_int=preimage_low_c_int,
        preimage_high_c_int=preimage_high_c_int,
        input_bounds_low_c_int=input_bounds_low_c_int,
        input_bounds_high_c_int=input_bounds_high_c_int,
        scale_factor=scale_factor,
        total_bits=total_bits,
        activation=activation,
        unsound_contract_tolerance=unsound_contract_tolerance,
        input_scale_factor=input_scale_factor,
        contract_tolerance_c_int=contract_tolerance_c_int,
    )


def render_output_target_program(
    output_size: int,
    input_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    target_label: int,
    scale_factor: int,
    total_bits: int,
    input_scale_factor: int | None = None,
    margin_cut_directions_c_int: str | None = None,
    margin_cut_low_c_int: str | None = None,
    margin_cut_high_c_int: str | None = None,
    margin_cut_scale: int | None = None,
    margin_cut_count: int = 0,
) -> str:
    """Render the exact deployed output property over a verified hidden box.

    Optional directional cuts are sound invariants of the deployed hidden vector.
    Their producer bounds each real direction over the original input region, then
    widens it by the inherited per-coordinate hidden error and by coefficient
    quantization error before converting outward to the integer product scale
    ``margin_cut_scale * input_scale_factor``. Consequently each emitted
    ``__ESBMC_assume`` contains every reachable deployed hidden vector; it removes
    only Cartesian-box combinations that cannot come from the input region.

    Output weights and biases are already quantized, and the body uses the shared
    deployed arithmetic kernel exactly. No output-layer error budget is needed.
    """

    return outerlayer_fixed_int(
        in_layer_layer_size=input_size,
        cur_layer_layer_size=output_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        input_bounds_low_int=input_bounds_low_c_int,
        input_bounds_high_int=input_bounds_high_c_int,
        targetCls=target_label,
        scale_factor=scale_factor,
        total_bits=total_bits,
        input_scale_factor=input_scale_factor,
        margin_cut_directions_c_int=margin_cut_directions_c_int,
        margin_cut_low_c_int=margin_cut_low_c_int,
        margin_cut_high_c_int=margin_cut_high_c_int,
        margin_cut_scale=margin_cut_scale,
        margin_cut_count=margin_cut_count,
    )


def render_output_valid_set_program(
    output_size: int,
    input_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    valid_classes: tuple[int, ...],
    scale_factor: int,
    total_bits: int,
    input_scale_factor: int | None = None,
) -> str:
    return outerlayer_fixed_int_multiclass(
        in_layer_layer_size=input_size,
        cur_layer_layer_size=output_size,
        weights_c_int=weights_c_int,
        biases_c_int=biases_c_int,
        input_bounds_low_int=input_bounds_low_c_int,
        input_bounds_high_int=input_bounds_high_c_int,
        valid_classes=valid_classes,
        scale_factor=scale_factor,
        total_bits=total_bits,
        input_scale_factor=input_scale_factor,
    )


def render_assumption_sentinel_program(
    input_size: int,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
) -> str:
    """Render a satisfiability sentinel for an integer input assumption box."""

    return f"""\
#include <stdint.h>

#define INPUT_SIZE {input_size}

extern long long nondet_longlong(void);

long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_c_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_c_int};

void __ESBMC_assert(_Bool, const char *);

int main(void)
{{
    long long input[INPUT_SIZE];
    for (int k = 0; k < INPUT_SIZE; ++k)
    {{
        input[k] = nondet_longlong();
        __ESBMC_assume(input[k] >= input_bounds_low[k] &&
                       input[k] <= input_bounds_high[k]);
    }}

    __ESBMC_assert(0, "sentinel");
    return 0;
}}
"""


def render_network_end_to_end_program(
    *,
    input_size: int,
    input_bounds_low_c_int: str,
    input_bounds_high_c_int: str,
    layers: list[dict[str, object]],
    target_label: int | None = None,
    valid_classes: tuple[int, ...] | None = None,
    inject_invariants: bool = True,
) -> str:
    """Render one exact deployed-semantics whole-network ESBMC harness."""

    if not layers:
        raise ValueError("The end-to-end harness requires at least one layer.")
    if target_label is None and not valid_classes:
        raise ValueError("Expected a target label or a non-empty valid-class set.")

    max_width = max(
        max(int(layer["input_size"]), int(layer["output_size"]))
        for layer in layers
    )
    declarations: list[str] = []
    steps: list[str] = []
    for index, layer in enumerate(layers):
        input_dim = int(layer["input_size"])
        output_dim = int(layer["output_size"])
        total_bits = int(layer["total_bits"])
        fractional_bits = int(layer["fractional_bits"])
        input_fractional_bits = int(layer["input_fractional_bits"])
        accumulator_type = str(layer.get("accumulator_c_type", "__int128"))
        declarations.append(
            f"""
static const int LAYER_{index}_IN = {input_dim};
static const int LAYER_{index}_OUT = {output_dim};
static const int LAYER_{index}_Q = {total_bits};
static const int LAYER_{index}_F = {fractional_bits};
static const int64_t LAYER_{index}_WEIGHTS[{output_dim}][{input_dim}] = {layer["weights_c_int"]};
static const int64_t LAYER_{index}_BIASES[{output_dim}] = {layer["biases_c_int"]};
static const int64_t LAYER_{index}_LOW[{output_dim}] = {layer["invariant_low_c_int"]};
static const int64_t LAYER_{index}_HIGH[{output_dim}] = {layer["invariant_high_c_int"]};
"""
        )
        input_buffer = "buffer_a" if index % 2 == 0 else "buffer_b"
        output_buffer = "buffer_b" if index % 2 == 0 else "buffer_a"
        relu = (
            "        if (value < 0) value = 0;\n"
            if index < len(layers) - 1
            else ""
        )
        invariant = (
            f"""\
        __ESBMC_assume(
            {output_buffer}[out_idx] >= LAYER_{index}_LOW[out_idx] &&
            {output_buffer}[out_idx] <= LAYER_{index}_HIGH[out_idx]
        );
"""
            if inject_invariants
            else ""
        )
        steps.append(
            f"""
    for (int out_idx = 0; out_idx < LAYER_{index}_OUT; ++out_idx)
    {{
        {accumulator_type} acc = 0;
        for (int in_idx = 0; in_idx < LAYER_{index}_IN; ++in_idx)
        {{
            acc = ({accumulator_type})mac_i128(
                (__int128)acc,
                LAYER_{index}_WEIGHTS[out_idx][in_idx],
                {input_buffer}[in_idx]
            );
        }}
        __int128 value = div_round_half_away_from_zero_i128(
            (__int128)acc,
            ((__int128)1 << {input_fractional_bits})
        ) + (__int128)LAYER_{index}_BIASES[out_idx];
        value = clamp_to_signed_range_i128(value, LAYER_{index}_Q);
{relu}        value = clamp_to_signed_range_i128(value, LAYER_{index}_Q);
        {output_buffer}[out_idx] = (int64_t)value;
{invariant}    }}
"""
        )

    output_size = int(layers[-1]["output_size"])
    final_buffer = "buffer_b" if (len(layers) - 1) % 2 == 0 else "buffer_a"
    if valid_classes:
        valid_values = "{" + ", ".join(str(int(value)) for value in valid_classes) + "}"
        property_declarations = f"""
#define NUM_VALID_CLASSES {len(valid_classes)}
static const int VALID_CLASSES[NUM_VALID_CLASSES] = {valid_values};

static int is_valid_class(int value)
{{
    for (int i = 0; i < NUM_VALID_CLASSES; ++i)
    {{
        if (VALID_CLASSES[i] == value) return 1;
    }}
    return 0;
}}
"""
        property_body = f"""
    int64_t max_valid = INT64_MIN;
    int64_t max_invalid = INT64_MIN;
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {{
        if (is_valid_class(i))
        {{
            if ({final_buffer}[i] > max_valid) max_valid = {final_buffer}[i];
        }}
        else
        {{
            if ({final_buffer}[i] > max_invalid) max_invalid = {final_buffer}[i];
        }}
    }}
    __ESBMC_assert(max_valid > max_invalid,
                   "End-to-end valid-set classification property violated");
"""
    else:
        property_declarations = f"#define TARGET_CLASS {int(target_label)}\n"
        property_body = f"""
    const int64_t target = {final_buffer}[TARGET_CLASS];
    for (int i = 0; i < OUTPUT_SIZE; ++i)
    {{
        if (i != TARGET_CLASS)
        {{
            __ESBMC_assert(
                target > {final_buffer}[i],
                "End-to-end target classification property violated"
            );
        }}
    }}
"""

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {int(input_size)}
#define OUTPUT_SIZE {output_size}
#define E2E_INVARIANTS {1 if inject_invariants else 0}

extern long long nondet_longlong(void);
void __ESBMC_assert(_Bool, const char *);
void __ESBMC_assume(_Bool);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}

static const int64_t INPUT_LOW[INPUT_SIZE] = {input_bounds_low_c_int};
static const int64_t INPUT_HIGH[INPUT_SIZE] = {input_bounds_high_c_int};
{''.join(declarations)}
{property_declarations}

int main(void)
{{
    int64_t input[INPUT_SIZE];
    int64_t buffer_a[{max(max_width, input_size)}] = {{0}};
    int64_t buffer_b[{max_width}] = {{0}};

    for (int i = 0; i < INPUT_SIZE; ++i)
    {{
        input[i] = nondet_longlong();
        __ESBMC_assume(input[i] >= INPUT_LOW[i] && input[i] <= INPUT_HIGH[i]);
        buffer_a[i] = input[i];
    }}

{''.join(steps)}
{property_body}
    return 0;
}}
"""
