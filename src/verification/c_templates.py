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
) -> str:
    return f"""\
#include <stdint.h>
#include <limits.h>


#define INPUT_SIZE   {in_layer_layer_size}
#define LAYER_SIZE   {cur_layer_layer_size}
#define TARGET_CLASS {targetCls}
#define SCALE_FACTOR {scale_factor}LL
#define TOTAL_BITS   {total_bits}

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

void __ESBMC_assert(_Bool, const char *);
#define QNN_ASSERT(cond, msg) __ESBMC_assert((cond), (msg))

{render_arith_kernel()}

/* Transformacao afim em ponto fixo: out[i] = (W[i] · in + b[i]) / SCALE */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{{
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        __int128 acc = 0; /* acumulador em SCALE_FACTOR^2 */

        for (int j = 0; j < INPUT_SIZE; ++j) {{
            /* w * x ambos em SCALE_FACTOR → produto em SCALE_FACTOR^2 */
            acc = mac_i128(acc, weights[i][j], in_[j]);
        }}

        __int128 value = div_round_half_away_from_zero_i128(
            acc,
            (__int128)SCALE_FACTOR
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
        "abs_tol + (rel_tol_num * range) / rel_tol_den"
        if unsound_contract_tolerance
        else "0"
    )

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {in_layer_layer_size}
#define LAYER_SIZE {cur_layer_layer_size}
#define SCALE_FACTOR {scale_factor}LL
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
    __ESBMC_assert(TOTAL_BITS > 1 && TOTAL_BITS < 127, "TOTAL_BITS must fit in __int128");

    /*
     * Contract tolerance is zero for sound assume-guarantee composition.
     * The legacy non-zero tolerance is only emitted by the explicit
     * --unsound-contract-tolerance debug flag.
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
        __int128 out_lb = div_round_half_away_from_zero_i128(s_lb, (__int128)SCALE_FACTOR)
            + (__int128)biases[i];
        __int128 out_ub = div_round_half_away_from_zero_i128(s_ub, (__int128)SCALE_FACTOR)
            + (__int128)biases[i];

        clamp_bounds_to_signed_range(&out_lb, &out_ub, TOTAL_BITS);
        apply_activation_bounds(&out_lb, &out_ub);
        clamp_bounds_to_signed_range(&out_lb, &out_ub, TOTAL_BITS);

        const __int128 accepted_low = pre_lo - preimage_tolerance;
        const __int128 accepted_high = pre_hi + preimage_tolerance;

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
) -> str:
    integer_bits_value = max(int(total_bits) - 1, 0) if integer_bits is None else int(integer_bits)
    fractional_bits_value = 0 if fractional_bits is None else int(fractional_bits)

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {input_size}
#define LAYER_SIZE {output_size}
#define SCALE_FACTOR {scale_factor}LL
#define TOTAL_BITS {total_bits}
#define INTEGER_BITS {integer_bits_value}
#define FRACTIONAL_BITS {fractional_bits_value}

/*
 * Formal no-saturation harness for one affine fixed-point layer.
 *
 * Backend arithmetic:
 *   acc = sum(input_int * weight_int)
 *   pre_clamp = rescale(acc, SCALE_FACTOR) + bias_int
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
        const __int128 lower_rescaled = div_round_half_away_from_zero_i128(lower, (__int128)SCALE_FACTOR);
        const __int128 upper_rescaled = div_round_half_away_from_zero_i128(upper, (__int128)SCALE_FACTOR);
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
) -> str:
    valid_classes_array = "{" + ", ".join(map(str, valid_classes)) + "}"
    num_valid_classes = len(valid_classes)

    return f"""\
#include <stdint.h>
#include <limits.h>
#include <stdbool.h>


#define INPUT_SIZE       {in_layer_layer_size}
#define LAYER_SIZE       {cur_layer_layer_size}
#define NUM_VALID_CLASSES {num_valid_classes}
#define SCALE_FACTOR     {scale_factor}LL
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
            (__int128)SCALE_FACTOR
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
) -> str:
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
    )
