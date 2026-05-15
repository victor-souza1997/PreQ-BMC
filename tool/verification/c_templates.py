from __future__ import annotations


def outerlayer_fixed_int(
    in_layer_layer_size: int,
    cur_layer_layer_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_int: str,
    input_bounds_high_int: str,
    targetCls: int,
    scale_factor: int,
) -> str:
    return f"""\
#include <stdint.h>
#include <limits.h>


#define INPUT_SIZE   {in_layer_layer_size}
#define LAYER_SIZE   {cur_layer_layer_size}
#define TARGET_CLASS {targetCls}
#define SCALE_FACTOR {scale_factor}LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

static inline long long llabs(long long x) {{
    return x < 0LL ? -x : x;
}}

/* Transformacao afim em ponto fixo: out[i] = (W[i] · in + b[i]) / SCALE */
static void affine_transform_fixed(const long long in_[INPUT_SIZE], long long out_[LAYER_SIZE])
{{
    for (int i = 0; i < LAYER_SIZE; ++i) {{
        long long acc = 0LL; /* acumulador em SCALE_FACTOR^2 */

        for (int j = 0; j < INPUT_SIZE; ++j) {{
            /* w * x ambos em SCALE_FACTOR → produto em SCALE_FACTOR^2 */
            long long prod = weights[i][j] * in_[j];
            acc += prod;
        }}

        /* Rescale para SCALE_FACTOR e adiciona bias (já em SCALE_FACTOR) */
        out_[i] = (acc / SCALE_FACTOR) + biases[i];
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
) -> str:
    return f"""\
#include <stdint.h>
#include <limits.h>


#define INPUT_SIZE   {in_layer_layer_size}
#define LAYER_SIZE   {cur_layer_layer_size}
#define SCALE_FACTOR {scale_factor}LL

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long preimage_low[LAYER_SIZE]  = {preimage_low_int};
long long preimage_high[LAYER_SIZE] = {preimage_high_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

static inline long long llabs(long long x) {{
    return x < 0LL ? -x : x;
}}

/* Camada afim em ponto fixo sobre um box de entrada: s_lb <= s_out <= s_ub */
static void check_affine_bounds_fixed(const long long in_[INPUT_SIZE])
{{
    /* tolerancia para preimagem */
    const long long abs_tol = (long long)(1e-3 * SCALE_FACTOR);
    const long long rel_tol_num = 1; /* 1% = 1/100 */
    const long long rel_tol_den = 100;

    for (int i = 0; i < LAYER_SIZE; ++i) {{
        long long s_out = 0LL;  /* saída exata na entrada atual */
        long long s_lb  = 0LL;  /* limite inferior usando box */
        long long s_ub  = 0LL;  /* limite superior usando box */

        /* Tolerance ao redor do intervalo de preimagem */
        const long long pre_lo = preimage_low[i];
        const long long pre_hi = preimage_high[i];

        __int128 pre_lo_i = (__int128)pre_lo;
        __int128 pre_hi_i = (__int128)pre_hi;

        __int128 range = pre_hi_i >= pre_lo_i
            ? pre_hi_i - pre_lo_i
            : pre_lo_i - pre_hi_i;

        __int128 eps = (__int128)abs_tol
            + ((__int128)rel_tol_num * range) / (__int128)rel_tol_den;

        __ESBMC_assert(
            out_lb >= pre_lo_i - eps &&
            out_ub <= pre_hi_i + eps,
            "affine bounds not within tolerated preimage"
        );
        int j = 0;

        while (j < INPUT_SIZE)
        {{
            //__ESBMC_loop_invariant(0 <= j && j <= INPUT_SIZE &&
            //                       s_lb <= s_out && s_out <= s_ub);

            const long long w  = weights[i][j];
            const long long lo = input_bounds_low[j];
            const long long hi = input_bounds_high[j];

            /* Passo exato na entrada nao determinística */
            s_out += w * in_[j];

            /* Contribuição baseada no sinal (imagem do box) */
            const long long cmin = (w >= 0LL) ? (w * lo) : (w * hi);
            const long long cmax = (w >= 0LL) ? (w * hi) : (w * lo);

            s_lb += cmin;
            s_ub += cmax;

            ++j;
        }}

        /* Rescale e adiciona bias */
        s_out = (s_out / SCALE_FACTOR) + biases[i];
        s_lb  = (s_lb  / SCALE_FACTOR) + biases[i];
        s_ub  = (s_ub  / SCALE_FACTOR) + biases[i];

        /* verifica se a saida esta dentro da preimagem esperada */
        __ESBMC_assert(s_out >= pre_lo - eps && s_out <= pre_hi + eps,
                       "affine output not within tolerated preimage");
    }}
}}

int main(void)
{{
    long long in_[INPUT_SIZE];

    /* Entrada nao deterministica dentro do conjunto de entrada */
    for (int j = 0; j < INPUT_SIZE; ++j) {{
        in_[j] = nondet_longlong();
        __ESBMC_assume(in_[j] >= input_bounds_low[j] &&
                       in_[j] <= input_bounds_high[j]);
    }}

    check_affine_bounds_fixed(in_);

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
    activation: str = "none",
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

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {in_layer_layer_size}
#define LAYER_SIZE {cur_layer_layer_size}
#define SCALE_FACTOR {scale_factor}LL
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
 * The affine computation is performed over intervals using __int128 to avoid
 * losing soundness due to intermediate integer overflows in the verifier model.
 */

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE] = {biases_c_int};

long long preimage_low[LAYER_SIZE] = {preimage_low_int};
long long preimage_high[LAYER_SIZE] = {preimage_high_int};

long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

static inline __int128 abs_i128(__int128 x)
{{
    return x < 0 ? -x : x;
}}

static inline __int128 div_floor_i128(__int128 a, __int128 d)
{{
    __ESBMC_assert(d > 0, "division denominator must be positive");

    if (a >= 0)
    {{
        return a / d;
    }}

    return -(((-a) + d - 1) / d);
}}

static inline __int128 div_ceil_i128(__int128 a, __int128 d)
{{
    __ESBMC_assert(d > 0, "division denominator must be positive");

    if (a >= 0)
    {{
        return (a + d - 1) / d;
    }}

    return -((-a) / d);
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
        /* ReLU6 interval transformer.
         * This assumes values are represented in the same fixed-point scale
         * as the layer output.
         */
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

    /*
     * Tolerance around the preimage interval.
     * abs_tol = 0.001 * scale
     * rel_tol = 1%
     */
    const __int128 abs_tol = (__int128)(SCALE_FACTOR / 1000);
    const __int128 rel_tol_num = 1;
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
        const __int128 eps = abs_tol + (rel_tol_num * range) / rel_tol_den;

        for (int j = 0; j < INPUT_SIZE; ++j)
        {{
            const __int128 w = (__int128)weights[i][j];
            const __int128 lo = (__int128)input_bounds_low[j];
            const __int128 hi = (__int128)input_bounds_high[j];

            __ESBMC_assert(lo <= hi, "invalid input interval");

            /*
             * Contribution of w*x over the input box.
             * If w >= 0: min = w*lo, max = w*hi.
             * If w <  0: min = w*hi, max = w*lo.
             */
            const __int128 cmin = (w >= 0) ? (w * lo) : (w * hi);
            const __int128 cmax = (w >= 0) ? (w * hi) : (w * lo);

            s_lb += cmin;
            s_ub += cmax;
        }}

        /*
         * Sound rescaling:
         *   lower bound uses floor
         *   upper bound uses ceil
         */
        __int128 out_lb = div_floor_i128(s_lb, (__int128)SCALE_FACTOR)
            + (__int128)biases[i];

        __int128 out_ub = div_ceil_i128(s_ub, (__int128)SCALE_FACTOR)
            + (__int128)biases[i];

        apply_activation_bounds(&out_lb, &out_ub);

        const __int128 accepted_low = pre_lo - eps;
        const __int128 accepted_high = pre_hi + eps;

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

def outerlayer_fixed_int_multiclass(
    in_layer_layer_size: int,
    cur_layer_layer_size: int,
    weights_c_int: str,
    biases_c_int: str,
    input_bounds_low_int: str,
    input_bounds_high_int: str,
    valid_classes: tuple[int, ...] | list[int],
    scale_factor: int,
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

extern long long nondet_longlong(void);

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE]              = {biases_c_int};

long long input_bounds_low[INPUT_SIZE]  = {input_bounds_low_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_int};

int valid_classes[NUM_VALID_CLASSES] = {valid_classes_array};

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
        long long acc = 0LL;

        for (int j = 0; j < INPUT_SIZE; ++j) {{
            long long prod = weights[i][j] * in_[j];
            acc += prod;
        }}

        out_[i] = (acc / SCALE_FACTOR) + biases[i];
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
    activation: str = "none",
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
        activation=activation,
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
    )
