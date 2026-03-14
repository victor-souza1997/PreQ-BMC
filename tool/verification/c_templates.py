from __future__ import annotations


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
) -> str:
    """Generate a bounds-only ESBMC program for a hidden affine layer."""

    return f"""\
#include <stdint.h>

#define INPUT_SIZE {input_size}
#define LAYER_SIZE {output_size}
#define SCALE_FACTOR {scale_factor}LL

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE] = {biases_c_int};
long long preimage_low[LAYER_SIZE] = {preimage_low_c_int};
long long preimage_high[LAYER_SIZE] = {preimage_high_c_int};
long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_c_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_c_int};

static inline long long llabs_ll(long long x) {{
    return x < 0LL ? -x : x;
}}

static inline __int128 div_floor_i128(__int128 a, long long d) {{
    if (a >= 0) return a / d;
    return -(((-a) + (d - 1)) / d);
}}

static inline __int128 div_ceil_i128(__int128 a, long long d) {{
    if (a >= 0) return (a + (d - 1)) / d;
    return -(((-a)) / d);
}}

int main(void) {{
    const long long abs_tol = (long long)(1e-3 * SCALE_FACTOR);
    const long long rel_tol_num = 1;
    const long long rel_tol_den = 100;

    for (int i = 0; i < LAYER_SIZE; ++i) {{
        __int128 s_lb = 0;
        __int128 s_ub = 0;

        const long long pre_lo = preimage_low[i];
        const long long pre_hi = preimage_high[i];
        const long long range = llabs_ll(pre_hi - pre_lo);
        const long long eps = abs_tol + (rel_tol_num * range) / rel_tol_den;

        for (int j = 0; j < INPUT_SIZE; ++j) {{
            const __int128 w = (__int128)weights[i][j];
            const __int128 lo = (__int128)input_bounds_low[j];
            const __int128 hi = (__int128)input_bounds_high[j];
            const __int128 cmin = (w >= 0) ? (w * lo) : (w * hi);
            const __int128 cmax = (w >= 0) ? (w * hi) : (w * lo);
            s_lb += cmin;
            s_ub += cmax;
        }}

        __int128 out_lb = div_floor_i128(s_lb, SCALE_FACTOR) + (__int128)biases[i];
        __int128 out_ub = div_ceil_i128(s_ub, SCALE_FACTOR) + (__int128)biases[i];

        __ESBMC_assert(
            out_lb >= (__int128)(pre_lo - eps) && out_ub <= (__int128)(pre_hi + eps),
            "affine bounds not within tolerated preimage");
    }}

    return 0;
}}
"""


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
    """Generate an ESBMC program asserting that `target_label` remains the argmax."""

    return f"""\
#include <stdint.h>
#include <limits.h>

#define INPUT_SIZE {input_size}
#define LAYER_SIZE {output_size}
#define TARGET_CLASS {target_label}
#define SCALE_FACTOR {scale_factor}LL

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE] = {biases_c_int};
long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_c_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_c_int};

static inline __int128 div_floor_i128(__int128 a, long long d) {{
    if (a >= 0) return a / d;
    return -(((-a) + (d - 1)) / d);
}}

static inline __int128 div_ceil_i128(__int128 a, long long d) {{
    if (a >= 0) return (a + (d - 1)) / d;
    return -(((-a)) / d);
}}

int main(void) {{
    __int128 target_acc_lb = 0;
    __int128 other_max_ub = -((__int128)1 << 120);

    for (int j = 0; j < INPUT_SIZE; ++j) {{
        const __int128 lo = (__int128)input_bounds_low[j];
        const __int128 hi = (__int128)input_bounds_high[j];
        const __int128 tw = (__int128)weights[TARGET_CLASS][j];
        target_acc_lb += (tw >= 0) ? (tw * lo) : (tw * hi);
    }}

    __int128 target_lb = div_floor_i128(target_acc_lb, SCALE_FACTOR) + (__int128)biases[TARGET_CLASS];

    for (int i = 0; i < LAYER_SIZE; ++i) {{
        if (i == TARGET_CLASS) {{
            continue;
        }}
        __int128 class_acc_ub = 0;
        for (int j = 0; j < INPUT_SIZE; ++j) {{
            const __int128 lo = (__int128)input_bounds_low[j];
            const __int128 hi = (__int128)input_bounds_high[j];
            const __int128 w = (__int128)weights[i][j];
            class_acc_ub += (w >= 0) ? (w * hi) : (w * lo);
        }}
        __int128 class_ub = div_ceil_i128(class_acc_ub, SCALE_FACTOR) + (__int128)biases[i];
        if (class_ub > other_max_ub) {{
            other_max_ub = class_ub;
        }}
    }}

    __ESBMC_assert(target_lb > other_max_ub, "classification property violated");
    return 0;
}}
"""


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
    """Generate an ESBMC program asserting the winning class stays within a valid set."""

    valid_classes_array = "{" + ", ".join(str(value) for value in valid_classes) + "}"
    return f"""\
#include <stdint.h>
#include <stdbool.h>

#define INPUT_SIZE {input_size}
#define LAYER_SIZE {output_size}
#define NUM_VALID_CLASSES {len(valid_classes)}
#define SCALE_FACTOR {scale_factor}LL

long long weights[LAYER_SIZE][INPUT_SIZE] = {weights_c_int};
long long biases[LAYER_SIZE] = {biases_c_int};
long long input_bounds_low[INPUT_SIZE] = {input_bounds_low_c_int};
long long input_bounds_high[INPUT_SIZE] = {input_bounds_high_c_int};
int valid_classes[NUM_VALID_CLASSES] = {valid_classes_array};

static bool is_valid_class(int class_id) {{
    for (int i = 0; i < NUM_VALID_CLASSES; ++i) {{
        if (valid_classes[i] == class_id) {{
            return true;
        }}
    }}
    return false;
}}

static inline __int128 div_floor_i128(__int128 a, long long d) {{
    if (a >= 0) return a / d;
    return -(((-a) + (d - 1)) / d);
}}

static inline __int128 div_ceil_i128(__int128 a, long long d) {{
    if (a >= 0) return (a + (d - 1)) / d;
    return -(((-a)) / d);
}}

int main(void) {{
    __int128 max_valid_lb = -((__int128)1 << 120);
    __int128 max_invalid_ub = -((__int128)1 << 120);

    for (int i = 0; i < LAYER_SIZE; ++i) {{
        __int128 acc_lb = 0;
        __int128 acc_ub = 0;
        for (int j = 0; j < INPUT_SIZE; ++j) {{
            const __int128 w = (__int128)weights[i][j];
            const __int128 lo = (__int128)input_bounds_low[j];
            const __int128 hi = (__int128)input_bounds_high[j];
            acc_lb += (w >= 0) ? (w * lo) : (w * hi);
            acc_ub += (w >= 0) ? (w * hi) : (w * lo);
        }}

        __int128 cur_lb = div_floor_i128(acc_lb, SCALE_FACTOR) + (__int128)biases[i];
        __int128 cur_ub = div_ceil_i128(acc_ub, SCALE_FACTOR) + (__int128)biases[i];

        if (is_valid_class(i)) {{
            if (cur_lb > max_valid_lb) max_valid_lb = cur_lb;
        }} else {{
            if (cur_ub > max_invalid_ub) max_invalid_ub = cur_ub;
        }}
    }}

    __ESBMC_assert(max_valid_lb > max_invalid_ub, "classification property violated");
    return 0;
}}
"""
