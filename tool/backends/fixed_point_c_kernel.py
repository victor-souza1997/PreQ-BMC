from __future__ import annotations


def fixed_point_c_kernel_source() -> str:
    """Return the shared C fixed-point arithmetic kernel.

    This kernel is intentionally used by both the deployed generated C backend
    and ESBMC harness templates. Keep arithmetic changes centralized here.
    """

    return """\
#ifndef QNN_ASSERT
#define QNN_ASSERT(cond, msg) ((void)0)
#endif

static inline __int128 clamp_to_signed_range(__int128 value, int total_bits) {
    const __int128 lower = -((__int128)1 << (total_bits - 1));
    const __int128 upper = (((__int128)1 << (total_bits - 1)) - 1);
    if (value < lower) return lower;
    if (value > upper) return upper;
    return value;
}

static inline __int128 div_round_half_away_from_zero_i128(__int128 numerator, __int128 denominator) {
    QNN_ASSERT(denominator > 0, "denominator must be positive");
    if (numerator >= 0) {
        return (numerator + denominator / 2) / denominator;
    }
    return -(((-numerator) + denominator / 2) / denominator);
}

static inline __int128 fixed_point_layer_value_i128(
    __int128 accumulator,
    __int128 scale,
    __int128 bias,
    int total_bits,
    int apply_relu
) {
    __int128 value = div_round_half_away_from_zero_i128(accumulator, scale) + bias;
    value = clamp_to_signed_range(value, total_bits);
    if (apply_relu && value < 0) {
        value = 0;
    }
    return clamp_to_signed_range(value, total_bits);
}
"""
