from __future__ import annotations


def render_arith_kernel() -> str:
    """Return the shared generated C fixed-point arithmetic kernel."""

    return """\
#ifndef QNN_ASSERT
#define QNN_ASSERT(cond, msg) ((void)0)
#endif

static inline __int128 div_round_half_away_from_zero_i128(__int128 num, __int128 den) {
    QNN_ASSERT(den > 0, "denominator must be positive");
    if (num >= 0) {
        return (num + den / 2) / den;
    }
    return -(((-num) + den / 2) / den);
}

static inline __int128 clamp_to_signed_range_i128(__int128 v, int total_bits) {
    const __int128 lower = -((__int128)1 << (total_bits - 1));
    const __int128 upper = (((__int128)1 << (total_bits - 1)) - 1);
    if (v < lower) return lower;
    if (v > upper) return upper;
    return v;
}

static inline __int128 mac_i128(__int128 acc, int64_t w, int64_t x) {
    return acc + ((__int128)w * (__int128)x);
}
"""
