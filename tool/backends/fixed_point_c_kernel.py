from __future__ import annotations

from verification.arith_kernel import render_arith_kernel


def fixed_point_c_kernel_source() -> str:
    """Compatibility wrapper for the verification-side arithmetic kernel."""

    return render_arith_kernel()
