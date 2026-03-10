# fp_vectors_generic.py
from __future__ import annotations

from sprouthdl.arithmetic.floating_point.fp_encoding import (
    bits_inf,
    bits_max_finite,
    bits_max_sub,
    bits_min_normal,
    bits_min_sub,
    bits_pow2,
    bits_qnan,
    bits_zero,
    fp_bias,
    fp_decode,
    fp_encode,
    fp_limits,
    fp_pack,
    fp_unpack,
)
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import run_vectors_aby
from sprouthdl.arithmetic.floating_point.sprout_hdl_float_sn import build_fp_mul_sn


def floatx_to_float(bits: int, EW: int, FW: int) -> float:
    """Alias for fp_decode."""
    return fp_decode(bits, EW, FW)


# ---------------------------------------------------------------------------
# Generic test vector builders
# ---------------------------------------------------------------------------


def build_fp_vectors(EW: int, FW: int):
    """Basic sanity vectors (normals/specials); works for any (EW, FW)."""
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    four = fp_encode(4.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    onept5 = fp_encode(1.5, EW, FW)
    neg2 = fp_encode(-2.0, EW, FW)
    pos0 = bits_zero(EW, FW, 0)
    neg0 = bits_zero(EW, FW, 1)
    pinf = bits_inf(EW, FW, 0)
    ninf = bits_inf(EW, FW, 1)
    qnan = bits_qnan(EW, FW)
    maxf = bits_max_finite(EW, FW)
    minN = bits_min_normal(EW, FW)

    return [
        ("1*2 = 2", one, two, fp_encode(2.0, EW, FW)),
        ("(-2)*2 = -4", neg2, two, fp_encode(-4.0, EW, FW)),
        ("1.5*1.5 = 2.25", onept5, onept5, fp_encode(2.25, EW, FW)),
        ("3*0.5 = 1.5", thr, half, onept5),
        ("0 * 1 = 0", pos0, one, pos0),
        ("(-0) * 2 = (-0)", neg0, two, neg0),
        ("Inf * 3 = Inf", pinf, thr, pinf),
        ("(-Inf) * (-2) = +Inf", ninf, neg2, pinf),
        ("Inf * 0 = NaN", pinf, pos0, qnan),
        ("NaN * 2 = NaN", qnan, two, qnan),
        ("Overflow: max*2 = Inf", maxf, two, pinf),
        ("Underflow: min*0.5 = 0", minN, half, pos0),
        ("(-1)*1 = -1", fp_encode(-1.0, EW, FW), one, fp_encode(-1.0, EW, FW)),
        ("4 * 0.5 = 2", four, half, two),
    ]


def build_fp_subnormal_vectors(EW: int, FW: int):
    """Subnormal-focused tests (assumes DUT supports subnormals)."""
    half = fp_encode(0.5, EW, FW)
    qtr = fp_encode(0.25, EW, FW)
    eigth = fp_encode(0.125, EW, FW)
    six = fp_encode(0.0625, EW, FW)
    minN = bits_min_normal(EW, FW)
    minS = bits_min_sub(EW, FW)
    maxS = bits_max_sub(EW, FW)
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)

    return [
        ("minNorm * 0.5  -> sub", minN, half, fp_encode(fp_decode(minN, EW, FW) * 0.5, EW, FW)),
        ("minNorm * 0.25 -> sub", minN, qtr, fp_encode(fp_decode(minN, EW, FW) * 0.25, EW, FW)),
        ("minNorm * 0.125-> sub", minN, eigth, fp_encode(fp_decode(minN, EW, FW) * 0.125, EW, FW)),
        ("minNorm * 0.0625-> sub", minN, six, fp_encode(fp_decode(minN, EW, FW) * 0.0625, EW, FW)),
        ("minSub * 2 -> next sub", minS, two, fp_encode(fp_decode(minS, EW, FW) * 2.0, EW, FW)),
        ("minSub * 3 -> 3*minSub", minS, thr, fp_encode(fp_decode(minS, EW, FW) * 3.0, EW, FW)),
        ("maxSub * 1.0 -> maxSub", maxS, one, maxS),
        ("minSub * 1.0 -> minSub", minS, one, minS),
        ("maxSub * 0.5  (tie->even)", maxS, half, fp_encode(fp_decode(maxS, EW, FW) * 0.5, EW, FW)),
        ("minSub * minSub -> 0", minS, minS, bits_zero(EW, FW, 0)),
    ]


# ---------------------------------------------------------------------------
# Convenience wrappers for common formats
# ---------------------------------------------------------------------------

def build_f16_vectors():
    return build_fp_vectors(5, 10)


def build_f16_subnormal_vectors():
    return build_fp_subnormal_vectors(5, 10)


def build_bf16_vectors():
    return build_fp_vectors(8, 7)


def build_bf16_subnormal_vectors():
    return build_fp_subnormal_vectors(8, 7)


# ---------------------------------------------------------------------------
# Manual test / demo
# ---------------------------------------------------------------------------

def main_test():
    from math import isinf, isnan, copysign, isfinite

    ew = 6
    fw = 9

    vecs = build_fp_vectors(ew, fw)
    for name, a, b, exp in vecs[:5]:
        print(name, hex(a), hex(b), hex(exp))

    print("Limits for (6,9):", fp_limits(ew, fw))

    print("Encode/decode test:")
    for x in [0.1, -2.5, 3.14159, 1e10, 1e-10, float("inf"), float("-inf"), float("nan")]:
        bits = fp_encode(x, ew, fw)
        x2 = fp_decode(bits, ew, fw)
        print(f"x={x} -> bits=0x{bits:04x} -> x2={x2}")

        x_proc = x
        fp_limit_dict = fp_limits(ew, fw)
        if x >= fp_limit_dict["pos_overflow_threshold"]:
            x_proc = float("inf")
        if x <= fp_limit_dict["neg_overflow_threshold"]:
            x_proc = float("-inf")

        assert (x2 == x_proc) or (isnan(x_proc) and isnan(x2)) or \
            (isinf(x) and isinf(x2) and (copysign(1.0, x) == copysign(1.0, x2))) or \
            (isfinite(x) and isfinite(x2) and abs(x2 - x_proc) <= abs(x) * 1e-3), \
            f"Mismatch: {x} -> {x2}"

    subnormals = True
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)
    run_vectors_aby(m, build_fp_vectors(ew, fw), label=f"float{ew+fw+1} normal cases",
                    decoder=lambda b: floatx_to_float(b, ew, fw))
    if subnormals:
        run_vectors_aby(m, build_fp_subnormal_vectors(ew, fw), label=f"float{ew+fw+1} subnormal cases",
                        decoder=lambda b: floatx_to_float(b, ew, fw))


if __name__ == "__main__":
    main_test()
