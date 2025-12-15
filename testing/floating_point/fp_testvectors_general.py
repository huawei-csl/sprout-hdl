# fp_vectors_generic.py
from __future__ import annotations
from math import frexp, copysign, isfinite, isnan, isinf, floor
from math import ldexp

from sprouthdl.arithmetic.floating_point.sprout_hdl_float import run_vectors_aby
from sprouthdl.arithmetic.floating_point.sprout_hdl_float_sn import build_fp_mul_sn

# -------------- core helpers: pack/unpack fields --------------


def fp_bias(EW: int) -> int:
    return (1 << (EW - 1)) - 1


def fp_pack(sign: int, exp: int, frac: int, EW: int, FW: int) -> int:
    return ((sign & 1) << (EW + FW)) | ((exp & ((1 << EW) - 1)) << FW) | (frac & ((1 << FW) - 1))


def fp_unpack(bits: int, EW: int, FW: int):
    sign = (bits >> (EW + FW)) & 1
    exp = (bits >> FW) & ((1 << EW) - 1)
    frac = bits & ((1 << FW) - 1)
    return sign, exp, frac


# -------------- decoder: bits -> Python float --------------


def fp_decode(bits: int, EW: int, FW: int) -> float:
    """IEEE-like decode (bias=2^(EW-1)-1), with subnormals and Inf/NaN."""
    s, e, f = fp_unpack(bits, EW, FW)
    bias = fp_bias(EW)
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        # subnormal
        return copysign((f / (1 << FW)) * 2.0 ** (1 - bias), -1.0 if s else 1.0)
    if e == (1 << EW) - 1:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    # normal
    return copysign((1.0 + f / (1 << FW)) * 2.0 ** (e - bias), -1.0 if s else 1.0)


# -------------- encoder: Python float -> bits --------------


def _round_to_even(x: float) -> int:
    """Round x to nearest int, ties to even (assumes |x| < ~2^53)."""
    lo = floor(x)
    frac = x - lo
    if frac < 0.5:
        return int(lo)
    if frac > 0.5:
        return int(lo + 1)
    # exactly .5 → to even
    return int(lo if (lo % 2 == 0) else lo + 1)


def fp_encode(x: float, EW: int, FW: int, *, subnormals: bool = True) -> int:
    """
    Encode Python float into IEEE-like (EW,FW) format with round-to-nearest-even.
    Saturates to ±Inf on overflow. If subnormals=False, underflow → ±0.
    """
    # Handle specials
    if isnan(x):
        # canonical qNaN: exp=all1, frac with MSB set
        return fp_pack(0, (1 << EW) - 1, 1 << (FW - 1 if FW > 0 else 0), EW, FW)
    if isinf(x):
        return fp_pack(1 if x < 0 else 0, (1 << EW) - 1, 0, EW, FW)
    if x == 0.0:
        # preserve sign of zero
        return fp_pack(1 if copysign(1.0, x) < 0 else 0, 0, 0, EW, FW)

    sgn = 1 if x < 0 else 0
    ax = abs(x)
    bias = fp_bias(EW)
    # frexp: ax = m * 2^e, m ∈ [0.5, 1.0)
    m, e = frexp(ax)
    # convert to [1.0, 2.0)
    m *= 2.0
    e -= 1

    # Compute encoded exponent for *normal* case
    E = e + bias
    # Check overflow to Inf
    if E >= (1 << EW) - 1:
        return fp_pack(sgn, (1 << EW) - 1, 0, EW, FW)

    # If E <= 0 we might be subnormal or underflow to zero
    if E <= 0:
        if not subnormals:
            return fp_pack(sgn, 0, 0, EW, FW)
        # subnormal: effectively shift the significand right by (1 - E)
        # target significand to round: m * 2^(FW + E - 1), because normal frac would be (m-1)*2^FW
        # Equivalently: ax / 2^(1-bias) has to be placed into FW fraction bits.
        # Direct way: exact subnormal fraction value (unrounded):
        # value = ax = (f / 2^FW) * 2^(1-bias)  => f = ax * 2^(FW + bias - 1)
        f_unrnd = ax * (2.0 ** (FW + bias - 1))
        f = max(0, min((1 << FW) - 1, _round_to_even(f_unrnd)))
        # If rounding overflowed the subnormal range, pop to min normal:
        if f == (1 << FW):
            return fp_pack(sgn, 1, 0, EW, FW)
        return fp_pack(sgn, 0, f, EW, FW)

    # Normal: fraction = round((m-1)*2^FW)
    frac_unrnd = (m - 1.0) * (1 << FW)
    f = _round_to_even(frac_unrnd)
    # Handle rounding overflow of fraction
    if f == (1 << FW):
        f = 0
        E += 1
        if E >= (1 << EW) - 1:
            return fp_pack(sgn, (1 << EW) - 1, 0, EW, FW)
    return fp_pack(sgn, int(E), int(f), EW, FW)


# -------------- convenience constructors for canonical constants --------------


def bits_zero(EW: int, FW: int, sign: int = 0) -> int:
    return fp_pack(sign, 0, 0, EW, FW)


def bits_inf(EW: int, FW: int, sign: int = 0) -> int:
    return fp_pack(sign, (1 << EW) - 1, 0, EW, FW)


def bits_qnan(EW: int, FW: int) -> int:
    return fp_pack(0, (1 << EW) - 1, 1 << (FW - 1 if FW > 0 else 0), EW, FW)


def bits_min_normal(EW: int, FW: int) -> int:
    return fp_pack(0, 1, 0, EW, FW)


def bits_max_finite(EW: int, FW: int, sign: int = 0) -> int:
    return fp_pack(sign, (1 << EW) - 2, (1 << FW) - 1, EW, FW)


def bits_min_sub(EW: int, FW: int) -> int:
    return fp_pack(0, 0, 1, EW, FW)


def bits_max_sub(EW: int, FW: int) -> int:
    return fp_pack(0, 0, (1 << FW) - 1, EW, FW)


# exact powers of two (when representable as normal)
def bits_pow2(k: int, EW: int, FW: int) -> int:
    E = k + fp_bias(EW)
    if 1 <= E <= (1 << EW) - 2:
        return fp_pack(0, E, 0, EW, FW)
    # handle overflow/underflow crudely
    if E >= (1 << EW) - 1:
        return bits_inf(EW, FW, 0)
    # subnormal or zero:
    # 2^k = (f/2^FW) * 2^(1-bias) => f = 2^(k - (1-bias)) * 2^FW
    f = int(round(2.0 ** (k - (1 - fp_bias(EW))) * (1 << FW)))
    if f <= 0:
        return bits_zero(EW, FW, 0)
    if f >= (1 << FW):
        return bits_min_normal(EW, FW)  # rounded up to min normal
    return fp_pack(0, 0, f, EW, FW)


# -------------- generic test vectors --------------


def build_fp_vectors(EW: int, FW: int):
    """
    Basic sanity vectors (normals/specials); uses encoder so it works for any (EW,FW).
    Returns: list of (name, a_bits, b_bits, expected_bits)
    """
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    four = fp_encode(4.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    onept5 = fp_encode(1.5, EW, FW)
    twopt25 = fp_encode(2.25, EW, FW)
    neg2 = fp_encode(-2.0, EW, FW)
    neg4 = fp_encode(-4.0, EW, FW)
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
        ("(-0) * 2 = (-0)", neg0, two, neg0),  # sign of zero: XOR of signs
        ("Inf * 3 = Inf", pinf, thr, pinf),
        ("(-Inf) * (-2) = +Inf", ninf, neg2, pinf),
        ("Inf * 0 = NaN", pinf, pos0, qnan),
        ("NaN * 2 = NaN", qnan, two, qnan),
        ("Overflow: max*2 = Inf", maxf, two, pinf),
        ("Underflow: min*0.5 = 0", minN, half, pos0),  # with subnormals disabled this holds
        ("(-1)*1 = -1", fp_encode(-1.0, EW, FW), one, fp_encode(-1.0, EW, FW)),
        ("4 * 0.5 = 2", four, half, two),
    ]


def build_fp_subnormal_vectors(EW: int, FW: int):
    """
    Subnormal-focused tests (assumes DUT supports subnormals).
    Uses exact power-of-two scalings so expectations are unambiguous.
    """
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

    # Expected results computed with encoder (subnormals=True)
    return [
        ("minNorm * 0.5  -> sub", minN, half, fp_encode(fp_decode(minN, EW, FW) * 0.5, EW, FW)),
        ("minNorm * 0.25 -> sub", minN, qtr, fp_encode(fp_decode(minN, EW, FW) * 0.25, EW, FW)),
        ("minNorm * 0.125-> sub", minN, eigth, fp_encode(fp_decode(minN, EW, FW) * 0.125, EW, FW)),
        ("minNorm * 0.0625-> sub", minN, six, fp_encode(fp_decode(minN, EW, FW) * 0.0625, EW, FW)),
        ("minSub * 2 -> next sub", minS, two, fp_encode(fp_decode(minS, EW, FW) * 2.0, EW, FW)),
        ("minSub * 3 -> 3*minSub", minS, thr, fp_encode(fp_decode(minS, EW, FW) * 3.0, EW, FW)),
        ("maxSub * 1.0 -> maxSub", maxS, one, maxS),
        ("minSub * 1.0 -> minSub", minS, one, minS),
        # Tie-to-even inside subnormal band: (odd * 0.5) → round to even
        ("maxSub * 0.5  (tie->even)", maxS, half, fp_encode(fp_decode(maxS, EW, FW) * 0.5, EW, FW)),
        ("minSub * minSub -> 0", minS, minS, bits_zero(EW, FW, 0)),
    ]


# -------------- convenience wrappers for common formats --------------


def build_f16_vectors():
    return build_fp_vectors(5, 10)


def build_f16_subnormal_vectors():
    return build_fp_subnormal_vectors(5, 10)


def build_bf16_vectors():
    return build_fp_vectors(8, 7)


def build_bf16_subnormal_vectors():
    return build_fp_subnormal_vectors(8, 7)


def floatx_to_float(bits: int, EW: int, FW: int) -> float:
    """alias decoder for readability"""
    return fp_decode(bits, EW, FW)


def fp_bias(EW: int) -> int:
    return (1 << (EW - 1)) - 1

def fp_limits(EW: int, FW: int):
    """
    Returns a dict with:
      - max_finite, min_finite: largest/smallest finite values
      - pos_overflow_threshold, neg_overflow_threshold:
          the smallest |x| such that encoding(x, EW, FW) → ±Inf
          (i.e., just above/below max_finite, taking rounding into account)
      - min_normal_pos: smallest positive *normal*
      - min_sub_pos:    smallest positive *subnormal* (0.0 if FW==0)
    All values are Python floats (use ldexp to avoid precision surprises).
    """
    bias = fp_bias(EW)
    # Largest finite exponent (encoded): Emax_enc = 2^EW - 2 (not all-ones)
    Emax_enc = (1 << EW) - 2
    e_unbiased = Emax_enc - bias

    # Largest finite mantissa = 2 - 2^{-FW}  ⇒ max finite value:
    max_finite = ldexp(2.0 - 2.0**(-FW), e_unbiased)
    min_finite = -max_finite

    # Overflow threshold (round-to-nearest-even):
    # Any x >= (2 - 2^{-(FW+1)}) * 2^{e_unbiased} encodes to +Inf
    pos_overflow_threshold = ldexp(2.0 - 2.0**(-(FW + 1)), e_unbiased)
    neg_overflow_threshold = -pos_overflow_threshold

    # Smallest positive *normal* and *subnormal*
    min_normal_pos = ldexp(1.0, 1 - bias)  # 1.0 * 2^{1-bias}
    min_sub_pos = 0.0 if FW == 0 else ldexp(1.0, 1 - bias - FW)  # 2^{1-bias-FW}

    return dict(
        max_finite=max_finite,
        min_finite=min_finite,
        pos_overflow_threshold=pos_overflow_threshold,
        neg_overflow_threshold=neg_overflow_threshold,
        min_normal_pos=min_normal_pos,
        min_sub_pos=min_sub_pos,
    )


def main_test():

    # Example: generic vectors for EW, FW:
    ew = 6
    fw = 9

    vecs = build_fp_vectors(ew, fw)
    for name, a, b, exp in vecs[:5]:
        print(name, hex(a), hex(b), hex(exp))

    print("Limits for (6,9):", fp_limits(ew, fw))

    # Decode to floats for logging:
    print("Encode/decode test:")

    # encode and decode some random floats
    for x in [0.1, -2.5, 3.14159, 1e10, 1e-10, float("inf"), float("-inf"), float("nan")]:
        bits = fp_encode(x, ew, fw)
        x2 = fp_decode(bits, ew, fw)
        print(f"x={x} -> bits=0x{bits:04x} -> x2={x2}")

        # handle pos and neg overflow
        x_proc = x
        fp_limit_dict = fp_limits(ew, fw)
        if x >= fp_limit_dict['pos_overflow_threshold']:
            x_proc = float("inf")
        if x <= fp_limit_dict['neg_overflow_threshold']:
            x_proc = float("-inf")

        # assert that it is close (or both NaN)
        assert (x2 == x_proc) or (isnan(x_proc) and isnan(x2)) or \
        (isinf(x) and isinf(x2) and (copysign(1.0,x)==copysign(1.0,x2))) or \
        (isfinite(x) and isfinite(x2) and abs(x2 - x_proc) <= abs(x)*1e-3), f"Mismatch: {x} -> {x2}"

    # simulation
    subnormals = True
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)
    run_vectors_aby(m, build_fp_vectors(ew, fw), label=f"float{ew+fw+1} normal cases", decoder=lambda b: floatx_to_float(b, ew, fw))
    if subnormals:
        run_vectors_aby(m, build_fp_subnormal_vectors(ew, fw), label=f"float{ew+fw+1} subnormal cases", decoder=lambda b: floatx_to_float(b, ew, fw))


if __name__ == "__main__":
    main_test()