"""Floating-point bit encoding/decoding utilities for arbitrary (EW, FW) formats.

Pure-Python, no hardware dependencies; safe to import from both src/ and testing/.
"""
from __future__ import annotations

from math import copysign, floor, frexp, isinf, isnan, ldexp


def fp_bias(ew: int) -> int:
    return (1 << (ew - 1)) - 1


def fp_pack(sign: int, exp: int, frac: int, ew: int, fw: int) -> int:
    return ((sign & 1) << (ew + fw)) | ((exp & ((1 << ew) - 1)) << fw) | (frac & ((1 << fw) - 1))


def fp_unpack(bits: int, ew: int, fw: int) -> tuple[int, int, int]:
    sign = (bits >> (ew + fw)) & 1
    exp = (bits >> fw) & ((1 << ew) - 1)
    frac = bits & ((1 << fw) - 1)
    return sign, exp, frac


def _round_to_even(x: float) -> int:
    lo = floor(x)
    frac = x - lo
    if frac < 0.5:
        return int(lo)
    if frac > 0.5:
        return int(lo + 1)
    return int(lo if lo % 2 == 0 else lo + 1)


def fp_decode(bits: int, ew: int, fw: int) -> float:
    """IEEE-like decode (bias=2^(EW-1)-1), with subnormals and Inf/NaN."""
    s, e, f = fp_unpack(bits, ew, fw)
    bias = fp_bias(ew)
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return copysign((f / (1 << fw)) * 2.0 ** (1 - bias), -1.0 if s else 1.0)
    if e == (1 << ew) - 1:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return copysign((1.0 + f / (1 << fw)) * 2.0 ** (e - bias), -1.0 if s else 1.0)


def fp_encode(x: float, ew: int, fw: int, *, subnormals: bool = True) -> int:
    """Encode Python float into IEEE-like (EW, FW) format with round-to-nearest-even."""
    if isnan(x):
        return fp_pack(0, (1 << ew) - 1, 1 << (fw - 1 if fw > 0 else 0), ew, fw)
    if isinf(x):
        return fp_pack(1 if x < 0 else 0, (1 << ew) - 1, 0, ew, fw)
    if x == 0.0:
        return fp_pack(1 if copysign(1.0, x) < 0 else 0, 0, 0, ew, fw)

    sgn = 1 if x < 0 else 0
    ax = abs(x)
    bias = fp_bias(ew)
    m, e = frexp(ax)
    m *= 2.0
    e -= 1

    E = e + bias
    if E >= (1 << ew) - 1:
        return fp_pack(sgn, (1 << ew) - 1, 0, ew, fw)

    if E <= 0:
        if not subnormals:
            return fp_pack(sgn, 0, 0, ew, fw)
        f_unrnd = ax * (2.0 ** (fw + bias - 1))
        f = max(0, min((1 << fw) - 1, _round_to_even(f_unrnd)))
        if f == (1 << fw):
            return fp_pack(sgn, 1, 0, ew, fw)
        return fp_pack(sgn, 0, f, ew, fw)

    frac_unrnd = (m - 1.0) * (1 << fw)
    f = _round_to_even(frac_unrnd)
    if f == (1 << fw):
        f = 0
        E += 1
        if E >= (1 << ew) - 1:
            return fp_pack(sgn, (1 << ew) - 1, 0, ew, fw)
    return fp_pack(sgn, int(E), int(f), ew, fw)


def fp_add_hw_ref(a_bits: int, b_bits: int, ew: int, fw: int) -> int:
    """Python reference matching hardware FpAdd: extends by 2 guard bits, then truncates.

    The hardware does cat(Const(0, UInt(2)), m) which in sprouthdl's LSB-first
    convention means m_ext = m << 2 (2 zero guard bits at LSB).  After addition
    and normalization, bits 0 and 1 are discarded without rounding.  This differs
    from fp_encode(a+b) which rounds to nearest-even.
    """
    max_e = (1 << ew) - 1

    sA, eA, fA = fp_unpack(a_bits, ew, fw)
    sB, eB, fB = fp_unpack(b_bits, ew, fw)

    is_nanA = eA == max_e and fA != 0
    is_nanB = eB == max_e and fB != 0
    is_infA = eA == max_e and fA == 0
    is_infB = eB == max_e and fB == 0

    if is_nanA or is_nanB or (is_infA and is_infB and sA != sB):
        nan_payload = (1 << (fw - 1)) if fw > 0 else 1
        return fp_pack(0, max_e, nan_payload, ew, fw)
    if is_infA:
        return fp_pack(sA, max_e, 0, ew, fw)
    if is_infB:
        return fp_pack(sB, max_e, 0, ew, fw)

    hA = 0 if eA == 0 else 1
    hB = 0 if eB == 0 else 1
    mA = (hA << fw) | fA
    mB = (hB << fw) | fB
    eA_eff = 1 if eA == 0 else eA
    eB_eff = 1 if eB == 0 else eB

    a_is_bigger = (eA_eff > eB_eff) or (eA_eff == eB_eff and mA >= mB)
    if a_is_bigger:
        exp_delta, m_big, m_small, s_big, s_small, e_big = eA_eff - eB_eff, mA, mB, sA, sB, eA_eff
    else:
        exp_delta, m_big, m_small, s_big, s_small, e_big = eB_eff - eA_eff, mB, mA, sB, sA, eB_eff

    m_big_ext = m_big << 2
    m_small_shift = (m_small << 2) >> exp_delta

    mant_mag = m_big_ext + m_small_shift if s_big == s_small else m_big_ext - m_small_shift

    if mant_mag == 0:
        return fp_pack(s_big & s_small, 0, 0, ew, fw)

    lead_pos = mant_mag.bit_length() - 1
    target_pos = fw + 2

    if lead_pos > target_pos:
        shift = lead_pos - target_pos
        mant_norm = mant_mag >> shift
        exp_out = e_big + shift
    elif lead_pos < target_pos:
        shift = target_pos - lead_pos
        mant_norm = mant_mag << shift
        exp_out = e_big - shift
    else:
        mant_norm = mant_mag
        exp_out = e_big

    if exp_out <= 0:
        return fp_pack(s_big, 0, 0, ew, fw)
    if exp_out >= max_e:
        return fp_pack(s_big, max_e, 0, ew, fw)

    frac_out = (mant_norm >> 2) & ((1 << fw) - 1)
    return fp_pack(s_big, exp_out, frac_out, ew, fw)


# ---------------------------------------------------------------------------
# Convenience constructors for canonical bit patterns
# ---------------------------------------------------------------------------

def bits_zero(ew: int, fw: int, sign: int = 0) -> int:
    return fp_pack(sign, 0, 0, ew, fw)


def bits_inf(ew: int, fw: int, sign: int = 0) -> int:
    return fp_pack(sign, (1 << ew) - 1, 0, ew, fw)


def bits_qnan(ew: int, fw: int) -> int:
    return fp_pack(0, (1 << ew) - 1, 1 << (fw - 1 if fw > 0 else 0), ew, fw)


def bits_min_normal(ew: int, fw: int) -> int:
    return fp_pack(0, 1, 0, ew, fw)


def bits_max_finite(ew: int, fw: int, sign: int = 0) -> int:
    return fp_pack(sign, (1 << ew) - 2, (1 << fw) - 1, ew, fw)


def bits_min_sub(ew: int, fw: int) -> int:
    return fp_pack(0, 0, 1, ew, fw)


def bits_max_sub(ew: int, fw: int) -> int:
    return fp_pack(0, 0, (1 << fw) - 1, ew, fw)


def bits_pow2(k: int, ew: int, fw: int) -> int:
    """Exact power of two 2^k, when representable as a normal."""
    E = k + fp_bias(ew)
    if 1 <= E <= (1 << ew) - 2:
        return fp_pack(0, E, 0, ew, fw)
    if E >= (1 << ew) - 1:
        return bits_inf(ew, fw, 0)
    f = int(round(2.0 ** (k - (1 - fp_bias(ew))) * (1 << fw)))
    if f <= 0:
        return bits_zero(ew, fw, 0)
    if f >= (1 << fw):
        return bits_min_normal(ew, fw)
    return fp_pack(0, 0, f, ew, fw)


def fp_limits(ew: int, fw: int) -> dict:
    """Return a dict with max_finite, min_finite, overflow thresholds, min_normal_pos, min_sub_pos."""
    bias = fp_bias(ew)
    Emax_enc = (1 << ew) - 2
    e_unbiased = Emax_enc - bias
    max_finite = ldexp(2.0 - 2.0 ** (-fw), e_unbiased)
    pos_overflow_threshold = ldexp(2.0 - 2.0 ** (-(fw + 1)), e_unbiased)
    return dict(
        max_finite=max_finite,
        min_finite=-max_finite,
        pos_overflow_threshold=pos_overflow_threshold,
        neg_overflow_threshold=-pos_overflow_threshold,
        min_normal_pos=ldexp(1.0, 1 - bias),
        min_sub_pos=0.0 if fw == 0 else ldexp(1.0, 1 - bias - fw),
    )
