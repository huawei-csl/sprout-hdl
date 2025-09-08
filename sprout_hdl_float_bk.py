# =========================
# Floating-point multiplier
# (now with optional subnormals)
# =========================
# Parameterizable by exponent width (EW) and fraction width (FW).
# - float16:  EW=5, FW=10
# - bfloat16: EW=8, FW=7
#
# Args:
#   name: module name
#   EW, FW: widths
#   subnormals: enable subnormal in/out (gradual underflow). Default False = flush-to-zero.
#
# Requires: Module, UInt, Bool, mux, cat from your sprout_hdl.

from typing import List
from sprout_hdl import *
from sprout_hdl_float import bf16_to_float, build_bf16_vectors, build_f16_vectors, build_fp_mul, half_to_float, run_vectors
from sprout_hdl_module import Module
from sprout_hdl_simulator import Simulator


# Uses: Module, UInt, Bool, mux, cat (from your sprout_hdl)


def _or_reduce_bits(vec_expr, hi: int, lo: int):
    """OR-reduce bits vec_expr[hi:lo] (inclusive). If hi < lo, returns 0."""
    if hi < lo:
        return 0
    acc = 0
    for i in range(lo, hi + 1):
        acc = acc | vec_expr[i]
    return acc


def _build_prefix_ors(vec_bits: List):
    """
    Given a bit-vector 'vec_bits' indexed [0..N-1] (LSB-first),
    returns prefix[i] = OR(vec_bits[0:i]) inclusive; i.e., prefix[0]=b0, prefix[1]=b0|b1, ...
    """
    prefix = []
    if len(vec_bits) == 0:
        return prefix
    acc = vec_bits[0]
    prefix.append(acc)
    for i in range(1, len(vec_bits)):
        acc = acc | vec_bits[i]
        prefix.append(acc)
    return prefix


def build_fp_mul_sn(name: str, EW: int, FW: int, *, subnormals: bool = False) -> "Module":
    W = 1 + EW + FW
    BIAS = (1 << (EW - 1)) - 1
    MAX_E = (1 << EW) - 1
    MAX_FINITE_E = MAX_E - 1

    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(W), "a")
    b = m.input(UInt(W), "b")
    y = m.output(UInt(W), "y")

    # ------------------
    # Field extraction
    # ------------------
    sA = a[W - 1]
    sB = b[W - 1]

    eA = a[W - 2 : FW]
    eB = b[W - 2 : FW]

    fA = a[FW - 1 : 0]
    fB = b[FW - 1 : 0]

    # ------------------
    # Classifiers
    # ------------------
    is_eA_zero = eA == 0
    is_eB_zero = eB == 0
    is_fA_zero = fA == 0
    is_fB_zero = fB == 0

    is_zeroA = is_eA_zero & is_fA_zero
    is_zeroB = is_eB_zero & is_fB_zero

    is_eA_all1 = eA == MAX_E
    is_eB_all1 = eB == MAX_E

    is_nanA = is_eA_all1 & (fA != 0)
    is_nanB = is_eB_all1 & (fB != 0)

    is_infA = is_eA_all1 & (fA == 0)
    is_infB = is_eB_all1 & (fB == 0)

    # NaN if: any NaN, or Inf * 0 (either way)
    is_nan_in = is_nanA | is_nanB | ((is_infA & is_zeroB) | (is_zeroA & is_infB))
    # Inf if: any Inf (and not NaN case)
    is_inf_in = is_infA | is_infB
    # Zero if: any zero (input) (we’ll combine later with underflow/subnormal case)
    is_zero_in = is_zeroA | is_zeroB

    # Sign
    sY = sA ^ sB

    # ------------------
    # Effective mantissas and exponents
    # ------------------
    # Hidden bit for normals; 0 for subnormals if enabled, else mask to zero.
    if subnormals:
        hiddenA = mux(is_eA_zero, 0, 1)
        hiddenB = mux(is_eB_zero, 0, 1)
        mA_eff = cat(hiddenA, fA)  # 1+FW bits
        mB_eff = cat(hiddenB, fB)
        # For exponent math, use e_eff = (e==0 ? 1 : e) to realize unbiased (1−BIAS) for subnormals.
        eA_eff = mux(is_eA_zero, 1, eA)
        eB_eff = mux(is_eB_zero, 1, eB)
    else:
        # Old behavior: subnormals treated as zero
        msk = (1 << (1 + FW)) - 1
        mA_eff = (cat(1, fA)) & mux(is_eA_zero, 0, msk)
        mB_eff = (cat(1, fB)) & mux(is_eB_zero, 0, msk)
        eA_eff = eA
        eB_eff = eB

    # ------------------
    # Multiply mantissas
    # ------------------
    prod = mA_eff * mB_eff  # width 2 + 2*FW
    PROD_W = 2 + 2 * FW

    # Normalization for mantissa (coarse)
    msb_high = prod[PROD_W - 1]  # 1 if ≥ 2.0

    top_hi = prod[PROD_W - 1 : FW + 1]  #  (1+FW) bits when msb_high=1 (right shift by 1)
    top_lo = prod[PROD_W - 2 : FW]  #  (1+FW) bits when msb_high=0
    mant_pre = mux(msb_high, top_hi, top_lo)  # (1+FW) bits, may be <1.0 if both operands subnormal

    # Rounding (round-to-nearest-even) for the *normal* path
    guard_hi = prod[FW]  # for msb_high=1
    guard_lo = prod[FW - 1]  # for msb_high=0
    guard_n = mux(msb_high, guard_hi, guard_lo)

    sticky_hi = _or_reduce_bits(prod, FW - 1, 0)
    sticky_lo = _or_reduce_bits(prod, FW - 2, 0)
    sticky_n = mux(msb_high, sticky_hi, sticky_lo)

    lsb_n = mant_pre[0]
    round_up_n = guard_n & (sticky_n | lsb_n)

    mant_round = mant_pre + mux(round_up_n, 1, 0)  # width (1+FW)+1
    carry = mant_round[FW + 1]  # rounding carry into MSB
    mant_post = mux(carry, mant_round[FW + 1 : 1], mant_round[FW:0])  # (1+FW)
    frac_norm = mant_post[FW - 1 : 0]

    # ------------------
    # Exponent path (biased result; works for normals & subnormals)
    # ------------------
    exp_sum = eA_eff + eB_eff  # EW+1
    inc1 = mux(msb_high, 1, 0)
    inc2 = mux(carry, 1, 0)
    exp_sum_inc = (exp_sum + inc1) + inc2  # EW+2

    bias_const = BIAS
    maxfinite_plus_bias = BIAS + MAX_FINITE_E

    underflow = exp_sum_inc <= bias_const  # E_out <= 0  → subnormal or zero if enabled
    overflow = exp_sum_inc > maxfinite_plus_bias

    # This is *already biased* exponent after normalization & rounding:
    exp_biased = exp_sum_inc - bias_const  # EW+2
    exp_norm = exp_biased[EW - 1 : 0]  # packable EW bits

    # ------------------
    # Subnormal output path (only if enabled and underflow)
    # ------------------
    if subnormals:
        # shift_amt = 1 - E_biased  = (BIAS + 1) - exp_sum_inc
        shift_amt = (BIAS + 1) - exp_sum_inc  # variable, >=1 when underflow is true

        # We’ll right-shift mant_post by 'shift_amt' to produce the subnormal fraction (no hidden 1).
        sig_pre = mant_post  # (1+FW) bits
        sig_shiftN = sig_pre >> shift_amt  # (1+FW) bits after shift
        frac_trunc_sub = sig_shiftN[FW - 1 : 0]  # FW bits (truncated)

        # guard bit at position (shift_amt - 1), sticky = OR of bits [0 .. shift_amt-2]
        shm1 = shift_amt - 1

        # Guard: bit 0 of (sig_pre >> (shift_amt - 1))
        guard_s = (sig_pre >> shm1)[0]

        # Sticky: prefix OR up to (shm1 - 1)
        # Build prefix ORs of sig_pre[0..FW]
        sig_bits = [sig_pre[i] for i in range(FW + 1)]  # indices 0..FW (LSB..MSB)
        prefix = _build_prefix_ors(sig_bits)  # prefix[i] = OR(sig_pre[0..i])
        # default sticky: if (shm1 > FW) → OR of all [0..FW], else 0
        sticky_default = mux(shm1 > FW, prefix[FW], 0)
        sticky_s = sticky_default
        # If shm1 == k (k in 1..FW), sticky = OR(sig_pre[0..k-1]) = prefix[k-1]
        for k in range(1, FW + 1):
            sticky_s = mux(shm1 == k, prefix[k - 1], sticky_s)

        lsb_s = sig_shiftN[0]
        round_up_s = guard_s & (sticky_s | lsb_s)

        # Add rounding increment to truncated FW bits
        frac_trunc_zext = cat(0, frac_trunc_sub)  # FW+1 bits
        frac_sum = frac_trunc_zext + mux(round_up_s, 1, 0)  # FW+1 bits
        carry_s = frac_sum[FW]  # overflow → becomes min normal
        frac_sub = frac_sum[FW - 1 : 0]

        # When rounding a subnormal overflows, pop to min normal: exp=1, frac=0
        exp_field_sub = mux(carry_s, 1, 0)
        frac_field_sub = mux(carry_s, 0, frac_sub)
    # ------------------
    # Pack (priority: NaN > Inf/overflow > Zero > Normal/Subnormal)
    # ------------------

    all1_E = (1 << EW) - 1
    qnan_payload = (1 << (FW - 1)) if FW > 0 else 1

    is_nan = is_nan_in
    is_inf = (~is_nan) & (is_inf_in | overflow)

    if subnormals:
        is_sub_out = (~is_nan) & (~is_inf) & underflow  # may end up zero if frac_sub==0 and no carry
        # Zero if true zero-in OR (subnormal path yields 0 fraction and no carry)
        frac_sub_is_zero = 0 if not subnormals else (frac_field_sub == 0)
        is_zero = (~is_nan) & (~is_inf) & (is_zero_in | (is_sub_out & frac_sub_is_zero))
        # Choose exponent/fraction fields
        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, mux(is_sub_out, exp_field_sub, exp_norm)))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, mux(is_sub_out, frac_field_sub, frac_norm)))
    else:
        # Original behavior: underflow → zero
        is_zero = (~is_nan) & (~is_inf) & (is_zero_in | underflow)
        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, exp_norm))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, frac_norm))

    sign_field = mux(is_nan, 0, sY)
    y <<= cat(sign_field, exp_field, frac_field)
    return m

# sanity_fp_mul_subnormals.py
# Requires your SproutHDL builders with subnormals enabled:
#   build_f16_mul(..., subnormals=True)
#   build_bf16_mul(..., subnormals=True)
# and your Simulator (inline or separate).


# --- decode helpers (for nice printouts) ---
# def half_to_float(h):
#     s = (h >> 15) & 1
#     e = (h >> 10) & 0x1F
#     f = h & 0x3FF
#     bias = 15
#     if e == 0:
#         if f == 0:
#             return -0.0 if s else 0.0
#         return ((-1.0) ** s) * (2 ** (1 - bias)) * (f / (1 << 10))
#     if e == 0x1F:
#         if f == 0:
#             return float("-inf") if s else float("inf")
#         return float("nan")
#     return ((-1.0) ** s) * (2 ** (e - bias)) * (1.0 + f / (1 << 10))

# def bf16_to_float(b):
#     s = (b >> 15) & 1
#     e = (b >> 7) & 0xFF
#     f = b & 0x7F
#     bias = 127
#     if e == 0:
#         if f == 0:
#             return -0.0 if s else 0.0
#         return ((-1.0) ** s) * (2 ** (1 - bias)) * (f / (1 << 7))
#     if e == 0xFF:
#         if f == 0:
#             return float("-inf") if s else float("inf")
#         return float("nan")
#     return ((-1.0) ** s) * (2 ** (e - bias)) * (1.0 + f / (1 << 7))

# # --- generic runner ---
# def run_vectors(mod, vectors, *, label="", decoder=None):
#     sim = Simulator(mod)
#     print(f"\n== {label} ==")
#     ok = 0
#     for name, a_hex, b_hex, exp_hex in vectors:
#         sim.set("a", a_hex).set("b", b_hex).eval()
#         got = sim.get("y")
#         pf = "PASS" if got == exp_hex else "FAIL"
#         extra = ""
#         if decoder is not None:
#             extra = f"  val={decoder(got):.8g} (exp {decoder(exp_hex):.8g})"
#         print(f"{pf:4s}  {name:35s} a=0x{a_hex:04X}  b=0x{b_hex:04X} -> y=0x{got:04X} (exp 0x{exp_hex:04X}){extra}")
#         ok += (got == exp_hex)
#     print(f"Summary: {ok}/{len(vectors)} passed.\n")

# -----------------------
# float16 (binary16) vectors with SUBNORMALS
# -----------------------
# Constants:
#   min normal   = 0x0400  (2^-14)
#   0.5          = 0x3800
#   0.25         = 0x3400
#   0.125        = 0x3000
#   0.0625       = 0x2C00
#   1.0          = 0x3C00
#   2.0          = 0x4000
#   3.0          = 0x4200
#   max subnorm  = 0x03FF
#   min subnorm  = 0x0001
def build_f16_subnormal_vectors():
    return [
        # Exact power-of-two scalings from min normal → subnormals
        ("minNorm * 0.5  -> sub",            0x0400, 0x3800, 0x0200),  # 2^-14 * 2^-1 = 2^-15
        ("minNorm * 0.25 -> sub",            0x0400, 0x3400, 0x0100),  # 2^-16
        ("minNorm * 0.125-> sub",            0x0400, 0x3000, 0x0080),  # 2^-17
        ("minNorm * 0.0625-> sub",           0x0400, 0x2C00, 0x0040),  # 2^-18

        # Subnormal scaling upward/downward by powers of two
        ("minSub * 2 -> next sub",           0x0001, 0x4000, 0x0002),
        ("minSub * 3 -> 3*minSub",           0x0001, 0x4200, 0x0003),

        # Subnormal * 1.0 should be identity (key correctness check)
        ("maxSub * 1.0 -> maxSub",           0x03FF, 0x3C00, 0x03FF),
        ("minSub * 1.0 -> minSub",           0x0001, 0x3C00, 0x0001),

        # Tie-to-even inside subnormal range
        # 0x03FF * 0.5 = (1023/2) → 511.5 → ties to even = 512 (0x0200)
        ("maxSub * 0.5  (tie->even)",        0x03FF, 0x3800, 0x0200),

        # Subnormal * subnormal → still subnormal or zero (exact small values)
        ("minSub * minSub -> 0",             0x0001, 0x0001, 0x0000),
    ]

# -----------------------
# bfloat16 vectors with SUBNORMALS
# -----------------------
# Constants:
#   min normal   = 0x0080  (2^-126)
#   0.5          = 0x3F00
#   0.25         = 0x3E80
#   0.125        = 0x3E00
#   1.0          = 0x3F80
#   2.0          = 0x4000
#   3.0          = 0x4040
#   max subnorm  = 0x007F
#   min subnorm  = 0x0001
def build_bf16_subnormal_vectors():
    return [
        # Exact power-of-two scalings from min normal → subnormals
        ("minNorm * 0.5  -> sub",            0x0080, 0x3F00, 0x0040),  # 2^-127
        ("minNorm * 0.25 -> sub",            0x0080, 0x3E80, 0x0020),  # 2^-128
        ("minNorm * 0.125-> sub",            0x0080, 0x3E00, 0x0010),  # 2^-129

        # Subnormal scaling
        ("minSub * 2 -> next sub",           0x0001, 0x4000, 0x0002),
        ("minSub * 3 -> 3*minSub",           0x0001, 0x4040, 0x0003),

        # Subnormal * 1.0 should be identity (key correctness check)
        ("maxSub * 1.0 -> maxSub",           0x007F, 0x3F80, 0x007F),
        ("minSub * 1.0 -> minSub",           0x0001, 0x3F80, 0x0001),

        # Tie-to-even in subnormal range: 127/2 = 63.5 → round to 64 (0x0040)
        ("maxSub * 0.5  (tie->even)",        0x007F, 0x3F00, 0x0040),

        # Subnormal * subnormal → zero (very tiny)
        ("minSub * minSub -> 0",             0x0001, 0x0001, 0x0000),
    ]

if __name__ == "__main__":
    #f16 = build_fp_mul_sn("F16Mul_Sub", EW=5, FW=10, subnormals=False)
    #bf16 = build_fp_mul_sn("BF16Mul_Sub", EW=8, FW=7, subnormals=False)
    f16 = build_fp_mul_sn("F16Mul_Sub", EW=5, FW=10, subnormals=True)
    bf16 = build_fp_mul_sn("BF16Mul_Sub", EW=8, FW=7, subnormals=True)

    run_vectors(f16, build_f16_vectors(), label="float16 default cases", decoder=half_to_float)
    run_vectors(bf16, build_bf16_vectors(), label="bfloat16 default cases", decoder=bf16_to_float)
    run_vectors(f16,  build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=half_to_float)
    run_vectors(bf16, build_bf16_subnormal_vectors(), label="bfloat16 subnormal cases", decoder=bf16_to_float)
