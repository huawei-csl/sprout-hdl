# =========================
# Floating-point multiplier (optimized)
# - Fast path for normal×normal: 0/1-bit normalize (no LZC, no left shifter)
# - Single-direction (right) shifting only
# - Safer dynamic-index helpers (no negative index hazards)
# - Leaner guard/sticky computation on the hot path
# =========================

from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import *
from sprouthdl.floating_point.sprout_hdl_float import bf16_to_float, build_bf16_vectors, build_f16_vectors, build_fp_mul, half_to_float, run_vectors


# ---- lean helpers ----
def _or_reduce_bits(vec_expr, hi: int, lo: int):
    """OR-reduce inclusive range [lo..hi]; returns 0 if range is empty."""
    if hi < lo:
        return 0
    acc = vec_expr[lo]
    for i in range(lo + 1, hi + 1):
        acc = acc | vec_expr[i]
    return acc

def _prefix_or_bits(x, width):
    """Return [ OR(x[0:0]), OR(x[0:1]), ..., OR(x[0:width-1]) ]"""
    out = []
    if width <= 0:
        return out
    acc = x[0]
    out.append(acc)
    for i in range(1, width):
        acc = acc | x[i]
        out.append(acc)
    return out

def build_fp_mul_sn(name: str, EW: int, FW: int, *, subnormals: bool = True) -> "Module":
    W = 1 + EW + FW
    BIAS = (1 << (EW - 1)) - 1
    MAX_E = (1 << EW) - 1
    MAX_FINITE_E = MAX_E - 1

    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(W), "a")
    b = m.input(UInt(W), "b")
    y = m.output(UInt(W), "y")

    # Fields (LSB..MSB; sign is MSB)
    sA = a[W - 1]
    sB = b[W - 1]
    eA = a[FW : W - 1]
    eB = b[FW : W - 1]
    fA = a[0:FW]
    fB = b[0:FW]

    # Classify
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

    is_nan_in = is_nanA | is_nanB | ((is_infA & is_zeroB) | (is_zeroA & is_infB))
    is_inf_in = is_infA | is_infB
    is_zero_in = is_zeroA | is_zeroB

    sY = sA ^ sB

    # Effective significands/exponents
    if subnormals:
        hiddenA = mux(is_eA_zero, 0, 1)
        hiddenB = mux(is_eB_zero, 0, 1)
        mA_eff = cat(fA, hiddenA)  # (1+FW)
        mB_eff = cat(fB, hiddenB)
        eA_eff = mux(is_eA_zero, 1, eA)  # subnormals behave as exponent=1 (unbiased 1-BIAS)
        eB_eff = mux(is_eB_zero, 1, eB)
    else:
        mask = (1 << (1 + FW)) - 1
        mA_eff = (cat(fA, 1)) & mux(is_eA_zero, 0, mask)
        mB_eff = (cat(fB, 1)) & mux(is_eB_zero, 0, mask)
        eA_eff = eA
        eB_eff = eB

    # ---------------- Product ----------------
    prod = mA_eff * mB_eff                  # width = 2 + 2*FW
    PROD_W = 2 + 2 * FW
    TOP = PROD_W - 1                        # = 2*FW + 1

    # === Fast normalize for normal×normal ===
    # Product of two [1,2) values is in [1,4) → top two bits decide 0/1-bit right shift.
    hi1 = prod[TOP]                         # top bit (1 → shift by 1)
    # Mantissa windows for the two cases (width 1+FW each):
    # case 0 (no shift): take [FW .. 2*FW]    → slice [FW : 2*FW+1)
    mant0 = prod[FW : 2 * FW + 1]
    # case 1 (shift by 1): take [FW+1 .. 2*FW+1] → slice [FW+1 : 2*FW+2)
    mant1 = prod[FW + 1 : 2 * FW + 2]
    mant_pre_fast = mux(hi1, mant1, mant0)

    # Guard/sticky (constant positions per case)
    if FW > 0:
        g0 = prod[FW - 1]
        g1 = prod[FW]
    else:
        g0 = 0
        g1 = 0
    sticky0 = _or_reduce_bits(prod, FW - 2, 0)          # below g0
    sticky1 = _or_reduce_bits(prod, FW - 1, 0)          # below g1
    guard_fast  = mux(hi1, g1, g0)
    sticky_fast = mux(hi1, sticky1, sticky0)

    # === (Optional) LZC for subnormal machinery only ===
    # Keep a generic LZC, but we do NOT use it on the hot path.
    msb_flags = []
    for i in range(PROD_W - 1, -1, -1):
        upper_zero = 1 if i == PROD_W - 1 else (prod[i + 1 : PROD_W] == 0)
        msb_flags.append(upper_zero & prod[i])
    lz = 0
    for idx, flag in enumerate(msb_flags):  # idx = leading zeros count
        lz = mux(flag, idx, lz)

    # === Choose mantissa/GRS source ===
    # Use fast pre-normalized mantissa by default. Only the subnormal OUT rounding
    # uses the (different) 'shift_amt' later.
    mant_pre = mant_pre_fast
    guard    = guard_fast
    sticky   = sticky_fast

    # Rounding to nearest-even on the normalized path
    lsb = mant_pre[0]
    round_up = guard & (sticky | lsb)
    mant_round = mant_pre + mux(round_up, 1, 0)         # (1+FW)+1 wide by type rules
    carry = mant_round[FW + 1]
    mant_post = mux(carry, mant_round[1 : FW + 2], mant_round[0 : FW + 1])  # (1+FW)
    frac_norm = mant_post[0:FW]

    # ---------------- Exponent / range checks ----------------
    exp_sum = eA_eff + eB_eff                        # EW+1
    # Fast-path exponent for the common case (matches general formula):
    # e_norm = (eA_eff+eB_eff - BIAS) + hi1 + carry
    e_norm_fast = (exp_sum - BIAS) + mux(hi1, 1, 0) + mux(carry, 1, 0)
    exp_field_norm = e_norm_fast[0:EW]

    # For exceptions/edge-range decisions (under/over) rely on the same 'lhs/limits'
    lhs = (exp_sum + 1) + mux(carry, 1, 0)          # = eA_eff + eB_eff + 1 + carry
    limit_under = BIAS + lz
    limit_over  = (BIAS + MAX_FINITE_E) + lz
    underflow = lhs <= limit_under
    overflow  = lhs >  limit_over

    # ---------------- Subnormal output path (when enabled AND underflow) ----------------
    if subnormals:
        # shift_amt = (BIAS + lz) - (eA_eff + eB_eff + carry)
        shift_amt = (BIAS + lz) - (exp_sum + mux(carry, 1, 0))
        sig_pre = mant_post                       # (1+FW)
        sig_shiftN = sig_pre >> shift_amt         # variable right shift for subnormal packing
        frac_trunc = sig_shiftN[0:FW]

        # Safe guard/sticky for subnormal rounding (avoid negative indices)
        pref_sig = _prefix_or_bits(sig_pre, FW + 1)

        def _bit_at_sig(idx_expr):
            acc = 0
            for k in range(FW + 1):
                acc = mux(idx_expr == k, sig_pre[k], acc)
            return acc

        def _pref_sig_at(idx_expr):
            acc = 0
            for k in range(FW + 1):
                acc = mux(idx_expr == k, pref_sig[k], acc)
            return acc

        has_g = shift_amt > 0
        has_s = shift_amt > 1
        guard_s  = mux(has_g, _bit_at_sig(shift_amt - 1), 0)
        sticky_s = mux(has_s, _pref_sig_at(shift_amt - 2), 0)

        lsb_s = frac_trunc[0]
        round_up_s = guard_s & (sticky_s | lsb_s)

        frac_trunc_zext = cat(frac_trunc, 0)  # FW+1
        frac_sum = frac_trunc_zext + mux(round_up_s, 1, 0)
        carry_s = frac_sum[FW]                # if overflow → becomes min normal
        frac_field_sub = frac_sum[0:FW]
        exp_field_sub  = mux(carry_s, 1, 0)

    # ---------------- Pack result ----------------
    all1_E = (1 << EW) - 1
    qnan_payload = (1 << (FW - 1)) if FW > 0 else 1

    is_nan = is_nan_in
    is_inf = (~is_nan) & (is_inf_in | overflow)

    if subnormals:
        is_sub_out = (~is_nan) & (~is_inf) & underflow
        sub_is_zero = is_sub_out & ((exp_field_sub == 0) & (frac_field_sub == 0))
        is_zero = (~is_nan) & (~is_inf) & (is_zero_in | sub_is_zero)

        exp_field = mux(is_nan | is_inf, all1_E,
                        mux(is_zero, 0,
                            mux(is_sub_out, exp_field_sub, exp_field_norm)))
        frac_field = mux(is_nan, qnan_payload,
                         mux(is_inf | is_zero, 0,
                             mux(is_sub_out, frac_field_sub, frac_norm)))
    else:
        is_zero = (~is_nan) & (~is_inf) & (is_zero_in | underflow)
        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, e_norm_fast[0:EW]))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, frac_norm))

    sign_field = mux(is_nan, 0, sY)
    y <<= cat(frac_field, exp_field, sign_field)
    return m

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

def build_f16_subnormal_ext_vectors():
    return [  # (value, 0.5) → expected
        # odd subnormals: N/2 is x.5 → tie-to-even
        ("0x0001 * 0.5 (tie->even)", 0x0001, 0x3800, 0x0000),  # 0.5 ulp → 0x0000
        ("0x0003 * 0.5 (tie->even)", 0x0003, 0x3800, 0x0002),  # 1.5 ulp → even 2
        ("0x0005 * 0.5 (tie->even)", 0x0005, 0x3800, 0x0002),  # 2.5 ulp → even 2
        ("0x0007 * 0.5 (tie->even)", 0x0007, 0x3800, 0x0004),
        ("0x03FD * 0.5 (tie->even)", 0x03FD, 0x3800, 0x01FE),  # 510.5 → 0x01FE (even)
        ("0x03FF * 0.5 (tie->even)", 0x03FF, 0x3800, 0x0200),  # 511.5 → 0x0200 (even)
        # near-tie neighbors:
        ("0x03FE * 0.5 (below tie)", 0x03FE, 0x3800, 0x01FF),  # 511.0 → 0x01FF
        ("0x0002 * 0.5 (below tie)", 0x0002, 0x3800, 0x0001),  # 1.0 → 0x0001
        ("0x0004 * 0.5 (below tie)", 0x0004, 0x3800, 0x0002),  # 2.0 → 0x0002
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
    run_vectors(f16, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=half_to_float)
