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
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import *
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import bf16_to_float, build_bf16_vectors, build_f16_vectors, build_fp_mul, half_to_float, run_vectors_local
from sprouthdl.sprouthdl_simulator import Simulator


# Uses: Module, UInt, Bool, mux, cat (from your sprout_hdl)


# ---- tiny helpers used inside build_fp_mul ----
def _or_reduce_bits(vec_expr, hi: int, lo: int):
    if hi < lo:
        return 0
    acc = 0
    for i in range(lo, hi + 1):
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

    # Fields
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

    # Effective significands / exponents
    if subnormals:
        hiddenA = mux(is_eA_zero, 0, 1)
        hiddenB = mux(is_eB_zero, 0, 1)
        mA_eff = cat(fA, hiddenA)  # (1+FW)
        mB_eff = cat(fB, hiddenB)
        eA_eff = mux(is_eA_zero, 1, eA)  # so e_eff - BIAS = 1 - BIAS for subnormals
        eB_eff = mux(is_eB_zero, 1, eB)
    else:
        mask = (1 << (1 + FW)) - 1
        mA_eff = (cat(fA, 1)) & mux(is_eA_zero, 0, mask)
        mB_eff = (cat(fB, 1)) & mux(is_eB_zero, 0, mask)
        eA_eff = eA
        eB_eff = eB

    # Product
    prod = mA_eff * mB_eff
    PROD_W = 2 + 2 * FW

    # ---------- Leading-zero count on 'prod' (duck-typed priority encoder) ----------
    # is_msb_i: "all bits above i are 0" & "bit i is 1"
    msb_flags = []
    for i in range(PROD_W - 1, -1, -1):
        if i == PROD_W - 1:
            upper_zero = 1  # True
        else:
            upper_zero = prod[i + 1 : PROD_W] == 0
        msb_flags.append(upper_zero & prod[i])  # Bool
    # lz = number of zeros before first '1' from MSB side
    # Build as a chain of muxes over constants (unique one-hot)
    lz = 0
    for idx, flag in enumerate(msb_flags):
        i = (PROD_W - 1) - idx  # actual bit index tested
        lz_const = (PROD_W - 1) - i  # = idx
        lz = mux(flag, lz_const, lz)  # UInt-like

    # Decide shift direction to align leading 1 to position FW
    # Right shift if lz <= FW+1, else left shift by (lz - (FW+1))
    need_right = lz <= (FW + 1)
    sr = (FW + 1) - lz
    sl = lz - (FW + 1)

    shifted_r = prod >> sr
    shifted_l = prod << sl
    shifted = mux(need_right, shifted_r, shifted_l)

    # Take normalized (1+FW) window (now leading 1 sits at bit FW)
    mant_pre = shifted[0:FW + 1]  # (1+FW) bits

    # Rounding (GRS)
    # Right-shift path: guard = prod[sr-1], sticky = OR(prod[0 .. sr-2])
    # Left-shift path:  guard=0, sticky=0 (left shift introduces zeros below)
    # Prefix ORs for 'prod' LSB..i
    pref = _prefix_or_bits(prod, PROD_W)

    # dynamic select helpers
    def _bit_at(vec, idx_expr):
        acc = 0
        for k in range(PROD_W):
            acc = mux(idx_expr == k, vec[k], acc)
        return acc

    def _pref_at(idx_expr):
        acc = 0
        for k in range(PROD_W):
            acc = mux(idx_expr == k, pref[k], acc)
        return acc

    guard_r = _bit_at(prod, sr - 1)
    sticky_r = _pref_at(sr - 2)
    guard = mux(need_right, guard_r, 0)
    sticky = mux(need_right, sticky_r, 0)

    lsb = mant_pre[0]
    round_up = guard & (sticky | lsb)

    mant_round = mant_pre + mux(round_up, 1, 0)  # width (1+FW)+1
    carry = mant_round[FW + 1]
    mant_post = mux(carry, mant_round[1 : FW + 2], mant_round[0:FW + 1])  # (1+FW)
    frac_norm = mant_post[0:FW]

    # ---------- Exponent / range checks with lz ----------
    exp_sum = eA_eff + eB_eff  # EW+1
    lhs = (exp_sum + 1) + mux(carry, 1, 0)  # = eA_eff + eB_eff + 1 + carry

    limit_under = BIAS + lz  # unsigned
    limit_over = (BIAS + MAX_FINITE_E) + lz

    underflow = lhs <= limit_under
    overflow = lhs > limit_over

    # Encoded exponent for normal case:
    e_norm = lhs - limit_under  # = (eA+eB+1+carry) - (BIAS+lz)
    exp_field_norm = e_norm[0:EW]

    # ---------- Subnormal output path (if enabled and underflow) ----------
    if subnormals:
        # shift_amt = (BIAS + lz) - (eA_eff + eB_eff + carry)
        shift_amt = (BIAS + lz) - (exp_sum + mux(carry, 1, 0))

        sig_pre = mant_post  # (1+FW)
        sig_shiftN = sig_pre >> shift_amt  # variable right shift
        frac_trunc = sig_shiftN[0:FW]  # FW bits

        # guard_s = bit at (shift_amt - 1) of sig_pre; sticky_s = OR(sig_pre[0 .. shift_amt-2])
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

        guard_s = _bit_at_sig(shift_amt - 1)
        sticky_s = _pref_sig_at(shift_amt - 2)

        lsb_s = frac_trunc[0]
        # lsb_s = 0 # reproduces the error
        round_up_s = guard_s & (sticky_s | lsb_s)

        frac_trunc_zext = cat(frac_trunc, 0)  # FW+1
        frac_sum = frac_trunc_zext + mux(round_up_s, 1, 0)
        carry_s = frac_sum[FW]  # if overflow → becomes min normal
        frac_field_sub = frac_sum[0:FW]
        exp_field_sub = mux(carry_s, 1, 0)

    # ---------- Pack with priority: NaN > Inf/overflow > Zero > Normal/Subnormal ----------
    all1_E = (1 << EW) - 1
    qnan_payload = (1 << (FW - 1)) if FW > 0 else 1

    is_nan = is_nan_in
    is_inf = (~is_nan) & (is_inf_in | overflow)

    if subnormals:
        is_sub_out = (~is_nan) & (~is_inf) & underflow
        sub_is_zero = is_sub_out & ((exp_field_sub == 0) & (frac_field_sub == 0))
        is_zero = (~is_nan) & (~is_inf) & (is_zero_in | sub_is_zero)

        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, mux(is_sub_out, exp_field_sub, exp_field_norm)))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, mux(is_sub_out, frac_field_sub, frac_norm)))
    else:
        is_zero = (~is_nan) & (~is_inf) & (is_zero_in | underflow)
        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, exp_field_norm))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, frac_norm))

    sign_field = mux(is_nan, 0, sY)
    y <<= cat(frac_field, exp_field, sign_field)
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

    run_vectors_local(f16, build_f16_vectors(), label="float16 default cases", decoder=half_to_float)
    run_vectors_local(bf16, build_bf16_vectors(), label="bfloat16 default cases", decoder=bf16_to_float)
    run_vectors_local(f16,  build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=half_to_float)
    run_vectors_local(bf16, build_bf16_subnormal_vectors(), label="bfloat16 subnormal cases", decoder=bf16_to_float)
    run_vectors_local(f16, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=half_to_float)