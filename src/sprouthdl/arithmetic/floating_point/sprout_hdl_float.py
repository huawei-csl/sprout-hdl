# =========================
# Floating-point multiplier
# =========================
# Parameterizable by exponent width (EW) and fraction width (FW).
# - float16:  EW=5, FW=10   (total 16)
# - bfloat16: EW=8, FW=7    (total 16)

from dataclasses import dataclass
from sprouthdl.sprouthdl import *
from sprouthdl.sprouthdl_module import Component, Module
from sprouthdl.sprouthdl_simulator import Simulator

# Uses: Module, UInt, Bool, mux, cat (from your sprout_hdl)


def _or_reduce_bits(vec_expr, hi: int, lo: int):
    """OR-reduce bits vec_expr[lo:hi+1] (inclusive). If hi < lo, returns 0."""
    if hi < lo:
        return 0
    acc = 0
    for i in range(lo, hi + 1):
        acc = acc | vec_expr[i]
    return acc


class FpMul(Component):

    def _extract_fields(self, a: Signal, b: Signal):
        W = self.W
        FW = self.FW
        sA = a[W - 1]  # sign
        sB = b[W - 1]

        eA = a[FW : W - 1]  # exponent EW bits
        eB = b[FW : W - 1]

        fA = a[0:FW]  # fraction FW bits
        fB = b[0:FW]
        return sA, sB, eA, eB, fA, fB

    def _classify_operands(self, eA: Signal, eB: Signal, fA: Signal, fB: Signal, exp_all_ones: int):
        is_eA_zero = eA == 0
        is_eB_zero = eB == 0
        is_fA_zero = fA == 0
        is_fB_zero = fB == 0

        is_zeroA = is_eA_zero & is_fA_zero
        is_zeroB = is_eB_zero & is_fB_zero

        is_eA_all1 = eA == exp_all_ones
        is_eB_all1 = eB == exp_all_ones

        is_nanA = is_eA_all1 & (fA != 0)
        is_nanB = is_eB_all1 & (fB != 0)

        is_infA = is_eA_all1 & (fA == 0)
        is_infB = is_eB_all1 & (fB == 0)

        # NaN if: any NaN, or Inf * 0 (either way)
        is_nan_in = is_nanA | is_nanB | ((is_infA & is_zeroB) | (is_zeroA & is_infB))
        # Inf if: any Inf (and not NaN case)
        is_inf_in = is_infA | is_infB
        # Zero if: any zero (and not NaN or Inf cases)
        is_zero_in = is_zeroA | is_zeroB

        return {
            "is_eA_zero": is_eA_zero,
            "is_eB_zero": is_eB_zero,
            "is_zeroA": is_zeroA,
            "is_zeroB": is_zeroB,
            "is_nanA": is_nanA,
            "is_nanB": is_nanB,
            "is_infA": is_infA,
            "is_infB": is_infB,
            "is_nan_in": is_nan_in,
            "is_inf_in": is_inf_in,
            "is_zero_in": is_zero_in,
        }

    def _effective_mantissas(self, fA: Signal, fB: Signal, is_eA_zero: Signal, is_eB_zero: Signal):
        FW = self.FW
        one_and_frac_A = cat(fA, 1)  # width 1+FW
        one_and_frac_B = cat(fB, 1)

        mask_1pFW = (1 << (1 + FW)) - 1
        mskA = mux(is_eA_zero, 0, mask_1pFW)
        mskB = mux(is_eB_zero, 0, mask_1pFW)

        mA_eff = one_and_frac_A & mskA  # (1+FW) bits
        mB_eff = one_and_frac_B & mskB
        return mA_eff, mB_eff

    def _normalize_and_round(self, prod: Signal):
        FW = self.FW
        PROD_W = 2 + 2 * FW

        msb_high = prod[PROD_W - 1]  # 1 if product ≥ 2.0

        # Select normalized top (1+FW) bits (pre-round)
        top_hi = prod[FW + 1 : PROD_W]  # when msb_high=1 → [FW+1 : 2*FW+1]
        top_lo = prod[FW : PROD_W - 1]  # when msb_high=0 → [FW : 2*FW]
        mant_pre = mux(msb_high, top_hi, top_lo)  # (1+FW) bits

        # Rounding bits (GRS: guard/round/sticky)
        guard_hi = prod[FW]  # when msb_high=1 → bit FW
        guard_lo = prod[FW - 1]  # when msb_high=0 → bit FW-1
        guard = mux(msb_high, guard_hi, guard_lo)

        sticky_hi = _or_reduce_bits(prod, FW - 1, 0)  # when msb_high=1 → [0:FW]
        sticky_lo = _or_reduce_bits(prod, FW - 2, 0)  # when msb_high=0 → [0:FW-1]
        sticky = mux(msb_high, sticky_hi, sticky_lo)

        lsb = mant_pre[0]  # LSB of current mantissa
        round_up = guard & (sticky | lsb)  # round-to-nearest-even

        # Add rounding increment (width grows by 1)
        mant_round = mant_pre + mux(round_up, 1, 0)  # width (1+FW)+1 = FW+2
        carry = mant_round[FW + 1]  # did rounding overflow?
        mant_post = mux(carry, mant_round[1 : FW + 2], mant_round[0:FW + 1])  # (1+FW) bits

        frac_norm = mant_post[0:FW]  # drop hidden 1 → FW bits
        return mant_post, frac_norm, msb_high, carry

    def _compute_exponent(self, eA: Signal, eB: Signal, msb_high: Signal, carry: Signal, BIAS: int, MAX_FINITE_E: int):
        EW = self.EW
        exp_sum = eA + eB  # width EW+1
        inc1 = mux(msb_high, 1, 0)
        inc2 = mux(carry, 1, 0)

        exp_sum_inc = (exp_sum + inc1) + inc2  # EW+2 bits
        bias_const = BIAS
        maxfinite_plus_bias = BIAS + MAX_FINITE_E

        # Underflow if exp_sum_inc <= BIAS (no subnormal output → zero)
        underflow = exp_sum_inc <= bias_const
        # Overflow if exp_sum_inc > BIAS + MAX_FINITE_E
        overflow = exp_sum_inc > maxfinite_plus_bias

        # Normal exponent field (EW bits): exp = exp_sum_inc - BIAS
        exp_unbiased = exp_sum_inc - bias_const  # width EW+2
        exp_norm = exp_unbiased[0:EW]  # truncate to EW
        return exp_norm, underflow, overflow

    def _pack_result(self, sY: Signal, exp_norm: Signal, frac_norm: Signal, is_nan_in: Signal, is_inf_in: Signal, is_zero_in: Signal):
        FW = self.FW
        EW = self.EW
        all1_E = (1 << EW) - 1
        qnan_payload = (1 << (FW - 1)) if FW > 0 else 1  # canonical quiet NaN

        # Priority: NaN > (Inf or overflow) > (Zero or underflow) > Normal
        is_nan = is_nan_in
        is_inf = (~is_nan) & is_inf_in
        is_zero = (~is_nan) & (~is_inf) & is_zero_in

        exp_field = mux(is_nan | is_inf, all1_E, mux(is_zero, 0, exp_norm))
        frac_field = mux(is_nan, qnan_payload, mux(is_inf | is_zero, 0, frac_norm))
        sign_field = mux(is_nan, 0, sY)  # sign is don't-care for NaN; choose 0
        return sign_field, exp_field, frac_field
    
    @dataclass
    class IO:
        a: Signal # input
        b: Signal # input
        y: Signal # output
    
    def __init__(self, EW: int, FW: int) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW

        self.io = self.IO(
            a=Signal(name="a", typ=UInt(self.W), kind="input"),
            b=Signal(name="b", typ=UInt(self.W), kind="input"),
            y=Signal(name="y", typ=UInt(self.W), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:
        EW = self.EW
        FW = self.FW
        W = self.W
        BIAS = (1 << (EW - 1)) - 1  # e.g. 15 for float16
        MAX_E = (1 << EW) - 1  # all-ones exponent
        MAX_FINITE_E = MAX_E - 1  # maximum finite exponent value

        a = self.io.a
        b = self.io.b
        y = self.io.y

        # ------------------
        # Field extraction
        # ------------------
        sA, sB, eA, eB, fA, fB = self._extract_fields(a, b)

        # ------------------
        # Classifiers
        # ------------------
        exp_all_ones = MAX_E
        cls = self._classify_operands(eA, eB, fA, fB, exp_all_ones)
        is_eA_zero = cls["is_eA_zero"]
        is_eB_zero = cls["is_eB_zero"]
        is_zeroA = cls["is_zeroA"]
        is_zeroB = cls["is_zeroB"]
        is_nan_in = cls["is_nan_in"]
        is_inf_in = cls["is_inf_in"]
        is_zero_in = cls["is_zero_in"]

        # ------------------
        # Sign
        # ------------------
        sY = sA ^ sB

        # ------------------
        # Effective mantissas with hidden '1' (mask out when exp==0 → treat as 0)
        # ------------------
        mA_eff, mB_eff = self._effective_mantissas(fA, fB, is_eA_zero, is_eB_zero)

        # ------------------
        # Multiply mantissas (unsigned product)
        # ------------------
        prod = mA_eff * mB_eff  # width 2 + 2*FW
        PROD_W = 2 + 2 * FW

        # Multiply two (1+FW)‑bit unsigned integers:
        # Product width PROD_W = 2 + 2*FW
        # Since each is in [1.0, 2.0) (or 0), product is in [1.0, 4.0) (or 0)
        # Therefore the product’s top two bits determine normalization:
        # If product ≥ 2.0 → MSB=1: we right‑shift by 1 and add +1 to exponent
        # Else product in [1.0, 2.0) → use as is

        # Normalization: product ∈ [1.000..., 3.111...] range in fixed-point.
        # If MSB (bit PROD_W-1) is 1 (≥2.0), shift right by 1 and increment exponent.
        mant_post, frac_norm, msb_high, carry = self._normalize_and_round(prod)

        # ------------------
        # Exponent path (unsigned math)
        # ------------------
        exp_norm, underflow, overflow = self._compute_exponent(
            eA, eB, msb_high, carry, BIAS, MAX_FINITE_E
        )

        # ------------------
        # Special-case selection & packing
        # ------------------
        sign_field, exp_field, frac_field = self._pack_result(
            sY, exp_norm, frac_norm, is_nan_in, is_inf_in | overflow, is_zero_in | underflow
        )

        y <<= cat(frac_field, exp_field, sign_field)


def build_fp_mul(name: str, EW: int, FW: int) -> Module:
    comp = FpMul(EW, FW)
    return comp.to_module(name, with_clock=False, with_reset=False)


def build_f16_mul(name: str = "F16Mul") -> Module:
    # IEEE-754 binary16: exponent=5, fraction=10
    return build_fp_mul(name, EW=5, FW=10)


def build_bf16_mul(name: str = "BF16Mul") -> Module:
    # bfloat16: exponent=8, fraction=7
    return build_fp_mul(name, EW=8, FW=7)

# sanity_fp_mul_tests.py
# Assumes:
#   from sprout_hdl import Simulator
#   from sprout_hdl import build_f16_mul, build_bf16_mul
# Or adapt the imports to your paths (e.g., from sprout_hdl import ...)

def half_to_float(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    bias = 15
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return ((-1.0) ** s) * (2 ** (1 - bias)) * (f / (1 << 10))
    if e == 0x1F:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return ((-1.0) ** s) * (2 ** (e - bias)) * (1.0 + f / (1 << 10))

def bf16_to_float(b):
    s = (b >> 15) & 1
    e = (b >> 7) & 0xFF
    f = b & 0x7F
    bias = 127
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        return ((-1.0) ** s) * (2 ** (1 - bias)) * (f / (1 << 7))
    if e == 0xFF:
        if f == 0:
            return float("-inf") if s else float("inf")
        return float("nan")
    return ((-1.0) ** s) * (2 ** (e - bias)) * (1.0 + f / (1 << 7))

def run_vectors_local(mod, vectors, *, label="", decoder=None):
    sim = Simulator(mod)
    print(f"\n== {label} ==")
    ok = 0
    for name, a_hex, b_hex, exp_hex in vectors:
        sim.set("a", a_hex).set("b", b_hex).eval()
        got = sim.get("y")
        pass_fail = "PASS" if got == exp_hex else "FAIL"
        extra = ""
        if decoder is not None:
            extra = f"  val={decoder(got):.8g} (exp {decoder(exp_hex):.8g})"
        print(f"{pass_fail:4s}  {name:25s}  a=0x{a_hex:04X}  b=0x{b_hex:04X}  -> y=0x{got:04X}  (exp 0x{exp_hex:04X}){extra}")
        ok += (got == exp_hex)
    print(f"Summary: {ok}/{len(vectors)} passed.\n")
    return ok

def build_f16_vectors():
    # float16 (binary16) constants
    #  +0    0x0000   -0    0x8000
    #  +1.0  0x3C00   +2.0  0x4000   +3.0  0x4200   +4.0  0x4400
    #  -1.0  0xBC00   -2.0  0xC000   -4.0  0xC400
    #  +Inf  0x7C00   -Inf  0xFC00   qNaN  0x7E00
    #  max finite (65504) = 0x7BFF
    #  min normal (2^-14) = 0x0400
    return [
        ("1*2 = 2",                0x3C00, 0x4000, 0x4000),
        ("(-2)*2 = -4",            0xC000, 0x4000, 0xC400),
        ("1.5*1.5 = 2.25",         0x3E00, 0x3E00, 0x4080),  # ← fixed
        ("3*0.5 = 1.5",            0x4200, 0x3800, 0x3E00),
        ("0 * 1 = 0",              0x0000, 0x3C00, 0x0000),
        ("(-0) * 2 = (-0)",        0x8000, 0x4000, 0x8000),  # sign of zero = XOR of inputs
        ("Inf * 3 = Inf",          0x7C00, 0x4200, 0x7C00),
        ("(-Inf) * (-2) = +Inf",   0xFC00, 0xC000, 0x7C00),
        ("Inf * 0 = NaN",          0x7C00, 0x0000, 0x7E00),
        ("NaN * 2 = NaN",          0x7E00, 0x4000, 0x7E00),
        ("Overflow: max*2 = Inf",  0x7BFF, 0x4000, 0x7C00),
        ("Underflow: min*0.5 = 0", 0x0400, 0x3800, 0x0000),
        ("(-1)*1 = -1",            0xBC00, 0x3C00, 0xBC00),
        ("4 * 0.5 = 2",            0x4400, 0x3800, 0x4000),
    ]

def build_bf16_vectors():
    # bfloat16 constants
    #  +0    0x0000   -0    0x8000
    #  +1.0  0x3F80   +1.5  0x3FC0   +2.0  0x4000   +3.0  0x4040   +4.0  0x4080
    #  -2.0  0xC000   -4.0  0xC080
    #  +Inf  0x7F80   -Inf  0xFF80   qNaN  0x7FC0
    #  max finite ≈ exponent 254: 0x7F7F
    #  min normal = 2^-126:      0x0080
    return [
        ("1*2 = 2",                0x3F80, 0x4000, 0x4000),
        ("(-2)*2 = -4",            0xC000, 0x4000, 0xC080),
        ("1.5*1.5 = 2.25",         0x3FC0, 0x3FC0, 0x4010),  # ← fixed
        ("3*0.5 = 1.5",            0x4040, 0x3F00, 0x3FC0),  # 0.5 in bf16 is 0x3F00
        ("0 * 1 = 0",              0x0000, 0x3F80, 0x0000),
        ("(-0) * 2 = (-0)",        0x8000, 0x4000, 0x8000),
        ("Inf * 3 = Inf",          0x7F80, 0x4040, 0x7F80),
        ("(-Inf) * (-2) = +Inf",   0xFF80, 0xC000, 0x7F80),
        ("Inf * 0 = NaN",          0x7F80, 0x0000, 0x7FC0),
        ("NaN * 2 = NaN",          0x7FC0, 0x4000, 0x7FC0),
        ("Overflow: max*2 = Inf",  0x7F7F, 0x4000, 0x7F80),
        ("Underflow: min*0.5 = 0", 0x0080, 0x3F00, 0x0000),
        ("(-1)*1 = -1",            0xBF80, 0x3F80, 0xBF80),
        ("4 * 0.5 = 2",            0x4080, 0x3F00, 0x4000),
    ]


if __name__ == "__main__":
    
    
    f16 = build_f16_mul("F16Mul")
    bf16 = build_bf16_mul("BF16Mul")
    
    run_vectors_local(f16, build_f16_vectors(), label="float16 (binary16)", decoder=half_to_float)
    run_vectors_local(bf16, build_bf16_vectors(), label="bfloat16", decoder=bf16_to_float)

    mul16 = build_f16_mul("F16Mul")
    #print(mul16.to_verilog())
    
    sim = Simulator(mul16)
    
    # 1.0 * 2.0 = 2.0
    sim.set("a", 0x3C00).set("b", 0x4000).eval()
    print("y = 0x%04X" % sim.get("y"))  # expect 0x4000
    
    # -2.0 * 2.0 = -4.0
    sim.set("a", 0xC000).set("b", 0x4000).eval()
    print("y = 0x%04X" % sim.get("y"))  # expect 0xC800
    
    # Inf * 0 → NaN (quiet)
    sim.set("a", 0x7C00).set("b", 0x0000).eval()
    print("NaN? y = 0x%04X" % sim.get("y"))  # exponent all ones; non-zero fraction
    
    #mulbf = build_bf16_mul("BF16Mul")
    #print(mulbf.to_verilog())
    
    # bfloat16 known patterns: 1.0=0x3F80, 2.0=0x4000, 4.0=0x4080
    #sim = Simulator(mulbf)
    #sim.set("a", 0x3F80).set("b", 0x4000).eval()
    #print("y = 0x%04X" % sim.get("y"))  # expect 0x4000 (1*2=2)
