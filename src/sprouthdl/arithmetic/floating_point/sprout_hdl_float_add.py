"""Floating-point adder (IEEE-like) for sprout_hdl.

A lightweight behavioural adder mirroring the interface of the multiplier
helpers. It supports configurable exponent/fraction widths and basic handling
of NaN/Inf/zero plus subnormal inputs by treating their implicit leading bit as
0 and using an exponent of 1 for alignment.
"""

from dataclasses import dataclass

from sprouthdl.sprouthdl_module import Component, Module
from sprouthdl.sprouthdl import Signal, UInt, cat, mux, Const


@dataclass
class FpAddIO:
    a: Signal
    b: Signal
    y: Signal

class FpAdd(Component):

    def __init__(self, EW: int, FW: int) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.BIAS = (1 << (EW - 1)) - 1
        self.MAX_E = (1 << EW) - 1

        self.io = FpAddIO(
            a=Signal(name="a", typ=UInt(self.W), kind="input"),
            b=Signal(name="b", typ=UInt(self.W), kind="input"),
            y=Signal(name="y", typ=UInt(self.W), kind="output"),
        )

        self.elaborate()

    # Helpers -------------------------------------------------------------
    def _shift_right(self, mant, shift):
        return mant >> shift

    def _leading_one_pos(self, mant, width):
        pos = 0
        for i in range(width - 1, -1, -1):
            pos = mux(mant[i], i, pos)
        return pos

    def _extract_fields(self, operand):
        sign = operand[self.W - 1]
        exp = operand[self.FW : self.W - 1]
        frac = operand[0 : self.FW]
        return sign, exp, frac

    def _classify_operand(self, exp, frac):
        is_e_zero = exp == 0
        is_f_zero = frac == 0
        is_zero = is_e_zero & is_f_zero
        is_e_all1 = exp == self.MAX_E
        is_nan = is_e_all1 & (frac != 0)
        is_inf = is_e_all1 & (frac == 0)
        return is_zero, is_nan, is_inf, is_e_zero, is_f_zero

    def _effective_fields(self, exp, frac, is_exp_zero):
        hidden = mux(is_exp_zero, 0, 1)
        mant = cat(hidden, frac)
        eff_exp = mux(is_exp_zero, 1, exp)
        return mant, eff_exp

    def _align_operands(self, mA, mB, eA_eff, eB_eff, sA, sB):
        eA_gt = eA_eff > eB_eff
        exp_delta = mux(eA_gt, eA_eff - eB_eff, eB_eff - eA_eff)

        m_big = mux(eA_gt, mA, mB)
        m_small = mux(eA_gt, mB, mA)
        s_big = mux(eA_gt, sA, sB)
        s_small = mux(eA_gt, sB, sA)
        e_big = mux(eA_gt, eA_eff, eB_eff)

        m_big_ext = cat(Const(0, UInt(2)), m_big)
        m_small_ext = cat(Const(0, UInt(2)), m_small)
        m_small_shift = self._shift_right(m_small_ext, exp_delta)

        return e_big, m_big_ext, m_small_shift, s_big, s_small

    def _combine_mantissas(self, m_big_ext, m_small_shift, s_big, s_small):
        same_sign = s_big == s_small
        mant_add = m_big_ext + m_small_shift
        mant_sub = m_big_ext - m_small_shift
        mant_mag = mux(same_sign, mant_add, mant_sub)
        sign_out = mux(same_sign, s_big, mux(mant_sub[self.FW + 2], s_small, s_big))
        return mant_mag, sign_out

    def _normalize(self, mant_mag, e_big):
        overflow = mant_mag[self.FW + 2]
        mant_post_over = mant_mag >> 1

        lead_pos = self._leading_one_pos(mant_mag, self.FW + 3)
        shift_norm = (self.FW + 1) - lead_pos
        mant_norm = mux(overflow, mant_post_over, mant_mag << shift_norm)
        exp_pre = e_big + mux(overflow, 1, 0) - mux(overflow, 0, shift_norm)
        return mant_norm, exp_pre

    def _select_special_result(
        self, sign_out, exp_out, frac_out, sA, sB, is_infA, is_infB, is_nanA, is_nanB
    ):
        nan_in = is_nanA | is_nanB
        inf_in = is_infA | is_infB

        use_nan = nan_in | (inf_in & (sA != sB))
        use_inf = inf_in & ~(sA != sB)
        sign_special = mux(is_infA & ~is_infB, sA, mux(is_infB & ~is_infA, sB, sign_out))

        exp_field = mux(use_nan, self.MAX_E, mux(use_inf, self.MAX_E, exp_out))
        nan_payload = (
            Const(1 << (self.FW - 1 if self.FW > 0 else 0), UInt(self.FW))
            if self.FW > 0
            else Const(1, UInt(1))
        )
        frac_field = mux(use_nan, nan_payload, mux(use_inf, 0, frac_out))
        sign_field = mux(use_nan | use_inf, sign_special, sign_out)

        return sign_field, exp_field, frac_field

    def elaborate(self) -> None:
        a = self.io.a
        b = self.io.b
        y = self.io.y

        sA, eA, fA = self._extract_fields(a)
        sB, eB, fB = self._extract_fields(b)

        (
            _,
            is_nanA,
            is_infA,
            is_eA_zero,
            _,
        ) = self._classify_operand(eA, fA)
        (
            _,
            is_nanB,
            is_infB,
            is_eB_zero,
            _,
        ) = self._classify_operand(eB, fB)

        mA, eA_eff = self._effective_fields(eA, fA, is_eA_zero)
        mB, eB_eff = self._effective_fields(eB, fB, is_eB_zero)

        e_big, m_big_ext, m_small_shift, s_big, s_small = self._align_operands(
            mA, mB, eA_eff, eB_eff, sA, sB
        )
        mant_mag, sign_out = self._combine_mantissas(m_big_ext, m_small_shift, s_big, s_small)

        mant_norm, exp_pre = self._normalize(mant_mag, e_big)

        frac_final = mant_norm[2 : self.FW + 2]
        exp_final = exp_pre

        overflow_exp = exp_final >= self.MAX_E
        underflow_exp = exp_final <= 0

        is_zero_res = underflow_exp | (mant_mag == 0)
        exp_out = mux(overflow_exp, self.MAX_E, mux(is_zero_res, 0, exp_final))
        frac_out = mux(is_zero_res, 0, frac_final)

        sign_field, exp_field, frac_field = self._select_special_result(
            sign_out, exp_out, frac_out, sA, sB, is_infA, is_infB, is_nanA, is_nanB
        )

        y <<= cat(frac_field, exp_field, sign_field)


def build_fp_add(name: str, EW: int, FW: int) -> Module:
    comp = FpAdd(EW, FW)
    return comp.to_module(name, with_clock=False, with_reset=False)


def build_f16_add(name: str = "F16Add") -> Module:
    return build_fp_add(name, EW=5, FW=10)


def build_bf16_add(name: str = "BF16Add") -> Module:
    return build_fp_add(name, EW=8, FW=7)
