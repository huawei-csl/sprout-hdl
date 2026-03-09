"""Floating-point adder (IEEE-like) for sprout_hdl.

A lightweight behavioural adder mirroring the interface of the multiplier
helpers. It supports configurable exponent/fraction widths and basic handling
of NaN/Inf/zero plus subnormal inputs by treating their implicit leading bit as
0 and using an exponent of 1 for alignment.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from sprouthdl.sprouthdl_module import Component, Module
from sprouthdl.sprouthdl import Expr, Signal, UInt, cat, mux, Const
from sprouthdl.arithmetic.int_arithmetic_config import (
    AdderConfig,
    build_adder,
)


@dataclass
class FpAddIO:
    a: Signal
    b: Signal
    y: Signal

class FpAdd(Component):

    def __init__(self, EW: int, FW: int, adder_cfg: Optional[AdderConfig] = None) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.BIAS = (1 << (EW - 1)) - 1
        self.MAX_E = (1 << EW) - 1
        self.adder_cfg = adder_cfg

        self.io: FpAddIO = FpAddIO(
            a=Signal(name="a", typ=UInt(self.W), kind="input"),
            b=Signal(name="b", typ=UInt(self.W), kind="input"),
            y=Signal(name="y", typ=UInt(self.W), kind="output"),
        )

        self.elaborate()

    # Helpers -------------------------------------------------------------
    def _shift_right(self, mant: Expr, shift: Expr) -> Expr:
        return mant >> shift

    def _leading_one_pos(self, mant: Expr, width: int) -> Expr:
        """
        Return index of the most-significant set bit in mant[0:width).
        If mant is zero, returns 0 (caller should gate with an explicit zero check).
        """
        pos_width = max(1, (width - 1).bit_length())
        pos = Const(0, UInt(pos_width))
        found = Const(0, UInt(1))
        for i in range(width - 1, -1, -1):
            idx = Const(i, UInt(pos_width))
            pos = mux(found, pos, mux(mant[i], idx, pos))
            found = found | mant[i]
        return pos

    def _extract_fields(self, operand: Expr) -> Tuple[Expr, Expr, Expr]:
        sign = operand[self.W - 1]
        exp = operand[self.FW : self.W - 1]
        frac = operand[0 : self.FW]
        return sign, exp, frac

    def _classify_operand(self, exp: Expr, frac: Expr) -> Tuple[Expr, Expr, Expr, Expr, Expr]:
        is_e_zero = exp == 0
        is_f_zero = frac == 0
        is_zero = is_e_zero & is_f_zero
        is_e_all1 = exp == self.MAX_E
        is_nan = is_e_all1 & (frac != 0)
        is_inf = is_e_all1 & (frac == 0)
        return is_zero, is_nan, is_inf, is_e_zero, is_f_zero

    def _effective_fields(self, exp: Expr, frac: Expr, is_exp_zero: Expr) -> Tuple[Expr, Expr]:
        hidden = mux(is_exp_zero, 0, 1)
        # cat() expects LSB-first ordering, so place the hidden/implicit bit last
        # to make it the MSB of the mantissa.
        mant = cat(frac, hidden)
        eff_exp = mux(is_exp_zero, 1, exp)
        return mant, eff_exp

    def _align_operands(self, mA: Expr, mB: Expr, eA_eff: Expr, eB_eff: Expr, sA: Expr, sB: Expr) -> Tuple[Expr, Expr, Expr, Expr, Expr]:
        eA_gt = eA_eff > eB_eff
        e_eq = eA_eff == eB_eff
        mA_ge = mA >= mB
        a_is_bigger = eA_gt | (e_eq & mA_ge)
        exp_delta = mux(a_is_bigger, eA_eff - eB_eff, eB_eff - eA_eff)

        m_big = mux(a_is_bigger, mA, mB)
        m_small = mux(a_is_bigger, mB, mA)
        s_big = mux(a_is_bigger, sA, sB)
        s_small = mux(a_is_bigger, sB, sA)
        e_big = mux(a_is_bigger, eA_eff, eB_eff)

        m_big_ext = cat(Const(0, UInt(2)), m_big)
        m_small_ext = cat(Const(0, UInt(2)), m_small)
        m_small_shift = self._shift_right(m_small_ext, exp_delta)

        return e_big, m_big_ext, m_small_shift, s_big, s_small

    def _combine_mantissas(self, m_big_ext: Expr, m_small_shift: Expr, s_big: Expr, s_small: Expr) -> Tuple[Expr, Expr]:
        same_sign = s_big == s_small
        mant_add = build_adder(m_big_ext, m_small_shift, self.adder_cfg) if self.adder_cfg is not None else m_big_ext + m_small_shift
        mant_sub = m_big_ext - m_small_shift
        mant_mag = mux(same_sign, mant_add, mant_sub)
        zero_mag = mant_mag == 0
        sign_out = mux(zero_mag, s_big & s_small, s_big)
        return mant_mag, sign_out

    def _normalize(self, mant_mag: Expr, e_big: Expr) -> Tuple[Expr, Expr, Expr, Expr]:
        # mant_mag width is FW+4: [FW+3] is possible carry-out, [FW+2] is the expected hidden bit.
        overflow = mant_mag[self.FW + 3]
        mant_post_over = mant_mag >> 1  # normalize when carry-out is set

        # Otherwise, shift left so that the leading 1 sits at bit (FW+2)
        lead_pos = self._leading_one_pos(mant_mag, mant_mag.typ.width)
        shift_norm = (self.FW + 2) - lead_pos
        mant_norm = mux(overflow, mant_post_over, mant_mag << shift_norm)
        exp_pre = e_big + mux(overflow, 1, 0) - mux(overflow, 0, shift_norm)
        return mant_norm, exp_pre, overflow, shift_norm

    def _select_special_result(
        self,
        sign_out: Expr,
        exp_out: Expr,
        frac_out: Expr,
        sA: Expr,
        sB: Expr,
        is_infA: Expr,
        is_infB: Expr,
        is_nanA: Expr,
        is_nanB: Expr,
    ) -> Tuple[Expr, Expr, Expr]:
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

        mant_norm, exp_pre, overflow_flag, shift_norm = self._normalize(mant_mag, e_big)

        frac_final = mant_norm[2 : self.FW + 2]
        exp_final = exp_pre

        overflow_exp = overflow_flag & (exp_final >= self.MAX_E)
        underflow_exp = (~overflow_flag) & (e_big <= shift_norm)

        is_zero_res = underflow_exp | (mant_mag == 0)
        exp_out = mux(is_zero_res, 0, mux(overflow_exp, self.MAX_E, exp_final))
        frac_out = mux(is_zero_res, 0, mux(overflow_exp, 0, frac_final))

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
