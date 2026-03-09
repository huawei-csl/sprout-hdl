from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from sprouthdl.sprouthdl import Expr


@dataclass
class MultiplierConfig:
    """Configuration for choosing between Sprout operator and explicit multiplier."""

    use_operator: bool = False
    multiplier_opt: MultiplierOption | None = None
    encodings: TwoInputAritEncodings | None = None
    ppg_opt: PPGOption | None = None
    ppa_opt: PPAOption | None = None
    fsa_opt: FSAOption | None = None
    optim_type: Literal["area", "speed"] = "area"


@dataclass
class AdderConfig:
    """Configuration for choosing between Sprout operator and explicit adder."""

    use_operator: bool = False
    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption | None = None
    full_output_bit: bool = True


def build_multiplier(a: Expr, b: Expr, mult_cfg: MultiplierConfig) -> Expr:
    if mult_cfg.use_operator:
        return a * b

    assert mult_cfg.multiplier_opt is not None, "multiplier_opt must be provided for explicit multipliers"
    assert mult_cfg.encodings is not None, "encodings must be provided for explicit multipliers"
    assert mult_cfg.ppg_opt is not None and mult_cfg.ppa_opt is not None and mult_cfg.fsa_opt is not None

    multiplier = mult_cfg.multiplier_opt.value(
        a_w=a.typ.width,
        b_w=b.typ.width,
        a_encoding=mult_cfg.encodings.a,
        b_encoding=mult_cfg.encodings.b,
        ppg_cls=mult_cfg.ppg_opt.value,
        ppa_cls=mult_cfg.ppa_opt.value,
        fsa_cls=mult_cfg.fsa_opt.value,
        optim_type=mult_cfg.optim_type,
    ).make_internal()
    multiplier.io.a <<= a
    multiplier.io.b <<= b
    return multiplier.io.y


def build_adder(a: Expr, b: Expr, adder_cfg: AdderConfig) -> Expr:
    if adder_cfg.use_operator:
        return a + b

    assert adder_cfg.fsa_opt is not None, "fsa_opt must be provided for explicit adders"
    signed = is_signed(adder_cfg.encoding)

    adder = StageBasedPrefixAdder(
        a_w=a.typ.width,
        b_w=b.typ.width,
        signed_a=signed,
        signed_b=signed,
        optim_type=adder_cfg.optim_type,
        fsa_cls=adder_cfg.fsa_opt.value,
        full_output_bit=adder_cfg.full_output_bit,
    ).make_internal()
    adder.io.a <<= a
    adder.io.b <<= b
    return adder.io.y


def adder_tree(values: Sequence[Expr], adder_cfg: AdderConfig) -> Expr:
    if len(values) == 0:
        raise ValueError("Adder tree requires at least one value")
    if len(values) == 1:
        return values[0]

    mid = len(values) // 2
    left = adder_tree(values[:mid], adder_cfg)
    right = adder_tree(values[mid:], adder_cfg)
    return build_adder(left, right, adder_cfg)
