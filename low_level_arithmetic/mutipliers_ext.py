import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, DefaultDict, Dict, List, Literal, Optional, Tuple, Type

import numpy as np

from low_level_arithmetic.multiplier_stage_core import Component, CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplier, StageBasedMultiplierIO
from low_level_arithmetic.test_vector_generation import Format
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from sprouthdl.sprouthdl_module import Module


class StageBasedExtMultiplier(Component):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        format_a: Format = Format.unsigned,
        format_b: Format = Format.unsigned,
        optim_type: Literal["area", "speed"] = "area",
        ppg_cls: Type[PartialProductGeneratorBase],
        ppa_cls: Type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: Type[FinalStageAdderBase] = RippleCarryFinalAdder,
    ) -> None:

        self.format_a = format_a
        self.format_b = format_b
        self.aw = a_w
        self.bw = b_w
        self.optim_type = optim_type  
        self.ppg_cls = ppg_cls
        self.ppa_cls = ppa_cls
        self.fsa_cls = fsa_cls


class StageBasedSignMagnitudeMultiplier(StageBasedExtMultiplier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.format_a == Format.sign_magnitude and self.format_b == Format.sign_magnitude, "Only sign-magnitude format is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(self.aw), kind="input"),
            b=Signal(name="b", typ=UInt(self.bw), kind="input"),
            y=Signal(name="y", typ=UInt(self.aw + self.bw - 1), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplier = StageBasedMultiplier(
            signed_a=False,
            signed_b=False,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
        ).make_internal()
        
        self.mult = mult

        W = self.aw  # assume square for now

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[W - 2 : 0]  # make magnitude unsigned
        mag_b = self.io.b[W - 2 : 0]  # make magnitude unsigned

        # mag_y = mag_a * mag_b
        # use the specified multiplier
        mult.io.a <<= mag_a
        mult.io.b <<= mag_b
        mag_y = mult.io.y

        # sign
        sy = sa ^ sb

        # make sure sign is positive if value is zero
        is_zero = mux(mag_y == 0, Const(True, Bool()), Const(False, Bool()))
        sy = mux(is_zero, Const(False, Bool()), sy)

        self.io.y <<= Concat([sy, mag_y[2*W-2-1:0]])  # sign + magnitude (drop overflow bit)


class StageBasedSignMagnitudeExtMultiplier(StageBasedExtMultiplier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.format_a == Format.sign_magnitude_ext and self.format_b == Format.sign_magnitude_ext, "Only sign-magnitude extended format is supported"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(self.aw), kind="input"),
            b=Signal(name="b", typ=UInt(self.bw), kind="input"),
            y=Signal(name="y", typ=UInt(self.aw + self.bw), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplier = StageBasedMultiplier(
            signed_a=False,
            signed_b=False,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            optim_type=self.optim_type,
        ).make_internal()

        self.mult = mult

        W = self.aw  # assume square for now

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[W - 2 : 0]  # make magnitude unsigned
        mag_b = self.io.b[W - 2 : 0]  # make magnitude unsigned

        # assign sc_a = operand_a_sign & (operand_a_mag == {(N_IN-1){1'b0} }); // special case when input is -0 (-8), 1000
        sign_a = self.io.a[W - 1]
        sign_b = self.io.b[W - 1]
        sc_a = sign_a & (mag_a == Const(0, UInt(W - 1)))
        sc_b = sign_b & (mag_b == Const(0, UInt(W - 1)))

        #  assign operand_a_mag_mod = sc_a
        #        ? {1'b1, operand_a_mag[N_IN-3:0]}  // 100 -> 4 ,  operand_a_mag[N_IN-3:0] is 00 in this case
        #          : operand_a_mag;

        mag_a_mod = mux(sc_a, Concat([Const(1, Bool()), mag_a[W - 3 : 0]]), mag_a)
        mag_b_mod = mux(sc_b, Concat([Const(1, Bool()), mag_b[W - 3 : 0]]), mag_b)

        # mag_y = mag_a * mag_b
        # use the specified multiplier
        mult.io.a <<= mag_a_mod
        mult.io.b <<= mag_b_mod
        mag_y = mult.io.y

        sel_crit = Concat([sc_a, sc_b])

        # assign mag_y_mod, shift one up
        mag_y_mod = mux_if(
            if_cond=((sel_crit == Const(0b10, UInt(2))) | (sel_crit == Const(0b01, UInt(2)))),
            then_expr=Concat(
                [
                    mag_y[2 * W - 3 : 0],
                    Const(0, Bool()),
                ]
            ),
            else_expr=mux_if(
                if_cond=(sel_crit == Const(0b11, UInt(2))),
                then_expr=Const(0, UInt(2 * W - 2)),
                else_expr=mag_y,
            ),
        )

        h = mux_if(
            if_cond=(sel_crit==Const(0b11, UInt(2))),
            then_expr=Const(1, Bool()),
            else_expr=Const(0, Bool()),
        )

        # sign
        sy = sa ^ sb

        # make sure sign is positive if value is zero
        is_zero = mux(mag_y == 0, Const(True, Bool()), Const(False, Bool()))
        sy = mux(is_zero, Const(False, Bool()), sy)

        self.io.y <<= Concat([sy, h, mag_y_mod[2 * W - 3 : 0]])  # sign + magnitude (drop overflow bit)
