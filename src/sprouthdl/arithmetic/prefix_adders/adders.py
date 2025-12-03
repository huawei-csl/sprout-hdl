import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, DefaultDict, Dict, List, Literal, Optional, Tuple, Type

import numpy as np

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierConfig, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, from_encoding, to_encoding
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from sprouthdl.sprouthdl_module import Module

@dataclass(frozen=True)
class AdderConfig(MultiplierConfig):
    a_width: int
    b_width: int
    signed_a: Optional[bool]
    signed_b: Optional[bool]
    optim_type: Literal["area", "speed"]
    full_output_bit: bool = False

    @property
    def out_width(self) -> int:
        return max(self.a_width, self.b_width) + (1 if self.full_output_bit else 0)


class StageBasedAdderBase(Component):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        optim_type: Literal["area", "speed"] = "area",
        fsa_cls: Optional[Type[FinalStageAdderBase]] = None,
        full_output_bit: bool = False,
    ) -> None:
        self.aw = a_w
        self.bw = b_w
        self.optim_type = optim_type  
        self.fsa_cls = fsa_cls
        self.config = AdderConfig(a_w, b_w, signed_a=None, signed_b=None, optim_type=optim_type, full_output_bit=full_output_bit)


class StageBasedPrefixAdder(StageBasedAdderBase):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        optim_type: Literal["area", "speed"] = "area",
        fsa_cls: Optional[Type[FinalStageAdderBase]] = None,
        full_output_bit: bool = True, # True corresponds to output type Encoding.unsigned_overflow
    ) -> None:
        super().__init__(a_w, b_w, optim_type=optim_type, fsa_cls=fsa_cls, full_output_bit=full_output_bit)
        # Additional initialization for prefix adder can go here

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(self.aw), kind="input"),
            b=Signal(name="b", typ=UInt(self.bw), kind="input"),
            y=Signal(name="y", typ=UInt(self.config.out_width), kind="output"),
        )

        self.fsa = self.fsa_cls(self.config) if self.fsa_cls is not None else RippleCarryFinalAdder(self.config)
        self.elaborate()

    def elaborate(self):

        reduced_columns = defaultdict(list)
        for i in range(self.io.y.typ.width):
            if i < self.aw:
                reduced_columns[i].append(self.io.a[i])
            if i < self.bw:
                reduced_columns[i].append(self.io.b[i])
            
        result_bits = self.fsa.resolve(reduced_columns)
        self.io.y <<= Concat(result_bits[:self.config.out_width])
        
        
def smoke_test():
    
    n_bits = 8
    adder = StageBasedPrefixAdder(
        a_w=n_bits,
        b_w=n_bits,
        optim_type="area",
        fsa_cls=RippleCarryFinalAdder,
    )
    
    
    # create module
    print(adder.to_module(f"PrefixAdder{n_bits}").to_verilog())
    
if __name__ == "__main__":
    smoke_test()
