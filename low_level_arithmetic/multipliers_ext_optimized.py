import abc
from collections import defaultdict
from dataclasses import dataclass
import re
from typing import ClassVar, DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple, Type

from aigverse import read_aiger_into_aig, write_aiger
import numpy as np

from low_level_arithmetic.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplier, StageBasedMultiplierIO
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier
from low_level_arithmetic.test_vector_generation import Encoding, MultiplierTestVectors, from_encoding, to_encoding
from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, file_to_lines
from sprouthdl.helpers import get_aig_stats, get_yosys_transistor_count, optimize_aag
from sprouthdl.sprouthdl_aiger import AigerImporter
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from sprouthdl.sprouthdl_module import Module
from testing.test_different_logic import aag_file_to_aag_lines, run_vectors_io, verilog_to_aag_lines_via_pyosys, verilog_to_aag_via_pyosys

class StageBasedMultiplierBasicOptmized(StageBasedExtMultiplier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.unsigned or self.b_encoding == Encoding.twos_complement, "Only unsigned or two's complement encoding is supported"
        assert self.b_encoding == Encoding.unsigned or self.b_encoding == Encoding.twos_complement, "Only unsigned or two's complement encoding is supported"
        y_encoding = Encoding.twos_complement if (self.a_encoding == Encoding.twos_complement or self.b_encoding == Encoding.twos_complement) else Encoding.unsigned

        def get_type(enc: Encoding,) -> Type:
            if enc == Encoding.unsigned:
                return UInt
            elif enc == Encoding.twos_complement:
                return SInt

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=get_type(self.a_encoding)(self.aw), kind="input"),
            b=Signal(name="b", typ=get_type(self.b_encoding)(self.bw), kind="input"),
            y=Signal(name="y", typ=get_type(y_encoding)(self.aw + self.bw), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:

        if self.aw == 4 and self.bw == 4 and self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned :
            verilog_file_name = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_2/analysis/minima_histogram/final_gen_design_files_best_design_aig_count/mydesign_mockturtle_cleaned.v"
            # verilog_file_name = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_2/analysis/minima_histogram/final_gen_design_files_best_design_nb_transistors/mydesign_mockturtle_cleaned.v"
        else:
            raise NotImplementedError("This optimized multiplier is only implemented for 4x4 unsigned multiplication.")

        aag_lines = verilog_to_aag_lines_via_pyosys(verilog_file_name, top="mydesign_comb", embed_symbols=True, no_startoffset=True)
        # aag_lines = optimize_aag(aag_lines)

        # aag_file = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_2/analysis/minima_histogram/final_mockturtle_design_best_design_aig_count/out_aiger_yosys_iter9_proc2_0.aig"
        aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/exp_test_3/data_collection/run_20251013_174847_MO2F"
        aag_file = aag_root + "/final_mockturtle_design/out_aiger_yosys_iter3_proc0_2.aig"
        map_file = aag_root + "/final_mockturtle_design/aiger_map_cleaned.map"
        
        aag_lines2 = aag_file_to_aag_lines(aag_file, map_file=map_file)
 

        m = AigerImporter(aag_lines2).get_sprout_module()
                              
        # c = Component().from_module(m, make_internal=True)
        # instantiate an unsigned multiplier for the magnitudes

        this_class = self

        class LoadedMultiplier(Component):

            def __init__(self):
                @dataclass
                class IO:
                    operand_a_i: Signal
                    operand_b_i: Signal
                    result_o: Signal
 
                self.io: IO = IO(
                    operand_a_i=Signal(name="operand_a_i", typ=UInt(this_class.aw), kind="input"),
                    operand_b_i=Signal(name="operand_b_i", typ=UInt(this_class.bw), kind="input"),
                    result_o=Signal(name="result_o", typ=UInt(this_class.aw + this_class.bw), kind="output"),
                )

        self.mult = LoadedMultiplier()
        self.mult.from_module(m, make_internal=True, group=True)

        # use the specified multiplier
        self.mult.io.operand_a_i <<= self.io.a
        self.mult.io.operand_b_i <<= self.io.b
        self.io.y <<= self.mult.io.result_o


if __name__ == "__main__":  # pragma: no cover - demonstration only

    n_bits = 4
    signed = False

    m = StageBasedMultiplierBasicOptmized(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )
    mod = m.to_module("multiplier_ext_optimized")
    print(mod)

    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"AIG-reported gate count: {aig_gates}")

    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        y_encoding=to_encoding(signed),
    ).generate()

    run_vectors_io(module, vecs)
    

