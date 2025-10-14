import abc
from collections import defaultdict
from dataclasses import dataclass
import re
import tempfile
from typing import ClassVar, DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple, Type

from aigverse import read_aiger_into_aig, write_aiger
import numpy as np

from low_level_arithmetic.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplier, StageBasedMultiplierIO

from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier
from low_level_arithmetic.test_vector_generation import Encoding, MultiplierTestVectors, from_encoding, to_encoding
from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, file_to_lines
from sprouthdl.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count, optimize_aag
from sprouthdl.sprouthdl_aiger import AigerImporter
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from sprouthdl.sprouthdl_module import Module

from testing.aag_conv.aig_to_aag import aig_file_to_aag_lines
from testing.test_different_logic import aig_file_to_aag_lines_via_yosys, run_vectors_io, verilog_to_aag_lines_via_yosys, verilog_to_aag_via_yosys


class OptimizedMultiplierBasic(StageBasedExtMultiplier):

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

    def get_aag_lines(self) -> List[str]:

        # verilog_file_name = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_4bit_star_1/analysis/minima_histogram/final_gen_design_files_best_design_aig_count/mydesign_mockturtle_cleaned.v"

        if self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned and self.aw == self.bw and self.aw == 4:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_4bit_star_1"
        elif self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned and self.aw == self.bw and self.aw == 3:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_3bit_star_1"
        elif self.a_encoding == Encoding.twos_complement and self.b_encoding == Encoding.twos_complement and self.aw == self.bw and self.aw == 4:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_4bit_star_1"
        elif self.a_encoding == Encoding.twos_complement and self.b_encoding == Encoding.twos_complement and self.aw == self.bw and self.aw == 3:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_3bit_star_1"
        else:
            raise NotImplementedError("No precomputed AIG for this configuration")

        aig_file = aag_root + "/analysis/minima_histogram/final_mockturtle_design_best_design_aig_count/out_aiger.aig"
        map_file = aag_root + "/analysis/minima_histogram/final_mockturtle_design_best_design_aig_count/aiger_map_cleaned.map"

        aag_lines = aig_file_to_aag_lines(aig_file, map_file=map_file)

        # other options:
        # aag_lines = verilog_to_aag_lines_via_yosys(verilog_file_name, top="mydesign_comb", embed_symbols=True, no_startoffset=True) # works but higher aig count
        # aag_lines = aig_file_to_aag_lines_via_yosys(aag_file, map_file=map_file) # works but does not preserve exactness -> higher aig count

        return aag_lines

    def elaborate(self) -> None:

        m = AigerImporter(self.get_aag_lines()).get_sprout_module()

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

        # todo: test from_verlog and from_aig_file

        # use the specified multiplier
        self.mult.io.operand_a_i <<= self.io.a
        self.mult.io.operand_b_i <<= self.io.b
        self.io.y <<= self.mult.io.result_o
        
        
class OptmizedSignMagnitudeMultiplier(StageBasedExtMultiplier):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.a_encoding == Encoding.sign_magnitude and self.b_encoding == Encoding.sign_magnitude, "Only sign-magnitude encoding is supported"
        assert self.ppg_cls is None, "PPG must be None"
        assert self.ppa_cls is None, "PPA must be None"
        assert self.fsa_cls is None, "FSA must be None"

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(self.aw), kind="input"),
            b=Signal(name="b", typ=UInt(self.bw), kind="input"),
            y=Signal(name="y", typ=UInt(self.aw + self.bw - 1), kind="output"),
        )

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplier = OptimizedMultiplierBasic(
            a_encoding=Encoding.unsigned,
            b_encoding=Encoding.unsigned,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
        ).make_internal()
        
        self.mult = mult

        W = self.aw  # assume square for now

        sa = self.io.a[W - 1]
        sb = self.io.b[W - 1]
        mag_a = self.io.a[0 : W - 1]  # make magnitude unsigned
        mag_b = self.io.b[0 : W - 1]  # make magnitude unsigned

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

        self.io.y <<= Concat([mag_y[0 : 2 * W - 2], sy])  # sign + magnitude (drop overflow bit)


if __name__ == "__main__":  # pragma: no cover - demonstration only

    n_bits = 4
    signed = True

    m = OptimizedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )
    mod = m.to_module("multiplier_ext_optimized")
    print(mod)

    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
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
    
    # sign magnitude version

    n_bits = 4
    
    from low_level_arithmetic.multiplier_stage_options_demo_lib import MultiplierOption, encoding_for_multiplier

    encodings = encoding_for_multiplier(MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER.value)[0]

    m = OptmizedSignMagnitudeMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
    )
    mod = m.to_module("multiplier_ext_optimized_sign_magnitude")
    print(mod)
    module = mod
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
    yosys_metrics = get_yosys_metrics(module)
    aig_gates = get_aig_stats(module)
    print(f"Yosys-reported transistor count: {transistor_count}")
    print(f"Yosys-reported metrics: {yosys_metrics}")
    print(f"AIG-reported gate count: {aig_gates}")
    
    vecs = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        y_w=2 * n_bits - 1,
        num_vectors=16,
        tb_sigma=None,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        y_encoding=encodings.y,
    ).generate()

    run_vectors_io(module, vecs)

