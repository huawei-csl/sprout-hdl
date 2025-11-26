import abc
from collections import defaultdict
from dataclasses import dataclass
import re
import tempfile
from typing import Callable, ClassVar, DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple, Type

from aigverse import read_aiger_into_aig, write_aiger
import numpy as np

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierConfig, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO

from sprouthdl.arithmetic.int_multipliers.multipliers.mutipliers_ext import StageBasedExtMultiplier
from sprouthdl.arithmetic.int_multipliers.stages.ppa_fsa_util import OutputConfig, compressor_sum
from sprouthdl.arithmetic.int_multipliers.stages.ppa_stages import CarrySaveAccumulator
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors, from_encoding, to_encoding
from sprouthdl.aig.aig_aigerverse import _get_aag_sym, file_to_lines
from sprouthdl.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count, optimize_aag, run_vectors
from sprouthdl.sprouthdl_aiger import AigerImporter
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from sprouthdl.sprouthdl_module import Module

from sprouthdl.aig.aig_to_aag import aig_file_to_aag_lines
from sprouthdl.aig.aig_yosys import verilog_to_aag_lines_via_yosys, verilog_to_aag_via_yosys
from sprouthdl.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

# ----- precomputed optimized multipliers stored as AIGs: ffile locations need to be adopted -----

def get_aag_lines_default(aw: int, bw: int, a_encoding: Encoding, b_encoding: Encoding) -> List[str]:

    # verilog_file_name = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_4bit_star_1/analysis/minima_histogram/final_gen_design_files_best_design_aig_count/mydesign_mockturtle_cleaned.v"

    def get_aig_files() -> Tuple[str, str]:
        if a_encoding == Encoding.unsigned and b_encoding == Encoding.unsigned and aw == bw and aw == 4:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_4bit_star_1"
        elif a_encoding == Encoding.unsigned and b_encoding == Encoding.unsigned and aw == bw and aw == 3:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_3bit_star_1"
        elif a_encoding == Encoding.twos_complement and b_encoding == Encoding.twos_complement and aw == bw and aw == 4:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_4bit_star_1"
        elif a_encoding == Encoding.twos_complement and b_encoding == Encoding.twos_complement and aw == bw and aw == 3:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_3bit_star_1"
        elif a_encoding == Encoding.unsigned and b_encoding == Encoding.unsigned and aw == bw and aw == 8:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/unsigned_optim_8bit_star_1"
        elif a_encoding == Encoding.twos_complement and b_encoding == Encoding.twos_complement and aw == bw and aw == 8:
            aag_root = "/scratch/farnold/eda_package/flow_sim2_merged/output/db/signed_optim_8bit_star_1"
        else:
            raise NotImplementedError("No precomputed AIG for this configuration")

        aig_file = aag_root + "/analysis/minima_histogram/final_mockturtle_design_best_design_aig_count/out_aiger.aig"
        map_file = aag_root + "/analysis/minima_histogram/final_mockturtle_design_best_design_aig_count/aiger_map_cleaned.map"
        return aig_file, map_file

    aig_file, map_file = get_aig_files()
    aag_lines = aig_file_to_aag_lines(aig_file, map_file=map_file)

    # other options
    # aag_lines = verilog_to_aag_lines_via_yosys(verilog_file_name, top="mydesign_comb", embed_symbols=True, no_startoffset=True) # works but higher aig count

    return aag_lines


def get_optimized_aag_lines_strong(aw: int, bw: int, a_encoding: Encoding, b_encoding: Encoding) -> List[str]:
    if not(a_encoding == Encoding.unsigned and b_encoding == Encoding.unsigned and aw == bw and aw == 4):
        raise NotImplementedError("Only optimized 4-bit unsigned multiplier is supported")
    aag_root = "data/NEW_multiplier-tc_unsigned-4b-aig100mt75ys-trans344/multiplier-tc_unsigned-4b-aig100mt75ys-trans44/"
    aig_file = aag_root + "out_aiger_yosys_iter1a6c22e9d2eb8ca7503a29263f7a1c56d31ae6b15d71a6e1b3d66d256eb4f0274_1_round_17.aig"
    map_file = aag_root + "out_aiger_map_mod.aig"
    aag_lines = aig_file_to_aag_lines(aig_file, map_file=map_file)
    return aag_lines


# ----- optimized multiplier classes -----


class OptimizedMultiplier(StageBasedExtMultiplier):

    def __init__(self, *args, f_aag_lines: Optional[Callable[[int, int, Encoding, Encoding], list[str]]] = None, **kwargs) -> None:
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

        self.f_aag_lines = f_aag_lines if f_aag_lines is not None else get_aag_lines_default

        self.elaborate()

    def elaborate(self) -> None:

        m = AigerImporter(self.f_aag_lines(self.aw, self.bw, self.a_encoding, self.b_encoding)).get_sprout_module()

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


class OptimizedSignMagnitudeMultiplier(StageBasedExtMultiplier):

    def __init__(self, *args, f_aag_lines: Optional[List[str]] = None, **kwargs) -> None:
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
        
        self.f_aag_lines = f_aag_lines

        self.elaborate()

    def elaborate(self) -> None:

        # instantiate an unsigned multiplier for the magnitudes
        mult: StageBasedMultiplierBasic = OptimizedMultiplier(
            a_encoding=Encoding.unsigned,
            b_encoding=Encoding.unsigned,
            a_w=self.aw - 1,
            b_w=self.bw - 1,
            ppg_cls=self.ppg_cls,
            ppa_cls=self.ppa_cls,
            fsa_cls=self.fsa_cls,
            f_aag_lines=self.f_aag_lines,
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


class OptimizedMultiplierFrom4BitBlocks(StageBasedExtMultiplier):

    def __init__(self, *args, f_aag_lines: Optional[List[str]] = None, use_compressor_tree=True, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Compose four 8x8 optimized unsigned multipliers into a 16x16->32 multiplier
        assert self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned, "Only unsigned encoding is supported"
        # assert aw and bw are >=4 and powers of two
        assert self.aw >= 4 and self.bw >= 4 and ((self.aw & (self.aw - 1)) == 0) and ((self.bw & (self.bw - 1)) == 0), "This composite multiplier expects input widths >=8 and powers of two"

        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(self.aw), kind="input"),
            b=Signal(name="b", typ=UInt(self.bw), kind="input"),
            y=Signal(name="y", typ=UInt(self.aw + self.bw), kind="output"),
        )

        # check class attribut f_aag_lines exists
        if hasattr(self, 'f_aag_lines'):
            if f_aag_lines is not None:
                raise ValueError("f_aag_lines class attribute exists, do not pass f_aag_lines per argument")                
        else:    
            self.f_aag_lines = f_aag_lines

        self.use_compressor_tree = use_compressor_tree # results are better with compressor tree

        self.elaborate()

    def elaborate(self) -> None:

        if self.aw == 4 and self.bw == 4:
            multiplier = OptimizedMultiplier(
                a_encoding=Encoding.unsigned,
                b_encoding=Encoding.unsigned,
                a_w=self.aw,
                b_w=self.bw,
                ppg_cls=None,
                ppa_cls=None,
                fsa_cls=None,
                f_aag_lines=self.f_aag_lines,
            ).make_internal()

            # wire inputs/outputs
            multiplier.io.a <<= self.io.a
            multiplier.io.b <<= self.io.b
            self.io.y <<= multiplier.io.y
            return
        else:

            # Split operands into 8-bit halves: a = a_hi<<8 + a_lo; b = b_hi<<8 + b_lo
            a_lo = self.io.a[0 : self.aw//2]
            a_hi = self.io.a[self.aw//2 : self.aw]
            b_lo = self.io.b[0 : self.bw//2]
            b_hi = self.io.b[self.bw//2 : self.bw]

            # Instantiate four optimized 8x8 multipliers
            multipliers: List[StageBasedExtMultiplier] = []
            mult_cls = OptimizedMultiplierFrom4BitBlocks if (self.aw//2 > 4 and self.bw//2 > 4) else OptimizedMultiplier

            for _ in range(4):
                multipliers.append(
                    mult_cls(
                        a_encoding=Encoding.unsigned,
                        b_encoding=Encoding.unsigned,
                        a_w=self.aw//2,
                        b_w=self.bw//2,
                        ppg_cls=None,
                        ppa_cls=None,
                        fsa_cls=None,
                        f_aag_lines=self.f_aag_lines,
                    ).make_internal()
                )
            m_ll, m_lh, m_hl, m_hh = multipliers

            # Wire inputs
            m_ll.io.a <<= a_lo
            m_ll.io.b <<= b_lo

            m_lh.io.a <<= a_lo
            m_lh.io.b <<= b_hi

            m_hl.io.a <<= a_hi
            m_hl.io.b <<= b_lo

            m_hh.io.a <<= a_hi
            m_hh.io.b <<= b_hi

            # Partial products (each 16 bits wide)
            p0 = m_ll.io.y  # a_lo * b_lo
            p1 = m_lh.io.y  # a_lo * b_hi
            p2 = m_hl.io.y  # a_hi * b_lo
            p3 = m_hh.io.y  # a_hi * b_hi

            if not self.use_compressor_tree:

                # Zero-extend to 2*aw bits and align via left shifts using Concat
                p0 = Concat([p0, Const(0, UInt(self.aw))])  # no shift

                p1 = Concat([p1, Const(0, UInt(self.aw))])
                p1_sh8 = Concat([Const(0, UInt(self.aw//2)), p1[0:self.aw*3//2]])  # (p1 << aw//2)

                p2 = Concat([p2, Const(0, UInt(self.aw))])
                p2_sh8 = Concat([Const(0, UInt(self.aw//2)), p2[0:self.aw*3//2]])  # (p2 << aw//2)

                p3_sh16 = Concat([Const(0, UInt(self.aw)), p3])  # (p3 << aw)

                # Sum partial products
                self.io.y <<= p0 + p1_sh8 + p2_sh8 + p3_sh16

            else:

                # use of compression tree
                # generate dict with Dict[int, Expr] with int indicating bit weight
                cols: DefaultDict[int, List[Expr]] = defaultdict(list)
                for i in range(p0.typ.width):
                    cols[i].append(p0[i])
                for i in range(p1.typ.width):
                    cols[i + self.aw//2].append(p1[i])
                for i in range(p2.typ.width):
                    cols[i + self.aw//2].append(p2[i])
                for i in range(p3.typ.width):
                    cols[i + self.aw].append(p3[i])

                config = MultiplierConfig(
                    a_width=self.aw,
                    b_width=self.bw,
                    signed_a=Encoding.unsigned,
                    signed_b=Encoding.unsigned,
                    optim_type=self.optim_type,
                )

                # ppg_cls = CompressorTreeAccumulator
                ppg_cls = CarrySaveAccumulator # smaller depth same aig count for 8 bit multiplier
                ppa = ppg_cls(config=config)
                ppa_cols = ppa.accumulate(cols)
                fsa = RippleCarryFinalAdder(config=config)
                fsa_bits = fsa.resolve(ppa_cols)

                # another option is to use compressor_sum for the summation
                # # Zero-extend to 2*aw bits and align via left shifts using Concat
                # p0 = Concat([p0, Const(0, UInt(self.aw))])  # no shift

                # p1 = Concat([p1, Const(0, UInt(self.aw))])
                # p1_sh8 = Concat([Const(0, UInt(self.aw//2)), p1[0:self.aw*3//2]])  # (p1 << aw//2)

                # p2 = Concat([p2, Const(0, UInt(self.aw))])
                # p2_sh8 = Concat([Const(0, UInt(self.aw//2)), p2[0:self.aw*3//2]])  # (p2 << aw//2)

                # p3_sh16 = Concat([Const(0, UInt(self.aw)), p3])  # (p3 << aw)

                # fsa_bits = compressor_sum(
                #     config=OutputConfig(
                #         out_width=self.aw + self.bw,
                #         optim_type=self.optim_type,
                #     ),
                #     partials=[p0, p1_sh8, p2_sh8, p3_sh16],
                #     ppg_cls=CarrySaveAccumulator,
                #     fsa_cls=RippleCarryFinalAdder,
                # )

                self.io.y <<= Concat(fsa_bits)


class OptimizedMultiplierFrom4BitBlocksStrong(OptimizedMultiplierFrom4BitBlocks):
    f_aag_lines = staticmethod(get_optimized_aag_lines_strong)


def test_multiplier_ext_optimized() -> None:
    n_bits = 8
    signed = False

    c = OptimizedMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )
    mod = c.to_module("multiplier_ext_optimized")
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

    run_vectors(module, vecs)

    # sign magnitude version

    n_bits = 4

    from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import MultiplierOption, encoding_for_multiplier

    encodings = encoding_for_multiplier(MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER.value)[0]

    c = OptimizedSignMagnitudeMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
    )
    mod = c.to_module("multiplier_ext_optimized_sign_magnitude")
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

    run_vectors(module, vecs)

    # assembled from 4 bit blocks

    n_bits = 8

    c = OptimizedMultiplierFrom4BitBlocks(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
    )
    mod = c.to_module("multiplier_ext_optimized_8x8_from_4x4")
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
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)

    # use a different file

    n_bits = 8
    signed = False

    c = OptimizedMultiplierFrom4BitBlocksStrong(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )

    mod = c.to_module("multiplier_ext_optimized")
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
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)
    
    n_bits = 16
    signed = False
    
    c = OptimizedMultiplierFrom4BitBlocksStrong(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=to_encoding(signed),
        b_encoding=to_encoding(signed),
        ppg_cls=None,
        ppa_cls=None,
        fsa_cls=None,
        optim_type="area",
    )
    
    mod = c.to_module("multiplier_ext_optimized")
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
        a_encoding=Encoding.unsigned,
        b_encoding=Encoding.unsigned,
        y_encoding=Encoding.unsigned,
    ).generate()
    run_vectors(module, vecs)


if __name__ == "__main__":
    test_multiplier_ext_optimized()