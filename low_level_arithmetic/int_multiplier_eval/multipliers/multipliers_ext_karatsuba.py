import abc
from collections import defaultdict
from dataclasses import dataclass
import re
import tempfile
from typing import Callable, ClassVar, DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple, Type

from aigverse import read_aiger_into_aig, write_aiger
import numpy as np

from low_level_arithmetic.int_multiplier_eval.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO

from low_level_arithmetic.int_multiplier_eval.multipliers.multipliers_ext_optimized import OptimizedMultiplier
from low_level_arithmetic.int_multiplier_eval.multipliers.mutipliers_ext import StageBasedExtMultiplier
from low_level_arithmetic.int_multiplier_eval.stages.ppa_fsa_util import OutputConfig, compressor_sum
from low_level_arithmetic.int_multiplier_eval.stages.ppa_stages import CarrySaveAccumulator
from low_level_arithmetic.int_multiplier_eval.testvector_generation import Encoding, MultiplierTestVectors, from_encoding, to_encoding
from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, file_to_lines
from sprouthdl.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count, optimize_aag
from sprouthdl.sprouthdl_aiger import AigerImporter
from sprouthdl.sprouthdl_module import Component
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux, mux_if
from sprouthdl.sprouthdl_module import Module

from testing.aag_conv.aig_to_aag import aig_file_to_aag_lines
from testing.test_different_logic import aig_file_to_aag_lines_via_yosys, run_vectors_io, verilog_to_aag_lines_via_yosys, verilog_to_aag_via_yosys


from typing import List, Optional

# Assumes the following are already imported in your file:
# from low_level_arithmetic.int_multiplier_eval.multipliers.multiplier_stage_core import (
#     StageBasedMultiplierIO,
# )
# from low_level_arithmetic.int_multiplier_eval.multipliers.mutipliers_ext import StageBasedExtMultiplier
# from low_level_arithmetic.int_multiplier_eval.testvector_generation import Encoding
# from sprouthdl.hdl import Signal, UInt, Const, Concat   # or your actual paths
# from low_level_arithmetic.int_multiplier_eval.multipliers.optimized_multiplier import OptimizedMultiplier


class KaratsubaMultiplierFrom4BitBlocks(StageBasedExtMultiplier):
    """
    Unsigned n×n multiplier built recursively from an optimized 4×4 block using Karatsuba.

    - Base case (width <= 4):
        Uses a single 4×4 OptimizedMultiplier, zero-extending inputs on the MSB side
        and slicing the low 2*width bits of the 8-bit product.

    - Recursive case (width > 4):
        Let n = width, k = n // 2, a = a_hi * 2^k + a_lo, b = b_hi * 2^k + b_lo.

            p0 = a_lo * b_lo
            p2 = a_hi * b_hi
            p1 = (a_lo + a_hi) * (b_lo + b_hi)

            middle = p1 - p0 - p2

            y = p0 + (middle << k) + (p2 << (2k))

        All intermediate results are explicitly kept/modded in 2*n bits using
        Concat + slicing to model shifts.
    """

    def __init__(self, *args, f_aag_lines: Optional[List[str]] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Only unsigned supported for now
        assert (
            self.a_encoding == Encoding.unsigned and self.b_encoding == Encoding.unsigned
        ), "KaratsubaMultiplierFrom4BitBlocks supports only unsigned encoding"

        # For now: square multipliers only
        assert self.aw == self.bw, "Karatsuba implementation assumes aw == bw"
        assert self.aw >= 1, "Operand width must be at least 1 bit"

        # Handle f_aag_lines as class attribute or constructor argument, but not both
        if hasattr(self, "f_aag_lines"):
            if f_aag_lines is not None:
                raise ValueError("f_aag_lines class attribute exists, do not pass f_aag_lines as argument")
        else:
            self.f_aag_lines = f_aag_lines

        # I/O interface
        self.io: StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=UInt(self.aw), kind="input"),
            b=Signal(name="b", typ=UInt(self.bw), kind="input"),
            y=Signal(name="y", typ=UInt(self.aw + self.bw), kind="output"),
        )
        
        self.use_compressor = True  # Set to False to use simple '+' for final sums

        self.elaborate()

    # ------------------------------------------------------------------ #
    # Leaf: implement width<=4 using the optimized 4×4 block
    # ------------------------------------------------------------------ #
    def _mul_leaf_4bit(self, a_sig, b_sig, width: int):
        """
        Multiply two unsigned signals of 'width' bits (1 <= width <= 4) using a
        4×4 OptimizedMultiplier:
          - zero-extend each operand on the MSB side to 4 bits
          - multiply using the 4×4 block
          - return the low 2*width bits of the 8-bit product
        """
        assert 1 <= width <= 4

        if width == 4:
            a_ext = a_sig
            b_ext = b_sig
        else:
            # Zero-extend on MSB side: [low_bits, zeros] so value is unchanged
            a_ext = Concat([a_sig, Const(0, UInt(4 - width))])
            b_ext = Concat([b_sig, Const(0, UInt(4 - width))])

        core = OptimizedMultiplier(
            a_encoding=Encoding.unsigned,
            b_encoding=Encoding.unsigned,
            a_w=4,
            b_w=4,
            ppg_cls=None,
            ppa_cls=None,
            fsa_cls=None,
            f_aag_lines=self.f_aag_lines,
        ).make_internal()

        core.io.a <<= a_ext
        core.io.b <<= b_ext

        # Core product is 8 bits; we only need 2*width bits
        return core.io.y
    
    def _mul_leave_optimized(self, a_sig, b_sig, width):
        core = OptimizedMultiplier(
            a_encoding=Encoding.unsigned,
            b_encoding=Encoding.unsigned,
            a_w=width,
            b_w=width,
            ppg_cls=None,
            ppa_cls=None,
            fsa_cls=None,
            f_aag_lines=self.f_aag_lines,
        ).make_internal()
        
        core.io.a <<= a_sig
        core.io.b <<= b_sig
        
        # Core product is 8 bits; we only need 2*width bits
        return core.io.y

    def _mul_5bit_from_4bit_block(self, a5, b5, width=5):
        """
        5x5 unsigned multiply using:
          - one 4x4 OptimizedMultiplier on the low 4 bits
          - simple AND-based cross terms for the top bits
        Returns a 10-bit Expr (lower 10 bits of the full product).
        """
        
        width_core = width-1

        # Split into high / low parts
        a_lo = a5[0:4]   # bits 3..0
        a_hi = a5[4]     # bit 4
        b_lo = b5[0:4]
        b_hi = b5[4]

        # 1) 4x4 core: a_lo * b_lo  -> 8 bits
        core = OptimizedMultiplier(
            a_encoding=Encoding.unsigned,
            b_encoding=Encoding.unsigned,
            a_w=4,
            b_w=4,
            ppg_cls=None,
            ppa_cls=None,
            fsa_cls=None,
            f_aag_lines=self.f_aag_lines,
        ).make_internal()
        core.io.a <<= a_lo
        core.io.b <<= b_lo
        p0_8 = core.io.y  # 8-bit product

        # Map to 10 bits: bits [7:0] = p0_8, bits [9:8] = 0
        t0 = Concat([p0_8, Const(0, UInt(2))])  # 10 bits

        # 2) Cross terms: a_hi * b_lo and b_hi * a_lo  (each 4 bits)
        # Assuming scalar&vector broadcasting in your DSL; if not, replace with Mux.
        term_a = mux(a_hi, b_lo, Const(0, UInt(4)))  # b_lo & a_hi   # 4 bits: if a_hi==1 -> b_lo else 0
        term_b = mux(b_hi, a_lo, Const(0, UInt(4)))  # a_lo & b_hi   # 4 bits: if b_hi==1 -> a_lo else 0

        # Place them at bits [7:4] (i.e., << 4) and zero-extend to 10 bits
        # bits 0..3 = 0, bits 4..7 = term_*, bits 8..9 = 0
        t1 = Concat([Const(0, UInt(4)), term_a, Const(0, UInt(2))])  # 10 bits
        t2 = Concat([Const(0, UInt(4)), term_b, Const(0, UInt(2))])  # 10 bits

        # 3) High-high term: a_hi * b_hi at bit 8 (<< 8), 10-bit wide
        term_hi = a_hi & b_hi  # 1 bit
        # bits 0..7 = 0, bit 8 = term_hi, bit 9 = 0
        t3 = Concat([Const(0, UInt(8)), term_hi, Const(0, UInt(1))])  # 10 bits

        # 4) Final sum in 10-bit ring (add may create a carry, slice back to 10)
        if not self.use_compressor:
            prod_11 = t0 + t1 + t2 + t3
        else:
            prod_11 = compressor_sum(
                config=OutputConfig(
                    out_width=self.aw + self.bw,
                    optim_type=self.optim_type,
                ),
                partials=[t0, t1, t2, t3],
                ppg_cls=CarrySaveAccumulator,
                fsa_cls=RippleCarryFinalAdder,
            )

        return prod_11[0:10]

    # def _mul_5bit_from_4bit_block(self, a5, b5):
    #     """
    #     5x5 unsigned multiply using:
    #       - one 4x4 OptimizedMultiplier on the low 4 bits
    #       - simple AND/mux-based cross terms for the top bits
    #     Returns exactly 10 bits.
    #     """
    #     # Split into high / low parts
    #     a_lo = a5[0:4]   # a[3:0]
    #     a_hi = a5[4]     # a[4]
    #     b_lo = b5[0:4]   # b[3:0]
    #     b_hi = b5[4]     # b[4]

    #     # 1) 4x4 core: a_lo * b_lo  -> 8 bits
    #     core = OptimizedMultiplier(
    #         a_encoding=Encoding.unsigned,
    #         b_encoding=Encoding.unsigned,
    #         a_w=4,
    #         b_w=4,
    #         ppg_cls=None,
    #         ppa_cls=None,
    #         fsa_cls=None,
    #         f_aag_lines=self.f_aag_lines,
    #     ).make_internal()
    #     core.io.a <<= a_lo
    #     core.io.b <<= b_lo
    #     p0_8 = core.io.y  # 8-bit product

    #     # Map to 10 bits: bits [7:0] = p0_8, bits [9:8] = 0
    #     t0 = Concat([p0_8, Const(0, UInt(2))])  # 10 bits

    #     # 2) Cross terms: a_hi * b_lo and b_hi * a_lo  (each 4 bits)
    #     # use mux to be explicit about gating
    #     term_a = mux(a_hi, b_lo, Const(0, UInt(4)))  # if a_hi==1 -> b_lo else 0
    #     term_b = mux(b_hi, a_lo, Const(0, UInt(4)))  # if b_hi==1 -> a_lo else 0

    #     # Place them at bits [7:4] (i.e., << 4), zero-extend to 10 bits:
    #     # bits 0..3 = 0, bits 4..7 = term_*, bits 8..9 = 0
    #     t1 = Concat([Const(0, UInt(4)), term_a, Const(0, UInt(2))])  # 10 bits
    #     t2 = Concat([Const(0, UInt(4)), term_b, Const(0, UInt(2))])  # 10 bits

    #     # 3) High-high term: a_hi * b_hi at bit 8 (<< 8), 10-bit wide
    #     term_hi = a_hi & b_hi  # 1 bit
    #     # bits 0..7 = 0, bit 8 = term_hi, bit 9 = 0
    #     t3 = Concat([Const(0, UInt(8)), term_hi, Const(0, UInt(1))])  # 10 bits

    #     # 4) Final sum (may grow by a carry; slice back to 10 LSBs)
    #     prod_11 = t0 + t1 + t2 + t3
    #     return prod_11[0:10]

    # def _mul_5bit_from_4bit_block(self, a5, b5):
    #     return a5 * b5  # Placeholder implementation; replace with actual logic as needed

    # ------------------------------------------------------------------ #
    # Recursive Karatsuba construction (pure-combinational in this module)
    # ------------------------------------------------------------------ #
    def _build_karatsuba(self, a_sig, b_sig, width: int):
        """
        Recursively build an 'width'×'width' unsigned multiplier from 4×4 blocks.
        Returns an Expr of width 2*width bits.
        """
        assert width >= 1

        # Base case: use 4×4 optimized core (with padding for widths < 4)
        if width <= 4:
            return self._mul_leaf_4bit(a_sig, b_sig, width)

        if width == 5:
            return self._mul_5bit_from_4bit_block(a_sig, b_sig)

        n = width
        total_w = 2 * n
        k = n // 2
        lo_w = k
        hi_w = n - k

        # Split operands: a = a_hi*2^k + a_lo
        a_lo = a_sig[0:lo_w]
        a_hi = a_sig[lo_w:n]
        b_lo = b_sig[0:lo_w]
        b_hi = b_sig[lo_w:n]

        # --- p0 = a_lo * b_lo  (lo_w x lo_w) ---------------------------
        p0_small = self._build_karatsuba(a_lo, b_lo, lo_w)  # width: 2*lo_w

        # --- p2 = a_hi * b_hi  (hi_w x hi_w) ---------------------------
        p2_small = self._build_karatsuba(a_hi, b_hi, hi_w)  # width: 2*hi_w

        # --- p1 = (a_lo + a_hi) * (b_lo + b_hi) -----------------------
        # Your DSL already widens the result by one bit to hold the carry,
        # so we don't need to manually extend the operands.
        a_sum = a_lo + a_hi  # width: sum_w = max(lo_w, hi_w) + 1
        b_sum = b_lo + b_hi
        sum_w = max(lo_w, hi_w) + 1

        p1_small = self._build_karatsuba(a_sum, b_sum, sum_w)  # width: 2*sum_w

        # --- Embed partial products into a common 2*n-bit domain -------
        # Zero-extend on the MSB side using Concat([value, zeros])
        p0_2n = Concat([p0_small, Const(0, UInt(total_w - 2 * lo_w))])
        p2_2n = Concat([p2_small, Const(0, UInt(total_w - 2 * hi_w))])
        p1_2n = Concat([p1_small, Const(0, UInt(total_w - 2 * sum_w))])

        # middle = p1 - p0 - p2 in 2*n bits (mod 2^(2n))
        middle_2n = p1_2n - p0_2n - p2_2n  # stays 2*n bits in your DSL

        # --- Shifts implemented via Concat + slicing -------------------
        # (x << k) mod 2^(2n) == take low 2n bits of Concat([zeros(k), x])

        # (middle << k) in 2*n bits
        middle_shift_big = Concat([Const(0, UInt(k)), middle_2n])  # width: 2*n + k
        middle_shift_2n = middle_shift_big[0:total_w]

        # (p2 << 2k) in 2*n bits
        p2_shift_big = Concat([Const(0, UInt(2 * k)), p2_2n])  # width: 2*n + 2k
        p2_shift_2n = p2_shift_big[0:total_w]

        # Final Karatsuba combination in 2*n-bit ring
        if not self.use_compressor:
            prod_2n = p0_2n + middle_shift_2n + p2_shift_2n  # width: 2*n (or 2*n+1 → we'll slice at top)
        else:
            prod_2n = compressor_sum(
                config=OutputConfig(
                    out_width=self.aw + self.bw,
                    optim_type=self.optim_type,
                ),
                # partials=[p0_2n, middle_shift_2n, p2_shift_2n],
                partials=[p0_2n, middle_shift_big, p2_shift_big],
                ppg_cls=CarrySaveAccumulator,
                fsa_cls=RippleCarryFinalAdder,
            )

        return prod_2n

    # ------------------------------------------------------------------ #
    # Top-level elaboration
    # ------------------------------------------------------------------ #
    def elaborate(self) -> None:
        n = self.aw
        assert n == self.bw, "Karatsuba implementation assumes aw == bw"

        prod = self._build_karatsuba(self.io.a, self.io.b, n)

        # Ensure we drive exactly 2*n bits (in case '+' returns an extra carry bit)
        self.io.y <<= prod[0 : 2 * n]


def test_multiplier_ext_optimized() -> None:

    n_bits = 16
    signed = False

    c = KaratsubaMultiplierFrom4BitBlocks(
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
    run_vectors_io(module, vecs)


if __name__ == "__main__":
    test_multiplier_ext_optimized()
