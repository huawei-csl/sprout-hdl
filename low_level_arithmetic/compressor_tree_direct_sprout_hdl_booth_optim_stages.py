from collections import defaultdict
from typing import DefaultDict, List

from low_level_arithmetic.compressor_tree_sprout_hdl import get_transistor_count_from_m_yosys
from low_level_arithmetic.multiplier_stage_core import (
    CompressorTreeAccumulator,
    FinalStageAdderBase,
    MultiplierTestVectors,
    PartialProductAccumulatorBase,
    PartialProductGeneratorBase,
    RippleCarryFinalAdder,
    StageBasedMultiplier,
)
from sprouthdl.sprouthdl import Bool, Const, Expr, SInt, cast
from sprouthdl.sprouthdl_module import Module
from testing.test_different_logic import run_vectors_io


class BoothOptimizedPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = (
        (False, False),
    )

    def generate_columns(self) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = self.core.io.a
        b = self.core.io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = self.core.io.y.typ.width
        b_signed = b.typ.signed

        mag1 = [a[i] for i in range(wa)] + [Const(False, Bool())]
        mag2 = [Const(False, Bool())] + mag1[:-1]
        mag1_inv = [~bit for bit in mag1]
        mag2_inv = [~bit for bit in mag2]

        mag1_neg = mag1_inv
        mag2_neg = mag2_inv

        # improved pre-computed negatives via two's complement
        mag1_invs = -cast(a, SInt(wa + 1))
        mag2_invs = mag1_invs << 1
        mag1_neg = [mag1_invs[i] for i in range(wa + 1)]
        mag2_neg = [mag2_invs[i] for i in range(wa + 1)]

        def bbit(k: int) -> Expr:
            if 0 <= k < wb:
                return b[k]
            return b[wb - 1] if b_signed else Const(False, Bool())

        n_groups = (wb + 2) // 2
        for group in range(n_groups):
            x = bbit(2 * group - 1)
            y = bbit(2 * group)
            z = bbit(2 * group + 1)

            nz, ny, nx = ~z, ~y, ~x
            eq000 = nz & ny & nx
            eq001 = nz & ny & x
            eq010 = nz & y & nx
            eq011 = nz & y & x
            eq100 = z & ny & nx
            eq101 = z & ny & x
            eq110 = z & y & nx
            eq111 = z & y & x

            sel_pos1 = eq001 | eq010
            sel_pos2 = eq011
            sel_neg1 = eq101 | eq110
            sel_neg2 = eq100

            neg = sel_neg1 | sel_neg2

            base_w = 2 * group
            max_len = (wa + 1) + 2
            for t in range(max_len):
                if t < wa + 1:
                    a_pos1 = mag1[t]
                    a_pos2 = mag2[t]
                    a_neg1 = mag1_neg[t]
                    a_neg2 = mag2_neg[t]

                    emit_bit = (
                        (a_pos1 & sel_pos1)
                        | (a_pos2 & sel_pos2)
                        | (a_neg1 & sel_neg1)
                        | (a_neg2 & sel_neg2)
                    )
                elif t == wa + 1:
                    emit_bit = ~neg
                elif t == wa + 2:
                    emit_bit = Const(True, Bool())
                else:
                    emit_bit = None

                if emit_bit is None:
                    continue

                weight = base_w + t
                if weight < out_bits:
                    cols[weight].append(emit_bit)

            if group == 0:
                correction_col = len(mag1)
                if correction_col < out_bits:
                    cols[correction_col].append(Const(True, Bool()))

        total_bits = sum(len(v) for v in cols.values())
        print(
            f"PPG (Booth optimised): generated {total_bits} bits across {len(cols)} columns"
        )
        return cols


class MultiplierCompressorTree(StageBasedMultiplier):
    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: str = "area",
        ppg_cls: type[PartialProductGeneratorBase] = BoothOptimizedPartialProductGenerator,
        ppa_cls: type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: type[FinalStageAdderBase] = RippleCarryFinalAdder,
    ) -> None:
        super().__init__(
            a_w,
            b_w,
            signed_a=signed_a,
            signed_b=signed_b,
            optim_type=optim_type,
            ppg_cls=ppg_cls,
            ppa_cls=ppa_cls,
            fsa_cls=fsa_cls,
        )

    def elaborate(self) -> None:
        super().elaborate()
        cfg = self.config
        print(
            f"MultiplierCompressorTree (Booth optimised): {cfg.a_width}x{cfg.b_width} -> {cfg.out_width} bits"
        )


def gen_sprout_module(mult: MultiplierCompressorTree) -> Module:
    return mult.to_module(f"Mul{mult.config.a_width}_ct_booth_opt")


def main() -> None:
    n_bits = 16
    signed = False

    mult = MultiplierCompressorTree(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
    )

    module = gen_sprout_module(mult)
    transistor_count = get_transistor_count_from_m_yosys(module, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {transistor_count}")

    specs, vecs, decoder = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        num_vectors=16,
        tb_sigma=None,
        signed_a=signed,
        signed_b=signed,
    ).generate()
    _ = specs
    run_vectors_io(module, vecs, decoder=decoder)


if __name__ == "__main__":
    main()
