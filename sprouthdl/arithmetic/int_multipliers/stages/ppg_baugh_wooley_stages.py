from collections import defaultdict
from typing import DefaultDict, Dict, List

from sprouthdl.helpers import get_yosys_transistor_count
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierTestVectors, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Module
from testing.test_different_logic import run_vectors_io


class BaughWooleyPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = ((True, True),)

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = io.a
        b = io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = io.y.typ.width

        for i in range(wa - 1):
            for j in range(wb - 1):
                weight = i + j
                if weight >= out_bits:
                    continue
                cols[weight].append(a[i] & b[j])

        i = wa - 1
        for j in range(wb - 1):
            cols[i + j].append(~(a[i] & b[j]))

        j = wb - 1
        for i in range(wa - 1):
            cols[i + j].append(~(a[i] & b[j]))

        cols[wa - 1 + wb - 1].append(a[wa - 1] & b[wb - 1])

        cols[wa - 1 + wb - 1 + 1].append(Const(True, Bool()))
        cols[wa - 1].append(Const(True, Bool()))
        cols[wb - 1].append(Const(True, Bool()))

        total_bits = sum(len(v) for v in cols.values())
        print(
            f"PPG (Baugh-Wooley): generated {total_bits} bits across {len(cols)} columns"
        )
        return cols


class ConfiguredMultiplier(StageBasedMultiplierBasic):
    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: str = "area",
        ppg_cls: type[PartialProductGeneratorBase] = BaughWooleyPartialProductGenerator,
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
            f"MultiplierCompressorTree: {cfg.a_width}x{cfg.b_width} -> {cfg.out_width} bits"
        )



def gen_sprout_module(mult: ConfiguredMultiplier) -> Module:
    return mult.to_module(f"Mul{mult.config.a_width}_ct")


def main() -> None:
    n_bits = 4
    signed = True

    mult = ConfiguredMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
    )

    module = gen_sprout_module(mult)
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10, deepsyn=False)
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
    run_vectors_io(module, vecs, decoder=decoder, use_signed=True)


if __name__ == "__main__":
    main()