from collections import defaultdict
from typing import DefaultDict, List

from sprouthdl.helpers import get_yosys_transistor_count
from low_level_arithmetic.int_multipliers.multipliers.multiplier_stage_core import CompressorTreeAccumulator, FinalStageAdderBase, MultiplierTestVectors, PartialProductAccumulatorBase, PartialProductGeneratorBase, RippleCarryFinalAdder, StageBasedMultiplierBasic, StageBasedMultiplierIO
from sprouthdl.sprouthdl import Concat, Expr
from sprouthdl.sprouthdl_module import Module
from testing.test_different_logic import run_vectors_io

# AND partial product generator (schoolbook method)

class AndPartialProductGenerator(PartialProductGeneratorBase):
    supported_signatures = (
        (False, False),
        (True, True),
        (True, False),
        (False, True),
    )

    def generate_columns(
        self, io: StageBasedMultiplierIO
    ) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = io.a
        b = io.b
        a_vec: Expr = a
        b_vec: Expr = b

        if a.typ.signed:
            sign_bit = a[a.typ.width - 1]
            a_vec = Concat([a] + [sign_bit] * a.typ.width)
        if b.typ.signed:
            sign_bit = b[b.typ.width - 1]
            b_vec = Concat([b] + [sign_bit] * b.typ.width)

        out_bits = io.y.typ.width

        for i in range(a_vec.typ.width):
            for j in range(b_vec.typ.width):
                weight = i + j
                if weight >= out_bits:
                    continue
                cols[weight].append(a_vec[i] & b_vec[j])

        total_bits = sum(len(v) for v in cols.values())
        print(
            f"PPG (Basic): generated {total_bits} bits across {len(cols)} columns"
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
        ppg_cls: type[PartialProductGeneratorBase] = AndPartialProductGenerator,
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
            f"MultiplierCompressorTree (Basic): {cfg.a_width}x{cfg.b_width} -> {cfg.out_width} bits"
        )


def gen_sprout_module(mult: ConfiguredMultiplier) -> Module:
    return mult.to_module(f"Mul{mult.config.a_width}_ct_basic")


def main() -> None:
    n_bits = 16
    signed = True

    mult = ConfiguredMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
    )

    module = gen_sprout_module(mult)
    transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
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