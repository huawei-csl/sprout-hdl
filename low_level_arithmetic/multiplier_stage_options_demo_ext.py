"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Tuple, Type

from low_level_arithmetic.multiplier_stage_options_demo import FSAOption, PPAOption, PPGOption
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeMultiplier
from low_level_arithmetic.ppa_stages import (
    CarrySaveAccumulator,
    DaddaTreeAccumulator,
    FourTwoCompressorAccumulator,
    WallaceTreeAccumulator,
)
from low_level_arithmetic.test_vector_generation import (
    Encoding,
    MultiplierTestVectors,
    to_format,
)
from low_level_arithmetic.ppg_baugh_wooley_stages import (
    BaughWooleyPartialProductGenerator,
)
from low_level_arithmetic.ppg_basic_stages import (
    BasicUnsignedPartialProductGenerator,
)
from low_level_arithmetic.ppg_booth_optim_signed_stages import (
    BoothOptimizedSignedPartialProductGenerator,
)
from low_level_arithmetic.ppg_booth_optim_stages import (
    BoothOptimizedPartialProductGenerator,
)
from low_level_arithmetic.ppg_booth_unoptim_stages import (
    BoothUnoptimizedPartialProductGenerator,
)
from low_level_arithmetic.multiplier_stage_core import (
    CompressorTreeAccumulator,
    RippleCarryFinalAdder,
    StageBasedMultiplier,
)
from low_level_arithmetic.fsa_stages import (
    BrentKungPrefixFinalStage,
    PrefixAdderFinalStage,
    RipplePrefixFinalStage,
    SklanskyPrefixFinalStage,
)
from sprouthdl.sprouthdl import reset_shared_cache
from testing.test_different_logic import run_vectors_io


def main() -> None:  # pragma: no cover - demonstration only
    
    class Demo(NamedTuple):
        multiplier_cls: Type[StageBasedExtMultiplier]
        encoding: Encoding
        ppg_opt: PPGOption
        ppa_opt: PPAOption
        fsa_opt: FSAOption

    # define some demo combinations to try
    demos: Tuple[Demo, ...] = (
        (StageBasedSignMagnitudeMultiplier, Encoding.sign_magnitude, PPGOption.BASIC, PPAOption.WALLACE_TREE, FSAOption.RIPPLE),
        (StageBasedSignMagnitudeExtMultiplier, Encoding.sign_magnitude_ext, PPGOption.BASIC, PPAOption.WALLACE_TREE, FSAOption.RIPPLE),
        # (PPGOption.BASIC, PPAOption.CARRY_SAVE_TREE, FSAOption.PREFIX_KS),
        # (PPGOption.BASIC, PPAOption.FOUR_TWO_COMPRESSOR, FSAOption.PREFIX_BK),
        # (PPGOption.BAUGH_WOOLEY, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_KS),
        # (PPGOption.BOOTH_UNOPTIMISED, PPAOption.DADDA_TREE, FSAOption.PREFIX_RCA),
        # (PPGOption.BOOTH_OPTIMISED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_BK),
        # (PPGOption.BOOTH_OPTIMISED_SIGNED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_SKLANSKY),
    )

    width = 16

    completed_demo_runs = 0

    num_vectors = 1000
    bitwidths = [4, 8, 16]

    for width in bitwidths:

        for multiplier_cls, format, ppg_opt, ppa_opt, fsa_opt in demos:

            reset_shared_cache()

            multiplier = multiplier_cls(
                a_w=width,
                b_w=width,
                a_format=format,
                b_format=format,
                ppg_cls=ppg_opt.value,
                ppa_cls=ppa_opt.value,
                fsa_cls=fsa_opt.value,
                optim_type="area",
            )

            module = multiplier.to_module(f"demo_{ppg_opt.name.lower()}_{format}_{format}_{fsa_opt.name.lower()}")
            print(f"Built module '{module.name}' using PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}")

            specs, vecs, decoder = MultiplierTestVectors(
                a_w=width,
                b_w=width,
                y_w=multiplier.io.y.typ.width,
                num_vectors=num_vectors,
                tb_sigma=None,
                a_format=format,
                b_format=format,
                y_format=format,
            ).generate()

            run_vectors_io(module, vecs, decoder=decoder)

            completed_demo_runs += 1
            print(f"Completed {completed_demo_runs} multiplier demos.")
            gr = module.module_analyze()
            print(f"Graph report: {gr}")


if __name__ == "__main__":
    main()
