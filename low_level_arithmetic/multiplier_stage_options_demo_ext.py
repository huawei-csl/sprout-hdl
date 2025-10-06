"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from low_level_arithmetic.multiplier_stage_options_demo import FSAOption, PPAOption, PPGOption
from low_level_arithmetic.ppa_stages import (
    CarrySaveAccumulator,
    DaddaTreeAccumulator,
    FourTwoCompressorAccumulator,
    WallaceTreeAccumulator,
)
from low_level_arithmetic.test_vector_generation import (
    Format,
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
    StageBasedSignMagnitudeMultiplier,
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

    # define some demo combinations to try
    demos: Tuple[Tuple[PPGOption, PPAOption, FSAOption], ...] = (
        (PPGOption.BASIC, PPAOption.WALLACE_TREE, FSAOption.RIPPLE),
        (PPGOption.BASIC, PPAOption.CARRY_SAVE_TREE, FSAOption.PREFIX_KS),
        (PPGOption.BASIC, PPAOption.FOUR_TWO_COMPRESSOR, FSAOption.PREFIX_BK),
        (PPGOption.BAUGH_WOOLEY, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_KS),
        (PPGOption.BOOTH_UNOPTIMISED, PPAOption.DADDA_TREE, FSAOption.PREFIX_RCA),
        (PPGOption.BOOTH_OPTIMISED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_BK),
        (PPGOption.BOOTH_OPTIMISED_SIGNED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_SKLANSKY),
    )

    width = 16

    completed_demo_runs = 0

    for width in (4, 8, 16):

        for ppg_opt, ppa_opt, fsa_opt in demos:
            
            for fromat in (Format.unsigned, Format.sign_magnitude_ext):
            #for signed_a, signed_b in ppg_opt.value.supported_signatures or ((False, False),):
                #print(f"Building multiplier with width={width}, signed_a={signed_a}, signed_b={signed_b}, PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}")
                reset_shared_cache()

                # multiplier = StageBasedMultiplier(
                #     a_w=width,
                #     b_w=width,
                #     signed_a=signed_a,
                #     signed_b=signed_b,
                #     ppg_cls=ppg_opt.value,
                #     ppa_cls=ppa_opt.value,
                #     fsa_cls=fsa_opt.value,
                #     optim_type="area",
                # )
                
                if (signed_a, signed_b) != (True, True):
                    print("Skipping non-signed Baugh-Wooley configuration")
                    continue
                
                multiplier = StageBasedSignMagnitudeMultiplier(
                    a_w=width,
                    b_w=width,
                    signed_a=False,
                    signed_b=False,
                    ppg_cls=ppg_opt.value,
                    ppa_cls=ppa_opt.value,
                    fsa_cls=fsa_opt.value,
                    optim_type="area",
                )
                format_a = Format.sign_magnitude
                format_b = Format.sign_magnitude
                
                module = multiplier.to_module(
                    f"demo_{ppg_opt.name.lower()}_{signed_a}_{signed_b}_{fsa_opt.name.lower()}"
                )
                print(
                    f"Built module '{module.name}' using PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}"
                )

                specs, vecs, decoder = MultiplierTestVectors(
                    a_w=width,
                    b_w=width,
                    num_vectors=8,
                    tb_sigma=None,
                    format_a=Format.sign_magnitude, #to_format(signed_a),
                    format_b=Format.sign_magnitude,#to_format(signed_b),
                ).generate()
                _ = specs
                run_vectors_io(module, vecs, decoder=decoder)

                completed_demo_runs += 1
                print(f"Completed {completed_demo_runs} multiplier demos.")
                gr = module.module_analyze()
                print(f"Graph report: {gr}")


if __name__ == "__main__":
    main()
