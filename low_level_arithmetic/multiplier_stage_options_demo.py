"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from enum import Enum
from typing import Tuple

from low_level_arithmetic.compressor_tree_direct_sprout_hdl_baugh_wooley import (
    MultiplierTestVectors,
)
from low_level_arithmetic.compressor_tree_direct_sprout_hdl_baugh_wooley_stages import (
    BaughWooleyPartialProductGenerator,
)
from low_level_arithmetic.compressor_tree_direct_sprout_hdl_basic_stages import (
    BasicUnsignedPartialProductGenerator,
)
from low_level_arithmetic.compressor_tree_direct_sprout_hdl_booth_optim_signed_stages import (
    BoothOptimizedSignedPartialProductGenerator,
)
from low_level_arithmetic.compressor_tree_direct_sprout_hdl_booth_optim_stages import (
    BoothOptimizedPartialProductGenerator,
)
from low_level_arithmetic.compressor_tree_direct_sprout_hdl_booth_unoptim_stages import (
    BoothUnoptimizedPartialProductGenerator,
)
from low_level_arithmetic.multiplier_stage_core import (
    CompressorTreeAccumulator,
    RippleCarryFinalAdder,
    StageBasedMultiplier,
)
from low_level_arithmetic.prefix_adder_stage import (
    BrentKungPrefixFinalStage,
    PrefixAdderFinalStage,
    RipplePrefixFinalStage,
    SklanskyPrefixFinalStage,
)
from testing.test_different_logic import run_vectors_io


class PPGOption(Enum):
    BASIC = BasicUnsignedPartialProductGenerator
    BAUGH_WOOLEY = BaughWooleyPartialProductGenerator
    BOOTH_UNOPTIMISED = BoothUnoptimizedPartialProductGenerator
    BOOTH_OPTIMISED = BoothOptimizedPartialProductGenerator
    BOOTH_OPTIMISED_SIGNED = BoothOptimizedSignedPartialProductGenerator


class PPAOption(Enum):
    COMPRESSOR_TREE = CompressorTreeAccumulator


class FSAOption(Enum):
    RIPPLE = RippleCarryFinalAdder
    PREFIX_KS = PrefixAdderFinalStage
    PREFIX_BK = BrentKungPrefixFinalStage
    PREFIX_SKLANSKY = SklanskyPrefixFinalStage
    PREFIX_RCA = RipplePrefixFinalStage





def main() -> None:  # pragma: no cover - demonstration only
    demos: Tuple[Tuple[PPGOption, PPAOption, FSAOption], ...] = (
        (PPGOption.BASIC, PPAOption.COMPRESSOR_TREE, FSAOption.RIPPLE),
        (PPGOption.BAUGH_WOOLEY, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_KS),
        (PPGOption.BOOTH_UNOPTIMISED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_RCA),
        (PPGOption.BOOTH_OPTIMISED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_BK),
        (PPGOption.BOOTH_OPTIMISED_SIGNED, PPAOption.COMPRESSOR_TREE, FSAOption.PREFIX_SKLANSKY),
    )

    width = 16
    completed = 0

    for ppg_opt, ppa_opt, fsa_opt in demos:
        for signed_a, signed_b in ppg_opt.value.supported_signatures or ((False, False),):
            multiplier = StageBasedMultiplier(
                a_w=width,
                b_w=width,
                signed_a=signed_a,
                signed_b=signed_b,
                ppg_cls=ppg_opt.value,
                ppa_cls=ppa_opt.value,
                fsa_cls=fsa_opt.value,
                optim_type="area",
            )
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
                signed_a=signed_a,
                signed_b=signed_b,
            ).generate()
            _ = specs
            run_vectors_io(module, vecs, decoder=decoder)

            completed += 1
            print(f"Completed {completed} multiplier demos.")


if __name__ == "__main__":
    main()
