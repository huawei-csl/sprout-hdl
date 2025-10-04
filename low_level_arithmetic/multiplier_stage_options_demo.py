"""Demonstrate how to mix and match multiplier stages."""

from typing import Dict, Tuple, Type

from low_level_arithmetic.compressor_tree_direct_sprout_hdl_baugh_wooley import MultiplierTestVectors
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
    FinalStageAdderBase,
    PartialProductAccumulatorBase,
    PartialProductGeneratorBase,
    RippleCarryFinalAdder,
    StageBasedMultiplier,
)
from testing.test_different_logic import run_vectors_io


PARTIAL_PRODUCT_GENERATORS: Dict[str, Type[PartialProductGeneratorBase]] = {
    "basic": BasicUnsignedPartialProductGenerator,
    "baugh_wooley": BaughWooleyPartialProductGenerator,
    "booth_unoptimised": BoothUnoptimizedPartialProductGenerator,
    "booth_optimised": BoothOptimizedPartialProductGenerator,
    "booth_optimised_signed": BoothOptimizedSignedPartialProductGenerator,
}

PARTIAL_PRODUCT_ACCUMULATORS: Dict[str, Type[PartialProductAccumulatorBase]] = {
    "compressor_tree": CompressorTreeAccumulator,
}

FINAL_STAGE_ADDERS: Dict[str, Type[FinalStageAdderBase]] = {
    "ripple": RippleCarryFinalAdder,
}


def build_multiplier(
    *,
    a_width: int,
    b_width: int,
    ppg: str,
    ppa: str = "compressor_tree",
    fsa: str = "ripple",
    signed_a: bool = False,
    signed_b: bool = False,
    optim_type: str = "area",
) -> StageBasedMultiplier:
    try:
        ppg_cls = PARTIAL_PRODUCT_GENERATORS[ppg]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown PPG '{ppg}'") from exc

    try:
        ppa_cls = PARTIAL_PRODUCT_ACCUMULATORS[ppa]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown PPA '{ppa}'") from exc

    try:
        fsa_cls = FINAL_STAGE_ADDERS[fsa]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown FSA '{fsa}'") from exc

    return StageBasedMultiplier(
        a_width,
        b_width,
        signed_a=signed_a,
        signed_b=signed_b,
        optim_type=optim_type,
        ppg_cls=ppg_cls,
        ppa_cls=ppa_cls,
        fsa_cls=fsa_cls,
    )


def main() -> None:  # pragma: no cover - demonstration only
    demos: Tuple[Tuple[str, str, str], ...] = (
        ("basic", "compressor_tree", "ripple"),
        ("baugh_wooley", "compressor_tree", "ripple"),
        ("booth_unoptimised", "compressor_tree", "ripple"),
        ("booth_optimised", "compressor_tree", "ripple"),
        ("booth_optimised_signed", "compressor_tree", "ripple"),
    )

    wa = wb = 16
    
    demo_count_f = 0

    for i, (ppg_name, ppa_name, fsa_name) in enumerate(demos):
            
        for signed_inputs in PARTIAL_PRODUCT_GENERATORS[ppg_name].supported_signatures:
            multiplier = build_multiplier(
                a_width=wa,
                b_width=wb,
                ppg=ppg_name,
                ppa=ppa_name,
                fsa=fsa_name,
                signed_a=signed_inputs[0],
                signed_b=signed_inputs[1],
            )
            module = multiplier.to_module(
                f"demo_{ppg_name}{signed_inputs}"
            )
            print(
                f"Built module '{module.name}' using PPG={ppg_name}, PPA={ppa_name}, FSA={fsa_name}"
            )

            specs, vecs, decoder = MultiplierTestVectors(
                a_w=wa,
                b_w=wb,
                num_vectors=16,
                tb_sigma=None,
                signed_a=signed_inputs[0],
                signed_b=signed_inputs[1],
            ).generate()
            
            run_vectors_io(module, vecs, decoder=decoder)
            
            demo_count_f += 1
            print(f"Completed {demo_count_f} multiplier demos.")


if __name__ == "__main__":
    main()
