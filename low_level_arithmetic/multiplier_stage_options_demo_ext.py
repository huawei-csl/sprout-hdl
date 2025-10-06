"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple, Self, Tuple, Type


from low_level_arithmetic.multiplier_stage_options_demo import FSAOption, PPAOption, PPGOption
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier, StageBasedMultiplierBasic, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeExtToTwosComplementMultiplier, StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier, StageBasedSignMagnitudeExtUpMultiplier, StageBasedSignMagnitudeMultiplier, StageBasedSignMagnitudeToTwosComplementMultiplier
from low_level_arithmetic.ppa_stages import (
    CarrySaveAccumulator,
    DaddaTreeAccumulator,
    FourTwoCompressorAccumulator,
    WallaceTreeAccumulator,
)
from low_level_arithmetic.test_vector_generation import (
    Encoding,
    MultiplierTestVectors,
    to_encoding,
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

    class MultiplierEncodings(NamedTuple):
        a: Encoding
        b: Encoding
        y: Encoding

        def with_(self, **changes) -> Self:
            return self._replace(**changes)

        def set_inputs(self, enc: Encoding) -> Self:
            return self.with_(a=enc, b=enc)

        def set_output(self, enc: Encoding) -> Self:
            return self.with_(y=enc)

        def set_all(self, enc: Encoding) -> Self:
            return self.with_(a=enc, b=enc, y=enc)

        @classmethod
        def with_enc(cls, enc: Encoding) -> Self:
            return cls(a=enc, b=enc, y=enc)

    class Demo(NamedTuple):
        multiplier_cls: Type[StageBasedExtMultiplier]
        encoding: MultiplierEncodings
        ppg_opt: PPGOption
        ppa_opt: PPAOption
        fsa_opt: FSAOption

    # class MultiplierFormat(NamedTuple):
    #     a: Encoding
    #     b: Encoding
    #     y: Encoding
    #     a_w: int
    #     b_w: int
    #     y_w: int

    #     def with_(self, **changes) -> Self:
    #         return self._replace(**changes)

    #     def set_all_encodings(self, enc: Encoding) -> Self:
    #         return self.with_(a=enc, b=enc, y=enc)

    #     def set_input_widths(self, input_w: int) -> Self:
    #         return self.with_(a_w=input_w, b_w=input_w)

    #     def set_output_width(self, output_w: int) -> Self:
    #         return self.with_(y_w=output_w)

    #     @classmethod
    #     def default(cls, input_w: int) -> Self:
    #         return cls(
    #             a=Encoding.unsigned,
    #             b=Encoding.unsigned,
    #             y=Encoding.unsigned,
    #             a_w=input_w,
    #             b_w=input_w,
    #             y_w=input_w * 2,
    #         )

    # define some demo combinations to try
    demos: Tuple[Demo, ...] = (
        (StageBasedSignMagnitudeMultiplier, MultiplierEncodings.with_enc(Encoding.sign_magnitude), PPGOption.BASIC, PPAOption.WALLACE_TREE, FSAOption.RIPPLE),
        (
            StageBasedSignMagnitudeExtMultiplier,
            MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext),
            PPGOption.BASIC,
            PPAOption.WALLACE_TREE,
            FSAOption.RIPPLE,
        ),
        (
            StageBasedSignMagnitudeExtUpMultiplier,
            MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.sign_magnitude_ext_up),
            PPGOption.BASIC,
            PPAOption.WALLACE_TREE,
            FSAOption.RIPPLE,
        ),
        (StageBasedMultiplierBasic, MultiplierEncodings.with_enc(Encoding.unsigned), PPGOption.BASIC, PPAOption.WALLACE_TREE, FSAOption.RIPPLE),
        (
            StageBasedMultiplierBasic,
            MultiplierEncodings.with_enc(Encoding.twos_complement),
            PPGOption.BOOTH_OPTIMISED_SIGNED,
            PPAOption.COMPRESSOR_TREE,
            FSAOption.PREFIX_SKLANSKY,
        ),
        (
            StageBasedSignMagnitudeToTwosComplementMultiplier,
            MultiplierEncodings.with_enc(Encoding.sign_magnitude).set_output(Encoding.twos_complement),
            PPGOption.BASIC,
            PPAOption.WALLACE_TREE,
            FSAOption.RIPPLE
        ),
        (
             StageBasedSignMagnitudeExtToTwosComplementMultiplier,
             MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement),
             PPGOption.BASIC,
             PPAOption.WALLACE_TREE,
             FSAOption.RIPPLE
         ),
        (
            StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier,
            MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement_upper),
            PPGOption.BASIC,
            PPAOption.WALLACE_TREE,
            FSAOption.RIPPLE
        ),
    
    )


    completed_demo_runs = 0

    num_vectors = 100
    bitwidths = [4, 8, 16]

    for width in bitwidths:

        for multiplier_cls, encodings, ppg_opt, ppa_opt, fsa_opt in demos:

            reset_shared_cache()

            multiplier = multiplier_cls(
                a_w=width,
                b_w=width,
                a_encoding=encodings.a,
                b_encoding=encodings.b,
                ppg_cls=ppg_opt.value,
                ppa_cls=ppa_opt.value,
                fsa_cls=fsa_opt.value,
                optim_type="area",
            )

            module = multiplier.to_module(f"demo_{ppg_opt.name.lower()}_{encodings.a.name.lower()}_{encodings.b.name.lower()}_{fsa_opt.name.lower()}")
            print(f"Built module '{module.name}' using PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}")

            specs, vecs, decoder = MultiplierTestVectors(
                a_w=width,
                b_w=width,
                y_w=multiplier.io.y.typ.width,
                num_vectors=num_vectors,
                tb_sigma=None,
                a_encoding=encodings.a,
                b_encoding=encodings.b,
                y_encoding=encodings.y,
            ).generate()

            run_vectors_io(module, vecs, decoder=decoder)

            completed_demo_runs += 1
            print(f"Completed {completed_demo_runs} multiplier demos.")
            gr = module.module_analyze()
            print(f"Graph report: {gr}")


if __name__ == "__main__":
    main()
