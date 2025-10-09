from enum import Enum
from typing import List, NamedTuple, Self, Tuple, Type
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier, StageBasedMultiplierBasic, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeExtToTwosComplementMultiplier, StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier, StageBasedSignMagnitudeExtUpMultiplier, StageBasedSignMagnitudeMultiplier, StageBasedSignMagnitudeToTwosComplementMultiplier, StarMultiplier
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


# Options for each stage

class PPGOption(Enum):
    BASIC = BasicUnsignedPartialProductGenerator
    BAUGH_WOOLEY = BaughWooleyPartialProductGenerator
    BOOTH_UNOPTIMISED = BoothUnoptimizedPartialProductGenerator
    BOOTH_OPTIMISED = BoothOptimizedPartialProductGenerator
    BOOTH_OPTIMISED_SIGNED = BoothOptimizedSignedPartialProductGenerator
    NONE = None


class PPAOption(Enum):
    COMPRESSOR_TREE = CompressorTreeAccumulator
    WALLACE_TREE = WallaceTreeAccumulator
    DADDA_TREE = DaddaTreeAccumulator
    CARRY_SAVE_TREE = CarrySaveAccumulator
    FOUR_TWO_COMPRESSOR = FourTwoCompressorAccumulator
    NONE = None


class FSAOption(Enum):
    RIPPLE = RippleCarryFinalAdder
    PREFIX_KOGGE_STONE = PrefixAdderFinalStage
    PREFIX_BRENT_KUNG = BrentKungPrefixFinalStage
    PREFIX_SKLANSKY = SklanskyPrefixFinalStage
    PREFIX_RCA = RipplePrefixFinalStage
    NONE = None

class StageMultiplier(Enum):
    STAGE_BASED_MULTIPLIER_BASIC = StageBasedMultiplierBasic
    STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER = StageBasedSignMagnitudeMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_MULTIPLIER = StageBasedSignMagnitudeExtMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_UP_MULTIPLIER = StageBasedSignMagnitudeExtUpMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_TO_TWOS_COMPLEMENT_MULTIPLIER = StageBasedSignMagnitudeToTwosComplementMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_TO_TWOS_COMPLEMENT_MULTIPLIER = StageBasedSignMagnitudeExtToTwosComplementMultiplier
    STAGE_BASED_SIGN_MAGNITUDE_EXT_TO_TWOS_COMPLEMENT_UPPER_MULTIPLIER = StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier
    STAR_STAR_MULTIPLIER = StarMultiplier

    @classmethod
    def get_list_with_string(cls, search_string: str) -> list["StageMultiplier"]:
        result = []
        for entry in StageMultiplier:
            if search_string.lower() in entry.name.lower():
                result += [entry]
        return result
    
    @classmethod
    def get_list_with_all(cls) -> list["StageMultiplier"]:
        result = []
        for entry in StageMultiplier:
            result += [entry]
        return result

def get_list_from_enum(enum_cls: Type[Enum]) -> list[Enum]:
    result = []
    for entry in enum_cls:
        result += [entry]
    return result


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
    encodings: MultiplierEncodings
    ppg_opt: PPGOption
    ppa_opt: PPAOption
    fsa_opt: FSAOption
    all_sigma: bool = True


def encoding_for_multiplier(multiplier_cls: type[StageBasedExtMultiplier]) -> List[MultiplierEncodings]:
    if multiplier_cls == StageBasedSignMagnitudeMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.sign_magnitude)]
    elif multiplier_cls == StageBasedSignMagnitudeExtMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext)]
    elif multiplier_cls == StageBasedSignMagnitudeExtUpMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.sign_magnitude_ext_up)]
    elif multiplier_cls == StageBasedMultiplierBasic:
        return [MultiplierEncodings.with_enc(Encoding.unsigned), MultiplierEncodings.with_enc(Encoding.twos_complement)]
    elif multiplier_cls == StageBasedSignMagnitudeToTwosComplementMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.sign_magnitude).set_output(Encoding.twos_complement)]
    elif multiplier_cls == StageBasedSignMagnitudeExtToTwosComplementMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement)]
    elif multiplier_cls == StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement_upper)]
    elif multiplier_cls == StarMultiplier:
        return [MultiplierEncodings.with_enc(Encoding.unsigned), MultiplierEncodings.with_enc(Encoding.twos_complement)]
    else:
        raise ValueError(f"Unknown multiplier class: {multiplier_cls}")
