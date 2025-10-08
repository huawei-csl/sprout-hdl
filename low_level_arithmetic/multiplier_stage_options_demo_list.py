from typing import List, Tuple
from low_level_arithmetic.multiplier_stage_options_demo_lib import Demo, FSAOption, MultiplierEncodings, PPAOption, PPGOption, StageMultiplier, encoding_for_multiplier
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier, StageBasedMultiplierBasic, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeExtToTwosComplementMultiplier, StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier, StageBasedSignMagnitudeExtUpMultiplier, StageBasedSignMagnitudeMultiplier, StageBasedSignMagnitudeToTwosComplementMultiplier, StarMultiplier
from low_level_arithmetic.test_vector_generation import Encoding


demos1: List[Demo] = [
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
        FSAOption.RIPPLE,
    ),
    (
        StageBasedSignMagnitudeExtToTwosComplementMultiplier,
        MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement),
        PPGOption.BASIC,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE,
    ),
    (
        StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier,
        MultiplierEncodings.with_enc(Encoding.sign_magnitude_ext).set_output(Encoding.twos_complement_upper),
        PPGOption.BASIC,
        PPAOption.WALLACE_TREE,
        FSAOption.RIPPLE,
    ),
    (
        StarMultiplier,
        MultiplierEncodings.with_enc(Encoding.twos_complement),
        PPGOption.NONE,
        PPAOption.NONE,
        FSAOption.NONE,
    ),
    (
        StarMultiplier,
        MultiplierEncodings.with_enc(Encoding.unsigned),
        PPGOption.NONE,
        PPAOption.NONE,
        FSAOption.NONE,
    ),
]




def get_selection1_list():

    demos = []
    
    ppg_opt = PPGOption.BASIC
    ppa_opt = PPAOption.WALLACE_TREE
    fsa_opt = FSAOption.PREFIX_BRENT_KUNG

    # get all multiplier options
    for sm in StageMultiplier.get_list_with_all():
        
        for encoding in encoding_for_multiplier(sm.value):
            demos.append((sm.value, encoding, ppg_opt, ppa_opt, fsa_opt))

    return demos

if __name__ == "__main__":
    demos1 = get_selection1_list()
    print(f"Defined {len(demos1)} demo configurations to try")