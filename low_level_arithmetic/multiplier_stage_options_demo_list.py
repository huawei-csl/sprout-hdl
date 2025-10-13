from typing import List, Tuple
from low_level_arithmetic.multiplier_stage_options_demo_lib import Demo, FSAOption, MultiplierEncodings, PPAOption, PPGOption, StageMultiplier, encoding_for_multiplier, get_list_from_enum
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier, StageBasedMultiplierBasic, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeExtToTwosComplementMultiplier, StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier, StageBasedSignMagnitudeExtUpMultiplier, StageBasedSignMagnitudeMultiplier, StageBasedSignMagnitudeToTwosComplementMultiplier, StarMultiplier
from low_level_arithmetic.test_vector_generation import Encoding, to_encoding


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


def get_selection1_list(large_sweep: bool = True, all_sigmas_sweep: bool = False) -> List[Demo]:

    demos : list[Demo] = []

    ppg_0 = PPGOption.BASIC
    ppa_0 = PPAOption.WALLACE_TREE
    fsa_0 = FSAOption.PREFIX_BRENT_KUNG

    # vary multiplier options
    for sm in get_list_from_enum(StageMultiplier):  # StageMultiplier.get_list_with_all():
        for encoding in encoding_for_multiplier(sm.value):
            demos.append(Demo(sm.value, encoding, ppg_0, ppa_0, fsa_0))

    # vary PPG options
    for ppg in get_list_from_enum(PPGOption):
        if ppg == PPGOption.NONE:
            continue
        for signed_a, signed_b in ppg.value.supported_signatures: # or ((False, False),):
            if signed_a != signed_b:
                continue 
            encoding = to_encoding(signed_a)
            demos.append(Demo(StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, MultiplierEncodings.with_enc(encoding), ppg, ppa_0, fsa_0))

    # vary PPA options
    for ppa in get_list_from_enum(PPAOption):
        if ppa == PPAOption.NONE:
            continue
        demos.append(Demo(StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, MultiplierEncodings.with_enc(Encoding.unsigned), ppg_0, ppa, fsa_0))
        demos.append(Demo(StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, MultiplierEncodings.with_enc(Encoding.twos_complement), ppg_0, ppa, fsa_0))

    # vary FSA options
    for fsa in get_list_from_enum(FSAOption):
        if fsa == FSAOption.NONE:
            continue
        demos.append(Demo(StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, MultiplierEncodings.with_enc(Encoding.unsigned), ppg_0, ppa_0, fsa))
        demos.append(Demo(StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, MultiplierEncodings.with_enc(Encoding.twos_complement), ppg_0, ppa_0, fsa))

    # add supposedly "best" option
    demos.append(Demo(StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, MultiplierEncodings.with_enc(Encoding.unsigned), PPGOption.BOOTH_OPTIMISED, PPAOption.DADDA_TREE, FSAOption.PREFIX_SKLANSKY))
    demos.append(Demo(multiplier_cls=StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, encodings=MultiplierEncodings.with_enc(Encoding.twos_complement), ppg_opt=PPGOption.BOOTH_OPTIMISED_SIGNED, ppa_opt=PPAOption.DADDA_TREE, fsa_opt=FSAOption.PREFIX_SKLANSKY))

    # now all combinations of PPG, PPA, FSA for basic multiplier with unsigned and twos_complement
    # with all_sigma = False to make it quicker
    from itertools import product
    for sm, ppg, ppa, fsa in product(get_list_from_enum(StageMultiplier), get_list_from_enum(PPGOption), get_list_from_enum(PPAOption), get_list_from_enum(FSAOption)):
        if ppg == PPGOption.NONE or ppa == PPAOption.NONE or fsa == FSAOption.NONE:
            continue
        if not large_sweep and sm != StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC:
            continue
        for signed_a, signed_b in ppg.value.supported_signatures: # or ((False, False),):
            if signed_a != signed_b:
                continue 
            # if already added above, skip
            if Demo(sm.value,  MultiplierEncodings.with_enc(to_encoding(signed_a)), ppg, ppa, fsa) in demos:
                continue
            encoding = to_encoding(signed_a)
            demos.append(Demo(multiplier_cls=StageMultiplier.STAGE_BASED_MULTIPLIER_BASIC.value, encodings=MultiplierEncodings.with_enc(encoding), ppg_opt=ppg, ppa_opt=ppa, fsa_opt=fsa, all_sigma=all_sigmas_sweep))
    return demos

if __name__ == "__main__":
    demos1 = get_selection1_list(large_sweep=True, all_sigmas_sweep=True)
    print(f"Defined {len(demos1)} demo configurations to try")
