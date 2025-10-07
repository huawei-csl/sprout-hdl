from low_level_arithmetic.multiplier_stage_options_demo import run_stage_multiplier_demo
from low_level_arithmetic.multiplier_stage_options_demo_ext import run_stage_multiplier_ext_demo


def test_stage_multipliers() -> None:

    run_stage_multiplier_ext_demo()

    run_stage_multiplier_demo()
    
if __name__ == "__main__":  # pragma: no cover - run as script
    test_stage_multipliers()
