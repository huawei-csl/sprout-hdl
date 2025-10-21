from low_level_arithmetic.stage_multiplier_eval.run_multiplier_stage_options_eval import run_stage_multiplier_demo
from low_level_arithmetic.stage_multiplier_eval.run_multiplier_stage_options_eval_ext import run_stage_multiplier_ext_demo


def test_stage_multipliers() -> None:

    run_stage_multiplier_ext_demo()

    run_stage_multiplier_demo()
    
if __name__ == "__main__":  # pragma: no cover - run as script
    test_stage_multipliers()