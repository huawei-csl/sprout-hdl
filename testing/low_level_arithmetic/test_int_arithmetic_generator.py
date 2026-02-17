import json
import os
from pathlib import Path
import subprocess
import sys

from sprouthdl.arithmetic.int_arithmetic_generator import (
    AdderGeneratorConfig,
    GenerationActions,
    MultiplierGeneratorConfig,
    generate_adder,
    generate_multiplier,
)
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding


def test_generate_multiplier_api_with_artifacts_and_sim(tmp_path: Path):
    cfg = MultiplierGeneratorConfig(
        n_bits=4,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.CARRY_SAVE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        input_encoding=Encoding.unsigned,
        with_clock=False,
    )
    actions = GenerationActions(
        verilog_out=tmp_path / "mul.v",
        aag_out=tmp_path / "mul.aag",
        simulate=True,
        num_vectors=24,
        yosys_stats=True,
        yosys_opt_iterations=1,
    )

    result = generate_multiplier(cfg, actions=actions)

    assert result.module.name.startswith("mul_")
    assert result.verilog_out == tmp_path / "mul.v"
    assert result.aag_out == tmp_path / "mul.aag"
    assert result.verilog_out.exists()
    assert result.aag_out.exists()
    assert result.simulation_failures == 0
    assert result.yosys_stats is not None
    assert result.transistor_count is not None
    assert result.transistor_count > 0


def test_generate_adder_api_with_sim(tmp_path: Path):
    cfg = AdderGeneratorConfig(
        n_bits=5,
        fsa_opt=FSAOption.PREFIX_BRENT_KUNG,
        input_encoding=Encoding.twos_complement,
        full_output_bit=True,
    )
    actions = GenerationActions(
        verilog_out=tmp_path / "add.v",
        simulate=True,
        num_vectors=20,
    )

    result = generate_adder(cfg, actions=actions)

    assert result.verilog_out == tmp_path / "add.v"
    assert result.verilog_out.exists()
    assert result.simulation_failures == 0
    assert result.input_encoding == Encoding.twos_complement
    assert result.output_encoding == Encoding.twos_complement


def test_cli_multiplier_smoke(tmp_path: Path):
    verilog_path = tmp_path / "cli_mul.v"
    aag_path = tmp_path / "cli_mul.aag"

    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

    cmd = [
        sys.executable,
        "-m",
        "sprouthdl.arithmetic.int_arithmetic_generator",
        "multiplier",
        "--n-bits",
        "4",
        "--multiplier-opt",
        "STAGE_BASED_MULTIPLIER",
        "--ppg-opt",
        "AND",
        "--ppa-opt",
        "ACCUMULATOR_TREE",
        "--fsa-opt",
        "RIPPLE_CARRY",
        "--encoding",
        "unsigned",
        "--simulate",
        "--num-vectors",
        "16",
        "--verilog-out",
        str(verilog_path),
        "--aag-out",
        str(aag_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    stdout = proc.stdout.strip()
    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    payload = json.loads(stdout[json_start : json_end + 1])

    assert payload["simulation_failures"] == 0
    assert payload["verilog_out"] == str(verilog_path)
    assert payload["aag_out"] == str(aag_path)
    assert verilog_path.exists()
    assert aag_path.exists()
