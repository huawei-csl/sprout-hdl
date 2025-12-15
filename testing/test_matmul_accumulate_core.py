from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Sequence

import numpy as np

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel, is_signed
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import AdderConfig, MultiplierConfig, build_matmul_accumulate, max_y_width_unsigned
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl import Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Component, Module
from sprouthdl.sprouthdl_simulator import Simulator





def test_mmac_core_basic_simulation():
    dim = 4
    a_width = 8
    b_width = 8
    c_width =  max_y_width_unsigned(a_width, b_width, dim, include_carry_from_add=False)
    signed_io_type = True

    # use sprout operators
    # mult_cfg = MultiplierConfig(use_operator=True)
    # add_cfg = AdderConfig(use_operator=True, full_output_bit=True)

    encoding = Encoding.twos_complement

    # use custom multiplier and adder
    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(encoding),
        ppg_opt=PPGOption.BAUGH_WOOLEY if is_signed(encoding) else PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    add_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True, encoding=encoding)

    core_build_out = build_matmul_accumulate(dim, a_width, b_width, c_width, mult_cfg, add_cfg, signed_io_type=signed_io_type)

    print(
        f"Output matrix Y has shape: ({dim}, {dim}) with element width {core_build_out.Y[0,0].typ.width} bits"
    )

    # For simulation, operate on the module built directly from the reusable component.
    sim = Simulator(core_build_out.module)

    # either use this
    # a_min, a_max = EncodingModel(encoding).value_range(a_width)
    # b_min, b_max = EncodingModel(encoding).value_range(b_width)
    # c_min, c_max = EncodingModel(encoding).value_range(c_width)
    # rng = np.random.default_rng(seed=42)
    # a_vals = rng.integers(a_min, a_max, size=(dim, dim), dtype=int)
    # b_vals = rng.integers(b_min, b_max, size=(dim, dim), dtype=int)
    # c_vals = rng.integers(c_min, c_max, size=(dim, dim), dtype=int)

    # or
    a_vals = EncodingModel(encoding).get_uniform_sample_np(a_width, size=(dim, dim))
    b_vals = EncodingModel(encoding).get_uniform_sample_np(b_width, size=(dim, dim))
    c_vals = EncodingModel(encoding).get_uniform_sample_np(c_width, size=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            sim.set(core_build_out.A[i, j], int(a_vals[i, j]))
            sim.set(core_build_out.B[i, j], int(b_vals[i, j]))
            sim.set(core_build_out.C[i, j], int(c_vals[i, j]))

    sim.eval()

    y_hw = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            y_hw[i, j] = sim.get(core_build_out.Y[i, j])

    y_np = a_vals @ b_vals + c_vals
    if signed_io_type:
        assert np.array_equal(y_hw, y_np), "Simulation mismatch for matmul accumulate core"
    else:
        # encode each element according to the encoding
        y_np_encoded = np.vectorize(
            lambda x: EncodingModel(encoding).encode_value(int(x), core_build_out.Y[0, 0].typ.width)
        )(y_np)
        assert np.array_equal(y_hw, y_np_encoded), "Simulation mismatch for matmul accumulate core"


    
    print("Matmul accumulate simulation passed. Y=\n", y_hw)

    # get yosys transistor count
    yosys_metrics = get_yosys_metrics(core_build_out.module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
