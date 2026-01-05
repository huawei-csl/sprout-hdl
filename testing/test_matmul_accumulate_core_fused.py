from __future__ import annotations

import numpy as np

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption, TwoInputAritEncodings
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import MMAcDims, MMAcWidths
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_fused import MMAcFusedCfg, MultiplierConfig, build_matmul_accumulate
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl_simulator import Simulator


def test_mmac_core_basic_simulation():
    dim = 4
    a_width = 4 #8
    b_width = 4 #8
    c_width = 10 #18

    encoding = Encoding.unsigned # for twos_complement see test_matmul_accumulate_core.py and make sure the input type into ppa is signed
    signed_io_type = True

    mult_cfg = MultiplierConfig(
        ppg_opt=PPGOption.BAUGH_WOOLEY if encoding == Encoding.twos_complement else PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )

    dims = MMAcDims(dim_m=dim, dim_n=dim, dim_k=dim)
    widths = MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width)
    cfg = MMAcFusedCfg(dims=dims, widths=widths, mult_cfg=mult_cfg, encoding=encoding)

    core_build_out = build_matmul_accumulate(cfg, signed_io_type=signed_io_type)

    core_build_out.module.to_verilog_file("mmac_core_fused.v")
    print(f"Output matrix Y has shape: ({dim}, {dim}) with element width {core_build_out.Y[0,0].typ.width} bits")

    sim = Simulator(core_build_out.module)

    # rng = np.random.default_rng(seed=42)
    # a_vals = rng.integers(0, 2**a_width, size=(dim, dim), dtype=int)
    # b_vals = rng.integers(0, 2**b_width, size=(dim, dim), dtype=int)
    # c_vals = rng.integers(0, 2**c_width, size=(dim, dim), dtype=int)

    # for i in range(dim):
    #     for j in range(dim):
    #         sim.set(core.A[i, j], int(a_vals[i, j]))
    #         sim.set(core.B[i, j], int(b_vals[i, j]))
    #         sim.set(core.C[i, j], int(c_vals[i, j]))

    # sim.eval()

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
        sim.peek(core_build_out.Y[0, 1][0])
        # encode each element according to the encoding
        y_np_encoded = np.vectorize(
            lambda x: EncodingModel(encoding).encode_value(int(x), core_build_out.Y[0, 0].typ.width)
        )(y_np)
        assert np.array_equal(y_hw, y_np_encoded), "Simulation mismatch for matmul accumulate core"

    # get yosys transistor count
    yosys_metrics = get_yosys_metrics(core_build_out.module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
