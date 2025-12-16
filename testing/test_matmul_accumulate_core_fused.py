from __future__ import annotations

import numpy as np

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import MMAcDims, MMAcWidths
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_fused import MMAcFusedCfg, MultiplierConfig, build_matmul_accumulate
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl_simulator import Simulator


def test_mmac_core_basic_simulation():
    dim = 4
    a_width = 8
    b_width = 8
    c_width = 20

    mult_cfg = MultiplierConfig(ppg_opt=PPGOption.AND, ppa_opt=PPAOption.WALLACE_TREE, fsa_opt=FSAOption.RIPPLE_CARRY)

    dims = MMAcDims(dim_m=dim, dim_n=dim, dim_k=dim)
    widths = MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width)
    cfg = MMAcFusedCfg(dims=dims, widths=widths, mult_cfg=mult_cfg)

    core = build_matmul_accumulate(cfg)

    print(
        f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.Y[0,0].typ.width} bits"
    )

    sim = Simulator(core.module)

    rng = np.random.default_rng(seed=42)
    a_vals = rng.integers(0, 2**a_width, size=(dim, dim), dtype=int)
    b_vals = rng.integers(0, 2**b_width, size=(dim, dim), dtype=int)
    c_vals = rng.integers(0, 2**c_width, size=(dim, dim), dtype=int)

    for i in range(dim):
        for j in range(dim):
            sim.set(core.A[i, j], int(a_vals[i, j]))
            sim.set(core.B[i, j], int(b_vals[i, j]))
            sim.set(core.C[i, j], int(c_vals[i, j]))

    sim.eval()

    y_hw = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            y_hw[i, j] = sim.get(core.Y[i, j])

    y_np = a_vals @ b_vals + c_vals
    assert np.array_equal(y_hw, y_np), "Simulation mismatch for matmul accumulate core"
    print("Matmul accumulate simulation passed. Y=\n", y_hw)

    # get yosys transistor count
    yosys_metrics = get_yosys_metrics(core.module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_basic_simulation()
