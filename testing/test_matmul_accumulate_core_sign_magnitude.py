from __future__ import annotations

import numpy as np

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import AdderConfig, MMAcCfg, MMAcDims, MMAcWidths, max_y_width_unsigned
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_sign_magnitude import (
    MultiplierConfig,
    SignMagnitudeEncoderConfig,
    build_matmul_accumulate,
)
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl_simulator import Simulator


def test_mmac_core_sign_magnitude_pipeline():
    dim = 4
    dim_k = dim
    a_width = 8
    b_width = 8
    c_width = max_y_width_unsigned(a_width, b_width, dim_k, include_carry_from_add=False)

    encoding = Encoding.twos_complement_symmetric

    # optional: disable encoders when using StageBasedSignMagnitudeToTwosComplementMultiplier
    encoding_cfg = SignMagnitudeEncoderConfig(
        encoder_clip_most_negative=False,
        decoder_clip_most_negative=False,
    )

    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_SIGN_MAGNITUDE_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(Encoding.sign_magnitude),
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        encoding_cfg=encoding_cfg,
    )
    add_cfg = AdderConfig(use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True, encoding=encoding)

    dims = MMAcDims(dim_m=dim, dim_n=dim, dim_k=dim_k)
    widths = MMAcWidths(a_width=a_width, b_width=b_width, c_width=c_width)
    cfg = MMAcCfg(dims=dims, widths=widths, mult_cfg=mult_cfg, add_cfg=add_cfg)

    core_build_out = build_matmul_accumulate(cfg, signed_io_type=True)

    print(
        f"Output matrix Y has shape: ({dim}, {dim}) with element width {core_build_out.Y[0,0].typ.width} bits"
    )

    sim = Simulator(core_build_out.module)

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
    assert np.array_equal(y_hw, y_np), "Simulation mismatch for matmul accumulate core with sign-magnitude wrappers"
    print("Matmul accumulate (sign-magnitude) simulation passed. Y=\n", y_hw)

    yosys_metrics = get_yosys_metrics(core_build_out.module)
    print(f"Yosys metrics: {yosys_metrics}")


if __name__ == "__main__":
    test_mmac_core_sign_magnitude_pipeline()
