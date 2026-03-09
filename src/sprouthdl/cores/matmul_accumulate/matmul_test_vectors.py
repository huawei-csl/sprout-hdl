from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, EncodingModel
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import MatmulAccumulateCore


def _shape2d(arr: Array) -> Tuple[int, int]:
    return len(arr), len(arr[0])


def generate_matmul_vectors(
    core: MatmulAccumulateCore,
    encoding: Encoding,
    num_vectors: int,
    sigma: Optional[float] = None,
    encoding_ab_inputs: Optional[Encoding] = None,
) -> List[Tuple[str, Dict[str, int], Dict[str, int]]]:
    """Build flat-dict test vectors for a matmul-accumulate core.

    Args:
        core: Component whose io.A/B/C/Y ports define the shapes and widths.
        encoding: Encoding used for C and Y (and A/B when encoding_ab_inputs is None).
        num_vectors: Number of random test vectors to generate.
        sigma: If given, use normal sampling with this sigma; otherwise uniform.
        encoding_ab_inputs: Override encoding for A and B (e.g. for mixed-precision).
    """
    a_width = core.io.A[0, 0].typ.width
    b_width = core.io.B[0, 0].typ.width
    c_width = core.io.C[0, 0].typ.width
    y_width = core.io.Y[0, 0].typ.width

    a_rows, a_cols = _shape2d(core.io.A)  # A: (m, k)
    b_rows, b_cols = _shape2d(core.io.B)  # B: (k, n)
    c_rows, c_cols = _shape2d(core.io.C)  # C/Y: (m, n)

    enc_model = EncodingModel(encoding)
    enc_model_inputs = EncodingModel(encoding_ab_inputs) if encoding_ab_inputs is not None else enc_model

    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

    for idx in range(num_vectors):
        if sigma is None:
            a_vals = enc_model_inputs.get_uniform_sample_np(a_width, size=(a_rows, a_cols))
            b_vals = enc_model_inputs.get_uniform_sample_np(b_width, size=(b_rows, b_cols))
            c_vals = enc_model.get_uniform_sample_np(c_width, size=(c_rows, c_cols))
        else:
            a_vals = enc_model_inputs.get_normal_sample_np(a_width, sigma, size=(a_rows, a_cols))
            b_vals = enc_model_inputs.get_normal_sample_np(b_width, sigma, size=(b_rows, b_cols))
            c_vals = enc_model.get_normal_sample_np(c_width, sigma, size=(c_rows, c_cols))

        y_vals = a_vals @ b_vals + c_vals

        a_vals = np.vectorize(lambda x: enc_model_inputs.encode_value(int(x), a_width))(a_vals)
        b_vals = np.vectorize(lambda x: enc_model_inputs.encode_value(int(x), b_width))(b_vals)
        c_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), c_width))(c_vals)
        y_vals = np.vectorize(lambda x: enc_model.encode_value(int(x), y_width))(y_vals)

        ins: Dict[str, int] = {}
        outs: Dict[str, int] = {}

        for i in range(a_rows):
            for k in range(a_cols):
                ins[core.io.A[i, k].name] = int(a_vals[i, k])

        for k in range(b_rows):
            for j in range(b_cols):
                ins[core.io.B[k, j].name] = int(b_vals[k, j])

        for i in range(c_rows):
            for j in range(c_cols):
                ins[core.io.C[i, j].name] = int(c_vals[i, j])
                outs[core.io.Y[i, j].name] = int(y_vals[i, j])

        vectors.append((f"vec_{idx}", ins, outs))

    return vectors
