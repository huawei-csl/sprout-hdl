from __future__ import annotations

import os
import sys

import numpy as np

from sprouthdl.helpers import get_yosys_metrics

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.aggregate.aggregate_floating_point import FloatingPoint, FloatingPointType
from sprouthdl.sprouthdl import UInt, as_expr
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator
from testing.floating_point.fp_testvectors_general import fp_decode, fp_encode


def build_fp_matmul_accumulate(dim: int, ft: FloatingPointType) -> tuple[Module, Array, Array, Array, Array]:
    m = Module("fp_matmul_accumulate", with_clock=False, with_reset=False)

    def build_io_matrix(prefix: str) -> Array:
        rows = []
        for i in range(dim):
            cols = []
            for j in range(dim):
                bits = m.input(UInt(ft.width_total), f"{prefix}_{i}_{j}")
                cols.append(FloatingPoint(ft, bits=as_expr(bits)))
            rows.append(Array(cols))
        return Array(rows)

    A = build_io_matrix("a")
    B = build_io_matrix("b")
    C = build_io_matrix("c")

    y_rows = []
    for i in range(dim):
        y_cols = []
        for j in range(dim):
            dot = None
            for k in range(dim):
                prod = A[i, k] * B[k, j]
                dot = prod if dot is None else dot + prod
            acc = dot + C[i, j]

            y_bits = m.output(UInt(ft.width_total), f"y_{i}_{j}")
            y_bits <<= acc.bits
            y_cols.append(FloatingPoint(ft, bits=as_expr(y_bits)))
        y_rows.append(Array(y_cols))
    Y = Array(y_rows)

    return m, A, B, C, Y


def _encode_matrix(values: np.ndarray, ft: FloatingPointType) -> np.ndarray:
    encode = np.vectorize(lambda x: fp_encode(float(np.float16(x)), ft.exponent_width, ft.fraction_width))
    return encode(values)


def _decode_matrix(bits: np.ndarray, ft: FloatingPointType) -> np.ndarray:
    decode = np.vectorize(lambda x: fp_decode(int(x), ft.exponent_width, ft.fraction_width))
    return decode(bits)


def _float16_matmul_accumulate(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    dim = a.shape[0]
    y = np.empty((dim, dim), dtype=np.float16)
    for i in range(dim):
        for j in range(dim):
            acc = np.float16(c[i, j])
            for k in range(dim):
                acc = np.float16(acc + np.float16(a[i, k] * b[k, j]))
            y[i, j] = acc
    return y


def test_fp_matmul_accumulate_simulation():
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=10)
    module, A, B, C, Y = build_fp_matmul_accumulate(dim, ft)

    sim = Simulator(module)

    # a_vals = np.array([[1.0, 2.0], [0.5, 1.5]], dtype=np.float16)
    # b_vals = np.array([[2.0, 0.25], [1.0, 3.0]], dtype=np.float16)
    # c_vals = np.array([[0.25, 0.5], [1.0, 0.75]], dtype=np.float16)

    rng = np.random.default_rng(seed=123)
    a_vals = rng.uniform(0.1, 2.0, size=(dim, dim))
    b_vals = rng.uniform(0.1, 2.0, size=(dim, dim))
    c_vals = rng.uniform(0.1, 2.0, size=(dim, dim))

    a_bits = _encode_matrix(a_vals, ft)
    b_bits = _encode_matrix(b_vals, ft)
    c_bits = _encode_matrix(c_vals, ft)

    for i in range(dim):
        for j in range(dim):
            sim.set(A[i, j].bits, int(a_bits[i, j]))
            sim.set(B[i, j].bits, int(b_bits[i, j]))
            sim.set(C[i, j].bits, int(c_bits[i, j]))

    sim.eval()

    y_hw_bits = np.zeros((dim, dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            y_hw_bits[i, j] = sim.get(Y[i, j].bits)

    y_hw = _decode_matrix(y_hw_bits, ft)
    y_expected = _float16_matmul_accumulate(a_vals, b_vals, c_vals)

    assert np.isclose(y_hw, y_expected, rtol=1e-2, atol=1e-2
                      ).all(), f"Expected {y_expected} but got {y_hw}"

    # get yosys transistor count
    # yosys_metrics = get_yosys_metrics(module)
    # print(f"Yosys metrics: {yosys_metrics}")

if __name__ == "__main__":
    test_fp_matmul_accumulate_simulation()
