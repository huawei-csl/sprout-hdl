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


