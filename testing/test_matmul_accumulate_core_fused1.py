from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Iterable, List, Literal

import numpy as np
from pyparsing import Optional

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, PPAOption, PPGOption
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import StageBasedMultiplierIO, TwoInputAritConfig
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl import Concat, Expr, Signal, UInt
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator


@dataclass
class MultiplierConfig:
    """Configuration for stage-based partial product flow."""

    ppg_opt: PPGOption
    ppa_opt: PPAOption
    fsa_opt: FSAOption
    optim_type: Literal["area", "speed"] = "area"


@dataclass(frozen=True)
class StageConfig(TwoInputAritConfig):

    output_width: Optional[int] = None
    
    @property
    def out_width(self) -> int:
        if self.output_width is None:
            raise ValueError("output_width must be specified for StageConfig")
        return self.output_width


def fused_inner_product(vec_a: Iterable[Expr], vec_b: Iterable[Expr], c_term: Expr, mult_cfg: MultiplierConfig) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")
    if len(a_list) == 0:
        raise ValueError("inner_product: no operands provided")

    a_width = a_list[0].typ.width
    b_width = b_list[0].typ.width
    if any(sig.typ.width != a_width for sig in a_list):
        raise ValueError("inner_product: inconsistent widths in vector A")
    if any(sig.typ.width != b_width for sig in b_list):
        raise ValueError("inner_product: inconsistent widths in vector B")

    product_width = a_width + b_width
    max_product_sum = len(a_list) * ((1 << product_width) - 1)
    max_c = (1 << c_term.typ.width) - 1
    result_width = max(product_width, (max_product_sum + max_c).bit_length())

    stage_cfg = StageConfig(
        a_width=a_width,
        b_width=b_width,
        output_width=result_width,
        optim_type=mult_cfg.optim_type,
    )

    ppg = mult_cfg.ppg_opt.value(stage_cfg)
    ppa = mult_cfg.ppa_opt.value(stage_cfg)
    fsa = mult_cfg.fsa_opt.value(stage_cfg)

    merged_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
    for idx, (a_sig, b_sig) in enumerate(zip(a_list, b_list)):
        io = StageBasedMultiplierIO(
            a=a_sig,
            b=b_sig,
            y=None #Signal(name=f"pp_{idx}", typ=UInt(result_width), kind="wire"),
        )
        cols = ppg.generate_columns(io)
        for weight, bits in cols.items():
            if weight < result_width:
                merged_cols[weight].extend(bits)

    for bit_idx in range(min(c_term.typ.width, result_width)):
        merged_cols[bit_idx].append(c_term[bit_idx])

    reduced_cols = ppa.accumulate(merged_cols)
    filtered_cols = {w: bits for w, bits in reduced_cols.items() if w < result_width}
    result_bits = fsa.resolve(filtered_cols)
    return Concat(result_bits[:result_width])


@dataclass
class MatmulAccumulateCore:
    module: Module
    A: Array
    B: Array
    C: Array
    Y: Array


def build_matmul_accumulate_core(
    dim: int,
    a_width: int,
    b_width: int,
    c_width: int,
    mult_cfg: MultiplierConfig,
) -> MatmulAccumulateCore:
    if dim <= 0 or dim & (dim - 1) != 0:
        raise ValueError("Matrix dimension must be a power of two")

    m = Module("matmul_accumulate_core")

    # not for now, but in the future we would like a module, e.g. with IOs. But to convert to a module we need basic hdl types (signals with kind input and output)
    # maybe have an inner componenet with arrays as input and output (need input output kind for aggregate hdl type?),
    # then wrap an outer comoponent or module with signals as input and output that connect to the inner component
    # io could be defined like this:
    # class MyRecord(AggregateRecord):
    #    a: Array = self.A
    #    b: Array = self.B
    #    c: Array = self.C
    #    y: Array = self.Y

    # self.io = MyRecord()

    def build_matrix(name: str, width: int) -> Array:
        return Array([Array([m.input(UInt(width), f"{name}_{i}_{j}") for j in range(dim)]) for i in range(dim)])

    A = build_matrix("a", a_width)
    B = build_matrix("b", b_width)
    C = build_matrix("c", c_width)

    rows = []
    for i in range(dim):
        row = []
        a_row = A[i, :]
        for j in range(dim):
            b_col = B[:, j]
            dot = fused_inner_product(a_row, b_col, C[i, j], mult_cfg)
            y_sig = m.output(dot.typ, f"y_{i}_{j}")
            y_sig <<= dot
            row.append(y_sig)
        rows.append(Array(row))
    Y = Array(rows)

    return MatmulAccumulateCore(module=m, A=A, B=B, C=C, Y=Y)


def test_mmac_core_basic_simulation():
    dim = 4
    a_width = 8
    b_width = 8
    c_width = 20

    mult_cfg = MultiplierConfig(ppg_opt=PPGOption.AND, ppa_opt=PPAOption.WALLACE_TREE, fsa_opt=FSAOption.RIPPLE_CARRY)

    core = build_matmul_accumulate_core(dim, a_width, b_width, c_width, mult_cfg)

    print(f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.Y[0,0].typ.width} bits")

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
