from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, NamedTuple, Sequence

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.aggregate.aggregate_record import AggregateRecord
from sprouthdl.aggregate.aggregate_record_dynamic import AggregateRecordDynamic
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from sprouthdl.sprouthdl import Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Component, Module


@dataclass
class MultiplierConfig:
    """Configuration for choosing between Sprout operator and explicit multiplier."""

    use_operator: bool = False
    multiplier_opt: MultiplierOption | None = None
    encodings: TwoInputAritEncodings | None = None
    ppg_opt: PPGOption | None = None
    ppa_opt: PPAOption | None = None
    fsa_opt: FSAOption | None = None
    optim_type: Literal["area", "speed"] = "area"


@dataclass
class AdderConfig:
    """Configuration for choosing between Sprout operator and explicit adder."""

    use_operator: bool = False
    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption | None = None
    full_output_bit: bool = True


def build_multiplier(a: Expr, b: Expr, mult_cfg: MultiplierConfig) -> Expr:
    if mult_cfg.use_operator:
        return a * b

    assert mult_cfg.multiplier_opt is not None, "multiplier_opt must be provided for explicit multipliers"
    assert mult_cfg.encodings is not None, "encodings must be provided for explicit multipliers"
    assert mult_cfg.ppg_opt is not None and mult_cfg.ppa_opt is not None and mult_cfg.fsa_opt is not None

    multiplier = mult_cfg.multiplier_opt.value(
        a_w=a.typ.width,
        b_w=b.typ.width,
        a_encoding=mult_cfg.encodings.a,
        b_encoding=mult_cfg.encodings.b,
        ppg_cls=mult_cfg.ppg_opt.value,
        ppa_cls=mult_cfg.ppa_opt.value,
        fsa_cls=mult_cfg.fsa_opt.value,
        optim_type=mult_cfg.optim_type,
    ).make_internal()
    multiplier.io.a <<= a
    multiplier.io.b <<= b
    return multiplier.io.y


def build_adder(a: Expr, b: Expr, adder_cfg: AdderConfig) -> Expr:
    if adder_cfg.use_operator:
        return a + b

    assert adder_cfg.fsa_opt is not None, "fsa_opt must be provided for explicit adders"
    signed = is_signed(adder_cfg.encoding)

    adder = StageBasedPrefixAdder(
        a_w=a.typ.width,
        b_w=b.typ.width,
        signed_a=signed,
        signed_b=signed,
        optim_type=adder_cfg.optim_type,
        fsa_cls=adder_cfg.fsa_opt.value,
        full_output_bit=adder_cfg.full_output_bit,
    ).make_internal()
    adder.io.a <<= a
    adder.io.b <<= b
    return adder.io.y


def adder_tree(values: Sequence[Expr], adder_cfg: AdderConfig) -> Expr:
    if len(values) == 0:
        raise ValueError("Adder tree requires at least one value")
    if len(values) == 1:
        return values[0]

    mid = len(values) // 2
    left = adder_tree(values[:mid], adder_cfg)
    right = adder_tree(values[mid:], adder_cfg)
    return build_adder(left, right, adder_cfg)


def inner_product(
    vec_a: Iterable[Expr], vec_b: Iterable[Expr], mult_cfg: MultiplierConfig, add_cfg: AdderConfig
) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")

    products = [build_multiplier(a, b, mult_cfg) for a, b in zip(a_list, b_list)]
    return adder_tree(products, add_cfg)


@dataclass
class MatmulAccumulateIO(AggregateRecordDynamic):
    A: Array  # input
    B: Array  # input
    C: Array  # input
    Y: Array  # output

@dataclass
class MMAcDims:
    dim_m: int  # rows of A/C/Y
    dim_n: int  # cols of B/C/Y
    dim_k: int  # shared dimension between A and B

@dataclass
class MMAcWidths:
    a_width: int
    b_width: int
    c_width: int

@dataclass
class MMAcCfg:
    dims: MMAcDims
    widths: MMAcWidths
    mult_cfg: MultiplierConfig
    add_cfg: AdderConfig

class MatmulAccumulateCore(Component):
    io: MatmulAccumulateIO

class MatmulAccumulateComponent(MatmulAccumulateCore):
    """Reusable component for matrix multiply-accumulate."""

    def __init__(
        self,
        cfg: MMAcCfg,
        signed_io_type: bool = False,
    ):

        self.cfg = cfg
        self.io_hdl_type = SInt if (is_signed(self.cfg.add_cfg.encoding) and signed_io_type) else UInt

        def build_matrix(name: str, width: int, rows: int, cols: int, kind: str = "wire") -> Array:
            return Array(
                [
                    Array(
                        [
                            Signal(name=f"{name}_{i}_{j}", typ=self.io_hdl_type(width), kind=kind)
                            for j in range(cols)
                        ]
                    )
                    for i in range(rows)
                ]
            )

        self.A = build_matrix("a", self.cfg.widths.a_width, self.cfg.dims.dim_m, self.cfg.dims.dim_k, "input")
        self.B = build_matrix("b", self.cfg.widths.b_width, self.cfg.dims.dim_k, self.cfg.dims.dim_n, "input")
        self.C = build_matrix("c", self.cfg.widths.c_width, self.cfg.dims.dim_m, self.cfg.dims.dim_n, "input")

        self.elaborate()

        self.io: MatmulAccumulateIO = MatmulAccumulateIO(A=self.A, B=self.B, C=self.C, Y=self.Y)

    def elaborate(self):
        rows = []
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = self.A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = self.B[:, j]
                dot = inner_product(a_row, b_col, self.cfg.mult_cfg, self.cfg.add_cfg)
                acc = build_adder(self.C[i, j], dot, self.cfg.add_cfg)
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(acc.typ.width), kind="output")
                y_sig <<= acc
                row.append(y_sig)
            rows.append(Array(row))
        self.Y = Array(rows)


# class MatmulAccuumulateComponentWrapper(Component):
#     """Wrapper component that has only basic signals as IO."""

#     def __init__(self, cfg: MMAcCfg, signed_io_type: bool = False):
#         self.cfg = cfg
#         self.signed_io_type = signed_io_type
#         self.elaborate()

#     def elaborate(self):
#         self.comp = MatmulAccumulateComponent(self.cfg, signed_io_type=self.signed_io_type)

#         io_list: List[Signal] = self.comp.io.A.to_list() + self.comp.io.B.to_list() + self.comp.io.C.to_list() + self.comp.io.Y.to_list()
#         io_dict = {sig.name: sig for sig in io_list}
#         self.io = namedtuple('MatmulAccumulateIOWrapper', io_dict.keys())(**io_dict)


@dataclass
class MatmulAccumulateBuildOut:
    component: MatmulAccumulateComponent
    module: Module
    A: Array
    B: Array
    C: Array
    Y: Array


def build_matmul_accumulate(
    cfg: MMAcCfg,
    signed_io_type: bool = False,
) -> MatmulAccumulateBuildOut:
    
    # component = MatmulAccumulateComponent(cfg, signed_io_type=signed_io_type)

    # def build_wrapper_module(name: str, wrapped: MatmulAccumulateComponent) -> tuple[Module, Array, Array, Array, Array]:
        
    #     comp = MatmulAccuumulateComponentWrapper(cfg, signed_io_type=signed_io_type)
    #     m = comp.to_module(name)
    #     A_ports, B_ports, C_ports, Y_ports = comp.comp.io.A, comp.comp.io.B, comp.comp.io.C, comp.comp.io.Y

    #     return m, A_ports, B_ports, C_ports, Y_ports

    # component_module, A, B, C, Y = build_wrapper_module("matmul_accumulate_core", component)
    
    component = MatmulAccumulateComponent(cfg, signed_io_type=signed_io_type)
    component_module = component.to_module("matmul_accumulate_core")
    A, B, C, Y = component.io.A, component.io.B, component.io.C, component.io.Y

    return MatmulAccumulateBuildOut(
        component=component,
        module=component_module,
        A=A,
        B=B,
        C=C,
        Y=Y,
    )


def ceil_log2(n: int) -> int:
    if n <= 1:
        return 0
    return (n - 1).bit_length()


def max_y_width_unsigned(
    a_width: int, b_width: int, dim_k: int, *, include_carry_from_add: bool = True
) -> int:
    carry = 1 if include_carry_from_add else 0
    return a_width + b_width + ceil_log2(dim_k) + carry
