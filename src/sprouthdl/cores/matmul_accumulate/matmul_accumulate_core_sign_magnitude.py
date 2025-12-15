from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Literal, Sequence, Type

import numpy as np

from sprouthdl.aggregate.aggregate_array import Array
from sprouthdl.arithmetic.encoding.sign_magnitude import (
    SignMagnitudeToTwosComplementDecoder,
    TwosComplementToSignMagnitudeEncoder,
)
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import (
    Encoding,
    EncodingModel,
    is_signed,
)
from sprouthdl.arithmetic.prefix_adders.adders import StageBasedPrefixAdder
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl import Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Component, Module
from sprouthdl.sprouthdl_simulator import Simulator
from testing.test_matmul_accumulate_core import max_y_width_unsigned


@dataclass
class SignMagnitudeEncoderConfig:
    """Configuration for optional sign-magnitude conversion wrappers."""

    encoder_cls: Type[Component] | None = TwosComplementToSignMagnitudeEncoder
    decoder_cls: Type[Component] | None = SignMagnitudeToTwosComplementDecoder
    encoder_clip_most_negative: bool = False
    decoder_clip_most_negative: bool = False


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
    encoding_cfg: SignMagnitudeEncoderConfig | None = field(default_factory=SignMagnitudeEncoderConfig)

    def build(self, a: Expr, b: Expr) -> Expr:
        if self.use_operator:
            return a * b

        assert self.encodings is not None, "encodings must be provided for explicit multipliers"
        assert self.ppg_opt is not None and self.ppa_opt is not None and self.fsa_opt is not None

        multiplier = self.multiplier_opt.value(
            a_w=a.typ.width,
            b_w=b.typ.width,
            a_encoding=self.encodings.a,
            b_encoding=self.encodings.b,
            ppg_cls=self.ppg_opt.value,
            ppa_cls=self.ppa_opt.value,
            fsa_cls=self.fsa_opt.value,
            optim_type=self.optim_type,
        ).make_internal()
        multiplier.io.a <<= a
        multiplier.io.b <<= b
        y_expr: Expr = multiplier.io.y

        enc_cfg = self.encoding_cfg
        if enc_cfg and enc_cfg.decoder_cls is not None:
            decoder = enc_cfg.decoder_cls(
                width=y_expr.typ.width, clip_most_negative=enc_cfg.decoder_clip_most_negative
            ).make_internal()
            decoder.io.i <<= y_expr
            return decoder.io.o

        return y_expr


@dataclass
class AdderConfig:
    """Configuration for choosing between Sprout operator and explicit adder."""

    use_operator: bool = False
    encoding: Encoding = Encoding.unsigned
    optim_type: Literal["area", "speed"] = "area"
    fsa_opt: FSAOption | None = None
    full_output_bit: bool = True

    def build(self, a: Expr, b: Expr) -> Expr:
        if self.use_operator:
            return a + b

        signed = is_signed(self.encoding)

        adder = StageBasedPrefixAdder(
            a_w=a.typ.width,
            b_w=b.typ.width,
            signed_a=signed,
            signed_b=signed,
            optim_type=self.optim_type,
            fsa_cls=self.fsa_opt.value,
            full_output_bit=self.full_output_bit,
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
    return adder_cfg.build(left, right)


def inner_product(
    vec_a: Iterable[Expr], vec_b: Iterable[Expr], mult_cfg: MultiplierConfig, add_cfg: AdderConfig
) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")

    products = [mult_cfg.build(a, b) for a, b in zip(a_list, b_list)]
    return adder_tree(products, add_cfg)


@dataclass
class MatmulAccumulateIO:
    A: Array  # input
    B: Array  # input
    C: Array  # input
    Y: Array  # output


class MatmulAccumulateComponent(Component):
    """Reusable component for matrix multiply-accumulate."""

    def __init__(
        self,
        dim: int,
        a_width: int,
        b_width: int,
        c_width: int,
        mult_cfg: MultiplierConfig,
        add_cfg: AdderConfig,
    ):

        self.dim = dim
        self.a_width = a_width
        self.b_width = b_width
        self.c_width = c_width
        self.mult_cfg = mult_cfg
        self.add_cfg = add_cfg

        self.io_hdl_type = SInt if is_signed(add_cfg.encoding) else UInt

        def build_matrix(name: str, width: int, kind: str = "wire") -> Array:
            return Array(
                [
                    Array(
                        [
                            Signal(name=f"{name}_{i}_{j}", typ=self.io_hdl_type(width), kind=kind)
                            for j in range(dim)
                        ]
                    )
                    for i in range(dim)
                ]
            )

        self.A = build_matrix("a", a_width)
        self.B = build_matrix("b", b_width)
        self.C = build_matrix("c", c_width)

        self.elaborate()

        self.io = MatmulAccumulateIO(A=self.A, B=self.B, C=self.C, Y=self.Y)

    def _encode_matrix(self, matrix: Array) -> Array:
        enc_cfg = self.mult_cfg.encoding_cfg
        if enc_cfg is None or enc_cfg.encoder_cls is None:
            return matrix

        encoded_rows = []
        for i in range(self.dim):
            encoded_row = []
            for j in range(self.dim):
                encoder = enc_cfg.encoder_cls(
                    width=matrix[i, j].typ.width,
                    clip_most_negative=enc_cfg.encoder_clip_most_negative,
                ).make_internal()
                encoder.io.i <<= matrix[i, j]
                encoded_row.append(encoder.io.o)
            encoded_rows.append(Array(encoded_row))

        return Array(encoded_rows)

    def elaborate(self):
        encoded_A = self._encode_matrix(self.A)
        encoded_B = self._encode_matrix(self.B)

        rows = []
        for i in range(self.dim):
            row = []
            a_row = encoded_A[i, :]
            for j in range(self.dim):
                b_col = encoded_B[:, j]
                dot = inner_product(a_row, b_col, self.mult_cfg, self.add_cfg)
                acc = self.add_cfg.build(self.C[i, j], dot)
                y_sig = Signal(name=f"y_{i}_{j}", typ=self.io_hdl_type(acc.typ.width), kind="output")
                y_sig <<= acc
                row.append(y_sig)
            rows.append(Array(row))
        self.Y = Array(rows)


@dataclass
class MatmulAccumulateBuildOut:
    component: MatmulAccumulateComponent
    module: Module
    A: Array
    B: Array
    C: Array
    Y: Array


def build_matmul_accumulate(
    dim: int,
    a_width: int,
    b_width: int,
    c_width: int,
    mult_cfg: MultiplierConfig,
    add_cfg: AdderConfig,
) -> MatmulAccumulateBuildOut:
    component = MatmulAccumulateComponent(dim, a_width, b_width, c_width, mult_cfg, add_cfg)

    def build_wrapper_module(name: str, wrapped: MatmulAccumulateComponent) -> tuple[Module, Array, Array, Array, Array]:
        m = Module(name)

        def make_io_matrix(template: Array, register_io_func: Callable[[Signal], None]) -> Array:
            ports = Array.wire_like(template)
            for i in range(dim):
                for j in range(dim):
                    register_io_func(ports[i, j])
            return ports

        # gen new matrices with ports and connect to wrapped component
        A_ports = make_io_matrix(wrapped.A, m.add_input)
        B_ports = make_io_matrix(wrapped.B, m.add_input)
        C_ports = make_io_matrix(wrapped.C, m.add_input)
        Y_ports = make_io_matrix(wrapped.Y, m.add_output)
        wrapped.A <<= A_ports
        wrapped.B <<= B_ports
        wrapped.C <<= C_ports
        Y_ports <<= wrapped.Y

        return m, A_ports, B_ports, C_ports, Y_ports

    component_module, A, B, C, Y = build_wrapper_module("matmul_accumulate_core_sign_mag", component)

    return MatmulAccumulateBuildOut(
        component=component,
        module=component_module,
        A=A,
        B=B,
        C=C,
        Y=Y,
    )


