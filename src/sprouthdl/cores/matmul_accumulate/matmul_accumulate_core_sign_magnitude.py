from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Type

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
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core import (
    AdderConfig,
    MatmulAccumulateIO,
    MMAcCfg,
    MMAcDims,
    MMAcWidths,
    MultiplierConfig as BaseMultiplierConfig,
    adder_tree,
    build_adder,
)
from sprouthdl.sprouthdl import Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Component, Module


@dataclass
class SignMagnitudeEncoderConfig:
    """Configuration for optional sign-magnitude conversion wrappers."""

    encoder_cls: Type[Component] | None = TwosComplementToSignMagnitudeEncoder
    decoder_cls: Type[Component] | None = SignMagnitudeToTwosComplementDecoder
    encoder_clip_most_negative: bool = False
    decoder_clip_most_negative: bool = False


@dataclass
class MultiplierConfig(BaseMultiplierConfig):
    """Configuration for choosing between Sprout operator and explicit multiplier."""

    encoding_cfg: SignMagnitudeEncoderConfig | None = field(default_factory=SignMagnitudeEncoderConfig)


def build_multiplier(a: Expr, b: Expr, mult_cfg: MultiplierConfig) -> Expr:
    if mult_cfg.use_operator:
        return a * b

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
    y_expr: Expr = multiplier.io.y

    enc_cfg = mult_cfg.encoding_cfg
    if enc_cfg and enc_cfg.decoder_cls is not None:
        decoder = enc_cfg.decoder_cls(
            width=y_expr.typ.width, clip_most_negative=enc_cfg.decoder_clip_most_negative
        ).make_internal()
        decoder.io.i <<= y_expr
        return decoder.io.o

    return y_expr


def inner_product(
    vec_a: Iterable[Expr], vec_b: Iterable[Expr], mult_cfg: MultiplierConfig, add_cfg: AdderConfig
) -> Expr:
    a_list: List[Expr] = list(vec_a)
    b_list: List[Expr] = list(vec_b)
    if len(a_list) != len(b_list):
        raise ValueError("inner_product: length mismatch")

    products = [build_multiplier(a, b, mult_cfg) for a, b in zip(a_list, b_list)]
    return adder_tree(products, add_cfg)


class MatmulAccumulateComponent(Component):
    """Reusable component for matrix multiply-accumulate."""

    def __init__(
        self,
        cfg: MMAcCfg,
        signed_io_type: bool = False,
    ):

        self.cfg = cfg
        self.mult_cfg = cfg.mult_cfg
        self.add_cfg = cfg.add_cfg
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

        self.A = build_matrix("a", self.cfg.widths.a_width, self.cfg.dims.dim_m, self.cfg.dims.dim_k)
        self.B = build_matrix("b", self.cfg.widths.b_width, self.cfg.dims.dim_k, self.cfg.dims.dim_n)
        self.C = build_matrix("c", self.cfg.widths.c_width, self.cfg.dims.dim_m, self.cfg.dims.dim_n)

        self.elaborate()

        self.io = MatmulAccumulateIO(A=self.A, B=self.B, C=self.C, Y=self.Y)

    def _encode_matrix(self, matrix: Array) -> Array:
        enc_cfg = self.mult_cfg.encoding_cfg
        if enc_cfg is None or enc_cfg.encoder_cls is None:
            return matrix

        encoded_rows = []
        rows = len(matrix)
        cols = len(matrix[0])
        for i in range(rows):
            encoded_row = []
            for j in range(cols):
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
        for i in range(self.cfg.dims.dim_m):
            row = []
            a_row = encoded_A[i, :]
            for j in range(self.cfg.dims.dim_n):
                b_col = encoded_B[:, j]
                dot = inner_product(a_row, b_col, self.mult_cfg, self.add_cfg)
                acc = build_adder(self.C[i, j], dot, self.add_cfg)
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
    cfg: MMAcCfg,
    signed_io_type: bool = False,
) -> MatmulAccumulateBuildOut:
    component = MatmulAccumulateComponent(cfg, signed_io_type=signed_io_type)

    def build_wrapper_module(name: str, wrapped: MatmulAccumulateComponent) -> tuple[Module, Array, Array, Array, Array]:
        m = Module(name)

        def make_io_matrix(template: Array, register_io_func: Callable[[Signal], None]) -> Array:
            ports = Array.wire_like(template)
            rows = len(template)
            cols = len(template[0])
            for i in range(rows):
                for j in range(cols):
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
