from dataclasses import dataclass
from typing import Optional, Union

from sprouthdl.aggregate.hdl_aggregate import HDLAggregate
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import FpMul
from sprouthdl.arithmetic.floating_point.sprout_hdl_float_sn import FpMulSN
from sprouthdl.arithmetic.floating_point.sprout_hdl_float_add import FpAdd
from sprouthdl.sprouthdl import Expr, ExprLike, HDLType, Signal, UInt, Wire, as_expr, fit_width


@dataclass(frozen=True)
class FloatingPointType:
    """Floating-point format description.

    exponent_width : number of exponent bits
    fraction_width : number of fraction bits (excluding hidden bit)
    """

    exponent_width: int
    fraction_width: int

    def __post_init__(self) -> None:
        if self.exponent_width < 1:
            raise ValueError("FloatingPointType.exponent_width must be >= 1")
        if self.fraction_width < 0:
            raise ValueError("FloatingPointType.fraction_width must be >= 0")

    @property
    def width_total(self) -> int:
        return 1 + self.exponent_width + self.fraction_width

    def to_hdl_type(self) -> HDLType:
        return UInt(self.width_total)


class FloatingPoint(HDLAggregate):
    """Flat floating-point aggregate backed by a single Expr/Signal."""

    def __init__(
        self,
        ftype: FloatingPointType,
        name: Optional[str] = None,
        bits: Optional[ExprLike] = None,
        sn_support: bool = False,
    ):
        self.ftype = ftype
        self._typ = ftype.to_hdl_type()

        if bits is None:
            sig = Wire(self._typ, name=name)
            self._bits: Expr = sig
        else:
            bits_e = fit_width(as_expr(bits), self._typ)
            self._bits = bits_e
        self.sn_support = sn_support

    # ---- Introspection ----
    @property
    def bits(self) -> Expr:
        return self._bits

    @property
    def typ(self) -> HDLType:
        return self._typ

    @property
    def width(self) -> int:
        return self.ftype.width_total

    @property
    def exponent(self) -> Expr:
        lo = self.ftype.fraction_width
        hi = lo + self.ftype.exponent_width
        return self._bits[lo:hi]

    @property
    def fraction(self) -> Expr:
        return self._bits[0 : self.ftype.fraction_width]

    @property
    def sign(self) -> Expr:
        return self._bits[self.width - 1]

    # ---- HDLAggregate API ----
    def to_bits(self) -> Expr:
        return self._bits

    @classmethod
    def wire_like(
        cls,
        arg: Union["FloatingPoint", FloatingPointType],
        name: Optional[str] = None,
    ) -> "FloatingPoint":
        if isinstance(arg, FloatingPoint):
            ftype = arg.ftype
        elif isinstance(arg, FloatingPointType):
            ftype = arg
        else:
            raise TypeError("FloatingPoint.wire_like expects FloatingPoint or FloatingPointType, " f"got {type(arg)}")
        return cls(ftype, name=name, bits=None)

    def _assign_from_bits(self, bits: Expr) -> None:
        target = self._bits
        if not isinstance(target, Signal):
            raise TypeError("FloatingPoint assignment target must be backed by a Signal")

        target <<= bits

    # ---------------------------------
    # Floating-point add helper
    # ---------------------------------
    def add(self, other: "FloatingPoint") -> "FloatingPoint":
        if not isinstance(other, FloatingPoint):
            raise TypeError(f"Expected FloatingPoint, got {type(other)}")
        if self.ftype != other.ftype:
            raise ValueError("FloatingPoint add requires matching types")

        core = FpAdd(EW=self.ftype.exponent_width, FW=self.ftype.fraction_width).make_internal()

        core.io.a <<= self.bits
        core.io.b <<= other.bits

        return FloatingPoint(self.ftype, bits=core.io.y)

    def __add__(self, other: "FloatingPoint") -> "FloatingPoint":
        return self.add(other)

    # ---------------------------------
    # Floating-point multiply helper
    # ---------------------------------
    def mul(self, other: "FloatingPoint") -> "FloatingPoint":
        if not isinstance(other, FloatingPoint):
            raise TypeError(f"Expected FloatingPoint, got {type(other)}")
        if self.ftype != other.ftype:
            raise ValueError("FloatingPoint multiply requires matching types")
        
        fp_cls = FpMulSN if self.sn_support else FpMul

        core = fp_cls(EW=self.ftype.exponent_width, FW=self.ftype.fraction_width).make_internal()

        core.io.a <<= self.bits
        core.io.b <<= other.bits

        return FloatingPoint(self.ftype, bits=core.io.y)

    def __mul__(self, other: "FloatingPoint") -> "FloatingPoint":
        return self.mul(other)

    def __repr__(self) -> str:
        t = self.ftype
        return f"FloatingPoint(exp={t.exponent_width}, frac={t.fraction_width})"
