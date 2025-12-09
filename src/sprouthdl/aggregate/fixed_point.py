from typing import Generic, Optional
from typing import Type, TypeVar, Union
from sprouthdl.aggregate.hdl_aggregate import HDLAggregate, T_Agg
from sprouthdl.sprouthdl import Expr, ExprLike, HDLType, Signal, Wire, as_expr, fit_width


class FixedPoint(HDLAggregate):
    """
    Simple fixed-point aggregate.

    Represents:
        raw bits (Expr) + metadata (total_width, frac_width, signed)

    Typical usage (owning wire):
        acc = FixedPoint(total_width=16, frac_width=8, signed=True, name="acc")
        acc <<= 0  # drive as usual

    View usage (reinterpret existing bits):
        slice_bits = some_bus[15:0]
        val = FixedPoint.from_bits(slice_bits, total_width=16, frac_width=8, signed=False)
    """

    def __init__(
        self,
        total_width: int,
        frac_width: int,
        signed: bool = True,
        name: Optional[str] = None,
        bits: Optional[ExprLike] = None,
    ):
        if total_width < 1:
            raise ValueError("total_width must be >= 1")
        if not (0 <= frac_width <= total_width):
            raise ValueError("frac_width must be in [0, total_width]")

        self.total_width = int(total_width)
        self.frac_width = int(frac_width)
        self.signed = bool(signed)

        self._typ = HDLType(self.total_width, signed=self.signed, is_bool=False)

        if bits is None:
            # Owning case: create a fresh wire of the right size
            sig = Wire(self._typ, name=name)
            self._bits: Expr = sig
        else:
            # View case: reinterpret existing bits as this fixed-point format
            bits_e = fit_width(as_expr(bits), self._typ)
            self._bits = bits_e

    # ---- Introspection ----

    @property
    def typ(self) -> HDLType:
        return self._typ

    @property
    def bits(self) -> Expr:
        """Underlying HDL Expr for the raw fixed-point bits."""
        return self._bits

    @property
    def int_width(self) -> int:
        """Number of integer bits (including sign if signed)."""
        return self.total_width - self.frac_width

    # ---- HDLAggregate API ----

    def to_bits(self) -> Expr:
        return self._bits

    @classmethod
    def from_bits(
        cls,
        bits: Expr,
        total_width: int,
        frac_width: int,
        signed: bool = True,
        name: Optional[str] = None,
    ) -> "FixedPoint":
        """
        Reinterpret 'bits' as a FixedPoint(total_width, frac_width, signed).
        """
        return cls(
            total_width=total_width,
            frac_width=frac_width,
            signed=signed,
            name=name,
            bits=bits,
        )

    @classmethod
    def wire_like(
        cls,
        total_width: int,
        frac_width: int,
        signed: bool = True,
        name: Optional[str] = None,
    ) -> "FixedPoint":
        """
        Create a wire-backed FixedPoint instance (owning case).

        Example:
            acc = FixedPoint.wire_like(16, 8, signed=True, name="acc")
        """
        return cls(
            total_width=total_width,
            frac_width=frac_width,
            signed=signed,
            name=name,
            bits=None,  # force own wire
        )

    def _assign_from_bits(self, bits: Expr) -> None:
        """
        Assign to the underlying leaf Signal (wire or reg).

        - If backed by a reg Signal → next-state assignment via <<=.
        - If backed by a wire Signal → combinational driver.
        - If backed by a non-Signal Expr → error on assignment.
        """
        target = self._bits
        if not isinstance(target, Signal):
            raise TypeError("FixedPoint assignment target must be backed by a Signal")
        if target._auto_generated:
            raise TypeError("Cannot assign to FixedPoint view backed by auto-generated shared wire")

        target <<= bits

    def __repr__(self) -> str:
        return f"FixedPoint(width={self.total_width}, " f"frac={self.frac_width}, signed={self.signed})"
