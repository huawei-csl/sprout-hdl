# -----------------------------
# High-level aggregates (Bundle, Array, FixedPoint, ...)
# -----------------------------

from abc import ABC, abstractmethod
from typing import Type, TypeVar, Union

from sprouthdl.sprouthdl import Expr, ExprLike, as_expr, fit_width


T_Agg = TypeVar("T_Agg", bound="HDLAggregate")


class HDLAggregate(ABC):
    """
    Base class for structured HDL values (Bundle, Array, FixedPoint, ...).

    Requirements for subclasses:
      - to_bits(self) -> Expr
          Flatten the structure into a single bitvector Expr.
      - from_bits(cls, bits: Expr, *shape_args, **shape_kwargs) -> instance
          Reconstruct the same structure from a bitvector.
      - wire_like(cls, *shape_args, **shape_kwargs) -> instance
          Create a 'wire-filled' instance of this structure (all leaves are wires).
      - _assign_from_bits(self, bits: Expr)
          Drive the underlying leaf Signals from the given bits.
    """

    @abstractmethod
    def to_bits(self) -> Expr:
        """Flatten this aggregate into a single Expr bitvector."""
        ...

    @classmethod
    @abstractmethod
    def from_bits(
        cls: Type[T_Agg],
        bits: Expr,
        *args,
        **kwargs,
    ) -> T_Agg:
        """
        Rebuild an instance of this aggregate type from a flat bitvector.
        *args / **kwargs carry any shape metadata (e.g. length, frac_width).
        """
        ...

    @classmethod
    @abstractmethod
    def wire_like(
        cls: Type[T_Agg],
        *args,
        **kwargs,
    ) -> T_Agg:
        """
        Create a 'wire-filled' instance of this aggregate type.
        The arguments should match what from_bits() and your constructor need.
        """
        ...

    @abstractmethod
    def _assign_from_bits(self, bits: Expr) -> None:
        """
        Drive the underlying leaf Signals from the flat bitvector 'bits'.

        For example, a leaf backed by:
          - a combinational wire Signal:   sig <<= bits_slice
          - a register Signal (kind='reg'): sig.next = bits_slice
        Bundles/arrays chunk 'bits' and recurse into fields/elements.
        """
        ...

    # -------- Convenience API shared by all aggregates --------

    @property
    def width(self) -> int:
        """Total bit-width of this aggregate."""
        return self.to_bits().typ.width

    def _coerce_rhs_to_bits(self, rhs: Union["HDLAggregate", ExprLike]) -> Expr:
        """
        Convert rhs into a bitvector Expr with the same width as self.
        - HDLAggregate → rhs.to_bits()
        - Expr/int/bool → as_expr + fit_width(...)
        """
        lhs_bits = self.to_bits()
        t = lhs_bits.typ

        if isinstance(rhs, HDLAggregate):
            rhs_bits = rhs.to_bits()
        else:
            rhs_bits = fit_width(as_expr(rhs), t)

        if rhs_bits.typ.width != t.width:
            raise ValueError(f"Width mismatch in aggregate assignment: " f"lhs width={t.width}, rhs width={rhs_bits.typ.width}")
        return rhs_bits

    def assign(self, rhs: Union["HDLAggregate", ExprLike]) -> None:
        """
        Structural assignment: drive this aggregate from rhs.

        Example:
            my_bundle.assign(other_bundle)
            my_array.assign(0)
        """
        bits = self._coerce_rhs_to_bits(rhs)
        self._assign_from_bits(bits)

    def __imatmul__(self, rhs: Union["HDLAggregate", ExprLike]) -> "HDLAggregate":
        """
        Sugar:  agg @= rhs

        Mirrors Magma/Spinal semantics:
            my_bundle @= other
        """
        self.assign(rhs)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(width={self.width})"
