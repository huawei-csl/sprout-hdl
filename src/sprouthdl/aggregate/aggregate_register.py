from typing import Generic, Optional
from typing import Type, TypeVar, Union
from sprouthdl.aggregate.hdl_aggregate import HDLAggregate, T_Agg
from sprouthdl.sprouthdl import Expr, ExprLike, Signal, as_expr, fit_width


class AggregateRegister(HDLAggregate, Generic[T_Agg]):
    """
    Generic register that stores a packed HDLAggregate value.

    Usage:
        # Register that holds a FixedPoint(16, frac=8)
        acc = AggregateRegister(FixedPoint, total_width=16, frac_width=8, signed=True, name="acc_reg")

        # Read as a FixedPoint:
        acc_val = acc.value

        # Assign packed:
        acc @= acc_val  # or acc @= some_other_fixed

    Internally:
      - A single Signal(kind='reg') bitvector is allocated.
      - value/Q provide a structured HDLAggregate view via agg_cls.from_bits(...).
      - @= drives the register's next-state (reg driver).
    """

    def __init__(
        self,
        agg_cls: Type[T_Agg],
        *agg_args,
        name: Optional[str] = None,
        init: Optional[Union[T_Agg, ExprLike]] = None,
        **agg_kwargs,
    ):
        # Store metadata so we can rebuild a structured view later
        self._agg_cls: Type[T_Agg] = agg_cls
        self._agg_args = agg_args
        self._agg_kwargs = dict(agg_kwargs)

        # Use a wire-like instance to infer shape/width
        proto: T_Agg = agg_cls.wire_like(*agg_args, **agg_kwargs)
        bits_typ = proto.to_bits().typ

        reg_name = name or f"reg_{agg_cls.__name__}_{id(self)}"
        self._reg = Signal(reg_name, bits_typ, kind="reg")

        # Optional init value (packed)
        if init is not None:
            if isinstance(init, HDLAggregate):
                init_bits = init.to_bits()
            else:
                init_bits = as_expr(init)
            self._reg._init = fit_width(init_bits, bits_typ)

    # ---- HDLAggregate API ----

    def to_bits(self) -> Expr:
        """Packed register contents as a flat bitvector Expr."""
        return self._reg

    @classmethod
    def from_bits(
        cls,
        bits: Expr,
        agg_cls: Type[T_Agg],
        *agg_args,
        **agg_kwargs,
    ) -> "AggregateRegister[T_Agg]":
        """
        Create an AggregateRegister *view* around an existing reg-like Expr.
        Typically not needed for normal user code.
        """
        obj = cls.__new__(cls)
        obj._agg_cls = agg_cls
        obj._agg_args = agg_args
        obj._agg_kwargs = dict(agg_kwargs)
        obj._reg = bits  # assume reg-like Signal or Expr
        return obj

    @classmethod
    def wire_like(cls, *args, **kwargs):
        raise TypeError("AggregateRegister.wire_like() is not meaningful")

    def _assign_from_bits(self, bits: Expr) -> None:
        """
        Drive the register's next-state with the packed bits.
        """
        if not isinstance(self._reg, Signal) or self._reg.kind != "reg":
            raise TypeError("AggregateRegister must wrap a register-like Signal")
        self._reg <<= bits

    # ---- Convenience views ----

    @property
    def value(self) -> T_Agg:
        """Structured view of the register contents."""
        return self._agg_cls.from_bits(self._reg, *self._agg_args, **self._agg_kwargs)

    @property
    def Q(self) -> T_Agg:
        """Alias for value (typical register naming)."""
        return self.value

    @property
    def bits(self) -> Expr:
        """Raw register bits as Expr."""
        return self._reg
