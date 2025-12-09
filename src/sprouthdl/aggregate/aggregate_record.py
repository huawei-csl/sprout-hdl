from __future__ import annotations
from typing import Iterable, List, Sequence, Union, Tuple

from sprouthdl.aggregate.hdl_aggregate import HDLAggregate
from sprouthdl.sprouthdl import Expr, ExprLike, Signal, Wire, Concat, as_expr, fit_width

from typing import Type, TypeVar, Union, Dict, Any

T_Record = TypeVar("T_Record", bound="AggregateRecord")


class AggregateRecord(HDLAggregate):
    """
    Class-based record aggregate.

    Usage:

        class MyRecord():
            a: Signal = Wire(UInt(8))
            b: Signal = Wire(SInt(4))
            c: Array  = Array([Wire(UInt(8)) for _ in range(4)])

    On subclass definition (__init_subclass__), we scan the class
    attributes and treat any 'wire template' or HDLAggregate as a field
    template. On instance creation, these templates are cloned so that
    each instance gets its own wires/aggregates (no sharing).
    """

    _record_field_templates: Dict[str, Union[Signal, HDLAggregate]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        templates: Dict[str, Union[Signal, HDLAggregate]] = {}

        for name, value in cls.__dict__.items():
            # Skip private / dunder / methods / descriptors
            if name.startswith("_"):
                continue

            # Only instances are interesting (class-level templates)
            if isinstance(value, Signal):
                if value.kind != "wire":
                    raise TypeError(f"Record field '{name}' must be a wire Signal (kind='wire'), " f"got kind={value.kind!r}")
                templates[name] = value
            elif isinstance(value, HDLAggregate):
                # For aggregates, we expect they are wire-backed; cloning
                # is done via type(value).wire_like(value).
                templates[name] = value

        cls._record_field_templates = templates

    def __init__(self, **overrides: Union[Signal, HDLAggregate]) -> None:
        """
        Instantiate a record.

        - For each template field on the class, create a new instance:
            * Signal (wire)   → new Wire with same HDLType
            * HDLAggregate    → type(template).wire_like(template)
        - If a field is passed via keyword argument, use that instead.

        Example:
            b = MyRecord()
            b2 = MyRecord(a=Wire(UInt(8)))  # override 'a'
        """
        # Clone all template fields
        for name, tmpl in self._record_field_templates.items():
            if name in overrides:
                # todo: check that override type matches template width
                val = overrides[name]
            else:
                if isinstance(tmpl, Signal):
                    if tmpl.kind != "wire":
                        raise TypeError(f"Record field '{name}' template must be a wire Signal, " f"got kind={tmpl.kind!r}")
                    # Clone: new Wire with same type
                    val = Wire(tmpl.typ)
                elif isinstance(tmpl, HDLAggregate):
                    # Clone aggregate via its wire_like(template) API
                    val = type(tmpl).wire_like(tmpl)
                else:
                    raise TypeError(f"Unsupported template type for record field '{name}': " f"{type(tmpl)}")

            setattr(self, name, val)

        # Check for unknown overrides
        for k in overrides:
            if k not in self._record_field_templates:
                raise AttributeError(f"Unknown record field override '{k}' " f"for {self.__class__.__name__}")

    # ---------------- HDLAggregate API ----------------

    def to_bits(self) -> Expr:
        """Flatten all fields into a single bitvector Expr in field order."""
        from sprouthdl.sprouthdl import Concat  # local import to avoid cycles

        parts = []
        for name in self._record_field_templates.keys():
            v = getattr(self, name)
            if isinstance(v, Expr):
                parts.append(v)
            elif isinstance(v, HDLAggregate):
                parts.append(v.to_bits())
            else:
                raise TypeError(f"Unsupported field type in AggregateRecord.to_bits(): " f"{name} -> {type(v)}")
        if not parts:
            raise ValueError(f"AggregateRecord {self.__class__.__name__} has no fields")
        if len(parts) == 1:
            return parts[0]
        return Concat(parts)

    @classmethod
    def wire_like(
        cls: Type[T_Record],
        *args: Any,
        **kwargs: Any,
    ) -> T_Record:
        """
        Create a 'wire-filled' instance.

        For records, the shape is fully defined by the class, so we ignore
        any template argument and simply construct a fresh instance.
        """
        return cls()

    def _assign_from_bits(self, bits: Expr) -> None:
        """
        Drive all fields from the flat bitvector 'bits' in field order.
        """
        bit_pos = 0
        for name in self._record_field_templates.keys():
            v = getattr(self, name)

            if isinstance(v, Expr):
                w = v.typ.width
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w
                if isinstance(v, Signal):
                    v <<= slice_bits
                else:
                    # Replace non-Signal Expr with width-adapted slice
                    setattr(self, name, fit_width(slice_bits, v.typ))

            elif isinstance(v, HDLAggregate):
                w = v.width
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w
                v.assign(slice_bits)

            else:
                raise TypeError(f"Unsupported field type in AggregateRecord._assign_from_bits(): " f"{name} -> {type(v)}")

        # Consistency check
        if bit_pos != bits.typ.width:
            raise ValueError(
                f"Bit-slice consumption mismatch in {self.__class__.__name__}: "
                f"used {bit_pos} of {bits.typ.width} bits"
            )

    def __repr__(self) -> str:
        fields = ", ".join(self._record_field_templates.keys())
        return f"{self.__class__.__name__}(fields=[{fields}], width={self.width})"

    def __imatmul__(self, rhs: "AggregateRecord") -> "AggregateRecord":
        """
        Element-wise assignment: rec @= other

        For each field:
          - Signal field: lhs_field <<= rhs_field
          - HDLAggregate field (e.g. Array, nested Record): lhs_field <<= rhs_field
            (which can itself be elementwise if the child overrides __ilshift__)
        """
        if not isinstance(rhs, self.__class__):
            raise TypeError(f"{self.__class__.__name__} @= expects same record type, " f"got {type(rhs)}")

        # We assume same field templates on both sides
        for name in self._record_field_templates.keys():  # or _record_field_templates
            lhs = getattr(self, name)
            r = getattr(rhs, name)

            if isinstance(lhs, Expr) and isinstance(r, Expr):
                if isinstance(lhs, Signal):
                    lhs <<= r
                else:
                    # Non-signal Expr → just replace reference
                    setattr(self, name, r)
            elif isinstance(lhs, HDLAggregate) and isinstance(r, HDLAggregate):
                # Delegate to child aggregate's elementwise (if any)
                lhs <<= r
            else:
                raise TypeError(f"Incompatible field types for elementwise assign in " f"{self.__class__.__name__}.{name}: " f"{type(lhs)} @= {type(r)}")

        return self
