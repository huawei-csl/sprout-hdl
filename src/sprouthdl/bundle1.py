"""Bundle container support for Sprout HDL.

This module introduces a ``bundle`` container that can aggregate multiple
``HDLType`` fields into a single logical value. Bundles can be converted to and
from bit representations and integrate with :class:`sprouthdl.sprouthdl_module.Module`
registers without requiring any modifications to the existing core code base.

Example usage::

    from sprouthdl.sprouthdl import Module, UInt
    from sprouthdl.bundle import bundle

    payload_t = bundle({
        "opcode": UInt(3),
        "data": UInt(13),
    })

    m = Module("example")
    payload_reg = m.reg(payload_t, "payload")
    payload_reg <<= payload_t(opcode=1, data=5)

The bundle register can be sliced through attribute access (``payload_reg.opcode``)
and its value can be packed/unpacked via :meth:`bundle.to_bits` and
:meth:`bundle.from_bits`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from sprouthdl.sprouthdl import (
    Expr,
    ExprLike,
    HDLType,
    Signal,
    UInt,
    as_expr,
    cat,
    fit_width,
)
from sprouthdl.sprouthdl_module import Module


@dataclass(frozen=True)
class _BundleField:
    """Container describing one field within a bundle."""

    name: str
    typ: HDLType


class BundleValue:
    """Concrete value stored in a :class:`bundle` instance.

    Parameters
    ----------
    bundle
        The :class:`bundle` specification the value conforms to.
    values
        Mapping from field name to expressions that should be associated with
        the bundle.
    """

    def __init__(self, bundle: "bundle", values: Mapping[str, ExprLike]):
        self._bundle = bundle
        resolved: Dict[str, Expr] = {}
        for field in self._bundle.fields:
            if field.name not in values:
                raise KeyError(f"Missing value for bundle field '{field.name}'")
            value_expr = fit_width(as_expr(values[field.name]), field.typ)
            resolved[field.name] = value_expr
        # Detect unknown keys
        for provided in values.keys():
            if provided not in self._bundle.field_names:
                raise KeyError(f"Unknown bundle field '{provided}'")
        self._values = resolved

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def bundle(self) -> "bundle":
        return self._bundle

    @property
    def values(self) -> Mapping[str, Expr]:
        return self._values

    def __getitem__(self, name: str) -> Expr:
        return self._values[name]

    def __getattr__(self, name: str) -> Expr:
        try:
            return self._values[name]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise AttributeError(name) from exc

    def __iter__(self) -> Iterator[Tuple[str, Expr]]:
        for field in self._bundle.fields:
            yield field.name, self._values[field.name]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        field_reprs = ", ".join(f"{name}={expr!r}" for name, expr in self)
        return f"BundleValue({field_reprs})"

    # ------------------------------------------------------------------
    # Bit packing helpers
    # ------------------------------------------------------------------
    def to_bits(self) -> Expr:
        """Pack the bundle into a single unsigned bit-vector expression."""

        parts = [self._values[field.name] for field in self._bundle.fields]
        # ``cat`` concatenates the provided expressions after reversing the
        # argument order, therefore the first field is treated as the least
        # significant portion.
        return cat(*parts)

    @classmethod
    def from_bits(cls, bundle: "bundle", bits: ExprLike) -> "BundleValue":
        """Unpack an unsigned bit-vector expression into a bundle value."""

        bit_expr = fit_width(as_expr(bits), UInt(bundle.width))
        values: Dict[str, Expr] = {}
        offset = 0
        for field in bundle.fields:
            width = field.typ.width
            slice_expr = bit_expr[offset : offset + width]
            values[field.name] = fit_width(slice_expr, field.typ)
            offset += width
        return cls(bundle, values)


class bundle:
    """Descriptor for a collection of Sprout HDL fields."""

    def __init__(
        self,
        fields: Mapping[str, HDLType]
        | Sequence[Tuple[str, HDLType]]
        | type,
        *,
        name: Optional[str] = None,
    ):
        self._definition: Optional[type] = None
        if isinstance(fields, type):
            items = list(self._fields_from_class(fields))
            self._definition = fields
            if name is None:
                name = fields.__name__
        elif isinstance(fields, Mapping):
            items = list(fields.items())
        else:
            items = list(fields)
        if not items:
            raise ValueError("bundle requires at least one field")
        seen: set[str] = set()
        normalized: list[_BundleField] = []
        for field_name, typ in items:
            if not isinstance(typ, HDLType):
                raise TypeError(
                    f"Field '{field_name}' must be an HDLType, not {type(typ)!r}"
                )
            if field_name in seen:
                raise ValueError(f"Duplicate bundle field '{field_name}'")
            seen.add(field_name)
            normalized.append(_BundleField(field_name, typ))
        self._fields = normalized
        self._width = sum(field.typ.width for field in self._fields)
        self._field_slices: Dict[str, Tuple[int, int]] = {}
        offset = 0
        for field in self._fields:
            self._field_slices[field.name] = (offset, offset + field.typ.width)
            offset += field.typ.width
        self._name = name

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def fields(self) -> Sequence[_BundleField]:
        return tuple(self._fields)

    @property
    def field_names(self) -> Sequence[str]:
        return tuple(field.name for field in self._fields)

    @property
    def width(self) -> int:
        return self._width

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def definition(self) -> Optional[type]:
        return self._definition

    # ------------------------------------------------------------------
    # Value helpers
    # ------------------------------------------------------------------
    def __call__(self, /, **values: ExprLike) -> BundleValue:
        return BundleValue(self, values)

    def coerce(self, value: Union[BundleValue, Mapping[str, ExprLike]]) -> BundleValue:
        if isinstance(value, BundleValue):
            if value.bundle is not self:
                # Allow re-wrapping compatible bundles by comparing field names
                if value.bundle.field_names != self.field_names:
                    raise TypeError("BundleValue belongs to a different bundle specification")
                return BundleValue(self, value.values)
            return value
        if isinstance(value, Mapping):
            return BundleValue(self, value)
        raise TypeError("Bundle assignment expects a mapping or BundleValue")

    def to_bits(self, value: Union[BundleValue, Mapping[str, ExprLike]]) -> Expr:
        return self.coerce(value).to_bits()

    def from_bits(self, bits: ExprLike) -> BundleValue:
        return BundleValue.from_bits(self, bits)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _slice_for(self, name: str) -> Tuple[int, int]:
        return self._field_slices[name]

    @staticmethod
    def _fields_from_class(definition: type) -> Sequence[Tuple[str, HDLType]]:
        items: list[Tuple[str, HDLType]] = []
        for attr_name, attr_value in definition.__dict__.items():
            if isinstance(attr_value, HDLType):
                items.append((attr_name, attr_value))
        if not items:
            raise ValueError(
                f"bundle class '{definition.__name__}' must define at least one HDLType attribute"
            )
        return items

    @classmethod
    def from_class(cls, definition: type) -> "bundle":
        spec = cls(definition)
        spec.__module__ = definition.__module__
        return spec

    @classmethod
    def define(cls, definition: Optional[type] = None):
        if definition is None:
            return lambda def_cls: cls.from_class(def_cls)
        return cls.from_class(definition)


class BundleRegister:
    """Wrapper around a register storing a :class:`bundle` value."""

    def __init__(self, signal: Signal, spec: bundle):
        self._signal = signal
        self._spec = spec

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        if name in self._spec.field_names:
            start, stop = self._spec._slice_for(name)
            return self._signal[start:stop]
        return getattr(self._signal, name)

    # ------------------------------------------------------------------
    # Register semantics
    # ------------------------------------------------------------------
    @property
    def next(self) -> Optional[BundleValue]:
        drv = self._signal._driver
        if drv is None:
            return None
        return self._spec.from_bits(drv)

    @next.setter
    def next(self, value: Union[BundleValue, Mapping[str, ExprLike]]):
        coerced = self._spec.coerce(value)
        self._signal <<= coerced.to_bits()

    def __ilshift__(self, value: Union[BundleValue, Mapping[str, ExprLike]]):
        self.next = value
        return self

    def set_init(self, value: Union[BundleValue, Mapping[str, ExprLike]]):
        coerced = self._spec.coerce(value)
        self._signal.set_init(coerced.to_bits())

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def to_bits(self) -> Signal:
        """Expose the underlying register signal as an unsigned bit vector."""

        return self._signal

    def from_bits(self, bits: ExprLike) -> BundleValue:
        return self._spec.from_bits(bits)

    @property
    def typ(self) -> HDLType:
        return self._signal.typ

    @property
    def name(self) -> str:  # pragma: no cover - simple delegation
        return self._signal.name


_original_module_reg = Module.reg


def _reg_with_bundle(self: Module, typ, name: str, init: Optional[Union[BundleValue, Mapping[str, ExprLike]]] = None):
    if isinstance(typ, bundle):
        init_bits = None
        if init is not None:
            init_bits = typ.coerce(init).to_bits()
        base_signal = _original_module_reg(self, UInt(typ.width), name, init_bits)
        return BundleRegister(base_signal, typ)
    return _original_module_reg(self, typ, name, init)


Module.reg = _reg_with_bundle  # type: ignore[assignment]
