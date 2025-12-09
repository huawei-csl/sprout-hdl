# sprouthdl/bundle.py

from __future__ import annotations

from typing import Any, ClassVar, Dict, Iterator, Mapping, Optional, Tuple, Type, TypeVar, Union

from sprouthdl.sprouthdl import (
    Expr,
    ExprLike,
    HDLType,
    Register,
    Signal,
    UInt,
    as_expr,
    cat,
    fit_width,
)
from sprouthdl.sprouthdl_module import Module


B = TypeVar("B", bound="Bundle")


class Bundle:
    """Base class for bundle *specifications*.

    Subclass this and declare HDLType fields directly, e.g.:

        class Payload(Bundle):
            opcode: UInt = UInt(3)
            data:   UInt = UInt(13)

    The class itself is the schema; you never change these attributes at runtime.
    """

    # Per-subclass caches (set in __init_subclass__)
    __field_names__: ClassVar[Tuple[str, ...]]
    __field_types__: ClassVar[Dict[str, HDLType]]
    __field_slices__: ClassVar[Dict[str, Tuple[int, int]]]
    __width__: ClassVar[int]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        items: list[Tuple[str, HDLType]] = []
        for name, val in cls.__dict__.items():
            # Treat any HDLType-valued attribute as a bundle field
            if isinstance(val, HDLType):
                items.append((name, val))

        if not items:
            raise ValueError(f"Bundle subclass {cls.__name__!r} must define at least one HDLType " "attribute like 'opcode: UInt = UInt(3)'")

        field_names: list[str] = []
        field_types: Dict[str, HDLType] = {}
        field_slices: Dict[str, Tuple[int, int]] = {}
        offset = 0

        for name, hdl in items:
            if name in field_types:
                raise ValueError(f"Duplicate bundle field name {name!r} in {cls.__name__}")
            field_names.append(name)
            field_types[name] = hdl
            w = hdl.width
            field_slices[name] = (offset, offset + w)
            offset += w

        cls.__field_names__ = tuple(field_names)
        cls.__field_types__ = field_types
        cls.__field_slices__ = field_slices
        cls.__width__ = offset

    # ------------------------------------------------------------------
    # Schema access
    # ------------------------------------------------------------------
    @classmethod
    def width(cls) -> int:
        return cls.__width__

    @classmethod
    def field_names(cls) -> Tuple[str, ...]:
        return cls.__field_names__

    @classmethod
    def field_type(cls, name: str) -> HDLType:
        return cls.__field_types__[name]

    @classmethod
    def field_slice(cls, name: str) -> Tuple[int, int]:
        return cls.__field_slices__[name]

    # ------------------------------------------------------------------
    # Packing / unpacking (class-level)
    # ------------------------------------------------------------------
    @classmethod
    def coerce_mapping(cls: Type[B], value: Union[Mapping[str, ExprLike], B]) -> Mapping[str, Expr]:
        """Accept either a mapping or another compatible Bundle *value* object.

        Here we only care about field -> Expr; the schema is fixed on the class.
        """
        # If someone passes an instance of the same Bundle subclass, treat it like a mapping.
        if isinstance(value, cls):
            # For simplicity, assume they stored Expr-like attributes on the instance.
            values: Dict[str, Expr] = {}
            for name in cls.__field_names__:
                expr = fit_width(as_expr(getattr(value, name)), cls.__field_types__[name])
                values[name] = expr
            return values

        if isinstance(value, Bundle) and type(value) is not cls:
            # Different Bundle subclass: require identical field names
            other = type(value)
            if other.__field_names__ != cls.__field_names__:
                raise TypeError(f"Incompatible bundle assignment: {other.__name__} -> {cls.__name__}")
            values = {}
            for name in cls.__field_names__:
                expr = fit_width(as_expr(getattr(value, name)), cls.__field_types__[name])
                values[name] = expr
            return values

        if isinstance(value, Mapping):
            # Validate keys
            extra = set(value.keys()) - set(cls.__field_names__)
            if extra:
                raise KeyError(f"Unknown bundle fields: {sorted(extra)}")
            missing = [n for n in cls.__field_names__ if n not in value]
            if missing:
                raise KeyError(f"Missing bundle fields: {missing}")
            return {name: fit_width(as_expr(value[name]), cls.__field_types__[name]) for name in cls.__field_names__}

        raise TypeError(f"Bundle assignment expects mapping or Bundle instance, got {type(value)!r}")

    @classmethod
    def to_bits(cls, value: Union[Mapping[str, ExprLike], "Bundle"]) -> Expr:
        """Pack a mapping or Bundle instance into a single bit-vector Expr."""
        resolved = cls.coerce_mapping(value)
        # Keep declaration order; first field becomes least-significant chunk
        parts = [resolved[name] for name in cls.__field_names__]
        return cat(*parts)

    @classmethod
    def from_bits(cls: Type[B], bits: ExprLike) -> Dict[str, Expr]:
        """Unpack a bit-vector into a dict mapping field -> Expr."""
        bit_expr = fit_width(as_expr(bits), UInt(cls.__width__))
        result: Dict[str, Expr] = {}
        offset = 0
        for name in cls.__field_names__:
            hdl = cls.__field_types__[name]
            w = hdl.width
            slice_expr = bit_expr[offset : offset + w]
            result[name] = fit_width(slice_expr, hdl)
            offset += w
        return result

    # Optional: convenience to iterate a value mapping in order
    @classmethod
    def iter_items(cls, value: Union[Mapping[str, ExprLike], "Bundle"]) -> Iterator[Tuple[str, Expr]]:
        resolved = cls.coerce_mapping(value)
        for name in cls.__field_names__:
            yield name, resolved[name]



class BundleRegister2:
    """Register wrapper for a Bundle spec class."""

    def __init__(self, typ: Bundle, init: Optional[Union[Mapping[str, ExprLike], Bundle]] = None, name: Optional[str] = None):

        width = typ.width()
        init_bits = None
        if init is not None:
            init_bits = typ.to_bits(init)
        base_signal = Register(UInt(width), init_bits, name)

        self._signal = base_signal
        self._cls = typ

    # Attribute: field slices or delegate to Signal
    def __getattr__(self, name: str):
        if name in self._cls.__field_names__:
            start, stop = self._cls.__field_slices__[name]
            return self._signal[start:stop]
        return getattr(self._signal, name)

    # Register semantics ------------------------------------------------
    @property
    def next(self) -> Optional[Dict[str, Expr]]:
        drv = self._signal._driver
        if drv is None:
            return None
        # Represent next value as dict[field -> Expr]
        return self._cls.from_bits(drv)

    @next.setter
    def next(self, value: Union[Mapping[str, ExprLike], Bundle]):
        self._signal <<= self._cls.to_bits(value)

    def __ilshift__(self, value: Union[Mapping[str, ExprLike], Bundle]):
        self.next = value
        return self

    def set_init(self, value: Union[Mapping[str, ExprLike], Bundle]):
        self._signal.set_init(self._cls.to_bits(value))

    # Convenience -------------------------------------------------------
    def to_bits(self) -> Signal:
        return self._signal

    @property
    def typ(self) -> HDLType:
        return self._signal.typ

    @property
    def name(self) -> str:
        return self._signal.name


# --- Monkey patch Module.reg -------------------------------------------
# _original_module_reg = Module.reg


# def _reg_with_bundle(
#     self: Module,
#     typ: Any,
#     name: str,
#     init: Optional[Union[Mapping[str, ExprLike], Bundle]] = None,
# ):
#     # Detect Bundle specs: subclasses of Bundle with HDLType attributes
#     if isinstance(typ, type) and issubclass(typ, Bundle):
#         width = typ.width()
#         init_bits = None
#         if init is not None:
#             init_bits = typ.to_bits(init)
#         base_signal = _original_module_reg(self, UInt(width), name, init_bits)
#         return BundleRegister(base_signal, typ)

#     # Fallback: normal behaviour
#     return _original_module_reg(self, typ, name, init)


# Module.reg = _reg_with_bundle  # type: ignore[assignment]
