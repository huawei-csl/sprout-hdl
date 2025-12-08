from typing import Union
from dataclasses import dataclass, fields, MISSING
from sprouthdl.sprouthdl import Concat, Const, Expr, ExprLike, HDLType, Signal, UInt, Wire

# maybe use ExprLike for assigned values?
class Bundle3:

    # def __post_init__(self) -> None:
    #     """
    #     Interpret dataclass defaults that are HDLType / Bundle3 as *schemas*:

    #     - If a field's default is an HDLType (e.g. UInt(3)) and the user did not
    #       override it in __init__, automatically replace it with a Wire(default).

    #     - If a field's default is a Bundle3 instance and the user did not override it,
    #       automatically replace it with a fresh nested Bundle3 instance.

    #     - If the user passed an explicit value (e.g. Const(...)), leave it untouched.
    #     """
    #     for f in fields(self):  # iterate dataclass fields
    #         default = f.default
    #         if default is MISSING:
    #             # Either no default or a default_factory:
    #             # in both cases, dataclasses has already produced a value we should respect.
    #             continue

    #         current = getattr(self, f.name)

    #         # If the user passed something explicitly, current != default,
    #         # so we must NOT replace it.
    #         if current is not default:
    #             continue

    #         # Case 1: default is an HDLType (e.g. UInt(3)) -> create a Wire of that type
    #         if isinstance(default, HDLType):
    #             setattr(self, f.name, Wire(default))

    #         # Case 2: default is a nested Bundle3 schema -> create a fresh instance
    #         elif isinstance(default, Bundle3):
    #             nested = type(default)()  # this will also run its __post_init__
    #             setattr(self, f.name, nested)

    #         # Anything else (Const, integers, etc.) we leave as-is
    #         # to allow "spec" instances with constant fields if you want them.

    # def __post_init__(self) -> None:
    #     """
    #     For any dataclass field whose default is a Wire/Signal or Bundle3 and that
    #     the user did not override in __init__, create a fresh instance.

    #     This gives you "factory"-like semantics while allowing:

    #         a: ExprLike = Wire(UInt(3))

    #     instead of requiring field(default_factory=...).
    #     """
    #     for f in fields(self):  # type(self) is a @dataclass subclass
    #         default = f.default
    #         if default is MISSING:
    #             continue  # no plain default; may be default_factory, which already creates fresh instances

    #         current = getattr(self, f.name)

    #         # If the user passed an explicit value, or we already changed it, skip.
    #         if current is not default:
    #             continue

    #         # Default is a nested bundle: make a fresh one
    #         if isinstance(default, Bundle3):
    #             setattr(self, f.name, type(default)())  # calls its own __post_init__
    #         # Default is a Signal (Wire or some other signal kind): clone shape
    #         elif isinstance(default, Signal):
    #             # if default.kind == "wire":
    #             #     # New wire with same HDLType
    #             #     setattr(self, f.name, Wire(default.typ))
    #             # else:
    #             #     # Fallback: shallow clone other signals; adjust if you need more nuance
    #                 setattr(self, f.name, Signal(default.name, default.typ, default.kind))
    # Everything else (Const, UInt, etc.) can safely stay shared.

    def get_expr_bundle_fields(self) -> dict[str, Union[Expr, "Bundle3"]]:
        """Return bundle fields in instance order (Expr or nested Bundle3)."""
        fields: dict[str, Union[Expr, "Bundle3"]] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Expr) or isinstance(value, Bundle3):
                fields[name] = value
        return fields

    def to_bits(self) -> Expr:
        """Pack all Expr/Bundle3 fields into a single bit-vector."""
        slices = []
        for _, value in self.get_expr_bundle_fields().items():
            if isinstance(value, Expr):
                slices.append(value)
            elif isinstance(value, Bundle3):
                slices.append(value.to_bits())
            else:  # defensive; should not happen
                raise TypeError(f"Unsupported field type in to_bits: {type(value)}")
        return Concat(slices)

    def from_bits(self, bits: Expr) -> 'Bundle3':
        """Unpack a bit-vector into this instance's fields (Signals / nested Bundles)."""
        bit_pos = 0
        for name, value in self.get_expr_bundle_fields().items():
            if isinstance(value, Signal):
                w = value.typ.width
                field_bits = bits[bit_pos : bit_pos + w]
                value <<= field_bits
                bit_pos += w
            elif isinstance(value, Bundle3):
                w = value.width()
                field_bits = bits[bit_pos : bit_pos + w]
                value.from_bits(field_bits)
                bit_pos += w
            else:
                raise TypeError(f"Unsupported field type for assignment: {type(value)}")
        return self

    # ------------------------------------------------------------------
    # Bundle-wise assignment
    # ------------------------------------------------------------------
    def __ilshift__(self, rhs: "Bundle3"):
        """Field-wise connect: bundle_wire <<= bundle_expr."""
        # assume same class = same field names
        for name, lhs_value in self.get_expr_bundle_fields().items():
            rhs_value = getattr(rhs, name)
            lhs_value <<= rhs_value  # Signal.__ilshift__ or Bundle3.__ilshift__
        return self  

    def width(self) -> int:
        total_width = 0
        for field_name, field_value in self.get_expr_bundle_fields().items():
            if isinstance(field_value, Expr):
                if not hasattr(field_value, 'typ'):
                    raise TypeError(f"Expr field {field_name} has no 'typ' attribute for width calculation.")
                field_width = field_value.typ.width
            elif isinstance(field_value, Bundle3):
                field_width = field_value.width()
            else:
                raise TypeError(f"Unsupported field type for width calculation: {type(field_value)}")
            total_width += field_width
        return total_width
