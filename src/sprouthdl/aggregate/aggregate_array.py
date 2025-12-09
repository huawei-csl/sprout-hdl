from __future__ import annotations
from typing import Iterable, List, Sequence, Union, Tuple

from sprouthdl.aggregate.hdl_aggregate import HDLAggregate
from sprouthdl.sprouthdl import (
    Expr,
    ExprLike,
    Signal,
    Wire,
    Concat,
    as_expr,
    fit_width
)


# Each element is either a plain Expr (Signal/Const/etc.)
# or any higher-level aggregate (Bundle, FixedPoint, nested Array, ...)
ArrayElem = Union[Expr, HDLAggregate]
InputElem = Union[ExprLike, HDLAggregate]


class Array(HDLAggregate):
    def __init__(
        self,
        values: Sequence[InputElem],
    ):
        """
        Construct an Array from:
          - ExprLike (int, bool, Expr) → Expr (via as_expr)
          - HDLAggregate                 → nested aggregate (Bundle, Array, FixedPoint, ...)
        """
        if not values:
            raise ValueError("Array requires at least one element")

        elems: List[ArrayElem] = []
        for v in values:
            if isinstance(v, HDLAggregate):
                # Any aggregate (Bundle, Array, FixedPoint, ...)
                elems.append(v)
            else:
                # ExprLike (int, bool, Expr) → Expr
                elems.append(as_expr(v))

        self._elems: List[ArrayElem] = elems

    # --------------------------------------------------------------
    # Basic iteration / len
    # --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._elems)

    def __iter__(self) -> Iterable[ArrayElem]:
        return iter(self._elems)

    # --------------------------------------------------------------
    # N-dimensional indexing
    # --------------------------------------------------------------
    def __getitem__(self, key):
        """
        Generic N-dimensional indexing via tuples:

          - 1D:
              arr[i]
              arr[i:j]

          - 2D:
              mat[i, j]
              mat[i, :]
              mat[:, j]
              mat[i0:i1, j0:j1]

          - ND:
              tensor[i, j, k]
              tensor[:, 1, 2:]
              tensor[1:3, :, 0]

        Works as long as the underlying structure is nested Arrays
        where appropriate.
        """
        if not isinstance(key, tuple):
            key = (key,)
        return self._get_nd(key)

    def _get_nd(self, keys: Tuple) -> Union[ArrayElem, "Array"]:
        # No more axes: return this sub-array as-is
        if not keys:
            return self

        idx, *rest = keys

        # -------- Case 1: integer index at this dimension --------
        if isinstance(idx, int):
            elem = self._elems[idx]
            if not rest:
                # Last axis: return scalar element / sub-array / aggregate
                return elem
            # More axes: we must recurse into a nested Array
            if not isinstance(elem, Array):
                raise TypeError(f"Too many indices: element at this axis is {type(elem)}, not Array")
            return elem._get_nd(tuple(rest))

        # -------- Case 2: slice at this dimension --------
        if isinstance(idx, slice):
            sliced_elems = self._elems[idx]

            if not rest:
                # Just slicing this dimension: return a 1D Array with those elements
                return Array(sliced_elems)

            # There are more axes: recurse into each element, collecting results
            sub_arrays: List[ArrayElem] = []
            for elem in sliced_elems:
                if not isinstance(elem, Array):
                    raise TypeError("Too many indices for Array: expected nested Array " f"but found {type(elem)}")
                sub_arrays.append(elem._get_nd(tuple(rest)))

            return Array(sub_arrays)

        raise TypeError(f"Unsupported index type at this axis: {type(idx)}")

    # --------------------------------------------------------------
    # __setitem__ (still 1D for now; can be extended similarly)
    # --------------------------------------------------------------
    def __setitem__(
        self,
        idx,
        value: InputElem,
    ) -> None:
        if isinstance(idx, tuple):
            # ND assignment could be added similarly to _get_nd; for now we keep 1D.
            raise NotImplementedError("N-dimensional __setitem__ not implemented")
        elif isinstance(value, HDLAggregate):
            coerced = value
        else:
            coerced = as_expr(value)

        self._elems[idx] = coerced

    # --------------------------------------------------------------
    # Width + bit packing
    # --------------------------------------------------------------
    @property
    def width(self) -> int:
        """
        Total bit width of this array (recursively).

        NOTE: This overrides HDLAggregate.width for Arrays but is equivalent
        to `self.to_bits().typ.width`.
        """
        total = 0
        for elem in self._elems:
            if isinstance(elem, Expr):
                total += elem.typ.width
            elif isinstance(elem, HDLAggregate):
                total += elem.width
            else:
                raise TypeError(f"Unsupported element type in width(): {type(elem)}")
        return total

    def to_bits(self) -> Expr:
        """
        Flatten this Array to a single Expr bitvector.
        Layout is consistent with _assign_from_bits.
        """
        parts: List[Expr] = []
        for elem in self._elems:
            if isinstance(elem, Expr):
                parts.append(elem)
            elif isinstance(elem, HDLAggregate):
                parts.append(elem.to_bits())
            else:
                raise TypeError(f"Unsupported element type in to_bits(): {type(elem)}")
        return Concat(parts)

    # --------------------------------------------------------------
    # HDLAggregate API: from_bits + wire_like + _assign_from_bits
    # --------------------------------------------------------------
    @classmethod
    def from_bits(cls, bits: Expr, template: "Array") -> "Array":
        """
        Reinterpret 'bits' as an Array with the same shape as 'template',
        assigning into that template (in-place) and returning it.

        Usage:
            Array.from_bits(some_bits, template_array)
        """
        template._assign_from_bits(bits)
        return template

    @classmethod
    def wire_like(cls, template: "Array") -> "Array":
        """
        Create a 'wire-filled' Array with the same recursive shape as 'template'.

        Convention:
          - Expr leaves        → new Wire(typ)
          - HDLAggregate leaves → type(elem).wire_like(elem)

        So any aggregate type you put inside Array (Bundle, FixedPoint, nested Array, ...)
        should implement `@classmethod wire_like(template_instance)`.
        """
        new_elems: List[ArrayElem] = []

        for elem in template._elems:
            if isinstance(elem, Expr):
                new_elems.append(Wire(elem.typ))
            elif isinstance(elem, HDLAggregate):
                new_elems.append(type(elem).wire_like(elem))
            else:
                raise TypeError(f"Unsupported element type in Array.wire_like: {type(elem)}")

        return cls(new_elems)

    def _assign_from_bits(self, bits: Expr) -> None:
        """
        Drive the underlying leaves from a flat bitvector 'bits'.
        This is what HDLAggregate.assign / <<= eventually calls.
        """
        bit_pos = 0
        for i, elem in enumerate(self._elems):
            if isinstance(elem, Expr):
                w = elem.typ.width
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w

                if isinstance(elem, Signal):
                    # Combinational driver
                    elem <<= slice_bits
                else:
                    # Replace non-Signal Expr with an Expr of the right width
                    self._elems[i] = fit_width(slice_bits, elem.typ)

            elif isinstance(elem, HDLAggregate):
                w = elem.width
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w
                # Delegate to the aggregate's packed assignment
                elem.assign(slice_bits)

            else:
                raise TypeError(f"Unsupported element type in _assign_from_bits(): {type(elem)}")

        # Optional sanity check:
        # if bit_pos != bits.typ.width:
        #     raise ValueError("Bit-slice consumption mismatch in Array._assign_from_bits")

    # --------------------------------------------------------------
    # Element-wise assignment (shape-agnostic)
    # --------------------------------------------------------------
    def __imatmul__(self, rhs: "Array"):
        """
        Element-wise assignment:
          lhs @= rhs

        This is orthogonal to HDLAggregate's packed assignment (lhs <<= rhs).
        """
        if len(self) != len(rhs):
            raise ValueError("Array.__imatmul__: length mismatch")

        for i in range(len(self)):
            lhs = self._elems[i]
            r = rhs._elems[i]

            if isinstance(lhs, Expr) and isinstance(r, Expr):
                if isinstance(lhs, Signal):
                    lhs <<= r
                else:
                    self._elems[i] = r

            elif isinstance(lhs, HDLAggregate) and isinstance(r, HDLAggregate):
                # Delegate to the aggregate's own element-wise semantics
                lhs <<= r

            else:
                raise TypeError("Incompatible element types in Array.__ilshift__: " f"{type(lhs)} <<= {type(r)}")
        return self

    def __repr__(self) -> str:
        return f"Array(len={len(self)}, width={self.width})"
