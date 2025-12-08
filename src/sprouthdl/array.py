from __future__ import annotations
from typing import Iterable, List, Sequence, Union

from sprouthdl.sprouthdl import (
    Expr,
    ExprLike,
    HDLType,
    Signal,
    Wire,
    Concat,
    as_expr,
    fit_width,
)
from sprouthdl.bundle3 import Bundle3  # your struct type


ArrayElem = Union[Expr, "Array", Bundle3]


class Array:
    def __init__(
        self,
        values: Sequence[Union[ExprLike, "Array", Bundle3, HDLType]],
    ):
        if not values:
            raise ValueError("Array requires at least one element")

        elems: List[ArrayElem] = []
        for v in values:
            if isinstance(v, HDLType):
                elems.append(Wire(v))  # HDLType -> Wire(typ)
            elif isinstance(v, Array):
                elems.append(v)  # nested array
            elif isinstance(v, Bundle3):
                elems.append(v)  # struct element
            else:
                elems.append(as_expr(v))  # ExprLike -> Expr

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

        Works as long as the underlying structure is nested Arrays.
        """
        # Normalize to a tuple of indices (axes)
        if not isinstance(key, tuple):
            key = (key,)

        return self._get_nd(key)

    def _get_nd(self, keys: tuple):
        # No more axes: return this sub-array as-is
        if not keys:
            return self

        idx, *rest = keys

        # -------- Case 1: integer index at this dimension --------
        if isinstance(idx, int):
            elem = self._elems[idx]
            if not rest:
                # Last axis: return scalar element / sub-array / bundle
                return elem
            # More axes: we must recurse into a nested Array
            if not isinstance(elem, Array):
                raise TypeError(f"Too many indices: element at this axis is {type(elem)}, " "not Array")
            return elem._get_nd(tuple(rest))

        # -------- Case 2: slice at this dimension --------
        if isinstance(idx, slice):
            # Slice this axis to get a sublist of elements
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

        # Optionally: handle Ellipsis, etc., here if you want
        raise TypeError(f"Unsupported index type at this axis: {type(idx)}")

    # --------------------------------------------------------------
    # __setitem__ (still 1D for now; can be extended similarly)
    # --------------------------------------------------------------
    def __setitem__(self, idx, value: Union[ExprLike, "Array", Bundle3, HDLType]) -> None:
        if isinstance(idx, tuple):
            # You *can* implement ND assignment similarly to _get_nd,
            # but you said you mainly need selection, so we keep this simple.
            raise NotImplementedError("N-dimensional __setitem__ not implemented")

        if isinstance(value, HDLType):
            coerced: ArrayElem = Wire(value)
        elif isinstance(value, Array):
            coerced = value
        elif isinstance(value, Bundle3):
            coerced = value
        else:
            coerced = as_expr(value)
        self._elems[idx] = coerced

    # --------------------------------------------------------------
    # Width + bit packing (already dimension-agnostic)
    # --------------------------------------------------------------
    def width(self) -> int:
        total = 0
        for elem in self._elems:
            if isinstance(elem, Expr):
                total += elem.typ.width
            elif isinstance(elem, Array):
                total += elem.width()
            elif isinstance(elem, Bundle3):
                total += elem.width()
            else:
                raise TypeError(f"Unsupported element type in width(): {type(elem)}")
        return total

    def to_bits(self) -> Expr:
        parts: List[Expr] = []
        for elem in self._elems:
            if isinstance(elem, Expr):
                parts.append(elem)
            elif isinstance(elem, Array):
                parts.append(elem.to_bits())
            elif isinstance(elem, Bundle3):
                parts.append(elem.to_bits())
            else:
                raise TypeError(f"Unsupported element type in to_bits(): {type(elem)}")
        return Concat(parts)

    def from_bits(self, bits: Expr) -> "Array":
        bit_pos = 0
        for i, elem in enumerate(self._elems):
            if isinstance(elem, Expr):
                w = elem.typ.width
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w
                if isinstance(elem, Signal):
                    elem <<= slice_bits
                else:
                    self._elems[i] = fit_width(slice_bits, elem.typ)
            elif isinstance(elem, Array):
                w = elem.width()
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w
                elem.from_bits(slice_bits)
            elif isinstance(elem, Bundle3):
                w = elem.width()
                slice_bits = bits[bit_pos : bit_pos + w]
                bit_pos += w
                elem.from_bits(slice_bits)
            else:
                raise TypeError(f"Unsupported element type in from_bits(): {type(elem)}")
        return self

    # --------------------------------------------------------------
    # Element-wise assignment (shape-agnostic)
    # --------------------------------------------------------------
    def __ilshift__(self, rhs: "Array"):
        if len(self) != len(rhs):
            raise ValueError("Array.__ilshift__: length mismatch")
        for i in range(len(self)):
            lhs = self._elems[i]
            r = rhs._elems[i]
            if isinstance(lhs, Expr) and isinstance(r, Expr):
                if isinstance(lhs, Signal):
                    lhs <<= r
                else:
                    self._elems[i] = r
            elif isinstance(lhs, Array) and isinstance(r, Array):
                lhs <<= r
            elif isinstance(lhs, Bundle3) and isinstance(r, Bundle3):
                lhs <<= r
            else:
                raise TypeError(f"Incompatible element types in Array.__ilshift__: " f"{type(lhs)} <<= {type(r)}")
        return self
