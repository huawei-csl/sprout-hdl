"""FSM state enumeration for Sprout-HDL.

Declare states as class variables using :func:`state` so the IDE can
see (and autocomplete) every state name::

    class MyFSM(State, encoding="binary"):
        IDLE = state()
        RUN  = state()
        DONE = state()

    reg = m.reg(MyFSM.typ, "state", init=MyFSM.IDLE)

Supported encodings: ``"binary"`` (default), ``"onehot"``, ``"gray"``.
"""

from __future__ import annotations

from sprouthdl.sprouthdl import Const, HDLType, UInt


class _StatePlaceholder:
    """Sentinel returned by :func:`state` to mark class-level state entries."""


def state() -> Const:
    """Declare a state entry inside a :class:`State` subclass.

    The return type is annotated as ``Const`` so that the IDE treats the
    attribute as an expression usable in ``switch_``/``case_``/``==``.
    At class-creation time the placeholder is replaced with the real
    ``Const`` value.
    """
    return _StatePlaceholder()  # type: ignore[return-value]


_ENCODINGS = {"binary", "onehot", "gray"}


class State:
    """Base class for FSM state enumerations.

    Subclass and use :func:`state` for each entry::

        class MyFSM(State, encoding="onehot"):
            IDLE = state()
            RUN  = state()
            DONE = state()

        MyFSM.IDLE   # Const – IDE autocompletes this
        MyFSM.typ    # HDLType matching the encoding width
    """

    typ: HDLType
    encoding: str
    names: list[str]
    _width: int
    _values: dict[str, int]

    def __init_subclass__(cls, encoding: str = "binary", **kwargs):
        super().__init_subclass__(**kwargs)

        if encoding not in _ENCODINGS:
            raise ValueError(f"Unknown encoding '{encoding}', expected one of {_ENCODINGS}")

        # Collect state names in declaration order
        names: list[str] = []
        for attr in list(vars(cls)):
            if isinstance(getattr(cls, attr), _StatePlaceholder):
                names.append(attr)

        if not names:
            return  # allow intermediate base classes with no states

        n = len(names)

        if encoding == "binary":
            values = list(range(n))
            width = max(1, (n - 1).bit_length())
        elif encoding == "onehot":
            values = [1 << i for i in range(n)]
            width = n
        elif encoding == "gray":
            values = [i ^ (i >> 1) for i in range(n)]
            width = max(1, (n - 1).bit_length())

        typ = UInt(width)

        cls.encoding = encoding
        cls.names = names
        cls._width = width
        cls._values = dict(zip(names, values))
        cls.typ = typ

        # Replace placeholders with real Const values
        for name, val in cls._values.items():
            setattr(cls, name, Const(val, typ))

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self.names)}, encoding={self.encoding!r}, width={self._width})"
