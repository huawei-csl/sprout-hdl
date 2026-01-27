"""FSM state enumeration for Sprout-HDL."""

from __future__ import annotations

from sprouthdl.sprouthdl import Const, UInt


class State:
    """Lightweight FSM state enumeration with automatic width and encoding.

    Usage::

        fsm = State("IDLE", "RUN", "DONE", encoding="binary")
        reg = m.reg(fsm.typ, "state", init=fsm.IDLE)
        with switch_(reg):
            with case_(fsm.IDLE):
                reg <<= fsm.RUN

    Supported encodings: ``"binary"`` (default), ``"onehot"``, ``"gray"``.
    """

    _ENCODINGS = {"binary", "onehot", "gray"}

    def __init__(self, *names: str, encoding: str = "binary"):
        if not names:
            raise ValueError("State requires at least one state name")
        if encoding not in self._ENCODINGS:
            raise ValueError(f"Unknown encoding '{encoding}', expected one of {self._ENCODINGS}")

        self.encoding = encoding
        self.names = list(names)
        n = len(names)

        # Compute encoded values and width
        if encoding == "binary":
            values = list(range(n))
            width = max(1, (n - 1).bit_length())
        elif encoding == "onehot":
            values = [1 << i for i in range(n)]
            width = n
        elif encoding == "gray":
            values = [i ^ (i >> 1) for i in range(n)]
            width = max(1, (n - 1).bit_length())

        self._width = width
        self.typ = UInt(width)
        self._values = dict(zip(names, values))

        # Expose each state as a Const attribute
        for name, val in self._values.items():
            setattr(self, name, Const(val, self.typ))

    def __len__(self) -> int:
        return len(self.names)

    def __repr__(self) -> str:
        return f"State({', '.join(self.names)}, encoding={self.encoding!r}, width={self._width})"
