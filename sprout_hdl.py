# sprout_hdl.py
# A tiny, SpinalHDL-inspired EDSL for Python → Verilog
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Union, Sequence

# -----------------------------
# Types
# -----------------------------


@dataclass
class HDLType:
    width: int
    signed: bool = False
    is_bool: bool = False

    def __post_init__(self):
        if self.is_bool:
            self.width = 1
        if self.width < 1:
            raise ValueError("Type width must be >= 1")

    def range_str(self) -> str:
        return "" if self.width == 1 else f"[{self.width-1}:0]"

    def decl_str(self) -> str:
        sign = "signed " if self.signed else ""
        rng = self.range_str()
        return f"{sign}{rng}".strip()


def Bool() -> HDLType:
    return HDLType(1, signed=False, is_bool=True)


def UInt(w: int) -> HDLType:
    return HDLType(w, signed=False, is_bool=False)


def SInt(w: int) -> HDLType:
    return HDLType(w, signed=True, is_bool=False)


# -----------------------------
# Expressions
# -----------------------------


class Expr:
    typ: HDLType

    # Prevent accidental use in Python control flow.
    def __bool__(self):
        raise TypeError("HDL expressions cannot be used as Python booleans. Use mux()/comparators/etc.")

    # Arithmetic
    def __add__(self, other: ExprLike) -> "Expr":
        return op_add(self, as_expr(other))

    def __radd__(self, other: ExprLike) -> "Expr":
        return op_add(as_expr(other), self)

    def __sub__(self, other: ExprLike) -> "Expr":
        return op_sub(self, as_expr(other))

    def __rsub__(self, other: ExprLike) -> "Expr":
        return op_sub(as_expr(other), self)

    def __mul__(self, other: ExprLike) -> "Expr":
        return op_mul(self, as_expr(other))

    def __rmul__(self, other: ExprLike) -> "Expr":
        return op_mul(as_expr(other), self)

    # Bitwise and logic
    def __and__(self, other: ExprLike) -> "Expr":
        return op_bit(self, as_expr(other), "&")

    def __rand__(self, other: ExprLike) -> "Expr":
        return op_bit(as_expr(other), self, "&")

    def __or__(self, other: ExprLike) -> "Expr":
        return op_bit(self, as_expr(other), "|")

    def __ror__(self, other: ExprLike) -> "Expr":
        return op_bit(as_expr(other), self, "|")

    def __xor__(self, other: ExprLike) -> "Expr":
        return op_bit(self, as_expr(other), "^")

    def __rxor__(self, other: ExprLike) -> "Expr":
        return op_bit(as_expr(other), self, "^")

    def __invert__(self) -> "Expr":
        return op_not(self)

    # Shifts (logical). For variable shifts, result width is source width.
    def __lshift__(self, other: ExprLike) -> "Expr":
        return op_shift(self, as_expr(other), "<<")

    def __rshift__(self, other: ExprLike) -> "Expr":
        return op_shift(self, as_expr(other), ">>")

    # Comparators → Bool(1)
    def __eq__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "==")

    def __ne__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "!=")

    def __lt__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "<")

    def __le__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), "<=")

    def __gt__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), ">")

    def __ge__(self, other: ExprLike) -> "Expr":
        return op_cmp(self, as_expr(other), ">=")

    # Indexing / slicing
    def __getitem__(self, sl: Union[int, slice]) -> "Expr":
        if isinstance(sl, int):
            return Slice(self, sl, sl)  # single bit
        if isinstance(sl, slice):
            msb = sl.start
            lsb = sl.stop
            if msb is None or lsb is None:
                raise ValueError("Slice must be [msb:lsb]")
            if msb < lsb:
                raise ValueError("Use descending slice [msb:lsb]")
            return Slice(self, msb, lsb)
        raise TypeError("Unsupported index type")

    def to_verilog(self) -> str:
        raise NotImplementedError


ExprLike = Union[Expr, int, bool]

# -----------------------------
# Leaf nodes
# -----------------------------


class Const(Expr):
    def __init__(self, value: int, typ: HDLType):
        self.value = int(value)
        self.typ = typ

    def to_verilog(self) -> str:
        if self.typ.is_bool:
            return "1'b1" if self.value != 0 else "1'b0"
        base = "sd" if self.typ.signed or self.value < 0 else "d"
        return f"{self.typ.width}'{base}{self.value}"


class Signal(Expr):
    def __init__(self, name: str, typ: HDLType, kind: str, module: "Module"):
        self.name = name
        self.typ = typ
        self.kind = kind  # 'input' | 'output' | 'wire' | 'reg'
        self.module = module
        self._driver: Optional[Expr] = None  # for wire/output
        self._next: Optional[Expr] = None  # for reg
        self._init: Optional[Expr] = None  # for reg

    def __ilshift__(self, rhs: ExprLike):
        """Connect combinational driver: y <<= expr"""
        rhs_e = fit_width(as_expr(rhs), self.typ)
        self._driver = rhs_e
        return self

    # For registers: set next-state
    @property
    def next(self) -> Optional[Expr]:
        return self._next

    @next.setter
    def next(self, rhs: ExprLike):
        if self.kind != "reg":
            raise TypeError("next can only be set on registers")
        self._next = fit_width(as_expr(rhs), self.typ)

    def set_init(self, init: ExprLike):
        if self.kind != "reg":
            raise TypeError("init can only be set on registers")
        self._init = fit_width(as_expr(init), self.typ)

    def to_verilog(self) -> str:
        return self.name
    
    def __repr__(self):
          return f"Signal(name={self.name!r}, kind={self.kind}, typ=<{self.typ.width}{'s' if self.typ.signed else 'u'}>)"


# -----------------------------
# Compound nodes
# -----------------------------


class Op2(Expr):
    def __init__(self, a: Expr, b: Expr, op: str, typ: HDLType):
        self.a = a
        self.b = b
        self.op = op
        self.typ = typ

    def to_verilog(self) -> str:
        return f"({self.a.to_verilog()} {self.op} {self.b.to_verilog()})"


class Op1(Expr):
    def __init__(self, a: Expr, op: str, typ: HDLType):
        self.a = a
        self.op = op
        self.typ = typ

    def to_verilog(self) -> str:
        return f"({self.op}{self.a.to_verilog()})"


class Ternary(Expr):
    def __init__(self, sel: Expr, a: Expr, b: Expr):
        self.sel = sel
        self.a = a
        self.b = b
        # widen to max width, signed if either signed
        w = max(self.a.typ.width, self.b.typ.width)
        s = self.a.typ.signed or self.b.typ.signed
        self.typ = HDLType(w, s, is_bool=False)

    def to_verilog(self) -> str:
        a = fit_width(self.a, self.typ).to_verilog()
        b = fit_width(self.b, self.typ).to_verilog()
        return f"({self.sel.to_verilog()} ? {a} : {b})"


class Concat(Expr):
    def __init__(self, parts: Sequence[Expr]):
        self.parts = list(parts)
        w = sum(p.typ.width for p in self.parts)
        self.typ = HDLType(w, signed=False, is_bool=False)

    def to_verilog(self) -> str:
        inner = ", ".join(p.to_verilog() for p in self.parts)
        return f"{{{inner}}}"


class Slice(Expr):
    def __init__(self, a: Expr, msb: int, lsb: int):
        if msb < lsb:
            raise ValueError("Slice msb must be >= lsb")
        self.a = a
        self.msb = msb
        self.lsb = lsb
        self.typ = HDLType(msb - lsb + 1, signed=False, is_bool=(msb == lsb))

    def to_verilog(self) -> str:
        return f"{self.a.to_verilog()}[{self.msb}:{self.lsb}]"


class Resize(Expr):
    def __init__(self, a: Expr, to_width: int):
        self.a = a
        self.to_width = to_width
        self.typ = HDLType(to_width, signed=a.typ.signed, is_bool=(to_width == 1))

    def to_verilog(self) -> str:
        aw = self.a.typ.width
        tw = self.to_width
        if aw == tw:
            return self.a.to_verilog()
        if aw > tw:
            # truncate LSBs kept (common hardware pattern)
            return f"{self.a.to_verilog()}[{tw-1}:0]"
        # extend
        ext_bits = tw - aw
        if self.a.typ.signed:
            signbit = f"{self.a.to_verilog()}[{aw-1}]"
            return f"{{{{{ext_bits}{{{signbit}}}}}, {self.a.to_verilog()}}}"
        else:
            return f"{{{{{ext_bits}{{1'b0}}}}, {self.a.to_verilog()}}}"


# -----------------------------
# Operator helpers
# -----------------------------


def bits_required(v: int) -> int:
    if v == 0:
        return 1
    if v > 0:
        return v.bit_length()
    return (-v).bit_length() + 1  # include sign


def as_expr(x: ExprLike) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool):
        return Const(1 if x else 0, Bool())
    if isinstance(x, int):
        signed = x < 0
        w = bits_required(x)
        return Const(x, HDLType(w, signed=signed))
    raise TypeError(f"Cannot convert {type(x)} to Expr")


def bitwise_result_type(a: Expr, b: Expr) -> HDLType:
    return HDLType(max(a.typ.width, b.typ.width), signed=False)


def add_result_type(a: Expr, b: Expr) -> HDLType:
    return HDLType(max(a.typ.width, b.typ.width) + 1, signed=a.typ.signed or b.typ.signed)


def mul_result_type(a: Expr, b: Expr) -> HDLType:
    return HDLType(a.typ.width + b.typ.width, signed=a.typ.signed or b.typ.signed)


def fit_width(e: Expr, t: HDLType) -> Expr:
    if e.typ.width == t.width:
        return e
    return Resize(e, t.width)


def op_add(a: Expr, b: Expr) -> Expr:
    t = add_result_type(a, b)
    return Op2(a, b, "+", t)


def op_sub(a: Expr, b: Expr) -> Expr:
    t = add_result_type(a, b)
    return Op2(a, b, "-", t)


def op_mul(a: Expr, b: Expr) -> Expr:
    t = mul_result_type(a, b)
    return Op2(a, b, "*", t)


def op_bit(a: Expr, b: Expr, sym: str) -> Expr:
    t = bitwise_result_type(a, b)
    return Op2(fit_width(a, t), fit_width(b, t), sym, t)


def op_not(a: Expr) -> Expr:
    return Op1(a, "~", HDLType(a.typ.width, signed=False, is_bool=a.typ.is_bool))


def op_shift(a: Expr, b: Expr, sym: str) -> Expr:
    # if b is const, widen on left shift; otherwise keep width
    if isinstance(b, Const) and sym == "<<":
        t = HDLType(a.typ.width + b.value, signed=a.typ.signed)
    else:
        t = HDLType(a.typ.width, signed=a.typ.signed)
    return Op2(a, b, sym, t)


def op_cmp(a: Expr, b: Expr, sym: str) -> Expr:
    return Op2(a, b, sym, Bool())


def mux(sel: ExprLike, a: ExprLike, b: ExprLike) -> Expr:
    return Ternary(as_expr(sel), as_expr(a), as_expr(b))


def cat(*parts: ExprLike) -> Expr:
    return Concat([as_expr(p) for p in parts])


# -----------------------------
# Module and codegen
# -----------------------------


class Module:
    def __init__(self, name: str, with_clock: bool = True, with_reset: bool = True):
        self.name = name
        self.with_clock = with_clock
        self.with_reset = with_reset
        self._signals: List[Signal] = []
        self._ports: List[Signal] = []
        # default clock/reset inputs
        if with_clock:
            self.clk = self.input(Bool(), "clk")
        else:
            self.clk = None
        if with_reset:
            self.rst = self.input(Bool(), "rst")
        else:
            self.rst = None

    # Signal constructors
    def input(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "input", self)
        self._signals.append(s)
        self._ports.append(s)
        return s

    def output(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "output", self)
        self._signals.append(s)
        self._ports.append(s)
        return s

    def wire(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "wire", self)
        self._signals.append(s)
        return s

    def reg(self, typ: HDLType, name: str, init: Optional[ExprLike] = None) -> Signal:
        s = Signal(name, typ, "reg", self)
        if init is not None:
            s.set_init(init)
        self._signals.append(s)
        return s

    # Introspection helpers
    def _ports_of(self, kind: str) -> List[Signal]:
        return [s for s in self._ports if s.kind == kind]

    def _is_port(self, s: "Signal") -> bool:
        # Use identity, not equality, so we don't trigger Expr.__eq__.
        return any(s is p for p in self._ports)

    def _internals_of(self, kind: str) -> List[Signal]:
        # Avoid `s not in self._ports` (it calls __eq__). Use identity instead.
        return [s for s in self._signals if s.kind == kind and not self._is_port(s)]

    # Verilog generation
    def to_verilog(self) -> str:
        # Basic checks
        for s in self._signals:
            if s.kind in ("wire", "output") and s._driver is None:
                # allow un-driven outputs if the user plans to wire them later,
                # but it's safer to warn early.
                if s.kind == "output":
                    raise ValueError(f"Output '{s.name}' has no driver.")
            if s.kind == "reg" and s._next is None:
                # regs must have next state
                raise ValueError(f"Register '{s.name}' has no next-state assignment. Set s.next = ...")

        lines: List[str] = []
        # Ports list
        port_names = [p.name for p in self._ports]
        ports_csv = ", ".join(port_names)
        lines.append(f"module {self.name} ({ports_csv});")

        # Declarations
        for p in self._ports:
            dir_ = "input" if p.kind == "input" else "output"
            sign = "signed " if p.typ.signed else ""
            rng = p.typ.range_str()
            lines.append(f"  {dir_} {sign}{rng} {p.name};")

        # Internals
        wires = self._internals_of("wire")
        regs = self._internals_of("reg")
        for w in wires:
            sign = "signed " if w.typ.signed else ""
            rng = w.typ.range_str()
            lines.append(f"  wire {sign}{rng} {w.name};")
        for r in regs:
            sign = "signed " if r.typ.signed else ""
            rng = r.typ.range_str()
            lines.append(f"  reg {sign}{rng} {r.name};")

        # Combinational assigns for wires/outputs
        for s in [*wires, *self._ports_of("output")]:
            if s._driver is not None:
                rhs = fit_width(s._driver, s.typ).to_verilog()
                lines.append(f"  assign {s.name} = {rhs};")

        # Sequential logic
        if regs:
            if not self.with_clock:
                raise ValueError("Registers present but module has no clock input.")
            sens = f"posedge {self.clk.name}"
            if self.with_reset:
                sens += f" or posedge {self.rst.name}"
            lines.append(f"  always @({sens}) begin")
            if self.with_reset:
                lines.append(f"    if ({self.rst.name}) begin")
                for r in regs:
                    init = r._init.to_verilog() if r._init is not None else f"{r.typ.width}'d0"
                    lines.append(f"      {r.name} <= {init};")
                lines.append("    end else begin")
                for r in regs:
                    lines.append(f"      {r.name} <= {fit_width(r._next, r.typ).to_verilog()};")
                lines.append("    end")
            else:
                for r in regs:
                    lines.append(f"    {r.name} <= {fit_width(r._next, r.typ).to_verilog()};")
            lines.append("  end")

        lines.append("endmodule")
        return "\n".join(lines)

# ----------------------------


def _mask(w: int) -> int:
    return (1 << w) - 1 if w > 0 else 0


def _to_bits(v: int, w: int) -> int:
    return int(v) & _mask(w)


def _from_bits_signed(bits: int, w: int) -> int:
    if w == 0:
        return 0
    sign = (bits >> (w - 1)) & 1
    return bits - (1 << w) if sign else bits


def _resize_bits(bits: int, from_w: int, to_w: int, signed: bool) -> int:
    """Truncate or extend a value in two's complement as needed."""
    bits = _to_bits(bits, from_w)
    if to_w == from_w:
        return bits
    if to_w < from_w:
        # Truncate LSBs kept (matches Verilog slicing)
        return _to_bits(bits, to_w)
    # Extend
    if signed:
        val = _from_bits_signed(bits, from_w)
        return _to_bits(val, to_w)
    return _to_bits(bits, to_w)


def _sid(s: "Signal") -> int:
    return id(s)

def _clsname(o) -> str:
    return o.__class__.__name__

