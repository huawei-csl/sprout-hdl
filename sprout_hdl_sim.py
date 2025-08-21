# -----------------------------
# Tiny Python-level simulator
# -----------------------------
from sprout_hdl import *

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


class Simulator:
    """
    Cycle-accurate simulator for a single Module.
    - step(): one rising edge of clk (if module has clock)
    - set(): set an input or (optionally) force a reg
    - get(): read any signal (inputs, wires, outputs, regs)
    - reset(): asynchronous reset (like posedge rst)
    - eval(): recompute combinational paths (lazy anyway)
    """

    def __init__(self, module: "Module"):
        self.m = module
        self.inputs = [s for s in self.m._ports if s.kind == "input"]
        self.outputs = [s for s in self.m._ports if s.kind == "output"]
        self.regs = [s for s in self.m._signals if s.kind == "reg"]
        self.wires = [s for s in self.m._signals if s.kind == "wire"]

        self._by_name = {s.name: s for s in self.m._signals}

        # ⬇️ Use ids instead of Signal objects
        self._in: dict[int, int] = {_sid(i): 0 for i in self.inputs}
        self._reg: dict[int, int] = {}

        for r in self.regs:
            init_bits = 0
            if r._init is not None:
                init_bits = self._eval_expr_bits(r._init)
                init_bits = _resize_bits(init_bits, r._init.typ.width, r.typ.width, r._init.typ.signed)
            self._reg[_sid(r)] = _to_bits(init_bits, r.typ.width)

        if self.m.with_reset:
            self._in[_sid(self.m.rst)] = 0
        if self.m.with_clock:
            self._in[_sid(self.m.clk)] = 0

        self._cache_expr: dict[int, int] = {}
        self._cache_sig: dict[int, int] = {}
        self._time_steps = 0

    # -----------------------------
    # Public API
    # -----------------------------

    def set(self, ref, value: int):
        s = self._resolve(ref)
        if s.kind == "input":
            self._in[_sid(s)] = _to_bits(value, s.typ.width)
        elif s.kind == "reg":
            self._reg[_sid(s)] = _to_bits(value, s.typ.width)
        else:
            raise ValueError("Only inputs and regs can be set directly.")
        self._invalidate()
        return self

    def get(self, ref, *, signed: bool | None = None) -> int:
        s = self._resolve(ref)
        bits = self._eval_signal_bits(s)
        if signed is None:
            signed = s.typ.signed
        return _from_bits_signed(bits, s.typ.width) if signed else bits

    def eval(self) -> "Simulator":
        """(Re)compute combinational network. (Lazy by default; this just clears caches.)"""
        self._invalidate()
        # Evaluate all outputs once to populate errors early.
        for y in self.outputs:
            _ = self._eval_signal_bits(y)
        return self

    def step(self, n: int = 1):
        for _ in range(n):
            if self.m.with_clock:
                self._in[_sid(self.m.clk)] = 0
            next_vals = self._compute_next_state()
            for r, v in next_vals.items():
                self._reg[_sid(r)] = v
            if self.m.with_clock:
                self._in[_sid(self.m.clk)] = 1
                self._in[_sid(self.m.clk)] = 0
            self._time_steps += 1
            self._invalidate()
        return self

    def reset(self, asserted: bool = True):
        if not self.m.with_reset:
            return self
        self._in[_sid(self.m.rst)] = 1 if asserted else 0
        if asserted:
            for r in self.regs:
                if r._init is not None:
                    v = self._eval_expr_bits(r._init)
                    v = _resize_bits(v, r._init.typ.width, r.typ.width, r._init.typ.signed)
                else:
                    v = 0
                self._reg[_sid(r)] = _to_bits(v, r.typ.width)
            self._invalidate()
        return self

    def deassert_reset(self):
        if self.m.with_reset:
            self._in[_sid(self.m.rst)] = 0
        return self

    # Convenience
    def peek_outputs(self, *, signed: bool = False) -> dict[str, int]:
        return {y.name: self.get(y, signed=signed) for y in self.outputs}

    # -----------------------------
    # Internals
    # -----------------------------

    def _resolve(self, ref: Union[str, Signal]) -> Signal:
        if isinstance(ref, Signal):
            return ref
        if isinstance(ref, str):
            try:
                return self._by_name[ref]
            except KeyError:
                raise KeyError(f"No signal named '{ref}' in module {self.m.name}.")
        raise TypeError(f"Expected Signal or str, got {type(ref)}")

    def _invalidate(self):
        self._cache_expr.clear()
        self._cache_sig.clear()

    def _compute_next_state(self) -> dict:
        """Compute next-state values for all regs without committing."""
        res: dict[Signal, int] = {}
        rst_high = (self.m.with_reset and self._in.get(_sid(self.m.rst), 0) != 0)

        for r in self.regs:
            if rst_high:
                # Reset path
                if r._init is not None:
                    init_bits = self._eval_expr_bits(r._init)
                    v = _resize_bits(init_bits, r._init.typ.width, r.typ.width, r._init.typ.signed)
                else:
                    v = 0
                res[r] = _to_bits(v, r.typ.width)
            else:
                if r._next is None:
                    raise ValueError(f"Register '{r.name}' has no next-state assignment. Set r.next = ...")
                nxt_bits = self._eval_expr_bits(r._next)
                res[r] = _resize_bits(nxt_bits, r._next.typ.width, r.typ.width, r._next.typ.signed)
        return res

    # ------- Expression evaluation (to bit patterns) -------

    def _eval_signal_bits(self, s: Signal, _visiting: Optional[set] = None) -> int:
        sid = id(s)
        if sid in self._cache_sig:
            return self._cache_sig[sid]

        if s.kind == "input":
            bits = _to_bits(self._in.get(_sid(s), 0), s.typ.width)

        elif s.kind == "reg":
            bits = _to_bits(self._reg.get(_sid(s), 0), s.typ.width)

        elif s.kind in ("wire", "output"):
            if s._driver is None:
                raise ValueError(f"Signal '{s.name}' ({s.kind}) has no driver.")
            if _visiting is None:
                _visiting = set()
            key = ("sig", sid)
            if key in _visiting:
                raise RuntimeError(f"Combinational loop detected involving '{s.name}'.")
            _visiting.add(key)
            drv_bits = self._eval_expr_bits(s._driver, _visiting)
            _visiting.remove(key)
            bits = _resize_bits(drv_bits, s._driver.typ.width, s.typ.width, s._driver.typ.signed)
        else:
            raise TypeError(f"Unknown signal kind: {s.kind}")

        self._cache_sig[sid] = bits
        return bits

    def _eval_expr_bits(self, e: Expr, _visiting: Optional[set] = None) -> int:
        """Evaluate expression e to a bit-pattern of width e.typ.width."""
        eid = id(e)
        if eid in self._cache_expr:
            return self._cache_expr[eid]

        if isinstance(e, Const):
            bits = _to_bits(e.value, e.typ.width)

        elif isinstance(e, Signal):
            bits = self._eval_signal_bits(e, _visiting)

        elif isinstance(e, Op1):
            a = self._eval_expr_bits(e.a, _visiting)
            if e.op == "~":
                bits = _to_bits(~a, e.typ.width)
            else:
                raise NotImplementedError(f"Unary op '{e.op}' not implemented.")

        elif isinstance(e, Op2):
            op = e.op
            tw = e.typ.width

            if op in ("&", "|", "^"):
                # Bitwise: inputs are already Resize'd by op_bit() to match widths
                av = self._eval_expr_bits(e.a, _visiting)
                bv = self._eval_expr_bits(e.b, _visiting)
                if op == "&":
                    bits = _to_bits(av & bv, tw)
                elif op == "|":
                    bits = _to_bits(av | bv, tw)
                else:
                    bits = _to_bits(av ^ bv, tw)

            elif op in ("+", "-"):
                # Extend both operands to result width using their own signedness
                aw = e.a.typ.width
                bw = e.b.typ.width
                av = _resize_bits(self._eval_expr_bits(e.a, _visiting), aw, tw, e.a.typ.signed)
                bv = _resize_bits(self._eval_expr_bits(e.b, _visiting), bw, tw, e.b.typ.signed)
                if op == "+":
                    bits = _to_bits(av + bv, tw)  # two's-complement add
                else:
                    bits = _to_bits(av - bv, tw)

            elif op == "*":
                # Multiply with signedness
                aw = e.a.typ.width
                bw = e.b.typ.width
                a_raw = self._eval_expr_bits(e.a, _visiting)
                b_raw = self._eval_expr_bits(e.b, _visiting)
                a_int = _from_bits_signed(a_raw, aw) if e.a.typ.signed else _to_bits(a_raw, aw)
                b_int = _from_bits_signed(b_raw, bw) if e.b.typ.signed else _to_bits(b_raw, bw)
                prod = a_int * b_int
                bits = _to_bits(prod, tw)

            elif op in ("<<", ">>"):
                av = self._eval_expr_bits(e.a, _visiting)
                bv = self._eval_expr_bits(e.b, _visiting)
                shift = _to_bits(bv, max(e.b.typ.width, 32))  # treat as non-negative small int
                if op == "<<":
                    bits = _to_bits(av << shift, tw)
                else:
                    # Logical right shift on bit pattern
                    # Compute using the source width (logical, not arithmetic)
                    src_w = e.a.typ.width
                    av_src = _to_bits(av, src_w)
                    bits = _to_bits(av_src >> shift, tw)

            elif op in ("==", "!=", "<", "<=", ">", ">="):
                # Compare after extending both to a common width.
                cw = max(e.a.typ.width, e.b.typ.width)
                av_bits = _resize_bits(self._eval_expr_bits(e.a, _visiting), e.a.typ.width, cw, e.a.typ.signed)
                bv_bits = _resize_bits(self._eval_expr_bits(e.b, _visiting), e.b.typ.width, cw, e.b.typ.signed)

                if op in ("==", "!="):
                    # Bitwise equality over common width
                    eq = av_bits == bv_bits
                    val = 1 if (eq if op == "==" else not eq) else 0
                else:
                    # Relational: signed if either is signed
                    signed = e.a.typ.signed or e.b.typ.signed
                    ai = _from_bits_signed(av_bits, cw) if signed else av_bits
                    bi = _from_bits_signed(bv_bits, cw) if signed else bv_bits
                    if op == "<":
                        val = 1 if ai < bi else 0
                    elif op == "<=":
                        val = 1 if ai <= bi else 0
                    elif op == ">":
                        val = 1 if ai > bi else 0
                    else:
                        val = 1 if ai >= bi else 0
                bits = _to_bits(val, e.typ.width)

            else:
                raise NotImplementedError(f"Binary op '{op}' not implemented.")

        elif isinstance(e, Ternary):
            sel = self._eval_expr_bits(e.sel, _visiting)
            # Evaluate chosen branch and resize to the ternary's result type
            chosen = e.a if sel != 0 else e.b
            cbits = self._eval_expr_bits(chosen, _visiting)
            from_w = chosen.typ.width
            bits = _resize_bits(cbits, from_w, e.typ.width, chosen.typ.signed)

        elif isinstance(e, Concat):
            acc = 0
            for p in e.parts:
                pv = self._eval_expr_bits(p, _visiting)
                acc = (acc << p.typ.width) | _to_bits(pv, p.typ.width)
            bits = _to_bits(acc, e.typ.width)

        elif isinstance(e, Slice):
            av = self._eval_expr_bits(e.a, _visiting)
            # Use the full source width, then slice
            shifted = av >> e.lsb
            width = e.typ.width
            bits = _to_bits(shifted, width)

        elif isinstance(e, Resize):
            av = self._eval_expr_bits(e.a, _visiting)
            bits = _resize_bits(av, e.a.typ.width, e.to_width, e.a.typ.signed)            

        else:
            raise TypeError(f"Unsupported Expr subclass: {type(e)}")

        self._cache_expr[eid] = bits
        return bits
