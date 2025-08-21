# sprout_to_aiger.py
# Export a SproutHDL Module to an AIG in AIGER ASCII (.aag) format
# Works with your 'sprout_hdl.py' DSL (duck-typing; no hard import required).
# Tested conceptually with arithmetic & logic-heavy designs (incl. float16/bfloat16 MACs).
from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any, Iterable

from aag_loader_writer import AbstractAdapter, conv_aag_into_graph
from sprout_hdl import Module, SInt, Signal, UInt, Bool, Const, Op1, Op2, cat, mux

# ---- AIGER literals helpers -------------------------------------------------

# Literals are non-negative ints; LSB=1 means inverted edge.
# 0  => const 0 (false)
# 1  => const 1 (true)  == ~0
# even numbers 2*k are regular variables (node indices start at 1)


def lit_const0() -> int:
    return 0


def lit_const1() -> int:
    return 1


def lit_not(a: int) -> int:
    return a ^ 1


# ---- AIG writer with structural hashing -------------------------------------


class _AIG:
    """
    Low-level AIG builder that only knows about 1-bit literals and 2-input ANDs.
    Builds an internal representation suitable for writing AIGER ASCII.
    """

    def __init__(self):
        self.next_var: int = 1  # AIGER variable indices start at 1
        self.inputs: List[int] = []  # list of even literals (2*var)
        self.latches: List[Tuple[int, int]] = []  # (curr_even_lit, next_lit)
        self.outputs: List[int] = []  # general literals
        self.ands: List[Tuple[int, int, int]] = []  # (lhs_even_lit, rhs_lit0, rhs_lit1)

        # symbol tables (same order as the corresponding lists)
        self.sym_inputs: List[str] = []
        self.sym_latches: List[str] = []
        self.sym_outputs: List[str] = []

        # structural hash for AND nodes
        self._and_cache: Dict[Tuple[int, int], int] = {}

    # variable allocation (returns even literal)
    def _new_var(self) -> int:
        v = self.next_var
        self.next_var += 1
        return 2 * v

    # 2-input AND with simplifications + structural hashing
    def mk_and(self, a: int, b: int) -> int:
        # trivial cases
        if a == lit_const0() or b == lit_const0():
            return lit_const0()
        if a == lit_const1():
            return b
        if b == lit_const1():
            return a
        if a == b:
            return a
        if a == lit_not(b):
            return lit_const0()

        # canonical order on RHS for caching
        if a > b:
            a, b = b, a
        key = (a, b)
        res = self._and_cache.get(key)
        if res is not None:
            return res

        lhs = self._new_var()
        self.ands.append((lhs, a, b))
        self._and_cache[key] = lhs
        return lhs

    # Derived ops via AND+NOT
    def mk_or(self, a: int, b: int) -> int:
        # a | b == ~(~a & ~b)
        return lit_not(self.mk_and(lit_not(a), lit_not(b)))

    def mk_xor(self, a: int, b: int) -> int:
        # (a & ~b) | (~a & b)
        t0 = self.mk_and(a, lit_not(b))
        t1 = self.mk_and(lit_not(a), b)
        return self.mk_or(t0, t1)

    def mk_xnor(self, a: int, b: int) -> int:
        return lit_not(self.mk_xor(a, b))

    def mk_maj(self, a: int, b: int, c: int) -> int:
        # (a&b) | (a&c) | (b&c)
        t0 = self.mk_and(a, b)
        t1 = self.mk_and(a, c)
        t2 = self.mk_and(b, c)
        return self.mk_or(self.mk_or(t0, t1), t2)

    def mk_ite(self, s: int, t: int, e: int) -> int:
        # s ? t : e  ==  (s & t) | (~s & e)
        return self.mk_or(self.mk_and(s, t), self.mk_and(lit_not(s), e))

    # bit-vector helpers (LSB-first lists of literals)
    def bv_zero(self, w: int) -> List[int]:
        return [lit_const0()] * w

    def bv_from_int(self, value: int, w: int) -> List[int]:
        return [lit_const1() if ((value >> i) & 1) else lit_const0() for i in range(w)]

    def bv_not(self, a: List[int]) -> List[int]:
        return [lit_not(x) for x in a]

    def bv_or(self, a: List[int], b: List[int]) -> List[int]:
        w = max(len(a), len(b))
        a = self._extend(a, w)
        b = self._extend(b, w)
        return [self.mk_or(a[i], b[i]) for i in range(w)]

    def bv_and(self, a: List[int], b: List[int]) -> List[int]:
        w = max(len(a), len(b))
        a = self._extend(a, w)
        b = self._extend(b, w)
        return [self.mk_and(a[i], b[i]) for i in range(w)]

    def bv_xor(self, a: List[int], b: List[int]) -> List[int]:
        w = max(len(a), len(b))
        a = self._extend(a, w)
        b = self._extend(b, w)
        return [self.mk_xor(a[i], b[i]) for i in range(w)]

    def bv_mux(self, s: int, a: List[int], b: List[int]) -> List[int]:
        w = max(len(a), len(b))
        a = self._extend(a, w)
        b = self._extend(b, w)
        return [self.mk_ite(s, a[i], b[i]) for i in range(w)]

    def bv_add(self, a: List[int], b: List[int], cin: int = lit_const0(), w_out: Optional[int] = None) -> Tuple[List[int], int]:
        w = max(len(a), len(b))
        a = self._extend(a, w)
        b = self._extend(b, w)
        if w_out is None:
            w_out = w + 1
        out: List[int] = []
        c = cin
        for i in range(w_out):
            ai = a[i] if i < len(a) else lit_const0()
            bi = b[i] if i < len(b) else lit_const0()
            xab = self.mk_xor(ai, bi)
            s = self.mk_xor(xab, c)
            out.append(s)
            c = self.mk_or(self.mk_and(ai, bi), self.mk_and(c, xab))
        return out[:w_out], c

    def bv_sub(self, a: List[int], b: List[int], w_out: Optional[int] = None) -> Tuple[List[int], int]:
        # a - b = a + (~b) + 1 ; returns (diff, carry_out). carry_out==0 -> borrow (a<b)
        nb = [lit_not(x) for x in b]
        return self.bv_add(a, nb, cin=lit_const1(), w_out=w_out)

    def bv_ult(self, a: List[int], b: List[int]) -> int:
        # Unsigned less-than based on borrow of a - b
        w = max(len(a), len(b))
        _, c_out = self.bv_sub(self._extend(a, w), self._extend(b, w), w_out=w)
        return lit_not(c_out)  # borrow if no carry

    def bv_slt(self, a: List[int], b: List[int]) -> int:
        # Signed less-than: if signs differ, a_sign; else unsigned lt
        wa = len(a)
        wb = len(b)
        w = max(wa, wb)
        a = self._sext(a, w)
        b = self._sext(b, w)
        sa = a[w - 1]
        sb = b[w - 1]
        sign_diff = self.mk_xor(sa, sb)
        u = self.bv_ult(a, b)
        return self.mk_ite(sign_diff, sa, u)

    def bv_eq(self, a: List[int], b: List[int]) -> int:
        w = max(len(a), len(b))
        a = self._extend(a, w)
        b = self._extend(b, w)
        eq = lit_const1()
        for i in range(w):
            eq = self.mk_and(eq, self.mk_xnor(a[i], b[i]))
        return eq

    def bv_shift_left(self, a: List[int], sh: List[int], w_out: int) -> List[int]:
        # Barrel shifter (logical), mux-based; sh is LSB-first
        res = a[:]
        res = self._zext(res, w_out)
        step = 1
        for bit in sh:
            if step >= w_out:
                break
            shifted = [lit_const0()] * step + res[: w_out - step]
            res = self.bv_mux(bit, shifted, res)
            step <<= 1
        return res

    def bv_shift_right(self, a: List[int], sh: List[int], w_out: int) -> List[int]:
        res = self._zext(a[:], w_out)
        step = 1
        for bit in sh:
            if step >= w_out:
                break
            shifted = res[step:] + [lit_const0()] * step
            res = self.bv_mux(bit, shifted, res)
            step <<= 1
        return res

    def bv_mul(self, a: List[int], b: List[int], w_out: int) -> List[int]:
        # Shift-add array multiplier (unsigned; two's complement works bitwise equally for product bits)
        # Produces full product then truncates to w_out LSBs.
        wa = len(a)
        wb = len(b)
        acc = self.bv_zero(wa + wb)
        for j in range(wb):
            row = [self.mk_and(ai, b[j]) for ai in a]
            row = [lit_const0()] * j + row + [lit_const0()] * j
            acc, _ = self.bv_add(acc, self._extend(row, len(acc)), w_out=len(acc))
        return acc[:w_out]

    # internal helpers
    def _extend(self, v: List[int], w: int) -> List[int]:
        if len(v) >= w:
            return v[:w]
        return v + [lit_const0()] * (w - len(v))

    def _zext(self, v: List[int], w: int) -> List[int]:
        return self._extend(v, w)

    def _sext(self, v: List[int], w: int) -> List[int]:
        if len(v) >= w:
            return v[:w]
        sign = v[-1] if v else lit_const0()
        return v + [sign] * (w - len(v))


# ---- Exporter: SproutHDL Module -> AIGER -------------------------------------


class AigerExporter:
    """
    Bit-blasts a SproutHDL Module into an AIG, then writes AIGER ASCII (.aag).
    - Supports: Const, Signal (input/output/wire/reg), Op1(~), Op2(& | ^ + - * << >> == != < <= > >=),
                Ternary (mux), Concat, Slice, Resize.
    - Registers become AIGER latches; async reset is encoded in next-state as ITE(rst, init, next).
      (Per AIGER v20071012, latches _initialize to 0_.)
    """

    def __init__(self, module: Any):
        self.m = module
        self.aig = _AIG()

        # id-based maps so we never invoke __eq__
        self._sig_bits: Dict[int, List[int]] = {}  # id(signal) -> bits (LSB-first)
        self._visiting: set = set()  # detect comb loops in wires
        self._reg_list: List[Any] = []  # actual Signal objects of kind 'reg' in allocation order

        # cache expressions to bits (by id)
        self._expr_cache: Dict[int, List[int]] = {}

    # ---- public API

    def write_aag(self, path: str, *, flatten_outputs: bool = True) -> None:
        """Builds AIG from module and writes an .aag file."""
        self._build_network(flatten_outputs=flatten_outputs)
        self._write_aag_file(path)
        
    def get_aag(self) -> List[str]:
        """Builds AIG from module and returns the AIGER ASCII lines."""
        self._build_network(flatten_outputs=True)
        return self._get_aag_lines()

    # ---- network construction

    def _build_network(self, *, flatten_outputs: bool):
        # 1) allocate all input bits first (AIGER requires inputs before gates)
        for p in self.m._ports:
            if p.kind == "input":
                bits = []
                for i in range(p.typ.width):
                    lit = self.aig._new_var()
                    self.aig.inputs.append(lit)
                    self.aig.sym_inputs.append(f"{p.name}[{i}]")
                    bits.append(lit)
                self._sig_bits[id(p)] = bits

        # 2) allocate latches (reg current bits)
        for s in self.m._signals:
            if s.kind == "reg":
                bits = []
                for i in range(s.typ.width):
                    q = self.aig._new_var()  # current state literal
                    bits.append(q)
                    self.aig.sym_latches.append(f"{s.name}[{i}]")
                self._sig_bits[id(s)] = bits
                self._reg_list.append(s)

        # 3) compute next-state for latches
        rst_lit = None
        if getattr(self.m, "with_reset", False):
            # 'rst' is a Bool input in Module constructor if with_reset=True
            rst_bits = self._bits_of_signal(self.m.rst)
            rst_lit = rst_bits[0]

        for s in self._reg_list:
            if s._next is None:
                raise ValueError(f"Register '{s.name}' has no next-state assignment.")
            next_bits = self._eval_expr_bits(s._next)
            next_bits = self._fit_bits(next_bits, s.typ.width, signed=getattr(s.typ, "signed", False))

            # build ITE(rst, init, next) if reset present
            if rst_lit is not None:
                if s._init is not None:
                    init_bits = self._eval_expr_bits(s._init)
                    init_bits = self._fit_bits(init_bits, s.typ.width, signed=getattr(s._init.typ, "signed", False))
                else:
                    init_bits = [lit_const0()] * s.typ.width
                next_bits = self.aig.bv_mux(rst_lit, init_bits, next_bits)

            q_bits = self._bits_of_signal(s)
            assert len(q_bits) == len(next_bits)
            for qb, dn in zip(q_bits, next_bits):
                self.aig.latches.append((qb, dn))

        # 4) build outputs (bit-blast drivers)
        for p in self.m._ports:
            if p.kind != "output":
                continue
            drv_bits = self._eval_expr_bits(p._driver) if p._driver is not None else [lit_const0()] * p.typ.width
            drv_bits = self._fit_bits(drv_bits, p.typ.width, signed=getattr(p.typ, "signed", False))

            if flatten_outputs or p.typ.width == 1:
                for i, b in enumerate(drv_bits):
                    self.aig.outputs.append(b)
                    self.aig.sym_outputs.append(f"{p.name}[{i}]")
            else:
                # If someone wants one literal per word, they'd need an encoder;
                # Spec requires single-bit outputs, so we default to flattened.
                for i, b in enumerate(drv_bits):
                    self.aig.outputs.append(b)
                    self.aig.sym_outputs.append(f"{p.name}[{i}]")

    # ---- expression bit-blasting

    def _bits_of_signal(self, s: Any) -> List[int]:
        key = id(s)
        if key in self._sig_bits:
            return self._sig_bits[key]
        # inputs & regs are pre-allocated; wires/outputs are computed from drivers
        if s.kind in ("wire", "output"):
            # protect against comb loops
            vk = ("sig", key)
            if vk in self._visiting:
                raise RuntimeError(f"Combinational loop involving '{s.name}'.")
            self._visiting.add(vk)
            bits = self._eval_expr_bits(s._driver)
            bits = self._fit_bits(bits, s.typ.width, signed=getattr(s.typ, "signed", False))
            self._visiting.remove(vk)
            self._sig_bits[key] = bits
            return bits
        raise TypeError(f"Unknown signal kind: {s.kind}")

    def _fit_bits(self, bits: List[int], w: int, *, signed: bool) -> List[int]:
        if len(bits) == w:
            return bits
        if len(bits) > w:
            return bits[:w]
        return self.aig._sext(bits, w) if signed else self.aig._zext(bits, w)

    def _eval_expr_bits(self, e: Any) -> List[int]:
        if e is None:
            return [lit_const0()]
        eid = id(e)
        if eid in self._expr_cache:
            return self._expr_cache[eid]

        k = e.__class__.__name__

        if k == "Const":
            w = e.typ.width
            bits = self.aig.bv_from_int(getattr(e, "value", 0), w)

        elif k == "Signal":
            bits = self._bits_of_signal(e)

        elif k == "Op1":
            assert e.op == "~", f"Unsupported unary op {e.op}"
            a = self._eval_expr_bits(e.a)
            bits = self.aig.bv_not(a)

        elif k == "Op2":
            op = e.op
            a = self._eval_expr_bits(e.a)
            b = self._eval_expr_bits(e.b)
            w_out = e.typ.width
            signed = getattr(e.a.typ, "signed", False) or getattr(e.b.typ, "signed", False)

            if op == "&":
                bits = self.aig.bv_and(a, b)
            elif op == "|":
                bits = self.aig.bv_or(a, b)
            elif op == "^":
                bits = self.aig.bv_xor(a, b)
            elif op == "+":
                bits, _ = self.aig.bv_add(a, b, w_out=w_out)
            elif op == "-":
                bits, _ = self.aig.bv_sub(a, b, w_out=w_out)
            elif op == "*":
                bits = self.aig.bv_mul(a, b, w_out=w_out)
            elif op == "<<":
                sh = b  # variable shift
                bits = self.aig.bv_shift_left(a, sh, w_out=w_out)
            elif op == ">>":
                sh = b
                bits = self.aig.bv_shift_right(a, sh, w_out=w_out)
            elif op in ("==", "!=", "<", "<=", ">", ">="):
                # produce Bool(1) -> single literal as 1-bit vector
                if op in ("==", "!="):
                    eq = self.aig.bv_eq(a, b)
                    lit = eq if op == "==" else lit_not(eq)
                else:
                    lt = self.aig.bv_slt(a, b) if signed else self.aig.bv_ult(a, b)
                    if op == "<":
                        lit = lt
                    elif op == "<=":
                        lit = lit_not(self.aig.bv_ult(b, a) if not signed else self.aig.bv_slt(b, a))
                    elif op == ">":
                        lit = self.aig.bv_ult(b, a) if not signed else self.aig.bv_slt(b, a)
                    else:
                        lit = lit_not(lt)
                bits = [lit]
            else:
                raise NotImplementedError(f"Unsupported binary op '{op}'")

            # fit result vector
            bits = self._fit_bits(bits, w_out, signed=signed if op not in ("==", "!=", "<", "<=", ">", ">=") else False)

        elif k == "Ternary":
            sel = self._eval_expr_bits(e.sel)[0]
            a = self._eval_expr_bits(e.a)
            b = self._eval_expr_bits(e.b)
            w_out = e.typ.width
            a = self._fit_bits(a, w_out, signed=getattr(e.a.typ, "signed", False))
            b = self._fit_bits(b, w_out, signed=getattr(e.b.typ, "signed", False))
            bits = self.aig.bv_mux(sel, a, b)

        elif k == "Concat":
            # parts = [MSB ... LSB] ; our vectors are LSB-first
            vec: List[int] = []
            for part in reversed(e.parts):
                pb = self._eval_expr_bits(part)
                vec.extend(pb)
            bits = vec

        elif k == "Slice":
            base = self._eval_expr_bits(e.a)
            # e.msb, e.lsb inclusive; our vectors LSB-first
            bits = base[e.lsb : e.msb + 1]

        elif k == "Resize":
            a = self._eval_expr_bits(e.a)
            bits = self._fit_bits(a, e.to_width, signed=getattr(e.a.typ, "signed", False))

        else:
            raise TypeError(f"Unsupported Expr subclass: {type(e)}")

        self._expr_cache[eid] = bits
        return bits

    # ---- file I/O
    def _write_aag_file(self, path: str) -> None:
        lines = self._get_aag_lines()
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            
    def _get_aag_lines(self) -> None:
        I = len(self.aig.inputs)
        L = len(self.aig.latches)
        O = len(self.aig.outputs)
        A = len(self.aig.ands)
        M = I + L + A  # by construction we never create unused vars

        lines: List[str] = []
        lines.append(f"aag {M} {I} {L} {O} {A}")

        # inputs, latches, outputs
        lines.extend(str(l) for l in self.aig.inputs)
        lines.extend(f"{l} {n}" for (l, n) in self.aig.latches)
        lines.extend(str(o) for o in self.aig.outputs)

        # AND gates
        lines.extend(f"{lhs} {r0} {r1}" for (lhs, r0, r1) in self.aig.ands)

        # symbol table
        if self.aig.sym_inputs or self.aig.sym_latches or self.aig.sym_outputs:
            for i, name in enumerate(self.aig.sym_inputs):
                lines.append(f"i{i} {name}")
            for i, name in enumerate(self.aig.sym_latches):
                lines.append(f"l{i} {name}")
            for i, name in enumerate(self.aig.sym_outputs):
                lines.append(f"o{i} {name}")

        # comment
        lines.append("c")
        lines.append(f"generated from SproutHDL module '{self.m.name}'")

        return lines


# ---- public convenience function --------------------------------------------


def export_module_to_aiger(module: Any, file_path: str, *, flatten_outputs: bool = True) -> None:
    """
    Usage:
        from sprout_to_aiger import export_module_to_aiger
        export_module_to_aiger(my_module, "out.aag")
    """
    AigerExporter(module).write_aag(file_path, flatten_outputs=flatten_outputs)


# ---------- SproutHDL adapter ----------
class SproutHDLAdapter(AbstractAdapter):
    """
    Build a SproutHDL Module from an AAG.
    - graph must be a Module instance.
    - nodes are Sprout expressions (Signal or Expr); we don’t force wires for every AND.
    """
    def __init__(self, module: Module):
        super().__init__(module)
        self.m: Module = module
        self._pi_count = 0
        self._po_count = 0

    def _bit_type(self):
        return Bool()

    def pi(self, name: Optional[str] = None):
        if name is None:
            name = f"pi{self._pi_count}"
        self._pi_count += 1
        return self.m.input(self._bit_type(), name)

    def po(self, node, name: Optional[str] = None):
        if name is None:
            name = f"po{self._po_count}"
        self._po_count += 1
        y = self.m.output(self._bit_type(), name)
        y <<= node
        return y

    def buf(self, node):
        # no-op in Sprout; expressions can be reused freely
        return node

    def const(self, value: bool):
        # Leverage Sprout’s int→Const coercion (1-bit)
        #return 1 if value else 0
        return Const(1, Bool()) if value else Const(0, Bool())

    def NOT(self, node):
        return ~node

    def AND(self, a, b):
        return a & b

class AigerImporter:

    def __init__(self, lines: List[str]):
        self.lines = lines 

    def get_sprout_module(self, name: str | None = None) -> Module:
        if name is None:
            name = "ImportedModule"
        return conv_aag_into_graph(self.lines,
                                   Module(name, with_clock=False, with_reset=False),
                                   adapter=SproutHDLAdapter)


def main_small_tst():

    m = Module("LogicDemo", with_clock=False, with_reset=False)
    x0 = m.input(Bool(), "x0")
    x1 = m.input(Bool(), "x1")
    x2 = m.input(Bool(), "x2")
    y = m.output(Bool(), "y")

    round = x0 & (x1 | x2)

    y <<= round

    # export to aiger lines
    exporter = AigerExporter(m)
    aag_lines = exporter.get_aag()

    # import back to SproutHDL Module
    importer = AigerImporter(aag_lines)
    sprout_module = importer.get_sprout_module()

    # print hdl
    print("Original Module Verilog:")
    print(m.to_verilog())
    print("\nImported Module Verilog:")
    print(sprout_module.to_verilog())


if __name__ == "__main__":
    main_small_tst()
