import abc
from dataclasses import dataclass, make_dataclass
import hashlib
import random
import time
from sprouthdl.sprouthdl import Bool, Expr, ExprLike, HDLType, Signal, UInt, cat, fit_width, _SHARED, reset_shared_cache


from typing import Any, Dict, List, Optional
from dataclasses import is_dataclass, fields

from sprouthdl.aig.aig_yosys import verilog_to_aag_lines_via_yosys

try:  # Python 3.10 compatibility
    from typing import Self  # type: ignore
except ImportError:
    from typing_extensions import Self  # type: ignore

from sprouthdl.sprouthdl_analyzer import _Analyzer, GraphReport


class Component(abc.ABC):

    io: dataclass | Dict

    # define attribute name
    @property
    def name(self) -> str:
        return self.__class__.__name__

    # @abc.abstractmethod
    def elaborate(self) -> None:  # pragma: no cover - structural hook
        # raise NotImplementedError
        pass

    # convenience helpers -------------------------------------------------------

    def to_module(self, name: Optional[str] = None, with_clock: bool = False, with_reset: bool = False) -> 'Module':
        module = Module(
            name or f"comp_{get_rand_hash()}",
            with_clock=with_clock,
            with_reset=with_reset,
        )

        def iter_io_values(io):
            if isinstance(io, dict):
                return io.values()
            if is_dataclass(io):
                return (getattr(io, f.name) for f in fields(io))
            if hasattr(io, "_fields"):              # namedtuple
                return (getattr(io, n) for n in io._fields)
            return vars(io).values()                # normal objects

        for sig in iter_io_values(self.io):
            sig: Signal

            # if is clock/reset assign to module clk/rst
            if sig.name == "clk":
                if module.clk is None:
                    module.clk = sig
                else:
                    raise(ValueError("Module already has a clock signal"))
                continue
            if sig.name == "rst":
                if module.rst is None:
                    module.rst = sig
                else:
                    raise(ValueError("Module already has a reset signal"))
                continue

            if sig.kind == "input":
                module.add_input(sig)
            elif sig.kind == "output":
                module.add_output(sig)
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        module.component = self # can be used for debugging
        module._collect_signals_from_outputs([s for s in iter_io_values(self.io) if s.kind == "output"])
        return module 

    def from_module(self, module: 'Module', make_internal=False, group=False) -> Self:
        if group:
            IOCollector().group(module, self.get_spec())

        # find signals in module and assign to io
        io_fields = {}
        for sig in module._signals:
            if sig.kind in ('input', 'output'):
                io_fields[sig.name] = sig
        # instance.io = dataclass(type('IO', (), io_fields))()  # type: ignore
        # instance.io = make_dataclass("IO", io_fields)
        # instance.io = io_fields
        for io_name, io_sig in io_fields.items():
            setattr(self.io, io_name, io_sig)
            # or self.io.__dict__[io_name] <<= io_sig
            # io_sig.kind = 'wire'  # change to wire in module
        self.elaborate()  # re-elaborate to rebuild internal structure
        if make_internal:
            self.make_internal()

    def from_verilog(self, verilog_str: str, top=None, group=True) -> Self:
        from sprouthdl.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

        aag_lines = verilog_to_aag_lines_via_yosys(verilog_str, top=top, embed_symbols=True, no_startoffset=True)
        self.from_aag_lines(aag_lines, group=group)

    def from_aig_file(self, aig_path: str, map_file: str|None = None, group=True, make_internal=False) -> Self:
        from sprouthdl.aig.aig_yosys import aig_file_to_aag_lines_via_yosys

        aag_lines = aig_file_to_aag_lines_via_yosys(aig_path, map_file=map_file)
        self.from_aag_lines(aag_lines, group=group, make_internal=make_internal)
        return self

    def from_aag_lines(self, aag_lines: List[str], group=True, make_internal=True) -> Self:
        from sprouthdl.sprouthdl_aiger import AigerImporter

        m = AigerImporter(aag_lines).get_sprout_module()
        self.from_module(m, make_internal=make_internal, group=group)

    def make_internal(self) -> Self:
        # go through all signals in io and change to 'wire'
        ios_dict = self.io if isinstance(self.io, dict) else self.io.__dict__
        for sig in ios_dict.values():
            if sig.kind in ('input', 'output'):
                sig.kind = 'wire'
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        return self

    def get_spec(self) -> Dict[str, UInt]:
        return gen_spec(self)


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
        self.component : Optional["Component"] = None

    # Signal constructors
    def input(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "input") #, self)
        self._signals.append(s)
        self._ports.append(s)
        return s
    
    def add_input(self, signal: Signal) -> None:
        if signal.kind != "input":
            # change to input
            signal.kind = "input"
        if id(signal) in [id(s) for s in self._signals]:
            raise ValueError("Signal already exists in module.")
        self._signals.append(signal)
        self._ports.append(signal)

    def output(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "output") #, self)
        self._signals.append(s)
        self._ports.append(s)
        return s
    
    def add_output(self, signal: Signal) -> None:
        if signal.kind != "output":
            # change to output
            signal.kind = "output"
        if id(signal) in [id(s) for s in self._signals]:
            raise ValueError("Signal already exists in module.")
        self._signals.append(signal)
        self._ports.append(signal)

    def wire(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "wire") #, self)
        self._signals.append(s)
        return s

    def reg(self, typ: HDLType, name: str, init: Optional[ExprLike] = None) -> Signal:
        s = Signal(name, typ, "reg") #, self)
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
    
    def get_spec(self) -> Dict[str, UInt]:
        spec = {}
        for p in self._ports:
            spec[p.name] = p.typ
        return spec
    
    def collect_signals(self) -> None:
        """
        Walk the design starting from outputs
        """
        self._collect_signals_from_outputs(self._ports_of("output"))

    def _collect_signals_from_outputs(self, outputs: List[Signal]) -> None:
        """
        Walk the design starting from outputs and registers, pulling every reachable
        Signal into _signals. Internal signals must be wire/reg; encountering an
        input/output that is not a port raises. Name collisions are avoided by
        suffixing internal signal names.
        """
        # Track names already in use (ports first, keep their names stable)
        name_to_sig: Dict[str, Signal] = {p.name: p for p in self._ports}
        visited_signals = set()
        visited_exprs = set()
        port_ids = {id(p) for p in self._ports}

        def uniquify_internal(sig: Signal) -> None:
            if id(sig) in port_ids:
                return
            base = sig.name
            if name_to_sig.get(base, None) is sig:
                return
            if base not in name_to_sig:
                name_to_sig[base] = sig
                return
            idx = 1
            while True:
                candidate = f"{base}_{idx}"
                if candidate not in name_to_sig:
                    sig.name = candidate
                    name_to_sig[candidate] = sig
                    return
                idx += 1

        def visit_expr(e: Expr | None) -> None:
            if e is None:
                return
            if isinstance(e, Signal):
                visit_signal(e)
                return
            if id(e) in visited_exprs:
                return
            visited_exprs.add(id(e))
            # Recurse through common expression fields
            for attr in ("a", "b", "sel"):
                if hasattr(e, attr):
                    visit_expr(getattr(e, attr))
            if hasattr(e, "parts"):
                for p in e.parts:
                    visit_expr(p)
            if hasattr(e, "_driver"):
                visit_expr(getattr(e, "_driver"))

        def visit_signal(sig: Signal) -> None:
            if id(sig) in visited_signals:
                return
            visited_signals.add(id(sig))

            if id(sig) not in port_ids:
                if sig.kind in ("input", "output"):
                    raise ValueError(f"Internal signal '{sig.name}' has port kind '{sig.kind}'. Use wire/reg for internals.")
                uniquify_internal(sig)
                if not any(sig is s for s in self._signals):
                    self._signals.append(sig)

            if sig._driver is not None:
                visit_expr(sig._driver)

        for out in outputs:
            visit_signal(out)
    
    def to_component(self) -> Component:
        """
        Create a lightweight Component wrapper that exposes this module's ports as
        the component IO. This mirrors Component.to_module() in spirit by sharing
        the same Signal objects for ports (no copying), so drivers and expressions
        remain intact.

        Notes:
        - Field names in the IO dataclass must be valid Python identifiers. If a
          port name contains characters like '[', ']', etc. (e.g., bit-ports
          "a[0]"), the field name is sanitized (e.g., "a_0"). The underlying
          Signal keeps its original .name, so codegen and analysis are unaffected.
        """
        # Build IO dataclass fields for inputs/outputs
        port_signals = [p for p in self._ports if p.kind in ("input", "output")]

        def sanitize(n: str) -> str:
            s = ''.join(c if (c.isalnum() or c == '_') else '_' for c in n)
            if not s or s[0].isdigit():
                s = f"p_{s}"
            return s

        # Ensure unique field names after sanitization
        used: Dict[str, int] = {}
        fields: List[tuple[str, type]] = []
        values: Dict[str, Signal] = {}
        for sig in port_signals:
            base = sanitize(sig.name)
            idx = used.get(base, 0)
            used[base] = idx + 1
            field_name = base if idx == 0 else f"{base}_{idx}"
            fields.append((field_name, Signal))
            values[field_name] = sig

        IO = make_dataclass("IO", fields)

        # Minimal concrete Component instance with populated IO
        #class _ModuleWrappedComponent(Component):
        #    pass

        comp = Component() #_ModuleWrappedComponent()
        comp.io = IO(**values)
        return comp

    # Verilog generation
    def to_verilog_lines(self) -> list[str]:
        # Basic checks
        for s in self._signals:
            if s.kind in ("wire", "output") and s._driver is None:
                if s.kind == "output":
                    raise ValueError(f"Output '{s.name}' has no driver.")
            if s.kind == "reg" and s._driver is None:
                raise ValueError(f"Register '{s.name}' has no next-state assignment.")

        lines: List[str] = []
        # Ports list
        port_names = [p.name for p in self._ports]
        ports_csv = ", ".join(port_names)
        lines.append(f"module {self.name} ({ports_csv});")

        # Declarations
        lines.append("// Ports")
        for p in self._ports:
            dir_ = "input" if p.kind == "input" else "output"
            sign = "signed " if p.typ.signed else ""
            rng = p.typ.range_str()
            lines.append(f"  {dir_} {sign}{rng} {p.name};")

        # Internals
        #wires = self._internals_of("wire") + _SHARED.wires
        # instead of the above merge to avoid duplication if called multiple times
        wires = self._internals_of("wire")
        wires += [s for s in _SHARED.wires if not any(s is w for w in wires)]
        
        regs = self._internals_of("reg")
        lines.append('// Wires')
        for w in wires:
            sign = "signed " if w.typ.signed else ""
            rng = w.typ.range_str()
            lines.append(f"  wire {sign}{rng} {w.name};")
        lines.append('// Registers')
        for r in regs:
            sign = "signed " if r.typ.signed else ""
            rng = r.typ.range_str()
            lines.append(f"  reg {sign}{rng} {r.name};")
        # Combinational assigns for wires/outputs
        lines.append("// Combinational assignments")
        for s in [*wires, *self._ports_of("output")]:
            if s._driver is not None:
                rhs = fit_width(s._driver, s.typ).to_verilog()
                lines.append(f"  assign {s.name} = {rhs};")

        # Sequential logic
        lines.append("// Sequential logic")
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
                    lines.append(f"      {r.name} <= {fit_width(r._driver, r.typ).to_verilog()};")
                lines.append("    end")
            else:
                for r in regs:
                    lines.append(f"    {r.name} <= {fit_width(r._driver, r.typ).to_verilog()};")
            lines.append("  end")

        lines.append("endmodule")
        return lines
    
    def to_verilog(self) -> str:
        lines = self.to_verilog_lines() + [""]  # final newline
        return "\n".join(lines)
    
    def to_verilog_file(self, filepath: str) -> None:
        verilog_str = self.to_verilog()
        with open(filepath, "w") as f:
            f.write(verilog_str)

    def module_analyze(self: "Module",
                        *,
                        include_wiring: bool = False,
                        include_consts: bool = False,
                        include_reg_cones: bool = True) -> GraphReport:
        """
        Analyze combinational cones of this module.
          - include_wiring=False → don't count Concat/Slice/Resize in node counts (still traversed)
          - include_consts=False → don't count Const in node counts
          - include_reg_cones=True → also traverse register driver cones (depth to sequential inputs)
        Depth model:
          - Op1/Op2/Ternary each add 1 level
          - Concat/Slice/Resize add 0 (transparent wiring)
          - Signals: inputs/regs are sources (depth=0); wires/outputs inline their driver
          - Const: depth=0
        """
        return _Analyzer(include_wiring, include_consts, include_reg_cones).run(self)
    
    
    def all_exprs(self) -> List[Expr]:
        """Depth-first traversal of every expression in the module."""
        seen = set()
        exprs = []
        
        def add_expr(e: Expr):
            if id(e) not in seen:
                seen.add(id(e))
                exprs.append(e)
    
        def visit(e: Expr):
            
            if id(e) in seen:
                return
            
            add_expr(e)
            # Recurse through children
            if hasattr(e, "a"):
                visit(e.a)
            if hasattr(e, "b"):
                visit(e.b)
            if hasattr(e, "sel"):
                visit(e.sel)
            if hasattr(e, "parts"):
                for p in e.parts:
                    visit(p)
            if hasattr(e, "_driver"):
                if e._driver is not None:
                    visit(e._driver)
    
        for s in self._signals:
            # outputs
            if s.kind == "output":
                add_expr(s)
            if s._driver is not None:
                visit(s._driver)
    
        return exprs

def gen_spec(class_instance: Component) -> Dict[str, UInt]:
    spec = {}
    for sig in class_instance.io.__dict__.values():
        spec[sig.name] = sig.typ
    return spec

def get_rand_hash() -> str:
    random_string = str(random.random()) + str(time.time())
    hash_object = hashlib.sha256(random_string.encode())
    name = str(hash_object.hexdigest())
    return name


class IOCollector:
    """
    Group scattered 1-bit ports like a[0], a[1], ..., a[N-1] into a wide UInt port 'a' of width N.
    Mutates the module in-place:
      - The old bit-ports are converted to internal 'wire's.
      - New aggregated ports are created and connected.
    API:
        IOCollector().group(m, {"a": UInt(16), "b": UInt(16), "y": UInt(16)})
    """

    def group(self, m: Module, spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        spec: { base_name -> sprout type (e.g., UInt(16)) }
        Returns a mapping { base_name -> aggregated Signal } for convenience.
        """
        out: Dict[str, Any] = {}
        for base, typ in spec.items():
            width = typ.width
            if width == 1:
                continue  # nothing to do for 1-bit ports
            # Gather all ports that look like base[i]
            bits = self._find_bit_ports(m, base, width)

            if not bits:
                raise ValueError(f"No ports found for base '{base}'")

            k = bits[0].kind
            if any(b.kind != k for b in bits):
                raise ValueError(f"Mixed directions for '{base}[i]': {[b.kind for b in bits]}")

            if k == "input":
                agg = self._create_agg_input_and_wire_bits(m, base, typ, bits)
            elif k == "output":
                agg = self._create_agg_output_from_bits(m, base, typ, bits)
            else:
                raise ValueError(f"Ports for '{base}[i]' are not inputs/outputs (found kind='{k}')")

            out[base] = agg
        return out

    # ---------------- internals ----------------

    def _find_bit_ports(self, m: Module, base: str, width: int):
        """Return ports [bit0, bit1, ..., bit{width-1}] by exact bracketed name."""
        # Build precise name map: "a[0]" -> Signal
        name_to_sig = {p.name: p for p in m._ports}
        bits = []
        for i in range(width):
            nm = f"{base}_{i}_"
            s = name_to_sig.get(nm)
            if s is None:
                raise ValueError(f"Missing bit-port '{nm}'")
            # sanity: 1-bit?
            if s.typ.width != 1:
                raise ValueError(f"Expected 1-bit for '{nm}', got {s.typ}")
            bits.append(s)
        return bits

    def _demote_port_to_wire(self, m: Module, s):
        """Turn an input/output port into an internal wire (keeps drivers/uses intact)."""
        # if s in m._ports:
        #     m._ports.remove(s)
        # s.kind = "wire"  # from input/output → wire
        m._ports[:] = [p for p in m._ports if p is not s]
        s.kind = "wire"

    def _create_agg_input_and_wire_bits(self, m: Module, base: str, typ: Any, bits: List[Any]):
        """Create 'input <typ> base' and drive each former port-bit (now wire) from base[i]."""
        # Name clash?
        if any(p.name == base for p in m._ports):
            raise ValueError(f"Port '{base}' already exists.")
        agg = m.input(typ, base)
        # Demote and connect LSB..MSB
        for i, b in enumerate(bits):
            self._demote_port_to_wire(m, b)
            b <<= agg[i]  # drive internal bit-wire from wide input
        return agg

    def _create_agg_output_from_bits(self, m: Module, base: str, typ: Any, bits: List[Any]):
        """Create 'output <typ> base' as concat of LSB..MSB of the (now internal) bit signals."""
        if any(p.name == base for p in m._ports):
            raise ValueError(f"Port '{base}' already exists.")
        # Demote old bit-ports first (they already have drivers from the existing logic)
        for b in bits:
            self._demote_port_to_wire(m, b)

        agg = m.output(typ, base)
        # Build y = cat(LSB ... MSB)
        parts_lsb_to_msb = [bits[i] for i in range(typ.width)]
        agg <<= cat(*parts_lsb_to_msb)
        return agg
