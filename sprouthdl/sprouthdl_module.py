import abc
from dataclasses import dataclass, make_dataclass
import hashlib
import random
import time
from sprouthdl.sprouthdl import Bool, Expr, ExprLike, HDLType, Signal, UInt, fit_width, _SHARED, reset_shared_cache


from typing import Dict, List, Optional, Self

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

    def to_module(self, name: Optional[str] = None) -> 'Module':
        module = Module(
            name or f"comp_{get_rand_hash()}",
            with_clock=False,
            with_reset=False,
        )
        for sig in self.io.__dict__.values():
            if sig.kind == "input":
                module.add_input(sig)
            elif sig.kind == "output":
                module.add_output(sig)
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        module.component = self # can be used for debugging
        return module 

    
    def from_module(self, module: 'Module', make_internal=False, group=False) -> Self:
        #if group:
        #    IOCollector().group(module, cls.get_spec())
        
        # find signals in module and assign to io
        io_fields = {}
        for sig in module._signals:
            if sig.kind in ('input', 'output'):
                io_fields[sig.name] = sig
        #instance.io = dataclass(type('IO', (), io_fields))()  # type: ignore
        #instance.io = make_dataclass("IO", io_fields)
        #instance.io = io_fields
        for io_name, io_sig in io_fields.items():
            setattr(self.io, io_name, io_sig)
        self.elaborate()  # re-elaborate to rebuild internal structure
        if make_internal:
            self.make_internal()


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
        wires = self._internals_of("wire") + _SHARED.wires
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

    def module_analyze(self: "Module",
                        *,
                        include_wiring: bool = False,
                        include_consts: bool = False,
                        include_reg_cones: bool = True) -> GraphReport:
        """
        Analyze combinational cones of this module.
          - include_wiring=False → don't count Concat/Slice/Resize in node counts (still traversed)
          - include_consts=False → don't count Const in node counts
          - include_reg_cones=True → also traverse reg.next cones (depth to sequential inputs)
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
    
        def visit(e: Expr):
            if id(e) in seen:
                return
            seen.add(id(e))
            exprs.append(e)
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
            if s._driver is not None:
                visit(s._driver)
            if s.kind == "reg" and s._next is not None:
                visit(s._next)
    
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
