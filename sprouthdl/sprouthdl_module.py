from sprouthdl.sprouthdl import Bool, ExprLike, HDLType, Signal, fit_width, _SHARED, reset_shared_cache


from typing import List, Optional

from sprouthdl.sprouthdl_analyzer import _Analyzer, GraphReport


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
        s = Signal(name, typ, "input") #, self)
        self._signals.append(s)
        self._ports.append(s)
        return s

    def output(self, typ: HDLType, name: str) -> Signal:
        s = Signal(name, typ, "output") #, self)
        self._signals.append(s)
        self._ports.append(s)
        return s

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