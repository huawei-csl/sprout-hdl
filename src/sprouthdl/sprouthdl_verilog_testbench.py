"""Verilog testbench generator mirroring the Simulator API.

This module provides a drop-in replacement for :class:`sprouthdl.sprouthdl_simulator.Simulator`
that records stimuli and expected results while delegating semantic evaluation to the Python
simulator.  The recorded trace can then be emitted as a synthesizable Verilog testbench that
replays the same interactions against the generated RTL (e.g. when running under Verilator).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

from sprouthdl.sprouthdl import Expr, Signal
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator as PythonSimulator


@dataclass
class _Event:
    action: str
    delay: float
    payload: dict


class VerilogTestbenchSimulator:
    """Record Simulator-like interactions and emit a Verilog testbench.

    The public API mirrors :class:`~sprouthdl.sprouthdl_simulator.Simulator`.  The methods
    delegate to the in-Python simulator to keep behaviour identical while logging the sequence
    of IO operations.  Once the trace is complete, call :meth:`write_testbench` to materialise
    the Verilog testbench that reproduces the stimuli and checks the observed outputs.
    """

    def __init__(
        self,
        module: Module,
        *,
        clock_period: float = 10.0,
        eval_delay: float = 1.0,
    ) -> None:
        self.m = module
        self._sim = PythonSimulator(module)
        self.clock_period = clock_period
        self.eval_delay = eval_delay
        self._events: List[_Event] = []
        self.inputs = [s for s in self.m._ports if s.kind == "input"]
        self.outputs = [s for s in self.m._ports if s.kind == "output"]
        self.regs = [s for s in self.m._signals if s.kind == "reg"]
        self._by_name = {s.name: s for s in self.m._signals}
        self._half_period = clock_period / 2.0 if module.with_clock else None

    # ------------------------------------------------------------------
    # Simulator compatible API
    # ------------------------------------------------------------------
    def set(self, ref: Union[str, Signal], value: int) -> "VerilogTestbenchSimulator":
        sig = self._resolve(ref)
        if sig.kind not in {"input", "reg"}:
            raise ValueError("Only inputs and regs can be set directly.")
        self._sim.set(sig, value)
        self._events.append(
            _Event(
                action="set",
                delay=0.0,
                payload={
                    "signal": sig,
                    "value": value,
                },
            )
        )
        return self

    def get(self, ref: Union[str, Signal], *, signed: Optional[bool] = None) -> int:
        return self._sim.get(ref, signed=signed)

    def eval(self) -> "VerilogTestbenchSimulator":
        self._sim.eval()
        self._events.append(
            _Event(
                action="eval",
                delay=self.eval_delay,
                payload={"snapshot": self._snapshot()},
            )
        )
        return self

    def step(self, n: int = 1) -> "VerilogTestbenchSimulator":
        if not self.m.with_clock:
            raise RuntimeError("step() requires a clocked module.")
        for _ in range(n):
            self._sim.step(1)
            # Rising edge then falling edge with optional sampling delay afterwards.
            self._events.append(
                _Event("clock_high", self._half_period, {"clock": self.m.clk})
            )
            self._events.append(
                _Event("clock_low", self._half_period, {"clock": self.m.clk})
            )
            self._events.append(
                _Event(
                    action="sample",
                    delay=self.eval_delay,
                    payload={"snapshot": self._snapshot()},
                )
            )
        return self

    def reset(self, asserted: bool = True) -> "VerilogTestbenchSimulator":
        if not self.m.with_reset:
            return self
        self._sim.reset(asserted)
        value = 1 if asserted else 0
        self._events.append(
            _Event(
                action="set",
                delay=0.0,
                payload={
                    "signal": self.m.rst,
                    "value": value,
                },
            )
        )
        if asserted:
            # allow combinational paths to settle with reset asserted
            self._events.append(
                _Event(
                    action="eval",
                    delay=self.eval_delay,
                    payload={"snapshot": self._snapshot()},
                )
            )
        return self

    def deassert_reset(self) -> "VerilogTestbenchSimulator":
        if self.m.with_reset:
            self.reset(False)
        return self

    def peek_outputs(self, *, signed: bool = False) -> Dict[str, int]:
        return self._sim.peek_outputs(signed=signed)

    # Watch support mirrors the Python simulator by direct delegation.
    def watch(self, what, alias: Optional[str] = None):
        self._sim.watch(what, alias)
        return self

    def get_watch(self, name: str) -> int:
        return self._sim.get_watch(name)

    def clear_watches(self) -> None:
        self._sim.clear_watches()

    def list_signals(self) -> List[str]:
        return self._sim.list_signals()

    def peek(self, what):
        return self._sim.peek(what)

    def peek_next(self, reg_name):
        return self._sim.peek_next(reg_name)

    def log_expression_states(self, expr_list: Iterable[Expr]):
        return self._sim._get_expression_states(expr_list)

    # ------------------------------------------------------------------
    # Testbench emission
    # ------------------------------------------------------------------
    def to_testbench_lines(
        self,
        *,
        tb_module_name: Optional[str] = None,
        timescale: Optional[str] = "1ns/1ps",
    ) -> Path:
        """Emit a Verilog testbench that replays the recorded events."""

        if not self._events:
            raise RuntimeError("No events recorded – nothing to write.")

        tb_name = tb_module_name or f"{self.m.name}_tb"

        lines: List[str] = []

        if timescale:
            lines.append(f"`timescale {timescale}")
            lines.append("")

        lines.append(f"module {tb_name};")

        for s in self.inputs:
            decl = self._decl_line("reg", s)
            lines.append(f"  {decl}")
        for s in self.outputs:
            decl = self._decl_line("wire", s)
            lines.append(f"  {decl}")
        lines.append("")

        # DUT instantiation
        ports = []
        for p in self.m._ports:
            ports.append(f"    .{p.name}({p.name})")
        ports_csv = ",\n".join(ports)
        lines.append(f"  {self.m.name} dut (\n{ports_csv}\n  );")
        lines.append("")

        lines.append("  initial begin")
        # Initialise all inputs to zero
        for s in self.inputs:
            init_literal = self._literal(0, s.typ.width)
            lines.append(f"    {s.name} = {init_literal};")
        lines.append("")

        for event in self._events:
            if event.delay > 0:
                lines.append(f"    #{self._format_delay(event.delay)};")
            if event.action == "set":
                sig: Signal = event.payload["signal"]
                literal = self._literal(event.payload["value"], sig.typ.width)
                if self.m._is_port(sig):
                    lines.append(f"    {sig.name} = {literal};")
                else:
                    lines.append(f"    dut.{sig.name} = {literal};")
            elif event.action == "eval":
                self._emit_snapshot(lines, event.payload["snapshot"], indent="    ")
            elif event.action in {"clock_high", "clock_low"}:
                clk = event.payload["clock"]
                val = "1'b1" if event.action == "clock_high" else "1'b0"
                lines.append(f"    {clk.name} = {val};")
            elif event.action == "sample":
                self._emit_snapshot(lines, event.payload["snapshot"], indent="    ")
            else:
                raise ValueError(f"Unknown event action: {event.action}")

        lines.append("    $display(\"Testbench completed successfully.\");")
        lines.append("    $finish;")
        lines.append("  end")
        lines.append("endmodule")

        return lines
    
    def write_testbench(
        self,
        path: Union[str, Path],
        *,
        tb_module_name: Optional[str] = None,
        timescale: Optional[str] = "1ns/1ps",
    ) -> Path:
        lines = self.to_testbench_lines(
            tb_module_name=tb_module_name,
            timescale=timescale,
        )
        path = Path(path)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _resolve(self, ref: Union[str, Signal]) -> Signal:
        if isinstance(ref, Signal):
            return ref
        if isinstance(ref, str):
            try:
                return self._by_name[ref]
            except KeyError as exc:
                raise KeyError(f"No signal named '{ref}' in module {self.m.name}.") from exc
        raise TypeError(f"Expected Signal or str, got {type(ref)}")

    def _snapshot(self) -> dict:
        outputs = self._sim.peek_outputs()
        watches = {}
        if hasattr(self._sim, "_watch_values"):
            watches = dict(self._sim._watch_values)
        return {"outputs": outputs, "watches": watches}

    def _literal(self, value: int, width: int) -> str:
        mask = (1 << width) - 1
        value &= mask
        if width == 1:
            return f"1'b{value & 1}"
        digits = max(1, (width + 3) // 4)
        return f"{width}'h{value:0{digits}x}"

    def _decl_line(self, kind: str, sig: Signal) -> str:
        sign = "signed " if sig.typ.signed else ""
        rng = sig.typ.range_str()
        if rng:
            return f"{kind} {sign}{rng} {sig.name};"
        return f"{kind} {sign}{sig.name};".replace("  ", " ")

    def _emit_snapshot(self, lines: List[str], snapshot: dict, *, indent: str) -> None:
        outputs = snapshot.get("outputs", {})
        if outputs:
            fmt_parts = []
            arg_names = ["$time"]
            for name, value in outputs.items():
                literal = self._literal(value, self._by_name[name].typ.width)
                lines.append(f"{indent}if ({name} !== {literal}) begin")
                lines.append(f"{indent}  $fatal(1, \"Expected {name}=%0h, got %0h\", {literal}, {name});")
                lines.append(f"{indent}end")
                fmt_parts.append(f" {name}=%0h")
                arg_names.append(name)
            fmt = "[%0t]" + "".join(fmt_parts)
            arg_list = ", ".join(arg_names)
            lines.append(f"{indent}$display(\"{fmt}\", {arg_list});")
        if snapshot.get("watches"):
            watch_names = ", ".join(sorted(snapshot["watches"].keys()))
            lines.append(f"{indent}// Watches captured in Python only: {watch_names}")

    def _format_delay(self, value: float) -> str:
        if abs(value - int(value)) < 1e-9:
            return str(int(value))
        return f"{value:g}"


__all__ = ["VerilogTestbenchSimulator"]
