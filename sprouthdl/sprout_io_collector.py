# io_collect.py
from typing import Dict, Tuple, List, Optional, Any
from sprouthdl.sprout_hdl import UInt, cat
from sprouthdl.sprout_hdl_module import Module


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
            nm = f"{base}[{i}]"
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
        """Create 'output <typ> base' as concat of MSB..LSB of the (now internal) bit signals."""
        if any(p.name == base for p in m._ports):
            raise ValueError(f"Port '{base}' already exists.")
        # Demote old bit-ports first (they already have drivers from the existing logic)
        for b in bits:
            self._demote_port_to_wire(m, b)

        agg = m.output(typ, base)
        # Build y = cat(MSB ... LSB)
        parts_msb_to_lsb = [bits[i] for i in reversed(range(typ.width))]
        agg <<= cat(*parts_msb_to_lsb)
        return agg