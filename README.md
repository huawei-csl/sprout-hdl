# Sprout-HDL

A hardware description library for digital design.

## Overview

Sprout-HDL is a hardware description library designed to streamline the process of digital circuit design. It provides a clean, intuitive interface for creating and manipulating hardware components.

## Features

- Component-based design
- Built-in simulation capabilities
- Synthesis support
- Easy integration with existing HDL workflows

## Installation

```bash
git clone https://github.com/username/sprout-hdl.git
cd sprout-hdl
pip install -e .
```

## Usage

```python
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import Bool, UInt, SInt, mux, cat

# declare Moudule
m = Module("LogicDemo", with_clock=False, with_reset=False)

# declare inputs
x = m.input(UInt(8), "x")
y = m.input(UInt(8), "y")
sg = m.input(SInt(8), "sg")

# declare outputs
f = m.input(Bool(), "f")
z = m.output(UInt(9), "z")  # 8+8 -> 9 bits
eq = m.output(Bool(), "eq")
hi = m.output(UInt(4), "hi")

sum_ = x + y  # sum is 9-bit
z <<= sum_  # auto-fit/truncate handled
eq <<= x == y  # Bool
hi <<= cat(x[7:6], y[7:6])  # concat 2+2 = 4 bits

# Mux on Bool
w = m.output(UInt(8), "w")
w <<= mux(f, x & y, x | y)

print(m.to_verilog())
```

## Todo

- create package out of this
- add pytest to gitlab to be run automatically
- remove _SHARED object (now used for verilog generation)
- remove is_bool flag, probably not necessary, just use length of 1
- parse wires and regs from graph
- test peek / watch logic
- add better hierarchy capablities / all in graph.  m.wire / m.reg not necessary.
- type conversions: sint, uint, etc
- Uint(value), optional length bit?

## License



## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.