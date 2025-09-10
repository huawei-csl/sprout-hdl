# Sprout-HDL

A hardware description library for digital design.

## Overview

Sprout-HDL is a hardware description library designed to streamline the process of digital circuit design. It provides a clean, intuitive interface for creating and manipulating hardware components.

## Features

- Component-based design
- Built-in simulation capabilities
- Synthesis support
- Comprehensive standard library
- Easy integration with existing HDL workflows

## Installation

```bash
git clone https://github.com/username/sprout-hdl.git
cd sprout-hdl
pip install -e .
```

## Usage

```python
from sprout_hdl import Module, Signal, Wire

# Create a simple module
class Adder(Module):
    def __init__(self):
        super().__init__()
        self.a = UInt(8)
        self.b = UInt(8)
        self.result = UInt(9)
        
    def elaborate(self):
        self.result <<= self.a + self.b
```

## Documentation

For more detailed documentation, see the [docs](docs/) folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.