# Sprout-HDL

Sprout-HDL is a Python embedded domain-specific language (EDSL) for building digital hardware in a concise, composable way.  It lets you describe logic with Python expressions, compile the result to synthesizable Verilog or AIG/AAG netlists, and iterate quickly with a built-in cycle-accurate simulator.

## Project overview

Sprout-HDL revolves around a small set of core modules:

- **`sprouthdl.sprouthdl`** – the expression DSL.  It provides bit-precise types such as `Bool`, `UInt`, and `SInt`, shared-expression caching, and the overloaded arithmetic / bitwise operators that make the Python syntax feel like an HDL.
- **`sprouthdl.sprouthdl_module`** – structural modeling helpers.  The `Module` class constructs ports, wires, and registers, produces Verilog, and exposes analysis utilities.  The `Component` base class lets you package reusable sub-designs and convert them to or from Sprout modules.  `IOCollector` can rebuild packed ports from bit-level signals when importing external netlists.
- **`sprouthdl.sprouthdl_simulator`** – a lightweight simulator that can drive inputs, tick clocks, inspect outputs or internal expressions, and capture probes for debugging—all without leaving Python.

Supporting packages add reusable arithmetic building blocks and importer utilities for external netlists when you need to mix handwritten Sprout code with pre-existing IP.【F:low_level_arithmetic/multipliers_ext.py†L1-L182】【F:low_level_arithmetic/multipliers_ext_optimized.py†L1-L182】

## Installation

```bash
git clone https://github.com/username/sprout-hdl.git
cd sprout-hdl
pip install -e .
```

The library relies on Python 3.10+ and the packages listed in `requirements.txt`.  Optional regression tests require Yosys/Pyosys and aigverse if you plan to exercise the external tooling integration flows.

## Quick start

### 1. Describe a module

```python
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import Bool, UInt, mux, cat

m = Module("LogicDemo", with_clock=False, with_reset=False)
a = m.input(UInt(8), "a")
b = m.input(UInt(8), "b")
sel = m.input(Bool(), "sel")
sum_ = m.output(UInt(9), "sum")
mask = m.output(UInt(4), "mask")
out = m.output(UInt(8), "out")

sum_ <<= a + b              # automatic width growth
top_bits = cat(a[7], b[7])
mask <<= top_bits           # concatenate slices
a_and_b = a & b
b_or_a = a | b
out <<= mux(sel, a_and_b, b_or_a)

print(m.to_verilog())
```

The `Module` API checks that every output has a driver and every register has a next-state assignment before emitting Verilog.【F:sprouthdl/sprouthdl_module.py†L169-L235】

### 2. Simulate the design

```python
from sprouthdl.sprouthdl_simulator import Simulator

sim = Simulator(m)
sim.set("a", 0x55).set("b", 0x0F).set("sel", 1)
sim.eval()                 # recompute combinational logic
print(sim.peek_outputs())   # {'sum': 0x64, 'mask': 0x9, 'out': 0x05}
```

The simulator keeps track of inputs, wires, outputs, and registers, supports `eval()` for combinational updates, `step()` for clocked designs, and exposes helpers such as `peek`, `peek_next`, and signal watching for deeper inspection.【F:sprouthdl/sprouthdl_simulator.py†L16-L420】

### 3. Integrate with external tooling

Modules can be exported to Verilog, AIG, or AAG for downstream synthesis, equivalence checking, or integration into larger verification environments.  Import helpers then let you bring optimized or third-party netlists back into Sprout for continued composition and simulation.【F:sprouthdl/sprouthdl_module.py†L169-L235】【F:low_level_arithmetic/multipliers_ext_optimized.py†L62-L132】

## Modules and components in detail

- `Component` subclasses package reusable structures.  They can materialize new modules (`to_module`), import designs from Verilog or AIG formats (`from_verilog`, `from_aag_lines`), and retag ports as internals (`make_internal`).  Components also expose `get_spec()` to drive `IOCollector` regrouping when you import flattened designs.【F:sprouthdl/sprouthdl_module.py†L14-L94】
- `Module` is typically used at the top level or as an intermediate representation while you are still wiring a design.  It offers constructors for inputs, outputs, wires, and registers; utilities for enumerating signals; Verilog emission with automatic width fitting; and a `module_analyze()` routine that reports combinational depth and node counts for timing exploration.【F:sprouthdl/sprouthdl_module.py†L96-L255】
- `IOCollector` helps rebuild packed buses (e.g., `a[0] … a[N-1]` → `a[N-1:0]`) after reading back designs from AIG/AAG files or external synthesizers.【F:sprouthdl/sprouthdl_module.py†L302-L360】

### Hierarchical design with components

Components are ideal for assembling hierarchical designs: they let you instantiate another component, adapt its IO, and even swap in a pre-synthesized netlist without leaving Python.  One common pattern wraps a reusable building block with `make_internal()` so that auxiliary logic can surround the core implementation while exposing a compact public interface.【F:low_level_arithmetic/multipliers_ext.py†L62-L152】  A related flow imports an external AIG module, converts it into a `Component`, and calls `from_module(..., make_internal=True)` so the imported logic behaves like a native Sprout block inside a larger generator.【F:low_level_arithmetic/multipliers_ext_optimized.py†L62-L132】  These techniques extend to Verilog importers and make it straightforward to mix Sprout-authored code with IP produced by external flows.

## Simulation notes

The simulator supports both combinational and sequential designs:

- `eval()` recomputes combinational logic and captures registered probes.
- `set()` and `get()` let you drive or inspect signals by name.
- `step()` advances the clock, committing register next-state expressions while honoring asynchronous resets.
- `watch()` and `peek_next()` provide scope-style visibility for debugging complex pipelines.【F:sprouthdl/sprouthdl_simulator.py†L49-L420】

These capabilities align with the standard Sprout development flow: express a design, validate it in Python, then export it to your synthesis or verification stack.

## Main development flow

1. **Model logic in Python.** Use `Module` and DSL expressions to capture datapaths, state machines, and control logic.
2. **Factor reusable pieces.** Wrap recurring structures in `Component` subclasses so they can be instantiated, parameterized, or replaced with imported implementations.
3. **Simulate early and often.** Drive stimuli with the simulator, observe register evolution, and iterate on the Python source before handing designs to downstream tools.
4. **Export netlists.** Emit Verilog or AIG/AAG when you are ready for synthesis, formal checking, or integration with external flows.【F:sprouthdl/sprouthdl_module.py†L169-L235】【F:low_level_arithmetic/multipliers_ext_optimized.py†L62-L132】

## Examples

Check out the `testing/examples/` directory for practical examples:

- **`simple_component.py`** – A minimal example showing how to define a Component with IO ports and generate Verilog
- **`component_example.py`** – Comprehensive examples including hierarchical design and simulation
- **`module_with_component.py`** – Shows how to integrate Components within Module-based designs

See the [examples README](examples/README.md) for detailed documentation and key concepts.

## Next steps

- Explore the `testing/examples/` directory to see working examples of components and modules
- Explore the `sprouthdl/floating_point` and `low_level_arithmetic` packages for more generators.
- Use `module_analyze()` to gauge combinational depth before synthesis.【F:sprouthdl/sprouthdl_module.py†L238-L255】
- Integrate the simulator into your verification harness to shorten debug cycles.

## Slices
We follow the indexing of pyton also in Sprout-HDL signals. For example `sig[4:7]` creates a new expression containing of bits 4 and 5 (counted from lsb) of the original expression `sig`.

# Running Scripts and Tests

For some scripts you might need to add `PYTHONPATH=$(pwd)` before running the command, e.g. `PYTHONPATH=$(pwd) pytest` or `PYTHONPATH=$(pwd) python <scriptname>`.

## Arithmetic Evaluations
### Integers
Run the evaluation script 
```bash
python sprouthdl/arithmetic/int_multipliers/eval/run_multiplier_stage_options_eval_ext_stat.py
```
This will generate a parquet file in the folder `data`. Visualization can be done with the plotly app via 
```bash
python sprouthdl/arithmetic/int_multipliers/eval/plot/plotly_app.py --file data/data_file.parquet
``` 
or with the script 
```bash
python sprouthdl/arithmetic/int_multipliers/eval/plot/multiplier_stage_plot.py --file  data/data_file.parquet`.
```
Replace `data_file.parquet` with the file produced by the evaluate script.

If desired, new multiplier options can be added here: `sprouthdl/arithmetic/int_multipliers/eval/multiplier_stage_options_demo_lib.py`.

#### Optimized Multipliers
The multipliers in `sprouthdl/arithmetic/int_multipliers/multipliers/multipliers_ext_optimized.py` rely on precomputed AIGs (and map files). By default they load packaged assets from `sprouthdl/arithmetic/int_multipliers/data/optimized/` using the following filenames:

- `unsigned_3b|4b|8b/{out_aiger.aig,aiger_map_cleaned.map}`
- `signed_3b|4b|8b/{out_aiger.aig,aiger_map_cleaned.map}`
- `unsigned_4b_strong/{out_aiger_strong.aig,out_aiger_map_strong.map}`

To point at your own artifacts, set `SPROUTHDL_OPT_MULT_DIR=/path/to/optimized` (keep the same subdirectory names/files) or pass a custom `f_aag_lines` callable when constructing the multiplier. Clear errors are raised if neither packaged nor user-supplied assets are found.

You can produce and optimize these files with Flowy; for 4-bit unsigned, for example:
```bash
# 4 bit unsigned star starting point
python flowy/flows/reinforce/run/statistical/run_flows_in_docker.py\
  --experiment unsigned_optim_8bit_star_1\
  --bitwidth 4\
  --iterations 50\
  --mockturtle_chains 5\
  --mockturtle_chain_workers 5\
  --mockturtle_chain_len 15\
  --compression_scripts_per_step 3\
  --scripts_per_step 2\
  --nb_runs 100\
  --nb_workers 50\
  --recipe_selection PERFORMANCE_SAMPLING\
  --selection_metric aig_count\
  --output_encoding unsigned\
  --input_encoding unsigned\
  --strategy_name equal\
  --verilog_file resources/sources/mydesign_comb_star_unsigned.v.template

PYTHONPATH=$(pwd) python flowy/flows/sim/visualize_histograms.py --experiment unsigned_optim_4bit_star_1
PYTHONPATH=$(pwd) python flowy/flows/sim/extract_best_design.py --experiment unsigned_optim_4bit_star_1
PYTHONPATH=$(pwd) python flowy/flows/reinforce/analysis/visualize_runs.py  --experiment unsigned_optim_4bit_star_1
```



## Todo

- create package out of this --> done
- add pytest to gitlab to be run automatically --> done
- remove _SHARED object (now used for verilog generation)
- remove is_bool flag, probably not necessary, just use length of 1
- parse wires and regs from graph
- test peek / watch logic --> done
- add better hierarchy capablities / all in graph.  m.wire / m.reg not necessary.
- type conversions: sint, uint, etc
- Uint(value), optional length bit?
- simulation: get any signal in graph wich is there implicitly, run simulation just on function and after setting starting nodes to a value
- unify get and peek (same thing but one is with sign conversion, the other is not)
- unify log_expression_states and watches in the Simulator.
- clean up two versions of testvector generation and the exhaustive testvector generation
- in module there is all_exprs and _collect_signals_from_outputs maybe this can be merged, especially by removing the _signal attribute (maybe still retain it as a cache)

Contributions are welcome—feel free to open issues or submit pull requests with improvements or new hardware components.
