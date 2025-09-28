# roundtrip_suite.py
from __future__ import annotations
import os, tempfile, random
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# --- your libs (adjust paths as needed) ---
from aigverse import equivalence_checking
from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, file_to_lines, read_aag_into_aig, conv_aag_into_aig
from sprouthdl.sprout_hdl import UInt, Bool, reset_shared_cache
from sprouthdl.sprout_hdl_aiger import AigerExporter, AigerImporter
from sprouthdl.sprout_hdl_module import Module
from sprouthdl.sprout_hdl_simulator import Simulator
from sprouthdl.sprout_io_collector import IOCollector
from testing.testvectors_general import build_fp_vectors, floatx_to_float  # generic EW,FW vectors/decoder
from sprout_hdl_float_sn import build_fp_mul_sn  # your FP mul (with/without subnormals)

# Pyosys
from pyosys import libyosys as ys

# -----------------------------
# Utilities
# -----------------------------


def write_temp_verilog(m: Module, top_name: str | None = None) -> str:
    """Write module Verilog to a temporary file and return the path."""
    if top_name and m.name != top_name:
        # rename for convenience
        v = m.to_verilog().replace(f"module {m.name} ", f"module {top_name} ")
    else:
        v = m.to_verilog()
    fd, path = tempfile.mkstemp(suffix=".v")
    os.close(fd)
    with open(path, "w") as f:
        f.write(v)
    return path


def verilog_to_aag_via_pyosys(
    verilog_path: str,
    *,
    top: str | None = None,
    aag_out_path: str | None = None,
    tie_undriven: str | None = None,  # {"zero","one","random"} or None
    embed_symbols: bool = True,
    no_startoffset: bool = True,
    map_out_path: str | None = None,  # write_aiger -map
) -> Tuple[str, str | None]:
    """Run yosys: read_verilog → synth -flatten → aigmap → write_aiger -ascii […]. Returns (aag_path, map_path_or_None)."""
    # reset yosys design to avoid crosstalk between runs
    ys.run_pass("design -reset")
    if aag_out_path is None:
        fd, aag_out_path = tempfile.mkstemp(suffix=".aag")
        os.close(fd)
    if map_out_path is None:
        fd, map_out_path = tempfile.mkstemp(suffix=".map")
        os.close(fd)

    ys.run_pass("design -reset")
    ys.run_pass(f"read_verilog {verilog_path}")
    ys.run_pass("hierarchy -check " + ("-auto-top" if top is None else f"-top {top}"))
    ys.run_pass("synth -flatten")
    if tie_undriven in {"zero", "one", "random"}:
        flag = {"zero": "-zero", "one": "-one", "random": "-random"}[tie_undriven]
        ys.run_pass(f"setundef -undriven {flag}")
    ys.run_pass("aigmap")

    opts = ["-ascii"]
    if embed_symbols:
        opts.append("-symbols")
    if no_startoffset:
        opts.append("-no-startoffset")
    if map_out_path:
        opts += ["-map", map_out_path]
    ys.run_pass(f"write_aiger {' '.join(opts)} {aag_out_path}")
    return aag_out_path, (map_out_path if os.path.getsize(map_out_path) > 0 else None)


def sprout_to_aig_via_exporter(m: Module):
    """Sprout → AAGER lines (ASCII) and AIG object (via read_aag_into_aig)."""
    aag_lines = AigerExporter(m).get_aag()
    # If you want an AIG object as well:
    fd, tmp = tempfile.mkstemp(suffix=".aag")
    os.close(fd)
    with open(tmp, "w") as f:
        f.write("\n".join(aag_lines) + "\n")
    aig = read_aag_into_aig(tmp)
    return aag_lines, aig


def roundtrip_aiger_back_to_sprout(aag_lines: List[str], *, name="Imported") -> Module:
    """Import AAG (with symbols kept) back into Sprout."""
    # Keep symbol table (last lines); leave as-is if already present
    aag_sym = _get_aag_sym(aag_lines)
    aag_for_import = aag_lines[:-2] + aag_sym if aag_sym else aag_lines
    return AigerImporter(aag_for_import).get_sprout_module(name)


def run_vectors_io(
    m: Module,
    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]],
    *,
    decoder: Callable[[int], float] | None = None,
) -> None:
    """
    Generic runner:
      vectors: list of (label, inputs{name->int}, expected{name->int})
    Prints mismatches; raises AssertionError at the end if any failed.
    """
    
    sim = Simulator(m)
    fails = 0
    for name, ins, outs in vectors:
        for k, v in ins.items():
            sim.set(k, v)
        sim.eval()
        bad = []
        for oname, exp in outs.items():
            got = sim.get(oname)
            if got != exp:
                if decoder and oname == "y":
                    bad.append(f"{oname}: got=0x{got:0X} ({decoder(got):.8g})  exp=0x{exp:0X} ({decoder(exp):.8g})")
                else:
                    bad.append(f"{oname}: got=0x{got:0X}  exp=0x{exp:0X}")
            else:
                if decoder and oname == "y":
                    print(f"PASS {name}: {oname}=0x{got:0X} ({decoder(got):.8g})")
                else:
                    print(f"PASS {name}: {oname}=0x{got:0X}")
        if bad:
            fails += 1
            print(f"FAIL  {name}:  " + " | ".join(bad))
    if fails:
        raise AssertionError(f"{fails}/{len(vectors)} vectors failed")
        #print(f"{fails}/{len(vectors)} vectors failed")

# -----------------------------
# Simple modules + vectors
# -----------------------------


def build_logic3() -> Tuple[Module, Dict[str, UInt]]:
    """y = x0 & (x1 | x2)"""
    m = Module("Logic3", with_clock=False, with_reset=False)
    x0 = m.input(Bool(), "x0")
    x1 = m.input(Bool(), "x1")
    x2 = m.input(Bool(), "x2")
    y = m.output(Bool(), "y")
    y <<= x0 & (x1 | x2)
    # vectors: exhaustive 3-bit
    vecs = []
    for v in range(8):
        ins = {"x0": (v >> 0) & 1, "x1": (v >> 1) & 1, "x2": (v >> 2) & 1}
        exp = {"y": ins["x0"] & (ins["x1"] | ins["x2"])}
        vecs.append((f"v={v:03b}", ins, exp))
    spec = {"x0": UInt(1), "x1": UInt(1), "x2": UInt(1), "y": UInt(1)}
    return m, spec, vecs


def build_adder(W: int = 8) -> Tuple[Module, Dict[str, UInt], List]:
    m = Module(f"Add{W}", with_clock=False, with_reset=False)
    a = m.input(UInt(W), "a")
    b = m.input(UInt(W), "b")
    y = m.output(UInt(W + 1), "y")
    y <<= a + b
    vecs = []
    for _ in range(64):
        va = random.getrandbits(W)
        vb = random.getrandbits(W)
        vecs.append((f"{va}+{vb}", {"a": va, "b": vb}, {"y": (va + vb) & ((1 << (W + 1)) - 1)}))
    spec = {"a": UInt(W), "b": UInt(W), "y": UInt(W + 1)}
    return m, spec, vecs


def build_fp_mul_case(EW: int, FW: int, *, subnormals: bool = False):
    bits_tot = 1 + EW + FW
    m = build_fp_mul_sn(f"FPMul_E{EW}_F{FW}_{'SN' if subnormals else 'NZ'}", EW=EW, FW=FW, subnormals=subnormals)
    # use your generic vector builder
    vecs_basic = []
    for name, a, b, exp in build_fp_vectors(EW, FW):
        vecs_basic.append((name, {"a": a, "b": b}, {"y": exp}))
    spec = {"a": UInt(bits_tot), "b": UInt(bits_tot), "y": UInt(bits_tot)}
    return m, spec, vecs_basic, (lambda bits: floatx_to_float(bits, EW, FW))

# primary pi permutation handler
def build_pi_order(aag_lines):
    """Return the PI name list in positional order using symbol lines (iN name)."""
    syms = {}
    for ln in aag_lines:
        if not ln or ln[0] in "ca":  # stop at 'c' or skip blanks
            if ln.startswith("c"): break
            continue
        if ln[0] == "i":
            # i<number> <name>
            try:
                n_str, name = ln[1:].lstrip().split(" ", 1)
            except ValueError:
                n_str, name = ln[1:].lstrip(), ""
            syms[int(n_str)] = name
    return [syms[i] for i in sorted(syms)]

def permute_aig_inputs(aig, current_order: list[str], target_order: list[str]):
    """
    Reorder AIG primary inputs (positionally) to match the target_order by names.
    Returns a new aig object with permuted inputs.  (Uses aigverse API semantics.)
    """
    name_to_pos = {nm: i for i, nm in enumerate(current_order)}
    perm = [name_to_pos[nm] for nm in target_order]  # current position for each desired position
    # Apply permutation: recreate a fresh AIG with inputs wired according to perm
    from aigverse import Aig
    new = Aig()
    # recreate PIs in target order
    pis = [new.create_pi(nm) for nm in target_order]
    # make a mapping from old PI nodes to new ones
    old_pis = list(aig.pis())           # positional
    old_to_new = {aig.get_node(old_pis[i]): pis[j] for j, i in enumerate(perm)}
    # Copy AND graph (topo) with node mapping, preserving complemented edges
    # Simplest: use write/read AIG with a temporary name-map, or rebuild with a small copier.
    # (Pseudocode here; adjust to your AIG API)
    def map_sig(sig):
        n = aig.get_node(sig)
        if aig.is_constant(n): return new.get_constant(aig.is_complemented(sig))
        if aig.is_pi(n):       # replace with permuted pi
            mapped = old_to_new[n]
            return new.create_buf(mapped, inverted=aig.is_complemented(sig))
        # AND node: map fanins
        f0, f1 = aig.fanins(n)
        m0 = map_sig(f0); m1 = map_sig(f1)
        out = new.create_and(m0, m1)
        return new.create_buf(out, inverted=aig.is_complemented(sig))
    # Recreate POs
    for s, nm in zip(aig.pos(), (getattr(aig, "po_names", None) or [])):
        new.create_po(map_sig(s), nm if nm else None)
    return new


# -----------------------------
# Round-trip test for ONE module case
# -----------------------------


def run_test_one_module(m: Module, spec: Dict[str, UInt], vectors, *, decoder=None, 
                    equivalence_check = True) -> None:
    print(f"\n=== {m.name} ===")

    # 1) Original sim
    print("Sim (original) …")
    run_vectors_io(m, vectors, decoder=decoder)

    # 2) Sprout → AIGER (exporter) → AIG
    aag_lines, aig_exp = sprout_to_aig_via_exporter(m)

    # optional agi to sprout module
    #m2 = roundtrip_aiger_back_to_sprout(aag_lines, name=m.name+"_exp")

    # 3) Verilog → Pyosys → AIGER → AIG
    vpath = write_temp_verilog(m, top_name=m.name)
    # for testing copy vpath to local dir
    os.system(f"cp {vpath} ./{m.name}.v")
    aag_path, map_path = verilog_to_aag_via_pyosys(vpath, top=m.name, embed_symbols=True, no_startoffset=True)
    aig_pyo = read_aag_into_aig(aag_path)

    aag_pyo_lines = file_to_lines(aag_path)
    
    # after you produced aag_path with yosys:
    aag_back_lines = file_to_lines(aag_path)
    m_back_raw = AigerImporter(aag_back_lines).get_sprout_module("BackRaw")    
    # exporter AIG from the raw imported module (no IOCollector yet)
    aag_from_raw, aig_from_raw = sprout_to_aig_via_exporter(m_back_raw)    
    # yosys AIG
    aig_yosys = read_aag_into_aig(aag_path)
        
    assert equivalence_checking(aig_from_raw, aig_yosys), \
        "Importer produced a non-equivalent Sprout netlist BEFORE regrouping"

    # Normalize PI order (by name) before equivalence
    # order_exp = build_pi_order(aag_lines)   # e.g., ["a[0]",...,"b[0]",...]
    # order_pyo = build_pi_order(aag_pyo_lines)   # e.g., ["b[0]",...,"a[0]",...]

    # if order_exp != order_pyo:
    #     aig_pyo = permute_aig_inputs(aig_pyo, current_order=order_pyo, target_order=order_exp)

    # 4) Equivalence check
    if equivalence_check:
        print("AIG equivalence (exporter vs pyosys) …")
        assert equivalence_checking(aig_exp, aig_pyo), "AIGs not equivalent!"

    # 5) AAG (with symbols) → Sprout
    aag_back = file_to_lines(aag_path)
    m_back = roundtrip_aiger_back_to_sprout(aag_back, name=m.name + "_back")

    # 6) Regroup I/Os to match original port widths
    IOCollector().group(m_back, spec)

    # 7) Re-sim round-tripped module
    print("Sim (round-tripped) …")
    run_vectors_io(m_back, vectors, decoder=decoder)


# -----------------------------
# Main: build a list and run
# -----------------------------


def test_run():
    random.seed(123)

    # cases: List[Tuple[Module, Dict[str, UInt], List, Callable | None]] = []

    # # Simple combinational
    # # m, spec, vecs = build_logic3()
    # # cases.append((m, spec, vecs, None))

    # # m, spec, vecs = build_adder(8)
    # # cases.append((m, spec, vecs, None))

    # # PickProbe (if you included its builder)
    # try:
    #     m, spec, vecs = build_pick_probe(11)
    #     cases.append((m, spec, vecs, None))
    # except Exception as e:
    #     print("Skip PickProbe (builder not available):", e)

    # # FP multipliers (a couple of small formats)
    # for EW, FW, SN in [(5, 3, False), (5, 10, False)]:  # add more as you wish
    #     m, spec, vecs, dec = build_fp_mul_case(EW, FW, subnormals=SN)
    #     cases.append((m, spec, vecs, dec))

    def gen_m_case(i:int) -> Tuple[Module, Dict[str, UInt], List, Callable | None]:
        reset_shared_cache()
        if i==0:
            return build_logic3() + (None,)
        elif i==1:
            return build_adder(8) + (None,)
        elif i==2:
            return build_fp_mul_case(5,3,subnormals=False)
        elif i==3:
            return build_fp_mul_case(5,10,subnormals=False)

    # Run everything
    n_cases = 4
    for i in range(n_cases):

        m, spec, vecs, dec = gen_m_case(i)

        print(f"Running case {i + 1}/{n_cases}")
        run_test_one_module(m, spec, vecs, decoder=dec, equivalence_check=True if i!=3 else False)  # skip equivalence for last case (FP mul E5F10) due to complexity

    print("\nAll module cases passed round-trip + sim checks.")
    print(f"Tested {n_cases} module(s).")


if __name__ == "__main__":
    test_run()