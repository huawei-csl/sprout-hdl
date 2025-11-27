import hashlib
import os
import random
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

from aigverse import DepthAig, aig_cut_rewriting, aig_resubstitution, balancing, sop_refactoring
from pyosys import libyosys as ys

from sprouthdl.aig.aig_aigerverse import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag
from sprouthdl.sprouthdl import Expr
from sprouthdl.sprouthdl_aiger import AigerExporter, AigerImporter
from sprouthdl.sprouthdl_module import IOCollector
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator

def optimize_aig_elaborate(aig_in) -> dict:

    # Track the best gate and depth
    best_gate = None
    best_depth = None
    
    best_stats = None
    best_aig = None

    # Two sequences are tested
    op_seq_tuple = ([aig_resubstitution, sop_refactoring, aig_cut_rewriting], [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing])

    # Loop through the two sequences
    for op_seq in op_seq_tuple:
        # Load the aig file
        aig = aig_in.clone()
        # Loop through the sequences 3 times
        for i in range(3):
            for op in op_seq:
                # Perform the operation
                op(aig)
                # Perform analysis
                depth_aig = DepthAig(aig)
                # Extract stats
                stats = {
                    "num_pis": len(aig.pis()),
                    "num_pos": len(aig.pos()),
                    "num_gates": len(aig.gates()),
                    "size": aig.size(),
                    "depth": depth_aig.num_levels(),
                }
                # Store best stats dict
                if best_gate is None or stats["num_gates"] < best_gate:
                    best_gate = stats["num_gates"]
                    best_depth = stats["depth"]
                    best_stats = stats
                    best_aig = aig
                elif stats["num_gates"] == best_gate and stats["depth"] < best_depth:
                    best_depth = stats["depth"]
                    best_stats = stats
                    best_aig = aig

    return best_aig, best_stats

def optimize_aig_simple(aig, n_iter_optimizations=10) -> dict:
    for i in range(n_iter_optimizations):
        for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting]: #, balancing]: balancing increases size
            optimization(aig)
    return aig

def optimize_aag(aag_lines: List[str], n_iter_optimizations=10, simple=False) -> List[str]:

    # convert to aigverse object
    aig = conv_aag_into_aig(aag_lines)    

    if simple:
        aig = optimize_aig_simple(aig, n_iter_optimizations=n_iter_optimizations)
    else:
        aig, _ = optimize_aig_elaborate(aig)

    # aig back to aag
    aag_optimized = conv_aig_into_aag(aig, symbols=_get_aag_sym(aag_lines))

    return aag_optimized
    aig = optimize_aig_elaborate(aig)[0]

    # aig back to aag
    aag_optimized = conv_aig_into_aag(aig, symbols=_get_aag_sym(aag_lines))
    
    return aag_optimized


def refactor_module_to_aig(module: Module, optimize=True, n_iter_optimizations=10) -> Module:
    # -- swact --
    optim = True
    # get AIG
    aag = AigerExporter(module).get_aag()
    if optim:
        aag = optimize_aag(aag, n_iter_optimizations=n_iter_optimizations)
    m_aig = AigerImporter(aag).get_sprout_module()
    try:
        spec = module.component.get_spec()
    except:
        spec = module.get_spec()
    IOCollector().group(m_aig, spec) # regroup I/Os to match original port widths
    return m_aig


def get_aig_stats(m: Module, n_iter_optimizations=10, simple=False) -> dict:
    aag_lines = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag_lines)

    if simple:
        aig = optimize_aig_simple(aig, n_iter_optimizations=n_iter_optimizations)
    else:
        aig, _ = optimize_aig_elaborate(aig)

    depth_aig = DepthAig(aig)

    stats = {
        'num_pis': len(aig.pis()),
        'num_pos': len(aig.pos()),
        'num_gates': len(aig.gates()),
        'size': aig.size(),
        'depth': depth_aig.num_levels(),
    }
    return stats


# -- sim

def run_vectors(
    m: Module, vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]], *, 
    decoder: Callable[[int], float] | None = None, exprs: Optional[List[Expr]] = None,
    use_signed: bool = False,
    raise_on_fail: bool = True,
    print_on_pass: bool = False,
) -> None:
    sim = Simulator(m)
    sim.trace_enabled = True if exprs is not None else False
    sim.traced_expressions = exprs if exprs is not None else []
    run_vectors_on_simulator(
        sim, vectors, decoder=decoder, use_signed=use_signed,
        raise_on_fail=raise_on_fail, print_on_pass=print_on_pass,
    )
    return sim.trace_history
    
    
def run_vectors_on_simulator(
    sim: Simulator, vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]], *, 
    decoder: Callable[[int], float] | None = None,
    use_signed: bool = False,
    raise_on_fail: bool = True,
    print_on_pass: bool = False,
) -> None:
    
    """
    Generic runner:
      vectors: list of (label, inputs{name->int}, expected{name->int})
    Prints mismatches; raises AssertionError at the end if any failed.
    """

    states_list = []

    fails = 0
    for name, ins, outs in vectors:
        for k, v in ins.items():
            if k[0] == "_":
                continue
            sim.set(k, v)
        sim.eval()
        bad = []
        for oname, exp in outs.items():
            if oname[0] == "_":
                continue
            got_raw = sim.peek(oname) #sim.get(oname)
            got_signed = sim.get(oname)
            got = got_signed if use_signed else got_raw
            if got != exp:
                if decoder and oname == "y":
                    bad.append(f"{oname}: got=0x{got:0X} ({decoder(got):.8g})  exp=0x{exp:0X} ({decoder(exp):.8g})")
                else:
                    bad.append(f"{oname}: got=0x{got:0X}  exp=0x{exp:0X}")
            else:
                if print_on_pass:
                    if decoder and oname == "y":
                        print(f"PASS {name}: {oname}=0x{got:0X} ({decoder(got):.8g})")
                    else:
                        print(f"PASS {name}: {oname}=0x{got:0X} ({got})")
                    pass
        if bad:
            fails += 1
            print(f"FAIL  {name}:  " + " | ".join(bad))
        # if exprs is not None:
        #     state = sim._get_expr_snapshot(exprs)
        #     # convert expr to id
        #     state = [(id(e), v) for e, v in state]
        #     # and convert to dict for easy comparison
        #     state = dict(state)
        #     states_list.append(state)
        
    print(f"Number of vectors: {len(vectors)}, {fails} failures")
    if fails and raise_on_fail:
        raise AssertionError(f"{fails}/{len(vectors)} vectors failed")


    #return states_list

def get_switch_count(states_list) -> float:
    # get all ids from step 0
    ids = set(states_list[0].keys())
    # count number of switches per id
    switches = {i: 0 for i in ids}
    for i in ids:
        last = states_list[0][i]
        for s in states_list[1:]:
            if s[i] != last:
                switches[i] += 1
                last = s[i]
    # sum up all switches
    total_switches = sum(switches.values())
    return total_switches / len(states_list)  # average per vector


def extract_yosys_metrics(aag_lines: list[str], deepsyn=False) -> dict:

    fd, aag_tmp_file = tempfile.mkstemp(suffix=".aag")
    os.close(fd)
    with open(aag_tmp_file, "w") as f:
        f.write("\n".join(aag_lines) + "\n")

    fd, stat_tmp_file = tempfile.mkstemp(suffix=".json")

    silence_output = True

    if silence_output:
        prepend = "tee -q "
    else:
        prepend = ""
    if deepsyn:
        # create abc script temp file
        fd_abc, abc_tmp_file = tempfile.mkstemp(suffix=".abc")
        os.close(fd_abc)
        with open(abc_tmp_file, "w") as f:
            # f.write("strash; &get -n; &deepsyn -I 500 -J 200 -T 10; &put\n") # I: stop after I iterations without any improvement, J: number of random initializations, default: I: 20 j: 500 Timeout T [seconds]
            f.write("strash; &get -n; &deepsyn J 5 -T 100; &put\n")
    ys.run_pass(f"{prepend}design -reset")
    ys.run_pass(f"{prepend}read_aiger {aag_tmp_file}")
    ys.run_pass(f"{prepend}rename -top top")
    # ys.run_pass("hierarchy -top top")
    ys.run_pass(f"{prepend}hierarchy -check")
    ys.run_pass(f"{prepend}proc; {prepend}opt; {prepend}fsm; {prepend}memory; {prepend}opt")
    if deepsyn:
        ys.run_pass(f"abc -script {abc_tmp_file}")
    ys.run_pass(f"{prepend}techmap; {prepend}opt; {prepend}abc -fast; {prepend}opt")
    ys.run_pass(f"{prepend}rename -wire -suffix _reg t:*DFF*")
    # ys.run_pass(f"{prepend}autoname t:*DFF* %n;")
    # ys.run_pass("stat  -tech cmos")
    ys.run_pass(f"tee -q -o {stat_tmp_file} stat -top top -tech cmos -json")

    # todo get aiger stat

    # read stats from json file
    import json
    with open(stat_tmp_file, "r") as f:
        stats = json.load(f)

    os.remove(stat_tmp_file)
    os.remove(aag_tmp_file)
    stats = stats["modules"]["\\top"]
    stats["estimated_num_transistors"] = int(stats["estimated_num_transistors"].replace("+", "")) # plus in case of registers, because they are not counted
    return stats


def get_yosys_metrics(m: Module, n_iter_optimizations=10, deepsyn=False) -> int:
    aag_lines = AigerExporter(m).get_aag()

    if n_iter_optimizations > 0:
        aag_lines = optimize_aag(aag_lines, n_iter_optimizations=n_iter_optimizations)

    stat = extract_yosys_metrics(aag_lines, deepsyn=deepsyn)
    return stat

def get_transistor_count_from_stats(stats: dict) -> int:
    return stats["estimated_num_transistors"]

def get_yosys_transistor_count(m: Module, n_iter_optimizations=10, deepsyn=False) -> int:
    stats = get_yosys_metrics(m, n_iter_optimizations=n_iter_optimizations, deepsyn=deepsyn)
    return get_transistor_count_from_stats(stats)