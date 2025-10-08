import hashlib
import os
import random
import tempfile
import time
from typing import Callable, Dict, List, Optional, Tuple

from aigverse import aig_cut_rewriting, aig_resubstitution, sop_refactoring
from pyosys import libyosys as ys

from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag
from sprouthdl.sprouthdl import Expr
from sprouthdl.sprouthdl_aiger import AigerExporter, AigerImporter
from sprouthdl.sprouthdl_io_collector import IOCollector
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator


def optimize_aag(aag_lines: List[str], n_iter_optimizations=10) -> List[str]:

    # convert to aigverse object
    aig = conv_aag_into_aig(aag_lines)
    
    for i in range(n_iter_optimizations):
        for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting]: #, balancing]: balancing increases size
            optimization(aig)

    # aig back to aag
    aag_optimized = conv_aig_into_aag(aig, symbols=_get_aag_sym(aag_lines))
    
    return aag_optimized


def get_rand_hash() -> str:
    random_string = str(random.random()) + str(time.time())
    hash_object = hashlib.sha256(random_string.encode())
    name = str(hash_object.hexdigest())
    return name


def refactor_module_to_aig(module: Module, optimize=True, n_iter_optimizations=10) -> Module:
    # -- swact --
    optim = True
    # get AIG
    aag = AigerExporter(module).get_aag()
    if optim:
        aag = optimize_aag(aag, n_iter_optimizations=n_iter_optimizations)
    m_aig = AigerImporter(aag).get_sprout_module()
    spec = module.component.get_spec()
    IOCollector().group(m_aig, spec) # regroup I/Os to match original port widths
    return m_aig

# -- sim

def run_vectors(
    m: Module, vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]], *, 
    decoder: Callable[[int], float] | None = None, exprs: List[Expr] = [],
    use_signed: bool = False,
) -> None:
    """
    Generic runner:
      vectors: list of (label, inputs{name->int}, expected{name->int})
    Prints mismatches; raises AssertionError at the end if any failed.
    """

    states_list = []

    sim = Simulator(m)
    fails = 0
    for name, ins, outs in vectors:
        for k, v in ins.items():
            sim.set(k, v)
        sim.eval()
        bad = []
        for oname, exp in outs.items():
            got_raw = sim.peek(oname) #sim.get(oname)
            got_signed = sim.get(oname)
            got = got_signed if use_signed else got_raw
            if got != exp:
                if decoder and oname == "y":
                    bad.append(f"{oname}: got=0x{got:0X} ({decoder(got):.8g})  exp=0x{exp:0X} ({decoder(exp):.8g})")
                else:
                    bad.append(f"{oname}: got=0x{got:0X}  exp=0x{exp:0X}")
            else:
                if decoder and oname == "y":
                    # print(f"PASS {name}: {oname}=0x{got:0X} ({decoder(got):.8g})")
                    pass
                else:
                    # print(f"PASS {name}: {oname}=0x{got:0X} ({got})")
                    pass
        if bad:
            fails += 1
            print(f"FAIL  {name}:  " + " | ".join(bad))
        state = sim.log_expression_states(exprs)
        # convert expr to id
        state = [(id(e), v) for e, v in state]
        # and convert to dict for easy comparison
        state = dict(state)
        states_list.append(state)
    if fails:
        raise AssertionError(f"{fails}/{len(vectors)} vectors failed")

    print(f"Number of vectors: {len(vectors)}, {fails} failures")

    return states_list

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
            f.write("strash; &get -n; &deepsyn -I 500 -J 200; &put\n") # I: stop after I iterations without any improvement, J: number of random initializations, default: I: 20 j: 500
    ys.run_pass(f"{prepend}design -reset")
    ys.run_pass(f"{prepend}read_aiger {aag_tmp_file}")
    ys.run_pass(f"{prepend}rename -top top")
    # ys.run_pass("hierarchy -top top")
    ys.run_pass(f"{prepend}hierarchy -check")
    ys.run_pass(f"{prepend}proc; {prepend}opt; {prepend}fsm; {prepend}memory; {prepend}opt")
    if deepsyn:
        ys.run_pass(f"abc -script {abc_tmp_file}")
    ys.run_pass(f"{prepend}techmap; {prepend}opt; {prepend}abc -fast; {prepend}opt")
    #ys.run_pass("stat  -tech cmos")
    ys.run_pass(f"tee -q -o {stat_tmp_file} stat -tech cmos -json")

    # todo get aiger stat

    # read stats from json file
    import json
    with open(stat_tmp_file, "r") as f:
        stats = json.load(f)

    os.remove(stat_tmp_file)
    os.remove(aag_tmp_file)
    stats = stats["modules"]["\\top"]
    stats['estimated_num_transistors'] = int(stats['estimated_num_transistors'])
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
