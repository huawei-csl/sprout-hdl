import os
import tempfile

# Pyosys
from pyosys import libyosys as ys


def extract_yosys_metrics(aag_lines: list[str]) -> dict:

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

    ys.run_pass(f"{prepend}design -reset")
    ys.run_pass(f"{prepend}read_aiger {aag_tmp_file}")
    ys.run_pass(f"{prepend}rename -top top")
    # ys.run_pass("hierarchy -top top")
    ys.run_pass(f"{prepend}hierarchy -check")
    ys.run_pass(f"{prepend}proc; {prepend}opt; {prepend}fsm; {prepend}memory; {prepend}opt")
    ys.run_pass(f"{prepend}techmap; {prepend}opt; {prepend}abc -fast; {prepend}opt")
    #ys.run_pass("stat  -tech cmos")
    ys.run_pass(f"tee -q -o {stat_tmp_file} stat -tech cmos -json")

    # read stats from json file
    import json
    with open(stat_tmp_file, "r") as f:
        stats = json.load(f)

    os.remove(stat_tmp_file)
    os.remove(aag_tmp_file)
    return int(stats["modules"]["\\top"]["estimated_num_transistors"])
