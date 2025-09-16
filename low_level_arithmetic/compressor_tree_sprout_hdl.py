import random
from typing import List
from low_level_arithmetic.compressor_tree_multiplier_ex3 import build_wallace_compressor_graph, random_compressor_tree
from low_level_arithmetic.prefix_adder import Vec
from sprout_hdl_module import Module
from sprout_hdl_simulator import Simulator


def build_multiplier_from_compressor_graph(name: str, A, nodes):
    """
    Build a SproutHDL combinational multiplier module from a compressor-tree graph.

    Inputs inferred from graph:
      - n is inferred from pp[i,j] signal names (square n×n multiplier).

    Graph assumptions (as used in this chat):
      - Node(kind) ∈ {'sig','FA','HA','sink'}
      - For 'sig' nodes: port ∈ {'pp','sum','carry'}; weight = i+j for pp; carry has weight+1
      - Edges go: sig(w) -> (FA/HA at w), and (FA/HA at w) -> sum(w) / carry(w+1), and sig(w)->sink(w)
      - Final two rows either via explicit 'sink' nodes named "final_row_[A|B]_w{w}"
        or implicitly as signals with no outgoing edge to adders.

    Returns:
      - sprout_hdl_module.Module with ports:
          a: UInt(n), b: UInt(n), y: UInt(2*n)
    """
    import re
    import numpy as np
    from collections import defaultdict

    from sprout_hdl_module import Module
    from sprout_hdl import UInt, cat

    # --- helpers ---------------------------------------------------------------
    def infer_n():
        pp_nodes = [nd for nd in nodes if nd.kind == "sig" and nd.port == "pp"]
        if not pp_nodes:
            raise ValueError("No partial products found (sig/pp).")
        mi = mj = 0
        for nd in pp_nodes:
            m = re.search(r"pp\[(\d+),(\d+)\]", nd.name)
            if not m:
                raise ValueError(f"Bad pp name: {nd.name}")
            i, j = int(m.group(1)), int(m.group(2))
            mi = max(mi, i); mj = max(mj, j)
        if mi != mj:
            raise ValueError(f"Inferred non-square pp grid: max_i={mi}, max_j={mj}")
        return mi + 1

    def parse_pp_indices(name: str):
        m = re.search(r"pp\[(\d+),(\d+)\]", name)
        if not m:
            raise ValueError(f"Cannot parse pp indices from '{name}'")
        return int(m.group(1)), int(m.group(2))

    # Build predecessor/successor lists once
    rows, cols = np.nonzero(A)
    succ = {i: [] for i in range(A.shape[0])}
    pred = {i: [] for i in range(A.shape[0])}
    for u, v in zip(rows.tolist(), cols.tolist()):
        succ[u].append(v)
        pred[v].append(u)

    n = infer_n()
    W = 2 * n  # product width

    # --- Sprout module skeleton ------------------------------------------------
    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(n), "a")
    b = m.input(UInt(n), "b")
    y = m.output(UInt(W), "y")

    # One-bit expression for each signal node
    sig_expr = {}  # node_id -> Sprout bit expr

    # Initialize partial products
    for nd in nodes:
        if nd.kind == "sig" and nd.port == "pp":
            i, j = parse_pp_indices(nd.name)
            sig_expr[nd.idx] = (a[i] & b[j])

    # Compute adder outputs stage-by-stage to respect dataflow
    max_stage = max((nd.stage or 0) for nd in nodes)
    for st in range(1, max_stage + 1):
        for nd in nodes:
            if nd.kind in ("FA", "HA") and nd.stage == st:
                # gather inputs (signals) in any order; graph guarantees correct wiring by weight
                ins = [u for u in pred[nd.idx] if nodes[u].kind == "sig"]
                if nd.kind == "FA":
                    if len(ins) != 3:
                        raise ValueError(f"FA at idx={nd.idx} does not have 3 signal inputs")
                    x, yb, z = (sig_expr[ins[0]], sig_expr[ins[1]], sig_expr[ins[2]])
                    s_bit = (x ^ yb) ^ z
                    c_bit = (x & yb) | (x & z) | (yb & z)
                else:  # HA
                    if len(ins) != 2:
                        raise ValueError(f"HA at idx={nd.idx} does not have 2 signal inputs")
                    x, yb = (sig_expr[ins[0]], sig_expr[ins[1]])
                    s_bit = x ^ yb
                    c_bit = x & yb

                # drive successor signal nodes
                outs = [v for v in succ[nd.idx] if nodes[v].kind == "sig"]
                for v in outs:
                    if nodes[v].port == "sum":
                        sig_expr[v] = s_bit
                    elif nodes[v].port == "carry":
                        sig_expr[v] = c_bit

    # Collect final two rows per weight
    rowA = [0] * W
    rowB = [0] * W

    # Prefer explicit sinks if present
    sinks = [nd for nd in nodes if nd.kind == "sink"]
    if sinks:
        for sk in sinks:
            mname = re.search(r"final_row_(A|B)_w(\d+)", sk.name)
            if not mname:
                continue
            row_name, w = mname.group(1), int(mname.group(2))
            # find the single signal predecessor
            p_sig = [u for u in pred[sk.idx] if nodes[u].kind == "sig"]
            if not p_sig:
                continue
            bit = sig_expr[p_sig[0]]
            if w < W:
                if row_name == "A":
                    rowA[w] = bit
                else:
                    rowB[w] = bit
    else:
        # Implicit: signals that no longer feed adders are final bits
        finals_by_w = defaultdict(list)
        for nd in nodes:
            if nd.kind == "sig" and nd.port in ("sum", "carry"):
                if not any(nodes[v].kind in ("FA", "HA") for v in succ[nd.idx]):
                    finals_by_w[nd.weight].append(nd.idx)
        for w, ids in finals_by_w.items():
            ids_sorted = sorted(ids)
            if w < W and len(ids_sorted) >= 1:
                rowA[w] = sig_expr[ids_sorted[0]]
            if w < W and len(ids_sorted) >= 2:
                rowB[w] = sig_expr[ids_sorted[1]]

        # Also include any passthrough pp bits that were never consumed and land directly in final columns
        # (e.g., w=0 LSB)
        for nd in nodes:
            if nd.kind == "sig" and nd.port == "pp":
                if not any(nodes[v].kind in ("FA", "HA") for v in succ[nd.idx]):
                    w = nd.weight
                    if w < W and rowA[w] == 0:
                        rowA[w] = sig_expr[nd.idx]
                    elif w < W and rowB[w] == 0:
                        rowB[w] = sig_expr[nd.idx]

    # Ripple-carry adder for the two rows -> product bits
    carry = 0
    bits = []
    for w in range(W):
        a_bit = rowA[w] if w < len(rowA) else 0
        b_bit = rowB[w] if w < len(rowB) else 0
        s_bit = (a_bit ^ b_bit) ^ carry
        c_out = (a_bit & b_bit) | (a_bit & carry) | (b_bit & carry)
        bits.append(s_bit)
        carry = c_out
    # (product fits in W bits; overflow carry is discarded by design)

    y <<= cat(*reversed(bits))
    return m


def gen_compressor_tree_graph_and_sprout_module(n: int, policy: str = "dadda") -> Module:
    
    """
    Convenience function to build a compressor-tree multiplier graph and
    corresponding SproutHDL module for an n-bit unsigned multiplier.

    Args:
      name: module name
      n: input bitwidth (n x n multiplier)
      policy: "dadda" or "wallace"

    Returns:
      sprout_hdl_module.Module
    """

    N = n  # change to try other bitwidths; for plotting, keep <= 6
    if policy == "wallace":
        A, nodes = build_wallace_compressor_graph(N)
    elif policy == "random":
        A, nodes = random_compressor_tree(n=N, seed=42, shrink_range=(0.6, 0.95), p_fa=0.65)
    else:
        raise ValueError(f"Unknown policy '{policy}'; must be 'dadda', 'wallace', or 'random'")
    m = build_multiplier_from_compressor_graph(policy, A, nodes)
    return m

def build_mul_verctor_rand(n: int, num_random: int = 512, seed: int = 0xADDEF) -> List[Vec]:
    M = (1 << n) - 1
    V: List[Vec] = []

        
    def mul(a: int, b: int) -> int:
        total = (a & M) * (b & M)
        y = total & ((1 << (2*n)) - 1)
        V.append((f"mul{a:02d}x{b:02d}", a & M, b & M, y))
    

    rng = random.Random(seed)
    for i in range(num_random):
        a = rng.randrange(1 << n)
        b = rng.randrange(1 << n)
        c = 0 #c = rng.randrange(2)
        mul(a, b)
        
    return V

def run_vectors(mod, vectors, *, label="") -> bool:
    sim = Simulator(mod)
    print(f"\n== {label} ==")
    ok = 0
    for name, a, b, y in vectors:
        sim.set("a", a).set("b", b).eval()
        goty = sim.get("y")
        cout_available = True
        pass_fail = "PASS" if goty == y else "FAIL"

        print(f"{pass_fail:4s}  {name:25s}  a=0x{a:04X}  b=0x{b:04X}  -> y=0x{goty:04X}  (exp 0x{y:04X})")
        if goty == y:
            ok += 1
    print(f"Summary: {ok}/{len(vectors)} passed.\n")
    return ok == len(vectors)  # True if all passed
    
def main():
    m = gen_compressor_tree_graph_and_sprout_module(8, policy="wallace")
    run_vectors(m, build_mul_verctor_rand(8), label="8x8 Wallace Multiplier")
    
    m = gen_compressor_tree_graph_and_sprout_module(8, policy="random")
    run_vectors(m, build_mul_verctor_rand(8), label="8x8 Wallace Multiplier")
    
    
if __name__ == "__main__":
    main()