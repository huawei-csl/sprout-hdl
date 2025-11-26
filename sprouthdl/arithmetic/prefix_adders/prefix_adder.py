# sprout_prefix_adder.py
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Set, Tuple, Iterable, Optional, TypeAlias

from aigverse import DepthAig, aig_cut_rewriting, aig_resubstitution, balancing, sop_refactoring
from matplotlib import pyplot as plt
from sprouthdl.arithmetic.prefix_adders.prefix_adder_transform import get_multiscan_nodes_24, get_multiscan_nodes_32, prefix_nodes_to_ranges, zcg_24, zcg_32
from sprouthdl.helpers import get_yosys_metrics
from sprouthdl.sprouthdl_analyzer import GraphReport
from sprouthdl.sprouthdl_module import Module
from sprouthdl.aigerverse_aag_loader_writer import conv_aag_into_aig, read_aag_into_aig
from sprouthdl.sprouthdl import UInt, cat
from sprouthdl.sprouthdl_aiger import AigerExporter
from sprouthdl.sprouthdl_simulator import Simulator

Pair = Tuple[int, int]

# ------------------------- Utilities -------------------------


def _normalize_P(n: int, P: Any) -> Set[Pair]:
    """
    Accepts:
      - set of (i,j),
      - list of (i,j),
      - NxN list-of-lists with truthy entries where nodes exist.
    Returns: set of (i,j) with 0 <= j <= i < n and j < i (combine nodes only).
    Leaves (i,i) are implicit and NOT stored.
    """
    nodes: Set[Pair] = set()
    if isinstance(P, (set, list, tuple)) and P and isinstance(next(iter(P)), tuple):
        for i, j in P:
            if not (0 <= j <= i < n):
                raise ValueError(f"Node {(i,j)} out of range for n={n}")
            if i != j:
                nodes.add((i, j))
        return nodes
    # assume matrix
    if not hasattr(P, "__len__"):
        raise TypeError("Unsupported P; pass set[(i,j)] or NxN list")
    if len(P) != n:
        raise ValueError(f"P must be {n}x{n}")
    for i in range(n):
        row = P[i]
        if len(row) != n:
            raise ValueError(f"P row {i} must have length {n}")
        for j in range(n):
            if j <= i and row[j]:
                if i != j:
                    nodes.add((i, j))
    return nodes


def _exists(nodes: Set[Pair], i: int, j: int) -> bool:
    """Does (i,j) exist as a node (combine) or leaf?"""
    return (i == j) or ((i, j) in nodes)

def legalize_P(n: int, nodes: Set[Pair]) -> Set[Pair]:
    """
    Ensure that the nodes are legal for a prefix adder:
      - (i,0) exists for all i=1..n-1
      - (i,i) exists for all i=0..n-1
      - (n-1,0) exists
    Returns: normalized set of nodes.
    """
    nodes = _normalize_P(n, nodes)

    # add all (i,0) for i=1..n-1
    for i in range(1, n):
        if not (i, 0) in nodes:
            nodes.add((i, 0))

    def exists(i, j):
        return (i == j) or ((i, j) in nodes)
    
    def fix_node(i: int, j: int):
        if i == j:
            return
        for k in range(j, i):  # look for a valid split
            if exists(i, k + 1) and exists(k, j):
                return
        # choose a k
        k_new = i - 1
        nodes.add((i, k_new))
        fix_node(i, k_new)  # recurse to fix the new node
        if k_new-1 != j:
            nodes.add((k_new-1, j))
            fix_node(k_new - 1, j)  # recurse to fix the new node

    for node in nodes.copy():
        fix_node(*node)
        
    for i in range(1, n):
        fix_node(i, 0)  # ensure (n-1,0) exists
        
    return nodes

def gen_random_P(n: int, num_nodes: int) -> Set[Pair]:
    """
    Generate a random set of combine nodes (i,j) for a prefix adder.
    The set will contain num_nodes nodes, with 0 <= j <= i < n and j < i.
    """
    if num_nodes > n * (n - 1) // 2:
        raise ValueError(f"Too many nodes {num_nodes} for n={n}; max is {n*(n-1)//2}")
    
    nodes: Set[Pair] = set()
    while len(nodes) < num_nodes:
        i = random.randint(1, n - 1)
        j = random.randint(0, i - 1)
        nodes.add((i, j))
        
    # Ensure (i,0) for all i=1..n-1
    for i in range(1, n):
        nodes.add((i, 0))
        
    nodes = legalize_P(n, nodes)
    
    return nodes


def _find_split(nodes: Set[Pair], i: int, j: int) -> Optional[int]:
    """
    Find a k (j ≤ k < i) such that left=(i,k+1) and right=(k,j) both exist
    (either as combine nodes in 'nodes' or as leaves when left has k+1==i or right has k==j).
    Greedy from the top (largest k) to encourage shallow cones (typical for KS/BK/Sklansky).
    """
    for k in reversed(range(i - 1, j - 1, -1)): # actually needs  to be reversed
        left_ok  = _exists(nodes, i, k + 1)
        right_ok = _exists(nodes, k, j)
        if left_ok and right_ok:
            return k
    return None

def _valid_splits(nodes: Set[Pair], i: int, j: int) -> List[int]:
    """All k with j ≤ k < i such that left=(i,k+1) and right=(k,j) both exist."""
    Ks: List[int] = []
    for k in range(j, i):
        if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
            Ks.append(k)
    return Ks

def analyze_prefix_matrix(
    n: int,
    nodes: Set[Pair],
    *,
    tie_break: str = "min_k",  # among equal-min-depth splits: "max_k" | "min_k" | "balanced"
) -> Tuple[Set[Pair], Dict[Pair, int], Dict[Pair, int], Dict[int, int], int]:
    """
    Compute minimal combine-depth for every (i,j) in P (and leaves), choose best splits, and
    return:
      nodes:   normalized set of (i,j)
      depth:   map (i,j) -> minimal black-cell depth
      best_k:  map (i,j) -> chosen split k that achieves depth[i,j] (only for i>j)
      carry_depth: map i -> depth(i,0)     (depth to compute G[i:0])
      worst_depth: max over i of depth(i,0)
    """
    depth: Dict[Pair, int] = {}
    best_k: Dict[Pair, int] = {}

    def node_depth(i: int, j: int) -> int:
        key = (i, j)
        if i == j:
            depth[key] = 0
            return 0
        if key in depth:
            return depth[key]
        if key not in nodes:
            raise ValueError(f"P does not declare combine node {(i,j)} but it is required.")
        Ks = _valid_splits(nodes, i, j)
        if not Ks:
            raise ValueError(f"No valid split for node {(i,j)}. Check your P matrix.")
        # Evaluate all splits: depth = 1 + max(depth(left), depth(right))
        cand: List[Tuple[int, int, int, int]] = []  # (depth, k, dl, dr)
        for k in Ks:
            dl = node_depth(i, k + 1)
            dr = node_depth(k, j)
            d = 1 + max(dl, dr)
            cand.append((d, k, dl, dr))
        # Choose min depth; tie-break as requested
        minD = min(c[0] for c in cand)
        best = [c for c in cand if c[0] == minD]
        if tie_break == "max_k":
            chosen = max(best, key=lambda t: t[1])
        elif tie_break == "min_k":
            chosen = min(best, key=lambda t: t[1])
        elif tie_break == "balanced":
            # minimize subtree imbalance; tie-break to larger k
            def imbalance(t):
                _, k, dl, dr = t
                return (abs(dl - dr), -k)
            chosen = min(best, key=imbalance)
        else:
            raise ValueError(f"Unknown tie_break '{tie_break}'")
        depth[key] = chosen[0]
        best_k[key] = chosen[1]
        return depth[key]

    # Ensure carries to bit 0 exist; compute depths for all required (i,0)
    for i in range(n):
        node_depth(i, 0)

    # Optionally compute depth for every declared node (fills depth map)
    for (i, j) in nodes:
        node_depth(i, j)

    carry_depth = {i: depth[(i, 0)] for i in range(n)}
    worst_depth = max(carry_depth.values()) if carry_depth else 0
    return depth, best_k, carry_depth, worst_depth


# ------------------------- Builder -------------------------

def build_prefix_adder_from_matrix(
    name: str,
    n: int,
    P: Any,
    *,
    with_cin: bool = False,
    with_cout: bool = True,
    depth_optimize: bool = True,
) -> Module:
    """
    Build an n-bit SproutHDL adder using a prefix tree specified by P.

    P: either set/list of (i,j) pairs (combine nodes) OR an n×n matrix (truthy at [i][j]).
       Leaves (i,i) are implicit and must NOT be listed.

    Ports:
      - a: UInt(n), b: UInt(n)
      - cin: UInt(1)  (optional)
      - y: UInt(n)
      - cout: UInt(1) (optional)

    Conventions:
      p_i = a_i ^ b_i, g_i = a_i & b_i
      combine: (G,P) ⊗ (G',P') = (G | (P & G'), P & P')
    """
    nodes = _normalize_P(n, P)
    if depth_optimize:
        depth, best_k, _, _ = analyze_prefix_matrix(n, nodes)
    m = Module(name, with_clock=False, with_reset=False)

    a = m.input(UInt(n), "a")
    b = m.input(UInt(n), "b")
    if with_cin:
        cin = m.input(UInt(1), "cin")
    else:
        cin = 0  # constant 0 (DSL const)

    y   = m.output(UInt(n), "y")
    if with_cout:
        cout = m.output(UInt(1), "cout")

    # bit locals
    p = [a[i] ^ b[i] for i in range(n)]  # XOR-propagate (good for sum)
    g = [a[i] & b[i] for i in range(n)]

    # memo for (G,P) at any (i,j) that exists (or leaf) - GP means generate, propagate
    GP: Dict[Pair, Tuple[Any, Any]] = {}

    def gp(i: int, j: int) -> Tuple[Any, Any]:
        """Return (G[i:j], P[i:j]) according to the matrix-defined tree."""
        key = (i, j)
        if key in GP:
            return GP[key]
        if i == j:
            GP[key] = (g[i], p[i])
            return GP[key]
        if (i, j) not in nodes:
            raise ValueError(f"Matrix P does not declare a combine node at {(i,j)}")
        if depth_optimize:
            k = best_k[(i, j)]
        else:
            k = _find_split(nodes, i, j)
        if k is None:
            raise ValueError(f"No valid split for node {(i,j)}. Check your P matrix.")
        Gl, Pl = gp(i, k + 1)
        Gr, Pr = gp(k, j)
        Gij = Gl | (Pl & Gr)
        Pij = Pl & Pr
        GP[key] = (Gij, Pij)
        print(f"produced node {(i,j)} out of ({i,k+1}) and ({k,j})")
        return GP[key]

    # sanity: for every i, we must be able to get the prefix up to bit 0 (directly or via its tree).
    # If (i,0) is not explicitly declared, we still require that the tree contains a path to combine to j=0.
    # The 'gp' will enforce presence of the needed nodes; if it misses, it will raise.
    # Carry chain:
    c = [None] * (n + 1)
    c[0] = cin
    for i in range(n):
        # carry into bit i+1:
        Gi0, Pi0 = gp(i, 0)
        c[i + 1] = Gi0 | (Pi0 & cin)

    # sum bits
    s = [p[i] ^ c[i] for i in range(n)]
    y <<= cat(*s)
    if with_cout:
        cout <<= c[n]
    return m

def validate_legality(n: int, nodes: Set[Pair]) -> bool:
    def exists(i,j): return (i == j) or ((i,j) in nodes)
    for (i,j) in nodes:
        ok = False
        if i==j:
            continue # no split needed
        for k in range(j, i):  # look for a valid split
            if exists(i, k+1) and exists(k, j):
                ok = True
                break
        if not ok:
            print(f"Node {(i,j)} has no legal split (i.e., no k with (i,k+1) and (k,j) existing).")
            return False
    # check that (n-1,0) exists (carry to bit 0)
    if not exists(n - 1, 0):
        print(f"Carry to bit 0 (node {(n-1,0)}) is not declared in P.")
        return False
    # check that all (i, 0) in nodes
    for i in range(n):
       if not (i, 0) in nodes:
           print(f"Node {(i,0)} is not declared in P (carry to bit 0).")
           return False
    return True


# ------------------------- Famous topologies -------------------------


# def P_kogge_stone(n: int) -> Set[Pair]:
#     """
#     Kogge–Stone nodes: at stage s (s=0..), span = 2^{s+1}-1, nodes (i, i-span) for i >= span.
#     """
#     nodes: Set[Pair] = set()
#     span = 1
#     while span < n:
#         # span = 2^{s+1}-1 sequence: 1,3,7,15,...
#         for i in range(span, n):
#             nodes.add((i, i - span))
#         span = (span << 1) | 1
#     return nodes

def P_ripple_carry(n: int) -> Set[Pair]:
    """Only (i,0) for i=1..n-1. Leaves (i,i) are implicit."""
    return {(i, 0) for i in range(1, n)}

def P_kogge_stone(n: int) -> Set[Pair]:
    """
    Kogge–Stone nodes over indices 0..n-1.
    At stage s, span = 2^{s+1} - 1 -> 1,3,7,15,...
    Add:
      - the KS pairs (i, i-span) for i >= span
      - (i, i) and (i, 0) for all i=1..n-1
    """
    nodes: Set[Pair] = set()

    # Ensure (i, i) and (i, 0) exist for i=1..n-1
    for i in range(1, n):
        #nodes.add((i, i)) # implicit in the tree
        nodes.add((i, 0))

    # Kogge–Stone spans: 1, 3, 7, 15, ...
    span = 1
    while span < n:
        for i in range(span, n):
            nodes.add((i, i - span))
        span = (span << 1) | 1  # 2*span + 1

    return nodes


def P_sklansky(n: int) -> Set[Pair]:
    """
    Sklansky (divide&conquer): for stage s, block=2^{s+1}, half=2^s.
    For each block starting at g, connect all i in [g+half .. g+block-1] to j=g.
    """
    nodes: Set[Pair] = set()
    s = 0
    while (1 << (s + 1)) <= n:
        block = 1 << (s + 1)
        half = 1 << s
        for g in range(0, n, block):
            j = g
            upper = min(g + block - 1, n - 1)
            for i in range(g + half, upper + 1):
                nodes.add((i, j))
        s += 1
    return nodes


def P_brent_kung(n: int) -> Set[Pair]:
    """
    Brent–Kung nodes as (i,j), where the node *outputs* the prefix over [j..i].
    We simulate upsweep (reduction) and downsweep (distribution) while tracking,
    for each bit index t, the current available prefix start j_at[t].
    """
    import math

    nodes: Set[Pair] = set()
    if n <= 1:
        return nodes

    L = math.ceil(math.log2(n))

    # j_at[t] = lowest index j such that we currently have prefix for [j..t]
    j_at = [i for i in range(n)]  # initially only leaves (i,i)

    # ---- Upsweep: compute group prefixes at block ends ----
    # At level s, step = 2^(s+1), half = 2^s.
    # For i = step-1, step-1+step, ... we combine (i, i-half+1) with prefix at r=i-half.
    for s in range(L):
        step = 1 << (s + 1)
        half = 1 << s
        for i in range(step - 1, n, step):
            r = i - half                # right index feeds the lower-half prefix
            j = j_at[r]                 # right child's j determines the output j
            nodes.add((i, j))           # this black cell outputs prefix [j..i]
            j_at[i] = j                 # update available prefix at i

    # ---- Downsweep: distribute prefixes to the missing positions ----
    # For level s = L-2 .. 0: positions i = 3*2^s - 1, then + step
    # Combine (i, i-half+1) with the prefix available at r=i-half (which already holds a wider j).
    for s in range(L - 2, -1, -1):
        step = 1 << (s + 1)
        half = 1 << s
        start = (3 << s) - 1           # 3*2^s - 1
        for i in range(start, n, step):
            r = i - half
            j = j_at[r]                 # right child's j propagates to the output
            nodes.add((i, j))
            j_at[i] = j

    return nodes

def ParallelScan_8_a(n: int) -> Set[Pair]:
    
    assert n == 8

    nodes: Set[Pair] = set()
    
    for i in range(1, n):        
        nodes.add((i, 0))
        
    for i in range(0, n):
        nodes.add((i, i))
    
    #nodes.add((1,0))
    nodes.add((3,2))
    nodes.add((5,4))
    nodes.add((7,6))
    #nodes.add((3, 0))
    #nodes.add((5,0))

    return nodes

def ParallelScan_8_b(n: int) -> Set[Pair]:
    
    assert n == 8

    nodes: Set[Pair] = set()
    
    for i in range(1, n):        
        nodes.add((i, 0))
        
    for i in range(0, n):
        nodes.add((i, i))    
        
    nodes.add((1,0))
    #nodes.add((3,0))
    nodes.add((4,3))
    nodes.add((5,3))
    nodes.add((7,6))
    #nodes.add((5,0))
    
    return nodes

def ParallelScan_16_a(n: int) -> Set[Pair]:
    
    assert n == 16

    nodes: Set[Pair] = set()
    
    for i in range(1, n):        
        nodes.add((i, 0))
        
    for i in range(0, n):
        nodes.add((i, i))
    
    #nodes.add((1,0))
    nodes.add((3,2))
    nodes.add((5,4))
    nodes.add((7,6))
    nodes.add((9,8))
    nodes.add((11,10))
    nodes.add((13,12))
    nodes.add((15,14))
    
    nodes.add((7,4))
    nodes.add((11,8))

    return nodes

def ParallelScan_16_b(n: int) -> Set[Pair]:
    
    assert n == 16

    nodes: Set[Pair] = set()
    
    for i in range(1, n):        
        nodes.add((i, 0))
        
    for i in range(0, n):
        nodes.add((i, i))
    
    nodes.add((1,0))
    nodes.add((3,0))
    nodes.add((4,3))
    nodes.add((5,3))
    nodes.add((7,6))
    nodes.add((8,6))
    nodes.add((10,9))
    nodes.add((11,9))
    nodes.add((13,12))
    nodes.add((14,12))
    #nodes.add((5,0))
    nodes.add((11,6))
    #nodes.add((11,0))

    return nodes

def get_num_nodes(P: Set[Pair], n: int, cleanup=True) -> int:
    
    # check if legal
    if cleanup:
        P = add_trivial_nodes(P, n)
    assert validate_legality(n, P)
    
    n_nodes = 0
    for p_elem in P:
        if p_elem[0] == p_elem[1]:
            continue
        n_nodes += 1
    return n_nodes

def add_trivial_nodes(P: Set[Pair], n: int) -> Set[Pair]:
    
    P_new = P.copy()
    
    for i in range(n):
        P_new.add((i,i))  # leaves
    for i in range(1,n):
        P_new.add((i,0))  # carry to bit 0
        
    return P_new

def get_depth(P: Set[Pair], n: int) -> int:
    
    def get_depth(current_node: Pair, current_depth: int) -> int:
        
        if current_node[0] == current_node[1]:
            return current_depth
        
        # find split
        k = _find_split(P, current_node[0], current_node[1])
        assert k is not None
        
        left_node = (current_node[0], k+1)
        right_node = (k, current_node[1])
        
        left_depth = get_depth(left_node, current_depth + 1)
        right_depth = get_depth(right_node, current_depth + 1)
        
        return max(left_depth, right_depth)
    
    # start with inputs and got thourgh all leaves
    max_depth_overall = 0
    for i in range(n):
        node = (i,0)
        depth = get_depth(node, 0)
        if depth > max_depth_overall:
            max_depth_overall = depth

    return max_depth_overall

# ------------------------- Test Vectors -------------------------

def build_prefix_adder_vectors_cin0():
    return [  # a, b, cin, y, ulp
        ("1+6", 1, 6, 0, 7),
         # ...
    ]

# (name, a, b, cin, y_exp, cout_exp)
Vec: TypeAlias = Tuple[str, int, int, int, int, int]

def build_adder_vectors16(num_random: int = 512, seed: int = 0xADDEF) -> List[Vec]:
    n = 8
    M = (1 << n) - 1
    V: List[Vec] = []

    def add(name: str, a: int, b: int, cin: int):
        total = (a & M) + (b & M) + (cin & 1)
        y = total & M
        co = (total >> n) & 1
        V.append((name, a & M, b & M, cin & 1, y, co))

    # --- Directed: extremes & big numbers ---
    add("zero+zero",        0x0000, 0x0000, 0)
    add("max+zero",         0xFFFF, 0x0000, 0)
    add("max+one",          0xFFFF, 0x0001, 0)
    add("max+max",          0xFFFF, 0xFFFF, 0)
    add("half+half",        0x8000, 0x8000, 0)
    add("near-msb-carry",   0x7FFF, 0x0001, 0)
    add("cin-only",         0x0000, 0x0000, 1)
    add("max+zero+cin",     0xFFFF, 0x0000, 1)
    add("max+max+cin",      0xFFFF, 0xFFFF, 1)

    # Famous hexes / “large numbers”
    add("DEAD+BEEF",        0xDEAD, 0xBEEF, 0)
    add("C0DE+F00D",        0xC0DE, 0xF00D, 0)
    add("FACE+FEED",        0xFACE, 0xFEED, 0)
    add("ACDC+1337+cin",    0xACDC, 0x1337, 1)
    add("8001+7FFE",        0x8001, 0x7FFE, 0)
    add("FF00+00FF",        0xFF00, 0x00FF, 0)

    # --- Patterns: alternating/blocks ---
    for a,b in [(0x5555,0xAAAA), (0x3333,0xCCCC), (0x0F0F,0xF0F0),
                (0xF00F,0x0FF0), (0xAA55,0x55AA), (0xFFFF,0x5555)]:
        add(f"pat {a:04X}+{b:04X}", a, b, 0)
        add(f"pat {a:04X}+{b:04X}+cin", a, b, 1)

    # --- Carry-ripple lengths: (2^k−1)+1 ---
    for k in range(1, 16):
        add(f"ripple_len_{k}", (1 << k) - 1, 0x0001, 0)

    # --- One-hot sweeps vs full/one-hot ---
    for k in range(16):
        add(f"onehot{k}+full",      1 << k, M, 0)
        add(f"onehot{k}+onehot{k}", 1 << k, 1 << k, 1)  # tests carry-in handling

    # --- Near-boundary big cases ---
    edges = [
        (0xFFFE, 0x0001, 0), (0xFFFE, 0x0001, 1),
        (0x7FFF, 0x8000, 0), (0x7FFF, 0x8000, 1),
        (0x8000, 0x7FFF, 1), (0x4000, 0x4000, 1),
        (0xFFF0, 0x0010, 0), (0xFFF0, 0x0010, 1),
        (0x8000, 0x0000, 1), (0x0001, 0x0001, 1),
    ]
    for a,b,c in edges:
        add(f"edge {a:04X}+{b:04X}+c{c}", a, b, c)

    # --- Reproducible randoms ---
    rng = random.Random(seed)
    for i in range(num_random):
        a = rng.randrange(1 << 16)
        b = rng.randrange(1 << 16)
        c = rng.randrange(2)
        add(f"rnd{i:04d}", a, b, c)

    # filter out vectors with cin not equal 0
    V = [v for v in V if v[3] == 0]

    return V

def build_adder_verctorsn_rand(n: int, num_random: int = 512, seed: int = 0xADDEF) -> List[Vec]:
    M = (1 << n) - 1
    V: List[Vec] = []

    def add(name: str, a: int, b: int, cin: int):
        total = (a & M) + (b & M) + (cin & 1)
        y = total & M
        co = (total >> n) & 1
        V.append((name, a & M, b & M, cin & 1, y, co))

    rng = random.Random(seed)
    for i in range(num_random):
        a = rng.randrange(1 << n)
        b = rng.randrange(1 << n)
        c = 0 #c = rng.randrange(2)
        add(f"rnd{i:04d}", a, b, c)

    return V

def run_vectors(mod, vectors, *, label="") -> bool:
    sim = Simulator(mod)
    print(f"\n== {label} ==")
    ok = 0
    for name, a, b, cin, y, cout in vectors:
        sim.set("a", a).set("b", b).eval()
        goty = sim.get("y")
        cout_available = True
        gotcout = sim.get("cout") if cout_available else None
        pass_fail = "PASS" if goty == y  and gotcout == cout else "FAIL"

        #print(f"{pass_fail:4s}  {name:25s}  a=0x{a:04X}  b=0x{b:04X}  cin=0x{cin:04X} -> y=0x{goty:04X}  (exp 0x{y:04X})  cout=0x{gotcout:04X} (exp 0x{cout:04X})")
        if goty == y and (gotcout == cout if gotcout is not None else cout is None):
            ok += 1
    print(f"Summary: {ok}/{len(vectors)} passed.\n")
    return ok == len(vectors)  # True if all passed


def P_to_matrix(n: int, nodes: Iterable[Pair]) -> List[List[int]]:
    """(Optional) helper to render a set of pairs into an n×n 0/1 matrix (for printing/debug)."""
    M = [[0] * n for _ in range(n)]
    for i, j in nodes:
        if 0 <= j <= i < n and i != j:
            M[i][j] = 1
    for i in range(n):
        M[i][i] = 1  # leaves shown as 1 for readability (not required by builder)
    return M


@dataclass
class AigReport:
    size: int
    depth: int
    optimized_size: int = 0
    optimized_depth: int = 0

def get_stats(nodes, n, name) -> Tuple[GraphReport, AigReport]:

    m = build_prefix_adder_from_matrix(name, n, nodes, with_cin=False, with_cout=True, depth_optimize=True)
    # print(m.to_verilog())

    sim = Simulator(m)
    sim.set("a", 3).set("b", 5).eval()
    print("y =", sim.get("y"))

    passed = run_vectors(m, build_adder_verctorsn_rand(n, num_random=20, seed=0xADDEF), label="Prefix Adder Test Vectors 16-bit Random")
    if not passed:
        raise RuntimeError("Prefix adder test vectors failed.")

    aag = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag)

    # Clone the AIG network for size comparison
    aig_clone = aig.clone()
    # Optimize the AIG with several optimization algorithms
    n_iter_optimizations = 10
    for i in range(n_iter_optimizations):
        for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
            optimization(aig)

    print(f"Results for {name} (n={n}):")
    print(f"Original AIG Size:  {aig_clone.size()}")
    print(f"Optimized AIG Size: {aig.size()}")
    print(f"Original AIG Depth: {DepthAig(aig_clone).num_levels()}")
    print(f"Optimized AIG Depth: {DepthAig(aig).num_levels()}")

    aig_report = AigReport(
        size=aig.size(),
        depth=DepthAig(aig).num_levels(),
        optimized_size=aig_clone.size(),
        optimized_depth=DepthAig(aig_clone).num_levels()
        )
    graph_report = m.module_analyze(include_wiring=True, include_consts=True)

    prefix_graph_stats = {"num_nodes": get_num_nodes(nodes, n), "depth": get_depth(nodes, n)}

    yosys_stats = get_yosys_metrics(m)

    return graph_report, aig_report, prefix_graph_stats, yosys_stats

def get_min_max_prefix_tree_range(results_vec):
    # extract from tuple
    min_num_nodes = None
    max_num_nodes = None
    min_depth = None
    max_depth = None
    for results in results_vec:
        name, graph_report, aig_report, prefix_graph_stats, yosys_stats = results
        if min_num_nodes is None:
            min_num_nodes = prefix_graph_stats["num_nodes"]
            max_num_nodes = prefix_graph_stats["num_nodes"]
            min_depth = prefix_graph_stats["depth"]
            max_depth = prefix_graph_stats["depth"]
        else:
            if prefix_graph_stats["num_nodes"] < min_num_nodes:
                min_num_nodes = prefix_graph_stats["num_nodes"]
            if prefix_graph_stats["num_nodes"] > max_num_nodes:
                max_num_nodes = prefix_graph_stats["num_nodes"]
            if prefix_graph_stats["depth"] < min_depth:
                min_depth = prefix_graph_stats["depth"]
            if prefix_graph_stats["depth"] > max_depth:
                max_depth = prefix_graph_stats["depth"]
    return min_num_nodes, max_num_nodes, min_depth, max_depth


def main_test():

    n_random = 100
    n_bits_vec = [8, 16, 24, 32]

    for n in n_bits_vec:
        print(f"\n\n\nTesting prefix adders with n={n} bits...")

        configs = [
            (P_ripple_carry(n), "Ripple Carry"),
            (P_kogge_stone(n), "Kogge-Stone"),
            (P_sklansky(n), "Sklansky"),
            (P_brent_kung(n), "Brent-Kung"),
        ]

        if n == 8:

            assert validate_legality(n, ParallelScan_8_a(n))
            assert validate_legality(n, ParallelScan_8_b(n))
            configs += [(ParallelScan_8_a(n), "Parallel Scan 8a")]
            configs += [(ParallelScan_8_b(n), "Parallel Scan 8b")]
            assert get_num_nodes(ParallelScan_8_a(n), n) == 10
            assert get_num_nodes(ParallelScan_8_b(n), n) == 10

        if n == 16:

            assert validate_legality(n, ParallelScan_16_a(n))
            assert validate_legality(n, ParallelScan_16_b(n))
            configs += [(ParallelScan_16_a(n), "Parallel Scan 16a")]
            configs += [(ParallelScan_16_b(n), "Parallel Scan 16b")]
            assert get_num_nodes(ParallelScan_16_a(n), n) == 24
            assert get_num_nodes(ParallelScan_16_b(n), n) == 24

        if n == 24:

            config = add_trivial_nodes(get_multiscan_nodes_24(), n)
            assert validate_legality(n, config)
            configs += [(config, "Parallel Scan 24")]

            config = add_trivial_nodes(prefix_nodes_to_ranges(zcg_24), n)
            assert validate_legality(n, config)
            configs += [(config, "ZCG 24")]

        if n == 32:

            config = add_trivial_nodes(get_multiscan_nodes_32(), n)
            assert validate_legality(n, config)
            configs += [(config, "Parallel Scan 32")]

            config = add_trivial_nodes(prefix_nodes_to_ranges(zcg_32), n)
            assert validate_legality(n, config)
            configs += [(config, "ZCG 32")]

        configs += [(gen_random_P(n, random.randint(5, 15)), f"Random {i}") for i in range(n_random)]

        results: List[Tuple[str, GraphReport, AigReport]] = []
        for nodes, name in configs:
            nodes = add_trivial_nodes(nodes, n)
            if not validate_legality(n, nodes):
                print(f"Invalid configuration for {name}. Skipping.")
                raise RuntimeError("Invalid configuration.")
            print(f"\nBuilding {name} prefix adder with n={n}...")
            graph_report, aig_report, prefix_graph_stats, yosys_stats = get_stats(nodes, n, name)
            results.append((name, graph_report, aig_report, prefix_graph_stats, yosys_stats))

        # plot results in scatter plot size vs depth, aig
        plt.figure(figsize=(6, 4))

        def plt_results(results_plot_list):
            markers = ['s', '^', 'D', 'v', 'P', '*', 'X', 'H', '<', '>']
            first_gray = True
            i_not_gray = 0

            x_rand = []
            y_rand = []

            for name, x, y in results_plot_list:
                if "Random" not in name:
                    continue  # plot randoms only in second pass
                x_rand.append(x)
                y_rand.append(y)

            plt.scatter(x_rand, y_rand, color="gray", label=f"Random ({n_random}x, legal)", marker="o", alpha=0.25)

            for name, x, y in results_plot_list:
                if "Random" in name:
                    continue  # plot randoms only in second pass

                label = name
                color = None
                # take from palette
                marker = markers[i_not_gray]
                i_not_gray = (i_not_gray + 1) % len(markers)
                alpha = 1.0
                plt.scatter(x, y, color=color, label=label, marker=marker, alpha=alpha)

        # transform results to

        results_plot_list = []
        for name, graph_report, aig_report, prefix_graph_stats, yosys_stats in results:
            results_plot_list.append((name, aig_report.size, aig_report.depth))
        plt_results(results_plot_list)

        plt.title(f"Prefix Adder AIG Size vs Depth ({n}-bit)")
        plt.xlabel("AIG Size")
        plt.ylabel("AIG Depth")
        # on upper right
        plt.legend(loc='upper right')
        plt.grid()
        plt.savefig(f"prefix_adder_aig_size_vs_depth_n{n}.png")

        # plot results in scatter plot size vs depth, optimized aig
        plt.figure(figsize=(6, 4))

        results_plot_list = []
        for name, graph_report, aig_report, prefix_graph_stats, yosys_stats in results:
            results_plot_list.append((name, aig_report.optimized_size, aig_report.optimized_depth))
        plt_results(results_plot_list)

        plt.title(f"Prefix Adder Optimized AIG Size vs Depth ({n}-bit)")
        plt.xlabel("Optimized AIG Size")
        plt.ylabel("Optimized AIG Depth")
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig(f"prefix_adder_optimized_aig_size_vs_depth_n{n}.png")

        plt.figure(figsize=(6, 4))
        results_plot_list = []
        for name, graph_report, aig_report, prefix_graph_stats, yosys_stats in results:
            results_plot_list.append((name, graph_report.op_nodes, graph_report.max_depth))
        plt_results(results_plot_list)

        plt.title(f"Prefix Adder Graph Size vs Depth ({n}-bit)")
        plt.xlabel("Graph Nodes")
        plt.ylabel("Graph Depth")
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig(f"prefix_adder_graph_size_vs_depth_n{n}.png")

        plt.figure(figsize=(6, 4))
        results_plot_list = []
        for name, graph_report, aig_report, prefix_graph_stats, yosys_stats in results:
            results_plot_list.append((name, prefix_graph_stats["num_nodes"], prefix_graph_stats["depth"]))
        plt_results(results_plot_list)
        # get min max
        min_num_nodes, max_num_nodes, min_depth, max_depth = get_min_max_prefix_tree_range(results)
        # plot the line y=2n-2-x in the range
        x_vals = list(range(min_num_nodes - 1, max_num_nodes + 2))
        y_vals = [2 * n - 2 - x for x in x_vals]
        # cut off values outside min_depth..max_depth
        x_vals = [x for x, y in zip(x_vals, y_vals) if min_depth <= y <= max_depth]
        y_vals = [y for y in y_vals if min_depth <= y <= max_depth]
        plt.plot(x_vals, y_vals, color="gray", linestyle="--", label="Bound: y=2*n_bit-2-x")
        plt.title(f"Prefix Adder Prefix Tree Size vs Depth ({n}-bit)")
        plt.xlabel("Prefix Tree Nodes")
        plt.ylabel("Prefix Tree Depth")
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig(f"prefix_adder_prefix_tree_size_vs_depth_n{n}.png")

        plt.figure(figsize=(6, 4))
        results_plot_list = []
        for name, graph_report, aig_report, prefix_graph_stats, yosys_stats in results:
            results_plot_list.append((name, yosys_stats["estimated_num_transistors"], graph_report.max_depth))
        plt_results(results_plot_list)
        plt.title(f"Prefix Adder Yosys Transistors vs Prefix Graph Depth ({n}-bit)")
        plt.xlabel("Yosys Transistors")
        plt.ylabel("Prefix Graph Depth")
        plt.legend(loc="upper right")
        plt.grid()
        plt.savefig(f"prefix_adder_yosys_transistors_vs_depth_n{n}.png")


if __name__ == "__main__":
    main_test()