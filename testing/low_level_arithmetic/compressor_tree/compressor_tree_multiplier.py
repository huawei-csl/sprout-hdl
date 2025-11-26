# Wallace compressor tree graph → adjacency matrix → plots (usage demo)
# You can tweak `N` below to try other sizes. For plotting, N<=6 is recommended.

from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import math
import random
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil


# -----------------------------
# Graph representation
# -----------------------------


@dataclass
class Node:
    idx: int
    name: str
    kind: str  # 'sig', 'FA', 'HA', 'sink'
    weight: Optional[int] = None
    stage: Optional[int] = None
    port: Optional[str] = None  # 'pp', 'sum', 'carry' for signals


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)
    A: List[List[int]] = field(default_factory=list)  # adjacency matrix

    def add_node(self, name, kind, weight=None, stage=None, port=None) -> int:
        idx = len(self.nodes)
        self.nodes.append(Node(idx, name, kind, weight, stage, port))
        for row in self.A:
            row.append(0)
        self.A.append([0] * len(self.nodes))
        return idx

    def add_edge(self, u: int, v: int):
        self.A[u][v] = 1


# -----------------------------
# Wallace compressor tree builder
# -----------------------------


def build_wallace_compressor_graph(n: int) -> Tuple[np.ndarray, List[Node]]:
    """
    Build Wallace-style compressor tree for an n-bit unsigned multiplier.
    Returns (adjacency_matrix, nodes).
    """
    g = Graph()
    cols: Dict[int, List[int]] = defaultdict(list)  # weight -> signal node indices

    # Stage 0: partial products
    for i in range(n):
        for j in range(n):
            w = i + j
            s = g.add_node(f"pp[{i},{j}]", "sig", weight=w, stage=0, port="pp")
            cols[w].append(s)

    max_w = 2 * (n - 1)
    stage = 0
    while True:
        stage += 1
        next_cols: Dict[int, List[int]] = defaultdict(list)

        for w in range(max_w + stage + 1):  # allow growth from carries
            inputs = cols.get(w, []).copy()

            # Use FAs for groups of 3
            while len(inputs) >= 3:
                a, b, c = inputs.pop(), inputs.pop(), inputs.pop()
                ad = g.add_node(f"FA_s{stage}_w{w}", "FA", weight=w, stage=stage)
                g.add_edge(a, ad)
                g.add_edge(b, ad)
                g.add_edge(c, ad)
                s = g.add_node(f"s(w{w},st{stage})", "sig", weight=w, stage=stage, port="sum")
                co = g.add_node(f"c(w{w+1},st{stage})", "sig", weight=w + 1, stage=stage, port="carry")
                g.add_edge(ad, s)
                g.add_edge(ad, co)
                next_cols[w].append(s)
                next_cols[w + 1].append(co)

            # Use HA for leftover pair
            if len(inputs) == 2:
                a, b = inputs.pop(), inputs.pop()
                ad = g.add_node(f"HA_s{stage}_w{w}", "HA", weight=w, stage=stage)
                g.add_edge(a, ad)
                g.add_edge(b, ad)
                s = g.add_node(f"s(w{w},st{stage})", "sig", weight=w, stage=stage, port="sum")
                co = g.add_node(f"c(w{w+1},st{stage})", "sig", weight=w + 1, stage=stage, port="carry")
                g.add_edge(ad, s)
                g.add_edge(ad, co)
                next_cols[w].append(s)
                next_cols[w + 1].append(co)

            # Single leftover passes through
            if len(inputs) == 1:
                next_cols[w].append(inputs.pop())

        # Stop when each column height <= 2
        if all(len(v) <= 2 for v in next_cols.values()):
            cols = next_cols
            break
        cols = next_cols

    # Optional: attach final sinks (two rows per weight)
    for w in range(max(cols.keys()) + 1):
        arr = cols.get(w, [])
        if len(arr) == 2:
            a = g.add_node(f"final_row_A_w{w}", "sink", weight=w, stage=stage + 1)
            b = g.add_node(f"final_row_B_w{w}", "sink", weight=w, stage=stage + 1)
            g.add_edge(arr[0], a)
            g.add_edge(arr[1], b)
        elif len(arr) == 1:
            a = g.add_node(f"final_row_A_w{w}", "sink", weight=w, stage=stage + 1)
            g.add_edge(arr[0], a)

    A = np.array(g.A, dtype=np.uint8)
    return A, g.nodes

# -----------------------------
# Random compressor-tree builder
# -----------------------------

def _weighted_choice(rng, pairs: List[Tuple[int, int]], p_fa: float) -> Tuple[int, int]:
    """Choose (F,H) with bias toward FAs controlled by p_fa in [0,1]."""
    eps = 1e-3
    weights = [(eps + p_fa)**F * (eps + (1 - p_fa))**H for (F, H) in pairs]
    s = sum(weights)
    r = rng.random() * s
    acc = 0.0
    for (pair, w) in zip(pairs, weights):
        acc += w
        if r <= acc:
            return pair
    return pairs[-1]


def random_compressor_tree(n: int,
                           seed: Optional[int] = None,
                           shrink_range: Tuple[float, float] = (0.55, 0.9),
                           p_fa: float = 0.7,
                           max_stages: int = 32,
                           attach_sinks: bool = True) -> Tuple[np.ndarray, List[Node]]:
    """
    Build a random (but legal) compressor tree for an n-bit unsigned multiplier.
    - shrink_range: per-stage random factor r; target limit L_w = max(2, floor(r * height_w))
    - p_fa: bias towards full adders in [0,1] when multiple solutions exist
    - max_stages: safety cap; if not finished, final stages force L_w=2
    Returns: (adjacency_matrix, nodes)
    """
    rng = random.Random(seed)
    g = Graph()
    cols: Dict[int, List[int]] = defaultdict(list)

    # Stage 0: partial products
    for i in range(n):
        for j in range(n):
            w = i + j
            s = g.add_node(f"pp[{i},{j}]", "sig", weight=w, stage=0, port="pp")
            cols[w].append(s)

    # Helper to choose random FA/HA counts to reach a limit L given h inputs
    def choose_FA_H(h: int, L: int) -> Tuple[int, int]:
        if h <= L:
            return 0, 0
        feasible = []
        maxF = h // 3
        for F in range(maxF + 1):
            maxH = (h - 3*F) // 2
            for H in range(maxH + 1):
                # Next-stage column height at w becomes h - 2F - H
                if h - 2*F - H <= L:
                    if F + H >= 1:  # must compress if h>L
                        feasible.append((F, H))
        if not feasible:
            # Greedy fallback: prioritize FAs
            F = min(h // 3, (h - L) // 2)
            H = min((h - 3*F) // 2, h - 2*F - L)
            if F + H == 0 and h > L:
                H = min(1, (h - 3*F) // 2)
            return F, H
        return _weighted_choice(rng, feasible, p_fa)

    stage = 0
    for s in range(1, max_stages + 1):
        stage = s
        next_cols: Dict[int, List[int]] = defaultdict(list)

        # Random per-weight limits
        limits: Dict[int, int] = {}
        all_ws = list(cols.keys())
        for w in all_ws:
            h = len(cols[w])
            if h <= 2:
                limits[w] = h
                continue
            r = rng.uniform(*shrink_range)
            L = max(2, int(math.floor(r * h)))
            L = min(L, h - 1) if h > 2 else L
            limits[w] = L

        rng.shuffle(all_ws)

        for w in all_ws:
            inputs = cols[w][:]
            rng.shuffle(inputs)
            h = len(inputs)
            L = limits[w]
            if h <= 2:
                next_cols[w].extend(inputs)
                continue

            F, H = choose_FA_H(h, L)

            # Realize FAs
            for _ in range(F):
                a = inputs.pop()
                b = inputs.pop()
                c = inputs.pop()
                ad = g.add_node(f"FA_s{stage}_w{w}", "FA", weight=w, stage=stage)
                g.add_edge(a, ad); g.add_edge(b, ad); g.add_edge(c, ad)
                s_sig = g.add_node(f"s(w{w},st{stage})", "sig", weight=w, stage=stage, port="sum")
                c_sig = g.add_node(f"c(w{w+1},st{stage})", "sig", weight=w+1, stage=stage, port="carry")
                g.add_edge(ad, s_sig); g.add_edge(ad, c_sig)
                next_cols[w].append(s_sig)
                next_cols[w+1].append(c_sig)

            # Realize HAs
            for _ in range(H):
                a = inputs.pop()
                b = inputs.pop()
                ad = g.add_node(f"HA_s{stage}_w{w}", "HA", weight=w, stage=stage)
                g.add_edge(a, ad); g.add_edge(b, ad)
                s_sig = g.add_node(f"s(w{w},st{stage})", "sig", weight=w, stage=stage, port="sum")
                c_sig = g.add_node(f"c(w{w+1},st{stage})", "sig", weight=w+1, stage=stage, port="carry")
                g.add_edge(ad, s_sig); g.add_edge(ad, c_sig)
                next_cols[w].append(s_sig)
                next_cols[w+1].append(c_sig)

            # Leftovers pass through
            next_cols[w].extend(inputs)

        if all(len(v) <= 2 for v in next_cols.values()):
            cols = next_cols
            break
        cols = next_cols

    # Force finish if necessary
    safety = 0
    while not all(len(v) <= 2 for v in cols.values()) and safety < 16:
        stage += 1
        next_cols: Dict[int, List[int]] = defaultdict(list)
        for w, inputs in sorted(cols.items()):
            inputs = inputs[:]
            # Greedy compression to L=2
            while len(inputs) > 2:
                if len(inputs) >= 3:
                    a = inputs.pop(); b = inputs.pop(); c = inputs.pop()
                    ad = g.add_node(f"FA_s{stage}_w{w}", "FA", weight=w, stage=stage)
                    g.add_edge(a, ad); g.add_edge(b, ad); g.add_edge(c, ad)
                    s_sig = g.add_node(f"s(w{w},st{stage})", "sig", weight=w, stage=stage, port="sum")
                    c_sig = g.add_node(f"c(w{w+1},st{stage})", "sig", weight=w+1, stage=stage, port="carry")
                    g.add_edge(ad, s_sig); g.add_edge(ad, c_sig)
                    next_cols[w].append(s_sig); next_cols[w+1].append(c_sig)
                elif len(inputs) == 2:
                    break
            next_cols[w].extend(inputs[:2])
        cols = next_cols
        safety += 1

    # Attach sinks
    if attach_sinks:
        max_w = max(cols.keys()) if cols else 0
        for w in range(max_w + 1):
            arr = cols.get(w, [])
            if len(arr) == 2:
                a = g.add_node(f"final_row_A_w{w}", "sink", weight=w, stage=stage+1)
                b = g.add_node(f"final_row_B_w{w}", "sink", weight=w, stage=stage+1)
                g.add_edge(arr[0], a); g.add_edge(arr[1], b)
            elif len(arr) == 1:
                a = g.add_node(f"final_row_A_w{w}", "sink", weight=w, stage=stage+1)
                g.add_edge(arr[0], a)

    A = np.array(g.A, dtype=np.uint8)
    return A, g.nodes

# -----------------------------
# Validators (formula-only + simulation)
# -----------------------------


def quick_formula_checks(A: np.ndarray, nodes) -> dict:
    n_nodes = A.shape[0]
    rows, cols = np.nonzero(A)
    succ = {i: [] for i in range(n_nodes)}
    pred = {i: [] for i in range(n_nodes)}
    for u, v in zip(rows, cols):
        succ[u].append(v)
        pred[v].append(u)

    # Tally adders by weight
    FA_w = defaultdict(int)
    HA_w = defaultdict(int)
    for nd in nodes:
        if nd.kind == "FA":
            FA_w[nd.weight] += 1
        elif nd.kind == "HA":
            HA_w[nd.weight] += 1

    # 0) Acyclicity (nilpotency): A^N == 0
    N = A.shape[0]
    An = A.copy().astype(np.uint8)
    for _ in range(1, N):
        An = (An @ A)  # integer matmul
        if not An.any():
            break
    nilpotent = not An.any()

    # 1) Degree constraints + weight consistency on edges
    deg_ok = True
    w_edge_ok = True
    sig_out_to_adder_le1 = True
    sumcarry_produced_once = True
    for nd in nodes:
        if nd.kind == "FA":
            ins = [u for u in pred[nd.idx] if nodes[u].kind == "sig"]
            outs = [v for v in succ[nd.idx] if nodes[v].kind == "sig"]
            deg_ok &= (len(ins) == 3 and len(outs) == 2)
        elif nd.kind == "HA":
            ins = [u for u in pred[nd.idx] if nodes[u].kind == "sig"]
            outs = [v for v in succ[nd.idx] if nodes[v].kind == "sig"]
            deg_ok &= (len(ins) == 2 and len(outs) == 2)
        elif nd.kind == "sig":
            # sum/carry should be produced exactly once (indegree 1), pp have indegree 0
            if nd.port in ("sum", "carry"):
                sumcarry_produced_once &= (len(pred[nd.idx]) == 1 and nodes[pred[nd.idx][0]].kind in ("FA","HA"))
            elif nd.port == "pp":
                sumcarry_produced_once &= (len(pred[nd.idx]) == 0)
            # at most one consumption by adder
            out_to_adders = [v for v in succ[nd.idx] if nodes[v].kind in ("FA","HA")]
            sig_out_to_adder_le1 &= (len(out_to_adders) <= 1)

    # Weight deltas on each edge
    for u, v in zip(rows, cols):
        U, V = nodes[u], nodes[v]
        if U.kind == "sig" and V.kind in ("FA", "HA"):
            w_edge_ok &= (U.weight == V.weight)
        elif U.kind in ("FA", "HA") and V.kind == "sig":
            if V.port == "sum":
                w_edge_ok &= (V.weight == U.weight)
            elif V.port == "carry":
                w_edge_ok &= (V.weight == U.weight + 1)
        elif U.kind == "sig" and V.kind == "sink":
            w_edge_ok &= (U.weight == V.weight)

    # 2) Column-count formulas per weight
    in_count_w = defaultdict(int)      # edges: sig(w)->adder(w)
    out_sum_w = defaultdict(int)       # edges: adder(w)->sum(w)
    out_carry_w = defaultdict(int)     # edges: adder(w-1)->carry(w)

    for u, v in zip(rows, cols):
        U, V = nodes[u], nodes[v]
        if U.kind == "sig" and V.kind in ("FA","HA") and U.weight == V.weight:
            in_count_w[U.weight] += 1
        if U.kind in ("FA","HA") and V.kind == "sig" and V.port == "sum" and U.weight == V.weight:
            out_sum_w[U.weight] += 1
        if U.kind in ("FA","HA") and V.kind == "sig" and V.port == "carry" and V.weight == (U.weight + 1):
            out_carry_w[V.weight] += 1  # carry contributes to column V.weight

    # Check equalities:
    #   in_count(w) == 3*FA_w + 2*HA_w
    #   out_sum(w)  ==   FA_w +   HA_w
    #   out_carry(w)==   FA_{w-1}+HA_{w-1}
    count_ok = True
    details = []
    all_w = set(list(FA_w.keys()) + list(HA_w.keys()) + list(in_count_w.keys()) + list(out_sum_w.keys()) + list(out_carry_w.keys()))
    for w in sorted(all_w):
        lhs_in = in_count_w[w]
        rhs_in = 3*FA_w[w] + 2*HA_w[w]
        lhs_sum = out_sum_w[w]
        rhs_sum = FA_w[w] + HA_w[w]
        lhs_carry = out_carry_w[w]
        rhs_carry = FA_w[w-1] + HA_w[w-1]
        ok_w = (lhs_in == rhs_in) and (lhs_sum == rhs_sum) and (lhs_carry == rhs_carry)
        count_ok &= ok_w
        if not ok_w:
            details.append({
                "w": w,
                "in_edges": lhs_in, "exp_in": rhs_in,
                "sum_edges": lhs_sum, "exp_sum": rhs_sum,
                "carry_into_w": lhs_carry, "exp_carry": rhs_carry,
                "FA_w": FA_w[w], "HA_w": HA_w[w],
                "FA_wm1": FA_w[w-1], "HA_wm1": HA_w[w-1],
            })

    # 3) Final two-row condition: signals that no longer feed adders: ≤2 per weight
    final_sig_by_w = defaultdict(int)
    for nd in nodes:
        if nd.kind == "sig":
            succ_adders = [v for v in succ[nd.idx] if nodes[v].kind in ("FA","HA")]
            if len(succ_adders) == 0:
                final_sig_by_w[nd.weight] += 1
    two_row_ok = all(c <= 2 for c in final_sig_by_w.values())
    
    pass_all = all([nilpotent, deg_ok, w_edge_ok, sig_out_to_adder_le1,
                             sumcarry_produced_once, count_ok, two_row_ok])
    
    assert pass_all, "One or more checks failed; see details in output"

    return {
        "nilpotent_DAG": nilpotent,
        "degree_ok": deg_ok,
        "weight_edges_ok": w_edge_ok,
        "signals_consumed_at_most_once": sig_out_to_adder_le1,
        "sumcarry_produced_once": sumcarry_produced_once,
        "column_count_equalities_ok": count_ok,
        "column_count_violations": details,
        "final_two_rows_ok": two_row_ok,
        "final_bits_per_weight": dict(sorted(final_sig_by_w.items())),
        "summary_pass": pass_all,
    }

def _build_pred_succ(A: np.ndarray):
    rows, cols = np.nonzero(A)
    succ = {i: [] for i in range(A.shape[0])}
    pred = {i: [] for i in range(A.shape[0])}
    for u, v in zip(rows, cols):
        succ[u].append(v)
        pred[v].append(u)
    return pred, succ

# --- Simulation-based functional check ---

import re

def _parse_pp_indices(name: str) -> Tuple[int, int]:
    m = re.search(r"pp\[(\d+),(\d+)\]", name)
    if not m: raise ValueError(f"Bad pp name: {name}")
    return int(m.group(1)), int(m.group(2))

def _infer_n(nodes) -> int:
    ps = [nd for nd in nodes if nd.kind == "sig" and nd.port == "pp"]
    mi = max(_parse_pp_indices(nd.name)[0] for nd in ps)
    mj = max(_parse_pp_indices(nd.name)[1] for nd in ps)
    n_i, n_j = mi+1, mj+1
    if n_i != n_j: raise ValueError("Non-square multiplier inferred.")
    return n_i

def _simulate_once(A: np.ndarray, nodes, a: int, b: int) -> int:
    pred, succ = _build_pred_succ(A)
    node_val: Dict[int, int] = {}
    # init pp
    for nd in nodes:
        if nd.kind == "sig" and nd.port == "pp":
            i, j = _parse_pp_indices(nd.name)
            node_val[nd.idx] = ((a >> i) & 1) & ((b >> j) & 1)
    # staged
    max_stage = max(nd.stage or 0 for nd in nodes)
    for st in range(1, max_stage + 1):
        for nd in nodes:
            if nd.kind in ("FA","HA") and nd.stage == st:
                ins = [u for u in pred[nd.idx] if nodes[u].kind == "sig"]
                vals = [node_val[u] for u in ins]
                ssum = sum(vals)
                if nd.kind == "HA": ssum = vals[0] + vals[1]
                sum_bit = ssum & 1; carry_bit = (ssum >> 1) & 1
                outs = [v for v in succ[nd.idx] if nodes[v].kind == "sig"]
                for v in outs:
                    if nodes[v].port == "sum": node_val[v] = sum_bit
                    elif nodes[v].port == "carry": node_val[v] = carry_bit
    # collect final rows
    n = _infer_n(nodes); W = 2*n+1
    rowA = [0]*W; rowB = [0]*W
    pred_succ = _build_pred_succ(A)[0]
    sinks = [nd for nd in nodes if nd.kind == "sink"]
    if sinks:
        for sk in sinks:
            m = re.search(r"final_row_(A|B)_w(\d+)", sk.name)
            if not m: continue
            row = m.group(1); w = int(m.group(2))
            preds = [u for u in pred_succ[sk.idx] if nodes[u].kind == "sig"]
            if preds:
                bit = node_val.get(preds[0], 0)
                if row == "A": rowA[w] = bit
                else: rowB[w] = bit
    else:
        succ = _build_pred_succ(A)[1]
        for nd in nodes:
            if nd.kind == "sig" and nd.port in ("sum","carry"):
                if not [v for v in succ[nd.idx] if nodes[v].kind in ("FA","HA")]:
                    w = nd.weight or 0
                    if rowA[w] == 0: rowA[w] = node_val.get(nd.idx, 0)
                    elif rowB[w] == 0: rowB[w] = node_val.get(nd.idx, 0)
    # ripple
    carry = 0; got = 0
    for w in range(W):
        s = (rowA[w] if w < len(rowA) else 0) + (rowB[w] if w < len(rowB) else 0) + carry
        bit = s & 1; carry = s >> 1
        got |= (bit << w)
    if carry: got |= (carry << W)
    return got

def verify_multiplier(A: np.ndarray, nodes, exhaustive: bool=False, samples: int=300, seed: int=0) -> dict:
    n = _infer_n(nodes)
    tvs = []
    if exhaustive and (1 << (2*n)) <= 100000:
        for a in range(1<<n):
            for b in range(1<<n):
                tvs.append((a,b))
    else:
        rng = random.Random(seed)
        for _ in range(samples):
            tvs.append((rng.randrange(1<<n), rng.randrange(1<<n)))
    mismatches = []
    for a, b in tvs:
        got = _simulate_once(A, nodes, a, b)
        want = a * b
        if got != want:
            mismatches.append((a, b, want, got))
            if len(mismatches) >= 20: break
    return {"n": n, "tested": len(tvs), "passed": len(mismatches)==0, "mismatches": mismatches}


# -----------------------------
# Helpers: table + plotting
# -----------------------------


def nodes_to_dataframe(nodes: List[Node]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"idx": nd.idx, "name": nd.name, "kind": nd.kind, "weight": nd.weight, "stage": nd.stage, "port": nd.port} for nd in nodes]
    ).sort_values(["stage", "weight", "kind", "idx"], ignore_index=True)


def compute_positions(nodes: List[Node]) -> Dict[int, Tuple[float, float]]:
    """
    Layered layout:
    - y-axis = stage (top: sources stage 0; bottom: sinks final)
    - x-axis = weight, with small horizontal offsets within each (stage, weight, kind) bucket
    """
    # Determine max stage for vertical inversion (stage 0 at top visually)
    max_stage = max(nd.stage or 0 for nd in nodes)
    buckets: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)  # (stage, weight, kind) -> node indices
    for nd in nodes:
        st = nd.stage or 0
        wt = nd.weight if nd.weight is not None else 0
        buckets[(st, wt, nd.kind)].append(nd.idx)
        
    # merge "FA" and "HA" buckets to reduce horizontal spread
    merged_buckets: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
    for (st, wt, kind), lst in buckets.items():
        if kind in ("FA", "HA"):
            merged_buckets[(st, wt, "adder")].extend(lst)
        else:
            merged_buckets[(st, wt, kind)].extend(lst)

    # Assign positions
    pos: Dict[int, Tuple[float, float]] = {}
    for (st, wt, kind), lst in merged_buckets.items():
        lst_sorted = sorted(lst)  # stable
        k = len(lst_sorted)
        # spread within [-0.35, +0.35] horizontally to avoid overlap
        if kind != "sink":
            #offsets = np.linspace(-0.35, 0.35, k) if k > 1 else [0.0]
            offsets = np.linspace(0, 0.6, k) if k > 1 else [0.0]
        else:
            offsets = [0] * k #np.linspace(0, 0.7, k) if k > 1 else [0.0]
        for i, nid in enumerate(lst_sorted):
            x = wt + float(offsets[i])
            y = (max_stage + 1) - st  # invert so stage 0 at top
            # Nudge by kind so adders and signals don't overlap exactly
            #if kind in ("FA", "HA"):
            if kind == "adder":
                #x += 0.12
                y += 0.25
            elif kind == "sink":
                #x += 0.12
                # y -= 0.05
                pass
            pos[nid] = (x, y)
    return pos


def plot_graph(A: np.ndarray, nodes: List[Node], title: str = "Compressor Tree (Wallace)", suffix: str = "") -> None:
    pos = compute_positions(nodes)

    # Separate node sets for styling
    sig_ids = [nd.idx for nd in nodes if nd.kind == "sig"]
    fa_ids = [nd.idx for nd in nodes if nd.kind == "FA"]
    ha_ids = [nd.idx for nd in nodes if nd.kind == "HA"]
    sink_ids = [nd.idx for nd in nodes if nd.kind == "sink"]

    # Plot nodes
    plt.figure(figsize=(10, 7))
    # Signals
    xs = [pos[i][0] for i in sig_ids]
    ys = [pos[i][1] for i in sig_ids]
    plt.scatter(xs, ys, s=30, label="signals")
    # Full adders
    xs = [pos[i][0] for i in fa_ids]
    ys = [pos[i][1] for i in fa_ids]
    plt.scatter(xs, ys, s=70, marker="s", label="FA")
    # Half adders
    xs = [pos[i][0] for i in ha_ids]
    ys = [pos[i][1] for i in ha_ids]
    plt.scatter(xs, ys, s=70, marker="D", label="HA")
    # Sinks
    xs = [pos[i][0] for i in sink_ids]
    ys = [pos[i][1] for i in sink_ids]
    plt.scatter(xs, ys, s=35, marker="v", label="sinks")

    # Plot edges
    n = A.shape[0]
    rows, cols = np.nonzero(A)
    for u, v in zip(rows, cols):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        plt.plot([x0, x1], [y0, y1], linewidth=0.6, alpha=0.6)

    # Cosmetic axes
    # x-axis: weights; y-axis: stages
    all_weights = sorted({nd.weight for nd in nodes if nd.weight is not None})
    all_stages = sorted({nd.stage for nd in nodes if nd.stage is not None})
    if all_weights:
        plt.xticks(all_weights)
    if all_stages:
        # show original stage numbers near ticks (remember we inverted when plotting)
        max_stage = max(all_stages)
        yticks = [(max_stage + 1) - st for st in all_stages]
        ylabels = [str(st) for st in all_stages]
        plt.yticks(yticks, ylabels)
        plt.ylabel("stage")
    plt.xlabel("weight (i+j)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'compressor_tree_graph{suffix}.png')
    
def get_node_kind_counts(nodes: List[Node], split=True) -> Dict[str, int]:
    counts = defaultdict(int)
    for nd in nodes:
        kind = nd.kind
        if split:
            if nd.kind == "signal" and nd.port == "pp":
                kind = "pp"
        counts[kind] += 1
    return dict(counts)


def plot_adjacency_spy(A: np.ndarray, title: str = "Adjacency Matrix (sparsity)", suffix: str = "") -> None:
    plt.figure(figsize=(6, 6))
    plt.spy(A, markersize=2)
    plt.title(title)
    plt.xlabel("to (node index)")
    plt.ylabel("from (node index)")
    plt.tight_layout()
    plt.savefig(f'compressor_tree_adjacency{suffix}.png')

# -----------------------------
# Contracted graph without signal nodes but labeled edges
# -----------------------------


def contract_signals_to_labeled_edges(G: Graph) -> Graph:
    """
    Contract a compressor-tree Graph by removing intermediate signal nodes and
    encoding edge types in the adjacency matrix:
        - sum edges   : 1
        - carry edges : 2
        - source edges: 3  (from input partial-products)

    Rules:
      - Keep all non-signal nodes ('FA', 'HA', 'sink').
      - Keep input signal nodes with port='pp', but rename their kind to 'sources'.
      - Remove all other 'sig' nodes ('sum', 'carry').
      - Connect producers directly to consumers, labeling edges as above.
        * For adder -> (adder|sink): follow through 'sum'/'carry' signals and label 1/2.
        * For sources(pp) -> (adder|sink): label 3.

    Returns a NEW Graph with reduced node set and an adjacency matrix whose entries
    are in {0,1,2,3}.
    """
    # Build predecessor/successor lists for the original graph
    N = len(G.nodes)
    succ = [[] for _ in range(N)]
    pred = [[] for _ in range(N)]
    for u in range(N):
        row = G.A[u]
        for v, val in enumerate(row):
            if val:
                succ[u].append(v)
                pred[v].append(u)

    # Decide which nodes to keep:
    #  - keep all non-signal nodes
    #  - keep 'sig' nodes only if they are partial-products (port == 'pp') -> rename kind='sources'
    keep_old_idxs = []
    for nd in G.nodes:
        if nd.kind != "sig":
            keep_old_idxs.append(nd.idx)
        elif nd.port == "pp":
            keep_old_idxs.append(nd.idx)
        # else drop 'sum'/'carry' signal nodes

    # Map old indices -> new indices
    old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_old_idxs)}

    # Create new graph and add kept nodes (renaming pp->sources)
    G2 = Graph()
    for old_idx in keep_old_idxs:
        nd = G.nodes[old_idx]
        G2.add_node(name=nd.name, kind=nd.kind, weight=nd.weight, stage=nd.stage, port=nd.port)

    # Helper to set labeled edge with conflict checking
    def set_edge(u_new: int, v_new: int, label: int):
        cur = G2.A[u_new][v_new]
        if cur == 0 or cur == label:
            G2.A[u_new][v_new] = label
        else:
            # This should not happen in a legal compressor tree (no multi-edges with different labels).
            raise ValueError(f"Edge label conflict on ({u_new}->{v_new}): have {cur}, tried {label}")

    # 1) Wire sources (pp) directly to their adder/sink consumers with label 3
    for old_idx in keep_old_idxs:
        nd = G.nodes[old_idx]
        if nd.kind == "sig" and nd.port == "pp":
            u_new = old2new[old_idx]
            for v in succ[old_idx]:
                kind_v = G.nodes[v].kind
                if kind_v in ("FA", "HA", "sink"):
                    if v in old2new:
                        set_edge(u_new, old2new[v], 3)
                # If pp -> sig (rare), skip; only connect to kept nodes

    # 2) Bypass 'sum'/'carry' signal nodes: adder -> (adder|sink) with labels 1/2
    for old_idx in range(N):
        nd = G.nodes[old_idx]
        if nd.kind in ("FA", "HA"):
            if old_idx not in old2new:
                continue
            u_new = old2new[old_idx]
            # Find outgoing signal nodes from this adder (sum/carry)
            out_sig_ids = [v for v in succ[old_idx] if G.nodes[v].kind == "sig"]
            # For each signal, forward its edges to final consumers with appropriate label
            for s in out_sig_ids:
                port = G.nodes[s].port  # 'sum' or 'carry'
                label = 1 if port == "sum" else 2 if port == "carry" else None
                if label is None:
                    continue
                for t in succ[s]:
                    kind_t = G.nodes[t].kind
                    if kind_t in ("FA", "HA", "sink") and t in old2new:
                        set_edge(u_new, old2new[t], label)

    G2.A = np.array(G2.A, dtype=np.uint8)
    return G2


def demo():
    # -----------------------------
    # Usage demo
    # -----------------------------

    N = 8  # change to try other bitwidths; for plotting, keep <= 6

    A, nodes = build_wallace_compressor_graph(N)
    df_nodes = nodes_to_dataframe(nodes)

    # Basic stats
    stats = {
        "n_bits": N,
        "num_nodes": len(nodes),
        "num_edges": int(A.sum()),
        "adj_shape": A.shape,
        "num_sigs": int(sum(1 for nd in nodes if nd.kind == "sig")),
        "num_FA": int(sum(1 for nd in nodes if nd.kind == "FA")),
        "num_HA": int(sum(1 for nd in nodes if nd.kind == "HA")),
        "num_sinks": int(sum(1 for nd in nodes if nd.kind == "sink")),
    }
    stats

    # Plots
    plot_graph(A, nodes, title=f"Compressor Tree (Wallace), n={N}")
    G_c = contract_signals_to_labeled_edges(Graph(nodes, A))
    plot_graph(G_c.A, G_c.nodes, title=f"Contracted Graph (Wallace) (n={N})", suffix="_contracted")
    plot_adjacency_spy(A, title=f"Adjacency Matrix (Wallace) (n={N})")
    plot_adjacency_spy(G_c.A, title=f"Contracted Adjacency Matrix (Wallace) (n={N})", suffix="_contracted")

    # Save CSVs for download
    nodes_csv_path = f"wallace_n{N}_nodes.csv"
    adj_csv_path = f"wallace_n{N}_adjacency.csv"
    df_nodes.to_csv(nodes_csv_path, index=False)
    # store adjacency as dense CSV for convenience (small N recommended)
    pd.DataFrame(A, columns=[f"n{j}" for j in range(A.shape[1])]).to_csv(adj_csv_path, index=False)

    nodes_csv_path, adj_csv_path

    print("Quick formula checks:")
    checks = quick_formula_checks(A, nodes)
    print(checks)

    # Run the demo now that functions are defined

    A_rand, nodes_rand = random_compressor_tree(n=N, seed=42, shrink_range=(0.6, 0.95), p_fa=0.65)

    stats = {
        "nodes": len(nodes_rand),
        "edges": int(A_rand.sum()),
        "signals": sum(1 for nd in nodes_rand if nd.kind == "sig"),
        "FAs": sum(1 for nd in nodes_rand if nd.kind == "FA"),
        "HAs": sum(1 for nd in nodes_rand if nd.kind == "HA"),
        "sinks": sum(1 for nd in nodes_rand if nd.kind == "sink"),
    }

    form = quick_formula_checks(A_rand, nodes_rand)
    ver = verify_multiplier(A_rand, nodes_rand, exhaustive=False, samples=500)

    print(f"Random compressor tree stats (n={N}): {stats}")
    print(f"Formula checks: {form}")
    print(f"Functional verification: {ver}")
    plot_graph(A_rand, nodes_rand, title=f"Compressor Tree (Random), n={N}", suffix="_random")
    plot_adjacency_spy(A_rand, title=f"Adjacency Matrix (Random, n={N})", suffix="_random")

if __name__ == "__main__":
    demo()