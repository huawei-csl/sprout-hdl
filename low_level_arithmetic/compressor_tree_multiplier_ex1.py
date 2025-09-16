from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal
import numpy as np

StagePolicy = Literal["wallace", "dadda"]


@dataclass
class Node:
    id: int
    kind: str  # "PP", "FA", "HA", "OUT"
    label: str
    col: int
    stage: int  # -1 for sources, big number for OUT


class CompTreeGraph:
    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: List[Tuple[int, int]] = []  # (producer -> consumer)

    def add_node(self, kind: str, label: str, col: int, stage: int) -> int:
        nid = len(self.nodes)
        self.nodes.append(Node(nid, kind, label, col, stage))
        return nid

    def add_edge(self, u: int, v: int):
        if u == v:
            raise ValueError("Self-loop not allowed in this DAG.")
        self.edges.append((u, v))


# Replace the Dadda part of build_compressor_tree with this in-place stage processing
def _dadda_thresholds(max_height: int) -> List[int]:
    if max_height <= 2:
        return [2]
    ts = [2]
    while ts[-1] < max_height:
        next_t = int(np.floor(1.5 * ts[-1]))
        if next_t == ts[-1]:
            next_t += 1
        ts.append(next_t)
    return list(reversed(ts))


def build_compressor_tree(n: int, policy: StagePolicy = "dadda") -> CompTreeGraph:
    """
    Build a Wallace/Dadda compressor-tree DAG for an n-bit unsigned multiplier.
    Nodes:
      - "PP": partial product sources (stage = -1)
      - "FA": full adder (3:2) nodes
      - "HA": half adder (2:2) nodes (used to fit Dadda thresholds)
      - "OUT": final row collectors (connected to remaining bits)
    Returns CompTreeGraph.
    """
    g = CompTreeGraph()

    # Initial columns of available bits: tokens are *producer node ids*
    cols: Dict[int, List[int]] = defaultdict(list)

    # Create PP sources
    for i in range(n):
        for j in range(n):
            w = i + j
            nid = g.add_node("PP", f"PP({i},{j})", col=w, stage=-1)
            cols[w].append(nid)

    max_col = 2 * n - 2  # highest PP column index
    # initial maximum stack height across columns
    max_height = max((len(cols[w]) for w in range(max_col + 1)), default=0)

    if policy == "wallace":
        thresholds = [2]  # reduce toward 2 repeatedly until done
    
    if policy == "dadda":
        thresholds = _dadda_thresholds(max_height)
        stage = 0
        # Process thresholds from large -> small; do reductions in-place so carries are immediately handled
        for T in thresholds:
            w = 0
            # iterate across columns; columns can grow as we push carries into w+1
            while w <= max(cols.keys(), default=0):
                stack = cols[w]  # this is the *live* list for that column
                # reduce until this column has <= T items
                while len(stack) > T:
                    if len(stack) >= 3:
                        a = stack.pop(); b = stack.pop(); c = stack.pop()
                        fa = g.add_node("FA", f"FA(s{stage},w{w})", col=w, stage=stage)
                        g.add_edge(a, fa); g.add_edge(b, fa); g.add_edge(c, fa)
                        # FA produces sum (same column) and carry (next column)
                        stack.append(fa)          # sum in same column (may be reduced further)
                        cols[w + 1].append(fa)   # carry to next column (will be processed when its turn comes)
                    elif len(stack) >= 2:
                        a = stack.pop(); b = stack.pop()
                        ha = g.add_node("HA", f"HA(s{stage},w{w})", col=w, stage=stage)
                        g.add_edge(a, ha); g.add_edge(b, ha)
                        stack.append(ha)
                        cols[w + 1].append(ha)
                    else:
                        break
                w += 1
            stage += 1

    stage = 0
    # For Wallace, we will loop until all columns have <= 2 bits
    while True:
        # Determine columns to iterate over (current columns present)
        current_max_w = max(cols.keys(), default=0)
        next_cols: Dict[int, List[int]] = defaultdict(list)

        # For each column, reduce to <= T using FAs (3:2) then HAs (2:2)
        # Note: We iterate the thresholds list; for Wallace thresholds=[2] but will loop until stable
        for w in range(current_max_w + 1):
            stack = cols.get(w, []).copy()  # producer ids at this column
            h = len(stack)
            T = thresholds[0] if policy == "wallace" else thresholds[min(stage, len(thresholds) - 1)]
            # Use FAs (3 -> 2)
            while h > T and h >= 3:
                a = stack.pop()
                b = stack.pop()
                c = stack.pop()
                fa = g.add_node("FA", f"FA(s{stage},w{w})", col=w, stage=stage)
                g.add_edge(a, fa)
                g.add_edge(b, fa)
                g.add_edge(c, fa)
                # FA node acts as producer in next stage: sum in same column, carry in w+1
                next_cols[w].append(fa)
                next_cols[w + 1].append(fa)
                h -= 2
            # Use HAs (2 -> 2 but frees one height)
            while h > T and h >= 2:
                a = stack.pop()
                b = stack.pop()
                ha = g.add_node("HA", f"HA(s{stage},w{w})", col=w, stage=stage)
                g.add_edge(a, ha)
                g.add_edge(b, ha)
                next_cols[w].append(ha)
                next_cols[w + 1].append(ha)
                h -= 1
            # Forward leftovers (<= T) unchanged to next_cols
            for prod in stack:
                next_cols[w].append(prod)

        cols = next_cols
        stage += 1

        # Wallace termination condition: repeat until all columns have <= 2
        if policy == "wallace":
            all_le_2 = all(len(cols[w]) <= 2 for w in cols.keys())
            if all_le_2:
                break
            # otherwise continue with same threshold (2)
            continue

        # Dadda policy: we finish after applying all thresholds (we stepped through stages
        # using threshold chosen above; continue until the final threshold stage)
        # Stop when we've reached final threshold and all columns <= final threshold.
        final_T = thresholds[0]
        if stage >= len(thresholds) and all(len(cols[w]) <= final_T for w in cols.keys()):
            break
        # else continue

    # Terminate: connect remaining bits to OUT nodes (rows)
    out_stage = stage
    for w in sorted(cols.keys()):
        bits = cols.get(w, [])
        for r, prod in enumerate(bits):
            out = g.add_node("OUT", f"ROW{r}_W{w}", col=w, stage=out_stage)
            g.add_edge(prod, out)

    return g


def adjacency_matrix(g: CompTreeGraph):
    N = len(g.nodes)
    A = np.zeros((N, N), dtype=np.uint8)
    for u, v in g.edges:
        A[u, v] = 1
    meta = [{"id": nd.id, "kind": nd.kind, "label": nd.label, "col": nd.col, "stage": nd.stage} for nd in g.nodes]
    return A, meta


# --- Example usage ---
if __name__ == "__main__":
    # build a 4x4 multiplier compressor tree (Dadda style)
    g = build_compressor_tree(4, policy="dadda")
    A, meta = adjacency_matrix(g)

    print(f"Nodes: {len(g.nodes)}")
    print(f"Edges: {len(g.edges)}")
    print("Adjacency matrix shape:", A.shape)

    # show first 20 node metadata entries
    for m in meta:#:20]:
        print(m)

    # optionally visualize adjacency (requires networkx + matplotlib if you want to draw it)
    import networkx as nx
    import matplotlib.pyplot as plt
    G_nx = nx.DiGraph()
    for nd in meta:
        G_nx.add_node(nd["id"], label=nd["label"], kind=nd["kind"], col=nd["col"], stage=nd["stage"])
    G_nx.add_edges_from(g.edges)
    pos = nx.spring_layout(G_nx)
    labels = {nd["id"]: nd["label"] for nd in meta}
    nx.draw(G_nx, pos, with_labels=True, labels=labels, node_size=300)
    plt.savefig("compressor_tree.png")
