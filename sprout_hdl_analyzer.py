# -----------------------------
# Graph analysis / metrics
# -----------------------------
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Iterable

#from sprout_hdl_module import Module
from sprout_hdl import Concat, Const, Expr, Op1, Op2, Resize, Signal, Slice, Ternary, _clsname
#from sprout_hdl_module import Module

@dataclass
class GraphReport:
    # Counts
    total_expr_nodes: int          # unique Expr nodes counted per options
    op_nodes: int                  # Op1 + Op2 + Ternary only (no wiring/consts/signals)
    by_type: Dict[str, int]        # class-name -> count
    # Depths
    output_depth: Dict[str, int]   # depth of each output driver cone
    reg_next_depth: Dict[str, int] # depth to each reg next-state (sequential inputs)
    max_depth: int                 # max over outputs and (optionally) reg.next cones

class _Analyzer:
    """Internal: walks expressions, counts nodes, computes depth with memoization."""
    def __init__(self, include_wiring: bool, include_consts: bool, include_reg_cones: bool):
        self.include_wiring = include_wiring
        self.include_consts = include_consts
        self.include_reg_cones = include_reg_cones
        # visited set for counting (ids, to avoid Expr.__eq__)
        self._seen_for_count: Set[int] = set()
        self._type_counts: Dict[str, int] = {}
        # memo for depth by expr id
        self._depth_memo: Dict[int, int] = {}

    # ---- node classification for counting ----
    def _is_wiring(self, e: Expr) -> bool:
        return isinstance(e, (Concat, Slice, Resize))

    def _is_const(self, e: Expr) -> bool:
        return isinstance(e, Const)

    def _is_signal(self, e: Expr) -> bool:
        return isinstance(e, Signal)

    def _is_logic_op(self, e: Expr) -> bool:
        return isinstance(e, (Op1, Op2, Ternary))

    def _should_count(self, e: Expr) -> bool:
        if self._is_signal(e):
            return False
        if self._is_const(e):
            return self.include_consts
        if self._is_wiring(e):
            return self.include_wiring
        return True  # Op1/Op2/Ternary

    # ---- structural children (transparent through wiring) ----
    def _children(self, e: Expr) -> Iterable[Expr]:
        """Return structural children of e. For Signals, dive into driver if it's a comb signal."""
        if isinstance(e, Const):
            return ()
        if isinstance(e, Signal):
            # Do NOT cross registers (seq boundary). Inputs with no driver are leaves.
            if e.kind in ("input", "reg"):
                return ()
            # 'wire' or 'output' with a driver: traverse into the cone
            return (e._driver,) if e._driver is not None else ()
        if isinstance(e, Op1):
            return (e.a,)
        if isinstance(e, Op2):
            return (e.a, e.b)
        if isinstance(e, Ternary):
            return (e.sel, e.a, e.b)
        if isinstance(e, Concat):
            return tuple(e.parts)
        if isinstance(e, Slice):
            return (e.a,)
        if isinstance(e, Resize):
            return (e.a,)
        # Fallback: no children
        return ()

    # ---- counting (unique nodes) ----
    def _count_walk(self, e: Expr):
        st = [e]
        while st:
            cur = st.pop()
            if cur is None:
                continue
            cid = id(cur)
            if cid in self._seen_for_count:
                continue
            self._seen_for_count.add(cid)
            if self._should_count(cur):
                cls = _clsname(cur)
                self._type_counts[cls] = self._type_counts.get(cls, 0) + 1
            # Always traverse children to reach ops behind wiring/signals
            for ch in self._children(cur):
                st.append(ch)

    # ---- depth (logic levels) ----
    def _depth(self, e: Expr, visiting: Set[int]) -> int:
        """Combinational depth. Logic ops add 1; wiring (Concat/Slice/Resize) add 0; Signals may inline driver; Const/inputs/regs are 0."""
        if e is None:
            return 0
        eid = id(e)
        if eid in self._depth_memo:
            return self._depth_memo[eid]
        if eid in visiting:
            raise RuntimeError("Combinational cycle detected while computing depth.")
        visiting.add(eid)

        if isinstance(e, Const):
            d = 0
        elif isinstance(e, Signal):
            if e.kind in ("input", "reg") or e._driver is None:
                d = 0
            else:
                d = self._depth(e._driver, visiting)
        elif isinstance(e, (Concat, Slice, Resize)):
            # transparent wiring
            d = max((self._depth(ch, visiting) for ch in self._children(e)), default=0)
        elif isinstance(e, (Op1, Op2, Ternary)):
            d = 1 + max((self._depth(ch, visiting) for ch in self._children(e)), default=0)
        else:
            # unknown node → treat as logic with +1
            d = 1 + max((self._depth(ch, visiting) for ch in self._children(e)), default=0)

        visiting.remove(eid)
        self._depth_memo[eid] = d
        return d

    # ---- public entry ----
    def run(self, m: "Module") -> GraphReport:
        # Roots: all outputs with drivers; optionally reg.next cones
        out_depth: Dict[str, int] = {}
        for s in m._ports_of("output"):
            if s._driver is not None:
                self._count_walk(s._driver)
                out_depth[s.name] = self._depth(s._driver, set())
            else:
                out_depth[s.name] = 0

        reg_depth: Dict[str, int] = {}
        if self.include_reg_cones:
            for r in m._internals_of("reg"):
                if r._next is not None:
                    self._count_walk(r._next)
                    reg_depth[r.name] = self._depth(r._next, set())
                else:
                    reg_depth[r.name] = 0

        # Summaries
        op_nodes = sum(self._type_counts.get(t, 0) for t in ("Op1", "Op2", "Ternary"))
        total_expr_nodes = sum(self._type_counts.values())
        max_depth = 0
        if out_depth:
            max_depth = max(max_depth, max(out_depth.values()))
        if self.include_reg_cones and reg_depth:
            max_depth = max(max_depth, max(reg_depth.values()))

        return GraphReport(
            total_expr_nodes=total_expr_nodes,
            op_nodes=op_nodes,
            by_type=dict(self._type_counts),
            output_depth=out_depth,
            reg_next_depth=reg_depth,
            max_depth=max_depth,
        )