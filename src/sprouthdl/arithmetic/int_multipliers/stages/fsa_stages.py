from __future__ import annotations

from typing import Callable, ClassVar, Dict, List, Set, Tuple

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import FinalStageAdderBase
from sprouthdl.arithmetic.prefix_adders.prefix_adder import P_brent_kung, P_kogge_stone, P_ripple_carry, P_sklansky, Pair, analyze_prefix_matrix, legalize_P
from sprouthdl.sprouthdl import Bool, Const, Expr


def _exists(nodes: Set[Pair], i: int, j: int) -> bool:
    return (i == j) or ((i, j) in nodes)


def _find_split(nodes: Set[Pair], i: int, j: int) -> int | None:
    for k in range(i - 1, j - 1, -1):
        if _exists(nodes, i, k + 1) and _exists(nodes, k, j):
            return k
    return None


class PrefixAdderFinalStage(FinalStageAdderBase):
    """Final stage adder that realises a chosen prefix network."""

    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(
        P_kogge_stone
    )
    depth_optimize: ClassVar[bool] = True

    def resolve(self, columns: Dict[int, List[Expr]]) -> List[Expr]:
        width = self.config.out_width
        max_col = max(columns.keys(), default=width - 1)
        working_width = max(width, max_col + 1)

        zero = Const(False, Bool())
        row_a: List[Expr] = []
        row_b: List[Expr] = []

        for idx in range(working_width):
            bits = list(columns.get(idx, []))
            if len(bits) > 2:
                raise ValueError(
                    f"Column {idx} has {len(bits)} bits; expected at most 2 before prefix addition"
                )
            if len(bits) == 0:
                row_a.append(zero)
                row_b.append(zero)
            elif len(bits) == 1:
                row_a.append(bits[0])
                row_b.append(zero)
            else:  # len == 2
                row_a.append(bits[0])
                row_b.append(bits[1])

        propagates = [a ^ b for a, b in zip(row_a, row_b)]
        generates = [a & b for a, b in zip(row_a, row_b)]

        raw_nodes = set(self.prefix_matrix_builder(working_width))
        nodes = legalize_P(working_width, raw_nodes)

        best_k: Dict[Pair, int] = {}
        if self.depth_optimize and nodes:
            _, best_k, _, _ = analyze_prefix_matrix(working_width, nodes)

        gp_cache: Dict[Pair, Tuple[Expr, Expr]] = {}

        def gp(i: int, j: int) -> Tuple[Expr, Expr]:
            key = (i, j)
            if key in gp_cache:
                return gp_cache[key]
            if i == j:
                result = (generates[i], propagates[i])
                gp_cache[key] = result
                return result
            if key not in nodes:
                raise ValueError(
                    f"Prefix matrix missing node {(i, j)} required for carry propagation"
                )
            if self.depth_optimize:
                k = best_k[key]
            else:
                k = _find_split(nodes, i, j)
                if k is None:
                    raise ValueError(f"No legal split for prefix node {(i, j)}")
            g_left, p_left = gp(i, k + 1)
            g_right, p_right = gp(k, j)
            result = (g_left | (p_left & g_right), p_left & p_right)
            gp_cache[key] = result
            return result

        carries: List[Expr] = [zero] * (working_width + 1)
        for idx in range(working_width):
            g_prefix, p_prefix = gp(idx, 0)
            carries[idx + 1] = g_prefix | (p_prefix & carries[0])

        sums = [propagates[idx] ^ carries[idx] for idx in range(working_width)]
        result_bits: List[Expr] = list(sums)
        result_bits.append(carries[working_width])
        return result_bits


class BrentKungPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(
        P_brent_kung
    )


class SklanskyPrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(
        P_sklansky
    )


class RipplePrefixFinalStage(PrefixAdderFinalStage):
    prefix_matrix_builder: ClassVar[Callable[[int], Set[Pair]]] = staticmethod(
        P_ripple_carry
    )
    depth_optimize: ClassVar[bool] = False