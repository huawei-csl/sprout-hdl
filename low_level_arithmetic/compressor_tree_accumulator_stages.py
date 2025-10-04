from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import TYPE_CHECKING, DefaultDict, Dict, List, Tuple

from low_level_arithmetic.multiplier_stage_core import (
    PartialProductAccumulatorBase,
    half_adder,
    full_adder_fast,
    full_adder_low_area,
)
from sprouthdl.sprouthdl import Expr


if TYPE_CHECKING:
    from low_level_arithmetic.multiplier_stage_core import StageBasedMultiplier


class WallaceTreeAccumulator(PartialProductAccumulatorBase):
    """Classic Wallace tree reduction of partial-product columns."""

    def __init__(self, core: "StageBasedMultiplier") -> None:
        super().__init__(core)
        self._full_adder = (
            full_adder_low_area
            if self.core.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            progress = False

            for weight in sorted(cols.keys()):
                bits = list(cols[weight])

                while len(bits) >= 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True

                if len(bits) == 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 1:
                    next_cols[weight].append(bits.pop())

            if not progress:
                return cols

            cols = next_cols


class DaddaTreeAccumulator(PartialProductAccumulatorBase):
    """Dadda tree reduction using progressively tighter column height thresholds."""

    def __init__(self, core: "StageBasedMultiplier") -> None:
        super().__init__(core)
        self._full_adder = (
            full_adder_low_area
            if self.core.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        max_height = max((len(bits) for bits in cols.values()), default=0)
        if max_height <= 2:
            return cols

        thresholds = self._build_thresholds(max_height)
        stage_limits = list(reversed(thresholds))[1:]  # skip the largest value

        for target in stage_limits:
            for weight in sorted(cols.keys()):
                reduced, carries = self._reduce_column_to_target(cols[weight], target)
                cols[weight] = reduced
                cols[weight + 1].extend(carries)

        return cols

    @staticmethod
    def _build_thresholds(max_height: int) -> List[int]:
        thresholds = [2]
        while thresholds[-1] < max_height:
            next_val = floor(3 * thresholds[-1] / 2)
            if next_val == thresholds[-1]:
                next_val += 1
            thresholds.append(next_val)
        return thresholds

    def _reduce_column_to_target(
        self, bits: List[Expr], target: int
    ) -> Tuple[List[Expr], List[Expr]]:
        working = list(bits)
        carries: List[Expr] = []

        while len(working) > target:
            if len(working) >= 3 and len(working) - 2 >= target:
                x = working.pop()
                y = working.pop()
                z = working.pop()
                s, c = self._full_adder(x, y, z)
                working.append(s)
                carries.append(c)
            else:
                x = working.pop()
                y = working.pop()
                s, c = half_adder(x, y)
                working.append(s)
                carries.append(c)

        return working, carries
