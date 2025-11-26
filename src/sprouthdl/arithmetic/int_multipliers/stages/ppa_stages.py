from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import DefaultDict, Dict, List, Tuple

from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import MultiplierConfig, PartialProductAccumulatorBase, half_adder, full_adder_fast, full_adder_low_area
from sprouthdl.sprouthdl import Bool, Const, Expr


class WallaceTreeAccumulator(PartialProductAccumulatorBase):
    """Classic Wallace tree reduction of partial-product columns."""

    def __init__(self, config: MultiplierConfig) -> None:
        super().__init__(config)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
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

    def __init__(self, config: MultiplierConfig) -> None:
        super().__init__(config)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
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


class CarrySaveAccumulator(PartialProductAccumulatorBase):
    """Iterative carry-save reduction using only full adders."""

    def __init__(self, config: MultiplierConfig) -> None:
        super().__init__(config)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
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
                if bits:
                    next_cols[weight].extend(bits)

            if not progress:
                return cols

            cols = next_cols


class FourTwoCompressorAccumulator(PartialProductAccumulatorBase):
    """Reduction based on 4:2 compressors backed by chained full adders."""

    def __init__(self, config: MultiplierConfig) -> None:
        super().__init__(config)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
            else full_adder_fast
        )
        self._zero = Const(False, Bool())

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            progress = False

            for weight in sorted(cols.keys()):
                bits = list(cols[weight])

                while len(bits) >= 4:
                    a = bits.pop()
                    b = bits.pop()
                    c = bits.pop()
                    d = bits.pop()
                    sum_bit, carry_low, carry_high = self._compress_4_2(a, b, c, d)
                    next_cols[weight].append(sum_bit)
                    next_cols[weight + 1].extend((carry_low, carry_high))
                    progress = True

                if len(bits) == 3:
                    x, y, z = bits.pop(), bits.pop(), bits.pop()
                    s, c = self._full_adder(x, y, z)
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 2:
                    s, c = half_adder(bits.pop(), bits.pop())
                    next_cols[weight].append(s)
                    next_cols[weight + 1].append(c)
                    progress = True
                elif len(bits) == 1:
                    next_cols[weight].append(bits.pop())

            if not progress:
                return cols

            cols = next_cols

    def _compress_4_2(self, a: Expr, b: Expr, c: Expr, d: Expr) -> Tuple[Expr, Expr, Expr]:
        s1, c1 = self._full_adder(a, b, c)
        s2, c2 = self._full_adder(s1, d, self._zero)
        return s2, c1, c2