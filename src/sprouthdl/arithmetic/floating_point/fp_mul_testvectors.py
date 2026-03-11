"""Test vector generation for the FP multiplier (FpMulSN).

Provides random and exhaustive vector generators following the same
tuple-based format as the integer arithmetic test vectors:
    [(name, {"a": bits, "b": bits}, {"y": bits}), ...]
"""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from sprouthdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode

TestVectors = List[Tuple[str, Dict[str, int], Dict[str, int]]]


def _is_subnormal(bits: int, EW: int, FW: int) -> bool:
    e = (bits >> FW) & ((1 << EW) - 1)
    f = bits & ((1 << FW) - 1)
    return e == 0 and f != 0


def _should_skip(
    a: int, b: int, product_val: float, y_enc: int,
    EW: int, FW: int, subnormals: bool, always_subnormal_rounding: bool,
) -> bool:
    """Return True if this vector should be skipped due to FTZ limitations."""
    if not subnormals:
        if _is_subnormal(a, EW, FW) or _is_subnormal(b, EW, FW):
            return True
        if not always_subnormal_rounding:
            min_normal_val = 2.0 ** (2 - (1 << (EW - 1)))
            if 0 < abs(product_val) < min_normal_val and y_enc != 0:
                return True
    return False


class FpMulTestVectors:
    """Random test vector generator for the FP multiplier."""

    def __init__(
        self,
        EW: int,
        FW: int,
        num_vectors: int = 10_000,
        subnormals: bool = True,
        always_subnormal_rounding: bool = False,
        seed: int = 42,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.num_vectors = num_vectors
        self.subnormals = subnormals
        self.always_subnormal_rounding = always_subnormal_rounding
        self.seed = seed

    def generate(self) -> TestVectors:
        rng = np.random.default_rng(self.seed)
        vectors: TestVectors = []
        while len(vectors) < self.num_vectors:
            a = int(rng.integers(0, 1 << self.W))
            b = int(rng.integers(0, 1 << self.W))
            product_val = fp_decode(a, self.EW, self.FW) * fp_decode(b, self.EW, self.FW)
            y = fp_encode(product_val, self.EW, self.FW, subnormals=self.subnormals)
            if _should_skip(a, b, product_val, y,
                            self.EW, self.FW, self.subnormals, self.always_subnormal_rounding):
                continue
            vectors.append((f"v{len(vectors)}", {"a": a, "b": b}, {"y": y}))
        return vectors


class FpMulTestVectorsExhaustive:
    """Exhaustive test vector generator for the FP multiplier (small formats)."""

    def __init__(
        self,
        EW: int,
        FW: int,
        subnormals: bool = True,
        always_subnormal_rounding: bool = False,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.subnormals = subnormals
        self.always_subnormal_rounding = always_subnormal_rounding

    def generate(self) -> TestVectors:
        vectors: TestVectors = []
        for a in range(1 << self.W):
            for b in range(1 << self.W):
                product_val = fp_decode(a, self.EW, self.FW) * fp_decode(b, self.EW, self.FW)
                y = fp_encode(product_val, self.EW, self.FW, subnormals=self.subnormals)
                if _should_skip(a, b, product_val, y,
                                self.EW, self.FW, self.subnormals, self.always_subnormal_rounding):
                    continue
                vectors.append((f"v{len(vectors)}", {"a": a, "b": b}, {"y": y}))
        return vectors
