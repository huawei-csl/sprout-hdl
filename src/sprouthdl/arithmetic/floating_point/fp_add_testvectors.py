"""Test vector generation for the FP adder (FpAdd).

Provides random and exhaustive vector generators following the same
tuple-based format as the integer arithmetic test vectors:
    [(name, {"a": bits, "b": bits}, {"y": bits}), ...]
"""
from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np

from sprouthdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode

TestVectors = List[Tuple[str, Dict[str, int], Dict[str, int]]]


class FpAddTestVectors:
    """Random test vector generator for the FP adder."""

    def __init__(
        self,
        EW: int,
        FW: int,
        num_vectors: int = 10_000,
        subnormals: bool = True,
        seed: int = 42,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.num_vectors = num_vectors
        self.subnormals = subnormals
        self.seed = seed

    def generate(self) -> TestVectors:
        rng = np.random.default_rng(self.seed)
        vectors: TestVectors = []
        for i in range(self.num_vectors):
            a = int(rng.integers(0, 1 << self.W))
            b = int(rng.integers(0, 1 << self.W))
            y = fp_encode(
                fp_decode(a, self.EW, self.FW) + fp_decode(b, self.EW, self.FW),
                self.EW, self.FW, subnormals=self.subnormals,
            )
            vectors.append((f"v{i}", {"a": a, "b": b}, {"y": y}))
        return vectors


class FpAddTestVectorsExhaustive:
    """Exhaustive test vector generator for the FP adder (small formats)."""

    def __init__(
        self,
        EW: int,
        FW: int,
        subnormals: bool = True,
    ) -> None:
        self.EW = EW
        self.FW = FW
        self.W = 1 + EW + FW
        self.subnormals = subnormals

    def generate(self) -> TestVectors:
        vectors: TestVectors = []
        for a in range(1 << self.W):
            for b in range(1 << self.W):
                y = fp_encode(
                    fp_decode(a, self.EW, self.FW) + fp_decode(b, self.EW, self.FW),
                    self.EW, self.FW, subnormals=self.subnormals,
                )
                vectors.append((f"v{len(vectors)}", {"a": a, "b": b}, {"y": y}))
        return vectors
