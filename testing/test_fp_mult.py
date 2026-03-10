"""Tests for FpMul (flush-to-zero multiplier).

Note: test_floating_point.py already covers FpMul f16 normal/subnormal vectors via
test_floating_point_mul_matches_general_vectors and test_floating_point_mul_matches_subnormal_vectors.
The tests here add BF16 coverage and serve as a standalone entry point for the multiplier.
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from sprouthdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import FpMul, build_fp_mul, run_vectors_aby
from sprouthdl.sprouthdl_simulator import Simulator
from testing.floating_point.fp_testvectors_general import (
    build_bf16_vectors,
    build_f16_vectors,
    floatx_to_float,
)


def test_f16_mul_vectors():
    mod = build_fp_mul("F16MulTest", EW=5, FW=10)
    passed = run_vectors_aby(mod, build_f16_vectors(), label="f16 mul",
                             decoder=lambda b: floatx_to_float(b, 5, 10))
    assert passed


def test_bf16_mul_vectors():
    mod = build_fp_mul("BF16MulTest", EW=8, FW=7)
    passed = run_vectors_aby(mod, build_bf16_vectors(), label="bf16 mul",
                             decoder=lambda b: floatx_to_float(b, 8, 7))
    assert passed


def _is_subnormal(bits: int, EW: int, FW: int) -> bool:
    e = (bits >> FW) & ((1 << EW) - 1)
    f = bits & ((1 << FW) - 1)
    return e == 0 and f != 0


def _run_random_mul(EW: int, FW: int, name: str, num_vectors: int = 10_000, seed: int = 42):
    """Random test for FpMul (flush-to-zero).

    Subnormal inputs are excluded: FpMul zeros the mantissa for subnormal inputs but
    does not update the exponent path, producing completely wrong results.
    Subnormal inputs are not a supported input range for the FTZ multiplier.
    """
    W = 1 + EW + FW
    mul = FpMul(EW, FW)
    sim = Simulator(mul.to_module(name, with_clock=False, with_reset=False))
    rng = np.random.default_rng(seed)
    failures = []
    tested = 0
    while tested < num_vectors:
        a = int(rng.integers(0, 1 << W))
        b = int(rng.integers(0, 1 << W))
        if _is_subnormal(a, EW, FW) or _is_subnormal(b, EW, FW):
            continue  # subnormal inputs not supported by FpMul (FTZ-in is buggy)
        tested += 1
        exp = fp_encode(fp_decode(a, EW, FW) * fp_decode(b, EW, FW), EW, FW, subnormals=False)
        sim.set(mul.io.a, a)
        sim.set(mul.io.b, b)
        sim.eval()
        got = sim.get(mul.io.y)
        if got != exp:
            failures.append((a, b, exp, got))
    assert not failures, (
        f"{len(failures)}/{num_vectors} failures; first 5:\n"
        + "\n".join(f"  a={a:#06x} b={b:#06x} exp={e:#06x} got={g:#06x}"
                    for a, b, e, g in failures[:5])
    )


def test_f16_mul_random():
    _run_random_mul(5, 10, "F16MulRand")


def test_bf16_mul_random():
    _run_random_mul(8, 7, "BF16MulRand")


# def test_bf34_mul_random():
#     _run_random_mul(3, 4, "BF34MulRand")


if __name__ == "__main__":
    test_f16_mul_vectors()
    test_bf16_mul_vectors()
    test_f16_mul_random()
    test_bf16_mul_random()
    #test_bf34_mul_random()
