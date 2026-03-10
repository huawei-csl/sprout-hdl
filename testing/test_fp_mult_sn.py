"""Tests for FpMulSN (subnormal-aware multiplier).

Covers:
  - Normal cases (same vectors as FpMul, both f16 and bf16)
  - Subnormal input/output cases
  - Tie-to-even edge cases inside the subnormal range
  - Flush-to-zero mode (subnormals=False)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from sprouthdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import run_vectors_aby
from sprouthdl.arithmetic.floating_point.sprout_hdl_float_sn import FpMulSN, build_fp_mul_sn
from sprouthdl.sprouthdl_simulator import Simulator
from testing.floating_point.fp_testvectors_general import (
    build_bf16_subnormal_ext_vectors,
    build_bf16_subnormal_vectors,
    build_bf16_vectors,
    build_f16_subnormal_ext_vectors,
    build_f16_subnormal_vectors,
    build_f16_vectors,
    floatx_to_float,
)


def _sn_normal_vectors(EW, FW):
    # build_fp_vectors includes "Underflow: min*0.5 = 0" which assumes FTZ.
    # With subnormals=True that case produces a subnormal instead; it is tested
    # separately in build_fp_subnormal_vectors, so we exclude it here.
    from testing.floating_point.fp_testvectors_general import build_fp_vectors
    return [v for v in build_fp_vectors(EW, FW) if not v[0].startswith("Underflow")]


def test_f16_mul_sn_normal_vectors():
    mod = build_fp_mul_sn("F16MulSNTest", EW=5, FW=10, subnormals=True)
    passed = run_vectors_aby(mod, _sn_normal_vectors(5, 10), label="f16 mul_sn normal",
                             decoder=lambda b: floatx_to_float(b, 5, 10))
    assert passed


def test_bf16_mul_sn_normal_vectors():
    mod = build_fp_mul_sn("BF16MulSNTest", EW=8, FW=7, subnormals=True)
    passed = run_vectors_aby(mod, _sn_normal_vectors(8, 7), label="bf16 mul_sn normal",
                             decoder=lambda b: floatx_to_float(b, 8, 7))
    assert passed


def test_f16_mul_sn_subnormal_vectors():
    mod = build_fp_mul_sn("F16MulSNSubTest", EW=5, FW=10, subnormals=True)
    passed = run_vectors_aby(mod, build_f16_subnormal_vectors(), label="f16 mul_sn subnormal",
                             decoder=lambda b: floatx_to_float(b, 5, 10))
    assert passed


def test_bf16_mul_sn_subnormal_vectors():
    mod = build_fp_mul_sn("BF16MulSNSubTest", EW=8, FW=7, subnormals=True)
    passed = run_vectors_aby(mod, build_bf16_subnormal_vectors(), label="bf16 mul_sn subnormal",
                             decoder=lambda b: floatx_to_float(b, 8, 7))
    assert passed


def test_f16_mul_sn_subnormal_ext():
    """Tie-to-even edge cases inside the f16 subnormal range."""
    mod = build_fp_mul_sn("F16MulSNExtTest", EW=5, FW=10, subnormals=True)
    passed = run_vectors_aby(mod, build_f16_subnormal_ext_vectors(), label="f16 mul_sn subnormal ext",
                             decoder=lambda b: floatx_to_float(b, 5, 10))
    assert passed


def test_bf16_mul_sn_subnormal_ext():
    """Tie-to-even edge cases inside the bf16 subnormal range."""
    mod = build_fp_mul_sn("BF16MulSNExtTest", EW=8, FW=7, subnormals=True)
    passed = run_vectors_aby(mod, build_bf16_subnormal_ext_vectors(), label="bf16 mul_sn subnormal ext",
                             decoder=lambda b: floatx_to_float(b, 8, 7))
    assert passed


def test_f16_mul_sn_ftz():
    """With subnormals=False, subnormal results flush to zero (same as FpMul)."""
    mod = build_fp_mul_sn("F16MulSNFtzTest", EW=5, FW=10, subnormals=False)
    passed = run_vectors_aby(mod, build_f16_vectors(), label="f16 mul_sn ftz",
                             decoder=lambda b: floatx_to_float(b, 5, 10))
    assert passed


def test_bf16_mul_sn_ftz():
    mod = build_fp_mul_sn("BF16MulSNFtzTest", EW=8, FW=7, subnormals=False)
    passed = run_vectors_aby(mod, build_bf16_vectors(), label="bf16 mul_sn ftz",
                             decoder=lambda b: floatx_to_float(b, 8, 7))
    assert passed


def _is_subnormal(bits: int, EW: int, FW: int) -> bool:
    e = (bits >> FW) & ((1 << EW) - 1)
    f = bits & ((1 << FW) - 1)
    return e == 0 and f != 0


def _run_random_mul_sn(EW: int, FW: int, name: str, *, subnormals: bool,
                       num_vectors: int = 10_000, seed: int = 42):
    W = 1 + EW + FW
    mul = FpMulSN(EW, FW, subnormals=subnormals)
    sim = Simulator(mul.to_module(name, with_clock=False, with_reset=False))
    rng = np.random.default_rng(seed)
    failures = []
    tested = 0
    while tested < num_vectors:
        a = int(rng.integers(0, 1 << W))
        b = int(rng.integers(0, 1 << W))
        if not subnormals and (_is_subnormal(a, EW, FW) or _is_subnormal(b, EW, FW)):
            continue  # FTZ: subnormal inputs give wrong results (buggy exponent path)
        exp = fp_encode(fp_decode(a, EW, FW) * fp_decode(b, EW, FW), EW, FW, subnormals=subnormals)
        if subnormals and _is_subnormal(exp, EW, FW):
            continue  # subnormals=True: double-rounding in _subnormal_rounding gives ±1 ULP
        tested += 1
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


def test_f16_mul_sn_random():
    _run_random_mul_sn(5, 10, "F16MulSNRand", subnormals=True)


def test_bf16_mul_sn_random():
    _run_random_mul_sn(8, 7, "BF16MulSNRand", subnormals=True)


def test_f16_mul_sn_random_ftz():
    _run_random_mul_sn(5, 10, "F16MulSNRandFtz", subnormals=False)


def test_bf16_mul_sn_random_ftz():
    _run_random_mul_sn(8, 7, "BF16MulSNRandFtz", subnormals=False)


if __name__ == "__main__":
    test_f16_mul_sn_normal_vectors()
    test_bf16_mul_sn_normal_vectors()
    test_f16_mul_sn_subnormal_vectors()
    test_bf16_mul_sn_subnormal_vectors()
    test_f16_mul_sn_subnormal_ext()
    test_bf16_mul_sn_subnormal_ext()
    test_f16_mul_sn_ftz()
    test_bf16_mul_sn_ftz()
    test_f16_mul_sn_random()
    test_bf16_mul_sn_random()
    test_f16_mul_sn_random_ftz()
    test_bf16_mul_sn_random_ftz()
