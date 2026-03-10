import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from sprouthdl.arithmetic.floating_point.sprout_hdl_float_add import FpAdd, build_fp_add
from sprouthdl.arithmetic.floating_point.fp_encoding import fp_decode, fp_encode, fp_unpack
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import run_vectors_aby
from sprouthdl.sprouthdl_simulator import Simulator
from testing.floating_point.fp_testvectors_general import (
    bits_inf,
    bits_min_normal,
    bits_max_sub,
    bits_min_sub,
    bits_qnan,
    bits_zero,
)


def build_add_vectors(EW: int, FW: int):
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    onept5 = fp_encode(1.5, EW, FW)
    onept25 = fp_encode(1.25, EW, FW)
    onept75 = fp_encode(1.75, EW, FW)
    two_pt_five = fp_encode(2.5, EW, FW)
    three_qtr = fp_encode(0.75, EW, FW)
    quarter = fp_encode(0.25, EW, FW)
    neg_onept25 = fp_encode(-1.25, EW, FW)
    neg_half = fp_encode(-0.5, EW, FW)
    neg_two = fp_encode(-2.0, EW, FW)
    min_norm = bits_min_normal(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    neg_min_norm = min_norm | (1 << (EW + FW))
    neg_max_sub = max_sub | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)
    neg0 = bits_zero(EW, FW, 1)
    pinf = bits_inf(EW, FW, 0)
    ninf = bits_inf(EW, FW, 1)
    qnan = bits_qnan(EW, FW)

    return [
        ("1+1=2", one, one, two),
        ("2+1=3", two, one, thr),
        ("1+0.5=1.5", one, half, onept5),
        ("0.75+0.75=1.5", three_qtr, three_qtr, onept5),
        ("1.25+0.5=1.75", onept25, half, onept75),
        ("1.5+(-1.25)=0.25", onept5, neg_onept25, quarter),
        ("1.75+(-0.5)=1.25", onept75, neg_half, onept25),
        ("1.5+1.5=3.0", onept5, onept5, fp_encode(3.0, EW, FW)),
        ("2.5+1.5=4.0", two_pt_five, onept5, fp_encode(4.0, EW, FW)),
        ("minNorm+(-maxSub)=minSub", min_norm, neg_max_sub, bits_min_sub(EW, FW)),
        ("minNorm+(-minNorm)=+0", min_norm, neg_min_norm, pos0),
        ("-2+2=0", neg_two, two, pos0),
        ("(-0)+0=0", neg0, pos0, pos0),
        ("inf+(-inf)=nan", pinf, ninf, qnan),
        ("inf+1=inf", pinf, one, pinf),
    ]


def test_f16_adder_vectors():
    mod = build_fp_add("F16AddTest", EW=5, FW=10)
    vectors = build_add_vectors(5, 10)
    passed = run_vectors_aby(mod, vectors, label="f16 add")
    assert passed


def test_f16_adder_random(num_vectors: int = 10_000, seed: int = 42):
    """IEEE 754 compliance: exhaustive random test against fp_encode/fp_decode reference."""
    EW, FW = 5, 10
    W = 1 + EW + FW
    adder = FpAdd(EW, FW)
    module = adder.to_module("F16RandTest", with_clock=False, with_reset=False)
    sim = Simulator(module)

    rng = np.random.default_rng(seed)
    failures = []
    for _ in range(num_vectors):
        a_bits = int(rng.integers(0, 1 << W))
        b_bits = int(rng.integers(0, 1 << W))
        exp_bits = fp_encode(fp_decode(a_bits, EW, FW) + fp_decode(b_bits, EW, FW), EW, FW)
        sim.set(adder.io.a, a_bits)
        sim.set(adder.io.b, b_bits)
        sim.eval()
        got_bits = sim.get(adder.io.y)
        if got_bits != exp_bits:
            failures.append((a_bits, b_bits, exp_bits, got_bits))

    assert not failures, (
        f"{len(failures)}/{num_vectors} failures; first 5:\n"
        + "\n".join(
            f"  a={a:#06x} b={b:#06x} exp={e:#06x} got={g:#06x}"
            for a, b, e, g in failures[:5]
        )
    )
    print(f"Random test passed: {num_vectors} vectors, 0 failures")


def test_f16_adder_subnormal_vectors():
    """Explicit subnormal input/output cases with subnormals=True (default)."""
    EW, FW = 5, 10
    min_norm = bits_min_normal(EW, FW)
    min_sub = bits_min_sub(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    neg_max_sub = max_sub | (1 << (EW + FW))
    neg_min_norm = min_norm | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)

    vectors = [
        # subnormal + subnormal = subnormal
        ("minSub+minSub=2*minSub", min_sub, min_sub,
         fp_encode(fp_decode(min_sub, EW, FW) * 2, EW, FW)),
        # subnormal + subnormal = normal (carry out of subnormal range)
        ("maxSub+minSub=minNorm", max_sub, min_sub, min_norm),
        # normal - normal = subnormal
        ("minNorm+(-maxSub)=minSub", min_norm, neg_max_sub, min_sub),
        # normal - normal = zero (exact cancellation)
        ("minNorm+(-minNorm)=+0", min_norm, neg_min_norm, pos0),
        # subnormal input, result is normal
        ("maxSub+maxSub", max_sub, max_sub,
         fp_encode(fp_decode(max_sub, EW, FW) * 2, EW, FW)),
    ]

    mod = build_fp_add("F16SubnormalTest", EW=EW, FW=FW, subnormals=True)
    passed = run_vectors_aby(mod, vectors, label="f16 subnormal")
    assert passed


def test_f16_adder_random_ftz(num_vectors: int = 10_000, seed: int = 42):
    """Random test with subnormals=False: results in subnormal range flush to +0."""
    EW, FW = 5, 10
    W = 1 + EW + FW
    adder = FpAdd(EW, FW, subnormals=False)
    module = adder.to_module("F16RandFtzTest", with_clock=False, with_reset=False)
    sim = Simulator(module)

    rng = np.random.default_rng(seed)
    failures = []
    for _ in range(num_vectors):
        a_bits = int(rng.integers(0, 1 << W))
        b_bits = int(rng.integers(0, 1 << W))
        exp_bits = fp_encode(
            fp_decode(a_bits, EW, FW) + fp_decode(b_bits, EW, FW), EW, FW, subnormals=False
        )
        sim.set(adder.io.a, a_bits)
        sim.set(adder.io.b, b_bits)
        sim.eval()
        got_bits = sim.get(adder.io.y)
        if got_bits != exp_bits:
            failures.append((a_bits, b_bits, exp_bits, got_bits))

    assert not failures, (
        f"{len(failures)}/{num_vectors} failures; first 5:\n"
        + "\n".join(
            f"  a={a:#06x} b={b:#06x} exp={e:#06x} got={g:#06x}"
            for a, b, e, g in failures[:5]
        )
    )
    print(f"Random FTZ test passed: {num_vectors} vectors, 0 failures")


def test_f16_adder_flush_to_zero():
    """With subnormals=False, results in the subnormal range flush to +0."""
    EW, FW = 5, 10
    min_norm = bits_min_normal(EW, FW)
    max_sub = bits_max_sub(EW, FW)
    min_sub = bits_min_sub(EW, FW)
    neg_max_sub = max_sub | (1 << (EW + FW))
    pos0 = bits_zero(EW, FW, 0)

    vectors = [
        # These would be subnormal with subnormals=True, but flush to zero
        ("minNorm+(-maxSub) flushes to 0", min_norm, neg_max_sub, pos0),
        ("minSub+minSub flushes to 0", min_sub, min_sub, pos0),
        # Normal results are unaffected
        ("1+1=2 still works", fp_encode(1.0, EW, FW), fp_encode(1.0, EW, FW), fp_encode(2.0, EW, FW)),
    ]

    mod = build_fp_add("F16FtzTest", EW=EW, FW=FW, subnormals=False)
    passed = run_vectors_aby(mod, vectors, label="f16 flush-to-zero")
    assert passed


if __name__ == "__main__":
    test_f16_adder_vectors()
    test_f16_adder_subnormal_vectors()
    test_f16_adder_flush_to_zero()
    test_f16_adder_random_ftz()
    test_f16_adder_random()
