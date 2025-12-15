import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))
sys.path.append(ROOT)

from sprouthdl.arithmetic.floating_point.sprout_hdl_float_add import build_fp_add
from sprouthdl.arithmetic.floating_point.sprout_hdl_float import run_vectors_local
from testing.floating_point.fp_testvectors_general import fp_encode, bits_zero, bits_inf, bits_qnan


def build_add_vectors(EW: int, FW: int):
    one = fp_encode(1.0, EW, FW)
    two = fp_encode(2.0, EW, FW)
    thr = fp_encode(3.0, EW, FW)
    half = fp_encode(0.5, EW, FW)
    neg_two = fp_encode(-2.0, EW, FW)
    pos0 = bits_zero(EW, FW, 0)
    neg0 = bits_zero(EW, FW, 1)
    pinf = bits_inf(EW, FW, 0)
    ninf = bits_inf(EW, FW, 1)
    qnan = bits_qnan(EW, FW)

    return [
        ("1+1=2", one, one, two),
        ("2+1=3", two, one, thr),
        ("1+0.5=1.5", one, half, fp_encode(1.5, EW, FW)),
        ("-2+2=0", neg_two, two, pos0),
        ("(-0)+0=0", neg0, pos0, pos0),
        ("inf+(-inf)=nan", pinf, ninf, qnan),
        ("inf+1=inf", pinf, one, pinf),
    ]


@pytest.mark.xfail(reason="Adder model is experimental and may not perfectly match IEEE rounding")
def test_f16_adder_vectors():
    mod = build_fp_add("F16AddTest", EW=5, FW=10)
    vectors = build_add_vectors(5, 10)
    passed = run_vectors_local(mod, vectors, label="f16 add")
    assert passed == len(vectors)
