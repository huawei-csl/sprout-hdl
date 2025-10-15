# =========================
# HiFloat8 reference utilities
# =========================
# This module provides a pure-Python reference model for the Ascend HiFloat8
# (HiF8) tapered floating-point format as defined in
#   "Ascend HiFloat8 Format for Deep Learning", arXiv:2409.16626.
#
# The focus is on accurate encode/decode helpers and a behavioural multiplier
# that observes "rounding-half-away-from-zero" (TA) as described in the paper.
# These utilities are useful for golden-model simulation and test-vector
# generation.  Hardware generation is intentionally left separate because
# synthesising the full tapered arithmetic datapath is a substantial effort
# beyond the scope of this helper.

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

from sprouthdl.helpers import get_yosys_transistor_count
from sprouthdl.sprouthdl import *
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator

# -----------------------------------------------------------------------------
# Shared constants / helpers for hardware path
# -----------------------------------------------------------------------------

EXP_INT_WIDTH = 8  # signed exponent math range [-128, 127]


def _const_sint(val: int) -> Expr:
    return Const(val, SInt(EXP_INT_WIDTH))


def _const_uint(width: int, val: int) -> Expr:
    return Const(val, UInt(width))


# -----------------------------------------------------------------------------
# Encoding helpers
# -----------------------------------------------------------------------------


def _is_nan(enc: int) -> bool:
    return enc == 0x80


def _is_inf(enc: int) -> bool:
    return enc in (0x6F, 0xEF)


def _is_zero(enc: int) -> bool:
    return enc == 0x00


def hif8_to_float(enc: int) -> float:
    """Decode an 8-bit HiFloat8 payload into a Python float."""
    if _is_nan(enc):
        return math.nan
    if _is_zero(enc):
        return 0.0
    if _is_inf(enc):
        return -math.inf if enc & 0x80 else math.inf

    sign = -1.0 if (enc & 0x80) else 1.0
    payload = enc & 0x7F

    # Dot-field prefix decoding (variable length)
    bits = payload
    if bits >> 5 == 0b11:  # D = 4  (2-bit dot)
        exp_field = (bits >> 1) & 0xF
        mant_field = bits & 0x1
        exp_sign = (exp_field >> 3) & 0x1
        mag = 8 + (exp_field & 0x7)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 2.0
    elif bits >> 5 == 0b10:  # D = 3
        exp_field = (bits >> 2) & 0x7
        mant_field = bits & 0x3
        exp_sign = (exp_field >> 2) & 0x1
        mag = 4 + (exp_field & 0x3)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 4.0
    elif bits >> 5 == 0b01:  # D = 2
        exp_field = (bits >> 3) & 0x3
        mant_field = bits & 0x7
        exp_sign = (exp_field >> 1) & 0x1
        mag = 2 + (exp_field & 0x1)
        exponent = -mag if exp_sign else mag
        significand = 1.0 + mant_field / 8.0
    elif bits >> 4 == 0b001:  # D = 1
        exp_sign = (bits >> 3) & 0x1
        exponent = -1 if exp_sign else 1
        mant_field = bits & 0x7
        significand = 1.0 + mant_field / 8.0
    elif bits >> 3 == 0b0001:  # D = 0
        exponent = 0
        mant_field = bits & 0x7
        significand = 1.0 + mant_field / 8.0
    else:  # DML denormals
        mant_field = bits & 0x7
        if mant_field == 0:
            return 0.0
        exponent = mant_field - 23  # M encodes exponent offset
        significand = 1.0

    return sign * math.ldexp(significand, exponent)


@dataclass(frozen=True)
class HiF8Value:
    encoding: int
    value: float


@lru_cache(maxsize=None)
def _positive_hif8_catalogue() -> List[HiF8Value]:
    """Enumerate all non-negative, non-NaN HiF8 encodings with their values."""
    catalogue: List[HiF8Value] = []
    for enc in range(0x00, 0x80):
        if _is_nan(enc):
            continue
        val = hif8_to_float(enc)
        if math.isnan(val):
            continue
        catalogue.append(HiF8Value(enc, val))
    # Sort by numeric value to ease binary search / tie-breaking.
    catalogue.sort(key=lambda v: v.value)
    return catalogue


def _tie_away_key(target: float, candidate: HiF8Value) -> Tuple[float, float]:
    diff = abs(candidate.value - target)
    # Tie-away-from-zero: prefer larger magnitude on equal difference.
    return (diff, -candidate.value)


def float_to_hif8(value: float) -> int:
    """Encode a Python float into HiFloat8 using TA rounding."""
    if math.isnan(value):
        return 0x80  # canonical NaN
    if math.isinf(value):
        return 0xEF if value < 0 else 0x6F

    sign = 1 if math.copysign(1.0, value) < 0 else 0
    abs_val = abs(value)

    if abs_val == 0.0:
        return 0x00

    catalogue = _positive_hif8_catalogue()
    best = min(catalogue, key=lambda cand: _tie_away_key(abs_val, cand))
    return best.encoding | (sign << 7)


def multiply_hif8(a: int, b: int) -> int:
    """HiF8 multiply using the reference encode/decode helpers."""
    fa = hif8_to_float(a)
    fb = hif8_to_float(b)

    if math.isnan(fa) or math.isnan(fb):
        return 0x80
    if (math.isinf(fa) and fb == 0.0) or (math.isinf(fb) and fa == 0.0):
        return 0x80

    product = fa * fb
    return float_to_hif8(product)


# -----------------------------------------------------------------------------
# Hardware module builder
# -----------------------------------------------------------------------------


def build_hif8_mul_module(name: str = "HiF8Mul") -> Module:
    m = Module(name, with_clock=False, with_reset=False)
    a = m.input(UInt(8), "a")
    b = m.input(UInt(8), "b")
    y = m.output(UInt(8), "y")

    table = [[multiply_hif8(av, bv) for bv in range(256)] for av in range(256)]

    a_low = a[0:4]
    a_high = a[4:8]
    b_low = b[0:4]
    b_high = b[4:8]

    result = _const_uint(8, 0)
    for ah in range(16):
        row_low = _const_uint(8, 0)
        for al in range(16):
            aval = (ah << 4) | al
            row = _const_uint(8, 0)
            for bh in range(16):
                group = _const_uint(8, 0)
                for bl in range(16):
                    bval = (bh << 4) | bl
                    prod = table[aval][bval]
                    group = mux(
                        b_low == _const_uint(4, bl),
                        _const_uint(8, prod),
                        group,
                    )
                row = mux(b_high == _const_uint(4, bh), group, row)
            row_low = mux(a_low == _const_uint(4, al), row, row_low)
        result = mux(a_high == _const_uint(4, ah), row_low, result)

    y <<= result
    return m

# -----------------------------------------------------------------------------
# Vector helpers for regression tests / demos
# -----------------------------------------------------------------------------

def catalogue_summary() -> Dict[str, int]:
    """Return basic counts for documentation / sanity checks."""
    cats = _positive_hif8_catalogue()
    normals = [v for v in cats if v.encoding not in (0x00,) and not (v.encoding >> 5 == 0)]
    denormals = [v for v in cats if (v.encoding >> 3) == 0 and v.encoding != 0x00]
    return {
        "total": len(cats),
        "normals": len(normals),
        "denormals": len(denormals),
    }


def build_basic_vectors() -> List[Tuple[str, int, int, int]]:
    """Simple regression vectors anchored on specification examples."""
    return [
        ("unity", 0x33, 0x33, multiply_hif8(0x33, 0x33)),
        ("2 * 0.5", 0x6C, 0x17, multiply_hif8(0x6C, 0x17)),
        ("zero * max", 0x00, 0x6E, multiply_hif8(0x00, 0x6E)),
        ("min denorm * 2", 0x01, 0x6C, multiply_hif8(0x01, 0x6C)),
        ("inf * finite", 0x6F, 0x33, multiply_hif8(0x6F, 0x33)),
        ("finite * inf", 0x33, 0x6F, multiply_hif8(0x33, 0x6F)),
    ]


# -----------------------------------------------------------------------------
# Standalone smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("HiF8 catalogue summary:", catalogue_summary())
    for name, a, b, exp in build_basic_vectors():
        fa = hif8_to_float(a)
        fb = hif8_to_float(b)
        fc = hif8_to_float(exp)
        prod = hif8_to_float(multiply_hif8(a, b))
        print(
            f"{name:12s}: a=0x{a:02X} ({fa:+.6e})  "
            f"b=0x{b:02X} ({fb:+.6e})  ->  "
            f"y=0x{exp:02X} ({fc:+.6e})  [recalc {prod:+.6e}]"
        )

    print("\nVerifying hardware multiplier against reference model...")
    dut = build_hif8_mul_module("HiF8Mul_Ref")
    sim = Simulator(dut)
    mismatches = 0
    for aval in range(256):
        for bval in range(256):
            sim.set("a", aval).set("b", bval).eval()
            got = sim.get("y")
            exp = multiply_hif8(aval, bval)
            if got != exp:
                if mismatches < 10:
                    print(
                        f"Mismatch a=0x{aval:02X} b=0x{bval:02X} "
                        f"got=0x{got:02X} exp=0x{exp:02X}"
                    )
                mismatches += 1
    if mismatches == 0:
        print("All 65,536 combinations match.")
    else:
        print(f"Found {mismatches} mismatches.")

    # get yosys transistor count
    
    n_t = get_yosys_transistor_count(dut)
    print("Yosys transistor count:", n_t)