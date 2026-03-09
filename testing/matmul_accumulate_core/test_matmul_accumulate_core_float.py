from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from sprouthdl.aggregate.aggregate_floating_point import FloatingPointType
from sprouthdl.arithmetic.int_arithmetic_config import AdderConfig, MultiplierConfig
from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_float import (
    FpMMAcCfg,
    FpMMAcDims,
    FpMatmulAccumulateComponent,
)
from sprouthdl.helpers import get_yosys_metrics, run_vectors_on_simulator
from sprouthdl.sprouthdl_simulator import Simulator
from testing.floating_point.fp_testvectors_general import fp_decode, fp_encode


def _fp_add_hw_ref(a_bits: int, b_bits: int, ew: int, fw: int) -> int:
    """Python reference matching hardware FpAdd: extends by 2 guard bits, then truncates.

    The hardware does cat(Const(0, UInt(2)), m) which in sprouthdl's LSB-first
    convention means m_ext = m << 2 (2 zero guard bits at LSB).  After addition
    and normalization, bits 0 and 1 are discarded without rounding.  This differs
    from fp_encode(a+b) which rounds to nearest-even.
    """
    max_e = (1 << ew) - 1

    def unpack(v):
        s = (v >> (ew + fw)) & 1
        e = (v >> fw) & max_e
        f = v & ((1 << fw) - 1)
        return s, e, f

    def pack(s, e, f):
        return ((s & 1) << (ew + fw)) | ((e & max_e) << fw) | (f & ((1 << fw) - 1))

    sA, eA, fA = unpack(a_bits)
    sB, eB, fB = unpack(b_bits)

    is_nanA = eA == max_e and fA != 0
    is_nanB = eB == max_e and fB != 0
    is_infA = eA == max_e and fA == 0
    is_infB = eB == max_e and fB == 0

    if is_nanA or is_nanB or (is_infA and is_infB and sA != sB):
        nan_payload = (1 << (fw - 1)) if fw > 0 else 1
        return pack(0, max_e, nan_payload)
    if is_infA:
        return pack(sA, max_e, 0)
    if is_infB:
        return pack(sB, max_e, 0)

    hA = 0 if eA == 0 else 1
    hB = 0 if eB == 0 else 1
    mA = (hA << fw) | fA
    mB = (hB << fw) | fB
    eA_eff = 1 if eA == 0 else eA
    eB_eff = 1 if eB == 0 else eB

    # Matches hardware: a_is_bigger = eA_gt | (e_eq & mA_ge)
    a_is_bigger = (eA_eff > eB_eff) or (eA_eff == eB_eff and mA >= mB)
    if a_is_bigger:
        exp_delta, m_big, m_small, s_big, s_small, e_big = eA_eff - eB_eff, mA, mB, sA, sB, eA_eff
    else:
        exp_delta, m_big, m_small, s_big, s_small, e_big = eB_eff - eA_eff, mB, mA, sB, sA, eB_eff

    # Hardware: cat(Const(0, UInt(2)), m) → m << 2 (LSB-first → 2 guard bits at bottom)
    m_big_ext = m_big << 2
    m_small_shift = (m_small << 2) >> exp_delta  # integer right shift: truncates low bits

    mant_mag = m_big_ext + m_small_shift if s_big == s_small else m_big_ext - m_small_shift

    if mant_mag == 0:
        return pack(s_big & s_small, 0, 0)

    lead_pos = mant_mag.bit_length() - 1
    target_pos = fw + 2  # hidden bit position after cat(0b00, mant)

    if lead_pos > target_pos:
        shift = lead_pos - target_pos
        mant_norm = mant_mag >> shift  # truncate low bits (matches hardware)
        exp_out = e_big + shift
    elif lead_pos < target_pos:
        shift = target_pos - lead_pos
        mant_norm = mant_mag << shift
        exp_out = e_big - shift
    else:
        mant_norm = mant_mag
        exp_out = e_big

    if exp_out <= 0:
        return pack(s_big, 0, 0)
    if exp_out >= max_e:
        return pack(s_big, max_e, 0)

    # frac_final = mant_norm[2:fw+2]: discard 2 guard bits, extract fw fraction bits
    frac_out = (mant_norm >> 2) & ((1 << fw) - 1)
    return pack(s_big, exp_out, frac_out)


def _generate_fp_matmul_vectors(
    core: FpMatmulAccumulateComponent,
    num_vectors: int,
    *,
    seed: int = 42,
) -> List[Tuple[str, Dict[str, int], Dict[str, int]]]:
    """Generate test vectors for an FpMatmulAccumulateComponent.

    The reference computation follows the same sequential left-to-right
    accumulation order as elaborate():
        dot = A[i,0]*B[0,j]
        dot = dot + A[i,1]*B[1,j]  ...
        acc = dot + C[i,j]
    Each operation is quantised to the target FP format, matching the hardware.
    """
    rng = np.random.default_rng(seed=seed)
    ft = core.cfg.ftype
    ew, fw = ft.exponent_width, ft.fraction_width
    dims = core.cfg.dims
    m, n, k = dims.dim_m, dims.dim_n, dims.dim_k

    def _encode(x: float) -> int:
        return fp_encode(float(x), ew, fw)

    def _decode(b: int) -> float:
        return fp_decode(int(b), ew, fw)

    vectors: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

    for idx in range(num_vectors):
        a_f = rng.uniform(0.1, 2.0, size=(m, k))
        b_f = rng.uniform(0.1, 2.0, size=(k, n))
        c_f = rng.uniform(0.1, 2.0, size=(m, n))

        a_bits = np.vectorize(_encode)(a_f)
        b_bits = np.vectorize(_encode)(b_f)
        c_bits = np.vectorize(_encode)(c_f)

        # Decode to quantised floats so multiplications use the same precision as hardware.
        # Multiplier uses round-to-nearest-even (matches fp_encode).
        # Adder truncates 2 guard bits (use _fp_add_hw_ref, not fp_encode).
        a_q = np.vectorize(_decode)(a_bits)
        b_q = np.vectorize(_decode)(b_bits)

        y_bits = np.zeros((m, n), dtype=int)
        for i in range(m):
            for j in range(n):
                dot_bits = _encode(a_q[i, 0] * b_q[0, j])
                for kk in range(1, k):
                    prod_bits = _encode(a_q[i, kk] * b_q[kk, j])
                    dot_bits = _fp_add_hw_ref(dot_bits, prod_bits, ew, fw)
                y_bits[i, j] = _fp_add_hw_ref(dot_bits, int(c_bits[i, j]), ew, fw)

        ins: Dict[str, int] = {}
        outs: Dict[str, int] = {}

        for i in range(m):
            for kk in range(k):
                ins[core.io.A[i, kk].bits.name] = int(a_bits[i, kk])
        for kk in range(k):
            for j in range(n):
                ins[core.io.B[kk, j].bits.name] = int(b_bits[kk, j])
        for i in range(m):
            for j in range(n):
                ins[core.io.C[i, j].bits.name] = int(c_bits[i, j])
                outs[core.io.Y[i, j].bits.name] = int(y_bits[i, j])

        vectors.append((f"vec_{idx}", ins, outs))

    return vectors


def _float16_matmul_accumulate(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Reference accumulation using numpy float16 arithmetic."""
    dim_m, dim_k = a.shape
    _, dim_n = b.shape
    y = np.empty((dim_m, dim_n), dtype=np.float16)
    for i in range(dim_m):
        for j in range(dim_n):
            acc = np.float16(c[i, j])
            for k in range(dim_k):
                acc = np.float16(acc + np.float16(a[i, k] * b[k, j]))
            y[i, j] = acc
    return y


def test_fp_matmul_accumulate_simulation():
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=8)

    cfg = FpMMAcCfg(dims=FpMMAcDims(dim_m=dim, dim_n=dim, dim_k=dim), ftype=ft)
    core = FpMatmulAccumulateComponent(cfg)
    module = core.to_module("fp_matmul_accumulate")

    print(f"Output matrix Y has shape: ({dim}, {dim}) with element width {core.io.Y[0, 0].bits.typ.width} bits")

    vectors = _generate_fp_matmul_vectors(core, num_vectors=128)

    sim = Simulator(module)
    failures = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)
    print(f"Simulation complete: {failures} failures")

    # yosys_metrics = get_yosys_metrics(module)
    # print(f"Yosys metrics: {yosys_metrics}")


def test_fp_matmul_accumulate_simulation_approx():
    """Approximate simulation test using numpy float16 as reference (np.isclose)."""
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=8)
    ew, fw = ft.exponent_width, ft.fraction_width

    cfg = FpMMAcCfg(dims=FpMMAcDims(dim_m=dim, dim_n=dim, dim_k=dim), ftype=ft)
    core = FpMatmulAccumulateComponent(cfg)
    module = core.to_module("fp_matmul_accumulate_approx")

    sim = Simulator(module)

    rng = np.random.default_rng(seed=123)
    a_vals = rng.uniform(0.1, 2.0, size=(dim, dim))
    b_vals = rng.uniform(0.1, 2.0, size=(dim, dim))
    c_vals = rng.uniform(0.1, 2.0, size=(dim, dim))

    def _encode(x: float) -> int:
        return fp_encode(float(x), ew, fw)

    def _decode(b: int) -> float:
        return fp_decode(int(b), ew, fw)

    a_bits = np.vectorize(_encode)(a_vals)
    b_bits = np.vectorize(_encode)(b_vals)
    c_bits = np.vectorize(_encode)(c_vals)

    # Quantise inputs to the target FP format before passing to the reference.
    a_q = np.vectorize(_decode)(a_bits)
    b_q = np.vectorize(_decode)(b_bits)
    c_q = np.vectorize(_decode)(c_bits)

    for i in range(dim):
        for k in range(dim):
            sim.set(core.io.A[i, k].bits, int(a_bits[i, k]))
    for k in range(dim):
        for j in range(dim):
            sim.set(core.io.B[k, j].bits, int(b_bits[k, j]))
    for i in range(dim):
        for j in range(dim):
            sim.set(core.io.C[i, j].bits, int(c_bits[i, j]))

    sim.eval()

    y_hw = np.vectorize(_decode)(
        np.array([[sim.get(core.io.Y[i, j].bits) for j in range(dim)] for i in range(dim)])
    )
    y_expected = _float16_matmul_accumulate(a_q, b_q, c_q)

    assert np.isclose(y_hw, y_expected, rtol=1e-2, atol=1e-2).all(), (
        f"Mismatch:\nhw      = {y_hw}\nexpected= {y_expected}"
    )


def test_fp_matmul_accumulate_simulation_with_cfg():
    """Bit-exact simulation with explicit stage-based multiplier and ripple-carry adder configs."""
    dim = 4
    ft = FloatingPointType(exponent_width=5, fraction_width=8)

    mult_cfg = MultiplierConfig(
        use_operator=False,
        multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
        encodings=TwoInputAritEncodings.with_enc(Encoding.unsigned),
        ppg_opt=PPGOption.AND,
        ppa_opt=PPAOption.WALLACE_TREE,
        fsa_opt=FSAOption.RIPPLE_CARRY,
    )
    adder_cfg = AdderConfig(
        use_operator=False,
        fsa_opt=FSAOption.RIPPLE_CARRY,
        full_output_bit=True,
        encoding=Encoding.unsigned,
    )

    cfg = FpMMAcCfg(
        dims=FpMMAcDims(dim_m=dim, dim_n=dim, dim_k=dim),
        ftype=ft,
        adder_cfg=adder_cfg,
        mult_cfg=mult_cfg,
    )
    core = FpMatmulAccumulateComponent(cfg)
    module = core.to_module("fp_matmul_accumulate_cfg")

    vectors = _generate_fp_matmul_vectors(core, num_vectors=32)

    sim = Simulator(module)
    failures = run_vectors_on_simulator(sim, vectors, use_signed=False, raise_on_fail=True, print_on_pass=False)
    print(f"Simulation complete (with cfg): {failures} failures")


if __name__ == "__main__":
    test_fp_matmul_accumulate_simulation()
    test_fp_matmul_accumulate_simulation_approx()
    test_fp_matmul_accumulate_simulation_with_cfg()
