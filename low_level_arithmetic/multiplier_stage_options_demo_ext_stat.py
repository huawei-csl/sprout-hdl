"""Demonstrate how to mix and match multiplier stages."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
import sys
import time
from typing import NamedTuple, Self, Tuple, Type
import uuid

import numpy as np
from tqdm import tqdm


from low_level_arithmetic.multiplier_stage_options_demo_lib import Demo, MultiplierEncodings, PPAOption, PPGOption
from low_level_arithmetic.multiplier_stage_options_demo_lib import FSAOption
from low_level_arithmetic.multiplier_stage_options_demo_ext_stat_helper import MultiplierRow, ParquetCollector, _flatten_op_nodes
from low_level_arithmetic.mutipliers_ext import StageBasedExtMultiplier, StageBasedMultiplierBasic, StageBasedSignMagnitudeExtMultiplier, StageBasedSignMagnitudeExtToTwosComplementMultiplier, StageBasedSignMagnitudeExtToTwosComplementUpperMultiplier, StageBasedSignMagnitudeExtUpMultiplier, StageBasedSignMagnitudeMultiplier, StageBasedSignMagnitudeToTwosComplementMultiplier, StarMultiplier
from low_level_arithmetic.ppa_stages import (
    CarrySaveAccumulator,
    DaddaTreeAccumulator,
    FourTwoCompressorAccumulator,
    WallaceTreeAccumulator,
)
from low_level_arithmetic.test_vector_generation import (
    Encoding,
    MultiplierTestVectors,
    to_encoding,
)
from low_level_arithmetic.ppg_baugh_wooley_stages import (
    BaughWooleyPartialProductGenerator,
)
from low_level_arithmetic.ppg_basic_stages import (
    BasicUnsignedPartialProductGenerator,
)
from low_level_arithmetic.ppg_booth_optim_signed_stages import (
    BoothOptimizedSignedPartialProductGenerator,
)
from low_level_arithmetic.ppg_booth_optim_stages import (
    BoothOptimizedPartialProductGenerator,
)
from low_level_arithmetic.ppg_booth_unoptim_stages import (
    BoothUnoptimizedPartialProductGenerator,
)
from low_level_arithmetic.multiplier_stage_core import (
    CompressorTreeAccumulator,
    RippleCarryFinalAdder,
    StageBasedMultiplier,
)
from low_level_arithmetic.fsa_stages import (
    BrentKungPrefixFinalStage,
    PrefixAdderFinalStage,
    RipplePrefixFinalStage,
    SklanskyPrefixFinalStage,
)
from sprouthdl.helpers import get_aig_stats, get_switch_count, get_yosys_metrics, get_yosys_transistor_count, refactor_module_to_aig, run_vectors
from sprouthdl.sprouthdl import Op2, reset_shared_cache
from sprouthdl.sprouthdl_aiger import AigerExporter, AigerImporter
from sprouthdl.sprouthdl_module import gen_spec
from sprouthdl.sprouthdl_io_collector import IOCollector
from testing.floating_point.sim_int import optimize_aag, run_vectors_io_log
from testing.test_different_logic import run_vectors_io
from low_level_arithmetic.multiplier_stage_options_demo_list import demos1, get_selection1_list, get_selection1_list

def get_target_sigma_index(sigmas: list, n_bits: int) -> int:

    def target_sigma_for(n_bits: int) -> float:
        """σ_target = 3/16 * 2**n_bits."""
        return (3.0 / 16.0) * (2 ** int(n_bits))

    tgt = target_sigma_for(n_bits)

    sigmas_dist = np.abs(sigmas.astype(float) - tgt)
    idx = np.argmin(sigmas_dist)
    return int(idx)


def run_configuration(
    multiplier_cls: Type[StageBasedExtMultiplier],
    encodings: MultiplierEncodings,
    ppg_opt: PPGOption,
    ppa_opt: PPAOption,
    fsa_opt: FSAOption,
    n_bits: int,
    num_vectors: int,
    sigmas: list,
    all_sigmas: bool = True,
) -> None:
    reset_shared_cache()

    multiplier = multiplier_cls(
        a_w=n_bits,
        b_w=n_bits,
        a_encoding=encodings.a,
        b_encoding=encodings.b,
        ppg_cls=ppg_opt.value,
        ppa_cls=ppa_opt.value,
        fsa_cls=fsa_opt.value,
        optim_type="area",
    )

    module = multiplier.to_module(f"demo_{ppg_opt.name.lower()}_{encodings.a.name.lower()}_{encodings.b.name.lower()}_{fsa_opt.name.lower()}")
    print(f"Built module '{module.name}' using PPG={ppg_opt.name}, PPA={ppa_opt.name}, FSA={fsa_opt.name}")

    vecs = MultiplierTestVectors(
                a_w=n_bits,
                b_w=n_bits,
                y_w=multiplier.io.y.typ.width,
                num_vectors=num_vectors,
                tb_sigma=None,
                a_encoding=encodings.a,
                b_encoding=encodings.b,
                y_encoding=encodings.y,
            ).generate()

    # run_vectors_io(module, vecs)

    # -- swact --
    m_aig = refactor_module_to_aig(module)

    # AIG network test sim
    print("Sim (AIG) …")
    # run_vectors_io(m_aig, vecs)

    exprs = m_aig.all_exprs()
    all_ands = [e for e in exprs if isinstance(e, Op2) and e.op == "&"]

    def run_and_count(vecs_run) -> int:
        states_list = run_vectors(m_aig, vecs_run, exprs=all_ands)
        return get_switch_count(states_list)             

    # switches = run_and_count(vecs)
    # print(f"Average AND switches: {switches}")

    switches = []
    if not all_sigmas:
        sigmas = [sigmas[get_target_sigma_index(np.array(sigmas), n_bits)]]
    for sigma in sigmas:
        vecs = MultiplierTestVectors(
                      a_w=n_bits,
                      b_w=n_bits,
                      y_w=multiplier.io.y.typ.width,
                      num_vectors=num_vectors,
                      tb_sigma=sigma,
                      a_encoding=encodings.a,
                      b_encoding=encodings.b,
                      y_encoding=encodings.y,
                  ).generate()
        switches.append(run_and_count(vecs))
        print(f"Average AND switches (sigma={sigma}): {switches[-1]}")

    gr = m_aig.module_analyze()
    tc = get_yosys_transistor_count(m_aig)
    ym = get_yosys_metrics(m_aig)
    num_aig_gates = len(all_ands) # aig.gates()
    aig_stats = get_aig_stats(m_aig)

    run_id = uuid.uuid4().hex
    t_now = time.time()

    rows = []
    for sigma_val, sw_val in zip(sigmas, switches):
        row = MultiplierRow(
                    run_id=run_id,
                    timestamp=t_now,
                    module_name=module.name,
            
                    n_bits=int(n_bits),
                    multiplier_cls=multiplier_cls.__name__,
                    ppg_opt=ppg_opt.name,
                    ppa_opt=ppa_opt.name,
                    fsa_opt=fsa_opt.name,
                    a_enc=encodings.a.name,
                    b_enc=encodings.b.name,
                    y_enc=encodings.y.name,
                    a_w=int(multiplier.io.a.typ.width),
                    b_w=int(multiplier.io.b.typ.width),
                    y_w=int(multiplier.io.y.typ.width),
                    num_vectors=int(num_vectors),
            
                    sigma=float(sigma_val),
                    switches=int(sw_val),
                    n_sigmas=len(sigmas),
                    multiple_sigmas=len(sigmas) > 1,
            
                    total_expr_nodes=int(gr.total_expr_nodes),
                    max_depth=int(gr.max_depth),
                    depth_y=int(gr.output_depth["y"]),
                    op_nodes_json=_flatten_op_nodes(gr.op_nodes),
            
                    num_wires=int(ym["num_wires"]),
                    num_cells=int(ym["num_cells"]),
                    estimated_num_transistors=int(ym["estimated_num_transistors"]),
                    transistor_count=int(tc),
                    
                    num_aig_gates=aig_stats["num_gates"],
                    aig_depth=aig_stats["depth"],
                )
        rows.append(row)
    return rows

def run_stage_multiplier_ext_demo(demos: list[Demo]) -> None:  # pragma: no cover - demonstration only

    num_vectors = 2000
    bitwidths = [4, 8, 16]
    sigma_factor = 0.5
    n_steps_sigma = 8
    parallel = False

    sys.setrecursionlimit(2000)

    # datecode
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_file = f"data/multiplier_runs_{run_id}.parquet"
    collector = ParquetCollector(out_file)
    
    if parallel:
        bitwidths = list(reversed(bitwidths))  # start with big ones first to get better load balancing

    for n_bits in bitwidths:

        sigma_max = 2**n_bits * sigma_factor
        sigma_start = sigma_max / n_steps_sigma        
        sigmas = np.linspace(sigma_start, sigma_max, n_steps_sigma)

        if not parallel:

            for demo in demos:                
                rows = run_configuration(demo.multiplier_cls, demo.encodings, demo.ppg_opt, demo.ppa_opt, demo.fsa_opt, n_bits, num_vectors, sigmas, all_sigmas=demo.all_sigma)
                collector.extend(rows)

        else:       

            max_workers = 80

            _worker = run_configuration

            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = []
                for demo in demos: 
                    futures.append(
                        ex.submit(_worker, demo.multiplier_cls, demo.encodings, demo.ppg_opt, demo.ppa_opt, demo.fsa_opt, n_bits, num_vectors, sigmas, all_sigmas=demo.all_sigma)
                    )
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Running demos", unit="demo"):
                    rec = fut.result()
                    # Safely aggregate on the main process:
                    if collector is not None and rec is not None:
                        collector.extend(rec)   # or whatever your API is
                    results.extend(rec)

    # get number of rows added
    print(f"Total number of configurations run: {len(demos) * len(bitwidths) * n_steps_sigma}")
    print(f"Total rows collected: {collector.n_rows()}")

    # save to file
    collector.save(append=True)

if __name__ == "__main__":
    # run_stage_multiplier_ext_demo(demos=demos1)
    run_stage_multiplier_ext_demo(demos=get_selection1_list(small_sweep=True))
