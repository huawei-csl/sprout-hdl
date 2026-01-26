"""
Evaluation script for sweeping dim and width parameters of the Winograd matmul accumulate core.
Uses multiprocessing with timeout to run evaluations in parallel.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import (
    FSAOption,
    MultiplierOption,
    PPAOption,
    PPGOption,
    TwoInputAritEncodings,
)
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, is_signed
from sprouthdl.cores.matmul_accumulate.matmul_accumulate_core_winograd import (
    AdderConfig,
    MMAcCfg,
    MMAcDims,
    MMAcWidths,
    MatmulAccumulateComponent,
    MultiplierConfig,
    max_y_width_unsigned,
)
from sprouthdl.helpers import get_yosys_metrics


@dataclass
class EvalConfig:
    dim_m: int
    dim_n: int
    dim_k: int
    a_width: int
    b_width: int


def run_single_evaluation(config: EvalConfig) -> Optional[dict[str, Any]]:
    """Run a single evaluation with the given configuration.

    Returns the yosys metrics dict or None if an error occurred.
    """
    try:
        c_width = max_y_width_unsigned(config.a_width, config.b_width, config.dim_k, include_carry_from_add=False)
        encoding = Encoding.twos_complement
        signed_io_type = True
        n_iter_optimizations = 0

        mult_cfg = MultiplierConfig(
            use_operator=False,
            multiplier_opt=MultiplierOption.STAGE_BASED_MULTIPLIER,
            encodings=TwoInputAritEncodings.with_enc(encoding),
            ppg_opt=PPGOption.BAUGH_WOOLEY if is_signed(encoding) else PPGOption.AND,
            ppa_opt=PPAOption.WALLACE_TREE,
            fsa_opt=FSAOption.RIPPLE_CARRY,
        )
        add_cfg = AdderConfig(
            use_operator=False, fsa_opt=FSAOption.RIPPLE_CARRY, full_output_bit=True, encoding=encoding
        )

        core_config = MMAcCfg(
            dims=MMAcDims(dim_m=config.dim_m, dim_n=config.dim_n, dim_k=config.dim_k),
            widths=MMAcWidths(a_width=config.a_width, b_width=config.b_width, c_width=c_width),
            mult_cfg=mult_cfg,
            add_cfg=add_cfg,
        )

        core = MatmulAccumulateComponent(core_config, signed_io_type=signed_io_type)
        module = core.to_module("matmul_accumulate_core")

        yosys_metrics = get_yosys_metrics(module, n_iter_optimizations=n_iter_optimizations)

        return {
            "config": {
                "dim_m": config.dim_m,
                "dim_n": config.dim_n,
                "dim_k": config.dim_k,
                "a_width": config.a_width,
                "b_width": config.b_width,
            },
            "metrics": yosys_metrics,
        }
    except Exception as e:
        print(f"Error for config {config}: {e}")
        return None


def worker_wrapper(args: tuple[EvalConfig, mp.Queue]) -> None:
    """Wrapper function for multiprocessing worker."""
    config, result_queue = args
    result = run_single_evaluation(config)
    result_queue.put((config, result))


def run_evaluation_with_timeout(config: EvalConfig, timeout_minutes: float) -> Optional[dict[str, Any]]:
    """Run evaluation with a timeout. Returns None if timeout or error."""
    result_queue = mp.Queue()
    process = mp.Process(target=worker_wrapper, args=((config, result_queue),))
    process.start()
    process.join(timeout=timeout_minutes * 60)

    if process.is_alive():
        print(f"Timeout for config: dim={config.dim_m}, width={config.a_width}")
        process.terminate()
        process.join()
        return None

    if not result_queue.empty():
        _, result = result_queue.get()
        return result
    return None


def run_sweep(
    dim_vals: list[int],
    width_vals: list[int],
    timeout_minutes: float = 5.0,
    n_workers: int = 4,
    output_file: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Run parameter sweep over dim_vals and width_vals.

    Args:
        dim_vals: List of dimension values (applied to dim_m, dim_n, dim_k)
        width_vals: List of width values (applied to a_width, b_width)
        timeout_minutes: Maximum time per evaluation in minutes
        n_workers: Number of parallel workers
        output_file: Path to save results (JSON format)

    Returns:
        List of result dictionaries
    """
    configs = []
    for dim_val in dim_vals:
        for width_val in width_vals:
            configs.append(EvalConfig(
                dim_m=dim_val,
                dim_n=dim_val,
                dim_k=dim_val,
                a_width=width_val,
                b_width=width_val,
            ))

    print(f"Running sweep with {len(configs)} configurations")
    print(f"Timeout: {timeout_minutes} minutes per evaluation")
    print(f"Workers: {n_workers}")
    print("-" * 60)

    results = []

    # Use a pool with timeout handling
    with mp.Pool(processes=n_workers) as pool:
        async_results = []
        for config in configs:
            ar = pool.apply_async(run_single_evaluation, (config,))
            async_results.append((config, ar))

        for config, ar in async_results:
            try:
                result = ar.get(timeout=timeout_minutes * 60)
                if result is not None:
                    results.append(result)
                    print(f"Completed: dim={config.dim_m}, width={config.a_width} -> "
                          f"transistors={result['metrics'].get('estimated_num_transistors', 'N/A')}")
                else:
                    results.append({
                        "config": {
                            "dim_m": config.dim_m,
                            "dim_n": config.dim_n,
                            "dim_k": config.dim_k,
                            "a_width": config.a_width,
                            "b_width": config.b_width,
                        },
                        "metrics": None,
                        "error": "Evaluation returned None",
                    })
                    print(f"Failed: dim={config.dim_m}, width={config.a_width}")
            except mp.TimeoutError:
                results.append({
                    "config": {
                        "dim_m": config.dim_m,
                        "dim_n": config.dim_n,
                        "dim_k": config.dim_k,
                        "a_width": config.a_width,
                        "b_width": config.b_width,
                    },
                    "metrics": None,
                    "error": "Timeout",
                })
                print(f"Timeout: dim={config.dim_m}, width={config.a_width}")
            except Exception as e:
                results.append({
                    "config": {
                        "dim_m": config.dim_m,
                        "dim_n": config.dim_n,
                        "dim_k": config.dim_k,
                        "a_width": config.a_width,
                        "b_width": config.b_width,
                    },
                    "metrics": None,
                    "error": str(e),
                })
                print(f"Error: dim={config.dim_m}, width={config.a_width} -> {e}")

    # Save results
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"winograd_sweep_results_{timestamp}.json"

    output_path = Path(__file__).parent / output_file
    with open(output_path, "w") as f:
        json.dump({
            "sweep_config": {
                "dim_vals": dim_vals,
                "width_vals": width_vals,
                "timeout_minutes": timeout_minutes,
                "n_workers": n_workers,
            },
            "results": results,
        }, f, indent=2)

    print("-" * 60)
    print(f"Results saved to: {output_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sweep parameters for Winograd matmul accumulate core")
    parser.add_argument("--test", action="store_true", help="Run with reduced parameters for testing")
    parser.add_argument("--timeout", type=float, default=2.0, help="Timeout in minutes per evaluation")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="winograd_sweep_results.json", help="Output file name")
    args = parser.parse_args()

    if args.test:
        # Reduced parameters for quick testing
        dim_vals = [2, 4]
        width_vals = [4, 8]
        timeout_minutes = 1.0
    else:
        # Full sweep
        dim_vals = [2, 4, 8, 16, 32]
        width_vals = [4, 8, 16, 32]
        timeout_minutes = args.timeout

    n_workers = args.workers if args.workers else min(4, mp.cpu_count())

    results = run_sweep(
        dim_vals=dim_vals,
        width_vals=width_vals,
        timeout_minutes=timeout_minutes,
        n_workers=n_workers,
        output_file=args.output,
    )

    print(f"\nCompleted {len([r for r in results if r.get('metrics') is not None])} / {len(results)} evaluations")


if __name__ == "__main__":
    main()
