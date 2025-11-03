import os
import tempfile
import time
from low_level_arithmetic.compressor_tree.compressor_tree_sprout_hdl import gen_compressor_tree_graph_and_sprout_module
from sprouthdl.floating_point.sprout_hdl_float_sn import build_fp_mul_sn

from flowy.flows.reinforce.data_collection.lib.definitions import Encodings, Policy, RecipeSelection, SelectionMetric, SimMode, StrategyName, Operations

import flowy.flows.reinforce.run.statistical.run_flows_in_docker as run_flows_in_docker
from flowy.flows.reinforce.analysis.visualize_run import visualize_run
from flowy.data_structures.database import RunDatabase, RunIdentifier, ExperimentIdentifier
from flowy.definitions import DatabaseConfig, ExperimentStages
from flowy.flows.sim.extract_best_design import extract_and_store_best_design
from flowy.flows.sim.visualize_histograms import visualize_histograms
from flowy.flows.reinforce.analysis.visualize_runs import visualize_main   

import argparse

from sprouthdl.helpers import get_aig_stats, get_yosys_metrics, get_yosys_transistor_count
from sprouthdl.sprouthdl_module import Component, Module

def flowy_optimize(m : Module | Component,
                   nb_runs=50,
                   nb_workers=10,
                   iterations=10,
                   mockturtle_chains=1,
                   mockturtle_chain_len=10,
                   mockturtle_chain_workers=1,
                   selection_metric=SelectionMetric.aig_count.value):

      name_initial = m.name
      m.name = "mydesign_comb" # default flowy name
      verilog_code = m.to_verilog()

      # Export to file
      # verilog_path = os.path.join(output_dir, filename)
      
      # not dependent on time
      random_hash = os.urandom(8).hex()
      filename = f"my_logical_design_{random_hash}.v"
      # use tempdir instead
      tempdir = tempfile.gettempdir()
      verilog_path = os.path.join(tempdir, filename)

      # os.makedirs(output_dir, exist_ok=True)
      with open(verilog_path, "w") as f:
           f.write(verilog_code)
      # print(f"Wrote {verilog_path} ({total_bits} bits)")

      datecode = time.strftime("%Y%m%d_%H%M%S")

      # run flowy optimization

      args = run_flows_in_docker.build_parser().parse_args([])

      selection_metric = SelectionMetric.aig_count.value
      experiment = f"exp_{name_initial}_{datecode}"

      args.experiment = experiment
      # add arguments below
      args.nb_runs = nb_runs
      args.nb_workers = nb_workers
      args.iterations = iterations
      args.mockturtle_chains = mockturtle_chains
      args.mockturtle_chain_len = mockturtle_chain_len
      args.mockturtle_chain_workers = mockturtle_chain_workers
      args.recipe_selection = RecipeSelection.PERFORMANCE_SAMPLING.value
      # args.env_option = "auto"
      args.strategy_name = "equal"
      args.debug = False
      args.selection_metric = selection_metric
      # args.input_encoding = Encodings.twos_complement
      # args.output_encoding = Encodings.twos_complement
      args.verilog_file = verilog_path
      args.compression_scripts_per_step = 3
      args.scripts_per_step = 2
      args.simulation_tb = False
      args.extra_files = ""
      args.verbose = False

      # run flowy
      results = run_flows_in_docker.run_with_args(args, commit_hash="2c8627681a246df748be4bf26c4ace4bb55190ce")
      # run_db_identifier = RunIdentifier(experiment_name=expert)
      extract_and_store_best_design(experiment=experiment, target_metrics=[SelectionMetric.aig_count])
      visualize_histograms(experiment=experiment)
      visualize_main(ExperimentIdentifier(root_database=DatabaseConfig.default_path, experiment=experiment))
      
      # delete temp verilog file
      os.remove(verilog_path)
      
      best_design = RunIdentifier(root_database="output/db", experiment=experiment, stage="analysis", run="best_designs")
      best_design_item = RunDatabase(best_design).load("final_mockturtle_design_best_design_aig_count")
      aig_count = best_design_item.get("aig_count").value
      aig_file_path = best_design_item.get("aiger_filepath").path
      aiger_map_file_path = best_design_item.get("aiger_map_filepath").path

      c_out = m.to_component().from_aig_file(aig_file_path, aiger_map_file_path, make_internal=False)
      module = c_out.to_module("optimized_design")

      if isinstance(m, Module):
          return module
      else:
          return c_out

def main():

    output_dir = "testing/floating_point/generated"

    source_design = "fp_multiplier"
    # source_design = "int_multiplier"

    if source_design == "fp_multiplier":
        # floating point multiplier
        # Configuration
        exponent_width = 4
        fraction_width = 3 # M
        subnormals = True

        total_bits = 1 + exponent_width + fraction_width
        filename = f"fxmul_E{exponent_width}_M{fraction_width}_subn{subnormals}.v"

        # Build multiplier
        m = build_fp_mul_sn(
            "mydesign_comb",
            EW=exponent_width,
            FW=fraction_width,
            subnormals=subnormals,
        )
        verilog_code = m.to_verilog()

    elif source_design == "int_multiplier":
        n_bits = 4
        total_bits = n_bits
        g, m = gen_compressor_tree_graph_and_sprout_module(n_bits, policy="wallace")
        m.name = "mydesign_comb"
        verilog_code = m.to_verilog()
        filename = f"compressor_tree_{n_bits}bits.v"

  


    m_optimized = flowy_optimize(m)

    # c = module.to_component()
    
    def get_module_stats(module: Module):
        transistor_count = get_yosys_transistor_count(module, n_iter_optimizations=10)
        yosys_metrics = get_yosys_metrics(module)
        aig_gates = get_aig_stats(module)

        print(f"Design stats: transistor_count={transistor_count}, yosys_metrics={yosys_metrics}, aig_gates={aig_gates}")
        
        return {
            "transistor_count": transistor_count,
            "yosys_metrics": yosys_metrics,
            "aig_gates": aig_gates,
        }

    print("Original design stats:")
    m_orig_stats = get_module_stats(m)
    print("Optimized design stats:")
    m_opt_stats = get_module_stats(m_optimized)

    print(f"Original vs Optimized transistor count: {m_orig_stats['transistor_count']} vs {m_opt_stats['transistor_count']}")
    print(f"Original vs Optimized AIG gates: {m_orig_stats['aig_gates']} vs {m_opt_stats['aig_gates']}")


if __name__ == "__main__":
    main()
