import os
import time
from low_level_arithmetic.compressor_tree.compressor_tree_sprout_hdl import gen_compressor_tree_graph_and_sprout_module
from sprouthdl.floating_point.sprout_hdl_float_sn import build_fp_mul_sn

from flowy.flows.reinforce.data_collection.lib.definitions import Encodings, Policy, RecipeSelection, SelectionMetric, SimMode, StrategyName, Operations
from flowy.flows.reinforce.run.statistical.run_flow import run_flow
import flowy.flows.reinforce.run.statistical.run_flows_in_docker as run_flows_in_docker
from flowy.flows.reinforce.analysis.visualize_run import visualize_run
from flowy.data_structures.database import RunDatabase, RunIdentifier, ExperimentIdentifier
from flowy.definitions import DatabaseConfig, ExperimentStages
from flowy.flows.sim.extract_best_design import extract_and_store_best_design
from flowy.flows.sim.visualize_histograms import visualize_histograms
from flowy.flows.reinforce.analysis.visualize_runs import visualize_main   

import argparse

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
        fp_multiplier = build_fp_mul_sn(
            "mydesign_comb",
            EW=exponent_width,
            FW=fraction_width,
            subnormals=subnormals,
        )
        verilog_code = fp_multiplier.to_verilog()

    elif source_design == "int_multiplier":
        n_bits = 4
        total_bits = n_bits
        g, m = gen_compressor_tree_graph_and_sprout_module(n_bits, policy="wallace")
        m.name = "mydesign_comb"
        verilog_code = m.to_verilog()
        filename = f"compressor_tree_{n_bits}bits.v"

    # Export to file
    verilog_path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(verilog_path, "w") as f:
        f.write(verilog_code)
    print(f"Wrote {verilog_path} ({total_bits} bits)")

    datecode = time.strftime("%Y%m%d_%H%M%S")

    # run flowy optimization

    args = run_flows_in_docker.build_parser().parse_args([])

    selection_metric = SelectionMetric.aig_count.value
    experiment = f"exp_test_3_{datecode}"

    args.experiment = experiment
    # add arguments below
    args.nb_runs = 50
    args.nb_workers = 50
    args.iterations = 10
    args.mockturtle_chains = 1
    args.mockturtle_chain_len = 10
    args.mockturtle_chain_workers = args.mockturtle_chains
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

    run = True
    if run:

        results = run_flows_in_docker.run_with_args(args, commit_hash="2c8627681a246df748be4bf26c4ace4bb55190ce")

        # run_db_identifier = RunIdentifier(experiment_name=expert)
        extract_and_store_best_design(experiment=experiment, target_metrics=[SelectionMetric.aig_count])
        visualize_histograms(experiment=experiment)
        visualize_main(ExperimentIdentifier(root_database=DatabaseConfig.default_path, experiment=experiment))

        print("done")

    experiment = "exp_test_3_20251029_133739"
    best_design = RunIdentifier(root_database="output/db", experiment=experiment, stage="analysis", run="best_designs")

    best_design_item = RunDatabase(best_design).load("final_mockturtle_design_best_design_aig_count")
    aig_count = best_design_item.get("aig_count").value
    aig_file_path = best_design_item.get("aiger_filepath").path
    aiger_map_file_path = best_design_item.get("aiger_map_filepath").path

    print(f"finished experiment {experiment}, best design stored in {best_design}")

    # visualize runs
    # visualize_histograms
    # extract_best

    # visualize_run(db_identifier)

    # load final best design which was loaded into the analysis db by extract_best

    # extract number of transistors
    # run_db = RunDatabase(db_identifier)
    # load final best design
    # run_data = run_db.load('final_mockturtle_design')

    # print(f"Final best design, nb_transistors: {run_data['nb_transistors'].value}")


if __name__ == "__main__":
    main()
