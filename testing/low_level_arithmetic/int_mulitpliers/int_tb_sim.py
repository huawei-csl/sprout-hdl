from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, MultiplierEncodings, PPAOption, PPGOption
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import MultiplierTestVectorsInt, StageBasedMultiplierBasic
from sprouthdl.helpers import run_vectors, run_vectors_on_simulator
from sprouthdl.sprouthdl_simulator import Simulator
from sprouthdl.sprouthdl_verilog_testbench import VerilogTestbenchSimulator
from sprouthdl.various.vcd_writer import write_vcd


def int_tb_sim():
    n_bits = 16
    signed = False  

    mult = StageBasedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        ppg_cls=PPGOption.BOOTH_OPTIMISED.value,  # PPGOption.AND.value,
        ppa_cls=PPAOption.CARRY_SAVE_TREE.value, #PPAOption.WALLACE_TREE.value,
        fsa_cls=FSAOption.PREFIX_MULTI_SCAN.value,  # FSAOption.RIPPLE.value,
    )
    module = mult.to_module(f"Mul{n_bits}")

    # specs, vecs, decoder = MultiplierTestVectorsInt(
    #     a_w=n_bits,
    #     b_w=n_bits,
    #     num_vectors=16,
    #     tb_sigma=None,
    #     signed_a=signed,
    #     signed_b=signed,
    # ).generate()

    decoder = None

    encodings = MultiplierEncodings.with_enc(Encoding.unsigned if not signed else Encoding.twos_complement,)
    vecs = MultiplierTestVectors(
                a_w=n_bits,
                b_w=n_bits,
                y_w=mult.io.y.typ.width,
                num_vectors=16,
                tb_sigma=None,
                a_encoding=encodings.a,
                b_encoding=encodings.b,
                y_encoding=encodings.y,
            ).generate()

    use_signed = False

    run_vectors(module, vecs, print_on_pass=True, use_signed=use_signed)

    sim = Simulator(module)
    sim.trace_enabled = True
    run_vectors_on_simulator(sim, vecs, decoder=decoder, use_signed=use_signed, print_on_pass=True)

    trace_history = sim.trace_history
    trace_names = sim.get_traced_expr_names()
    # print(f"Traced signals: {list(trace_names.values())}")

    vcd_filename = "int_multiplier_tb_sim.vcd"
    write_vcd(trace_history=trace_history, 
              trace_names=trace_names, 
              filename=vcd_filename,
              top_module=module.name, 
              timescale="1ns")

    sim_tb = VerilogTestbenchSimulator(module)
    run_vectors_on_simulator(sim_tb, vecs, decoder=decoder, use_signed=use_signed, print_on_pass=False)

    print("\n".join(sim_tb.to_testbench_lines()))

    tb_filename = "int_multiplier_tb_sim.v"
    verilog_filename = "int_multiplier.v"
    sim_tb.to_testbench_file(tb_filename, tb_module_name=module.name+"_tb")
    module.to_verilog_file(verilog_filename)

    # tb = VerilogTestbenchSimulator.from_multiplier_module(
    #     module,
    #     specs,
    #     vecs,
    #     decoder,
    #     clock_period=10.0,
    #     eval_delay=1.0,
    # )

    # tb.reset(True)
    # tb.step()  # capture reset behaviour
    # tb.deassert_reset()

    # for vec in vecs:
    #     tb.set_inputs_from_vector(vec)
    #     tb.eval()
    #     tb.step()
    #     tb.check_outputs_against_vector(vec)

    # print("All test vectors passed!")

    return

if __name__ == "__main__":
    int_tb_sim()
