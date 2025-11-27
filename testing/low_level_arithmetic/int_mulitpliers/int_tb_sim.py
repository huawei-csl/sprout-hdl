from sprouthdl.arithmetic.int_multipliers.eval.multiplier_stage_options_demo_lib import FSAOption, MultiplierEncodings, PPAOption, PPGOption
from sprouthdl.arithmetic.int_multipliers.eval.testvector_generation import Encoding, MultiplierTestVectors
from sprouthdl.arithmetic.int_multipliers.multipliers.multiplier_stage_core import MultiplierTestVectorsInt, StageBasedMultiplierBasic
from sprouthdl.helpers import run_vectors, run_vectors_on_simulator
from sprouthdl.sprouthdl_simulator import Simulator
from sprouthdl.sprouthdl_verilog_testbench import VerilogTestbenchSimulator


def int_tb_sim():
    n_bits = 4
    signed = False  

    mult = StageBasedMultiplierBasic(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
        ppg_cls=PPGOption.AND.value,
        ppa_cls=PPAOption.WALLACE_TREE.value,
        fsa_cls=FSAOption.RIPPLE.value,
    )
    module = mult.to_module(f"Mul{n_bits}")

    specs, vecs, decoder = MultiplierTestVectorsInt(
        a_w=n_bits,
        b_w=n_bits,
        num_vectors=16,
        tb_sigma=None,
        signed_a=signed,
        signed_b=signed,
    ).generate()

    encodings = MultiplierEncodings.with_enc(Encoding.unsigned)
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

    run_vectors(module, vecs, print_on_pass=True, use_signed=True)
    
    sim = Simulator(module)
    sim.state_logging = True
    run_vectors_on_simulator(sim, vecs, decoder=decoder, use_signed=True, print_on_pass=True)
    
    print("All test vectors passed!")

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
