from aigverse import Aig, DepthAig
from sprouthdl.sprout_hdl import UInt
from sprouthdl.floating_point.sprout_hdl_float import build_f16_mul, build_f16_vectors, build_fp_mul, half_to_float, run_vectors
from sprout_hdl_float_bk import build_f16_subnormal_vectors
from sprouthdl.floating_point.sprout_hdl_float_sn import build_f16_subnormal_ext_vectors, build_fp_mul_sn
from sprouthdl.sprout_hdl_aiger import AigerExporter, AigerImporter, export_module_to_aiger
from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, conv_aag_into_aig, conv_aig_into_aag, read_aag_into_aig

from aigverse import aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing

from sprouthdl.sprout_io_collector import IOCollector

def get_size_mult(ew: int, subnormals=False) -> int:

    ew=5
    subnormals = True

    fw = 16 - ew - 1
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)

    aag = AigerExporter(m).get_aag()
    aag_sym = _get_aag_sym(aag)
    m_right_back = AigerImporter(aag[:-2]+aag_sym).get_sprout_module()

    collector = IOCollector()
    sprout_collected = collector.group(
        m_right_back,
        {
            "a": UInt(16),
            "b": UInt(16),
            "y": UInt(16),
        },
    )

    aig = conv_aag_into_aig(aag, Aig())

    aag_bak = conv_aig_into_aag(aig)
    aig_bak = conv_aag_into_aig(aag_bak, Aig())

    # assert equivalence
    from aigverse import equivalence_checking
    assert equivalence_checking(aig, aig_bak), "AIGs are not equivalent after conversion!"

    # aig = read_aiger_into_aig("i10.aig")  # or any AIGER file
    print("nodes:", aig.size(), "PIs:", aig.num_pis(), "POs:", aig.num_pos())

    # Clone the AIG network for size comparison
    aig_clone = aig.clone()

    # Optimize the AIG with several optimization algorithms
    # n_iter_optimizations = 100
    # for i in range(n_iter_optimizations):
    #     for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
    #         optimization(aig)

    # Print the size of the unoptimized and optimized AIGs
    print(f"Original AIG Size:  {aig_clone.size()}")
    print(f"Optimized AIG Size: {aig.size()}")

    aag = conv_aig_into_aag(aig)

    sprout = AigerImporter(aag[:-2]+aag_sym).get_sprout_module()

    collector = IOCollector()
    sprout_collected = collector.group(sprout, {
        "a": UInt(16),
        "b": UInt(16),
        "y": UInt(16),
    })

    aag2 = AigerExporter(sprout).get_aag()
    aig2 = conv_aag_into_aig(aag2, Aig())

    from aigverse import equivalence_checking
    assert equivalence_checking(aig_clone, aig2), "AIGs are not equivalent after conversion!"

    run_vectors(m, build_f16_vectors(), label="float16 default cases", decoder=half_to_float)
    run_vectors(m, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=half_to_float)

    run_vectors(sprout, build_f16_vectors(), label="float16 default cases", decoder=half_to_float)
    run_vectors(sprout, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=half_to_float)
    run_vectors(sprout, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=half_to_float)

    return aig.size(), DepthAig(aig).num_levels()


def main():
    
    # sweep and plot
    
    import matplotlib.pyplot as plt
    import numpy as np
    ew_values = list(range(1, 14))
    size_depth= [get_size_mult(ew) for ew in ew_values]
    sizes = [size for size, _ in size_depth]
    depths = [depth for _, depth in size_depth]
    #print("EW sizes:", ew_values)
    #print("Sizes:", sizes)
    
    #sizes_sn = [get_size_mult(ew, subnormals=True) for ew in ew_values]
    #print("EW sizes (subnormals):", ew_values)
    #print("Sizes (subnormals):", sizes_sn)
    
    plt.figure(figsize=(10, 6))
    plt.plot(ew_values, sizes, marker='o', label='F16Mul (No Subnormals)')
    #plt.plot(ew_values, sizes_sn, marker='x', label='F16Mul (With Subnormals)')
    plt.legend()
    plt.title('AIG Size vs Exponent Width (EW) for F16Mul')
    plt.xlabel('Exponent Width (EW)')
    plt.ylabel('AIG Size')
    plt.xticks(ew_values)
    plt.grid()
    plt.savefig('aig_size_vs_ew.png')
    print('Saved plot to aig_size_vs_ew.png')

if __name__ == "__main__":
    main()