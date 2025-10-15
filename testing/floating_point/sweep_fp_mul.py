from aigverse import Aig, DepthAig
from sprouthdl.floating_point.sprout_hdl_float import build_f16_mul, build_fp_mul
from sprouthdl.floating_point.sprout_hdl_float_sn import build_fp_mul_sn
from sprouthdl.sprouthdl_aiger import export_module_to_aiger
from sprouthdl.aigerverse_aag_loader_writer import conv_aag_into_aig, read_aag_into_aig

from aigverse import aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing

def get_size_mult(ew: int, subnormals=False, optim_steps = 200) -> int:
    fw = 16 - ew - 1
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)

    export_module_to_aiger(m, "f16mul.aag")  # writes ASCII AIGER

    aig = Aig()
    aig = read_aag_into_aig("f16mul.aag", aig)
    # aig = read_aiger_into_aig("i10.aig")  # or any AIGER file
    print("nodes:", aig.size(), "PIs:", aig.num_pis(), "POs:", aig.num_pos())

    # Clone the AIG network for size comparison
    aig_clone = aig.clone()

    # Optimize the AIG with several optimization algorithms
    for i in range(optim_steps):
        for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
            optimization(aig)

    # Print the size of the unoptimized and optimized AIGs
    print(f"Original AIG Size:  {aig_clone.size()}")
    print(f"Optimized AIG Size: {aig.size()}")

    return aig.size(), DepthAig(aig).num_levels()


def main():

    # sweep and plot

    import matplotlib.pyplot as plt
    import numpy as np
    ew_values = list(range(1, 14))
    size_depth= [get_size_mult(ew) for ew in ew_values]
    sizes = [size for size, _ in size_depth]
    depths = [depth for _, depth in size_depth]
    # print("EW sizes:", ew_values)
    # print("Sizes:", sizes)

    size_depth_sn = [get_size_mult(ew, subnormals=True) for ew in ew_values]
    sizes_sn = [size for size, _ in size_depth_sn]
    depths_sn = [depth for _, depth in size_depth_sn]
    # print("EW sizes (subnormals):", ew_values)
    # print("Sizes (subnormals):", sizes_sn)

    # plot with two subplots, one below the other
    _, ax = plt.subplots(2, 1, figsize=(7, 7)) #, sharex=True)
    ax[0].plot(ew_values, sizes, marker='o', label='F16Mul (No Subnormals)')
    ax[0].plot(ew_values, sizes_sn, marker='x', label='F16Mul (With Subnormals)')
    ax[0].legend()
    ax[0].set_title('AIG Size vs Exponent Width (EW) for F16Mul')
    ax[0].set_xlabel('Exponent Width (EW)')
    ax[0].set_ylabel('AIG Size')
    ax[0].set_xticks(ew_values)
    ax[0].grid()

    ax[1].plot(ew_values, depths, marker='o', label='F16Mul (No Subnormals)')
    ax[1].plot(ew_values, depths_sn, marker='x', label='F16Mul (With Subnormals)')
    ax[1].legend()
    ax[1].set_title('AIG Depth vs Exponent Width (EW) for F16Mul')
    ax[1].set_xlabel('Exponent Width (EW)')
    ax[1].set_ylabel('AIG Depth')
    ax[1].set_xticks(ew_values)
    ax[1].grid()    

    plt.tight_layout()

    plt.savefig('aig_size_vs_ew.png')
    print('Saved plot to aig_size_vs_ew.png')

if __name__ == "__main__":
    main()