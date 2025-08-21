from aigverse import Aig
from sprout_hdl_float import build_f16_mul  # or your own module builder
from sprout_hdl_aiger import export_module_to_aiger
from hdl_test.aag_loader_writer import conv_aag_into_aig

from aigverse import aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing

m = build_f16_mul("F16Mul")
export_module_to_aiger(m, "f16mul.aag")  # writes ASCII AIGER

# Later, in Python (with aigverse installed):

aig = Aig()
aig = conv_aag_into_aig("f16mul.aag", aig)
# aig = read_aiger_into_aig("i10.aig")  # or any AIGER file
print("nodes:", aig.size(), "PIs:", aig.num_pis(), "POs:", aig.num_pos())

# Clone the AIG network for size comparison
aig_clone = aig.clone()

# Optimize the AIG with several optimization algorithms
n_iter_optimizations = 10
for i in range(n_iter_optimizations):
    for optimization in [aig_resubstitution, sop_refactoring, aig_cut_rewriting, balancing]:
        optimization(aig)


# Print the size of the unoptimized and optimized AIGs
print(f"Original AIG Size:  {aig_clone.size()}")
print(f"Optimized AIG Size: {aig.size()}")
