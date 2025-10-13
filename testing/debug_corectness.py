from aigverse import equivalence_checking
from sprouthdl.aigerverse_aag_loader_writer import _get_aag_sym, conv_aag_into_aig
from sprouthdl.sprouthdl import UInt
from sprouthdl.sprouthdl_aiger import AigerExporter, AigerImporter
from sprouthdl.floating_point.sprout_hdl_float import half_to_float, run_vectors
from sprouthdl.floating_point.sprout_hdl_float_sn import build_f16_subnormal_ext_vectors, build_f16_subnormal_vectors, build_fp_mul_sn
from sprouthdl.sprouthdl_module import IOCollector


def main():
    ew=5
    subnormals = True

    fw = 16 - ew - 1
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)
    # m3 = build_fp_mul_sn("F16Mul3", EW=ew+1, FW=fw-1, subnormals=subnormals)

    aag = AigerExporter(m).get_aag()
    aig = conv_aag_into_aig(aag)

    aag_sym = _get_aag_sym(aag)
    m2 = AigerImporter(aag[:-2]+aag_sym).get_sprout_module()

    aag2 = AigerExporter(m2).get_aag()
    aig2 = conv_aag_into_aig(aag2)

    assert equivalence_checking(aig, aig2), "AIGs are not equivalent after conversion!"

    collector = IOCollector()
    collector.group(m2, {
        "a": UInt(16),
        "b": UInt(16),
        "y": UInt(16),
    })

    run_vectors(m, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=half_to_float)
    run_vectors(m, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=half_to_float)

    run_vectors(m2, build_f16_subnormal_vectors(), label="float16 subnormal cases", decoder=half_to_float)
    run_vectors(m2, build_f16_subnormal_ext_vectors(), label="float16 subnormal ext cases", decoder=half_to_float)


if __name__ == "__main__":
    main()