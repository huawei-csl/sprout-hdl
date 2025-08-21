
    ew=5
    subnormals = True

    fw = 16 - ew - 1
    m = build_fp_mul_sn("F16Mul", EW=ew, FW=fw, subnormals=subnormals)

    aag = AigerExporter(m).get_aag()
    aag_sym = _get_aag_sym(aag)
    m_right_back = AigerImporter(aag[:-2]+aag_sym).get_sprout_module()