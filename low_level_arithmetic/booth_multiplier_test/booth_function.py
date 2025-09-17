def booth_radix4(a: int, b: int, width: int, trace: bool = False) -> int:
    """
    Radix-4 Booth multiplication of signed two's-complement integers.

    Args:
        a: multiplicand (interpreted as signed 'width'-bit two's complement)
        b: multiplier   (interpreted as signed 'width'-bit two's complement)
        width: bit-width of inputs a and b
        trace: if True, prints per-group decisions

    Returns:
        Signed product as a Python int, interpreted in 2*width bits.
    """
    def to_signed(x, w):
        x &= (1 << w) - 1
        return x - (1 << w) if (x >> (w - 1)) & 1 else x

    def bit(xu, k):
        return (xu >> k) & 1

    # Interpret inputs as width-bit signed, but keep also unsigned views for bit access
    a_s = to_signed(a, width)
    b_u = b & ((1 << width) - 1)

    # Sign bit of multiplier for sign-extension above MSB
    msb_b = (b_u >> (width - 1)) & 1

    # Helper to read sign-extended multiplier bits with an implicit y[-1] = 0
    def y(k):
        if k < 0:
            return 0
        if k >= width:
            return msb_b  # sign-extend
        return bit(b_u, k)

    # Radix-4 Booth recode table:
    # (y_{2i+1} y_{2i} y_{2i-1}) -> coeff in {-2,-1,0,+1,+2}
    def booth_coeff(b2, b1, b0):
        code = (b2 << 2) | (b1 << 1) | b0
        if code in (0b000, 0b111): return 0
        if code in (0b001, 0b010): return +1
        if code == 0b011:          return +2
        if code == 0b100:          return -2
        if code in (0b101, 0b110): return -1
        raise RuntimeError("Invalid Booth code")

    acc = 0  # we will accumulate into an (unbounded) Python int, then truncate
    n_groups = (width + 2) // 2  # = ceil((width+1)/2)

    for i in range(n_groups):
        b2, b1, b0 = y(2*i + 1), y(2*i), y(2*i - 1)
        c = booth_coeff(b2, b1, b0)  # -2, -1, 0, +1, +2
        if c != 0:
            term = (c * a_s) << (2 * i)
            acc += term
            if trace:
                print(f"i={i:2d} patt={b2}{b1}{b0} coeff={c:+d} "
                      f"add ( {c:+d} * a ) << {2*i} = {term}")
        elif trace:
            print(f"i={i:2d} patt={b2}{b1}{b0} coeff= 0  (skip)")

    # Truncate to 2*width bits, then return as signed
    mask2 = (1 << (2 * width)) - 1
    acc_u = acc & mask2
    if (acc_u >> (2 * width - 1)) & 1:  # sign bit set
        return acc_u - (1 << (2 * width))
    else:
        return acc_u


# ---- quick sanity checks ----
if __name__ == "__main__":
    for w in (4, 8, 12):
        for a in range(-(1<<(w-1)), 1<<(w-1)):
            for b in range(-(1<<(w-1)), 1<<(w-1), max(1, (1<<w)//7)):  # sample a few b's
                p = booth_radix4(a, b, w)
                assert p == a*b or (p & ((1<<(2*w))-1)) == (a*b) & ((1<<(2*w))-1)
    # Example with trace
    print("Example:")
    A, B, W = -13, 11, 6
    print(f"{A} * {B} = {booth_radix4(A,B,W,trace=True)} (python: {A*B})")
