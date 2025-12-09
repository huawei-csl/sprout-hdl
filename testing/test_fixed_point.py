

from sprouthdl.aggregate.aggregate_register import AggregateRegister
from sprouthdl.aggregate.fixed_point import FixedPoint
from sprouthdl.sprouthdl import Expr, Resize, Signal, UInt, Wire, reset_shared_cache


def test_fixedpoint_wire_creation_and_assign():
    reset_shared_cache()

    # Create a 16-bit signed fixed-point with 8 fractional bits
    acc = FixedPoint(total_width=16, frac_width=8, signed=True, name="acc")

    # Underlying bits should be a wire Signal
    assert isinstance(acc.bits, Signal)
    assert acc.bits.kind == "wire"
    assert acc.width == 16
    assert acc.typ.width == 16
    assert acc.int_width == 8  # 16 total - 8 frac

    # Assign an integer to it
    acc @= 42

    # The wire should now have a combinational driver
    assert acc.bits._driver is not None
    assert isinstance(acc.bits._driver, Expr)
    # And that driver should be widened to 16 bits
    assert acc.bits._driver.typ.width == 16

    # Assign an Expr (here, just reusing the current bits)
    acc @= acc.bits
    assert acc.bits._driver.typ.width == 16
    
def test_fixedpoint_view_from_bits():
    reset_shared_cache()

    # A 32-bit bus
    bus = Wire(UInt(32), name="bus")

    # View the lower 16 bits as an unsigned Q8.8 fixed-point
    lower16 = bus[0:16]  # Slice returns [15:0] in Verilog
    val = FixedPoint.from_bits(
        bits=lower16,
        total_width=16,
        frac_width=8,
        signed=False,
        name="bus_lo_as_fp",
    )

    assert val.width == 16
    assert val.bits.typ.width == 16
    assert val.frac_width == 8
    assert val.signed is False

    # This is a *view* backed by a Slice Expr, not a Signal,
    # so assignments should fail at runtime.
    try:
        val @= 0
        assert False, "Expected TypeError when assigning to FixedPoint view backed by non-Signal"
    except TypeError:
        pass
    
def test_aggregate_register_fixedpoint():
    reset_shared_cache()

    # A register that holds a FixedPoint(16, frac=8, signed)
    acc_reg = AggregateRegister(
        FixedPoint,
        total_width=16,
        frac_width=8,
        signed=True,
        name="acc_reg",
    )

    # Raw underlying register bits
    reg_bits = acc_reg.bits
    assert isinstance(reg_bits, Signal)
    assert reg_bits.kind == "reg"
    assert reg_bits.typ.width == 16

    # Structured view
    val_view = acc_reg.value
    assert isinstance(val_view, FixedPoint)
    assert val_view.width == 16
    assert val_view.frac_width == 8
    assert val_view.signed is True

    # Drive next state from an integer
    acc_reg @= 5
    assert reg_bits._driver is not None
    assert isinstance(reg_bits._driver, Expr)
    assert reg_bits._driver.typ.width == 16

    # Drive from another FixedPoint
    src = FixedPoint(16, 8, signed=True, name="src")
    acc_reg @= src
    assert reg_bits._driver.typ.width == 16

    # Drive from a FixedPoint view (from_bits) is also fine;
    # only the *target's* backing matters, not the source.
    bus = Wire(UInt(32), name="bus2")
    view = FixedPoint.from_bits(bus[0:16], 16, 8, signed=True)
    acc_reg @= view
    assert reg_bits._driver.typ.width == 16
    
def build_fixedpoint_mac():
    # Inputs (here as wires for simplicity; in a real Module they’d be ports)
    a = FixedPoint(16, 8, signed=True, name="a")
    b = FixedPoint(16, 8, signed=True, name="b")

    # Multiply at Expr level, using the raw bits
    prod_bits = a.bits * b.bits  # Expr, width = 32

    # Interpret product as Q16.16 for accumulation
    prod_fp = FixedPoint.from_bits(prod_bits, total_width=32, frac_width=16, signed=True)

    # Accumulator register: 32-bit Q16.16
    acc = AggregateRegister(
        FixedPoint,
        total_width=32,
        frac_width=16,
        signed=True,
        name="acc_reg",
    )

    # Next-state: acc_next = acc.value + prod_fp (on Expr layer for now)
    # (With a true FixedPoint arithmetic API you'd do acc.value + prod_fp, but for
    # now we manually use .bits and reinterpret.)
    acc_sum_bits = acc.value.bits + prod_fp.bits  # Expr, 33 bits

    # Truncate/resize back to 32 bits and drive the register
    acc @= Resize(acc_sum_bits, to_width=32)

    return a, b, acc

def test_aggregate_assign_width_mismatch():
    reset_shared_cache()

    fx16 = FixedPoint(16, 8, signed=True, name="fx16")
    fx12 = FixedPoint(12, 4, signed=True, name="fx12")

    assert fx16.width == 16
    assert fx12.width == 12

    try:
        fx16 @= fx12
        assert False, "Expected ValueError on width mismatch"
    except ValueError:
        pass

    # But assigning an int should be widened automatically
    fx16 @= 1
    assert fx16.bits._driver is not None




    
    
if __name__ == "__main__":
    test_fixedpoint_wire_creation_and_assign()
    test_fixedpoint_view_from_bits()
    test_aggregate_register_fixedpoint()
    build_fixedpoint_mac()
    
