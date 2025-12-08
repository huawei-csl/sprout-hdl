from dataclasses import dataclass, field

from sprouthdl.bundle3 import Bundle3
from sprouthdl.sprouthdl import Const, ExprLike, HDLType, Signal, UInt, Wire
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator


def test_bundle_basic():

    @dataclass
    class Bundle_test(Bundle3):
        # use factory to init defaults to avoid shared instances
        a: ExprLike = field(default_factory=lambda: Wire(UInt(3)))  # use wires as default since they are assignable, factory to avoid shared instances
        b: ExprLike = field(default_factory=lambda: Wire(UInt(5)))
        
    b1 = Bundle_test(a=Const(value=2, typ=UInt(3)), b=Const(value=4, typ=UInt(5)))
    b2 = Bundle_test()
    b3 = Bundle_test()

    bits = b1.to_bits()
    b2.from_bits(bits)
    b3 <<= b1

    assert isinstance(b2, Bundle3)
    assert b2.a.typ.width == 3
    assert b2.b.typ.width == 5

    assert Simulator(Module(name="")).peek(b2.a) == 2
    assert Simulator(Module(name="")).peek(b2.b) == 4
    assert b2.width() == 8

    # same for b3
    assert isinstance(b3, Bundle3)
    assert b3.a.typ.width == 3
    assert b3.b.typ.width == 5
    assert Simulator(Module(name="")).peek(b3.a) == 2
    assert Simulator(Module(name="")).peek(b3.b) == 4
    assert b3.width() == 8

if __name__ == "__main__":
    test_bundle_basic()
