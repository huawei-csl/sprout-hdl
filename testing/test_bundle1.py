from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest

from sprouthdl.bundle1 import BundleRegister, BundleValue, bundle
from sprouthdl.sprouthdl import UInt
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator


def test_bundle_to_from_bits_roundtrip():
    spec = bundle({"a": UInt(3), "b": UInt(5)})
    value = spec(a=1, b=7)
    bits = value.to_bits()

    assert bits.typ.width == spec.width

    unpacked = spec.from_bits(bits)
    assert isinstance(unpacked, BundleValue)
    assert unpacked["a"].typ.width == 3
    assert unpacked["b"].typ.width == 5


def test_bundle_register_integration():
    spec = bundle({"a": UInt(3), "b": UInt(5)})
    m = Module("bundle_reg_test")

    reg = m.reg(spec, "state")
    assert isinstance(reg, BundleRegister)

    reg.next = spec(a=1, b=2)

    a_out = m.output(UInt(3), "a_out")
    b_out = m.output(UInt(5), "b_out")
    a_out <<= reg.a
    b_out <<= reg.b

    sim = Simulator(m)
    sim.reset()
    sim.deassert_reset()
    sim.step()

    assert sim.get("a_out") == 1
    assert sim.get("b_out") == 2


def test_bundle_from_class_factory():
    class PayloadSpec:
        opcode = UInt(3)
        data = UInt(5)
        label = "ignored"

    spec = bundle.from_class(PayloadSpec)

    assert isinstance(spec, bundle)
    assert spec.name == "PayloadSpec"
    value = spec(opcode=1, data=2)
    assert value.opcode.typ.width == 3
    assert value.data.typ.width == 5


def test_bundle_define_decorator():
    @bundle.define
    class DecoratedSpec:
        opcode = UInt(2)
        immediate = UInt(6)

    assert isinstance(DecoratedSpec, bundle)
    assert DecoratedSpec.definition.__name__ == "DecoratedSpec"

    instance = DecoratedSpec(opcode=1, immediate=3)
    bits = DecoratedSpec.to_bits(instance)
    unpacked = DecoratedSpec.from_bits(bits)
    assert unpacked.opcode.typ.width == 2
    assert unpacked.immediate.typ.width == 6


def test_bundle_define_requires_fields():
    class EmptySpec:
        pass

    with pytest.raises(ValueError):
        bundle.from_class(EmptySpec)

