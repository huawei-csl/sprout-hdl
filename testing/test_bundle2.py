# user code

from sprouthdl.sprouthdl import Register
from sprouthdl.sprouthdl_module import Module, UInt
from sprouthdl.bundle2 import Bundle, BundleRegister2 as BundleRegister


class Payload(Bundle):
    opcode: UInt = UInt(3)
    data: UInt = UInt(13)


m = Module("example")

# Create a register that *logically* holds a Payload, but physically UInt(Payload.width())
payload_reg1: BundleRegister =  m.reg(Payload, "payload") #m.reg(Register(Payload)
payload_reg = BundleRegister(Payload, name="payload")

# Access a field as a slice of the underlying signal:
opcode_sig = payload_reg.opcode  # Expr slice
data_sig = payload_reg.data  # Expr slice

payload1 = Payload()
payload1.opcode = 1
payload1.data = 5

# Set next value via a simple mapping:
payload_reg <<= {"opcode": 1, "data": 5}

# Or via another Bundle subclass instance (if you choose to create one):
my_payload = Payload()
my_payload.opcode = 1  # if you want to store Exprs on an instance
my_payload.data = 5
payload_reg <<= my_payload

# Inspect next as dict[field -> Expr]
next_val = payload_reg.next  # either dict or None
bits = payload_reg.to_bits()  # underlying Signal (UInt(width))

print(f"Payload register width: {Payload.width()} bits, with value bits: {bits._driver}")
