from sprout_hdl_module import Module
from sprout_hdl import UInt

mac = Module("Mac32", with_clock=True, with_reset=True)
a = mac.input(UInt(16), "a")
b = mac.input(UInt(16), "b")
acc_out = mac.output(UInt(32), "acc_out")

acc = mac.reg(UInt(32), "acc", init=0)
acc.next = acc + (a * b)
acc_out <<= acc

print(mac.to_verilog())
