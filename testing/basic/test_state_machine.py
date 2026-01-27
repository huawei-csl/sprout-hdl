from sprouthdl.sprouthdl import Bool, UInt
from sprouthdl.sprouthdl_state import State
from sprouthdl.sprouthdl_control_strutures import case_, default, if_, switch_
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl_simulator import Simulator, _sid


def test_state_binary_encoding():
    fsm = State("IDLE", "RUN", "DONE", encoding="binary")
    assert fsm._width == 2
    assert fsm.IDLE.value == 0
    assert fsm.RUN.value == 1
    assert fsm.DONE.value == 2
    assert len(fsm) == 3


def test_state_onehot_encoding():
    fsm = State("A", "B", "C", "D", encoding="onehot")
    assert fsm._width == 4
    assert fsm.A.value == 0b0001
    assert fsm.B.value == 0b0010
    assert fsm.C.value == 0b0100
    assert fsm.D.value == 0b1000


def test_state_gray_encoding():
    fsm = State("S0", "S1", "S2", "S3", encoding="gray")
    assert fsm._width == 2
    assert fsm.S0.value == 0b00
    assert fsm.S1.value == 0b01
    assert fsm.S2.value == 0b11
    assert fsm.S3.value == 0b10


def test_state_machine_simulation():
    """Simple 3-state FSM: IDLE -> RUN -> DONE -> IDLE, controlled by 'go' input."""
    fsm = State("IDLE", "RUN", "DONE")

    m = Module("FSM", with_clock=True, with_reset=False)
    go = m.input(Bool(), "go")
    state = m.reg(fsm.typ, "state", init=fsm.IDLE)
    out = m.output(UInt(2), "out")

    out <<= 0

    with switch_(state):
        with case_(fsm.IDLE):
            out <<= 0
            with if_(go):
                state <<= fsm.RUN
        with case_(fsm.RUN):
            out <<= 1
            state <<= fsm.DONE
        with case_(fsm.DONE):
            out <<= 2
            state <<= fsm.IDLE
        with default():
            state <<= fsm.IDLE

    sim = Simulator(m)
    sim.eval()

    # Initial state: IDLE, out=0
    assert sim.get("state") == fsm.IDLE.value
    assert sim.get("out") == 0

    # No go signal: stays in IDLE
    sim.set("go", 0)
    ns = sim._compute_next_state()
    assert ns[_sid(state)] == fsm.IDLE.value

    # Assert go: next state is RUN
    sim.set("go", 1)
    ns = sim._compute_next_state()
    assert ns[_sid(state)] == fsm.RUN.value

    # Advance to RUN
    sim.set("state", fsm.RUN.value)
    sim.eval()
    assert sim.get("out") == 1
    ns = sim._compute_next_state()
    assert ns[_sid(state)] == fsm.DONE.value

    # Advance to DONE
    sim.set("state", fsm.DONE.value)
    sim.eval()
    assert sim.get("out") == 2
    ns = sim._compute_next_state()
    assert ns[_sid(state)] == fsm.IDLE.value

if __name__ == "__main__":
    print("Running tests...")
    test_state_machine_simulation()
    test_state_binary_encoding()
    test_state_onehot_encoding()
    test_state_gray_encoding()
    test_state_machine_simulation()
        
