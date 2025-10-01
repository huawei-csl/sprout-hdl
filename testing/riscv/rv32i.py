# rv32i_sprout.py
# Minimal RV32I (single-cycle) core in SproutHDL + tiny assembler + simple simulation.

from typing import List, Tuple, Dict

# ---- imports that work with both of your Sprout variants ----

    # package variant
from sprouthdl.sprouthdl_module import Module
from sprouthdl.sprouthdl import UInt, Bool, Const, mux, cat, fit_width

# Simulator import (support both names you used)

from sprouthdl.sprouthdl_simulator import Simulator



# -----------------------
# Helpers: immediates
# -----------------------
def _sext_cat(msb, bits_expr, total_w):
    """Sign-extend 'bits_expr' to 'total_w' via concatenating copies of 'msb'."""
    need = total_w - bits_expr.typ.width
    if need <= 0:
        return bits_expr
    return cat(*([msb] * need), bits_expr)

def imm_i32(instr):
    # I-immediate: [31:20], sign-extended to 32
    imm12 = instr[31:20]
    s = instr[31:31]
    return _sext_cat(s, imm12, 32)

def imm_s32(instr):
    # S-immediate: [31:25] | [11:7], sign-extended to 32
    bits = cat(instr[31:25], instr[11:7])
    s = instr[31:31]
    return _sext_cat(s, bits, 32)

def imm_b32(instr):
    # B-immediate: imm[12|10:5|4:1|11|0]
    #   12 = [31], 11 = [7], 10:5 = [30:25], 4:1 = [11:8], 0 = 0
    s = instr[31:31]
    bits13 = cat(instr[31:31], instr[7:7], instr[30:25], instr[11:8], Const(0, UInt(1)))
    return _sext_cat(s, bits13, 32)


# -----------------------
# Helpers: regfile read
# -----------------------
def read_reg(regs: List, idx_expr):
    """32→1 mux over regs[0..31]."""
    acc = Const(0, UInt(32))
    for i in range(32):
        acc = mux(idx_expr == i, regs[i], acc)
    return acc


# -----------------------
# Core builder
# -----------------------
def build_rv32i_simple(name="RV32I_Simple"):
    """
    Ports:
      in  instr[31:0]            -- instruction at imem_addr (combinational)
      in  dmem_rdata[31:0]       -- read word from data memory @ dmem_addr (combinational)
      out imem_addr[31:0]        -- PC (byte address)
      out dmem_addr[31:0]        -- byte address for load/store
      out dmem_wdata[31:0]       -- store data
      out dmem_we                -- store enable
      out halt                   -- 1 on EBREAK
    """
    m = Module(name, with_clock=True, with_reset=True)

    # ---- Ports ----
    instr       = m.input(UInt(32), "instr")
    dmem_rdata  = m.input(UInt(32), "dmem_rdata")

    imem_addr   = m.output(UInt(32), "imem_addr")
    dmem_addr   = m.output(UInt(32), "dmem_addr")
    dmem_wdata  = m.output(UInt(32), "dmem_wdata")
    dmem_we     = m.output(Bool(),   "dmem_we")
    halt        = m.output(Bool(),   "halt")

    # ---- State ----
    pc = m.reg(UInt(32), "pc", init=0)

    # 32 general-purpose registers (x0 is hard-wired to zero by construction)
    regs = [m.reg(UInt(32), f"x{i}", init=0) for i in range(32)]

    # ---- Decode fields ----
    opcode = instr[6:0]          # [6:0]
    rd     = instr[11:7]         # [11:7]
    funct3 = instr[14:12]        # [14:12]
    rs1    = instr[19:15]        # [19:15]
    rs2    = instr[24:20]        # [24:20]
    funct7 = instr[31:25]        # [31:25]

    # Opcodes
    is_rtype  = (opcode == 0x33)
    is_itype  = (opcode == 0x13)
    is_load   = (opcode == 0x03)  # LW (funct3=010)
    is_store  = (opcode == 0x23)  # SW (funct3=010)
    is_branch = (opcode == 0x63)  # BEQ/BNE
    is_system = (opcode == 0x73)  # EBREAK/ECALL/CSR
    is_ebreak = (instr == 0x00100073)  # exact encoding

    # Read sources
    rs1_val = read_reg(regs, rs1)
    rs2_val = read_reg(regs, rs2)

    # Immediates
    i_imm = imm_i32(instr)
    s_imm = imm_s32(instr)
    b_imm = imm_b32(instr)

    # ALU (minimal ops: ADD, SUB; ADDI)
    is_sub = (is_rtype & (funct3 == 0) & (funct7 == 0x20))
    # operand B: for I-type arith use immediate, else rs2
    opB = mux(is_itype, i_imm, rs2_val)

    add_res = fit_width(rs1_val + opB, UInt(32))
    sub_res = fit_width(rs1_val - rs2_val, UInt(32))
    alu_res = mux(is_sub, sub_res, add_res)

    # Data memory interface
    addr_imm = mux(is_store, s_imm, i_imm)           # S uses s_imm; I (LW) uses i_imm
    daddr    = fit_width(rs1_val + addr_imm, UInt(32))

    dmem_addr <<= daddr
    dmem_wdata <<= rs2_val
    dmem_we <<= is_store

    # Instruction memory address = PC
    imem_addr <<= pc

    # Branch decision (BEQ/BNE only)
    beq = (funct3 == 0) & (rs1_val == rs2_val)
    bne = (funct3 == 1) & (rs1_val != rs2_val)
    take_branch = is_branch & (beq | bne)

    pc_plus4   = fit_width(pc + 4, UInt(32))
    pc_br_tgt  = fit_width(pc + b_imm, UInt(32))
    pc_next    = mux(is_ebreak, pc, mux(take_branch, pc_br_tgt, pc_plus4))
    pc.next    = pc_next

    # Register writeback:
    wb_from_mem = is_load
    wb_data     = mux(wb_from_mem, dmem_rdata, alu_res)
    reg_we      = (is_rtype | is_itype | is_load) & (~is_system)

    for i in range(32):
        if i == 0:
            # x0 is hard-wired zero
            regs[i].next = Const(0, UInt(32))
        else:
            we_i = reg_we & (rd == i)
            regs[i].next = mux(we_i, wb_data, regs[i])

    # Halt signal on EBREAK
    halt <<= is_ebreak

    return m


# -----------------------
# Encoders (machine words)
# -----------------------
def ADDI(rd, rs1, imm):
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x13

def ADD(rd, rs1, rs2):
    return (0x00 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x33

def SUB(rd, rs1, rs2):
    return (0x20 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (rd << 7) | 0x33

def LW(rd, rs1, imm):
    # funct3=010 for LW
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (2 << 12) | (rd << 7) | 0x03

def SW(rs2, rs1, imm):
    # funct3=010 for SW; imm split into [11:5] and [4:0]
    imm12 = imm & 0xFFF
    imm11_5 = (imm12 >> 5) & 0x7F
    imm4_0  = imm12 & 0x1F
    return (imm11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | (imm4_0 << 7) | 0x23

def BEQ(rs1, rs2, bytes_off):
    off = bytes_off
    imm12  = (off >> 12) & 0x1
    imm10_5= (off >> 5) & 0x3F
    imm4_1 = (off >> 1) & 0xF
    imm11  = (off >> 11) & 0x1
    return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (0 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0x63

def BNE(rs1, rs2, bytes_off):
    off = bytes_off
    imm12  = (off >> 12) & 0x1
    imm10_5= (off >> 5) & 0x3F
    imm4_1 = (off >> 1) & 0xF
    imm11  = (off >> 11) & 0x1
    return (imm12 << 31) | (imm10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (1 << 12) | (imm4_1 << 8) | (imm11 << 7) | 0x63

def EBREAK():
    return 0x00100073


# -----------------------
# Tiny assembler
# -----------------------
_REG_ALIAS: Dict[str, int] = {
    # base names
    **{f"x{i}": i for i in range(32)},
    # canonical aliases
    "zero":0,"ra":1,"sp":2,"gp":3,"tp":4,
    "t0":5,"t1":6,"t2":7,"s0":8,"s1":9,
    "a0":10,"a1":11,"a2":12,"a3":13,"a4":14,"a5":15,"a6":16,"a7":17,
    "s2":18,"s3":19,"s4":20,"s5":21,"s6":22,"s7":23,"s8":24,"s9":25,"s10":26,"s11":27,
    "t3":28,"t4":29,"t5":30,"t6":31,
    # extra alias
    "fp":8,
}

def _parse_reg(tok: str) -> int:
    t = tok.strip().lower()
    if t in _REG_ALIAS:
        return _REG_ALIAS[t]
    raise ValueError(f"Unknown register '{tok}'")

def _parse_int(tok: str) -> int:
    tok = tok.strip().lower()
    # allow +/- decimal, hex 0x, binary 0b
    if tok.startswith(("+0x","-0x","0x","+0b","-0b","0b")):
        return int(tok, 0)
    # plain int (decimal)
    return int(tok, 0)

def _strip_comment(line: str) -> str:
    # strip // or # or ; comments
    for sep in ("//", "#", ";"):
        p = line.find(sep)
        if p != -1:
            line = line[:p]
    return line.strip()

def _split_operands(opstr: str) -> List[str]:
    # Split by commas, trim spaces
    # e.g. "x1, x2, 0(x3)" -> ["x1","x2","0(x3)"]
    return [t.strip() for t in opstr.split(",") if t.strip()]

def _parse_mem(operand: str) -> Tuple[int, int]:
    # offset(rs) syntax, e.g. "16(x2)" or "-4(x10)"
    s = operand.replace(" ", "")
    if "(" not in s or not s.endswith(")"):
        raise ValueError(f"Bad memory operand '{operand}', expected off(rs)")
    off_str, rs_str = s.split("(", 1)
    rs_str = rs_str[:-1]
    off = _parse_int(off_str)
    rs = _parse_reg(rs_str)
    return off, rs

def assemble(lines: List[str], *, origin: int = 0) -> List[int]:
    """
    Two-pass assembler for a tiny RV32I subset.
    Returns a list of machine words (ints).
    """
    # Pass 1: collect labels
    labels: Dict[str, int] = {}
    pc = origin
    items: List[Tuple[int, str]] = []  # (pc, normalized_inst)
    for raw in lines:
        s = _strip_comment(raw)
        if not s:
            continue
        while True:
            # support multiple labels on the same line: L1: L2: ADDI x1,x0,1
            if ":" in s:
                head, tail = s.split(":", 1)
                label = head.strip()
                if label:
                    if label in labels:
                        raise ValueError(f"Duplicate label '{label}'")
                    labels[label] = pc
                    s = tail.strip()
                    if not s:
                        break
                    continue
            break
        if s:
            items.append((pc, s))
            pc += 4

    # Pass 2: encode
    words: List[int] = []
    for pc, s in items:
        # separate mnemonic and operands
        parts = s.split(None, 1)
        mnem = parts[0].upper()
        ops  = _split_operands(parts[1]) if len(parts) > 1 else []

        def beq_off(tok: str) -> int:
            # label or immediate (bytes). RISC-V branch adds immediate to current PC.
            if tok in labels:
                return labels[tok] - pc
            v = _parse_int(tok)
            return v

        if mnem == "NOP":
            words.append(ADDI(0, 0, 0))
        elif mnem in ("EBREAK","HALT"):
            words.append(EBREAK())

        elif mnem == "ADDI":
            if len(ops) != 3: raise ValueError("ADDI rd, rs1, imm")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); imm = _parse_int(ops[2])
            if not (-2048 <= imm <= 2047): raise ValueError("ADDI imm out of 12-bit signed range")
            words.append(ADDI(rd, rs1, imm))

        elif mnem == "ADD":
            if len(ops) != 3: raise ValueError("ADD rd, rs1, rs2")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); rs2 = _parse_reg(ops[2])
            words.append(ADD(rd, rs1, rs2))

        elif mnem == "SUB":
            if len(ops) != 3: raise ValueError("SUB rd, rs1, rs2")
            rd  = _parse_reg(ops[0]); rs1 = _parse_reg(ops[1]); rs2 = _parse_reg(ops[2])
            words.append(SUB(rd, rs1, rs2))

        elif mnem == "LW":
            if len(ops) != 2: raise ValueError("LW rd, offset(rs1)")
            rd = _parse_reg(ops[0])
            off, rs1 = _parse_mem(ops[1])
            if not (-2048 <= off <= 2047): raise ValueError("LW offset out of 12-bit signed range")
            words.append(LW(rd, rs1, off))

        elif mnem == "SW":
            if len(ops) != 2: raise ValueError("SW rs2, offset(rs1)")
            rs2 = _parse_reg(ops[0])
            off, rs1 = _parse_mem(ops[1])
            if not (-2048 <= off <= 2047): raise ValueError("SW offset out of 12-bit signed range")
            words.append(SW(rs2, rs1, off))

        elif mnem in ("BEQ","BNE"):
            if len(ops) != 3: raise ValueError(f"{mnem} rs1, rs2, label|imm")
            rs1 = _parse_reg(ops[0]); rs2 = _parse_reg(ops[1]); off = beq_off(ops[2])
            if (off & 0x1) != 0:
                raise ValueError("Branch offset must be 2-byte aligned")
            if not (-4096 <= off <= 4094):
                raise ValueError("Branch offset out of range (~±4KB)")
            words.append(BEQ(rs1, rs2, off) if mnem == "BEQ" else BNE(rs1, rs2, off))

        else:
            raise ValueError(f"Unknown mnemonic '{mnem}' in: {s}")

    return words


# -----------------------
# Simple simulation (now with assembly program)
# -----------------------
def simulate_demo():
    if Simulator is None:
        raise RuntimeError("Simulator class not found. Import your sprout simulator and retry.")

    # Program (assembly):
    prog_asm = [
        "ADDI x1, x0, 2",         # x1 = 2
        "ADDI x2, x0, 3",         # x2 = 3
        "ADD  x3, x1, x2",        # x3 = 5
        "SW   x3, 0(x0)",         # mem[0] = 5
        "LW   x4, 0(x0)",         # x4 = 5
        "BEQ  x3, x4, done",      # if equal, skip next
        "ADDI x5, x0, 99",        # would be skipped
        "done: EBREAK",           # halt
    ]
    prog = assemble(prog_asm)

    # Instantiate core
    m = build_rv32i_simple("RV32I_Simple")
    sim = Simulator(m)

    # Memories
    dmem = bytearray(256)  # tiny data memory
    max_cycles = 100

    # Reset: assert then deassert
    sim.reset(True).eval()
    sim.deassert_reset().eval()

    def imem_fetch(pc_val: int) -> int:
        idx = (pc_val >> 2)
        return prog[idx] if 0 <= idx < len(prog) else 0

    # Run
    print("Assembled program:")
    for i, w in enumerate(prog):
        print(f"  {4*i:04x}: 0x{w:08x}   | {prog_asm[i] if i < len(prog_asm) else ''}")

    for cycle in range(max_cycles):
        # Phase A: drive instr for current PC
        pc_val = sim.get("pc")
        inst = imem_fetch(pc_val)
        sim.set("instr", inst)

        # First eval to compute dmem address/control
        sim.eval()

        # Phase B: drive data memory read combinationally
        addr = sim.get("dmem_addr") & ~0x3
        rword = int.from_bytes(dmem[addr:addr+4], "little")
        sim.set("dmem_rdata", rword)

        # Evaluate again to let LW data propagate into writeback
        sim.eval()

        # Snapshot some signals
        we   = sim.get("dmem_we")
        wdat = sim.get("dmem_wdata")
        halt = sim.get("halt")

        # Rising edge: commit store (use values from this cycle)
        if we:
            dmem[addr:addr+4] = (wdat & 0xFFFFFFFF).to_bytes(4, "little")

        # Advance state
        sim.step(1)

        # Print a brief trace
        x1 = sim.get("x1"); x2 = sim.get("x2"); x3 = sim.get("x3"); x4 = sim.get("x4")
        print(f"t={cycle:02d}  PC=0x{pc_val:08x}  instr=0x{inst:08x}  x1={x1} x2={x2} x3={x3} x4={x4}  we={we} addr={addr} w=0x{wdat&0xffffffff:08x}")
        if halt:
            break

    # Final check: mem[0..3] should hold 5
    final_word = int.from_bytes(dmem[0:4], "little")
    print(f"DMEM[0..3] = {list(dmem[0:4])}  (word={final_word})")
    print("Registers:",\
          {f"x{i}": sim.get(f'x{i}') for i in range(8)})

if __name__ == "__main__":
    simulate_demo()
