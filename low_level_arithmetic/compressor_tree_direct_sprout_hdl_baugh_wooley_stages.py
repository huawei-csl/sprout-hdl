import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple

import numpy as np

from low_level_arithmetic.compressor_tree_sprout_hdl import get_transistor_count_from_m_yosys
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, SInt, Signal, UInt
from sprouthdl.sprouthdl_module import Module
from testing.test_different_logic import run_vectors_io


def half_adder(x: Expr, y: Expr) -> Tuple[Expr, Expr]:
    return x ^ y, x & y  # sum, carry


def full_adder_low_area(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    s1 = x ^ y
    return s1 ^ z, (s1 & z) | (x & y)


def full_adder_fast(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    s = x ^ y ^ z
    return s, (x & y) | (y & z) | (z & x)


class Component(abc.ABC):
    io: dataclass

    @abc.abstractmethod
    def elaborate(self) -> None:
        raise NotImplementedError


@dataclass(frozen=True)
class MultiplierConfig:
    a_width: int
    b_width: int
    signed_a: bool
    signed_b: bool
    optim_type: Literal["area", "speed"]

    @property
    def out_width(self) -> int:
        return self.a_width + self.b_width


class StageBase(abc.ABC):
    def __init__(self, core: "ConfigurableMultiplier") -> None:
        self.core = core


class PartialProductGeneratorBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def generate_columns(self) -> DefaultDict[int, List[Expr]]:
        """Return a mapping from weight to generated partial product bits."""
        raise NotImplementedError


class PartialProductAccumulatorBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        """Reduce columns so that each column contains at most two bits."""
        raise NotImplementedError


class FinalStageAdderBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def resolve(self, columns: Dict[int, List[Expr]]) -> List[Expr]:
        """Return the result bits (LSB first) after the final adder."""
        raise NotImplementedError


class BaughWooleyPartialProductGenerator(PartialProductGeneratorBase):
    def generate_columns(self) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)

        a = self.core.io.a
        b = self.core.io.b
        wa = a.typ.width
        wb = b.typ.width
        out_bits = self.core.io.y.typ.width

        # ordinary partial products
        for i in range(wa - 1):
            for j in range(wb - 1):
                weight = i + j
                if weight >= out_bits:
                    continue
                cols[weight].append(a[i] & b[j])

        # sign row (i = wa-1)
        i = wa - 1
        for j in range(wb - 1):
            cols[i + j].append(~(a[i] & b[j]))

        # sign column (j = wb-1)
        j = wb - 1
        for i in range(wa - 1):
            cols[i + j].append(~(a[i] & b[j]))

        # sign corner (i = wa-1, j = wb-1)
        cols[wa - 1 + wb - 1].append(a[wa - 1] & b[wb - 1])

        # correction bits
        cols[wa - 1 + wb - 1 + 1].append(Const(True, Bool()))
        cols[wa - 1].append(Const(True, Bool()))
        cols[wb - 1].append(Const(True, Bool()))

        total_bits = sum(len(v) for v in cols.values())
        print(
            f"PPG (Baugh-Wooley): generated {total_bits} bits across {len(cols)} columns"
        )
        return cols


class CompressorTreeAccumulator(PartialProductAccumulatorBase):
    def __init__(self, core: "ConfigurableMultiplier") -> None:
        super().__init__(core)
        self._full_adder = (
            full_adder_low_area
            if self.core.config.optim_type == "area"
            else full_adder_fast
        )

    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        cols: DefaultDict[int, List[Expr]] = defaultdict(list)
        for weight, bits in columns.items():
            cols[weight].extend(bits)

        while True:
            next_cols: DefaultDict[int, List[Expr]] = defaultdict(list)
            reduced = True
            for weight in sorted(cols.keys()):
                bits = cols[weight]
                if len(bits) > 2:
                    reduced = False
                    sum_bits, carry_bits = self._compress_column(bits)
                    next_cols[weight].extend(sum_bits)
                    next_cols[weight + 1].extend(carry_bits)
                else:
                    next_cols[weight].extend(bits)
            cols = next_cols
            if reduced:
                break

        print("PPA (Compressor tree): reduced columns to width <= 2")
        return cols

    def _compress_column(self, bits: List[Expr]) -> Tuple[List[Expr], List[Expr]]:
        sum_bits: List[Expr] = []
        carry_bits: List[Expr] = []
        work_bits = list(bits)
        while len(work_bits) >= 3:
            x, y, z = work_bits[:3]
            work_bits = work_bits[3:]
            s, c = self._full_adder(x, y, z)
            sum_bits.append(s)
            carry_bits.append(c)
        if len(work_bits) == 2:
            s, c = half_adder(work_bits[0], work_bits[1])
            sum_bits.append(s)
            carry_bits.append(c)
        elif len(work_bits) == 1:
            sum_bits.append(work_bits[0])
        return sum_bits, carry_bits


class RippleCarryFinalAdder(FinalStageAdderBase):
    def resolve(self, columns: Dict[int, List[Expr]]) -> List[Expr]:
        max_weight = self.core.io.y.typ.width
        result_bits: List[Expr] = []
        carry: Optional[Expr] = None

        for weight in range(max_weight):
            bits = list(columns.get(weight, []))
            if carry is not None:
                bits.append(carry)

            if len(bits) == 0:
                result_bits.append(Const(False, Bool()))
                carry = None
            elif len(bits) == 1:
                result_bits.append(bits[0])
                carry = None
            elif len(bits) == 2:
                s, carry = half_adder(bits[0], bits[1])
                result_bits.append(s)
            elif len(bits) == 3:
                s, carry = full_adder_fast(bits[0], bits[1], bits[2])
                result_bits.append(s)
            else:
                raise ValueError(
                    f"Unexpected number of bits ({len(bits)}) in column {weight} during final addition"
                )

        if carry is not None:
            result_bits.append(carry)

        print(f"FSA (Ripple-carry): produced {len(result_bits)} result bits")
        return result_bits


class ConfigurableMultiplier(Component):
    def __init__(
        self,
        a_w: int,
        b_w: int,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: Literal["area", "speed"] = "area",
        ppg_cls: type[PartialProductGeneratorBase] = BaughWooleyPartialProductGenerator,
        ppa_cls: type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: type[FinalStageAdderBase] = RippleCarryFinalAdder,
    ) -> None:
        self.config = MultiplierConfig(a_w, b_w, signed_a, signed_b, optim_type)

        self.ppg_cls = ppg_cls
        self.ppa_cls = ppa_cls
        self.fsa_cls = fsa_cls

        @dataclass
        class IO:
            a: Signal
            b: Signal
            y: Signal

        base_typ_a = SInt if signed_a else UInt
        base_typ_b = SInt if signed_b else UInt
        base_type_y = SInt if (signed_a or signed_b) else UInt

        self.io: IO = IO(
            a=Signal(name="a", typ=base_typ_a(a_w), kind="input"),
            b=Signal(name="b", typ=base_typ_b(b_w), kind="input"),
            y=Signal(name="y", typ=base_type_y(self.config.out_width), kind="output"),
        )

        self.ppg = self.ppg_cls(self)
        self.ppa = self.ppa_cls(self)
        self.fsa = self.fsa_cls(self)

        self.elaborate()

    def elaborate(self) -> None:
        columns = self.ppg.generate_columns()
        reduced_columns = self.ppa.accumulate(columns)
        result_bits = self.fsa.resolve(reduced_columns)
        expected_width = self.io.y.typ.width
        self.io.y <<= Concat(reversed(result_bits[:expected_width]))

        print(
            f"MultiplierCompressorTree: {self.config.a_width}x{self.config.b_width} -> {expected_width} bits"
        )


class MultiplierTestVectors:
    def __init__(
        self,
        a_w: int,
        b_w: int,
        num_vectors: int = 64,
        tb_sigma: Optional[float] = None,
        signed_a: bool = False,
        signed_b: bool = False,
    ) -> None:
        self.a_w = a_w
        self.b_w = b_w
        self.y_w = a_w + b_w
        self.num_vectors = num_vectors
        self.tb_sigma = tb_sigma
        self.a_signed = signed_a
        self.b_signed = signed_b

    def generate(self) -> Tuple:
        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:

                def rand_gen_unsigned(n: int) -> int:
                    return int(np.round(np.random.normal((1 << (n - 1)), self.tb_sigma)))

                def rand_gen_signed(n: int) -> int:
                    return int(np.round(np.random.normal(0, self.tb_sigma)))

                va = rand_gen_signed(self.a_w) if self.a_signed else rand_gen_unsigned(self.a_w)
                vb = rand_gen_signed(self.b_w) if self.b_signed else rand_gen_unsigned(self.b_w)

                def clamp_unsigned(v: int, n: int) -> int:
                    return max(min(v, (1 << n) - 1), 0)

                def clamp_signed(v: int, n: int) -> int:
                    return max(min(v, (1 << (n - 1)) - 1), -(1 << (n - 1)))

                va = clamp_signed(va, self.a_w) if self.a_signed else clamp_unsigned(va, self.a_w)
                vb = clamp_signed(vb, self.b_w) if self.b_signed else clamp_unsigned(vb, self.b_w)
            else:
                va = np.random.randint(-(1 << (self.a_w - 1)), 1 << (self.a_w - 1)) if self.a_signed else np.random.randint(0, 1 << self.a_w)
                vb = np.random.randint(-(1 << (self.b_w - 1)), 1 << (self.b_w - 1)) if self.b_signed else np.random.randint(0, 1 << self.b_w)

            vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": va * vb}))

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.y_w)}
        return spec, vecs, None


def gen_sprout_module(class_instance: ConfigurableMultiplier) -> Module:
    m = Module(f"Mul{class_instance.config.a_width}_ct", with_clock=False, with_reset=False)
    for sig in class_instance.io.__dict__.values():
        if sig.kind == "input":
            m.add_input(sig)
        elif sig.kind == "output":
            m.add_output(sig)
        else:
            raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
    return m


def gen_spec(class_instance: Component) -> Dict[str, UInt]:
    spec: Dict[str, UInt] = {}
    for sig in class_instance.io.__dict__.values():
        spec[sig.name] = sig.typ
    return spec


def main() -> None:
    n_bits = 16
    signed = True

    mult = ConfigurableMultiplier(
        a_w=n_bits,
        b_w=n_bits,
        signed_a=signed,
        signed_b=signed,
        optim_type="area",
    )

    module = gen_sprout_module(mult)
    transistor_count = get_transistor_count_from_m_yosys(module, n_iter_optimizations=10)
    print(f"Yosys-reported transistor count: {transistor_count}")

    specs, vecs, decoder = MultiplierTestVectors(
        a_w=n_bits,
        b_w=n_bits,
        num_vectors=16,
        tb_sigma=None,
        signed_a=signed,
        signed_b=signed,
    ).generate()
    _ = specs  # unused in this smoke test but returned for completeness

    run_vectors_io(module, vecs, decoder=decoder)


if __name__ == "__main__":
    main()
