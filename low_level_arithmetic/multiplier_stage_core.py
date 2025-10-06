from __future__ import annotations

import abc
from collections import defaultdict
from dataclasses import dataclass
from typing import ClassVar, DefaultDict, Dict, List, Literal, Optional, Self, Tuple, Type

import numpy as np

from low_level_arithmetic.test_vector_generation import Encoding
from sprouthdl.sprouthdl import Bool, Concat, Const, Expr, Signal, SInt, UInt, mux
from sprouthdl.sprouthdl_module import Module


# ---- common arithmetic helpers -------------------------------------------------

def half_adder(x: Expr, y: Expr) -> Tuple[Expr, Expr]:
    return x ^ y, x & y  # sum, carry


def full_adder_low_area(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    s1 = x ^ y
    return s1 ^ z, (s1 & z) | (x & y)


def full_adder_fast(x: Expr, y: Expr, z: Expr) -> Tuple[Expr, Expr]:
    s = x ^ y ^ z
    return s, (x & y) | (y & z) | (z & x)


# ---- abstract component/stage definitions --------------------------------------


class Component(abc.ABC):
    io: dataclass

    @abc.abstractmethod
    def elaborate(self) -> None:  # pragma: no cover - structural hook
        raise NotImplementedError
    
    # convenience helpers -------------------------------------------------------
    
    def to_module(self, name: Optional[str] = None) -> Module:
        module = Module(
            name or f"Mul{self.config.a_width}x{self.config.b_width}_ct",
            with_clock=False,
            with_reset=False,
        )
        for sig in self.io.__dict__.values():
            if sig.kind == "input":
                module.add_input(sig)
            elif sig.kind == "output":
                module.add_output(sig)
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        module.component = self # can be used for debugging
        return module
    
    def make_internal(self) -> Self:
        # go through all signals in io and change to 'wire'
        for sig in self.io.__dict__.values():
            if sig.kind in ('input', 'output'):
                sig.kind = 'wire'
            else:
                raise ValueError(f"Signal {sig.name} has unsupported kind '{sig.kind}'")
        return self


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
    def __init__(self, config: MultiplierConfig) -> None:
        self.config = config
        self.multiplier_config = config


class PartialProductGeneratorBase(StageBase, abc.ABC):
    supported_signatures: ClassVar[Optional[Tuple[Tuple[bool, bool], ...]]] = None

    @abc.abstractmethod
    def generate_columns(self, io: "StageBasedMultiplierIO") -> DefaultDict[int, List[Expr]]:
        raise NotImplementedError


class PartialProductAccumulatorBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def accumulate(self, columns: Dict[int, List[Expr]]) -> DefaultDict[int, List[Expr]]:
        raise NotImplementedError


class FinalStageAdderBase(StageBase, abc.ABC):
    @abc.abstractmethod
    def resolve(self, columns: Dict[int, List[Expr]]) -> List[Expr]:
        raise NotImplementedError


class CompressorTreeAccumulator(PartialProductAccumulatorBase):
    def __init__(self, config: MultiplierConfig) -> None:
        super().__init__(config)
        self._full_adder = (
            full_adder_low_area
            if self.config.optim_type == "area"
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
        max_weight = self.config.out_width
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

        return result_bits


@dataclass
class StageBasedMultiplierIO:
    a: Signal
    b: Signal
    y: Signal


class StageBasedMultiplier(Component):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        *,
        signed_a: bool = False,
        signed_b: bool = False,
        optim_type: Literal["area", "speed"] = "area",
        ppg_cls: Type[PartialProductGeneratorBase],
        ppa_cls: Type[PartialProductAccumulatorBase] = CompressorTreeAccumulator,
        fsa_cls: Type[FinalStageAdderBase] = RippleCarryFinalAdder,
    ) -> None:
        self.config = MultiplierConfig(a_w, b_w, signed_a, signed_b, optim_type)

        supported = ppg_cls.supported_signatures
        if supported is not None and (signed_a, signed_b) not in supported:
            raise ValueError(
                f"{ppg_cls.__name__} does not support signed_a={signed_a}, signed_b={signed_b}"
            )

        base_typ_a = SInt if signed_a else UInt
        base_typ_b = SInt if signed_b else UInt
        base_type_y = SInt if (signed_a or signed_b) else UInt

        self.io : StageBasedMultiplierIO = StageBasedMultiplierIO(
            a=Signal(name="a", typ=base_typ_a(a_w), kind="input"),
            b=Signal(name="b", typ=base_typ_b(b_w), kind="input"),
            y=Signal(name="y", typ=base_type_y(self.config.out_width), kind="output"),
        )
        
        self.ppg = ppg_cls(self.config)
        self.ppa = ppa_cls(self.config)
        self.fsa = fsa_cls(self.config)
        
        self.elaborate()

    def elaborate(self) -> None:
        columns = self.ppg.generate_columns(self.io)
        reduced_columns = self.ppa.accumulate(columns)
        result_bits = self.fsa.resolve(reduced_columns)
        expected_width = self.io.y.typ.width
        self.io.y <<= Concat(reversed(result_bits[:expected_width]))
        
        # debugging
        self.colums = columns
        self.reduced_columns = reduced_columns

def gen_spec(component: Component) -> Dict[str, UInt]:
    spec: Dict[str, UInt] = {}
    for sig in component.io.__dict__.values():
        spec[sig.name] = sig.typ
    return spec


@dataclass
class MultiplierTestVectors:
    a_w: int
    b_w: int
    num_vectors: int = 64
    tb_sigma: Optional[float] = None
    signed_a: bool = False
    signed_b: bool = False

    def generate(self) -> Tuple[Dict[str, UInt], List[Tuple[str, Dict[str, int], Dict[str, int]]], None]:
        vecs: List[Tuple[str, Dict[str, int], Dict[str, int]]] = []

        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:

                def rand_unsigned(width: int) -> int:
                    return int(np.round(np.random.normal((1 << (width - 1)), self.tb_sigma)))

                def rand_signed(width: int) -> int:
                    return int(np.round(np.random.normal(0, self.tb_sigma)))

                va = rand_signed(self.a_w) if self.signed_a else rand_unsigned(self.a_w)
                vb = rand_signed(self.b_w) if self.signed_b else rand_unsigned(self.b_w)

                def clamp_unsigned(value: int, width: int) -> int:
                    return max(min(value, (1 << width) - 1), 0)

                def clamp_signed(value: int, width: int) -> int:
                    return max(min(value, (1 << (width - 1)) - 1), -(1 << (width - 1)))

                va = clamp_signed(va, self.a_w) if self.signed_a else clamp_unsigned(va, self.a_w)
                vb = clamp_signed(vb, self.b_w) if self.signed_b else clamp_unsigned(vb, self.b_w)
            else:
                if self.signed_a:
                    va = np.random.randint(-(1 << (self.a_w - 1)), 1 << (self.a_w - 1))
                else:
                    va = np.random.randint(0, 1 << self.a_w)
                if self.signed_b:
                    vb = np.random.randint(-(1 << (self.b_w - 1)), 1 << (self.b_w - 1))
                else:
                    vb = np.random.randint(0, 1 << self.b_w)

            vecs.append((f"{va}*{vb}", {"a": va, "b": vb}, {"y": va * vb}))

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.a_w + self.b_w)}
        return spec, vecs, None
