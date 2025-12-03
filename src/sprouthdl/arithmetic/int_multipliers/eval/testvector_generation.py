from enum import Enum
import random
from typing import Optional, Tuple, Union

import numpy as np


from sprouthdl.sprouthdl import UInt

# enum for


class Encoding(Enum):
    twos_complement = "twos_complement"
    twos_complement_symmetric = "twos_complement_symmetric"
    twos_complement_upper = "twos_complement_upper"
    sign_magnitude = "sign_magnitude"
    sign_magnitude_ext = "sign_magnitude_ext"
    gray = "gray"
    unsigned = "unsigned"
    unsigned_overflow = "unsigned_overflow"
    twos_complement_overflow = "twos_complement_overflow"
    onehot = "onehot"
    sign_magnitude_ext_up = "sign_magnitude_ext_up"
    
_SIGNED_encodings = {
    Encoding.twos_complement,
    Encoding.twos_complement_symmetric,
    Encoding.sign_magnitude,
    Encoding.sign_magnitude_ext,
    Encoding.sign_magnitude_ext_up,
    Encoding.twos_complement_overflow
}

def is_signed(fmt: Encoding) -> bool:
    return fmt in _SIGNED_encodings

def to_encoding(signed: bool | Encoding) -> Encoding:
    if isinstance(signed, Encoding):
        return signed
    return Encoding.twos_complement if signed else Encoding.unsigned

def from_encoding(fmt: Encoding) -> bool:
    assert fmt in {Encoding.twos_complement, Encoding.unsigned}, f"Cannot convert encoding {fmt} to bool"
    return fmt == Encoding.twos_complement

class TwoInputArithmeticTestVectors:
    


    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: Optional[int] = None,
        tb_sigma: Optional[float] = None,
        a_encoding: Encoding = Encoding.unsigned,
        b_encoding: Encoding = Encoding.unsigned,
        y_encoding: Encoding = Encoding.unsigned,
    ):
        self.a_w = a_w
        self.b_w = b_w
        self.y_w = a_w + b_w if y_w is None else y_w
        self.num_vectors = num_vectors
        self.tb_sigma = tb_sigma
        self.a_encoding = a_encoding
        self.b_encoding = b_encoding
        self.y_encoding = y_encoding           
    
    @staticmethod
    def _value_range(fmt: Encoding, width: int) -> Tuple[int, int]:
        if fmt in [Encoding.twos_complement, Encoding.sign_magnitude_ext]:
            return (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        if fmt == Encoding.twos_complement_symmetric:
            limit = (1 << (width - 1)) - 1
            return (-limit, limit)
        if fmt == Encoding.twos_complement_upper: # extend on the upper side
            return (-(1 << (width - 1))+1, (1 << (width - 1)))
        if fmt in [Encoding.sign_magnitude]:
            limit = (1 << (width - 1)) - 1
            return (-limit, limit)
        if fmt== Encoding.sign_magnitude_ext_up: # extend on the upper side
            return (-(1 << (width - 1) + 1), (1 << (width - 1)))
        if fmt == Encoding.onehot:
            return (0, max(width - 1, 0))
        # unsigned-like encodings (unsigned, gray)
        return (0, (1 << width) - 1)
    
    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(min(value, hi), lo)
    
    @staticmethod
    def _encode_value(fmt: Encoding, value: int, width: int) -> int:
        # to be sure lets clamp
        clamped = MultiplierTestVectors._clamp(value, *MultiplierTestVectors._value_range(fmt, width))
        if fmt == Encoding.onehot:
            return 1 << clamped
        if fmt == Encoding.gray:            
            return clamped ^ (clamped >> 1)
        if fmt in [Encoding.sign_magnitude, Encoding.sign_magnitude_ext]:
            sign_bit = 1 if clamped < 0 else 0
            magnitude = abs(clamped)
            magnitude = magnitude & ((1 << (width - 1)) - 1)  # mask to width-1 bits
            return (sign_bit << (width - 1)) | magnitude
        if fmt == Encoding.sign_magnitude_ext_up:
            sign_bit = 1 if clamped < 0 else 0
            magnitude = abs(clamped)
            magnitude = magnitude & ((1 << (width - 1)) - 1)  # mask to width-1 bits
            if clamped == (1 << (width - 1)):
                sign_bit = 1 # 100..0 represents the upper limit
            return (sign_bit << (width - 1)) | magnitude
        if fmt in [Encoding.twos_complement, Encoding.twos_complement_symmetric, Encoding.twos_complement_upper]:
            if clamped < 0:
                return (1 << width) + clamped  # two's complement representation
            else:
                return clamped
        if fmt == Encoding.unsigned_overflow or fmt == Encoding.twos_complement_overflow:
            return value % (1 << width)
        return clamped
    
    def get_normal_sample(self, fmt: Encoding, width: int) -> int:
        lo, hi = self._value_range(fmt, width)
        if fmt == Encoding.onehot:
            center = (width - 1) / 2 if width > 0 else 0
            raw_value = int(np.round(np.random.normal(center, self.tb_sigma)))
            raw_value = self._clamp(raw_value, lo, hi)
            return raw_value
    
        mean = 0 if is_signed(fmt) else (lo + hi) / 2
        raw_value = int(np.round(np.random.normal(mean, self.tb_sigma)))
        raw_value = self._clamp(raw_value, lo, hi)
        return raw_value
    
    def get_uniform_sample(self, fmt: Encoding, width: int) -> int:
        lo, hi = self._value_range(fmt, width)
        if fmt == Encoding.onehot:
            if width <= 0:
                return 0
            raw_value = random.randrange(width)
            return raw_value
        raw_value = random.randint(lo, hi)
        return raw_value

class MultiplierTestVectors(TwoInputArithmeticTestVectors):

    def generate(self) -> Tuple:

        if self.num_vectors is None:
            self.num_vectors = 64  # default number of vectors

        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:
                va_value = self.get_normal_sample(self.a_encoding, self.a_w)
                vb_value = self.get_normal_sample(self.b_encoding, self.b_w)
            else:
                va_value = self.get_uniform_sample(self.a_encoding, self.a_w)
                vb_value = self.get_uniform_sample(self.b_encoding, self.b_w)

            va_encoded = self._encode_value(self.a_encoding, va_value, self.a_w)
            vb_encoded = self._encode_value(self.b_encoding, vb_value, self.b_w)

            y_value = va_value * vb_value
            y_encoded = self._encode_value(self.y_encoding, y_value, self.y_w)

            # append test vector
            vecs.append(
                (f"{va_value}*{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs


class MultiplierTestVectorsExhaustive(TwoInputArithmeticTestVectors):

    def generate_encoding_tables(self) -> Tuple[dict, dict, dict]:
        a_table = {}
        b_table = {}
        y_table = {}

        a_lo, a_hi = self._value_range(self.a_encoding, self.a_w)
        for val in range(a_lo, a_hi + 1):
            enc = self._encode_value(self.a_encoding, val, self.a_w)
            a_table[enc] = val

        b_lo, b_hi = self._value_range(self.b_encoding, self.b_w)
        for val in range(b_lo, b_hi + 1):
            enc = self._encode_value(self.b_encoding, val, self.b_w)
            b_table[enc] = val

        y_lo, y_hi = self._value_range(self.y_encoding, self.y_w)
        for val in range(y_lo, y_hi + 1):
            enc = self._encode_value(self.y_encoding, val, self.y_w)
            y_table[enc] = val

        return a_table, b_table, y_table

    def generate(self) -> Tuple:

        if self.num_vectors is not None or self.tb_sigma is not None:
            raise ValueError("num_vectors must be None for exhaustive test vector generation")       

        total_combinations = (1 << self.a_w) * (1 << self.b_w)
        self.num_vectors = total_combinations
        a_table, b_table, y_table = self.generate_encoding_tables()

        vecs = []
        for i in range(self.num_vectors):

            # get encoded value
            va_encoded = i % (1 << self.a_w)
            vb_encoded = i // (1 << self.a_w)

            va_value = a_table[va_encoded]
            vb_value = b_table[vb_encoded]
            y_value = va_value * vb_value
            y_encoded = y_table[y_value]

            # append test vector
            vecs.append(
                (f"{va_value}*{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs


class AdderTestVectors(TwoInputArithmeticTestVectors):

    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: Optional[int] = None,
        tb_sigma: Optional[float] = None,
        a_encoding: Encoding = Encoding.unsigned,
        b_encoding: Encoding = Encoding.unsigned,
        y_encoding: Encoding = Encoding.unsigned,
    ):
        y_width = max(a_w, b_w) + 1 if y_w is None else y_w
        super().__init__(
            a_w=a_w,
            b_w=b_w,
            y_w=y_width,
            num_vectors=num_vectors,
            tb_sigma=tb_sigma,
            a_encoding=a_encoding,
            b_encoding=b_encoding,
            y_encoding=y_encoding,
        )

    def generate(self) -> Tuple:
        if self.num_vectors is None:
            self.num_vectors = 64

        vecs = []
        for _ in range(self.num_vectors):
            if self.tb_sigma is not None:
                va_value = self.get_normal_sample(self.a_encoding, self.a_w)
                vb_value = self.get_normal_sample(self.b_encoding, self.b_w)
            else:
                va_value = self.get_uniform_sample(self.a_encoding, self.a_w)
                vb_value = self.get_uniform_sample(self.b_encoding, self.b_w)

            va_encoded = self._encode_value(self.a_encoding, va_value, self.a_w)
            vb_encoded = self._encode_value(self.b_encoding, vb_value, self.b_w)

            y_value = va_value + vb_value
            y_encoded = self._encode_value(self.y_encoding, y_value, self.y_w)

            vecs.append(
                (f"{va_value}+{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        return vecs
