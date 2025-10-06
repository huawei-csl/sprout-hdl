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
    onehot = "onehot"
    sign_magnitude_ext_up = "sign_magnitude_ext_up"

def to_encoding(signed: bool | Encoding) -> Encoding:
    if isinstance(signed, Encoding):
        return signed
    return Encoding.twos_complement if signed else Encoding.unsigned

def from_encoding(fmt: Encoding) -> bool:
    assert fmt in {Encoding.twos_complement, Encoding.unsigned}, f"Cannot convert encoding {fmt} to bool"
    return fmt == Encoding.twos_complement


class MultiplierTestVectors:

    _SIGNED_encodings = {
        Encoding.twos_complement,
        Encoding.twos_complement_symmetric,
        Encoding.sign_magnitude,
        Encoding.sign_magnitude_ext,
    }

    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: int = 64,
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

    @classmethod
    def _is_signed(cls, fmt: Encoding) -> bool:
        return fmt in cls._SIGNED_encodings

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
        return value

    def get_normal_sample(self, fmt: Encoding, width: int) -> int:
        lo, hi = self._value_range(fmt, width)
        if fmt == Encoding.onehot:
            center = (width - 1) / 2 if width > 0 else 0
            raw_value = int(np.round(np.random.normal(center, self.tb_sigma)))
            raw_value = self._clamp(raw_value, lo, hi)
            return raw_value

        mean = 0 if self._is_signed(fmt) else (lo + hi) / 2
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

    def generate(self) -> Tuple:

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

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.y_w)}
        return spec, vecs, None
