from enum import Enum
import random
from typing import Optional, Tuple, Union

import numpy as np


from sprouthdl.sprouthdl import UInt

# enum for


class Format(Enum):
    twos_complement = "twos_complement"
    twos_complement_symmetric = "twos_complement_symmetric"
    sign_magnitude = "sign_magnitude"
    sign_magnitude_ext = "sign_magnitude_ext"
    gray = "gray"
    unsigned = "unsigned"
    onehot = "onehot"

def to_format(signed: bool | Format) -> Format:
    if isinstance(signed, Format):
        return signed
    return Format.twos_complement if signed else Format.unsigned


class MultiplierTestVectors:

    _SIGNED_FORMATS = {
        Format.twos_complement,
        Format.twos_complement_symmetric,
        Format.sign_magnitude,
        Format.sign_magnitude_ext,
    }

    def __init__(
        self,
        a_w: int,
        b_w: int,
        y_w: Optional[int] = None,
        num_vectors: int = 64,
        tb_sigma: Optional[float] = None,
        format_a: Format = Format.unsigned,
        format_b: Format = Format.unsigned,
    ):
        self.a_w = a_w
        self.b_w = b_w
        self.y_w = a_w + b_w if y_w is None else y_w
        self.num_vectors = num_vectors
        self.tb_sigma = tb_sigma
        self.a_format = format_a
        self.b_format = format_b

    @classmethod
    def _is_signed(cls, fmt: Format) -> bool:
        return fmt in cls._SIGNED_FORMATS

    @staticmethod
    def _value_range(fmt: Format, width: int) -> Tuple[int, int]:
        if fmt in [Format.twos_complement, Format.sign_magnitude_ext]:
            return (-(1 << (width - 1)), (1 << (width - 1)) - 1)
        if fmt == Format.twos_complement_symmetric:
            limit = (1 << (width - 1)) - 1
            return (-limit, limit)
        if fmt in [Format.sign_magnitude]:
            limit = (1 << (width - 1)) - 1
            return (-limit, limit)
        if fmt == Format.onehot:
            return (0, max(width - 1, 0))
        # unsigned-like formats (unsigned, gray)
        return (0, (1 << width) - 1)

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(min(value, hi), lo)

    @staticmethod
    def _encode_value(fmt: Format, value: int, width: int) -> int:
        # to be sure lets clamp
        clamped = MultiplierTestVectors._clamp(value, *MultiplierTestVectors._value_range(fmt, width))
        if fmt == Format.onehot:
            return 1 << clamped
        if fmt == Format.gray:            
            return clamped ^ (clamped >> 1)
        if fmt in [Format.sign_magnitude, Format.sign_magnitude_ext]:
            sign_bit = 1 if clamped < 0 else 0
            magnitude = abs(clamped)
            magnitude = magnitude & ((1 << (width - 1)) - 1)  # mask to width-1 bits
            return (sign_bit << (width - 1)) | magnitude
        return value

    def get_normal_sample(self, fmt: Format, width: int) -> int:
        lo, hi = self._value_range(fmt, width)
        if fmt == Format.onehot:
            center = (width - 1) / 2 if width > 0 else 0
            raw_value = int(np.round(np.random.normal(center, self.tb_sigma)))
            raw_value = self._clamp(raw_value, lo, hi)
            return raw_value

        mean = 0 if self._is_signed(fmt) else (lo + hi) / 2
        raw_value = int(np.round(np.random.normal(mean, self.tb_sigma)))
        raw_value = self._clamp(raw_value, lo, hi)
        return raw_value

    def get_uniform_sample(self, fmt: Format, width: int) -> int:
        lo, hi = self._value_range(fmt, width)
        if fmt == Format.onehot:
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
                va_value = self.get_normal_sample(self.a_format, self.a_w)
                vb_value = self.get_normal_sample(self.b_format, self.b_w)
            else:
                va_value = self.get_uniform_sample(self.a_format, self.a_w)
                vb_value = self.get_uniform_sample(self.b_format, self.b_w)
            
            va_encoded = self._encode_value(self.a_format, va_value, self.a_w)
            vb_encoded = self._encode_value(self.b_format, vb_value, self.b_w)

            y_value = va_value * vb_value
            y_encoded = self._encode_value(self.a_format, y_value, self.y_w)

            # append test vector
            vecs.append(
                (f"{va_value}*{vb_value}", {"a": va_encoded, "b": vb_encoded}, {"y": y_encoded})
            )

        spec = {"a": UInt(self.a_w), "b": UInt(self.b_w), "y": UInt(self.y_w)}
        return spec, vecs, None
