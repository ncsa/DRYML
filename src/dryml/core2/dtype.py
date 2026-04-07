from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re
import numpy as np


_VALID_KINDS = {
    "bool",
    "int",
    "uint",
    "float",
    "bfloat",
    "complex",
    "string",
    "object",
}


@dataclass(frozen=True, slots=True)
class DType:
    """
    Canonical backend-independent tensor dtype.

    Examples
    --------
    DType("float", 32)    -> float32
    DType("int", 64)      -> int64
    DType("bool", None)   -> bool
    DType("bfloat", 16)   -> bfloat16
    """
    kind: str
    bits: int | None = None

    def __post_init__(self):
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"Invalid dtype kind {self.kind!r}.")
        if self.kind in {"bool", "string", "object"}:
            if self.bits is not None:
                raise ValueError(f"{self.kind} does not take a bit width.")
        else:
            if not isinstance(self.bits, int) or self.bits <= 0:
                raise ValueError(f"{self.kind} requires a positive integer bit width.")

    @property
    def name(self) -> str:
        if self.kind in {"bool", "string", "object"}:
            return self.kind
        return f"{self.kind}{self.bits}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"DType({self.name!r})" 


_DTYPE_RE = re.compile(
    r"^(?:(bool|string|object)|((?:u?int|float|bfloat|complex))(\d+))$"
)


def normalize_dtype(x: Any) -> DType:
    if isinstance(x, DType):
        return x

    if isinstance(x, str):
        m = _DTYPE_RE.fullmatch(x)
        if m is None:
            raise ValueError(f"Unrecognized dtype string {x!r}.")

        simple_kind = m.group(1)
        if simple_kind is not None:
            return DType(simple_kind)

        kind = m.group(2)
        bits = int(m.group(3))
        return DType(kind, bits)

    # numpy scalar type / numpy dtype
    try:
        np_dtype = np.dtype(x)
        return normalize_dtype(np_dtype.name)
    except Exception:
        pass

    # generic ".name" fallback for tf dtypes and similar
    name = getattr(x, "name", None)
    if isinstance(name, str):
        return normalize_dtype(name)

    # torch.dtype often stringifies as "torch.float32"
    s = str(x)
    if s.startswith("torch."):
        return normalize_dtype(s.removeprefix("torch."))

    raise TypeError(f"Cannot normalize dtype from {x!r}.")
