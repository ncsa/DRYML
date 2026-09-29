from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering


class CardinalityKind(Enum):
    FINITE = auto()
    INFINITE = auto()
    UNKNOWN = auto()


@total_ordering
@dataclass(frozen=True, slots=True)
class Cardinality:
    kind: CardinalityKind
    value: int | None = None

    def __post_init__(self) -> None:
        if self.kind is CardinalityKind.FINITE:
            if self.value is None:
                raise ValueError("Finite cardinality requires an integer value.")
            if self.value < 0:
                raise ValueError("Cardinality must be a non-negative integer.")
        else:
            if self.value is not None:
                raise ValueError("Only finite cardinality may have a value.")

    @classmethod
    def finite(cls, n: int) -> Cardinality:
        return cls(CardinalityKind.FINITE, n)

    @classmethod
    def infinite(cls) -> Cardinality:
        return cls(CardinalityKind.INFINITE)

    @classmethod
    def unknown(cls) -> Cardinality:
        return cls(CardinalityKind.UNKNOWN)

    @property
    def is_finite(self) -> bool:
        return self.kind is CardinalityKind.FINITE

    @property
    def is_infinite(self) -> bool:
        return self.kind is CardinalityKind.INFINITE

    @property
    def is_unknown(self) -> bool:
        return self.kind is CardinalityKind.UNKNOWN

    def require_finite(self) -> int:
        if not self.is_finite:
            raise ValueError(f"Expected finite cardinality, got {self}.")
        assert self.value is not None
        return self.value

    def __int__(self) -> int:
        return self.require_finite()

    def __repr__(self) -> str:
        if self.kind is CardinalityKind.FINITE:
            return f"Cardinality({self.value})"
        if self.kind is CardinalityKind.INFINITE:
            return "Cardinality.INFINITE"
        return "Cardinality.UNKNOWN"

    def __stable_leaf_bytes__(self):
        return str(self).encode("utf-8")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cardinality):
            return NotImplemented
        return (self.kind, self.value) == (other.kind, other.value)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Cardinality):
            return NotImplemented

        # UNKNOWN is intentionally not ordered.
        if self.is_unknown or other.is_unknown:
            raise TypeError("Unknown cardinality is not orderable.")

        if self.is_finite and other.is_finite:
            assert self.value is not None and other.value is not None
            return self.value < other.value

        if self.is_finite and other.is_infinite:
            return True

        if self.is_infinite and other.is_finite:
            return False

        return False  # infinite < infinite is False


Cardinality.INFINITE = Cardinality.infinite()
Cardinality.UNKNOWN = Cardinality.unknown()
