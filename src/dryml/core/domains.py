"""Immutable indexed domain capabilities for template generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable, Protocol, runtime_checkable

from .errors import TemplateError
from .freeze import FrozenTuple


@runtime_checkable
class Distribution(Protocol):
    """Provide sampling and optional finite indexed support for a binding.

    Implementations are trusted runtime providers. They are not template values
    and are not implicitly discovered from arbitrary objects.
    """

    def sample(self, rng: random.Random, /) -> object:
        """Draw one supported value using ``rng``."""

    def cardinality(self) -> int | None:
        """Return finite support size, or ``None`` when it cannot be indexed."""

    def value_at(self, index: int, /) -> object:
        """Return the value at a finite support index.

        Raises:
            TemplateError: If ``index`` is invalid or outside finite support.
        """

    def contains(self, value: object, /) -> bool | None:
        """Return exact membership, or ``None`` when membership is unavailable."""

    def bounds(self) -> tuple[int | float, int | float] | None:
        """Return numeric support bounds when they are exact and available."""


def _index(index: int, cardinality: int) -> int:
    if type(index) is not int or index < 0 or index >= cardinality:
        raise TemplateError("distribution index is outside finite support")
    return index


def _same_choice(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


@dataclass(frozen=True, slots=True)
class UniformIntRange:
    """Inclusive immutable uniform distribution over exact Python integers.

    Args:
        lo: Inclusive lower integer bound, excluding ``bool``.
        hi: Inclusive upper integer bound, excluding ``bool``.

    Raises:
        TemplateError: If bounds are not exact integers or ``hi`` precedes
            ``lo``.
    """

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if type(self.lo) is not int or type(self.hi) is not int:
            raise TemplateError("integer range bounds must be exact int values, not bool or coercible numerics")
        if self.hi < self.lo:
            raise TemplateError("integer range upper bound must not precede lower bound")

    def sample(self, rng: random.Random, /) -> int:
        """Draw an inclusive integer using the supplied random generator."""
        return rng.randint(self.lo, self.hi)

    def cardinality(self) -> int:
        """Return the inclusive support size without allocating the range."""
        return self.hi - self.lo + 1

    def value_at(self, index: int, /) -> int:
        """Return an indexed support value.

        Raises:
            TemplateError: If ``index`` is outside the inclusive range.
        """
        return self.lo + _index(index, self.cardinality())

    def contains(self, value: object, /) -> bool:
        """Return whether ``value`` is an exact integer within this range."""
        return type(value) is int and self.lo <= value <= self.hi

    def bounds(self) -> tuple[int, int]:
        """Return this range's inclusive numeric bounds."""
        return self.lo, self.hi


@dataclass(frozen=True, slots=True)
class UniformFromSet:
    """Uniform immutable distribution over a detached ordered choice support.

    Args:
        values: Nonempty iterable of supported template input values.

    Raises:
        TemplateError: If the support is empty or contains a type-aware
            duplicate.
    """

    values: FrozenTuple

    def __init__(self, values: Iterable[Any]) -> None:
        from .canonical import freeze_def_value

        frozen = FrozenTuple(freeze_def_value(value) for value in values)
        if not frozen:
            raise TemplateError("choice distribution requires at least one value")
        for index, value in enumerate(frozen):
            if any(_same_choice(value, prior) for prior in frozen[:index]):
                raise TemplateError("choice distribution values must be type-aware duplicate-free")
        object.__setattr__(self, "values", frozen)

    def sample(self, rng: random.Random, /) -> object:
        """Draw one frozen support value using the supplied random generator."""
        return self.values[rng.randrange(self.cardinality())]

    def cardinality(self) -> int:
        """Return the finite number of detached choices."""
        return len(self.values)

    def value_at(self, index: int, /) -> object:
        """Return an indexed choice.

        Raises:
            TemplateError: If ``index`` is outside finite support.
        """
        return self.values[_index(index, self.cardinality())]

    def contains(self, value: object, /) -> bool:
        """Return whether a type-aware exact choice equals ``value``."""
        return any(_same_choice(value, choice) for choice in self.values)

    def bounds(self) -> tuple[int | float, int | float] | None:
        """Return bounds only when every choice is a finite exact numeric value."""
        if not all(
            type(value) in {int, float}
            and (type(value) is int or math.isfinite(value))
            for value in self.values
        ):
            return None
        return min(self.values), max(self.values)


__all__ = ["Distribution", "UniformFromSet", "UniformIntRange"]
