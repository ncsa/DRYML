from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol
import random

from .freeze import FrozenTuple


class Matcher(Protocol):
    """Predicate used by Selector verification."""

    def matches(self, value: Any, *, present: bool = True) -> bool:
        ...


class Generator(Protocol):
    """Value generator used by SearchSpace sampling/grid expansion."""

    def sample(self, rng: random.Random) -> Any:
        ...

    def support_matcher(self) -> Matcher:
        ...

    def grid(self) -> tuple[Any, ...]:
        ...


@dataclass(frozen=True, slots=True)
class Par:
    """Parameterized placeholder with query matcher and optional generator."""

    name: str | None
    matcher: Matcher
    generator: Generator | None = None

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return self.matcher.matches(value, present=present)


@dataclass(frozen=True, slots=True)
class PresentMatcher:
    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present


@dataclass(frozen=True, slots=True)
class MissingMatcher:
    def matches(self, value: Any, *, present: bool = True) -> bool:
        return not present


@dataclass(frozen=True, slots=True)
class AnyMatcher:
    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present


@dataclass(frozen=True, slots=True)
class ExactMatcher:
    value: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_def_value

        object.__setattr__(self, "value", freeze_def_value(self.value))

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and value == self.value


@dataclass(frozen=True, slots=True)
class ChoiceMatcher:
    values: FrozenTuple

    def __init__(self, values: Iterable[Any]):
        from .canonical import freeze_def_value

        object.__setattr__(self, "values", FrozenTuple(freeze_def_value(v) for v in values))

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and any(value == choice for choice in self.values)


@dataclass(frozen=True, slots=True)
class IntRangeMatcher:
    lo: int
    hi: int

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and isinstance(value, int) and self.lo <= value <= self.hi


@dataclass(frozen=True, slots=True)
class SubclassMatcher:
    cls: type

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and isinstance(value, type) and issubclass(value, self.cls)


@dataclass(frozen=True, slots=True)
class SatisfiesMatcher:
    predicate: Callable[[Any], bool]
    name: str | None = None

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and bool(self.predicate(value))


@dataclass(frozen=True, slots=True)
class UniformIntRangeGenerator:
    lo: int
    hi: int

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.lo, self.hi)

    def support_matcher(self) -> Matcher:
        return IntRangeMatcher(self.lo, self.hi)

    def grid(self) -> tuple[int, ...]:
        return tuple(range(self.lo, self.hi + 1))


@dataclass(frozen=True, slots=True)
class UniformFromSetGenerator:
    values: FrozenTuple

    def __init__(self, values: Iterable[Any]):
        from .canonical import freeze_def_value

        object.__setattr__(self, "values", FrozenTuple(freeze_def_value(v) for v in values))

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(tuple(self.values))

    def support_matcher(self) -> Matcher:
        return ChoiceMatcher(self.values)

    def grid(self) -> tuple[Any, ...]:
        return tuple(self.values)


def Present(name: str | None = None) -> Par:
    return Par(name, PresentMatcher())


def Missing(name: str | None = None) -> Par:
    return Par(name, MissingMatcher())


def AnyValue(name: str | None = None) -> Par:
    return Par(name, AnyMatcher())


def Exact(value: Any, name: str | None = None) -> Par:
    return Par(name, ExactMatcher(value))


def Choice(values: Iterable[Any], name: str | None = None) -> Par:
    return Par(name, ChoiceMatcher(values))


def IntRange(lo: int, hi: int, name: str | None = None) -> Par:
    return Par(name, IntRangeMatcher(lo, hi))


def SubclassOf(cls: type, name: str | None = None) -> Par:
    return Par(name, SubclassMatcher(cls))


def Satisfies(predicate: Callable[[Any], bool], name: str | None = None) -> Par:
    return Par(name, SatisfiesMatcher(predicate, name=name))


def UniformIntRange(lo: int, hi: int, name: str | None = None) -> Par:
    gen = UniformIntRangeGenerator(lo, hi)
    return Par(name, gen.support_matcher(), gen)


def UniformFromSet(values: Iterable[Any], name: str | None = None) -> Par:
    gen = UniformFromSetGenerator(values)
    return Par(name, gen.support_matcher(), gen)
