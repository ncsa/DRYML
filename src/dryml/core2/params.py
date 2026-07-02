from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol
import random

from .freeze import FrozenTuple


class Matcher(Protocol):
    """Predicate used by Selector verification."""

    def matches(self, value: Any, *, present: bool = True) -> bool:
        ...

    def stable_key(self) -> Any:
        ...


class Generator(Protocol):
    """Value generator used by SearchSpace sampling/grid expansion."""

    def sample(self, rng: random.Random) -> Any:
        ...

    def support_matcher(self) -> Matcher:
        ...

    def grid(self) -> tuple[Any, ...]:
        ...

    def stable_key(self) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class Par:
    """Parameterized placeholder with query matcher and optional generator."""

    name: str | None
    matcher: Matcher
    generator: Generator | None = None

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return self.matcher.matches(value, present=present)

    def stable_key(self) -> Any:
        gen_key = None if self.generator is None else self.generator.stable_key()
        return ("par", self.name, self.matcher.stable_key(), gen_key)


@dataclass(frozen=True, slots=True)
class PresentMatcher:
    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present

    def stable_key(self) -> Any:
        return ("present",)


@dataclass(frozen=True, slots=True)
class MissingMatcher:
    def matches(self, value: Any, *, present: bool = True) -> bool:
        return not present

    def stable_key(self) -> Any:
        return ("missing",)


@dataclass(frozen=True, slots=True)
class AnyMatcher:
    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present

    def stable_key(self) -> Any:
        return ("any",)


@dataclass(frozen=True, slots=True)
class ExactMatcher:
    value: Any

    def __post_init__(self) -> None:
        from .canonical import freeze_def_value

        object.__setattr__(self, "value", freeze_def_value(self.value))

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and value == self.value

    def stable_key(self) -> Any:
        return ("exact", self.value)


@dataclass(frozen=True, slots=True)
class ChoiceMatcher:
    values: FrozenTuple

    def __init__(self, values: Iterable[Any]):
        from .canonical import freeze_def_value

        object.__setattr__(self, "values", FrozenTuple(freeze_def_value(v) for v in values))

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and any(value == choice for choice in self.values)

    def stable_key(self) -> Any:
        return ("choice", self.values)


@dataclass(frozen=True, slots=True)
class IntRangeMatcher:
    lo: int
    hi: int

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and isinstance(value, int) and self.lo <= value <= self.hi

    def stable_key(self) -> Any:
        return ("int-range", self.lo, self.hi)


@dataclass(frozen=True, slots=True)
class SubclassMatcher:
    cls: type

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and isinstance(value, type) and issubclass(value, self.cls)

    def stable_key(self) -> Any:
        return ("subclass", self.cls)


@dataclass(frozen=True, slots=True)
class SatisfiesMatcher:
    """Predicate matcher whose optional name is a stable semantic identity."""

    predicate: Callable[[Any], bool]
    name: str | None = None

    def matches(self, value: Any, *, present: bool = True) -> bool:
        return present and bool(self.predicate(value))

    def stable_key(self) -> Any:
        if self.name is not None:
            return ("satisfies", self.name)
        from .symbol import maybe_symbol_ref

        ref = maybe_symbol_ref(self.predicate, functions=True)
        if ref is not None:
            return ("satisfies", ref)
        raise TypeError("Anonymous Satisfies predicates are not stable-hashable; provide name=...")


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

    def stable_key(self) -> Any:
        return ("uniform-int-range", self.lo, self.hi)


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

    def stable_key(self) -> Any:
        return ("uniform-from-set", self.values)


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
