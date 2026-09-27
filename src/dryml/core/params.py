from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from .freeze import FrozenTuple


class Matcher(Protocol):
    """Predicate used by Selector verification."""

    def matches(self, value: Any, *, present: bool = True) -> bool:
        ...

    def stable_key(self) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class Match:
    """Independent query predicate leaf with optional descriptive metadata.

    Args:
        matcher: The predicate implementation used for selector matching.
        name: Optional metadata retained in the stable predicate identity. It
            does not bind a template parameter or connect query fields.

    ``Match`` is independent from template expressions. Its optional name is
    query metadata and never links fields or template bindings.
    """

    matcher: Matcher
    name: str | None = None

    def __init__(self, matcher: Matcher, name: str | None = None) -> None:
        object.__setattr__(self, "matcher", matcher)
        object.__setattr__(self, "name", name)

    def matches(self, value: Any, *, present: bool = True) -> bool:
        """Return whether the wrapped matcher accepts ``value``.

        Args:
            value: Candidate selector value.
            present: Whether the candidate field exists.

        Returns:
            ``True`` when the wrapped matcher accepts the candidate.
        """

        return self.matcher.matches(value, present=present)

    def stable_key(self) -> Any:
        """Return the stable query identity supplied by the wrapped matcher.

        Returns:
            An immutable matcher-derived identity used by selector persistence.

        Raises:
            TypeError: If the wrapped matcher has no stable identity.
        """

        return ("match", self.name, self.matcher.stable_key())


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


def Present(name: str | None = None) -> Match:
    """Return a predicate matching present values without creating a binding.

    Args:
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.
    """
    return Match(PresentMatcher(), name)


def Missing(name: str | None = None) -> Match:
    """Return a predicate matching absent values without creating a binding.

    Args:
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.
    """
    return Match(MissingMatcher(), name)


def AnyValue(name: str | None = None) -> Match:
    """Return a predicate matching every present value.

    Args:
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.
    """
    return Match(AnyMatcher(), name)


def Exact(value: Any, name: str | None = None) -> Match:
    """Return a predicate matching exactly one frozen value.

    Args:
        value: Supported value to freeze into the predicate.
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.

    Raises:
        TypeError: If ``value`` is not a supported definition value.
    """
    return Match(ExactMatcher(value), name)


def Choice(values: Iterable[Any], name: str | None = None) -> Match:
    """Return a predicate matching one of the supplied frozen values.

    Args:
        values: Supported values to freeze into the predicate.
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.
    """
    return Match(ChoiceMatcher(values), name)


def IntRange(lo: int, hi: int, name: str | None = None) -> Match:
    """Return a predicate matching an inclusive integer range.

    Args:
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.
    """
    return Match(IntRangeMatcher(lo, hi), name)


def SubclassOf(cls: type, name: str | None = None) -> Match:
    """Return a predicate matching subclasses of ``cls``.

    Args:
        cls: Required superclass.
        name: Optional stable query metadata.

    Returns:
        An independent ``Match`` leaf.
    """
    return Match(SubclassMatcher(cls), name)


def Satisfies(predicate: Callable[[Any], bool], name: str | None = None) -> Match:
    """Return a trusted predicate leaf for ordinary selector matching.

    Args:
        predicate: Trusted callable evaluated by ordinary selector matching.
        name: Optional stable query metadata; required for an anonymous callable
            that must be stable-hashed.

    Returns:
        An independent ``Match`` leaf.

    Side Effects:
        The predicate is not called until selector matching.
    """
    return Match(SatisfiesMatcher(predicate, name=name), name)
