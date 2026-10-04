from __future__ import annotations

from collections.abc import ItemsView, Iterable, Mapping, ValuesView
from dataclasses import dataclass
from typing import Any, Iterator

from ..definition import ConcreteDefinition
from ..object import Object
from ..policies import CachePolicy
from .model import QueryCardinalityError, QueryExplanation


def _sort_cdefs(cdefs: Iterable[ConcreteDefinition]) -> tuple[ConcreteDefinition, ...]:
    return tuple(sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


@dataclass(frozen=True, slots=True)
class ObjectResultSet(Mapping):
    """Immutable retained live-Object mapping returned by non-query lookups.

    Args:
        repo: Repo owning the retained Objects and materialization admission.
        objects: Complete definitions mapped to already-live Objects.
        domain: Diagnostic lookup domain, normally ``"stored"`` or ``"known"``.
        explanation: Optional detached explanation for the lookup.

    Iteration yields complete-definition keys in deterministic order. Value
    access and cardinality terminals enter materialization admission but do not
    construct, restore, or query for additional Objects. Cardinality terminals
    raise :class:`QueryCardinalityError` when their contracts are not met.
    """

    repo: Any
    _objects: dict[ConcreteDefinition, Object]
    domain: str = "stored"
    explanation: QueryExplanation | None = None

    def __init__(
            self,
            repo,
            objects: Mapping[ConcreteDefinition, Object],
            *,
            domain: str = "stored",
        explanation: QueryExplanation | None = None,
    ):
        object.__setattr__(self, "repo", repo)
        ordered = {cdef: objects[cdef] for cdef in _sort_cdefs(objects.keys())}
        object.__setattr__(self, "_objects", ordered)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "explanation", explanation)

    def __getitem__(self, key: ConcreteDefinition) -> Object:
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_getitem"):
            return self._objects[key]

    def __iter__(self) -> Iterator[ConcreteDefinition]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def count(self) -> int:
        """Return the number of retained definition/Object pairs."""

        return len(self)

    def exists(self) -> bool:
        """Return whether at least one retained Object exists."""

        return len(self) > 0

    def one(self) -> Object:
        """Return the sole Object under admission, raising on other cardinality."""

        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_one"):
            if len(self) != 1:
                raise QueryCardinalityError(
                    f"Expected exactly one object, found {len(self)}."
                )
            return next(iter(self._objects.values()))

    def one_or_none(self) -> Object | None:
        """Return zero or one Object under admission, raising on ambiguity."""

        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_one_or_none"):
            if len(self) > 1:
                raise QueryCardinalityError(
                    f"Expected zero or one object, found {len(self)}."
                )
            return next(iter(self._objects.values())) if self._objects else None

    def first(self) -> Object | None:
        """Return the first deterministic Object under admission, or ``None``."""

        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_first"):
            return next(iter(self._objects.values())) if self._objects else None

    def items(self):
        """Return a repeatable item view whose iteration holds one admission lease."""

        return _GuardedObjectItemsView(self)

    def values(self):
        """Return a repeatable value view whose iteration holds one admission lease."""

        return _GuardedObjectValuesView(self)

    def apply(self, func, *args, **kwargs) -> "ObjectResultSet":
        """Apply ``func`` to every retained Object under one admission lease.

        Positional and keyword arguments are forwarded to ``func`` after each
        Object. The same result set is returned. Callback exceptions propagate
        after any earlier Objects have already been visited.
        """

        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_apply"):
            for obj in self._objects.values():
                func(obj, *args, **kwargs)
            return self


class _GuardedObjectItemsView(ItemsView):
    """Mapping items view that admits retained Objects for each full iteration."""

    def __iter__(self):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_items"):
            yield from self._mapping._objects.items()


class _GuardedObjectValuesView(ValuesView):
    """Mapping values view that admits retained Objects for each full iteration."""

    def __iter__(self):
        from dryml.runtime import materialization_admission

        with materialization_admission(operation="object_result_set_values"):
            yield from self._mapping._objects.values()
