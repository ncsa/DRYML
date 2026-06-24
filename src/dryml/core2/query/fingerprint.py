from __future__ import annotations

from collections import Counter
from typing import Any

from ..definition import ConcreteDefinition, Definition
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..object import Object
from .local_structure import canonical_class_key, scalar_key, walk_local_structure
from .model import (
    ClassMatchPolicy,
    DefinitionFingerprint,
    ExactSubtreeConstraint,
    FeatureRequirement,
    FeatureToken,
)
from .path import Arg, DefinitionPath, Index, Key, Kwarg


def target_fingerprint(cdef: ConcreteDefinition) -> DefinitionFingerprint:
    counts: Counter[FeatureToken] = Counter()
    walk_local_structure(cdef, _FeatureCounter(counts), mode="target-full")
    return DefinitionFingerprint(dict(counts))


def target_local_fingerprint(cdef: ConcreteDefinition) -> DefinitionFingerprint:
    counts: Counter[FeatureToken] = Counter()
    walk_local_structure(cdef, _FeatureCounter(counts, include_edge_class=True), mode="target-local")
    return DefinitionFingerprint(dict(counts))


def selector_requirements(
        selector: Any,
        *,
        class_match: ClassMatchPolicy = "selector") -> tuple[FeatureRequirement, ...]:
    counts: Counter[FeatureToken] = Counter()
    walk_local_structure(selector, _FeatureCounter(counts), mode="selector-full", class_match=class_match)
    return tuple(
        FeatureRequirement(token, count)
        for token, count in sorted(counts.items(), key=lambda item: repr(item[0]))
    )


def selector_local_requirements(
        selector: Any,
        *,
        class_match: ClassMatchPolicy = "selector") -> tuple[FeatureRequirement, ...]:
    counts: Counter[FeatureToken] = Counter()
    walk_local_structure(selector, _FeatureCounter(counts), mode="selector-local", class_match=class_match)
    return tuple(
        FeatureRequirement(token, count)
        for token, count in sorted(counts.items(), key=lambda item: repr(item[0]))
    )


def collect_exact_constraints(selector: Any) -> tuple[ExactSubtreeConstraint, ...]:
    constraints: list[ExactSubtreeConstraint] = []
    _collect_exact_constraints(selector, DefinitionPath(), constraints)
    return tuple(constraints)


def _add(counts: Counter[FeatureToken], kind: str, path: DefinitionPath | None, payload: Any = None) -> None:
    counts[FeatureToken(kind, path, payload)] += 1


class _FeatureCounter:
    def __init__(self, counts: Counter[FeatureToken], *, include_edge_class: bool = False):
        self.counts = counts
        self.include_edge_class = include_edge_class

    def feature(self, kind: str, path: DefinitionPath | None, payload: Any = None) -> None:
        _add(self.counts, kind, path, payload)

    def definition_boundary(
            self,
            path: DefinitionPath,
            definition: Definition | ConcreteDefinition,
            *,
            unordered: bool = False) -> None:
        _add(self.counts, "CDEF_EDGE_AT_PATH", path, None)
        if isinstance(definition, ConcreteDefinition):
            _add(self.counts, "CDEF_EDGE_EXACT", path, definition.stable_hash())
            if self.include_edge_class:
                _add(self.counts, "CDEF_EDGE_CLASS", path, canonical_class_key(definition.cls))


def _collect_exact_constraints(value: Any, path: DefinitionPath, out: list[ExactSubtreeConstraint]) -> None:
    if isinstance(value, Object):
        value = value.definition

    if isinstance(value, ConcreteDefinition):
        out.append(ExactSubtreeConstraint(path, value))
        for idx, child in enumerate(value.args):
            _collect_exact_constraints(child, path.child(Arg(idx)), out)
        for key, child in value.kwargs.items():
            _collect_exact_constraints(child, path.child(Kwarg(key)), out)
        return

    if isinstance(value, Definition):
        if value.args is not None:
            for idx, child in enumerate(value.args):
                _collect_exact_constraints(child, path.child(Arg(idx)), out)
        for key, child in value.kwargs.items():
            _collect_exact_constraints(child, path.child(Kwarg(key)), out)
        return

    if isinstance(value, (dict, FrozenDict)):
        for key, child in value.items():
            _collect_exact_constraints(child, path.child(Key(key)), out)
        return

    if isinstance(value, (list, tuple, FrozenList, FrozenTuple)):
        for idx, child in enumerate(value):
            _collect_exact_constraints(child, path.child(Index(idx)), out)
        return

    if isinstance(value, (set, frozenset, FrozenSet)):
        for child in value:
            if isinstance(child, Object):
                child = child.definition
            if isinstance(child, ConcreteDefinition):
                out.append(ExactSubtreeConstraint(path, child, unordered_member=True))
        return


def requirements_satisfied(fingerprint: DefinitionFingerprint, requirements: tuple[FeatureRequirement, ...]) -> bool:
    for req in requirements:
        if fingerprint.counts.get(req.token, 0) < req.count:
            return False
    return True
