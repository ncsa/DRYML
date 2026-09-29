from __future__ import annotations

from collections import Counter
from typing import Any

from ..cdef_graph import EdgeKind
from ..definition import ConcreteDefinition, Definition
from .local_structure import canonical_class_key, scalar_key, walk_local_structure
from .model import (
    ClassMatchPolicy,
    DefinitionFingerprint,
    FeatureRequirement,
    FeatureToken,
)
from .path import DefinitionPath


def legacy_target_fingerprint(cdef: ConcreteDefinition) -> DefinitionFingerprint:
    """Legacy recursive fingerprint retained as a scan-only test oracle."""
    counts: Counter[FeatureToken] = Counter()
    walk_local_structure(cdef, _FeatureCounter(counts), mode="target-full")
    return DefinitionFingerprint(dict(counts))


def target_local_fingerprint(cdef: ConcreteDefinition) -> DefinitionFingerprint:
    counts: Counter[FeatureToken] = Counter()
    walk_local_structure(cdef, _FeatureCounter(counts, include_edge_class=True), mode="target-local")
    return DefinitionFingerprint(dict(counts))


def legacy_selector_requirements(
        selector: Any,
        *,
        class_match: ClassMatchPolicy = "selector") -> tuple[FeatureRequirement, ...]:
    """Legacy recursive selector requirements retained as a scan-only test oracle."""
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
            edge_kind: EdgeKind = EdgeKind.MATERIALIZE,
            unordered: bool = False) -> None:
        _add(self.counts, f"CDEF_EDGE_AT_PATH:{edge_kind.value}", path, None)
        if edge_kind is EdgeKind.MATERIALIZE:
            _add(self.counts, "CDEF_EDGE_AT_PATH", path, None)
        if isinstance(definition, ConcreteDefinition):
            _add(self.counts, f"CDEF_EDGE_EXACT:{edge_kind.value}", path, definition.stable_hash())
            if edge_kind is EdgeKind.MATERIALIZE:
                _add(self.counts, "CDEF_EDGE_EXACT", path, definition.stable_hash())
            if self.include_edge_class:
                _add(self.counts, f"CDEF_EDGE_CLASS:{edge_kind.value}", path, canonical_class_key(definition.cls))
                if edge_kind is EdgeKind.MATERIALIZE:
                    _add(self.counts, "CDEF_EDGE_CLASS", path, canonical_class_key(definition.cls))

def legacy_requirements_satisfied(fingerprint: DefinitionFingerprint, requirements: tuple[FeatureRequirement, ...]) -> bool:
    for req in requirements:
        if fingerprint.counts.get(req.token, 0) < req.count:
            return False
    return True
