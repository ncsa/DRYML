from __future__ import annotations

from collections import Counter
from typing import Any

from ..definition import ConcreteDefinition, Definition
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..object import Object
from ..symbol import maybe_symbol_ref
from ..utils.stable_hash import stable_hash_function
from .model import (
    ClassMatchPolicy,
    DefinitionFingerprint,
    ExactSubtreeConstraint,
    FeatureRequirement,
    FeatureToken,
)
from .path import Arg, DefinitionPath, Index, Key, Kwarg, SetMember, iter_set_members


def canonical_class_key(cls: Any) -> str:
    ref = maybe_symbol_ref(cls, functions=False)
    if ref is not None:
        return stable_hash_function(ref)
    return stable_hash_function(cls)


def scalar_key(value: Any) -> str | None:
    try:
        return stable_hash_function(value)
    except TypeError:
        return None


def target_fingerprint(cdef: ConcreteDefinition) -> DefinitionFingerprint:
    counts: Counter[FeatureToken] = Counter()
    _fingerprint_target_value(cdef, DefinitionPath(), counts)
    return DefinitionFingerprint(dict(counts))


def target_local_fingerprint(cdef: ConcreteDefinition) -> DefinitionFingerprint:
    counts: Counter[FeatureToken] = Counter()
    _fingerprint_target_cdef_local(cdef, DefinitionPath(), counts)
    return DefinitionFingerprint(dict(counts))


def selector_requirements(
        selector: Any,
        *,
        class_match: ClassMatchPolicy = "selector") -> tuple[FeatureRequirement, ...]:
    counts: Counter[FeatureToken] = Counter()
    _fingerprint_selector_value(selector, DefinitionPath(), counts, class_match=class_match)
    return tuple(
        FeatureRequirement(token, count)
        for token, count in sorted(counts.items(), key=lambda item: repr(item[0]))
    )


def selector_local_requirements(
        selector: Any,
        *,
        class_match: ClassMatchPolicy = "selector") -> tuple[FeatureRequirement, ...]:
    counts: Counter[FeatureToken] = Counter()
    _fingerprint_selector_value(selector, DefinitionPath(), counts, class_match=class_match, local=True)
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


def _container_family(value: Any) -> str | None:
    families = _target_container_families(value)
    return families[0] if families else None


def _target_container_families(value: Any) -> tuple[str, ...]:
    if isinstance(value, FrozenList):
        # FrozenList is a tuple subclass. Current selector matching lets both
        # list and tuple selectors match it, so target fingerprints must expose
        # both families to avoid false negatives.
        return ("list", "tuple")
    if isinstance(value, list):
        return ("list",)
    if isinstance(value, (tuple, FrozenTuple)):
        return ("tuple",)
    if isinstance(value, (set, frozenset, FrozenSet)):
        return ("set",)
    if isinstance(value, (dict, FrozenDict)):
        return ("dict",)
    return ()


def _selector_container_families(value: Any) -> tuple[str, ...]:
    if isinstance(value, FrozenList):
        # A FrozenList selector is compatible with multiple sequence families
        # through Python tuple subclassing. The current inverted index only
        # supports conjunctive requirements, so no single family token is safe.
        return ()
    if isinstance(value, list):
        return ("list",)
    if isinstance(value, (tuple, FrozenTuple)):
        return ("tuple",)
    if isinstance(value, (set, frozenset, FrozenSet)):
        return ("set",)
    if isinstance(value, (dict, FrozenDict)):
        return ("dict",)
    return ()


def _mapping_items(value: Any):
    if isinstance(value, (dict, FrozenDict)):
        return value.items()
    return None


def _sequence_items(value: Any):
    if isinstance(value, (list, tuple, FrozenList, FrozenTuple)):
        return enumerate(value)
    return None


def _fingerprint_target_value(value: Any, path: DefinitionPath, counts: Counter[FeatureToken]) -> None:
    if isinstance(value, Object):
        value = value.definition

    if isinstance(value, ConcreteDefinition):
        _add(counts, "EXACT_SUBTREE", path, value.stable_hash())
        _add(counts, "CLASS_KEY", path, canonical_class_key(value.cls))
        for idx, child in enumerate(value.args):
            _fingerprint_target_value(child, path.child(Arg(idx)), counts)
        for key, child in value.kwargs.items():
            _add(counts, "HAS_KWARG", path, key)
            _fingerprint_target_value(child, path.child(Kwarg(key)), counts)
        return

    families = _target_container_families(value)
    if families:
        for family in families:
            _add(counts, "CONTAINER_KIND", path, family)
        primary_family = families[0]
        if primary_family in {"list", "tuple", "set"}:
            _add(counts, "SEQUENCE_LENGTH", path, len(value))
        if primary_family == "dict":
            for key, child in _mapping_items(value):
                key_hash = scalar_key(key)
                if key_hash is not None:
                    _add(counts, "HAS_MAPPING_KEY", path, key_hash)
                _fingerprint_target_value(child, path.child(Key(key)), counts)
        elif primary_family != "set":
            for idx, child in _sequence_items(value):
                _fingerprint_target_value(child, path.child(Index(idx)), counts)
        return

    key = scalar_key(value)
    if key is not None:
        _add(counts, "SCALAR_VALUE", path, key)


def _fingerprint_target_cdef_local(cdef: ConcreteDefinition, path: DefinitionPath, counts: Counter[FeatureToken]) -> None:
    _add(counts, "EXACT_NODE", path, cdef.stable_hash())
    _add(counts, "EXACT_SUBTREE", path, cdef.stable_hash())
    _add(counts, "CLASS_KEY", path, canonical_class_key(cdef.cls))
    for idx, child in enumerate(cdef.args):
        _fingerprint_target_local_value(child, path.child(Arg(idx)), counts)
    for key, child in cdef.kwargs.items():
        _add(counts, "HAS_KWARG", path, key)
        _fingerprint_target_local_value(child, path.child(Kwarg(key)), counts)


def _fingerprint_target_local_value(value: Any, path: DefinitionPath, counts: Counter[FeatureToken]) -> None:
    if isinstance(value, Object):
        value = value.definition

    if isinstance(value, ConcreteDefinition):
        _add(counts, "CDEF_EDGE_AT_PATH", path, None)
        _add(counts, "CDEF_EDGE_EXACT", path, value.stable_hash())
        _add(counts, "CDEF_EDGE_CLASS", path, canonical_class_key(value.cls))
        return

    families = _target_container_families(value)
    if families:
        for family in families:
            _add(counts, "CONTAINER_KIND", path, family)
        primary_family = families[0]
        if primary_family in {"list", "tuple", "set"}:
            _add(counts, "SEQUENCE_LENGTH", path, len(value))
        if primary_family == "dict":
            for key, child in _mapping_items(value):
                key_hash = scalar_key(key)
                if key_hash is not None:
                    _add(counts, "HAS_MAPPING_KEY", path, key_hash)
                _fingerprint_target_local_value(child, path.child(Key(key)), counts)
        elif primary_family == "set":
            for seg, child in iter_set_members(value):
                _fingerprint_target_local_value(child, path.child(seg), counts)
        else:
            for idx, child in _sequence_items(value):
                _fingerprint_target_local_value(child, path.child(Index(idx)), counts)
        return

    key = scalar_key(value)
    if key is not None:
        _add(counts, "SCALAR_VALUE", path, key)


def _fingerprint_selector_value(
        value: Any,
        path: DefinitionPath,
        counts: Counter[FeatureToken],
        *,
        class_match: ClassMatchPolicy,
        local: bool = False) -> None:
    from ..utils.types import is_nonclass_callable

    if isinstance(value, Object):
        value = value.definition

    if isinstance(value, ConcreteDefinition):
        if local and path:
            _add(counts, "CDEF_EDGE_AT_PATH", path, None)
            _add(counts, "CDEF_EDGE_EXACT", path, value.stable_hash())
        else:
            _add(counts, "EXACT_SUBTREE", path, value.stable_hash())
        return

    if isinstance(value, Definition):
        if value.cls is not None and class_match == "exact" and not is_nonclass_callable(value.cls):
            try:
                _add(counts, "CLASS_KEY", path, canonical_class_key(value.cls))
            except TypeError:
                pass
        if value.args is not None:
            for idx, child in enumerate(value.args):
                _fingerprint_selector_value(child, path.child(Arg(idx)), counts, class_match=class_match, local=local)
        for key, child in value.kwargs.items():
            _add(counts, "HAS_KWARG", path, key)
            _fingerprint_selector_value(child, path.child(Kwarg(key)), counts, class_match=class_match, local=local)
        return

    families = _selector_container_families(value)
    if families or isinstance(value, FrozenList):
        primary_family = families[0] if families else "list"
        for family in families:
            _add(counts, "CONTAINER_KIND", path, family)
        if primary_family in {"list", "tuple", "set"}:
            _add(counts, "SEQUENCE_LENGTH", path, len(value))
        if primary_family == "dict":
            for key, child in _mapping_items(value):
                key_hash = scalar_key(key)
                if key_hash is not None:
                    _add(counts, "HAS_MAPPING_KEY", path, key_hash)
                _fingerprint_selector_value(child, path.child(Key(key)), counts, class_match=class_match, local=local)
        elif primary_family != "set":
            for idx, child in _sequence_items(value):
                _fingerprint_selector_value(child, path.child(Index(idx)), counts, class_match=class_match, local=local)
        elif local:
            for seg, child in iter_set_members(value):
                if isinstance(child, Object):
                    child = child.definition
                if isinstance(child, (ConcreteDefinition, Definition)):
                    _fingerprint_selector_value(child, path.child(seg), counts, class_match=class_match, local=local)
        return

    if is_nonclass_callable(value):
        return

    key = scalar_key(value)
    if key is not None:
        _add(counts, "SCALAR_VALUE", path, key)


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
