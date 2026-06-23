from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ..definition import ConcreteDefinition, Definition, categorical_definition, selector_match
from ..object import Object
from .fingerprint import collect_exact_constraints, selector_requirements
from .model import (
    ClassMatchPolicy,
    DefinitionOccurrence,
    QueryDomain,
    QueryDomainError,
    QueryExplanation,
    QueryProjection,
    QueryStats,
    RefreshPolicy,
    ResultUniverse,
)
from .path import DefinitionPathLike, QueryPathError, get_subtree, normalize_path, replace_subtree
from .result import DefinitionResultSet, ObjectResultSet, OccurrenceResultSet


@dataclass(frozen=True, slots=True)
class DefinitionQuery:
    repo: Any
    original: Definition | ConcreteDefinition | None
    selector: Definition | ConcreteDefinition | None
    domain: QueryDomain | None = None
    projection: QueryProjection | None = None
    class_match_policy: ClassMatchPolicy = "selector"
    strict_policy: bool = False
    refresh_policy: RefreshPolicy = "auto"
    reuse_weak_policy: bool = True
    universe: ResultUniverse | None = None

    @classmethod
    def from_source(
            cls,
            repo,
            source=None,
            *,
            domain: QueryDomain | None = None,
            universe: ResultUniverse | None = None) -> "DefinitionQuery":
        original = _snapshot_source(source)
        return cls(repo=repo, original=original, selector=original, domain=domain, universe=universe)

    def categorical(
            self,
            *,
            path: DefinitionPathLike = "$",
            recursive: bool = False) -> "DefinitionQuery":
        if self.selector is None:
            raise QueryPathError("Cannot apply categorical() to an unconstrained query.")
        norm = normalize_path(path)
        subtree = get_subtree(self.selector, norm)
        projected = categorical_definition(subtree, recursive=recursive)
        return replace(self, selector=replace_subtree(self.selector, norm, projected))

    def restore(self, *, path: DefinitionPathLike = "$") -> "DefinitionQuery":
        if self.original is None or self.selector is None:
            raise QueryPathError("Cannot restore() on an unconstrained query.")
        norm = normalize_path(path)
        subtree = get_subtree(self.original, norm)
        replacement = deepcopy(subtree) if isinstance(subtree, Definition) else subtree
        return replace(self, selector=replace_subtree(self.selector, norm, replacement))

    def exact(
            self,
            definition: ConcreteDefinition | Object | None = None,
            *,
            path: DefinitionPathLike = "$") -> "DefinitionQuery":
        if self.selector is None:
            raise QueryPathError("Cannot apply exact() to an unconstrained query.")
        norm = normalize_path(path)
        if definition is None:
            if self.original is None:
                raise QueryPathError(f"Cannot infer exact subtree at {norm!s}; query has no original source.")
            definition = get_subtree(self.original, norm)
        if isinstance(definition, Object):
            definition = definition.definition
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError(f"Exact constraint at {norm!s} requires a ConcreteDefinition, got {type(definition).__name__}.")
        return replace(self, selector=replace_subtree(self.selector, norm, definition))

    def class_match(self, policy: ClassMatchPolicy) -> "DefinitionQuery":
        if policy not in {"selector", "exact"}:
            raise ValueError("class_match policy must be 'selector' or 'exact'.")
        return replace(self, class_match_policy=policy)

    def strict(self, enabled: bool = True) -> "DefinitionQuery":
        return replace(self, strict_policy=bool(enabled))

    def refresh(self, policy: RefreshPolicy = "auto") -> "DefinitionQuery":
        if policy not in {False, "auto", True}:
            raise ValueError("refresh policy must be False, 'auto', or True.")
        return replace(self, refresh_policy=policy)

    def reuse_weak(self, enabled: bool = True) -> "DefinitionQuery":
        return replace(self, reuse_weak_policy=bool(enabled))

    def stored(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        q = replace(self, domain="stored", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def cached(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        q = replace(self, domain="cached", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def known(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        q = replace(self, domain="known", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def nested(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        q = replace(self, domain="nested", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def definitions(self) -> "DefinitionQuery":
        if self.domain != "nested":
            return self
        return replace(self, projection="definitions")

    def owners(self) -> "DefinitionQuery":
        if self.domain != "nested":
            raise QueryDomainError("owners() is only valid for nested queries.")
        return replace(self, projection="owners")

    def execute(self):
        self._require_domain()
        if self.domain == "nested":
            occs, stats = self._execute_nested_occurrences()
            explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
            raw = OccurrenceResultSet(self.repo, occs, explanation=explanation)
            if self.projection == "definitions":
                return raw.definitions()
            if self.projection == "owners":
                return raw.owners()
            return raw

        cdefs, stats = self._execute_definition_domain()
        explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
        materializable = True
        domain = self.domain or "stored"
        if self.universe is not None:
            materializable = self.universe.materializable
            domain = self.universe.domain
        return DefinitionResultSet(
            self.repo,
            cdefs,
            materializable=materializable,
            domain=domain,
            explanation=explanation,
        )

    def defs(self):
        result = self.execute()
        if isinstance(result, DefinitionResultSet):
            return result
        raise QueryDomainError("Raw nested queries return occurrences; use .definitions().defs() or .owners().defs().")

    def objects(self, **load_options) -> ObjectResultSet:
        result = self.execute()
        if isinstance(result, OccurrenceResultSet):
            raise QueryDomainError("Raw nested occurrences cannot be materialized directly; use .owners().objects().")
        return result.objects(**load_options)

    def count(self) -> int:
        return self.execute().count()

    def exists(self) -> bool:
        return self.count() > 0

    def explain(self) -> QueryExplanation:
        return self.execute().explanation

    def _require_domain(self) -> None:
        if self.domain is None:
            raise QueryDomainError("Select a query domain with stored(), cached(), known(), or nested() before executing.")

    def _domain_label(self) -> str:
        if self.domain == "nested" and self.projection is not None:
            return f"nested-{self.projection}"
        return self.domain or "unset"

    def _execute_definition_domain(self) -> tuple[tuple[ConcreteDefinition, ...], QueryStats]:
        stats = QueryStats()
        if self.universe is not None:
            if self.universe.kind != "definitions":
                raise QueryDomainError("A definition terminal cannot execute over an occurrence universe.")
            stats.universe_size = len(self.universe.definitions)
            matches = self._verify_cdefs(tuple(self.universe.definitions), stats=stats)
            stats.result_count = len(matches)
            return matches, stats

        catalog = self.repo._query_catalog
        exact_root = self.selector if isinstance(self.selector, ConcreteDefinition) else None
        if self.refresh_policy is True:
            catalog.refresh(True, stats=stats)
        elif exact_root is not None and self.domain in {"stored", "known"}:
            catalog.ensure_exact_stored(exact_root, stats=stats)
        elif self.domain in {"stored", "known"}:
            catalog.refresh(self.refresh_policy, stats=stats)

        if self.domain == "stored":
            universe_ids = catalog.stored_ids()
        elif self.domain == "cached":
            universe_ids = catalog.cached_ids(reuse_weak=self.reuse_weak_policy)
        elif self.domain == "known":
            universe_ids = catalog.known_ids(reuse_weak=self.reuse_weak_policy)
        else:
            raise QueryDomainError(f"Unsupported definition domain {self.domain!r}.")

        stats.universe_size = len(universe_ids)
        requirements = selector_requirements(self.selector, class_match=self.class_match_policy) if self.selector is not None else ()
        candidate_ids = catalog.candidate_ids(universe_ids, requirements, stats=stats)
        cdefs = catalog.ids_to_cdefs(candidate_ids)
        matches = self._verify_cdefs(cdefs, stats=stats)
        stats.result_count = len(matches)
        return matches, stats

    def _execute_nested_occurrences(self) -> tuple[tuple[DefinitionOccurrence, ...], QueryStats]:
        stats = QueryStats()
        if self.universe is not None:
            if self.universe.kind != "occurrences":
                raise QueryDomainError("A nested query cannot execute over a definition universe.")
            stats.universe_size = len(self.universe.occurrences)
            verified_nested = self._verify_cdefs(
                tuple({occ.definition for occ in self.universe.occurrences}),
                stats=stats,
            )
            verified = set(verified_nested)
            out = tuple(occ for occ in self.universe.occurrences if occ.definition in verified)
            stats.result_count = len(out)
            return out, stats

        catalog = self.repo._query_catalog
        catalog.refresh(self.refresh_policy, stats=stats)
        universe_ids = catalog.nested_ids()
        stats.universe_size = len(universe_ids)
        requirements = selector_requirements(self.selector, class_match=self.class_match_policy) if self.selector is not None else ()
        candidate_ids = catalog.candidate_ids(universe_ids, requirements, stats=stats)
        cdefs = catalog.ids_to_cdefs(candidate_ids)
        matches = self._verify_cdefs(cdefs, stats=stats)
        match_ids = {catalog.cdef_id(cdef) for cdef in matches}
        match_ids.discard(None)
        occurrences = catalog.occurrences_for_nested_ids(match_ids)
        stats.result_count = len(occurrences)
        return occurrences, stats

    def _verify_cdefs(
            self,
            cdefs: tuple[ConcreteDefinition, ...],
            *,
            stats: QueryStats) -> tuple[ConcreteDefinition, ...]:
        if self.selector is None:
            stats.verified_count += len(cdefs)
            return tuple(sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))

        constraints = collect_exact_constraints(self.selector)
        out: list[ConcreteDefinition] = []
        for cdef in cdefs:
            stats.verified_count += 1
            if not _structural_match(
                    self.selector,
                    cdef,
                    strict=self.strict_policy,
                    class_match=self.class_match_policy):
                continue
            if not _exact_constraints_match(cdef, constraints):
                continue
            out.append(cdef)
        return tuple(sorted(out, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))


def _snapshot_source(source):
    if source is None:
        return None
    if isinstance(source, Object):
        return source.definition
    if isinstance(source, ConcreteDefinition):
        return source
    if isinstance(source, Definition):
        return deepcopy(source)
    raise TypeError(f"Query source must be Definition, ConcreteDefinition, Object, or None, not {type(source).__name__}.")


def _structural_match(selector, cdef: ConcreteDefinition, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    if not selector_match(selector, cdef, strict=strict):
        return False
    if class_match == "exact" and not _exact_class_match(selector, cdef):
        return False
    return True


def _exact_class_match(selector, target) -> bool:
    from .fingerprint import canonical_class_key
    from ..freeze import FrozenDict, FrozenList, FrozenTuple

    if isinstance(selector, Object):
        selector = selector.definition
    if isinstance(target, Object):
        target = target.definition

    if isinstance(selector, ConcreteDefinition):
        return True

    if isinstance(selector, Definition):
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        if selector.cls is not None:
            try:
                if canonical_class_key(selector.cls) != canonical_class_key(target.cls):
                    return False
            except TypeError:
                pass
        if selector.args is not None:
            if target.args is None or len(selector.args) != len(target.args):
                return False
            for sel_child, tgt_child in zip(selector.args, target.args):
                if not _exact_class_match(sel_child, tgt_child):
                    return False
        for key, sel_child in selector.kwargs.items():
            if key not in target.kwargs:
                return False
            if not _exact_class_match(sel_child, target.kwargs[key]):
                return False
        return True

    if isinstance(selector, (dict, FrozenDict)):
        if not isinstance(target, (dict, FrozenDict)):
            return False
        for key, sel_child in selector.items():
            if key not in target:
                return False
            if not _exact_class_match(sel_child, target[key]):
                return False
        return True

    if isinstance(selector, (list, tuple, FrozenList, FrozenTuple)):
        if not isinstance(target, (list, tuple, FrozenList, FrozenTuple)):
            return False
        if len(selector) != len(target):
            return False
        for sel_child, tgt_child in zip(selector, target):
            if not _exact_class_match(sel_child, tgt_child):
                return False
        return True

    # Set selectors are matched unordered by selector_match; no path-stable class
    # constraint can be derived here without reimplementing the set matcher.
    return True


def _exact_constraints_match(cdef: ConcreteDefinition, constraints) -> bool:
    for constraint in constraints:
        try:
            candidate = get_subtree(cdef, constraint.path)
        except QueryPathError:
            return False
        if not isinstance(candidate, ConcreteDefinition):
            return False
        if candidate.stable_hash() != constraint.cdef.stable_hash():
            return False
        if candidate != constraint.cdef:
            return False
    return True
