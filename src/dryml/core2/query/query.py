from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from ..canonical import matching_container_family
from ..cdef_identity import same_cdef
from ..definition import ConcreteDefinition, Definition, categorical_definition, selector_match
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..object import Object
from ..symbol import maybe_symbol_ref, resolve_symbol
from ..utils.types import is_nonclass_callable
from .fingerprint import collect_exact_constraints, selector_requirements
from .graph_plan import graph_candidate_ids
from .domain import CachedDomain, KnownDomain, NestedDomain, StoredDomain
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
from .selector_graph import compile_selector_graph


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
        self._check_universe_domain_switch("stored")
        q = replace(self, domain="stored", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def cached(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("cached")
        q = replace(self, domain="cached", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def known(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("known")
        q = replace(self, domain="known", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def nested(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._check_universe_domain_switch("nested")
        q = replace(self, domain="nested", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def _check_universe_domain_switch(self, requested: str) -> None:
        if self.universe is None:
            return
        if requested != self.universe.domain:
            raise QueryDomainError(
                f"Cannot switch a fixed ResultSet universe from domain {self.universe.domain!r} to {requested!r}."
            )

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
            if self.universe is None and self.projection == "definitions":
                cdefs, stats = self._execute_nested_definitions()
                explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=False,
                    domain="nested-definitions",
                    explanation=explanation,
                )
            if self.universe is None and self.projection == "owners":
                cdefs, stats = self._execute_nested_owners()
                explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=True,
                    domain="owners",
                    explanation=explanation,
                )
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
        replicas = None
        if self.universe is not None and self.universe.replicas is not None:
            replicas = {cdef: self.universe.replicas.get(cdef, ()) for cdef in cdefs}
        return DefinitionResultSet(
            self.repo,
            cdefs,
            materializable=materializable,
            domain=domain,
            explanation=explanation,
            replicas=replicas,
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
        elif exact_root is not None and self.domain in {"stored", "known"} and self.refresh_policy is not False:
            catalog.ensure_exact_stored(exact_root, stats=stats)
        elif self.domain in {"stored", "known"}:
            catalog.refresh(self.refresh_policy, stats=stats)

        if exact_root is not None and self.domain in {"stored", "cached", "known"}:
            candidate_ids = self._exact_domain_candidate_ids(catalog, exact_root)
            stats.universe_size = len(candidate_ids)
            stats.candidate_count = len(candidate_ids)
            cdefs = catalog.ids_to_cdefs(candidate_ids)
            matches = self._verify_cdefs(cdefs, stats=stats)
            stats.result_count = len(matches)
            return matches, stats

        domain = self._definition_domain(catalog)
        domain.prepare(stats=stats)
        stats.universe_size = domain.estimate_size() or 0
        selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
        if selector_graph is not None:
            candidate_ids = graph_candidate_ids(catalog, selector_graph, domain, stats=stats)
        else:
            requirements = selector_requirements(self.selector, class_match=self.class_match_policy) if self.selector is not None else ()
            candidate_ids = catalog.candidate_ids_for_domain(domain, requirements, stats=stats)
        cdefs = catalog.ids_to_cdefs(candidate_ids)
        matches = self._verify_cdefs(cdefs, stats=stats)
        stats.result_count = len(matches)
        return matches, stats

    def _definition_domain(self, catalog):
        if self.domain == "stored":
            return StoredDomain(catalog)
        if self.domain == "cached":
            return CachedDomain(catalog, reuse_weak=self.reuse_weak_policy)
        if self.domain == "known":
            return KnownDomain(catalog, reuse_weak=self.reuse_weak_policy)
        raise QueryDomainError(f"Unsupported definition domain {self.domain!r}.")

    def _exact_domain_candidate_ids(self, catalog, exact_root: ConcreteDefinition) -> set[str]:
        if self.domain in {"cached", "known"} and self.repo.get_cached(exact_root, reuse_weak=self.reuse_weak_policy) is not None:
            catalog.register_cached(exact_root)
        exact_ids = catalog.exact_ids(exact_root)
        if self.domain == "cached":
            if self.repo.get_cached(exact_root, reuse_weak=self.reuse_weak_policy) is None:
                return set()
            return exact_ids
        if self.domain == "stored":
            return catalog.filter_stored_ids(exact_ids)
        return {
            did
            for did in exact_ids
            if catalog.is_stored_id(did)
            or self.repo.get_cached(catalog.definition_for_id(did), reuse_weak=self.reuse_weak_policy) is not None
        }

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
        selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
        if selector_graph is not None:
            candidate_ids = graph_candidate_ids(catalog, selector_graph, None, stats=stats)
            candidate_ids = catalog.filter_nested_ids(candidate_ids)
            stats.universe_size = len(candidate_ids)
            stats.candidate_count = len(candidate_ids)
        else:
            domain = NestedDomain(catalog)
            requirements = selector_requirements(self.selector, class_match=self.class_match_policy) if self.selector is not None else ()
            if requirements:
                candidate_ids = catalog.candidate_ids_for_domain(domain, requirements, stats=stats)
            else:
                candidate_ids = domain.all_ids()
                stats.candidate_count = len(candidate_ids)
            stats.universe_size = domain.estimate_size() or len(candidate_ids)
        cdefs = catalog.ids_to_cdefs(candidate_ids)
        matches = self._verify_cdefs(cdefs, stats=stats)
        match_ids = {catalog.cdef_id(cdef) for cdef in matches}
        match_ids.discard(None)
        occurrences = tuple(catalog.iter_occurrences_for_nested_ids(match_ids))
        stats.result_count = len(occurrences)
        return occurrences, stats

    def _execute_nested_definitions(self) -> tuple[tuple[ConcreteDefinition, ...], QueryStats]:
        matches, _, stats = self._execute_nested_definition_matches()
        stats.result_count = len(matches)
        return matches, stats

    def _execute_nested_owners(self) -> tuple[tuple[ConcreteDefinition, ...], QueryStats]:
        matches, match_ids, stats = self._execute_nested_definition_matches()
        catalog = self.repo._query_catalog
        owner_ids = catalog.owner_ids_for_nested_ids(match_ids)
        owners = catalog.ids_to_cdefs(owner_ids)
        stats.result_count = len(owners)
        return tuple(sorted(owners, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))), stats

    def _execute_nested_definition_matches(self) -> tuple[tuple[ConcreteDefinition, ...], set[str], QueryStats]:
        stats = QueryStats()
        catalog = self.repo._query_catalog
        catalog.refresh(self.refresh_policy, stats=stats)
        selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
        if selector_graph is not None:
            candidate_ids = graph_candidate_ids(catalog, selector_graph, None, stats=stats)
            candidate_ids = catalog.filter_nested_ids(candidate_ids)
            stats.universe_size = len(candidate_ids)
            stats.candidate_count = len(candidate_ids)
        else:
            domain = NestedDomain(catalog)
            requirements = selector_requirements(self.selector, class_match=self.class_match_policy) if self.selector is not None else ()
            if requirements:
                candidate_ids = catalog.candidate_ids_for_domain(domain, requirements, stats=stats)
            else:
                candidate_ids = domain.all_ids()
                stats.candidate_count = len(candidate_ids)
            stats.universe_size = domain.estimate_size() or len(candidate_ids)
        cdefs = catalog.ids_to_cdefs(candidate_ids)
        matches = self._verify_cdefs(cdefs, stats=stats)
        match_ids = {catalog.cdef_id(cdef) for cdef in matches}
        match_ids.discard(None)
        return matches, match_ids, stats

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
    if not _query_match(selector, cdef, strict=strict, class_match=class_match):
        return False
    return True


def _query_match(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    """Query-layer verifier: selector semantics plus exact ConcreteDefinition anchors."""
    if isinstance(selector, Object):
        selector = selector.definition
    if isinstance(target, Object):
        target = target.definition

    if isinstance(selector, ConcreteDefinition):
        return isinstance(target, ConcreteDefinition) and same_cdef(selector, target)

    if isinstance(selector, Definition):
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        if selector.cls is not None:
            if not _query_match_class(selector.cls, target.cls, strict=strict, class_match=class_match):
                return False
        if selector.args is not None:
            if target.args is None:
                return False
            if not _query_match(selector.args, target.args, strict=strict, class_match=class_match):
                return False
        for key, child in selector.kwargs.items():
            if key not in target.kwargs:
                return False
            if not _query_match(child, target.kwargs[key], strict=strict, class_match=class_match):
                return False
        return True

    if isinstance(selector, (dict, FrozenDict)):
        if not isinstance(target, (dict, FrozenDict)):
            return False
        for key, child in selector.items():
            if key not in target:
                return False
            if not _query_match(child, target[key], strict=strict, class_match=class_match):
                return False
        return True

    family = matching_container_family(selector, target)
    if family in {"list", "tuple"}:
        if len(selector) != len(target):
            return False
        return all(
            _query_match(sel_child, tgt_child, strict=strict, class_match=class_match)
            for sel_child, tgt_child in zip(selector, target)
        )

    if family == "set":
        return _unordered_match(selector, target, lambda sel_child, tgt_child: _query_match(
            sel_child,
            tgt_child,
            strict=strict,
            class_match=class_match,
        ))

    return _query_match_leaf(selector, target, strict=strict, class_match=class_match)


def _query_match_class(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    selector_ref = maybe_symbol_ref(selector, functions=False)
    target_ref = maybe_symbol_ref(target, functions=False)
    if selector_ref is not None and target_ref is not None:
        if selector_ref == target_ref:
            return True
        if strict or class_match == "exact":
            return False
        try:
            selector_obj = resolve_symbol(selector_ref)
            target_obj = resolve_symbol(target_ref)
        except Exception:
            return False
        if isinstance(selector_obj, type) and isinstance(target_obj, type):
            return issubclass(target_obj, selector_obj)
        return False

    if isinstance(selector, type) and isinstance(target, type):
        if strict or class_match == "exact":
            return selector is target
        return issubclass(target, selector)

    return _query_match_leaf(selector, target, strict=strict, class_match=class_match)


def _query_match_leaf(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    if is_nonclass_callable(selector):
        if strict:
            raise TypeError("Callable selectors are not allowed in strict query matching.")
        return bool(selector(target))

    selector_ref = maybe_symbol_ref(selector, functions=False)
    target_ref = maybe_symbol_ref(target, functions=False)
    if selector_ref is not None and target_ref is not None:
        if selector_ref == target_ref:
            return True
        if strict or class_match == "exact":
            return False
        try:
            selector_obj = resolve_symbol(selector_ref)
            target_obj = resolve_symbol(target_ref)
        except Exception:
            return False
        if isinstance(selector_obj, type) and isinstance(target_obj, type):
            return issubclass(target_obj, selector_obj)
        return False

    try:
        return selector_match(selector, target, strict=strict)
    except TypeError:
        return False


def _unordered_match(selector_values, target_values, edge_predicate) -> bool:
    selector_list = list(selector_values)
    target_list = list(target_values)
    if len(selector_list) != len(target_list):
        return False

    edges = [
        [idx for idx, tgt in enumerate(target_list) if edge_predicate(sel, tgt)]
        for sel in selector_list
    ]
    order = sorted(range(len(selector_list)), key=lambda idx: len(edges[idx]))
    matched_to_selector: dict[int, int] = {}

    def augment(sel_idx: int, seen: set[int]) -> bool:
        for tgt_idx in edges[sel_idx]:
            if tgt_idx in seen:
                continue
            seen.add(tgt_idx)
            if tgt_idx not in matched_to_selector or augment(matched_to_selector[tgt_idx], seen):
                matched_to_selector[tgt_idx] = sel_idx
                return True
        return False

    for sel_idx in order:
        if not augment(sel_idx, set()):
            return False
    return True


def _exact_class_match(selector, target) -> bool:
    from .fingerprint import canonical_class_key
    from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
    from ..symbol import maybe_symbol_ref

    if isinstance(selector, Object):
        selector = selector.definition
    if isinstance(target, Object):
        target = target.definition

    selector_ref = maybe_symbol_ref(selector, functions=False)
    target_ref = maybe_symbol_ref(target, functions=False)
    if selector_ref is not None and target_ref is not None:
        try:
            return canonical_class_key(selector) == canonical_class_key(target)
        except TypeError:
            return True

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

    if isinstance(selector, (set, frozenset, FrozenSet)):
        if not isinstance(target, (set, frozenset, FrozenSet)):
            return False
        return _unordered_match(selector, target, lambda sel_child, tgt_child: selector_match(
            sel_child,
            tgt_child,
            strict=False,
        ) and _exact_class_match(sel_child, tgt_child))

    return True


def _exact_constraints_match(cdef: ConcreteDefinition, constraints) -> bool:
    for constraint in constraints:
        try:
            candidate = get_subtree(cdef, constraint.path)
        except QueryPathError:
            return False
        if constraint.unordered_member:
            if not _unordered_set_contains_exact(candidate, constraint.cdef):
                return False
            continue
        if not isinstance(candidate, ConcreteDefinition):
            return False
        if candidate.stable_hash() != constraint.cdef.stable_hash():
            return False
        if candidate != constraint.cdef:
            return False
    return True


def _unordered_set_contains_exact(candidate, expected: ConcreteDefinition) -> bool:
    from ..freeze import FrozenSet

    if not isinstance(candidate, (set, frozenset, FrozenSet)):
        return False
    for member in candidate:
        if isinstance(member, Object):
            member = member.definition
        if not isinstance(member, ConcreteDefinition):
            continue
        if same_cdef(member, expected):
            return True
    return False
