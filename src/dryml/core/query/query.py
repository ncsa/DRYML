from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cmp_to_key
from typing import Any

from ..canonical import matching_container_family
from ..definition import ConcreteDefinition, Definition, selector_match
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..links import DefLink
from ..object import Object
from ..params import Match
from ..quoted import QuotedDef, SelectorSpec
from ..selector import Selector
from ..symbol import maybe_symbol_ref, resolve_symbol
from ..errors import ParameterizationLimitError
from ..utils.types import is_nonclass_callable
from .model import (
    ClassMatchPolicy,
    QueryDomainError,
    QueryExplanation,
    QueryCardinalityError,
    QueryIndexUnavailable,
    QueryVerifyBudgetExceeded,
    QueryStats,
    QueryWouldScanError,
    RefreshPolicy,
)
from .path import (
    DefinitionPath,
    Kwarg,
    Parameter,
    QueryPathError,
    iter_value_edges,
)
from .utils import cdef_equal


_DEFAULT_GENERATOR_WITNESS_LIMIT = 65_536


@dataclass(slots=True)
class _GeneratorWitnessBudget:
    limit: int | None
    visited: int = 0

    def consume(self) -> None:
        self.visited += 1
        if self.limit is not None and self.visited > self.limit:
            raise ParameterizationLimitError("generator query witness limit exceeded")


def _query_match(
    selector, target, *, strict: bool, class_match: ClassMatchPolicy
) -> bool:
    """Query-layer verifier: selector semantics plus exact ConcreteDefinition anchors."""
    if isinstance(selector, Object):
        selector = selector.definition
    if isinstance(target, Object):
        target = target.definition
    if isinstance(selector, Selector):
        selector = selector.root

    from ..cdef_graph import EdgeKind

    if (
        isinstance(selector, DefLink)
        and selector.kind is EdgeKind.REF
        and isinstance(selector.target, (QuotedDef, SelectorSpec))
    ):
        selector = selector.target
    if (
        isinstance(target, DefLink)
        and target.kind is EdgeKind.REF
        and isinstance(target.target, (QuotedDef, SelectorSpec))
    ):
        target = target.target

    if isinstance(selector, ConcreteDefinition):
        return isinstance(target, ConcreteDefinition) and cdef_equal(selector, target)

    from ..factory import FactorySpec

    if isinstance(selector, FactorySpec):
        return _query_match_factory(
            selector, target, strict=strict, class_match=class_match
        )

    if isinstance(selector, DefLink):
        if selector.kind is EdgeKind.MATERIALIZE:
            target_value = (
                target.target
                if isinstance(target, DefLink) and target.kind is EdgeKind.MATERIALIZE
                else target
            )
            return _query_match(
                selector.target, target_value, strict=strict, class_match=class_match
            )
        if selector.kind is EdgeKind.REF:
            if not isinstance(target, DefLink) or target.kind is not EdgeKind.REF:
                return False
            return _query_match(
                selector.target, target.target, strict=strict, class_match=class_match
            )
        return False

    if isinstance(selector, (QuotedDef, SelectorSpec)):
        if not isinstance(target, (QuotedDef, SelectorSpec, Selector, Definition)):
            return False
        sel_value = (
            selector.value if isinstance(selector, QuotedDef) else selector.selector
        )
        tgt_value = (
            target.value
            if isinstance(target, QuotedDef)
            else target.selector if isinstance(target, SelectorSpec) else target
        )
        if isinstance(sel_value, Selector):
            sel_value = sel_value.root
        if isinstance(tgt_value, Selector):
            tgt_value = tgt_value.root
        return _query_match(
            sel_value, tgt_value, strict=strict, class_match=class_match
        )

    if isinstance(selector, Match):
        return selector.matches(target, present=True)

    if isinstance(selector, Definition):
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        if selector.cls is not None:
            if not _query_match_class(
                selector.cls, target.cls, strict=strict, class_match=class_match
            ):
                return False
        if isinstance(target, ConcreteDefinition):
            # V2 identities persist semantic names rather than a particular
            # positional/keyword call spelling.  Partial binding deliberately
            # omits defaults, so only supplied parameters constrain matching.
            try:
                selector_parameters = selector.parameters
            except TypeError:
                # Missing/Present selectors may intentionally mention a
                # parameter absent from the live constructor. Bind the known
                # portion, then retain those structural absence constraints.
                from ..bound_args import _constructor_signature, bind_partial_arguments

                if selector.cls is None or not isinstance(selector.cls, type):
                    raise
                signature = _constructor_signature(selector.cls)
                known_kwargs = {
                    name: value
                    for name, value in selector.kwargs.items()
                    if name in signature.parameters
                }
                unknown_kwargs = {
                    name: value
                    for name, value in selector.kwargs.items()
                    if name not in signature.parameters
                }
                args = () if selector.args is None else tuple(selector.args)
                bound = bind_partial_arguments(selector.cls, args, known_kwargs)
                selector_parameters = dict(bound.items())
                selector_parameters.update(unknown_kwargs)
            for name, child in selector_parameters.items():
                if name not in target.parameters:
                    if isinstance(child, Match) and child.matches(None, present=False):
                        continue
                    return False
                if not _query_match(
                    child,
                    target.parameters[name],
                    strict=strict,
                    class_match=class_match,
                ):
                    return False
            return True
        from ..categorical import _is_prepared_selector_definition, _semantic_parameters

        if _is_prepared_selector_definition(selector):
            target_parameters = _semantic_parameters(target)
            for name, child in selector.parameters.items():
                if name not in target_parameters:
                    if isinstance(child, Match) and child.matches(None, present=False):
                        continue
                    return False
                if not _query_match(
                    child,
                    target_parameters[name],
                    strict=strict,
                    class_match=class_match,
                ):
                    return False
            return True
        if selector.args is not None:
            if target.args is None:
                return False
            if not _query_match(
                selector.args, target.args, strict=strict, class_match=class_match
            ):
                return False
        for key, child in selector.kwargs.items():
            if key not in target.kwargs:
                if isinstance(child, Match) and child.matches(None, present=False):
                    continue
                return False
            if not _query_match(
                child, target.kwargs[key], strict=strict, class_match=class_match
            ):
                return False
        return True

    if isinstance(selector, (dict, FrozenDict)):
        if not isinstance(target, (dict, FrozenDict)):
            return False
        for key, child in selector.items():
            if key not in target:
                if isinstance(child, Match) and child.matches(None, present=False):
                    continue
                return False
            if not _query_match(
                child, target[key], strict=strict, class_match=class_match
            ):
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
        return _unordered_match(
            selector,
            target,
            lambda sel_child, tgt_child: _query_match(
            sel_child,
            tgt_child,
            strict=strict,
            class_match=class_match,
            ),
        )

    return _query_match_leaf(selector, target, strict=strict, class_match=class_match)


def _query_match_factory(
    selector, target, *, strict: bool, class_match: ClassMatchPolicy
) -> bool:
    """Verify exact or Match-bearing partial FactorySpec call patterns."""

    from ..factory import FactorySpec, _contains_match

    if not isinstance(target, FactorySpec):
        return False
    if not _query_match(
        selector.target, target.target, strict=strict, class_match=class_match
    ):
        return False
    if len(selector.args) != len(target.args):
        return False
    if not all(
        _query_match(left, right, strict=strict, class_match=class_match)
        for left, right in zip(selector.args, target.args)
    ):
        return False

    is_pattern = _contains_match(selector)
    if not is_pattern and tuple(selector.kwargs) != tuple(target.kwargs):
        return False
    return all(
        key in target.kwargs
        and _query_match(
            value, target.kwargs[key], strict=strict, class_match=class_match
        )
        for key, value in selector.kwargs.items()
    )


def _query_match_class(
    selector, target, *, strict: bool, class_match: ClassMatchPolicy
) -> bool:
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


def _query_match_leaf(
    selector, target, *, strict: bool, class_match: ClassMatchPolicy
) -> bool:
    if is_nonclass_callable(selector):
        if strict:
            raise TypeError(
                "Callable selectors are not allowed in strict query matching."
            )
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
            if tgt_idx not in matched_to_selector or augment(
                matched_to_selector[tgt_idx], seen
            ):
                matched_to_selector[tgt_idx] = sel_idx
                return True
        return False

    for sel_idx in order:
        if not augment(sel_idx, set()):
            return False
    return True


def _projection_origin_paths(
        projected: Any,
        source: Any,
        origins: Any,
) -> dict[DefinitionPath, tuple[DefinitionPath | None, Any]]:
    """Map each projected occurrence to its exact preceding source occurrence.

    The map follows the transformation itself instead of matching values. This
    keeps distinct aliases, interned scalars, and changed set-member addresses
    separate while a query composes multiple projections.
    """

    from ..categorical import _is_prepared_selector_definition, _semantic_parameters

    paths: dict[DefinitionPath, tuple[DefinitionPath | None, Any]] = {}
    active: set[tuple[int, int]] = set()

    def source_parameter_path(value: Any, name: str) -> DefinitionPath | None:
        if isinstance(value, ConcreteDefinition):
            return DefinitionPath((Parameter(name),))
        if _is_prepared_selector_definition(value):
            return DefinitionPath((Kwarg(name),))
        # Authored calls can synthesize positional and variadic semantic
        # buckets, so only the value itself is authoritative at this boundary.
        return None

    def source_edge_path(value: Any, child: Any) -> DefinitionPath:
        matches = [
            edge.segment for edge in iter_value_edges(value) if edge.value is child
        ]
        if len(matches) != 1:
            raise QueryPathError(
                "Semantic projection lost an exact source occurrence correspondence."
            )
        return DefinitionPath((matches[0],))

    def visit(
            projected_value: Any,
            source_value: Any,
            projected_path: DefinitionPath,
        source_path: DefinitionPath | None,
    ) -> None:
        paths[projected_path] = (source_path, source_value)
        projected_edges = iter_value_edges(projected_value)
        if not projected_edges:
            return
        key = (id(projected_value), id(source_value))
        if key in active:
            raise QueryPathError(
                "Cycle while recording semantic query projection origins."
            )
        active.add(key)
        try:
            if _is_prepared_selector_definition(projected_value):
                if not isinstance(source_value, (Definition, ConcreteDefinition)):
                    raise QueryPathError(
                        "Semantic projection has no definition source for prepared parameters."
                    )
                source_parameters = _semantic_parameters(source_value)
                for edge in projected_edges:
                    if (
                        not isinstance(edge.segment, Kwarg)
                        or edge.segment.name not in source_parameters
                    ):
                        raise QueryPathError(
                            "Semantic projection lost a prepared parameter source."
                        )
                    relative_source_path = source_parameter_path(
                        source_value,
                        edge.segment.name,
                    )
                    child_source_path = (
                        None
                        if source_path is None or relative_source_path is None
                        else source_path.join(relative_source_path)
                    )
                    visit(
                        edge.value,
                        source_parameters[edge.segment.name],
                        projected_path.child(edge.segment),
                        child_source_path,
                    )
                return

            member_origins = origins.set_members.get(id(projected_value))
            if member_origins is not None:
                for edge in projected_edges:
                    matches = [
                        source_member
                        for projected_member, source_member in member_origins
                        if projected_member is edge.value
                    ]
                    if len(matches) != 1:
                        raise QueryPathError(
                            "Semantic projection lost a transformed set-member correspondence."
                        )
                    relative_source_path = source_edge_path(source_value, matches[0])
                    visit(
                        edge.value,
                        matches[0],
                        projected_path.child(edge.segment),
                        (
                            None
                            if source_path is None
                            else source_path.join(relative_source_path)
                        ),
                    )
                return

            source_edges = {
                edge.segment: edge for edge in iter_value_edges(source_value)
            }
            for edge in projected_edges:
                source_edge = source_edges.get(edge.segment)
                if source_edge is None:
                    raise QueryPathError(
                        "Semantic projection changed an occurrence without recording its source."
                    )
                visit(
                    edge.value,
                    source_edge.value,
                    projected_path.child(edge.segment),
                    None if source_path is None else source_path.child(edge.segment),
                )
        finally:
            active.remove(key)

    visit(projected, source, DefinitionPath(), DefinitionPath())
    return paths


@dataclass(frozen=True, slots=True)
class _V3Restriction:
    """One immutable V3 membership restriction and its bound authority scope."""

    kind: str
    value: Any
    scope: Any | None = None


def _same_v3_scope(left, right) -> bool:
    """Return whether two defaults name the same authority object, not a key."""

    if left is None or right is None:
        return False
    from .source import RepoSource, StoreSource

    return (
        isinstance(left, StoreSource)
        and isinstance(right, StoreSource)
        and left.store is right.store
    ) or (
        isinstance(left, RepoSource)
        and isinstance(right, RepoSource)
        and left.repo is right.repo
        and left.weak == right.weak
    )


def _bounded_identity_entries(entries, limit: int):
    """Retain only the canonical prefix without sorting the whole result domain."""

    selected = []
    for key, entry in entries.items():
        _retain_bounded_identity_entry(selected, key, entry, limit)
    return {key: entry for key, entry in selected}


def _retain_bounded_identity_entry(selected, key, entry, limit: int) -> None:
    """Insert one candidate into a small canonical-prefix selection buffer."""

    if limit == 0:
        return
    selected.append((key, entry))
    selected.sort(key=cmp_to_key(_compare_bounded_identity_entries))
    if len(selected) > limit:
        selected.pop()


def _compare_bounded_identity_entries(left, right) -> int:
    """Use canonical graph encodings only for colliding identity digests."""

    from .identity import _canonical_tie

    left_key, right_key = left[0], right[0]
    left_prefix = (left_key.kind, left_key.digest)
    right_prefix = (right_key.kind, right_key.digest)
    if left_prefix != right_prefix:
        return -1 if left_prefix < right_prefix else 1
    left_tie = _canonical_tie(left_key._value)
    right_tie = _canonical_tie(right_key._value)
    return (left_tie > right_tie) - (left_tie < right_tie)


def _compare_bounded_occurrences(left, right) -> int:
    """Compare typed occurrence keys, encoding roots only on digest ties."""

    from .identity import _occurrence_path_key

    left_key, right_key = left[0], right[0]
    first, second = left_key.owner, right_key.owner
    prefix_a, prefix_b = (first.kind, first.digest), (second.kind, second.digest)
    if prefix_a != prefix_b:
        return -1 if prefix_a < prefix_b else 1
    if first._value is not second._value and first.sort_key != second.sort_key:
        return -1 if first.sort_key < second.sort_key else 1
    path_a, path_b = _occurrence_path_key(left_key.path), _occurrence_path_key(right_key.path)
    if path_a != path_b:
        return -1 if path_a < path_b else 1
    first, second = left_key.target, right_key.target
    prefix_a, prefix_b = (first.kind, first.digest), (second.kind, second.digest)
    if prefix_a != prefix_b:
        return -1 if prefix_a < prefix_b else 1
    if first._value is second._value:
        return 0
    return (first.sort_key > second.sort_key) - (first.sort_key < second.sort_key)


@dataclass(frozen=True, slots=True)
class _AlgebraSource:
    """Deferred identity algebra whose operands share one terminal capture."""

    left: "IdentityQuery"
    right: "IdentityQuery"
    operation: str

    def __post_init__(self) -> None:
        if self.operation not in {"union", "intersection"}:
            raise ValueError("Query V3 algebra operation is unsupported.")


def union(left, right, *others):
    """Combine fixed and live identity universes without assigning a new scope.

    Args:
        left: IdentityQuery or IdentitySet supplying initial members.
        right: Second IdentityQuery or IdentitySet operand.
        *others: Additional identity operands, evaluated in the supplied order.

    Returns:
        A fixed IdentitySet when all operands are fixed; otherwise a deferred
        IdentityQuery that merges complete identities and detached evidence.

    Raises:
        TypeError: If any operand is not an identity query or fixed set.

    Side Effects:
        Fixed-only combinations never read a source; source-backed queries
        remain unevaluated until an explicit terminal.
    """

    from .identity import IdentitySet

    result = left
    for other in (right, *others):
        if isinstance(result, IdentitySet):
            result = result.union(other) if isinstance(other, IdentitySet) else result.query().union(other)
        elif isinstance(result, IdentityQuery):
            result = result.union(other)
        else:
            raise TypeError("Query V3 union requires identity queries or fixed sets.")
    return result


def intersection(left, right, *others):
    """Intersect fixed and live identity universes under the algebra scope rules.

    Args:
        left: Initial IdentityQuery or IdentitySet.
        right: Second identity operand.
        *others: Additional operands evaluated in order.

    Returns:
        A source-free IdentitySet for fixed-only operands, otherwise a deferred
        IdentityQuery with conservative default authority.

    Raises:
        TypeError: If an operand is not an identity query or fixed set.

    Side Effects:
        No source access occurs until a query terminal; fixed-only work stays
        detached from Stores.
    """

    from .identity import IdentitySet

    result = left
    for other in (right, *others):
        if isinstance(result, IdentitySet):
            result = result.intersection(other) if isinstance(other, IdentitySet) else result.query().intersection(other)
        elif isinstance(result, IdentityQuery):
            result = result.intersection(other)
        else:
            raise TypeError("Query V3 intersection requires identity queries or fixed sets.")
    return result


@dataclass(frozen=True, slots=True)
class IdentityQuery:
    """Compose deferred restrictions over complete CDef and reference identities.

    Instances come from :meth:`dryml.core.Repo.query`, :meth:`Store.query`, or a
    fixed :class:`IdentitySet`. Restriction methods return immutable plans; an
    explicit terminal captures required Store facts once and returns an
    :class:`IdentitySet` or scalar identity. Query evaluation reads no payloads
    and never constructs Objects, allocates ObjectIds, writes authority, observes
    host state, or executes workloads.

    Attributes:
        source: Producer or fixed membership that supplies the initial universe.
        restrictions: Ordered immutable identity, metadata, and source filters.
        default_scope: Optional unambiguous authority scope for later operations.
        max_witness_limit: Optional exact GeneratorSelector verification bound.
        refresh_policy: Existing derived-index refresh policy.
        scan_policy_mode: Authority-inventory scan policy.
        max_verify_limit: Optional detached-candidate verification bound.
        max_depth_limit: Optional later relationship traversal depth bound.
        indexed_required: Whether unproved V3 index coverage is rejected.
        take_limit: Optional explicit bounded terminal prefix size.
    """

    source: Any
    restrictions: tuple[_V3Restriction, ...] = ()
    default_scope: Any | None = None
    max_witness_limit: int | None = _DEFAULT_GENERATOR_WITNESS_LIMIT
    refresh_policy: RefreshPolicy = "auto"
    scan_policy_mode: str = "allow"
    max_verify_limit: int | None = None
    max_depth_limit: int | None = None
    indexed_required: bool = False
    take_limit: int | None = None

    @classmethod
    def from_store(cls, store) -> "IdentityQuery":
        """Create an unevaluated V3 universe from one Store producer.

        Args:
            store: Store whose authoritative and derived knowledge seeds the plan.

        Returns:
            A query with ``store`` as its default authority scope.

        Raises:
            TypeError: Immediately if ``store`` is unsupported.

        Side Effects:
            None. The Store is not read until a terminal executes.
        """

        from .source import StoreSource

        source = StoreSource(store)
        return cls(source, default_scope=source)

    @classmethod
    def from_repo(cls, repo, *, weak: bool = True) -> "IdentityQuery":
        """Create an unevaluated V3 universe from one Repo producer.

        Args:
            repo: Repo supplying connected Store and retained-cache knowledge.
            weak: Include weak cache entries as well as strong entries.

        Returns:
            A query with ``repo`` as its default authority scope.

        Raises:
            TypeError: If ``weak`` is not an exact bool.

        Side Effects:
            None. Stores and caches are not read until a terminal executes.
        """

        from .source import RepoSource

        source = RepoSource(repo, weak=weak)
        return cls(source, default_scope=source)

    @classmethod
    def from_retained_repo_lookup(cls, repo, *, weak: bool = True) -> "IdentityQuery":
        """Create the private root-or-cache universe for retained Repo lookups.

        Args:
            repo: Repo supplying explicit roots and retained cache entries.
            weak: Include weak cache entries as well as strong entries.

        Returns:
            An unevaluated identity query for retained non-query Repo APIs.

        Raises:
            TypeError: If ``weak`` is not an exact bool.

        Side Effects:
            None until a terminal executes.
        """

        from .source import RetainedRepoLookupSource

        source = RetainedRepoLookupSource(repo, weak=weak)
        return cls(source, default_scope=source)

    @classmethod
    def from_set(cls, members) -> "IdentityQuery":
        """Create an unevaluated V3 refinement over a fixed IdentitySet.

        Args:
            members: Detached :class:`IdentitySet` membership and evidence.

        Returns:
            A query restricted to the supplied fixed identities.

        Raises:
            TypeError: If ``members`` is not an IdentitySet.

        Side Effects:
            None. No external authority is adopted or read.
        """

        from .identity import IdentitySet

        if not isinstance(members, IdentitySet):
            raise TypeError("IdentityQuery.from_set requires an IdentitySet.")
        return cls(members)

    def sel(self, value=None) -> "IdentityQuery":
        """Restrict this fixed universe by one structural or exact selector.

        Definitions remain structural even when complete; ConcreteDefinitions,
        exact references, and GeneratorSelector support retain their stricter
        established meanings.  ``None`` is an immutable no-op.

        Args:
            value: Definition, complete CDef, Selector, GeneratorSelector, exact
                ObjectRef or StateRef, or ``None``.

        Returns:
            An immutable query with the selector appended, or this query for
            ``None``.

        Raises:
            TypeError: If the selector type is unsupported or contains a
                StateSelectorRef without an explicit selector scope.

        Side Effects:
            None. Selection remains deferred until a terminal executes.
        """

        if value is None:
            return self
        from ..definition import ConcreteDefinition, Definition
        from ..generator import GeneratorSelector
        from ..reference_values import ObjectRef, StateRef
        from ..selector import Selector, _contains_state_selector

        if not isinstance(value, (Definition, ConcreteDefinition, Selector, GeneratorSelector, ObjectRef, StateRef)):
            raise TypeError("IdentityQuery.sel requires a Definition, selector, generator, or exact reference.")
        selector_root = (
            value.root if isinstance(value, Selector)
            else value.prefilter.root if isinstance(value, GeneratorSelector)
            else value
        )
        if _contains_state_selector(selector_root):
            raise TypeError(
                "StateSelectorRef values supplied to IdentityQuery.sel require selector(value, scope=repo)."
            )
        return self._append("sel", value)

    def cdefs(self) -> "IdentityQuery":
        """Restrict retained members to CDefs without expanding membership."""

        return self._append("kind", "cdef")

    def object_refs(self) -> "IdentityQuery":
        """Restrict retained members to ObjectRefs without projection or expansion."""

        return self._append("kind", "object_ref")

    def state_refs(self) -> "IdentityQuery":
        """Restrict retained members to StateRefs without projection or expansion."""

        return self._append("kind", "state_ref")

    def closure(self, edges=None) -> "IdentityQuery":
        """Expand retained identities through literal directed relationships.

        Args:
            edges: An immutable :class:`EdgePolicy`; omitted selects every
                retained association, materializing edge, and reference edge.

        Returns:
            An immutable identity query whose terminal includes these input
            identities and every identity reachable through the selected policy.

        Raises:
            TypeError: If ``edges`` is not an EdgePolicy.

        Side Effects:
            None until a terminal evaluates the input query. Expansion only
            inspects detached identity values and never loads payloads or finds
            reverse saved-state relationships.
        """

        from .relationships import normalize_edge_policy

        return replace(
            self,
            source=_RelationshipClosure(
                self, normalize_edge_policy(edges), self.max_depth_limit,
            ),
            restrictions=(),
        )

    def nested(self, selector=None, *, edges=None) -> "OccurrenceQuery":
        """Create strict non-empty typed relationship occurrences from these roots.

        Args:
            selector: Optional target selector with the established V3 identity
                meanings. ``None`` retains every reachable target.
            edges: Immutable relationship policy, defaulting to all retained
                directed relationship kinds.

        Returns:
            An immutable occurrence query retaining root identities, typed paths,
            and exact target identities.

        Raises:
            TypeError: If the selector or policy is unsupported.

        Side Effects:
            None. Evaluation remains deferred and never searches for reverse
            ObjectRef-to-StateRef relationships.
        """

        from .relationships import normalize_edge_policy

        return OccurrenceQuery(
            self, selector=selector, edges=normalize_edge_policy(edges),
            max_depth_limit=self.max_depth_limit,
        )

    def object_id(self, value) -> "IdentityQuery":
        """Restrict ObjectRef and StateRef candidates containing one ObjectId.

        Args:
            value: Exact ObjectId required at any retained reference path.

        Returns:
            An immutable restricted query.

        Raises:
            TypeError: If ``value`` is not an ObjectId.

        Side Effects:
            None until a terminal executes.
        """

        from ..reference_values import ObjectId

        if not isinstance(value, ObjectId):
            raise TypeError("object_id requires an ObjectId.")
        return self._append("object_id", value)

    def namespace(self, prefix) -> "IdentityQuery":
        """Restrict reference candidates to a normalized namespace prefix.

        Args:
            prefix: Iterable namespace prefix accepted by ObjectRef validation.

        Returns:
            An immutable restricted query.

        Raises:
            TypeError: If the namespace is not iterable or has invalid elements.
            ValueError: If namespace validation rejects the prefix.

        Side Effects:
            None until a terminal executes.
        """

        from ..reference_values import _normalize_namespace

        return self._append("namespace", _normalize_namespace(tuple(prefix)))

    def contains(self, value) -> "IdentityQuery":
        """Restrict aggregates to a proper owned subtree exact ObjectRef.

        Args:
            value: Exact ObjectRef required below an aggregate root.

        Returns:
            An immutable restricted query.

        Raises:
            TypeError: If ``value`` is not an ObjectRef.

        Side Effects:
            None until a terminal executes.
        """

        from ..reference_values import ObjectRef

        if not isinstance(value, ObjectRef):
            raise TypeError("contains requires an ObjectRef.")
        return self._append("contains", value)

    def state_hash(self, value: str) -> "IdentityQuery":
        """Restrict StateRefs to an existing complete local state hash.

        Args:
            value: Canonical local-state hash.

        Returns:
            An immutable restricted query.

        Raises:
            TypeError: If the hash is not a string.
            ValueError: If the hash format is invalid.

        Side Effects:
            None until a terminal executes.
        """

        from ..reference_values import _validate_state_hash

        _validate_state_hash(value)
        return self._append("state_hash", value)

    def alias(self, value: str, *, scope=None) -> "IdentityQuery":
        """Restrict by aliases captured from the authority scope bound now.

        Args:
            value: Non-empty alias name.
            scope: Store or Repo authority, defaulting to the producer scope.

        Returns:
            An immutable alias restriction.

        Raises:
            ValueError: If ``value`` is not a non-empty string.
            QueryDomainError: If no unambiguous scope is available.
            TypeError: If ``scope`` is unsupported.

        Side Effects:
            None until a terminal captures aliases under an authority fence.
        """

        if not isinstance(value, str) or not value:
            raise ValueError("alias requires a non-empty string.")
        return self._append("alias", value, self._bound_scope(scope))

    def stored(self, *, scope=None) -> "IdentityQuery":
        """Restrict members by their kind-specific captured stored authority.

        Args:
            scope: Store or Repo authority, defaulting to the producer scope.

        Returns:
            An immutable stored-membership restriction.

        Raises:
            QueryDomainError: If no unambiguous scope is available.
            TypeError: If ``scope`` is unsupported.

        Side Effects:
            None until a terminal reads authority or verified derived candidates.
        """

        return self._append("stored", None, self._bound_scope(scope))

    def cached(self, *, scope=None, weak: bool = True) -> "IdentityQuery":
        """Retain members found in one selected Repo cache view, without adding any.

        Args:
            scope: Repo whose cache membership is tested, or the input Repo
                producer's bound default. Fixed and Store sources require it.
            weak: Include weak and strong entries when true; strong only when false.

        Returns:
            An unevaluated identity restriction over the existing input universe.

        Raises:
            TypeError: For a non-Repo scope or non-boolean tier selection.
            QueryDomainError: If a fixed or Store producer lacks explicit scope.

        Side Effects:
            Terminal evaluation reads retained cache entries once; it never
            constructs Objects, captures states, or populates a cache.
        """

        from .source import RepoSource

        if type(weak) is not bool:
            raise TypeError("cached weak must be an exact bool.")
        selected = self.default_scope if scope is None else self._normalize_scope(scope)
        if not isinstance(selected, RepoSource):
            raise QueryDomainError("cached requires an explicit Repo scope.")
        return self._append("cached", weak, selected)

    def where(self, predicate, *, scope=None) -> "IdentityQuery":
        """Conjoin typed metadata without changing the identity query family.

        Args:
            predicate: Typed Query V3 metadata predicate.
            scope: Store or Repo authority, defaulting to the producer scope.

        Returns:
            An immutable metadata restriction.

        Raises:
            TypeError: If the predicate or scope is unsupported.
            QueryDomainError: If no unambiguous scope is available.

        Side Effects:
            None until a terminal captures metadata under authority fences.
        """

        from .metadata import _require_predicate

        return self._append("metadata", _require_predicate(predicate), self._bound_scope(scope))

    def in_source(self, source) -> "IdentityQuery":
        """Restrict contribution evidence and bind later authority defaults.

        Args:
            source: Contributing Store or Repo represented in captured evidence.

        Returns:
            An immutable source-evidence restriction.

        Raises:
            TypeError: If ``source`` is not a Store or Repo producer.

        Side Effects:
            None until a terminal executes.
        """

        scope = self._normalize_scope(source)
        return replace(self._append("source", source), default_scope=scope)

    def union(self, other: "IdentityQuery | IdentitySet") -> "IdentityQuery":
        """Return a deferred complete-identity union with shared source cuts.

        The result keeps a default authority scope only when both operand plans
        name the same Store or Repo object.  Different producers therefore
        compose without choosing authority by source order.

        Args:
            other: Identity query or fixed IdentitySet to combine.

        Returns:
            A deferred complete-identity union.

        Raises:
            TypeError: If ``other`` is unsupported.
            QueryDomainError: If execution policies conflict.

        Side Effects:
            None until a terminal evaluates both operands under shared cuts.
        """

        return self._combine(other, "union")

    def intersection(self, other: "IdentityQuery | IdentitySet") -> "IdentityQuery":
        """Return a deferred complete-identity intersection with shared cuts.

        Matching identities retain the evidence supplied by both operands.  As
        with :meth:`union`, an ambiguous authority default is deliberately
        removed rather than selected from operand order.

        Args:
            other: Identity query or fixed IdentitySet to combine.

        Returns:
            A deferred complete-identity intersection.

        Raises:
            TypeError: If ``other`` is unsupported.
            QueryDomainError: If execution policies conflict.

        Side Effects:
            None until a terminal evaluates both operands under shared cuts.
        """

        return self._combine(other, "intersection")

    def take(self, limit: int) -> "IdentitySet":
        """Evaluate an explicitly bounded canonical identity prefix.

        Args:
            limit: Non-negative exact number of globally deduplicated members.

        Returns:
            A fixed IdentitySet whose requested limit remains visible through
            later fixed-set refinement; use its query() to compose further.

        Raises:
            ValueError: If ``limit`` is not a non-negative exact integer.

        Side Effects:
            Evaluates the query under authority fences. Eligible stored-CDef
            plans may read derived keyset pages but never mutate authority.
        """

        if type(limit) is not int or limit < 0:
            raise ValueError("take limit must be a non-negative exact int.")
        return replace(self, take_limit=limit).collect()

    def max_witnesses(self, limit: int | None) -> "IdentityQuery":
        """Set the shared GeneratorSelector verification witness budget.

        Args:
            limit: Positive exact witness count, or ``None`` for no bound.

        Returns:
            An immutable query carrying the verification budget.

        Raises:
            ValueError: If ``limit`` is not positive, exact, or ``None``.

        Side Effects:
            None until a terminal verifies generator support.
        """

        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_witnesses limit must be a positive exact int or None.")
        return replace(self, max_witness_limit=limit)

    def refresh(self, policy: RefreshPolicy = "auto") -> "IdentityQuery":
        """Return a plan with the requested derived-index refresh policy.

        ``False`` leaves derived indexes untouched, ``"auto"`` reconciles stale
        indexes when safe, and ``True`` forces reconciliation before each
        terminal source cut. Authoritative Store records are never rewritten.

        Args:
            policy: Exact ``False``, ``"auto"``, or exact ``True`` behavior.

        Returns:
            An immutable query carrying the selected refresh policy.

        Raises:
            ValueError: If ``policy`` is not ``False``, ``"auto"``, or ``True``.

        Side Effects:
            None until a terminal executes. Terminal evaluation may open,
            validate, rebuild, or replace derived query-index sidecars.
        """

        if policy is not False and policy is not True and policy != "auto":
            raise ValueError("refresh policy must be False, 'auto', or True.")
        return replace(self, refresh_policy=policy)

    def scan_policy(self, policy: str) -> "IdentityQuery":
        """Return a plan controlling required authoritative inventory scans.

        ``"allow"`` scans silently, ``"warn"`` emits one ``RuntimeWarning`` per
        terminal requiring a scan, and ``"forbid"`` raises
        :class:`QueryWouldScanError` before scanning.

        Args:
            policy: One of ``"allow"``, ``"warn"``, or ``"forbid"``.

        Returns:
            An immutable query carrying the selected authority-scan policy.

        Raises:
            ValueError: If ``policy`` is not ``"allow"``, ``"warn"``, or
                ``"forbid"``.

        Side Effects:
            None until a terminal executes; ``"warn"`` may then emit a warning.
        """

        if not isinstance(policy, str) or policy not in {"allow", "warn", "forbid"}:
            raise ValueError("scan policy must be 'allow', 'warn', or 'forbid'.")
        return replace(self, scan_policy_mode=policy)

    def require_indexed(self) -> "IdentityQuery":
        """Require proved V3 index coverage rather than authority fallback.

        Returns:
            An immutable query that also forbids authority inventory scans.

        Raises:
            None while building the plan. Terminals raise QueryIndexUnavailable
            when the required coverage or direct authority proof is absent.

        Side Effects:
            None until a terminal validates the derived index.
        """

        return replace(self, scan_policy_mode="forbid", indexed_required=True)

    def max_verify(self, limit: int | None) -> "IdentityQuery":
        """Set the maximum detached identity candidates verified by a terminal.

        Args:
            limit: Non-negative exact candidate count, or ``None`` for no bound.

        Returns:
            An immutable query carrying the verification budget.

        Raises:
            ValueError: If ``limit`` is negative, inexact, or not ``None``.

        Side Effects:
            None until a terminal verifies candidates.
        """

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_verify limit must be a non-negative exact int or None.")
        return replace(self, max_verify_limit=limit)

    def max_depth(self, limit: int | None) -> "IdentityQuery":
        """Set the relationship-depth safety budget for closure or nesting.

        Args:
            limit: Non-negative exact depth, or ``None`` for no bound.

        Returns:
            An immutable query carrying the traversal budget.

        Raises:
            ValueError: If ``limit`` is negative, inexact, or not ``None``.

        Side Effects:
            None until a relationship terminal executes.
        """

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_depth limit must be a non-negative exact int or None.")
        return replace(self, max_depth_limit=limit)

    def explain(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        """Describe whether this immutable V3 plan requires an authority scan.

        ``sql`` is accepted for interface parity but reports no backend detail:
        U3 evaluates authoritative detached records rather than a SQL result.
        ``analyze`` evaluates the plan and includes its final result count.

        Args:
            analyze: Execute the plan to include runtime counts and refresh work.
            sql: Request backend detail; currently retained for interface parity.

        Returns:
            A disclosure-safe structural QueryExplanation.

        Raises:
            QueryError: If analyzed execution, authority capture, or required
                index validation fails.

        Side Effects:
            With ``analyze=True``, reads sources and may refresh derived indexes
            according to the plan. Otherwise no query source is read.
        """

        del sql
        exact = self._exact_state_selector() is not None and not self._metadata_scopes()
        scan_required = not exact and self._requires_authority_inventory()
        result_count = None
        source_cuts = capture_rounds = demand_recaptures = 0
        if analyze:
            result_count, capture = self._run_terminal(
                lambda cut: sum(1 for _ in self._iter_terminal_members(cut)),
                return_capture=True,
            )
            source_cuts = capture.source_cuts
            capture_rounds = capture.capture_rounds
            demand_recaptures = capture.demand_recaptures
        return QueryStats(
            refresh_action=(capture.refresh_action if analyze else "none"),
            result_count=result_count,
            fast_path="exact-state-ref" if exact else None,
            scan_required=scan_required,
            scan_reason=("V3 identity inventory requires authoritative record enumeration" if scan_required else None),
            candidate_rows_read=(capture.candidate_rows_read if analyze else 0),
            cdef_blobs_decoded=(capture.cdef_blobs_decoded if analyze else 0),
            pages_fetched=(capture.pages_fetched if analyze else 0),
            source_cuts=source_cuts,
            capture_rounds=capture_rounds,
            demand_recaptures=demand_recaptures,
            instability_retries=capture.instability_retries if analyze else 0,
        ).explanation(domain="identity", refresh=self.refresh_policy)

    def categorical(self, **kwargs) -> "IdentityQuery":
        """Append a categorical edit of the latest selector restriction.

        Args:
            **kwargs: Keyword arguments accepted by ``Selector.categorical``.

        Returns:
            An immutable query with the edited selector appended.

        Raises:
            QueryDomainError: If no selector restriction exists.
            TypeError: If selector editing rejects the arguments.

        Side Effects:
            None. Selection remains deferred.
        """

        selector = self._last_selector()
        if selector is None:
            raise QueryDomainError("categorical() requires an existing Selector restriction.")
        return self.sel(selector.categorical(**kwargs))

    def restore(self, **kwargs) -> "IdentityQuery":
        """Append the restored form of the latest selector restriction.

        Args:
            **kwargs: Keyword arguments accepted by ``Selector.restore``.

        Returns:
            An immutable query with the restored selector appended.

        Raises:
            QueryDomainError: If no selector restriction exists.
            TypeError: If selector editing rejects the arguments.

        Side Effects:
            None. Selection remains deferred.
        """

        selector = self._last_selector()
        if selector is None:
            raise QueryDomainError("restore() requires an existing Selector restriction.")
        return self.sel(selector.restore(**kwargs))

    def exact(self, definition=None, **kwargs) -> "IdentityQuery":
        """Append an exact subtree edit without discarding restrictions.

        Args:
            definition: Optional subtree definition accepted by ``Selector.exact``.
            **kwargs: Path edits accepted by ``Selector.exact``.

        Returns:
            An immutable query with the exact selector appended.

        Raises:
            QueryDomainError: If no selector restriction exists.
            TypeError: If selector editing rejects the arguments.

        Side Effects:
            None. Selection remains deferred.
        """

        selector = self._last_selector()
        if selector is None:
            raise QueryDomainError("exact() requires an existing Selector restriction.")
        return self.sel(selector.exact(definition, **kwargs))

    def collect(self):
        """Finish detached membership evaluation into a fixed IdentitySet.

        Returns:
            A deterministic fixed result with detached evidence and visible bounds.

        Raises:
            QueryError: If validation, authority capture, budgets, or required
                index coverage fails.

        Side Effects:
            Reads source authority and caches under retryable cuts and may refresh
            derived indexes according to the plan.
        """

        return self._run_terminal(self._collect_with_capture)

    def count(self) -> int:
        """Return the distinct identity count after all validation.

        Returns:
            Number of identities retained by the plan and any explicit bound.

        Raises:
            QueryError: If validation, authority capture, budgets, or required
                index coverage fails.

        Side Effects:
            Executes the query without constructing a complete fixed result.
        """

        if self.take_limit is not None:
            return self._run_terminal(lambda cut: len(self._bounded_terminal_entries(cut)[0]))
        return self._run_terminal(lambda cut: sum(1 for _ in self._iter_terminal_members(cut)))

    def exists(self) -> bool:
        """Return whether one fully valid identity remains.

        Returns:
            ``True`` after the first conclusive valid member, otherwise ``False``.

        Raises:
            QueryError: If validation or source/index access fails.

        Side Effects:
            Executes only the source work needed for a conclusive answer.
        """

        if self.take_limit == 0:
            return False
        return self._run_terminal(lambda cut: next(self._iter_terminal_members(cut), None) is not None)

    def one(self):
        """Return the sole retained identity.

        Returns:
            The one complete CDef, ObjectRef, or StateRef.

        Raises:
            QueryCardinalityError: If zero or multiple identities remain.
            QueryError: If validation or source/index access fails.

        Side Effects:
            Executes enough source work to prove singleton cardinality.
        """

        if self.take_limit is not None:
            items = self._run_terminal(lambda cut: tuple(self._bounded_terminal_entries(cut)[0].values()))
            if len(items) != 1:
                raise QueryCardinalityError(f"Expected exactly one result, found {len(items)}.")
            return items[0][0]
        items = self._run_terminal(self._up_to_two_terminal_members)
        if len(items) != 1:
            raise QueryCardinalityError(f"Expected exactly one result, found {len(items)}.")
        return items[0][1][0]

    def one_or_none(self):
        """Return zero or one identity while rejecting ambiguity.

        Returns:
            The sole identity, or ``None`` when no identity remains.

        Raises:
            QueryCardinalityError: If multiple identities remain.
            QueryError: If validation or source/index access fails.

        Side Effects:
            Executes enough source work to prove at-most-one cardinality.
        """

        if self.take_limit is not None:
            items = self._run_terminal(lambda cut: tuple(self._bounded_terminal_entries(cut)[0].values()))
            if len(items) > 1:
                raise QueryCardinalityError(f"Expected zero or one result, found {len(items)}.")
            return items[0][0] if items else None
        items = self._run_terminal(self._up_to_two_terminal_members)
        if len(items) > 1:
            raise QueryCardinalityError(f"Expected zero or one result, found {len(items)}.")
        return items[0][1][0] if items else None

    def __bool__(self) -> bool:
        raise TypeError("IdentityQuery requires an explicit terminal.")

    def _append(self, kind, value, scope=None) -> "IdentityQuery":
        return replace(self, restrictions=(*self.restrictions, _V3Restriction(kind, value, scope)))

    def _run_terminal(self, evaluate, *, return_capture=False):
        """Restart all dependent stages on demand growth or an invalidated cut."""

        from .model import QueryError
        from .source import SourceCapture, _SourceRecapture

        if (
                self.scan_policy_mode == "forbid"
                and not self.indexed_required
                and self._requires_authority_scan()
                and not self._can_try_bounded_index()
                and self._exact_stored_cdef_selector() is None
                and not (
                    self._exact_state_selector() is not None
                    and not self._metadata_scopes()
                    and not any(
                        item.kind in {"alias", "stored"}
                        for item in self.restrictions
                    )
                )):
            raise QueryWouldScanError("Query V3 requires an authoritative inventory scan.")
        refresh_action = self._refresh_derived_indexes()
        capture = SourceCapture()
        capture.refresh_action = refresh_action
        capture.active = True
        while True:
            try:
                answer = evaluate(capture)
                return (answer, capture) if return_capture else answer
            except _SourceRecapture as recapture:
                if recapture.invalidated:
                    capture.instability_retries += 1
                    if capture.instability_retries >= 3:
                        raise QueryError("Query V3 source changed during evaluation.") from None

    def _refresh_derived_indexes(self) -> str:
        """Apply the terminal's refresh policy once to producer-owned indexes."""

        if self.refresh_policy is False or self.take_limit == 0:
            return "none"
        if self.refresh_policy == "auto" and (
                self._exact_stored_cdef_selector() is not None
                or (
                    self._exact_state_selector() is not None
                    and not self._metadata_scopes()
                    and not any(
                        item.kind in {"alias", "stored"}
                        for item in self.restrictions
                    )
                )):
            return "none"
        from .source import RepoSource, RetainedRepoLookupSource, StoreSource

        seen_repos = set()
        seen_stores = set()
        refreshed = False

        def refresh_store(store):
            nonlocal refreshed
            key = store.authority_fence_key()
            if key in seen_stores:
                return
            seen_stores.add(key)
            if getattr(store, "query_index_policy", None) in {"memory", "none"}:
                return
            try:
                index = store.open_query_index()
            except QueryIndexUnavailable:
                if self.refresh_policy is True:
                    raise
                return
            if index is not None:
                index.refresh(self.refresh_policy)
                refreshed = True

        def visit(query):
            nonlocal refreshed
            source = query.source
            if isinstance(source, StoreSource):
                refresh_store(source.store)
            elif isinstance(source, (RepoSource, RetainedRepoLookupSource)):
                repo = source.repo
                if id(repo) not in seen_repos:
                    seen_repos.add(id(repo))
                    if any(
                            getattr(store, "query_index_policy", None)
                            not in {"memory", "none"}
                            for store in repo.stores):
                        refreshed = (
                            repo._query_index.refresh(self.refresh_policy)
                            or refreshed
                        )
                    seen_stores.update(
                        store.authority_fence_key() for store in repo.stores
                    )
            elif isinstance(source, _RelationshipClosure):
                visit(source.input_query)
            elif isinstance(source, _RelationshipProjection):
                visit(source.occurrence_query.roots)
            elif isinstance(source, _AlgebraSource):
                visit(source.left)
                visit(source.right)

        visit(self)
        if not refreshed:
            return "none"
        return "forced" if self.refresh_policy is True else "auto"

    def _combine(self, other: "IdentityQuery", operation: str) -> "IdentityQuery":
        """Validate compatible terminal controls and defer one algebra stage."""

        from .identity import IdentitySet

        if isinstance(other, IdentitySet):
            other = other.query()
        if not isinstance(other, IdentityQuery):
            raise TypeError("Query V3 algebra requires an IdentityQuery or IdentitySet.")
        controls = (
            "refresh_policy", "scan_policy_mode", "max_verify_limit",
            "max_witness_limit", "max_depth_limit", "indexed_required",
        )
        if any(getattr(self, name) != getattr(other, name) for name in controls):
            raise QueryDomainError(
                "Cannot combine Query V3 plans with conflicting execution policies."
            )
        return IdentityQuery(
            _AlgebraSource(self, other, operation),
            default_scope=(
                self.default_scope
                if _same_v3_scope(self.default_scope, other.default_scope)
                else None
            ),
            max_witness_limit=self.max_witness_limit,
            refresh_policy=self.refresh_policy,
            scan_policy_mode=self.scan_policy_mode,
            max_verify_limit=self.max_verify_limit,
            max_depth_limit=self.max_depth_limit,
            indexed_required=self.indexed_required,
        )

    def _bound_scope(self, scope):
        if scope is None:
            if self.default_scope is None:
                raise QueryDomainError("Collection-backed authority restrictions require an explicit scope.")
            return self.default_scope
        return self._normalize_scope(scope)

    @staticmethod
    def _normalize_scope(scope):
        from ..store.store import Store
        from .source import RepoSource, StoreSource

        if isinstance(scope, (StoreSource, RepoSource)):
            return scope
        if isinstance(scope, Store):
            return StoreSource(scope)
        if hasattr(scope, "stores") and hasattr(scope, "retain_topology"):
            return RepoSource(scope)
        raise TypeError("Query V3 authority scope must be a Store or Repo producer.")

    def _metadata_scopes(self) -> frozenset[str]:
        from .metadata import _leaves

        return frozenset(
            leaf.field.scope
            for restriction in self.restrictions if restriction.kind == "metadata"
            for leaf in _leaves(restriction.value)
        )

    def _collect_with_capture(self, capture):
        """Collect this plan using a terminal-owned capture registry."""

        from .identity import IdentitySet

        if self.take_limit is None:
            members, bounded = self._terminal_entries(capture)
        else:
            members, bounded = self._bounded_terminal_entries(capture)
        limit = self.take_limit
        limit_conflict = False
        if limit is None:
            limit, limit_conflict = self._retained_requested_limit_state()
        return IdentitySet._from_entries(
            members,
            bounded=bounded,
            requested_limit=limit,
            requested_limit_conflict=limit_conflict,
            query_work=self._retained_query_work(),
            query_stats=(
                capture.candidate_rows_read,
                capture.cdef_blobs_decoded,
                capture.pages_fetched,
            ),
        )

    def _retained_query_work(self):
        """Return deduplicated work provenance captured by fixed-set sources."""

        from .identity import IdentitySet

        if isinstance(self.source, IdentitySet):
            return dict(self.source._query_work)
        if isinstance(self.source, _AlgebraSource):
            return {
                **self.source.left._retained_query_work(),
                **self.source.right._retained_query_work(),
            }
        if isinstance(self.source, _RelationshipClosure):
            return self.source.input_query._retained_query_work()
        if isinstance(self.source, _RelationshipProjection):
            return self.source.occurrence_query.roots._retained_query_work()
        return {}

    def _retained_requested_limit_state(self):
        """Return visible prefix-limit state retained by fixed-set sources."""

        from .identity import IdentitySet, _combined_requested_limit_state

        if isinstance(self.source, IdentitySet):
            return (
                self.source.requested_limit,
                self.source._requested_limit_conflict,
            )
        if isinstance(self.source, _AlgebraSource):
            return _combined_requested_limit_state(
                *self.source.left._retained_requested_limit_state(),
                *self.source.right._retained_requested_limit_state(),
            )
        if isinstance(self.source, _RelationshipClosure):
            return self.source.input_query._retained_requested_limit_state()
        if isinstance(self.source, _RelationshipProjection):
            return self.source.occurrence_query.roots._retained_requested_limit_state()
        return None, False

    def _evaluate_entries(self, capture):
        """Return complete private entries for an algebra operand, never public API."""

        members, _ = self._evaluated_entries(capture)
        return members

    def _evaluated_entries(self, capture):
        """Evaluate this plan while preserving any operand-local explicit bound."""

        return (
            self._terminal_entries(capture)
            if self.take_limit is None
            else self._bounded_terminal_entries(capture)
        )

    def _terminal_entries(self, capture):
        """Evaluate all fallible validation, then return qualified private entries."""

        if self.indexed_required and self._requires_authority_inventory():
            raise QueryIndexUnavailable("Query V3 index coverage is not available for this plan.")
        captured = self._captured_members(capture)
        return {
            key: entry for key, entry in self._iter_terminal_members(capture, captured)
        }, captured[2]

    def _bounded_terminal_entries(self, capture):
        """Select a canonical prefix while retaining at most its requested size."""

        assert self.take_limit is not None
        if self.take_limit == 0:
            return {}, True
        indexed = self._bounded_index_entries(capture)
        if indexed is not None:
            return indexed, True
        selected = []
        for key, entry in self._iter_terminal_members(capture):
            _retain_bounded_identity_entry(selected, key, entry, self.take_limit)
        return (
            {key: entry for key, entry in selected},
            True,
        )

    def _bounded_index_entries(self, capture):
        """Return a keyset-paged global CDef prefix when every source supports it."""

        import heapq

        from .identity import IdentitySet
        from .model import (
            QueryIndexDirty,
            QueryIndexUnavailable,
            QueryStats,
            QueryWouldScanError,
        )

        stores = self._bounded_index_stores(capture)
        if stores is None:
            return None
        streams = []
        for store in stores:
            if self.refresh_policy is False:
                status = store.query_index_status()
                if status.state != "ready":
                    reason = (
                        "The query index changed outside managed index publication."
                        if status.state == "dirty"
                        else "The query index is not ready."
                    )
                    if self.indexed_required:
                        raise QueryIndexUnavailable(
                            f"{reason} Query V3 requires an already-ready index when refresh is disabled."
                        )
                    if self.scan_policy_mode == "forbid":
                        raise QueryWouldScanError(
                            f"{reason} Query V3 cannot fall back when scans are forbidden."
                        )
                    return None
            try:
                index = store.open_query_index()
            except QueryIndexUnavailable:
                return None
            if index is None:
                return None

            def pages(index=index):
                after = None
                while True:
                    page_stats = QueryStats()
                    with index.read_view(include_cached=False) as view:
                        page = getattr(view, "iter_stored_identity_cdef_batches", None)
                        if page is None:
                            return
                        batch = next(page(
                            after=after,
                            # One lookahead row proves the last retained digest
                            # bucket is complete without fetching another page.
                            batch_size=self.take_limit + 1,
                            stats=page_stats,
                        ), None)
                    capture.candidate_rows_read += page_stats.candidate_rows_read
                    capture.cdef_blobs_decoded += page_stats.cdef_blobs_decoded
                    capture.pages_fetched += page_stats.pages_fetched
                    if batch is None:
                        return
                    yield from batch.cdefs
                    after = batch.next_cursor

            stream = iter(pages())
            streams.append((store, stream))

        try:
            frontier = []
            for position, (store, stream) in enumerate(streams):
                value = next(stream, None)
                if value is not None:
                    heapq.heappush(
                        frontier, (value.graph_hash(), position, value, store, stream),
                    )
            selected = {}
            while frontier and len(selected) < self.take_limit:
                digest = frontier[0][0]
                bucket = []
                while frontier and frontier[0][0] == digest:
                    _, position, value, store, stream = heapq.heappop(frontier)
                    bucket.append((value, store))
                    following = next(stream, None)
                    if following is not None:
                        heapq.heappush(
                            frontier,
                            (following.graph_hash(), position, following, store, stream),
                        )
                verified = []
                for value, store in bucket:
                    direct = capture.read_exact_stored_cdef(store, value)
                    if direct is None:
                        if self.indexed_required:
                            raise QueryIndexUnavailable(
                                "The derived index candidate lacks direct stored-root authority proof."
                            )
                        if self.scan_policy_mode == "forbid":
                            raise QueryWouldScanError(
                                "Verifying the derived index candidate requires an authority scan."
                            )
                        return None
                    verified.append((direct, self._source_evidence(store)))
                fixed = IdentitySet(verified)
                for key in sorted(fixed._entries, key=lambda item: item.sort_key):
                    selected[key] = fixed._entries[key]
                    if len(selected) == self.take_limit:
                        break
            return selected
        except QueryIndexDirty:
            if self.indexed_required or self.scan_policy_mode == "forbid":
                raise
            return None

    def _bounded_index_stores(self, capture):
        """Return producer Stores for the narrow paged stored-CDef plan shape."""

        from .source import RepoSource, StoreSource

        if not self._can_try_bounded_index():
            return None
        if isinstance(self.source, StoreSource):
            stores = (self.source.store,)
        else:
            stores = capture.repo_stores(self.source.repo)
        if not stores or any(
                getattr(store, "query_index_policy", None) in {"memory", "none"}
                for store in stores):
            return None
        return stores

    def _can_try_bounded_index(self) -> bool:
        """Return whether this plan can use canonical stored-CDef index pages."""

        from .source import RepoSource, StoreSource

        if self.take_limit is None or self.take_limit <= 0:
            return False
        if not isinstance(self.source, (StoreSource, RepoSource)):
            return False
        if not self.restrictions or not all(
                item.kind in {"kind", "stored"} for item in self.restrictions):
            return False
        if not any(
                item.kind == "kind" and item.value == "cdef"
                for item in self.restrictions):
            return False
        if any(
                item.kind == "kind" and item.value != "cdef"
                for item in self.restrictions):
            return False
        stored = tuple(item for item in self.restrictions if item.kind == "stored")
        if not stored:
            return False
        if isinstance(self.source, StoreSource):
            return all(
                isinstance(item.scope, StoreSource)
                and item.scope.store is self.source.store
                for item in stored
            )
        return all(
            isinstance(item.scope, RepoSource)
            and item.scope.repo is self.source.repo
            for item in stored
        )

    def _iter_terminal_members(self, capture, captured=None):
        """Yield qualified entries without constructing a final IdentitySet.

        Metadata predicates are validated across their eligible candidate domain
        before this iterator can stop.  That preserves late authority failures
        while allowing structural cardinality sinks to retain only one or two
        members.
        """

        if self.indexed_required and self._requires_authority_inventory():
            raise QueryIndexUnavailable("Query V3 index coverage is not available for this plan.")
        members, _, _ = self._captured_members(capture) if captured is None else captured
        if self.max_verify_limit is not None and len(members) > self.max_verify_limit:
            raise QueryVerifyBudgetExceeded("Query V3 verification budget exceeded.")
        generator_budget = _GeneratorWitnessBudget(self.max_witness_limit)
        validation_domain = members
        metadata_matches = {}
        for restriction in self.restrictions:
            if restriction.kind == "metadata":
                facts = self._scope_facts(restriction.scope, capture)
                metadata_matches[id(restriction)] = {
                    key: self._matches_metadata(entry[0], restriction.value, facts)
                    for key, entry in validation_domain.items()
                }
            elif restriction.kind == "source":
                validation_domain = {
                    key: entry for key, entry in validation_domain.items()
                    if self._source_evidence(restriction.value) in entry[1].sources
                }
        for key, entry in members.items():
            for restriction in self.restrictions:
                if restriction.kind == "metadata":
                    if not metadata_matches[id(restriction)].get(key, False):
                        break
                elif not self._matches(
                    restriction, entry[0], entry[1], capture, generator_budget,
                ):
                    break
            else:
                yield key, entry

    def _up_to_two_terminal_members(self, capture):
        """Keep only cardinality evidence required by singleton sinks."""

        items = []
        for item in self._iter_terminal_members(capture):
            items.append(item)
            if len(items) == 2:
                break
        return items

    def _requires_authority_inventory(self) -> bool:
        """Return whether this plan can reach Store authority through its source."""

        from .identity import IdentitySet

        if isinstance(self.source, IdentitySet):
            return False
        if isinstance(self.source, _RelationshipClosure):
            return self.source.input_query._requires_authority_inventory()
        if isinstance(self.source, _RelationshipProjection):
            return self.source.occurrence_query.roots._requires_authority_inventory()
        if isinstance(self.source, _AlgebraSource):
            return (
                self.source.left._requires_authority_inventory()
                or self.source.right._requires_authority_inventory()
            )
        return True

    def _requires_authority_scan(self) -> bool:
        """Return whether this plan requires broad authoritative inventory."""

        from .identity import IdentitySet

        if self.take_limit == 0 or isinstance(self.source, IdentitySet):
            return False
        if isinstance(self.source, _RelationshipClosure):
            return self.source.input_query._requires_authority_scan()
        if isinstance(self.source, _RelationshipProjection):
            return self.source.occurrence_query.roots._requires_authority_scan()
        if isinstance(self.source, _AlgebraSource):
            return (
                self.source.left._requires_authority_scan()
                or self.source.right._requires_authority_scan()
            )
        exact_state = self._exact_state_selector()
        needs_inventory = any(
            item.kind in {"alias", "stored"} for item in self.restrictions
        )
        return exact_state is None or bool(self._metadata_scopes()) or needs_inventory

    def _captured_members(self, capture=None):
        from .identity import IdentitySet
        from .source import (
            RepoSource,
            RetainedRepoLookupSource,
            SourceCapture,
            StoreSource,
        )
        from ..definition import ConcreteDefinition

        capture = SourceCapture() if capture is None else capture
        scopes = self._metadata_scopes()
        exact_state = self._exact_state_selector()
        needs_inventory = any(
            restriction.kind in {"alias", "stored"} for restriction in self.restrictions
        )
        exact_cdef = self._exact_stored_cdef_selector()
        if isinstance(self.source, StoreSource) and exact_cdef is not None:
            direct = capture.read_exact_stored_cdef(self.source, exact_cdef)
            if direct is not None:
                return IdentitySet(((direct, self._source_evidence(self.source)),))._entries, capture, False
        if isinstance(self.source, RepoSource) and exact_cdef is not None:
            repo = self.source.repo
            with repo.retain_topology(allow_physical_duplicates=True):
                stores = capture.repo_stores(repo)
                if stores and all(
                    capture.read_exact_stored_cdef(store, exact_cdef) is not None
                    for store in stores
                ):
                    members = [
                        (exact_cdef, self._source_evidence(store))
                        for store in stores
                    ]
                    if exact_cdef in capture.cache_knowledge(repo, weak=self.source.weak):
                        members.append((exact_cdef, self._source_evidence(repo)))
                    return IdentitySet(members)._entries, capture, False
        requires_scan = self._requires_authority_scan()
        if self.scan_policy_mode == "forbid" and requires_scan:
            raise QueryWouldScanError("Query V3 requires an authoritative inventory scan.")
        if (
                self.scan_policy_mode == "warn"
                and requires_scan
                and not getattr(capture, "scan_warning_emitted", False)):
            import warnings

            warnings.warn(
                "Query V3 requires an authoritative inventory scan.",
                RuntimeWarning,
                stacklevel=4,
            )
            capture.scan_warning_emitted = True
        if exact_state is not None and not scopes and not needs_inventory:
            if isinstance(self.source, StoreSource):
                state = capture.read_exact_state(self.source, exact_state)
                members = () if state is None else ((state, self._source_evidence(self.source)),)
                return IdentitySet(members)._entries, capture, False
            if isinstance(self.source, RepoSource):
                with self.source.repo.retain_topology(allow_physical_duplicates=True):
                    members = [
                        (state, self._source_evidence(store))
                        for store in capture.repo_stores(self.source.repo)
                        if (state := capture.read_exact_state(store, exact_state)) is not None
                    ]
                    if exact_state in capture.cache_knowledge(
                        self.source.repo, weak=self.source.weak,
                    ):
                        members.append((exact_state, self._source_evidence(self.source.repo)))
                return IdentitySet(members)._entries, capture, False
        if isinstance(self.source, IdentitySet):
            result = self.source
        elif isinstance(self.source, _AlgebraSource):
            left, left_bounded = self.source.left._evaluated_entries(capture)
            right, right_bounded = self.source.right._evaluated_entries(capture)
            if self.source.operation == "union":
                entries = dict(left)
                for key, (value, evidence) in right.items():
                    existing = entries.get(key)
                    entries[key] = (
                        value if existing is None else existing[0],
                        evidence if existing is None else existing[1].merged_with(evidence),
                    )
            else:
                entries = {
                    key: (value, evidence.merged_with(right[key][1]))
                    for key, (value, evidence) in left.items() if key in right
                }
            return entries, capture, left_bounded or right_bounded
        elif isinstance(self.source, _RelationshipClosure):
            result = self.source.collect(capture=capture)
        elif isinstance(self.source, _RelationshipProjection):
            result = self.source.collect(capture=capture)
        elif isinstance(self.source, StoreSource):
            result = capture.capture_store(self.source, metadata_scopes=scopes).knowledge()
        elif isinstance(self.source, RepoSource):
            result = capture.capture_repo(self.source, metadata_scopes=scopes)
        elif isinstance(self.source, RetainedRepoLookupSource):
            exact_cdefs = {
                restriction.value
                for restriction in self.restrictions
                if restriction.kind == "sel" and isinstance(restriction.value, ConcreteDefinition)
            }
            result = capture.capture_retained_lookup_repo(
                self.source,
                exact_cdef=next(iter(exact_cdefs)) if len(exact_cdefs) == 1 else None,
            )
        else:
            raise TypeError(
                "IdentityQuery source must be an IdentitySet, StoreSource, RepoSource, "
                "or retained Repo lookup source."
            )
        return dict(result._entries), capture, result.bounded

    def _exact_state_selector(self):
        """Return one terminal-selective StateRef filter, if the plan has one."""

        from ..reference_values import StateRef

        states = {
            restriction.value for restriction in self.restrictions
            if restriction.kind == "sel" and isinstance(restriction.value, StateRef)
        }
        return next(iter(states)) if len(states) == 1 else None

    def _exact_stored_cdef_selector(self):
        """Return one CDef eligible for the direct stored-root authority path."""

        from ..definition import ConcreteDefinition
        from .source import RepoSource, StoreSource

        if self._metadata_scopes() or not self.restrictions:
            return None
        exact = {
            item.value for item in self.restrictions
            if item.kind == "sel" and isinstance(item.value, ConcreteDefinition)
        }
        if len(exact) != 1 or not any(
                item.kind == "kind" and item.value == "cdef"
                for item in self.restrictions):
            return None
        if not all(
                item.kind in {"sel", "kind", "stored", "source"}
                for item in self.restrictions):
            return None
        if isinstance(self.source, StoreSource):
            same_scope = any(
                item.kind == "stored"
                and isinstance(item.scope, StoreSource)
                and item.scope.store is self.source.store
                for item in self.restrictions
            )
        elif isinstance(self.source, RepoSource):
            same_scope = any(
                item.kind == "stored"
                and isinstance(item.scope, RepoSource)
                and item.scope.repo is self.source.repo
                for item in self.restrictions
            )
        else:
            return None
        return next(iter(exact)) if same_scope else None

    def _scope_facts(self, scope, capture):
        from .source import RepoSource, StoreSource

        scopes = self._metadata_scopes()
        if isinstance(scope, StoreSource):
            return (capture.capture_store(scope, metadata_scopes=scopes),)
        if isinstance(scope, RepoSource):
            repo = scope.repo
            with repo.retain_topology(allow_physical_duplicates=True):
                return capture.capture_stores(capture.repo_stores(repo), metadata_scopes=scopes)
        raise TypeError("Query V3 authority scope is invalid.")

    def _matches(self, restriction, value, evidence, capture, generator_budget) -> bool:
        from ..definition import ConcreteDefinition, Definition
        from ..generator import GeneratorSelector
        from ..reference_values import ObjectRef, StateRef
        from ..selector import Selector
        from .identity import SourceEvidence

        if restriction.kind == "kind":
            return (
                (restriction.value == "cdef" and isinstance(value, ConcreteDefinition))
                or (restriction.value == "object_ref" and isinstance(value, ObjectRef))
                or (restriction.value == "state_ref" and isinstance(value, StateRef))
            )
        if restriction.kind == "source":
            return self._source_evidence(restriction.value) in evidence.sources
        if restriction.kind == "sel":
            selector = restriction.value
            if isinstance(selector, ObjectRef):
                return isinstance(value, (ObjectRef, StateRef)) and (
                    value.object if isinstance(value, StateRef) else value
                ) == selector
            if isinstance(selector, StateRef):
                return isinstance(value, StateRef) and value == selector
            target = value.definition if isinstance(value, (ObjectRef, StateRef)) else value
            if isinstance(selector, GeneratorSelector):
                return self._generator_matches(selector, target, generator_budget)
            if isinstance(selector, Selector):
                if isinstance(selector.root, ConcreteDefinition):
                    return selector.root.graph_equal(target)
                return _query_match(
                    selector.root, target, strict=selector.strict,
                    class_match=selector.cls_policy,
                )
            if isinstance(selector, ConcreteDefinition):
                return isinstance(target, ConcreteDefinition) and selector.graph_equal(target)
            return isinstance(selector, Definition) and _query_match(
                selector, target, strict=False, class_match="selector"
            )
        if restriction.kind == "object_id":
            ref = value.object if isinstance(value, StateRef) else value
            return isinstance(ref, ObjectRef) and restriction.value in ref.objects.values()
        if restriction.kind == "namespace":
            ref = value.object if isinstance(value, StateRef) else value
            return isinstance(ref, ObjectRef) and any(
                item.namespace[:len(restriction.value)] == restriction.value
                for item in ref.objects.values()
            )
        if restriction.kind == "contains":
            ref = value.object if isinstance(value, StateRef) else value
            return isinstance(ref, ObjectRef) and any(
                path and ref.at(path) == restriction.value for path in ref.objects
            )
        if restriction.kind == "state_hash":
            return isinstance(value, StateRef) and restriction.value in value.states.values()
        if restriction.kind == "stored":
            from .source import RepoSource, StoreSource

            if (
                isinstance(value, ConcreteDefinition)
                and isinstance(restriction.scope, StoreSource)
                and capture.has_exact_stored_cdef(restriction.scope, value)
            ):
                return True
            if (
                isinstance(value, ConcreteDefinition)
                and isinstance(restriction.scope, RepoSource)
                and capture.has_exact_stored_cdef_for_repo(restriction.scope.repo, value)
            ):
                return True
            return any(fact.is_stored(value) for fact in self._scope_facts(restriction.scope, capture))
        if restriction.kind == "cached":
            return value in capture.cache_knowledge(
                restriction.scope.repo, weak=restriction.value,
            )
        if restriction.kind == "alias":
            return self._matches_alias(value, restriction.value, self._scope_facts(restriction.scope, capture))
        if restriction.kind == "metadata":
            return self._matches_metadata(value, restriction.value, self._scope_facts(restriction.scope, capture))
        raise QueryDomainError("Query V3 restriction is unsupported.")

    @staticmethod
    def _generator_matches(selector, target, budget) -> bool:
        budget.consume()
        if not _query_match(selector.prefilter.root, target, strict=False, class_match="exact"):
            return False
        return selector.matches(target)

    @staticmethod
    def _matches_alias(value, alias, facts) -> bool:
        from ..reference_values import ObjectRef, StateRef
        from .model import QueryError

        targets = set()
        for fact in facts:
            targets.update(record.object_ref for record in fact.object_aliases if record.alias == alias)
            for record in fact.state_aliases:
                if record.alias != alias:
                    continue
                state = next((item.state_ref for item in fact.state_refs if item.digest == record.state_ref_digest), None)
                if state is None:
                    raise QueryError("Query V3 alias authority is unavailable.")
                targets.add(state)
        if len(targets) > 1:
            raise QueryError("Query V3 alias authority conflicts.")
        if not targets:
            return False
        target = next(iter(targets))
        if isinstance(target, ObjectRef):
            return (isinstance(value, ObjectRef) and value == target) or (
                isinstance(value, StateRef) and value.object == target
            )
        return isinstance(value, StateRef) and value == target

    @staticmethod
    def _source_evidence(source):
        """Return the same detached token used by U2 Store contribution capture."""

        from ..store.store import Store
        from .source import RepoSource, StoreSource
        from .identity import SourceEvidence

        if isinstance(source, StoreSource):
            source = source.store
        if isinstance(source, RepoSource):
            source = source.repo
        if isinstance(source, Store):
            return SourceEvidence.from_source(source.authority_fence_key())
        return SourceEvidence.from_source(source)

    @staticmethod
    def _matches_metadata(value, predicate, facts) -> bool:
        from ..reference_values import ObjectRef, StateRef
        from .metadata import (
            _MISSING, _leaves, _lineage_projection, _snapshot_projection,
            evaluate_captured_metadata_predicate, predicate_requires_state,
        )
        from .model import QueryError
        from .source import _MetadataReadFailure

        if predicate_requires_state(predicate) and not isinstance(value, StateRef):
            return False
        if not isinstance(value, (ObjectRef, StateRef)):
            return False
        projections = {}
        for scope in {leaf.field.scope for leaf in _leaves(predicate)}:
            values = []
            for fact in facts:
                target = value.object if scope in {"object", "lineage"} and isinstance(value, StateRef) else value
                try:
                    captured = fact.captured_metadata(target, scope)
                except KeyError:
                    continue
                values.append(captured)
            if not values:
                raise QueryError("Query V3 metadata authority is unavailable.")
            if any(isinstance(item, _MetadataReadFailure) for item in values):
                raise QueryError("Query V3 metadata read failed.")
            if any(item != values[0] for item in values[1:]):
                raise QueryError("Query V3 metadata authority conflicts.")
            captured = values[0]
            if scope in {"object", "state"}:
                projections[scope] = _MISSING if captured is None else captured
            elif scope == "lineage":
                projections[scope] = _lineage_projection(captured)
            else:
                projections[scope] = _snapshot_projection(captured)
        return evaluate_captured_metadata_predicate(predicate, value, projections)

    def _last_selector(self):
        from ..definition import ConcreteDefinition, Definition
        from ..selector import Selector

        for restriction in reversed(self.restrictions):
            if restriction.kind != "sel":
                continue
            value = restriction.value
            if isinstance(value, Selector):
                return value
            if isinstance(value, (Definition, ConcreteDefinition)):
                return Selector(
                    value,
                    exact_root=value if isinstance(value, ConcreteDefinition) else None,
                )
        return None


@dataclass(frozen=True, slots=True)
class _RelationshipClosure:
    """Deferred relationship expansion applied after an input identity plan."""

    input_query: IdentityQuery
    edges: Any
    max_depth: int | None

    def collect(self, *, capture=None):
        """Expand captured input identities without introducing source reads."""

        from .identity import IdentitySet
        from .source import SourceCapture
        from .relationships import relationship_closure

        roots = self.input_query._collect_with_capture(
            SourceCapture() if capture is None else capture
        )
        members = []
        for root in roots:
            for value in relationship_closure(
                    (root,), edges=self.edges, max_depth=self.max_depth,
            ):
                members.extend((value, source) for source in roots.evidence_for(root).sources)
                if not roots.evidence_for(root).sources:
                    members.append(value)
        return IdentitySet(members, bounded=roots.bounded)


@dataclass(frozen=True, slots=True)
class _RelationshipProjection:
    """Deferred existential owner or target projection over an occurrence plan."""

    occurrence_query: "OccurrenceQuery"
    projection: str

    def collect(self, *, capture=None):
        """Project identities without materializing raw occurrence paths."""

        from .identity import IdentitySet
        from .source import SourceCapture

        roots = self.occurrence_query.roots._collect_with_capture(
            SourceCapture() if capture is None else capture
        )
        members = []
        for value, sources in self.occurrence_query._project(self.projection, roots, capture):
            members.extend((value, source) for source in sources)
            if not sources:
                members.append(value)
        fixed = self.occurrence_query.fixed
        return IdentitySet(
            members, bounded=roots.bounded or (fixed.bounded if fixed is not None else False),
        )


@dataclass(frozen=True, slots=True)
class OccurrenceQuery:
    """Compose deferred strict non-empty Query V3 relationship occurrences.

    The query traverses only identities retained by ``roots``. Its raw terminal
    preserves every non-empty typed path; ``owners`` and ``targets`` use direct
    existential traversal and therefore do not inherit ``max_occurrences``.

    Attributes:
        roots: Identity query supplying explicit relationship roots.
        selector: Optional target selector.
        edges: Retained directed relationship policy.
        through_kinds: Required kinds on each matching typed path.
        path_filter: Optional exact typed or graph-field path restriction.
        occurrence_limit: Optional visible raw-output cap.
        max_depth_limit: Optional traversal depth cap.
        target_kind_filter: Optional complete identity kinds retained at terminals.

    Evaluation remains deferred until an explicit terminal and only inspects
    detached identities plus captured authority required by metadata filters.
    """

    roots: IdentityQuery
    selector: Any | None = None
    edges: Any = None
    through_kinds: frozenset[Any] = frozenset()
    path_filter: Any | None = None
    occurrence_limit: int | None = None
    max_depth_limit: int | None = None
    target_kind_filter: frozenset[str] | None = None
    fixed: Any | None = None
    metadata_restrictions: tuple[tuple[Any, Any], ...] = ()
    additional_selectors: tuple[Any, ...] = ()

    @classmethod
    def from_set(cls, occurrences) -> "OccurrenceQuery":
        """Refine captured occurrences without adopting a live producer.

        Args:
            occurrences: Detached :class:`OccurrenceSet` membership and evidence.

        Returns:
            A query restricted to the supplied fixed occurrences.

        Raises:
            TypeError: If ``occurrences`` is not an OccurrenceSet.

        Side Effects:
            None. No external authority is adopted or read.
        """

        from .identity import IdentitySet, OccurrenceSet
        from .relationships import EdgePolicy

        if not isinstance(occurrences, OccurrenceSet):
            raise TypeError("OccurrenceQuery.from_set requires an OccurrenceSet.")
        return cls(IdentityQuery.from_set(IdentitySet()), edges=EdgePolicy.ALL, fixed=occurrences)

    def __post_init__(self) -> None:
        from .identity import IdentitySet, OccurrenceSet
        from .relationships import EdgePolicy, RelationshipKind, RelationshipPath
        from ..utils.graph.path import GraphPath

        if not isinstance(self.roots, IdentityQuery):
            raise TypeError("OccurrenceQuery roots must be an IdentityQuery.")
        if self.fixed is not None and not isinstance(self.fixed, OccurrenceSet):
            raise TypeError("OccurrenceQuery fixed source must be an OccurrenceSet.")
        if not isinstance(self.edges, EdgePolicy):
            raise TypeError("OccurrenceQuery edges must be an EdgePolicy.")
        if not isinstance(self.through_kinds, frozenset) or not all(
                isinstance(kind, RelationshipKind) for kind in self.through_kinds
        ):
            raise TypeError("OccurrenceQuery through kinds must be RelationshipKind values.")
        if self.path_filter is not None and not isinstance(
                self.path_filter, (RelationshipPath, GraphPath)
        ):
            raise TypeError("OccurrenceQuery path must be a RelationshipPath or GraphPath.")
        if self.occurrence_limit is not None and (
                type(self.occurrence_limit) is not int or self.occurrence_limit < 0
        ):
            raise ValueError("max_occurrences must be a non-negative exact int or None.")
        if self.max_depth_limit is not None and (
                type(self.max_depth_limit) is not int or self.max_depth_limit < 0
        ):
            raise ValueError("max_depth must be a non-negative exact int or None.")
        if (
                self.target_kind_filter is not None
                and not self.target_kind_filter <= {"cdef", "object_ref", "state_ref"}):
            raise ValueError("OccurrenceQuery target kinds are unsupported.")
        if self.selector is not None:
            # Reuse the U3 selector validator without evaluating any source.
            IdentityQuery.from_set(IdentitySet()).sel(self.selector)
        for selector in self.additional_selectors:
            IdentityQuery.from_set(IdentitySet()).sel(selector)

    def sel(self, value=None) -> "OccurrenceQuery":
        """Conjoin a V3-compatible target selector without changing roots.

        Args:
            value: Any selector accepted by :meth:`IdentityQuery.sel`, or
                ``None`` for an immutable no-op.

        Returns:
            An immutable occurrence query with the target restriction appended.

        Raises:
            TypeError: Immediately if the selector is unsupported.

        Side Effects:
            None. Target matching remains deferred until a terminal executes.
        """

        if value is None:
            return self
        if self.selector is not None:
            return replace(self, additional_selectors=(*self.additional_selectors, value))
        return replace(self, selector=value)

    def where(self, predicate, *, scope=None) -> "OccurrenceQuery":
        """Restrict eligible occurrence targets using one captured authority scope.

        Args:
            predicate: Valid object, state, lineage, or snapshot predicate.
            scope: Store or Repo authority; defaults only for a live producer.

        Returns:
            An immutable target-restricted occurrence query.

        Raises:
            QueryDomainError: If a fixed source has no explicit authority scope.
            TypeError: If the predicate or source is unsupported.

        Side Effects:
            None until a terminal captures target metadata under source fences.
        """

        from .metadata import _require_predicate

        selected = self.roots.default_scope if scope is None else IdentityQuery._normalize_scope(scope)
        if selected is None:
            raise QueryDomainError("Fixed occurrences require an explicit scope for metadata.")
        return replace(
            self, metadata_restrictions=(*self.metadata_restrictions, (_require_predicate(predicate), selected)),
        )

    def cdefs(self) -> "OccurrenceQuery":
        """Restrict occurrence targets to CDefs without changing paths."""

        return replace(
            self,
            target_kind_filter=self._narrow_target_kinds("cdef"),
        )

    def object_refs(self) -> "OccurrenceQuery":
        """Restrict occurrence targets to ObjectRefs without expansion."""

        return replace(
            self,
            target_kind_filter=self._narrow_target_kinds("object_ref"),
        )

    def state_refs(self) -> "OccurrenceQuery":
        """Restrict occurrence targets to StateRefs without expansion."""

        return replace(
            self,
            target_kind_filter=self._narrow_target_kinds("state_ref"),
        )

    def target_kind(self, kind: str) -> "OccurrenceQuery":
        """Restrict targets to one explicit V3 identity kind.

        Args:
            kind: ``"cdef"``, ``"object_ref"``, or ``"state_ref"``.

        Returns:
            An immutable narrowed occurrence query.

        Raises:
            ValueError: If ``kind`` is unsupported.

        Side Effects:
            None until a terminal executes.
        """

        if kind not in {"cdef", "object_ref", "state_ref"}:
            raise ValueError("OccurrenceQuery target kind is unsupported.")
        return replace(
            self,
            target_kind_filter=self._narrow_target_kinds(kind),
        )

    def _narrow_target_kinds(self, kind: str) -> frozenset[str]:
        current = (
            frozenset(("cdef", "object_ref", "state_ref"))
            if self.target_kind_filter is None
            else self.target_kind_filter
        )
        return current & frozenset((kind,))

    def through(self, kind) -> "OccurrenceQuery":
        """Require one relationship kind on each retained typed path.

        Args:
            kind: RelationshipKind that must occur on the path.

        Returns:
            An immutable path-restricted occurrence query.

        Raises:
            TypeError: If ``kind`` is not a RelationshipKind.

        Side Effects:
            None until a terminal traverses relationships.
        """

        from .relationships import RelationshipKind

        if not isinstance(kind, RelationshipKind):
            raise TypeError("OccurrenceQuery through requires a RelationshipKind.")
        return replace(self, through_kinds=self.through_kinds | frozenset((kind,)))

    def path(self, value) -> "OccurrenceQuery":
        """Require an exact RelationshipPath or concatenated GraphPath.

        Args:
            value: Exact typed relationship path or graph-field path.

        Returns:
            An immutable path-restricted occurrence query.

        Raises:
            TypeError: If ``value`` is not a RelationshipPath or GraphPath.

        Side Effects:
            None until a terminal traverses relationships.
        """

        from .relationships import RelationshipPath
        from ..utils.graph.path import GraphPath

        if not isinstance(value, (RelationshipPath, GraphPath)):
            raise TypeError("OccurrenceQuery path requires a RelationshipPath or GraphPath.")
        return replace(self, path_filter=value)

    def max_occurrences(self, limit: int | None) -> "OccurrenceQuery":
        """Set the raw-occurrence cap without bounding direct projections.

        Args:
            limit: Non-negative exact occurrence count, or ``None`` for no cap.

        Returns:
            An immutable occurrence query carrying the output cap.

        Raises:
            ValueError: If ``limit`` is negative, inexact, or not ``None``.

        Side Effects:
            None until a raw occurrence terminal executes.
        """

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_occurrences must be a non-negative exact int or None.")
        return replace(self, occurrence_limit=limit)

    def take(self, limit: int) -> "OccurrenceSet":
        """Return a fixed canonical prefix without collecting all raw paths.

        Args:
            limit: Non-negative exact number of globally distinct occurrences.

        Returns:
            A bounded OccurrenceSet with its requested limit and captured evidence.

        Raises:
            ValueError: If the limit is not an exact non-negative integer.

        Side Effects:
            Evaluates selected roots and necessary metadata under a retryable cut;
            zero-limit requests validate statically without reading sources.
        """

        from .identity import OccurrenceSet

        if type(limit) is not int or limit < 0:
            raise ValueError("take limit must be a non-negative exact int.")
        if limit == 0:
            return OccurrenceSet(bounded=True, requested_limit=0)
        return self.roots._run_terminal(lambda cut: self._take_with_capture(cut, limit))

    def _take_with_capture(self, capture, limit):
        from .identity import OccurrenceSet

        roots = self.roots._collect_with_capture(capture)
        selected = []
        for occurrence in self._iter_occurrences(capture, roots=roots):
            sources = (
                self.fixed.evidence_for(occurrence)
                if self.fixed is not None else roots.evidence_for(occurrence.owner)
            )
            selected.append((occurrence.key, (occurrence, sources)))
            selected.sort(key=cmp_to_key(_compare_bounded_occurrences))
            if len(selected) > limit:
                selected.pop()
        return OccurrenceSet._from_entries(
            dict(selected), bounded=True, requested_limit=limit,
        )

    def max_depth(self, limit: int | None) -> "OccurrenceQuery":
        """Set a failing relationship-depth safety budget for traversal.

        Args:
            limit: Non-negative exact depth, or ``None`` for no bound.

        Returns:
            An immutable occurrence query carrying the traversal budget.

        Raises:
            ValueError: If ``limit`` is negative, inexact, or not ``None``.

        Side Effects:
            None until a terminal traverses relationships.
        """

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_depth must be a non-negative exact int or None.")
        return replace(self, max_depth_limit=limit)

    def collect(self):
        """Evaluate typed raw occurrences into a fixed OccurrenceSet.

        Returns:
            A deterministic fixed result with detached evidence and visible bounds.

        Raises:
            QueryError: If root capture, metadata validation, or traversal budgets
                fail.

        Side Effects:
            Reads root authority and required target metadata under retryable cuts.
        """

        return self.roots._run_terminal(self._collect_with_capture)

    def _collect_with_capture(self, capture):
        """Capture roots and target metadata in the same retryable terminal."""

        from .identity import OccurrenceSet
        from .relationships import iter_relationship_occurrences

        roots = self.roots._collect_with_capture(capture)
        matches = self._target_matcher(roots, capture)
        if self.fixed is not None:
            members = []
            for occurrence in self._iter_fixed_occurrences(matches=matches):
                sources = self.fixed.evidence_for(occurrence).sources
                members.extend((occurrence, source) for source in sources)
                if not sources:
                    members.append(occurrence)
            return OccurrenceSet(
                members,
                bounded=self.fixed.bounded or self.occurrence_limit is not None,
                requested_limit=self.fixed.requested_limit if self.occurrence_limit is None else self.occurrence_limit,
            )
        occurrences = iter_relationship_occurrences(
            roots,
            matches,
            edges=self.edges,
            through=self.through_kinds,
            path=self.path_filter,
            max_occurrences=self.occurrence_limit,
            max_depth=self.max_depth_limit,
        )
        members = []
        for occurrence in occurrences:
            sources = roots.evidence_for(occurrence.owner).sources
            members.extend((occurrence, source) for source in sources)
            if not sources:
                members.append(occurrence)
        return OccurrenceSet(
            members, bounded=self.occurrence_limit is not None or roots.bounded,
        )

    def count(self) -> int:
        """Stream the distinct raw typed-occurrence count.

        Returns:
            Number of retained occurrences after any explicit raw-output cap.

        Raises:
            QueryError: If root capture, metadata validation, or traversal fails.

        Side Effects:
            Executes traversal without constructing a complete fixed result.
        """

        return self.roots._run_terminal(lambda cut: sum(1 for _ in self._iter_occurrences(cut)))

    def exists(self) -> bool:
        """Return whether one qualified occurrence exists.

        Returns:
            ``True`` after the first valid occurrence, otherwise ``False``.

        Raises:
            QueryError: If root capture, metadata validation, or traversal fails.

        Side Effects:
            Executes only traversal work needed for a conclusive answer.
        """

        return self.roots._run_terminal(lambda cut: next(self._iter_occurrences(cut), None) is not None)

    def one(self):
        """Return the sole qualified occurrence.

        Returns:
            The one complete typed occurrence.

        Raises:
            QueryCardinalityError: If zero or multiple occurrences remain.
            QueryError: If root capture, metadata validation, or traversal fails.

        Side Effects:
            Executes enough traversal to prove singleton cardinality.
        """

        occurrences = self.roots._run_terminal(self._up_to_two_occurrences)
        if len(occurrences) != 1:
            raise QueryCardinalityError(
                f"Expected exactly one occurrence, found {len(occurrences)}."
            )
        return occurrences[0]

    def one_or_none(self):
        """Return zero or one occurrence while rejecting ambiguity.

        Returns:
            The sole occurrence, or ``None`` when no occurrence remains.

        Raises:
            QueryCardinalityError: If multiple occurrences remain.
            QueryError: If root capture, metadata validation, or traversal fails.

        Side Effects:
            Executes enough traversal to prove at-most-one cardinality.
        """

        occurrences = self.roots._run_terminal(self._up_to_two_occurrences)
        if len(occurrences) > 1:
            raise QueryCardinalityError(
                f"Expected zero or one occurrence, found {len(occurrences)}."
            )
        return occurrences[0] if occurrences else None

    def owners(self) -> IdentityQuery:
        """Return a deferred direct existential projection of qualified roots."""

        return IdentityQuery(
            _RelationshipProjection(self, "owners"), default_scope=self.roots.default_scope,
        )

    def targets(self) -> IdentityQuery:
        """Return a deferred direct existential projection of qualified targets."""

        return IdentityQuery(
            _RelationshipProjection(self, "targets"), default_scope=self.roots.default_scope,
        )

    def _project(self, projection: str, roots, capture):
        from .relationships import iter_relationship_owners, iter_relationship_targets

        matches = self._target_matcher(roots, capture)
        if self.fixed is not None:
            for occurrence in self._iter_fixed_occurrences(ignore_limit=True, matches=matches):
                yield (
                    occurrence.owner if projection == "owners" else occurrence.target,
                    self.fixed.evidence_for(occurrence).sources,
                )
            return
        walker = (
            iter_relationship_owners if projection == "owners" else iter_relationship_targets
        )
        for root in roots:
            for value in walker(
                (root,), matches, edges=self.edges,
                through=self.through_kinds, path=self.path_filter,
                max_depth=self.max_depth_limit,
            ):
                yield value, roots.evidence_for(root).sources

    def _iter_occurrences(self, capture, *, roots=None):
        """Stream raw occurrences while retaining source evaluation once per terminal."""

        from .relationships import iter_relationship_occurrences

        roots = self.roots._collect_with_capture(capture) if roots is None else roots
        matches = self._target_matcher(roots, capture)
        if self.fixed is not None:
            yield from self._iter_fixed_occurrences(matches=matches)
            return
        yield from iter_relationship_occurrences(
            roots,
            matches,
            edges=self.edges,
            through=self.through_kinds,
            path=self.path_filter,
            max_occurrences=self.occurrence_limit,
            max_depth=self.max_depth_limit,
        )

    def _iter_fixed_occurrences(self, *, ignore_limit=False, matches=None):
        """Restrict captured paths without expanding them or reading a source."""

        from .relationships import RelationshipPath, _path_matches, _through_matches

        emitted = 0
        if self.occurrence_limit == 0 and not ignore_limit:
            return
        if matches is None:
            matches = self._matches_target
        for occurrence in self.fixed:
            if not matches(occurrence.target):
                continue
            path = occurrence.path
            if self.path_filter is not None:
                if isinstance(path, RelationshipPath):
                    if not _path_matches(path, self.path_filter):
                        continue
                elif path != self.path_filter:
                    continue
            if self.through_kinds and (
                not isinstance(path, RelationshipPath)
                or not _through_matches(path, self.through_kinds)
            ):
                continue
            yield occurrence
            emitted += 1
            if not ignore_limit and self.occurrence_limit is not None and emitted >= self.occurrence_limit:
                return

    def _up_to_two_occurrences(self, capture):
        """Retain only the cardinality evidence needed by singleton terminals."""

        occurrences = []
        for occurrence in self._iter_occurrences(capture):
            occurrences.append(occurrence)
            if len(occurrences) == 2:
                break
        return occurrences

    def _target_matcher(self, roots, capture):
        """Validate eligible target metadata before any scalar can stop early."""

        if not self.metadata_restrictions:
            return self._matches_target
        from .identity import IdentitySet, identity_key
        from .relationships import iter_relationship_targets

        if self.fixed is not None:
            candidates = (
                occ.target for occ in self._iter_fixed_occurrences(
                    ignore_limit=True, matches=lambda value: True,
                )
            )
        else:
            candidates = iter_relationship_targets(
                roots, lambda value: True, edges=self.edges,
                through=self.through_kinds, path=self.path_filter,
                max_depth=self.max_depth_limit,
            )
        query = IdentityQuery.from_set(IdentitySet(candidates))
        for predicate, scope in self.metadata_restrictions:
            query = query.where(predicate, scope=scope)
        qualified = {identity_key(value) for value in query._collect_with_capture(capture)}
        return lambda value: self._matches_target(value) and identity_key(value) in qualified

    def _matches_target(self, value) -> bool:
        from .identity import IdentitySet
        from ..definition import ConcreteDefinition
        from ..reference_values import ObjectRef, StateRef

        kind = (
            "cdef" if isinstance(value, ConcreteDefinition)
            else "object_ref" if isinstance(value, ObjectRef)
            else "state_ref" if isinstance(value, StateRef)
            else None
        )
        if kind is None or (
                self.target_kind_filter is not None
                and kind not in self.target_kind_filter):
            return False
        if self.selector is None and not self.additional_selectors:
            return True
        query = IdentityQuery.from_set(IdentitySet((value,)))
        for selector in ((self.selector,) if self.selector is not None else ()) + self.additional_selectors:
            query = query.sel(selector)
        return query.exists()

    def __bool__(self) -> bool:
        raise TypeError("OccurrenceQuery requires an explicit terminal.")
