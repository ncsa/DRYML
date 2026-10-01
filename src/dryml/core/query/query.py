from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from heapq import merge
from typing import Any
import warnings

from ..canonical import matching_container_family
from ..definition import ConcreteDefinition, Definition, selector_match
from ..freeze import FrozenDict, FrozenList, FrozenSet, FrozenTuple
from ..links import DefLink
from ..object import Object
from ..params import Match
from ..quoted import QuotedDef, SelectorSpec
from ..selector import Selector
from ..symbol import maybe_symbol_ref, resolve_symbol
from ..errors import TemplateLimitError
from ..utils.types import is_nonclass_callable
from .graph_plan import graph_candidate_ids
from .lowering import ScanPolicy
from .domain import CachedDomain, KnownDomain, NestedDomain, StoredDomain
from .model import (
    ClassMatchPolicy,
    ContainmentContext,
    ContainmentEdgePolicy,
    DefinitionId,
    QueryDomain,
    QueryDomainError,
    QueryExplanation,
    QueryCardinalityError,
    QueryIndexError,
    QueryVerifyBudgetExceeded,
    QueryProjection,
    SourceQueryPlan,
    QueryStats,
    QueryWouldScanError,
    RefreshPolicy,
    ResultUniverse,
    containment_witness_key,
    is_exact_reference_target,
    validate_containment_policy,
    validate_containment_target,
)
from .containment import (
    iter_containment_occurrences_matching,
    iter_containment_owners_matching,
    iter_containment_targets_matching,
)
from .path import DefinitionPath, DefinitionPathLike, Kwarg, Parameter, QueryPathError, get_subtree, iter_value_edges, normalize_path, replace_subtree
from .result import DefinitionResultSet, ObjectResultSet, OccurrenceResultSet
from .selector_graph import compile_selector_graph
from .utils import cdef_equal


@dataclass(frozen=True, slots=True)
class CapturedNestedCandidates:
    generation: int
    cdefs_by_id: dict[DefinitionId, ConcreteDefinition]
    stats: QueryStats


class _QueryGenerationChanged(Exception):
    pass


_MAX_NESTED_QUERY_RETRIES = 3
_DEFAULT_TEMPLATE_WITNESS_LIMIT = 65_536
_DEFAULT_MAX_WITNESSES = object()


@dataclass(slots=True)
class _TemplateWitnessBudget:
    limit: int | None
    visited: int = 0

    def consume(self) -> None:
        self.visited += 1
        if self.limit is not None and self.visited > self.limit:
            raise TemplateLimitError("template query witness limit exceeded")


@dataclass(frozen=True, slots=True)
class DefinitionQuery:
    """Immutable structural query builder with exact-reference preservation.

    A query snapshots and resolves soft ``StateSelectorRef`` leaves before it is
    finalized, then executes against the selected structural domain. Reference
    leaves remain complete ``ObjectRef`` or ``StateRef`` values; callers use
    :meth:`references` for authority-verified reference projections.

    Attributes:
        repo: Managing Repo used for selector resolution and domain access.
        original: Resolved source retained for restore operations.
        selector: Current resolved structural selector.
        max_witness_limit: Exact-template authoritative-visit cap. The internal
            default is 65,536; visits consume it before safe prefilter rejection
            or duplicate suppression, and ``None`` permits an unbounded scan.
        template_selector: Optional exact TemplateSelector residual applied
            after structural prefiltering. Residual queries reject rewrites
            that cannot preserve exact support semantics.
    """
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
    occurrence_limit: int | None = None
    scan_policy_mode: str = "allow"
    max_verify_limit: int | None = None
    max_witness_limit: int | None | object = _DEFAULT_MAX_WITNESSES
    template_selector: Any | None = None
    containment_target: Any | None = None
    containment_edges: ContainmentEdgePolicy = "materialize"
    contains_ref: bool = False
    source_store: Any | None = None
    _original_values: tuple[tuple[DefinitionPath, Any], ...] = ()

    @classmethod
    def from_source(
            cls,
            repo,
            source=None,
            *,
            domain: QueryDomain | None = None,
            universe: ResultUniverse | None = None) -> "DefinitionQuery":
        """Create a finalized query from a soft structural source.

        Args:
            repo: Managing Repo required to resolve StateSelectorRef leaves.
            source: Optional Definition, CDef, ObjectRef, StateRef, Selector,
                TemplateSelector, or Object source. Exact references are lazy
                containment targets and cannot select ordinary authority domains.
                TemplateSelector support retains an exact residual beside its
                ordinary structural prefilter.
            domain: Optional initial query domain.
            universe: Optional fixed result universe for refinement.

        Returns:
            A query whose selector contains no soft StateSelectorRef.

        Raises:
            TypeError: If ``source`` has an unsupported type or no selector
                resolver is available.
            KeyError: If a selected state alias is missing.
            ValueError: If a selector resolves outside its ObjectRef scope.
            RepoLoadError: If connected Stores provide conflicting alias authority.
        """
        from ..reference_values import ObjectRef, StateRef
        from ..template_selector import TemplateSelector

        if isinstance(source, (ObjectRef, StateRef)):
            return cls(
                repo=repo,
                original=None,
                selector=None,
                domain=domain,
                universe=universe,
                containment_target=validate_containment_target(source),
            )

        if isinstance(source, TemplateSelector):
            prefilter = _resolve_query_state_selectors(source.prefilter.root, repo)
            return cls(
                repo=repo,
                original=prefilter,
                selector=prefilter,
                domain=domain,
                universe=universe,
                class_match_policy="exact",
                template_selector=source,
            )
        if isinstance(source, Selector):
            root = _resolve_query_state_selectors(source.root, repo)
            return cls(
                repo=repo,
                original=root,
                selector=root,
                domain=domain,
                universe=universe,
                strict_policy=source.strict,
                class_match_policy=source.cls_policy,
            )
        original = _resolve_query_state_selectors(_snapshot_source(source), repo)
        target = original if isinstance(original, ConcreteDefinition) else None
        return cls(
            repo=repo,
            original=original,
            selector=original,
            domain=domain,
            universe=universe,
            containment_target=target,
        )

    def references(self):
        """Start an authority-verified lightweight reference query.

        Returns:
            A ReferenceQuery whose optional structural source filters referenced
            CDefs without materializing Objects or loading state payloads.

        Raises:
            QueryDomainError: If an exact TemplateSelector residual is attached;
                ReferenceQuery cannot retain topology witness semantics, if this
                query has an exact reference containment target, or if nested
                containment has already been selected.

        Side Effects:
            None. This conversion does not query Store authority.
        """

        if self.template_selector is not None:
            raise QueryDomainError("TemplateSelector queries cannot be converted to ReferenceQuery.")
        if is_exact_reference_target(self.containment_target):
            raise QueryDomainError(
                "Exact-reference containment queries cannot be converted to reference authority."
            )
        if self.domain == "nested":
            raise QueryDomainError(
                "Nested containment queries cannot be converted to reference authority."
            )
        from .reference import ReferenceQuery

        return ReferenceQuery(self.repo, definition=self.selector)

    def where(self, predicate):
        """Conjoin a typed metadata predicate with this query's references.

        Args:
            predicate: A :class:`MetadataPredicate` constructed with
                :func:`dryml.core.query.field`.

        Returns:
            An immutable ReferenceQuery with this query's structural selector and
            the supplied metadata constraint.

        Raises:
            TypeError: If ``predicate`` is not a MetadataPredicate.
            ValueError: If predicate construction bounds are invalid.

        Side Effects:
            None until a ReferenceQuery terminal runs; that terminal never
            materializes Objects or opens state payloads.
        """

        return self.references().where(predicate)

    def categorical(
            self,
            *,
            path: DefinitionPathLike = "$",
            recursive: bool = False,
            drop=(),
            drop_args: bool = False,
            drop_class: bool = False) -> "DefinitionQuery":
        """Project one query occurrence onto named categorical constraints.

        Args:
            path: Root-relative occurrence to project.
            recursive: Whether to project nested definitions.
            drop: Constructor parameter names to omit from the selected traversal.
            drop_args: Whether to omit every argument constraint.
            drop_class: Whether to omit class constraints.

        Returns:
            An immutable query retaining original authority for :meth:`exact` and
            :meth:`restore`. The resulting selector is a selection expression,
            not a construction request.

        Raises:
            QueryPathError: If the query or selected occurrence is unconstrained.
            TypeError: If projection controls or selected values are invalid.
            ValueError: If requested names are absent, a cycle is encountered, or
                projection collapses a set.

        Side Effects:
            Does not materialize Objects or alter source definitions, Store
            records, or references. It may resolve an authored class only when
            required to name supplied call constraints; CDef parameter projection
            reads stored names without resolving its class.
        """

        self._reject_exact_reference_rewrite("categorical")
        self._reject_residual_rewrite("categorical")
        if self.selector is None:
            raise QueryPathError("Cannot apply semantic categorical projection to an unconstrained query.")
        from ..categorical import _project_categorical_definition_with_origins

        norm = normalize_path(path)
        subtree = get_subtree(self.selector, norm)
        projected, origins = _project_categorical_definition_with_origins(
            subtree,
            recursive=recursive,
            drop=drop,
            drop_args=drop_args,
            drop_class=drop_class,
        )
        selector = replace_subtree(
            self.selector,
            norm,
            projected,
            _origins=origins,
        )
        previous_values = dict(self._original_values)
        original_values = {}
        for selector_path, (source_path, source_value) in _projection_origin_paths(
                selector, self.selector, origins).items():
            original_values[selector_path] = (
                previous_values[source_path]
                if source_path is not None and source_path in previous_values
                else source_value
            )
        return replace(
            self,
            selector=selector,
            _original_values=tuple(original_values.items()),
        )

    def restore(self, *, path: DefinitionPathLike = "$") -> "DefinitionQuery":
        self._reject_exact_reference_rewrite("restore")
        self._reject_residual_rewrite("restore")
        if self.original is None or self.selector is None:
            raise QueryPathError("Cannot restore() on an unconstrained query.")
        norm = normalize_path(path)
        subtree = self._original_value(norm)
        replacement = deepcopy(subtree) if isinstance(subtree, Definition) else subtree
        return replace(self, selector=replace_subtree(self.selector, norm, replacement))

    def exact(
            self,
            definition: ConcreteDefinition | Object | None = None,
            *,
            path: DefinitionPathLike = "$") -> "DefinitionQuery":
        self._reject_exact_reference_rewrite("exact")
        self._reject_residual_rewrite("exact")
        if self.selector is None:
            raise QueryPathError("Cannot apply exact() to an unconstrained query.")
        norm = normalize_path(path)
        if definition is None:
            if self.original is None:
                raise QueryPathError(f"Cannot infer exact subtree at {norm!s}; query has no original source.")
            definition = self._original_value(norm)
        if isinstance(definition, Object):
            definition = definition.definition
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError(f"Exact constraint at {norm!s} requires a ConcreteDefinition, got {type(definition).__name__}.")
        return replace(self, selector=replace_subtree(self.selector, norm, definition))

    def _original_value(self, path: DefinitionPath) -> Any:
        """Return the original authority retained for one selector occurrence."""

        original_values = dict(self._original_values)
        if path in original_values:
            return original_values[path]
        if self.original is None:
            raise QueryPathError("Cannot resolve an original value without a source query.")
        return get_subtree(self.original, self._original_path(path))

    def _original_path(self, path: DefinitionPath) -> DefinitionPath:
        """Translate selector path segments to the original's V2 boundaries.

        Categorical projections retain legacy call spelling, while each V2 CDef
        in the original is addressed by semantic ``Parameter`` segments.

        Args:
            path: Path expressed against the current selector.

        Returns:
            The equivalent path through the original source.

        Raises:
            QueryPathError: If the path cannot be resolved through either tree.
        """

        from .selector_graph import _semantic_selector_path

        original = self.original
        selector = self.selector
        translated = []
        for segment in path:
            original_path = DefinitionPath((segment,))
            if isinstance(original, ConcreteDefinition):
                semantic = _semantic_selector_path(selector, DefinitionPath((segment,)))
                if semantic is not None:
                    original_path = semantic
            translated.extend(original_path.segments)
            original = get_subtree(original, original_path)
            selector = get_subtree(selector, DefinitionPath((segment,)))
        return DefinitionPath(tuple(translated))

    def class_match(self, policy: ClassMatchPolicy) -> "DefinitionQuery":
        self._reject_exact_reference_rewrite("class_match")
        if policy not in {"selector", "exact"}:
            raise ValueError("class_match policy must be 'selector' or 'exact'.")
        # Exact support comparisons intentionally do not resolve inheritance.
        return replace(self, class_match_policy="exact" if self.template_selector is not None else policy)

    def strict(self, enabled: bool = True) -> "DefinitionQuery":
        self._reject_exact_reference_rewrite("strict")
        return replace(self, strict_policy=bool(enabled))

    def refresh(self, policy: RefreshPolicy = "auto") -> "DefinitionQuery":
        if policy not in {False, "auto", True}:
            raise ValueError("refresh policy must be False, 'auto', or True.")
        return replace(self, refresh_policy=policy)

    def reuse_weak(self, enabled: bool = True) -> "DefinitionQuery":
        return replace(self, reuse_weak_policy=bool(enabled))

    def stored(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._reject_exact_reference_domain("stored")
        self._check_universe_domain_switch("stored")
        q = replace(self, domain="stored", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def cached(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._reject_exact_reference_domain("cached")
        self._check_universe_domain_switch("cached")
        q = replace(self, domain="cached", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def known(self, *, refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        self._reject_exact_reference_domain("known")
        self._check_universe_domain_switch("known")
        q = replace(self, domain="known", projection=None)
        return q if refresh is None else q.refresh(refresh)

    def nested(
            self,
            *,
            edges: ContainmentEdgePolicy = "materialize",
            contains_ref: bool = False,
            refresh: RefreshPolicy | None = None) -> "DefinitionQuery":
        """Select immutable stored-root containment with literal edge controls.

        Concrete CDef targets match structurally; exact ObjectRef and StateRef
        targets retain complete typed identity as terminal graph values. Selecting
        containment neither resolves nor loads a reference target.

        Args:
            edges: ``"materialize"`` (the compatibility default), ``"ref"``,
                or ``"all"``. The selected edge kind applies at every hop.
            contains_ref: Exact boolean retaining only qualifying paths that
                include at least one retained reference edge.
            refresh: Optional existing derived-index refresh policy.

        Returns:
            An immutable nested query preserving its target, policy, and later
            authoritative source scope.

        Raises:
            ValueError: If ``edges`` or ``refresh`` is unsupported.
            TypeError: If ``contains_ref`` is not an exact bool.

        Side Effects:
            None. This records containment intent without scanning Stores,
            resolving references, or materializing Objects.
        """

        edges, contains_ref = validate_containment_policy(edges, contains_ref)
        self._check_universe_domain_switch("nested")
        if self.universe is not None and self.universe.containment is not None:
            context = self.universe.containment
            if context.edges != edges or context.contains_ref != contains_ref:
                raise QueryDomainError(
                    "Cannot change traversal policy for a fixed containment result universe."
                )
            if context.target_kind != "definition" and not (
                    self.containment_target is None
                    or is_exact_reference_target(self.containment_target)):
                raise QueryDomainError(
                    "Exact-reference containment results require exact reference refinement."
                )
        q = replace(
            self,
            domain="nested",
            projection=None,
            containment_edges=edges,
            contains_ref=contains_ref,
        )
        return q if refresh is None else q.refresh(refresh)

    def in_store(self, store) -> "DefinitionQuery":
        """Restrict a nested containment query to one connected Store handle.

        Args:
            store: Exact Store instance already connected to this query's Repo.

        Returns:
            An immutable containment query scoped to ``store``.

        Raises:
            QueryDomainError: If this is not a nested query.
            ValueError: If ``store`` is not currently connected to the Repo.

        Side Effects:
            None. Store authority is not captured until supported containment
            execution is selected.
        """

        if self.domain != "nested":
            raise QueryDomainError("in_store() is only valid for nested containment queries.")
        if self.universe is not None and self.universe.containment is not None:
            raise QueryDomainError(
                "Cannot change source scope for a fixed containment result universe."
            )
        if not any(candidate is store for candidate in self.repo.stores):
            raise ValueError("in_store() requires a connected Store handle.")
        return replace(self, source_store=store)

    def _check_universe_domain_switch(self, requested: str) -> None:
        if self.universe is None:
            return
        if requested != self.universe.domain:
            raise QueryDomainError(
                f"Cannot switch a fixed ResultSet universe from domain {self.universe.domain!r} to {requested!r}."
            )

    def definitions(self) -> "DefinitionQuery":
        """Project a nested CDef query onto matching contained definitions.

        Returns:
            An immutable query selecting the nested-definition projection.

        Raises:
            QueryDomainError: If an exact ObjectRef or StateRef target would be
                coerced into a CDef projection.

        Side Effects:
            None. Projection selection does not execute the query.
        """

        if self.domain != "nested":
            return self
        self._require_cdef_target_projection("definitions")
        return replace(self, projection="definitions")

    def owners(self) -> "DefinitionQuery":
        """Project a nested containment query onto enclosing stored CDefs.

        Returns:
            An immutable query selecting authoritative stored-root owners.

        Raises:
            QueryDomainError: If nested containment has not been selected.

        Side Effects:
            None. Projection selection does not execute the query.
        """

        if self.domain != "nested":
            raise QueryDomainError("owners() is only valid for nested queries.")
        return replace(self, projection="owners")

    def object_refs(self):
        """Return exact ObjectRef containment values when execution supports them.

        Fixed containment universes project their retained complete ObjectRef
        evidence without rescanning Stores. Backend-owned fresh containment
        execution remains unavailable until an authority traversal is selected.

        Raises:
            QueryDomainError: If the target is not an ObjectRef, nested domain is
                not selected, or fresh exact-reference containment execution is
                unavailable.

        Returns:
            An ObjectRefResultSet for fixed containment evidence.

        Side Effects:
            Fixed-universe projection has no Store side effects.
        """

        self._require_reference_target_projection("object_refs", "ObjectRef")
        return replace(self, projection="object_refs").execute()

    def state_refs(self):
        """Return exact StateRef containment values when execution supports them.

        Fixed containment universes project their retained complete StateRef
        evidence without rescanning Stores. Backend-owned fresh containment
        execution remains unavailable until an authority traversal is selected.

        Raises:
            QueryDomainError: If the target is not a StateRef, nested domain is
                not selected, or fresh exact-reference containment execution is
                unavailable.

        Returns:
            A StateRefResultSet for fixed containment evidence.

        Side Effects:
            Fixed-universe projection has no Store side effects.
        """

        self._require_reference_target_projection("state_refs", "StateRef")
        return replace(self, projection="state_refs").execute()

    def max_occurrences(self, limit: int | None) -> "DefinitionQuery":
        if limit is not None and limit < 0:
            raise ValueError("max_occurrences limit must be non-negative or None.")
        # This bounds path enumeration. Capturing the candidate ancestor subgraph is terminal-specific.
        return replace(self, occurrence_limit=limit)

    def scan_policy(self, policy: str) -> "DefinitionQuery":
        if policy not in {"allow", "warn", "forbid"}:
            raise ValueError("scan policy must be 'allow', 'warn', or 'forbid'.")
        return replace(self, scan_policy_mode=policy)

    def require_indexed(self) -> "DefinitionQuery":
        return self.scan_policy("forbid")

    def max_verify(self, limit: int | None) -> "DefinitionQuery":
        if limit is not None and limit < 0:
            raise ValueError("max_verify limit must be non-negative or None.")
        return replace(self, max_verify_limit=limit)

    def max_witnesses(self, limit: int | None) -> "DefinitionQuery":
        """Set the residual witness-discovery cap for one terminal execution.

        Args:
            limit: A positive exact integer, or ``None`` to disable the cap.

        Returns:
            A query copy retaining every existing structural and exact constraint.

        Raises:
            ValueError: If ``limit`` is bool, zero, negative, or non-integral.
        """

        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_witnesses limit must be a positive exact int or None.")
        return replace(self, max_witness_limit=limit)

    def _reject_residual_rewrite(self, operation: str) -> None:
        """Reject selector rewrites that cannot preserve an exact residual."""

        if self.template_selector is not None:
            raise QueryDomainError(f"TemplateSelector queries cannot apply {operation}().")

    def _reject_exact_reference_rewrite(self, operation: str) -> None:
        """Reject structural rewrites for an exact reference containment target."""

        if is_exact_reference_target(self.containment_target):
            raise QueryDomainError(
                f"Exact-reference containment queries cannot apply {operation}()."
            )

    def _reject_exact_reference_domain(self, domain: str) -> None:
        """Reject ordinary definition authority domains for exact references."""

        if is_exact_reference_target(self.containment_target):
            raise QueryDomainError(
                f"Exact-reference containment queries can only select nested(), not {domain}()."
            )

    def _require_cdef_target_projection(self, projection: str) -> None:
        """Require a CDef-compatible target for a nested definition projection."""

        if is_exact_reference_target(self.containment_target):
            raise QueryDomainError(
                f"{projection}() is unavailable for exact {type(self.containment_target).__name__} containment targets."
            )

    def _require_reference_target_projection(self, projection: str, expected: str) -> None:
        """Validate an exact-reference value terminal before unsupported execution."""

        if self.domain != "nested":
            raise QueryDomainError(f"{projection}() is only valid for nested containment queries.")
        if type(self.containment_target).__name__ != expected:
            actual = (
                type(self.containment_target).__name__
                if self.containment_target is not None else "CDef selector"
            )
            raise QueryDomainError(f"{projection}() requires an exact {expected} containment target, got {actual}.")

    @property
    def lowering_scan_policy(self) -> ScanPolicy:
        return ScanPolicy(self.scan_policy_mode, self.max_verify_limit)

    def execute(self):
        self._require_domain()
        if self.template_selector is not None:
            return self._execute_template_selector()
        if self.domain == "nested":
            if self.universe is None and self.projection == "definitions":
                cdefs, stats = self._execute_nested_definitions()
                explanation = self._explanation(stats)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=False,
                    domain="nested-definitions",
                    explanation=explanation,
                    replicas={},
                )
            if self.universe is None and self.projection == "owners":
                cdefs, stats, replicas = self._execute_nested_owners()
                explanation = self._explanation(stats)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=True,
                    domain="owners",
                    explanation=explanation,
                    replicas=replicas,
                )
            occs, stats, owner_replicas = self._execute_nested_occurrences()
            explanation = self._explanation(stats)
            containment = None if self.universe is None else self.universe.containment
            containment_witnesses = () if self.universe is None else self.universe.containment_witnesses
            if containment is None and self.universe is None:
                from .model import containment_target_kind

                target_kind = (
                    containment_target_kind(self.containment_target)
                    if self.containment_target is not None else "definition"
                )
                containment = ContainmentContext(
                    target_kind=target_kind,
                    edges=self.containment_edges,
                    contains_ref=self.contains_ref,
                    source_scope=self._containment_source_scope(),
                    complete=True,
                    bounded=(
                        self.occurrence_limit is not None
                        and self.projection not in {"object_refs", "state_refs"}
                    ),
                )
            if (
                    containment is not None
                    and self.occurrence_limit is not None
                    and self.projection not in {"object_refs", "state_refs"}
            ):
                containment = replace(containment, bounded=True)
            if callable(occs):
                raw = OccurrenceResultSet(
                    self.repo,
                    occurrence_factory=occs,
                    explanation=explanation,
                    owner_replicas=owner_replicas,
                    containment=containment,
                    containment_witnesses=containment_witnesses,
                )
            else:
                raw = OccurrenceResultSet(
                    self.repo,
                    occs,
                    explanation=explanation,
                    owner_replicas=owner_replicas,
                    containment=containment,
                    containment_witnesses=containment_witnesses,
                )
            if self.projection == "definitions":
                return raw.definitions()
            if self.projection == "owners":
                return raw.owners()
            if self.projection == "object_refs":
                return raw.object_refs()
            if self.projection == "state_refs":
                return raw.state_refs()
            return raw

        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            query_backed = getattr(self.repo._query_index, "query_backed_definition_result_set", None)
            if query_backed is not None:
                result_set = query_backed(self)
                if result_set is not None:
                    return result_set

        cdefs, stats, query_replicas = self._execute_definition_domain()
        explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
        materializable = True
        domain = self.domain or "stored"
        if self.universe is not None:
            materializable = self.universe.materializable
            domain = self.universe.domain
        replicas = {} if query_replicas is None else query_replicas
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
        return self._execute_count()

    def exists(self) -> bool:
        if self.template_selector is not None:
            return self._execute_count() > 0
        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            count, _ = self.repo._query_index.count_definition_domain(self, stop_after=1)
            return count > 0
        if self.universe is None and self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            return bool(self._execute_federated_known_domain(stop_after=1)[0])
        if self.universe is None and self.domain == "nested":
            return bool(self._execute_terminal_items(stop_after=1))
        return self._execute_count() > 0

    def one(self):
        items = self._execute_terminal_items(stop_after=2)
        if len(items) != 1:
            label = "occurrence" if self.domain == "nested" and self.projection is None else "result"
            raise QueryCardinalityError(f"Expected exactly one {label}, found {len(items)}.")
        return items[0]

    def one_or_none(self):
        items = self._execute_terminal_items(stop_after=2)
        if len(items) > 1:
            label = "occurrence" if self.domain == "nested" and self.projection is None else "result"
            raise QueryCardinalityError(f"Expected zero or one {label}, found {len(items)}.")
        return items[0] if items else None

    def explain(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        return self._execute_explanation(analyze=analyze, sql=sql)

    def _execute_count(self) -> int:
        self._require_domain()
        if self.template_selector is not None:
            return len(self._execute_template_selector())
        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            count, _ = self.repo._query_index.count_definition_domain(self)
            return count
        if self.universe is None and self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            cdefs, _, _ = self._execute_federated_known_domain()
            return len(cdefs)
        if self.domain == "nested":
            if self.projection in {"object_refs", "state_refs"}:
                return len(self.execute())
            if self.universe is None and self.projection == "definitions":
                if not self._uses_authoritative_containment_residual() and self.repo._query_index.can_execute_query_domain("nested"):
                    cdefs, _ = self.repo._query_index.execute_nested_definitions(self)
                else:
                    cdefs, _ = self._execute_nested_definitions()
                return len(cdefs)
            if self.universe is None and self.projection == "owners":
                if not self._uses_authoritative_containment_residual() and self.repo._query_index.can_execute_query_domain("nested"):
                    cdefs, _, _ = self.repo._query_index.execute_nested_owners(self)
                else:
                    cdefs, _, _ = self._execute_nested_owners()
                return len(cdefs)
            occurrences, _, _ = self._execute_nested_occurrences()
            if callable(occurrences):
                return sum(1 for _ in occurrences())
            return len(occurrences)

        cdefs, _, _ = self._execute_definition_domain()
        return len(cdefs)

    def _execute_explanation(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        self._require_domain()
        if self.template_selector is not None:
            if not analyze:
                stats = QueryStats(
                    scan_required=self.universe is None,
                    scan_reason=(
                        "TemplateSelector queries require complete witness scanning."
                        if self.universe is None else None
                    ),
                )
                return self._explanation(stats)
            result = self._execute_template_selector()
            return result.explanation or QueryStats(result_count=len(result)).explanation(
                domain=self._domain_label(), refresh=self.refresh_policy,
            )
        if analyze:
            result = self.execute()
            explanation = result.explanation
            if explanation is None:
                return QueryStats(result_count=len(result)).explanation(domain=self._domain_label(), refresh=self.refresh_policy)
            return explanation
        if self.domain == "nested":
            if self._uses_authoritative_containment_residual() and not analyze:
                stats = QueryStats(
                    scan_required=True,
                    scan_reason="reference-aware containment requires authoritative root verification",
                )
                return self._explanation(stats)
            if self.universe is None and self.projection == "definitions":
                _, stats = self._execute_nested_definitions()
            elif self.universe is None and self.projection == "owners":
                _, stats, _ = self._execute_nested_owners()
            else:
                _, stats, _ = self._execute_nested_occurrences()
            return self._explanation(stats)

        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            stats = self.repo._query_index.explain_definition_domain(self, sql=sql)
            return stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)

        _, stats, _ = self._execute_definition_domain()
        return stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)

    def _require_domain(self) -> None:
        if self.domain is None:
            raise QueryDomainError("Select a query domain with stored(), cached(), known(), or nested() before executing.")

    def _domain_label(self) -> str:
        if self.domain == "nested" and self.projection is not None:
            return f"nested-{self.projection}"
        return self.domain or "unset"

    def _explanation(self, stats: QueryStats) -> QueryExplanation:
        """Attach immutable containment policy to an execution explanation."""

        explanation = stats.explanation(
            domain=self._domain_label(), refresh=self.refresh_policy,
        )
        if self.domain != "nested":
            return explanation
        from .model import containment_target_kind

        target_kind = (
            containment_target_kind(self.containment_target)
            if self.containment_target is not None else "definition"
        )
        return replace(
            explanation,
            containment_target_kind=target_kind,
            containment_edges=self.containment_edges,
            containment_contains_ref=self.contains_ref,
            containment_source_scope=self._containment_source_scope(),
        )

    def _execute_definition_domain(self):
        stats = QueryStats()
        if self.universe is not None:
            if self.universe.kind != "definitions":
                raise QueryDomainError("A definition terminal cannot execute over an occurrence universe.")
            stats.universe_size = len(self.universe.definitions)
            matches = self._verify_cdefs(tuple(self.universe.definitions), stats=stats)
            stats.result_count = len(matches)
            replicas = {}
            if self.universe.replicas is not None:
                replicas = {cdef: self.universe.replicas.get(cdef, ()) for cdef in matches}
            return matches, stats, replicas

        if self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            return self.repo._query_index.execute_definition_domain(self)
        if self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            return self._execute_federated_known_domain()

        catalog = self.repo._query_catalog
        exact_root = self.selector if isinstance(self.selector, ConcreteDefinition) else None
        if self.refresh_policy is True:
            catalog.refresh(True, stats=stats)
        elif exact_root is not None and self.domain in {"stored", "known"} and self.refresh_policy is not False:
            catalog.ensure_exact_stored(exact_root, stats=stats)
        elif self.domain in {"stored", "known"}:
            catalog.refresh(self.refresh_policy, stats=stats)

        live_domain = self._definition_domain(catalog)
        live_domain.prepare(stats=stats)
        with catalog.read_view(include_cached=self.domain in {"cached", "known"}) as snapshot:
            domain = live_domain.with_catalog(snapshot)
            if exact_root is not None and self.domain in {"stored", "cached", "known"}:
                candidate_ids = domain.filter(snapshot.exact_ids(exact_root))
                stats.universe_size = len(candidate_ids)
                stats.candidate_count = len(candidate_ids)
            else:
                stats.universe_size = domain.estimate_size()
                selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
                if selector_graph is not None:
                    candidate_ids = graph_candidate_ids(snapshot, selector_graph, domain, stats=stats)
                else:
                    candidate_ids = domain.all_ids()
                    stats.candidate_count = len(candidate_ids)
                    stats.universe_size = len(candidate_ids)
            cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            replicas = snapshot.replica_map(candidate_ids)
        cdefs = tuple(cdefs_by_id.values())
        matches = self._verify_cdefs(cdefs, stats=stats)
        stats.result_count = len(matches)
        replicas = {cdef: replicas.get(cdef, ()) for cdef in matches}
        return matches, stats, replicas

    def _execute_template_selector(self):
        """Verify exact support against complete graph-distinct domain witnesses.

        Query indexes intentionally collapse structural CDefs, so they cannot be
        used as terminal authority for topology-sensitive template support.  This
        path scans authoritative roots (or retained immutable result witnesses)
        before restoring the public structural-deduplication result semantics.
        """

        if self.scan_policy_mode == "forbid" and self.universe is None:
            raise QueryDomainError("TemplateSelector queries require complete witness scanning.")
        if self.domain == "nested" and not (
                self.universe is not None and self.universe.kind == "definitions"):
            return self._execute_template_selector_nested()

        stats = QueryStats(refresh_action="template-witness-scan")
        witness_budget = self._template_witness_budget()
        witnesses = self._template_definition_witnesses(stats, witness_budget)
        matches, replicas, verified_witnesses = self._verify_template_witnesses(witnesses, stats)
        stats.result_count = len(matches)
        explanation = stats.explanation(domain=self._domain_label(), refresh=self.refresh_policy)
        materializable = self.universe.materializable if self.universe is not None else True
        domain = self.universe.domain if self.universe is not None else self.domain or "stored"
        return DefinitionResultSet(
            self.repo,
            matches,
            materializable=materializable,
            domain=domain,
            explanation=explanation,
            replicas=replicas,
            witnesses=verified_witnesses,
            witness_complete=True,
        )

    def _template_witness_budget(self) -> _TemplateWitnessBudget:
        limit = (
            _DEFAULT_TEMPLATE_WITNESS_LIMIT
            if self.max_witness_limit is _DEFAULT_MAX_WITNESSES
            else self.max_witness_limit
        )
        return _TemplateWitnessBudget(limit)

    def _template_definition_witnesses(
            self,
            stats: QueryStats,
            witness_budget: _TemplateWitnessBudget):
        """Stream complete roots after a conservative structural index prefilter."""

        if self.universe is not None:
            if self.universe.kind != "definitions":
                raise QueryDomainError("A definition terminal cannot execute over an occurrence universe.")
            if not self.universe.witness_complete:
                raise QueryDomainError("TemplateSelector refinement requires complete immutable witness evidence.")
            replicas = self.universe.replicas or {}

            def universe_witnesses():
                for cdef in self.universe.witnesses:
                    witness_budget.consume()
                    yield cdef, tuple(replicas.get(cdef, ()))

            return universe_witnesses()

        indexed_candidates = None
        if (
            self.domain in {"stored", "known"}
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            prefilter_query = replace(
                self,
                domain="stored",
                template_selector=None,
                max_witness_limit=_DEFAULT_MAX_WITNESSES,
            )
            candidates, index_stats, _ = self.repo._query_index.execute_definition_domain(
                prefilter_query
            )
            indexed_candidates = set(candidates)
            stats.refresh_action = index_stats.refresh_action
            stats.universe_size = index_stats.universe_size
            stats.selected_features = index_stats.selected_features
            stats.posting_sizes = index_stats.posting_sizes
            stats.source_plans = index_stats.source_plans
            stats.generation_vector = index_stats.generation_vector
            stats.lowering_strategy = index_stats.lowering_strategy
            stats.scan_required = index_stats.scan_required
            stats.scan_reason = index_stats.scan_reason

        def roots():
            seen_cached: set[int] = set()
            if self.domain in {"stored", "known"}:
                for store in self.repo.stores:
                    iterate = getattr(store, "iter_authoritative_root_definitions", None)
                    if not callable(iterate):
                        raise QueryDomainError(
                            "TemplateSelector stored queries require authoritative root enumeration."
                        )
                    for cdef in self._iter_template_store_roots(store, iterate):
                        witness_budget.consume()
                        if indexed_candidates is None or cdef in indexed_candidates:
                            yield cdef, (store,)
            if self.domain in {"cached", "known"}:
                caches = (self.repo.strong_obj_cache, self.repo.weak_obj_cache)
                for cache in caches:
                    for _, obj in cache.items():
                        witness_budget.consume()
                        marker = id(obj)
                        if marker in seen_cached:
                            continue
                        seen_cached.add(marker)
                        yield obj.definition, ()

        if self.domain not in {"stored", "cached", "known"}:
            raise QueryDomainError(f"Unsupported TemplateSelector definition domain {self.domain!r}.")
        return roots()

    @staticmethod
    def _iter_template_store_roots(store, iterate):
        """Validate one Store's streamed exact-template witness authority."""

        try:
            for cdef in iterate():
                if not isinstance(cdef, ConcreteDefinition):
                    raise QueryDomainError(
                        "TemplateSelector Store enumeration yielded a non-CDef root."
                    )
                yield cdef
        except QueryDomainError:
            raise
        except Exception as error:
            raise QueryDomainError(
                f"TemplateSelector Store enumeration failed for {store!r}."
            ) from error

    def _verify_template_witnesses(self, witnesses, stats: QueryStats):
        """Verify each witness before structural result representative selection."""

        from ..template_selector import _AssignmentBudget

        assignment_budget = _AssignmentBudget(self.template_selector._max_assignments)
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        verified_witnesses = []
        for cdef, stores in witnesses:
            stats.candidate_count += 1
            matched = self._verify_cdefs(
                (cdef,), stats=stats, template_budget=assignment_budget
            )
            if not matched:
                continue
            match = matched[0]
            canonical = merged.setdefault(match, match)
            replicas.setdefault(canonical, []).extend(stores)
            verified_witnesses.append(match)
        ordered = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        return ordered, {cdef: tuple(dict.fromkeys(replicas.get(cdef, ()))) for cdef in ordered}, tuple(verified_witnesses)

    def _execute_template_selector_nested(self):
        """Apply exact support over shared root-local containment witnesses.

        Template support must inspect every selected candidate occurrence before a
        raw-result cap is applied: each visit participates in the existing
        witness and assignment budgets.  The containment walker supplies the
        literal edge policy and complete hops without collapsing private graph
        topology into the materialization-only graph helper.
        """

        stats = QueryStats(refresh_action="template-witness-scan")
        witness_budget = self._template_witness_budget()
        from ..template_selector import _AssignmentBudget

        assignment_budget = _AssignmentBudget(self.template_selector._max_assignments)

        def matches(value) -> bool:
            # Exact selectors only accept CDefs. Exact references remain terminal
            # containment values and never become template-selector candidates.
            if not isinstance(value, ConcreteDefinition):
                return False
            witness_budget.consume()
            stats.candidate_count += 1
            return bool(self._verify_cdefs(
                (value,), stats=stats, template_budget=assignment_budget,
            ))

        if self.universe is not None:
            context = self.universe.containment
            complete = self.universe.witness_complete or (
                context is not None and context.complete
            )
            if self.universe.kind != "occurrences" or not complete:
                raise QueryDomainError("TemplateSelector refinement requires complete immutable witness evidence.")
            replicas = self.universe.replicas or {}
            evidence = (
                self.universe.containment_witnesses
                if context is not None else self.universe.witnesses
            )

            def candidates():
                if self.projection == "owners":
                    for owner in self.universe.witnesses:
                        if matches(owner):
                            for occurrence in evidence:
                                if occurrence.owner is owner:
                                    yield occurrence, tuple(replicas.get(owner, ()))
                    return
                for occurrence in evidence:
                    if matches(occurrence.definition):
                        yield occurrence, tuple(replicas.get(occurrence.owner, ()))
        else:
            reason = "TemplateSelector queries require complete witness scanning."
            stats.scan_required = True
            stats.scan_reason = reason
            if self.scan_policy_mode == "warn":
                warnings.warn(
                    f"DRYML query requires scan fallback: {reason}",
                    RuntimeWarning,
                    stacklevel=3,
                )
            stores = self._containment_stores()

            def candidates():
                with self.repo._authority_read_fences(stores):
                    for store in stores:
                        iterate = getattr(store, "iter_authoritative_root_definitions", None)
                        if not callable(iterate):
                            raise QueryDomainError(
                                "TemplateSelector nested queries require authoritative root enumeration."
                            )
                        stats.store_scan_count += 1
                        for root in self._iter_template_store_roots(store, iterate):
                            yield from (
                                (occurrence, (store,))
                                for occurrence in iter_containment_occurrences_matching(
                                    (root,), matches,
                                    edges=self.containment_edges,
                                    contains_ref=self.contains_ref,
                                )
                            )

            context = ContainmentContext(
                target_kind="definition",
                edges=self.containment_edges,
                contains_ref=self.contains_ref,
                source_scope=self._containment_source_scope(),
                complete=True,
                bounded=self.occurrence_limit is not None,
            )

        merged: dict[tuple[Any, ...], Any] = {}
        owner_replicas: dict[ConcreteDefinition, list[Any]] = {}
        for occurrence, stores in candidates():
            owner_replicas.setdefault(occurrence.owner, []).extend(stores)
            merged.setdefault(containment_witness_key(occurrence), occurrence)
        occurrences = tuple(merged[key] for key in sorted(merged))
        owner_replicas = {
            owner: tuple(dict.fromkeys(stores))
            for owner, stores in owner_replicas.items()
        }
        if self.projection == "definitions":
            definitions = tuple(occ.definition for occ in occurrences)
            stats.result_count = len(dict.fromkeys(definitions))
            return DefinitionResultSet(
                self.repo, definitions, materializable=False, domain="nested-definitions",
                explanation=self._explanation(stats), replicas={}, witnesses=definitions, witness_complete=True,
                containment=context, containment_witnesses=occurrences,
            )
        if self.projection == "owners":
            owners = tuple(occ.owner for occ in occurrences)
            stats.result_count = len(dict.fromkeys(owners))
            return DefinitionResultSet(
                self.repo, owners, materializable=True, domain="owners", explanation=self._explanation(stats),
                replicas=owner_replicas, witnesses=owners, witness_complete=True,
                containment=context, containment_witnesses=occurrences,
                containment_carrier="owner",
            )
        visible = occurrences
        if self.projection not in {"object_refs", "state_refs"} and self.occurrence_limit is not None:
            visible = visible[:self.occurrence_limit]
        stats.result_count = len(visible)
        return OccurrenceResultSet(
            self.repo, visible, explanation=self._explanation(stats), owner_replicas=owner_replicas,
            witnesses=occurrences, witness_complete=True,
            containment=context, containment_witnesses=occurrences,
        )

    def _execute_federated_known_domain(self, *, stop_after: int | None = None):
        from .federation import CACHE_SOURCE_KEY

        stored_query = replace(self, domain="stored")
        stored_cdefs, stored_stats, stored_replicas = self.repo._query_index.execute_definition_domain(stored_query, stop_after=stop_after)

        if stop_after is not None and len(stored_cdefs) >= stop_after:
            stats = QueryStats(refresh_action="federated-known")
            stats.store_scan_count = stored_stats.store_scan_count
            stats.candidate_count = stored_stats.candidate_count
            stats.verified_count = stored_stats.verified_count
            stats.result_count = len(stored_cdefs)
            stats.universe_size = None
            stats.generation_vector = dict(stored_stats.generation_vector or {})
            stats.source_plans = stored_stats.source_plans
            return stored_cdefs, stats, stored_replicas

        cached_query = replace(self, domain="cached")
        cached_cdefs, cached_stats, cached_replicas = cached_query._execute_definition_domain()
        cache_generation = self.repo._query_catalog.current_generation()

        merged = {cdef: cdef for cdef in stored_cdefs}
        for cdef in cached_cdefs:
            merged.setdefault(cdef, cdef)
            if stop_after is not None and len(merged) >= stop_after:
                break

        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        replicas = {}
        for cdef in out:
            if cdef in stored_replicas:
                replicas[cdef] = stored_replicas[cdef]
            else:
                replicas[cdef] = cached_replicas.get(cdef, ())

        stats = QueryStats(refresh_action="federated-known")
        stats.store_scan_count = stored_stats.store_scan_count + cached_stats.store_scan_count
        stats.candidate_count = stored_stats.candidate_count + cached_stats.candidate_count
        stats.verified_count = stored_stats.verified_count + cached_stats.verified_count
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = dict(stored_stats.generation_vector or {})
        stats.generation_vector[CACHE_SOURCE_KEY] = cache_generation
        stats.source_plans = (*stored_stats.source_plans, SourceQueryPlan(
            source_key=CACHE_SOURCE_KEY,
            backend="memory-cache",
            generation=cache_generation,
            candidate_count=cached_stats.candidate_count,
            verified_count=cached_stats.verified_count,
            result_count=cached_stats.result_count,
            refresh_action=cached_stats.refresh_action,
        ))
        return out, stats, replicas

    def _execute_terminal_items(self, *, stop_after: int):
        self._require_domain()
        if self.template_selector is not None:
            return tuple(item for _, item in zip(range(stop_after), self._execute_template_selector()))
        if self.domain == "nested":
            if (
                    self.universe is None
                    and self.projection == "definitions"
                    and not self._uses_authoritative_containment_residual()
                    and self.repo._query_index.can_execute_query_domain("nested")
            ):
                return self.repo._query_index.execute_nested_definitions(self, stop_after=stop_after)[0]
            if (
                    self.universe is None
                    and self.projection == "owners"
                    and not self._uses_authoritative_containment_residual()
                    and self.repo._query_index.can_execute_query_domain("nested")
            ):
                return self.repo._query_index.execute_nested_owners(self, stop_after=stop_after)[0]
            if self.universe is None and self.projection is None:
                limit = stop_after if self.occurrence_limit is None else min(self.occurrence_limit, stop_after)
                occurrences, _, _ = replace(self, occurrence_limit=limit)._execute_nested_occurrences()
                if callable(occurrences):
                    return tuple(item for _, item in zip(range(stop_after), occurrences()))
                return tuple(occurrences[:stop_after])
            result = self.execute()
            return tuple(item for _, item in zip(range(stop_after), result))

        if self.universe is None and self.domain == "stored" and self.repo._query_index.can_execute_query_domain("stored"):
            return self.repo._query_index.execute_definition_domain(self, stop_after=stop_after)[0]
        if self.universe is None and self.domain == "known" and self.repo._query_index.can_execute_query_domain("stored"):
            return self._execute_federated_known_domain(stop_after=stop_after)[0]
        cdefs, _, _ = self._execute_definition_domain()
        return tuple(cdefs[:stop_after])

    def _definition_domain(self, catalog):
        if self.domain == "stored":
            return StoredDomain(catalog)
        if self.domain == "cached":
            return CachedDomain(catalog, reuse_weak=self.reuse_weak_policy)
        if self.domain == "known":
            return KnownDomain(catalog, reuse_weak=self.reuse_weak_policy)
        raise QueryDomainError(f"Unsupported definition domain {self.domain!r}.")

    def _execute_nested_occurrences(self):
        if self.universe is not None:
            stats = QueryStats()
            if self.universe.kind != "occurrences":
                raise QueryDomainError("A nested query cannot execute over a definition universe.")
            evidence = self.universe.containment_witnesses or self.universe.occurrences
            stats.universe_size = len(evidence)
            if self.universe.containment is not None:
                context = self.universe.containment
                target = self.containment_target
                if target is not None:
                    from .model import containment_target_kind

                    if containment_target_kind(target) != context.target_kind:
                        raise QueryDomainError(
                            "Fixed containment refinement cannot change the exact target kind."
                        )
                    if is_exact_reference_target(target):
                        out = tuple(item for item in evidence if item.target == target)
                    else:
                        out = tuple(
                            item for item in evidence
                            if isinstance(item.target, ConcreteDefinition)
                            and _query_match(
                                self.selector,
                                item.target,
                                strict=self.strict_policy,
                                class_match=self.class_match_policy,
                            )
                        )
                elif context.target_kind == "definition" and self.selector is not None:
                    out = tuple(
                        item for item in evidence
                        if isinstance(item.target, ConcreteDefinition)
                        and _query_match(
                            self.selector,
                            item.target,
                            strict=self.strict_policy,
                            class_match=self.class_match_policy,
                        )
                    )
                else:
                    out = tuple(evidence)
                if self.occurrence_limit is not None:
                    out = out[:self.occurrence_limit]
                stats.result_count = len(out)
                return out, stats, self.universe.replicas
            verified_nested = self._verify_cdefs(
                tuple({occ.definition for occ in self.universe.occurrences}),
                stats=stats,
            )
            verified = set(verified_nested)
            out = tuple(occ for occ in self.universe.occurrences if occ.definition in verified)
            if self.occurrence_limit is not None:
                out = out[:self.occurrence_limit]
            stats.result_count = len(out)
            return out, stats, self.universe.replicas

        if self._is_deterministic_empty_containment():
            stats = QueryStats(fast_path="deterministic-empty-containment")
            stats.result_count = 0
            return (), stats, {}

        if self._uses_authoritative_containment_residual():
            return self._execute_authoritative_containment_occurrences()

        if self.repo._query_index.can_execute_query_domain("nested"):
            return self.repo._query_index.execute_nested_occurrences(self)

        catalog = self.repo._query_catalog
        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            stats = QueryStats()
            captured = self._capture_nested_candidates(catalog, stats)
            _, match_ids = self._verify_cdefs_by_id(captured.cdefs_by_id, stats=stats)
            try:
                traversal = self._capture_occurrence_traversal(catalog, match_ids, captured.generation)
                break
            except _QueryGenerationChanged:
                continue
        else:
            raise QueryIndexError("Catalog generation changed repeatedly during nested occurrence query.")

        def occurrence_factory():
            return traversal.iter_occurrences(max_occurrences=self.occurrence_limit)

        return occurrence_factory, stats, traversal.owner_replicas

    def _execute_nested_definitions(self) -> tuple[tuple[ConcreteDefinition, ...], QueryStats]:
        if self._is_deterministic_empty_containment():
            stats = QueryStats(fast_path="deterministic-empty-containment", result_count=0)
            return (), stats
        if self._uses_authoritative_containment_residual():
            return self._execute_authoritative_containment_definitions()
        if self.universe is None and self.repo._query_index.can_execute_query_domain("nested"):
            return self.repo._query_index.execute_nested_definitions(self)
        matches, _, stats, _ = self._execute_nested_definition_matches()
        stats.result_count = len(matches)
        return matches, stats

    def _execute_nested_owners(self):
        if self._is_deterministic_empty_containment():
            stats = QueryStats(fast_path="deterministic-empty-containment", result_count=0)
            return (), stats, {}
        if self._uses_authoritative_containment_residual():
            return self._execute_authoritative_containment_owners()
        if self.universe is None and self.repo._query_index.can_execute_query_domain("nested"):
            return self.repo._query_index.execute_nested_owners(self)
        catalog = self.repo._query_catalog
        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            matches, match_ids, stats, generation = self._execute_nested_definition_matches()
            try:
                projection = self._project_owners(catalog, match_ids, generation)
                break
            except _QueryGenerationChanged:
                continue
        else:
            raise QueryIndexError("Catalog generation changed repeatedly during nested owner query.")
        owners = projection.cdefs
        owner_replicas = projection.replicas
        stats.result_count = len(owners)
        owners = tuple(sorted(owners, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        return owners, stats, {cdef: owner_replicas.get(cdef, ()) for cdef in owners}

    def _is_deterministic_empty_containment(self) -> bool:
        """Return whether literal traversal policy makes nested membership empty.

        Materialize-only traversal cannot contain a reference hop.  Classifying
        this before touching a sidecar or Store lets strict indexed policy retain
        its valid no-scan empty result.
        """

        return (
            self.universe is None
            and self.domain == "nested"
            and self.containment_edges == "materialize"
            and self.contains_ref
        )

    def _uses_authoritative_containment_residual(self) -> bool:
        """Return whether nested execution needs root-local authority evidence.

        The legacy materialize-only CDef paths remain index-backed.  Ref-aware,
        exact-reference, and source-restricted queries have no sidecar
        completeness certificate, so their results must be derived from a
        fenced authoritative root cut.
        """

        return (
            self.universe is None
            and self.domain == "nested"
            and not self._is_deterministic_empty_containment()
            and (
                self.containment_edges != "materialize"
                or self.contains_ref
                or is_exact_reference_target(self.containment_target)
                or self.source_store is not None
            )
        )

    def _containment_source_scope(self) -> tuple[str, ...]:
        """Return canonical source keys selected for fresh containment evidence."""

        return tuple(sorted(self._containment_source_key(store) for store in self._containment_stores()))

    def _containment_stores(self) -> tuple[Any, ...]:
        """Return selected physical sources once, in Repo priority order."""

        selected = (self.source_store,) if self.source_store is not None else tuple(self.repo.stores)
        stores = []
        seen = set()
        for store in selected:
            key = self._containment_source_key(store)
            if key not in seen:
                seen.add(key)
                stores.append(store)
        return tuple(stores)

    @staticmethod
    def _containment_source_key(store) -> str:
        """Return the stable source identity used by containment evidence."""

        if hasattr(store, "catalog_key"):
            return store.catalog_key()
        return f"{type(store).__module__}.{type(store).__qualname__}:id:{id(store)}"

    def _refresh_authoritative_containment_sources(self, stores, stats: QueryStats) -> None:
        """Perform allowed sidecar recovery before taking an authority cut."""

        if self.refresh_policy is False:
            return
        for store in stores:
            index = store.open_query_index()
            if index is not None:
                index.refresh(self.refresh_policy, stats=stats)

    def _capture_authoritative_containment_roots(self, stats: QueryStats):
        """Detach complete selected roots under existing Store authority fences.

        Sidecars can be stale, incomplete, or absent, therefore they are not
        consulted to choose roots.  This U4 capture deliberately retains no
        open index view or Store fence after returning.
        """

        stores = self._containment_stores()
        reason = "reference-aware containment requires authoritative root verification"
        stats.scan_required = True
        stats.scan_reason = reason
        if self.scan_policy_mode == "forbid":
            raise QueryWouldScanError(reason)
        if self.scan_policy_mode == "warn":
            warnings.warn(
                f"DRYML query requires scan fallback: {reason}",
                RuntimeWarning,
                stacklevel=3,
            )

        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            # Recovery has its own short fence; capture only after it releases it.
            self._refresh_authoritative_containment_sources(stores, stats)
            before = {
                self._containment_source_key(store): store.query_index_status()
                for store in stores
            }
            entries = []
            source_plans = []
            generations = {}
            generation_changed = False
            with self.repo._authority_read_fences(stores):
                for store in stores:
                    roots = tuple(store.authoritative_root_definitions())
                    stats.store_scan_count += 1
                    source_key = self._containment_source_key(store)
                    status = store.query_index_status()
                    previous = before[source_key]
                    if (
                            previous.state == "ready"
                            and status.state == "ready"
                            and (
                                previous.backend != status.backend
                                or previous.generation != status.generation
                            )
                    ):
                        generation_changed = True
                        break
                    generation = status.generation
                    if generation is not None:
                        generations[source_key] = generation
                    source_plans.append(SourceQueryPlan(
                        source_key=source_key,
                        backend=status.backend,
                        generation=generation,
                        candidate_count=len(roots),
                        refresh_action="authority-root-residual",
                    ))
                    entries.extend((root, store) for root in roots)
            if generation_changed:
                continue
            prior_action = stats.refresh_action
            stats.refresh_action = (
                "authority-root-residual"
                if prior_action == "none"
                else f"{prior_action}+authority-root-residual"
            )
            stats.universe_size = len(entries)
            stats.generation_vector = generations or None
            stats.source_plans = tuple(source_plans)
            return tuple(entries)
        raise QueryIndexError(
            "Derived index generation changed repeatedly during authoritative containment capture."
        )

    def _containment_matcher(self, stats: QueryStats):
        """Build a typed residual predicate while preserving verify budgets."""

        exact_target = self.containment_target
        if is_exact_reference_target(exact_target):
            return lambda value: type(value) is type(exact_target) and value == exact_target

        verified: set[ConcreteDefinition] = set()

        def matches(value) -> bool:
            if not isinstance(value, ConcreteDefinition):
                return False
            if value not in verified:
                verified.add(value)
                stats.verified_count += 1
                stats.python_verifications += 1
                if self.max_verify_limit is not None and stats.verified_count > self.max_verify_limit:
                    raise QueryVerifyBudgetExceeded(
                        f"Query exceeded max_verify budget {self.max_verify_limit}: "
                        f"verified {stats.verified_count} CDefs."
                    )
            return _query_match(
                self.selector, value,
                strict=self.strict_policy,
                class_match=self.class_match_policy,
            )

        return matches

    def _execute_authoritative_containment_definitions(self):
        """Project qualified CDef terminals from a fenced authoritative cut."""

        stats = QueryStats()
        entries = self._capture_authoritative_containment_roots(stats)
        matches = self._containment_matcher(stats)
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        for value in iter_containment_targets_matching(
                (root for root, _ in entries), matches,
                edges=self.containment_edges, contains_ref=self.contains_ref,
        ):
            if isinstance(value, ConcreteDefinition):
                merged.setdefault(value, value)
        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        return out, stats

    def _execute_authoritative_containment_owners(self):
        """Project qualified stored roots without enumerating raw paths."""

        stats = QueryStats()
        entries = self._capture_authoritative_containment_roots(stats)
        matches = self._containment_matcher(stats)
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        for root, store in entries:
            for owner in iter_containment_owners_matching(
                    (root,), matches, edges=self.containment_edges,
                    contains_ref=self.contains_ref,
            ):
                canonical = merged.setdefault(owner, owner)
                replicas.setdefault(canonical, []).append(store)
        out = tuple(sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef))))
        stats.result_count = len(out)
        return out, stats, {
            owner: tuple(dict.fromkeys(replicas.get(owner, ()))) for owner in out
        }

    def _execute_authoritative_containment_occurrences(self):
        """Collect canonically ordered root-local witnesses under one global cap."""

        from .model import containment_witness_key

        stats = QueryStats()
        entries = self._capture_authoritative_containment_roots(stats)
        matches = self._containment_matcher(stats)
        limit = None if self.projection in {"object_refs", "state_refs"} else self.occurrence_limit
        if limit == 0:
            stats.result_count = 0
            return (), stats, {}
        occurrences = []
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        seen = set()
        grouped_entries: dict[str, list[tuple[ConcreteDefinition, list[Any]]]] = {}
        for root, store in entries:
            groups = grouped_entries.setdefault(root.graph_hash(), [])
            for representative, stores in groups:
                if representative.graph_equal(root):
                    stores.append(store)
                    break
            else:
                groups.append((root, [store]))
        root_groups = (
            group for groups in grouped_entries.values() for group in groups
        )
        def root_occurrences(root, stores):
            for occurrence in iter_containment_occurrences_matching(
                    (root,), matches, edges=self.containment_edges,
                    contains_ref=self.contains_ref,
            ):
                yield occurrence, stores

        ordered = merge(
            *(root_occurrences(root, stores) for root, stores in root_groups),
            key=lambda item: containment_witness_key(item[0]),
        )
        for occurrence, stores in ordered:
            key = containment_witness_key(occurrence)
            replicas.setdefault(occurrence.owner, []).extend(stores)
            if key in seen:
                continue
            seen.add(key)
            occurrences.append(occurrence)
            if limit is not None and len(occurrences) >= limit:
                break
        stats.result_count = len(occurrences)
        return tuple(occurrences), stats, {
            owner: tuple(dict.fromkeys(stores)) for owner, stores in replicas.items()
        }

    def _execute_nested_definition_matches(self):
        stats = QueryStats()
        catalog = self.repo._query_catalog
        captured = self._capture_nested_candidates(catalog, stats)
        matches, match_ids = self._verify_cdefs_by_id(captured.cdefs_by_id, stats=stats)
        return matches, match_ids, stats, captured.generation

    def _capture_nested_candidates(self, catalog, stats: QueryStats) -> CapturedNestedCandidates:
        catalog.refresh(self.refresh_policy, stats=stats)
        with catalog.read_view(include_cached=False) as snapshot:
            selector_graph = compile_selector_graph(self.selector, class_match=self.class_match_policy)
            if selector_graph is not None:
                candidate_ids = graph_candidate_ids(snapshot, selector_graph, None, stats=stats)
                candidate_ids = snapshot.filter_nested_ids(candidate_ids)
                stats.candidate_count = len(candidate_ids)
            else:
                domain = NestedDomain(snapshot)
                candidate_ids = domain.all_ids()
                stats.candidate_count = len(candidate_ids)
                stats.universe_size = None
            cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            generation = snapshot.generation
        return CapturedNestedCandidates(generation=generation, cdefs_by_id=cdefs_by_id, stats=stats)

    def _capture_occurrence_traversal(
            self,
            catalog,
            ids: set[DefinitionId] | frozenset[DefinitionId],
            generation: int):
        with catalog.read_view(include_cached=False) as snapshot:
            if snapshot.generation != generation:
                raise _QueryGenerationChanged
            return snapshot.occurrence_snapshot_for_nested_ids(set(ids))

    def _project_owners(
            self,
            catalog,
            ids: set[DefinitionId] | frozenset[DefinitionId],
            generation: int):
        with catalog.read_view(include_cached=False) as snapshot:
            if snapshot.generation != generation:
                raise _QueryGenerationChanged
            return snapshot.project_owners(set(ids))

    def _verify_cdefs_by_id(
            self,
            cdefs_by_id: dict[DefinitionId, ConcreteDefinition],
            *,
            stats: QueryStats) -> tuple[tuple[ConcreteDefinition, ...], set[DefinitionId]]:
        matches = self._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
        match_set = set(matches)
        match_ids = {did for did, cdef in cdefs_by_id.items() if cdef in match_set}
        return matches, match_ids

    def _verify_cdefs(
            self,
            cdefs: tuple[ConcreteDefinition, ...],
            *,
            stats: QueryStats,
            template_budget=None) -> tuple[ConcreteDefinition, ...]:
        if self.selector is None:
            stats.verified_count += len(cdefs)
            stats.python_verifications += len(cdefs)
            if self.max_verify_limit is not None and stats.verified_count > self.max_verify_limit:
                raise QueryVerifyBudgetExceeded(
                    f"Query exceeded max_verify budget {self.max_verify_limit}: verified {stats.verified_count} CDefs."
                )
            return tuple(sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef))))

        out: list[ConcreteDefinition] = []
        for cdef in cdefs:
            stats.verified_count += 1
            stats.python_verifications += 1
            if self.max_verify_limit is not None and stats.verified_count > self.max_verify_limit:
                raise QueryVerifyBudgetExceeded(
                    f"Query exceeded max_verify budget {self.max_verify_limit}: verified {stats.verified_count} CDefs."
                )
            if not _structural_match(
                    self.selector,
                    cdef,
                    strict=self.strict_policy,
                    class_match=self.class_match_policy):
                continue
            if self.template_selector is not None:
                if template_budget is None:
                    matched = self.template_selector.matches(cdef)
                else:
                    matched = self.template_selector._matches(cdef, template_budget)
                if not matched:
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
    if isinstance(source, Selector):
        return source.root
    from ..template_selector import TemplateSelector
    if isinstance(source, TemplateSelector):
        return source.prefilter.root
    if isinstance(source, Definition):
        return deepcopy(source)
    raise TypeError(f"Query source must be Selector, Definition, ConcreteDefinition, Object, or None, not {type(source).__name__}.")


def _resolve_query_state_selectors(source, repo):
    """Resolve every soft StateSelectorRef before a DefinitionQuery exists.

    Query selectors may remain partial and contain ``Par`` values, so they cannot
    use Definition-to-CDef concretization as a resolver.  This graph-preserving
    rewrite only replaces soft selector leaves and retains all selector syntax.
    """

    from ..cdef_graph import EdgeKind
    from ..definition import Definition, SKIP_ARGS
    from ..links import DefLink
    from ..reference_values import StateSelectorRef

    memo = {}
    active: set[int] = set()

    def visit(value):
        if isinstance(value, StateSelectorRef):
            key = id(value)
            if key not in memo:
                resolver = getattr(repo, "resolve_state_selector", None)
                if not callable(resolver):
                    raise TypeError("StateSelectorRef query values require a managing Repo.")
                resolved = resolver(value)
                if resolved.object != value.object:
                    raise ValueError("StateSelectorRef query resolution returned a StateRef outside its ObjectRef scope.")
                memo[key] = resolved
            return memo[key]
        if not isinstance(value, (DefLink, Definition, dict, FrozenDict, list,
                                  FrozenList, tuple, FrozenTuple, set,
                                  FrozenSet)):
            return value
        key = id(value)
        if key in memo:
            return memo[key]
        if key in active:
            raise ValueError("Cycle while resolving query state selectors.")
        active.add(key)
        try:
            if isinstance(value, DefLink):
                target = visit(value.target)
                result = target if value.kind is EdgeKind.MATERIALIZE else DefLink.finalized(value.kind, target)
            elif isinstance(value, Definition):
                args = (SKIP_ARGS,) if value.args is None else tuple(visit(item) for item in value.args)
                kwargs = {name: visit(item) for name, item in value.kwargs.items()}
                result = (
                    Definition(*args, **kwargs)
                    if value.cls is None
                    else Definition(value.cls, *args, **kwargs)
                )
            elif isinstance(value, (dict, FrozenDict)):
                result = type(value)({name: visit(item) for name, item in value.items()})
            elif isinstance(value, (list, FrozenList, tuple, FrozenTuple)):
                result = type(value)(visit(item) for item in value)
            else:
                result = type(value)(visit(item) for item in value)
                if len(result) != len(value):
                    raise ValueError("Resolving query state selectors collapsed set members.")
            memo[key] = result
            return result
        finally:
            active.remove(key)

    return visit(source)


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
        return _query_match_factory(selector, target, strict=strict, class_match=class_match)

    if isinstance(selector, DefLink):
        if selector.kind is EdgeKind.MATERIALIZE:
            target_value = target.target if isinstance(target, DefLink) and target.kind is EdgeKind.MATERIALIZE else target
            return _query_match(selector.target, target_value, strict=strict, class_match=class_match)
        if selector.kind is EdgeKind.REF:
            if not isinstance(target, DefLink) or target.kind is not EdgeKind.REF:
                return False
            return _query_match(selector.target, target.target, strict=strict, class_match=class_match)
        return False

    if isinstance(selector, (QuotedDef, SelectorSpec)):
        if not isinstance(target, (QuotedDef, SelectorSpec, Selector, Definition)):
            return False
        sel_value = selector.value if isinstance(selector, QuotedDef) else selector.selector
        tgt_value = target.value if isinstance(target, QuotedDef) else target.selector if isinstance(target, SelectorSpec) else target
        if isinstance(sel_value, Selector):
            sel_value = sel_value.root
        if isinstance(tgt_value, Selector):
            tgt_value = tgt_value.root
        return _query_match(sel_value, tgt_value, strict=strict, class_match=class_match)

    if isinstance(selector, Match):
        return selector.matches(target, present=True)

    if isinstance(selector, Definition):
        if not isinstance(target, (Definition, ConcreteDefinition)):
            return False
        if selector.cls is not None:
            if not _query_match_class(selector.cls, target.cls, strict=strict, class_match=class_match):
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
                if not _query_match(child, target.parameters[name], strict=strict, class_match=class_match):
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
                if not _query_match(child, target_parameters[name], strict=strict, class_match=class_match):
                    return False
            return True
        if selector.args is not None:
            if target.args is None:
                return False
            if not _query_match(selector.args, target.args, strict=strict, class_match=class_match):
                return False
        for key, child in selector.kwargs.items():
            if key not in target.kwargs:
                if isinstance(child, Match) and child.matches(None, present=False):
                    continue
                return False
            if not _query_match(child, target.kwargs[key], strict=strict, class_match=class_match):
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


def _query_match_factory(selector, target, *, strict: bool, class_match: ClassMatchPolicy) -> bool:
    """Verify exact or Match-bearing partial FactorySpec call patterns."""

    from ..factory import FactorySpec, _contains_match

    if not isinstance(target, FactorySpec):
        return False
    if not _query_match(selector.target, target.target, strict=strict, class_match=class_match):
        return False
    if len(selector.args) != len(target.args):
        return False
    if not all(_query_match(left, right, strict=strict, class_match=class_match) for left, right in zip(selector.args, target.args)):
        return False

    is_pattern = _contains_match(selector)
    if not is_pattern and tuple(selector.kwargs) != tuple(target.kwargs):
        return False
    return all(
        key in target.kwargs and _query_match(value, target.kwargs[key], strict=strict, class_match=class_match)
        for key, value in selector.kwargs.items()
    )


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
            edge.segment for edge in iter_value_edges(value)
            if edge.value is child
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
            source_path: DefinitionPath | None) -> None:
        paths[projected_path] = (source_path, source_value)
        projected_edges = iter_value_edges(projected_value)
        if not projected_edges:
            return
        key = (id(projected_value), id(source_value))
        if key in active:
            raise QueryPathError("Cycle while recording semantic query projection origins.")
        active.add(key)
        try:
            if _is_prepared_selector_definition(projected_value):
                if not isinstance(source_value, (Definition, ConcreteDefinition)):
                    raise QueryPathError(
                        "Semantic projection has no definition source for prepared parameters."
                    )
                source_parameters = _semantic_parameters(source_value)
                for edge in projected_edges:
                    if not isinstance(edge.segment, Kwarg) or edge.segment.name not in source_parameters:
                        raise QueryPathError(
                            "Semantic projection lost a prepared parameter source."
                        )
                    relative_source_path = source_parameter_path(
                        source_value, edge.segment.name,
                    )
                    child_source_path = (
                        None if source_path is None or relative_source_path is None
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
                        None if source_path is None else source_path.join(relative_source_path),
                    )
                return

            source_edges = {edge.segment: edge for edge in iter_value_edges(source_value)}
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
