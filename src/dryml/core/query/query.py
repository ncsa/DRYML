from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import cmp_to_key
from heapq import heappop, heappush
from typing import Any
import warnings

from ..canonical import matching_container_family
from ..cdef_identity import cdef_node_key
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
    QueryIndexUnavailable,
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
    iter_containment_projection_occurrences_matching,
    iter_containment_targets_matching,
)
from .path import (
    DefinitionPath,
    DefinitionPathLike,
    Kwarg,
    Parameter,
    QueryPathError,
    get_subtree,
    iter_value_edges,
    normalize_path,
    replace_subtree,
)
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
_DEFAULT_GENERATOR_WITNESS_LIMIT = 65_536
_DEFAULT_MAX_WITNESSES = object()


@dataclass(slots=True)
class _GeneratorWitnessBudget:
    limit: int | None
    visited: int = 0

    def consume(self) -> None:
        self.visited += 1
        if self.limit is not None and self.visited > self.limit:
            raise ParameterizationLimitError("generator query witness limit exceeded")


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
        generator_selector: Optional exact GeneratorSelector residual applied
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
    generator_selector: Any | None = None
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
        universe: ResultUniverse | None = None,
    ) -> "DefinitionQuery":
        """Create a finalized query from a soft structural source.

        Args:
            repo: Managing Repo required to resolve StateSelectorRef leaves.
            source: Optional Definition, CDef, ObjectRef, StateRef, Selector,
                GeneratorSelector, or Object source. Exact references are lazy
                containment targets and cannot select ordinary authority domains.
                GeneratorSelector support retains an exact residual beside its
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
        from ..generator import GeneratorSelector

        if isinstance(source, (ObjectRef, StateRef)):
            return cls(
                repo=repo,
                original=None,
                selector=None,
                domain=domain,
                universe=universe,
                containment_target=validate_containment_target(source),
            )

        if isinstance(source, GeneratorSelector):
            prefilter = _resolve_query_state_selectors(source.prefilter.root, repo)
            return cls(
                repo=repo,
                original=prefilter,
                selector=prefilter,
                domain=domain,
                universe=universe,
                class_match_policy="exact",
                generator_selector=source,
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
            QueryDomainError: If an exact GeneratorSelector residual is attached;
                ReferenceQuery cannot retain topology witness semantics, if this
                query has an exact reference containment target, or if nested
                containment has already been selected.

        Side Effects:
            None. This conversion does not query Store authority.
        """

        if self.generator_selector is not None:
            raise QueryDomainError(
                "GeneratorSelector queries cannot be converted to ReferenceQuery."
            )
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
        drop_class: bool = False,
    ) -> "DefinitionQuery":
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
            raise QueryPathError(
                "Cannot apply semantic categorical projection to an unconstrained query."
            )
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
            selector, self.selector, origins
        ).items():
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
        path: DefinitionPathLike = "$",
    ) -> "DefinitionQuery":
        self._reject_exact_reference_rewrite("exact")
        self._reject_residual_rewrite("exact")
        if self.selector is None:
            raise QueryPathError("Cannot apply exact() to an unconstrained query.")
        norm = normalize_path(path)
        if definition is None:
            if self.original is None:
                raise QueryPathError(
                    f"Cannot infer exact subtree at {norm!s}; query has no original source."
                )
            definition = self._original_value(norm)
        if isinstance(definition, Object):
            definition = definition.definition
        if not isinstance(definition, ConcreteDefinition):
            raise TypeError(
                f"Exact constraint at {norm!s} requires a ConcreteDefinition, got {type(definition).__name__}."
            )
        return replace(self, selector=replace_subtree(self.selector, norm, definition))

    def _original_value(self, path: DefinitionPath) -> Any:
        """Return the original authority retained for one selector occurrence."""

        original_values = dict(self._original_values)
        if path in original_values:
            return original_values[path]
        if self.original is None:
            raise QueryPathError(
                "Cannot resolve an original value without a source query."
            )
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
        return replace(
            self,
            class_match_policy=(
                "exact" if self.generator_selector is not None else policy
            ),
        )

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
        refresh: RefreshPolicy | None = None,
    ) -> "DefinitionQuery":
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
            if (
                context.target_kind != "definition"
                and self.generator_selector is not None
            ):
                raise QueryDomainError(
                    "Exact selector refinement is unavailable for fixed reference containment results."
                )
            if (
                    context.target_kind != "definition"
                    and self.universe.containment_carrier != "owner"
                    and not (
                    self.containment_target is None
                    or is_exact_reference_target(self.containment_target)
                )
            ):
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
            raise QueryDomainError(
                "in_store() is only valid for nested containment queries."
            )
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
        self._require_fixed_containment_carrier("target")
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
        self._require_fixed_containment_carrier("owner")
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
            raise ValueError(
                "max_witnesses limit must be a positive exact int or None."
            )
        return replace(self, max_witness_limit=limit)

    def _reject_residual_rewrite(self, operation: str) -> None:
        """Reject selector rewrites that cannot preserve an exact residual."""

        if self.generator_selector is not None:
            raise QueryDomainError(
                f"GeneratorSelector queries cannot apply {operation}()."
            )

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

    def _require_fixed_containment_carrier(self, carrier: str) -> None:
        """Reject projection changes that would widen a fixed containment result."""

        if (
                self.universe is not None
                and self.universe.containment is not None
                and self.universe.kind == "definitions"
                and self.universe.containment_carrier != carrier
        ):
            raise QueryDomainError(
                "Cannot change the projection of a fixed containment result universe."
            )

    def _require_reference_target_projection(
        self, projection: str, expected: str
    ) -> None:
        """Validate an exact-reference value terminal before unsupported execution."""

        if self.domain != "nested":
            raise QueryDomainError(
                f"{projection}() is only valid for nested containment queries."
            )
        if type(self.containment_target).__name__ != expected:
            actual = (
                type(self.containment_target).__name__
                if self.containment_target is not None
                else "CDef selector"
            )
            raise QueryDomainError(
                f"{projection}() requires an exact {expected} containment target, got {actual}."
            )

    @property
    def lowering_scan_policy(self) -> ScanPolicy:
        return ScanPolicy(self.scan_policy_mode, self.max_verify_limit)

    def execute(self):
        self._require_domain()
        if self.generator_selector is not None:
            return self._execute_generator_selector()
        if self.domain == "nested":
            if self.universe is not None and self.universe.kind == "definitions":
                return self._execute_fixed_containment_definition_result()
            if self.universe is None and self.projection == "definitions":
                if self._uses_authoritative_containment_residual():
                    cdefs, stats, _, evidence, private_evidence, private_replicas = (
                        self._execute_authoritative_containment_projection_evidence(
                            "target"
                        )
                    )
                    return DefinitionResultSet(
                        self.repo,
                        cdefs,
                        materializable=False,
                        domain="nested-definitions",
                        explanation=self._explanation(stats),
                        replicas={},
                        containment=self._fresh_containment_context(complete=True),
                        containment_witnesses=evidence,
                        witnesses=(item.target for item in private_evidence),
                        witness_complete=True,
                        containment_private_witnesses=private_evidence,
                        containment_private_replicas=private_replicas,
                        containment_carrier="target",
                    )
                cdefs, stats = self._execute_nested_definitions()
                explanation = self._explanation(stats)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=False,
                    domain="nested-definitions",
                    explanation=explanation,
                    replicas={},
                    containment=self._fresh_containment_context(complete=False),
                    containment_carrier="target",
                )
            if self.universe is None and self.projection == "owners":
                if self._uses_authoritative_containment_residual():
                    (
                        cdefs,
                        stats,
                        replicas,
                        evidence,
                        private_evidence,
                        private_replicas,
                    ) = self._execute_authoritative_containment_projection_evidence(
                        "owner"
                    )
                    return DefinitionResultSet(
                        self.repo,
                        cdefs,
                        materializable=True,
                        domain="owners",
                        explanation=self._explanation(stats),
                        replicas=replicas,
                        containment=self._fresh_containment_context(complete=True),
                        containment_witnesses=evidence,
                        witnesses=(item.owner for item in private_evidence),
                        witness_complete=True,
                        containment_private_witnesses=private_evidence,
                        containment_private_replicas=private_replicas,
                        containment_carrier="owner",
                    )
                cdefs, stats, replicas = self._execute_nested_owners()
                explanation = self._explanation(stats)
                return DefinitionResultSet(
                    self.repo,
                    cdefs,
                    materializable=True,
                    domain="owners",
                    explanation=explanation,
                    replicas=replicas,
                    containment=self._fresh_containment_context(complete=False),
                    containment_carrier="owner",
                )
            occs, stats, owner_replicas, private_witnesses, private_replicas = (
                self._execute_nested_occurrences()
            )
            explanation = self._explanation(stats)
            containment = None if self.universe is None else self.universe.containment
            containment_witnesses = (
                None
                if self.universe is None
                else (
                    occs
                    if self.universe.containment is not None
                    else self.universe.containment_witnesses
                )
            )
            if containment is None and self.universe is None:
                from .model import containment_target_kind

                target_kind = (
                    containment_target_kind(self.containment_target)
                    if self.containment_target is not None
                    else "definition"
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
                    and (
                        self.universe is not None
                        or self.projection not in {"object_refs", "state_refs"}
                    )
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
                    witnesses=private_witnesses or (),
                    witness_complete=private_witnesses is not None
                    and (containment is None or not containment.bounded),
                    containment_private_witnesses=private_witnesses or (),
                    containment_private_replicas=private_replicas,
                )
            else:
                raw = OccurrenceResultSet(
                    self.repo,
                    occs,
                    explanation=explanation,
                    owner_replicas=owner_replicas,
                    containment=containment,
                    containment_witnesses=containment_witnesses,
                    witnesses=private_witnesses or (),
                    witness_complete=private_witnesses is not None
                    and (containment is None or not containment.bounded),
                    containment_private_witnesses=private_witnesses or (),
                    containment_private_replicas=private_replicas,
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

        if (
            self.universe is None
            and self.domain == "stored"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            query_backed = getattr(
                self.repo._query_index, "query_backed_definition_result_set", None
            )
            if query_backed is not None:
                result_set = query_backed(self)
                if result_set is not None:
                    return result_set

        cdefs, stats, query_replicas = self._execute_definition_domain()
        explanation = stats.explanation(
            domain=self._domain_label(), refresh=self.refresh_policy
        )
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
        raise QueryDomainError(
            "Raw nested queries return occurrences; use .definitions().defs() or .owners().defs()."
        )

    def objects(self, **load_options) -> ObjectResultSet:
        result = self.execute()
        if isinstance(result, OccurrenceResultSet):
            raise QueryDomainError(
                "Raw nested occurrences cannot be materialized directly; use .owners().objects()."
            )
        return result.objects(**load_options)

    def count(self) -> int:
        return self._execute_count()

    def exists(self) -> bool:
        if self.generator_selector is not None:
            return self._execute_count() > 0
        if (
            self.universe is None
            and self.domain == "stored"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            count, _ = self.repo._query_index.count_definition_domain(
                self, stop_after=1
            )
            return count > 0
        if (
            self.universe is None
            and self.domain == "known"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            return bool(self._execute_federated_known_domain(stop_after=1)[0])
        if self.universe is None and self.domain == "nested":
            return bool(self._execute_terminal_items(stop_after=1))
        return self._execute_count() > 0

    def one(self):
        items = self._execute_terminal_items(stop_after=2)
        if len(items) != 1:
            label = (
                "occurrence"
                if self.domain == "nested" and self.projection is None
                else "result"
            )
            raise QueryCardinalityError(
                f"Expected exactly one {label}, found {len(items)}."
            )
        return items[0]

    def one_or_none(self):
        items = self._execute_terminal_items(stop_after=2)
        if len(items) > 1:
            label = (
                "occurrence"
                if self.domain == "nested" and self.projection is None
                else "result"
            )
            raise QueryCardinalityError(
                f"Expected zero or one {label}, found {len(items)}."
            )
        return items[0] if items else None

    def explain(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        return self._execute_explanation(analyze=analyze, sql=sql)

    def _execute_count(self) -> int:
        self._require_domain()
        if self.generator_selector is not None:
            return len(self._execute_generator_selector())
        if (
            self.universe is None
            and self.domain == "stored"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            count, _ = self.repo._query_index.count_definition_domain(self)
            return count
        if (
            self.universe is None
            and self.domain == "known"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            cdefs, _, _ = self._execute_federated_known_domain()
            return len(cdefs)
        if self.domain == "nested":
            if self.universe is not None and self.universe.kind == "definitions":
                return len(self._execute_fixed_containment_definition_result())
            if self.projection in {"object_refs", "state_refs"}:
                return len(self.execute())
            if self.universe is None and self.projection == "definitions":
                if (
                    not self._uses_authoritative_containment_residual()
                    and self.repo._query_index.can_execute_query_domain("nested")
                ):
                    cdefs, _ = self.repo._query_index.execute_nested_definitions(self)
                else:
                    cdefs, _ = self._execute_nested_definitions()
                return len(cdefs)
            if self.universe is None and self.projection == "owners":
                if (
                    not self._uses_authoritative_containment_residual()
                    and self.repo._query_index.can_execute_query_domain("nested")
                ):
                    cdefs, _, _ = self.repo._query_index.execute_nested_owners(self)
                else:
                    cdefs, _, _ = self._execute_nested_owners()
                return len(cdefs)
            occurrences, _, _, _, _ = self._execute_nested_occurrences()
            if callable(occurrences):
                return sum(1 for _ in occurrences())
            return len(occurrences)

        cdefs, _, _ = self._execute_definition_domain()
        return len(cdefs)

    def _execute_explanation(
        self, *, analyze: bool = False, sql: bool = False
    ) -> QueryExplanation:
        self._require_domain()
        if self.generator_selector is not None:
            if self._is_deterministic_empty_containment():
                return self._explanation(
                    QueryStats(
                        fast_path="deterministic-empty-containment",
                        result_count=0,
                    )
                )
            if not analyze:
                stats = QueryStats(
                    scan_required=self.universe is None,
                    scan_reason=(
                        "GeneratorSelector queries require complete witness scanning."
                        if self.universe is None
                        else None
                    ),
                )
                return self._explanation(stats)
            result = self._execute_generator_selector()
            return result.explanation or QueryStats(
                result_count=len(result)
            ).explanation(
                domain=self._domain_label(),
                refresh=self.refresh_policy,
            )
        if analyze:
            result = self.execute()
            explanation = result.explanation
            if explanation is None:
                return QueryStats(result_count=len(result)).explanation(
                    domain=self._domain_label(), refresh=self.refresh_policy
                )
            return explanation
        if self.domain == "nested":
            if self.universe is not None and self.universe.kind == "definitions":
                result = self._execute_fixed_containment_definition_result()
                return result.explanation or self._explanation(
                    QueryStats(result_count=len(result))
                )
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
                _, stats, _, _, _ = self._execute_nested_occurrences()
            return self._explanation(stats)

        if (
            self.universe is None
            and self.domain == "stored"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            stats = self.repo._query_index.explain_definition_domain(self, sql=sql)
            return stats.explanation(
                domain=self._domain_label(), refresh=self.refresh_policy
            )

        _, stats, _ = self._execute_definition_domain()
        return stats.explanation(
            domain=self._domain_label(), refresh=self.refresh_policy
        )

    def _require_domain(self) -> None:
        if self.domain is None:
            raise QueryDomainError(
                "Select a query domain with stored(), cached(), known(), or nested() before executing."
            )

    def _domain_label(self) -> str:
        if self.domain == "nested" and self.projection is not None:
            return f"nested-{self.projection}"
        return self.domain or "unset"

    def _explanation(self, stats: QueryStats) -> QueryExplanation:
        """Attach immutable containment policy to an execution explanation."""

        explanation = stats.explanation(
            domain=self._domain_label(),
            refresh=self.refresh_policy,
        )
        if self.domain != "nested":
            return explanation
        from .model import containment_target_kind

        target_kind = (
            self.universe.containment.target_kind
            if self.universe is not None and self.universe.containment is not None
            else (
                containment_target_kind(self.containment_target)
                if self.containment_target is not None
                else "definition"
            )
        )
        return replace(
            explanation,
            containment_target_kind=target_kind,
            containment_edges=self.containment_edges,
            containment_contains_ref=self.contains_ref,
            containment_source_scope=(
                self.universe.containment.source_scope
                if self.universe is not None and self.universe.containment is not None
                else self._containment_source_scope()
            ),
        )

    def _execute_definition_domain(self):
        stats = QueryStats()
        if self.universe is not None:
            if self.universe.kind != "definitions":
                raise QueryDomainError(
                    "A definition terminal cannot execute over an occurrence universe."
                )
            stats.universe_size = len(self.universe.definitions)
            matches = self._verify_cdefs(tuple(self.universe.definitions), stats=stats)
            stats.result_count = len(matches)
            replicas = {}
            if self.universe.replicas is not None:
                replicas = {
                    cdef: self.universe.replicas.get(cdef, ()) for cdef in matches
                }
            return matches, stats, replicas

        if self.domain == "stored" and self.repo._query_index.can_execute_query_domain(
            "stored"
        ):
            return self.repo._query_index.execute_definition_domain(self)
        if self.domain == "known" and self.repo._query_index.can_execute_query_domain(
            "stored"
        ):
            return self._execute_federated_known_domain()

        catalog = self.repo._query_catalog
        exact_root = (
            self.selector if isinstance(self.selector, ConcreteDefinition) else None
        )
        if self.refresh_policy is True:
            catalog.refresh(True, stats=stats)
        elif (
            exact_root is not None
            and self.domain in {"stored", "known"}
            and self.refresh_policy is not False
        ):
            catalog.ensure_exact_stored(exact_root, stats=stats)
        elif self.domain in {"stored", "known"}:
            catalog.refresh(self.refresh_policy, stats=stats)

        live_domain = self._definition_domain(catalog)
        live_domain.prepare(stats=stats)
        with catalog.read_view(
            include_cached=self.domain in {"cached", "known"}
        ) as snapshot:
            domain = live_domain.with_catalog(snapshot)
            if exact_root is not None and self.domain in {"stored", "cached", "known"}:
                candidate_ids = domain.filter(snapshot.exact_ids(exact_root))
                stats.universe_size = len(candidate_ids)
                stats.candidate_count = len(candidate_ids)
            else:
                stats.universe_size = domain.estimate_size()
                selector_graph = compile_selector_graph(
                    self.selector, class_match=self.class_match_policy
                )
                if selector_graph is not None:
                    candidate_ids = graph_candidate_ids(
                        snapshot, selector_graph, domain, stats=stats
                    )
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

    def _execute_generator_selector(self):
        """Verify exact support against complete graph-distinct domain witnesses.

        Query indexes intentionally collapse structural CDefs, so they cannot be
        used as terminal authority for topology-sensitive template support.  This
        path scans authoritative roots (or retained immutable result witnesses)
        before restoring the public structural-deduplication result semantics.
        """

        if (
                self.scan_policy_mode == "forbid"
                and self.universe is None
                and not self._is_deterministic_empty_containment()
        ):
            raise QueryDomainError(
                "GeneratorSelector queries require complete witness scanning."
            )
        if (
                self.domain == "nested"
                and self.universe is not None
                and self.universe.kind == "definitions"
                and self.universe.containment is not None
        ):
            return self._execute_fixed_containment_definition_result()
        if self.domain == "nested" and not (
            self.universe is not None and self.universe.kind == "definitions"
        ):
            return self._execute_generator_selector_nested()

        stats = QueryStats(refresh_action="generator-witness-scan")
        witness_budget = self._generator_witness_budget()
        witnesses = self._generator_definition_witnesses(stats, witness_budget)
        matches, replicas, verified_witnesses = self._verify_generator_witnesses(
            witnesses, stats
        )
        stats.result_count = len(matches)
        explanation = stats.explanation(
            domain=self._domain_label(), refresh=self.refresh_policy
        )
        materializable = (
            self.universe.materializable if self.universe is not None else True
        )
        domain = (
            self.universe.domain
            if self.universe is not None
            else self.domain or "stored"
        )
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

    def _generator_witness_budget(self) -> _GeneratorWitnessBudget:
        limit = (
            _DEFAULT_GENERATOR_WITNESS_LIMIT
            if self.max_witness_limit is _DEFAULT_MAX_WITNESSES
            else self.max_witness_limit
        )
        return _GeneratorWitnessBudget(limit)

    def _generator_definition_witnesses(
        self, stats: QueryStats, witness_budget: _GeneratorWitnessBudget
    ):
        """Stream complete roots after a conservative structural index prefilter."""

        if self.universe is not None:
            if self.universe.kind != "definitions":
                raise QueryDomainError(
                    "A definition terminal cannot execute over an occurrence universe."
                )
            if not self.universe.witness_complete:
                raise QueryDomainError(
                    "GeneratorSelector refinement requires complete immutable witness evidence."
                )
            replicas = self.universe.replicas or {}

            def universe_witnesses():
                for cdef in self.universe.witnesses:
                    witness_budget.consume()
                    yield cdef, tuple(replicas.get(cdef, ()))

            return universe_witnesses()

        indexed_candidates = None
        if self.domain in {
            "stored",
            "known",
        } and self.repo._query_index.can_execute_query_domain("stored"):
            prefilter_query = replace(
                self,
                domain="stored",
                generator_selector=None,
                max_witness_limit=_DEFAULT_MAX_WITNESSES,
            )
            candidates, index_stats, _ = (
                self.repo._query_index.execute_definition_domain(prefilter_query)
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
                    iterate = getattr(
                        store, "iter_authoritative_root_definitions", None
                    )
                    if not callable(iterate):
                        raise QueryDomainError(
                            "GeneratorSelector stored queries require authoritative root enumeration."
                        )
                    for cdef in self._iter_generator_store_roots(store, iterate):
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
            raise QueryDomainError(
                f"Unsupported GeneratorSelector definition domain {self.domain!r}."
            )
        return roots()

    @staticmethod
    def _iter_generator_store_roots(store, iterate):
        """Validate one Store's streamed exact GeneratorSelector witness authority."""

        try:
            for cdef in iterate():
                if not isinstance(cdef, ConcreteDefinition):
                    raise QueryDomainError(
                        "GeneratorSelector Store enumeration yielded a non-CDef root."
                    )
                yield cdef
        except QueryDomainError:
            raise
        except Exception as error:
            raise QueryDomainError(
                f"GeneratorSelector Store enumeration failed for {store!r}."
            ) from error

    def _verify_generator_witnesses(self, witnesses, stats: QueryStats):
        """Verify each witness before structural result representative selection."""

        from ..generator import _AssignmentBudget

        assignment_budget = _AssignmentBudget(self.generator_selector._max_assignments)
        merged: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        verified_witnesses = []
        for cdef, stores in witnesses:
            stats.candidate_count += 1
            matched = self._verify_cdefs(
                (cdef,), stats=stats, generator_budget=assignment_budget
            )
            if not matched:
                continue
            match = matched[0]
            canonical = merged.setdefault(match, match)
            replicas.setdefault(canonical, []).extend(stores)
            verified_witnesses.append(match)
        ordered = tuple(
            sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
        )
        return (
            ordered,
            {cdef: tuple(dict.fromkeys(replicas.get(cdef, ()))) for cdef in ordered},
            tuple(verified_witnesses),
        )

    def _execute_generator_selector_nested(self):
        """Apply exact support over shared root-local containment witnesses.

        Generator support visits every selected candidate before a raw-result cap
        is applied, preserving the witness and assignment budgets. The shared
        containment walker supplies literal edge-policy and hop evidence without
        collapsing private graph topology.
        """

        stats = QueryStats(refresh_action="generator-witness-scan")
        witness_budget = self._generator_witness_budget()
        from ..generator import _AssignmentBudget

        assignment_budget = _AssignmentBudget(self.generator_selector._max_assignments)

        def matches(value) -> bool:
            if not isinstance(value, ConcreteDefinition):
                return False
            witness_budget.consume()
            stats.candidate_count += 1
            return bool(
                self._verify_cdefs(
                    (value,),
                    stats=stats,
                    generator_budget=assignment_budget,
                )
            )

        matched_candidates: dict[object, bool] = {}

        def matches_once(value: ConcreteDefinition) -> bool:
            """Verify each private CDef node once while retaining all its paths."""

            node_key = cdef_node_key(value)
            if node_key not in matched_candidates:
                matched_candidates[node_key] = matches(value)
            return matched_candidates[node_key]

        if self._is_deterministic_empty_containment():
            candidates = lambda: ()
            context = ContainmentContext(
                target_kind="definition",
                edges=self.containment_edges,
                contains_ref=self.contains_ref,
                source_scope=self._containment_source_scope(),
                complete=True,
                bounded=self.projection is None and self.occurrence_limit is not None,
            )
        elif self.universe is not None:
            context = self.universe.containment
            complete = self.universe.witness_complete
            if self.universe.kind != "occurrences" or not complete:
                raise QueryDomainError(
                    "GeneratorSelector refinement requires complete immutable witness evidence."
                )
            retained_private_replicas = self.universe.containment_private_replicas
            evidence = (
                self.universe.containment_private_witnesses
                if context is not None
                else self.universe.witnesses
            )

            def candidates():
                if self.projection == "owners":
                    if retained_private_replicas is None:
                        raise QueryDomainError(
                            "GeneratorSelector refinement requires complete retained witness evidence."
                        )
                    for occurrence in evidence:
                        if matches_once(occurrence.owner):
                            yield occurrence, tuple(
                                retained_private_replicas.get(
                                    cdef_node_key(occurrence.owner), ()
                                )
                            )
                    return
                for occurrence in evidence:
                    if matches_once(occurrence.definition):
                        yield occurrence, (
                            ()
                            if retained_private_replicas is None
                            else tuple(
                                retained_private_replicas.get(
                                    cdef_node_key(occurrence.owner), ()
                                )
                            )
                        )

        else:
            reason = "GeneratorSelector queries require complete witness scanning."
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
                        iterate = getattr(
                            store, "iter_authoritative_root_definitions", None
                        )
                        if not callable(iterate):
                            raise QueryDomainError(
                                "GeneratorSelector nested queries require authoritative root enumeration."
                            )
                        stats.store_scan_count += 1
                        for root in self._iter_generator_store_roots(store, iterate):
                            yield from (
                                (occurrence, (store,))
                                for occurrence in iter_containment_occurrences_matching(
                                    (root,),
                                    matches,
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
                bounded=self.projection is None and self.occurrence_limit is not None,
            )

        private_occurrences = []
        private_replicas: dict[object, list[Any]] = {}
        for occurrence, stores in candidates():
            private_occurrences.append(occurrence)
            private_replicas.setdefault(cdef_node_key(occurrence.owner), []).extend(
                stores
            )
        merged: dict[tuple[Any, ...], Any] = {}
        for occurrence in private_occurrences:
            merged.setdefault(containment_witness_key(occurrence), occurrence)
        occurrences = tuple(merged[key] for key in sorted(merged))
        visible = occurrences
        if self.projection is None and self.occurrence_limit is not None:
            visible = visible[: self.occurrence_limit]
        visible_keys = {containment_witness_key(occurrence) for occurrence in visible}
        private_occurrences = tuple(
            occurrence
            for occurrence in private_occurrences
            if containment_witness_key(occurrence) in visible_keys
        )
        private_owner_keys = {
            cdef_node_key(occurrence.owner) for occurrence in private_occurrences
        }
        private_replicas = {
            key: tuple(dict.fromkeys(stores))
            for key, stores in private_replicas.items()
            if key in private_owner_keys
        }
        owner_replicas: dict[ConcreteDefinition, list[Any]] = {}
        for occurrence in private_occurrences:
            owner_replicas.setdefault(occurrence.owner, []).extend(
                private_replicas.get(cdef_node_key(occurrence.owner), ())
            )
        owner_replicas = {
            owner: tuple(dict.fromkeys(stores))
            for owner, stores in owner_replicas.items()
        }
        if self.projection == "definitions":
            definitions = tuple(occ.definition for occ in occurrences)
            stats.result_count = len(dict.fromkeys(definitions))
            return DefinitionResultSet(
                self.repo,
                definitions,
                materializable=False,
                domain="nested-definitions",
                explanation=self._explanation(stats),
                replicas={},
                witnesses=(item.target for item in private_occurrences),
                witness_complete=True,
                containment=context,
                containment_witnesses=occurrences,
                containment_private_witnesses=private_occurrences,
            )
        if self.projection == "owners":
            owners = tuple(occ.owner for occ in occurrences)
            stats.result_count = len(dict.fromkeys(owners))
            return DefinitionResultSet(
                self.repo,
                owners,
                materializable=True,
                domain="owners",
                explanation=self._explanation(stats),
                replicas=owner_replicas,
                witnesses=(item.owner for item in private_occurrences),
                witness_complete=True,
                containment=context,
                containment_witnesses=occurrences,
                containment_private_witnesses=private_occurrences,
                containment_private_replicas=private_replicas,
                containment_carrier="owner",
            )
        stats.result_count = len(visible)
        return OccurrenceResultSet(
            self.repo,
            visible,
            explanation=self._explanation(stats),
            owner_replicas=owner_replicas,
            witnesses=private_occurrences,
            witness_complete=not context.bounded,
            containment=context,
            containment_witnesses=visible,
            containment_private_witnesses=private_occurrences,
            containment_private_replicas=private_replicas,
        )

    def _execute_federated_known_domain(self, *, stop_after: int | None = None):
        from .federation import CACHE_SOURCE_KEY

        stored_query = replace(self, domain="stored")
        stored_cdefs, stored_stats, stored_replicas = (
            self.repo._query_index.execute_definition_domain(
                stored_query, stop_after=stop_after
            )
        )

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
        cached_cdefs, cached_stats, cached_replicas = (
            cached_query._execute_definition_domain()
        )
        cache_generation = self.repo._query_catalog.current_generation()

        merged = {cdef: cdef for cdef in stored_cdefs}
        for cdef in cached_cdefs:
            merged.setdefault(cdef, cdef)
            if stop_after is not None and len(merged) >= stop_after:
                break

        out = tuple(
            sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
        )
        replicas = {}
        for cdef in out:
            if cdef in stored_replicas:
                replicas[cdef] = stored_replicas[cdef]
            else:
                replicas[cdef] = cached_replicas.get(cdef, ())

        stats = QueryStats(refresh_action="federated-known")
        stats.store_scan_count = (
            stored_stats.store_scan_count + cached_stats.store_scan_count
        )
        stats.candidate_count = (
            stored_stats.candidate_count + cached_stats.candidate_count
        )
        stats.verified_count = stored_stats.verified_count + cached_stats.verified_count
        stats.result_count = len(out)
        stats.universe_size = None
        stats.generation_vector = dict(stored_stats.generation_vector or {})
        stats.generation_vector[CACHE_SOURCE_KEY] = cache_generation
        stats.source_plans = (
            *stored_stats.source_plans,
            SourceQueryPlan(
            source_key=CACHE_SOURCE_KEY,
            backend="memory-cache",
            generation=cache_generation,
            candidate_count=cached_stats.candidate_count,
            verified_count=cached_stats.verified_count,
            result_count=cached_stats.result_count,
            refresh_action=cached_stats.refresh_action,
            ),
        )
        return out, stats, replicas

    def _execute_terminal_items(self, *, stop_after: int):
        self._require_domain()
        if self.generator_selector is not None:
            return tuple(
                item
                for _, item in zip(
                    range(stop_after), self._execute_generator_selector()
                )
            )
        if self.domain == "nested":
            if (
                    self.universe is None
                    and self.projection == "definitions"
                    and not self._uses_authoritative_containment_residual()
                    and self.repo._query_index.can_execute_query_domain("nested")
            ):
                return self.repo._query_index.execute_nested_definitions(
                    self, stop_after=stop_after
                )[0]
            if (
                    self.universe is None
                    and self.projection == "owners"
                    and not self._uses_authoritative_containment_residual()
                    and self.repo._query_index.can_execute_query_domain("nested")
            ):
                return self.repo._query_index.execute_nested_owners(
                    self, stop_after=stop_after
                )[0]
            if self.universe is None and self.projection is None:
                limit = (
                    stop_after
                    if self.occurrence_limit is None
                    else min(self.occurrence_limit, stop_after)
                )
                occurrences, _, _, _, _ = replace(
                    self,
                    occurrence_limit=limit,
                )._execute_nested_occurrences()
                if callable(occurrences):
                    return tuple(
                        item for _, item in zip(range(stop_after), occurrences())
                    )
                return tuple(occurrences[:stop_after])
            result = self.execute()
            return tuple(item for _, item in zip(range(stop_after), result))

        if (
            self.universe is None
            and self.domain == "stored"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
            return self.repo._query_index.execute_definition_domain(
                self, stop_after=stop_after
            )[0]
        if (
            self.universe is None
            and self.domain == "known"
            and self.repo._query_index.can_execute_query_domain("stored")
        ):
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
                raise QueryDomainError(
                    "A nested query cannot execute over a definition universe."
                )
            evidence = self.universe.containment_witnesses
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
                            item
                            for item in evidence
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
                        item
                        for item in evidence
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
                visible_keys = {containment_witness_key(item) for item in out}
                private_evidence = tuple(
                    item
                    for item in self.universe.containment_private_witnesses
                    if containment_witness_key(item) in visible_keys
                )
                private_replicas = self.universe.containment_private_replicas
                if private_replicas is not None:
                    private_owner_keys = {
                        cdef_node_key(item.owner) for item in private_evidence
                    }
                    private_replicas = {
                        key: stores
                        for key, stores in private_replicas.items()
                        if key in private_owner_keys
                    }
                    replicas = {}
                    for item in private_evidence:
                        replicas.setdefault(item.owner, []).extend(
                            private_replicas.get(cdef_node_key(item.owner), ())
                        )
                    replicas = {
                        owner: tuple(dict.fromkeys(stores))
                        for owner, stores in replicas.items()
                    }
                else:
                    replicas = {
                        item.owner: self.universe.replicas.get(item.owner, ())
                        for item in out
                    }
                stats.result_count = len(out)
                return out, stats, replicas, private_evidence, private_replicas
            verified_nested = self._verify_cdefs(
                tuple({occ.definition for occ in self.universe.occurrences}),
                stats=stats,
            )
            verified = set(verified_nested)
            out = tuple(
                occ for occ in self.universe.occurrences if occ.definition in verified
            )
            if self.occurrence_limit is not None:
                out = out[:self.occurrence_limit]
            stats.result_count = len(out)
            return out, stats, self.universe.replicas, (), None

        if self._is_deterministic_empty_containment():
            stats = QueryStats(fast_path="deterministic-empty-containment")
            stats.result_count = 0
            return (), stats, {}, (), {}

        if self._uses_authoritative_containment_residual():
            return self._execute_authoritative_containment_occurrences()

        if self.repo._query_index.can_execute_query_domain("nested"):
            occurrences, stats, replicas = (
                self.repo._query_index.execute_nested_occurrences(self)
            )
            return occurrences, stats, replicas, None, None

        catalog = self.repo._query_catalog
        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            stats = QueryStats()
            captured = self._capture_nested_candidates(catalog, stats)
            _, match_ids = self._verify_cdefs_by_id(captured.cdefs_by_id, stats=stats)
            try:
                traversal = self._capture_occurrence_traversal(
                    catalog, match_ids, captured.generation
                )
                break
            except _QueryGenerationChanged:
                continue
        else:
            raise QueryIndexError(
                "Catalog generation changed repeatedly during nested occurrence query."
            )

        def occurrence_factory():
            return traversal.iter_occurrences(max_occurrences=self.occurrence_limit)

        return occurrence_factory, stats, traversal.owner_replicas, None, None

    def _execute_fixed_containment_definition_result(self) -> DefinitionResultSet:
        """Refine one projected fixed universe without reacquiring Store authority.

        CDef selectors refine the result's projected carrier.  Exact reference
        targets instead filter the retained owner/terminal ledger, because an
        owner result must retain its target evidence for later exact requery.
        """

        universe = self.universe
        if (
            universe is None
            or universe.kind != "definitions"
            or universe.containment is None
        ):
            raise QueryDomainError(
                "A nested definition result requires containment universe metadata."
            )
        context = universe.containment
        carrier = universe.containment_carrier
        if carrier not in {"target", "owner"}:
            raise QueryDomainError(
                "Fixed containment result has an unsupported carrier."
            )
        stats = QueryStats(universe_size=len(universe.definitions))
        evidence = universe.containment_witnesses
        private_evidence = universe.containment_private_witnesses
        private_replicas = universe.containment_private_replicas
        if self.generator_selector is not None and (
            not context.complete or not universe.witness_complete
        ):
            raise QueryDomainError(
                "GeneratorSelector refinement requires complete retained witness evidence."
            )
        if is_exact_reference_target(self.containment_target):
            if not context.complete:
                raise QueryDomainError(
                    "Exact containment refinement requires complete retained witness evidence."
                )
            evidence = tuple(
                item for item in evidence if item.target == self.containment_target
            )
            if carrier != "owner":
                raise QueryDomainError(
                    "Exact-reference containment cannot refine a definition-target result."
                )
            retained = tuple(dict.fromkeys(item.owner for item in evidence))
            private_evidence = tuple(
                item
                for item in private_evidence
                if item.target == self.containment_target
            )
        else:
            if self.generator_selector is None:
                retained = self._verify_cdefs(tuple(universe.definitions), stats=stats)
            else:
                from ..generator import _AssignmentBudget

                witness_budget = self._generator_witness_budget()
                assignment_budget = _AssignmentBudget(
                    self.generator_selector._max_assignments
                )
                retained_items = []
                seen_candidates = set()
                for candidate in universe.witnesses:
                    node_key = cdef_node_key(candidate)
                    if node_key in seen_candidates:
                        continue
                    seen_candidates.add(node_key)
                    witness_budget.consume()
                    retained_items.extend(
                        self._verify_cdefs(
                            (candidate,),
                            stats=stats,
                            generator_budget=assignment_budget,
                        )
                    )
                if not private_evidence and universe.definitions:
                    raise QueryDomainError(
                        "GeneratorSelector refinement requires complete retained witness evidence."
                    )
                matched_nodes = {
                    cdef_node_key(candidate) for candidate in retained_items
                }
                private_evidence = tuple(
                    item
                    for item in private_evidence
                    if cdef_node_key(item.target if carrier == "target" else item.owner)
                    in matched_nodes
                )
                public_evidence = {}
                for item in private_evidence:
                    public_evidence.setdefault(containment_witness_key(item), item)
                evidence = tuple(
                    public_evidence[key] for key in sorted(public_evidence)
                )
                merged = {}
                for candidate in retained_items:
                    merged.setdefault(candidate, candidate)
                retained = tuple(
                    sorted(
                        merged.values(),
                        key=lambda cdef: (cdef.stable_hash(), repr(cdef)),
                )
                )
                matched_replicas = {}
                if carrier == "owner":
                    if private_replicas is None and private_evidence:
                        raise QueryDomainError(
                            "GeneratorSelector refinement requires complete retained witness evidence."
                        )
                    for candidate in retained_items:
                        canonical = merged[candidate]
                        matched_replicas.setdefault(canonical, []).extend(
                            (private_replicas or {}).get(cdef_node_key(candidate), ())
                        )
            retained_set = set(retained)
            if evidence and self.generator_selector is None:
                evidence = tuple(
                    item
                    for item in evidence
                    if (item.target if carrier == "target" else item.owner)
                    in retained_set
                )
                private_evidence = tuple(
                    item
                    for item in private_evidence
                    if (item.target if carrier == "target" else item.owner)
                    in retained_set
                )
        if private_replicas is not None:
            private_owner_keys = {
                cdef_node_key(item.owner) for item in private_evidence
            }
            private_replicas = {
                key: stores
                for key, stores in private_replicas.items()
                if key in private_owner_keys
            }
        replicas = {}
        if carrier == "owner" and private_replicas is not None:
            for item in private_evidence:
                replicas.setdefault(item.owner, []).extend(
                    private_replicas.get(cdef_node_key(item.owner), ())
                )
            replicas = {
                cdef: tuple(dict.fromkeys(replicas.get(cdef, ()))) for cdef in retained
            }
        elif universe.replicas is not None:
            replicas = {cdef: universe.replicas.get(cdef, ()) for cdef in retained}
        witnesses = (
            tuple(
                item.target
                for item in private_evidence
                if isinstance(item.target, ConcreteDefinition)
            )
            if carrier == "target"
            else tuple(item.owner for item in private_evidence)
        )
        stats.result_count = len(retained)
        return DefinitionResultSet(
            self.repo,
            retained,
            materializable=universe.materializable,
            domain=("nested-definitions" if carrier == "target" else "owners"),
            explanation=self._explanation(stats),
            replicas=replicas,
            containment=context,
            containment_witnesses=evidence,
            witnesses=witnesses,
            witness_complete=(
                context.complete and universe.witness_complete and not context.bounded
            ),
            containment_private_witnesses=private_evidence,
            containment_private_replicas=private_replicas,
            containment_carrier=carrier,
        )

    def _execute_nested_definitions(
        self,
    ) -> tuple[tuple[ConcreteDefinition, ...], QueryStats]:
        if self._is_deterministic_empty_containment():
            stats = QueryStats(
                fast_path="deterministic-empty-containment", result_count=0
            )
            return (), stats
        if self._uses_authoritative_containment_residual():
            return self._execute_authoritative_containment_definitions()
        if self.universe is None and self.repo._query_index.can_execute_query_domain(
            "nested"
        ):
            return self.repo._query_index.execute_nested_definitions(self)
        matches, _, stats, _ = self._execute_nested_definition_matches()
        stats.result_count = len(matches)
        return matches, stats

    def _execute_nested_owners(self):
        if self._is_deterministic_empty_containment():
            stats = QueryStats(
                fast_path="deterministic-empty-containment", result_count=0
            )
            return (), stats, {}
        if self._uses_authoritative_containment_residual():
            return self._execute_authoritative_containment_owners()
        if self.universe is None and self.repo._query_index.can_execute_query_domain(
            "nested"
        ):
            return self.repo._query_index.execute_nested_owners(self)
        catalog = self.repo._query_catalog
        for _ in range(_MAX_NESTED_QUERY_RETRIES):
            matches, match_ids, stats, generation = (
                self._execute_nested_definition_matches()
            )
            try:
                projection = self._project_owners(catalog, match_ids, generation)
                break
            except _QueryGenerationChanged:
                continue
        else:
            raise QueryIndexError(
                "Catalog generation changed repeatedly during nested owner query."
            )
        owners = projection.cdefs
        owner_replicas = projection.replicas
        stats.result_count = len(owners)
        owners = tuple(
            sorted(owners, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
        )
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

        return tuple(
            sorted(
                self._containment_source_key(store)
                for store in self._containment_stores()
            )
        )

    def _fresh_containment_context(self, *, complete: bool) -> ContainmentContext:
        """Build fixed-result policy metadata for one fresh containment terminal."""

        from .model import containment_target_kind

        return ContainmentContext(
            target_kind=(
                containment_target_kind(self.containment_target)
                if self.containment_target is not None
                else "definition"
            ),
            edges=self.containment_edges,
            contains_ref=self.contains_ref,
            source_scope=self._containment_source_scope(),
            complete=complete,
            bounded=False,
        )

    def _containment_stores(self) -> tuple[Any, ...]:
        """Return selected physical sources once, in Repo priority order."""

        selected = (
            (self.source_store,)
            if self.source_store is not None
            else tuple(self.repo.stores)
        )
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

    def _refresh_authoritative_containment_sources(
        self, stores, stats: QueryStats
    ) -> None:
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
                    source_plans.append(
                        SourceQueryPlan(
                        source_key=source_key,
                        backend=status.backend,
                        generation=generation,
                        candidate_count=len(roots),
                        refresh_action="authority-root-residual",
                        )
                    )
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
            return (
                lambda value: type(value) is type(exact_target)
                and value == exact_target
            )

        verified: set[ConcreteDefinition] = set()

        def matches(value) -> bool:
            if not isinstance(value, ConcreteDefinition):
                return False
            if value not in verified:
                verified.add(value)
                stats.verified_count += 1
                stats.python_verifications += 1
                if (
                    self.max_verify_limit is not None
                    and stats.verified_count > self.max_verify_limit
                ):
                    raise QueryVerifyBudgetExceeded(
                        f"Query exceeded max_verify budget {self.max_verify_limit}: "
                        f"verified {stats.verified_count} CDefs."
                    )
            return _query_match(
                self.selector,
                value,
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
            (root for root, _ in entries),
            matches,
            edges=self.containment_edges,
            contains_ref=self.contains_ref,
        ):
            if isinstance(value, ConcreteDefinition):
                merged.setdefault(value, value)
        out = tuple(
            sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
        )
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
                (root,),
                matches,
                edges=self.containment_edges,
                contains_ref=self.contains_ref,
            ):
                canonical = merged.setdefault(owner, owner)
                replicas.setdefault(canonical, []).append(store)
        out = tuple(
            sorted(merged.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
        )
        stats.result_count = len(out)
        return (
            out,
            stats,
            {owner: tuple(dict.fromkeys(replicas.get(owner, ()))) for owner in out},
        )

    def _execute_authoritative_containment_projection_evidence(self, carrier: str):
        """Capture minimal complete owner/terminal ledgers for a direct projection.

        The projection walker contributes one actual path per matching terminal
        per root.  This supports fixed carrier and exact-target refinement while
        avoiding raw enumeration of every path through a shared DAG.
        """

        if carrier not in {"target", "owner"}:
            raise ValueError("Containment projection carrier must be target or owner.")
        stats = QueryStats()
        entries = self._capture_authoritative_containment_roots(stats)
        matches = self._containment_matcher(stats)
        selected: dict[ConcreteDefinition, ConcreteDefinition] = {}
        replicas: dict[ConcreteDefinition, list[Any]] = {}
        witness_map = {}
        private_witnesses = []
        private_replicas: dict[object, list[Any]] = {}
        grouped_entries: dict[str, list[tuple[ConcreteDefinition, list[Any]]]] = {}
        for root, store in entries:
            groups = grouped_entries.setdefault(root.graph_hash(), [])
            for representative, stores in groups:
                if (
                    all(
                        self._containment_source_key(source)
                        != self._containment_source_key(store)
                        for source in stores
                    )
                    and representative.graph_equal(root)
                ):
                    stores.append(store)
                    break
            else:
                groups.append((root, [store]))
        root_groups = (group for groups in grouped_entries.values() for group in groups)
        for root, stores in root_groups:
            for occurrence in iter_containment_projection_occurrences_matching(
                (root,),
                matches,
                edges=self.containment_edges,
                    contains_ref=self.contains_ref,
            ):
                if carrier == "target":
                    if not isinstance(occurrence.target, ConcreteDefinition):
                        continue
                    private_witnesses.append(occurrence)
                    private_replicas.setdefault(cdef_node_key(occurrence.owner), []).extend(
                        stores
                    )
                    selected.setdefault(occurrence.target, occurrence.target)
                    owner_target = (occurrence.owner, occurrence.target)
                    existing = witness_map.get(owner_target)
                    if existing is None or containment_witness_key(
                        occurrence
                    ) < containment_witness_key(existing):
                        witness_map[owner_target] = occurrence
                    continue
                owner = selected.setdefault(occurrence.owner, occurrence.owner)
                replicas.setdefault(owner, []).extend(stores)
                private_witnesses.append(occurrence)
                private_replicas.setdefault(cdef_node_key(occurrence.owner), []).extend(
                    stores
                )
                witness_map.setdefault(containment_witness_key(occurrence), occurrence)
        out = tuple(
            sorted(selected.values(), key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
        )
        evidence = tuple(
            witness_map[key]
            for key in sorted(
                witness_map, key=(lambda key: containment_witness_key(witness_map[key]))
            )
        )
        stats.result_count = len(out)
        return (
            out,
            stats,
            {cdef: tuple(dict.fromkeys(replicas.get(cdef, ()))) for cdef in out},
            evidence,
            tuple(private_witnesses),
            {
                key: tuple(dict.fromkeys(stores))
                for key, stores in private_replicas.items()
            },
        )

    def _execute_authoritative_containment_occurrences(self):
        """Collect canonically ordered root-local witnesses under one global cap."""

        from .model import containment_witness_key

        stats = QueryStats()
        entries = self._capture_authoritative_containment_roots(stats)
        matches = self._containment_matcher(stats)
        limit = (
            None
            if self.projection in {"object_refs", "state_refs"}
            else self.occurrence_limit
        )
        if limit == 0:
            stats.result_count = 0
            return (), stats, {}, (), {}
        occurrences = []
        private_occurrences = []
        private_replicas: dict[object, list[Any]] = {}
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
        root_groups = (group for groups in grouped_entries.values() for group in groups)

        def root_occurrences(root, stores):
            for occurrence in iter_containment_occurrences_matching(
                (root,),
                matches,
                edges=self.containment_edges,
                    contains_ref=self.contains_ref,
            ):
                yield occurrence, stores

        def retain_private(occurrence, stores):
            private_occurrences.append(occurrence)
            private_replicas.setdefault(cdef_node_key(occurrence.owner), []).extend(
                stores
            )

        def same_public_witness(left, right):
            return (
                left.owner == right.owner
                and left.path == right.path
                and left.hops == right.hops
                and type(left.target) is type(right.target)
                and left.target == right.target
            )

        pending = []
        for index, (root, stores) in enumerate(root_groups):
            iterator = root_occurrences(root, stores)
            try:
                occurrence, stores = next(iterator)
            except StopIteration:
                continue
            heappush(
                pending,
                (containment_witness_key(occurrence), index, occurrence, stores, iterator),
            )
        while pending:
            key, index, occurrence, stores, iterator = heappop(pending)
            retain_private(occurrence, stores)
            if key not in seen:
                seen.add(key)
                occurrences.append(occurrence)
                if limit is not None and len(occurrences) >= limit:
                    for _, _, pending_occurrence, pending_stores, _ in pending:
                        if same_public_witness(occurrence, pending_occurrence):
                            retain_private(pending_occurrence, pending_stores)
                    break
            try:
                occurrence, stores = next(iterator)
            except StopIteration:
                continue
            heappush(
                pending,
                (containment_witness_key(occurrence), index, occurrence, stores, iterator),
            )
        private_occurrences = tuple(
            occurrence
            for occurrence in private_occurrences
            if any(
                same_public_witness(visible, occurrence)
                for visible in occurrences
            )
        )
        private_owner_keys = {
            cdef_node_key(occurrence.owner) for occurrence in private_occurrences
        }
        private_replicas = {
            key: tuple(dict.fromkeys(stores))
            for key, stores in private_replicas.items()
            if key in private_owner_keys
        }
        replicas = {}
        for occurrence in private_occurrences:
            for visible in occurrences:
                if same_public_witness(visible, occurrence):
                    replicas.setdefault(visible.owner, []).extend(
                        private_replicas.get(cdef_node_key(occurrence.owner), ())
                    )
        store_priority = {
            self._containment_source_key(store): index
            for index, store in enumerate(self._containment_stores())
        }
        stats.result_count = len(occurrences)
        return (
            tuple(occurrences),
            stats,
            {
                owner: tuple(
                    sorted(
                        dict.fromkeys(stores),
                        key=lambda store: store_priority[
                            self._containment_source_key(store)
                        ],
                    )
                )
                for owner, stores in replicas.items()
            },
            private_occurrences,
            private_replicas,
        )

    def _execute_nested_definition_matches(self):
        stats = QueryStats()
        catalog = self.repo._query_catalog
        captured = self._capture_nested_candidates(catalog, stats)
        matches, match_ids = self._verify_cdefs_by_id(captured.cdefs_by_id, stats=stats)
        return matches, match_ids, stats, captured.generation

    def _capture_nested_candidates(
        self, catalog, stats: QueryStats
    ) -> CapturedNestedCandidates:
        catalog.refresh(self.refresh_policy, stats=stats)
        with catalog.read_view(include_cached=False) as snapshot:
            selector_graph = compile_selector_graph(
                self.selector, class_match=self.class_match_policy
            )
            if selector_graph is not None:
                candidate_ids = graph_candidate_ids(
                    snapshot, selector_graph, None, stats=stats
                )
                candidate_ids = snapshot.filter_nested_ids(candidate_ids)
                stats.candidate_count = len(candidate_ids)
            else:
                domain = NestedDomain(snapshot)
                candidate_ids = domain.all_ids()
                stats.candidate_count = len(candidate_ids)
                stats.universe_size = None
            cdefs_by_id = snapshot.cdefs_by_id(candidate_ids)
            generation = snapshot.generation
        return CapturedNestedCandidates(
            generation=generation, cdefs_by_id=cdefs_by_id, stats=stats
        )

    def _capture_occurrence_traversal(
        self, catalog, ids: set[DefinitionId] | frozenset[DefinitionId], generation: int
    ):
        with catalog.read_view(include_cached=False) as snapshot:
            if snapshot.generation != generation:
                raise _QueryGenerationChanged
            return snapshot.occurrence_snapshot_for_nested_ids(set(ids))

    def _project_owners(
        self, catalog, ids: set[DefinitionId] | frozenset[DefinitionId], generation: int
    ):
        with catalog.read_view(include_cached=False) as snapshot:
            if snapshot.generation != generation:
                raise _QueryGenerationChanged
            return snapshot.project_owners(set(ids))

    def _verify_cdefs_by_id(
        self, cdefs_by_id: dict[DefinitionId, ConcreteDefinition], *, stats: QueryStats
    ) -> tuple[tuple[ConcreteDefinition, ...], set[DefinitionId]]:
        matches = self._verify_cdefs(tuple(cdefs_by_id.values()), stats=stats)
        match_set = set(matches)
        match_ids = {did for did, cdef in cdefs_by_id.items() if cdef in match_set}
        return matches, match_ids

    def _verify_cdefs(
            self,
            cdefs: tuple[ConcreteDefinition, ...],
            *,
            stats: QueryStats,
        generator_budget=None,
    ) -> tuple[ConcreteDefinition, ...]:
        if self.selector is None:
            stats.verified_count += len(cdefs)
            stats.python_verifications += len(cdefs)
            if (
                self.max_verify_limit is not None
                and stats.verified_count > self.max_verify_limit
            ):
                raise QueryVerifyBudgetExceeded(
                    f"Query exceeded max_verify budget {self.max_verify_limit}: verified {stats.verified_count} CDefs."
                )
            return tuple(
                sorted(cdefs, key=lambda cdef: (cdef.stable_hash(), repr(cdef)))
            )

        out: list[ConcreteDefinition] = []
        for cdef in cdefs:
            stats.verified_count += 1
            stats.python_verifications += 1
            if (
                self.max_verify_limit is not None
                and stats.verified_count > self.max_verify_limit
            ):
                raise QueryVerifyBudgetExceeded(
                    f"Query exceeded max_verify budget {self.max_verify_limit}: verified {stats.verified_count} CDefs."
                )
            if not _structural_match(
                    self.selector,
                    cdef,
                    strict=self.strict_policy,
                class_match=self.class_match_policy,
            ):
                continue
            if self.generator_selector is not None:
                if generator_budget is None:
                    matched = self.generator_selector.matches(cdef)
                else:
                    matched = self.generator_selector._matches(cdef, generator_budget)
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
    from ..generator import GeneratorSelector

    if isinstance(source, GeneratorSelector):
        return source.prefilter.root
    if isinstance(source, Definition):
        return deepcopy(source)
    raise TypeError(
        f"Query source must be Selector, Definition, ConcreteDefinition, Object, or None, not {type(source).__name__}."
    )


def _resolve_query_state_selectors(source, repo):
    """Resolve every soft StateSelectorRef before a DefinitionQuery exists.

    Query selectors may remain partial and contain ``Par`` values, so they cannot
    use Definition-to-CDef concretization as a resolver.  This graph-preserving
    rewrite only replaces soft selector leaves and retains all selector syntax.
    """

    from ..selector import _resolve_state_selectors

    return _resolve_state_selectors(source, repo)


def _structural_match(
    selector, cdef: ConcreteDefinition, *, strict: bool, class_match: ClassMatchPolicy
) -> bool:
    if not _query_match(selector, cdef, strict=strict, class_match=class_match):
        return False
    return True


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


@dataclass(frozen=True, slots=True)
class IdentityQuery:
    """Private composable Query V3 restriction plan over complete identities.

    The class is intentionally not yet produced by ``Repo.query`` or
    ``Store.query``.  It provides U3's shared restriction semantics while the
    public cutover remains owned by U7.  A terminal captures all requested Store
    facts once, evaluates only detached identities and facts, and returns an
    immutable :class:`IdentitySet` without constructing or saving Objects.
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
        """Create an unevaluated V3 universe from one Store producer."""

        from .source import StoreSource

        source = StoreSource(store)
        return cls(source, default_scope=source)

    @classmethod
    def from_repo(cls, repo, *, weak: bool = True) -> "IdentityQuery":
        """Create an unevaluated V3 universe from one Repo producer."""

        from .source import RepoSource

        source = RepoSource(repo, weak=weak)
        return cls(source, default_scope=source)

    @classmethod
    def from_retained_repo_lookup(cls, repo, *, weak: bool = True) -> "IdentityQuery":
        """Create the private V3 root-or-cache universe for retained Repo APIs."""

        from .source import RetainedRepoLookupSource

        source = RetainedRepoLookupSource(repo, weak=weak)
        return cls(source, default_scope=source)

    @classmethod
    def from_set(cls, members) -> "IdentityQuery":
        """Create an unevaluated V3 refinement over a fixed IdentitySet."""

        from .identity import IdentitySet

        if not isinstance(members, IdentitySet):
            raise TypeError("IdentityQuery.from_set requires an IdentitySet.")
        return cls(members)

    def sel(self, value=None) -> "IdentityQuery":
        """Restrict this fixed universe by one structural or exact selector.

        Definitions remain structural even when complete; ConcreteDefinitions,
        exact references, and GeneratorSelector support retain their stricter
        established meanings.  ``None`` is an immutable no-op.
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
        """Restrict ObjectRef and StateRef candidates containing ``value``."""

        from ..reference_values import ObjectId

        if not isinstance(value, ObjectId):
            raise TypeError("object_id requires an ObjectId.")
        return self._append("object_id", value)

    def namespace(self, prefix) -> "IdentityQuery":
        """Restrict reference candidates to an already validated namespace prefix."""

        from ..reference_values import _normalize_namespace

        return self._append("namespace", _normalize_namespace(tuple(prefix)))

    def contains(self, value) -> "IdentityQuery":
        """Restrict aggregates to a proper owned subtree exact ObjectRef."""

        from ..reference_values import ObjectRef

        if not isinstance(value, ObjectRef):
            raise TypeError("contains requires an ObjectRef.")
        return self._append("contains", value)

    def state_hash(self, value: str) -> "IdentityQuery":
        """Restrict StateRefs to an existing complete local state hash."""

        from ..reference_values import _validate_state_hash

        _validate_state_hash(value)
        return self._append("state_hash", value)

    def alias(self, value: str, *, scope=None) -> "IdentityQuery":
        """Restrict by aliases captured from the authority scope bound now."""

        if not isinstance(value, str) or not value:
            raise ValueError("alias requires a non-empty string.")
        return self._append("alias", value, self._bound_scope(scope))

    def stored(self, *, scope=None) -> "IdentityQuery":
        """Restrict members by their kind-specific captured stored authority."""

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
        """Conjoin typed metadata without changing the identity query family."""

        from .metadata import _require_predicate

        return self._append("metadata", _require_predicate(predicate), self._bound_scope(scope))

    def in_source(self, source) -> "IdentityQuery":
        """Restrict captured contribution evidence and bind later defaults to it."""

        scope = self._normalize_scope(source)
        return replace(self._append("source", source), default_scope=scope)

    def union(self, other: "IdentityQuery | IdentitySet") -> "IdentityQuery":
        """Return a deferred complete-identity union with shared source cuts.

        The result keeps a default authority scope only when both operand plans
        name the same Store or Repo object.  Different producers therefore
        compose without choosing authority by source order.
        """

        return self._combine(other, "union")

    def intersection(self, other: "IdentityQuery | IdentitySet") -> "IdentityQuery":
        """Return a deferred complete-identity intersection with shared cuts.

        Matching identities retain the evidence supplied by both operands.  As
        with :meth:`union`, an ambiguous authority default is deliberately
        removed rather than selected from operand order.
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
        """

        if type(limit) is not int or limit < 0:
            raise ValueError("take limit must be a non-negative exact int.")
        return replace(self, take_limit=limit).collect()

    def max_witnesses(self, limit: int | None) -> "IdentityQuery":
        """Set the shared GeneratorSelector verification witness budget."""

        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_witnesses limit must be a positive exact int or None.")
        return replace(self, max_witness_limit=limit)

    def refresh(self, policy: RefreshPolicy = "auto") -> "IdentityQuery":
        """Record the existing refresh policy without changing V3 authority facts."""

        if policy not in {False, "auto", True}:
            raise ValueError("refresh policy must be False, 'auto', or True.")
        return replace(self, refresh_policy=policy)

    def scan_policy(self, policy: str) -> "IdentityQuery":
        """Control whether a required V3 source inventory scan is permitted."""

        if policy not in {"allow", "warn", "forbid"}:
            raise ValueError("scan policy must be 'allow', 'warn', or 'forbid'.")
        return replace(self, scan_policy_mode=policy)

    def require_indexed(self) -> "IdentityQuery":
        """Require proved V3 index coverage, not merely a selective Store read."""

        return replace(self, scan_policy_mode="forbid", indexed_required=True)

    def max_verify(self, limit: int | None) -> "IdentityQuery":
        """Set the maximum detached identity candidates verified by a terminal."""

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_verify limit must be a non-negative exact int or None.")
        return replace(self, max_verify_limit=limit)

    def max_depth(self, limit: int | None) -> "IdentityQuery":
        """Set the relationship-depth safety budget for later closure or nesting."""

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_depth limit must be a non-negative exact int or None.")
        return replace(self, max_depth_limit=limit)

    def explain(self, *, analyze: bool = False, sql: bool = False) -> QueryExplanation:
        """Describe whether this immutable V3 plan requires an authority scan.

        ``sql`` is accepted for interface parity but reports no backend detail:
        U3 evaluates authoritative detached records rather than a SQL result.
        ``analyze`` evaluates the plan and includes its final result count.
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
            result_count=result_count,
            fast_path="exact-state-ref" if exact else None,
            scan_required=scan_required,
            scan_reason=("V3 identity inventory requires authoritative record enumeration" if scan_required else None),
            source_cuts=source_cuts,
            capture_rounds=capture_rounds,
            demand_recaptures=demand_recaptures,
            instability_retries=capture.instability_retries if analyze else 0,
        ).explanation(domain="identity", refresh=self.refresh_policy)

    def categorical(self, **kwargs) -> "IdentityQuery":
        """Append the categorical edit of this query's latest selector restriction."""

        selector = self._last_selector()
        if selector is None:
            raise QueryDomainError("categorical() requires an existing Selector restriction.")
        return self.sel(selector.categorical(**kwargs))

    def restore(self, **kwargs) -> "IdentityQuery":
        """Append the restored form of this query's latest selector restriction."""

        selector = self._last_selector()
        if selector is None:
            raise QueryDomainError("restore() requires an existing Selector restriction.")
        return self.sel(selector.restore(**kwargs))

    def exact(self, definition=None, **kwargs) -> "IdentityQuery":
        """Append an exact subtree edit without discarding earlier restrictions."""

        selector = self._last_selector()
        if selector is None:
            raise QueryDomainError("exact() requires an existing Selector restriction.")
        return self.sel(selector.exact(definition, **kwargs))

    def collect(self):
        """Finish detached membership evaluation and return a fixed IdentitySet."""

        return self._run_terminal(self._collect_with_capture)

    def count(self) -> int:
        """Return the distinct V3 identity count after all validation."""

        if self.take_limit is not None:
            return self._run_terminal(lambda cut: len(self._bounded_terminal_entries(cut)[0]))
        return self._run_terminal(lambda cut: sum(1 for _ in self._iter_terminal_members(cut)))

    def exists(self) -> bool:
        """Return whether one fully valid V3 identity remains."""

        if self.take_limit == 0:
            return False
        return self._run_terminal(lambda cut: next(self._iter_terminal_members(cut), None) is not None)

    def one(self):
        """Return one V3 identity or raise the normal cardinality error."""

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
        """Return zero or one V3 identity, rejecting ambiguous matches."""

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

        capture = SourceCapture()
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
        if limit is None and isinstance(self.source, IdentitySet):
            limit = self.source.requested_limit
        return IdentitySet._from_entries(
            members, bounded=bounded, requested_limit=limit,
        )

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
        selected = []
        for key, entry in self._iter_terminal_members(capture):
            _retain_bounded_identity_entry(selected, key, entry, self.take_limit)
        return (
            {key: entry for key, entry in selected},
            True,
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
        fixed_source = isinstance(
            self.source, (IdentitySet, _RelationshipClosure, _RelationshipProjection),
        )
        needs_inventory = any(
            restriction.kind in {"alias", "stored"} for restriction in self.restrictions
        )
        if isinstance(self.source, StoreSource) and not scopes and self.restrictions:
            exact_cdefs = [
                restriction.value for restriction in self.restrictions
                if restriction.kind == "sel" and isinstance(restriction.value, ConcreteDefinition)
            ]
            same_stored_scope = any(
                restriction.kind == "stored"
                and isinstance(restriction.scope, StoreSource)
                and restriction.scope.store is self.source.store
                for restriction in self.restrictions
            )
            if (
                exact_cdefs and same_stored_scope
                and any(restriction.kind == "kind" and restriction.value == "cdef"
                        for restriction in self.restrictions)
                and all(restriction.kind in {"sel", "kind", "stored", "source"}
                        for restriction in self.restrictions)
            ):
                direct = capture.read_exact_stored_cdef(self.source, exact_cdefs[0])
                if direct is not None:
                    return IdentitySet(((direct, self._source_evidence(self.source)),))._entries, capture, False
        if (
            self.scan_policy_mode == "forbid"
            and not fixed_source
            and (exact_state is None or scopes or needs_inventory)
        ):
            raise QueryWouldScanError("Query V3 requires an authoritative inventory scan.")
        if exact_state is not None and not scopes and not needs_inventory:
            if isinstance(self.source, StoreSource):
                state = capture.read_exact_state(self.source, exact_state)
                members = () if state is None else ((state, self._source_evidence(self.source)),)
                return IdentitySet(members)._entries, capture, False
            if isinstance(self.source, RepoSource):
                with self.source.repo.retain_topology():
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

    def _scope_facts(self, scope, capture):
        from .source import RepoSource, StoreSource

        scopes = self._metadata_scopes()
        if isinstance(scope, StoreSource):
            return (capture.capture_store(scope, metadata_scopes=scopes),)
        if isinstance(scope, RepoSource):
            repo = scope.repo
            with repo.retain_topology():
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
                return isinstance(value, ObjectRef) and value == selector
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
            from .source import StoreSource

            if (
                isinstance(value, ConcreteDefinition)
                and isinstance(restriction.scope, StoreSource)
                and capture.has_exact_stored_cdef(restriction.scope, value)
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
        from ..selector import Selector

        for restriction in reversed(self.restrictions):
            if restriction.kind == "sel" and isinstance(restriction.value, Selector):
                return restriction.value
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
    """Private immutable Query V3 relationship occurrence plan.

    The query traverses only identities retained by ``roots``. Its raw terminal
    preserves every non-empty typed path; ``owners`` and ``targets`` use direct
    existential traversal and therefore do not inherit ``max_occurrences``.
    """

    roots: IdentityQuery
    selector: Any | None = None
    edges: Any = None
    through_kinds: frozenset[Any] = frozenset()
    path_filter: Any | None = None
    occurrence_limit: int | None = None
    max_depth_limit: int | None = None
    target_kind_filter: frozenset[str] = frozenset()
    fixed: Any | None = None
    metadata_restrictions: tuple[tuple[Any, Any], ...] = ()
    additional_selectors: tuple[Any, ...] = ()

    @classmethod
    def from_set(cls, occurrences) -> "OccurrenceQuery":
        """Refine only captured occurrences without adopting a live producer."""

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
        if not self.target_kind_filter <= {"cdef", "object_ref", "state_ref"}:
            raise ValueError("OccurrenceQuery target kinds are unsupported.")
        if self.selector is not None:
            # Reuse the U3 selector validator without evaluating any source.
            IdentityQuery.from_set(IdentitySet()).sel(self.selector)
        for selector in self.additional_selectors:
            IdentityQuery.from_set(IdentitySet()).sel(selector)

    def sel(self, value=None) -> "OccurrenceQuery":
        """Conjoin a U3-compatible target selector without changing roots."""

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

        return replace(self, target_kind_filter=frozenset(("cdef",)))

    def object_refs(self) -> "OccurrenceQuery":
        """Restrict occurrence targets to ObjectRefs without expansion."""

        return replace(self, target_kind_filter=frozenset(("object_ref",)))

    def state_refs(self) -> "OccurrenceQuery":
        """Restrict occurrence targets to StateRefs without expansion."""

        return replace(self, target_kind_filter=frozenset(("state_ref",)))

    def target_kind(self, kind: str) -> "OccurrenceQuery":
        """Restrict targets to one explicit V3 identity kind."""

        if kind not in {"cdef", "object_ref", "state_ref"}:
            raise ValueError("OccurrenceQuery target kind is unsupported.")
        return replace(self, target_kind_filter=frozenset((kind,)))

    def through(self, kind) -> "OccurrenceQuery":
        """Require one relationship kind on the same retained typed path."""

        from .relationships import RelationshipKind

        if not isinstance(kind, RelationshipKind):
            raise TypeError("OccurrenceQuery through requires a RelationshipKind.")
        return replace(self, through_kinds=self.through_kinds | frozenset((kind,)))

    def path(self, value) -> "OccurrenceQuery":
        """Require an exact RelationshipPath or concatenated legacy GraphPath."""

        from .relationships import RelationshipPath
        from ..utils.graph.path import GraphPath

        if not isinstance(value, (RelationshipPath, GraphPath)):
            raise TypeError("OccurrenceQuery path requires a RelationshipPath or GraphPath.")
        return replace(self, path_filter=value)

    def max_occurrences(self, limit: int | None) -> "OccurrenceQuery":
        """Set the visible raw-occurrence cap without bounding direct projections."""

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
        """Set a failing relationship-depth safety budget for this traversal."""

        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("max_depth must be a non-negative exact int or None.")
        return replace(self, max_depth_limit=limit)

    def collect(self):
        """Evaluate typed raw occurrences into a bounded or complete OccurrenceSet."""

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
        """Stream the distinct raw typed-occurrence count without a fixed set."""

        return self.roots._run_terminal(lambda cut: sum(1 for _ in self._iter_occurrences(cut)))

    def exists(self) -> bool:
        """Return whether one qualified occurrence exists without collection."""

        return self.roots._run_terminal(lambda cut: next(self._iter_occurrences(cut), None) is not None)

    def one(self):
        """Return one streamed occurrence or raise the standard cardinality error."""

        occurrences = self.roots._run_terminal(self._up_to_two_occurrences)
        if len(occurrences) != 1:
            raise QueryCardinalityError(
                f"Expected exactly one occurrence, found {len(occurrences)}."
            )
        return occurrences[0]

    def one_or_none(self):
        """Return one streamed occurrence, ``None``, or raise on ambiguity."""

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
        if kind is None or (self.target_kind_filter and kind not in self.target_kind_filter):
            return False
        if self.selector is None and not self.additional_selectors:
            return True
        query = IdentityQuery.from_set(IdentitySet((value,)))
        for selector in ((self.selector,) if self.selector is not None else ()) + self.additional_selectors:
            query = query.sel(selector)
        return query.exists()

    def __bool__(self) -> bool:
        raise TypeError("OccurrenceQuery requires an explicit terminal.")
