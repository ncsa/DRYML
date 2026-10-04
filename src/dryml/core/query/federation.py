"""Manage derived query indexes attached to a multi-Store Repo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model import (
    QueryIndexStatus,
    QueryIndexUnavailable,
    QueryStats,
    RefreshPolicy,
    ValidationIssue,
    ValidationReport,
)


@dataclass(frozen=True, slots=True)
class StoreIndexBinding:
    """Associate one Repo Store with its derived query-index handle."""

    store: Any
    source_key: str
    priority: int
    index: Any | None = None


class RepoQueryIndex:
    """Coordinate derived-index lifecycle operations across a Repo's Stores."""

    def __init__(self, repo, *, authority_only: bool = False):
        self.repo = repo
        self._authority_only = authority_only
        self._opened_indexes: dict[str, Any] = {}
        self._bindings: tuple[StoreIndexBinding, ...] = ()
        self.refresh_bindings()

    @property
    def store_bindings(self) -> tuple[StoreIndexBinding, ...]:
        """Return current Store bindings in Repo priority order."""

        return self._bindings

    def refresh_bindings(self) -> tuple[StoreIndexBinding, ...]:
        """Rebuild and return bindings from the Repo's current Store topology."""

        bindings: list[StoreIndexBinding] = []
        seen: set[str] = set()
        for priority, store in enumerate(self.repo.stores):
            source_key = _store_source_key(store)
            if source_key in seen:
                continue
            seen.add(source_key)
            bindings.append(StoreIndexBinding(
                store=store,
                source_key=source_key,
                priority=priority,
                index=self._opened_indexes.get(source_key),
            ))
        self._bindings = tuple(bindings)
        return self._bindings

    def open_store_index(self, binding: StoreIndexBinding):
        """Open and retain a Store's configured derived index when available."""

        if self._authority_only:
            raise QueryIndexUnavailable(
                "This private state-IO Repo view cannot open persistent query indexes."
            )
        existing = self._opened_indexes.get(binding.source_key)
        if existing is not None:
            return existing
        opener = getattr(binding.store, "open_query_index", None)
        if opener is None:
            return None
        index = opener()
        if index is not None:
            self._opened_indexes[binding.source_key] = index
            self.refresh_bindings()
        return index

    def index_status(self, store=None) -> tuple[QueryIndexStatus, ...]:
        """Return derived-index status for one Store or every attached Store."""

        return tuple(
            self._status_for_binding(binding)
            for binding in self._bindings_for_store(store)
        )

    def validate(
            self,
            store=None,
            *,
            thorough: bool = False) -> tuple[ValidationReport, ...]:
        """Validate available derived indexes without changing Store authority."""

        reports = []
        for binding in self._bindings_for_store(store):
            policy = getattr(binding.store, "query_index_policy", "memory")
            if policy == "none":
                reports.append(ValidationReport("none", binding.source_key, True))
                continue
            if policy == "memory":
                reports.append(ValidationReport("memory", binding.source_key, True))
                continue
            try:
                index = self.open_store_index(binding)
            except QueryIndexUnavailable as exc:
                reports.append(ValidationReport(
                    str(policy),
                    binding.source_key,
                    False,
                    (ValidationIssue("error", str(exc)),),
                ))
                continue
            if index is None:
                reports.append(ValidationReport("memory", binding.source_key, True))
                continue
            validate = getattr(index, "validate", None)
            reports.append(
                ValidationReport(type(index).__name__, binding.source_key, True)
                if validate is None else validate(thorough=thorough)
            )
        return tuple(reports)

    def refresh(
            self,
            policy: RefreshPolicy,
            *,
            stats: QueryStats | None = None) -> bool:
        """Refresh configured persistent indexes under one policy.

        Args:
            policy: ``False``, ``"auto"``, or ``True`` refresh behavior.
            stats: Optional mutable execution counters passed to each backend.

        Returns:
            ``True`` when at least one configured backend completed refresh;
            ``False`` when no persistent backend was available.

        Raises:
            QueryIndexUnavailable: If forced refresh cannot reach a backend.

        Side Effects:
            May validate, rebuild, or replace derived sidecars. Store authority
            is not modified.
        """

        self.refresh_bindings()
        refreshed = False
        for binding in self._bindings:
            if getattr(binding.store, "query_index_policy", None) in {"memory", "none"}:
                continue
            try:
                index = self.open_store_index(binding)
                if index is None:
                    continue
                refresh = getattr(index, "refresh", None)
                if refresh is None:
                    continue
                refresh(policy, stats=stats)
            except QueryIndexUnavailable:
                if policy is True:
                    raise
                continue
            refreshed = True
        return refreshed

    def rebuild(self, store=None) -> None:
        """Rebuild configured persistent indexes for selected Stores."""

        for binding in self._bindings_for_store(store):
            if getattr(binding.store, "query_index_policy", None) in {"memory", "none"}:
                continue
            index = self.open_store_index(binding)
            if index is None:
                continue
            rebuild = getattr(index, "rebuild", None)
            if rebuild is not None:
                rebuild()

    def register_saved_graph(
            self,
            graph,
            roots_by_store: Mapping[Any, Sequence[Any]],
            state_refs_by_store: Mapping[Any, Sequence[Any]] | None = None) -> None:
        """Register one completed authoritative save with derived Store indexes.

        Args:
            graph: Complete query graph published by the save.
            roots_by_store: Stored roots grouped by owning Store.
            state_refs_by_store: Newly published StateRefs grouped by Store for
                incremental advisory-reference registration.

        Raises:
            Exception: Propagates a configured index registration failure after
                marking the Store index dirty when supported.

        Side Effects:
            Updates available derived indexes or leaves their durable dirty
            markers in place when registration fails.
        """

        if self._authority_only:
            return
        self.refresh_bindings()
        binding_by_key = {binding.source_key: binding for binding in self._bindings}
        for store, roots in roots_by_store.items():
            roots = tuple(roots)
            if not roots:
                continue
            binding = binding_by_key.get(_store_source_key(store))
            if binding is None:
                continue
            if getattr(store, "query_index_policy", None) in {"memory", "none"}:
                continue
            index = self.open_store_index(binding)
            if index is None:
                continue
            register_saved = getattr(index, "register_saved_graph", None)
            register = register_saved or getattr(index, "register_stored_roots", None)
            if register is None:
                continue
            try:
                if register_saved is not None:
                    references = () if state_refs_by_store is None else tuple(
                        state_refs_by_store.get(store, ())
                    )
                    register(graph, roots, references)
                else:
                    register(graph, roots)
            except QueryIndexUnavailable:
                continue
            except Exception:
                marker = getattr(store, "mark_query_index_dirty", None)
                if marker is not None:
                    marker()
                raise

    def close(self) -> None:
        """Detach this Repo's views without closing Store-owned indexes."""

        # DirStore caches each persistent index. A second Repo can legitimately
        # borrow the same Store, so only Store.close() owns its connections.
        self._opened_indexes.clear()
        self._bindings = ()

    def _bindings_for_store(self, store) -> tuple[StoreIndexBinding, ...]:
        self.refresh_bindings()
        if store is None:
            return self._bindings
        source_key = _store_source_key(store)
        return tuple(
            binding for binding in self._bindings
            if binding.source_key == source_key
        )

    def _status_for_binding(self, binding: StoreIndexBinding) -> QueryIndexStatus:
        policy = getattr(binding.store, "query_index_policy", "memory")
        if policy == "none":
            return QueryIndexStatus(
                backend="none",
                store_key=binding.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="disabled",
            )
        if policy == "memory":
            return QueryIndexStatus(
                backend="memory",
                store_key=binding.source_key,
                generation=self.repo._query_catalog.generation,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        try:
            index = self.open_store_index(binding)
        except QueryIndexUnavailable:
            return QueryIndexStatus(
                backend=str(policy),
                store_key=binding.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="unavailable",
            )
        if index is None:
            return QueryIndexStatus(
                backend="memory",
                store_key=binding.source_key,
                generation=self.repo._query_catalog.generation,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        status = getattr(index, "status", None)
        if status is None:
            return QueryIndexStatus(
                backend=type(index).__name__,
                store_key=binding.source_key,
                generation=None,
                schema_version=None,
                semantic_versions={},
                state="ready",
            )
        return status()


def _store_source_key(store) -> str:
    catalog_key = getattr(store, "catalog_key", None)
    if catalog_key is not None:
        return catalog_key()
    return f"{type(store).__module__}.{type(store).__qualname__}:id:{id(store)}"
