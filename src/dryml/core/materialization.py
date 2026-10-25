"""Node materialization and one-shot state restoration with fresh leased sources.

Normalized construction and retained Node facets own selection semantics. Runtime
binding projections and legacy in-memory adapters do not prove payload availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, ExitStack
from typing import Any, Literal, Mapping

from .canonical import from_canonical
from .cdef_graph import ConcreteDefinitionGraph, EdgeKind
from .definition import ConcreteDefinition, Definition
from .object import (
    Object,
    Serializable,
    _AbstractObjectAdmissionError,
    _admit_materialization_class,
)
from .policies import CachePolicy, LiveReusePolicy
from .symbol import ImportRef, resolve_symbol
from .cdef_identity import cdef_node_key
from .repo_plan import _NodeBindings, attach_runtime_binding, realization_scope
from .utils.graph.path import GraphPath


MaterializationActionKind = Literal["reuse", "construct"]
MaterializationReuseSource = Literal["memo", "cache", None]


class _NodeStateSources(dict):
    """Operation-owned borrowed sources plus the same retained Store topology."""

    def __init__(self, sources, stores):
        super().__init__(sources)
        self.stores = tuple(stores)


def _node_metadata_evidence(repo, node, owned, sources, prepared, *, source_store=None, source_stores=None, reservation=None):
    """Read selected capture/lineage authority afresh, never retain it in a plan.

    Records are metadata-only and independently cut-revalidated outside writer
    fences. Exact capture replicas must agree unless a provider was selected;
    lineage evidence must agree with that capture and existing live identities.
    Missing object declarations are not invented from a donor LocalState.
    Unrelated cooperating publications may invalidate a detached cut; retry only
    this metadata preparation, at most eight times, never constructors or hooks.
    """
    from .store.authority import StoreCutChanged
    for attempt in range(8):
        try:
            return _read_node_metadata_evidence(repo, node, owned, sources, prepared,
                source_store=source_store, source_stores=source_stores, reservation=reservation)
        except StoreCutChanged:
            if attempt == 7:
                raise


def _read_node_metadata_evidence(repo, node, owned, sources, prepared, *, source_store, source_stores, reservation):
    import json
    from .repo import MetadataConflictError
    from .metadata import timestamp_from_seconds
    from .reference_values import ObjectId
    stores = getattr(sources, "stores", tuple(repo.stores))
    overrides = {} if source_stores is None else source_stores
    captured = {}
    selected_nodes = (node, *(current for _, current in owned if current is not node))
    for current in selected_nodes:
        from .node import require_saved
        try:
            require_saved(current)
        except (TypeError, ValueError):
            continue
        provider = (overrides.get(current.state.key) if current.state is not None else None) or source_store
        captures = []
        for store in ((provider,) if provider is not None else stores):
            with store._metadata_read():
                selected = store._read_node_capture_for_selection(current, prepared)
                if selected is not None:
                    captures.append(selected[0].content)
        if captures and any(content != captures[0] for content in captures[1:]):
            raise MetadataConflictError("Selected snapshot evidence conflicts across Stores.")
        for row in (() if not captures else json.loads(captures[0])["lineages"]):
            if row["identity"] is None:
                continue
            identity = ObjectId.from_data(row["identity"])
            previous = captured.setdefault(identity, row)
            if (previous["creation_status"], previous["created_at"]) != (row["creation_status"], row["created_at"]):
                raise MetadataConflictError("Selected lineage evidence conflicts across saved graphs.")
    facts = {}
    for _, current in owned:
        identity = current.object_id
        if identity is None:
            continue
        provider = (overrides.get(current.state.key) if current.state is not None else None) or source_store
        candidates = (provider,) if provider is not None else stores
        rows = []
        root_fact = captured.get(identity)
        if root_fact is not None:
            rows.append((root_fact["creation_status"], root_fact["created_at"]))
        for store in candidates:
            with store._metadata_read():
                if not store._holds_node_metadata(current, "lineage"):
                    continue
                record = store._read_node_metadata(current, "lineage")
                row = None if record is None else json.loads(record.content)
                rows.append(("unknown", None) if row is None else (row["creation_status"], row["created_at"]))
        if rows:
            if any(row != rows[0] for row in rows[1:]):
                raise MetadataConflictError("Selected lineage evidence conflicts across Stores.")
            facts[identity] = timestamp_from_seconds(rows[0][1])
    for candidate in repo._all_live_candidates(reservation=reservation):
        identity = candidate.object_id
        if (identity in facts and getattr(candidate, "_lineage_fact_object_id", None) == identity
                and getattr(candidate, "_lineage_created_at", None) != facts[identity]):
            raise MetadataConflictError("Live cached identity disagrees with selected lineage evidence.")
    return facts


def _runtime_recipe(node):
    """Project Node facets off the retained legacy runtime-binding adapter."""
    from .node import _clone, _Facets, _rewrite

    def strip(source, result):
        if result.object_id is None and result.state is None and result._construction is None:
            return result
        concrete = isinstance(result, ConcreteDefinition)
        return _clone(result, None if concrete else result.args,
                      result.parameters if concrete else result.kwargs, _Facets(), reset_construction=True)

    return _rewrite(node, enter_quoted=False, node_transform=strip)


def _same_creation(candidate, node):
    from .node_views import CREATION_VIEW, view_equal
    return view_equal(getattr(candidate, "_construction_node", candidate.definition), node, CREATION_VIEW)


@contextmanager
def _selected_state_sources(repo, associations, *, source_store=None, source_stores=None):
    """Preflight each destination; borrow each distinct full key once per Store."""
    from .repo import RepoLoadError
    from .repo_plan import state_use_leases, _unique_stores
    from .state_values import StateKey
    from .store.store import Store

    associations = tuple(associations)
    selections = {state.key for _, state, _, _ in associations}
    if source_store is not None and not isinstance(source_store, Store):
        raise TypeError("source_store must be a connected Store.")
    if source_stores is not None and not isinstance(source_stores, Mapping):
        raise TypeError("source_stores must map full StateKeys to connected Stores.")
    overrides = {} if source_stores is None else dict(source_stores)
    if any(not isinstance(key, StateKey) or not isinstance(store, Store) for key, store in overrides.items()):
        raise TypeError("source_stores must map full StateKeys to connected Stores.")
    if set(overrides) - selections:
        raise ValueError("source_stores contains a key outside the selected state closure.")
    with repo.retain_topology(allow_physical_duplicates=True):
        with repo._configuration_lock:
            stores = repo._normalize_store_handles(repo.stores, reject_physical=False)
        if any(store not in stores for store in (*overrides.values(), *((source_store,) if source_store is not None else ()))):
            raise ValueError("State source Store must be connected.")
        for path, state, destination, cls in associations:
            try:
                state.validate_for(destination, codec=cls.state_codec)
            except (TypeError, ValueError) as error:
                raise RepoLoadError(f"Selected state is incompatible at {path!s}: {error}") from error
        candidates = {key: (overrides.get(key) or source_store,) if overrides.get(key) or source_store else stores
                      for key in selections}
        with ExitStack() as lifetime:
            lifetime.enter_context(state_use_leases((store, key) for key, stores in candidates.items() for store in stores))
            sources = {}
            for store in _unique_stores(s for stores in candidates.values() for s in stores):
                keys = [key for key, stores in candidates.items() if store in stores and key not in sources]
                if keys:
                    try:
                        from .store.authority import StoreCutChanged
                        for attempt in range(8):
                            try:
                                selected_sources = lifetime.enter_context(store.read_states(keys))
                            except StoreCutChanged:
                                if attempt == 7:
                                    raise
                            else:
                                sources.update(selected_sources)
                                break
                    except Exception as error:
                        raise RepoLoadError("Selected state preflight is incomplete: invalid source authority or payload.") from error
            missing = [(path, state.key) for path, state, _, _ in associations if state.key not in sources]
            if missing:
                details = "; ".join(f"{path!s} ({key.cdef_digest}/{key.codec_hash})" for path, key in missing)
                raise RepoLoadError("Selected state preflight is incomplete: unavailable state at " + details + ".")
            for path, state, _, _ in associations:
                source = sources.get(state.key)
                if source.state != state or source.manifest.codec_version != 1:
                    raise RepoLoadError(f"Selected state preflight is incomplete: incompatible source evidence at {path!s}.")
            yield _NodeStateSources(sources, stores)


def _node_inputs(node, *, classes=None, preparation=None):
    from .definition import ConcreteNode
    from .node import _owned_nodes
    from .cdef_codec import validate_cdef_stateful_role
    from .node_prepare import NodePreparation

    if not isinstance(node, ConcreteNode):
        raise TypeError("Repo.build requires a ConcreteNode; concretize the recipe first.")
    prepared = NodePreparation((node,)) if preparation is None else preparation
    prepared._slot(node)
    owned = tuple(_owned_nodes(node, strict=True))
    classes = _NodeBindings() if classes is None else classes
    identities = set()
    for path, current in owned:
        cls = classes.get(current)
        if cls is None:
            cls = _resolve_materialization_class(current)
        _admit_materialization_class(cls)
        validate_cdef_stateful_role(current, cls)
        if (current.object_id is not None or current.state is not None) and not current._stateful_role:
            raise ValueError("Stateless Nodes cannot select an ObjectId or private state.")
        if current.object_id is not None:
            if current.object_id in identities:
                raise ValueError("Independent Nodes cannot share an ObjectId.")
            identities.add(current.object_id)
        classes[current] = cls
    return prepared, owned, classes


def realize_node(repo, node, *, cache="weak", reuse_live="matching", exact=False, source_store=None, source_stores=None,
                 _memo=None, _sources=None, _hook_state=None, _effective=None, _inputs=None,
                 _preparation=None, _identity_reservation=None):
    """Realize a normalized Node graph with selected state exactly once.

    node is a ConcreteNode; exact requires saved completeness. cache selects
    weak/strong/none publication after success; reuse_live selects matching,
    greedy or never candidates. Missing IDs alone are allocated. Every selected
    association and current payload preflights before constructors/hooks. Returns
    the live root. Raises TypeError/ValueError for invalid inputs or RepoLoadError
    for unavailable state/construction/restore failure. Hooks run dependency-first
    under full-key leases, outside Store fences. A post-hook failure invalidates
    affected live nodes; no rollback is attempted or implied.
    """
    import inspect
    from dryml.runtime import materialization_admission
    from .node import require_saved
    from .repo import RepoLoadError
    from .repo_plan import apply_exact_reference_identity
    from .reference_values import ObjectRef
    from .state import construction_identity

    if cache not in {"weak", "strong", "none"} or reuse_live not in {"matching", "greedy", "never"}:
        raise ValueError("Invalid cache or reuse_live policy.")
    if exact:
        require_saved(node)
    with materialization_admission(operation="repo_node_materialization"), realization_scope():
        prepared, owned, classes = _node_inputs(node, preparation=_preparation) if _inputs is None else _inputs
        if _memo is not None and node in _memo:
            return _memo[node]
        original = _NodeBindings((current._node_id, current) for _, current in owned)
        runtime = _runtime_recipe(node)
        graph = ConcreteDefinitionGraph.from_root(runtime)
        selected = _NodeBindings() if _memo is None else _memo
        constrained = [current if _effective is None else _effective[current] for _, current in owned]
        if not exact and cache != "none" and all(current.object_id is None and current.state is None for current in constrained):
            candidate = repo.get_cached(runtime)
            if candidate is not None and _same_creation(candidate, node) and all(current not in selected or selected[current] is candidate.graph_at(path)
                                             for path, current in owned):
                for path, current in owned:
                    selected[current] = candidate.graph_at(path)
                _publish_cache(repo, candidate, cache)
                return candidate
        completed = []
        associations = [(path, current.state, current, classes[current]) for path, current in owned if current.state is not None]
        hooks_started = False
        phase, active_path = "construction", GraphPath()
        paths = {current._node_id: path for path, current in owned}
        publish_structural = not exact and all(current.state is None for current in constrained)
        from contextlib import nullcontext
        source_context = (_selected_state_sources(repo, associations, source_store=source_store, source_stores=source_stores)
                          if _sources is None else nullcontext(_sources))
        with source_context as sources, ExitStack() as reservations:
            lineage_facts = (_node_metadata_evidence(repo, node, owned, sources, prepared, source_store=source_store,
                source_stores=source_stores, reservation=_identity_reservation)
                if exact or any(current.object_id is not None for _, current in owned) else {})
            try:
                identity_reservation = _identity_reservation
                if exact:
                    from .state import reserve
                    identities = tuple(current.object_id for _, current in owned if current._stateful_role)
                    if identity_reservation is None:
                        identity_reservation = reservations.enter_context(reserve(node, (), identities))
                    else:
                        identity_reservation._covers_object_ids(identities)
                preferred = _NodeBindings()
                if exact and reuse_live != "never":
                    candidates = []
                    for candidate in repo._all_live_candidates(reservation=identity_reservation):
                        if not candidate.definition.graph_equal(runtime):
                            continue
                        if all(candidate.graph_at(path).object_id == source.object_id
                               and _same_creation(candidate.graph_at(path), source)
                               and (reuse_live == "greedy" or source.state is None
                                    or candidate.graph_at(path)._last_state_hash == source.state.key.codec_hash)
                               for path, source in owned):
                            candidates.append(candidate)
                    if len(candidates) == 1:
                        # Choose the whole live topology together. Stateless Method
                        # dependencies need not have independent cache entries.
                        preferred = _NodeBindings((source._node_id, candidates[0].graph_at(path))
                                                  for path, source in owned)
                for cdef in graph.topological_order(dependencies_first=True):
                    if cdef in selected:
                        continue
                    source = original.get(cdef) if _effective is None else _effective.get(cdef)
                    if source is None:
                        continue
                    obj = preferred.get(cdef)
                    active_path, phase = paths[source._node_id], "construction"
                    if obj is None and exact and reuse_live != "never" and source.object_id is not None:
                        candidates = [candidate for candidate in repo._all_live_candidates(reservation=identity_reservation)
                                      if candidate.object_id == source.object_id
                                      and candidate.definition.graph_equal(cdef)
                                      and _same_creation(candidate, source)
                                      and (reuse_live == "greedy" or candidate._last_state_hash == source.state.key.codec_hash)
                                      and all(candidate.graph_at(edge.path) is selected[edge.child]
                                              for edge in graph.outgoing(cdef) if edge.kind is EdgeKind.MATERIALIZE)]
                        if len(candidates) == 1:
                            obj = candidates[0]
                    elif obj is None and not exact and source.object_id is None and source.state is None and cache != "none":
                        obj = repo.get_cached(cdef)
                        if obj is not None and (not _same_creation(obj, source) or any(obj.graph_at(edge.path) is not selected[edge.child]
                                                   for edge in graph.outgoing(cdef) if edge.kind is EdgeKind.MATERIALIZE)):
                            obj = None
                    if obj is None:
                        cls = classes[source]
                        args, kwargs = project_cdef_call(source, cls=cls)
                        rt_args = from_canonical_local(args, repo=repo, resolve_cdef=lambda child: selected[child])
                        rt_kwargs = from_canonical_local(kwargs, repo=repo, resolve_cdef=lambda child: selected[child])
                        with construction_identity(source.object_id):
                            obj = cls(*rt_args, repo=repo, __cdef__=cdef, **rt_kwargs)
                        repo._num_constructions += 1
                        bound = inspect.signature(cls.__init__).bind(obj, *rt_args, **rt_kwargs)
                        bound.apply_defaults()
                        parameters = dict(bound.arguments)
                        parameters.pop("self", None)
                        selected[cdef] = obj
                        attach_runtime_binding(repo, cdef, obj, selected, parameters)
                        from .repo_plan import _record_runtime_values
                        from .utils.graph.path import Parameter
                        for name, value in source.parameters.items():
                            _record_runtime_values(value, parameters[name], GraphPath((Parameter(name),)), obj._runtime_projection)
                    selected[cdef] = obj
                    completed.append(obj)
                    # Retain construction selections, not newly captured states.
                    obj._construction_node = source
                    if source.object_id in lineage_facts:
                        from .snapshot_capture import install_lineage_fact
                        install_lineage_fact(obj, source.object_id, lineage_facts[source.object_id])
                    if source.state is not None:
                        from .state import reserve_node
                        if identity_reservation is None:
                            reservations.enter_context(reserve_node(obj))
                        else:
                            identity_reservation._covers_object_ids((obj.object_id,))
                        lock = obj._save_load_reservation
                        if not lock.acquire(blocking=False):
                            raise RepoLoadError("Live selected node is already reserved by save or restore.")
                        reservations.callback(lock.release)
                        hooks_started = True
                        phase = "restore"
                        if _hook_state is not None:
                            _hook_state[0] = True
                        _restore_state_source(obj, sources[source.state.key])
                        obj._last_state_hash = source.state.key.codec_hash
                    if publish_structural:
                        _publish_cache(repo, obj, cache)
                root = selected[runtime]
                identities = {path: selected[current].object_id for path, current in owned if current._stateful_role}
                for _, current in owned:
                    if current._stateful_role and current.object_id is None:
                        target = selected[current]
                        lineage_facts[target.object_id] = getattr(target, "_lineage_created_at", None)
                apply_exact_reference_identity(root, ObjectRef(runtime, identities), lineage_facts=lineage_facts)
                if not publish_structural:
                    for obj in completed:
                        _publish_cache(repo, obj, cache)
                if exact:
                    root._last_state_ref = node
                return root
            except BaseException as error:
                if hooks_started:
                    repo._invalidate_restoration_target(completed)
                if not isinstance(error, Exception):
                    raise
                operation = "Exact restore" if phase == "restore" else "Error constructing"
                recovery = "; affected live graph requires fresh materialization" if hooks_started else ""
                raise RepoLoadError(f"{operation} at {active_path!s} failed: {error}{recovery}") from error


def materialize_nodes(repo, roots, *, cache="weak", reuse_live="matching", declaration_stores=None):
    """Admit one Node signature boundary before any selected constructor or hook.

    roots is an ordered tuple of Nodes, live Objects or supported containers.
    Ref/quotation values remain terminal. Returns runtime values in the same
    container families with occurrence aliases retained across roots. All selected
    LocalStates preflight together under fresh full-key leases; conflicting state
    demands for one ObjectId raise RepoLoadError before effects. cache/reuse_live
    follow realize_node. Failures after a hook invalidate completed affected live
    graphs, including earlier roots. This owns no managed lifecycle policy.
    declaration_stores optionally maps selected exact Node digests to connected
    Store handles; pinned state/metadata sources never fall back to another replica.
    """
    from .canonical import node_kind, is_value_container_kind, rebuild_sequence_kind, rebuild_dict_kind
    from .node import Node
    from .repo import RepoLoadError
    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec

    normalization = {}
    nodes = {}

    def container(value, visit):
        if type(value) is frozenset:
            return frozenset(visit(child) for child in value)
        kind = node_kind(value)
        if isinstance(value, Mapping):
            return rebuild_dict_kind(kind, {key: visit(child) for key, child in value.items()})
        return rebuild_sequence_kind(kind, (visit(child) for child in value))

    def normalize(value):
        marker = id(value)
        if marker in normalization:
            return normalization[marker]
        if isinstance(value, Definition):
            result = value.concretize(repo=repo)
            nodes.setdefault(result._view, result)
        elif isinstance(value, ConcreteDefinition):
            from .node import _as_node_view
            result = _as_node_view(value)
            nodes.setdefault(result._view, result)
        elif isinstance(value, (Object, QuotedDef, SelectorSpec)):
            if isinstance(value, Object):
                from .repo import RepoSaveError
                try:
                    repo._state_graph_evidence(value)
                except RepoSaveError as error:
                    raise RepoLoadError("Live graph is unusable for materialization; use fresh materialization.") from error
            result = value
        elif isinstance(value, DefLink):
            result = value if value.kind is EdgeKind.REF else normalize(value.target)
        elif is_value_container_kind(node_kind(value)):
            result = container(value, normalize)
        else:
            result = value
        normalization[marker] = result
        return result

    normalized = tuple(normalize(root) for root in roots)
    declaration_stores = {} if declaration_stores is None else declaration_stores
    selected_stores = {}
    associations, demands = [], {}
    effective, paths, inputs = _NodeBindings(), {}, {}
    shared_classes = _NodeBindings()
    from .node import _semantic_equal
    for root in nodes.values():
        inputs[root._view] = _node_inputs(root, classes=shared_classes)
        _, owned, classes = inputs[root._view]
        provider = declaration_stores.get(root.exact_digest()) if declaration_stores else None
        if provider is not None:
            for _, current in owned:
                if current.state is not None:
                    previous = selected_stores.setdefault(current.state.key, provider)
                    if previous is not provider:
                        raise RepoLoadError("Boundary has conflicting selected StateKey source Stores.")
        for path, current in owned:
            previous = effective.get(current)
            paths.setdefault(current._node_id, path)
            if previous is not None:
                if not _semantic_equal(previous, current, topology=True, facets=False):
                    raise RepoLoadError("Boundary has incompatible effective construction views for one occurrence.")
                for left, right in ((previous.object_id, current.object_id), (previous.state, current.state)):
                    if left is not None and right is not None and left != right:
                        raise RepoLoadError("Boundary has incompatible effective identity/state demands for one occurrence.")
                current = previous.with_realization(
                    identities={GraphPath(): previous.object_id or current.object_id},
                    states={GraphPath(): previous.state or current.state})
            effective[current] = current
    for current in effective.values():
        if current.state is not None:
            if current.object_id is not None:
                previous = demands.setdefault(current.object_id, current.state)
                if previous != current.state:
                    raise RepoLoadError("Boundary has incompatible effective state demands for one ObjectId.")
            associations.append((paths[current._node_id], current.state, current, shared_classes[current]))
    for _, owned, _ in inputs.values():
        identities = [effective[current].object_id for _, current in owned if effective[current].object_id is not None]
        if len(set(identities)) != len(identities):
            raise RepoLoadError("Independent boundary Nodes cannot share an ObjectId.")
    memo = _NodeBindings()
    hook_state = [False]
    from .node import require_saved

    def realize(value):
        if isinstance(value, Node):
            try:
                require_saved(value)
                exact = True
            except ValueError:
                exact = False
            return realize_node(repo, value, cache=cache, reuse_live=reuse_live, exact=exact,
                                source_store=declaration_stores.get(value.exact_digest()) if declaration_stores else None,
                                source_stores=selected_stores,
                                _memo=memo, _sources=sources, _hook_state=hook_state,
                                _effective=effective, _inputs=inputs[value._view])
        if isinstance(value, (Object, QuotedDef, SelectorSpec)):
            return value
        if isinstance(value, DefLink):
            return value.target if value.kind is EdgeKind.REF else realize(value.target)
        if is_value_container_kind(node_kind(value)):
            return container(value, realize)
        return value

    with _selected_state_sources(repo, associations, source_stores=selected_stores) as sources:
        try:
            return tuple(realize(root) for root in normalized)
        except BaseException:
            # An earlier root may already have restored user state. There is no
            # transaction across hooks; retire the retained boundary on failure.
            if hook_state[0]:
                repo._invalidate_restoration_target(tuple(memo.values()))
            raise


class PreparedRestoration:
    """One-shot, owner-thread restoration handoff retaining sources and live ownership.

    Create with prepare_restoration. apply() returns the requested SavedNode for
    full restore, or None for partial apply; close() abandons unused preparation.
    Both release leases/reservations. Reuse, foreign-thread use and closed use
    raise RuntimeError. Preflight failure leaves the live graph usable; any failure
    after hook entry invalidates its entire retained graph. No rollback is promised.

    Args:
        repo: Repo owning connected source handles and live binding evidence.
        obj: Usable live root Object to restore without identity replacement.
        states: Optional rooted StateSpec or exact path/LocalState mapping.
        saved: Alternative complete SavedNode checked against the original live
            object projection; selections and lineage destinations are rebased to
            receiver occurrences, and construction pins are adopted after success.
        reservation: Optional active graph reservation covering this exact target.
        source_store, source_stores: Optional connected full-key source overrides.

    Raises:
        TypeError, ValueError: For malformed constraints or projection mismatch.
        RepoLoadError: For unavailable/incompatible state or unusable live targets.
        RepoSaveError: For conflicting, foreign or inactive graph ownership.
        MetadataConflictError: If selected capture/lineage evidence disagrees with
            another selected source or the supplied live target before hooks.

    Preparation retains leases and live reservations without running hooks. The
    factory prepare_restoration validates mutually exclusive partial/full inputs.
    """

    def __init__(self, repo, obj, states=None, *, saved=None, reservation=None, source_store=None, source_stores=None):
        import os
        from threading import get_ident
        from .node import _owned_nodes
        from .node_views import adopt_saved_view
        from .repo import RepoLoadError

        self._owner = os.getpid(), get_ident()
        self._closed = False
        self._lifetime = ExitStack()
        self.repo, self.obj, self.saved = repo, obj, saved
        self.plan = None
        try:
            if getattr(obj, "_restore_failed", False):
                raise RepoLoadError("Cannot restore an invalidated live graph; use fresh materialization.")
            self.plan, self.nodes, ids = repo._state_graph_evidence(obj)
            receiver = self.plan.node
            self.adopted = None if saved is None else adopt_saved_view(receiver, saved)
            if saved is None and states is not None:
                from .state_values import LocalState
                if not isinstance(states, Mapping) or any(not isinstance(state, LocalState) for state in states.values()):
                    raise TypeError("Partial apply requires a rooted mapping of exact LocalStates.")
                owned_occurrences = {current._occurrence for _, current in _owned_nodes(receiver)}
                for path, state in states.items():
                    destination = receiver.at(path)
                    if destination._occurrence not in owned_occurrences or not destination._stateful_role:
                        raise ValueError("Selected state is outside the live stateful ownership closure.")
            self.partial_view = receiver.with_realization(states=states) if saved is None else None
            selected = receiver.with_realization(states={p: None for p, _ in _owned_nodes(receiver)})
            # Unordered path ordinals address only one occurrence graph. Adoption
            # matches exact object projection before rebasing the supplied facets.
            selected = selected.with_realization(states=self.adopted.state_spec() if saved is not None else states)
            _, owned, classes = _node_inputs(selected)
            self.actions = [(path, current.state, obj.graph_at(path)) for path, current in owned if current.state is not None]
            if any(not isinstance(target, Serializable) for _, _, target in self.actions):
                raise RepoLoadError("Selected state has no retained stateful live destination.")
            if reservation is None:
                self._lifetime.enter_context(repo._reserve_state_graph_evidence(self.plan, self.nodes, ids))
            else:
                reservation._covers(self.nodes, ids)
            associations = [(path, current.state, current, classes[current]) for path, current in owned if current.state is not None]
            self.sources = self._lifetime.enter_context(_selected_state_sources(repo, associations, source_store=source_store, source_stores=source_stores))
            self._lineage_updates = []
            if saved is not None:
                metadata_prepared, metadata_owned, _ = _node_inputs(saved)
                facts = _node_metadata_evidence(repo, saved, metadata_owned, self.sources, metadata_prepared,
                                                source_store=source_store, source_stores=source_stores,
                                                reservation=reservation)
                from .repo import MetadataConflictError
                for path, current in _owned_nodes(self.adopted):
                    if current.object_id not in facts:
                        continue
                    target = obj.graph_at(path)
                    expected = facts[current.object_id]
                    if getattr(target, "_lineage_fact_object_id", None) == current.object_id:
                        if getattr(target, "_lineage_created_at", None) != expected:
                            raise MetadataConflictError("Target live identity disagrees with selected lineage evidence.")
                    else:
                        self._lineage_updates.append((target, current.object_id, expected))
            for target in sorted((t for _, _, t in self.actions), key=id):
                lock = target._save_load_reservation
                if not lock.acquire(blocking=False):
                    raise RepoLoadError("Live state is already reserved by save or restore.")
                self._lifetime.callback(lock.release)
        except BaseException:
            self.close()
            raise

    def _check(self):
        import os
        from threading import get_ident
        if self._closed or self._owner != (os.getpid(), get_ident()):
            raise RuntimeError("Prepared restoration is closed, consumed or belongs to another process/thread.")

    def __enter__(self):
        """Return this active preparation; reject consumed or foreign ownership."""
        self._check()
        return self

    def __exit__(self, *exc):
        """Release unused preparation without suppressing exceptions."""
        self.close()

    def close(self):
        """Release this owner's unused preparation idempotently; run no hooks."""
        if not self._closed:
            self._check()
            self._closed = True
            self._lifetime.close()

    def apply(self):
        """Consume prepared evidence once, restore dependency-first and release it.

        Returns the supplied SavedNode for full restoration, otherwise None.
        Raises RepoLoadError on hook failure with fresh-materialization guidance;
        the entire live graph is then invalidated. No historical receipt bypasses
        hooks. Source leases remain held until every hook and adoption completes.
        """
        from .repo import RepoLoadError
        from .repo_plan import validate_retained_save_plan
        from .node import _owned_nodes
        hooks_started = False
        self._check()
        try:
            validate_retained_save_plan(self.plan, self.obj)
            from .snapshot_capture import install_lineage_fact
            for target, identity, created_at in self._lineage_updates:
                install_lineage_fact(target, identity, created_at)
            order = {id(self.plan.binding.objects[current]): index
                     for index, current in enumerate(self.plan.graph.topological_order(dependencies_first=True))
                     if current in self.plan.binding.objects}
            for path, state, target in sorted(self.actions, key=lambda item: order[id(item[2])]):
                hooks_started = True
                _restore_state_source(target, self.sources[state.key])
                target._last_state_hash = state.key.codec_hash
            construction = self.adopted if self.adopted is not None else self.partial_view
            if construction is not None:
                for path, current in _owned_nodes(construction):
                    self.obj.graph_at(path)._construction_node = current
            if self.adopted is not None:
                self.obj._last_state_ref = self.saved
            return self.saved
        except BaseException as error:
            if hooks_started:
                self.repo._invalidate_restoration_target(self.nodes)
            if not isinstance(error, Exception):
                raise
            raise RepoLoadError("Targeted exact restore failed; affected live graph requires fresh materialization.") from error
        finally:
            self.close()


def _restore_state_source(obj, source):
    """Restore under native protection and let deferred consumers detach selection.

    The optional private binder must retain metadata only, never the borrowed
    source. Later payload access requires a fresh native reader of that selection.
    """
    obj.restore_state_from_dir(source.data_directory, codec=source.state.codec)
    bind = getattr(obj, "_bind_restored_state_source", None)
    if bind is not None:
        bind(source)


def prepare_restoration(repo, obj, states=None, *, saved=None, reservation=None, source_store=None, source_stores=None):
    """Prepare a one-shot partial/full restore without entering any state hook.

    repo/obj select the live ownership scope. states is a rooted LocalState map;
    saved instead requires the original projection-compatible SavedNode and adopts
    its construction pins on success. Optional reservation must already cover the
    graph. Source overrides map full StateKeys to connected Stores. Returns an
    owned PreparedRestoration that must be applied or closed. Validation failures
    propagate before hooks and release every borrowed resource.
    """
    if saved is not None and states is not None:
        raise ValueError("Select partial states or a full saved graph, not both.")
    return PreparedRestoration(repo, obj, states, saved=saved, reservation=reservation,
                               source_store=source_store, source_stores=source_stores)


def preflight_symbolic_materialization(*roots: Any, repo: Any = None) -> None:
    """Admit symbolic materializing structure before any Repo or target effect.

    Args:
        *roots: Raw Definition or container values selected for materialization.
        repo: Optional existing Repo used only for ordinary Ref authority
            selection while inspecting constructor signatures.

    Raises:
        UnresolvedDefinitionError: If active expressions remain in materializing
            structure after owning Template slots establish quotation barriers.
        ValueError: If a resolved Definition has no class authority.
        SignatureError: If an activated role rejects its selected authority.

    Side Effects:
        Resolves materializing class symbols solely to inspect their signatures.
        It never invokes constructors, factories, providers, CDef construction,
        cache mutation, claims, or Store writes.
    """

    from .bound_args import BoundArguments
    from .errors import UnresolvedDefinitionError
    from .links import DefLink
    from .params import Match
    from .quoted import QuotedDef, SelectorSpec
    from .signatures import compile_signature
    from .template import Expr
    from .utils.graph.value import iter_value_edges

    seen: set[int] = set()

    def visit_definition(definition: Definition) -> None:
        marker = id(definition)
        if marker in seen:
            return
        seen.add(marker)
        if definition.cls is None:
            raise ValueError("materializing Definition has no class authority")
        cls = resolve_symbol(definition.cls)
        if not isinstance(cls, type):
            raise TypeError("materializing Definition class authority must resolve to a type")
        plan = compile_signature(cls, constructor=True)
        bound = plan.signature.bind_partial(*(definition.args or ()), **definition.kwargs)
        boundary = plan.prepare_bound(
            BoundArguments(bound.arguments.items()), partial=True, repo=repo,
        )
        for name, value in boundary.authority.items():
            role = plan.slots[name].role
            if role in {"template", "ref"}:
                continue
            visit(value)

    def visit(value: Any) -> None:
        if isinstance(value, (QuotedDef, SelectorSpec, Object, Match)):
            return
        if isinstance(value, Expr):
            raise UnresolvedDefinitionError(
                "materializing Definition contains an active symbolic expression"
            )
        if isinstance(value, ConcreteDefinition):
            # Persisted CDefs were already role-normalized; their direct Ref
            # quotations are terminal, while materializing descendants remain live.
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            for edge in iter_value_edges(value):
                visit(edge.value)
            return
        if isinstance(value, Definition):
            visit_definition(value)
            return
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                visit(value.target)
            return
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)
        for edge in iter_value_edges(value):
            visit(edge.value)

    for root in roots:
        visit(root)


def _resolve_materialization_class(cdef: ConcreteDefinition) -> type:
    """Resolve one CDef class while explaining retired mixin authority."""

    try:
        return resolve_symbol(cdef.cls)
    except (ImportError, AttributeError) as error:
        if (
                isinstance(cdef.cls, ImportRef)
                and cdef.cls.module == "dryml.core.object"
                and cdef.cls.qualname in {"Metadata", "UniqueID"}):
            raise TypeError(
                f"ConcreteDefinition references retired {cdef.cls.qualname} mixin "
                "authority; reconstruct it with an ordinary Object class."
            ) from error
        raise


def preflight_materialization_classes(*roots) -> _NodeBindings:
    """Resolve and admit every class in a selected materializing closure.

    Args:
        *roots: CDefs, ObjectRefs, StateRefs, or nested values selected for live
            materialization.

    Returns:
        A private-node-keyed manifest retaining each resolved runtime class.

    Raises:
        _AbstractObjectAdmissionError: If any selected target is abstract.
        Exception: If a selected materializing class cannot be resolved.

    Ref links, quotations, and reference-as-value data remain terminal. The
    helper is deliberately separate from definition-only planning so inert CDef
    construction never resolves classes.
    """

    from .links import DefLink
    from .quoted import QuotedDef, SelectorSpec
    from .reference_values import ObjectRef, StateRef
    from .utils.graph.value import iter_value_edges

    manifest = _NodeBindings()

    def visit_cdef(cdef: ConcreteDefinition) -> None:
        if cdef in manifest:
            return
        try:
            cls = _resolve_materialization_class(cdef)
        except Exception as error:
            from .repo import RepoLoadError

            raise RepoLoadError("Materialization class preflight failed.") from error
        _admit_materialization_class(cls)
        manifest[cdef] = cls
        for edge in iter_value_edges(cdef):
            visit_value(edge.value)

    def visit_value(value) -> None:
        if isinstance(value, ConcreteDefinition):
            visit_cdef(value)
            return
        if isinstance(value, (ObjectRef, StateRef)):
            visit_cdef(value.definition)
            return
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                visit_value(value.target)
            return
        if isinstance(value, (QuotedDef, SelectorSpec)):
            return
        for edge in iter_value_edges(value):
            visit_value(edge.value)

    for root in roots:
        visit_value(root)
    return manifest


@dataclass(frozen=True, slots=True)
class MaterializationAction:
    """Definition-only recipe for one runtime materialization step.

    Reuse records only its source; execution retrieves any live Object after
    admission instead of retaining one in the plan.
    """

    definition: ConcreteDefinition
    kind: MaterializationActionKind
    primary_path: str
    reuse_source: MaterializationReuseSource = None
    cache: CachePolicy = "weak"


@dataclass(slots=True)
class MaterializationPlan:
    """Definition-only ordered materialization graph and per-node actions."""

    graph: ConcreteDefinitionGraph
    actions: _NodeBindings
    order: tuple[ConcreteDefinition, ...]
    cache: CachePolicy


def build_materialization_plan(
        repo,
        cdef: ConcreteDefinition,
        *,
        cache: CachePolicy = "weak",
        require_store: bool = False,
        memo: dict | None = None,
        path: list[str | int] | None = None) -> MaterializationPlan:
    """Build a definition-only plan without acquiring live cached Objects.

    Plan construction remains available during strict orchestration. It may
    inspect cache and Store availability metadata but never resolves classes,
    restores state, or retains a live Object in the returned plan.
    """

    if memo is None:
        memo = {}
    graph = ConcreteDefinitionGraph.from_root(cdef)
    if require_store and repo._first_store_with(cdef) is None:
        from .repo import RepoLoadError

        raise RepoLoadError("No connected Store contains this structural CDef; use load_or_build to construct it.")
    included = _included_nodes(repo, graph, cdef, memo)
    order = tuple(
        node for node in graph.topological_order(dependencies_first=True)
        if cdef_node_key(node) in included
    )
    primary_paths = _primary_paths(graph)
    root_path = _format_error_path(path)
    actions = _NodeBindings()
    for node in order:
        memo_reuse = memo.get(cdef_node_key(node), memo.get(node)) is not None
        cache_reuse = repo.has_cached(node)
        reuse_source: MaterializationReuseSource = (
            "memo" if memo_reuse else ("cache" if cache_reuse else None)
        )
        kind: MaterializationActionKind = "reuse" if reuse_source is not None else "construct"
        actions[node] = MaterializationAction(
            definition=node,
            kind=kind,
            primary_path=root_path if cdef_node_key(node) is cdef_node_key(cdef) else str(primary_paths.get(node, "<unknown>")),
            reuse_source=reuse_source,
            cache=cache,
        )
    return MaterializationPlan(graph=graph, actions=actions, order=order, cache=cache)


def execute_materialization_plan(
        repo,
        plan: MaterializationPlan,
        *,
        memo: dict,
        root: ConcreteDefinition,
        class_manifest: _NodeBindings | None = None):
    from dryml.runtime import materialization_admission

    with materialization_admission(operation="execute_materialization_plan"):
        if class_manifest is None:
            class_manifest = preflight_materialization_classes(root)
        with realization_scope():
            return _execute_materialization_plan(
                repo, plan, memo=memo, root=root, class_manifest=class_manifest,
            )


def _execute_materialization_plan(
        repo,
        plan: MaterializationPlan,
        *,
        memo: dict,
        root: ConcreteDefinition,
        class_manifest: _NodeBindings):
    """Execute an already admitted plan without resolving classes beforehand."""

    from .repo import RepoLoadError

    local_memo = _NodeBindings()
    for key, obj in memo.items():
        local_memo[key] = obj
    for cdef in plan.order:
        if cdef in local_memo:
            continue

        action = plan.actions[cdef]
        if action.kind == "reuse":
            obj = local_memo.get(cdef) if action.reuse_source == "memo" else repo.get_cached(
                cdef
            )
            if obj is None:
                if action.reuse_source == "cache":
                    refreshed = build_materialization_plan(
                        repo,
                        root,
                        cache=plan.cache,
                        memo=local_memo,
                    )
                    return _execute_materialization_plan(
                        repo,
                        refreshed,
                        memo=memo,
                        root=root,
                        class_manifest=class_manifest,
                    )
                source = "memoized" if action.reuse_source == "memo" else "cached"
                raise RepoLoadError(
                    f"Materialization plan requested {source} reuse for {cdef} at {action.primary_path}, "
                    "but no reusable object is available."
                )
            local_memo[cdef] = obj
            memo[cdef_node_key(cdef)] = obj
            continue

        if action.kind != "construct":
            raise RepoLoadError(f"Unknown materialization action kind {action.kind!r} at {action.primary_path}.")

        try:
            cls = class_manifest[cdef]
        except KeyError as error:
            raise RepoLoadError("Materialization class preflight is incomplete.") from error

        from .cdef_codec import CDefGraphCodecError, validate_cdef_stateful_role

        try:
            validate_cdef_stateful_role(cdef, cls)
        except CDefGraphCodecError as error:
            raise RepoLoadError(f"Incompatible definition authority at {action.primary_path}: {error}") from error
        canonical_args, canonical_kwargs = project_cdef_call(cdef, cls=cls)
        rt_args = from_canonical_local(canonical_args, resolve_cdef=lambda child: local_memo[child], repo=repo)
        rt_kwargs = from_canonical_local(canonical_kwargs, resolve_cdef=lambda child: local_memo[child], repo=repo)

        try:
            obj = cls(*rt_args, repo=repo, __cdef__=cdef, **rt_kwargs)
            repo._num_constructions += 1
        except _AbstractObjectAdmissionError:
            raise
        except Exception as e:
            cls_name = getattr(cdef.cls, "__name__", repr(cdef.cls))
            raise RepoLoadError(f"Error constructing {cls_name} at {action.primary_path}: {e}") from e

        local_memo[cdef] = obj
        import inspect

        bound = inspect.signature(cls.__init__).bind(obj, *rt_args, **rt_kwargs)
        parameters = dict(bound.arguments)
        parameters.pop("self", None)
        attach_runtime_binding(repo, cdef, obj, local_memo, parameters)
        memo[cdef_node_key(cdef)] = obj
        _publish_cache(repo, obj, action.cache)

    return local_memo[root]


def from_canonical_local(value: Any, *, resolve_cdef, repo, resolve_reference=None):
    """Decode constructor values using already-selected graph dependencies.

    Args:
        value: Canonical constructor value to decode.
        resolve_cdef: Callback returning selected direct CDef dependencies.
        repo: Repo owning the realization.
        resolve_reference: Optional exact StateRef resolver for materializing
            reference leaves.

    Returns:
        Runtime constructor data retaining selected Object identity.
    """
    return from_canonical(
        value, repo=repo, resolve_cdef=resolve_cdef,
        resolve_reference=resolve_reference,
    )


@dataclass(frozen=True, slots=True)
class NodeStateSelection:
    """Detached selected association; path/state/store grant no payload validity.

    path is a primary GraphPath, state a LocalState and store the selected connected
    Store. This metadata carries no reader, bytes, lease or live Object. A later
    consumer must acquire native leases and freshly validate the full StateKey.
    """

    path: GraphPath
    state: Any
    store: Any


@dataclass(frozen=True, slots=True)
class NodeLoadPlan:
    """SavedNode metadata and pure preparation for a future exact realization.

    state_ref is the original SavedNode; actions retain detached full-key source
    choices. preparation owns immutable graph facts only. Current availability,
    class admission and active-use protection are revalidated by execution; a plan
    cannot authorize skipping restoration. source_store/source_stores retain the
    explicit caller source scope, not a borrowed payload lifetime.
    """

    state_ref: Any
    actions: tuple[NodeStateSelection, ...]
    preparation: Any
    source_store: Any = None
    source_stores: Any = None


def build_exact_state_load_plan(
        repo, state_ref, *, source_store=None, source_stores=None,
        _defer_payload: bool = False, _read_cut=None) -> NodeLoadPlan:
    """Preflight complete SavedNode state without constructing or restoring Objects.

    Node inputs return a NodeLoadPlan retaining pure preparation and detached
    selected associations. Fresh full-key leases cover this preflight only;
    execute_exact_state_load_plan revalidates current sources and class admission.
    source_stores maps StateKeys to connected Stores for Node inputs. Legacy
    snapshot read cuts are rejected for Nodes, and no borrowed payload reader is
    retained in their detached plan. Incomplete metadata raises TypeError/ValueError
    and missing/incompatible selected state raises RepoLoadError before hooks.

    Args:
        repo (Repo): Repo containing every Store required by the exact closure.
        state_ref (SavedNode): Requested complete immutable Node graph.
        source_store (Store or None): Connected selected provider.
        source_stores (Mapping[StateKey, Store] or None): Exact source overrides.
        _defer_payload (bool): Retired private option; cannot weaken fresh validation.
        _read_cut: Retired private snapshot cut; any supplied value is rejected.

    Returns:
        Detached NodeLoadPlan with full-key source choices and pure preparation,
        without borrowed payload paths or an availability proof for future use.

    Raises:
        TypeError: If the reference or selector mapping/handle types are invalid.
        ValueError: If a selector is disconnected, inconsistent, or outside the
            materializing exact closure.
        RepoLoadError: If selected creation/manifest/capture or payload is unavailable.
        MetadataConflictError: If unqualified source evidence or a
            cached exact identity disagrees about persisted metadata.
        RuntimeError: If supplied evidence is expired, cleared, foreign-owned,
            or the Repo is closing, closed, or inherited from another process.

    Side Effects:
        Acquires full-key source leases only during preflight. No constructors or
        restore hooks run, and no evidence authorizes a later read without fresh
        validation. Current reader overrides remain authoritative. Old snapshots
        are rejected rather than converted or sent through a parallel loader.
    """
    from .node import require_saved
    from types import MappingProxyType
    if _read_cut is not None:
        raise ValueError("Node state preflight owns fresh full-key readers, not legacy snapshot read cuts.")
    require_saved(state_ref)
    prepared, owned, classes = _node_inputs(state_ref)
    associations = [(path, node.state, node, classes[node]) for path, node in owned if node.state is not None]
    with _selected_state_sources(repo, associations, source_store=source_store, source_stores=source_stores) as sources:
        _node_metadata_evidence(repo, state_ref, owned, sources, prepared, source_store=source_store, source_stores=source_stores)
        actions = tuple(NodeStateSelection(path, state, sources[state.key].store) for path, state, _, _ in associations)
    return NodeLoadPlan(state_ref, actions, prepared, source_store,
                        None if source_stores is None else MappingProxyType(dict(source_stores)))


def execute_exact_state_load_plan(
        repo,
        plan: NodeLoadPlan,
        *,
        reuse_live: LiveReusePolicy,
        cache: CachePolicy = "weak",
        _identity_reservation=None):
    """Revalidate and realize a detached NodeLoadPlan without partial cache publication.

    Args:
        repo: Repo owning live candidates and the completed result cache.
        plan: Detached NodeLoadPlan; legacy snapshot plans are incompatible.
        reuse_live: Exact candidate reuse policy.
        cache: Cache tier used only after every node has completed successfully.
        _identity_reservation: Optional internal token already owning every
            ObjectId in this exact load closure.

    Returns:
        A live root matching ``plan.state_ref``.

    Raises:
        RepoLoadError: If construction, reuse validation, or restoration fails.

    Side Effects:
        May restore or cache live Objects. The outer exact-load call installs its
        requested SavedNode receipt on the returned root after completion; nested
        materializing references do not install descendant receipts.
    """
    if not isinstance(plan, NodeLoadPlan):
        raise TypeError("Exact execution requires a NodeLoadPlan; legacy snapshot plans are incompatible.")
    return realize_node(repo, plan.state_ref, exact=True, reuse_live=reuse_live,
                        cache=cache, source_store=plan.source_store,
                        source_stores=plan.source_stores, _preparation=plan.preparation,
                        _identity_reservation=_identity_reservation)


def project_cdef_call(
        cdef: ConcreteDefinition,
        *,
        cls: type | None = None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Project a V2 identity onto its runtime constructor call surface.

    Args:
        cdef: Exact V2 identity to invoke.
        cls: Optional already-resolved current runtime class.

    Returns:
        Canonical positional and keyword values suitable for runtime decoding.

    Raises:
        TypeError: If the record is incompatible with the current class
            signature.
        Exception: If resolving a V2 class fails.

    The stored semantic record is projected through the current class signature
    without invoking preparation or applying defaults.
    """

    from dryml.runtime import materialization_admission

    with materialization_admission(operation="project_cdef_constructor_call"):
        if cls is None:
            cls = _resolve_materialization_class(cdef)
        from .bound_args import project_bound_arguments

        return project_bound_arguments(cls, cdef._bound_args)


def _included_nodes(repo, graph: ConcreteDefinitionGraph, root: ConcreteDefinition, memo: dict) -> set[ConcreteDefinition]:
    included: set[object] = set()

    def visit(cdef: ConcreteDefinition) -> None:
        key = cdef_node_key(cdef)
        if key in included:
            return
        included.add(key)
        if cdef_node_key(cdef) in memo or cdef in memo:
            return
        if repo.has_cached(cdef):
            return
        for edge in graph.outgoing(cdef):
            if edge.kind is EdgeKind.MATERIALIZE:
                visit(edge.child)

    visit(root)
    return included


def _primary_paths(graph: ConcreteDefinitionGraph) -> _NodeBindings:
    paths = _NodeBindings()
    for root in graph.roots:
        paths[root] = "$"
    for occ in graph.iter_occurrences():
        if occ.definition not in paths:
            paths[occ.definition] = str(occ.path)
    return paths


def _format_error_path(path: list[str | int] | None) -> str:
    if not path:
        return "<root>"
    return "/".join(map(str, path))


def _publish_cache(repo, obj: Object, cache: CachePolicy) -> None:
    if cache == "strong":
        repo.cache_strong(obj)
    elif cache == "weak":
        repo.cache_weak(obj)
    elif cache == "none":
        return
    else:
        raise ValueError(f"Unknown cache policy: {cache!r}")
