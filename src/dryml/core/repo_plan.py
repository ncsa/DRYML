from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import copy
import hashlib
import os
import re
import shutil
from types import MappingProxyType
from threading import Lock
import uuid
from typing import Any, Callable, Generic, Iterable, Iterator, TypeVar

from .canonical import NodeKind, is_runtime_leaf, node_kind
from .cdef_graph import ConcreteDefinitionGraph, EdgeKind
from .definition import ConcreteDefinition
from .object import Object, Serializable
from .policies import RepoGraphOptions
from .utils.graph.path import GraphPath, graph_path_sort_key
from .utils.graph.value import iter_value_edges
from .utils.graph.path import Parameter
from .cdef_identity import cdef_node_key


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RealizationScope:
    """Opaque identifier and ephemeral labels for one live realization."""

    token: str
    workspace_label: str = field(
        default_factory=lambda: f"scope-{uuid.uuid4().hex}",
        repr=False,
    )
    _workspace_node_labels: dict[object, str] = field(
        default_factory=dict,
        repr=False,
        compare=False,
        hash=False,
    )
    _workspace_lock: Lock = field(
        default_factory=Lock,
        repr=False,
        compare=False,
        hash=False,
    )

    def workspace_node_label(self, node_key: object) -> str:
        """Return one token-neutral label for a private node in this scope."""

        with self._workspace_lock:
            label = self._workspace_node_labels.get(node_key)
            if label is None:
                label = f"node-{uuid.uuid4().hex}"
                self._workspace_node_labels[node_key] = label
            return label


_CURRENT_REALIZATION_SCOPE: ContextVar[RealizationScope | None] = ContextVar(
    "dryml_realization_scope", default=None
)


@contextmanager
def realization_scope() -> Iterator[RealizationScope]:
    """Install a new scope unless a nested realization already owns one."""

    existing = _CURRENT_REALIZATION_SCOPE.get()
    if existing is not None:
        yield existing
        return
    scope = RealizationScope(uuid.uuid4().hex)
    token = _CURRENT_REALIZATION_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_REALIZATION_SCOPE.reset(token)


def current_realization_scope() -> RealizationScope | None:
    """Return the active scope without creating a new realization."""

    return _CURRENT_REALIZATION_SCOPE.get()


class _NodeBindings(dict):
    """Map CDefs by private node token rather than structural equality."""

    @staticmethod
    def _key(key):
        return cdef_node_key(key) if isinstance(key, ConcreteDefinition) else key

    def __contains__(self, key):
        return super().__contains__(self._key(key))

    def __getitem__(self, key):
        return super().__getitem__(self._key(key))

    def get(self, key, default=None):
        return super().get(self._key(key), default)

    def __setitem__(self, key, value):
        super().__setitem__(self._key(key), value)


class RuntimeBindingConflict(KeyError):
    def __init__(self, *, definition: ConcreteDefinition, first: Object, second: Object, path: GraphPath):
        self.definition = definition
        self.first = first
        self.second = second
        self.path = path
        super().__init__(
            f"Repo already has a different object matching {definition} at {path.legacy_str()}!"
        )


@dataclass(frozen=True, slots=True)
class RuntimeRoot:
    definition: ConcreteDefinition
    path: GraphPath
    obj: Object | None = None


@dataclass(slots=True)
class RuntimeGraphBinding:
    graph: ConcreteDefinitionGraph
    roots: tuple[RuntimeRoot, ...]
    objects: _NodeBindings
    missing: frozenset[object] = frozenset()
    scope: RealizationScope | None = None


@dataclass(frozen=True, slots=True)
class GraphObjectOccurrence:
    path: GraphPath
    definition: ConcreteDefinition
    obj: Object


@dataclass(frozen=True, slots=True)
class GraphApplyResult(Generic[T]):
    path: GraphPath
    definition: ConcreteDefinition
    value: T


@dataclass(frozen=True, slots=True)
class SaveAction:
    """One owned Serializable node selected for graph-state publication."""

    path: GraphPath
    definition: ConcreteDefinition
    obj: Object
    state_hash: str | None = None
    source_store: Any = None


@dataclass(slots=True)
class SavePlan:
    """Preflighted live graph evidence used to publish one exact StateRef."""

    graph: ConcreteDefinitionGraph
    binding: RuntimeGraphBinding
    actions: tuple[SaveAction, ...]


@dataclass(frozen=True, slots=True)
class StoreReport:
    """Ephemeral Store selections required to resolve one saved StateRef.

    Attributes:
        target_store: Store containing the enclosing StateRef record.
        state_stores: One selected Store for each canonical StateRef path.
        required_stores: Deduplicated complete Store set needed for resolution.
    """

    target_store: Any
    state_stores: MappingProxyType
    required_stores: tuple[Any, ...]

    def __init__(self, target_store: Any, state_stores: dict[GraphPath, Any], required_stores: Iterable[Any]):
        object.__setattr__(self, "target_store", target_store)
        object.__setattr__(self, "state_stores", MappingProxyType(dict(state_stores)))
        object.__setattr__(self, "required_stores", tuple(_unique_stores(required_stores)))


def collect_runtime_roots(value: Any) -> tuple[RuntimeRoot, ...]:
    roots: list[RuntimeRoot] = []
    _collect_runtime_roots(value, GraphPath(), roots)
    return tuple(roots)


def build_runtime_binding(
        repo,
        value: Any) -> RuntimeGraphBinding:
    roots = collect_runtime_roots(value)
    graph = ConcreteDefinitionGraph.from_roots(root.definition for root in roots)
    materialize_nodes = _materialize_reachable_nodes(graph, roots)
    objects = _NodeBindings()
    for root in roots:
        if root.obj is not None:
            bind_runtime_object(objects, root.definition, root.obj, path=root.path)
    for node in graph.nodes():
        if node.definition not in materialize_nodes:
            continue
        if node.definition in objects:
            continue
        for root in roots:
            if root.obj is None:
                continue
            relative = graph.primary_path(root.definition, node.definition)
            if relative is None:
                continue
            # Bindings belong to the realized Object, so they are rooted at
            # that receiver rather than any outer input container occurrence.
            bound = getattr(root.obj, "_runtime_bindings", {}).get(relative)
            if isinstance(bound, Object):
                bind_runtime_object(
                    objects, node.definition, bound,
                    path=root.path.join(relative),
                )
                break
        if node.definition in objects:
            continue
        obj = _cached_object(repo, node.definition)
        if obj is not None:
            path = _node_primary_path(graph, roots, node.definition)
            bind_runtime_object(objects, node.definition, obj, path=path)
    missing = frozenset(
        cdef_node_key(cdef) for cdef in materialize_nodes if cdef not in objects
    )
    return RuntimeGraphBinding(
        graph=graph,
        roots=roots,
        objects=objects,
        missing=missing,
        scope=current_realization_scope(),
    )


def bind_runtime_object(
        bindings: _NodeBindings,
        cdef: ConcreteDefinition,
        obj: Object,
        *,
        path: GraphPath) -> None:
    existing = bindings.get(cdef)
    if existing is None:
        bindings[cdef] = obj
        return
    if existing is not obj:
        raise RuntimeBindingConflict(definition=cdef, first=existing, second=obj, path=path)


def attach_runtime_binding(
        repo, cdef: ConcreteDefinition, obj: Object, memo,
        runtime_parameters: dict[str, Any] | None = None) -> None:
    """Attach completed realization evidence to a successfully built Object.

    Args:
        repo: Repo that owns the completed realization.
        cdef: Exact private-node CDef for ``obj``.
        obj: Successfully initialized live Object.
        memo: Current private-node materialization memo containing dependencies.
        runtime_parameters: Optional bound runtime values supplied to the
            constructor. They are authoritative for materializing reference
            leaves and avoid decoding an exact StateRef a second time.

    Side Effects:
        Stores immutable construction bindings, defensive runtime projections,
        scope, ObjectId/ObjectRef metadata, and the current Store affinity on
        ``obj``. No user attributes are inspected.
    """

    from .canonical import from_canonical
    from .reference_values import ObjectId, ObjectRef

    bindings = _NodeBindings()
    bindings[cdef] = obj
    graph = ConcreteDefinitionGraph.from_root(cdef)
    for occurrence in graph.iter_occurrences(include_roots=True):
        candidate = memo.get(occurrence.definition)
        if candidate is not None:
            bindings[occurrence.definition] = candidate

    runtime_values: dict[GraphPath, Any] = {GraphPath(): obj}
    projection = runtime_parameters or from_canonical(
        cdef.parameters,
        repo=repo,
        resolve_cdef=lambda child: bindings[child],
    )
    for name, canonical_value in cdef.parameters.items():
        runtime_value = projection.get(name)
        if name not in projection:
            runtime_value = from_canonical(
                canonical_value, repo=repo,
                resolve_cdef=lambda child: bindings[child],
            )
        _record_runtime_values(
            canonical_value, runtime_value,
            GraphPath((Parameter(name),)),
            runtime_values,
        )
    for occurrence in graph.iter_occurrences(include_roots=True):
        bound = bindings.get(occurrence.definition)
        if bound is not None:
            runtime_values[occurrence.path] = bound
    _rebase_imported_runtime_projections(cdef, obj, GraphPath(), runtime_values)

    object_ids = {}
    for node in graph.nodes():
        if not getattr(node.definition, "_stateful_role", False):
            continue
        bound = bindings.get(node.definition)
        if bound is None:
            continue
        object_id = getattr(bound, "_object_id", None)
        if object_id is None:
            object_id = ObjectId()
            bound._object_id = object_id
        path = GraphPath() if node.definition is cdef else graph.primary_path(cdef, node.definition)
        object_ids[path] = object_id

    for name, value in cdef.parameters.items():
        _collect_imported_object_ids(value, GraphPath((Parameter(name),)), object_ids)

    obj._realization_scope = current_realization_scope()
    obj._runtime_bindings = runtime_values
    obj._runtime_projection = runtime_values
    obj._store_affinity = repo.obj_default_store.get(cdef)
    obj._last_state_hash = getattr(obj, "_last_state_hash", None)
    obj._object_id = getattr(obj, "_object_id", None) if isinstance(obj, Serializable) else None
    obj._object_ref = ObjectRef(cdef, object_ids)


def _collect_imported_object_ids(value: Any, path: GraphPath, out: dict) -> None:
    """Expand immutable materializing reference IDs under an outer occurrence.

    Repeated materializing exact references retain one ObjectId entry at the
    minimum canonical path, matching ObjectRef's alias representation.
    """

    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import ObjectRef, StateRef

    if isinstance(value, StateRef):
        value = value.object
    if isinstance(value, ObjectRef):
        for child_path, object_id in value.objects.items():
            candidate_path = path.join(child_path)
            existing_path = next(
                (known_path for known_path, known_id in out.items()
                 if known_id == object_id),
                None,
            )
            if existing_path is None:
                out[candidate_path] = object_id
            elif graph_path_sort_key(candidate_path) < graph_path_sort_key(existing_path):
                del out[existing_path]
                out[candidate_path] = object_id
        return
    if isinstance(value, DefLink):
        if value.kind is EdgeKind.MATERIALIZE:
            _collect_imported_object_ids(value.target, path, out)
        return
    for edge in iter_value_edges(value):
        _collect_imported_object_ids(edge.value, path.child(edge.segment), out)


def apply_exact_reference_identity(obj: Object, reference) -> None:
    """Rebind a completed materialized subtree to supplied exact ObjectIds.

    Args:
        obj: Freshly materialized root for ``reference.definition``.
        reference: ObjectRef whose topology and ObjectIds are authoritative.

    Raises:
        ValueError: If a reference path does not resolve to the expected live
            Object or the supplied topology cannot be retained.

    Side Effects:
        Replaces IDs and subtree ObjectRefs only after construction has
        completed. State restoration remains the later exact-load boundary.
    """

    from .reference_values import ObjectRef

    if not isinstance(reference, ObjectRef):
        raise TypeError("Exact runtime materialization requires an ObjectRef.")
    if not obj.definition.graph_equal(reference.definition):
        raise ValueError("Materialized exact reference topology does not match its ObjectRef.")
    for path, object_id in reference.objects.items():
        bound = obj.graph_at(path)
        if not isinstance(bound, Object):
            raise ValueError(f"Exact ObjectRef path {path!s} did not resolve to an Object.")
        bound._object_id = object_id
        try:
            bound._object_ref = reference.at(path)
        except ValueError:
            # An alias occurrence still names the same completed Object; its
            # primary ObjectRef remains attached by the corresponding path.
            pass
    obj._object_ref = reference
    obj._object_id = reference.object_id


def _record_runtime_values(canonical: Any, runtime: Any, path: GraphPath, out: dict[GraphPath, Any]) -> None:
    """Record runtime-form values without traversing into materialized Objects."""

    from .links import DefLink
    from .cdef_graph import EdgeKind

    out[path] = _copy_runtime_value(runtime)
    if isinstance(canonical, ConcreteDefinition):
        return
    if isinstance(canonical, DefLink):
        if canonical.kind is EdgeKind.REF:
            out[path] = canonical.target
        return
    canonical_edges = tuple(iter_value_edges(canonical))
    runtime_edges = {edge.segment: edge.value for edge in iter_value_edges(runtime)} if canonical_edges else {}
    for edge in canonical_edges:
        if edge.segment in runtime_edges:
            _record_runtime_values(edge.value, runtime_edges[edge.segment], path.child(edge.segment), out)


def _rebase_imported_runtime_projections(
        canonical: Any, runtime: Any, path: GraphPath, out: dict[GraphPath, Any]) -> None:
    """Expose materialized exact-reference bindings below their outer occurrence.

    Bare, materializing ``Mat(ObjectRef)``, and ``StateRef`` values construct a
    live subtree. Its private runtime projection remains authoritative, but its
    paths must also be visible from the enclosing root for exact graph saves.
    ``Ref`` links deliberately retain their immutable reference value and never
    contribute owned runtime bindings.
    """

    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import ObjectRef, StateRef

    if isinstance(canonical, (ObjectRef, StateRef)):
        if not isinstance(runtime, Object):
            raise ValueError(f"Materialized exact reference at {path!s} did not produce an Object.")
        for inner_path, value in runtime._runtime_projection.items():
            target = path.join(inner_path)
            previous = out.get(target)
            if previous is not None and previous is not value:
                raise ValueError(f"Conflicting exact runtime binding at {target!s}.")
            out[target] = value
        return
    if isinstance(canonical, ConcreteDefinition):
        for edge in iter_value_edges(canonical):
            child_path = path.child(edge.segment)
            runtime_child = out.get(child_path)
            if runtime_child is not None:
                _rebase_imported_runtime_projections(
                    edge.value, runtime_child, child_path, out
                )
        return
    if isinstance(canonical, DefLink):
        if canonical.kind is EdgeKind.MATERIALIZE:
            _rebase_imported_runtime_projections(canonical.target, runtime, path, out)
        return
    canonical_edges = tuple(iter_value_edges(canonical))
    runtime_edges = {edge.segment: edge.value for edge in iter_value_edges(runtime)} if canonical_edges else {}
    for edge in canonical_edges:
        if edge.segment in runtime_edges:
            _rebase_imported_runtime_projections(edge.value, runtime_edges[edge.segment], path.child(edge.segment), out)


def _copy_runtime_value(value: Any) -> Any:
    """Copy mutable runtime data while retaining Object and exact-reference identity."""

    if isinstance(value, Object):
        return value
    if isinstance(value, dict):
        return {key: _copy_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_runtime_value(item) for item in value)
    if isinstance(value, set):
        return {_copy_runtime_value(item) for item in value}
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def iter_graph_objects(repo, root: Any, options: RepoGraphOptions) -> Iterator[Object]:
    _validate_graph_options(options)
    binding = build_runtime_binding(repo, root)
    for occurrence in _iter_bound_graph_object_occurrences(repo, binding, options):
        yield occurrence.obj


def _iter_bound_graph_objects(
        repo,
        binding: RuntimeGraphBinding,
        options: RepoGraphOptions) -> tuple[Object, ...]:
    return tuple(
        occurrence.obj
        for occurrence in _iter_bound_graph_object_occurrences(
            repo,
            binding,
            options,
        )
    )


def _iter_bound_graph_object_occurrences(
        repo,
        binding: RuntimeGraphBinding,
        options: RepoGraphOptions) -> tuple[GraphObjectOccurrence, ...]:
    out: list[GraphObjectOccurrence] = []
    load_memo = _NodeBindings()
    seen: set[object] = set()

    def visit(cdef: ConcreteDefinition, path: GraphPath, explicit_obj: Object | None = None) -> None:
        obj = explicit_obj if explicit_obj is not None else binding.objects.get(cdef)
        if obj is None:
            obj = _resolve_missing_for_traversal(repo, cdef, path, options, load_memo)
            if obj is None:
                return
            bind_runtime_object(binding.objects, cdef, obj, path=path)

        if options.dedupe:
            if cdef_node_key(cdef) in seen:
                return
            seen.add(cdef_node_key(cdef))

        should_apply = options.include_root or bool(path)
        if options.order == "pre" and should_apply:
            out.append(GraphObjectOccurrence(path, cdef, obj))

        for edge in binding.graph.outgoing(cdef):
            if edge.kind is not EdgeKind.MATERIALIZE:
                continue
            visit(edge.child, path.join(edge.path))

        if options.order == "post" and should_apply:
            out.append(GraphObjectOccurrence(path, cdef, obj))

    for root_occ in binding.roots:
        visit(root_occ.definition, root_occ.path, root_occ.obj)
    return tuple(out)


def apply_graph_objects(
        repo,
        root: Any,
        func: Callable[[Object], T],
        options: RepoGraphOptions) -> dict[ConcreteDefinition, T] | tuple[GraphApplyResult[T], ...]:
    _validate_graph_options(options)
    binding = build_runtime_binding(repo, root)
    occurrences = _iter_bound_graph_object_occurrences(repo, binding, options)
    if not options.dedupe:
        return tuple(
            GraphApplyResult(occ.path, occ.definition, func(occ.obj))
            for occ in occurrences
        )
    results: dict[ConcreteDefinition, T] = {}
    for occ in occurrences:
        results[occ.definition] = func(occ.obj)
    return results


def add_objects(repo, values: Iterable[Any], *, store=None) -> None:
    for value in values:
        binding = build_runtime_binding(repo, value)
        for obj in _iter_bound_graph_objects(
                repo,
                binding,
                RepoGraphOptions(include_root=True, order="post", missing="raise", dedupe=False)):
            _add_object_single(repo, obj, store=store)
        for missing in binding.missing:
            if _cached_object(repo, missing) is None:
                raise KeyError(f"No object linked to definition {missing} found in repo!")


def build_save_plan(
        repo,
        value: Object) -> SavePlan:
    """Build complete owned-state save evidence from retained U3 bindings.

    Args:
        repo: Repository owning live bindings.
        value: Live root Object to publish.
    Returns:
        A graph plan with each unique owned Serializable node exactly once.

    Raises:
        RepoSaveError: If retained bindings cannot prove a total ObjectRef map.
    """
    if not isinstance(value, Object):
        raise TypeError("Graph save requires one live Object root.")
    binding = build_runtime_binding(repo, value)
    from .repo import RepoSaveError

    if binding.missing:
        missing = next(iter(binding.missing))
        raise RepoSaveError(f"Definition of object {missing} is not reachable in this repo!")
    reference = getattr(value, "_object_ref", None)
    if reference is None or not reference.definition.graph_equal(value.definition):
        raise RepoSaveError("Live root lacks complete retained ObjectRef evidence.")
    actions: list[SaveAction] = []
    seeds = _seed_state_refs(value.definition)
    for path in reference.objects:
        try:
            obj = value.graph_at(path)
        except Exception:
            obj = None
        if isinstance(obj, Serializable):
            actions.append(SaveAction(path, obj.definition, obj))
            continue
        seed = seeds.get(path)
        if seed is not None:
            definition, state_hash = seed
            actions.append(SaveAction(path, definition, value, state_hash=state_hash))
            continue
        raise RepoSaveError(f"No retained Serializable binding or exact seed state at {path!s}.")
    return SavePlan(graph=binding.graph, binding=binding, actions=tuple(actions))


def execute_save_plan(
        repo,
        plan: SavePlan,
        *,
        store,
        deep_capture: bool = False,
        federated: bool = False,
        report_stores: bool = False,
        capture_memo: set[object] | None = None):
    """Publish local states then the complete enclosing StateRef record last.

    Args:
        repo: Repo owning the retained live bindings and claim fence hooks.
        plan: Complete preflighted graph save plan.
        store: Writable StateRef target Store.
        deep_capture: Whether to serialize every uncaptured live state node.
        federated: Whether reusable immutable state may remain in connected Stores.
        report_stores: Whether to return an ephemeral StoreReport with the StateRef.
        capture_memo: Internal ObjectId set already captured by pending-declaration
            completion; those states are adopted instead of serialized again.

    Returns:
        Complete StateRef, optionally paired with StoreReport.

    Raises:
        RepoSaveError: If retained bindings, codecs, closure, or publication fails.
        StoreAuthorityError: If a Store rejects immutable state or record authority.

    Side Effects:
        Publishes local state directories and then the enclosing StateRef. A later
        hook or immutable install failure can leave only unreferenced completed
        local-state directories.

    Concurrency:
        The final StateRef publication executes under the target Store writer
        fence and validates a matching initial-construction claim when present.

    Store Requirements:
        ``store`` must support writable same-Store staging for captures, immutable
        StateRef installation, atomic small-record replacement, and writer locks.
    """
    from .reference_values import StateRef
    from .store.records import DefinitionRecord, StateRefRecord

    reference = plan.binding.roots[0].obj.object_ref
    state_actions = list(plan.actions)
    _validate_codecs(state_actions)
    embedded = _resolve_embedded_state_refs(repo, plan.binding.roots[0].definition)
    capture_memo = set() if capture_memo is None else capture_memo
    reusable: dict[GraphPath, Any] = {}
    captures: list[SaveAction] = []
    for action in state_actions:
        if action.state_hash is not None:
            source = _find_local_state(repo, action.definition, action.state_hash)
            if source is None:
                raise _save_error(action.path, "exact seed local state is not available in a connected Store")
            reusable[action.path] = source
            continue
        state_hash = getattr(action.obj, "_last_state_hash", None)
        source = (
            None if (deep_capture and action.obj.object_id not in capture_memo) or state_hash is None
            else _find_local_state(repo, action.definition, state_hash)
        )
        if source is None:
            captures.append(action)
        else:
            reusable[action.path] = source

    # Capability and closure checks deliberately precede every user hook.
    store.preflight_publication("save graph", local_state=bool(captures or (not federated and (reusable or embedded))))
    for _, record, state_sources in embedded:
        if record is None:
            raise _save_error(GraphPath(), "embedded StateRef has no authority record in a connected Store")
        for _, _, source in state_sources:
            if source is None:
                raise _save_error(GraphPath(), "embedded StateRef local state is not available in a connected Store")

    for node in plan.graph.nodes():
        store.write_definition_record(DefinitionRecord(node.definition))
    for action in state_actions:
        store.write_definition_record(DefinitionRecord(action.definition))

    selected: dict[GraphPath, Any] = {}
    states: dict[GraphPath, str] = {}
    for action in captures:
        state_hash = _publish_local_state(action.obj, action.definition, store, action.path)
        states[action.path] = state_hash
        selected[action.path] = store
    for action in state_actions:
        if action.path in states:
            continue
        source = reusable[action.path]
        state_hash = action.state_hash or getattr(action.obj, "_last_state_hash")
        if not federated and source is not store:
            store.copy_local_state_from(source, action.definition, state_hash)
            source = store
        states[action.path] = state_hash
        selected[action.path] = source

    for _, record, state_sources in embedded:
        if not federated:
            for definition, state_hash, source in state_sources:
                if source is not store:
                    store.copy_local_state_from(source, definition, state_hash)
            store.write_state_ref_record(record)

    state_ref = StateRef(reference, states)
    # This is the graph-level publication boundary. Nothing mutable is updated
    # until the immutable total StateRef record has been installed.
    with store.writer_lock():
        repo._complete_initial_state_ref(state_ref, store)
        store.write_state_ref_record(StateRefRecord(state_ref))
        repo._mark_initial_state_ref_complete(state_ref, store)
    repo._num_saves += 1
    required = [store]
    required.extend(selected.values())
    if federated:
        for _, _, state_sources in embedded:
            required.extend(source for _, _, source in state_sources)
        required.extend(record_store for record_store, _, _ in embedded)
    report = StoreReport(store, selected, required)
    return (state_ref, report) if report_stores else state_ref


_CODEC_RE = re.compile(r"^[A-Za-z0-9]{1,32}$")


def _validate_codecs(actions: Iterable[SaveAction]) -> None:
    """Validate every selected developer codec before any serializer runs."""
    for action in actions:
        if action.state_hash is not None:
            continue
        codec = getattr(action.obj, "state_codec", None)
        if not isinstance(codec, str) or not _CODEC_RE.fullmatch(codec):
            raise _save_error(action.path, "state_codec must match [A-Za-z0-9]{1,32}")


def _publish_local_state(obj: Object, definition: ConcreteDefinition, store, path: GraphPath) -> str:
    """Write, manifest, and atomically install one local Serializable payload.

    Args:
        obj: Live Serializable node that contributes the payload.
        definition: Exact graph definition for the local state.
        store: Target Store that owns staging and immutable installation.
        path: Enclosing exact StateRef path used in diagnostics.

    Returns:
        The installed codec-qualified local state hash.

    Raises:
        RepoSaveError: If hooks, manifest creation, or installation fails.

    Side Effects:
        Creates and removes Store-owned staging as needed. A completed immutable
        local state may remain unreferenced if a later graph publication fails.
    """
    from .store.records import DefinitionRecord, LocalStateManifest

    codec = obj.state_codec
    stage = os.fspath(store.create_local_state_staging())
    reservation = getattr(obj, "_save_load_reservation", None)
    if reservation is None or not reservation.acquire(blocking=False):
        shutil.rmtree(stage, ignore_errors=True)
        raise _save_error(path, "local state is already reserved by save or restore")
    data_dir = os.path.join(stage, "data")
    try:
        os.makedirs(data_dir, exist_ok=True)
        if os.listdir(data_dir):
            raise ValueError("local-state staging data directory is not empty")
        obj.save_state_to_dir(data_dir, codec=codec)
        definition_record = DefinitionRecord(definition)
        definition_bytes = definition_record.to_bytes()
        with open(os.path.join(stage, "def.pkl"), "wb") as target:
            target.write(definition_bytes)
        manifest = LocalStateManifest(
            codec, definition_record.graph_hash, definition_record.digest,
            hashlib.sha256(definition_bytes).hexdigest(),
            _manifest_files(data_dir),
        )
        with open(os.path.join(stage, "manifest.record"), "wb") as target:
            target.write(manifest.to_bytes())
        store.install_local_state(stage, manifest)
        obj._last_state_hash = manifest.state_hash
    except BaseException as error:
        raise _save_error(path, f"local state publication failed for codec {codec!r}", error) from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        reservation.release()
    return manifest.state_hash


def _manifest_files(data_dir: str) -> tuple[tuple[str, int, str], ...]:
    """Return the exhaustive regular payload manifest for a staging data tree."""
    files: list[tuple[str, int, str]] = []
    for root, directories, names in os.walk(data_dir, followlinks=False):
        for directory in directories:
            path = os.path.join(root, directory)
            if os.path.islink(path) or not os.path.isdir(path):
                raise ValueError(f"unsupported payload directory entry {path!r}")
        for name in names:
            path = os.path.join(root, name)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"unsupported payload file entry {path!r}")
            digest = hashlib.sha256()
            with open(path, "rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            relative = os.path.relpath(path, data_dir).replace(os.sep, "/")
            files.append((relative, os.path.getsize(path), digest.hexdigest()))
    return tuple(sorted(files))


def _find_local_state(repo, definition: ConcreteDefinition, state_hash: str):
    """Return the connected Store carrying one verified reusable local state."""
    for candidate in repo.stores:
        try:
            candidate.validate_local_state(definition, state_hash)
        except Exception:
            continue
        return candidate
    return None


def _seed_state_refs(definition: ConcreteDefinition) -> dict[GraphPath, tuple[ConcreteDefinition, str]]:
    """Expand embedded materializing exact StateRefs into enclosing state paths."""
    result: dict[GraphPath, tuple[ConcreteDefinition, str]] = {}
    for outer, reference in _embedded_state_refs(definition):
        for path, state_hash in reference.states.items():
            result[outer.join(path)] = (reference.object.at(path).definition, state_hash)
    return result


def _embedded_state_refs(definition: ConcreteDefinition):
    """Yield materializing StateRefs with their paths in one enclosing CDef."""
    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .reference_values import StateRef

    result = []
    visited: set[object] = set()

    def visit_value(value: Any, path: GraphPath) -> None:
        if isinstance(value, StateRef):
            result.append((path, value))
            return
        if isinstance(value, ConcreteDefinition):
            visit_cdef(value, path)
            return
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                visit_value(value.target, path)
            return
        for edge in iter_value_edges(value):
            visit_value(edge.value, path.child(edge.segment))

    def visit_cdef(cdef: ConcreteDefinition, path: GraphPath) -> None:
        key = cdef_node_key(cdef)
        if key in visited:
            return
        visited.add(key)
        for edge in iter_value_edges(cdef):
            visit_value(edge.value, path.child(edge.segment))

    visit_cdef(definition, GraphPath())
    return tuple(result)


def _resolve_embedded_state_refs(repo, definition: ConcreteDefinition):
    """Resolve exact-reference record and local-state authority before hooks run."""
    resolved = []
    for _, reference in _embedded_state_refs(definition):
        record_store = None
        for candidate in repo.stores:
            try:
                record = candidate.read_state_ref_record(reference.digest())
            except Exception:
                continue
            if record is not None and record.state_ref == reference:
                record_store = candidate
                break
        state_sources = []
        for path, state_hash in reference.states.items():
            source = _find_local_state(repo, reference.object.at(path).definition, state_hash)
            state_sources.append((reference.object.at(path).definition, state_hash, source))
        resolved.append((record_store, None if record_store is None else record_store.read_state_ref_record(reference.digest()), tuple(state_sources)))
    return tuple(resolved)


def _unique_stores(stores: Iterable[Any]) -> tuple[Any, ...]:
    """Deduplicate Store objects by identity without relying on their equality."""
    result = []
    seen = set()
    for store in stores:
        if store is None or id(store) in seen:
            continue
        seen.add(id(store))
        result.append(store)
    return tuple(result)


def _save_error(path: GraphPath, message: str, cause: BaseException | None = None):
    """Return a path-specific RepoSaveError retaining a useful failure cause."""
    from .repo import RepoSaveError

    error = RepoSaveError(f"Graph save at {path!s}: {message}.")
    if cause is not None:
        error.__cause__ = cause
    return error


def _collect_runtime_roots(value: Any, path: GraphPath, roots: list[RuntimeRoot]) -> None:
    if is_runtime_leaf(value):
        return
    kind = node_kind(value)
    if kind is NodeKind.OBJECT:
        roots.append(RuntimeRoot(value.definition, path, value))
        return
    if kind is NodeKind.CONCRETE_DEFINITION:
        roots.append(RuntimeRoot(value, path, None))
        return
    if kind is NodeKind.DEFINITION:
        from .repo import RepoGraphError

        raise RepoGraphError("Plain Definitions aren't allowed here.")
    if kind in {
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.DICT,
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
    }:
        for edge in iter_value_edges(value):
            _collect_runtime_roots(edge.value, path.child(edge.segment), roots)
        return
    from .repo import RepoGraphError

    raise RepoGraphError(f"Unexpected object of type {type(value).__name__} at {path.legacy_str()}!")


def _cached_object(repo, cdef: ConcreteDefinition) -> Object | None:
    """Return one unambiguous candidate from the explicitly supplied Repo."""

    return repo.get_cached(cdef)


def _resolve_missing_for_traversal(
        repo,
        cdef: ConcreteDefinition,
        path: GraphPath,
        options: RepoGraphOptions,
        load_memo: dict[ConcreteDefinition, Object]) -> Object | None:
    obj = repo.get_cached(cdef)
    if obj is not None:
        return obj
    if options.missing == "skip":
        return None
    if options.missing == "load":
        return repo._materialize_cdef(cdef, memo=load_memo, path=list(path.legacy_tuple()))
    from .repo import RepoGraphError

    raise RepoGraphError(
        f"Definition {cdef} is not reachable as a live object in this repo at {path.legacy_str()}."
    )


def _node_primary_path(
        graph: ConcreteDefinitionGraph,
        roots: tuple[RuntimeRoot, ...],
        cdef: ConcreteDefinition) -> GraphPath:
    for root in roots:
        rel_path = graph.primary_path(root.definition, cdef)
        if rel_path is not None:
            return root.path.join(rel_path)
    return GraphPath()


def _add_object_single(repo, obj: Object, *, store=None) -> None:
    repo.pin(obj)
    if store is not None:
        repo.set_object_store(obj, store)
    elif obj.definition not in repo.obj_default_store:
        if repo.default_store is not None:
            repo.set_object_store(obj, repo.default_store)


def _materialize_reachable_nodes(graph: ConcreteDefinitionGraph, roots: tuple[RuntimeRoot, ...]) -> set[ConcreteDefinition]:
    out: set[ConcreteDefinition] = set()
    stack = [root.definition for root in roots]
    while stack:
        cdef = stack.pop()
        if cdef in out:
            continue
        out.add(cdef)
        for edge in graph.outgoing(cdef):
            if edge.kind is EdgeKind.MATERIALIZE:
                stack.append(edge.child)
    return out


def _validate_graph_options(options: RepoGraphOptions) -> None:
    if options.order not in ("pre", "post"):
        raise ValueError("Repo graph order must be 'pre' or 'post'.")
    if options.missing not in ("raise", "skip", "load"):
        raise ValueError("Repo graph missing policy must be 'raise', 'skip', or 'load'.")
