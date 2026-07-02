from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, TypeVar

from .canonical import NodeKind, is_runtime_leaf, node_kind
from .cdef_graph import ConcreteDefinitionGraph, EdgeKind
from .definition import ConcreteDefinition, Definition
from .object import Object, Serializable
from .policies import RepoGraphOptions, RepoLoadOptions
from .utils.graph.path import GraphPath
from .utils.graph.value import iter_value_edges


T = TypeVar("T")


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
    objects: dict[ConcreteDefinition, Object]
    missing: frozenset[ConcreteDefinition] = frozenset()


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
    definition: ConcreteDefinition
    obj: Object
    store: Any
    revision: str | None
    minimum_root_depth: int
    reason: Literal["serializable", "explicit-root", "ephemeral-depth"]


@dataclass(slots=True)
class SavePlan:
    graph: ConcreteDefinitionGraph
    binding: RuntimeGraphBinding
    actions: tuple[SaveAction, ...]


def collect_runtime_roots(value: Any) -> tuple[RuntimeRoot, ...]:
    roots: list[RuntimeRoot] = []
    _collect_runtime_roots(value, GraphPath(), roots)
    return tuple(roots)


def build_runtime_binding(
        repo,
        value: Any,
        *,
        resolve_global: bool = False) -> RuntimeGraphBinding:
    roots = collect_runtime_roots(value)
    graph = ConcreteDefinitionGraph.from_roots(root.definition for root in roots)
    materialize_nodes = _materialize_reachable_nodes(graph, roots)
    objects: dict[ConcreteDefinition, Object] = {}
    for root in roots:
        if root.obj is not None:
            bind_runtime_object(objects, root.definition, root.obj, path=root.path)
    for node in graph.nodes():
        if node.definition not in materialize_nodes:
            continue
        if node.definition in objects:
            continue
        obj = _cached_object(repo, node.definition, reuse_weak=True, resolve_global=resolve_global)
        if obj is not None:
            path = _node_primary_path(graph, roots, node.definition)
            bind_runtime_object(objects, node.definition, obj, path=path)
    missing = frozenset(cdef for cdef in materialize_nodes if cdef not in objects)
    return RuntimeGraphBinding(graph=graph, roots=roots, objects=objects, missing=missing)


def bind_runtime_object(
        bindings: dict[ConcreteDefinition, Object],
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


def iter_graph_objects(repo, root: Any, options: RepoGraphOptions) -> Iterator[Object]:
    _validate_graph_options(options)
    binding = build_runtime_binding(repo, root)
    for occurrence in _iter_bound_graph_object_occurrences(repo, binding, options):
        yield occurrence.obj


def _iter_bound_graph_objects(
        repo,
        binding: RuntimeGraphBinding,
        options: RepoGraphOptions,
        *,
        resolve_global: bool = False) -> tuple[Object, ...]:
    return tuple(
        occurrence.obj
        for occurrence in _iter_bound_graph_object_occurrences(
            repo,
            binding,
            options,
            resolve_global=resolve_global,
        )
    )


def _iter_bound_graph_object_occurrences(
        repo,
        binding: RuntimeGraphBinding,
        options: RepoGraphOptions,
        *,
        resolve_global: bool = False) -> tuple[GraphObjectOccurrence, ...]:
    out: list[GraphObjectOccurrence] = []
    load_memo: dict[ConcreteDefinition, Object] = {}
    seen: set[ConcreteDefinition] = set()

    def visit(cdef: ConcreteDefinition, path: GraphPath, explicit_obj: Object | None = None) -> None:
        obj = explicit_obj if explicit_obj is not None else binding.objects.get(cdef)
        if obj is None and resolve_global:
            obj = _cached_object(repo, cdef, reuse_weak=options.load.reuse_weak, resolve_global=True)
        if obj is None:
            obj = _resolve_missing_for_traversal(repo, cdef, path, options, load_memo)
            if obj is None:
                return
            bind_runtime_object(binding.objects, cdef, obj, path=path)

        if options.dedupe:
            if cdef in seen:
                return
            seen.add(cdef)

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
        binding = build_runtime_binding(repo, value, resolve_global=True)
        for obj in _iter_bound_graph_objects(
                repo,
                binding,
                RepoGraphOptions(include_root=True, order="post", missing="raise", dedupe=False),
                resolve_global=True):
            _add_object_single(repo, obj, store=store)
        for missing in binding.missing:
            if _cached_object(repo, missing, reuse_weak=True, resolve_global=True) is None:
                raise KeyError(f"No object linked to definition {missing} found in repo!")


def build_save_plan(
        repo,
        value: Any,
        *,
        store=None,
        revision: dict[ConcreteDefinition, str] | None = None,
        ephemeral_depth: int | None = 0) -> SavePlan:
    _validate_ephemeral_depth(ephemeral_depth)
    revision = revision or {}
    binding = build_runtime_binding(repo, value)
    if binding.missing:
        from .repo import RepoSaveError

        missing = next(iter(binding.missing))
        raise RepoSaveError(f"Definition of object {missing} is not reachable in this repo!")

    explicit_roots = {root.definition for root in binding.roots}
    min_depth = _minimum_depths(binding.graph, binding.roots)
    actions: list[SaveAction] = []
    for cdef in binding.graph.topological_order(dependencies_first=True):
        if cdef not in binding.objects:
            continue
        obj = binding.objects[cdef]
        reason = _save_reason(
            obj,
            cdef,
            explicit_roots=explicit_roots,
            min_depth=min_depth,
            ephemeral_depth=ephemeral_depth,
        )
        if reason is None:
            continue
        action_store = _select_save_store(repo, cdef, store)
        actions.append(SaveAction(
            definition=cdef,
            obj=obj,
            store=action_store,
            revision=revision.get(cdef),
            minimum_root_depth=min_depth.get(cdef, 0),
            reason=reason,
        ))
    return SavePlan(graph=binding.graph, binding=binding, actions=tuple(actions))


def execute_save_plan(repo, plan: SavePlan) -> None:
    saved_objs: dict[str, set[ConcreteDefinition]] = {}
    saved_roots_by_store: dict[Any, set[ConcreteDefinition]] = {}
    graph_registered = False
    for action in plan.actions:
        store_key = repo._query_catalog.store_id(action.store)
        saved = saved_objs.setdefault(store_key, set())
        if action.definition in saved:
            continue
        action.store.save_object(action.obj, revision=action.revision)
        saved.add(action.definition)
        saved_roots_by_store.setdefault(action.store, set()).add(action.definition)
        if not graph_registered:
            repo._query_catalog.register_graph(plan.graph)
            graph_registered = True
        repo._query_catalog.register_stored_root(action.definition, action.store)
        repo._num_saves += 1
    if saved_roots_by_store:
        repo._query_index.register_saved_graph(plan.graph, saved_roots_by_store)


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


def _cached_object(repo, cdef: ConcreteDefinition, *, reuse_weak: bool, resolve_global: bool = False) -> Object | None:
    obj = repo.get_cached(cdef, reuse_weak=reuse_weak)
    if obj is not None:
        return obj
    if resolve_global:
        from .repo import _global_repo

        if repo is not _global_repo:
            return _global_repo.get_cached(cdef, reuse_weak=reuse_weak)
    return None


def _resolve_missing_for_traversal(
        repo,
        cdef: ConcreteDefinition,
        path: GraphPath,
        options: RepoGraphOptions,
        load_memo: dict[ConcreteDefinition, Object]) -> Object | None:
    obj = repo.get_cached(cdef, reuse_weak=options.load.reuse_weak)
    if obj is not None:
        return obj
    if options.missing == "skip":
        return None
    if options.missing == "load":
        return repo._materialize_cdef(
            cdef,
            options=options.load,
            memo=load_memo,
            path=list(path.legacy_tuple()),
        )
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
    cdef = obj.definition
    if cdef in repo.strong_obj_cache and (obj is not repo.strong_obj_cache[cdef]):
        raise KeyError(f"Repo already has a different object matching {cdef}!")
    repo.pin(obj)
    if store is not None:
        repo.set_object_store(obj, store)
    elif obj.definition not in repo.obj_default_store:
        if repo.default_store is not None:
            repo.set_object_store(obj, repo.default_store)


def _minimum_depths(graph: ConcreteDefinitionGraph, roots: tuple[RuntimeRoot, ...]) -> dict[ConcreteDefinition, int]:
    depths: dict[ConcreteDefinition, int] = {}
    queue = deque((root.definition, 0) for root in roots)
    while queue:
        cdef, depth = queue.popleft()
        old = depths.get(cdef)
        if old is not None and old <= depth:
            continue
        depths[cdef] = depth
        for edge in graph.outgoing(cdef):
            if edge.kind is not EdgeKind.MATERIALIZE:
                continue
            queue.append((edge.child, depth + 1))
    return depths


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


def _save_reason(
        obj: Object,
        cdef: ConcreteDefinition,
        *,
        explicit_roots: set[ConcreteDefinition],
        min_depth: dict[ConcreteDefinition, int],
        ephemeral_depth: int | None) -> Literal["serializable", "explicit-root", "ephemeral-depth"] | None:
    if isinstance(obj, Serializable):
        return "serializable"
    if cdef in explicit_roots:
        return "explicit-root"
    if ephemeral_depth is None:
        return "ephemeral-depth"
    if min_depth.get(cdef, 0) <= ephemeral_depth:
        return "ephemeral-depth"
    return None


def _select_save_store(repo, cdef: ConcreteDefinition, explicit_store):
    store = explicit_store
    if store is None:
        store = repo.obj_default_store.get(cdef)
    if store is None:
        store = repo.default_store
    if store is None:
        from .repo import RepoSaveError

        raise RepoSaveError("No store available to save object!")
    return store


def _validate_graph_options(options: RepoGraphOptions) -> None:
    if options.order not in ("pre", "post"):
        raise ValueError("Repo graph order must be 'pre' or 'post'.")
    if options.missing not in ("raise", "skip", "load"):
        raise ValueError("Repo graph missing policy must be 'raise', 'skip', or 'load'.")


def _validate_ephemeral_depth(ephemeral_depth: int | None) -> None:
    if ephemeral_depth is not None:
        if not isinstance(ephemeral_depth, int):
            raise TypeError("ephemeral_depth must be a non-negative integer or None.")
        if ephemeral_depth < 0:
            raise ValueError("ephemeral_depth must be a non-negative integer or None.")
