from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .canonical import from_canonical
from .cdef_graph import ConcreteDefinitionGraph
from .definition import ConcreteDefinition, Definition
from .object import Object, Serializable
from .policies import CachePolicy, RepoLoadOptions
from .symbol import resolve_symbol


MaterializationActionKind = Literal["reuse", "construct"]


@dataclass(frozen=True, slots=True)
class MaterializationAction:
    definition: ConcreteDefinition
    kind: MaterializationActionKind
    primary_path: str
    obj: Object | None = None
    restore_state: bool = True
    store: Any | None = None
    revision: str | None = None
    build_missing: bool = False
    cache: CachePolicy = "weak"
    instance: str = "reuse"


@dataclass(slots=True)
class MaterializationPlan:
    graph: ConcreteDefinitionGraph
    actions: dict[ConcreteDefinition, MaterializationAction]
    order: tuple[ConcreteDefinition, ...]
    options: RepoLoadOptions


def build_materialization_plan(
        repo,
        cdef: ConcreteDefinition,
        options: RepoLoadOptions,
        *,
        revision: dict[ConcreteDefinition, str] | None = None,
        memo: dict | None = None,
        path: list[str | int] | None = None) -> MaterializationPlan:
    if memo is None:
        memo = {}
    if options.instance == "new" and options.cache != "none":
        raise ValueError("instance='new' requires cache='none' (caches are keyed by cdef)")

    graph = ConcreteDefinitionGraph.from_root(cdef)
    included = _included_nodes(repo, graph, cdef, options, memo)
    order = tuple(node for node in graph.topological_order(dependencies_first=True) if node in included)
    primary_paths = _primary_paths(graph)
    root_path = _format_error_path(path)
    actions = {}
    revision = {} if revision is None else revision
    for node in order:
        cached_obj = repo.get_cached(node, reuse_weak=options.reuse_weak) if options.instance == "reuse" else None
        memo_obj = memo.get(node)
        reuse_obj = memo_obj if memo_obj is not None else cached_obj
        kind: MaterializationActionKind = "reuse" if reuse_obj is not None else "construct"
        selected_store = repo._first_store_with(node) if options.restore_state else None
        actions[node] = MaterializationAction(
            definition=node,
            kind=kind,
            primary_path=root_path if node == cdef else str(primary_paths.get(node, "<unknown>")),
            obj=reuse_obj,
            restore_state=options.restore_state,
            store=selected_store,
            revision=revision.get(node),
            build_missing=options.build_missing,
            cache=options.cache,
            instance=options.instance,
        )
    return MaterializationPlan(graph=graph, actions=actions, order=order, options=options)


def execute_materialization_plan(
        repo,
        plan: MaterializationPlan,
        *,
        memo: dict,
        revision: dict[ConcreteDefinition, str],
        root: ConcreteDefinition):
    from .repo import RepoLoadError

    local_memo = dict(memo)
    for cdef in plan.order:
        if cdef in local_memo:
            continue

        action = plan.actions[cdef]
        revision_str = action.revision if action.revision is not None else revision.get(cdef, None)

        if action.kind == "reuse":
            obj = action.obj if action.obj is not None else repo.get_cached(cdef, reuse_weak=plan.options.reuse_weak)
            if obj is None:
                raise RepoLoadError(
                    f"Materialization plan requested cached reuse for {cdef} at {action.primary_path}, "
                    "but no cached object is available."
                )
            if action.restore_state:
                _restore_cached_if_needed(
                    repo,
                    cdef,
                    obj,
                    is_serializable=isinstance(obj, Serializable),
                    store=action.store,
                    revision_str=revision_str,
                    build_missing=action.build_missing,
                )
            local_memo[cdef] = obj
            memo[cdef] = obj
            continue

        if action.kind != "construct":
            raise RepoLoadError(f"Unknown materialization action kind {action.kind!r} at {action.primary_path}.")

        try:
            cls = resolve_symbol(cdef.cls)
        except Exception as e:
            cls_name = getattr(cdef.cls, "__name__", repr(cdef.cls))
            raise RepoLoadError(f"Error resolving {cls_name} at {action.primary_path}: {e}") from e

        is_serializable = isinstance(cls, type) and issubclass(cls, Serializable)

        in_store = action.store is not None
        if action.restore_state and is_serializable and (not in_store) and (not action.build_missing):
            raise RepoLoadError(
                f"Missing stored state for {cdef} at {action.primary_path} "
                f"(set build_missing=True to allow fresh construction)"
            )

        rt_args = from_canonical_local(cdef.args, resolve_cdef=lambda child: local_memo[child], repo=repo)
        rt_kwargs = from_canonical_local(cdef.kwargs, resolve_cdef=lambda child: local_memo[child], repo=repo)

        try:
            obj = cls(*rt_args, repo=repo, __cdef__=cdef, **rt_kwargs)
            repo._num_constructions += 1
        except Exception as e:
            cls_name = getattr(cdef.cls, "__name__", repr(cdef.cls))
            raise RepoLoadError(f"Error constructing {cls_name} at {action.primary_path}: {e}") from e

        if action.restore_state and in_store:
            st = action.store
            if st is None:
                if not action.build_missing:
                    raise RepoLoadError(f"Inconsistent store index for {cdef}")
            else:
                try:
                    st.restore_object(obj, revision=revision_str)
                    repo.set_object_store(cdef, st)
                except Exception as e:
                    raise RepoLoadError(f"Error restoring state for {cdef} at {action.primary_path}: {e}") from e

        local_memo[cdef] = obj
        memo[cdef] = obj
        _publish_cache(repo, obj, action.cache, action.instance)

    return local_memo[root]


def from_canonical_local(value: Any, *, resolve_cdef, repo):
    return from_canonical(value, repo=repo, resolve_cdef=resolve_cdef, restore_state=False)


def _included_nodes(repo, graph: ConcreteDefinitionGraph, root: ConcreteDefinition, options: RepoLoadOptions, memo: dict) -> set[ConcreteDefinition]:
    included: set[ConcreteDefinition] = set()

    def visit(cdef: ConcreteDefinition) -> None:
        if cdef in included:
            return
        included.add(cdef)
        if cdef in memo:
            return
        cached = options.instance == "reuse" and repo.get_cached(cdef, reuse_weak=options.reuse_weak) is not None
        if cached and not options.restore_state:
            return
        for edge in graph.outgoing(cdef):
            visit(edge.child)

    visit(root)
    return included


def _primary_paths(graph: ConcreteDefinitionGraph) -> dict[ConcreteDefinition, str]:
    paths = {root: "$" for root in graph.roots}
    for occ in graph.iter_occurrences():
        paths.setdefault(occ.definition, str(occ.path))
    return paths


def _format_error_path(path: list[str | int] | None) -> str:
    if not path:
        return "<root>"
    return "/".join(map(str, path))


def _restore_cached_if_needed(
        repo,
        cdef: ConcreteDefinition,
        obj: Object,
        *,
        is_serializable: bool,
        store,
        revision_str: str | None,
        build_missing: bool) -> None:
    from .repo import RepoLoadError

    if revision_str is not None:
        st = store
        if st is None:
            if is_serializable and not build_missing:
                raise RepoLoadError(f"No store has requested object ({cdef})")
            return
        try:
            st.restore_object(obj, revision=revision_str)
        except Exception as e:
            raise RepoLoadError(f"Store can't restore requested revision ({revision_str}) for object ({cdef})") from e
    st = repo.obj_default_store.get(cdef) or store
    if st is not None:
        repo.set_object_store(cdef, st)


def _publish_cache(repo, obj: Object, cache: CachePolicy, instance: str) -> None:
    if instance != "reuse":
        return
    if cache == "strong":
        repo.cache_strong(obj)
    elif cache == "weak":
        repo.cache_weak(obj)
    elif cache == "none":
        return
    else:
        raise ValueError(f"Unknown cache policy: {cache!r}")
