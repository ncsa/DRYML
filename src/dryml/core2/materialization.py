from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .canonical import NodeKind, node_kind, transform_container
from .cdef_graph import ConcreteDefinitionGraph
from .definition import ConcreteDefinition, Definition
from .freeze import FrozenNDArray
from .object import Object, Serializable
from .policies import CachePolicy, RepoLoadOptions
from .symbol import resolve_symbol


MaterializationActionKind = Literal["reuse", "construct"]


@dataclass(frozen=True, slots=True)
class MaterializationAction:
    definition: ConcreteDefinition
    kind: MaterializationActionKind
    primary_path: str


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
    for node in order:
        cached = options.instance == "reuse" and repo.get_cached(node, reuse_weak=options.reuse_weak) is not None
        kind: MaterializationActionKind = "reuse" if cached or node in memo else "construct"
        actions[node] = MaterializationAction(
            definition=node,
            kind=kind,
            primary_path=root_path if node == cdef else str(primary_paths.get(node, "<unknown>")),
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

    for cdef in plan.order:
        if cdef in memo:
            continue

        action = plan.actions[cdef]
        try:
            cls = resolve_symbol(cdef.cls)
        except Exception as e:
            cls_name = getattr(cdef.cls, "__name__", repr(cdef.cls))
            raise RepoLoadError(f"Error resolving {cls_name} at {action.primary_path}: {e}") from e

        is_serializable = isinstance(cls, type) and issubclass(cls, Serializable)
        revision_str = revision.get(cdef, None)

        if plan.options.instance == "reuse":
            obj = repo.get_cached(cdef, reuse_weak=plan.options.reuse_weak)
            if obj is not None:
                memo[cdef] = obj
                if plan.options.restore_state:
                    _restore_cached_if_needed(
                        repo,
                        cdef,
                        obj,
                        is_serializable=is_serializable,
                        revision_str=revision_str,
                        build_missing=plan.options.build_missing,
                    )
                continue

        in_store = repo.has_cdef_light(cdef)
        if plan.options.restore_state and is_serializable and (not in_store) and (not plan.options.build_missing):
            raise RepoLoadError(
                f"Missing stored state for {cdef} at {action.primary_path} "
                f"(set build_missing=True to allow fresh construction)"
            )

        rt_args = from_canonical_local(cdef.args, resolve_cdef=lambda child: memo[child], repo=repo)
        rt_kwargs = from_canonical_local(cdef.kwargs, resolve_cdef=lambda child: memo[child], repo=repo)

        try:
            obj = cls(*rt_args, repo=repo, __cdef__=cdef, **rt_kwargs)
            repo._num_constructions += 1
        except Exception as e:
            cls_name = getattr(cdef.cls, "__name__", repr(cdef.cls))
            raise RepoLoadError(f"Error constructing {cls_name} at {action.primary_path}: {e}") from e

        memo[cdef] = obj

        if plan.options.restore_state and in_store:
            st = repo._first_store_with(cdef)
            if st is None:
                if not plan.options.build_missing:
                    raise RepoLoadError(f"Inconsistent store index for {cdef}")
            else:
                try:
                    st.restore_object(obj, revision=revision_str)
                    repo.set_object_store(cdef, st)
                except Exception as e:
                    raise RepoLoadError(f"Error restoring state for {cdef} at {action.primary_path}: {e}") from e

        _publish_cache(repo, obj, plan.options.cache, plan.options.instance)

    return memo[root]


def from_canonical_local(value: Any, *, resolve_cdef, repo):
    kind = node_kind(value)
    if kind in {NodeKind.POD, NodeKind.TYPE, NodeKind.IDENTITY_VALUE}:
        return value
    if kind is NodeKind.FROZEN_NDARRAY:
        return value.thaw() if hasattr(value, "thaw") else np.array(value, copy=True)
    if kind is NodeKind.NDARRAY:
        return np.array(value, copy=True)
    if kind is NodeKind.CONCRETE_DEFINITION:
        return resolve_cdef(value)
    if kind is NodeKind.DEFINITION:
        cdef = value.concretize(repo=repo)
        return resolve_cdef(cdef)
    if kind is NodeKind.OBJECT:
        return resolve_cdef(value.definition)
    if kind in {NodeKind.IMPORT_REF, NodeKind.SOURCE_SPEC}:
        return resolve_symbol(value)
    if kind in {
        NodeKind.FROZEN_LIST,
        NodeKind.FROZEN_TUPLE,
        NodeKind.FROZEN_SET,
        NodeKind.FROZEN_DICT,
        NodeKind.LIST,
        NodeKind.TUPLE,
        NodeKind.SET,
        NodeKind.DICT,
    }:
        return transform_container(
            value,
            lambda _, child: from_canonical_local(child, resolve_cdef=resolve_cdef, repo=repo),
            target="runtime",
        )
    raise TypeError(f"Cannot de-canonicalize value of type {type(value).__name__}")


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
        revision_str: str | None,
        build_missing: bool) -> None:
    from .repo import RepoLoadError

    if revision_str is not None:
        st = repo._first_store_with(cdef)
        if st is None:
            if is_serializable and not build_missing:
                raise RepoLoadError(f"No store has requested object ({cdef})")
            return
        try:
            st.restore_object(obj, revision=revision_str)
        except Exception as e:
            raise RepoLoadError(f"Store can't restore requested revision ({revision_str}) for object ({cdef})") from e
    st = repo.obj_default_store.get(cdef) or repo._first_store_with(cdef)
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
