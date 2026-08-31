from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .canonical import from_canonical
from .cdef_identity import V1_IDENTITY_VERSION
from .cdef_graph import ConcreteDefinitionGraph, EdgeKind
from .definition import ConcreteDefinition, Definition
from .object import Object, Serializable
from .policies import CachePolicy, RepoLoadOptions
from .symbol import resolve_symbol
from .cdef_identity import cdef_node_key
from .repo_plan import _NodeBindings, attach_runtime_binding, realization_scope


MaterializationActionKind = Literal["reuse", "construct"]
MaterializationReuseSource = Literal["memo", "cache", None]


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
    restore_state: bool = True
    store: Any | None = None
    revision: str | None = None
    build_missing: bool = False
    cache: CachePolicy = "weak"
    instance: str = "reuse"


@dataclass(slots=True)
class MaterializationPlan:
    """Definition-only ordered materialization graph and per-node actions."""

    graph: ConcreteDefinitionGraph
    actions: _NodeBindings
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
    """Build a definition-only plan without acquiring live cached Objects.

    Plan construction remains available during strict orchestration. It may
    inspect cache and Store availability metadata but never resolves classes,
    restores state, or retains a live Object in the returned plan.
    """

    if memo is None:
        memo = {}
    if options.instance == "new" and options.cache != "none":
        raise ValueError("instance='new' requires cache='none' (caches are keyed by cdef)")

    graph = ConcreteDefinitionGraph.from_root(cdef)
    included = _included_nodes(repo, graph, cdef, options, memo)
    order = tuple(
        node for node in graph.topological_order(dependencies_first=True)
        if cdef_node_key(node) in included
    )
    primary_paths = _primary_paths(graph)
    root_path = _format_error_path(path)
    actions = _NodeBindings()
    revision = {} if revision is None else revision
    for node in order:
        memo_reuse = memo.get(cdef_node_key(node), memo.get(node)) is not None
        cache_reuse = options.instance == "reuse" and repo.has_cached(
            node, reuse_weak=options.reuse_weak
        )
        reuse_source: MaterializationReuseSource = (
            "memo" if memo_reuse else ("cache" if cache_reuse else None)
        )
        kind: MaterializationActionKind = "reuse" if reuse_source is not None else "construct"
        selected_store = repo._first_store_with(node) if options.restore_state else None
        actions[node] = MaterializationAction(
            definition=node,
            kind=kind,
            primary_path=root_path if cdef_node_key(node) is cdef_node_key(cdef) else str(primary_paths.get(node, "<unknown>")),
            reuse_source=reuse_source,
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
    from dryml.runtime import materialization_admission
    from .repo import RepoLoadError

    with materialization_admission(operation="execute_materialization_plan"):
        with realization_scope():
            return _execute_materialization_plan(repo, plan, memo=memo, revision=revision, root=root)


def _execute_materialization_plan(
        repo,
        plan: MaterializationPlan,
        *,
        memo: dict,
        revision: dict[ConcreteDefinition, str],
        root: ConcreteDefinition):
    """Execute an already admitted plan without resolving classes beforehand."""

    from .repo import RepoLoadError

    local_memo = _NodeBindings()
    for key, obj in memo.items():
        local_memo[key] = obj
    for cdef in plan.order:
        if cdef in local_memo:
            continue

        action = plan.actions[cdef]
        revision_str = action.revision if action.revision is not None else revision.get(cdef, None)

        if action.kind == "reuse":
            obj = local_memo.get(cdef) if action.reuse_source == "memo" else repo.get_cached(
                cdef, reuse_weak=plan.options.reuse_weak
            )
            if obj is None:
                if action.reuse_source == "cache":
                    refreshed = build_materialization_plan(
                        repo,
                        root,
                        plan.options,
                        revision=revision,
                        memo=local_memo,
                    )
                    return _execute_materialization_plan(
                        repo,
                        refreshed,
                        memo=memo,
                        revision=revision,
                        root=root,
                    )
                source = "memoized" if action.reuse_source == "memo" else "cached"
                raise RepoLoadError(
                    f"Materialization plan requested {source} reuse for {cdef} at {action.primary_path}, "
                    "but no reusable object is available."
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
            memo[cdef_node_key(cdef)] = obj
            continue

        if action.kind != "construct":
            raise RepoLoadError(f"Unknown materialization action kind {action.kind!r} at {action.primary_path}.")

        try:
            cls = resolve_symbol(cdef.cls)
        except Exception as e:
            cls_name = getattr(cdef.cls, "__name__", repr(cdef.cls))
            raise RepoLoadError(f"Error resolving {cls_name} at {action.primary_path}: {e}") from e

        from .cdef_codec import CDefGraphCodecError, validate_cdef_stateful_role

        try:
            validate_cdef_stateful_role(cdef, cls)
        except CDefGraphCodecError as error:
            raise RepoLoadError(f"Incompatible definition authority at {action.primary_path}: {error}") from error
        is_serializable = issubclass(cls, Serializable)

        in_store = action.store is not None
        if action.restore_state and is_serializable and (not in_store) and (not action.build_missing):
            raise RepoLoadError(
                f"Missing stored state for {cdef} at {action.primary_path} "
                f"(set build_missing=True to allow fresh construction)"
            )

        canonical_args, canonical_kwargs = project_cdef_call(cdef, cls=cls)
        rt_args = from_canonical_local(canonical_args, resolve_cdef=lambda child: local_memo[child], repo=repo)
        rt_kwargs = from_canonical_local(canonical_kwargs, resolve_cdef=lambda child: local_memo[child], repo=repo)

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
        attach_runtime_binding(repo, cdef, obj, local_memo)
        memo[cdef_node_key(cdef)] = obj
        _publish_cache(repo, obj, action.cache, action.instance)

    return local_memo[root]


def from_canonical_local(value: Any, *, resolve_cdef, repo):
    return from_canonical(value, repo=repo, resolve_cdef=resolve_cdef, restore_state=False)


def project_cdef_call(
        cdef: ConcreteDefinition,
        *,
        cls: type | None = None) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Project an exact identity onto its runtime constructor call surface.

    Args:
        cdef: Exact V1 or V2 identity to invoke.
        cls: Optional already-resolved current runtime class.

    Returns:
        Canonical positional and keyword values suitable for runtime decoding.

    Raises:
        TypeError: If a V2 record is incompatible with the current class
            signature.
        Exception: If resolving a V2 class fails.

    V1 identities retain their persisted raw call surface. V2 identities use
    their persisted semantic record and the current class signature without
    invoking preparation or applying defaults.
    """

    from dryml.runtime import materialization_admission

    with materialization_admission(operation="project_cdef_constructor_call"):
        if cdef.identity_version == V1_IDENTITY_VERSION:
            return tuple(cdef._args), dict(cdef._kwargs)
        if cls is None:
            cls = resolve_symbol(cdef.cls)
        from .bound_args import project_bound_arguments

        return project_bound_arguments(cls, cdef._bound_args)


def _included_nodes(repo, graph: ConcreteDefinitionGraph, root: ConcreteDefinition, options: RepoLoadOptions, memo: dict) -> set[ConcreteDefinition]:
    included: set[object] = set()

    def visit(cdef: ConcreteDefinition) -> None:
        key = cdef_node_key(cdef)
        if key in included:
            return
        included.add(key)
        if cdef_node_key(cdef) in memo or cdef in memo:
            return
        cached = options.instance == "reuse" and repo.has_cached(
            cdef, reuse_weak=options.reuse_weak
        )
        if cached and not options.restore_state:
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
