from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .canonical import from_canonical
from .cdef_identity import V1_IDENTITY_VERSION
from .cdef_graph import ConcreteDefinitionGraph, EdgeKind
from .definition import ConcreteDefinition, Definition
from .object import Object, Serializable
from .policies import CachePolicy, LiveReusePolicy, RepoLoadOptions
from .symbol import resolve_symbol
from .cdef_identity import cdef_node_key
from .repo_plan import _NodeBindings, attach_runtime_binding, realization_scope
from .utils.graph.path import GraphPath


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
        import inspect

        bound = inspect.signature(cls.__init__).bind(obj, *rt_args, **rt_kwargs)
        parameters = dict(bound.arguments)
        parameters.pop("self", None)
        attach_runtime_binding(repo, cdef, obj, local_memo, parameters)
        memo[cdef_node_key(cdef)] = obj
        _publish_cache(repo, obj, action.cache, action.instance)

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
        resolve_reference=resolve_reference, restore_state=False,
    )


@dataclass(frozen=True, slots=True)
class ExactStateAction:
    """One verified stateful node selected by an exact StateRef load plan."""

    reference: Any
    definition: ConcreteDefinition
    path: Any
    object_id: Any
    state_hash: str
    store: Any


@dataclass(frozen=True, slots=True)
class ExactStateLoadPlan:
    """Definition/reference-only closure evidence required before exact loading."""

    state_ref: Any
    actions: tuple[ExactStateAction, ...]


def build_exact_state_load_plan(repo, state_ref) -> ExactStateLoadPlan:
    """Verify complete StateRef authority without constructing or reserving Objects.

    Args:
        repo: Repo containing every Store required by the exact closure.
        state_ref: Requested immutable StateRef value.

    Returns:
        A complete plan mapping every local state to a validated Store.

    Raises:
        RepoLoadError: If the requested record, definition closure, embedded
            materializing StateRefs, or any local payload is unavailable.
    """
    from .repo import RepoLoadError
    from .cdef_graph import EdgeKind
    from .links import DefLink
    from .store.records import DefinitionRecord
    from .reference_values import ObjectRef, StateRef
    from .utils.graph.value import iter_value_edges

    if not isinstance(state_ref, StateRef):
        raise TypeError("Exact load requires a StateRef.")
    missing = []
    actions = []
    seen = set()
    definition_seen = set()

    def locate_record(reference):
        records = []
        for store in repo.stores:
            try:
                record = store.read_state_ref_record(reference.digest())
            except Exception as error:
                missing.append(f"StateRef {reference.digest()} in {store!r}: {error}")
                continue
            if record is not None and record.state_ref == reference:
                records.append(store)
            elif record is not None:
                missing.append(f"StateRef {reference.digest()} has incompatible authority in {store!r}")
        if not records:
            missing.append(f"authoritative StateRefRecord {reference.digest()}")

    def validate_definition(definition, label):
        expected = DefinitionRecord(definition)
        valid = False
        for store in repo.stores:
            try:
                record = store.read_definition_record(expected.digest)
            except Exception as error:
                missing.append(f"DefinitionRecord {label} in {store!r}: {error}")
                continue
            if record is not None and record.definition.graph_equal(definition):
                valid = True
        if not valid:
            missing.append(f"DefinitionRecord {label}")

    def visit_value(value):
        if isinstance(value, StateRef):
            visit(value)
            return
        if isinstance(value, ObjectRef):
            visit_definition_closure(value.definition)
            return
        if isinstance(value, ConcreteDefinition):
            visit_definition_closure(value)
            return
        if isinstance(value, DefLink):
            if value.kind is EdgeKind.MATERIALIZE:
                visit_value(value.target)
            return
        for edge in iter_value_edges(value):
            visit_value(edge.value)

    def visit_definition_closure(definition):
        key = id(definition)
        if key in definition_seen:
            return
        definition_seen.add(key)
        graph = ConcreteDefinitionGraph.from_root(definition)
        for node in graph.nodes():
            validate_definition(node.definition, node.definition.graph_hash())
        # Definition graph edges intentionally stop at exact references. Walk
        # values separately so materializing ObjectRef/StateRef seed topology is
        # also current DefinitionRecord authority before any realization.
        for node in graph.nodes():
            for edge in iter_value_edges(node.definition):
                visit_value(edge.value)

    def visit(reference):
        digest = reference.digest()
        if digest in seen:
            return
        seen.add(digest)
        locate_record(reference)
        visit_definition_closure(reference.definition)
        for path, state_hash in reference.states.items():
            definition = reference.object.at(path).definition
            sources = []
            for store in repo.stores:
                try:
                    store.validate_local_state(definition, state_hash)
                except Exception:
                    continue
                sources.append(store)
            if not sources:
                missing.append(f"local state {state_hash} at {path!s}")
            else:
                actions.append(ExactStateAction(
                    reference, definition, path, reference.object.objects[path], state_hash,
                    sources[0],
                ))

    visit(state_ref)
    if missing:
        raise RepoLoadError("Exact StateRef preflight is incomplete: " + "; ".join(dict.fromkeys(missing)))
    return ExactStateLoadPlan(state_ref, tuple(actions))


def execute_exact_state_load_plan(
        repo,
        plan: ExactStateLoadPlan,
        *,
        reuse_live: LiveReusePolicy,
        cache: CachePolicy = "weak",
        _reference_memo: dict[str, Object] | None = None,
        _greedy_touched: list[tuple[Object, GraphPath]] | None = None):
    """Realize a verified StateRef dependency-first without partial cache publication.

    Args:
        repo: Repo owning live candidates and the completed result cache.
        plan: Successful definition/reference-only preflight result.
        reuse_live: Exact candidate reuse policy.
        cache: Cache tier used only after every node has completed successfully.
        _reference_memo: Internal realization-scoped memo for repeated
            materializing StateRefs. Entries are added only after their complete
            exact realization succeeds.
        _greedy_touched: Internal realization-scoped list of live candidates
            mutated by greedy restoration. The outer exact realization clears
            and evicts these candidates if any nested seed or parent fails.

    Returns:
        A live root matching ``plan.state_ref``.

    Raises:
        RepoLoadError: If construction, reuse validation, or restoration fails.
    """
    from .repo import RepoLoadError
    from .repo_plan import _NodeBindings, apply_exact_reference_identity, attach_runtime_binding, realization_scope

    if reuse_live not in {"matching", "greedy", "never"}:
        raise ValueError("reuse_live must be 'matching', 'greedy', or 'never'.")
    reference_memo = {} if _reference_memo is None else _reference_memo
    known = reference_memo.get(plan.state_ref.digest())
    if known is not None:
        return known
    action_by_path = {
        action.path: action for action in plan.actions
        if action.reference == plan.state_ref
    }
    selected = _NodeBindings()
    completed = []
    owns_greedy_touched = _greedy_touched is None
    greedy_touched = [] if owns_greedy_touched else _greedy_touched

    def exact_action(cdef, graph):
        path = GraphPath() if cdef is plan.state_ref.definition else graph.primary_path(
            plan.state_ref.definition, cdef
        )
        return action_by_path.get(path)

    def eligible(cdef, action, dependencies):
        candidates = []
        for candidate in repo._all_live_candidates():
            if not isinstance(candidate, Serializable):
                continue
            if candidate.object_id != action.object_id or not candidate.definition.graph_equal(cdef):
                continue
            valid = True
            for edge in dependencies:
                try:
                    if candidate.graph_at(edge.path) is not selected[edge.child]:
                        valid = False
                        break
                except Exception:
                    valid = False
                    break
            if valid:
                candidates.append(candidate)
        return candidates

    def reserve_unique(candidates, *, matching_hash=None):
        retained = []
        for candidate in candidates:
            reservation = getattr(candidate, "_save_load_reservation", None)
            if reservation is None or not reservation.acquire(blocking=False):
                continue
            if matching_hash is None or candidate._last_state_hash == matching_hash:
                retained.append(candidate)
            else:
                reservation.release()
        if len(retained) != 1:
            for candidate in retained:
                candidate._save_load_reservation.release()
            return None
        return retained[0]

    def restore(obj, action, path):
        codec = action.state_hash.split("-", 1)[0]
        try:
            import os

            payload = action.store.validate_local_state(action.definition, action.state_hash)
            obj.restore_state_from_dir(os.path.join(os.fspath(payload), "data"), codec=codec)
        except BaseException as error:
            raise RepoLoadError(
                f"Exact restore at {path!s} with codec {codec!r} failed: {error}"
            ) from error
        obj._last_state_hash = action.state_hash

    try:
        with realization_scope():
            graph = ConcreteDefinitionGraph.from_root(plan.state_ref.definition)
            for cdef in graph.topological_order(dependencies_first=True):
                if cdef in selected:
                    continue
                direct = tuple(edge for edge in graph.outgoing(cdef) if edge.kind is EdgeKind.MATERIALIZE)
                action = exact_action(cdef, graph)
                if action is not None and reuse_live != "never":
                    candidates = eligible(cdef, action, direct)
                    candidate = reserve_unique(
                        candidates,
                        matching_hash=action.state_hash if reuse_live == "matching" else None,
                    )
                    if candidate is not None:
                        if reuse_live == "greedy" and candidate._last_state_hash != action.state_hash:
                            greedy_touched.append((candidate, action.path))
                            restore(candidate, action, action.path)
                        candidate._save_load_reservation.release()
                        selected[cdef] = candidate
                        completed.append(candidate)
                        continue

                try:
                    cls = resolve_symbol(cdef.cls)
                    from .cdef_codec import validate_cdef_stateful_role
                    validate_cdef_stateful_role(cdef, cls)
                    args, kwargs = project_cdef_call(cdef, cls=cls)
                    def resolve_reference(reference):
                        known_reference = reference_memo.get(reference.digest())
                        if known_reference is not None:
                            return known_reference
                        return execute_exact_state_load_plan(
                            repo, build_exact_state_load_plan(repo, reference),
                            reuse_live=reuse_live, cache="none",
                            _reference_memo=reference_memo,
                            _greedy_touched=greedy_touched,
                        )

                    runtime_args = from_canonical_local(
                        args, repo=repo, resolve_cdef=lambda child: selected[child],
                        resolve_reference=resolve_reference,
                    )
                    runtime_kwargs = from_canonical_local(
                        kwargs, repo=repo, resolve_cdef=lambda child: selected[child],
                        resolve_reference=resolve_reference,
                    )
                    obj = cls(*runtime_args, repo=repo, __cdef__=cdef, **runtime_kwargs)
                    repo._num_constructions += 1
                    import inspect

                    bound = inspect.signature(cls.__init__).bind(obj, *runtime_args, **runtime_kwargs)
                    parameters = dict(bound.arguments)
                    parameters.pop("self", None)
                    attach_runtime_binding(
                        repo, cdef, obj, selected, parameters
                    )
                    if action is not None:
                        reservation = obj._save_load_reservation
                        if not reservation.acquire(blocking=False):
                            raise RepoLoadError(f"Fresh exact node at {action.path!s} could not reserve itself.")
                        try:
                            restore(obj, action, action.path)
                        finally:
                            reservation.release()
                    selected[cdef] = obj
                    completed.append(obj)
                except RepoLoadError:
                    raise
                except BaseException as error:
                    raise RepoLoadError(f"Exact construction at {exact_action(cdef, graph).path if action else '$'} failed: {error}") from error
            root = selected[plan.state_ref.definition]
            apply_exact_reference_identity(root, plan.state_ref.object)
            for obj in completed:
                if cache == "strong":
                    repo.cache_strong(obj)
                elif cache == "weak":
                    repo.cache_weak(obj)
            reference_memo[plan.state_ref.digest()] = root
            return root
    except BaseException as error:
        if not owns_greedy_touched:
            raise
        mutated_paths = tuple(dict.fromkeys(str(path) for _, path in greedy_touched))
        for candidate, path in greedy_touched:
            candidate._last_state_hash = None
            repo._evict_live(candidate)
        if mutated_paths:
            error.args = (
                f"{error} Greedy restore mutated and evicted candidates at "
                f"{', '.join(mutated_paths)}.",
            )
        raise


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
