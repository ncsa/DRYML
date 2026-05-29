from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

from dryml.core2 import Repo
from dryml.core2.canonical import to_canonical
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.object import Object
from dryml.core2.utils.general import get_unique_concrete_definitions
from dryml.core2.utils.recurse import iter_leaves

from .protocol import StoreRef


@dataclass(slots=True)
class PreparedCall:
    args_canonical: Any
    kwargs_canonical: Any
    transfer_store: StoreRef
    result_store: StoreRef
    transfer_tmp: tempfile.TemporaryDirectory | None = None
    result_tmp: tempfile.TemporaryDirectory | None = None


def _coerce_store_ref(store, *, tmp_prefix: str):
    if store is None:
        tmp = tempfile.TemporaryDirectory(prefix=tmp_prefix)
        return StoreRef.directory(tmp.name), tmp

    if isinstance(store, StoreRef):
        return store, None

    from dryml.core2.store.dir import DirStore
    if isinstance(store, DirStore):
        return StoreRef.directory(store.base_dir), None

    from dryml.core2.store.store import Store
    if isinstance(store, Store):
        raise TypeError(
            "Local execution currently supports only directory stores as "
            f"StoreRefs, got {type(store).__name__}."
        )

    return StoreRef.directory(store), None


def _default_store_ref_candidate(store):
    if store is None:
        return None
    from dryml.core2.store.dir import DirStore
    if isinstance(store, DirStore):
        return store
    return None


def _target_cdef(item) -> ConcreteDefinition:
    if isinstance(item, Object):
        return item.definition
    if isinstance(item, ConcreteDefinition):
        return item
    raise TypeError(
        "Update targets must be DRYML Objects or ConcreteDefinitions, "
        f"got {type(item).__name__}."
    )


def _store_matches_ref(store, store_ref: StoreRef) -> bool:
    return (
        store is not None
        and store_ref.kind == "directory"
        and getattr(store, "base_dir", None) == store_ref.uri
    )


def _save_live_objects(value, source_repo, transfer_repo: Repo, transfer_ref: StoreRef) -> None:
    objects_by_cdef = {}
    available_cdefs = set()

    for leaf in iter_leaves(value):
        if isinstance(leaf, Object):
            objects_by_cdef[leaf.definition] = leaf
            if _store_matches_ref(source_repo._first_store_with(leaf.definition), transfer_ref):
                available_cdefs.add(leaf.definition)

    for cdef in get_unique_concrete_definitions(value):
        if _store_matches_ref(source_repo._first_store_with(cdef), transfer_ref):
            available_cdefs.add(cdef)
        if cdef in objects_by_cdef:
            continue
        obj = source_repo.get_cached(cdef)
        if obj is not None:
            objects_by_cdef[cdef] = obj

    for cdef in available_cdefs:
        if cdef not in transfer_repo.light_index:
            transfer_repo.light_index.add(cdef)

    for obj in objects_by_cdef.values():
        transfer_repo.cache_weak(obj)
    for obj in objects_by_cdef.values():
        if obj.definition in available_cdefs:
            continue
        transfer_repo.save_object(obj)


def update_cdefs(update=None) -> tuple[ConcreteDefinition, ...]:
    if update is None or update is False:
        return ()
    if update is True:
        raise NotImplementedError("update=True is not implemented yet; pass explicit objects.")
    if isinstance(update, (Object, ConcreteDefinition)):
        update = (update,)
    return tuple(_target_cdef(item) for item in update)


def update_targets(update=None) -> tuple[Object, ...]:
    if update is None or update is False:
        return ()
    if update is True:
        raise NotImplementedError("update=True is not implemented yet; pass explicit objects.")
    if isinstance(update, Object):
        update = (update,)
    return tuple(item for item in update if isinstance(item, Object))


def prepare_call(
        args,
        kwargs,
        *,
        repo=None,
        transfer_store=None,
        result_store=None) -> PreparedCall:
    from dryml.core2.repo import manage_repo

    with manage_repo(repo=repo) as source_repo:
        if transfer_store is None:
            transfer_store = _default_store_ref_candidate(source_repo.default_store)
        if result_store is None:
            result_store = _default_store_ref_candidate(source_repo.default_store)

        transfer_ref, transfer_tmp = _coerce_store_ref(
            transfer_store,
            tmp_prefix="dryml-transfer-",
        )
        result_ref, result_tmp = _coerce_store_ref(
            result_store,
            tmp_prefix="dryml-result-",
        )

        transfer_repo = Repo(stores=transfer_ref.open())
        _save_live_objects((args, kwargs), source_repo, transfer_repo, transfer_ref)

        args_canonical = to_canonical(args, repo=source_repo)
        kwargs_canonical = to_canonical(kwargs, repo=source_repo)
        transfer_repo.flush()

    return PreparedCall(
        args_canonical=args_canonical,
        kwargs_canonical=kwargs_canonical,
        transfer_store=transfer_ref,
        result_store=result_ref,
        transfer_tmp=transfer_tmp,
        result_tmp=result_tmp,
    )


def restore_result(response, *, repo=None, result_store: StoreRef):
    store = result_store.open()
    if repo is None:
        result_repo = Repo(stores=store)
    else:
        from dryml.core2.repo import manage_repo
        with manage_repo(repo=repo) as result_repo:
            result_repo.add_store(store)
            return result_repo.load_object(
                response.result_canonical,
                restore_state=True,
                build_missing=True,
            )

    return result_repo.load_object(
        response.result_canonical,
        restore_state=True,
        build_missing=True,
    )


def restore_updates(targets, *, result_store: StoreRef) -> None:
    targets = update_targets(targets)
    if not targets:
        return
    result_repo = Repo(stores=result_store.open())
    for target in targets:
        store = result_repo._first_store_with(target.definition)
        if store is None:
            continue
        store.restore_object(target)
