from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

from dryml.core2 import Repo
from dryml.core2.canonical import to_canonical
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.object import Object

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

    return StoreRef.directory(store), None


def _target_cdef(item) -> ConcreteDefinition:
    if isinstance(item, Object):
        return item.definition
    if isinstance(item, ConcreteDefinition):
        return item
    raise TypeError(
        "Update targets must be DRYML Objects or ConcreteDefinitions, "
        f"got {type(item).__name__}."
    )


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
    transfer_ref, transfer_tmp = _coerce_store_ref(
        transfer_store,
        tmp_prefix="dryml-transfer-",
    )
    result_ref, result_tmp = _coerce_store_ref(
        result_store,
        tmp_prefix="dryml-result-",
    )

    transfer_repo = Repo(stores=transfer_ref.open())
    args_canonical = to_canonical(args, repo=transfer_repo)
    kwargs_canonical = to_canonical(kwargs, repo=transfer_repo)
    transfer_repo.save_object((args_canonical, kwargs_canonical))
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
        result_repo.load_object(target.definition, restore_state=True, build_missing=False)
        refreshed = result_repo.get_cached(target.definition)
        if refreshed is not None:
            for key, value in refreshed.__dict__.items():
                if key in {"__cdef__", "__ws__"}:
                    continue
                target.__dict__[key] = value
