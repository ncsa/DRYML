from __future__ import annotations

from io import BytesIO

import pytest

from dryml.core import Repo
from dryml.core.repo import default_repo
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipExportStore, ZipStore
from dryml.formats.refs import format_cdef_id
from dryml.managed import (
    AmbiguousManagedStoreError,
    ManagedOperationStore,
    ManagedStateError,
    ManagedStoreUnsupportedError,
    OperationKey,
    resolve_managed_store,
)


KEY = OperationKey(format_cdef_id("2" * 64), "compute")
FP1 = "managed-declaration-v1-" + "3" * 64
FP2 = "managed-declaration-v1-" + "4" * 64


def test_explicit_store_repo_and_active_default_resolution(tmp_path):
    store = DirStore(tmp_path / "store")

    assert resolve_managed_store(store=store) is store
    assert resolve_managed_store(repo=Repo(store)) is store
    with default_repo(Repo(store)):
        assert resolve_managed_store() is store


def test_store_selection_fails_closed_for_absent_or_competing_stores(tmp_path):
    with default_repo(Repo()):
        with pytest.raises(AmbiguousManagedStoreError, match="no managed-capable Store"):
            resolve_managed_store()

    repo = Repo([DirStore(tmp_path / "left"), DirStore(tmp_path / "right")])
    with pytest.raises(AmbiguousManagedStoreError, match="multiple managed-capable Stores"):
        resolve_managed_store(repo=repo)

    assert resolve_managed_store(repo=repo, store=repo.stores[1]) is repo.stores[1]


def test_zip_store_explicitly_rejects_live_managed_control():
    store = ZipStore(BytesIO())
    try:
        assert store.supports_store_capability("managed-snapshot-v1")
        assert not store.supports_store_capability("managed-control-v1")
        with pytest.raises(ManagedStoreUnsupportedError, match="DirStore"):
            ManagedOperationStore(store)
        snapshot = ManagedOperationStore(store, writable=False)
        with pytest.raises(ManagedStoreUnsupportedError, match="snapshot Stores"):
            snapshot.operation(KEY, FP1).acquire()
        with pytest.raises(NotImplementedError, match="managed control"):
            store.managed_control_root()
    finally:
        store.close()


def test_zip_store_rejects_incomplete_managed_snapshot(tmp_path):
    source = DirStore(tmp_path / "source")
    operation = ManagedOperationStore(source).operation(KEY, FP1)
    with operation.acquire() as lease:
        pending = lease.prepare(resumable=True)
        lease.interrupt(pending.realization.realization_id)
    archive = BytesIO()
    ZipExportStore(
        archive,
        source.base_dir,
        include_paths={".dryml/managed-v1"},
    ).commit()
    snapshot = ZipStore(archive)
    try:
        snapshot_operation = ManagedOperationStore(
            snapshot, writable=False
        ).operation(KEY, FP1)
        with pytest.raises(ManagedStateError, match="incomplete control"):
            snapshot_operation.status()
        with pytest.raises(ManagedStateError, match="incomplete realization"):
            snapshot_operation.history()
    finally:
        snapshot.close()


def test_declaration_rollover_keeps_old_generation_history_across_reopen(tmp_path):
    path = tmp_path / "store"
    managed = ManagedOperationStore(DirStore(path))
    old = managed.operation(KEY, FP1)
    with old.acquire() as lease:
        first = lease.prepare(resumable=True)
        lease._complete_control_only(first.realization.realization_id)
        lease._activate_control_only(first.realization.realization_id)

    current = managed.operation(KEY, FP2)
    with pytest.raises(ManagedStateError, match="declaration fingerprint"):
        current.acquire()
    with current.acquire(advance_declaration=True) as lease:
        second = lease.prepare(resumable=False)

    assert managed.generations(KEY) == (FP1, FP2)
    assert [item.realization_id for item in managed.history(KEY)] == [
        first.realization.realization_id,
        second.realization.realization_id,
    ]
    assert old.active().realization_id == first.realization.realization_id
    with pytest.raises(ManagedStateError, match="current declaration generation"):
        old.acquire()
    with pytest.raises(ManagedStateError, match="older declaration generation"):
        old.acquire(advance_declaration=True)

    reopened = ManagedOperationStore(DirStore(path))
    assert reopened.generations(KEY) == (FP1, FP2)
    assert len(reopened.history(KEY)) == 2


def test_malformed_or_truncated_control_fails_closed(tmp_path):
    operation = ManagedOperationStore(DirStore(tmp_path / "store")).operation(KEY, FP1)
    with operation.acquire() as lease:
        lease.prepare(resumable=True)

    operation.control_path.write_bytes(b'{"schema_version":1')
    with pytest.raises(ManagedStateError, match="could not be read"):
        operation.status()
