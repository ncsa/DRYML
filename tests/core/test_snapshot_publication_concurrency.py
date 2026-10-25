"""Format-4 publication/name CAS overlap and detached metadata read cuts.

V3 snapshot capture/current-annotation consumers remain explicitly refused;
these cases exercise their replacement Store-owned authority boundaries.
"""

from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event, Thread

import pytest

from dryml.core import Definition, Object, ObjectId
from dryml.core.store.authority import StoreCutChanged
from dryml.core.store.dir import DirStore
from dryml.core.store.records import NodeNameRecord
from dryml.core.store.store import StoreAliasConflictError
from dryml.core.store.zip import ZipStore, ZipStoreConflictError


class ConcurrentPayload(Object):
    def __init__(self, value=1):
        self.value = value


def creation(value=1):
    return Definition(ConcurrentPayload, value=value).concretize()


def test_concurrent_publications_preserve_all_declaration_contributions(tmp_path, monkeypatch):
    stores = [DirStore(tmp_path / "store", query_index="none") for _ in range(2)]
    identity = ObjectId()
    start = Barrier(2)
    records, errors = [], []
    # Recipe and saved membership contribute independently for the same identity.
    root = creation().with_identity(identity)
    for store in stores:
        original = store._prepare_node_publication
        def prepared(*args, original=original, **kwargs):
            result = original(*args, **kwargs)
            start.wait(10)
            return result
        monkeypatch.setattr(store, "_prepare_node_publication", prepared)
    def publish(store, kind):
        try:
            records.append((kind, store.publish_node(root, kind=kind)))
        except BaseException as error:
            errors.append(error)
    workers = [Thread(target=publish, args=(store, kind)) for store, kind in zip(stores, ("recipe", "saved"))]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    assert not errors and len(records) == 2
    assert set(stores[0].read_declaration_anchor(identity).sources) == {(k, r.digest) for k, r in records}
    for kind, record in records:
        assert stores[0].read_node(record.digest, kind=kind).object_id == identity


def test_two_name_cas_writers_cannot_lose_an_update(tmp_path):
    stores = [DirStore(tmp_path / "store", query_index="none") for _ in range(2)]
    roots = [stores[0].publish_node(creation(i)) for i in (1, 2)]
    names = [NodeNameRecord("latest", "recipe", r.digest) for r in roots]
    start = Barrier(3)
    completed, conflicts, errors = [], [], []
    def write(index):
        start.wait(10)
        try:
            completed.append(stores[index].compare_and_set_node_name(names[index], expected=None))
        except StoreAliasConflictError as error:
            conflicts.append(error)
        except BaseException as error:
            errors.append(error)
    workers = [Thread(target=write, args=(i,)) for i in range(2)]
    for worker in workers:
        worker.start()
    start.wait(10)
    for worker in workers:
        worker.join(10)
        assert not worker.is_alive()
    assert not errors and len(completed) == len(conflicts) == 1
    assert stores[0].read_node_name("latest") == completed[0]


def test_detached_expansion_does_not_block_writer_and_refuses_changed_cut(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    other = DirStore(store.base_dir, query_index="none")
    root = store.publish_node(creation())
    entered, release = Event(), Event()
    original = store._expand_node_root
    def expand(record):
        entered.set()
        assert release.wait(10)
        return original(record)
    monkeypatch.setattr(store, "_expand_node_root", expand)
    values, errors = [], []
    def read():
        try:
            values.append(store.read_node(root.digest))
        except BaseException as error:
            errors.append(error)
    reader = Thread(target=read)
    reader.start()
    assert entered.wait(10)
    # This completes before releasing graph expansion; no timing race is needed.
    other.install_creation(creation(2))
    release.set()
    reader.join(10)
    assert not reader.is_alive() and not values
    assert len(errors) == 1 and isinstance(errors[0], StoreCutChanged)


def test_distinct_zip_handles_keep_distinct_buffered_cuts_and_conflict_at_commit(tmp_path):
    archive = tmp_path / "store.zip"
    writer = ZipStore(archive)
    try:
        root = writer.publish_node(creation())
        writer.commit()
    finally:
        writer.close()
    first, second = ZipStore(archive), ZipStore(archive)
    try:
        assert first.authority_fence_key() != second.authority_fence_key()
        first.compare_and_set_node_name(NodeNameRecord("first", "recipe", root.digest), expected=None)
        second.compare_and_set_node_name(NodeNameRecord("second", "recipe", root.digest), expected=None)
        assert first.read_node_name("second") is None
        assert second.read_node_name("first") is None
        first.commit()
        with pytest.raises(ZipStoreConflictError, match="changed since open"):
            second.commit()
        assert second._archive_dirty
        assert second.read_node_name("second") is not None
    finally:
        first.close()
        second.close()
