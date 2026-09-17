"""Focused shared-DirStore snapshot proofs for the core execution adapter."""

from __future__ import annotations

import threading

import pytest

from dryml.core import Object, Repo, RepoDefinition
from dryml.core.execute import CoreOptions, SharedDirStoreStrategy, prepare_shared_storage
from dryml.core.execute_codec import CoreCallCodecError, decode_invocation
from dryml.core.repo_plan import SaveRouting
from dryml.core.session import config
from dryml.core.selector import Selector
from dryml.core.definition import Definition
from dryml.core.signatures import ReferenceSelection
from dryml.core.store.dir import DirStore
from dryml.core.store.zip import ZipStore


class StorageTarget:
    """Minimal selector target used to retain an ordered route in a Repo export."""


class PinnedStorageTarget(Object):
    """Minimal persisted value used to check frozen selected-Store bindings."""


def _pinned_value():
    """Provide a callable root for detached selection-table encoding."""
    return None


def test_storage_snapshot_uses_one_export_and_survives_later_topology_changes(tmp_path, monkeypatch):
    """An export and selected-Store table remain one locked configuration generation."""

    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    repo = Repo(
        [first, second],
        save_routing=SaveRouting(((Selector(Definition(StorageTarget)), second),)),
    )
    pinned = repo.save(PinnedStorageTarget(repo=repo), deep_capture=True)
    original_export = repo.to_definition
    calls = []

    entered = threading.Event()
    release = threading.Event()
    result = []
    reordered = threading.Event()

    def export_then_wait():
        definition = original_export()
        calls.append(definition)
        entered.set()
        assert release.wait(timeout=5)
        return definition

    monkeypatch.setattr(repo, "to_definition", export_then_wait)

    def prepare():
        result.append(prepare_shared_storage(CoreOptions(repo=repo)))

    thread = threading.Thread(target=prepare)
    thread.start()
    assert entered.wait(timeout=5)
    reorder = threading.Thread(target=lambda: (repo.set_default_store(second), reordered.set()))
    reorder.start()
    assert not reordered.wait(timeout=0.1)
    release.set()
    thread.join(timeout=5)
    reorder.join(timeout=5)
    assert not thread.is_alive()
    assert not reorder.is_alive()
    snapshot = result.pop()
    try:
        assert len(calls) == 1
        assert [store["path"] for store in snapshot.storage_setup["repo"]["stores"]] == [
            store["path"] for store in calls[0].to_data()["stores"]
        ]
        assert snapshot.storage_setup["repo"]["routing"]["routes"][0]["store"] == 1
        assert snapshot.recovery_repo.to_definition().to_data() == calls[0].to_data()
        assert [str(store.base_dir) for store in snapshot.frozen_storage.source_stores] == [
            descriptor["path"] for descriptor in calls[0].to_data()["stores"]
        ]
        assert repo.default_store is second

        prepared = SharedDirStoreStrategy().prepare(
            _pinned_value, (), {}, repo=repo, control_store=None, update_args=False,
            selections={"pin": ReferenceSelection(pinned.object, first)},
            _frozen_storage=snapshot.frozen_storage,
        )
        _, _, _, _, selections, _ = decode_invocation(
            prepared.invocation, repo=snapshot.recovery_repo,
        )
        assert selections["pin"].store is snapshot.recovery_repo.stores[0]

        wrong = DirStore(tmp_path / "wrong", query_index="none")
        try:
            with pytest.raises(CoreCallCodecError, match="unsupported selected Store"):
                SharedDirStoreStrategy().prepare(
                    _pinned_value, (), {}, repo=repo, control_store=None, update_args=False,
                    selections={"pin": ReferenceSelection(pinned.object, wrong)},
                    _frozen_storage=snapshot.frozen_storage,
                )
        finally:
            wrong.close()
    finally:
        snapshot.close()


def test_storage_snapshot_encodes_same_and_separate_control_store_roles(tmp_path):
    """Control authority uses a table reference once or an explicit direct descriptor."""

    state = DirStore(tmp_path / "state", query_index="none")
    same_physical = DirStore.open_existing(tmp_path / "state", query_index="none")
    external = DirStore(tmp_path / "external", query_index="none")
    repo = Repo(state)

    same = prepare_shared_storage(CoreOptions(repo=repo, control_store=same_physical))
    separate = prepare_shared_storage(CoreOptions(repo=repo, control_store=external))
    try:
        assert same.storage_setup["control_store"] == {"repo_store": 0}
        assert separate.storage_setup["control_store"] == {
            "kind": "dir", "path": str(tmp_path / "external"), "query_index": "none",
        }
    finally:
        same.close()
        separate.close()


def test_storage_snapshot_owns_reconstruction_without_closing_borrowed_handles(tmp_path, monkeypatch):
    """Closing a submission snapshot only releases handles that it reconstructed."""

    store = DirStore(tmp_path / "state", query_index="none")
    repo = Repo(store)
    closed = []
    original_close = DirStore.close

    def observe_close(self):
        closed.append(self)
        return original_close(self)

    monkeypatch.setattr(DirStore, "close", observe_close)
    snapshot = prepare_shared_storage(CoreOptions(repo=repo))
    snapshot.close()

    assert store not in closed
    assert closed


def test_storage_snapshot_rejects_missing_definition_storage_without_creating_it(tmp_path):
    """A detached definition must reopen existing authority rather than create a Store."""

    source = DirStore(tmp_path / "source", query_index="none")
    data = Repo(source).to_definition().to_data()
    missing = tmp_path / "missing"
    data["stores"][0]["path"] = str(missing)
    definition = RepoDefinition.from_data(data)

    with pytest.raises(ValueError, match="shared Store"):
        prepare_shared_storage(CoreOptions(repo=definition))

    assert not missing.exists()


def test_storage_snapshot_rejects_zip_definition_before_reconstruction(tmp_path, monkeypatch):
    """The initial strategy rejects every ZipStore descriptor without opening it."""

    archive = ZipStore(tmp_path / "state.zip")
    archive._archive_dirty = True
    archive.commit()
    definition = Repo(archive).to_definition()
    monkeypatch.setattr(
        Repo,
        "from_definition",
        classmethod(lambda cls, value: (_ for _ in ()).throw(AssertionError("must not reconstruct"))),
    )

    with pytest.raises(ValueError, match="DirStores"):
        prepare_shared_storage(CoreOptions(repo=definition))

    archive.close()


def test_storage_snapshot_retains_session_defaults_after_acceptance(tmp_path):
    """Later caller defaults cannot change retained recovery or result policy."""

    first = Repo(DirStore(tmp_path / "first", query_index="none"))
    second = Repo(DirStore(tmp_path / "second", query_index="none"))
    with config(repo=first, cache="strong"):
        snapshot = prepare_shared_storage(CoreOptions(return_objects=False))
        try:
            with config(repo=second, cache="none"):
                assert snapshot.cache == "strong"
                assert snapshot.return_objects is False
                assert snapshot.recovery_repo.to_definition().to_data() == first.to_definition().to_data()
        finally:
            snapshot.close()
