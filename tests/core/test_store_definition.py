"""Standalone Store definition and Session reconstruction-cache contracts."""

from copy import deepcopy
from io import BytesIO
import os

import pytest

import dryml
from dryml import session
from dryml.core import Object, Repo, RepoDefinitionError, RepoReconstructionError
from dryml.core import session as core_session
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord
from dryml.core.store.store import Store, StoreAuthorityError
from dryml.core.store.zip import ZipStore


class StoreDefinitionRecordObject(Object):
    """Small Object type used only to create immutable Store test records."""

    def __init__(self, value="record"):
        super().__init__()
        self.value = value


def _record(value="record"):
    """Return one small immutable record suitable for Store transaction tests."""

    return DefinitionRecord(StoreDefinitionRecordObject(value).definition)


def _committed_zip(path):
    """Create and return a clean path-backed ZipStore with current authority."""

    store = ZipStore(path)
    store._archive_dirty = True
    store.commit()
    return store


@pytest.fixture(autouse=True)
def _reset_core_session():
    """Keep Store-cache selection and ownership independent between tests."""

    dryml.reset_config()
    try:
        yield
    finally:
        dryml.reset_config()


def test_store_definition_round_trip_detaches_dir_and_clean_zip(tmp_path):
    """Portable Store descriptors are detached and reopen existing authority only."""

    directory = DirStore(tmp_path / "directory", query_index="none")
    archive = _committed_zip(tmp_path / "store.zip")
    try:
        directory_definition = directory.to_definition()
        archive_definition = archive.to_definition()

        assert directory_definition == {
            "kind": "dir", "path": str(tmp_path / "directory"), "query_index": "none",
        }
        assert archive_definition == {"kind": "zip", "path": str(tmp_path / "store.zip")}
        directory_definition["query_index"] = "memory"
        assert directory.to_definition()["query_index"] == "none"

        reopened_directory = Store.from_definition(directory.to_definition())
        reopened_archive = Store.from_definition(archive_definition)
        try:
            assert type(reopened_directory) is DirStore
            assert type(reopened_archive) is ZipStore
            assert reopened_directory is not directory
            assert reopened_archive is not archive
        finally:
            reopened_directory.close()
            reopened_archive.close()
    finally:
        directory.close()
        archive.close()


@pytest.mark.parametrize(
    "definition",
    [
        {"kind": "other", "path": "/tmp/store"},
        {"kind": "dir", "path": "/tmp/store"},
        {"kind": "dir", "path": "/tmp/store", "query_index": "other"},
        {"kind": "zip", "path": "relative.zip"},
        {"kind": "zip", "path": "/tmp/store", "extra": True},
    ],
)
def test_store_definition_rejects_malformed_descriptors_without_opening(definition):
    """Closed descriptor validation rejects wrong tags, fields, and settings."""

    with pytest.raises(RepoDefinitionError):
        Store.from_definition(definition)

    with pytest.raises(TypeError):
        Store.from_definition([definition])


@pytest.mark.parametrize("kind", ("missing", "malformed"))
def test_store_definition_requires_existing_valid_authority(tmp_path, kind):
    """Standalone reconstruction neither creates nor repairs unavailable authority."""

    path = tmp_path / kind
    if kind == "malformed":
        path.mkdir()
        (path / "store-format.record").write_bytes(b"malformed")
    definition = {"kind": "dir", "path": str(path), "query_index": "none"}

    with pytest.raises(StoreAuthorityError):
        Store.from_definition(definition)

    assert not path.exists() if kind == "missing" else not (path / "definitions").exists()


def test_store_definition_rejects_nonportable_zip_and_dir_settings(tmp_path):
    """Export rejects dirty/file-like archives and unsupported opening settings."""

    archive = _committed_zip(tmp_path / "store.zip")
    archive._archive_dirty = True
    file_like = ZipStore(BytesIO())
    from dryml.core.query.sqlite import SQLiteQueryIndexConfig

    custom = DirStore(tmp_path / "custom", query_index=SQLiteQueryIndexConfig())
    try:
        with pytest.raises(RepoDefinitionError, match="clean committed archive"):
            archive.to_definition()
        with pytest.raises(RepoDefinitionError, match="clean committed archive"):
            file_like.to_definition()
        with pytest.raises(RepoDefinitionError, match="query policy"):
            custom.to_definition()
    finally:
        archive.close()
        file_like.close()
        custom.close()


def test_dirstore_close_releases_connections_after_reopening(tmp_path):
    """Every close releases currently reopened derived-index connections."""

    class Connection:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    store = DirStore(tmp_path / "store", query_index="none")
    first = Connection()
    second = Connection()
    store._query_index_instance = first
    store.close()
    store._query_index_instance = second
    store.close()

    assert first.closed
    assert second.closed


def test_open_existing_custom_sqlite_configuration_remains_fresh_and_uncached(tmp_path):
    """Custom SQLite settings retain the existing uncached opening contract."""

    from dryml.core.query.sqlite import SQLiteQueryIndexConfig

    store = DirStore(tmp_path / "store", query_index="none")
    config = SQLiteQueryIndexConfig(journal_mode="delete")
    try:
        first = DirStore.open_existing(store.base_dir, query_index=config)
        second = DirStore.open_existing(store.base_dir, query_index=config)
        try:
            assert first is not second
            assert first.query_index is config
            assert second.query_index is config
            with session.resource_cache() as cache:
                cached_first = DirStore.open_existing(store.base_dir, query_index=config)
                cached_second = DirStore.open_existing(store.base_dir, query_index=config)
                try:
                    assert cached_first is not cached_second
                    assert cached_first not in cache.stores
                    assert cached_second not in cache.stores
                finally:
                    cached_first.close()
                    cached_second.close()
        finally:
            first.close()
            second.close()
    finally:
        store.close()


def test_store_definition_rejects_unhashable_query_policy(tmp_path):
    """Malformed descriptor values raise the owning validation error."""

    with pytest.raises(RepoDefinitionError):
        Store.from_definition({
            "kind": "dir", "path": str(tmp_path / "store"), "query_index": [],
        })


def test_store_cache_reuses_matching_physical_identity_and_preserves_settings(tmp_path):
    """Store, Repo, and existing opens share matching cache entries only."""

    directory = DirStore(tmp_path / "directory", query_index="none")
    alias = directory.base_dir + "/."
    repo = Repo(directory)
    dryml.configure(repo=repo)
    source = directory.to_definition()
    alias_definition = deepcopy(source)
    alias_definition["path"] = alias
    different_policy = deepcopy(source)
    different_policy["query_index"] = "memory"

    rebuilt = None
    with session.resource_cache():
        assert Store.from_definition(alias_definition) is directory
        assert DirStore.open_existing(directory.base_dir, query_index="none") is directory
        rebuilt = Repo.from_definition(repo.to_definition())
        assert rebuilt.default_store is directory
        with pytest.raises(RuntimeError, match="resource cache"):
            rebuilt.close(flush=False)
        different = Store.from_definition(different_policy)
        assert different is not directory
        assert Store.from_definition(different_policy) is different
        with pytest.raises(RuntimeError, match="resource cache"):
            different.close()

    assert rebuilt is repo

    first = Store.from_definition(source)
    second = Store.from_definition(source)
    try:
        assert first is not second
        assert first is not directory
        assert second is not directory
    finally:
        first.close()
        second.close()
        repo.close(flush=False)
        directory.close()


def test_cached_zip_hit_keeps_dirty_transaction_and_cache_cleanup_never_commits(tmp_path):
    """Cache hits retain dirty Zip transactions while owned cleanup discards buffers."""

    path = tmp_path / "store.zip"
    archive = _committed_zip(path)
    before = path.read_bytes()
    definition = archive.to_definition()
    repo = Repo(archive)
    dryml.configure(repo=repo)
    record = _record("dirty")

    with session.resource_cache():
        archive.write_definition_record(record)
        assert archive._archive_dirty
        assert Store.from_definition(definition) is archive

    assert path.read_bytes() == before
    archive.close()
    repo.close(flush=False)
    dryml.configure(repo=None)
    reopened = ZipStore.open_existing(path)
    try:
        assert reopened.read_definition_record(record.digest) is None
    finally:
        reopened.close()

    with session.resource_cache():
        owned = Store.from_definition(definition)
        assert owned is not archive
        owned.write_definition_record(record)
        owned.commit()

    reopened = ZipStore.open_existing(path)
    try:
        assert reopened.read_definition_record(record.digest) == record
    finally:
        reopened.close()

    committed = path.read_bytes()
    discarded = _record("discarded")

    with session.resource_cache():
        owned = Store.from_definition(definition)
        owned.write_definition_record(discarded)
        assert owned._archive_dirty

    assert path.read_bytes() == committed
    reopened = ZipStore.open_existing(path)
    try:
        assert reopened.read_definition_record(record.digest) == record
        assert reopened.read_definition_record(discarded.digest) is None
    finally:
        reopened.close()


def test_cached_zip_commit_rekeys_its_live_transaction(tmp_path):
    """An owned commit remains the one reusable transaction after replacement."""

    archive = _committed_zip(tmp_path / "store.zip")
    definition = archive.to_definition()
    try:
        with session.resource_cache():
            cached = Store.from_definition(definition)
            cached.write_definition_record(_record("committed"))
            cached.commit()

            assert Store.from_definition(definition) is cached
    finally:
        archive.close()


def test_cached_zip_does_not_rekey_after_external_archive_replacement(tmp_path):
    """A replaced archive cannot make an old buffered handle a cache hit."""

    archive = _committed_zip(tmp_path / "store.zip")
    definition = archive.to_definition()
    replacement = ZipStore.open_existing(archive.archive_path)
    try:
        with session.resource_cache():
            cached = Store.from_definition(definition)
            replacement.write_definition_record(_record("replacement"))
            replacement.commit()

            assert Store.from_definition(definition) is not cached
    finally:
        replacement.close()
        archive.close()


def test_cached_dirstore_does_not_rekey_after_root_replacement(tmp_path):
    """An externally replaced direct root cannot rekey its old live handle."""

    source = DirStore(tmp_path / "store", query_index="none")
    replacement = DirStore(tmp_path / "replacement", query_index="none")
    definition = source.to_definition()
    displaced = tmp_path / "displaced"
    try:
        with session.resource_cache():
            cached = Store.from_definition(definition)
            os.rename(source.base_dir, displaced)
            os.rename(replacement.base_dir, source.base_dir)

            assert Store.from_definition(definition) is not cached
    finally:
        source.close()
        replacement.close()


def test_cache_teardown_preserves_primary_error_and_retains_failed_store(tmp_path, monkeypatch):
    """A teardown close failure transfers retry ownership without replacing work errors."""

    store = DirStore(tmp_path / "store", query_index="none")
    definition = store.to_definition()
    original_close = DirStore.close
    failed = set()

    def fail_once(self):
        if self is not store and id(self) not in failed:
            failed.add(id(self))
            raise OSError("close failed")
        return original_close(self)

    monkeypatch.setattr(DirStore, "close", fail_once)
    try:
        with pytest.raises(ValueError, match="primary") as raised:
            with session.resource_cache():
                Store.from_definition(definition)
                raise ValueError("primary")

        cleanup = raised.value.repo_cleanup_error
        assert isinstance(cleanup, RepoReconstructionError)
        assert cleanup.cleanup() is None
    finally:
        store.close()


def test_cache_teardown_preserves_primary_control_flow_and_retains_failed_store(tmp_path, monkeypatch):
    """Teardown failure does not replace interruption with its cleanup error."""

    store = DirStore(tmp_path / "store", query_index="none")
    definition = store.to_definition()
    original_close = DirStore.close
    failed = set()

    def fail_once(self):
        if self is not store and id(self) not in failed:
            failed.add(id(self))
            raise OSError("close failed")
        return original_close(self)

    monkeypatch.setattr(DirStore, "close", fail_once)
    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            with session.resource_cache():
                Store.from_definition(definition)
                raise KeyboardInterrupt

        cleanup = raised.value.repo_cleanup_error
        assert isinstance(cleanup, RepoReconstructionError)
        assert cleanup.cleanup() is None
    finally:
        store.close()


def test_cache_provisional_failure_retains_failed_store_for_retry(tmp_path, monkeypatch):
    """A failed cache admission leaves a single cleanup owner for its Store."""

    store = DirStore(tmp_path / "store", query_index="none")
    definition = store.to_definition()
    original_close = DirStore.close

    def reject_lease(cache, resource):
        raise RuntimeError("lease rejected")

    def reject_close(self):
        if self is not store:
            raise OSError("close failed")
        return original_close(self)

    monkeypatch.setattr(core_session, "_lease_resource", reject_lease)
    monkeypatch.setattr(DirStore, "close", reject_close)
    try:
        with session.resource_cache():
            with pytest.raises(RuntimeError, match="lease rejected") as raised:
                Store.from_definition(definition)

        cleanup = raised.value.repo_cleanup_error
        assert isinstance(cleanup, RepoReconstructionError)
        monkeypatch.setattr(DirStore, "close", original_close)
        assert cleanup.cleanup() is None
    finally:
        store.close()
