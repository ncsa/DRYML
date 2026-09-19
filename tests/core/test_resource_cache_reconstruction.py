"""Session resource-cache Repo semantic reuse and cleanup contracts."""

from copy import deepcopy
from pathlib import Path

import pytest

import dryml
from dryml import session
from dryml.core import Repo, RepoReconstructionError
from dryml.core import repo_definition
from dryml.core.repo_plan import SaveRouting
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.core.store.zip import ZipStore


@pytest.fixture(autouse=True)
def _reset_core_session():
    """Keep cache leases and selected Repos isolated between tests."""

    dryml.reset_config()
    try:
        yield
    finally:
        dryml.reset_config()


def _definition(tmp_path, name, *, config=None, stores=1, routing=None):
    """Create portable existing Store authority and one detached Repo definition."""

    handles = [DirStore(tmp_path / f"{name}-{index}", query_index="none") for index in range(stores)]
    repo = Repo(handles, config=config, save_routing=routing)
    return repo, handles, repo.to_definition()


def test_cache_reuses_equivalent_repos_and_leases_the_reconstructed_handle(tmp_path):
    """Equivalent detached Repo requests return one cache-owned leased Repo."""

    source, handles, definition = _definition(tmp_path, "source", config={"nested": {"value": [1]}})
    try:
        with session.resource_cache() as cache:
            first = Repo.from_definition(definition)
            second = Repo.from_definition(definition)

            assert second is first
            assert first in cache.repos
            assert first.default_store in cache.stores
            with pytest.raises(RuntimeError, match="resource cache"):
                first.close(flush=False)
        assert first._closed
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_cache_reuses_current_borrowed_repo_after_config_mutation(tmp_path):
    """A selected borrowed Repo keeps a semantic record as its config changes."""

    source, handles, _ = _definition(tmp_path, "source", config={"mode": "before"})
    dryml.configure(repo=source)
    try:
        with session.resource_cache() as cache:
            source.set_config("mode", "after")

            assert Repo.from_definition(source.to_definition()) is source
            assert cache.repos == (source,)
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_cache_rejects_unsupported_borrowed_selection_without_changing_session(tmp_path):
    """An invalid selected Repo is neither admitted nor installed on failure."""

    current = Repo(DirStore(tmp_path / "current", query_index="none"))
    unsupported = Repo(
        DirStore(tmp_path / "unsupported", query_index="none"), config={"value": object()},
    )
    dryml.configure(repo=current)
    try:
        with session.resource_cache() as cache:
            with pytest.raises(RuntimeError, match="unsupported or closed"):
                dryml.configure(repo=unsupported)

            assert dryml.status()["repo"] is current
            assert cache.repos == (current,)
    finally:
        current.close(flush=False)
        current.default_store.close()
        unsupported.close(flush=False)
        unsupported.default_store.close()


def test_cache_rejects_closed_borrowed_selection_without_changing_session(tmp_path):
    """A closed selected Repo is neither admitted nor installed on failure."""

    current = Repo(DirStore(tmp_path / "current", query_index="none"))
    closed = Repo(DirStore(tmp_path / "closed", query_index="none"))
    closed.close(flush=False)
    dryml.configure(repo=current)
    try:
        with session.resource_cache() as cache:
            with pytest.raises(RuntimeError, match="unsupported or closed"):
                dryml.configure(repo=closed)

            assert dryml.status()["repo"] is current
            assert cache.repos == (current,)
    finally:
        current.close(flush=False)
        current.default_store.close()
        closed.default_store.close()


def test_distinct_repo_settings_share_stores_but_not_repo_identity(tmp_path):
    """Repo matching includes mutable settings while Store identity remains shared."""

    source, handles, definition = _definition(tmp_path, "source", config={"mode": "one"}, stores=2)
    changed = deepcopy(definition.to_data())
    changed["settings"]["config"] = {"mode": "two"}
    changed["settings"]["lease_duration"] = 12.0
    changed["settings"]["save_objs_on_deletion"] = True
    changed["stores"] = list(reversed(changed["stores"]))
    changed_definition = type(definition).from_data(changed)
    try:
        with session.resource_cache():
            first = Repo.from_definition(definition)
            second = Repo.from_definition(changed_definition)

            assert first is not second
            assert set(first.stores) == set(second.stores)
            assert second.config == {"mode": "two"}
            assert second._lease_duration == 12.0
            assert second.save_objs_on_deletion
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_repo_reconciliation_handles_nested_mutation_before_first_new_key_lookup(tmp_path):
    """A changed Repo cannot create a duplicate when a canonical Y entry exists."""

    source, handles, definition = _definition(tmp_path, "source", config={"nested": {"mode": "x"}})
    changed = deepcopy(definition.to_data())
    changed["settings"]["config"] = {"nested": {"mode": "y"}}
    changed_definition = type(definition).from_data(changed)
    try:
        with session.resource_cache():
            x = Repo.from_definition(definition)
            y = Repo.from_definition(changed_definition)
            x.config["nested"]["mode"] = "y"

            assert Repo.from_definition(changed_definition) is y
            assert x is not y
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_repo_reconciliation_rekeys_two_changed_records_without_false_collision(tmp_path):
    """A full snapshot handles an X/Y key swap before either lookup is resolved."""

    source, handles, definition = _definition(tmp_path, "source", config={"mode": "x"})
    changed = deepcopy(definition.to_data())
    changed["settings"]["config"] = {"mode": "y"}
    changed_definition = type(definition).from_data(changed)
    try:
        with session.resource_cache():
            x = Repo.from_definition(definition)
            y = Repo.from_definition(changed_definition)
            x.set_config("mode", "y")
            y.set_config("mode", "x")

            assert Repo.from_definition(definition) is y
            assert Repo.from_definition(changed_definition) is x
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_dynamic_distinct_repo_request_reuses_only_its_overlapping_store(tmp_path):
    """A later [S1, S2] request shares S1 without preopening its branch resources."""

    first_source = DirStore(tmp_path / "s0", query_index="none")
    shared = DirStore(tmp_path / "s1", query_index="none")
    later_source = DirStore(tmp_path / "s2", query_index="none")
    first = Repo([first_source, shared])
    later = Repo([shared, later_source], config={"later": True}, save_routing="per-object")
    try:
        with session.resource_cache():
            rebuilt_first = Repo.from_definition(first.to_definition())
            rebuilt_later = Repo.from_definition(later.to_definition())

            assert rebuilt_first is not rebuilt_later
            assert rebuilt_first.stores[1] is rebuilt_later.stores[0]
            assert len({id(store) for store in (*rebuilt_first.stores, *rebuilt_later.stores)}) == 3
            assert rebuilt_later.save_routing.graph_mode == "per-object"
    finally:
        first.close(flush=False)
        later.close(flush=False)
        first_source.close()
        shared.close()
        later_source.close()


def test_same_key_reentrancy_rejects_without_losing_the_outer_request(tmp_path, monkeypatch):
    """A recursive request fails explicitly instead of waiting on its own staging key."""

    source, handles, definition = _definition(tmp_path, "source")
    original_init = Repo.__init__
    attempted = False

    def recursive_init(self, *args, **kwargs):
        nonlocal attempted
        if not attempted:
            attempted = True
            with pytest.raises(RuntimeError, match="Recursive reconstruction"):
                Repo.from_definition(definition)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(Repo, "__init__", recursive_init)
    try:
        with session.resource_cache():
            assert isinstance(Repo.from_definition(definition), Repo)
        assert attempted
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_same_store_key_reentrancy_rejects_without_losing_the_outer_request(tmp_path, monkeypatch):
    """A Store opener cannot recursively reopen its pending physical key."""

    source = DirStore(tmp_path / "source", query_index="none")
    definition = source.to_definition()
    original_init = DirStore.__init__
    attempted = False

    def recursive_init(self, *args, **kwargs):
        nonlocal attempted
        if kwargs.get("_existing_only") and not attempted:
            attempted = True
            with pytest.raises(RuntimeError, match="Recursive reconstruction.*Store"):
                Store.from_definition(definition)
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(DirStore, "__init__", recursive_init)
    try:
        with session.resource_cache():
            assert isinstance(Store.from_definition(definition), DirStore)
        assert attempted
    finally:
        source.close()


def test_failed_staged_repo_request_preserves_prior_hits_without_new_members(tmp_path, monkeypatch):
    """A later failed request closes only its provisional miss and retains prior hits."""

    first_source = DirStore(tmp_path / "first", query_index="none")
    later_source = ZipStore(tmp_path / "later.zip")
    later_source._archive_dirty = True
    later_source.commit()
    first = Repo(first_source)
    later = Repo([first_source, later_source])
    original_open = ZipStore.open_existing
    try:
        with session.resource_cache() as cache:
            cached = Repo.from_definition(first.to_definition())
            monkeypatch.setattr(
                ZipStore,
                "open_existing",
                classmethod(lambda cls, path: (_ for _ in ()).throw(OSError("later open failed"))),
            )
            with pytest.raises(Exception, match="required Store authority"):
                Repo.from_definition(later.to_definition())

            assert Repo.from_definition(first.to_definition()) is cached
            assert cache.stores == (cached.default_store,)
    finally:
        monkeypatch.setattr(ZipStore, "open_existing", original_open)
        first.close(flush=False)
        later.close(flush=False)
        first_source.close()
        later_source.close()


def test_cached_repo_rejects_physical_membership_changes_before_opening(tmp_path):
    """Cache leases allow reordering but reject physical additions before opening."""

    source, handles, definition = _definition(tmp_path, "source", stores=2)
    missing = tmp_path / "must-not-open.zip"
    new_handle = DirStore(tmp_path / "new-handle", query_index="none")
    try:
        with session.resource_cache():
            rebuilt = Repo.from_definition(definition)
            with pytest.raises(RuntimeError, match="topology"):
                rebuilt.add_store(missing)
            with pytest.raises(RuntimeError, match="topology"):
                rebuilt.set_default_store(missing)
            with pytest.raises(RuntimeError, match="topology"):
                rebuilt.add_store(new_handle)
            assert not missing.exists()

            rebuilt.set_default_store(rebuilt.stores[1])
            rebuilt.set_config("updated", {"nested": True})
            rebuilt.set_save_routing("per-object")
            assert Repo.from_definition(rebuilt.to_definition()) is rebuilt
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()
        new_handle.close()


def test_cached_repo_teardown_closes_repo_before_its_owned_store(tmp_path, monkeypatch):
    """Cache cleanup closes dependent Repos before cache-owned Store handles."""

    source, handles, definition = _definition(tmp_path, "source")
    order = []
    original_repo_close = Repo.close
    original_store_close = DirStore.close
    rebuilt = None

    def observe_repo_close(self, *args, **kwargs):
        if self is rebuilt:
            order.append("repo")
        return original_repo_close(self, *args, **kwargs)

    def observe_store_close(self):
        if rebuilt is not None and self in rebuilt.stores:
            order.append("store")
        return original_store_close(self)

    monkeypatch.setattr(Repo, "close", observe_repo_close)
    monkeypatch.setattr(DirStore, "close", observe_store_close)
    try:
        with session.resource_cache():
            rebuilt = Repo.from_definition(definition)
        assert order.index("repo") < order.index("store")
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_cached_repo_close_failure_retains_repo_and_store_for_retry(tmp_path, monkeypatch):
    """A failed Repo close transfers one retry owner with its Store dependency."""

    source, handles, definition = _definition(tmp_path, "source")
    original_close = Repo.close
    rebuilt = None
    failed = False

    def fail_once(self, *args, **kwargs):
        nonlocal failed
        if self is rebuilt and not failed:
            failed = True
            raise OSError("repo close failed")
        return original_close(self, *args, **kwargs)

    monkeypatch.setattr(Repo, "close", fail_once)
    try:
        with pytest.raises(RepoReconstructionError) as raised:
            with session.resource_cache():
                rebuilt = Repo.from_definition(definition)

        cleanup = raised.value
        assert cleanup.cleanup() is None
        assert rebuilt._closed
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_failed_repo_publication_closes_repo_first_and_retains_staged_store(tmp_path, monkeypatch):
    """A post-build failure keeps one failed Repo and its staged Store for retry."""

    first_source, first_handles, first_definition = _definition(tmp_path, "first")
    second_source, second_handles, second_definition = _definition(tmp_path, "second")
    original_key = repo_definition._repo_cache_key_from_repo
    original_close = Repo.close
    rebuilt = None
    rebuilt_key_calls = 0

    try:
        with session.resource_cache() as cache:
            prior = Repo.from_definition(first_definition)

            original_init = Repo.__init__

            def capture_init(self, *args, **kwargs):
                nonlocal rebuilt
                result = original_init(self, *args, **kwargs)
                rebuilt = self
                return result

            def reject_rebuilt_key(repo):
                nonlocal rebuilt_key_calls
                if repo is rebuilt:
                    rebuilt_key_calls += 1
                    if rebuilt_key_calls == 3:
                        raise RuntimeError("publication validation failed")
                return original_key(repo)

            def fail_rebuilt_close(self, *args, **kwargs):
                if self is rebuilt:
                    raise OSError("repo close failed")
                return original_close(self, *args, **kwargs)

            with monkeypatch.context() as patched:
                patched.setattr(Repo, "__init__", capture_init)
                patched.setattr(repo_definition, "_repo_cache_key_from_repo", reject_rebuilt_key)
                patched.setattr(Repo, "close", fail_rebuilt_close)
                with pytest.raises(RuntimeError, match="publication validation failed") as raised:
                    Repo.from_definition(second_definition)

            cleanup = raised.value.repo_cleanup_error
            assert cleanup._retained_repos == [rebuilt]
            assert cleanup._retained_stores == [rebuilt.default_store]
            assert Repo.from_definition(first_definition) is prior
            assert cache.stores == (prior.default_store,)

        assert cleanup.cleanup() is None
    finally:
        first_source.close(flush=False)
        second_source.close(flush=False)
        for handle in (*first_handles, *second_handles):
            handle.close()


def test_builder_failure_retains_constructed_repo_and_staged_store_once(tmp_path, monkeypatch):
    """A constructor failure cannot lose a partially built Repo or its Store."""

    source, handles, definition = _definition(tmp_path, "source")
    original_init = Repo.__init__
    original_close = Repo.close
    rebuilt = None
    cache = None
    try:
        def fail_after_init(self, *args, **kwargs):
            nonlocal rebuilt
            original_init(self, *args, **kwargs)
            rebuilt = self
            raise RuntimeError("builder failed")

        def fail_rebuilt_close(self, *args, **kwargs):
            if self is rebuilt:
                raise OSError("repo close failed")
            return original_close(self, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(Repo, "__init__", fail_after_init)
            patched.setattr(Repo, "close", fail_rebuilt_close)
            with pytest.raises(RepoReconstructionError, match="cleanup requires retry") as raised:
                with session.resource_cache() as cache:
                    Repo.from_definition(definition)

        cleanup = raised.value
        assert cleanup._retained_repos == [rebuilt]
        assert cleanup._retained_stores == [rebuilt.default_store]
        assert cache.repos == ()
        assert cache.stores == ()
        assert cleanup.cleanup() is None
    finally:
        source.close(flush=False)
        for handle in handles:
            handle.close()


def test_cache_retry_owner_deduplicates_shared_store_dependencies(tmp_path, monkeypatch):
    """Two failed cached Repos transfer their one shared Store to one retry owner."""

    store = DirStore(tmp_path / "shared", query_index="none")
    first_source = Repo(store, config={"repo": "first"})
    second_source = Repo(store, config={"repo": "second"})
    rebuilt = []
    original_close = Repo.close
    try:
        def fail_rebuilt_close(self, *args, **kwargs):
            if any(self is repo for repo in rebuilt):
                raise OSError("repo close failed")
            return original_close(self, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(Repo, "close", fail_rebuilt_close)
            with pytest.raises(RepoReconstructionError) as raised:
                with session.resource_cache():
                    rebuilt.extend((
                        Repo.from_definition(first_source.to_definition()),
                        Repo.from_definition(second_source.to_definition()),
                    ))

        cleanup = raised.value
        shared = rebuilt[0].default_store
        assert cleanup._retained_repos == list(reversed(rebuilt))
        assert cleanup._retained_stores == [shared]

        closed = 0
        original_store_close = DirStore.close

        def count_shared_close(self):
            nonlocal closed
            if self is shared:
                closed += 1
            return original_store_close(self)

        monkeypatch.setattr(DirStore, "close", count_shared_close)
        assert cleanup.cleanup() is None
        assert closed == 1
    finally:
        first_source.close(flush=False)
        second_source.close(flush=False)
        store.close()


def test_cache_repo_cleanup_discards_dirty_zip_without_deletion_save(tmp_path, monkeypatch):
    """Non-flushing cache teardown neither commits buffered state nor invokes deletion save."""

    archive = ZipStore(tmp_path / "source.zip")
    archive._archive_dirty = True
    archive.commit()
    source = Repo(archive)
    definition = source.to_definition()
    before = archive.archive_path
    before_bytes = Path(before).read_bytes()
    saved = []
    try:
        with session.resource_cache():
            rebuilt = Repo.from_definition(definition)
            rebuilt.save_objs_on_deletion = True
            rebuilt.save = lambda value: saved.append(value)
            rebuilt.stores[0]._archive_dirty = True

        assert saved == []
        assert Path(before).read_bytes() == before_bytes
        reopened = ZipStore.open_existing(before)
        try:
            assert not reopened._archive_dirty
        finally:
            reopened.close()
    finally:
        source.close(flush=False)
        archive.close()
