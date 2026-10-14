from io import BytesIO
import os
from contextlib import contextmanager
from threading import Thread

import pytest

from dryml.core.store.store import StoreCapabilityError, StorePublicationCapabilities
from dryml.core.store.zip import ZipStore
from dryml.core import Repo, Object
from dryml.core.store.dir import DirStore


class ArchiveRoot(Object):
    """Stateless snapshot root for archive authority/lifetime regression checks."""

    def __init__(self):
        pass


def test_capability_preflight_fails_closed_before_local_state_hooks(tmp_path):
    caps = StorePublicationCapabilities(False, False, False, False, False)
    with pytest.raises(StoreCapabilityError, match="writable"):
        caps.require_writable("save", local_state=True)

    store = ZipStore(BytesIO())
    try:
        with pytest.raises(StoreCapabilityError, match="writable"):
            store.preflight_publication("save", local_state=True)
    finally:
        store.close()


def test_read_cut_expiry_clear_close_and_error_cleanup(tmp_path):
    """Cut evidence expires without changing borrowed handles or live cache flags."""
    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    with pytest.raises(KeyboardInterrupt):
        with repo._read_cut() as cut:
            assert repo._topology_leases == 1
            with pytest.raises(RuntimeError, match="topology lease"):
                repo.close(flush=False)
            with pytest.raises(RuntimeError, match="topology lease"):
                repo.add_store(DirStore(tmp_path / "other", query_index="none"))
            raise KeyboardInterrupt()
    assert repo._topology_leases == 0
    assert not cut.active
    with pytest.raises(RuntimeError, match="inactive"):
        cut.snapshot(store, "a" * 64)
    with pytest.raises(RuntimeError, match="re-entered"):
        cut.__enter__()
    with repo._read_cut() as cleared:
        repo.clear_cache(strong=False, weak=False)
        with pytest.raises(RuntimeError, match="cleared"):
            cleared.check()
    repo.close(flush=False)
    with pytest.raises(RuntimeError, match="close"):
        repo._read_cut().__enter__()
    assert store.read_state_ref_record("a" * 64) is None


def test_foreign_thread_cannot_read_or_unwind_cut(tmp_path):
    """A rejected foreign unwind preserves the owning thread's resources."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    errors = []
    with repo._read_cut() as cut:
        def foreign():
            for operation in (cut.check, lambda: cut.__exit__(None, None, None)):
                try:
                    operation()
                except RuntimeError as error:
                    errors.append(str(error))
        thread = Thread(target=foreign)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(errors) == 2 and all("thread" in error for error in errors)
        cut.check()
        assert repo._topology_leases == 1
    assert repo._topology_leases == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires inherited-process boundary")
def test_inherited_cut_rejects_reads_entry_and_unwind_before_owner_locks(tmp_path):
    """A child cannot consume proof, acquire inherited locks, or release the parent."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    with repo._read_cut() as cut:
        pid = os.fork()
        if pid == 0:
            class ForbiddenLock:
                def __enter__(self):
                    os._exit(2)
                def __exit__(self, *args):
                    os._exit(3)
            repo._configuration_lock = ForbiddenLock()
            for operation in (
                cut.check, lambda: cut.__exit__(None, None, None),
                lambda: repo._read_cut().__enter__(), repo.clear_cache,
                lambda: repo.close(flush=False),
            ):
                try:
                    operation()
                except RuntimeError:
                    continue
                except BaseException:
                    os._exit(4)
                os._exit(5)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        cut.check()
        assert repo._topology_leases == 1
    assert repo._topology_leases == 0


def test_fence_failure_unwinds_prior_fences_and_topology(tmp_path, monkeypatch):
    """Failed ordered fence admission releases only successfully entered owners."""
    first = DirStore(tmp_path / "first", query_index="none")
    second = DirStore(tmp_path / "second", query_index="none")
    events = []

    @contextmanager
    def first_fence():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    @contextmanager
    def fail_fence():
        raise KeyboardInterrupt()
        yield

    monkeypatch.setattr(first, "authority_fence_key", lambda: "1")
    monkeypatch.setattr(second, "authority_fence_key", lambda: "2")
    monkeypatch.setattr(first, "authority_read_fence", first_fence)
    monkeypatch.setattr(second, "authority_read_fence", fail_fence)
    repo = Repo([second, first])
    with pytest.raises(KeyboardInterrupt):
        with repo._read_cut():
            pytest.fail("fence failure must precede reads")
    assert events == ["enter", "exit"]
    assert repo._topology_leases == 0


def test_unsupported_fence_is_not_replaced_by_validation_memo(tmp_path, monkeypatch):
    """An unsupported authority cut keeps its established capability failure."""
    store = DirStore(tmp_path / "store", query_index="none")

    def unsupported():
        raise StoreCapabilityError("no stable authority cut")

    monkeypatch.setattr(store, "authority_fence_key", unsupported)
    repo = Repo(store)
    with pytest.raises(StoreCapabilityError, match="no stable"):
        with repo._read_cut():
            pytest.fail("unsupported fence must precede reads")
    assert repo._topology_leases == 0


def test_zip_cut_keeps_extractions_distinct_and_checks_cached_liveness(tmp_path, fixed_snapshot_environment):
    """Equal archive names never imply equal transaction views or live handles."""
    first = ZipStore(tmp_path / "store.zip")
    writer = Repo(first)
    state = writer.save_object(ArchiveRoot(repo=writer))
    first.commit()
    second = ZipStore.open_existing(tmp_path / "store.zip")
    repo = Repo([first, second])
    try:
        with repo._read_cut() as cut:
            assert first._can_reuse_exact_validation()
            assert second._can_reuse_exact_validation()
            assert first.authority_fence_key() != second.authority_fence_key()
            assert first._cut_snapshot_key(cut, state.digest()) != second._cut_snapshot_key(cut, state.digest())
            assert cut.snapshot(first, state.digest())[0].state_ref == state
            assert cut.snapshot(second, state.digest())[0].state_ref == state
            assert len(cut.snapshots) == 2
            second.close()
            with pytest.raises(RuntimeError, match="closed"):
                cut.snapshot(second, state.digest())
    finally:
        first.close()
        second.close()


def test_exact_inputs_and_captured_selections_precede_fence_admission(tmp_path, monkeypatch, fixed_snapshot_environment):
    """Invalid selectors fail before unsupported fences or authority reads."""
    from dryml.core.materialization import build_exact_state_load_plan

    store = DirStore(tmp_path / "store", query_index="none")
    repo = Repo(store)
    state = repo.save_object(ArchiveRoot(repo=repo))
    detached = DirStore(tmp_path / "detached", query_index="none")
    attempts = []

    def unsupported():
        attempts.append(True)
        raise StoreCapabilityError("unexpected fence admission")

    monkeypatch.setattr(store, "authority_fence_key", unsupported)
    with pytest.raises(TypeError, match="StateRef"):
        build_exact_state_load_plan(repo, object())
    with pytest.raises(TypeError, match="Store handle"):
        build_exact_state_load_plan(repo, state, source_stores={state: None})
    with pytest.raises(ValueError, match="connected"):
        build_exact_state_load_plan(repo, state, source_store=detached)
    assert not attempts
    assert repo._topology_leases == 0


def test_closing_repo_cut_rejected_before_configuration_lock(tmp_path):
    """Effective closing state is checked before entering an inherited owner lock."""
    repo = Repo(DirStore(tmp_path / "store", query_index="none"))
    original_lock = repo._configuration_lock

    class ForbiddenLock:
        def __enter__(self):
            pytest.fail("closing cut acquired configuration lock")
        def __exit__(self, *args):
            pytest.fail("closing cut released configuration lock")

    repo._configuration_lock = ForbiddenLock()
    repo._closing = True
    try:
        with pytest.raises(RuntimeError, match="close"):
            repo._read_cut().__enter__()
    finally:
        repo._closing = False
        repo._configuration_lock = original_lock
