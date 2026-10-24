"""Full-key retention, native active use and incarnation-safe cleanup proofs."""

from contextlib import closing
import multiprocessing
import os
from pathlib import Path
from uuid import uuid4

import pytest

from dryml.core import Definition
from dryml.core.store.dir import DirStore
from dryml.core.store.store import StoreAuthorityError


def creation(value=1):
    return Definition(dict, value=value).concretize()


def install(store, value=1, payload=b"state"):
    stage = store.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(payload)
    return store.install_state(creation(value), stage, codec="Test", codec_version=1,
                               interpretation={}).state.key


def retained(owner, store, key, *, token=None, adopt=True):
    token = str(uuid4()) if token is None else token
    owner.retention.prepare(token, "a" * 64, ((store, (key,)),))
    owner.retention.protect(token, (store,))
    if adopt:
        owner.retention.adopt(token, (store,))
    return token


def test_shared_read_and_same_operation_cleanup_never_upgrade(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    before = store.authority_cut()
    with store.read_state(key) as source:
        assert store.delete_state(key).status == "in-use"
        assert Path(source.data_directory, "value").read_bytes() == b"state"
        assert store.authority_cut() == before
    assert store.delete_state(key).status == "removed"
    assert store.inspect_states(key.cdef_digest)[0].available is False


def test_protect_associate_release_and_full_key_store_isolation(tmp_path):
    a, b = (DirStore(tmp_path / name, query_index="none") for name in ("a", "b"))
    key, sibling, other = install(a), install(a, payload=b"sibling"), install(a, value=2)
    replica = install(b)
    assert key.codec_hash == other.codec_hash
    first = retained(a, a, key)
    recovery = retained(a, a, key)
    a.retention.release(first, (a,))
    assert a.delete_state(key).status == "retained"
    assert a.delete_state(sibling).status == "removed"
    assert a.delete_state(other).status == "removed"
    assert b.delete_state(replica).status == "removed"
    a.retention.release(recovery, (a,))
    assert a.delete_state(key).status == "removed"
    assert Path(a._bucket_path(key.cdef_digest), "def.pkl").is_file()
    assert Path(a.state_data_path(key)).parent.joinpath("manifest.record").is_file()


def test_grant_addresses_normalize_equivalent_paths_but_reject_misplaced_records(tmp_path, monkeypatch):
    """Native and portable path spellings denote one grant, never another address."""
    from dryml.core.store import retention

    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    retained(store, store, key)
    expected = retention.grants(store, key)
    assert len(expected) == 1
    grant_path = retention._grant_path
    monkeypatch.setattr(retention, "_grant_path", lambda *args: os.path.join(
        os.path.dirname(grant_path(*args)), ".", os.path.basename(grant_path(*args))))
    assert retention.grants(store, key) == expected
    assert store.delete_state(key).status == "retained"
    path = Path(grant_path(store, key, expected[0]))
    path.rename(path.with_name("misplaced.record"))
    with pytest.raises(StoreAuthorityError, match="grant"):
        retention.grants(store, key)


def test_uncertain_owner_keeps_grants_and_abandonment_fences_adoption(tmp_path):
    owner, target = (DirStore(tmp_path / name, query_index="none") for name in ("owner", "target"))
    key = install(target)
    token = retained(owner, target, key, adopt=False)
    owner.close()
    assert target.delete_state(key).status == "retained"
    owner = DirStore(tmp_path / "owner", query_index="none")
    owner.retention.abandon(token)
    with pytest.raises(StoreAuthorityError, match="abandon"):
        owner.retention.adopt(token, (target,))
    with pytest.raises(StoreAuthorityError, match="abandon"):
        owner.retention.protect(token, (target,))
    owner.retention.release(token, (target,))
    assert target.delete_state(key).status == "removed"


def test_release_pending_restarts_exact_targets_after_second_store_failure(tmp_path, monkeypatch):
    owner, a, b = (DirStore(tmp_path / name, query_index="none") for name in ("owner", "a", "b"))
    ka, kb = install(a), install(b)
    token = str(uuid4())
    owner.retention.prepare(token, "b" * 64, ((a, (ka,)), (b, (kb,))))
    # A succeeds; an unavailable second capture/target cannot establish adoption.
    owner.retention.protect(token, (a,))
    with pytest.raises(StoreAuthorityError, match="target"):
        owner.retention.adopt(token, (a,))
    owner.retention.abandon(token)
    original = a._prepare_and_apply
    monkeypatch.setattr(a, "_prepare_and_apply", lambda *args, **kw: (_ for _ in ()).throw(OSError("release denied")))
    with pytest.raises(OSError, match="release denied"):
        owner.retention.release(token, (a, b))
    assert owner.retention.read(token).disposition == "release-pending"
    assert owner.retention.read(token).released_targets == (b.retention_domain,)
    monkeypatch.setattr(a, "_prepare_and_apply", original)
    assert a.delete_state(ka).status == "retained"
    owner.close()
    owner = DirStore(tmp_path / "owner", query_index="none")
    assert owner.retention.pending() == (token,)
    monkeypatch.setattr(os, "walk", lambda *a, **kw: pytest.fail("all-history scan"))
    monkeypatch.setattr(b, "_prepare_and_apply", lambda *a, **kw: pytest.fail("acknowledged target was revisited"))
    assert owner.retention.release(token, (a, b)).disposition == "released"
    assert owner.retention.pending() == ()
    assert a.delete_state(ka).status == "removed"


def test_garbage_failure_then_new_incarnation_obsoletes_old_cleanup(tmp_path, monkeypatch):
    import dryml.core.store.retention as retention
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    original = retention.remove_private_data
    with monkeypatch.context() as patch:
        patch.setattr(retention, "remove_private_data", lambda *args: (_ for _ in ()).throw(OSError("garbage denied")))
        outcome = store.delete_state(key)
    assert outcome.status == "cleanup-pending" and outcome.cleanup_error
    assert not store.inspect_states(key.cdef_digest)[0].available
    assert install(store) == key
    newer = store.inspect_states(key.cdef_digest)[0].incarnation
    assert newer != outcome.incarnation
    assert store.retry_state_removal(key, outcome.incarnation).status == "obsolete"
    with store.read_state(key) as source:
        assert Path(source.data_directory, "value").read_bytes() == b"state"
    assert original is retention.remove_private_data


def test_retention_and_withdrawal_feed_index_and_survive_reverse_index_loss(tmp_path):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    token = retained(store, store, key)
    index = store.open_query_index()
    assert len(index.projection_rows("retention")) == 1
    index.close()
    index.path.unlink()
    assert store.delete_state(key).status == "retained"
    store.retention.release(token, (store,))
    assert index.projection_rows("retention") == ()
    assert store.delete_state(key).status == "removed"
    import json
    assert not json.loads(index.projection_rows("availability")[0][2])["available"]
    assert not store.authority_cut().pending


def _holding_reader(path, key, ready, release):
    store = DirStore(path, query_index="none")
    with store.read_state(key):
        ready.set()
        if not release.wait(10):
            raise RuntimeError("reader barrier timed out")


@pytest.mark.timeout(25)
def test_spawned_reader_deletion_is_nonblocking(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    worker = ctx.Process(target=_holding_reader, args=(store.base_dir, key, ready, release))
    worker.start()
    try:
        assert ready.wait(10)
        assert store.delete_state(key).status == "in-use"
    finally:
        release.set()
        worker.join(10)
        if worker.is_alive():
            worker.kill()
            worker.join()
    assert worker.exitcode == 0
    assert store.delete_state(key).status == "removed"


def _finish(worker, release=None, expected=0):
    if release is not None:
        release.set()
    worker.join(10)
    if worker.is_alive():
        worker.kill()
        worker.join()
        pytest.fail("spawned retention worker failed to exit at its barrier")
    assert worker.exitcode == expected


def _protecting_worker(path, token, ready, release):
    store = DirStore(path, query_index="none")
    original = store._prepare_and_apply
    def apply(prepare, **kwargs):
        def paused():
            result = prepare()
            if "retention" in result[1]:
                ready.set()
                if not release.wait(10):
                    raise RuntimeError("grant barrier timed out")
            return result
        return original(paused, **kwargs)
    store._prepare_and_apply = apply
    store.retention.protect(token, (store,))


@pytest.mark.timeout(25)
def test_spawned_grant_acquisition_holds_key_before_writer(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    token = str(uuid4())
    store.retention.prepare(token, "a" * 64, ((store, (key,)),))
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    worker = ctx.Process(target=_protecting_worker, args=(store.base_dir, token, ready, release))
    worker.start()
    try:
        assert ready.wait(10)
        assert store.delete_state(key).status == "in-use"
        from dryml.locking import FileLock
        probe = FileLock(store._writer_lock_path)
        assert probe.acquire(blocking=False), "grant preparation retained the writer fence"
        probe.release()
    finally:
        _finish(worker, release)
    assert store.delete_state(key).status == "retained"


def _withdrawing_worker(path, key, ready, release, crash):
    import dryml.core.store.retention as retention
    store = DirStore(path, query_index="none")
    original = retention.remove_private_data
    def remove(directory):
        if directory == store.state_data_path(key):
            ready.set()
            if crash:
                Path(directory, "value").unlink()  # partial physical removal
                os._exit(23)
            if not release.wait(10):
                raise RuntimeError("withdrawal barrier timed out")
        return original(directory)
    retention.remove_private_data = remove
    store.delete_state(key)


def _publishing_worker(path, entered, finished):
    import dryml.locking as locking
    store = DirStore(path, query_index="none")
    original = locking.FileLock.acquire
    def acquire(self, **kwargs):
        if ".state-leases" in self.path:
            entered.set()
        return original(self, **kwargs)
    locking.FileLock.acquire = acquire
    install(store)
    finished.set()


@pytest.mark.timeout(30)
def test_spawned_publication_waits_outside_writer_until_withdrawal_removal_finishes(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    ctx = multiprocessing.get_context("spawn")
    ready, release, entered, finished = (ctx.Event() for _ in range(4))
    cleanup = ctx.Process(target=_withdrawing_worker, args=(store.base_dir, key, ready, release, False))
    publisher = ctx.Process(target=_publishing_worker, args=(store.base_dir, entered, finished))
    cleanup.start()
    try:
        assert ready.wait(10)
        publisher.start()
        assert entered.wait(10)
        assert not finished.is_set()
        from dryml.locking import FileLock
        probe = FileLock(store._writer_lock_path)
        assert probe.acquire(blocking=False), "publisher waited for a key while holding writer fence"
        probe.release()
    finally:
        _finish(cleanup, release)
        if publisher.pid is not None:
            _finish(publisher)
    assert finished.is_set()
    with store.read_state(key) as source:
        assert Path(source.data_directory, "value").read_bytes() == b"state"


@pytest.mark.timeout(25)
def test_spawned_crash_after_partial_removal_cannot_delete_new_retained_incarnation(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    old = store.inspect_states(key.cdef_digest)[0].incarnation
    ctx = multiprocessing.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    cleanup = ctx.Process(target=_withdrawing_worker, args=(store.base_dir, key, ready, release, True))
    cleanup.start()
    assert ready.wait(10)
    _finish(cleanup, expected=23)
    assert install(store) == key
    retained(store, store, key)
    assert store.retry_state_removal(key, old).status == "obsolete"
    with store.read_state(key) as source:
        assert Path(source.data_directory, "value").read_bytes() == b"state"


def test_abandoned_staging_checks_exact_owner_and_live_lease(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    active = store.create_local_state_staging()
    unrelated = store.create_local_state_staging()
    Path(active, "data", "prefix").write_bytes(b"partial capture")
    token = str(uuid4())
    store.retention.prepare(token, "c" * 64, (), staging=((store, active),))
    store.retention.abandon(token)
    assert store.retention.release(token, (store,)).disposition == "release-pending"
    assert Path(active).is_dir() and Path(unrelated).is_dir()
    store.release_local_state_staging(active)
    assert store.retention.release(token, (store,)).disposition == "released"
    assert not Path(active).exists() and Path(unrelated).is_dir()
    other = DirStore(store.base_dir, query_index="none")
    with pytest.raises(StoreAuthorityError, match="another"):
        other.discard_local_state_staging(unrelated)
    store.discard_local_state_staging(unrelated)


def test_same_operation_publication_reports_in_use_and_does_not_queue(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    authenticate = store._authenticate_state_data
    observed = []
    def check(*args):
        observed.append(store.delete_state(key).status)
        return authenticate(*args)
    monkeypatch.setattr(store, "_authenticate_state_data", check)
    assert install(store) == key
    assert observed == ["in-use"]
    assert not Path(store.state_data_path(key)).parent.joinpath("removals").exists()


@pytest.mark.parametrize("transition", ["protect", "release", "withdraw"])
def test_index_ack_oserror_keeps_intent_and_never_becomes_cleanup_success(tmp_path, monkeypatch, transition):
    import dryml.core.store.authority as authority
    from dryml.core.store.authority import StoreMutationError
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    token = str(uuid4())
    if transition in {"protect", "release"}:
        store.retention.prepare(token, "a" * 64, ((store, (key,)),))
    if transition == "release":
        store.retention.protect(token, (store,))
        store.retention.adopt(token, (store,))
    def fail(*args):
        raise OSError("ack failed")
    with monkeypatch.context() as patch:
        patch.setattr(authority, "acknowledge", fail)
        with pytest.raises(StoreMutationError):
            if transition == "withdraw":
                store.delete_state(key)
            else:
                getattr(store.retention, transition)(token, (store,))
    assert store.authority_cut().pending
    assert Path(store.state_data_path(key), "value").read_bytes() == b"state"
    store.open_query_index().maintain_store()
    if transition == "protect":
        assert store.delete_state(key).status == "retained"
    elif transition == "release":
        assert store.retention.read(token).disposition == "release-pending"
        assert store.retention.release(token, (store,)).disposition == "released"
    else:
        old = store.inspect_states(key.cdef_digest)[0].incarnation
        assert store.retry_state_removal(key, old).status == "removed"


def test_plan_leases_are_globally_ordered_and_growing_sets_require_revalidation(tmp_path, monkeypatch):
    from dryml.core.repo_plan import state_use_leases
    from dryml.locking import FileLock
    a, b = (DirStore(tmp_path / name, query_index="none") for name in ("a", "b"))
    ka, kb = install(a), install(b)
    observed = []
    original = FileLock.acquire
    def acquire(self, **kwargs):
        if ".state-leases" in self.path:
            observed.append(self.path)
        return original(self, **kwargs)
    monkeypatch.setattr(FileLock, "acquire", acquire)
    with state_use_leases(((b, kb), (a, ka))):
        with a.read_state(ka), b.read_state(kb):
            assert a.delete_state(ka).status == "in-use"
    expected = [store._state_lease_path(key) for store, key in sorted(
        ((a, ka), (b, kb)), key=lambda item: item[0].store_uuid)]
    assert observed == expected
    with a.state_leases((ka,)):
        with pytest.raises(StoreAuthorityError, match="release/reacquire"):
            with state_use_leases(((a, ka), (b, kb))):
                pass
    with a.writer_lock():
        with pytest.raises(StoreAuthorityError, match="before Store writer"):
            with a.state_leases((ka,)):
                pass


def test_async_child_does_not_inherit_parent_native_lease_lifetime(tmp_path):
    """Each overlapping task retains its own native reader after its parent releases."""
    import asyncio
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def child():
            with store.read_state(key):
                entered.set()
                await release.wait()
        with store.read_state(key):
            task = asyncio.create_task(child())
            await entered.wait()
        assert store.delete_state(key).status == "in-use"
        release.set()
        await task
    asyncio.run(run())
    assert store.delete_state(key).status == "removed"


def test_zip_leases_cleanup_are_view_local_and_durable_grants_are_refused(tmp_path):
    from dryml.core.store.zip import ZipStore
    from dryml.core.store.store import StoreCapabilityError
    archive = tmp_path / "store.zip"
    first = ZipStore(archive)
    key = install(first)
    first.commit()
    second = ZipStore(archive)
    with first.read_state(key) as source:
        assert first.delete_state(key).status == "in-use"
        from threading import Thread
        outcomes = []
        worker = Thread(target=lambda: outcomes.append(second.delete_state(key)))
        worker.start()
        worker.join(5)
        assert not worker.is_alive() and outcomes[0].status == "removed"
        assert Path(source.data_directory, "value").read_bytes() == b"state"
        with pytest.raises(StoreCapabilityError, match="transaction-local"):
            first.retention
        first.close()
        with pytest.raises(RuntimeError):
            source.data_directory
    assert not Path(first.base_dir).exists()
    second.close()


def test_foreign_live_staging_is_refused_before_publication_or_consumption(tmp_path):
    owner = DirStore(tmp_path / "store", query_index="none")
    other = DirStore(owner.base_dir, query_index="none")
    stage = owner.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(b"active capture")
    before = owner.authority_cut()
    with pytest.raises(StoreAuthorityError):
        other.install_state(creation(), stage, codec="Test", codec_version=1, interpretation={})
    assert Path(stage, "data", "value").read_bytes() == b"active capture"
    assert owner.authority_cut() == before


def test_replay_cannot_consume_another_allocators_live_staging(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    from dryml.core.store.authority import StoreMutationError
    store = DirStore(tmp_path / "store", query_index="none")
    stage = store.create_local_state_staging()
    Path(stage, "data", "value").write_bytes(b"active capture")
    def fail(*args):
        raise OSError("variant denied")
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_directory", fail)
        with pytest.raises(StoreMutationError) as raised:
            store.install_state(creation(), stage, codec="Test", codec_version=1, interpretation={})
    other = DirStore(store.base_dir, query_index="none")
    with pytest.raises(StoreAuthorityError, match="staging.*in-use"):
        other.replay_mutation(raised.value.mutation_id)
    assert Path(stage, "data", "value").is_file()
    store.replay_mutation(raised.value.mutation_id)


def test_partial_owner_transition_cannot_lose_release_pending_resource_ledger(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    from dryml.core.store.authority import StoreMutationError
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    token = retained(store, store, key)
    original = fs.publish_file
    def fail(source, destination, **kwargs):
        if Path(destination) == Path(store.base_dir, "retention", "pending.record"):
            raise OSError("pending ledger unavailable")
        return original(source, destination, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError):
            store.retention.release(token, (store,))
    store.recover_authority()
    for record in store.pending_mutations():
        store.acknowledge_mutation(record.intent["mutation_id"], record.intent["epoch"])
    if store.retention.read(token).disposition == "release-pending":
        assert store.retention.pending() == (token,)
    assert store.delete_state(key).status == "retained"


def test_generic_old_grant_replay_cannot_resurrect_abandoned_released_token(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    from dryml.core.store.authority import StoreMutationError
    from dryml.core.store.store import StoreCapabilityError
    owner, target = (DirStore(tmp_path / name, query_index="none") for name in ("owner", "target"))
    key = install(target)
    token = str(uuid4())
    owner.retention.prepare(token, "a" * 64, ((target, (key,)),))
    original = fs.publish_file
    def fail(source, destination, **kwargs):
        if "grants" in Path(destination).parts:
            raise OSError("grant denied")
        return original(source, destination, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError) as raised:
            owner.retention.protect(token, (target,))
    with pytest.raises(StoreAuthorityError, match="abandonment"):
        owner.retention.release(token, (target,))
    owner.retention.abandon(token)
    assert owner.retention.release(token, (target,)).disposition == "released"
    with pytest.raises(StoreCapabilityError, match="owner"):
        target.replay_mutation(raised.value.mutation_id)
    assert target.delete_state(key).status == "removed"


def test_pending_pages_and_idempotent_retention_have_no_history_or_noop_churn(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    tokens = [str(uuid4()) for _ in range(3)]
    for token in tokens:
        store.retention.prepare(token, "a" * 64, ((store, (key,)),))
    assert store.retention.pending(limit=2) == tuple(sorted(tokens)[:2])
    assert store.retention.pending(after=sorted(tokens)[1], limit=2) == (sorted(tokens)[2],)
    with monkeypatch.context() as patch:
        patch.setattr(store, "_metadata_read", lambda: pytest.fail("zero page acquired authority"))
        assert store.retention.pending(limit=0) == ()
    token = tokens[0]
    store.retention.protect(token, (store,))
    store.retention.adopt(token, (store,))
    cut = store.authority_cut()
    index = store.open_query_index()
    changes = index._connections.connection(readonly=False).total_changes
    store.retention.adopt(token, (store,))
    assert store.authority_cut() == cut
    assert index._connections.connection(readonly=False).total_changes == changes
    store.retention.release(token, (store,))
    cut = store.authority_cut()
    store.retention.release(token, (store,))
    assert store.authority_cut() == cut


def test_corrupt_grant_is_not_release_proof(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    retained(store, store, key)
    path, = Path(store.state_data_path(key)).parent.joinpath("grants").glob("*.record")
    path.write_bytes(b"truncated authoritative grant")
    with pytest.raises(StoreAuthorityError, match="Malformed"):
        store.delete_state(key)
    assert Path(store.state_data_path(key), "value").read_bytes() == b"state"


def test_prior_schema10_unsupported_retention_coverage_recovers_explicitly(tmp_path):
    import sqlite3
    from dryml.core.query.model import QueryIndexDirty, QueryIndexIncompatible
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    retained(store, store, key)
    index = store.open_query_index()
    index.close()
    with closing(sqlite3.connect(index.path)) as con, con:
        con.execute("UPDATE store_projection_coverage SET supported=0 WHERE family='retention'")
    with pytest.raises((QueryIndexDirty, QueryIndexIncompatible)):
        index.projection_rows("retention")
    index.maintain_store()
    assert len(index.projection_rows("retention")) == 1
    assert store.delete_state(key).status == "retained"


def test_in_use_deletion_probes_key_before_any_index_or_writer_fence(tmp_path, monkeypatch):
    store = DirStore(tmp_path / "store", query_index="sqlite")
    key = install(store)
    with store.read_state(key):
        monkeypatch.setattr(store, "open_query_index", lambda: pytest.fail("busy deletion opened index before key probe"))
        monkeypatch.setattr(store, "writer_lock", lambda: pytest.fail("busy deletion waited for writer before key probe"))
        assert store.delete_state(key).status == "in-use"


def test_partial_republication_leaves_exact_retry_authority_for_detached_remnants(tmp_path, monkeypatch):
    import dryml.filesystem as fs
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    assert store.delete_state(key).status == "removed"
    original = fs.publish_directory
    def fail_after_visibility(source, destination):
        original(source, destination)
        raise OSError("republication barrier failed")
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_directory", fail_after_visibility)
        with pytest.raises(OSError, match="barrier failed"):
            install(store)
    partial = store.inspect_states(key.cdef_digest)[0]
    assert not partial.available
    assert install(store) == key
    assert store.retry_state_removal(key, partial.incarnation).status == "obsolete"
    with store.read_state(key) as source:
        assert Path(source.data_directory, "value").read_bytes() == b"state"


def test_second_capture_failure_releases_only_owner_abandoned_provisional_work(tmp_path):
    from dryml.core.store.retention import grants
    owner, second = (DirStore(tmp_path / name, query_index="none") for name in ("owner", "second"))
    key = install(owner)
    durable = retained(owner, owner, key)
    stage = second.create_local_state_staging()
    token = str(uuid4())
    owner.retention.prepare(token, "b" * 64, ((owner, (key,)), (second, ())),
                            staging=((second, stage),))
    owner.retention.protect(token, (owner,))
    def capture_second():
        Path(stage, "data", "prefix").write_bytes(b"partial second capture")
        raise OSError("second capture failed")
    with pytest.raises(OSError, match="second capture"):
        capture_second()
    with pytest.raises(StoreAuthorityError, match="abandonment"):
        owner.retention.release(token, (owner, second))
    owner.retention.abandon(token)
    second.release_local_state_staging(stage)
    assert owner.retention.release(token, (owner, second)).disposition == "released"
    assert {grant.token for grant in grants(owner, key)} == {durable}
    assert not Path(stage).exists()
    assert owner.delete_state(key).status == "retained"


@pytest.mark.parametrize("policy", ["none", "sqlite"])
def test_partial_owner_ledger_repairs_from_exact_journal_without_capture(tmp_path, monkeypatch, policy):
    import dryml.filesystem as fs
    from dryml.core.store.authority import StoreMutationError
    store = DirStore(tmp_path / "store", query_index=policy)
    key = install(store)
    token = str(uuid4())
    original = fs.publish_file
    def fail(source, destination, **kwargs):
        if Path(destination) == Path(store.base_dir, "retention", "intents", token + ".record"):
            raise OSError("owner intent denied")
        return original(source, destination, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(fs, "publish_file", fail)
        with pytest.raises(StoreMutationError) as raised:
            store.retention.prepare(token, "a" * 64, ((store, (key,)),))
    with pytest.raises(StoreAuthorityError, match="recovery"):
        store.retention.pending()
    monkeypatch.setattr(store, "_prepare_node_roots", lambda *a: pytest.fail("owner repair captured/prepared graph"))
    store.replay_mutation(raised.value.mutation_id)
    assert store.retention.pending() == (token,)
    assert store.retention.read(token).resources[0].keys == (key,)
    store.retention.protect(token, (store,))
    store.retention.adopt(token, (store,))
    assert store.delete_state(key).status == "retained"


def test_cleanup_cannot_reuse_its_operations_exclusive_recovery_lease(tmp_path):
    store = DirStore(tmp_path / "store", query_index="none")
    key = install(store)
    before = store.authority_cut()
    with store.state_leases((key,), exclusive=True, blocking=False) as acquired:
        assert acquired
        assert store.delete_state(key).status == "in-use"
        assert store.authority_cut() == before
    assert store.delete_state(key).status == "removed"


def test_repo_federated_delete_continues_and_reports_every_full_key(tmp_path, monkeypatch):
    from dryml.core import Repo
    stores = [DirStore(tmp_path / name, query_index="none") for name in ("a", "b", "c")]
    keys = [install(store) for store in stores]
    assert keys[0] == keys[1] == keys[2]
    def fail(key):
        raise OSError("B withdrawal failed")
    monkeypatch.setattr(stores[1], "delete_state", fail)
    outcomes = Repo(stores).delete_states([keys[0]])
    assert [item.status for item in outcomes] == ["removed", "failed", "removed"]
    assert outcomes[1].error.args == ("B withdrawal failed",)
    assert [Path(store.state_data_path(keys[0])).exists() for store in stores] == [False, True, False]


def test_repo_saved_root_retain_release_and_inventory_without_reserialization(tmp_path, monkeypatch):
    from dryml.core import Repo
    from tests.managed.u16_fixtures import Value
    store = DirStore(tmp_path / "store", query_index="sqlite")
    repo = Repo(store)
    saved, report = repo.save_object(Value(repo=repo), report_stores=True)
    publication, = report.publications
    def forbidden(*args, **kwargs):
        pytest.fail("retain/release serialized or scanned historical inventory")
    monkeypatch.setattr(store, "iter_state_ref_records", forbidden)
    monkeypatch.setattr(store, "iter_stored_root_records", forbidden)
    monkeypatch.setattr(Value, "save_state_to_dir_imp", forbidden)
    root = repo.retain(saved)
    assert [item.status for item in repo.release(publication.retention_token, owner_store=store)] == ["retained"]
    assert [item.status for item in repo.release(root)] == ["removed"]
    inventory, = repo.inspect_states(saved.state.key.cdef_digest)
    assert inventory.domain == store.retention_domain
    assert inventory.descriptor.state.key == saved.state.key and not inventory.descriptor.available
    assert repo.inspect_states(saved.state.key.cdef_digest, available_only=True) == ()
    from dryml.core.node_prepare import NodePreparation
    digest = NodePreparation((saved,)).digest(saved, domain="dryml-saved-root/v1")
    assert store.read_node(digest, kind="saved").graph_equal(saved)


def test_repo_interrupted_cleanup_reports_completed_and_unattempted_targets(tmp_path, monkeypatch):
    from dryml.core import Repo
    stores = [DirStore(tmp_path / name, query_index="none") for name in ("a", "b", "c")]
    key = install(stores[0])
    for store in stores[1:]:
        assert install(store) == key
    def interrupt(key):
        raise KeyboardInterrupt("withdrawal interrupted")
    monkeypatch.setattr(stores[1], "delete_state", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        Repo(stores).delete_states([key])
    assert [item.status for item in caught.value.cleanup_outcomes] == ["removed", "failed", "unattempted"]
    assert Path(stores[2].state_data_path(key)).is_dir()
