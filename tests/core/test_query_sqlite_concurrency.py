import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from contextlib import closing, contextmanager

import pytest

import dryml.locking as locking_module
from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.bound_args import BoundArguments
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.model import QueryIndexBusy, QueryIndexError
from dryml.core.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
import dryml.core.query.sqlite.index as sqlite_index_module
from dryml.core.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core.query.sqlite.schema import SQLITE_QUERY_INDEX_SCHEMA_VERSION
from dryml.core.store.dir import DirStore
from dryml.core.store.records import DefinitionRecord
from dryml.core.symbol import ImportRef


pytestmark = pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")


_COMMON_WORKER_CODE = r'''
import json
from pathlib import Path
import time

from dryml.core.cdef_graph import ConcreteDefinitionGraph
from dryml.core.bound_args import BoundArguments
from dryml.core.definition import ConcreteDefinition
from dryml.core.freeze import FrozenDict, FrozenTuple
from dryml.core.query.sqlite import SQLiteQueryIndexConfig
from dryml.core.query.sqlite.index import (
    SQLiteStoreQueryIndex,
    _EncodedNode,
    _bump_generation,
    _read_generation,
    _relative_def_path,
    _resolve_definition_id,
)
from dryml.core.query.utils import stable_hash_to_blob
from dryml.core.store.dir import DirStore
from dryml.core.symbol import ImportRef


def cdef(name):
    return ConcreteDefinition._from_bound_record(
        ImportRef("builtins", "dict"),
        BoundArguments((("name", name),)),
    )


def index(path, *, timeout=0.05, retries=20):
    return SQLiteStoreQueryIndex(
        source_key="concurrency-store",
        path=path,
        config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=timeout, max_write_retries=retries),
    )


def register_root(path, name, *, timeout=0.05, retries=20):
    idx = index(path, timeout=timeout, retries=retries)
    root = cdef(name)
    result = idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
    idx.close()
    return result


def store_index(base_dir):
    store = DirStore(
        base_dir,
        query_index=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=0.05, max_write_retries=20),
    )
    return store.open_query_index()


def emit(value):
    print(json.dumps(value), flush=True)
'''


def _worker_env() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    paths = [str(root / "src"), str(root / "tests" / "core")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _run_worker(code: str, *, timeout: float = 10.0) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _COMMON_WORKER_CODE + code],
        check=False,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_worker_env(),
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def _start_worker(code: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _COMMON_WORKER_CODE + code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_worker_env(),
    )


def _collect_worker(proc: subprocess.Popen, *, timeout: float = 10.0) -> dict:
    stdout, stderr = proc.communicate(timeout=timeout)
    assert proc.returncode == 0, stderr
    lines = [line for line in stdout.splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else {}


def _spawn_register_worker(path: str, queue) -> None:
    idx = _index(Path(path), timeout=0.05, retries=20)
    try:
        root = _cdef("spawn-child")
        before = idx._connections.connection(readonly=False)
        result = idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
        after = idx._connections.connection(readonly=False)
        queue.put({"generation": result.generation, "same_connection": before is after})
    except BaseException as exc:
        queue.put({"error": repr(exc)})
    finally:
        idx.close()


def _cdef(name: str) -> ConcreteDefinition:
    return ConcreteDefinition._from_bound_record(
        ImportRef("builtins", "dict"),
        BoundArguments((("name", name),)),
    )


def _index(path: Path, *, timeout: float = 0.05, retries: int = 20) -> SQLiteStoreQueryIndex:
    return SQLiteStoreQueryIndex(
        source_key="concurrency-store",
        path=path,
        config=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=timeout, max_write_retries=retries),
    )


def _stored_root_count(path: Path) -> int:
    sqlite3 = require_sqlite()
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT COUNT(*) FROM stored_roots").fetchone()[0]
    finally:
        con.close()


def _generation(path: Path) -> int:
    sqlite3 = require_sqlite()
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT generation FROM catalog_state WHERE singleton = 1").fetchone()[0]
    finally:
        con.close()


def _save_root_definition(store: DirStore, cdef: ConcreteDefinition) -> None:
    store.write_definition_record(DefinitionRecord(cdef))


def _request_store(path):
    store = DirStore(path, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store.write_definition_record(DefinitionRecord(_cdef("request-root")))
    store.rebuild_query_index()
    return store


def _query_thread(query, stores, outcomes, *, name=None):
    stores = stores if isinstance(stores, tuple) else (stores,)

    def run():
        try:
            outcomes.append(query.explain(analyze=True))
        except BaseException as error:
            outcomes.append(error)
        finally:
            for store in stores:
                store.open_query_index()._connections.close_current()

    thread = threading.Thread(target=run, daemon=True, name=name)
    thread.start()
    return thread


@pytest.mark.timeout(40)
def test_u2_single_flight_waits_outside_resources_and_rejects_pending_admission(tmp_path, monkeypatch):
    """A waiter holds no cut or lease while its owner validates real authority."""
    import dryml.core.query.query as query_module
    from dryml.core.query.model import QueryIndexUnavailable, QueryWouldScanError

    store = _request_store(tmp_path / "store")
    index = store.open_query_index()
    query = store.query().refresh(True)
    entered, release, joined = threading.Event(), threading.Event(), threading.Event()
    calls, owner, waiter = [], [], []
    original = index.refresh
    validate = index._validate_store_roots
    original_fence = store.authority_read_fence
    held = threading.local()
    identity = index._connections._path_identity()
    generation = index.status().generation
    wait = query_module._RefreshAttempt.wait

    def refresh(policy, *, stats=None):
        calls.append(policy)
        return original(policy, stats=stats)

    @contextmanager
    def fence():
        with original_fence():
            held.depth = getattr(held, "depth", 0) + 1
            try:
                yield
            finally:
                held.depth -= 1

    def compare(con, issues, **kwargs):
        assert con.in_transaction and index._connections._leases
        assert held.depth > 0
        entered.set()
        assert release.wait(15)
        return validate(con, issues, **kwargs)

    def observe(attempt):
        request = query._refresh_bindings[0].request
        assert request._process_lock().acquire(blocking=False)
        request._process_lock().release()
        assert getattr(held, "depth", 0) == 0
        assert (os.getpid(), threading.get_ident(), True) not in index._connections._leases
        assert index._connections._leases  # only the owner retains its validation cut
        joined.set()
        return wait(attempt)

    monkeypatch.setattr(index, "refresh", refresh)
    monkeypatch.setattr(index, "_validate_store_roots", compare)
    monkeypatch.setattr(store, "authority_read_fence", fence)
    monkeypatch.setattr(query_module._RefreshAttempt, "wait", observe)
    first = _query_thread(query, store, owner)
    second = None
    try:
        assert entered.wait(10)
        with pytest.raises(QueryWouldScanError):
            query.cdefs().stored().scan_policy("forbid").take(1)
        with pytest.raises(QueryIndexUnavailable):
            query.cdefs().stored().require_indexed().take(1)
        assert not joined.is_set()
        second = _query_thread(query.cdefs(), store, waiter)
        assert joined.wait(10)
    finally:
        release.set()
        first.join(15)
        if second is not None:
            second.join(15)
    assert not first.is_alive() and second is not None and not second.is_alive()
    assert calls == [True]
    assert owner[0].refresh_action == "validate" and owner[0].store_scan_count == 2
    assert waiter[0].refresh_action == "waited" and waiter[0].store_scan_count == 1
    assert not index._connections._leases
    assert not query._refresh_bindings[0].request.in_flight
    assert index._connections._path_identity() == identity and index.status().generation == generation
    assert all(not con.in_transaction for con in index._connections._connections.values())
    probe = locking_module.FileLock(store._writer_lock_path)
    assert probe.acquire(blocking=False)
    probe.release()


@pytest.mark.timeout(40)
@pytest.mark.parametrize("interrupted", (False, True))
def test_u2_old_waiter_observes_failed_attempt_after_fast_retry(tmp_path, monkeypatch, interrupted):
    """A completed A survives B's activated-but-unacknowledged failed cohort."""
    import dryml.core.query.query as query_module
    from dryml.core.query.source import SourceCapture
    from dryml.core.repo import Repo

    a, b = (_request_store(tmp_path / name) for name in ("a", "b"))
    stores = (a, b)
    repo = Repo(b)
    query = a.query().union(repo.query()).refresh(True)
    indexes = tuple(store.open_query_index() for store in stores)
    index = indexes[1]
    uuids = tuple(item.status().diagnostics["index_uuid"] for item in indexes)
    generation = _generation(index.path)
    for name, store in zip(("a", "b"), stores):
        store.write_definition_record(DefinitionRecord(_cdef(name)))
    tokens = tuple(backend._dirty_markers() for backend in indexes)
    primary = KeyboardInterrupt("owner interrupted") if interrupted else PermissionError("owner failed")
    entered, fail, joined, observe = (threading.Event() for _ in range(4))
    original = index._clear_dirty
    wait = query_module._RefreshAttempt.wait
    calls, owner, waiter = [], [], []
    captures = {}
    capture_init = SourceCapture.__init__

    def retain(capture, controls=None):
        capture_init(capture, controls)
        if controls is not None:
            captures[threading.current_thread().name] = capture

    for store, backend in zip(stores, indexes):
        refresh = backend.refresh

        def cycle(policy, *, stats=None, _store=store, _refresh=refresh):
            calls.append((_store, policy))
            return _refresh(policy, stats=stats)

        monkeypatch.setattr(backend, "refresh", cycle)

    acknowledgements = []

    def acknowledge(*args, **kwargs):
        acknowledgements.append(args[0])
        assert index._build_claim_path().exists()
        assert _generation(index.path) > generation
        assert not index._connections._leases
        if len(acknowledgements) == 1:
            assert not a.query_index_is_dirty() and b.query_index_is_dirty()
            assert set(args[0]) == set(tokens[1])
            entered.set()
            assert fail.wait(15)
            raise primary
        return original(*args, **kwargs)

    def delayed(attempt):
        request = query._refresh_bindings[0].request
        assert request._process_lock().acquire(blocking=False)
        request._process_lock().release()
        assert repo._configuration_lock.acquire(blocking=False)
        repo._configuration_lock.release()
        assert repo._topology_leases == 0
        for store, backend in zip(stores, indexes):
            probe = locking_module.FileLock(store._writer_lock_path)
            assert probe.acquire(blocking=False)
            probe.release()
            assert (os.getpid(), threading.get_ident(), True) not in backend._connections._leases
        joined.set()
        assert observe.wait(15)
        return wait(attempt)

    monkeypatch.setattr(index, "_clear_dirty", acknowledge)
    monkeypatch.setattr(SourceCapture, "__init__", retain)
    monkeypatch.setattr(query_module._RefreshAttempt, "wait", delayed)
    first = _query_thread(query, stores, owner, name="failed-owner")
    second = None
    try:
        assert entered.wait(10), owner
        second = _query_thread(query, stores, waiter, name="old-waiter")
        assert joined.wait(10)
        assert _stored_root_count(index.path) == 2
        b.write_definition_record(DefinitionRecord(_cdef("late-b")))
        later_tokens = set(index._dirty_markers())
        assert set(tokens[1]) < later_tokens
        fail.set()
        first.join(10)
        assert not first.is_alive() and owner == [primary]
        assert set(index._dirty_markers()) == later_tokens
        assert not index._build_claim_path().exists()
        successful_a = indexes[0].status().diagnostics["index_uuid"]
        assert successful_a != uuids[0]
        retry = query.explain(analyze=True)
        assert retry.refresh_action == "rebuild" and retry.store_scan_count == 3
        assert retry.result_count == 4 and _stored_root_count(index.path) == 3
        assert calls == [(a, True), (b, True), (b, True)]
        assert indexes[0].status().diagnostics["index_uuid"] == successful_a
    finally:
        fail.set()
        observe.set()
        first.join(15)
        if second is not None:
            second.join(15)
    assert second is not None and not second.is_alive()
    assert len(waiter) == 1 and isinstance(waiter[0], QueryIndexError)
    assert ("interrupted" if interrupted else "failed") in str(waiter[0])
    assert captures["failed-owner"].store_scan_count == 2
    assert captures["failed-owner"].refresh_action == "rebuild"
    assert captures["old-waiter"].store_scan_count == 0
    assert captures["old-waiter"].source_cuts == 0
    assert query.explain(analyze=True).refresh_action == "reused"
    assert calls == [(a, True), (b, True), (b, True)]
    assert not query._refresh_bindings[0].request.in_flight
    for store, backend in zip(stores, indexes):
        assert backend.status().state == "ready" and not store.query_index_is_dirty()
        assert not backend._connections._leases and not backend._build_claim_path().exists()
        assert all(not con.in_transaction for con in backend._connections._connections.values())
        assert not tuple(backend.path.parent.glob(f"{backend.path.name}.rebuild-*.tmp*"))
        probe = locking_module.FileLock(store._writer_lock_path)
        assert probe.acquire(blocking=False)
        probe.release()


@pytest.mark.timeout(40)
def test_u2_waiter_interruption_detaches_and_self_reentry_refuses(tmp_path, monkeypatch):
    """Detaching a waiter leaves real replacement ownership and its retry intact."""
    store = _request_store(tmp_path / "store")
    query = store.query().refresh(True)
    index = store.open_query_index()
    store.mark_query_index_dirty()
    original = index._capture_authority_cut
    identity = index._connections._path_identity()
    entered, release = threading.Event(), threading.Event()
    owner, waiter, reentered = [], [], []

    def capture():
        cut = original()
        assert index._build_claim_path().exists()
        assert not index._connections._leases
        with pytest.raises(QueryIndexBusy, match="reentr"):
            query.count()
        reentered.append(True)
        entered.set()
        assert release.wait(15)
        return cut

    def interrupt(timeout=None):
        raise KeyboardInterrupt("waiter detached")

    monkeypatch.setattr(index, "_capture_authority_cut", capture)
    first = _query_thread(query, store, owner)
    second = None
    try:
        assert entered.wait(10)
        request = query._refresh_bindings[0].request
        with request._process_lock():
            attempt = next(iter(request.in_flight.values()))[1]
        monkeypatch.setattr(attempt.condition, "wait", interrupt)
        second = _query_thread(query, store, waiter)
        second.join(10)
        assert len(waiter) == 1 and isinstance(waiter[0], KeyboardInterrupt)
        assert attempt.condition.acquire(blocking=False)
        attempt.condition.release()
        assert first.is_alive()
    finally:
        release.set()
        first.join(15)
        if second is not None:
            second.join(15)
    assert not first.is_alive() and second is not None and not second.is_alive()
    assert reentered == [True] and owner[0].refresh_action == "rebuild"
    assert query.explain(analyze=True).refresh_action == "reused"
    assert index._connections._path_identity() != identity and not store.query_index_is_dirty()
    assert not query._refresh_bindings[0].request.in_flight
    assert not index._connections._leases and not index._build_claim_path().exists()


@pytest.mark.timeout(50)
def test_u2_multiple_flights_reversed_cohorts_and_late_request(tmp_path, monkeypatch):
    """Join every old attempt, then co-credit only one newly owned pending cohort."""
    import dryml.core.query.query as query_module

    store = _request_store(tmp_path / "store")
    a, b = store.query().refresh(True), store.query().refresh(True)
    index = store.open_query_index()
    original = index.refresh
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    joined = threading.Condition()
    observations, calls, outcomes = [], [], [[], [], [], []]
    wait = query_module._RefreshAttempt.wait
    original_fence = store.authority_read_fence
    held = threading.local()

    @contextmanager
    def fence():
        with original_fence():
            held.depth = getattr(held, "depth", 0) + 1
            try:
                yield
            finally:
                held.depth -= 1

    def refresh(policy, *, stats=None):
        position = len(calls)
        calls.append(policy)
        if position < 2:
            entered[position].set()
            assert release[position].wait(20)
        return original(policy, stats=stats)

    def observe(attempt):
        for query in (a, b, fresh):
            request = query._refresh_bindings[0].request
            assert request._process_lock().acquire(timeout=5)
            request._process_lock().release()
        assert getattr(held, "depth", 0) == 0
        assert (os.getpid(), threading.get_ident(), True) not in index._connections._leases
        with joined:
            observations.append(attempt)
            joined.notify_all()
        return wait(attempt)

    monkeypatch.setattr(index, "refresh", refresh)
    monkeypatch.setattr(query_module._RefreshAttempt, "wait", observe)
    monkeypatch.setattr(store, "authority_read_fence", fence)
    threads = [_query_thread(a, store, outcomes[0])]
    try:
        assert entered[0].wait(10)
        threads.append(_query_thread(b, store, outcomes[1]))
        assert entered[1].wait(10)
        fresh = store.query().refresh(True)  # created after both old admissions
        forward, reverse = a.union(b).union(fresh), fresh.union(b).union(a)
        threads.extend((_query_thread(forward, store, outcomes[2]), _query_thread(reverse, store, outcomes[3])))
        with joined:
            assert joined.wait_for(lambda: len(observations) >= 2, timeout=10)
        assert observations[0] is observations[1]
        first_position = 0 if observations[0].owner == threads[0].ident else 1
        assert observations[0].owner == threads[first_position].ident
        assert calls == [True, True]  # no claiming fresh while either older pair flies
        release[first_position].set()
        with joined:
            assert joined.wait_for(lambda: len(observations) >= 3, timeout=10)
        assert calls == [True, True]
        release[1 - first_position].set()
    finally:
        for event in release:
            event.set()
        for thread in threads:
            thread.join(20)
    assert all(not thread.is_alive() for thread in threads)
    assert all(len(items) == 1 and not isinstance(items[0], BaseException) for items in outcomes), outcomes
    assert calls == [True, True, True]
    assert fresh.explain(analyze=True).refresh_action == "reused"
    assert all(not query._refresh_bindings[0].request.in_flight for query in (a, b, fresh))
    assert not index._connections._leases and not index._build_claim_path().exists()


@pytest.mark.timeout(50)
def test_u2_successful_joins_recheck_later_cohort_before_claiming_remainder(tmp_path, monkeypatch):
    import dryml.core.query.query as query_module

    store = _request_store(tmp_path / "store")
    a, b, c = (store.query().refresh(True) for _ in range(3))
    entered, release = [threading.Event() for _ in range(2)], [threading.Event() for _ in range(2)]
    joined_a, resumed_a, joined_b = threading.Event(), threading.Event(), threading.Event()
    proceed = threading.Event()
    original = store.open_query_index().refresh
    wait = query_module._RefreshAttempt.wait
    calls, outcomes, frozen_attempts = [], [[], [], []], []

    def refresh(policy, *, stats=None):
        position = len(calls)
        calls.append(policy)
        if position < 2:
            entered[position].set()
            assert release[position].wait(20)
        return original(policy, stats=stats)

    def observe(attempt):
        if threading.current_thread().name != "frozen-obligations":
            return wait(attempt)
        frozen_attempts.append(attempt)
        if len(frozen_attempts) == 1:
            joined_a.set()
            wait(attempt)
            resumed_a.set()
            assert proceed.wait(20)
        else:
            joined_b.set()
            return wait(attempt)

    monkeypatch.setattr(store.open_query_index(), "refresh", refresh)
    monkeypatch.setattr(query_module._RefreshAttempt, "wait", observe)
    threads = [_query_thread(a, store, outcomes[0])]
    try:
        assert entered[0].wait(10)
        threads.append(_query_thread(a.union(b).union(c), store, outcomes[1], name="frozen-obligations"))
        assert joined_a.wait(10)
        threads.append(_query_thread(a.union(b), store, outcomes[2]))
        release[0].set()
        assert resumed_a.wait(10) and entered[1].wait(10)
        proceed.set()
        assert joined_b.wait(10)
        assert calls == [True, True]
        release[1].set()
    finally:
        proceed.set()
        for event in release:
            event.set()
        for thread in threads:
            thread.join(20)
    assert all(not thread.is_alive() for thread in threads)
    assert all(len(items) == 1 and not isinstance(items[0], BaseException) for items in outcomes), outcomes
    assert len(frozen_attempts) == 2 and frozen_attempts[0] is not frozen_attempts[1]
    assert calls == [True, True, True]
    assert outcomes[1][0].store_scan_count == 2  # owned C cycle + fresh evaluation only


@pytest.mark.timeout(40)
def test_u2_owner_baseexception_notifies_all_blocked_waiters(tmp_path, monkeypatch):
    """Unwind a real fenced validation transaction before notifying every waiter."""
    import dryml.core.query.query as query_module

    store = _request_store(tmp_path / "store")
    query = store.query().refresh(True)
    entered, release = threading.Event(), threading.Event()
    joined = threading.Condition()
    observations, outcomes = [], [[], [], []]
    wait = query_module._RefreshAttempt.wait
    primary = SystemExit("injected owner unwind")
    index = store.open_query_index()
    original = index._validate_store_roots
    generation = index.status().generation

    def validate(con, issues, **kwargs):
        assert con.in_transaction and index._connections._leases
        entered.set()
        assert release.wait(15)
        raise primary

    def observe(attempt):
        with joined:
            observations.append(attempt)
            joined.notify_all()
        return wait(attempt)

    monkeypatch.setattr(index, "_validate_store_roots", validate)
    monkeypatch.setattr(query_module._RefreshAttempt, "wait", observe)
    threads = [_query_thread(query, store, outcomes[0])]
    try:
        assert entered.wait(10)
        threads.extend(_query_thread(query, store, outcomes[position]) for position in (1, 2))
        with joined:
            assert joined.wait_for(lambda: len(observations) == 2, timeout=10)
        assert observations[0] is observations[1]
    finally:
        release.set()
        for thread in threads:
            thread.join(15)
    assert all(not thread.is_alive() for thread in threads)
    assert outcomes[0] == [primary]
    assert all(len(items) == 1 and isinstance(items[0], QueryIndexError)
               and "interrupted" in str(items[0]) for items in outcomes[1:])
    assert not query._refresh_bindings[0].request.in_flight
    assert not index._connections._leases and not index._build_claim_path().exists()
    assert index.status().generation == generation
    probe = locking_module.FileLock(store._writer_lock_path)
    assert probe.acquire(blocking=False)
    probe.release()
    monkeypatch.setattr(index, "_validate_store_roots", original)
    assert query.explain(analyze=True).refresh_action == "validate"


@pytest.mark.timeout(40)
def test_u2_published_attempt_outcome_survives_later_owner_interruption(tmp_path, monkeypatch):
    """Unwind repeats a lost success notification for an already-blocked terminal."""
    store = _request_store(tmp_path / "store")
    query = store.query().refresh(True)
    request = query._refresh_bindings[0].request
    index = store.open_query_index()
    identity = index._connections._path_identity()
    generation = index.status().generation
    validate = index._validate_store_roots
    entered, release, blocked = (threading.Event() for _ in range(3))
    primary = KeyboardInterrupt("owner interrupted before success notification")
    owner, waiting, interruptions, validations = [], [], [], []

    def compare(con, issues, **kwargs):
        validations.append(True)
        assert con.in_transaction and index._connections._leases
        entered.set()
        assert release.wait(15)
        return validate(con, issues, **kwargs)

    monkeypatch.setattr(index, "_validate_store_roots", compare)
    first = _query_thread(query, store, owner)
    second = attempt = notify = None
    try:
        assert entered.wait(10), owner
        with request._process_lock():
            attempt = next(iter(request.in_flight.values()))[1]
        wait, notify = attempt.condition.wait, attempt.condition.notify_all

        def observe(timeout=None):
            assert not attempt.done
            assert request._process_lock().acquire(blocking=False)
            request._process_lock().release()
            assert (os.getpid(), threading.get_ident(), True) not in index._connections._leases
            # The owner must acquire this same condition after wait releases it.
            blocked.set()
            return wait(timeout)

        def interrupt_notification():
            assert blocked.is_set() and attempt.done and attempt.failure is None
            assert not index._connections._leases and not index._build_claim_path().exists()
            probe = locking_module.FileLock(store._writer_lock_path)
            assert probe.acquire(blocking=False)
            probe.release()
            if not interruptions:
                interruptions.append(primary)
                raise primary
            return notify()

        monkeypatch.setattr(attempt.condition, "wait", observe)
        monkeypatch.setattr(attempt.condition, "notify_all", interrupt_notification)
        second = _query_thread(query.cdefs(), store, waiting)
        assert blocked.wait(10)
        release.set()
        first.join(10)
        assert not first.is_alive() and owner == [primary]
        second.join(10)
        assert not second.is_alive(), "published success stranded its already-blocked terminal"
        assert len(waiting) == 1 and waiting[0].refresh_action == "waited", waiting
        assert waiting[0].result_count == 1 and waiting[0].store_scan_count == 1
    finally:
        release.set()
        first.join(15)
        # Release a stranded waiter on regression so a failed test leaves no thread.
        if second is not None:
            if second.is_alive() and notify is not None:
                with attempt.condition:
                    notify()
            second.join(15)
    assert not first.is_alive() and second is not None and not second.is_alive()
    assert interruptions == [primary] and attempt.done and attempt.failure is None
    assert query.explain(analyze=True).refresh_action == "reused"
    assert validations == [True] and len(request.completed) == 1 and not request.in_flight
    assert index._connections._path_identity() == identity and index.status().generation == generation
    assert not index._connections._leases and not index._build_claim_path().exists()
    assert all(not con.in_transaction for con in index._connections._connections.values())
    assert not tuple(index.path.parent.glob(f"{index.path.name}.rebuild-*.tmp*"))
    probe = locking_module.FileLock(store._writer_lock_path)
    assert probe.acquire(blocking=False)
    probe.release()


def test_u2_fork_resets_held_request_lock_and_preserves_derivative_sharing(tmp_path):
    """Composed fork resets query/SQLite owners and refuses inherited Repo topology."""
    result = _run_worker(f'''
import os
import select
import signal
import threading
from dryml.core.repo import Repo
from dryml.core.query.sqlite.connection import SQLiteConnectionManager
if not hasattr(os, "fork"):
    emit({{"skipped": True}})
else:
    store = DirStore({str(tmp_path / "request-fork")!r}, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    store.write_definition_record(__import__("dryml.core.store.records", fromlist=["DefinitionRecord"]).DefinitionRecord(cdef("root")))
    store.rebuild_query_index()
    query = store.query().refresh(True)
    derived = query.cdefs()
    query.count()
    derived.explain(analyze=True)  # preload the exercised selector path before fork
    repo = Repo(store)
    repo_query = repo.query().refresh(True)
    repo_query.refresh(False).count()
    idx = store.open_query_index()
    request = query._refresh_bindings[0].request
    held, release = threading.Event(), threading.Event()
    def hold():
        with (request._process_lock(), SQLiteConnectionManager._instances_lock,
              idx._connections._process_lock(), repo._configuration_lock):
            held.set()
            assert release.wait(15)
    thread = threading.Thread(target=hold, daemon=True)
    with idx._retained_read_view() as parent_view:
        thread.start()
        assert held.wait(2)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                first = derived.explain(analyze=True)
                second = query.explain(analyze=True)
                try:
                    repo_query.count()
                except RuntimeError as error:
                    assert "process" in str(error)
                else:
                    raise AssertionError("inherited Repo was accepted")
                fresh_store = DirStore(store.base_dir, query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
                fresh = Repo(fresh_store)
                fresh_query = fresh.query().refresh(True)
                third = fresh_query.explain(analyze=True)
                fourth = fresh_query.explain(analyze=True)
                for backend in (idx, fresh_store.open_query_index()):
                    assert not backend._connections._leases
                    assert all(key[0] == os.getpid() for key in backend._connections._connections)
                    assert not backend._build_claim_path().exists()
                    backend.close()
                assert not request.in_flight and fresh._topology_leases == 0
                os.write(write_fd, json.dumps([first.refresh_action, second.refresh_action,
                                             third.refresh_action, fourth.refresh_action]).encode())
                os._exit(0)
            except BaseException as error:
                os.write(write_fd, repr(error).encode())
                os._exit(1)
        os.close(write_fd)
        try:
            ready = bool(select.select([read_fd], [], [], 10)[0])
            if not ready:
                os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            payload = os.read(read_fd, 4096).decode()
        finally:
            os.close(read_fd)
            release.set()
            thread.join(2)
        assert not thread.is_alive()
        assert parent_view._active and parent_view._con.in_transaction
        assert len(idx._connections._leases) == 1
        assert next(parent_view.iter_stored_identity_cdef_batches(batch_size=1)).cdefs == (cdef("root"),)
    assert ready and status == 0, payload
    assert not idx._connections._leases and not request.in_flight
    assert repo._topology_leases == 0 and repo_query.refresh(False).count() == 1
    emit({{"ready": ready, "status": status, "actions": json.loads(payload),
           "parent": query.explain(analyze=True).refresh_action}})
''', timeout=30)
    if result.get("skipped"):
        pytest.skip("requires fork")
    assert result == {"ready": True, "status": 0,
                      "actions": ["validate", "reused", "validate", "reused"], "parent": "reused"}


@pytest.mark.timeout(30)
def test_explicit_cycle_declined_live_claim_validates_actual_candidate(tmp_path, monkeypatch):
    """Old ready-looking data under a live builder confers no completion credit."""
    from dryml.core.query.model import QueryStats
    from dryml.core.store.records import MainRefRecord

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    root, main = _cdef("root"), _cdef("main")
    store.write_definition_record(DefinitionRecord(root))
    store.write_definition_record(DefinitionRecord(main), stored_root=False)
    store.rebuild_query_index()
    store.write_main_ref(MainRefRecord(DefinitionRecord(main).digest))
    index = store.open_query_index()
    claimed, release = threading.Event(), threading.Event()
    errors = []
    original_capture = index._capture_authority_cut
    original_claim = index._build_claim
    original_fence = store.authority_read_fence
    held = threading.local()

    @contextmanager
    def fence():
        with original_fence():
            held.depth = getattr(held, "depth", 0) + 1
            try:
                yield
            finally:
                held.depth -= 1

    def capture():
        if threading.current_thread() is worker:
            claimed.set()
            assert release.wait(10)
        return original_capture()

    @contextmanager
    def claim(**kwargs):
        if threading.current_thread() is not worker:
            assert getattr(held, "depth", 0) == 0
            assert not index._connections._leases
        with original_claim(**kwargs) as acquired:
            yield acquired

    def rebuild():
        try:
            index.rebuild()
        except BaseException as error:
            errors.append(error)
        finally:
            index._connections.close_current()

    worker = threading.Thread(target=rebuild, daemon=True)
    monkeypatch.setattr(store, "authority_read_fence", fence)
    monkeypatch.setattr(index, "_capture_authority_cut", capture)
    monkeypatch.setattr(index, "_build_claim", claim)
    worker.start()
    try:
        assert claimed.wait(10)
        stats = QueryStats()
        with pytest.raises(QueryIndexBusy, match="diverges"):
            index.refresh(True, stats=stats)
        assert stats.store_scan_count == 2
        assert stats.refresh_action == "sqlite-validate"
        report = index.reconcile()
        assert not report.changed and report.action == "validate"
        assert report.diagnostics["store_scan_count"] == 2
        assert any("diverges" in (issue.detail or "") for issue in report.issues)
        assert not index._connections._leases
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive() and not errors
    assert not index._build_claim_path().exists()
    index.refresh(True)


def test_explicit_cycle_declined_claim_accepts_substantively_valid_candidate(tmp_path, monkeypatch):
    """A non-owner may succeed by validating, never by claiming a replacement."""
    from dryml.core.query.model import QueryStats
    from dryml.core.store.records import MainRefRecord

    store = DirStore(tmp_path / "store", query_index="sqlite")
    record = store.write_definition_record(DefinitionRecord(_cdef("main")), stored_root=False)
    store.rebuild_query_index()
    index = store.open_query_index()
    store.write_main_ref(MainRefRecord(record.digest))
    other = SQLiteStoreQueryIndex(source_key=index.source_key, path=index.path,
                                  config=index.config, store=store, dirty_path=index.dirty_path)
    original_claim = index._build_claim

    @contextmanager
    def claim(**kwargs):
        assert not index._connections._leases
        with other._build_claim(force=True) as acquired:
            assert acquired
            other._rebuild_owned()
            with original_claim(**kwargs) as ours:
                assert not ours
                yield ours

    monkeypatch.setattr(index, "_build_claim", claim)
    report = index.reconcile()
    assert not report.changed and report.action == "validate"
    assert report.generation_after > report.generation_before
    assert report.diagnostics == {"store_scan_count": 2, "refresh_action": "sqlite-validate"}
    assert any("roots differ" in issue.message for issue in report.issues)
    assert not index._connections._leases and not index._build_claim_path().exists()
    other.close()


def test_explicit_cycle_detects_wal_generation_change_in_active_destination(tmp_path, monkeypatch):
    """The retained transaction alone cannot prove continuity behind a WAL writer."""
    from dryml.core.query.model import QueryStats
    from dryml.core.query.sqlite.utils import wal_runtime_is_known_safe

    if not wal_runtime_is_known_safe(require_sqlite().sqlite_version_info):
        pytest.skip("installed SQLite is not qualified for WAL")
    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="wal"))
    store.rebuild_query_index()
    index = store.open_query_index()
    original = index._validate_store_roots
    mutations = []

    def validate(con, issues, **kwargs):
        original(con, issues, **kwargs)
        if not mutations:
            identity = index._connections._path_identity()
            with closing(require_sqlite().connect(index.path)) as writer, writer:
                writer.execute("UPDATE catalog_state SET generation = generation + 1")
            assert index._connections._path_identity() == identity
            mutations.append(1)

    monkeypatch.setattr(index, "_validate_store_roots", validate)
    stats = QueryStats()
    index.refresh(True, stats=stats)
    assert mutations == [1]
    assert stats.store_scan_count == 2 and stats.refresh_action == "sqlite-rebuild"
    assert not index._connections._leases


def test_explicit_cycle_destination_change_discards_validation_cut(tmp_path, monkeypatch):
    """Replacing a destination during comparison cannot validate the old inode."""
    from dryml.core.query.model import QueryStats

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    _save_root_definition(store, _cdef("root"))
    store.rebuild_query_index()
    index = store.open_query_index()
    other = SQLiteStoreQueryIndex(source_key=index.source_key, path=index.path,
                                  config=index.config, store=store, dirty_path=index.dirty_path)
    original = index._validate_store_roots
    original_claim = index._build_claim
    replaced = []

    def validate(con, issues, **kwargs):
        original(con, issues, **kwargs)
        if not replaced:
            assert con.in_transaction and index._connections._leases
            other.rebuild()
            replaced.append(other.status().diagnostics["index_uuid"])

    @contextmanager
    def claim(**kwargs):
        assert not index._connections._leases
        with original_claim(**kwargs) as acquired:
            yield acquired

    monkeypatch.setattr(index, "_validate_store_roots", validate)
    monkeypatch.setattr(index, "_build_claim", claim)
    stats = QueryStats()
    index.refresh(True, stats=stats)
    assert len(replaced) == 1
    assert stats.store_scan_count == 2 and stats.refresh_action == "sqlite-rebuild"
    assert index.status().diagnostics["index_uuid"] != replaced[0]
    assert not index._connections._leases and not index._build_claim_path().exists()
    other.close()


@pytest.mark.timeout(30)
def test_explicit_validation_holds_authority_fence_through_comparison(tmp_path, monkeypatch):
    """A main writer cannot cross the validation cut; later authority is not credit."""
    from dryml.core.store.records import MainRefRecord

    store = DirStore(tmp_path / "store", query_index="sqlite")
    store.write_definition_record(DefinitionRecord(_cdef("root")))
    main = store.write_definition_record(DefinitionRecord(_cdef("main")), stored_root=False)
    store.rebuild_query_index()
    index = store.open_query_index()
    generation = index.status().generation
    attempted, entered = threading.Event(), threading.Event()
    original_validate = index._validate_store_roots
    errors = []
    checks = []

    def writer():
        try:
            probe = locking_module.FileLock(store._writer_lock_path)
            acquired = probe.acquire(blocking=False)
            if acquired:
                probe.release()
            checks.append(acquired)
            attempted.set()
            store.write_main_ref(MainRefRecord(main.digest))
            entered.set()
        except BaseException as error:
            errors.append(error)
            attempted.set()

    worker = threading.Thread(target=writer, daemon=True)

    def validate(con, issues, **kwargs):
        worker.start()
        assert attempted.wait(10)
        assert not entered.is_set()
        assert con.in_transaction and index._connections._leases
        original_validate(con, issues, **kwargs)

    monkeypatch.setattr(index, "_validate_store_roots", validate)
    try:
        index.refresh(True)
    finally:
        if worker.ident is not None:
            worker.join(10)
    assert not worker.is_alive() and not errors and entered.is_set()
    assert checks == [False]
    assert index.status().generation == generation
    assert not index._connections._leases
    monkeypatch.setattr(index, "_validate_store_roots", original_validate)
    index.refresh(True)
    assert index.status().generation > generation


@pytest.mark.timeout(30)
@pytest.mark.parametrize("schema_version", (SQLITE_QUERY_INDEX_SCHEMA_VERSION, 7), ids=("current", "known-older"))
@pytest.mark.parametrize("operation", (True, "always", "auto", "rebuild"))
def test_overlapping_foreign_destination_owner_is_refused_after_claim(tmp_path, monkeypatch, schema_version, operation):
    """Post-preflight foreign ownership outranks even known-older recovery under claim."""
    from dryml.core.query.model import QueryIndexIncompatible, QueryStats

    path = tmp_path / "shared.sqlite"
    config = SQLiteQueryIndexConfig(path=path, journal_mode="delete")
    first, second = (DirStore(tmp_path / name, query_index=config) for name in ("first", "second"))
    records = tuple(store.write_definition_record(DefinitionRecord(_cdef(name)))
                    for store, name in ((first, "first"), (second, "second")))
    authority_paths = tuple(Path(store._definition_path(record.digest)) for store, record in zip((first, second), records))
    authority_bytes = tuple(path.read_bytes() for path in authority_paths)
    contender = second.open_query_index()
    preflight, release = threading.Event(), threading.Event()
    original_claim = contender._build_claim
    outcomes = []
    stats = QueryStats()

    @contextmanager
    def claim(**kwargs):
        preflight.set()
        assert release.wait(10)
        with original_claim(**kwargs) as acquired:
            assert acquired
            yield acquired

    def refresh():
        try:
            if operation == "rebuild":
                contender.rebuild(stats=stats)
            else:
                contender.refresh(operation, stats=stats)
        except BaseException as error:
            outcomes.append(error)
        finally:
            contender._connections.close_current()

    monkeypatch.setattr(contender, "_build_claim", claim)
    monkeypatch.setattr(contender, "_capture_authority_cut", lambda: pytest.fail("foreign owner scanned"))
    worker = threading.Thread(target=refresh, daemon=True)
    worker.start()
    try:
        assert preflight.wait(10)
        first.rebuild_query_index()
        owner = first.open_query_index()
        owner.close()
        with closing(require_sqlite().connect(path)) as con, con:
            con.execute(f"PRAGMA user_version = {schema_version}")
            assert con.execute("SELECT store_key FROM catalog_state WHERE singleton = 1").fetchone() == (owner.source_key,)
        first.mark_query_index_dirty()
        identity = (path.stat().st_dev, path.stat().st_ino)
        contents = path.read_bytes()
        tokens = tuple(backend._dirty_markers() for backend in (owner, contender))
        assert all(tokens)
    finally:
        release.set()
        worker.join(10)
    assert not worker.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], QueryIndexIncompatible)
    assert str(outcomes[0]) == "SQLite query-index destination has a foreign Store owner; refusing maintenance."
    assert stats.store_scan_count == 0
    assert (path.stat().st_dev, path.stat().st_ino) == identity
    assert path.read_bytes() == contents
    assert tuple(backend._dirty_markers() for backend in (owner, contender)) == tokens
    assert tuple(path.read_bytes() for path in authority_paths) == authority_bytes
    assert not contender._connections._leases and not contender._build_claim_path().exists()
    assert all(not con.in_transaction for con in contender._connections._connections.values())
    assert not tuple(path.parent.glob(f"{path.name}.rebuild-*.tmp*"))


def test_fresh_forked_manager_resets_held_class_registry_lock(tmp_path):
    """Fresh child construction and path barriers never acquire the parent's lock."""
    result = _run_worker(f'''
import os
import select
import signal
import threading
from dryml.core.query.sqlite.connection import SQLiteConnectionManager

if not hasattr(os, "fork"):
    emit({{"skipped": True}})
else:
    idx = index({str(tmp_path / "fork-registry.sqlite")!r})
    idx.initialize_empty()
    held, release = threading.Event(), threading.Event()
    def hold():
        with SQLiteConnectionManager._instances_lock:
            held.set()
            assert release.wait(8)
    worker = threading.Thread(target=hold, daemon=True)
    with idx.read_view() as parent_view:
        worker.start()
        assert held.wait(2)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                fresh = index({str(tmp_path / "fork-registry.sqlite")!r})
                with fresh._retained_read_view() as view:
                    assert view.generation == 0
                SQLiteConnectionManager._close_current_thread_for_path(fresh.path)
                with idx._retained_read_view() as view:
                    assert all(key[0] == os.getpid() for key in idx._connections._connections)
                os.write(write_fd, b"ok")
                os._exit(0)
            except BaseException as error:
                os.write(write_fd, repr(error).encode())
                os._exit(1)
        os.close(write_fd)
        ready = False
        try:
            ready = bool(select.select([read_fd], [], [], 3)[0])
            if not ready:
                os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            payload = os.read(read_fd, 4096).decode()
        finally:
            os.close(read_fd)
            release.set()
            worker.join(2)
        emit({{"ready": ready, "status": status, "payload": payload,
               "parent_active": parent_view._active and parent_view._con.in_transaction}})
''')
    if result.get("skipped"):
        pytest.skip("requires os.fork")
    assert result == {"ready": True, "status": 0, "payload": "ok", "parent_active": True}


@pytest.mark.timeout(30)
def test_u7_readiness_waits_for_build_claim_before_authority_fence(tmp_path, monkeypatch):
    """A claimed rebuilder can capture while a query waits outside every fence."""

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    root = _cdef("ready")
    _save_root_definition(store, root)
    store.rebuild_query_index()
    index = store.open_query_index()
    store.mark_query_index_dirty()
    claimed, preflight = threading.Event(), threading.Event()
    held = threading.local()
    errors = []
    waits = []
    original_cut = index._capture_authority_cut
    original_claim = index._build_claim
    original_fence = store.authority_read_fence

    @contextmanager
    def fence():
        with original_fence():
            held.depth = getattr(held, "depth", 0) + 1
            try:
                yield
            finally:
                held.depth -= 1

    def capture():
        if threading.current_thread() is worker:
            claimed.set()
            assert preflight.wait(10)
        return original_cut()

    @contextmanager
    def claim(**kwargs):
        if threading.current_thread() is not worker:
            assert getattr(held, "depth", 0) == 0
            assert not index._connections._leases
            waits.append("outside")
            preflight.set()
        with original_claim(**kwargs) as acquired:
            yield acquired

    def rebuild():
        try:
            index.rebuild(force=True)
        except BaseException as error:
            errors.append(error)
        finally:
            index._connections.close_current()

    worker = threading.Thread(target=rebuild, daemon=True)
    monkeypatch.setattr(store, "authority_read_fence", fence)
    monkeypatch.setattr(index, "_capture_authority_cut", capture)
    monkeypatch.setattr(index, "_build_claim", claim)
    worker.start()
    try:
        assert claimed.wait(10)
        assert tuple(store.query().cdefs().stored().take(1)) == (root,)
    finally:
        preflight.set()
        worker.join(10)
    assert not worker.is_alive() and not errors and waits == ["outside"]
    assert not index._connections._leases and not index._build_claim_path().exists()


@pytest.mark.timeout(30)
def test_u7_replacement_preserves_active_same_process_reader(tmp_path):
    """Build-claim cache barriers cannot tear down another leased read view."""

    store = DirStore(tmp_path / "store", query_index=SQLiteQueryIndexConfig(journal_mode="delete"))
    root = _cdef("pinned")
    _save_root_definition(store, root)
    store.rebuild_query_index()
    index = store.open_query_index()
    errors = []

    def rebuild():
        try:
            index.rebuild(force=False)  # ready-claim short circuit also closes caches
            index.rebuild(force=True)
        except BaseException as error:
            errors.append(error)
        finally:
            index._connections.close_current()

    with index.read_view() as view:
        old = view.index_uuid, view.generation, view.file_identity
        worker = threading.Thread(target=rebuild, daemon=True)
        worker.start()
        try:
            worker.join(10)
            assert not worker.is_alive()
            if os.name == "nt":
                assert len(errors) == 1 and isinstance(errors[0], QueryIndexBusy)
            else:
                assert not errors
            assert view._active and view._con.in_transaction
            assert next(view.iter_stored_identity_cdef_batches(batch_size=1)).cdefs == (root,)
            assert (view.index_uuid, view.generation, view.file_identity) == old
        finally:
            worker.join(10)
    assert not index._connections._leases
    if errors:
        index.rebuild(force=True)
    with index._retained_read_view() as fresh:
        assert fresh.generation > old[1]
        assert fresh.index_uuid != old[0] and fresh.file_identity != old[2]
        assert next(fresh.iter_stored_identity_cdef_batches(batch_size=1)).cdefs == (root,)


@pytest.mark.timeout(30)
def test_u7_main_writer_waits_through_paging_and_direct_verification(tmp_path, monkeypatch):
    """A supported main publication cannot change the cut between pages and proof."""

    from dryml.core.query.source import SourceCapture
    from dryml.core.query.sqlite.index import SQLiteQueryIndexReadView
    from dryml.core.store.records import MainRefRecord

    store = DirStore(tmp_path / "store", query_index="sqlite")
    root, main = sorted((_cdef("first"), _cdef("second")), key=lambda value: value.graph_hash(), reverse=True)
    _save_root_definition(store, root)
    store.write_definition_record(DefinitionRecord(main), stored_root=False)
    store.write_main_ref(MainRefRecord(DefinitionRecord(root).digest))
    store.rebuild_query_index()
    attempting, done = threading.Event(), threading.Event()
    errors = []
    original_page = SQLiteQueryIndexReadView.iter_stored_identity_cdef_batches
    original_direct = SourceCapture.read_exact_stored_cdef
    checks = []

    def publish():
        try:
            probe = locking_module.FileLock(store._writer_lock_path)
            acquired = probe.acquire(blocking=False)
            if acquired:
                probe.release()
            checks.append(acquired)
            attempting.set()
            store.write_main_ref(MainRefRecord(DefinitionRecord(main).digest))
            done.set()
        except BaseException as error:
            errors.append(error)
            attempting.set()

    worker = threading.Thread(target=publish, daemon=True)

    def page(view, **kwargs):
        if not worker.is_alive() and not attempting.is_set():
            worker.start()
        assert attempting.wait(10) and not done.is_set()
        yield from original_page(view, **kwargs)

    def direct(capture, source, target):
        assert not done.is_set()
        return original_direct(capture, source, target)

    with monkeypatch.context() as scoped:
        scoped.setattr(SQLiteQueryIndexReadView, "iter_stored_identity_cdef_batches", page)
        scoped.setattr(SourceCapture, "read_exact_stored_cdef", direct)
        try:
            assert tuple(store.query().cdefs().stored().refresh(False).require_indexed().take(1)) == (root,)
        finally:
            if worker.ident is not None:
                worker.join(10)
    assert not worker.is_alive() and not errors and checks == [False] and done.is_set()
    assert tuple(store.query().cdefs().stored().refresh(False).take(1)) == (main,)
    assert not store.open_query_index()._connections._leases


def test_u7_cancelled_retained_entry_releases_transaction_and_fence(tmp_path, monkeypatch):
    """Cancellation after BEGIN unwinds both connection lease and authority cut."""

    store = DirStore(tmp_path / "store", query_index="sqlite")
    _save_root_definition(store, _cdef("cancel"))
    store.rebuild_query_index()
    index = store.open_query_index()

    def interrupt(con):
        assert con.in_transaction and index._connections._leases
        raise KeyboardInterrupt("retained entry cancelled")

    with monkeypatch.context() as scoped:
        scoped.setattr(sqlite_index_module, "_read_generation", interrupt)
        with pytest.raises(KeyboardInterrupt, match="entry cancelled"):
            store.query().cdefs().stored().refresh(False).take(1)
    assert not index._connections._leases
    assert all(not con.in_transaction for con in index._connections._connections.values())
    _save_root_definition(store, _cdef("after"))  # a leaked fence would block this writer


def test_u7_fork_does_not_inherit_held_manager_lock_or_reader_lease(tmp_path):
    """The new reader guard remains process-local even if fork overlaps its owner."""

    result = _run_worker(f'''
import os
import select
import signal
import threading

if not hasattr(os, "fork"):
    emit({{"skipped": True}})
else:
    idx = index({str(tmp_path / "fork.sqlite")!r})
    idx.initialize_empty()
    held, release = threading.Event(), threading.Event()
    def hold():
        with idx._connections._process_lock():
            held.set()
            assert release.wait(8)
    with idx.read_view() as parent_view:
        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        assert held.wait(2)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                with idx._retained_read_view() as child_view:
                    payload = {{"generation": child_view.generation,
                               "leases": len(idx._connections._leases)}}
                os.write(write_fd, json.dumps(payload).encode())
                os._exit(0)
            except BaseException as error:
                os.write(write_fd, json.dumps({{"error": repr(error)}}).encode())
                os._exit(1)
        os.close(write_fd)
        ready = False
        try:
            ready = bool(select.select([read_fd], [], [], 3)[0])
            if not ready:
                os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            payload = os.read(read_fd, 4096)
        finally:
            os.close(read_fd)
            release.set()
            worker.join(2)
        assert ready, "child inherited held manager lock"
        data = json.loads(payload)
        data["status"] = status
        data["parent_active"] = parent_view._active and parent_view._con.in_transaction
        emit(data)
''')
    if result.get("skipped"):
        pytest.skip("requires os.fork")
    assert result == {"generation": 0, "leases": 1, "status": 0, "parent_active": True}


@pytest.mark.parametrize("entry", ("read_view", "_retained_read_view"))
@pytest.mark.parametrize("exit_mode", ("normal", "exceptional", "access"))
def test_forked_read_view_unwinds_without_parent_handle_cleanup(tmp_path, entry, exit_mode):
    """Natural child unwind preserves its cause, fresh views, and the parent cut."""

    result = _run_worker(f'''
import os
import select
import signal
from dryml.core.query.model import QueryIndexError

if not hasattr(os, "fork"):
    emit({{"skipped": True}})
else:
    idx = index({str(tmp_path / "inherited.sqlite")!r})
    root = cdef("parent-root")
    idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
    pid = None
    payload = {{"error_type": None, "primary_preserved": False}}
    primary = RuntimeError("child body failed")
    try:
        with getattr(idx, {entry!r})() as parent_view:
            parent_con = parent_view._con
            parent_ids = parent_view.exact_ids(root)
            temp = parent_view.create_temp_relation(parent_ids)
            statements = []
            parent_con.set_trace_callback(statements.append)
            read_fd, write_fd = os.pipe()
            pid = os.fork()
            if pid == 0:
                os.close(read_fd)
                statements.clear()
                if {exit_mode!r} == "exceptional":
                    raise primary
                if {exit_mode!r} == "access":
                    rejected = []
                    operations = (
                        lambda: parent_view.exact_ids(root),
                        lambda: tuple(parent_view.iter_stored_identity_cdef_batches(batch_size=1)),
                        parent_view.all_occurrence_snapshot,
                        parent_view.drop_temp_relations,
                    )
                    for operation in operations:
                        try:
                            operation()
                        except QueryIndexError as error:
                            rejected.append("process" in str(error))
                        else:
                            rejected.append(False)
                    payload["rejected"] = rejected
                    payload["access_sql"] = list(statements)
                    with idx._retained_read_view() as fresh:
                        payload["fresh_inside"] = (
                            fresh._con is not parent_con
                            and fresh.exact_ids(root) == parent_ids
                            and len(idx._connections._leases) == 1
                        )
                    payload["fresh_released"] = not idx._connections._leases and not fresh._con.in_transaction
                    statements.clear()
                    parent_view.close()
            else:
                os.close(write_fd)
                try:
                    ready = bool(select.select([read_fd], [], [], 5)[0])
                    if not ready:
                        os.kill(pid, signal.SIGKILL)
                    _, status = os.waitpid(pid, 0)
                    assert ready, "child did not finish inherited context unwind"
                    data = json.loads(os.read(read_fd, 4096))
                    data["status"] = status
                    data["parent_preserved"] = (
                        parent_view._active and parent_con.in_transaction
                        and parent_view.exact_ids(root) == parent_ids
                        and len(idx._connections._leases) == 1
                        and parent_con.execute(
                            "SELECT name FROM sqlite_temp_master WHERE name = ?", (temp,)
                        ).fetchone() == (temp,)
                    )
                finally:
                    os.close(read_fd)
    except BaseException as error:
        if pid != 0:
            raise
        payload["error_type"] = type(error).__name__
        payload["primary_preserved"] = error is primary

    if pid == 0:
        # This is reached only after the inherited outer with statement unwinds.
        payload["inherited_sql"] = statements
        try:
            with idx.read_view() as fresh:
                payload["fresh_after"] = (
                    fresh._con is not parent_con
                    and fresh.exact_ids(root) == parent_ids
                    and len(idx._connections._leases) == 1
                )
            payload["child_clean"] = not idx._connections._leases and not fresh._con.in_transaction
        except BaseException as error:
            payload["fresh_error"] = repr(error)
        os.write(write_fd, json.dumps(payload).encode())
        os.close(write_fd)
        os._exit(0)
    data["parent_clean"] = not idx._connections._leases and not parent_con.in_transaction
    data["parent_temp_dropped"] = parent_con.execute(
        "SELECT name FROM sqlite_temp_master WHERE name = ?", (temp,)
    ).fetchone() is None
    with idx.read_view() as fresh:
        data["parent_reusable"] = fresh._con is parent_con and fresh.exact_ids(root) == parent_ids
    idx.close()
    emit(data)
''', timeout=15)
    if result.get("skipped"):
        pytest.skip("requires os.fork")
    assert result["error_type"] == ("RuntimeError" if exit_mode == "exceptional" else None), result
    assert result["primary_preserved"] is (exit_mode == "exceptional"), result
    assert result["inherited_sql"] == [], result
    assert result["status"] == 0 and result["fresh_after"] and result["child_clean"], result
    assert result["parent_preserved"] and result["parent_clean"], result
    assert result["parent_temp_dropped"] and result["parent_reusable"], result
    if exit_mode == "access":
        assert result["rejected"] == [True] * 4, result
        assert result["access_sql"] == [], result
        assert result["fresh_inside"] and result["fresh_released"], result


def test_cross_process_commit_visible_without_reconnect(tmp_path):
    path = tmp_path / "index.sqlite"
    idx = _index(path)
    idx.initialize_empty()
    with idx.read_view() as view:
        assert view.generation == 0

    result = _run_worker(f'''
result = register_root({str(path)!r}, "visible")
emit({{"generation": result.generation, "changed": result.changed}})
''')

    assert result == {"generation": 1, "changed": True}
    with idx.read_view() as view:
        ids = view.exact_ids(_cdef("visible"))
        assert len(ids) == 1
        assert view.generation == 1


def test_uncommitted_write_is_invisible_until_commit(tmp_path):
    path = tmp_path / "index.sqlite"
    ready = tmp_path / "writer-ready"
    release = tmp_path / "release-writer"
    idx = _index(path)
    idx.initialize_empty()

    proc = _start_worker(f'''
idx = index({str(path)!r}, timeout=0.05, retries=0)
root = cdef("pending")
con = idx._connections.connection(readonly=False)
con.execute("BEGIN IMMEDIATE")
generation = _read_generation(con)
next_generation = generation + 1
encoded = _EncodedNode.from_cdef(root)
did, _ = _resolve_definition_id(con, encoded, generation=next_generation)
con.execute(
    "INSERT OR IGNORE INTO stored_roots (def_id, storage_hash, relative_def_path, def_size, def_mtime_ns, indexed_generation) VALUES (?, ?, ?, NULL, NULL, ?)",
    (did, stable_hash_to_blob(root.stable_hash()), _relative_def_path(root.stable_hash()), next_generation),
)
_bump_generation(con, next_generation)
Path({str(ready)!r}).write_text("ready")
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
con.execute("COMMIT")
idx.close()
emit({{"committed": True}})
''')

    try:
        _wait_for(ready)
        with idx.read_view() as view:
            assert view.exact_ids(_cdef("pending")) == set()
            assert view.generation == 0
        release.write_text("go")
        assert _collect_worker(proc) == {"committed": True}
        with idx.read_view() as view:
            assert len(view.exact_ids(_cdef("pending"))) == 1
            assert view.generation == 1
    finally:
        release.write_text("go")
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_read_transaction_keeps_snapshot_until_next_read_view(tmp_path):
    path = tmp_path / "index.sqlite"
    idx = SQLiteStoreQueryIndex(
        source_key="concurrency-store",
        path=path,
        config=SQLiteQueryIndexConfig(journal_mode="wal", busy_timeout=0.05, max_write_retries=20),
    )
    try:
        idx.initialize_empty()
    except QueryIndexError as exc:
        pytest.skip(f"WAL mode unavailable in this environment: {exc}")

    with idx.read_view() as view:
        assert view.generation == 0
        result = _run_worker(f'''
result = SQLiteStoreQueryIndex(
    source_key="concurrency-store",
    path={str(path)!r},
    config=SQLiteQueryIndexConfig(journal_mode="wal", busy_timeout=0.05, max_write_retries=20),
).register_stored_roots(ConcreteDefinitionGraph.from_root(cdef("snapshot")), [cdef("snapshot")])
emit({{"generation": result.generation, "changed": result.changed}})
''')
        assert result == {"generation": 1, "changed": True}
        assert view.generation == 0
        assert view.exact_ids(_cdef("snapshot")) == set()

    with idx.read_view() as view:
        assert view.generation == 1
        assert len(view.exact_ids(_cdef("snapshot"))) == 1


def test_many_readers_one_writer_in_wal_mode(tmp_path):
    path = tmp_path / "index.sqlite"
    idx = SQLiteStoreQueryIndex(
        source_key="concurrency-store",
        path=path,
        config=SQLiteQueryIndexConfig(journal_mode="wal", busy_timeout=0.05, max_write_retries=20),
    )
    try:
        idx.initialize_empty()
    except QueryIndexError as exc:
        pytest.skip(f"WAL mode unavailable in this environment: {exc}")

    release = tmp_path / "release-readers"
    reader_procs = []
    for reader_idx in range(3):
        ready = tmp_path / f"reader-{reader_idx}-ready"
        reader_procs.append(_start_worker(f'''
idx = SQLiteStoreQueryIndex(
    source_key="concurrency-store",
    path={str(path)!r},
    config=SQLiteQueryIndexConfig(journal_mode="wal", busy_timeout=0.05, max_write_retries=20),
)
with idx.read_view() as view:
    Path({str(ready)!r}).write_text("ready")
    while not Path({str(release)!r}).exists():
        time.sleep(0.01)
    emit({{"generation": view.generation, "count": len(view.exact_ids(cdef("wal-writer")))}})
idx.close()
'''))
    try:
        for reader_idx in range(3):
            _wait_for(tmp_path / f"reader-{reader_idx}-ready")
        result = _run_worker(f'''
idx = SQLiteStoreQueryIndex(
    source_key="concurrency-store",
    path={str(path)!r},
    config=SQLiteQueryIndexConfig(journal_mode="wal", busy_timeout=0.05, max_write_retries=20),
)
root = cdef("wal-writer")
write_result = idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
idx.close()
emit({{"generation": write_result.generation, "changed": write_result.changed}})
''')
        assert result == {"generation": 1, "changed": True}
        release.write_text("go")
        assert [_collect_worker(proc) for proc in reader_procs] == [
            {"generation": 0, "count": 0},
            {"generation": 0, "count": 0},
            {"generation": 0, "count": 0},
        ]
    finally:
        release.write_text("go")
        for proc in reader_procs:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)

    with idx.read_view() as view:
        assert view.generation == 1
        assert len(view.exact_ids(_cdef("wal-writer"))) == 1


def test_concurrent_different_writers_both_commit(tmp_path):
    path = tmp_path / "index.sqlite"
    _index(path).initialize_empty()
    procs = [
        _start_worker(f'''
result = register_root({str(path)!r}, {name!r}, timeout=0.01, retries=50)
emit({{"name": {name!r}, "generation": result.generation, "changed": result.changed}})
''')
        for name in ("left", "right")
    ]

    results = [_collect_worker(proc) for proc in procs]

    assert {result["name"] for result in results} == {"left", "right"}
    assert all(result["changed"] for result in results)
    assert _stored_root_count(path) == 2
    assert _generation(path) == 2
    assert _index(path).validate(thorough=True).ok


def test_concurrent_identical_registration_is_idempotent(tmp_path):
    path = tmp_path / "index.sqlite"
    _index(path).initialize_empty()
    procs = [
        _start_worker(f'''
result = register_root({str(path)!r}, "same", timeout=0.01, retries=50)
emit({{"generation": result.generation, "changed": result.changed}})
''')
        for _ in range(2)
    ]

    results = [_collect_worker(proc) for proc in procs]

    assert _stored_root_count(path) == 1
    assert _generation(path) == 1
    assert sorted(result["changed"] for result in results) == [False, True]
    sqlite3 = require_sqlite()
    con = sqlite3.connect(path)
    try:
        assert set(row[0] for row in con.execute("SELECT document_frequency FROM feature_tokens")) == {1}
    finally:
        con.close()


def test_writer_retry_covers_connection_initialization(tmp_path, monkeypatch):
    """A transient connection-configuration lock uses the write retry budget."""

    index = _index(tmp_path / "index.sqlite", retries=1)
    index.initialize_empty()
    sqlite3 = require_sqlite()
    original_connection = index._connections.connection
    attempts = 0

    def connection(*, readonly=False):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        return original_connection(readonly=readonly)

    monkeypatch.setattr(index._connections, "connection", connection)
    index._run_write_transaction(lambda con: None)

    assert attempts == 2
    index.close()


def test_busy_retry_exhaustion_reports_query_index_busy(tmp_path):
    path = tmp_path / "index.sqlite"
    ready = tmp_path / "writer-ready"
    release = tmp_path / "release-writer"
    _index(path).initialize_empty()
    proc = _start_worker(f'''
idx = index({str(path)!r}, timeout=1.0, retries=0)
con = idx._connections.connection(readonly=False)
con.execute("BEGIN IMMEDIATE")
Path({str(ready)!r}).write_text("ready")
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
con.execute("ROLLBACK")
idx.close()
emit({{"released": True}})
''')

    try:
        _wait_for(ready)
        idx = _index(path, timeout=0.01, retries=0)
        root = _cdef("busy")
        with pytest.raises(QueryIndexBusy):
            idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
    finally:
        release.write_text("go")
        assert _collect_worker(proc) == {"released": True}


@pytest.mark.parametrize("captured_token", (False, True))
def test_rebuild_preserves_same_root_token_published_after_capture(tmp_path, monkeypatch, captured_token):
    """A later publication survives both empty and nonempty acknowledgement cuts."""
    from dryml.core.query.model import QueryStats

    store = _request_store(tmp_path / "store")
    index = store.open_query_index()
    root = _cdef("request-root")
    if captured_token:
        store.mark_query_index_dirty(root)
    captured = set(index._dirty_markers())
    assert bool(captured) is captured_token
    original_capture = index._capture_authority_cut
    original_clear = index._clear_dirty
    later, errors, acknowledgements = [], [], []

    def publish():
        try:
            later.append(Path(store.mark_query_index_dirty(root)))
        except BaseException as error:
            errors.append(error)

    def capture():
        cut = original_capture()
        assert set(cut[3]) == captured
        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        worker.join(10)
        assert not worker.is_alive() and not errors
        return cut

    def clear(markers, **kwargs):
        assert index._build_claim_path().exists()
        assert _stored_root_count(index.path) == 1
        assert not index._connections._leases
        acknowledgements.append(set(markers))
        return original_clear(markers, **kwargs)

    monkeypatch.setattr(index, "_capture_authority_cut", capture)
    monkeypatch.setattr(index, "_clear_dirty", clear)
    stats = QueryStats()
    index.rebuild(stats=stats)

    assert acknowledgements == [captured]
    assert stats.store_scan_count == 1 and stats.result_count == 1
    assert len(later) == 1 and set(index._dirty_markers()) == set(later)
    assert not any(marker.exists() for marker in captured)
    assert index.status().state == "dirty"
    assert not index._build_claim_path().exists() and not index._connections._leases


def test_root_registration_fails_dirty_while_rebuild_claim_is_active(tmp_path):
    path = tmp_path / "index.sqlite"
    dirty_path = tmp_path / "index.dirty"
    idx = SQLiteStoreQueryIndex(
        source_key="concurrency-store",
        path=path,
        config=SQLiteQueryIndexConfig(journal_mode="delete"),
        dirty_path=dirty_path,
    )
    idx.initialize_empty()
    idx._build_claim_path().write_text("active\n")
    root = _cdef("overlap")

    with pytest.raises(QueryIndexBusy, match="rebuild"):
        idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])

    assert idx._is_dirty()
    assert idx.status().state == "dirty"


def test_ensure_ready_does_not_initialize_while_build_claim_is_active(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    owner = _index(path)
    contender = _index(path)
    monkeypatch.setattr(sqlite_index_module, "_BUILD_CLAIM_WAIT_SECONDS", 0.05)

    with owner._build_claim(force=True) as acquired:
        assert acquired
        with pytest.raises(QueryIndexBusy, match="Timed out"):
            contender._ensure_ready()

    assert not path.exists()


def test_live_stale_build_claim_is_not_stolen(tmp_path, monkeypatch):
    path = tmp_path / "index.sqlite"
    ready = tmp_path / "claim-owner-ready"
    release = tmp_path / "release-claim-owner"
    contender = _index(path)
    monkeypatch.setattr(sqlite_index_module, "_BUILD_CLAIM_WAIT_SECONDS", 0.05)
    proc = _start_worker(f'''
import os

idx = index({str(path)!r})
with idx._build_claim(force=True) as acquired:
    assert acquired
    old_time = time.time() - 1000
    os.utime(idx._build_claim_path(), (old_time, old_time))
    Path({str(ready)!r}).write_text("ready")
    while not Path({str(release)!r}).exists():
        time.sleep(0.01)
emit({{"released": True}})
''')

    try:
        _wait_for(ready)
        with pytest.raises(QueryIndexBusy, match="Timed out"):
            with contender._build_claim(force=True):
                pass
    finally:
        release.write_text("go")
        assert _collect_worker(proc) == {"released": True}

    assert not contender._build_claim_path().exists()


def test_build_claim_owner_does_not_remove_successor_claim(tmp_path):
    path = tmp_path / "index.sqlite"
    index = _index(path)
    claim_path = index._build_claim_path()
    claim_path.write_text("owner\n", encoding="utf-8")
    owner_identity = claim_path.stat()
    successor_path = claim_path.with_suffix(".successor")
    successor_path.write_text("successor\n", encoding="utf-8")
    os.replace(successor_path, claim_path)

    index._unlink_claim_if_owned(claim_path, owner_identity)

    assert claim_path.exists()
    claim_path.unlink()


def test_windows_claim_seam_unlocks_and_closes_before_cleanup(tmp_path, monkeypatch):
    """The Windows claim path releases its handle before unlinking the claim."""

    class FakeMSVCRT:
        LK_NBLCK = 1
        LK_UNLCK = 2
        calls = []

        @classmethod
        def locking(cls, fd, operation, size):
            cls.calls.append((fd, operation, size))

    index = _index(tmp_path / "index.sqlite")
    closed = False
    original_close = os.close
    original_unlink = index._unlink_claim_if_owned

    def close(fd):
        nonlocal closed
        closed = True
        original_close(fd)

    def unlink_after_close(path, owner_identity):
        assert closed
        original_unlink(path, owner_identity)

    monkeypatch.setattr(locking_module, "fcntl", None)
    monkeypatch.setattr(locking_module, "msvcrt", FakeMSVCRT)
    monkeypatch.setattr(sqlite_index_module.os, "close", close)
    monkeypatch.setattr(index, "_unlink_claim_if_owned", unlink_after_close)

    with index._build_claim(force=True) as acquired:
        assert acquired

    assert [call[1] for call in FakeMSVCRT.calls] == [FakeMSVCRT.LK_NBLCK, FakeMSVCRT.LK_UNLCK]
    assert not index._build_claim_path().exists()


def test_rebuild_closes_managed_canonical_handles_before_checkpoint(tmp_path, monkeypatch):
    """Rebuild releases pooled canonical handles before sidecar cleanup."""

    store = DirStore(
        tmp_path / "store",
        query_index=SQLiteQueryIndexConfig(journal_mode="delete"),
    )
    _save_root_definition(store, _cdef("checkpoint-barrier"))
    index = store.open_query_index()
    index.initialize_empty()
    assert index._connections._connections
    original_checkpoint = index._checkpoint_and_cleanup_sidecars
    observed_canonical = False

    def checkpoint(path, *, label, allow_corrupt=False):
        nonlocal observed_canonical
        if label == "canonical":
            observed_canonical = True
            assert not index._connections._connections
        return original_checkpoint(path, label=label, allow_corrupt=allow_corrupt)

    monkeypatch.setattr(index, "_checkpoint_and_cleanup_sidecars", checkpoint)
    index.rebuild()

    assert observed_canonical


def test_wal_rebuild_waits_for_reader_before_replacing_canonical_sidecar(tmp_path):
    """A live WAL reader prevents mixed-generation sidecar activation."""

    store = DirStore(
        tmp_path / "store",
        query_index=SQLiteQueryIndexConfig(journal_mode="wal", busy_timeout=0.01, max_write_retries=0),
    )
    root = _cdef("wal-barrier")
    _save_root_definition(store, root)
    index = store.open_query_index()
    try:
        index.initialize_empty()
    except QueryIndexError as exc:
        pytest.skip(f"WAL mode unavailable in this environment: {exc}")
    store.mark_query_index_dirty()
    sqlite3 = require_sqlite()
    reader = sqlite3.connect(store.query_index_path, isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM catalog_state").fetchall()
        with pytest.raises(QueryIndexBusy, match="canonical sidecar checkpoint"):
            store.rebuild_query_index()
        assert store.query_index_is_dirty()
        assert not list(Path(store.query_index_path).parent.glob(f"{Path(store.query_index_path).name}.rebuild-*.tmp*"))
    finally:
        reader.execute("ROLLBACK")
        reader.close()

    store.rebuild_query_index()
    assert not list(Path(store.query_index_path).parent.glob(f"{Path(store.query_index_path).name}.rebuild-*.tmp*"))
    assert store.query_index_status().state == "ready"


def test_writer_busy_retry_succeeds_after_lock_release(tmp_path):
    path = tmp_path / "index.sqlite"
    ready = tmp_path / "writer-ready"
    release = tmp_path / "release-writer"
    _index(path).initialize_empty()
    proc = _start_worker(f'''
idx = index({str(path)!r}, timeout=1.0, retries=0)
con = idx._connections.connection(readonly=False)
con.execute("BEGIN IMMEDIATE")
Path({str(ready)!r}).write_text("ready")
while not Path({str(release)!r}).exists():
    time.sleep(0.01)
con.execute("ROLLBACK")
idx.close()
emit({{"released": True}})
''')

    try:
        _wait_for(ready)
        writer = _start_worker(f'''
result = register_root({str(path)!r}, "retry-success", timeout=0.01, retries=50)
emit({{"generation": result.generation, "changed": result.changed}})
''')
        time.sleep(0.05)
        release.write_text("go")
        assert _collect_worker(proc) == {"released": True}
        assert _collect_worker(writer) == {"generation": 1, "changed": True}
        assert _stored_root_count(path) == 1
    finally:
        release.write_text("go")
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=5)


def test_spawned_worker_opens_process_local_connection(tmp_path):
    path = tmp_path / "index.sqlite"
    idx = _index(path)
    idx.initialize_empty()
    parent_con = idx._connections.connection(readonly=False)
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_spawn_register_worker, args=(str(path), queue))

    process.start()
    process.join(timeout=10)

    assert process.exitcode == 0
    result = queue.get(timeout=1)
    assert result == {"generation": 1, "same_connection": True}
    assert idx._connections.connection(readonly=False) is parent_con
    with idx.read_view() as view:
        assert len(view.exact_ids(_cdef("spawn-child"))) == 1


def test_forked_child_uses_child_process_connection(tmp_path):
    path = tmp_path / "index.sqlite"
    result = _run_worker(f'''
import os

if not hasattr(os, "fork"):
    emit({{"skipped": True}})
else:
    idx = index({str(path)!r}, timeout=0.05, retries=20)
    idx.initialize_empty()
    parent_con = idx._connections.connection(readonly=False)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            child_con = idx._connections.connection(readonly=False)
            root = cdef("fork-child")
            write_result = idx.register_stored_roots(ConcreteDefinitionGraph.from_root(root), [root])
            payload = json.dumps({{"new_connection": child_con is not parent_con, "generation": write_result.generation}}).encode()
            os.write(write_fd, payload)
            os._exit(0)
        except BaseException as exc:
            os.write(write_fd, json.dumps({{"error": repr(exc)}}).encode())
            os._exit(1)
    os.close(write_fd)
    _, status = os.waitpid(pid, 0)
    payload = os.read(read_fd, 4096)
    os.close(read_fd)
    data = json.loads(payload.decode())
    with idx.read_view() as view:
        data["visible"] = len(view.exact_ids(cdef("fork-child"))) == 1
    data["status"] = status
    emit(data)
''')

    if result.get("skipped"):
        pytest.skip("requires os.fork")
    assert result == {"new_connection": True, "generation": 1, "visible": True, "status": 0}


def test_process_termination_before_commit_rolls_back_partial_rows(tmp_path):
    path = tmp_path / "index.sqlite"
    ready = tmp_path / "writer-ready"
    idx = _index(path)
    idx.initialize_empty()

    proc = _start_worker(f'''
idx = index({str(path)!r}, timeout=0.05, retries=0)
root = cdef("crash-before-commit")
con = idx._connections.connection(readonly=False)
con.execute("BEGIN IMMEDIATE")
generation = _read_generation(con)
next_generation = generation + 1
encoded = _EncodedNode.from_cdef(root)
did, _ = _resolve_definition_id(con, encoded, generation=next_generation)
con.execute(
    "INSERT OR IGNORE INTO stored_roots (def_id, storage_hash, relative_def_path, def_size, def_mtime_ns, indexed_generation) VALUES (?, ?, ?, NULL, NULL, ?)",
    (did, stable_hash_to_blob(root.stable_hash()), _relative_def_path(root.stable_hash()), next_generation),
)
_bump_generation(con, next_generation)
Path({str(ready)!r}).write_text("ready")
while True:
    time.sleep(1)
''')

    _wait_for(ready)
    proc.kill()
    proc.wait(timeout=5)

    assert _stored_root_count(path) == 0
    assert _generation(path) == 0
    assert idx.validate(thorough=True).ok
    with idx.read_view() as view:
        assert view.exact_ids(_cdef("crash-before-commit")) == set()


def test_dirty_marker_recovers_object_committed_without_index_in_separate_process(tmp_path):
    store = DirStore(
        tmp_path,
        query_index=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=0.05, max_write_retries=20),
    )
    root = _cdef("object-without-index")
    _save_root_definition(store, root)
    idx = store.open_query_index()
    idx.initialize_empty()
    store.mark_query_index_dirty()
    idx.close()

    result = _run_worker(f'''
idx = store_index({str(tmp_path)!r})
idx.refresh("auto")
with idx.read_view() as view:
    ids = view.exact_ids(cdef("object-without-index"))
    emit({{"count": len(ids), "generation": view.generation}})
''')

    assert result["count"] == 1
    assert result["generation"] > 0
    assert not store.query_index_is_dirty()
    with store.open_query_index().read_view() as view:
        assert len(view.exact_ids(root)) == 1


def test_crash_mid_rebuild_leaves_building_state_and_recovers(tmp_path):
    ready = tmp_path / "rebuild-started"
    store = DirStore(
        tmp_path,
        query_index=SQLiteQueryIndexConfig(journal_mode="delete", busy_timeout=0.05, max_write_retries=20),
    )
    root = _cdef("building-recovery")
    _save_root_definition(store, root)
    store.clear_query_index_dirty()

    proc = _start_worker(f'''
idx = store_index({str(tmp_path)!r})
idx.initialize_empty(build_state="building")
Path({str(ready)!r}).write_text("ready")
while True:
    time.sleep(1)
''')

    _wait_for(ready)
    proc.kill()
    proc.wait(timeout=5)

    idx = store.open_query_index()

    assert idx.status().state == "building"
    report = idx.validate(thorough=True)
    assert not report.ok
    assert any("not ready" in issue.message for issue in report.issues)
    idx.close()

    result = _run_worker(f'''
idx = store_index({str(tmp_path)!r})
idx.refresh("auto")
with idx.read_view() as view:
    emit({{"state": idx.status().state, "count": len(view.exact_ids(cdef("building-recovery"))), "generation": view.generation}})
''')

    assert result["state"] == "ready"
    assert result["count"] == 1
    assert result["generation"] > 0


def _wait_for(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {path}")
