import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.model import QueryIndexBusy, QueryIndexError
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig, require_sqlite, sqlite_available
from dryml.core2.query.sqlite.index import SQLiteStoreQueryIndex
from dryml.core2.repo_plan import SaveAction, SavePlan, execute_save_plan
from dryml.core2.store.dir import DirStore
from dryml.core2.symbol import ImportRef
from dryml.core2.utils.general import pickle_save


pytestmark = pytest.mark.skipif(not sqlite_available(), reason="sqlite3 is unavailable")


_COMMON_WORKER_CODE = r'''
import json
from pathlib import Path
import time

from dryml.core2.cdef_graph import ConcreteDefinitionGraph
from dryml.core2.definition import ConcreteDefinition
from dryml.core2.freeze import FrozenDict, FrozenTuple
from dryml.core2.query.sqlite import SQLiteQueryIndexConfig
from dryml.core2.query.sqlite.index import (
    SQLiteStoreQueryIndex,
    _EncodedNode,
    _bump_generation,
    _read_generation,
    _relative_def_path,
    _resolve_definition_id,
)
from dryml.core2.query.utils import stable_hash_to_blob
from dryml.core2.store.dir import DirStore
from dryml.core2.symbol import ImportRef


def cdef(name):
    return ConcreteDefinition._from_persisted_record(
        ImportRef("builtins", "dict"),
        FrozenTuple((name,)),
        FrozenDict({}),
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
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=_worker_env(),
    )
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
    return ConcreteDefinition._from_persisted_record(
        ImportRef("builtins", "dict"),
        FrozenTuple((name,)),
        FrozenDict({}),
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
    Path(store.object_dir(cdef)).mkdir(parents=True, exist_ok=True)
    pickle_save(cdef, store._def_file(cdef))


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

    result = _run_worker(f'''
idx = store_index({str(tmp_path)!r})
idx.refresh("auto")
with idx.read_view() as view:
    emit({{"state": idx.status().state, "count": len(view.exact_ids(cdef("building-recovery"))), "generation": view.generation}})
''')

    assert result["state"] == "ready"
    assert result["count"] == 1
    assert result["generation"] > 0


def test_save_plan_does_not_register_index_before_object_publication():
    root = _cdef("save-order")
    graph = ConcreteDefinitionGraph.from_root(root)

    class FailingStore:
        def save_object(self, obj, *, revision=None):
            raise RuntimeError("object publication failed")

    class QueryCatalog:
        def __init__(self):
            self.calls = []

        def store_id(self, store):
            return "failing-store"

        def register_graph(self, graph):
            self.calls.append("register_graph")

        def register_stored_root(self, definition, store):
            self.calls.append("register_stored_root")

    class QueryIndex:
        def __init__(self):
            self.calls = []

        def register_saved_graph(self, graph, roots_by_store):
            self.calls.append((graph, roots_by_store))

    class RepoStub:
        def __init__(self):
            self._query_catalog = QueryCatalog()
            self._query_index = QueryIndex()
            self._num_saves = 0

    repo = RepoStub()
    store = FailingStore()
    plan = SavePlan(
        graph=graph,
        binding=None,
        actions=(SaveAction(root, object(), store, None, 0, "explicit-root"),),
    )

    with pytest.raises(RuntimeError, match="object publication failed"):
        execute_save_plan(repo, plan)

    assert repo._query_catalog.calls == []
    assert repo._query_index.calls == []
    assert repo._num_saves == 0


def _wait_for(path: Path, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise TimeoutError(f"Timed out waiting for {path}")
