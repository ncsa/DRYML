import asyncio
import contextvars
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

import dryml.core.execute as execute_module
from dryml.core import Repo
from dryml.core.execute import ExecutionContext, core_worker_setup, current_context, worker_context
from dryml.core.session import config, get_config
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml import session
from dryml.execute import Executor, WorkerSetup
from dryml.execute.models import WorkerSetupContext
from dryml.execute.subprocess import SubProcessConfig
from dryml.formats import make_envelope, semantic_id
from dryml.runtime import RuntimeContextSpec, RuntimeMode, active_runtime


def test_worker_context_is_task_owned_and_does_not_leak():
    repo = object()
    control = object()
    with pytest.raises(RuntimeError, match="outside an active worker setup"):
        current_context()

    with worker_context(ExecutionContext(repo, control)):
        assert current_context().repo is repo
        assert current_context().control_store is control

    with pytest.raises(RuntimeError, match="outside an active worker setup"):
        current_context()


def test_copied_async_task_cannot_use_worker_context():
    async def check():
        with worker_context(ExecutionContext(object(), None)):
            task = asyncio.create_task(read_context())
            return await task

    async def read_context():
        with pytest.raises(RuntimeError, match="different task"):
            current_context()

    asyncio.run(check())


def test_copied_thread_cannot_use_worker_context():
    captured = []

    def read_context():
        with pytest.raises(RuntimeError, match="different thread"):
            current_context()
        captured.append(True)

    with worker_context(ExecutionContext(object(), None)):
        copied = contextvars.copy_context()
        thread = threading.Thread(target=lambda: copied.run(read_context))
        thread.start()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert captured == [True]


def _setup_data(tmp_path, *, control_store=None):
    store = DirStore(tmp_path / "state", query_index="none")
    repo = Repo(store)
    definition = repo.to_definition().to_data()
    repo.close(flush=False)
    store.close()
    payload = {
        "runtime": RuntimeContextSpec(RuntimeMode.INLINE).to_data(),
        "repo": definition,
        "role": "main",
        "replica": 0,
        "control_store": control_store,
    }
    return make_envelope(
        schema="dryml.core.execute.v1.1",
        kind="worker_setup",
        prefix="core_setup",
        payload=payload,
        semantic_id=semantic_id(
            "core_setup", "dryml.core.execute.v1.1", "worker_setup", payload,
            max_depth=64, max_nodes=65_536, max_entries=65_536,
        ),
        max_depth=64,
        max_nodes=65_536,
        max_entries=65_536,
    )


def _ray_context():
    return WorkerSetupContext(
        submission_id="worker-context", backend="ray", environment=None,
        allocation=None, native_grant={"kind": "ray", "resources": {"CPU": 1}},
    )


def _worker_context_snapshot():
    """Return worker-local core/runtime facts without serializing live handles."""
    context = current_context()
    allocation = active_runtime().allocation
    return {
        "repo_stores": len(context.repo.stores),
        "control_store": context.control_store is None,
        "mode": active_runtime().mode.value,
        "cpus": allocation.cpus,
        "world_allocation_id": allocation.world_allocation_id,
        "grant_provenance": allocation.grant_provenance,
    }


def test_reconstruction_failure_unwinds_runtime_before_payload_delivery(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def activation(*args, **kwargs):
        events.append("runtime-enter")
        try:
            yield None
        finally:
            events.append("runtime-exit")

    def fail_reconstruction(cls, definition):
        events.append("reconstruct")
        raise OSError("state unavailable")

    monkeypatch.setattr(execute_module, "activation_scope", activation)
    monkeypatch.setattr(Repo, "from_definition", classmethod(fail_reconstruction))

    with pytest.raises(OSError, match="state unavailable"):
        with core_worker_setup(_ray_context(), _setup_data(tmp_path)):
            pass
    assert events == ["runtime-enter", "reconstruct", "runtime-exit"]


def test_control_store_failure_closes_reconstructed_repo_before_runtime(tmp_path, monkeypatch):
    events = []
    data = _setup_data(
        tmp_path,
        control_store={"kind": "dir", "path": str(tmp_path / "control"), "query_index": "none"},
    )

    @contextmanager
    def activation(*args, **kwargs):
        events.append("runtime-enter")
        try:
            yield None
        finally:
            events.append("runtime-exit")

    original_from_definition = Repo.from_definition.__func__
    original_close = Repo.close

    def reconstruct(cls, definition):
        events.append("reconstruct")
        return original_from_definition(cls, definition)

    def close_repo(self, *, flush):
        assert flush is False
        events.append("repo-close")
        return original_close(self, flush=flush)

    def open_control(cls, definition):
        events.append("control-open")
        raise OSError("control unavailable")

    monkeypatch.setattr(execute_module, "activation_scope", activation)
    monkeypatch.setattr(Repo, "from_definition", classmethod(reconstruct))
    monkeypatch.setattr(Repo, "close", close_repo)
    monkeypatch.setattr(Store, "from_definition", classmethod(open_control))

    with pytest.raises(OSError, match="control unavailable"):
        with core_worker_setup(_ray_context(), data):
            pass
    assert events == ["runtime-enter", "reconstruct", "control-open", "repo-close", "runtime-exit"]


def test_setup_reconstructs_one_repo_and_deduplicates_its_control_store(tmp_path):
    before = get_config()
    data = _setup_data(tmp_path, control_store={"repo_store": 0})
    with core_worker_setup(_ray_context(), data):
        active = current_context()
        assert active.control_store is active.repo.stores[0]
        assert get_config().repo is active.repo
    assert get_config() is before


def test_setup_cache_reuses_the_reconstructed_repo_and_shared_control_store(tmp_path):
    """Worker reconstruction uses the Session cache already active for setup."""
    data = _setup_data(tmp_path, control_store={"repo_store": 0})
    with core_worker_setup(_ray_context(), data):
        active = current_context()
        cache = session.current_resource_cache()
        assert cache is not None
        assert active.repo in cache.repos
        assert active.control_store in cache.stores
        assert Repo.from_definition(active.repo.to_definition()) is active.repo


def test_setup_cache_reuses_a_separate_control_store(tmp_path):
    """A separate worker control role is registered through the same cache."""
    control_path = tmp_path / "control"
    DirStore(control_path, query_index="none").close()
    data = _setup_data(
        tmp_path,
        control_store={"kind": "dir", "path": str(control_path), "query_index": "none"},
    )
    with core_worker_setup(_ray_context(), data):
        control = current_context().control_store
        assert control is not None
        assert control in session.current_resource_cache().stores
        assert Store.from_definition(control.to_definition()) is control


def test_reused_worker_setup_starts_with_fresh_cache_and_selection(tmp_path):
    """Serial worker setup scopes do not retain prior cache or Repo selection."""
    before = get_config()
    caches = []
    repos = []
    for _ in range(2):
        with core_worker_setup(_ray_context(), _setup_data(tmp_path)):
            caches.append(session.current_resource_cache())
            repos.append(current_context().repo)
            assert get_config().repo is repos[-1]
        assert session.current_resource_cache() is None
        assert get_config() is before
        with pytest.raises(RuntimeError, match="outside an active worker setup"):
            current_context()
    assert caches[0] is not caches[1]
    assert repos[0] is not repos[1]


@pytest.mark.parametrize("phase", ("cache", "repo", "control", "session", "context"))
def test_setup_failures_restore_every_worker_local_context(tmp_path, monkeypatch, phase):
    """Every setup acquisition failure restores selection, cache, context, and runtime."""
    control_path = tmp_path / "control"
    DirStore(control_path, query_index="none").close()
    data = _setup_data(
        tmp_path,
        control_store={"kind": "dir", "path": str(control_path), "query_index": "none"},
    )
    before_config = get_config()
    before_runtime = active_runtime()

    @contextmanager
    def fail_scope(*args, **kwargs):
        raise RuntimeError(f"{phase} setup failure")
        yield None

    if phase == "cache":
        monkeypatch.setattr(session, "resource_cache", fail_scope)
    elif phase == "repo":
        monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: (_ for _ in ()).throw(
            RuntimeError("repo setup failure"),
        )))
    elif phase == "control":
        monkeypatch.setattr(execute_module, "_control_store", lambda repo, descriptor: (_ for _ in ()).throw(
            RuntimeError("control setup failure"),
        ))
    elif phase == "session":
        monkeypatch.setattr(execute_module, "config", fail_scope)
    else:
        monkeypatch.setattr(execute_module, "worker_context", fail_scope)

    with pytest.raises(RuntimeError, match=f"{phase} setup failure"):
        with core_worker_setup(_ray_context(), data):
            pass
    assert get_config() is before_config
    assert active_runtime() == before_runtime
    assert session.current_resource_cache() is None
    with pytest.raises(RuntimeError, match="outside an active worker setup"):
        current_context()


def test_setup_unwinds_every_acquired_resource_in_lifo_order(tmp_path, monkeypatch):
    events = []

    @contextmanager
    def activation(*args, **kwargs):
        events.append("runtime-enter")
        try:
            yield None
        finally:
            events.append("runtime-exit")

    @contextmanager
    def session_config(*args, **kwargs):
        events.append("session-enter")
        try:
            yield None
        finally:
            events.append("session-exit")

    original_resource_cache = session.resource_cache

    @contextmanager
    def cache_scope():
        events.append("cache-enter")
        with original_resource_cache() as cache:
            yield cache
        events.append("cache-exit")

    @contextmanager
    def context(value):
        events.append("context-enter")
        try:
            yield value
        finally:
            events.append("context-exit")

    control_path = tmp_path / "control"
    DirStore(control_path, query_index="none").close()
    data = _setup_data(
        tmp_path,
        control_store={"kind": "dir", "path": str(control_path), "query_index": "none"},
    )
    original_repo_close = Repo.close
    original_store_close = DirStore.close

    def close_repo(self, *, flush):
        assert flush is False
        events.append("repo-close")
        return original_repo_close(self, flush=flush)

    def close_store(self):
        events.append("control-close" if Path(self.base_dir) == control_path else "repo-store-close")
        return original_store_close(self)

    monkeypatch.setattr(execute_module, "activation_scope", activation)
    monkeypatch.setattr(session, "resource_cache", cache_scope)
    monkeypatch.setattr(execute_module, "config", session_config)
    monkeypatch.setattr(execute_module, "worker_context", context)
    monkeypatch.setattr(Repo, "close", close_repo)
    monkeypatch.setattr(DirStore, "close", close_store)

    with core_worker_setup(
        _ray_context(), data,
    ):
        events.append("body")

    assert events == [
        "runtime-enter", "cache-enter", "session-enter", "context-enter", "body",
        "context-exit", "session-exit", "repo-close", "control-close",
        "repo-store-close", "cache-exit", "runtime-exit",
    ]


def test_core_setup_rejects_legacy_aliases_before_runtime_or_store_opening(tmp_path, monkeypatch):
    opened = []
    activated = []
    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: opened.append(definition)))
    monkeypatch.setattr(execute_module, "activation_scope", lambda *args: activated.append(args))
    legacy = {
        "runtime_spec": RuntimeContextSpec(RuntimeMode.INLINE).to_data(),
        "repo_definition": _setup_data(tmp_path)["payload"]["repo"],
    }

    with pytest.raises(ValueError, match="v1.1 envelope"):
        with core_worker_setup(_ray_context(), legacy):
            pass
    assert opened == []
    assert activated == []


def test_core_setup_rejects_preinitialized_session_before_store_opening(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: opened.append(definition)))

    with config(repo=Repo()):
        with pytest.raises(RuntimeError, match="pristine core session"):
            with core_worker_setup(_ray_context(), _setup_data(tmp_path)):
                pass
    assert opened == []


def test_core_setup_rejects_preloaded_framework_before_store_opening(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: opened.append(definition)))
    monkeypatch.setitem(__import__("sys").modules, "torch", object())

    with pytest.raises(Exception, match="framework was imported"):
        with core_worker_setup(_ray_context(), _setup_data(tmp_path)):
            pass
    assert opened == []


def test_default_subprocess_core_setup_uses_a_no_extra_controls_grant(tmp_path):
    """A normal generic subprocess does not need an otherwise-unused world."""
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = Executor(SubProcessConfig(spool_directory=spool))
    setup = WorkerSetup(
        factory="dryml.core.execute:core_worker_setup",
        data=_setup_data(tmp_path),
    )
    try:
        future = executor.submit(_worker_context_snapshot, worker_setup=setup)
        assert future.result(timeout=10) == {
            "repo_stores": 1,
            "control_store": True,
            "mode": "inline",
            "cpus": (),
            "world_allocation_id": None,
            "grant_provenance": "baseline",
        }
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=5)
