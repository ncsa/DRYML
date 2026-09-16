import asyncio
import contextvars
import threading
from contextlib import contextmanager

import pytest

import dryml.core.execute as execute_module
from dryml.core import Repo
from dryml.core.execute import ExecutionContext, core_worker_setup, current_context, worker_context
from dryml.core.session import config, get_config
from dryml.core.store.dir import DirStore
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

    @contextmanager
    def activation(*args, **kwargs):
        events.append("runtime-enter")
        try:
            yield None
        finally:
            events.append("runtime-exit")

    class ReconstructedRepo:
        stores = ()

        def close(self, *, flush):
            assert flush is False
            events.append("repo-close")

    def open_control(cls, path, *, query_index):
        events.append("control-open")
        raise OSError("control unavailable")

    monkeypatch.setattr(execute_module, "activation_scope", activation)
    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: ReconstructedRepo()))
    monkeypatch.setattr(DirStore, "open_existing", classmethod(open_control))
    data = _setup_data(
        tmp_path,
        control_store={"kind": "dir", "path": str(tmp_path / "control"), "query_index": "none"},
    )

    with pytest.raises(OSError, match="control unavailable"):
        with core_worker_setup(_ray_context(), data):
            pass
    assert events == ["runtime-enter", "control-open", "repo-close", "runtime-exit"]


def test_setup_reconstructs_one_repo_and_deduplicates_its_control_store(tmp_path):
    before = get_config()
    data = _setup_data(tmp_path, control_store={"repo_store": 0})
    with core_worker_setup(_ray_context(), data):
        active = current_context()
        assert active.control_store is active.repo.stores[0]
        assert get_config().repo is active.repo
    assert get_config() is before


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

    @contextmanager
    def context(value):
        events.append("context-enter")
        try:
            yield value
        finally:
            events.append("context-exit")

    class ReconstructedRepo:
        stores = ()

        def close(self, *, flush):
            assert flush is False
            events.append("repo-close")

    class ControlStore:
        def close(self):
            events.append("control-close")

    monkeypatch.setattr(execute_module, "activation_scope", activation)
    monkeypatch.setattr(execute_module, "config", session_config)
    monkeypatch.setattr(execute_module, "worker_context", context)
    monkeypatch.setattr(execute_module, "_control_store", lambda repo, descriptor: (ControlStore(), True))
    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: ReconstructedRepo()))

    with core_worker_setup(
        _ray_context(),
        _setup_data(tmp_path, control_store={"kind": "dir", "path": str(tmp_path / "control"), "query_index": "none"}),
    ):
        events.append("body")

    assert events == [
        "runtime-enter", "session-enter", "context-enter", "body", "context-exit",
        "session-exit", "control-close", "repo-close", "runtime-exit",
    ]


def test_core_setup_rejects_legacy_aliases_before_runtime_or_store_opening(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(Repo, "from_definition", classmethod(lambda cls, definition: opened.append(definition)))
    legacy = {
        "runtime_spec": RuntimeContextSpec(RuntimeMode.INLINE).to_data(),
        "repo_definition": _setup_data(tmp_path)["payload"]["repo"],
    }

    with pytest.raises(ValueError, match="v1.1 envelope"):
        with core_worker_setup(_ray_context(), legacy):
            pass
    assert opened == []


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
