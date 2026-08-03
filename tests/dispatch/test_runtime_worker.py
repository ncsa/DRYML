import sys

from dryml import session
from dryml.core.store.dir import DirStore
from dryml.dispatch import Dispatcher
from dryml.environments import PythonExecutableSpec
from dryml.operations import attach_operation_id, make_function_call_spec
from dryml.runtime import RuntimeEnforcement, RuntimeState
from dryml.runtime.publication import PublicationService


def test_worker_runtime_active_before_target_import(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    op = attach_operation_id(make_function_call_spec("dispatch_target:runtime_status"))

    result = Dispatcher(store=store).run(op, environment=env)

    assert result.result_canonical["mode"] == "worker"
    assert result.result_canonical["bootstrap"] == "1"
    assert result.result_canonical["import_mode"] == "worker"
    assert result.result_canonical["enforcement"] == "strict"
    assert result.result_canonical["selected_environment"] == "python"
    assert result.result_canonical["selected_world"] == ["main"]
    assert result.result_canonical["selected_runtime"] == "worker"
    assert result.result_canonical["compatibility_policy"] == "strict"
    assert result.result_canonical["compatibility_axes"] == ["environment", "world", "runtime"]


def test_worker_configured_framework_import_succeeds_with_runtime_spec(tmp_path, target_module):
    store = DirStore(tmp_path / "store", query_index="none")
    env = PythonExecutableSpec(sys.executable, pythonpath_policy="explicit", extra_pythonpath=(str(target_module.parent),)).to_data()
    runtime = {"mode": "worker", "frameworks": {"torch": {"num_threads": 3}}, "device_visibility": {"policy": "assigned"}}
    op = attach_operation_id(make_function_call_spec("dispatch_target:configured_torch_import_status"))

    result = Dispatcher(store=store).run(op, environment=env, runtime=runtime)

    assert result.status == "ok"
    assert result.result_canonical["mode"] == "worker"
    assert result.result_canonical["bootstrap"] == "1"
    assert result.result_canonical["marker"] == "fake-dispatch-torch"
    assert result.result_canonical["threads"] == 3
    assert result.result_canonical["cuda_visible_devices"] == ""


def test_worker_session_is_published_before_store_and_handshake_work(
    tmp_path, monkeypatch
):
    import dryml.dispatch.worker as worker
    import dryml.runtime.context as context
    import dryml.runtime.guards as guards
    import dryml.session.state as state

    affinity = {0, 1}
    service = PublicationService(
        environ={},
        affinity_getter=lambda: affinity,
        affinity_setter=lambda cpus: (affinity.clear(), affinity.update(cpus)),
    )
    service.initialize(RuntimeState(enforcement=RuntimeEnforcement.OFF))
    monkeypatch.setattr(state, "publication", service)
    monkeypatch.setattr(context, "publication", service)
    monkeypatch.setattr(guards, "publication", service)
    monkeypatch.setattr(state, "_loaded_framework_roots", lambda: ())

    store = DirStore(tmp_path / "store", query_index="none")
    plan = Dispatcher(store=store).plan(
        attach_operation_id(make_function_call_spec("operator:add", args=[1, 2])),
        record_policy="none",
    )
    request = tmp_path / "request.json"
    handshake = tmp_path / "handshake.json"
    response = tmp_path / "response.json"
    worker.write_json_file(str(request), plan.envelope.to_json())
    events = []

    def inspect_before_store(envelope, *, features):
        snapshot = session.current()
        assert snapshot.runtime.spec is not None
        assert snapshot.runtime.spec.to_data() == envelope.runtime_spec
        assert snapshot.selected_runtime.to_data() == envelope.runtime_spec
        assert snapshot.compatibility_axes.to_data() == list(envelope.requirement_axes)
        assert service.current().metadata["framework_results"]
        events.append("store")
        return [], {}, False, ({"message": "stop before Store access"},)

    real_write = worker.write_json_file

    def record_write(path, data):
        if path == str(handshake):
            events.append("handshake")
        real_write(path, data)

    monkeypatch.setattr(worker, "_open_and_validate_stores", inspect_before_store)
    monkeypatch.setattr(worker, "write_json_file", record_write)

    status = worker.main(
        [
            "--request",
            str(request),
            "--handshake",
            str(handshake),
            "--response",
            str(response),
        ]
    )

    assert status == 1
    assert events == ["store", "handshake"]
