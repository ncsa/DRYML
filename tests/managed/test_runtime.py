from __future__ import annotations

from pathlib import Path

import pytest

from dryml.core2 import Object, Repo
from dryml.core2.repo import default_repo
from dryml.core2.store.dir import DirStore
from dryml.managed import (
    AmbiguousManagedStoreError,
    ControlRequest,
    ManagedCallback,
    ManagedCapabilityError,
    ManagedInterruptedError,
    ManagedOutput,
    ManagedRerunRequiredError,
    ManagedStateError,
    OperationPreflight,
    OperationResult,
    StaleManagedResultError,
    current_operation_context,
    managed,
)
from dryml.records import make_representation_spec


REPRESENTATION = make_representation_spec("fake.bytes", version="1", storage_kinds=("product-dir",))


class FakeOperation(Object):
    def __init__(self, family="compute", resumable=True, early_completion=True):
        super().__init__()
        self.family = family
        self.resumable = resumable
        self.early_completion = early_completion

    def __dryml_managed_preflight__(self, method, args, kwargs):
        return OperationPreflight(
            resumable=self.resumable,
            checkpoint_schema="fake-checkpoint-v1" if self.resumable else None,
            early_completion=self.early_completion,
        )

    def _run(self, value=b"value", fail=None):
        context = current_operation_context()
        context.progress(1, total=2, message=self.family)

        def checkpoint():
            if fail == "checkpoint":
                raise OSError("checkpoint production failed")
            context.write_checkpoint("cursor.txt", (b"1",))

        control = context.safe_point(checkpoint=checkpoint)
        if fail == "interrupt-before":
            raise KeyboardInterrupt("managed-interrupt-secret-sentinel-41c8")
        if fail == "before":
            raise RuntimeError("failed before checkpoint")
        if fail == "after":
            checkpoint()
            context.commit_checkpoint()
            raise RuntimeError("failed after checkpoint")
        if fail == "interrupt-after":
            checkpoint()
            context.commit_checkpoint()
            raise KeyboardInterrupt("managed-interrupt-secret-sentinel-41c8")
        if fail is not None:
            raise RuntimeError(str(fail))
        early = control is ControlRequest.GRACEFUL_STOP
        context.write_output(
            "result",
            "value.bin",
            (value,),
            representation=REPRESENTATION,
        )
        context.progress(2, total=2, message=self.family)
        return OperationResult(early_completed=early)

    @managed(
        outputs=(ManagedOutput("result", primary=True, kind="data"),),
        resumable=True,
        checkpoint_schema="fake-checkpoint-v1",
        early_completion=True,
    )
    def compute(self, value=b"value", fail=None):
        return self._run(value=value, fail=fail)

    @managed(
        outputs=(ManagedOutput("result", primary=True, kind="data"),),
        resumable=True,
        checkpoint_schema="fake-checkpoint-v1",
        early_completion=True,
    )
    def train(self, value=b"value", fail=None):
        return self._run(value=value, fail=fail)


class Consumer(Object):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def __dryml_managed_inputs__(self, method, args, kwargs):
        return (self.source,)

    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        context = current_operation_context()
        context.write_output("result", "value.bin", (b"consumed",), representation=REPRESENTATION)


class NonResumable(Object):
    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        context = current_operation_context()
        context.progress(1, total=1)
        context.safe_point()
        context.write_output("result", "value.bin", (b"value",), representation=REPRESENTATION)


class MissingOutput(Object):
    @managed(outputs=(ManagedOutput("result", primary=True, kind="data"),))
    def compute(self):
        return None


def _result_path(store, invocation):
    record = store.records.read_record(invocation.outputs["result"].record_id)
    return store.records.resolve_storage_ref(
        record["payload"]["storage"][0], record_id=record["id"]
    ).joinpath("value.bin")


def _read_result(store, invocation):
    return _result_path(store, invocation).read_bytes()


def test_ambiguous_default_repo_does_not_bypass_managed_runtime(tmp_path):
    repo = Repo((
        DirStore(tmp_path / "first"),
        DirStore(tmp_path / "second"),
    ))

    with default_repo(repo), pytest.raises(AmbiguousManagedStoreError):
        FakeOperation().compute()


@pytest.mark.parametrize("method", ["compute", "train"])
def test_complete_reuse_status_progress_results_and_history_have_lifecycle_parity(tmp_path, method):
    store = DirStore(tmp_path / method)
    operation = getattr(FakeOperation(family=method), method)

    completed = operation(store=store, value=b"first")
    reused = operation(store=store, value=b"first")

    assert completed.action == "start"
    assert reused.action == "reuse"
    assert reused.realization_id == completed.realization_id
    assert _read_result(store, reused) == b"first"
    assert operation.status(store=store).status == "completed"
    assert operation.progress(store=store).current == 2
    assert operation.results(store=store)["result"].record_id == completed.outputs["result"].record_id
    assert [item.realization_id for item in operation.history(store=store)] == [completed.realization_id]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_reuse_rejects_invalid_active_output_without_mutating_active_state(
    tmp_path, damage
):
    store = DirStore(tmp_path / damage)
    operation = FakeOperation().compute
    completed = operation(store=store)
    output_path = _result_path(store, completed)
    before_status = operation.status(store=store)
    before_history = operation.history(store=store)

    if damage == "missing":
        output_path.unlink()
    else:
        output_path.write_bytes(b"corrupt")

    with pytest.raises(Exception, match="integrity"):
        operation(store=store)

    assert operation.status(store=store) == before_status
    assert operation.history(store=store) == before_history
    assert operation.results(store=store) == completed.outputs


def test_changed_logical_input_is_stale_until_explicit_rerun(tmp_path):
    store = DirStore(tmp_path / "store")
    producer = FakeOperation()
    first_source = producer.compute(store=store, value=b"one")
    consumer = Consumer(producer.compute.result)
    first = consumer.compute(store=store)
    second_source = producer.compute.rerun(store=store, value=b"two")

    assert second_source.realization_id != first_source.realization_id
    with pytest.raises(StaleManagedResultError):
        consumer.compute(store=store)
    rerun = consumer.compute.rerun(store=store)
    assert rerun.realization_id != first.realization_id


def test_resumable_interrupt_preserves_checkpoint_and_normal_call_resumes(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    once = {"requested": False}

    def interrupt(event):
        if event.kind == "progress" and not once["requested"]:
            once["requested"] = True
            return ControlRequest.INTERRUPT
        return None

    callback = ManagedCallback(interrupt, controls={ControlRequest.INTERRUPT})
    with pytest.raises(ManagedInterruptedError):
        operation(store=store, callbacks=(callback,))

    interrupted = operation.status(store=store)
    assert interrupted.status == "interrupted"
    assert interrupted.checkpoint_head.startswith("checkpoint-v1-")
    resumed = operation(store=store)
    assert resumed.action == "resume"
    assert len(operation.history(store=store)[0].attempt_ids) == 2


def test_non_resumable_interrupt_requires_explicit_rerun(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = NonResumable().compute
    callback = ManagedCallback(
        lambda event: ControlRequest.INTERRUPT,
        controls={ControlRequest.INTERRUPT},
        fail_soft=True,
    )
    with pytest.raises(ManagedInterruptedError):
        operation(store=store, callbacks=(callback,))
    with pytest.raises(ManagedRerunRequiredError):
        operation(store=store)


def test_graceful_stop_requires_declared_capability_and_valid_early_result(tmp_path):
    callback = ManagedCallback(
        lambda event: ControlRequest.GRACEFUL_STOP,
        controls={ControlRequest.GRACEFUL_STOP},
        fail_soft=True,
    )
    store = DirStore(tmp_path / "allowed")
    result = FakeOperation().compute(store=store, callbacks=(callback,))
    assert result.early_completed

    unsupported = DirStore(tmp_path / "unsupported")
    operation = FakeOperation(early_completion=False).compute
    with pytest.raises(ManagedCapabilityError, match="early completion"):
        operation(store=unsupported, callbacks=(callback,))
    assert not Path(unsupported.managed_control_root()).exists()


def test_strict_callback_failure_checkpoints_and_is_resumable(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    callback = ManagedCallback(lambda event: (_ for _ in ()).throw(RuntimeError("observer")))

    with pytest.raises(Exception, match="RuntimeError"):
        operation(store=store, callbacks=(callback,))
    status = operation.status(store=store)
    assert status.status == "failed"
    assert status.checkpoint_head.startswith("checkpoint-v1-")
    assert operation(store=store).action == "resume"


def test_strict_completion_callback_fails_before_activation_with_checkpoint(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute

    def fail_on_completion(event):
        if event.kind == "completed":
            raise RuntimeError("completion observer")

    with pytest.raises(Exception, match="RuntimeError"):
        operation(store=store, callbacks=(ManagedCallback(fail_on_completion),))

    status = operation.status(store=store)
    assert status.status == "failed"
    assert status.active_realization_id is None
    assert status.checkpoint_head.startswith("checkpoint-v1-")
    assert operation(store=store).action == "resume"


def test_strict_callback_checkpoint_failure_is_not_resumable(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    callback = ManagedCallback(lambda event: (_ for _ in ()).throw(RuntimeError("observer")))

    with pytest.raises(OSError, match="checkpoint production failed"):
        operation(store=store, callbacks=(callback,), fail="checkpoint")
    assert operation.status(store=store).status == "failed"
    assert operation.status(store=store).checkpoint_head is None
    with pytest.raises(ManagedRerunRequiredError):
        operation(store=store)


def test_checkpoint_requests_coalesce_to_one_commit_per_safe_point(tmp_path):
    store = DirStore(tmp_path / "store")
    callback = ManagedCallback(
        lambda event: ControlRequest.CHECKPOINT,
        controls={ControlRequest.CHECKPOINT},
    )

    result = FakeOperation().compute(store=store, callbacks=(callback, callback, callback))
    checkpoints = tuple(
        (Path(store.managed_control_root()) / "operations").glob(
            "**/attempts/*/checkpoints/checkpoint-v1-*"
        )
    )

    assert result.action == "start"
    assert len(checkpoints) == 1


def test_strict_callback_on_non_resumable_operation_rejects_before_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    callback = ManagedCallback(lambda event: None)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        NonResumable().compute(store=store, callbacks=(callback,))
    assert not Path(store.managed_control_root()).exists()


def test_whole_pipeline_capability_downgrade_rejects_strict_guarantee_before_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    callback = ManagedCallback(lambda event: None)

    with pytest.raises(ManagedCapabilityError, match="strict callback"):
        FakeOperation(resumable=False).compute(store=store, callbacks=(callback,))
    assert not Path(store.managed_control_root()).exists()


def test_missing_logical_input_fails_preflight_without_consumer_mutation(tmp_path):
    store = DirStore(tmp_path / "store")
    consumer = Consumer(FakeOperation().compute.result)

    with pytest.raises(Exception, match="active"):
        consumer.compute(store=store)
    assert not Path(store.managed_control_root()).exists()


def test_fail_soft_observer_continues_and_records_bounded_diagnostics(tmp_path):
    store = DirStore(tmp_path / "store")
    secret = "managed-callback-secret-sentinel-a97d"
    callback = ManagedCallback(
        lambda event: (_ for _ in ()).throw(RuntimeError(secret)),
        fail_soft=True,
    )
    operation = FakeOperation().compute

    result = operation(store=store, callbacks=(callback,))

    assert result.action == "start"
    assert len(result.diagnostics) <= 32
    assert result.diagnostics
    assert all(item == f"callback RuntimeError: {secret}" for item in result.diagnostics)
    durable_data = (
        *store.records.iter_records(),
        *store.records.iter_specs(),
        *(item.to_json() for item in operation.history(store=store)),
    )
    assert secret not in str(durable_data)
    assert all(
        secret.encode() not in path.read_bytes()
        for path in Path(store.managed_control_root()).rglob("*")
        if path.is_file()
    )


@pytest.mark.parametrize("point,resumable", [("before", False), ("after", True)])
def test_operation_failure_is_resumable_only_after_compatible_checkpoint(tmp_path, point, resumable):
    store = DirStore(tmp_path / point)
    operation = FakeOperation().compute

    with pytest.raises(RuntimeError, match=f"failed {point}"):
        operation(store=store, fail=point)
    if resumable:
        assert operation(store=store).action == "resume"
    else:
        with pytest.raises(ManagedRerunRequiredError):
            operation(store=store)


def test_operation_failure_keeps_exception_message_transient(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    secret = "managed-local-secret-sentinel-73b4"

    with pytest.raises(RuntimeError, match=secret):
        operation(store=store, fail=secret)

    failed = operation.history(store=store)[0]
    assert failed.diagnostics == ("RuntimeError: execution_failed",)
    assert secret not in str(failed.to_json())


@pytest.mark.parametrize(
    "point,resumable",
    [("interrupt-before", False), ("interrupt-after", True)],
)
def test_keyboard_interrupt_is_resumable_only_after_compatible_checkpoint(
    tmp_path, point, resumable
):
    store = DirStore(tmp_path / point)
    operation = FakeOperation().compute
    terminal_secret = "managed-terminal-secret-sentinel-1ed9"

    def interrupt_terminal_publication(event):
        if event.kind == "interrupted":
            raise SystemExit(terminal_secret)

    callback = ManagedCallback(interrupt_terminal_publication)
    with pytest.raises(
        KeyboardInterrupt, match="managed-interrupt-secret-sentinel-41c8"
    ):
        operation(store=store, callbacks=(callback,), fail=point)

    interrupted = operation.history(store=store)[0]
    assert interrupted.status == "interrupted"
    assert interrupted.resumable is resumable
    assert interrupted.diagnostics == ("operation interrupted by process control",)
    assert "managed-interrupt-secret-sentinel-41c8" not in str(interrupted.to_json())
    assert terminal_secret not in str(interrupted.to_json())
    if resumable:
        assert interrupted.checkpoint_head is not None
        assert operation(store=store).action == "resume"
    else:
        assert interrupted.checkpoint_head is None
        with pytest.raises(ManagedRerunRequiredError):
            operation(store=store)


def test_missing_required_output_fails_without_activation(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = MissingOutput().compute

    with pytest.raises(Exception, match="required output"):
        operation(store=store)
    assert operation.status(store=store).status == "failed"
    assert operation.results(store=store) == {}


def test_failed_rerun_keeps_old_active_readable(tmp_path):
    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    first = operation(store=store, value=b"old")

    with pytest.raises(RuntimeError):
        operation.rerun(store=store, fail="before")

    assert operation.results(store=store)["result"].record_id == first.outputs["result"].record_id
    assert _read_result(store, first) == b"old"


def test_activation_event_failure_keeps_completed_rerun_inactive(
    tmp_path, monkeypatch
):
    import dryml.managed.store as store_module

    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    first = operation(store=store, value=b"old")
    original_write_json = store_module._write_json
    injected = False

    def fail_event_publication(path, data, *, immutable=False):
        nonlocal injected
        if Path(path).parent.name == "activations" and not injected:
            injected = True
            raise ManagedStateError("simulated activation event publication failure")
        return original_write_json(path, data, immutable=immutable)

    monkeypatch.setattr(store_module, "_write_json", fail_event_publication)
    with pytest.raises(ManagedStateError, match="event publication failure"):
        operation.rerun(store=store, value=b"new")

    assert (
        operation.results(store=store)["result"].record_id
        == first.outputs["result"].record_id
    )
    assert [item.status for item in operation.history(store=store)] == [
        "completed",
        "completed",
    ]


def test_post_commit_activation_event_read_failure_returns_committed_rerun(
    tmp_path, monkeypatch
):
    import dryml.managed.store as store_module

    store = DirStore(tmp_path / "store")
    operation = FakeOperation().compute
    operation(store=store, value=b"old")
    original_write_json = store_module._write_json
    original_read_json = store_module._read_json
    published_event = None
    injected = False

    def track_event_publication(path, data, *, immutable=False):
        nonlocal published_event
        result = original_write_json(path, data, immutable=immutable)
        if Path(path).parent.name == "activations":
            published_event = Path(path)
        return result

    def fail_first_event_read(path, name):
        nonlocal injected
        if published_event == Path(path) and not injected:
            injected = True
            raise OSError("simulated post-commit activation event read failure")
        return original_read_json(path, name)

    monkeypatch.setattr(store_module, "_write_json", track_event_publication)
    monkeypatch.setattr(store_module, "_read_json", fail_first_event_read)
    result = operation.rerun(store=store, value=b"new")

    assert injected
    assert (
        operation.results(store=store)["result"].record_id
        == result.outputs["result"].record_id
    )
