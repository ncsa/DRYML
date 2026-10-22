"""End-to-end core Execute coverage for managed-operation transport."""

from __future__ import annotations

import functools
from abc import abstractmethod
from pathlib import Path

import pytest

from dryml import F
from dryml.artifacts import CachedDataset
from dryml.core import Definition, Executor as CoreExecutor
from dryml.core import Mat, Object, Repo, StateRef, function, signatures
from dryml.core.execute import CoreExecutionError, CoreOptions
from dryml.core.execute_codec import CoreCallCodecError, decode_outcome, encode_invocation, invoke_invocation
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.core.store.store import Store
from dryml.core.utils.graph.path import GraphPath, Parameter
from dryml.data import ArrayDataset, as_supervised
from dryml.execute.subprocess import SubProcessConfig
from dryml.managed import ManagedConfig, managed_operation
from dryml.managed import InvocationResult
from tests.core.u19_execution_fixtures import WorkerValue
from dryml.models import Experiment, TrainFunction
from dryml.models.experiment import _notify_host_observers
from tests.managed.execution_fixtures import (
    InnerFunctionWrappedManagedValue,
    ManagedMatrixValue,
    MatrixArgument,
    ORDERS,
    OuterWrappedManagedValue,
    matrix_config,
    nested_managed_call,
    nested_managed_advance,
    nested_unrelated_function,
)


def _add_wrapper(target):
    """Retain a visible ordinary wrapper around one managed declaration."""

    @functools.wraps(target)
    def wrapped(self, value, *, managed):
        self.events.append("before")
        try:
            result = target(self, value, managed=managed)
            return result if isinstance(result, InvocationResult) else result + 10
        finally:
            self.events.append("after")

    return wrapped


def _default_wrapper(target):
    """Retain a managed target in a keyword-only default for transport coverage."""

    @functools.wraps(target)
    def wrapped(self, value, *, managed, _target=target):
        result = _target(self, value, managed=managed)
        return result if isinstance(result, InvocationResult) else result + 10

    return wrapped


class FrozenWrapperValue(Pickleable):
    """Managed receiver whose accepted outer wrapper must remain transport-stable."""

    def __init__(self, value=0):
        """Initialize deterministic state and wrapper observations."""

        self.value = value
        self.events = []

    @_add_wrapper
    @managed_operation()
    def advance(self, value, *, managed):
        """Mutate once through the inner managed lifecycle."""

        self.events.append("body")
        self.value += value
        return self.value


class DefaultCapturedWrapperValue(Pickleable):
    """Fixture whose outer wrapper retains its managed target in a default."""

    def __init__(self):
        """Initialize the state changed by the retained managed body."""

        self.value = 0

    @_default_wrapper
    @managed_operation()
    def advance(self, value, *, managed):
        """Mutate once before the default-captured wrapper adjusts the result."""

        self.value += value
        return self.value


class DirectConfigValue(Pickleable):
    """Non-composite managed receiver used to prove direct config transport."""

    def __init__(self) -> None:
        """Initialize state and callback observations for each fresh attempt."""

        self.value = 0
        self.checkpoint_callbacks = 0

    @managed_operation()
    def advance(self, *, managed) -> int:
        """Advance once and create the callback-visible checkpoint."""

        self.value += 1
        managed.checkpoint()
        return self.value


class StoreControlValue(Pickleable):
    """Worker fixture that requires a declaration-owned Store control."""

    def __init__(self) -> None:
        """Initialize the small state published through the selected Store."""

        self.value = 0

    @managed_operation(store_parameter="store")
    def advance(self, *, store: Store | None = None, managed) -> None:
        """Require the decoded Store control before mutating worker-local state."""

        if store is None:
            raise AssertionError("worker lost declared Store control")
        self.value += 1


class WorkerTelemetryModel(Pickleable):
    """Small worker-owned state proving telemetry never becomes model state."""

    def __init__(self) -> None:
        """Initialize the one accepted-update counter."""

        self.value = 0


class WorkerTelemetryTrainer(TrainFunction):
    """Emit one worker host event after accepted accounting and safe points."""

    supports_observers = True

    def __call__(self, exp, *, callbacks=(), observer_session=None):
        """Apply one update and report its retained position."""

        exp.model.value += 1
        exp.state.record_update(examples=1, loss=1.0)
        for callback in callbacks:
            callback()
        _notify_host_observers(observer_session, {
            "event": "train_batch_end",
            "step": exp.state.step,
        })


class WorkerFileObserver:
    """Reconstructible observer recording construction, delivery, and cleanup."""

    def __init__(self, path: str) -> None:
        """Record worker-side construction at the admitted invocation boundary."""

        self.path = path
        Path(path).write_text("constructed\n", encoding="ascii")

    def __call__(self, logs) -> None:
        """Append one host event without retaining the external file in DRYML state."""

        with Path(self.path).open("a", encoding="ascii") as stream:
            stream.write(f"{logs['event']}:{logs['step']}\n")

    def close(self) -> None:
        """Record bounded invocation cleanup."""

        with Path(self.path).open("a", encoding="ascii") as stream:
            stream.write("closed\n")


def _count_checkpoint_callback(value, context) -> None:
    """Record callback delivery on the checkpointed managed receiver."""

    del context
    value.checkpoint_callbacks += 1


def test_u19_subprocess_completed_target_replays_after_receiver_cleanup(tmp_path):
    """Directly transported completion admits before any receiver hook or IO."""
    from dryml.managed.control import ManagedControlStore
    from dryml.managed.results import result_path
    from dryml.managed.errors import ManagedResultError
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    marker = tmp_path / "events"
    subject = WorkerValue(str(marker), repo=repo)
    old, report = repo.save(subject, report_stores=True)
    first = subject.advance(managed=ManagedConfig(state_repo=repo))
    control = ManagedControlStore(repo.stores[0], repo)
    control.cleanup_checkpoints(first.execution.operation_id)
    for publication in report.publications:
        if publication.retention_token is not None:
            repo.release(publication.retention_token, owner_store=repo.stores[0])
    assert not Path(repo.stores[0].state_data_path(old.state.key)).exists()
    assert not Path(repo.stores[0].state_data_path(first.final_state_ref.state.key)).exists()
    before = marker.read_text()
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False))
    try:
        future = executor.submit(subject.advance)
        replay = future.result(timeout=30)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)
    repo.close(flush=False)
    assert type(replay) is InvocationResult
    assert replay.status.state == "completed"
    assert replay.execution.run_id == first.execution.run_id
    assert replay.execution.producer_invocation_id == first.execution.invocation_id
    assert replay.execution.invocation_id != first.execution.invocation_id
    assert replay.final_state_ref.graph_equal(first.final_state_ref)
    assert replay.value == {"value": 3, "resuming": False}
    Path(result_path(repo.stores[0], replay.result_reference.key)).unlink()
    with pytest.raises(ManagedResultError, match="result_unavailable"):
        replay.value
    assert marker.read_text() == before


def test_u19_transport_rerun_start_and_resume_restore_once_under_owner(tmp_path):
    """Supplied saved start seeds recovery and only resume restores latest."""
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    marker = tmp_path / "events"
    subject = WorkerValue(str(marker), repo=repo)
    start = repo.save(subject)
    subject.value = 90
    Path(str(marker) + ".fail").touch()
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False))
    try:
        before = marker.read_text().splitlines()
        failed = executor.submit(subject.advance, kwargs={"managed": ManagedConfig(rerun=True, start_from=start)})
        with pytest.raises(CoreExecutionError):
            failed.result(timeout=30)
        failed.cleanup(timeout=5)
        status = subject.advance.status(state_repo=repo)
        assert status.state == "failed"
        delta = marker.read_text().splitlines()[len(before):]
        assert delta.count("restore") == delta.count("construct") == delta.count("body") == delta.count("save") == 1
        before = marker.read_text().splitlines()
        resumed = executor.submit(subject.advance)
        result = resumed.result(timeout=30)
        resumed.cleanup(timeout=5)
        assert result.value == {"value": 6, "resuming": True}
        assert result.execution.run_id == status.run_id
        assert result.execution.invocation_id != status.invocation_id
        delta = marker.read_text().splitlines()[len(before):]
        assert delta.count("restore") == delta.count("construct") == delta.count("body") == delta.count("save") == 1
    finally:
        executor.close(cancel=True, timeout=10)


def test_u19_decoded_target_and_copied_worker_context_expire(tmp_path):
    """Detached receiver metadata cannot extend a closed or inherited owner scope."""
    import contextvars
    from dryml.core.execute import ExecutionContext, worker_context
    from dryml.core.execute_codec import decode_invocation
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    marker = tmp_path / "events"
    subject = WorkerValue(str(marker), repo=repo)
    repo.save(subject)
    wire = encode_invocation(subject.advance, (), {}, repo=repo)
    target, *_ = decode_invocation(wire, repo=repo)
    with worker_context(ExecutionContext(repo, None)):
        copied = contextvars.copy_context()
    before = marker.read_text()
    with pytest.raises(RuntimeError, match="outside an active worker"):
        copied.run(target)
    repo.close(flush=False)
    with pytest.raises(RuntimeError, match="expired"):
        target()
    assert marker.read_text() == before


def test_u19_execution_realization_retains_the_admitting_owner(tmp_path, monkeypatch):
    """The constructor handoff cannot release/reacquire the operation lifetime lock."""
    from dryml.managed.control import ManagedControlStore
    from dryml.managed.identity import AdmissionIdentity
    from dryml.managed.errors import ManagedConflictError
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = WorkerValue(str(tmp_path / "events"), repo=repo)
    repo.save(subject)
    identity = AdmissionIdentity.from_call(repo.describe(subject), subject.advance._descriptor, (), {})
    control = ManagedControlStore(repo.stores[0], repo)
    original = repo.materialize_boundary
    observed = []
    def realize(roots, **kwargs):
        with pytest.raises(ManagedConflictError, match="already_running"):
            with control.admission(identity):
                pytest.fail("execution owner was released")
        observed.append(True)
        return original(roots, **kwargs)
    monkeypatch.setattr(repo, "materialize_boundary", realize)
    outcome = decode_outcome(invoke_invocation(encode_invocation(subject.advance, (), {}, repo=repo), repo=repo), repo=repo)
    assert outcome["success"] and outcome["result"].value["value"] == 3
    assert observed == [True]


def test_u19_active_owner_and_start_without_rerun_do_not_materialize_worker(tmp_path):
    """Busy admission and invalid start controls refuse before receiver construction."""
    from dryml.managed.control import ManagedControlStore
    from dryml.managed.identity import AdmissionIdentity
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    marker = tmp_path / "events"
    subject = WorkerValue(str(marker), repo=repo)
    saved = repo.save(subject)
    identity = AdmissionIdentity.from_call(saved, subject.advance._descriptor, (), {})
    control = ManagedControlStore(repo.stores[0], repo)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False))
    before = marker.read_text()
    try:
        with control.admission(identity):
            busy = executor.submit(subject.advance)
            with pytest.raises(CoreExecutionError, match="ManagedConflictError"):
                busy.result(timeout=30)
            busy.cleanup(timeout=5)
        invalid = executor.submit(subject.advance, kwargs={"managed": ManagedConfig(start_from=saved)})
        with pytest.raises(CoreExecutionError, match="ManagedConfigError"):
            invalid.result(timeout=30)
        invalid.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)
    assert marker.read_text() == before


def test_u19_two_worker_branches_preserve_shared_dataset_and_independent_models(tmp_path):
    """Separate operation/run owners retain shared input IDs and exact alias edges."""
    from tests.core.u19_execution_fixtures import Branch
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    data = WorkerValue(str(tmp_path / "data-events"), repo=repo)
    first_model = WorkerValue(str(tmp_path / "first-events"), repo=repo)
    second_model = WorkerValue(str(tmp_path / "second-events"), repo=repo)
    first, second = Branch(data, data, first_model, repo=repo), Branch(data, data, second_model, repo=repo)
    repo.save(first)
    repo.save(second)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False))
    try:
        a, b = executor.submit(first.run), executor.submit(second.run)
        left, right = a.result(timeout=30), b.result(timeout=30)
        a.cleanup(timeout=5)
        b.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)
    assert left.value == (str(data.object_id), str(first_model.object_id), 1)
    assert right.value == (str(data.object_id), str(second_model.object_id), 1)
    assert left.execution.operation_id != right.execution.operation_id
    assert left.execution.run_id != right.execution.run_id
    for result in (left, right):
        live = repo.load_state_ref(result.final_state_ref, reuse_live="never")
        assert live.data is live.repeated and live.data.object_id == data.object_id
        assert live.model.value == 1


def test_u19_inherited_core_worker_resources_refuse_a_forked_process(tmp_path):
    """An inherited ContextVar cannot authorize resource use in a new process."""
    import os
    from dryml.core.execute import ExecutionContext, current_context, worker_context
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    with worker_context(ExecutionContext(repo, None)):
        pid = os.fork()
        if pid == 0:
            try:
                current_context()
            except RuntimeError as error:
                os._exit(0 if "different process" in str(error) else 2)
            os._exit(1)
        _, status = os.waitpid(pid, 0)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
        assert current_context().repo is repo


def test_u19_source_cut_retry_is_metadata_only_and_does_not_repeat_hooks(tmp_path, monkeypatch):
    """A changed detached read cut retries before construction/restore delivery."""
    from contextlib import contextmanager
    from dryml.core.store.authority import StoreCutChanged
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    marker = tmp_path / "events"
    subject = WorkerValue(str(marker), repo=repo)
    saved = repo.save(subject)
    store = repo.stores[0]
    original = store.read_states
    calls = []
    @contextmanager
    def changed_once(keys):
        calls.append(True)
        if len(calls) == 1:
            raise StoreCutChanged("controlled unrelated publication")
        with original(keys) as sources:
            yield sources
    before = marker.read_text().splitlines()
    monkeypatch.setattr(store, "read_states", changed_once)
    restored = repo.load_state_ref(saved, reuse_live="never")
    assert restored.value == 0
    assert len(calls) == 2
    assert marker.read_text().splitlines()[len(before):] == ["construct", "restore"]


def test_u19_worker_recovery_transports_exact_current_witness_without_execution(tmp_path, monkeypatch):
    """Abandoned rerun recovery terminates and preserves original current CAS bytes."""
    from dryml.managed.control import ManagedControlStore
    from dryml.managed.runtime import AdmissionOutcome
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    marker = tmp_path / "events"
    subject = WorkerValue(str(marker), repo=repo)
    first = subject.advance(managed=ManagedConfig(state_repo=repo))
    control = ManagedControlStore(repo.stores[0], repo)
    original = control.inspect(first.execution.operation_id).to_bytes()
    wire = encode_invocation(subject.advance, (), {"managed": ManagedConfig(rerun=True, start_from=first.final_state_ref)}, repo=repo)
    with monkeypatch.context() as patch:
        patch.setattr(repo, "materialize_boundary", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("preparation failed")))
        failed = decode_outcome(invoke_invocation(wire, repo=repo), repo=repo)
    assert not failed["success"]
    assert not control.read_preparation(first.execution.operation_id).settled
    before = marker.read_text()
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(SubProcessConfig(spool_directory=spool), core=CoreOptions(repo=repo, return_objects=False))
    try:
        future = executor.submit(subject.advance)
        result = future.result(timeout=30)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)
    assert type(result) is AdmissionOutcome
    assert result.disposition == "preparation_abandoned"
    assert result.current.to_bytes() == original
    assert result.run_id == first.execution.run_id
    assert result.invocation_id != first.execution.invocation_id
    assert marker.read_text() == before


class InheritedDirectConfigValue(DirectConfigValue):
    """Child fixture retaining its base class's normal inherited declaration."""


class WorkerDependent(Object):
    """Record construction if an abstract worker graph reaches its dependency."""

    def __init__(self, marker: str) -> None:
        """Write the marker only when this dependency is actually materialized."""

        Path(marker).write_text("constructed", encoding="ascii")


class AbstractWorkerTarget(Object):
    """Abstract target used to prove worker-side preflight rejects its graph."""

    def __init__(self, dependency) -> None:
        """Retain the dependency that must stay unmaterialized on rejection."""

        self.dependency = dependency

    @abstractmethod
    def required(self) -> None:
        """Declare the unresolved ordinary abstract obligation."""


@function
def _materialize_worker_target(value: Mat[Object]) -> str:
    """Force worker-side Mat delivery only after Execute has accepted the call."""

    return type(value).__name__


@function
def _train_materialized_experiment(value: Mat[Object]) -> StateRef:
    """Train one materialized Experiment and return its managed terminal state."""

    return value.train()


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_abstract_worker_materialization_keeps_remote_type_evidence_and_dependencies_inert(tmp_path):
    """A worker rejects abstract Mat delivery before any dependent Object is built."""

    marker = tmp_path / "dependent-built"
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    target = Definition(
        AbstractWorkerTarget,
        Definition(WorkerDependent, str(marker)),
    ).concretize(repo=repo)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_materialize_worker_target, target)
        with pytest.raises(CoreExecutionError) as raised:
            future.result(timeout=15)
        assert type(raised.value) is CoreExecutionError
        assert "TypeError" in str(raised.value)
        assert not marker.exists()
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)


def test_accepted_composite_uses_captured_wrapper_after_class_mutation(tmp_path):
    """A caller mutation after encoding cannot replace the accepted wrapper graph."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    value = FrozenWrapperValue(repo=repo)
    repo.save_object(value, deep_capture=True)
    invocation = encode_invocation(
        value.advance, (2,), {"managed": ManagedConfig(state_repo=repo)}, repo=repo,
    )

    def replacement(self, value, *, managed):
        """Fail if decoder performs a late receiver-member lookup."""

        raise AssertionError("accepted wrapper was bypassed")

    original = FrozenWrapperValue.__dict__["advance"]
    type.__setattr__(FrozenWrapperValue, "advance", replacement)
    try:
        outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    finally:
        type.__setattr__(FrozenWrapperValue, "advance", original)

    assert outcome["success"]
    assert outcome["result"].value == 2


@pytest.mark.parametrize(
    ("subject_type", "capture"),
    ((FrozenWrapperValue, "closure"), (DefaultCapturedWrapperValue, "default")),
)
def test_accepted_composite_snapshots_closure_and_default_wrapper_captures(
        tmp_path, subject_type, capture):
    """Later wrapper-capture mutation cannot alter a previously encoded call."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    value = subject_type(repo=repo)
    repo.save_object(value, deep_capture=True)
    invocation = encode_invocation(
        value.advance, (2,), {"managed": ManagedConfig(state_repo=repo)}, repo=repo,
    )
    outer = subject_type.__dict__["advance"]._outer
    target = outer.__wrapped__

    def replacement(self, value, *, managed):
        """Fail if the decoder reads a mutable source wrapper after acceptance."""

        raise AssertionError("accepted wrapper capture was bypassed")

    if capture == "closure":
        cell = next(cell for cell in outer.__closure__ if cell.cell_contents is target)
        cell.cell_contents = replacement
        restore = lambda: setattr(cell, "cell_contents", target)
    else:
        defaults = outer.__kwdefaults__
        assert defaults is not None
        defaults["_target"] = replacement
        restore = lambda: defaults.__setitem__("_target", target)
    try:
        outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)
    finally:
        restore()

    assert outcome["success"]
    assert outcome["result"].value == 2


@pytest.mark.parametrize(
    ("member", "written_order"), ORDERS,
)
@pytest.mark.parametrize("placement", ("local", "subprocess"))
def test_managed_decorator_orders_materialize_once_and_publish_one_final_state(
        tmp_path, member, written_order, placement,
        monkeypatch: pytest.MonkeyPatch,
        request: pytest.FixtureRequest):
    """Each order has one Mat boundary; subprocess rows retain host observation."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((state_store, control_store))
    subject = ManagedMatrixValue(repo=repo)
    argument = MatrixArgument(3, repo=repo)
    if placement == "local":
        request.getfixturevalue("fixed_managed_snapshot_environment")
    argument_state = repo.save_object(argument, deep_capture=True)
    operation = getattr(subject, member)
    from dryml.annotations import annotations_for_method

    annotations = annotations_for_method(ManagedMatrixValue, member)
    assert ("test.integration", "present") in [
        (annotation.key, annotation.value) for annotation in annotations
    ]
    if placement == "local":
        observed = []
        original_args = signatures.SignaturePlan._prepare_bound
        original_return = signatures.SignaturePlan._prepare_return

        def observe_args(plan, arguments, *, partial, controls):
            observed.append("args")
            return original_args(
                plan, arguments, partial=partial, controls=controls,
            )

        def observe_return(plan, value, controls, **kwargs):
            observed.append("return")
            return original_return(plan, value, controls, **kwargs)

        monkeypatch.setattr(
            signatures.SignaturePlan, "_prepare_bound", observe_args,
        )
        monkeypatch.setattr(
            signatures.SignaturePlan, "_prepare_return", observe_return,
        )
        result = operation(
            argument_state, managed=matrix_config(repo, control_store),
        )
        assert result.value.graph_equal(argument_state)
        assert (subject.calls, subject.value) == (1, 3)
        assert subject.save_calls == 1
        assert observed == ["args", "return"]
    else:
        repo.save_object(subject, deep_capture=True)
        spool = tmp_path / "spool"
        spool.mkdir()
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(operation, argument_state)
            assert future.result(timeout=15).value.graph_equal(argument_state)
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)
        restored = repo.load_state_ref(
            operation.status(
                state_repo=repo, control_store=control_store,
            ).final,
        )
        assert (restored.calls, restored.value, restored.save_calls) == (1, 3, 2)
    assert operation.status(
        state_repo=repo, control_store=control_store,
    ).state == "completed"


@pytest.mark.parametrize(
    ("subject_type", "persisted_events"),
    (
        (FrozenWrapperValue, ["before", "body"]),
        (InnerFunctionWrappedManagedValue, ["before", "body", "after"]),
    ),
)
def test_ordinary_wrappers_run_in_authored_order_on_first_subprocess_call(
        tmp_path, subject_type, persisted_events):
    """W(M(method)) and M(W(F(method))) retain wrapper bodies before local use."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = subject_type(repo=repo)
    repo.save_object(subject, deep_capture=True)
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(subject.advance, 2)
        assert future.result(timeout=15).value == (2 if subject_type is FrozenWrapperValue else 12)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)
    restored = repo.load_state_ref(
        subject.advance.status(state_repo=repo).final, reuse_live="never",
    )
    assert restored.events == persisted_events
    assert restored.value == 2


@pytest.mark.parametrize("placement", ("codec", "subprocess"))
@pytest.mark.parametrize("subject_type", (DirectConfigValue, InheritedDirectConfigValue))
def test_direct_bound_operation_transports_explicit_managed_config(
        tmp_path, placement: str, subject_type: type[DirectConfigValue]) -> None:
    """Direct bound worker calls retain explicit authority, callbacks, and rerun."""

    state_store = DirStore(tmp_path / "state", query_index="none")
    control_store = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((state_store, control_store))
    subject = subject_type(repo=repo)
    repo.save_object(subject, deep_capture=True)
    initial = ManagedConfig(
        state_repo=repo, control_store=control_store,
        callbacks=[_count_checkpoint_callback],
    )
    assert subject.advance(managed=initial).value == 1
    rerun = ManagedConfig(
        state_repo=repo, control_store=control_store, rerun=True,
        callbacks=[_count_checkpoint_callback],
    )

    if placement == "codec":
        outcome = decode_outcome(
            invoke_invocation(
                encode_invocation(subject.advance, (), {"managed": rerun}, repo=repo),
                repo=repo, raw_traceback=True,
            ),
            repo=repo, allow_raw_traceback=True,
        )
        assert outcome["success"], outcome["traceback"]
        assert outcome["result"].value == 2
    else:
        spool = tmp_path / "spool"
        spool.mkdir()
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(subject.advance, kwargs={"managed": rerun})
            assert future.result(timeout=15).value == 2
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)

    status = subject.advance.status(state_repo=repo, control_store=control_store)
    restored = repo.load_state_ref(status.final, reuse_live="never")
    assert status.state == "completed"
    assert (restored.value, restored.checkpoint_callbacks) == (2, 2)


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_direct_bound_operation_rejects_malformed_config_before_mutation(tmp_path) -> None:
    """Invalid explicit policy is rejected during capture before a worker can mutate."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = DirectConfigValue(repo=repo)
    repo.save_object(subject, deep_capture=True)
    malformed = ManagedConfig(state_repo=repo)
    object.__setattr__(malformed, "rerun", 1)

    with pytest.raises(CoreCallCodecError, match="malformed managed config"):
        encode_invocation(subject.advance, (), {"managed": malformed}, repo=repo)

    assert subject.value == 0
    assert subject.advance.status(state_repo=repo).state == "not_started"


def test_execute_builds_experiment_telemetry_factory_in_worker(
        tmp_path, fixed_managed_snapshot_environment) -> None:
    """Execute transports inert observer configuration, not a live client."""

    del fixed_managed_snapshot_environment
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control = DirStore(tmp_path / "control", query_index="none")
    exp = Experiment(
        WorkerTelemetryModel(), WorkerTelemetryTrainer(), repo=repo,
    )
    repo.save_object(exp, deep_capture=True)
    marker = tmp_path / "telemetry.log"
    invocation = encode_invocation(
        exp.train,
        (),
        {
            "callbacks": [F(WorkerFileObserver, str(marker))],
            "managed": ManagedConfig(
                state_repo=repo, control_store=control,
            ),
        },
        repo=repo,
    )
    assert not marker.exists()
    outcome = decode_outcome(invoke_invocation(invocation, repo=repo), repo=repo)

    assert outcome["success"]
    result = outcome["result"]
    assert isinstance(result, StateRef)
    assert marker.read_text(encoding="ascii").splitlines() == [
        "constructed", "train_batch_end:1", "closed",
    ]
    restored = repo.load_state_ref(result, reuse_live="never")
    assert (restored.model.value, restored.state.step) == (1, 1)


def test_execute_saves_experiment_with_nested_cached_test_data(tmp_path) -> None:
    """Worker terminal capture retains a materialized test Dataset's saved source."""

    import numpy as np

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    cached = CachedDataset(ArrayDataset({
        "cart": np.asarray([[1.0, 2.0, 3.0]]),
    }, repo=repo))
    source = cached.compute(
        codec="numpy", managed=ManagedConfig(state_repo=repo),
    )
    experiment = Definition(
        Experiment,
        Definition(WorkerTelemetryModel),
        Definition(WorkerTelemetryTrainer),
        test_data=as_supervised(source, "cart", input_as_target=True),
    )
    spool = tmp_path / "spool"
    spool.mkdir()
    executor = CoreExecutor(
        SubProcessConfig(spool_directory=spool),
        core=CoreOptions(repo=repo, return_objects=False),
    )
    try:
        future = executor.submit(_train_materialized_experiment, experiment)
        final = future.result(timeout=30)
        future.cleanup(timeout=5)
    finally:
        executor.close(cancel=True, timeout=10)

    assert final.at(GraphPath((
        Parameter("test_data"), Parameter("src"),
    ))) == source


def test_execute_rejects_live_experiment_telemetry_before_training(tmp_path) -> None:
    """A live local observer cannot cross Execute's worker transport boundary."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    exp = Experiment(
        WorkerTelemetryModel(), WorkerTelemetryTrainer(), repo=repo,
    )
    repo.save_object(exp, deep_capture=True)

    with pytest.raises(CoreCallCodecError, match="live telemetry observer"):
        encode_invocation(
            exp.train,
            (),
            {"callbacks": [lambda logs: None]},
            repo=repo,
        )
    with pytest.raises(CoreCallCodecError, match="live observer factory configuration"):
        encode_invocation(
            exp.train,
            (),
            {"callbacks": [F(WorkerFileObserver, object())]},
            repo=repo,
        )

    assert (exp.model.value, exp.state.step, exp.state.phase) == (0, 0, None)


def test_managed_store_control_uses_definition_transport_and_returns_state_ref(tmp_path) -> None:
    """A worker rebuilds only the declared Store control and returns its final ref."""

    source = DirStore(tmp_path / "source", query_index="none")
    override = DirStore(tmp_path / "override", query_index="none")
    control = DirStore(tmp_path / "control", query_index="none")
    repo = Repo((source,))
    subject = StoreControlValue(repo=repo)
    repo.save_object(subject, deep_capture=True)

    outcome = decode_outcome(
        invoke_invocation(
            encode_invocation(
                subject.advance, (), {
                    "store": override,
                    "managed": ManagedConfig(
                        state_repo=repo, control_store=control,
                    ),
                }, repo=repo,
            ),
            repo=repo, raw_traceback=True,
        ),
        repo=repo, allow_raw_traceback=True,
    )

    assert outcome["success"], outcome["traceback"]
    from dryml.core.node_prepare import NodePreparation
    result = outcome["result"]
    assert type(result) is InvocationResult
    final = result.final_state_ref
    digest = NodePreparation((final,)).digest(final, domain="dryml-saved-root/v1")
    assert override.read_node(digest, kind="saved").graph_equal(final)


@pytest.mark.usefixtures("fixed_managed_snapshot_environment")
def test_unrelated_nested_function_keeps_its_own_normalization_boundary(
        tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A managed handoff never suppresses a separate nested function boundary."""

    class NestedFunctionValue(Pickleable):
        """Exercise an independent function inside an M(F(method)) handoff."""

        @managed_operation()
        @function
        def run(self, *, managed) -> int:
            """Normalize the nested call while the owning lifecycle is active."""

            assert managed.active
            return nested_unrelated_function(2)

    observed = []
    original_args = signatures.SignaturePlan._prepare_ambient_args
    original_return = signatures.SignaturePlan._prepare_ambient_return

    def observe_args(plan, args, kwargs):
        observed.append(("args", plan))
        return original_args(plan, args, kwargs)

    def observe_return(plan, value, **kwargs):
        observed.append(("return", plan))
        return original_return(plan, value, **kwargs)

    monkeypatch.setattr(
        signatures.SignaturePlan, "_prepare_ambient_args", observe_args,
    )
    monkeypatch.setattr(
        signatures.SignaturePlan, "_prepare_ambient_return", observe_return,
    )
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    subject = NestedFunctionValue(repo=repo)
    try:
        assert subject.run(managed=ManagedConfig(state_repo=repo)).value == 3
        assert [stage for stage, _ in observed] == ["args", "return"]
        nested_plan = observed[0][1]
        assert observed[1][1] is nested_plan
        assert nested_plan.target.__qualname__.startswith(
            f"{nested_unrelated_function.__qualname__}."
        )
        assert subject.run.status(state_repo=repo).state == "completed"
    finally:
        repo.close(flush=False)


@pytest.mark.parametrize("placement", ("local", "subprocess"))
def test_nested_managed_config_preserves_selected_state_ref_authority(
        tmp_path, placement: str) -> None:
    """Nested local and worker calls retain explicit config and StateRef authority."""

    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    control_store = DirStore(tmp_path / "control", query_index="none")
    subject = ManagedMatrixValue(repo=repo)
    argument = MatrixArgument(3, repo=repo)
    argument_state = repo.save_object(argument, deep_capture=True)
    repo.save_object(subject, deep_capture=True)
    nested_config = {"config": [matrix_config(repo, control_store)]}
    if placement == "local":
        assert nested_managed_call(subject, argument_state, nested_config).graph_equal(argument_state)
    else:
        spool = tmp_path / "spool"
        spool.mkdir()
        executor = CoreExecutor(
            SubProcessConfig(spool_directory=spool),
            core=CoreOptions(
                repo=repo, control_store=control_store, return_objects=False,
            ),
        )
        try:
            future = executor.submit(
                nested_managed_call, subject, argument_state, nested_config,
            )
            assert future.result(timeout=15).graph_equal(argument_state)
            future.cleanup(timeout=5)
        finally:
            executor.close(cancel=True, timeout=10)
    status = subject.operation_afm.status(
        state_repo=repo, control_store=control_store,
    )
    assert status.state == "completed"
    restored = repo.load_state_ref(status.final, reuse_live="never")
    assert (restored.calls, restored.value) == (1, 3)
