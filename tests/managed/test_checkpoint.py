"""U7 checkpoint safe-point publication coverage."""

from __future__ import annotations

import os
import sys
from threading import Event, Thread

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.reference_values import StateRef
from dryml.core.signatures import Ref
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedConflictError, ManagedContextError, managed_operation
from dryml.managed.errors import ManagedRecoveryError


class CheckpointValue(Pickleable):
    """Stateful receiver whose checkpoints capture an intermediate value."""

    seen_context = None
    entered = Event()
    release = Event()

    def __init__(self, value=0):
        self.value = value

    @managed_operation(resumable=True)
    def save_then_change(self, *, managed) -> Ref[StateRef]:
        """Publish value one before mutating the final state to value two."""

        self.value = 1
        checkpoint = managed.checkpoint()
        self.value = 2
        return checkpoint

    @managed_operation()
    def capture_context(self, *, managed):
        """Expose the active context to a coordinating test thread."""

        type(self).seen_context = managed
        type(self).entered.set()
        assert type(self).release.wait(10)

    @managed_operation()
    def recursive(self, *, managed):
        """Attempt a recursive checkpoint through an observer callback."""

        managed.checkpoint()

    @managed_operation()
    def fork_context(self, *, managed):
        """Prove an inherited child cannot use the parent's active context."""

        child = os.fork()
        if child == 0:  # pragma: no cover - result is asserted by the parent process.
            try:
                managed.checkpoint()
            except ManagedContextError:
                os._exit(0)
            os._exit(1)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0


def test_checkpoint_saves_exact_intermediate_state_and_associates_it(tmp_path):
    """A checkpoint remains recoverable after later ordinary mutation and final save."""

    store = DirStore(tmp_path / "store")
    repo = Repo((store,))
    value = CheckpointValue(repo=repo)

    checkpoint = value.save_then_change(managed=ManagedConfig(state_repo=store))

    status = value.save_then_change.status(state_repo=store)
    assert value.value == 2
    assert status.checkpoint_state_ref == checkpoint
    assert status.final_state_ref != checkpoint

    loaded = Repo._for_state_io((store,))
    try:
        assert loaded.load_state_ref(checkpoint, reuse_live="never").value == 1
    finally:
        loaded.close()


@pytest.mark.parametrize("immediate_recheck, share_direct, share_control, extra_store, expected", [
    (True, False, False, False, 11),
    (False, False, False, False, 9),
    (False, True, False, False, 7),
    (False, True, True, False, 6),
    (False, False, True, False, 8),
    (False, True, True, True, 8),
])
def test_checkpoint_changed_final_semantic_validation_budget(tmp_path, monkeypatch, immediate_recheck, share_direct, share_control, extra_store, expected):
    """Count complete closure work without replacing guarded Store readers."""

    from dryml.core.materialization import _build_exact_state_load_plan
    from dryml.managed import storage
    from dryml.managed.control import ManagedControlStore, _OwnerFields

    if immediate_recheck:
        consume = storage._PublicationEvent.consume

        def consume_then_recheck(self, repo, state_ref, report, invocation):
            consume(self, repo, state_ref, report, invocation)
            storage.validate_state_ref(repo, state_ref)

        monkeypatch.setattr(storage._PublicationEvent, "consume", consume_then_recheck)
    if not share_control:
        def separate_fields(self, operation_id, *, attempt_id, owner_id, kind, digest=None, force_interrupt=False):
            fields = _OwnerFields(kind, digest, force_interrupt)
            return self.transition_running_owner(
                operation_id, attempt_id=attempt_id, owner_id=owner_id, build=fields.apply,
            )

        monkeypatch.setattr(ManagedControlStore, "_transition_running_fields", separate_fields)

    if not share_direct:
        monkeypatch.setattr(storage, "_matching_direct_publication_coverage", lambda *args: False)

    store = DirStore(tmp_path / "store")
    repo = Repo((store, DirStore(tmp_path / "unselected")) if extra_store else (store,))
    value = CheckpointValue(repo=repo)
    calls = []
    accelerated = []
    readers = (DirStore.read_state_ref_record, DirStore.read_snapshot_metadata,
               DirStore.open_local_state, DirStore._read_snapshot)
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is _build_exact_state_load_plan.__code__:
            calls.append(frame.f_locals["state_ref"])
        if event == "return" and frame.f_code is DirStore._can_reuse_exact_validation.__code__:
            accelerated.append(arg)

    sys.setprofile(observe)
    try:
        checkpoint = value.save_then_change(managed=ManagedConfig(state_repo=repo))
    finally:
        sys.setprofile(previous)
    assert len(calls) == expected
    assert accelerated and all(accelerated)
    assert readers == (DirStore.read_state_ref_record, DirStore.read_snapshot_metadata,
                       DirStore.open_local_state, DirStore._read_snapshot)
    assert value.value == 2
    status = value.save_then_change.status(state_repo=repo)
    assert status.checkpoint_state_ref == checkpoint
    assert status.final_state_ref != checkpoint
    if expected == 6:
        assert calls == [checkpoint, checkpoint, checkpoint,
                         status.final_state_ref, checkpoint, status.final_state_ref]


def test_context_rejects_cross_thread_and_inactive_checkpoint_use(tmp_path):
    """Only the creating active invocation thread may publish a checkpoint."""

    CheckpointValue.seen_context = None
    CheckpointValue.entered, CheckpointValue.release = Event(), Event()
    store = DirStore(tmp_path / "store")
    value = CheckpointValue(repo=Repo((store,)))
    outcome = []
    worker = Thread(target=lambda: outcome.append(value.capture_context(managed=ManagedConfig(state_repo=store))))
    worker.start()
    assert CheckpointValue.entered.wait(10)
    try:
        errors = []
        foreign = Thread(target=lambda: errors.append(pytest.raises(ManagedContextError, CheckpointValue.seen_context.checkpoint).value.reason))
        foreign.start()
        foreign.join(10)
        assert not foreign.is_alive()
        assert errors == ["inactive_context"]
    finally:
        CheckpointValue.release.set()
        worker.join(10)
    assert outcome == [None]
    with pytest.raises(ManagedContextError, match="inactive_context"):
        CheckpointValue.seen_context.checkpoint()


def test_recursive_checkpoint_is_rejected_and_the_saved_checkpoint_is_retained(tmp_path):
    """Observers cannot recursively enter the same context safe-point protocol."""

    store = DirStore(tmp_path / "store")
    value = CheckpointValue(repo=Repo((store,)))

    with pytest.raises(ManagedContextError, match="recursive_checkpoint"):
        value.recursive(managed=ManagedConfig(state_repo=store, callbacks=[lambda obj, context: context.checkpoint()]))
    status = value.recursive.status(state_repo=store)
    assert status.state == "failed"
    assert status.checkpoint_state_ref is not None


@pytest.mark.skipif(os.name == "nt" or not hasattr(os, "fork"), reason="fork context invalidation requires POSIX fork")
def test_forked_child_cannot_use_parent_context(tmp_path):
    """PID validation fails before an inherited child can touch parent ownership."""

    store = DirStore(tmp_path / "store")
    value = CheckpointValue(repo=Repo((store,)))
    assert value.fork_context(managed=ManagedConfig(state_repo=store)) is None


@pytest.fixture
def publication_handoff(tmp_path, fixed_managed_snapshot_environment):
    """Produce a real event under real invocation graph ownership, without control."""

    from dryml.managed.context import _create_context
    from dryml.managed import storage

    repo = Repo(DirStore(tmp_path / "state"))
    value = CheckpointValue(repo=repo)
    with storage._acquire_state_ownership(repo, value) as ownership:
        context = _create_context(
            control_store=repo.default_store, operation_id="operation",
            attempt_id="attempt", owner_id="owner", is_resuming=False,
            checkpoint_state_ref=None, obj=value, state_repo=repo,
            ownership=ownership, control=None, callbacks=(),
            snapshot_observer=lambda: fixed_managed_snapshot_environment,
        )
        try:
            result = storage._publish_managed_state_result(
                repo, value, reservation=ownership.reservation,
                snapshot_observer=context._snapshot_observer, invocation=context,
            )
            yield repo, context, result
        finally:
            context._deactivate()
    repo.close(flush=False)


@pytest.mark.parametrize("mismatch", ["target", "collision", "report", "repo", "invocation", "coverage", "graph", "ownership", "expired", "exit", "reuse"])
def test_publication_event_requires_exact_completed_event(publication_handoff, monkeypatch, mismatch):
    """Reports, digests, or surviving handles alone cannot authorize event reuse."""

    from dataclasses import replace

    repo, context, result = publication_handoff
    event = result.event
    target, report, caller = result.state_ref, result.report, context
    selected_repo = repo
    if mismatch in {"target", "collision"}:
        target = StateRef(target.object, {
            path: state_hash.rsplit("-", 1)[0] + "-" + "0" * 64
            for path, state_hash in target.states.items()
        })
        if mismatch == "collision":
            digest = result.state_ref.digest()
            monkeypatch.setattr(StateRef, "digest", lambda self: digest)
            assert target.digest() == result.state_ref.digest()
        assert target != result.state_ref
    elif mismatch == "report":
        report = replace(report)
        assert report == result.report and report is not result.report
    elif mismatch == "repo":
        selected_repo = Repo._for_state_io(repo.stores)
    elif mismatch == "invocation":
        caller = object()
    elif mismatch == "coverage":
        event._coverage = replace(event._coverage, required=())
    elif mismatch == "graph":
        monkeypatch.setattr(context, "_obj", object())
    elif mismatch == "ownership":
        monkeypatch.setattr(context._ownership, "_active", False)
    elif mismatch == "expired":
        context._expire_publication_event()
    elif mismatch == "exit":
        context._deactivate()
    else:
        event.consume(repo, target, report, context)
    try:
        with pytest.raises((ManagedRecoveryError, ManagedConflictError), match="invalid_publication_event|invalid_state_owner"):
            event.consume(selected_repo, target, report, caller)
        assert not event._active
        assert context._publication_event is None
    finally:
        if mismatch == "ownership":
            context._ownership._active = True
        if selected_repo is not repo:
            selected_repo.close(flush=False)


@pytest.mark.parametrize("foreign", ["thread", "pid"])
def test_publication_event_rejects_foreign_caller_without_retiring_owner(publication_handoff, monkeypatch, foreign):
    """Ownership is checked before a foreign caller can consume or expire evidence."""

    from dryml.managed import storage

    repo, context, result = publication_handoff
    errors = []

    def attempt():
        with pytest.raises(ManagedRecoveryError, match="invalid_publication_event"):
            result.event.consume(repo, result.state_ref, result.report, context)
        errors.append(True)

    if foreign == "thread":
        worker = Thread(target=attempt)
        worker.start()
        worker.join(10)
        assert not worker.is_alive()
    else:
        pid = os.getpid()
        with monkeypatch.context() as patch:
            patch.setattr(storage.os, "getpid", lambda: pid + 1)
            attempt()
    assert errors == [True]
    assert result.event._active
    result.event.consume(repo, result.state_ref, result.report, context)
    assert not result.event._active


def test_public_publisher_keeps_tuple_and_does_not_mint_event(publication_handoff):
    """The public projection remains a two-item tuple from the same implementation."""

    from dryml.managed.storage import publish_managed_state

    repo, context, result = publication_handoff
    result.event.consume(repo, result.state_ref, result.report, context)
    published = publish_managed_state(
        repo, context._obj, reservation=context._ownership.reservation,
        snapshot_observer=context._snapshot_observer,
    )
    assert type(published) is tuple and len(published) == 2
    assert published[0].object == result.state_ref.object
    assert published[0] == context._obj.last_state_ref
    assert published[1].snapshots
    assert context._publication_event is None


def test_publication_event_cannot_be_serialized(publication_handoff):
    """Completed-event reuse creates no transferable or persistent authority."""

    import pickle

    _, _, result = publication_handoff
    with pytest.raises(TypeError, match="cannot be serialized"):
        pickle.dumps(result.event)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cut_joined_publication_preserves_exact_validation_exception_chain(publication_handoff, monkeypatch, error_type):
    """Joining a cut retains the validator's existing BaseException classification."""

    from dryml.core import materialization
    from dryml.managed import storage

    repo, _, result = publication_handoff
    original = error_type("exact preflight stopped")

    def fail_preflight(*args, **kwargs):
        raise original

    monkeypatch.setattr(materialization, "build_exact_state_load_plan", fail_preflight)
    with repo._read_cut() as cut:
        with pytest.raises(ManagedRecoveryError) as caught:
            storage._validate_publication_closure(repo, result.state_ref, cut)
    assert caught.value.__cause__ is original
    assert not cut.active


def test_event_expires_before_boundaries_and_callbacks_without_immediate_validation(tmp_path, monkeypatch):
    """Publisher cuts close and handoffs retire before control or callback hooks."""

    from dryml.managed import context as context_module, runtime, storage

    repo = Repo(DirStore(tmp_path / "state"))
    value = CheckpointValue(repo=repo)
    events = []
    original = storage._publish_managed_state_result

    def publish(*args, **kwargs):
        result = original(*args, **kwargs)
        assert repo._topology_leases == 1
        events.append(result.event)
        return result

    def boundary(stage):
        if stage in {"state_published", "final_state_published", "callbacks_started"}:
            assert events and all(not event._active for event in events)
            assert repo._topology_leases == 1

    def no_immediate(*args, **kwargs):
        raise AssertionError("immediate caller validation must not run")

    monkeypatch.setattr(storage, "_publish_managed_state_result", publish)
    monkeypatch.setattr(runtime, "_publish_managed_state_result", publish)
    monkeypatch.setattr(runtime, "validate_state_ref", no_immediate)
    monkeypatch.setattr(context_module, "_checkpoint_boundary", boundary)
    monkeypatch.setattr(runtime, "_completion_boundary", boundary)
    value.save_then_change(managed=ManagedConfig(
        state_repo=repo, callbacks=[lambda *args: boundary("callbacks_started")],
    ))
    assert len(events) == 2
    assert repo._topology_leases == 0
