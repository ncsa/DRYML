"""U7 checkpoint safe-point publication coverage."""

from __future__ import annotations

import os
from threading import Event, Thread

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.reference_values import StateRef
from dryml.core.signatures import Ref
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedContextError, managed_operation


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
