"""End-to-end public managed-operation lifecycle acceptance coverage."""

from __future__ import annotations

import pytest

from dryml import Object, Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import (
    ManagedConfig,
    ManagedInterrupted,
    ManagedRerunRequiredError,
    managed_operation,
)


class AcceptanceCounter(Pickleable):
    """Counter whose first attempt checkpoints and cooperatively interrupts."""

    def __init__(self, value=0):
        self.value = value
        self.should_interrupt = True

    @managed_operation(resumable=True)
    def advance(self, amount, *, managed):
        """Advance once per entry and interrupt only the first attempt."""

        self.value += amount
        if self.should_interrupt:
            managed.checkpoint()
            self.should_interrupt = False
            self.value += amount
            managed.interrupt()
        return self.value


class AcceptanceRoot(Object):
    """Stateless root retaining a stateful child for graph acceptance evidence."""

    def __init__(self, child):
        self.child = child
        self.unsaved_marker = "root-only"

    @managed_operation()
    def advance_child(self, amount, *, managed):
        """Mutate the stateful child without making root-only data checkpoint state."""

        self.child.value += amount
        return self.child.value


def test_checkpoint_interrupt_fresh_resume_completion_and_rerun_chain(tmp_path):
    """Resume a retained attempt from a fresh graph, then require explicit rerun."""

    store = DirStore(tmp_path / "state")
    counter = AcceptanceCounter(repo=Repo((store,)))
    config = ManagedConfig(state_repo=store)

    with pytest.raises(ManagedInterrupted):
        counter.advance(2, managed=config)
    interrupted = counter.advance.status(state_repo=store)
    assert interrupted.state == "interrupted"
    assert interrupted.checkpoint_state_ref is not None

    recovery_repo = Repo((store,))
    try:
        recovered = recovery_repo.load_state_ref(
            interrupted.checkpoint_state_ref, reuse_live="never",
        )
        assert recovered.advance(2, managed=config) == 6
        completed = recovered.advance.status(state_repo=store)
        assert (completed.state, completed.attempt_id) == ("completed", interrupted.attempt_id)
        with pytest.raises(ManagedRerunRequiredError, match="already_completed"):
            recovered.advance(2, managed=config)
        assert recovered.advance(2, managed=ManagedConfig(state_repo=store, rerun=True)) == 8
        assert recovered.advance.status(state_repo=store).attempt_id != interrupted.attempt_id
    finally:
        recovery_repo.close()


def test_stateless_root_publishes_descendant_state_without_root_payload(tmp_path):
    """Persist a descendant graph while retaining ordinary stateless-root fields."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = AcceptanceRoot(AcceptanceCounter(1, repo=repo), repo=repo)

    assert root.advance_child(4, managed=ManagedConfig(state_repo=store)) == 5
    assert root.last_state_ref is not None
    assert root.child.last_state_ref is None
    assert root.unsaved_marker == "root-only"
