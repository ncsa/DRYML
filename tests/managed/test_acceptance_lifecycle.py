"""End-to-end public managed-operation lifecycle acceptance coverage."""

from __future__ import annotations

from dataclasses import replace

import pytest

pytestmark = pytest.mark.usefixtures("fixed_managed_snapshot_environment")

from dryml import Object, Repo
from dryml.core.object import Pickleable
from dryml.core.node import require_saved
from dryml.core.utils.graph.path import Parameter
from dryml.core.store.dir import DirStore
from dryml.managed import (
    ManagedConfig,
    InvocationResult,
    ManagedInterrupted,
    ManagedRerunRequiredError,
    managed_operation,
)


class AcceptanceCounter(Pickleable):
    """Counter whose first invocation checkpoints and cooperatively interrupts."""

    bodies = captures = restores = 0

    def __init__(self, value=0):
        self.value = value
        self.should_interrupt = True

    def save_state_to_dir_imp(self, dest_dir, *, codec):
        """Count and delegate pickle-state capture; propagate codec/IO errors."""
        type(self).captures += 1
        return super().save_state_to_dir_imp(dest_dir, codec=codec)

    def restore_state_from_dir_imp(self, src_dir, *, codec):
        """Count and delegate pickle-state restoration; propagate codec/IO errors."""
        type(self).restores += 1
        return super().restore_state_from_dir_imp(src_dir, codec=codec)

    @managed_operation(resumable=True)
    def advance(self, amount, *, managed):
        """Advance once per entry and interrupt only the first invocation."""

        type(self).bodies += 1
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
    """Fresh resume keeps its run; replay is inert and explicit rerun starts a new run."""

    store = DirStore(tmp_path / "state")
    counter = AcceptanceCounter(repo=Repo((store,)))
    config = ManagedConfig(state_repo=store)

    with pytest.raises(ManagedInterrupted):
        counter.advance(2, managed=config)
    interrupted = counter.advance.status(state_repo=store)
    assert interrupted.state == "interrupted"
    assert require_saved(interrupted.checkpoint) is interrupted.checkpoint

    recovery_repo = Repo((store,))
    try:
        recovered = recovery_repo.load_state_ref(
            interrupted.checkpoint, reuse_live="never",
        )
        assert recovered is not counter
        assert recovered.object_id == counter.object_id
        assert recovered.value == 4
        result = recovered.advance(2, managed=config)
        assert type(result) is InvocationResult and result.value == 6
        completed = recovered.advance.status(state_repo=store)
        assert (completed.state, completed.run_id) == ("completed", interrupted.run_id)
        assert completed.invocation_id != interrupted.invocation_id
        assert result.execution.operation_id == interrupted.operation_id
        assert result.execution.invocation_id == completed.invocation_id
        assert completed.final.graph_equal(result.final_state_ref)
        counts = AcceptanceCounter.bodies, AcceptanceCounter.captures, AcceptanceCounter.restores
        replay = recovered.advance(2, managed=config)
        assert type(replay) is InvocationResult and replay.value == 6
        assert replay.execution.run_id == completed.run_id
        assert replay.execution.producer_invocation_id == completed.invocation_id
        assert replay.execution.invocation_id not in {interrupted.invocation_id, completed.invocation_id}
        assert replay.final_state_ref.graph_equal(result.final_state_ref)
        after_replay = recovered.advance.status(state_repo=store)
        assert after_replay.checkpoint.graph_equal(completed.checkpoint)
        assert after_replay.final.graph_equal(completed.final)
        assert replace(after_replay, checkpoint=completed.checkpoint, final=completed.final) == completed
        with pytest.raises(ManagedRerunRequiredError, match="argument_mismatch"):
            recovered.advance(3, managed=config)
        assert (AcceptanceCounter.bodies, AcceptanceCounter.captures, AcceptanceCounter.restores) == counts
        rerun = recovered.advance(2, managed=ManagedConfig(state_repo=store, rerun=True))
        assert type(rerun) is InvocationResult and rerun.value == 8
        assert rerun.execution.operation_id == interrupted.operation_id
        assert rerun.execution.run_id != completed.run_id
        assert rerun.execution.invocation_id not in {
            interrupted.invocation_id, completed.invocation_id, replay.execution.invocation_id,
        }
        assert recovered.advance.status(state_repo=store).run_id == rerun.execution.run_id
    finally:
        recovery_repo.close()


def test_stateless_root_publishes_descendant_state_without_root_payload(tmp_path):
    """Persist a descendant graph while retaining ordinary stateless-root fields."""

    store = DirStore(tmp_path / "state")
    repo = Repo((store,))
    root = AcceptanceRoot(AcceptanceCounter(1, repo=repo), repo=repo)

    result = root.advance_child(4, managed=ManagedConfig(state_repo=store))
    assert type(result) is InvocationResult and result.value == 5
    final = result.final_state_ref
    assert require_saved(final) is final
    assert root.last_state_ref.graph_equal(final)
    assert final.state is None
    child = final.at((Parameter("child"),))
    assert child.state is not None and child.object_id == root.child.object_id
    assert list(final.state_spec().values()) == [child.state]
    assert root.child.last_state_ref.graph_equal(child)
    assert root.unsaved_marker == "root-only"
    loaded = repo.load_state_ref(final, reuse_live="never")
    assert loaded.child.value == 5 and loaded.unsaved_marker == "root-only"
    assert loaded.child.object_id == root.child.object_id
    assert loaded.child.last_state_ref is None
