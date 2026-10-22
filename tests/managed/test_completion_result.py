"""Public durable completion and fresh lazy-source qualification."""

import inspect
import shutil

import pytest

from dryml.core import Repo
from dryml.core.object import Pickleable
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, managed_operation


class CompletionValue(Pickleable):
    """Count real body/capture/restore and consumer boundaries."""

    bodies = captures = restores = handoffs = 0

    def __init__(self, value=0):
        self.value = value

    def save_state_to_dir_imp(self, directory, *, codec):
        type(self).captures += 1
        return super().save_state_to_dir_imp(directory, codec=codec)

    def restore_state_from_dir_imp(self, directory, *, codec):
        type(self).restores += 1
        return super().restore_state_from_dir_imp(directory, codec=codec)

    @managed_operation(resumable=True)
    def advance(self, amount=3, *, managed):
        type(self).bodies += 1
        self.value += amount
        return {"value": self.value}

    def _managed_post_publication(self, final_state, state_repo, report):
        type(self).handoffs += 1


def test_public_result_detaches_and_replays_old_live_receiver_without_work(tmp_path):
    from dryml.managed import InvocationResult
    store = DirStore(tmp_path / "state", query_index="none")
    repo = Repo(store)
    live = CompletionValue(repo=repo)
    seed = repo.save(live)
    CompletionValue.bodies = CompletionValue.captures = CompletionValue.restores = CompletionValue.handoffs = 0
    first = live.advance(managed=ManagedConfig(state_repo=repo))
    assert type(first) is InvocationResult
    assert inspect.signature(live.advance).return_annotation is InvocationResult
    assert first.status.state == "completed" and first.final_state_ref.state is not None
    assert first.value == {"value": 3}
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == (1, 1, 0, 1)
    old = repo.load(seed, cache="none")
    counts = CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs
    replay = old.advance(3, managed=ManagedConfig(state_repo=repo))
    assert old.value == 0 and replay.value == first.value
    assert replay.execution.run_id == first.execution.run_id
    assert replay.execution.producer_invocation_id == first.execution.invocation_id
    assert replay.execution.invocation_id != first.execution.invocation_id
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == counts
    shutil.rmtree(store.state_data_path(first.final_state_ref.state.key))
    shutil.rmtree(store.state_data_path(seed.state.key))
    repo.close(flush=False)
    assert first.status.state == "completed" and first.final_state_ref.graph_equal(replay.final_state_ref)
    assert first.value == {"value": 3}
    fresh = Repo(DirStore(tmp_path / "state", query_index="none"))
    again = old.advance(managed=ManagedConfig(state_repo=fresh))
    assert again.value == {"value": 3} and old.value == 0
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores, CompletionValue.handoffs) == counts


@pytest.mark.parametrize("fault", ["missing", "corrupt", "replacement"])
def test_each_lazy_access_revalidates_source_even_after_decode(tmp_path, fault):
    from dryml.managed import ManagedResultError
    from dryml.managed.results import result_path
    root = tmp_path / "state"
    store = DirStore(root, query_index="none")
    repo = Repo(store)
    live = CompletionValue(repo=repo)
    result = live.advance(managed=ManagedConfig(state_repo=repo))
    assert result.value == {"value": 3}
    counts = CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores
    path = result_path(store, result.result_reference.key)
    if fault == "missing":
        from pathlib import Path
        Path(path).unlink()
    elif fault == "corrupt":
        from pathlib import Path
        Path(path).write_bytes(b"{}")
    else:
        from pathlib import Path
        payload = Path(path).read_bytes()
        repo.close(flush=False)
        root.rename(tmp_path / "old-state")
        replacement = DirStore(root, query_index="none")
        replacement_path = Path(result_path(replacement, result.result_reference.key))
        replacement_path.parent.mkdir(parents=True)
        replacement_path.write_bytes(payload)
    assert result.status.state == "completed"
    with pytest.raises(ManagedResultError) as error:
        result.value
    assert error.value.reason == {"missing": "result_unavailable", "corrupt": "result_corrupt", "replacement": "result_source_changed"}[fault]
    assert (CompletionValue.bodies, CompletionValue.captures, CompletionValue.restores) == counts


def test_unsupported_return_fails_before_automatic_capture(tmp_path):
    from dataclasses import dataclass
    from dryml.managed import ManagedResultError
    from tests.managed.u16_fixtures import Value, setup
    @dataclass(frozen=True)
    class Summary:
        value: int
    control, live = setup(tmp_path)
    Value.action = lambda obj, ctx: Summary(1)
    with pytest.raises(ManagedResultError) as error:
        live.run(managed=ManagedConfig(state_repo=control.state_repo))
    assert Value.captures == 0 and Value.bodies == 1
    assert error.value.execution and error.value.failed_result.status.state == "failed"
    assert live.run.status(state_repo=control.state_repo).failure_code == "result_error"
