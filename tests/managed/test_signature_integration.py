"""Actual public result cutover through owning signature and nested boundaries."""

from __future__ import annotations

import pytest

from dryml.core import Repo, SavedNode, function
from dryml.core.object import Pickleable
from dryml.core.signatures import Mat, Ref, SignatureError, SignaturePlan
from dryml.core.store.dir import DirStore
from dryml.managed import ManagedConfig, ManagedResultError, managed_operation


@function
def nested(amount: int) -> int:
    return amount + 1


class SignatureValue(Pickleable):
    """Deliver supported Mat inputs and explicit inert return metadata."""
    captures = 0

    def __init__(self, value=0):
        self.value = value
        self.calls = 0

    def save_state_to_dir_imp(self, directory, *, codec):
        type(self).captures += 1
        return super().save_state_to_dir_imp(directory, codec=codec)

    @managed_operation()
    @function
    def normalized(self, amount: int = 2, *, managed) -> int:
        self.calls += 1
        return nested(amount)

    @managed_operation()
    def receive(self, value: Mat[SignatureValue], *, managed) -> int:
        self.calls += 1
        return value.value

    @managed_operation()
    def metadata(self, *, managed) -> Ref[SavedNode]:
        return Ref(managed.checkpoint())

    @managed_operation()
    def return_conflict(self, *, managed) -> Mat[SignatureValue]:
        self.calls += 1
        return Ref(managed.checkpoint())

    @managed_operation()
    def excluded(self: Mat[SignatureValue], *, managed: Mat[SignatureValue]):
        assert managed.active
        self.calls += 1
        return self


def test_return_and_argument_normalization_once_nested_independent_replay_zero_return(tmp_path, monkeypatch):
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = SignatureValue(repo=repo)
    counts = []
    prepare_args, prepare_return = SignaturePlan.prepare_bound, SignaturePlan._prepare_return
    def args(self, *a, **kw):
        counts.append(("args", self))
        return prepare_args(self, *a, **kw)
    def returns(self, *a, **kw):
        counts.append(("return", self))
        return prepare_return(self, *a, **kw)
    monkeypatch.setattr(SignaturePlan, "prepare_bound", args)
    monkeypatch.setattr(SignaturePlan, "_prepare_return", returns)
    result = live.normalized(managed=ManagedConfig(state_repo=repo))
    assert result.value == 3 and live.calls == 1
    owner = SignatureValue.normalized.signature_plan()
    assert sum(kind == "args" and plan is owner for kind, plan in counts) == 1
    assert sum(kind == "return" and plan is owner for kind, plan in counts) == 1
    assert sum(kind == "return" and plan is not owner for kind, plan in counts) == 1
    counts.clear()
    replay = live.normalized(2, managed=ManagedConfig(state_repo=repo))
    assert replay.value == 3 and live.calls == 1
    assert counts == [("args", owner)]


def test_mat_input_is_delivered_live_but_result_is_supported_data(tmp_path):
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live, source = SignatureValue(repo=repo), SignatureValue(7, repo=repo)
    saved = repo.save(source)
    source.value = 99
    result = live.receive(saved, managed=ManagedConfig(state_repo=repo))
    assert result.value == 7 and live.calls == 1


def test_opposing_input_fails_before_body_and_control_creation(tmp_path):
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = SignatureValue(repo=repo)
    with pytest.raises(SignatureError): live.receive(Ref(repo.save(live)), managed=ManagedConfig(state_repo=repo))
    assert live.calls == 0 and live.receive.status(state_repo=repo).state == "not_started"


def test_return_normalization_failure_keeps_checkpoint_and_avoids_final_capture(tmp_path):
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = SignatureValue(repo=repo)
    SignatureValue.captures = 0
    with pytest.raises(SignatureError): live.return_conflict(managed=ManagedConfig(state_repo=repo))
    status = live.return_conflict.status(state_repo=repo)
    assert status.state == "failed" and status.checkpoint is not None and status.final is None
    assert SignatureValue.captures == 1


def test_metadata_return_is_normalized_once_and_codec_never_materializes_it(tmp_path):
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = SignatureValue(repo=repo)
    result = live.metadata(managed=ManagedConfig(state_repo=repo))
    assert result.value.graph_equal(result.status.checkpoint)


def test_excluded_receiver_and_context_are_not_converted_but_live_results_refuse(tmp_path):
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    live = SignatureValue(repo=repo)
    with pytest.raises(ManagedResultError): live.excluded(managed=ManagedConfig(state_repo=repo))
    assert live.calls == 1 and live.excluded.status(state_repo=repo).state == "failed"
