"""Accepted input cuts through real Experiment/Core persistence boundaries."""

from concurrent.futures import ThreadPoolExecutor
import pickle

import numpy as np
import pytest

from dryml.core import Repo, TensorSpec
from dryml.core.object import Pickleable
from dryml.core.repo import RepoSaveError
from dryml.core.store.dir import DirStore
from dryml.data import (ArrayDataset, ContinuationMismatchError, DatasetState,
                        DatasetTraversal, GeneratorDataset, RecoverableGenerator,
                        TraversalOwnershipError, Zip)
from dryml.models import Experiment, TrainFunction, TrainState
from dryml.models.utils import record_train_update


class Counter(Pickleable):
    """Persist a tiny mutable native-owner stand-in."""

    def __init__(self):
        self.value = 0


class Procedure(TrainFunction):
    """Reusable procedure; invocation input belongs exclusively to exp.state."""

    def __call__(self, exp, *, callbacks=()):
        with self.input_session(exp) as session:
            view = session.traversal.next_epoch()
            while True:
                try:
                    candidate = view.reserve()
                except StopIteration:
                    session.sync_boundary()
                    break
                with session.update(candidate):
                    exp.model.value += int(candidate.value)
                    record_train_update(exp, candidate.value, 1., batched=False,
                                        input_candidate=candidate)
                for callback in callbacks:
                    callback()


def experiment(tmp_path=None, *, trainer=None, values=(1, 2, 3)):
    repo = Repo() if tmp_path is None else Repo(DirStore(tmp_path / "state", query_index="none"))
    exp = Experiment(Counter(), trainer or Procedure(), ArrayDataset(np.asarray(values)), repo=repo)
    return repo, exp


def accept_one(exp, session):
    candidate = session.traversal.next_epoch().reserve()
    with session.update(candidate):
        exp.model.value += int(candidate.value)
        record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)
    return candidate


def test_positive_completion_captures_once_without_per_update_capture(monkeypatch):
    from dryml.data import DatasetTraversal
    calls = []
    original = DatasetTraversal.capture_state
    def capture(owner):
        calls.append(owner.position.sequence)
        return original(owner)
    monkeypatch.setattr(DatasetTraversal, "capture_state", capture)
    _, exp = experiment(values=tuple(range(40)))
    exp.train_fn(exp, callbacks=(lambda: calls.append("log"),))
    assert calls == ["log"] * 40 + [40]
    assert (exp.model.value, exp.state.step, exp.state.input_sequence) == (780, 40, 40)
    assert exp.state.dataset_state.position == exp.state.input_position
    assert exp.state._input_provider is None


def test_fresh_zero_work_is_source_inert_and_value_encoding_is_v3(tmp_path, monkeypatch):
    repo, exp = experiment(tmp_path)
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("prepared zero work"))
    with exp.train_fn.input_session(exp, positive=False) as session:
        assert session.traversal is None
    encoded = exp.state.__getstate__()
    assert encoded["version"] == 3
    assert encoded["fields"]["input_started"] is False
    assert encoded["fields"]["dataset_state"] is None
    assert exp.state._input_provider is None
    restored = repo.load(repo.save_object(exp), cache="none")
    assert restored.state.input_started is False
    assert restored.state.input_sequence == 0 and restored.state.dataset_state is None
    assert restored.state._input_provider is None


def test_exception_without_current_capture_rejects_retry_before_prepare(monkeypatch):
    _, exp = experiment()
    with pytest.raises(RuntimeError, match="callback"):
        with exp.train_fn.input_session(exp) as session:
            accept_one(exp, session)
            raise RuntimeError("callback")
    assert exp.state.step == 1 and exp.model.value == 1
    assert not exp.model._restore_failed
    assert exp.state._input_provider is None
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("prepared stale exact input"))
    with pytest.raises(ValueError, match="current.*record|durable reload"):
        with exp.train_fn.input_session(exp):
            pytest.fail("admitted stale exact retry")
    with pytest.raises(ValueError, match="current.*record|durable reload"):
        exp.state.__getstate__()


def test_bundle_has_one_continuation_and_restores_pair_atomically(tmp_path):
    _, exp = experiment()
    exp.train_fn(exp)
    payload = tmp_path / "payload"
    payload.mkdir()
    exp.save_state_to_dir_imp(payload, codec="pkl")
    from dryml.core.utils.general import pickle_load
    encoded = pickle_load(payload / "experiment_state.pkl")
    assert encoded["version"] == 3
    assert "dataset_state" not in encoded["state"]["fields"]
    assert encoded["continuation"]["path"] == "dataset-state"
    assert DatasetState.load(payload / "dataset-state").position == exp.state.input_position
    _, target = experiment()
    target.restore_state_from_dir_imp(payload, codec="pkl")
    assert target.state.input_position == exp.state.input_position
    before = target.state
    (payload / "dataset-state" / "record.json").write_bytes(b"truncated")
    with pytest.raises(ValueError):
        target.restore_state_from_dir_imp(payload, codec="pkl")
    assert target.state is before


def test_native_update_reservation_excludes_other_thread_save(tmp_path):
    repo, exp = experiment(tmp_path)
    with exp.train_fn.input_session(exp, repo=repo) as session:
        candidate = session.traversal.next_epoch().reserve()
        with session.update(candidate):
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(repo.save_object, exp)
                with pytest.raises(RepoSaveError, match="reserved"):
                    future.result(timeout=5)
            exp.model.value = 1
            record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)
        saved = repo.save_object(exp)
    restored = repo.load(saved, cache="none")
    assert (restored.model.value, restored.state.step, restored.state.input_sequence) == (1, 1, 1)


def test_standalone_value_roundtrip_and_in_memory_validation_do_not_capture():
    _, exp = experiment()
    exp.train_fn(exp)
    restored = pickle.loads(pickle.dumps(exp.state))
    assert restored.dataset_state.position == exp.state.input_position
    assert restored._input_provider is None
    encoded = exp.state.__getstate__()
    record = encoded["fields"]["dataset_state"]
    assert set(record) == {"version", "metadata", "payloads"}
    assert record["metadata"] is exp.state.dataset_state._metadata
    assert record["payloads"] is exp.state.dataset_state._payloads
    assert "_input_provider" not in encoded["fields"]
    exp.state.validate()


def test_v2_train_state_rejected_without_assignment():
    state = TrainState(step=7)
    before = state.__getstate__()
    payload = TrainState().__getstate__()
    payload["version"] = 2
    with pytest.raises(ValueError, match="version"):
        state.__setstate__(payload)
    assert state.__getstate__() == before


def test_value_encoding_rejects_inconsistent_v3_sequence_atomically():
    _, exp = experiment()
    exp.train_fn(exp)
    encoded = exp.state.__getstate__()
    encoded["fields"]["input_sequence"] += 1
    target = TrainState(step=9)
    before = target.__getstate__()
    with pytest.raises(ValueError, match="current matching record"):
        target.__setstate__(encoded)
    assert target.__getstate__() == before


@pytest.mark.parametrize("stage", ["after_accounting", "after_frontier"])
def test_accounting_frontier_interruption_invalidates_every_native_owner(tmp_path, monkeypatch, stage):
    from dryml.models import training_transaction
    repo, exp = experiment(tmp_path)
    durable = repo.save_object(exp)
    def fail(current):
        if current == stage:
            raise KeyboardInterrupt("split cut")
    monkeypatch.setattr(training_transaction, "_training_input_boundary", fail)
    with pytest.raises(KeyboardInterrupt, match="split cut"):
        with exp.train_fn.input_session(exp, repo=repo) as session:
            accept_one(exp, session)
    assert exp._restore_failed and exp.model._restore_failed
    assert exp.state._input_provider is None
    for owner in (exp, exp.model):
        with pytest.raises(RepoSaveError, match="invalidated"):
            repo.save_object(owner)
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("invalid owner prepared"))
    with pytest.raises(RepoSaveError, match="invalidated"):
        with exp.train_fn.input_session(exp, repo=repo):
            pass
    loaded = repo.load(durable, cache="none")
    assert loaded.model.value == loaded.state.step == 0


def test_callback_failure_with_current_record_allows_exact_retry(tmp_path):
    repo, exp = experiment(tmp_path)
    with pytest.raises(RuntimeError, match="callback"):
        with exp.train_fn.input_session(exp, repo=repo) as session:
            accept_one(exp, session)
            saved = repo.save_object(exp)
            record = exp.state.dataset_state
            raise RuntimeError("callback")
    assert exp.state._input_provider is None
    assert not exp.model._restore_failed
    with exp.train_fn.input_session(exp, repo=repo) as session:
        assert session.traversal.position == record.position
        candidate = session.traversal.next_epoch().reserve()
        assert candidate.value == 2
        with session.update(candidate):
            exp.model.value += 2
            record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)
    restored = repo.load(saved, cache="none")
    assert (restored.state.step, restored.model.value) == (1, 1)


def test_invalidated_native_owner_rejects_even_with_current_record(tmp_path, monkeypatch):
    repo, exp = experiment(tmp_path)
    with exp.train_fn.input_session(exp, repo=repo) as session:
        accept_one(exp, session)
        session.capture()
    repo._invalidate_restoration_target((exp.model,))
    monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("invalid native owner prepared"))
    with pytest.raises(RepoSaveError, match="invalidated"):
        with exp.train_fn.input_session(exp, repo=repo):
            pass


def test_dynamic_capture_failure_keeps_accepted_facts_and_prior_durable_checkpoint(tmp_path, monkeypatch):
    repo, exp = experiment(tmp_path)
    before = repo.save_object(exp)
    def fail(_):
        raise ValueError("dynamic buffer cannot encode")
    with pytest.raises(RepoSaveError, match="snapshot-local") as failure:
        with exp.train_fn.input_session(exp, repo=repo) as session:
            accept_one(exp, session)
            monkeypatch.setattr(DatasetTraversal, "capture_state", fail)
            repo.save_object(exp)
    assert isinstance(failure.value.__cause__, ValueError)
    assert str(failure.value.__cause__) == "dynamic buffer cannot encode"
    assert (exp.state.step, exp.state.input_sequence, exp.model.value) == (1, 1, 1)
    assert not exp.model._restore_failed
    assert exp.state._input_provider is None
    loaded = repo.load(before, cache="none")
    assert (loaded.state.step, loaded.model.value, loaded.state.input_started) == (0, 0, False)


def test_final_capture_failure_retires_provider_and_keeps_prior_record(monkeypatch):
    _, exp = experiment()
    def fail(_):
        raise ValueError("final capture failed")
    with pytest.raises(ValueError, match="final capture failed"):
        with exp.train_fn.input_session(exp) as session:
            prior = session.capture()
            accept_one(exp, session)
            monkeypatch.setattr(DatasetTraversal, "capture_state", fail)
    assert exp.state.dataset_state is prior and prior.position.sequence == 0
    assert (exp.state.step, exp.state.input_sequence, exp.model.value) == (1, 1, 1)
    assert exp.state._input_provider is None and not exp.model._restore_failed
    with pytest.raises(ValueError, match="current matching record"):
        exp.state.__getstate__()


def test_cross_thread_save_at_accepted_cut_rejects_without_pulling_source(tmp_path, monkeypatch):
    repo, exp = experiment(tmp_path)
    with exp.train_fn.input_session(exp, repo=repo) as session:
        accept_one(exp, session)
        def fail(_):
            pytest.fail("wrong-thread source capture")
        with monkeypatch.context() as patch:
            patch.setattr(DatasetTraversal, "capture_state", fail)
            with ThreadPoolExecutor(1) as pool:
                future = pool.submit(repo.save_object, exp)
                with pytest.raises(RepoSaveError, match="snapshot-local") as failure:
                    future.result(timeout=5)
                assert isinstance(failure.value.__cause__, TraversalOwnershipError)
                assert "owner thread" in str(failure.value.__cause__)
        saved = repo.save_object(exp)
    assert repo.load(saved, cache="none").state.input_sequence == 1


def test_stable_owner_evidence_is_prepared_once_with_fresh_binding_checks(monkeypatch):
    repo, exp = experiment()
    builds = []
    original = repo._state_graph_evidence
    def evidence(obj):
        builds.append(obj)
        return original(obj)
    monkeypatch.setattr(repo, "_state_graph_evidence", evidence)
    with exp.train_fn.input_session(exp, repo=repo) as session:
        view = session.traversal.next_epoch()
        for _ in range(2):
            candidate = view.reserve()
            with session.update(candidate):
                record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)
        assert builds == [exp]
        prior = exp.model
        exp.model = Counter()
        candidate = view.reserve()
        with pytest.raises(RepoSaveError, match="binding"):
            with session.update(candidate):
                pytest.fail("changed graph accepted")
        exp.model = prior
        with session.update(candidate):
            record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)


def test_two_runs_share_procedure_but_no_cursor_provider_or_record():
    trainer = Procedure()
    _, first = experiment(trainer=trainer)
    _, second = experiment(trainer=trainer)
    before = set(vars(trainer))
    with trainer.input_session(first) as a, trainer.input_session(second) as b:
        accept_one(first, a)
        assert second.state.input_sequence == 0
        a.capture()
        b.capture()
        assert first.state._input_provider is a and second.state._input_provider is b
        assert first.state.dataset_state is not second.state.dataset_state
        assert b.traversal.position.sequence == 0
    assert first.state._input_provider is second.state._input_provider is None
    assert set(vars(trainer)) == before


def test_exact_mismatched_target_rejects_before_producer_acquisition(monkeypatch):
    _, exp = experiment()
    with exp.train_fn.input_session(exp):
        pass
    exp.train_data = ArrayDataset(np.asarray([7, 8, 9]))
    monkeypatch.setattr(DatasetTraversal, "_raw", lambda _: pytest.fail("acquired mismatched source"))
    with pytest.raises(ContinuationMismatchError):
        with exp.train_fn.input_session(exp, exact=False):
            pytest.fail("failed exact restore fell back")
    assert exp.state.input_recovery == "exact" and exp.state._input_provider is None


class DirectCounter(RecoverableGenerator):
    """Direct-only factory with visible acquisition and snapshot counts."""

    opens = captures = 0

    def __init__(self):
        type(self).opens += 1
        self.position = 0

    def __next__(self):
        if self.position == 3:
            raise StopIteration
        self.position += 1
        return self.position

    def capture_state(self):
        type(self).captures += 1
        return self.position

    def restore_state(self, state):
        self.position = state


def ordinary_counter():
    yield from (1, 2, 3)


@pytest.mark.parametrize("kind", ["indexed", "direct", "mixed"])
def test_consumer_recovery_setup_is_source_agnostic(kind):
    data = ArrayDataset(np.arange(3))
    if kind == "direct":
        data = GeneratorDataset(DirectCounter, spec=TensorSpec("int64", shape=()))
    elif kind == "mixed":
        data = Zip(ArrayDataset(np.arange(3)),
                   GeneratorDataset(ordinary_counter, spec=TensorSpec("int64", shape=())),
                   GeneratorDataset(DirectCounter, spec=TensorSpec("int64", shape=())))
    exp = Experiment(Counter(), Procedure(), data)
    DirectCounter.opens = DirectCounter.captures = 0
    with exp.train_fn.input_session(exp) as session:
        assert DirectCounter.opens == DirectCounter.captures == 0
        candidate = session.traversal.next_epoch().reserve()
        with session.update(candidate):
            record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)
        assert DirectCounter.captures == 0
        assert session.traversal.replay_recovery_enabled is (kind == "mixed")
    record = exp.state.dataset_state
    assert record.replay_recovery_enabled is (kind == "mixed")
    with data.open(state=record, acceptance="explicit") as restored:
        value = restored.next_epoch().reserve().value
        assert value == ((1, 2, 2) if kind == "mixed" else 2 if kind == "direct" else 1)


def test_initial_and_zero_new_update_completion_capture_and_started_zero_work_preserves(monkeypatch):
    _, exp = experiment()
    captures = []
    original = DatasetTraversal.capture_state
    def capture(owner):
        captures.append(owner.position)
        return original(owner)
    monkeypatch.setattr(DatasetTraversal, "capture_state", capture)
    with exp.train_fn.input_session(exp):
        pass
    first = exp.state.dataset_state
    assert len(captures) == 1 and first.position.sequence == 0
    with exp.train_fn.input_session(exp):
        pass
    assert len(captures) == 1 and exp.state.dataset_state is first
    with exp.train_fn.input_session(exp, positive=False, exact=False):
        pass
    assert exp.state.dataset_state is first and exp.state.input_recovery == "exact"


def unknown_counter():
    yield 1


def test_boundary_only_resume_refreshes_record_before_next_pass(monkeypatch):
    data = GeneratorDataset(unknown_counter, spec=TensorSpec("int64", shape=()))
    exp = Experiment(Counter(), Procedure(), data)
    with exp.train_fn.input_session(exp) as session:
        accept_one(exp, session)
    old = exp.state.dataset_state
    assert (old.position.epoch, old.position.start, old.position.sequence) == (0, 1, 1)
    with exp.train_fn.input_session(exp) as session:
        with pytest.raises(StopIteration):
            session.traversal.next_epoch()
        session.sync_boundary()
        assert (exp.state.epoch, exp.state.next_batch, exp.state.input_sequence) == (1, 0, 2)
        assert exp.state.pending_epoch_postlude == 0
        session.sync_boundary()
        assert exp.state.input_sequence == 2
    assert exp.state.step == 1
    assert exp.state.dataset_state is not old
    assert exp.state.dataset_state.position.sequence == 2


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_provider_retires_on_exception_before_any_update(error):
    _, exp = experiment()
    with pytest.raises(error):
        with exp.train_fn.input_session(exp):
            raise error("stop")
    assert exp.state._input_provider is None
    assert exp.state.input_started and exp.state.dataset_state is None


def test_positional_lane_keeps_weaker_retry_and_has_no_record():
    _, exp = experiment()
    with pytest.raises(RuntimeError):
        with exp.train_fn.input_session(exp, exact=False) as session:
            accept_one(exp, session)
            raise RuntimeError("positional callback")
    with exp.train_fn.input_session(exp) as session:
        assert session.traversal.next_epoch().reserve().value == 2
    assert exp.state.input_recovery == "positional" and exp.state.dataset_state is None


def test_metadata_validation_preserves_stale_record_without_value_encoding(monkeypatch):
    _, exp = experiment()
    with exp.train_fn.input_session(exp) as session:
        session.capture()
        prior = exp.state.dataset_state
        accept_one(exp, session)
        with monkeypatch.context() as patch:
            patch.setattr(TrainState, "__getstate__", lambda _: pytest.fail("candidate encoded continuation"))
            exp.state.validate()
            from dryml.models.jax.base import _restore_train_state
            candidate = TrainState()
            _restore_train_state(candidate, exp.state)
            candidate.validate()
            assert candidate.dataset_state is prior
        with pytest.raises(ValueError, match="current matching record"):
            exp.state.__getstate__()


def test_callback_from_accounting_endpoint_releases_native_guard_and_keeps_acceptance():
    _, exp = experiment()
    def callback():
        assert exp.state.step == exp.state.input_sequence == 1
        assert not exp.model._restore_failed
        raise RuntimeError("accepted callback")
    with pytest.raises(RuntimeError, match="accepted callback"):
        with exp.train_fn.input_session(exp) as session:
            candidate = session.traversal.next_epoch().reserve()
            with session.update(candidate):
                exp.model.value = 1
                record_train_update(exp, candidate.value, 1., batched=False,
                                    input_candidate=candidate, callbacks=(callback,))
    assert not exp.model._restore_failed and exp.state.step == exp.state.input_sequence == 1
    assert exp.state._input_provider is None


def test_delayed_native_failure_invalidates_even_after_matching_capture(tmp_path):
    repo, exp = experiment(tmp_path)
    with pytest.raises(RuntimeError, match="delayed native"):
        with exp.train_fn.input_session(exp, repo=repo) as session:
            accept_one(exp, session)
            session.capture()
            session.invalidate_native()
            raise RuntimeError("delayed native")
    assert exp.model._restore_failed and exp._restore_failed
    assert exp.state.dataset_state.position == exp.state.input_position
    with pytest.raises(RepoSaveError, match="invalidated"):
        repo.save_object(exp.model)


def test_conflicting_admission_rejects_before_input_preparation(monkeypatch):
    repo, exp = experiment()
    with repo.reserve_state_graph(exp):
        monkeypatch.setattr(type(exp.train_data), "prepare", lambda _: pytest.fail("conflicting preparation"))
        with pytest.raises(RepoSaveError, match="reserved"):
            with exp.train_fn.input_session(exp, repo=repo):
                pytest.fail("conflicting input admitted")
    assert exp.state.input_started is False and exp.state._input_provider is None


def test_provider_retirement_survives_replaced_train_state_binding():
    _, exp = experiment()
    prior = exp.state
    with pytest.raises(RepoSaveError, match="owner binding"):
        with exp.train_fn.input_session(exp):
            exp.state = TrainState()
    assert prior._input_provider is None and exp.state._input_provider is None


def test_quiescent_save_reuses_current_capture_until_next_accepted_update(tmp_path, monkeypatch):
    repo, exp = experiment(tmp_path)
    captures = []
    original = DatasetTraversal.capture_state
    def capture(owner):
        captures.append(owner.position.sequence)
        return original(owner)
    monkeypatch.setattr(DatasetTraversal, "capture_state", capture)
    with exp.train_fn.input_session(exp, repo=repo) as session:
        view = session.traversal.next_epoch()
        first = view.reserve()
        with session.update(first):
            record_train_update(exp, first.value, 1., batched=False, input_candidate=first)
        repo.save_object(exp)
        record = exp.state.dataset_state
        repo.save_object(exp)
        assert exp.state.dataset_state is record and captures == [1]
        second = view.reserve()
        with session.update(second):
            record_train_update(exp, second.value, 1., batched=False, input_candidate=second)
        repo.save_object(exp)
        assert captures == [1, 2]
    assert captures == [1, 2]


def test_unrecorded_new_input_boundary_cannot_be_silently_repaired_by_save():
    exp = Experiment(Counter(), Procedure(), GeneratorDataset(
        unknown_counter, spec=TensorSpec("int64", shape=())))
    with exp.train_fn.input_session(exp) as session:
        accept_one(exp, session)
        view = session.traversal._active
        with pytest.raises(StopIteration):
            view.reserve()
        with pytest.raises(ValueError, match="synchronize the boundary"):
            session.capture()
        session.sync_boundary()
        assert session.capture().position == exp.state.input_position


@pytest.mark.usefixtures("fixed_snapshot_environment")
def test_required_source_state_selection_is_checked_by_consumer_before_pull(tmp_path, monkeypatch):
    from dryml.data import Map
    from dryml.numpy import NumpyArray
    from tests.data.test_dataset_state_validation import PayloadScale
    repo = Repo(DirStore(tmp_path / "state", query_index="none"))
    payload = NumpyArray((1,), "int64", repo=repo)
    payload.array = np.asarray([2], dtype=np.int64)
    data = Map(ArrayDataset(np.arange(3)), PayloadScale(payload), repo=repo)
    required = repo.save_object(data)
    exp = Experiment(Counter(), Procedure(), data, repo=repo)
    with exp.train_fn.input_session(exp, repo=repo):
        pass
    assert exp.state.dataset_state.state_ref.graph_equal(required)
    with exp.train_fn.input_session(exp, repo=repo) as session:
        candidate = session.traversal.next_epoch().reserve()
        with session.update(candidate):
            record_train_update(exp, candidate.value, 1., batched=False, input_candidate=candidate)
    payload.array = np.asarray([3], dtype=np.int64)
    changed = repo.save_object(data)
    assert not changed.graph_equal(required)
    monkeypatch.setattr(DatasetTraversal, "_raw", lambda _: pytest.fail("mismatched required source pulled"))
    with pytest.raises(ContinuationMismatchError, match="SavedNode selection"):
        with exp.train_fn.input_session(exp, repo=repo):
            pass
    assert exp.train_data is data and exp.state._input_provider is None
